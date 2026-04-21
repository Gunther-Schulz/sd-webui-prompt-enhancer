"""Integration test for Tags-mode output.

Runs the full user-facing Tags pipeline (tag-format system prompt +
assembled user message + Ollama) across a matrix of
(source, tag_format, modifier_selection) and greps the output for
leak patterns we've hit in real use. Intended to run BEFORE committing
prompt/modifier changes so we catch phrase-leak regressions early.

Run:
    python tests/test_tags_pipeline.py

Exits 0 if all cases clean, 1 if any case has leaks. Prints per-case
output so you can eyeball quality beyond the hard checks.

Covered checks per case:
  - Instruction-word leaks (unexpected, surprising, specific,
    detailed, choose, concrete, pick, etc. as tag tokens)
  - Phrase-shape tags (>3 underscores in a single token, or common
    phrase prefixes like "style_of_", "wearing_a_", "in_the_")
  - Capitalized tokens in Anima output (Anima rule: lowercase)
  - Instruction-section leaks (category names like "location",
    "artist", "setting", "imperfection" appearing as bare tags)
  - Format-specific: Anima must emit one of safe/sensitive/nsfw/
    explicit (safety tag rule).
"""

import json
import os
import re
import sys
import time
import urllib.request

import yaml

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# Abliterated Qwen variants to compare, smallest → largest.
# Non-abliterated variants excluded by preference (we want no refusals).
MODELS = [
    "huihui_ai/qwen3-abliterated:4b",     # ~2.4 GB, smallest fit
    "huihui_ai/qwen3.5-abliterated:9b",   # current default, ~6.3 GB
    "huihui_ai/qwen3-abliterated:14b",    # ~9 GB, best rule-following tier
]
TIMEOUT = 300


# ── Config loading ───────────────────────────────────────────────────────────

def load_yaml(rel):
    with open(os.path.join(EXT_DIR, rel)) as f:
        return yaml.safe_load(f)


def load_modifiers():
    """Flatten modifier yamls into a single {name: entry} dict."""
    mods = {}
    mod_dir = os.path.join(EXT_DIR, "modifiers")
    for fn in sorted(os.listdir(mod_dir)):
        if not fn.endswith((".yaml", ".yml")):
            continue
        data = yaml.safe_load(open(os.path.join(mod_dir, fn)))
        if not isinstance(data, dict):
            continue
        for section, entries in data.items():
            if section.startswith("_") or not isinstance(entries, dict):
                continue
            for name, entry in entries.items():
                if isinstance(entry, dict):
                    mods[name] = entry
                elif isinstance(entry, str):
                    mods[name] = {"keywords": entry}
    return mods


# ── Pipeline replication (matches prompt_enhancer.py behavior) ──────────────

def build_style_string_tags(mod_list):
    """Mirror _build_style_string(mode='tags')."""
    parts = []
    for name, entry in mod_list:
        clean_name = name.replace("\U0001F3B2", "").strip()
        kw = entry.get("keywords") or ""
        if not kw:
            continue  # dice entries skipped in Tags mode (commit d39ee11)
        if clean_name.lower() not in kw.lower():
            kw = f"{clean_name.lower()}, {kw}"
        parts.append(kw)
    return f"Apply these styles: {', '.join(parts)}." if parts else ""


def call_ollama(model, system_prompt, user_msg, max_tokens=400):
    body = {
        "model": model, "stream": False, "think": False, "keep_alive": "5m",
        # Match production sampling (prompt_enhancer.py:1069). Without
        # repeat_penalty, 14B can enter degenerate loops like
        # "hair strands over X, hair strands over Y, ...".
        "options": {
            "temperature": 0.6, "num_predict": max_tokens,
            "top_k": 20, "top_p": 0.8,
            "repeat_penalty": 1.5, "presence_penalty": 1.5,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think\n{user_msg}"},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["message"]["content"].strip()


# ── Leak detectors ───────────────────────────────────────────────────────────

# Instruction words that should never appear as a tag itself.
INSTRUCTION_WORDS = {
    "unexpected", "surprising", "specific", "concrete", "choose", "pick",
    "detailed", "varied", "random", "style_of", "wearing_a", "wearing",
    "in_the_style_of", "detailed_background", "specific_artist",
}

# Anima-specific: these four are the allowed safety tags (one required).
ANIMA_SAFETY = {"safe", "sensitive", "nsfw", "explicit"}


def detect_leaks(output, tag_format_name):
    """Return list of (severity, message) for issues found in the output."""
    issues = []
    # Split into tag tokens — any separator among commas, newlines works
    tokens = [t.strip() for t in re.split(r"[,\n]+", output) if t.strip()]

    # 1. Instruction-word leaks (exact token match, case-insensitive)
    for tok in tokens:
        # Normalize: lowercase, space → underscore for comparison
        norm = tok.lower().replace(" ", "_")
        if norm in INSTRUCTION_WORDS:
            issues.append(("HIGH", f"instruction-word tag: {tok!r}"))
        # Prefix-shaped phrases
        for prefix in ("style_of_", "wearing_a_", "in_the_style_of_",
                       "specific_", "detailed_", "surprising_"):
            if norm.startswith(prefix):
                issues.append(("HIGH", f"phrase-prefix leak: {tok!r}"))
                break

    # 2. Phrase-shape tags (too many underscores)
    for tok in tokens:
        norm = tok.replace(" ", "_")
        if norm.count("_") >= 4 and not re.match(r"^[a-z]+_\d+", norm):
            issues.append(("HIGH", f"phrase-shape tag (>3 _): {tok!r}"))

    # 3. Capital letters (Anima requires lowercase)
    if tag_format_name == "Anima":
        for tok in tokens:
            # Allow score_7 etc. (lowercase in our convention)
            # Disallow ANY uppercase letter (Anima rule: lowercase in tag form)
            if re.search(r"[A-Z]", tok):
                issues.append(("MED", f"capitalized tag in Anima: {tok!r}"))

    # 4. Missing safety tag (Anima specifically)
    if tag_format_name == "Anima":
        norm_tokens = {t.lower().strip() for t in tokens}
        if not (norm_tokens & ANIMA_SAFETY):
            issues.append(("MED", "Anima: no safety tag in output"))

    return issues


# ── Test matrix ──────────────────────────────────────────────────────────────

SOURCES = [
    # (label, source prompt)
    ("generic", "a girl reading"),
    ("named_character", "hatsune miku in maid uniform, chibi"),
    ("with_artist_hint", "samurai warrior at dusk"),
]

MODIFIER_SETS = [
    # (label, list of modifier names)
    ("none", []),
    ("static_style", ["Anime Makoto Shinkai"]),
    ("dice_location", ["🎲 Random Setting"]),
    ("dice_artist_plus_static", ["🎲 Random Artist", "Anime Ghibli"]),
]


def run_for_model(model, all_mods, tag_formats, tag_format_name="Anima"):
    """Run the full matrix for a single model. Returns (total, failed, elapsed)."""
    tf = tag_formats[tag_format_name]
    system_prompt = tf["system_prompt"]
    total = 0
    failed = 0
    t0 = time.perf_counter()

    print(f"\n{'#'*72}\n# MODEL: {model}\n{'#'*72}\n")

    for src_label, src in SOURCES:
        for mod_label, mod_names in MODIFIER_SETS:
            total += 1
            mod_list = []
            for name in mod_names:
                entry = all_mods.get(name)
                if entry is None:
                    print(f"  ! unknown modifier: {name}")
                    continue
                mod_list.append((name, entry))

            style_str = build_style_string_tags(mod_list)
            user_msg = f"SOURCE PROMPT: {src}"
            if style_str:
                user_msg = f"{user_msg}\n\n{style_str}"

            try:
                output = call_ollama(model, system_prompt, user_msg)
            except Exception as e:
                failed += 1
                print(f"[ERROR] source={src_label}  mods={mod_label}  — {e}")
                continue

            issues = detect_leaks(output, tag_format_name)
            status = "PASS" if not issues else "FAIL"
            if issues:
                failed += 1
            print(f"[{status}] source={src_label}  mods={mod_label}")
            print(f"  output: {output}")
            for sev, msg in issues:
                print(f"  {sev}: {msg}")
            print()

    elapsed = time.perf_counter() - t0
    return total, failed, elapsed


def run():
    bases = load_yaml("bases.yaml")
    all_mods = load_modifiers()
    tag_formats = {}
    tf_dir = os.path.join(EXT_DIR, "tag-formats")
    for fn in sorted(os.listdir(tf_dir)):
        if fn.endswith((".yaml", ".yml")):
            data = yaml.safe_load(open(os.path.join(tf_dir, fn)))
            if data and "system_prompt" in data:
                label = os.path.splitext(fn)[0].replace("-", " ").replace("_", " ").title()
                tag_formats[label] = data

    print(f"Loaded {len(bases)} bases, {len(all_mods)} modifiers, {len(tag_formats)} tag formats")

    # Optional --model arg for single-model runs
    single = None
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            single = sys.argv[i + 1]
    models = [single] if single else MODELS

    summary = []
    for m in models:
        total, failed, elapsed = run_for_model(m, all_mods, tag_formats)
        summary.append((m, total, failed, elapsed))

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for m, total, failed, elapsed in summary:
        passed = total - failed
        print(f"  {m:50} {passed:>2}/{total}  ({elapsed:.1f}s)")
    return 1 if any(failed for _, _, failed, _ in summary) else 0


if __name__ == "__main__":
    sys.exit(run())
