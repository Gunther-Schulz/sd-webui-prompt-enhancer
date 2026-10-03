import json
import logging
import os
import re
import socket
import threading
import time
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory ─────────────────────────────────────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODIFIERS_DIR = os.path.join(_EXT_DIR, "modifiers")

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"


# ── File loading ─────────────────────────────────────────────────────────────

def _load_file(path):
    """Load a JSON or YAML file and return parsed content."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        print(f"[PromptEnhancer] ERROR: {path} must be a YAML/JSON mapping (dict), got {type(data).__name__}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[PromptEnhancer] ERROR: Failed to load {path}: {e}")
    return {}


# Configured Local Overrides folders that do not exist, as of the last load.
_missing_local_dirs = []

_STARTER_DIR = os.path.join(_EXT_DIR, "starter")


def _seed_local_dir(directory):
    """Copy the starter files into a Local Overrides folder that holds no
    config file yet. They are comments only: they load nothing, and they
    put the file names and the format where the user edits. Never creates
    the folder itself (that is the launcher's decision, not ours) and never
    touches a folder that already has a file in it."""
    try:
        if any(n.endswith((".yaml", ".yml", ".json")) for n in os.listdir(directory)):
            return
        import shutil
        copied = []
        for name in sorted(os.listdir(_STARTER_DIR)):
            if name.endswith(".yaml"):
                shutil.copyfile(os.path.join(_STARTER_DIR, name), os.path.join(directory, name))
                copied.append(name)
        if copied:
            print(f"[PromptEnhancer] Local Overrides folder was empty; starter files written to {directory}: {', '.join(copied)}")
    except OSError as e:
        print(f"[PromptEnhancer] Could not write starter files to {directory}: {e}")


def _get_local_dirs(ui_path=""):
    """Resolve local overrides directories.

    Supports comma-separated paths in UI field or env var.
    Returns list of valid directory paths.
    """
    raw = (ui_path or "").strip()
    if not raw:
        raw = os.environ.get("PROMPT_ENHANCER_LOCAL", "").strip()
    if not raw:
        return []
    dirs = []
    _missing_local_dirs.clear()
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        if os.path.isdir(p):
            _seed_local_dir(p)
            dirs.append(p)
        else:
            # A configured folder that is not there loads nothing. Say so:
            # skipped silently, it reads exactly like "no overrides wanted".
            _missing_local_dirs.append(p)
            print(f"[PromptEnhancer] Local Overrides folder not found, nothing loaded from it: {p}")
    return dirs


def _load_local_bases(local_dirs):
    """Load _bases.yaml from all local directories.

    Entries may be either a string (legacy: body only) or a dict with
    keys like 'body', 'target', 'description'.
    """
    merged = {}
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, BASES_FILENAME + ext)
            if os.path.isfile(path):
                merged.update({k: v for k, v in _load_file(path).items() if isinstance(v, (str, dict))})
    return merged


def _scan_modifier_files(directory):
    """Scan a directory for modifier YAML/JSON files.

    Returns dict: dropdown_name -> {category: {modifier: keywords}}.
    Skips _bases.* files. Uses _label field from YAML if present,
    otherwise derives label from filename.
    """
    result = {}
    if not directory or not os.path.isdir(directory):
        return result
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        # Underscore-prefixed files are special (_bases, _prompts), never
        # modifier dropdowns. _prompts used to fall through to the _label
        # warning below on every load.
        if stem.startswith("_"):
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(directory, name))
        if data:
            # Require _label in YAML for dropdown name
            label = data.pop("_label", None)
            if not label:
                print(f"[PromptEnhancer] WARNING: Skipping {os.path.join(directory, name)}: missing '_label' field. Add '_label: Your Label' to the YAML file.")
                continue
            result[label] = data
    return result


def _merge_modifier_dicts(base, override):
    """Merge two dropdown-level dicts. Same dropdown name -> merge categories."""
    merged = {}
    for label, categories in base.items():
        merged[label] = {}
        for cat, items in categories.items():
            if isinstance(items, dict):
                merged[label][cat] = dict(items)
    for label, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if label not in merged:
            merged[label] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[label]:
                merged[label][cat] = {}
            merged[label][cat].update(items)
    return merged


def _normalize_modifier(entry):
    """Normalize a modifier entry to dict form with keys:
      behavioral: prose instruction. The prose itself tells the LLM
                  whether to apply a named style or pick one ("The scene
                  is lit by golden hour..." vs "Choose a specific lighting
                  condition..."). No separate type flag — the behavioral
                  text carries the meaning.
      keywords:   comma-separated keyword string. May be empty for
                  random-choice entries.

    Accepts:
      - str: legacy format. Treated as keywords; behavioral is synthesized.
      - dict: new format.
    """
    if isinstance(entry, str):
        kw = entry.strip()
        return {
            "behavioral": f"Apply this style to the scene — describe the qualities through prose, do not list them as keywords: {kw}.",
            "keywords": kw,
        }
    if not isinstance(entry, dict):
        return None
    norm = {
        "behavioral": (entry.get("behavioral") or "").strip(),
        "keywords": (entry.get("keywords") or "").strip(),
    }
    if not norm["behavioral"] and norm["keywords"]:
        norm["behavioral"] = f"Apply this style to the scene — describe the qualities through prose, do not list them as keywords: {norm['keywords']}."
    if not norm["behavioral"] and not norm["keywords"]:
        return None
    return norm


def _build_dropdown_data(categories_dict):
    """Build flat lookup and choice list from a single dropdown's categories.

    Values in the returned flat dict are normalized modifier dicts (see
    _normalize_modifier).

    An entry carrying a `source:` key picked its value from the tag
    database, which this extension no longer ships. Such an entry (a
    local override written for the old extension, for example) is
    skipped with a console line naming it.
    """
    flat = {}
    choices = []
    for cat_name, items in categories_dict.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {cat_name.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, entry in items.items():
            if isinstance(entry, dict) and "source" in entry:
                print(f"[PromptEnhancer] Skipping modifier '{name}': it has a 'source:' entry, "
                      f"which drew its value from the tag database that is no longer part of this extension.")
                continue
            norm = _normalize_modifier(entry)
            if norm is None:
                continue
            flat[name] = norm
            choices.append(name)
    return flat, choices


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}          # flat: name -> keywords (for lookup across all dropdowns)
_dropdown_order = []         # list of dropdown labels in display order
_dropdown_choices = {}       # label -> [choice_list with separators]
_prompts = {}                # operational prompts loaded from prompts.yaml


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _dropdown_order, _dropdown_choices, _prompts

    local_dirs = _get_local_dirs(local_dir_path)

    # Bases (YAML, with local overrides)
    _bases = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(_EXT_DIR, "bases" + ext)
        if os.path.isfile(path):
            _bases = {k: v for k, v in _load_file(path).items() if isinstance(v, (str, dict))}
            break
    _bases.update(_load_local_bases(local_dirs))

    # Modifiers: scan extension modifiers/ dir + all local dirs, merge
    all_mods = _scan_modifier_files(_MODIFIERS_DIR)
    for local_dir in local_dirs:
        local_mods = _scan_modifier_files(local_dir)
        all_mods = _merge_modifier_dicts(all_mods, local_mods)

    _all_modifiers = {}
    _dropdown_order = []
    _dropdown_choices = {}
    for label in sorted(all_mods.keys()):
        flat, choices = _build_dropdown_data(all_mods[label])
        if choices:
            _dropdown_order.append(label)
            _dropdown_choices[label] = choices
            _all_modifiers.update(flat)

    # Prompts (YAML, with local overrides)
    _prompts = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(_EXT_DIR, "prompts" + ext)
        if os.path.isfile(path):
            data = _load_file(path) or {}
            _prompts = {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}
            break
    # Merge local prompt overrides
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, "_prompts" + ext)
            if os.path.isfile(path):
                local_p = _load_file(path) or {}
                for k, v in local_p.items():
                    if isinstance(v, str):
                        _prompts[k] = v.strip()


_reload_all()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_body(entry):
    """Return the body string from a base entry (str or dict with 'body')."""
    if isinstance(entry, dict):
        return entry.get("body", "")
    return entry or ""


def _base_meta(name):
    """Return metadata dict (target, description, etc.) for a base, or {}."""
    entry = _bases.get(name)
    if isinstance(entry, dict):
        return {k: v for k, v in entry.items() if k != "body"}
    return {}


def _base_names():
    """Return ordered (label, value) tuples for the Base dropdown.

    Label text comes from each base's yaml: an optional `label:` string
    (used verbatim in the paren) takes precedence over auto-derivation
    from the `target:` list (first 3 entries joined). Curated bases appear
    first in a fixed order; user-added bases follow in yaml order.
    """
    CURATED_ORDER = ["Default"]

    def _label(value):
        meta = _base_meta(value)
        paren = meta.get("label")
        if not paren:
            target = meta.get("target", [])
            if target and isinstance(target, list):
                paren = ", ".join(str(t) for t in target[:3])
        return f"{value} ({paren})" if paren else value

    result = []
    seen = set()
    for value in CURATED_ORDER:
        if value in _bases:
            result.append((_label(value), value))
            seen.add(value)
    for value in _bases.keys():
        if value.startswith("_") or value in seen:
            continue
        result.append((_label(value), value))
    return result


def _collect_modifiers(dropdown_selections):
    """Collect all selected modifiers into a list of (name, normalized_entry) tuples."""
    result = []
    for selections in dropdown_selections:
        for name in (selections or []):
            entry = _all_modifiers.get(name)
            if entry:
                result.append((name, entry))
    return result


def _build_style_string(mod_list):
    """Build the style block for the user message.

    Uses each modifier's behavioral field and emits a comma-separated
    list of style directives. The active base prompt governs HOW these
    get applied (voice, structure); we just name the styles.
    """
    if not mod_list:
        return ""
    # Short behaviorals concatenate into a compact directive.
    # Qwen recognizes style/mood/setting concepts directly — we don't need
    # to teach it, just name them. Base prompt handles voice.
    behaviorals = []
    for name, entry in mod_list:
        text = (entry.get("behavioral") or "").strip()
        if text:
            # Strip trailing punctuation so items join cleanly with commas.
            text = text.rstrip(".!?: ")
            if text:
                behaviorals.append(text)
    if not behaviorals:
        return ""
    return f"Apply these styles to the scene: {', '.join(behaviorals)}."


# ── Ollama ───────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"

# Placeholder used in the user message when Source Prompt is empty (dice roll).
# Styles and wildcards still flow through normally; this only replaces the
# "SOURCE PROMPT: {source}" line so the LLM knows to invent rather than expand.
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _to_ollama_base(api_url):
    base = api_url
    for suffix in ("/v1/chat/completions", "/v1", "/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or DEFAULT_OLLAMA_BASE


def _fetch_ollama_models(api_url):
    try:
        base = _to_ollama_base(api_url)
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        models.sort()
        return models
    except Exception:
        return []


def _refresh_models(api_url, current_model):
    models = _fetch_ollama_models(api_url)
    if not models:
        return gr.update()
    value = current_model if current_model in models else models[0]
    return gr.update(choices=models, value=value)


def _get_ollama_status(api_url):
    """Get Ollama status info: running, model loaded, GPU/CPU."""
    try:
        base = _to_ollama_base(api_url)
        req = urllib.request.Request(f"{base}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            version_data = json.loads(resp.read().decode("utf-8"))
        version = version_data.get("version", "?")

        # Check loaded models
        req = urllib.request.Request(f"{base}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ps_data = json.loads(resp.read().decode("utf-8"))

        models = ps_data.get("models", [])
        if models:
            m = models[0]
            name = m.get("name", "?")
            vram = m.get("size_vram", 0)
            total = m.get("size", 0)
            if vram > 0 and total > 0:
                gpu_pct = int(vram / total * 100)
                mode = f"GPU ({gpu_pct}%)" if gpu_pct > 50 else f"CPU ({100-gpu_pct}% offloaded)"
            elif vram > 0:
                mode = "GPU"
            else:
                mode = "CPU"
            return f"<span style='color:#6c6'>Ollama v{version} \u2022 {name} \u2022 {mode}</span>"
        else:
            return f"<span style='color:#6c6'>Ollama v{version} \u2022 connected</span>"
    except Exception:
        return "<span style='color:#c66'>Ollama not running</span>"


# ── Core logic ───────────────────────────────────────────────────────────────

def _strip_think_blocks(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _has_inline_wildcards(text):
    return bool(re.search(r"\{[^}]+\?\}", text))


def _clean_output(text, strip_underscores=True):
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    if strip_underscores:
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    return text.strip()


def _split_positive_negative(text):
    """Split LLM output at POSITIVE:/NEGATIVE: markers.

    Returns (positive, negative).  If no markers found, returns (text, "").
    """
    # Case-insensitive search for markers
    pos_match = re.search(r"(?i)^POSITIVE:\s*\n?", text, re.MULTILINE)
    neg_match = re.search(r"(?i)^NEGATIVE:\s*\n?", text, re.MULTILINE)
    if not neg_match:
        # No NEGATIVE marker — treat entire text as positive
        return text.strip(), ""
    if pos_match:
        positive = text[pos_match.end():neg_match.start()].strip()
    else:
        # NEGATIVE marker but no POSITIVE marker — everything before is positive
        positive = text[:neg_match.start()].strip()
    negative = text[neg_match.end():].strip()
    return positive, negative


# ── Keep LoRAs ───────────────────────────────────────────────────────────────
# Forge's main prompt field can carry `<...>` tokens (`<lora:name:0.8>`,
# `<hypernet:...>`). An enhancer run overwrites that field, so with
# "Keep LoRAs" on the tokens are carried across the overwrite, and kept
# away from the model on the Remix path, which sends the field's text.
#
# Any `<...>` counts, so new token kinds ride along without a code
# change. Refusing nested brackets is what stops one match swallowing a
# whole prompt that holds two tokens. Known limit, accepted: prose with
# bare comparison brackets ("5 < 10 > 3") yields a spurious token.
_ANGLE_TOKEN_RE = re.compile(r"<[^<>]+>")


def _extract_angle_tokens(text):
    """Return the `<...>` tokens in text, in order of first appearance."""
    tokens = []
    for token in _ANGLE_TOKEN_RE.findall(text or ""):
        if token not in tokens:
            tokens.append(token)
    return tokens


def _strip_angle_tokens(text):
    """Remove `<...>` tokens and collapse the whitespace they leave behind."""
    text = _ANGLE_TOKEN_RE.sub(" ", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def _append_missing_tokens(text, tokens):
    """Append every token not already present in text.

    Idempotent: a token the text already holds is not added again, so a
    second run does not double it. Position is not preserved — Forge
    strips these tokens wherever they sit, so placement carries no meaning.
    """
    text = text or ""
    missing = [token for token in tokens if token not in text]
    if not missing:
        return text
    return f"{text} {' '.join(missing)}" if text else " ".join(missing)


_STALL_TIMEOUT = int(os.environ.get("PROMPT_ENHANCER_STALL_TIMEOUT", "10"))
# Max Ollama stream chunks (≈ tokens) before we cap and treat as
# truncation. Historical 1000 was too low once word limits were removed
# from base prompts — rich prose can reach ~1500 chunks. 4000
# leaves ample headroom without opening the door to actual runaways.
_MAX_TOKENS = int(os.environ.get("PROMPT_ENHANCER_MAX_TOKENS", "4000"))
# Total wall time cap. 60s was tight on slower models and rich prose;
# 180s catches genuine runaway loops without truncating legitimate calls.
_MAX_TIME = int(os.environ.get("PROMPT_ENHANCER_MAX_TIME", "180"))


def _detect_repetition(text):
    """Detect repetitive output by tracking unique content ratio.

    Splits text into comma/period-separated segments. Triggers on either:
      (a) Any single segment appears 4+ times — catches short degenerate
          loops the 30-segment window misses (e.g. "the girl's hair is a
          deep chestnut" appearing 10x in a 19-segment paragraph).
      (b) Of the last 20 segments, 70%+ are duplicates of earlier ones
          (requires ≥ 30 total segments — original longer-text detector).

    Returns trimmed text if loop detected, None otherwise.
    """
    segments = [s.strip().lower() for s in re.split(r"[,.\n]+", text) if s.strip()]
    if len(segments) < 6:
        return None
    # Signal (a): any segment verbatim 4+ times → unambiguous loop
    from collections import Counter as _C
    counts = _C(segments)
    worst_seg, worst_n = counts.most_common(1)[0]
    if worst_n >= 4 and len(worst_seg) >= 10:
        # Trim back to before the second-to-last occurrence of the repeated segment
        # so we keep the first full use + some context.
        positions = [i for i, s in enumerate(segments) if s == worst_seg]
        trim_before_idx = positions[1] if len(positions) >= 2 else positions[0]
        # Reconstruct text up to that segment
        parts = re.split(r"([,.\n]+)", text)
        seg_count = 0
        char_pos = 0
        for part in parts:
            if part.strip() and not re.match(r"^[,.\n]+$", part):
                if seg_count >= trim_before_idx:
                    break
                seg_count += 1
            char_pos += len(part)
        trimmed = text[:char_pos].rstrip(" ,.\n")
        print(f"[PromptEnhancer] Aborting: repetition detected "
              f"(segment {worst_seg!r} appeared {worst_n}x)")
        return trimmed
    # Signal (b): original longer-window detector
    if len(segments) < 30:
        return None
    # Check: of the last 20 segments, how many are duplicates of earlier segments?
    early = set(segments[:-20])
    recent = segments[-20:]
    if not early:
        return None
    repeated = sum(1 for s in recent if s in early)
    ratio = repeated / len(recent)
    if ratio >= 0.7:
        # 70%+ of recent segments already appeared — it's looping
        # Trim to the point where content was still fresh
        # Walk backwards to find where repetition started
        seen = set()
        trim_idx = len(segments)
        dup_streak = 0
        for i in range(len(segments) - 1, -1, -1):
            if segments[i] in seen:
                dup_streak += 1
            else:
                if dup_streak > 5:
                    trim_idx = i + 1
                    break
                dup_streak = 0
            seen.add(segments[i])
        # Rebuild text from non-repeated segments
        # Find character position of the trim point
        parts = re.split(r"([,.\n]+)", text)
        seg_count = 0
        char_pos = 0
        for part in parts:
            if part.strip() and not re.match(r"^[,.\n]+$", part):
                seg_count += 1
                if seg_count > trim_idx:
                    break
            char_pos += len(part)
        trimmed = text[:char_pos].rstrip(" ,.\n")
        print(f"[PromptEnhancer] Aborting: repetition detected ({ratio:.0%} of last 20 segments are repeats)")
        return trimmed
    return None


_cancel_flag = threading.Event()
_last_seed = -1
# Which PE button produced the currently-staged prompt. Set by each
# handler entry point, consumed by process() to write PE Mode into image
# metadata and shown in every status line via _MODE_* prefixes below.
_last_pe_mode: str | None = None

# Status line prefix per mode — matches the button glyphs so the status
# message is self-identifying at a glance.
_MODE_ENHANCE = "\u2728 Enhance"   # ✨ Enhance
_MODE_REMIX = "\U0001F500 Remix"   # 🔀 Remix


class _TruncatedError(Exception):
    """LLM output was truncated (stall, max tokens, or max time)."""
    pass


def _call_llm(prompt, api_url, model, system_prompt, temperature, think=False, timeout=None, seed=-1, _progress=None, num_predict=1024):
    global _last_seed
    import random as _random
    if seed == -1:
        seed = _random.randint(0, 2**31 - 1)
    _last_seed = seed

    base = _to_ollama_base(api_url)
    # Prepend /no_think to user message for Qwen3 models that ignore think:false
    user_content = prompt if think else f"/no_think\n{prompt}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": float(temperature),
            "seed": int(seed),
            "top_k": 20,
            "top_p": 0.95 if think else 0.8,
            # Both penalties act on tokens in the recent context, and the
            # source prompt IS the recent context. repeat_penalty 1.5 swapped
            # the user's own words (handrail -> escalator, "two" -> "three",
            # on her back -> prone) and garbled the grammar; presence_penalty
            # 1.5 still pushed explicit source words out. Ollama's default
            # repeat_penalty is the loop guard that remains: no loop or
            # truncation in ~600 outputs. Measured 2026-10-03,
            # experiments/PROSE-TUNING.md.
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            # Explicit output cap. Without this, Ollama falls back to
            # whatever the model's Modelfile specifies (often 128 for
            # instruct variants) — which produces ~60 words of prose,
            # then a truncated tag draft of ~15 tokens, then ~7 final
            # tags after validator + rule-layer dedup. 1024 gives room
            # for rich prose + multi-dozen-tag drafts.
            "num_predict": int(num_predict),
        },
        "think": bool(think),
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base}/api/chat"
    stall_timeout = timeout or _STALL_TIMEOUT
    print(f"[PromptEnhancer] LLM call: model={model}, think={think}, temp={temperature}, stall={stall_timeout}s, prompt_len={len(prompt)}, system_len={len(system_prompt)}")

    last_err = None
    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"[PromptEnhancer] Ollama retry attempt {attempt + 1}")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Initial connection timeout
            resp = urllib.request.urlopen(req, timeout=30)
            # Set socket timeout so reads unblock periodically for cancel checks.
            # Uses CPython internals — wrapped for safety.
            try:
                resp.fp.raw._sock.settimeout(2.0)
            except (AttributeError, OSError):
                pass
            content_parts = []
            completed = False
            start_time = time.monotonic()
            last_token_time = start_time
            thinking_detected = False

            try:
                while True:
                    try:
                        line = resp.readline()
                    except (socket.timeout, TimeoutError):
                        # Socket timeout — no data yet, check cancel and continue
                        if _cancel_flag.is_set():
                            print(f"[PromptEnhancer] Cancelled by user")
                            break
                        continue
                    if not line:
                        break  # EOF
                    # Check cancel flag
                    if _cancel_flag.is_set():
                        print(f"[PromptEnhancer] Cancelled by user")
                        break
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check for thinking tokens (Qwen3 ignoring think:false)
                    thinking_text = chunk.get("message", {}).get("thinking", "")
                    if thinking_text and not think:
                        if not thinking_detected:
                            thinking_detected = True
                            print(f"[PromptEnhancer] WARNING: Model is thinking despite think=false")
                        # Don't reset timer for thinking tokens — let it stall out
                        if time.monotonic() - last_token_time > stall_timeout:
                            print(f"[PromptEnhancer] Aborting: thinking exceeded {stall_timeout}s")
                            break
                        continue

                    # Content tokens
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        content_parts.append(token)
                        last_token_time = time.monotonic()
                        # Repetition detection: check every 20 tokens
                        if len(content_parts) % 20 == 0 and len(content_parts) >= 40:
                            text = "".join(content_parts)
                            repetition = _detect_repetition(text)
                            if repetition:
                                content_parts = [repetition]
                                break
                        # Update shared progress for live status
                        if _progress is not None:
                            elapsed = last_token_time - start_time
                            _progress["words"] = len("".join(content_parts).split())
                            _progress["tokens"] = len(content_parts)
                            _progress["elapsed"] = elapsed
                            _progress["tps"] = len(content_parts) / elapsed if elapsed > 0 else 0
                        # Hard cap on total tokens
                        if len(content_parts) > _MAX_TOKENS:
                            print(f"[PromptEnhancer] Aborting: exceeded {_MAX_TOKENS} tokens")
                            break

                    # Check for completion
                    if chunk.get("done", False):
                        completed = True
                        break

                    # Check stall (no content token for too long)
                    if time.monotonic() - last_token_time > stall_timeout:
                        print(f"[PromptEnhancer] Aborting: no tokens for {stall_timeout}s")
                        break

                    # Total time cap (catches thinking disguised as content)
                    if time.monotonic() - start_time > _MAX_TIME:
                        print(f"[PromptEnhancer] Aborting: exceeded {_MAX_TIME}s total time")
                        break
            finally:
                try:
                    resp.close()
                except OSError:
                    pass

            content = "".join(content_parts)
            was_cancelled = _cancel_flag.is_set()
            print(f"[PromptEnhancer] Done: {len(content.split())} words, thinking={'yes' if thinking_detected else 'no'}, cancelled={'yes' if was_cancelled else 'no'}")
            result = _strip_think_blocks(content)
            if was_cancelled:
                raise InterruptedError(result)
            if not completed and result:
                raise _TruncatedError(result)
            return result

        except urllib.error.URLError as e:
            last_err = e
            print(f"[PromptEnhancer] Ollama connection failed (attempt {attempt + 1}): {e.reason}")
            if attempt == 0:
                time.sleep(2)
        except TimeoutError as e:
            last_err = urllib.error.URLError(str(e))
            print(f"[PromptEnhancer] Timeout (attempt {attempt + 1}): {e}")
            if attempt == 0:
                time.sleep(2)

    raise last_err


def _build_inline_wildcard_text(source):
    """Return the inline-wildcard directive if source contains {name?} placeholders.

    The previous selected-wildcards system was folded into the modifier
    system (🎲 Random X entries). This helper only handles the remaining
    inline {name?} syntax in the source prompt itself.
    """
    if source and _has_inline_wildcards(source):
        return _prompts.get("inline_wildcard", "")
    return ""


def _assemble_system_prompt(base_name):
    """Assemble the system prompt for a base (no modifiers/wildcards).

    Wraps the per-base body with the shared _preamble (prepended) and
    _format (appended) blocks from bases.yaml when present.
    """
    body = _base_body(_bases.get(base_name))
    if not body:
        return None
    parts = [
        _base_body(_bases.get("_preamble")).strip(),
        body.strip(),
        _base_body(_bases.get("_format")).strip(),
    ]
    return "\n\n".join(p for p in parts if p) or None


# ── `@` / `@@` sigils in the source prompt ───────────────────────────────────
# `prompt @ extra`        appends `extra` to the system prompt, under the
#                         joining line `sigil_append` from prompts.yaml.
# `prompt @@ replacement` replaces the system prompt with `replacement`.
# The sigil and everything after it never reach the user message. An
# empty suffix changes nothing, so a half-typed `foo @@` behaves like
# `foo`. There is no escape for a literal `@`: `a poster for @midnight`
# splits, and the effect is visible in the output rather than silent.
# Same behaviour as the WanGP prompt enhancer this is ported from.

def _split_sigil(prompt):
    """Split a source prompt at its sigil: (body, suffix, replace).

    `@@` is tested before `@`, the split is on the first occurrence,
    and both halves are stripped.
    """
    prompt = str(prompt or "").strip()
    body, separator, suffix = prompt.partition("@@")
    if separator == "@@":
        return body.strip(), suffix.strip(), True
    body, separator, suffix = prompt.partition("@")
    if separator == "":
        return prompt, "", False
    return body.strip(), suffix.strip(), False


def _merge_system_prompt(system_prompt, suffix, replace=False):
    """Apply a sigil suffix to a system prompt.

    Empty suffix: the system prompt unchanged. replace: the suffix
    alone. Otherwise the suffix appended under the joining line.
    """
    system_prompt = str(system_prompt or "").rstrip()
    suffix = str(suffix or "").strip()
    if not suffix:
        return system_prompt
    if replace:
        return suffix
    return f"{system_prompt}\n{_prompts['sigil_append']}\n{suffix}"


# ── Streaming progress ──────────────────────────────────────────────────────

def _call_llm_progress(prompt, api_url, model, system_prompt, temperature,
                       think=False, timeout=None, seed=-1):
    """Run _call_llm in a thread, yielding progress dicts every ~1s.

    Final yield is the result string. Exceptions propagate normally.
    Must be iterated from a generator function wired to a Gradio .click() handler.
    """
    progress = {"words": 0, "tokens": 0, "elapsed": 0.0, "tps": 0.0}
    result_box = [None]
    error_box = [None]

    def _worker():
        try:
            result_box[0] = _call_llm(prompt, api_url, model, system_prompt,
                                      temperature, think=think, timeout=timeout,
                                      seed=seed, _progress=progress)
        except Exception as e:
            error_box[0] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=1.0)
        if thread.is_alive():
            yield dict(progress)

    if error_box[0] is not None:
        raise error_box[0]
    yield result_box[0]


# ── UI ───────────────────────────────────────────────────────────────────────

class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        initial_models = _fetch_ollama_models(DEFAULT_API_URL)
        if not initial_models:
            initial_models = [DEFAULT_MODEL]

        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt ──
            source_prompt = gr.Textbox(
                label="Source Prompt", lines=3,
                placeholder="Type your prompt here, or leave empty to roll the dice. Use {name?} for inline wildcards, 'prompt @ extra instructions' to add to the system prompt, 'prompt @@ instructions' to replace it.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(value="\u2728 Enhance", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_enhance_btn")
                refine_btn = gr.Button(value="\U0001f500 Remix", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_refine_btn")
                cancel_btn = gr.Button(value="\u274c Cancel", scale=0, min_width=80, elem_id=f"{tab}_pe_cancel_btn")
                prepend_source = gr.Checkbox(label="Prepend", value=False, scale=0, min_width=60)
                prepend_source.do_not_save_to_config = True
                negative_prompt_cb = gr.Checkbox(label="+ Negative", value=False, scale=0, min_width=110)
                negative_prompt_cb.do_not_save_to_config = True
                motion_cb = gr.Checkbox(label="+ Motion and Audio", value=False, scale=0, min_width=150)
                motion_cb.do_not_save_to_config = True
                keep_loras = gr.Checkbox(label="Keep LoRAs", value=True, scale=0, min_width=110)
                keep_loras.do_not_save_to_config = True
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            # ── Base ──
            with gr.Row():
                base = gr.Dropdown(label="Base", choices=_base_names(), value="Default", scale=1, info="Prose voice — matches the image model family.")

            def _base_description_html(name):
                meta = _base_meta(name)
                desc = meta.get("description", "")
                if not desc:
                    return ""
                return f"<div style='color:#888; font-size:0.9em; margin-top:-8px; padding-left:4px'>{desc}</div>"

            base_description = gr.HTML(value=_base_description_html("Default"))
            base.change(fn=_base_description_html, inputs=[base], outputs=[base_description], show_progress=False)

            # ── Auto-generated modifier dropdowns (one per file) ──
            dd_components = []
            dd_labels = list(_dropdown_order)

            # Layout: 3 dropdowns per row, pad incomplete rows
            for i in range(0, len(dd_labels), 3):
                row_labels = dd_labels[i:i+3]
                with gr.Row():
                    for label in row_labels:
                        d = gr.Dropdown(
                            label=label,
                            choices=_dropdown_choices.get(label, []),
                            value=[], multiselect=True, scale=1,
                        )
                        d.do_not_save_to_config = True
                        dd_components.append(d)
                    # Pad incomplete rows so dropdowns don't stretch
                    for _ in range(3 - len(row_labels)):
                        gr.HTML(value="", visible=True, scale=1)

            # ── Temperature + Think + Seed ──
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.8, step=0.05, scale=2, info="0 = deterministic, 2 = creative")
                seed = gr.Number(label="Seed", value=-1, minimum=-1, step=1, scale=1, info="-1 = random", precision=0, elem_id=f"{tab}_pe_seed")
                seed.do_not_save_to_config = True
                seed_random_btn = ToolButton(value="\U0001f3b2", elem_id=f"{tab}_pe_seed_random")
                seed_reuse_btn = ToolButton(value="\u267b", elem_id=f"{tab}_pe_seed_reuse")
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)
                think.do_not_save_to_config = True
                seed_random_btn.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
                seed_reuse_btn.click(fn=lambda: _last_seed, inputs=[], outputs=[seed], show_progress=False)

            # ── API + reload ──
            with gr.Row():
                api_url = gr.Textbox(label="API URL", value=DEFAULT_API_URL, scale=3)
                model = gr.Dropdown(label="Model", choices=initial_models, value=DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0], allow_custom_value=True, scale=2)
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                # Pre-filled with the folder the launcher passed in
                # (PROMPT_ENHANCER_LOCAL), so what is in use is what is shown.
                local_dir_path = gr.Textbox(
                    label="Local Overrides",
                    value=_env_local,
                    placeholder="Comma-separated dirs (refreshes content only, restart for new dropdowns)",
                    info=("Folder not found: " + ", ".join(_missing_local_dirs)) if _missing_local_dirs else None,
                    scale=3,
                )
                local_dir_path.do_not_save_to_config = True
                reload_btn = gr.Button(value="\U0001f504 Reload", scale=0, min_width=100)
                refresh_models_btn = gr.Button(value="\U0001f504 Models", scale=0, min_width=100)
            ollama_status = gr.HTML(value=_get_ollama_status(DEFAULT_API_URL))

            # ── Reload wiring ──
            # Note: reload rebuilds dropdowns but can't add/remove them dynamically.
            # New files require a Forge restart. Existing dropdown contents are refreshed.
            def _do_refresh(current_base, *args):
                # Last arg is local_dir_path
                local_path = args[-1]
                dd_vals = args[:-1]

                _reload_all(local_path)
                results = [gr.update(choices=_base_names(), value=current_base if current_base in _bases else "Default")]
                for i, label in enumerate(dd_labels):
                    choices = _dropdown_choices.get(label, [])
                    old_val = dd_vals[i] if i < len(dd_vals) else []
                    results.append(gr.update(choices=choices, value=[v for v in (old_val or []) if v in _all_modifiers]))
                msg = (f"<span style='color:#6c6'>Reloaded: {len(_bases)} bases, "
                       f"{len(_dropdown_order)} modifier groups, "
                       f"{len(_all_modifiers)} modifiers, "
                       f"{len(_prompts)} prompts</span>")
                if _missing_local_dirs:
                    msg += (f" <span style='color:#c66'>Local Overrides folder not found: "
                            f"{', '.join(_missing_local_dirs)}</span>")
                results.append(msg)
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base] + dd_components + [local_dir_path],
                outputs=[base] + dd_components + [status],
                show_progress=False,
            )
            def _refresh_models_and_status(api_url, current_model):
                return _refresh_models(api_url, current_model), _get_ollama_status(api_url)
            refresh_models_btn.click(fn=_refresh_models_and_status, inputs=[api_url, model], outputs=[model, ollama_status], show_progress=False)

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")
            negative_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_in")
            negative_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_out")

            # First step of both buttons: copy the main prompt fields into
            # the hidden inputs. Remix edits that text; Enhance only reads
            # it for the `<...>` tokens to keep.
            pull_main_fields_js = f"""function(x, y) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    var neg = document.querySelector('#{tab}_neg_prompt textarea');
                    return [ta ? ta.value : x, neg ? neg.value : y];
                }}"""

            def _pull_main_fields(x, y):
                _cancel_flag.clear()
                return x, y

            # ── Enhance ──
            def _enhance(existing, source, api_url, model, base_name, *args):
                global _last_pe_mode
                _last_pe_mode = "Enhance"
                keep_loras = args[-1]
                motion_cb = args[-2]
                neg_cb, temp = args[-3], args[-4]
                prepend, sd, th = args[-7], args[-6], args[-5]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()
                # From here on `source` is the prompt without its sigil part.
                source, sigil_suffix, sigil_replace = _split_sigil(source)
                replaced = sigil_replace and bool(sigil_suffix)
                # `<...>` tokens in the main prompt field, to carry across
                # the overwrite. The field's text itself is not used here.
                kept_tokens = _extract_angle_tokens(existing) if keep_loras else []

                mods = _collect_modifiers(dd_vals)
                sp = _assemble_system_prompt(base_name)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_ENHANCE}: No system prompt configured.</span>"
                    return

                sp = _merge_system_prompt(sp, sigil_suffix, sigil_replace)

                # After an `@@` replace the user's text is the whole system
                # prompt: nothing below is appended to it.
                if not replaced:
                    if motion_cb:
                        sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                    if neg_cb:
                        sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                # Build user message with modifiers + inline wildcards
                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                initial_status = "\U0001F3B2 Rolling dice (prose)..." if not source else "Generating prose..."
                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_ENHANCE}: {initial_status}</span>"

                print(f"[PromptEnhancer] Enhance: model={model}, think={th}, mods={len(mods)}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_ENHANCE}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_ENHANCE}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw)
                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""
                    if prepend and source:
                        result = f"{source}\n\n{result}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield _append_missing_tokens(result, kept_tokens), negative, f"<span style='color:#6c6'>{_MODE_ENHANCE}: OK - {len(result.split())} words, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield _append_missing_tokens(partial, kept_tokens), "", f"<span style='color:#c66'>{_MODE_ENHANCE}: Cancelled - {len(partial.split())} words (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_ENHANCE}: Cancelled</span>"
                except _TruncatedError as e:
                    result = _clean_output(str(e))
                    yield _append_missing_tokens(result, kept_tokens), "", f"<span style='color:#ca6'>{_MODE_ENHANCE}: Truncated - {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_ENHANCE}: {msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_ENHANCE}: {msg}</span>"

            enhance_btn.click(
                fn=_pull_main_fields,
                _js=pull_main_fields_js,
                inputs=[prompt_in, negative_in], outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_enhance,
                inputs=[prompt_in, source_prompt, api_url, model, base]
                       + dd_components
                       + [prepend_source, seed, think, temperature, negative_prompt_cb, motion_cb, keep_loras],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Remix ──
            def _refine(existing, existing_neg, source, api_url, model, *args):
                global _last_pe_mode
                _last_pe_mode = "Remix"
                keep_loras = args[-1]
                motion_cb = args[-2]
                neg_cb, temp = args[-3], args[-4]
                prepend, sd, th = args[-7], args[-6], args[-5]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()

                # Remix sends the main field's text to the model, so the
                # `<...>` tokens come out before the send and go back on
                # after it.
                kept_tokens = _extract_angle_tokens(existing) if keep_loras else []
                if keep_loras:
                    existing = _strip_angle_tokens(existing)
                existing = (existing or "").strip()
                existing_neg = (existing_neg or "").strip()
                print(f"[PromptEnhancer] Remix: existing_len={len(existing)}, source_len={len((source or '').strip())}, neg={neg_cb}")
                if not existing:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: No prompt to remix. Generate one first with Enhance.</span>"
                    return

                # From here on `source` is the instruction without its sigil part.
                source, sigil_suffix, sigil_replace = _split_sigil(source)
                replaced = sigil_replace and bool(sigil_suffix)
                mods = _collect_modifiers(dd_vals)
                print(f"[PromptEnhancer] Remix: mods={len(mods)}, source={'yes' if source else 'no'}")

                if not mods and not source and not sigil_suffix:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Select modifiers or update source prompt.</span>"
                    return

                # The sigil acts on the editor prompt. The user's own
                # instruction and styles are appended after it either way.
                sp = _merge_system_prompt(_prompts.get("remix_prose", ""), sigil_suffix, sigil_replace)

                if not replaced:
                    if motion_cb:
                        sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                    if neg_cb:
                        sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                if source:
                    sp = f"{sp}\n\nInstruction:\n{source}"
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"

                # Build user message — include current negative if checkbox is on
                user_msg = existing
                if neg_cb and existing_neg:
                    user_msg = f"{user_msg}\n\nCurrent negative prompt:\n{existing_neg}"

                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw)

                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""

                    if prepend and source:
                        result = f"{source}\n\n{result}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield _append_missing_tokens(result, kept_tokens), negative, f"<span style='color:#6c6'>{_MODE_REMIX}: OK - remixed to {len(result.split())} words, {elapsed}</span>"
                except InterruptedError:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Cancelled</span>"
                except _TruncatedError as e:
                    # Truncated prose is still prose — surface it so the
                    # user can use or edit it.
                    yield _append_missing_tokens(_clean_output(str(e)), kept_tokens), "", f"<span style='color:#ca6'>{_MODE_REMIX}: Truncated</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: {type(e).__name__}: {e}</span>"

            refine_btn.click(
                fn=_pull_main_fields,
                _js=pull_main_fields_js,
                inputs=[prompt_in, negative_in], outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_refine,
                inputs=[prompt_in, negative_in, source_prompt, api_url, model]
                       + dd_components
                       + [prepend_source, seed, think, temperature, negative_prompt_cb, motion_cb, keep_loras],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Cancel ──
            # Only sets the threading flag. The generation function detects
            # it via InterruptedError and returns the "Cancelled" status
            # through Gradio's normal .then() output delivery.
            # - No cancels=: that kills the asyncio task, orphaning the return value
            # - No _js: DOM manipulation can desync Svelte component state
            # - No outputs: avoids racing with the generation function's status output
            # - trigger_mode="multiple": default "once" silently drops repeat clicks
            cancel_btn.click(
                fn=lambda: _cancel_flag.set(),
                inputs=[], outputs=[],
                queue=False,
                show_progress=False,
                trigger_mode="multiple",
            )

            # ── Write to main prompt textarea ──
            prompt_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[prompt_out], outputs=[prompt_out], show_progress=False,
            )

            # ── Write to negative prompt textarea ──
            negative_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_neg_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[negative_out], outputs=[negative_out], show_progress=False,
            )

        # ── Metadata ──
        def _parse_modifiers(params):
            """Parse PE Modifiers string into a set of names."""
            raw = params.get("PE Modifiers", "")
            return {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

        def _make_dd_restore(dd_label):
            """Create a restore function for a specific dropdown."""
            dd_choices = _dropdown_choices.get(dd_label, [])
            def restore(params):
                saved = _parse_modifiers(params)
                return [m for m in saved if m in dd_choices and m in _all_modifiers]
            return restore

        def _restore_temperature(params):
            raw = params.get("PE Temperature", "")
            if not raw:
                return gr.update()
            try:
                return max(0.0, min(2.0, float(raw)))
            except (TypeError, ValueError):
                return gr.update()

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (think, "PE Think"),
            (seed, lambda params: int(params.get("PE Seed", -1)) if params.get("PE Seed") else -1),
            (temperature, _restore_temperature),
            (prepend_source, lambda params: params.get("PE Prepend", "").lower() == "true"),
            (motion_cb, lambda params: params.get("PE Motion", "").lower() == "true"),
        ]
        # Add each modifier dropdown
        for i, label in enumerate(dd_labels):
            self.infotext_fields.append((dd_components[i], _make_dd_restore(label)))

        self.paste_field_names = [
            "PE Source", "PE Base", "PE Modifiers",
            "PE Think", "PE Seed",
            "PE Temperature", "PE Prepend", "PE Motion", "PE Mode",
        ]

        return [source_prompt, api_url, model, base,
                *dd_components, prepend_source, seed, think, temperature,
                negative_prompt_cb, motion_cb]

    def process(self, p, source_prompt, api_url, model, base, *args):
        # args = *dd_values, prepend_source, seed, think, temperature, negative_prompt_cb, motion_cb
        motion = args[-1]
        neg_cb = args[-2]
        temp = args[-3]
        think = args[-4]
        prepend = args[-6]
        dd_vals = args[:-6]

        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base

        all_mod_names = []
        for dd_val in dd_vals:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)
        if think:
            p.extra_generation_params["PE Think"] = True
        if neg_cb:
            p.extra_generation_params["PE Negative"] = True
        if _last_seed >= 0:
            p.extra_generation_params["PE Seed"] = _last_seed
        if temp is not None:
            p.extra_generation_params["PE Temperature"] = round(float(temp), 3)
        if prepend:
            p.extra_generation_params["PE Prepend"] = True
        if motion:
            p.extra_generation_params["PE Motion"] = True
        if _last_pe_mode:
            p.extra_generation_params["PE Mode"] = _last_pe_mode
