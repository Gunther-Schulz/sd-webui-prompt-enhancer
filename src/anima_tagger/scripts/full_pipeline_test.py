"""End-to-end test of the real Hybrid pipeline with modifiers + Ollama.

Simulates what the Forge extension will do in Hybrid mode once the
anima_tagger module is integrated:

  1. Build a shortlist of real artists/characters/series from the
     source prompt + active modifiers (via retriever).
  2. Run LLM pass 1 (Detailed base + modifier behaviorals + shortlist
     injection) → rich prose.
  3. Run LLM pass 3 (Anima tag system prompt + rich prose) → tag draft.
  4. Validate draft + apply Anima rule layer → final tag list.

Scenarios exercise real modifier combinations that have surfaced
quality issues historically (🎲 Random Artist / 🎲 Random Setting).

Run:
    python src/anima_tagger/scripts/full_pipeline_test.py
"""

import json
import os
import sys
import time
import urllib.request

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import load_all


EXT_DIR = os.path.abspath(os.path.join(_SRC, ".."))
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "huihui_ai/qwen3.5-abliterated:9b"


# ── Config loading (mimics what prompt_enhancer.py does on startup) ────

def load_yaml(rel: str) -> dict:
    with open(os.path.join(EXT_DIR, rel)) as f:
        return yaml.safe_load(f)


def load_modifiers() -> dict:
    mods: dict = {}
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


def build_style_prose(mod_names: list[str], all_mods: dict) -> str:
    """Mirror _build_style_string(mode='prose') from prompt_enhancer.py."""
    parts = []
    for name in mod_names:
        entry = all_mods.get(name) or {}
        bh = (entry.get("behavioral") or "").strip().rstrip(".!?: ")
        if bh:
            parts.append(bh)
    return f"Apply these styles to the scene: {', '.join(parts)}." if parts else ""


def assemble_base_prompt(bases: dict, name: str) -> str:
    preamble = bases.get("_preamble", {}).get("body", "")
    fmt = bases.get("_format", {}).get("body", "")
    entry = bases[name]
    body = entry["body"] if isinstance(entry, dict) else entry
    return "\n\n".join(p.strip() for p in [preamble, body, fmt] if p and p.strip())


# ── Ollama helpers ────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_msg: str, max_tokens: int = 600) -> str:
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": "5m",
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
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


# ── Scenarios ─────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label": "reading + Random Setting",
        "source": "a girl reading in a cafe",
        "modifiers": ["🎲 Random Setting"],
    },
    {
        "label": "Miku + Random Artist",
        "source": "hatsune miku holding a cake, chibi",
        "modifiers": ["🎲 Random Artist"],
    },
    {
        "label": "samurai + Random Artist + Random Setting",
        "source": "a lone samurai at dusk with a sword",
        "modifiers": ["🎲 Random Artist", "🎲 Random Setting"],
    },
    {
        "label": "dragon + Anime Ghibli + Random Setting",
        "source": "a dragon perched on a tower at sunset",
        "modifiers": ["Anime Ghibli", "🎲 Random Setting"],
    },
]


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    print("Opening anima_tagger stack (no models yet) …")
    t0 = time.perf_counter()
    stack = load_all()
    print(f"  opened in {time.perf_counter()-t0:.1f}s")

    bases = load_yaml("bases.yaml")
    prompts = load_yaml("prompts.yaml")
    tf_anima = load_yaml("tag-formats/anima.yaml")
    all_mods = load_modifiers()
    detailed_sp = assemble_base_prompt(bases, "Detailed")
    anima_tag_sp = tf_anima["system_prompt"]

    print("Loading bge-m3 + reranker …")
    t0 = time.perf_counter()
    with stack.models():
        print(f"  models up in {time.perf_counter()-t0:.1f}s")

        for sc in SCENARIOS:
            print("\n" + "=" * 80)
            print(f"SCENARIO: {sc['label']}")
            print(f"SOURCE:   {sc['source']}")
            print(f"MODIFIERS: {sc['modifiers']}")

            # 1. Shortlist from source
            t = time.perf_counter()
            sl = stack.build_shortlist(sc["source"])
            print(f"\nSHORTLIST ({time.perf_counter()-t:.2f}s)")
            print(f"  artists:    {sl.artists}")
            print(f"  characters: {sl.characters}")
            print(f"  series:     {sl.series}")

            # 2. Build prose-pass user message with modifier directives + shortlist
            style = build_style_prose(sc["modifiers"], all_mods)
            shortlist_frag = sl.as_system_prompt_fragment()
            user_msg = f"SOURCE PROMPT: {sc['source']}\n\n{style}"
            prose_sp = detailed_sp
            if shortlist_frag:
                prose_sp = prose_sp + "\n\n" + shortlist_frag
            t = time.perf_counter()
            prose = call_llm(prose_sp, user_msg, max_tokens=500)
            print(f"\nPROSE ({time.perf_counter()-t:.1f}s):")
            print("  " + prose.replace("\n", "\n  ")[:800])
            if len(prose) > 800:
                print("  …")

            # 3. Tag draft from rich prose
            t = time.perf_counter()
            draft = call_llm(anima_tag_sp, prose, max_tokens=400)
            print(f"\nDRAFT ({time.perf_counter()-t:.1f}s):")
            print("  " + draft)

            # 4. Validator + rule layer
            t = time.perf_counter()
            final = stack.tagger.tag_from_draft(draft, safety="safe")
            print(f"\nFINAL TAGS ({time.perf_counter()-t:.2f}s):")
            print("  " + ", ".join(final))

    print("\nModels unloaded; VRAM freed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
