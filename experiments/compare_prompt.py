"""Standalone prompt-construction harness for comparing extension output
with ad-hoc Ollama tests.

Mirrors what scripts/prompt_enhancer.py does in Prose mode (the path
z-image uses): assembles the system prompt via pe._assemble_system_prompt,
optionally appends motion/negative directives, builds the user message
with style modifiers and inline wildcards, calls Ollama with the same
options the extension uses.

Usage:
    python -m experiments.compare_prompt \\
        --source "girl in a meadow" \\
        --base Default \\
        --detail 0

    # With modifiers, motion, negative
    python -m experiments.compare_prompt \\
        --source "knight reading by candlelight" \\
        --base Default \\
        --modifiers "🎲 Random Artist,Dramatic" \\
        --motion --neg \\
        --seed 137 --temp 0.7

    # Just print the constructed SP (no LLM call)
    python -m experiments.compare_prompt \\
        --source "abandoned space station" --base Default --dry

The full system prompt and user message are printed before the LLM call
so you can compare with what Forge sends. The output is the actual
LLM response — with the same SP construction the extension uses.

This catches drift between standalone tests and extension behavior:
modifiers, motion, prepend, detail level, etc. all flow through the
real assembly functions.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from anima_tagger.scripts._pe_bootstrap import pe  # noqa: E402


def call_ollama(sp: str, user_msg: str, model: str, seed: int, temp: float,
                think: bool, num_predict: int = 1024,
                api_url: str = "http://127.0.0.1:11434") -> str:
    """Call Ollama with the same body the extension uses (stream:true,
    think configurable). Returns the assembled content."""
    body = {
        "model": model,
        "stream": True,
        "think": bool(think),
        "options": {
            "temperature": float(temp),
            "seed": int(seed),
            "num_predict": int(num_predict),
            "top_k": 20,
            "top_p": 0.95 if think else 0.8,
            "repeat_penalty": 1.5,
            "presence_penalty": 1.5,
        },
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": user_msg},
        ],
    }
    req = urllib.request.Request(
        api_url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            d = json.loads(line)
            msg = d.get("message", {})
            if msg.get("content"):
                chunks.append(msg["content"])
            if d.get("done"):
                break
    return "".join(chunks).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare extension prompt construction with ad-hoc tests.")
    ap.add_argument("--source", required=True, help="Source prompt string")
    ap.add_argument("--base", default="Default", help="Base name (e.g. Default, Detailed, Cinematic)")
    ap.add_argument("--detail", type=int, default=0, help="Detail level 0-3 (0=match base, 3=most detailed)")
    ap.add_argument("--custom-sp", default=None, help="Custom system prompt (overrides base)")
    ap.add_argument("--modifiers", default="", help="Comma-separated modifier names (e.g. 'Dramatic,🎲 Random Artist')")
    ap.add_argument("--motion", action="store_true", help="Append motion+audio directive (mirrors '+ Motion and Audio' checkbox)")
    ap.add_argument("--neg", action="store_true", help="Append negative-prompt directive (mirrors '+ Negative' checkbox)")
    ap.add_argument("--prepend", action="store_true", help="Prepend source to output (mirrors 'Prepend source' checkbox)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--think", action="store_true", help="Enable Ollama 'think' mode")
    ap.add_argument("--model", default="huihui_ai/qwen3.5-abliterated:9b")
    ap.add_argument("--api-url", default="http://127.0.0.1:11434")
    ap.add_argument("--dry", action="store_true", help="Print constructed SP + user_msg, do NOT call LLM")
    args = ap.parse_args()

    # Build modifiers list via pe._collect_modifiers — same path the extension uses
    mod_names = [m.strip() for m in args.modifiers.split(",") if m.strip()]
    if mod_names:
        # pe._collect_modifiers takes list-of-lists (one per dropdown)
        mods = pe._collect_modifiers([mod_names], seed=args.seed)
    else:
        mods = []

    # Assemble system prompt via the same function the extension calls
    sp = pe._assemble_system_prompt(args.base, args.custom_sp, args.detail)
    if not sp:
        print(f"[ERROR] No system prompt for base={args.base!r}. Available bases:")
        for name in pe._bases:
            if not name.startswith("_"):
                print(f"  - {name}")
        return 1

    # Append motion + negative directives mirroring _enhance/_hybrid
    if args.motion:
        sp = f"{sp}\n\n{pe._prompts.get('motion', '')}"
    if args.neg:
        sp = f"{sp}\n\n{pe._prompts.get('negative', '')}"

    # Build user message — source + style modifiers + inline wildcards
    source = args.source.strip()
    user_msg = f"SOURCE PROMPT: {source}" if source else pe._prompts.get("empty_source_signal", "")
    style_str = pe._build_style_string(mods)
    if style_str:
        user_msg = f"{user_msg}\n\n{style_str}"
    inline_text = pe._build_inline_wildcard_text(source)
    if inline_text:
        user_msg = f"{user_msg}\n\n{inline_text}"

    # Print the assembly so user can verify what gets sent
    print("=" * 78)
    print(f"BASE:       {args.base}")
    print(f"DETAIL:     {args.detail}")
    print(f"MODIFIERS:  {mod_names if mod_names else '(none)'}")
    print(f"MOTION:     {args.motion}")
    print(f"NEGATIVE:   {args.neg}")
    print(f"PREPEND:    {args.prepend}")
    print(f"SEED:       {args.seed}")
    print(f"TEMP:       {args.temp}")
    print(f"THINK:      {args.think}")
    print(f"MODEL:      {args.model}")
    print("=" * 78)
    print(f"\n--- SYSTEM PROMPT ({len(sp.split())} words) ---\n{sp}")
    print(f"\n--- USER MESSAGE ({len(user_msg.split())} words) ---\n{user_msg}")
    print()

    if args.dry:
        print("--- DRY RUN — no LLM call ---")
        return 0

    print("--- LLM OUTPUT ---")
    out = call_ollama(
        sp, user_msg,
        model=args.model, seed=args.seed, temp=args.temp,
        think=args.think, api_url=args.api_url,
    )
    if args.prepend and source:
        out = f"{source}\n\n{out}"
    print(out)
    print(f"\n--- {len(out.split())} words ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
