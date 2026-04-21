"""Load all runs for a variant, print side-by-side view per prompt +
seed so the rater can quickly score against the rubric.

Usage:
    python experiments/rate_variant.py v11 [v8]
      — v11: variant to review
      — v8:  optional reference variant; shown alongside for each
             (prompt, seed) so the rater can A/B deltas at a glance.

Prints to stdout. The rater writes scores into a companion JSON
(currently by hand — this script just assembles the view).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent


def load_runs(variant: str) -> dict:
    """Return {prompt_id: {seed: trace_dict}} for one variant."""
    out: dict = defaultdict(dict)
    for p in sorted((_REPO / ".ai" / "experiments" / variant).glob(f"{variant}__*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        pid = d.get("prompt_id")
        seed = d.get("seed")
        if pid and seed is not None:
            out[pid][seed] = d
    return dict(out)


def fmt_tags(t: str, w: int = 100) -> str:
    """Wrap a long comma-separated tag line to width w."""
    if not t:
        return "(empty)"
    out, line = [], ""
    for tok in (x.strip() for x in t.split(",")):
        if not tok:
            continue
        piece = f"{tok}, "
        if len(line) + len(piece) > w:
            out.append(line.rstrip())
            line = piece
        else:
            line += piece
    if line:
        out.append(line.rstrip(", "))
    return "\n    ".join(out)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    variant = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None

    runs = load_runs(variant)
    ref_runs = load_runs(ref) if ref else {}

    if not runs:
        print(f"No runs found for variant {variant!r} in .ai/experiments/{variant}/")
        sys.exit(1)

    print(f"=" * 80)
    print(f"Variant: {variant}  |  prompts: {len(runs)}  |  runs: {sum(len(s) for s in runs.values())}")
    if ref:
        print(f"Reference: {ref}  |  prompts: {len(ref_runs)}  |  runs: {sum(len(s) for s in ref_runs.values())}")
    print("=" * 80)

    for prompt_id in sorted(runs.keys()):
        seeds = sorted(runs[prompt_id].keys())
        print(f"\n\n{'#' * 80}")
        print(f"# prompt: {prompt_id}")
        trace0 = runs[prompt_id][seeds[0]]
        print(f"# source: {trace0.get('prompt_source','')!r}")
        mods = trace0.get("modifiers", [])
        if mods:
            print(f"# modifiers: {mods}")
        print("#" * 80)

        for seed in seeds:
            trace = runs[prompt_id][seed]
            print(f"\n── seed={seed} ──")
            print(f"  {variant}: {fmt_tags(trace.get('final_tags',''))}")
            if ref_runs.get(prompt_id, {}).get(seed):
                ref_trace = ref_runs[prompt_id][seed]
                print(f"  {ref}: {fmt_tags(ref_trace.get('final_tags',''))}")


if __name__ == "__main__":
    main()
