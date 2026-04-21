"""Interactive rating tool — walks through every trace in a variant's
output directory, displays the relevant slice (source, prose, draft,
final tags, shortlist, slot_fill trace), and prompts for rubric scores.

Saves ratings to `<trace_dir>/_ratings.json`. Resumable — skips traces
already rated unless `--rerate` passed. Note-taking baked in: the
rater can flag "rubric felt wrong" and those notes drive the next
rubric revision.

Usage (interactive):
    python -m experiments.rate --variant v1

Non-interactive (batch auto-rate by agent):
    python -m experiments.rate --variant v1 --agent  # reads each
    trace and the agent fills ratings non-interactively using its own
    judgment (this is how an LLM agent self-rates).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent

# Make experiments + src importable when invoked as module
sys.path.insert(0, str(_REPO_ROOT))


def _load_rubric() -> Dict[str, Any]:
    with (_EXP_DIR / "rubric.yaml").open() as f:
        return yaml.safe_load(f)


def _list_traces(variant: str) -> List[Path]:
    d = _REPO_ROOT / ".ai" / "experiments" / variant
    if not d.exists():
        raise SystemExit(f"no traces found at {d} — run the variant first")
    return sorted(p for p in d.glob("*.json") if p.name != "_index.json")


def _show_trace(trace: Dict[str, Any]) -> None:
    """Print the rater-relevant slice: source, modifiers, prose, draft,
    final tags, shortlist, slot_fill. Not the full trace — only what the
    rubric asks about."""
    print("=" * 78)
    print(f"run_id:     {trace.get('run_id')}")
    print(f"prompt_id:  {trace.get('prompt_id')}")
    print(f"seed:       {trace.get('seed')}")
    print(f"outcome:    {trace.get('outcome')}")
    print(f"source:     {trace.get('prompt_source')!r}")
    print(f"modifiers:  {trace.get('modifiers')}")
    sl = trace.get("shortlist") or {}
    if sl:
        print(f"shortlist:  a={sl.get('artists', [])[:3]}… c={sl.get('characters', [])[:3]}… s={sl.get('series', [])[:3]}…")
    prose = trace.get("prose") or ""
    print(f"\nprose ({len(prose.split())} words):")
    print("  " + prose.replace("\n", "\n  "))
    draft = trace.get("draft") or ""
    draft_tokens = [t.strip() for t in draft.split(",") if t.strip()]
    print(f"\ndraft ({len(draft_tokens)} tokens): {draft_tokens}")
    # slot_fill trace
    for rec in trace.get("steps", []):
        if rec["step"] == "slot_fill":
            sf = rec.get("outputs_written", {}).get("slot_fill_trace")
            if sf:
                print(f"\nslot_fill_trace: {sf}")
    print(f"\nFINAL TAGS:\n  {trace.get('final_tags')}")
    print()


def _prompt_rating(dims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Interactive prompt for rubric scores + notes."""
    print("Rate each dimension 1-5 (or s to skip, q to quit). Anchors above.")
    scores: Dict[str, int] = {}
    for dim in dims:
        while True:
            ans = input(f"  {dim['id']:15s} [1-5 / s / q]: ").strip().lower()
            if ans == "q":
                raise KeyboardInterrupt
            if ans == "s":
                break
            if ans in "12345":
                scores[dim["id"]] = int(ans)
                break
            print(f"    invalid; enter 1-5, s, or q")
    # Overall
    while True:
        ans = input(f"  {'overall':15s} [1-5 / s]: ").strip().lower()
        if ans == "s":
            break
        if ans in "12345":
            scores["overall"] = int(ans)
            break
        print(f"    invalid; enter 1-5 or s")
    notes = input("  notes (what failed, what surprised you, rubric gaps): ").strip()
    return {"scores": scores, "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--only", default="",
                    help="comma-separated prompt_ids to rate")
    ap.add_argument("--rerate", action="store_true",
                    help="re-rate runs that already have ratings")
    ap.add_argument("--show-only", action="store_true",
                    help="just print the traces, don't prompt for ratings")
    args = ap.parse_args()

    rubric = _load_rubric()
    trace_paths = _list_traces(args.variant)
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        trace_paths = [p for p in trace_paths
                       if any(o in p.stem for o in only)]

    ratings_path = _REPO_ROOT / ".ai" / "experiments" / args.variant / "_ratings.json"
    existing: Dict[str, Any] = {}
    if ratings_path.exists():
        with ratings_path.open() as f:
            existing = json.load(f)

    if args.show_only:
        for p in trace_paths:
            with p.open() as f:
                trace = json.load(f)
            _show_trace(trace)
        return 0

    # Print rubric anchors once at the top
    print(f"\nRubric v{rubric['version']}:")
    for dim in rubric["dimensions"]:
        print(f"  {dim['id']}: {dim['title']}")
        anchors = dim.get("anchors", {})
        for score_level in (1, 3, 5):
            if score_level in anchors:
                a = anchors[score_level].strip().replace("\n", " ")[:90]
                print(f"    {score_level}: {a}")
    print(f"\n  overall anchors:")
    for level, desc in rubric.get("overall_anchors", {}).items():
        print(f"    {level}: {desc}")

    print(f"\nRating {len(trace_paths)} trace(s) for variant={args.variant}")
    try:
        for p in trace_paths:
            run_id = p.stem
            if run_id in existing and not args.rerate:
                continue
            with p.open() as f:
                trace = json.load(f)
            _show_trace(trace)
            rating = _prompt_rating(rubric["dimensions"])
            rating["run_id"] = run_id
            rating["prompt_id"] = trace.get("prompt_id")
            rating["seed"] = trace.get("seed")
            existing[run_id] = rating
            # Persist after every rating so Ctrl-C doesn't lose work
            with ratings_path.open("w") as f:
                json.dump(existing, f, indent=2)
    except KeyboardInterrupt:
        print("\n\nRating interrupted. Progress saved.")

    print(f"\nRatings saved to {ratings_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
