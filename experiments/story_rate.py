"""Interactive rater for story-mode traces.

Walks every trace under `.ai/experiments/story/<variant>/`, displays
the relevant slice (source, plan, per-panel prompts, error trace if
any), and prompts for rubric scores from `experiments/story-rubric.yaml`.

Saves ratings to `.ai/experiments/story/<variant>/_ratings.json`.
Resumable: skips already-rated runs unless --rerate.

Usage:
    python -m experiments.story_rate --variant V1_one_pass_yaml
    python -m experiments.story_rate --variant V3a_two_pass_yaml --only lighthouse_6_linear
    python -m experiments.story_rate --variant V1_one_pass_yaml --show-only

After rating across variants, a separate (TODO: write) summary script
can aggregate _ratings.json files and produce a per-variant report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_rubric() -> Dict[str, Any]:
    with (_EXP_DIR / "story-rubric.yaml").open() as f:
        return yaml.safe_load(f)


def _list_traces(variant: str) -> List[Path]:
    d = _REPO_ROOT / ".ai" / "experiments" / "story" / variant
    if not d.exists():
        raise SystemExit(
            f"No traces at {d}. Run the variant first:\n"
            f"  python -m experiments.story_runner --variant {variant}"
        )
    return sorted(p for p in d.glob("*.json") if p.name != "_index.json")


def _print_panels(plan: Dict[str, Any]) -> None:
    print(f"\nstyle_anchor: {plan.get('style_anchor', '')!r}")
    print(f"\nroles ({len(plan.get('roles', []))}):")
    for r in plan.get("roles", []):
        desc = r.get("description", "").strip().replace("\n", " ")
        print(f"  - {r.get('id')}: {desc[:200]}{'...' if len(desc) > 200 else ''}")
    print(f"\npanels ({len(plan.get('panels', []))}):")
    for p in plan.get("panels", []):
        print(f"  Panel {p.get('n')} [{p.get('mode')}] roles={p.get('roles_present')} "
              f"refs={p.get('ref_panels')}")
        print(f"    caption: {p.get('caption', '')!r}")
        prompt = p.get("t2i_prompt") or p.get("edit_prompt") or "(no prompt — terse plan or pass 2 not run)"
        prompt_lines = prompt.strip().split("\n")
        for line in prompt_lines[:8]:
            print(f"    > {line}")
        if len(prompt_lines) > 8:
            print(f"    > … ({len(prompt_lines) - 8} more lines)")


def _show_trace(trace: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"run_id:        {trace.get('run_id')}")
    print(f"variant:       {trace.get('variant_id')} ({trace.get('variant_shape')}, {trace.get('variant_format')})")
    print(f"seed:          {trace.get('seed_id')}")
    print(f"source:        {trace.get('seed_source')!r}")
    print(f"length:        {trace.get('seed_length')}  mode: {trace.get('seed_story_mode')}")
    print(f"modifiers:     {trace.get('seed_modifiers')}")
    print(f"llm_seed:      {trace.get('llm_seed')}")
    print(f"outcome:       {trace.get('outcome')}  elapsed: {trace.get('elapsed_s')}s")

    if trace.get("outcome") != "ok":
        print(f"\n--- FAILURE ---")
        for rec in trace.get("passes", []):
            if rec.get("error"):
                print(f"\n[{rec.get('pass')}] {rec['error']}")
                if rec.get("raw_output"):
                    print(f"\nRaw LLM output (first 1500 chars):")
                    print(rec["raw_output"][:1500])
                if rec.get("traceback"):
                    print(f"\nPython traceback:\n{rec['traceback']}")
                break
    else:
        plan = trace.get("plan") or {}
        _print_panels(plan)

    print()


def _prompt_rating(rubric: Dict[str, Any], trace_failed: bool) -> Dict[str, Any]:
    """Interactive rubric prompt. If the trace failed at the schema layer,
    auto-records 1 for schema_validity and asks only for notes (other
    dimensions are unrateable on a failed plan)."""
    print("Rate each dimension 1-5 (or 's' to skip, 'q' to quit). Anchors above.")
    scores: Dict[str, int] = {}

    if trace_failed:
        scores["schema_validity"] = 1
        print("  (schema_validity auto-set to 1 — trace failed at validation)")
        notes = input("  notes (what failed, root cause guess): ").strip()
        return {"scores": scores, "notes": notes}

    for dim in rubric["dimensions"]:
        while True:
            ans = input(f"  {dim['id']:24s} [1-5 / s / q]: ").strip().lower()
            if ans == "q":
                raise KeyboardInterrupt
            if ans == "s":
                break
            if ans in "12345":
                scores[dim["id"]] = int(ans)
                break
            print(f"    invalid; enter 1-5, s, or q")
    while True:
        ans = input(f"  {'overall':24s} [1-5 / s]: ").strip().lower()
        if ans == "s":
            break
        if ans in "12345":
            scores["overall"] = int(ans)
            break
        print(f"    invalid; enter 1-5 or s")
    notes = input("  notes (what failed, what surprised you, rubric gaps): ").strip()
    return {"scores": scores, "notes": notes}


def _print_rubric(rubric: Dict[str, Any]) -> None:
    print(f"\nRubric v{rubric['version']}  ({rubric.get('target', '')})\n")
    for dim in rubric["dimensions"]:
        print(f"  {dim['id']}: {dim['title']}")
        if dim.get("note"):
            print(f"    note: {dim['note']}")
        anchors = dim.get("anchors", {})
        for level in (1, 3, 5):
            if level in anchors:
                a = anchors[level].strip().replace("\n", " ")
                print(f"    {level}: {a[:120]}{'...' if len(a) > 120 else ''}")
        print()
    if rubric.get("overall_anchors"):
        print("  overall:")
        for level, desc in rubric["overall_anchors"].items():
            print(f"    {level}: {desc}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--only", default="",
                    help="comma-separated seed_ids to rate")
    ap.add_argument("--rerate", action="store_true",
                    help="re-rate runs that already have ratings")
    ap.add_argument("--show-only", action="store_true",
                    help="just print traces, don't prompt for ratings")
    args = ap.parse_args()

    rubric = _load_rubric()
    trace_paths = _list_traces(args.variant)
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        trace_paths = [p for p in trace_paths if any(o in p.stem for o in only)]

    ratings_path = _REPO_ROOT / ".ai" / "experiments" / "story" / args.variant / "_ratings.json"
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

    _print_rubric(rubric)

    print(f"Rating {len(trace_paths)} trace(s) for variant={args.variant}\n")
    try:
        for p in trace_paths:
            run_id = p.stem
            if run_id in existing and not args.rerate:
                continue
            with p.open() as f:
                trace = json.load(f)
            _show_trace(trace)
            rating = _prompt_rating(rubric, trace_failed=(trace.get("outcome") != "ok"))
            rating["run_id"] = run_id
            rating["seed_id"] = trace.get("seed_id")
            rating["llm_seed"] = trace.get("llm_seed")
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
