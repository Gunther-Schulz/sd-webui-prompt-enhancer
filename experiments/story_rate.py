"""Interactive rater for story-mode traces.

The story-mode rating flow puts the human LAST in the loop:

    1. Agent rates first (Claude Code reads traces directly per
       experiments/AGENT_RATER_PROMPT.md → _ratings_agent.json)
    2. Human reviews the agent's ratings via --review-agent mode below.
       Hit Enter to accept the agent's score for a dimension; type
       1-5 to override; 's' to skip; 'q' to quit.
    3. summarize compares both rating files; agent → human disagreements
       become the rubric-revision conversation.

Final ship/no-ship decision is the human's. The agent does the tedious
per-trace reading; the human signs off.

You can also rate from scratch (without seeing agent scores) via the
plain --variant invocation — useful if you want maximum independence
on a sample, then switch to --review-agent for the rest.

Usage:
    # Review the agent's ratings (recommended workflow)
    python -m experiments.story_rate --variant V1_one_pass_yaml --review-agent

    # Rate from scratch (no agent defaults)
    python -m experiments.story_rate --variant V1_one_pass_yaml

    # Just look at traces, don't rate
    python -m experiments.story_rate --variant V1_one_pass_yaml --show-only
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


def _prompt_rating(rubric: Dict[str, Any], trace_failed: bool,
                   agent_rating: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Interactive rubric prompt.

    If the trace failed at the schema layer, auto-records 1 for
    schema_validity and asks only for notes (other dimensions are
    unrateable on a failed plan).

    If agent_rating is provided (--review-agent mode), shows the
    agent's score per dimension as a default. Enter accepts the
    agent's score; 1-5 overrides; 's' skips; 'q' quits.
    """
    review_mode = agent_rating is not None
    if review_mode:
        print("Review agent's ratings: Enter to accept, 1-5 to override, 's' skip, 'q' quit.")
    else:
        print("Rate each dimension 1-5 (or 's' to skip, 'q' to quit). Anchors above.")
    scores: Dict[str, int] = {}

    if trace_failed:
        scores["schema_validity"] = 1
        print("  (schema_validity auto-set to 1 — trace failed at validation)")
        if review_mode and agent_rating:
            agent_notes = (agent_rating.get("notes") or "").strip()
            if agent_notes:
                print(f"  agent's notes: {agent_notes}")
        notes = input("  notes (what failed, root cause guess): ").strip()
        return {"scores": scores, "notes": notes}

    agent_scores = (agent_rating or {}).get("scores", {})

    def _ask(dim_id: str, allow_skip: bool = True) -> Optional[int]:
        agent_score = agent_scores.get(dim_id) if review_mode else None
        if agent_score is not None:
            label = f"  {dim_id:24s} [agent: {agent_score}] [Enter accept / 1-5 override / s / q]: "
        else:
            extras = " / s" if allow_skip else ""
            label = f"  {dim_id:24s} [1-5{extras} / q]: "
        while True:
            ans = input(label).strip().lower()
            if ans == "q":
                raise KeyboardInterrupt
            if ans == "" and agent_score is not None:
                return int(agent_score)
            if ans == "s" and allow_skip:
                return None
            if ans in "12345":
                return int(ans)
            print(f"    invalid input; try again")

    for dim in rubric["dimensions"]:
        score = _ask(dim["id"])
        if score is not None:
            scores[dim["id"]] = score
    overall = _ask("overall")
    if overall is not None:
        scores["overall"] = overall

    if review_mode and agent_rating:
        agent_notes = (agent_rating.get("notes") or "").strip()
        if agent_notes:
            print(f"  agent's notes: {agent_notes}")
    notes = input("  notes (what disagreed with agent, rubric gaps, surprises): ").strip()
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
    ap.add_argument("--review-agent", action="store_true",
                    help="show the agent's _ratings_agent.json scores as defaults; "
                         "Enter accepts, 1-5 overrides. Recommended workflow.")
    args = ap.parse_args()

    rubric = _load_rubric()
    trace_paths = _list_traces(args.variant)
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        trace_paths = [p for p in trace_paths if any(o in p.stem for o in only)]

    variant_dir = _REPO_ROOT / ".ai" / "experiments" / "story" / args.variant
    ratings_path = variant_dir / "_ratings.json"
    existing: Dict[str, Any] = {}
    if ratings_path.exists():
        with ratings_path.open() as f:
            existing = json.load(f)

    agent_ratings: Dict[str, Any] = {}
    if args.review_agent:
        agent_path = variant_dir / "_ratings_agent.json"
        if not agent_path.exists():
            raise SystemExit(
                f"--review-agent requires {agent_path.relative_to(_REPO_ROOT)} "
                f"to exist. Run the agent rating pass first per "
                f"experiments/AGENT_RATER_PROMPT.md."
            )
        with agent_path.open() as f:
            agent_ratings = json.load(f)
        print(f"Loaded {len(agent_ratings)} agent rating(s) — Enter accepts agent score, 1-5 overrides.\n")

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
            agent_rating = agent_ratings.get(run_id) if args.review_agent else None
            if args.review_agent and agent_rating is None:
                print(f"  (no agent rating found for this run — rating from scratch)")
            rating = _prompt_rating(
                rubric,
                trace_failed=(trace.get("outcome") != "ok"),
                agent_rating=agent_rating,
            )
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
