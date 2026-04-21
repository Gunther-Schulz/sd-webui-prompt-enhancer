"""Experiment runner.

Executes one or more variants against a prompt set × seed grid.
Produces per-run JSON traces saved under `.ai/experiments/<variant>/`
so outputs can be inspected, re-rated, or diffed across variants
without re-running the LLM.

Usage:
    python -m experiments.runner --variant v1 --seeds 1000 1001 1002 \
        --prompts experiments/prompts/v1_smoke.yaml

    python -m experiments.runner --variant v1 --only miku,girl_ra

No rating here. Rating is a separate interactive tool
(`experiments/rate.py`, built after runner is validated).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent

# Make top-level `experiments` and `src` importable
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _load_variant(name: str):
    """Import experiments.variants.<name> and return its .build() Pipeline."""
    try:
        mod = importlib.import_module(f"experiments.variants.{name}")
    except ImportError as e:
        raise SystemExit(f"Unknown variant {name!r}: {e}") from e
    if not hasattr(mod, "build"):
        raise SystemExit(f"Variant {name!r} has no build() function")
    return mod.build


def _load_prompts(path: Path) -> List[Dict[str, Any]]:
    """Load a test prompt YAML. Each entry is {id, source, modifiers, note?}.

    Fail-loud on missing required fields — a mistyped prompt file
    produces a skipped run silently otherwise.
    """
    if not path.exists():
        raise SystemExit(f"prompts file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise SystemExit(f"prompts YAML must be a top-level list; got {type(data).__name__}")
    prompts = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise SystemExit(f"prompts[{i}] must be a dict")
        if "id" not in entry:
            raise SystemExit(f"prompts[{i}] missing 'id'")
        if "source" not in entry:
            raise SystemExit(f"prompts[{i}] (id={entry['id']}) missing 'source'")
        prompts.append({
            "id": str(entry["id"]),
            "source": str(entry["source"]),
            "modifiers": list(entry.get("modifiers", []) or []),
            "note": entry.get("note", ""),
        })
    return prompts


def _collect_modifiers(names: List[str], seed: int = -1):
    """Resolve modifier names via the shipped pe._collect_modifiers so
    source: mechanism and badge-stripping fire identically here and
    in Forge. Passing the seed lets db_pattern source entries resolve
    immediately (db_retrieve entries stay deferred — they need the
    anima stack which isn't up at collect-time).

    Fails if any name isn't registered — avoids silent "0 modifiers
    active" when a YAML typo names a non-existent modifier.
    """
    from experiments.steps.common import pe
    # Validate up-front
    for name in names:
        clean = pe._strip_mechanism_badges(name) if hasattr(pe, "_strip_mechanism_badges") else name
        if pe._all_modifiers.get(clean) is None:
            raise SystemExit(
                f"Unknown modifier: {name!r}. "
                f"Check for typos; available modifiers: {len(pe._all_modifiers)} registered."
            )
    # pe._collect_modifiers expects a list-of-lists (one inner list per
    # dropdown). The runner has a single flat list — wrap it.
    return pe._collect_modifiers([names], seed=seed)


def run_one(variant_build, variant_name: str, variant_params: Dict[str, Any],
            prompt: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Execute one (variant, prompt, seed) run. Returns the trace as dict."""
    pipeline = variant_build(variant_params)
    mods = _collect_modifiers(prompt["modifiers"], seed=seed)
    initial_state = {
        "source": prompt["source"],
        "mods": mods,
        "seed": seed,
    }
    run_id = f"{variant_name}__{prompt['id']}__seed{seed}__{uuid.uuid4().hex[:6]}"
    from experiments.pipeline import PipelineError
    try:
        final_state, trace = pipeline.run(initial_state, run_id=run_id, variant_name=variant_name)
    except PipelineError as e:
        # Fail-loud: record failure, but keep going across the grid so
        # one bad seed doesn't abort a whole run
        tdict = e.trace.to_dict()
        tdict["prompt_id"] = prompt["id"]
        tdict["seed"] = seed
        tdict["modifiers"] = prompt["modifiers"]
        tdict["prompt_source"] = prompt["source"]
        return tdict
    tdict = trace.to_dict()
    tdict["prompt_id"] = prompt["id"]
    tdict["seed"] = seed
    tdict["modifiers"] = prompt["modifiers"]
    tdict["prompt_source"] = prompt["source"]
    # Surface the final tag list at the top so the rater doesn't have
    # to dig into the last step's record
    tdict["final_tags"] = final_state.get("final_tags") or final_state.get("tags_after_validate")
    tdict["prose"] = final_state.get("prose")
    tdict["draft"] = final_state.get("draft")
    tdict["shortlist"] = {
        "artists": final_state["shortlist"].artists,
        "characters": final_state["shortlist"].characters,
        "series": final_state["shortlist"].series,
    } if final_state.get("shortlist") else None
    return tdict


def _out_dir(variant_name: str) -> Path:
    d = _REPO_ROOT / ".ai" / "experiments" / variant_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="variant module under experiments.variants")
    ap.add_argument("--prompts", default=None, help="path to prompts YAML")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000],
                    help="one or more seeds (space-separated)")
    ap.add_argument("--only", default="",
                    help="comma-separated prompt IDs to filter to")
    ap.add_argument("--param", action="append", default=[],
                    help="variant params as key=value (repeatable, e.g. --param model=qwen3:8b)")
    args = ap.parse_args()

    # Default prompts file
    if args.prompts:
        prompts_path = Path(args.prompts)
    else:
        prompts_path = _EXP_DIR / "prompts" / "v1_smoke.yaml"

    prompts = _load_prompts(prompts_path)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        prompts = [p for p in prompts if p["id"] in only]
        if not prompts:
            raise SystemExit(f"no prompts matched --only={args.only}")

    variant_params: Dict[str, Any] = {}
    for p in args.param:
        if "=" not in p:
            raise SystemExit(f"--param must be key=value; got {p!r}")
        k, v = p.split("=", 1)
        # Try to parse booleans and numbers; fall through to str
        if v.lower() in ("true", "false"):
            variant_params[k] = v.lower() == "true"
        else:
            try:
                variant_params[k] = int(v)
            except ValueError:
                try:
                    variant_params[k] = float(v)
                except ValueError:
                    variant_params[k] = v

    variant_build = _load_variant(args.variant)
    out_dir = _out_dir(args.variant)

    print(f"Running variant={args.variant}  prompts={len(prompts)}  seeds={args.seeds}  "
          f"total_runs={len(prompts)*len(args.seeds)}  out={out_dir}")

    all_runs: List[Dict[str, Any]] = []
    for prompt in prompts:
        for seed in args.seeds:
            print(f"\n── {prompt['id']} seed={seed} ── source={prompt['source']!r} mods={prompt['modifiers']}")
            trace = run_one(variant_build, args.variant, variant_params, prompt, seed)
            outcome = trace.get("outcome", "?")
            tags = trace.get("final_tags")
            print(f"  outcome={outcome}")
            if tags:
                print(f"  tags: {tags}")
            if outcome == "failed":
                # Show the failing step's error
                for rec in trace.get("steps", []):
                    if rec.get("error"):
                        print(f"  [{rec['step']}] FAILED:\n    {rec['error'].splitlines()[0]}")
                        break
            # Save per-run trace
            out_path = out_dir / f"{trace['run_id']}.json"
            with out_path.open("w") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
            all_runs.append({
                "run_id": trace["run_id"],
                "prompt_id": prompt["id"],
                "seed": seed,
                "outcome": outcome,
                "path": str(out_path.relative_to(_REPO_ROOT)),
            })

    # Write index
    index_path = out_dir / "_index.json"
    with index_path.open("w") as f:
        json.dump({"variant": args.variant, "params": variant_params, "runs": all_runs},
                  f, indent=2)
    print(f"\nSaved {len(all_runs)} traces. Index: {index_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
