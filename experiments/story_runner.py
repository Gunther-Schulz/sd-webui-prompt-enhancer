"""Story-mode experiment runner.

Loads a variant YAML + a seed story, runs the LLM passes via the
in-process llm_runner module, validates each output, saves the trace
to `.ai/experiments/story/<variant_id>/<run_id>.json`.

Uses the same llm_runner singleton the production extension uses, so
runs here are apples-to-apples with what Forge produces (CLAUDE.md
"no island tests"). Only needs the GGUF cached locally — auto-downloads
on first use.

Usage:
    # Run V1 against the canonical lighthouse seed
    python -m experiments.story_runner --variant V1_one_pass_yaml \\
        --seeds-file experiments/story-seeds.yaml \\
        --only lighthouse_6_linear

    # Run all variants against all seeds (the full A/B grid)
    for v in V1_one_pass_yaml V2_one_pass_json V3a_two_pass_yaml \\
             V3b_two_pass_json V5_terse_yaml; do
        python -m experiments.story_runner --variant $v \\
            --seeds-file experiments/story-seeds.yaml
    done

Override the model / quant / compute via CLI flags:
    --llm-repo / --llm-quant / --llm-model-path / --compute / --n-ctx

Fail-loud: schema-malformed LLM output, template-rendering failure,
or model-load error aborts the run with a visible trace entry. No
silent fallbacks (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments import story_validators as sv
from llm_runner import LLMRunner, TruncatedOutput
from llm_runner.config import config_override

# ── modifier resolution ─────────────────────────────────────────────────


def _load_story_modifiers() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load every YAML under experiments/story-modifiers/ AND modifiers/.
    Returns a flat lookup: name → {"behavioral": str, "keywords": str}.

    Names are matched case-sensitively against the modifier key. The
    `🎲 Random X` modifiers are not used in story mode (story-mode wants
    deterministic plans across the variant grid).
    """
    flat: Dict[str, Dict[str, str]] = {}
    sources = [_EXP_DIR / "story-modifiers", _REPO_ROOT / "modifiers"]
    for src in sources:
        if not src.exists():
            continue
        for yaml_path in sorted(src.glob("*.yaml")):
            with yaml_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            for category, mods in data.items():
                if category.startswith("_"):
                    continue
                if not isinstance(mods, dict):
                    continue
                for name, body in mods.items():
                    if not isinstance(body, dict):
                        continue
                    flat[name] = {
                        "behavioral": str(body.get("behavioral", "")).strip(),
                        "keywords": str(body.get("keywords", "")).strip(),
                    }
    return flat


def _resolve_modifiers(names: List[str], lookup: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """Return per-name {behavioral, keywords}. Fail-loud on unknown names."""
    out = []
    for name in names:
        if name not in lookup:
            available = sorted(lookup.keys())
            raise SystemExit(
                f"Unknown modifier {name!r}. "
                f"Did you mean one of: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
        out.append({"name": name, **lookup[name]})
    return out


def _resolve_story_meta(seed: Dict[str, Any], lookup: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Resolve length / story_mode entries to their behavioral text.
    Returns:
        {
          "length_n": <int — 6 panels → 6>,
          "length_behavioral": str,
          "story_mode_behavioral": str,
        }
    """
    length_name = seed["length"]
    if length_name not in lookup:
        raise SystemExit(
            f"Unknown story length {length_name!r}. "
            f"Edit experiments/story-modifiers/story-length.yaml to add it, "
            f"or fix the seed."
        )
    # Parse N from "<N> panels"
    m = re.match(r"^\s*(\d+)\s*panels?\s*$", length_name)
    if not m:
        raise SystemExit(
            f"length {length_name!r} doesn't match expected pattern "
            f"'<N> panels' — please rename to 'X panels' so the runner can "
            f"extract N."
        )
    length_n = int(m.group(1))

    mode_name = seed["story_mode"]
    if mode_name not in lookup:
        raise SystemExit(
            f"Unknown story mode {mode_name!r}. "
            f"Edit experiments/story-modifiers/story-mode.yaml to add it, "
            f"or fix the seed."
        )

    return {
        "length_n": length_n,
        "length_behavioral": lookup[length_name]["behavioral"],
        "story_mode_behavioral": lookup[mode_name]["behavioral"],
    }


# ── LLM client (in-process via llm_runner) ──────────────────────────────


def call_llm(runner: LLMRunner, system_prompt: str, user_prompt: str,
             temperature: float, num_predict: int, seed: int,
             stall_timeout_s: float = 60.0) -> Dict[str, Any]:
    """Make one generation against the in-process runner. Returns dict
    with keys: content, tokens, elapsed_ms, model, completed.

    `completed` is True iff the stream ended naturally; False on
    truncation/cancel/stall (in which case `content` is the partial).

    Fails loud on model-load errors. Truncation is captured in the
    return dict (not raised) so per-pass traces can record what
    happened without an exception unwinding the run.
    """
    t0 = time.monotonic()
    chunks = []
    last = None
    try:
        for chunk in runner.stream(
            user_prompt, system_prompt,
            temperature=temperature,
            seed=seed,
            num_predict=num_predict,
            stall_timeout_s=stall_timeout_s,
        ):
            chunks.append(chunk.text)
            last = chunk
    except TruncatedOutput as e:
        # Convert to in-band signal — we want the trace to record the
        # partial, not unwind the whole run.
        return {
            "content": e.partial,
            "tokens": 0,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "model": runner.config.model_ref,
            "completed": False,
            "truncation_reason": e.reason,
        }

    content = "".join(chunks)
    if not content.strip():
        raise RuntimeError(
            f"Runner returned empty content. tokens={last.tokens if last else 0}"
        )
    return {
        "content": content,
        "tokens": last.tokens if last else 0,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        "model": runner.config.model_ref,
        "completed": last.is_complete if last else False,
        "truncation_reason": None,
    }


# ── template rendering ──────────────────────────────────────────────────


def render(template: str, ctx: Dict[str, Any]) -> str:
    """Render a template using {key} substitution. Missing keys raise
    KeyError loudly (fail-loud — a typo in a template should NOT silently
    produce a partial prompt)."""
    # str.format-style with strict missing-key behavior
    class _StrictDict(dict):
        def __missing__(self, key):
            raise KeyError(
                f"template variable {key!r} not provided in context "
                f"(available: {sorted(self.keys())[:12]}...)"
            )
    return template.format_map(_StrictDict(ctx))


def _format_role_descriptions(roles: List[Dict[str, str]],
                              filter_ids: Optional[List[str]] = None) -> str:
    """Format roles as a multi-line block for prompt templates. If
    filter_ids is given, only include those roles; preserves the order
    from filter_ids."""
    if filter_ids is not None:
        rmap = {r["id"]: r for r in roles}
        roles = [rmap[i] for i in filter_ids if i in rmap]
    lines = []
    for r in roles:
        lines.append(f"- {r['id']}: {r['description'].strip()}")
    return "\n".join(lines) if lines else "(no roles in this panel)"


def _format_modifier_behaviorals(modifiers: List[Dict[str, str]]) -> str:
    if not modifiers:
        return "(none)"
    parts = [m["behavioral"] for m in modifiers if m.get("behavioral")]
    return " ".join(parts) if parts else "(none)"


# ── variant runners (one per shape) ─────────────────────────────────────


_LLM_META_KEYS = ("tokens", "elapsed_ms", "model", "completed", "truncation_reason")


def run_one_pass(variant: Dict[str, Any], seed: Dict[str, Any], meta: Dict[str, Any],
                 modifiers: List[Dict[str, str]], llm_seed: int,
                 runner: LLMRunner) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pass_cfg = variant["full_pass"]
    user_prompt = render(pass_cfg["user_prompt_template"], {
        "source": seed["source"],
        "length": seed["length"],
        "story_mode_behavioral": meta["story_mode_behavioral"],
        "modifier_behaviorals": _format_modifier_behaviorals(modifiers),
    })
    pass_record = {
        "pass": "full",
        "system_prompt": pass_cfg["system_prompt"],
        "user_prompt": user_prompt,
        "temperature": pass_cfg["temperature"],
        "num_predict": pass_cfg["num_predict"],
        "llm_seed": llm_seed,
    }
    try:
        resp = call_llm(
            runner, pass_cfg["system_prompt"], user_prompt,
            pass_cfg["temperature"], pass_cfg["num_predict"], llm_seed,
        )
    except Exception as e:
        pass_record["error"] = f"{type(e).__name__}: {e}"
        return None, [pass_record]
    pass_record["raw_output"] = resp["content"]
    pass_record["llm_meta"] = {k: resp.get(k) for k in _LLM_META_KEYS}

    validator = sv.get_validator(variant["validator"])
    try:
        plan = validator(resp["content"], expected_panel_count=meta["length_n"])
    except sv.ValidationError as e:
        pass_record["error"] = f"ValidationError: {e}"
        return None, [pass_record]
    pass_record["parsed_ok"] = True
    return plan, [pass_record]


def run_two_pass_per_panel(variant: Dict[str, Any], seed: Dict[str, Any],
                           meta: Dict[str, Any], modifiers: List[Dict[str, str]],
                           llm_seed: int, runner: LLMRunner
                           ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    # ── Pass 1: plan ───────────────────────────────────────────
    plan_cfg = variant["plan_pass"]
    plan_user = render(plan_cfg["user_prompt_template"], {
        "source": seed["source"],
        "length": seed["length"],
        "story_mode_behavioral": meta["story_mode_behavioral"],
        "modifier_behaviorals": _format_modifier_behaviorals(modifiers),
    })
    plan_rec = {
        "pass": "plan",
        "system_prompt": plan_cfg["system_prompt"],
        "user_prompt": plan_user,
        "temperature": plan_cfg["temperature"],
        "num_predict": plan_cfg["num_predict"],
        "llm_seed": llm_seed,
    }
    try:
        resp = call_llm(
            runner, plan_cfg["system_prompt"], plan_user,
            plan_cfg["temperature"], plan_cfg["num_predict"], llm_seed,
        )
    except Exception as e:
        plan_rec["error"] = f"{type(e).__name__}: {e}"
        records.append(plan_rec)
        return None, records
    plan_rec["raw_output"] = resp["content"]
    plan_rec["llm_meta"] = {k: resp.get(k) for k in _LLM_META_KEYS}

    plan_validator = sv.get_validator(variant["validator_plan"])
    try:
        plan = plan_validator(resp["content"], expected_panel_count=meta["length_n"])
    except sv.ValidationError as e:
        plan_rec["error"] = f"ValidationError: {e}"
        records.append(plan_rec)
        return None, records
    plan_rec["parsed_ok"] = True
    records.append(plan_rec)

    # ── Pass 2: per-panel ──────────────────────────────────────
    panel_cfg = variant["panel_pass"]
    panel_validator = sv.get_validator(variant["validator_panel"])
    for panel in plan["panels"]:
        panel_user = render(panel_cfg["user_prompt_template"], {
            "source": seed["source"],
            "style_anchor": plan["style_anchor"],
            "role_descriptions": _format_role_descriptions(
                plan["roles"], filter_ids=panel["roles_present"]
            ),
            "panel_n": panel["n"],
            "panel_mode": panel["mode"],
            "panel_caption": panel["caption"],
            "panel_roles_present": ", ".join(panel["roles_present"]) or "(none)",
            "panel_ref_panels": ", ".join(str(x) for x in panel["ref_panels"]) or "(none)",
        })
        # Derive a per-panel sub-seed so panel calls don't all sample the
        # same trajectory at the same temperature
        panel_llm_seed = (llm_seed + panel["n"] * 1000003) % (2**31)
        prec = {
            "pass": f"panel_{panel['n']}",
            "system_prompt": panel_cfg["system_prompt"],
            "user_prompt": panel_user,
            "temperature": panel_cfg["temperature"],
            "num_predict": panel_cfg["num_predict"],
            "llm_seed": panel_llm_seed,
        }
        try:
            resp = call_llm(
                runner, panel_cfg["system_prompt"], panel_user,
                panel_cfg["temperature"], panel_cfg["num_predict"], panel_llm_seed,
            )
        except Exception as e:
            prec["error"] = f"{type(e).__name__}: {e}"
            records.append(prec)
            return None, records
        prec["raw_output"] = resp["content"]
        prec["llm_meta"] = {k: resp.get(k) for k in _LLM_META_KEYS}
        try:
            panel_prompt = panel_validator(resp["content"])
        except sv.ValidationError as e:
            prec["error"] = f"ValidationError: {e}"
            records.append(prec)
            return None, records
        prec["parsed_ok"] = True
        records.append(prec)

        # Mutate plan: install the panel's prompt
        if panel["mode"] == "t2i":
            panel["t2i_prompt"] = panel_prompt
        else:
            panel["edit_prompt"] = panel_prompt

    return plan, records


def run_terse_one_pass(variant: Dict[str, Any], seed: Dict[str, Any],
                       meta: Dict[str, Any], modifiers: List[Dict[str, str]],
                       llm_seed: int, runner: LLMRunner
                       ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """V5: LLM produces terse plan; image prompts are template-assembled
    at the end (no second LLM call)."""
    plan, records = run_one_pass(variant, seed, meta, modifiers, llm_seed, runner)
    if plan is None:
        return None, records

    # Assemble prompts via templates
    t2i_tpl = variant.get("panel_prompt_template_t2i", "")
    edit_tpl = variant.get("panel_prompt_template_edit", "")
    if not t2i_tpl or not edit_tpl:
        records.append({
            "pass": "template_assembly",
            "error": "ValidationError: variant missing panel_prompt_template_t2i / "
                     "panel_prompt_template_edit (required for terse_one_pass)",
        })
        return None, records

    for panel in plan["panels"]:
        ctx = {
            "role_descriptions": _format_role_descriptions(
                plan["roles"], filter_ids=panel["roles_present"]
            ),
            "style_anchor": plan["style_anchor"],
            "caption": panel["caption"],
            "panel_n": panel["n"],
            "panel_ref_panels": ", ".join(str(x) for x in panel["ref_panels"]) or "(none)",
        }
        try:
            if panel["mode"] == "t2i":
                panel["t2i_prompt"] = render(t2i_tpl, ctx)
            else:
                panel["edit_prompt"] = render(edit_tpl, ctx)
        except KeyError as e:
            records.append({
                "pass": "template_assembly",
                "panel_n": panel["n"],
                "error": f"KeyError in template: {e}",
            })
            return None, records

    records.append({"pass": "template_assembly", "parsed_ok": True})
    return plan, records


SHAPE_RUNNERS = {
    "one_pass": run_one_pass,
    "two_pass_per_panel": run_two_pass_per_panel,
    "terse_one_pass": run_terse_one_pass,
}


# ── orchestration ───────────────────────────────────────────────────────


def run_one(variant: Dict[str, Any], seed: Dict[str, Any], llm_seed: int,
            llm_runner: LLMRunner) -> Dict[str, Any]:
    lookup = _load_story_modifiers()
    meta = _resolve_story_meta(seed, lookup)
    modifiers = _resolve_modifiers(seed.get("modifiers", []), lookup)

    shape = variant["shape"]
    shape_fn = SHAPE_RUNNERS.get(shape)
    if shape_fn is None:
        raise SystemExit(
            f"Unknown variant shape {shape!r}. Available: {sorted(SHAPE_RUNNERS.keys())}"
        )

    run_id = f"{variant['variant_id']}__{seed['id']}__seed{llm_seed}__{uuid.uuid4().hex[:6]}"
    print(f"\n── running {run_id}")
    print(f"   shape={shape} format={variant.get('format')} length_n={meta['length_n']}")
    print(f"   source: {seed['source']!r}")
    print(f"   modifiers: {[m['name'] for m in modifiers]}")

    t0 = time.monotonic()
    try:
        plan, records = shape_fn(variant, seed, meta, modifiers, llm_seed, llm_runner)
    except Exception as e:
        # Unexpected runner-level crash (not a per-pass error). Capture
        # and continue with a clear failure record.
        plan = None
        records = [{
            "pass": "runner_crash",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }]
    elapsed = time.monotonic() - t0

    outcome = "ok" if plan is not None else "failed"
    print(f"   outcome={outcome}  elapsed={elapsed:.1f}s  passes={len(records)}")
    if outcome == "failed":
        for rec in records:
            if rec.get("error"):
                first = rec["error"].splitlines()[0]
                print(f"   [{rec['pass']}] {first}")
                break

    return {
        "run_id": run_id,
        "variant_id": variant["variant_id"],
        "variant_shape": shape,
        "variant_format": variant.get("format"),
        "seed_id": seed["id"],
        "seed_source": seed["source"],
        "seed_length": seed["length"],
        "seed_story_mode": seed["story_mode"],
        "seed_modifiers": [m["name"] for m in modifiers],
        "llm_seed": llm_seed,
        "outcome": outcome,
        "elapsed_s": round(elapsed, 2),
        "passes": records,
        "plan": plan,  # None if any pass failed
    }


def _load_variant(name: str) -> Dict[str, Any]:
    path = _EXP_DIR / "story-variants" / f"{name}.yaml"
    if not path.exists():
        avail = sorted(p.stem for p in (_EXP_DIR / "story-variants").glob("*.yaml"))
        raise SystemExit(f"Unknown variant {name!r}. Available: {avail}")
    with path.open() as f:
        v = yaml.safe_load(f)
    if not isinstance(v, dict) or "variant_id" not in v:
        raise SystemExit(f"variant {name!r} is malformed (no variant_id)")
    return v


def _load_seeds(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise SystemExit(f"seeds file must be a top-level list; got {type(data).__name__}")
    for i, s in enumerate(data):
        for required in ("id", "source", "length", "story_mode"):
            if required not in s:
                raise SystemExit(f"seeds[{i}] missing {required!r}")
    return data


def _out_dir(variant_id: str) -> Path:
    d = _REPO_ROOT / ".ai" / "experiments" / "story" / variant_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="variant id (filename stem under experiments/story-variants/)")
    ap.add_argument("--seeds-file", default=str(_EXP_DIR / "story-seeds.yaml"))
    ap.add_argument("--only", default="",
                    help="comma-separated seed ids to filter to")
    ap.add_argument("--llm-seeds", type=int, nargs="+", default=[42],
                    help="one or more LLM seeds (default: [42])")
    # In-process LLM controls (overrides defaults from llm_runner.config)
    ap.add_argument("--llm-repo", default=None,
                    help="HF repo id for the GGUF (default: from llm_runner.config)")
    ap.add_argument("--llm-quant", default=None,
                    help="quantization (e.g. Q4_K_M, Q5_K_M, Q6_K)")
    ap.add_argument("--llm-model-path", default=None,
                    help="absolute path to a local GGUF (overrides --llm-repo)")
    ap.add_argument("--compute", choices=("gpu", "cpu", "shared"), default=None,
                    help="compute target")
    ap.add_argument("--n-gpu-layers", type=int, default=None,
                    help="number of GPU layers when --compute=shared")
    ap.add_argument("--n-ctx", type=int, default=None, help="context size")
    ap.add_argument("--no-dry", action="store_true",
                    help="disable low-level DRY sampler (use high-level samplers only)")
    args = ap.parse_args()

    variant = _load_variant(args.variant)
    seeds = _load_seeds(Path(args.seeds_file))
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        seeds = [s for s in seeds if s["id"] in only]
        if not seeds:
            raise SystemExit(f"No seeds matched --only={args.only!r}")

    # Build the runner config from CLI overrides (only set what's specified;
    # the rest comes from defaults in llm_runner.config).
    cfg_overrides = {}
    if args.llm_repo is not None:        cfg_overrides["repo_id"] = args.llm_repo
    if args.llm_quant is not None:       cfg_overrides["quant"] = args.llm_quant
    if args.llm_model_path is not None:  cfg_overrides["model_path"] = args.llm_model_path
    if args.compute is not None:         cfg_overrides["compute"] = args.compute
    if args.n_gpu_layers is not None:    cfg_overrides["n_gpu_layers"] = args.n_gpu_layers
    if args.n_ctx is not None:           cfg_overrides["n_ctx"] = args.n_ctx
    if args.no_dry:                      cfg_overrides["use_low_level_dry"] = False
    runner_cfg = config_override(**cfg_overrides)
    runner = LLMRunner(runner_cfg)

    out_dir = _out_dir(variant["variant_id"])
    print(f"Variant: {variant['variant_id']}  Seeds: {len(seeds)}  LLM seeds: {args.llm_seeds}")
    print(f"Total runs: {len(seeds) * len(args.llm_seeds)}  Out: {out_dir}")
    print(f"Model: {runner_cfg.model_ref}  compute={runner_cfg.compute}  "
          f"n_ctx={runner_cfg.n_ctx}  dry={runner_cfg.use_low_level_dry}")

    all_runs = []
    try:
        for seed in seeds:
            for llm_seed in args.llm_seeds:
                trace = run_one(variant, seed, llm_seed, runner)
                out_path = out_dir / f"{trace['run_id']}.json"
                with out_path.open("w") as f:
                    json.dump(trace, f, indent=2, ensure_ascii=False)
                all_runs.append({
                    "run_id": trace["run_id"],
                    "seed_id": seed["id"],
                    "llm_seed": llm_seed,
                    "outcome": trace["outcome"],
                    "elapsed_s": trace["elapsed_s"],
                    "path": str(out_path.relative_to(_REPO_ROOT)),
                })
    finally:
        # Always release the model when done so subsequent runs / image
        # generation get the VRAM back.
        runner.unload()

    index_path = out_dir / "_index.json"
    with index_path.open("w") as f:
        json.dump({
            "variant_id": variant["variant_id"],
            "model": runner_cfg.model_ref,
            "runs": all_runs,
        }, f, indent=2)

    n_ok = sum(1 for r in all_runs if r["outcome"] == "ok")
    print(f"\nDone. {n_ok}/{len(all_runs)} OK. Index: {index_path.relative_to(_REPO_ROOT)}")
    return 0 if n_ok == len(all_runs) else 1


if __name__ == "__main__":
    sys.exit(main())
