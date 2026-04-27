"""Cross-variant summary of story-mode test runs.

Reads all traces under `.ai/experiments/story/<variant>/` plus any
ratings files (`_ratings.json` from human rating via story_rate.py,
`_ratings_agent.json` from Claude Code agent rating per
experiments/AGENT_RATER_PROMPT.md), and prints comparison tables.

Two tables:
  1. Per-variant headline: parse rate, mean wall time, mean tokens,
     mean LLM-call count, mean overall rating (human + agent if both
     present).
  2. Per-dimension mean ratings: each rubric dim × each variant, so
     you can see which variant wins on which dim. Lets you choose a
     winner that trades off quality vs. cost differently per use case.

Optional: per-length breakdown shows where each variant's parse rate
collapses (the length threshold).

Usage:
    # All variants found under .ai/experiments/story/
    python -m experiments.story_summarize

    # Specific variants
    python -m experiments.story_summarize --variants V1_one_pass_yaml V3a_two_pass_yaml

    # Markdown output (paste into a doc / commit message / GitHub)
    python -m experiments.story_summarize --md

    # Per-length breakdown of parse rate
    python -m experiments.story_summarize --by-length
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import yaml

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
_TRACES_ROOT = _REPO_ROOT / ".ai" / "experiments" / "story"


def _list_variants() -> List[str]:
    if not _TRACES_ROOT.exists():
        return []
    return sorted(d.name for d in _TRACES_ROOT.iterdir() if d.is_dir())


def _load_traces(variant: str) -> List[Dict[str, Any]]:
    d = _TRACES_ROOT / variant
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        with p.open() as f:
            out.append(json.load(f))
    return out


def _load_ratings(variant: str, kind: str) -> Dict[str, Any]:
    """kind: 'human' (=_ratings.json) | 'agent' (=_ratings_agent.json)."""
    fn = "_ratings.json" if kind == "human" else "_ratings_agent.json"
    p = _TRACES_ROOT / variant / fn
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _trace_total_calls(trace: Dict[str, Any]) -> int:
    """Number of LLM passes (excluding template_assembly which isn't an LLM call)."""
    return sum(
        1 for r in trace.get("passes", [])
        if r.get("pass") not in ("template_assembly", "runner_crash")
    )


def _trace_total_tokens(trace: Dict[str, Any]) -> int:
    total = 0
    for r in trace.get("passes", []):
        meta = r.get("llm_meta") or {}
        total += int(meta.get("eval_count") or 0)
        total += int(meta.get("prompt_eval_count") or 0)
    return total


def _length_n_from_trace(trace: Dict[str, Any]) -> Optional[int]:
    sl = trace.get("seed_length", "")
    m = re.match(r"^\s*(\d+)\s*panels?\s*$", sl)
    return int(m.group(1)) if m else None


def _load_rubric_dims() -> List[str]:
    """Read rubric dimension ids in canonical order (for the per-dim table)."""
    p = _EXP_DIR / "story-rubric.yaml"
    if not p.exists():
        return []
    with p.open() as f:
        rb = yaml.safe_load(f)
    return [d["id"] for d in rb.get("dimensions", [])]


# ── per-variant aggregates ─────────────────────────────────────────────


def _summarize_variant(variant: str) -> Dict[str, Any]:
    traces = _load_traces(variant)
    ratings_human = _load_ratings(variant, "human")
    ratings_agent = _load_ratings(variant, "agent")

    n_total = len(traces)
    n_ok = sum(1 for t in traces if t.get("outcome") == "ok")
    wall_times = [t.get("elapsed_s") for t in traces if t.get("elapsed_s") is not None]
    tokens = [_trace_total_tokens(t) for t in traces if t.get("outcome") == "ok"]
    calls = [_trace_total_calls(t) for t in traces]

    # Mean ratings per dim for each rater
    def _mean_per_dim(ratings: Dict[str, Any]) -> Dict[str, float]:
        per_dim: Dict[str, List[int]] = {}
        for run_id, r in ratings.items():
            for dim_id, score in (r.get("scores") or {}).items():
                per_dim.setdefault(dim_id, []).append(int(score))
        return {d: round(mean(v), 2) for d, v in per_dim.items() if v}

    return {
        "variant": variant,
        "n_total": n_total,
        "n_ok": n_ok,
        "parse_rate": (n_ok / n_total) if n_total else 0.0,
        "mean_wall_s": round(mean(wall_times), 1) if wall_times else None,
        "mean_tokens": round(mean(tokens), 0) if tokens else None,
        "mean_calls": round(mean(calls), 1) if calls else None,
        "human": _mean_per_dim(ratings_human),
        "agent": _mean_per_dim(ratings_agent),
        "n_human_ratings": len(ratings_human),
        "n_agent_ratings": len(ratings_agent),
    }


# ── headline table ─────────────────────────────────────────────────────


def _format_headline(rows: List[Dict[str, Any]], md: bool) -> str:
    headers = ["variant", "parsed", "wall(s)", "tokens", "calls", "overall(h)", "overall(a)", "ratings(h/a)"]
    out = []
    for r in rows:
        overall_h = r["human"].get("overall")
        overall_a = r["agent"].get("overall")
        out.append([
            r["variant"],
            f"{r['n_ok']}/{r['n_total']}",
            f"{r['mean_wall_s']:.1f}" if r['mean_wall_s'] is not None else "-",
            f"{int(r['mean_tokens'])}" if r['mean_tokens'] is not None else "-",
            f"{r['mean_calls']:.1f}" if r['mean_calls'] is not None else "-",
            f"{overall_h:.2f}" if overall_h is not None else "-",
            f"{overall_a:.2f}" if overall_a is not None else "-",
            f"{r['n_human_ratings']}/{r['n_agent_ratings']}",
        ])
    return _render_table(headers, out, md)


# ── per-dimension table ────────────────────────────────────────────────


def _format_per_dim(rows: List[Dict[str, Any]], dims: List[str], md: bool,
                    rater: str = "human") -> str:
    """Column per variant, row per rubric dim. Cell = mean score."""
    if not dims:
        return ""
    if not any(r[rater] for r in rows):
        return f"(no {rater} ratings yet)"
    headers = ["dimension"] + [r["variant"] for r in rows]
    table_rows = []
    for dim in dims + ["overall"]:
        row = [dim]
        for r in rows:
            score = r[rater].get(dim)
            row.append(f"{score:.2f}" if score is not None else "-")
        table_rows.append(row)
    return _render_table(headers, table_rows, md)


# ── per-length breakdown ───────────────────────────────────────────────


def _format_by_length(rows: List[Dict[str, Any]], variants: List[str], md: bool) -> str:
    """Parse rate broken out by panel count."""
    # Group: per (variant, length) → (n_ok, n_total)
    grid: Dict[Tuple[str, int], Tuple[int, int]] = {}
    lengths_seen: set = set()
    for variant in variants:
        traces = _load_traces(variant)
        for t in traces:
            n = _length_n_from_trace(t)
            if n is None:
                continue
            lengths_seen.add(n)
            ok, total = grid.get((variant, n), (0, 0))
            grid[(variant, n)] = (ok + (1 if t.get("outcome") == "ok" else 0), total + 1)

    headers = ["length"] + variants
    table_rows = []
    for n in sorted(lengths_seen):
        row = [str(n)]
        for variant in variants:
            ok, total = grid.get((variant, n), (0, 0))
            row.append(f"{ok}/{total}" if total else "-")
        table_rows.append(row)
    return _render_table(headers, table_rows, md)


# ── shared table renderer ──────────────────────────────────────────────


def _render_table(headers: List[str], rows: List[List[str]], md: bool) -> str:
    if md:
        sep = "|" + "|".join("---" for _ in headers) + "|"
        lines = ["|" + "|".join(headers) + "|", sep]
        for row in rows:
            lines.append("|" + "|".join(row) + "|")
        return "\n".join(lines)
    # Fixed-width terminal table
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variant ids to summarize (default: all under .ai/experiments/story/)")
    ap.add_argument("--md", action="store_true",
                    help="output markdown tables (paste into commit messages / docs)")
    ap.add_argument("--by-length", action="store_true",
                    help="also print per-length parse-rate breakdown")
    args = ap.parse_args()

    available = _list_variants()
    if not available:
        print(f"No traces under {_TRACES_ROOT}. Run story_runner.py first.")
        return 1

    variants = args.variants or available
    missing = [v for v in variants if v not in available]
    if missing:
        print(f"Variants not found in traces: {missing}")
        print(f"Available: {available}")
        return 1

    rows = [_summarize_variant(v) for v in variants]
    dims = _load_rubric_dims()

    sep = "" if args.md else "\n" + "=" * 78 + "\n"

    print(sep + "Story-mode variant headline")
    print(sep)
    print(_format_headline(rows, args.md))

    print(sep + "Per-dimension means — human ratings")
    print(sep)
    print(_format_per_dim(rows, dims, args.md, rater="human"))

    print(sep + "Per-dimension means — agent ratings")
    print(sep)
    print(_format_per_dim(rows, dims, args.md, rater="agent"))

    if args.by_length:
        print(sep + "Parse rate by panel count")
        print(sep)
        print(_format_by_length(rows, variants, args.md))

    # Footer notes
    if not args.md:
        print()
        print("Legend:")
        print("  parsed:     # of runs whose plan parsed and validated")
        print("  wall(s):    mean wall-clock seconds per run")
        print("  tokens:     mean total tokens (prompt + output) per run")
        print("  calls:      mean LLM calls per run (1 for one-pass; N+1 for two-pass)")
        print("  (h)/(a):    human / agent rater")
        print("  ratings:    count of rated runs (human/agent)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
