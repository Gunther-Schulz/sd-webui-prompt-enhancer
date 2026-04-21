"""Pipeline abstraction for the tagging-system experiment runner.

A Pipeline is an ordered sequence of named Steps. Each Step transforms
a `State` dict in-place and appends a record to a `Trace`. The Trace
captures every step's inputs, outputs, params, timings, and any error —
so an experiment run is fully reproducible and introspectable.

Design principles (enforced, not aspirational):

1. **Shared with the extension.** The same Pipeline runs inside the
   Forge extension's event handlers. A variant promoted to production
   is a config change, not a reimplementation. See CLAUDE.md
   "No island tests".

2. **Fail-loud.** Steps raise on unexpected input (empty LLM output,
   zero retrieval candidates, etc.). Silent defaults are forbidden.
   The Trace captures the exception visibly. CLAUDE.md
   "No silent fallbacks".

3. **State is a plain dict.** Each Step documents what keys it reads
   and writes. No hidden attributes. Cheap to serialize for the trace.

4. **Traces are append-only records.** Never mutate a prior step's
   record. Every Step gets its own record in order.
"""

from __future__ import annotations

import json
import time
import traceback as _tb
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class PipelineError(RuntimeError):
    """A Step raised or a structural contract was violated.

    Carries the step name + partial trace so the caller can introspect
    what happened before the failure.
    """

    def __init__(self, step_name: str, message: str, trace: Trace):
        super().__init__(f"[{step_name}] {message}")
        self.step_name = step_name
        self.trace = trace


# ── Step interface ───────────────────────────────────────────────────

# A Step is any callable that accepts (state, params) and returns the
# modified state. It must NOT return None. Side effects that aren't
# reflected in state or added to `trace` are forbidden — violating
# this makes the pipeline non-introspectable.
StepFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


@dataclass
class Step:
    """A named pipeline stage.

    `fn` is the transform. `params` is a dict of static configuration
    passed into fn alongside state — the same fn can be reused in
    multiple variants with different params.
    """
    name: str
    fn: StepFn
    params: Dict[str, Any] = field(default_factory=dict)


# ── Trace ────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """One step's execution, captured for post-run analysis.

    Only records `outputs_written` (the delta this step produced) plus
    `inputs_seen` (just the *names* of pre-existing state keys — not
    their values, which are derivable by replaying prior deltas).
    This keeps trace JSON compact; full state at any step is
    reconstructible by replaying deltas from pipeline_input.
    """
    step_name: str
    params: Dict[str, Any]
    started_at: float
    elapsed_s: float
    inputs_seen: List[str]          # key names only — values derivable
    outputs_written: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step": self.step_name,
            "params": _json_safe(self.params),
            "elapsed_s": round(self.elapsed_s, 3),
            "inputs_seen": sorted(self.inputs_seen),
            "outputs_written": _json_safe(self.outputs_written),
        }
        if self.notes:
            d["notes"] = list(self.notes)
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class Trace:
    """Ordered record of every step's execution.

    The Trace is the primary artifact of an experiment run. Rate outputs
    should be judged by reading the trace, not by counting metrics.
    """
    run_id: str
    variant_name: str
    pipeline_input: Dict[str, Any]
    records: List[StepRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    outcome: str = "in_progress"   # in_progress | ok | failed

    # Current step may add arbitrary notes to the in-progress record
    # via trace.note(str). Useful for diagnostics that aren't state.
    _current: Optional[StepRecord] = None

    def start_step(self, step: Step, inputs_seen: List[str]) -> None:
        rec = StepRecord(
            step_name=step.name,
            params=dict(step.params),
            started_at=time.perf_counter(),
            elapsed_s=0.0,
            inputs_seen=list(inputs_seen),
            outputs_written={},
        )
        self._current = rec

    def finish_step(self, outputs_written: Dict[str, Any]) -> None:
        if self._current is None:
            raise RuntimeError("finish_step called without matching start_step")
        rec = self._current
        rec.elapsed_s = time.perf_counter() - rec.started_at
        rec.outputs_written = dict(outputs_written)
        self.records.append(rec)
        self._current = None

    def fail_step(self, error: str) -> None:
        if self._current is None:
            raise RuntimeError("fail_step called without matching start_step")
        rec = self._current
        rec.elapsed_s = time.perf_counter() - rec.started_at
        rec.error = error
        self.records.append(rec)
        self._current = None

    def note(self, msg: str) -> None:
        if self._current is None:
            raise RuntimeError("note called outside a step")
        self._current.notes.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "variant": self.variant_name,
            "pipeline_input": _json_safe(self.pipeline_input),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": self.outcome,
            "steps": [r.to_dict() for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── Pipeline ─────────────────────────────────────────────────────────


@dataclass
class Pipeline:
    """Ordered list of Steps. Immutable after construction.

    `run(initial_state, variant_name, run_id)` executes the full
    sequence, producing (final_state, trace). Raises PipelineError
    on first step failure (fail-loud) — partial trace accessible via
    the exception for diagnostics.
    """
    name: str
    steps: List[Step]

    def run(self, initial_state: Dict[str, Any],
            run_id: str,
            variant_name: Optional[str] = None) -> tuple[Dict[str, Any], Trace]:
        trace = Trace(
            run_id=run_id,
            variant_name=variant_name or self.name,
            pipeline_input=dict(initial_state),
        )
        state = dict(initial_state)
        try:
            for step in self.steps:
                pre_keys = sorted(state.keys())
                trace.start_step(step, pre_keys)
                # Snapshot references only (not deep copy) so we can
                # compute the delta after the step runs
                pre_refs = {k: id(state[k]) for k in state}
                try:
                    new_state = step.fn(state, step.params)
                except Exception as exc:  # fail-loud
                    err = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
                    trace.fail_step(err)
                    trace.outcome = "failed"
                    trace.ended_at = time.time()
                    raise PipelineError(step.name, str(exc), trace) from exc
                if new_state is None:
                    err = f"step {step.name!r} returned None (must return state dict)"
                    trace.fail_step(err)
                    trace.outcome = "failed"
                    trace.ended_at = time.time()
                    raise PipelineError(step.name, err, trace)
                # Delta = keys that are new or whose object identity changed
                delta = {
                    k: v for k, v in new_state.items()
                    if k not in pre_refs or pre_refs[k] != id(v)
                }
                trace.finish_step(delta)
                state = new_state
            trace.outcome = "ok"
            trace.ended_at = time.time()
            return state, trace
        except PipelineError:
            raise
        except Exception as exc:
            # Shouldn't happen — Pipeline itself errored. Record + reraise.
            trace.outcome = "failed"
            trace.ended_at = time.time()
            raise


# ── helpers ──────────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    """Coerce objects into something json-serializable. Keeps strings,
    numbers, bools, None as-is. Lists/dicts recursed. Other objects
    become their repr + type so the trace stays useful without crashing
    on a sentence-transformer reference."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(_json_safe(x) for x in obj)
    # Fall back to type + short repr to avoid leaking huge objects
    s = repr(obj)
    if len(s) > 300:
        s = s[:300] + "…"
    return {"_type": type(obj).__name__, "_repr": s}
