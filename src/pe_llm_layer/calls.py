"""Implementation of the LLM-call layer.

All inter-process LLM concerns (samplers, streaming, cancel, stall,
JSON grammar, model lifecycle) live in src.llm_runner. This module
adapts that to the historical call signatures used in
scripts/prompt_enhancer.py.

Notes:
- The old _call_llm / _call_llm_progress had api_url and model
  parameters because they talked HTTP to Ollama. The new in-process
  runner reads its config from shared.opts (Forge settings) so api_url
  and model are not needed at the call layer. The historical
  signatures are not preserved at the function level — callers in
  prompt_enhancer.py are updated to drop those args.
- The 4000-token hard cap (_MAX_TOKENS) from the old code is gone.
  num_predict at the sampler level enforces output length; nothing
  needs the post-hoc cap.
- The 90 LOC _detect_repetition heuristic from the old code is gone.
  The runner uses the DRY sampler + working repeat/freq/presence
  penalties at the sampler level (Ollama's Go runner silently dropped
  these for Qwen 3.5 — issue ollama/ollama#14493 — which is what
  forced the post-hoc trim).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("prompt_enhancer.llm_layer")

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from llm_runner import get_runner
from llm_runner import TruncatedOutput, LLMError, ModelLoadError


# Module-level cancel flag — shared between prompt_enhancer.py's cancel
# button and the runner's stream loop. The cancel button calls .set();
# call_llm/stream_llm default to checking this flag if no explicit
# cancel_flag is passed. prompt_enhancer.py imports this object so
# its existing _cancel_flag references go through the same Event.
cancel_flag = threading.Event()


# ── Sync call ───────────────────────────────────────────────────────────


def call_llm(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    *,
    seed: int = -1,
    num_predict: int = 1024,
    json_schema: Optional[dict] = None,
    timeout: Optional[float] = None,
    cancel_flag: Optional[threading.Event] = None,
) -> str:
    """Single LLM call, returns the full content string.

    Raises:
        TruncatedOutput: stream ended without natural completion.
            .partial holds whatever was generated. Some callers want
            the partial (Prose mode shows it as "Truncated").
        LLMError / ModelLoadError: hard runner failure.
    """
    runner = get_runner()
    stall_timeout_s = float(timeout) if timeout else 30.0
    return runner.call(
        prompt, system_prompt,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
        json_schema=json_schema,
        cancel_flag=cancel_flag if cancel_flag is not None else globals()["cancel_flag"],
        stall_timeout_s=stall_timeout_s,
    )


# ── Streaming call (yields progress dicts then final string) ────────────


def stream_llm(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    *,
    seed: int = -1,
    num_predict: int = 1024,
    json_schema: Optional[dict] = None,
    timeout: Optional[float] = None,
    cancel_flag: Optional[threading.Event] = None,
    progress_interval_s: float = 1.0,
) -> Iterator[Any]:
    """Generator: yields progress dicts during generation, then yields
    the final string as the last value.

    Yielded values:
      - dict {"words": int, "tokens": int, "elapsed": float, "tps": float}
        every progress_interval_s seconds
      - str (final): the accumulated output text

    Truncation is signaled by raising InterruptedError (cancel) or
    TruncatedOutput (cap/stall) — same as the old _call_llm_progress.
    The Gradio handlers in prompt_enhancer.py catch these and surface
    "Truncated" / "Cancelled" status accordingly.
    """
    runner = get_runner()
    accumulated = []
    last_yielded = time.monotonic()
    last_chunk = None
    stall_timeout_s = float(timeout) if timeout else 30.0
    effective_cancel = cancel_flag if cancel_flag is not None else globals()["cancel_flag"]

    for chunk in runner.stream(
        prompt, system_prompt,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
        json_schema=json_schema,
        cancel_flag=effective_cancel,
        stall_timeout_s=stall_timeout_s,
    ):
        if chunk.text:
            accumulated.append(chunk.text)
        last_chunk = chunk
        # Yield progress on a timer so UI updates without per-token spam
        now = time.monotonic()
        if now - last_yielded >= progress_interval_s:
            last_yielded = now
            yield {
                "words": chunk.words,
                "tokens": chunk.tokens,
                "elapsed": chunk.elapsed_s,
                "tps": chunk.tps,
            }

    text = "".join(accumulated)

    # Decide final disposition
    if effective_cancel.is_set():
        # Cancelled by user — surface partial via InterruptedError
        # (matches old _call_llm behavior at line ~2078)
        raise InterruptedError(text)
    if last_chunk is None or not last_chunk.is_complete:
        # Stream didn't finish naturally — truncated
        raise TruncatedOutput(text, "stream ended without completion")

    yield text


# ── Multi-sample prose (N candidates + picker) ──────────────────────────


def multi_sample_prose(
    user_msg: str,
    system_prompt: str,
    temperature: float,
    seed: int,
    n_samples: int,
    *,
    num_predict: int = 1024,
    picker_system_prompt: str = "",
) -> Tuple[str, List[str], int]:
    """Generate n_samples prose passages and return the picker-chosen one.

    Each sample uses (seed + i) so the sampling trajectories diverge.
    The picker call runs at near-deterministic temperature (0.1) and
    is asked to pick a single number 1..n_samples.

    Returns (picked_prose, all_samples, choice_index_1based).

    Raises ValueError if the picker returns no parsable digit — fail-loud
    per CLAUDE.md, no silent fallback to sample #0.

    Note: the old version had a _detect_repetition pre-filter that
    rejected looping samples before showing them to the picker. With
    the in-process runner using DRY + working repeat penalties, looping
    samples should be rare. If they still happen, re-add a pre-filter
    here — but expect the rate to drop dramatically vs the Ollama path.
    """
    samples: List[str] = []
    for i in range(n_samples):
        sample_seed = seed + i if seed != -1 else -1
        try:
            content = call_llm(
                user_msg, system_prompt, temperature,
                seed=sample_seed, num_predict=num_predict,
            )
        except TruncatedOutput as e:
            # Use the partial — multi-sample mode prefers a slightly
            # short sample over an outright failed one.
            content = e.partial
        samples.append(content)

    if not samples:
        raise ValueError("multi_sample_prose: no samples generated")

    # Picker: number-per-line + brief instruction
    picker_parts = [f"SOURCE: {user_msg!r}"]
    for i, s in enumerate(samples, 1):
        s_trim = s if len(s) < 800 else s[:800] + "…"
        picker_parts.append(f"\n--- Prose {i} ---\n{s_trim}")
    picker_parts.append("\nWhich is best? Respond with only the number.")
    picker_msg = "\n".join(picker_parts)

    pick_raw = call_llm(
        picker_msg, picker_system_prompt, temperature=0.1,
        seed=seed, num_predict=10,
    )
    # Parse first digit in 1..n_samples
    choice = None
    for ch in pick_raw:
        if ch.isdigit() and 1 <= int(ch) <= n_samples:
            choice = int(ch)
            break
    if choice is None:
        raise ValueError(
            f"multi-sample picker returned no valid choice. "
            f"raw={pick_raw!r} n_samples={n_samples}"
        )
    return samples[choice - 1], samples, choice


# ── Status display ──────────────────────────────────────────────────────


def get_llm_status() -> str:
    """LLM-runner status as HTML for the Forge UI status line.

    Cheap (no network, no model load) — reads from runner.info().
    Safe to call from UI refresh handlers.
    """
    try:
        runner = get_runner()
        info = runner.info()
        loaded = "loaded" if info["loaded"] else "not loaded"
        compute = info["compute"]
        n_gpu = info["n_gpu_layers"]
        if compute == "shared":
            compute = f"shared ({n_gpu} GPU layers)"
        dry = " · DRY" if info["low_level_dry"] else ""
        ref = info["model_ref"]
        if len(ref) > 70:
            ref = ref[:67] + "..."
        color = "#6c6" if info["loaded"] else "#888"
        return (f"<span style='color:{color}'>"
                f"LLM • {ref} • {loaded} • {compute}{dry}</span>")
    except Exception as e:
        return (f"<span style='color:#c66'>LLM runner unavailable: "
                f"{type(e).__name__}: {e}</span>")
