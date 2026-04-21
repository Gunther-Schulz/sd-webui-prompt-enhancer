"""V8 multi-sample prose + picker step.

Instead of one prose call, generate N samples at temp 0.8 (variety),
then a single LLM "picker" call chooses the sample that best preserves
source intent. Downstream steps use the picked sample.

This attacks variance structurally: per-seed failures only kill the
output if ALL samples fail. For the girl_sex hot case in V1/V5, 60% of
seeds produced a preserving prose — if we sample 3 times, the chance
all 3 fail is ~6%. If picker can identify the best, we nearly eliminate
full-sanitization outputs.

Cost: 4 LLM calls at prose step (3 samples + 1 picker) vs 1 for V5.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.steps.common import call_llm, pe
from experiments.steps import v1_baseline as v1s


_PICKER_SP = """\
You are a prose quality picker. You will see:
  - a user SOURCE prompt (may be terse or contain explicit/adult content)
  - N candidate PROSE passages, each an attempt to expand the source

Your job: choose the prose that most faithfully preserves the source's
concrete content. In particular:
  - If the source contains explicit, mature, or adult content (sex,
    nudity, specific acts, named adult scenarios), choose the prose
    that preserves that content literally.
  - If the source names specific subjects, choose the prose that
    features those exact subjects.
  - Penalize proses that sanitize, euphemize, or redirect the source.
  - All else equal, prefer the prose with richer concrete detail.

Output format: respond with ONLY the number of the best prose (1, 2, 3, …).
No explanation. No other text.
"""


def multi_sample_prose(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: sp (str), source (str), style_prose (str), seed (int).
    Write: prose (str — the picked sample),
           prose_samples (list[str] — all samples),
           prose_picker_choice (int — 1-indexed chosen sample).

    Generates `n_samples` prose passages with different derived seeds
    (seed+0, seed+1, seed+2), then asks a picker LLM to select the best.
    """
    n_samples = int(params.get("n_samples", 3))
    sp = state["sp"]
    source = state["source"]
    style_prose = state.get("style_prose", "")
    base_seed = state.get("seed", -1)

    user_msg = f"SOURCE PROMPT: {source}" if source else pe._EMPTY_SOURCE_SIGNAL
    if style_prose:
        user_msg = f"{user_msg}\n\n{style_prose}"

    samples: List[str] = []
    for i in range(n_samples):
        # Derive a distinct seed per sample so samples genuinely differ
        sample_seed = base_seed + i if base_seed != -1 else -1
        sample = call_llm(
            sp, user_msg,
            seed=sample_seed,
            temperature=params.get("temperature", 0.8),
            num_predict=params.get("num_predict", 1024),
            model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
        )
        samples.append(sample)

    # Picker call: LLM sees source + numbered samples, picks one
    picker_user_msg_parts = [f"SOURCE: {source!r}" if source else "SOURCE: (empty — dice roll)"]
    for i, s in enumerate(samples, 1):
        # Trim samples to keep picker context manageable
        s_trim = s if len(s) < 800 else s[:800] + "…"
        picker_user_msg_parts.append(f"\n--- Prose {i} ---\n{s_trim}")
    picker_user_msg_parts.append("\nWhich is best? Respond with only the number.")
    picker_user_msg = "\n".join(picker_user_msg_parts)

    pick_raw = call_llm(
        _PICKER_SP, picker_user_msg,
        seed=base_seed,
        temperature=0.1,   # picker should be near-deterministic
        num_predict=10,
        model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
    )
    # Parse — take first digit in the response
    choice = None
    for ch in pick_raw:
        if ch.isdigit() and 1 <= int(ch) <= n_samples:
            choice = int(ch)
            break
    if choice is None:
        # Picker didn't return a valid digit — fail loud
        raise ValueError(
            f"picker returned no valid choice. raw response: {pick_raw!r}"
        )

    picked = samples[choice - 1]
    return {
        **state,
        "prose": picked,
        "prose_samples": samples,
        "prose_picker_choice": choice,
        "prose_picker_raw": pick_raw,
    }
