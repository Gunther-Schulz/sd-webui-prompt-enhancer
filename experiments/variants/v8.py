"""Variant V8 — V5 with multi-sample prose + picker.

STRUCTURAL change from V5: the `prose` step is replaced with
`multi_sample_prose` (3 samples + picker LLM call). Downstream steps
receive the picked sample as `prose`.

Hypothesis: per-seed variance on explicit content is the LLM
occasionally landing on a sanitizing trajectory. Sampling 3 times and
picking the best should cut full-failure rate from ~20% (V1) to near
zero, since P(all-3-fail) ≈ 0.2^3 = 0.008 if P(one-fail) is 0.2.
Assumes the picker can identify the best sample reliably.

Cost: 4 LLM calls at prose step (3 samples + 1 picker) vs 1. Slower
but GPU is cheap per user's note.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.variants.v5 import _conditional_assemble_sp
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v8_multisample_prose",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),

            # Structural change: 3 prose samples + picker
            Step("multi_sample_prose", s8.multi_sample_prose, {
                "n_samples": p.get("n_samples", 3),
                "temperature": p.get("temperature", 0.8),
                "num_predict": p.get("num_predict", 1024),
                "model": p.get("model"),
            }),

            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
