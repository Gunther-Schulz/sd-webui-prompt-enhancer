"""Variant V7 — V5 pipeline with lowered prose-pass temperature.

SINGLE FACTOR from V5: prose temperature 0.8 → 0.3. All other steps
identical, directive identical, gating identical.

Hypothesis: per-seed LLM variance on explicit content is driven by
sampling noise at temperature 0.8. Lower temperature produces more
deterministic prose per seed — less variance across seeds.

Expected tradeoffs: dice-roll creativity may drop (less diverse scenes
on empty source). girl_sex should become more consistent in either
direction — whichever way temp=0.3 pushes is consistent.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.variants.v5 import _conditional_assemble_sp
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v7_lower_prose_temp",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
            # Single factor: temperature 0.8 → 0.3 ONLY on prose step
            Step("prose",             s.prose,             {"temperature": 0.3, "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
