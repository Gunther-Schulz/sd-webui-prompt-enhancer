"""Variant V5 — V4's source-adherence directive, gated on non-empty source.

SINGLE FACTOR from V4: the directive is applied conditionally — only
when `state['source']` is non-empty. For empty-source (dice roll),
the directive is skipped, recovering V1's creative-freedom behavior.

Hypothesis: V4 showed that generic adherence helps explicit content
(+0.4 mean, variance halved) but hurts dice-roll creativity (-0.9 on
empty_random_artist_random_setting). The directive's intent is
"preserve source content" — meaningless when there's no source. Gating
it on source presence should preserve V4's win without the regression.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def _conditional_assemble_sp(state, params):
    """Same as assemble_sp but applies prose_directive only when
    state['source'] is non-empty."""
    # Pop the stored directive so base assemble_sp doesn't apply it
    # unconditionally — we'll inject it ourselves based on state.
    directive = params.get("prose_directive")
    source = (state.get("source") or "").strip()
    params_without_directive = {k: v for k, v in params.items() if k != "prose_directive"}
    result = s.assemble_sp(state, params_without_directive)
    if directive and source:
        result["sp"] = f"{result['sp']}\n\n{directive}"
    return result


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v5_conditional_adherence",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
            Step("prose",             s.prose,             {"temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
