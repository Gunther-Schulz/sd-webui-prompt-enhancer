"""Variant V4 — V1 with a generic source-adherence directive.

SINGLE FACTOR CHANGED from V1: `prose_directive` on assemble_sp — a
different text than V2's (V4 is generic adherence, V2 was
NSFW-specific). Everything else identical.

Hypothesis: V1's explicit-content failure comes from LLM softening
source content generally (not just NSFW). A generic "preserve every
source element" directive should reduce sanitization across the board
without narrowing LLM focus (V2's coverage regression).
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s


_PROSE_DIRECTIVE_V4 = (
    "STRICT SOURCE ADHERENCE: the user's source prompt is ground truth. "
    "Every concrete element of the source — subjects, actions, states, "
    "objects, settings, attributes — must appear in your prose. Do not "
    "drop any element. Do not substitute with a softer or more abstract "
    "equivalent. Do not re-interpret. If the source uses a specific word "
    "(including explicit, adult, violent, or otherwise mature words), "
    "your prose uses that word or its direct concrete equivalent — never "
    "a euphemism and never a sanitized rephrasing. You may add concrete "
    "detail around what the source provides, but the source is inviolate."
)


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v4_strict_adherence",
        steps=[
            Step("assemble_sp",       s.assemble_sp,       {
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
