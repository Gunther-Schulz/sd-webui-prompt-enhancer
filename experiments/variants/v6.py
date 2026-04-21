"""Variant V6 — V5 with a softened directive that self-conditions on
source content.

SINGLE FACTOR from V5: directive text. Instead of "every concrete
source element must appear", it's "if the source contains specific
subjects/actions/descriptions, preserve them — especially mature
content — otherwise expand creatively as normal".

Hypothesis: V5's regression on girl_random_artist_random_era (-0.4) is
caused by the strict-adherence directive firing on thin sources (just
"girl") where the user's intent lives in the modifiers, not the source.
A softer directive that defers adherence to the LLM's judgment of
source content should preserve V5's girl_sex win without the thin-
source regression.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.variants.v5 import _conditional_assemble_sp


_PROSE_DIRECTIVE_V6 = (
    "SOURCE FIDELITY: if the user's source prompt contains specific "
    "subjects, actions, objects, states, or explicit/mature content, "
    "your prose preserves those elements as described — do not sanitize, "
    "euphemize, or substitute. You may add concrete detail around what "
    "the source provides, but do not replace or drop its specifics. "
    "If the source is sparse (just a subject word, or a vague cue), "
    "expand it creatively in whatever direction the modifiers and "
    "context suggest."
)


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v6_softer_adherence",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V6,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
            Step("prose",             s.prose,             {"temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
