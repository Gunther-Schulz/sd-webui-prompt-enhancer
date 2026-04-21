"""Variant V9 — V8 with a compound directive: strict preservation AND
creative expansion on thin sources.

SINGLE FACTOR from V8: the prose directive text. Instead of pure
strict adherence, the directive combines two separable instructions:
  (1) preserve every concrete source element (V4/V5 strictness)
  (2) when source is thin, expand creatively to fill in pose, clothing,
      lighting, setting, mood — produce a rich scene regardless of how
      terse the source was (NEW)

Hypothesis: V8's remaining regression on girl_random_artist_random_era
(-0.4 vs V1) comes from the directive firing on thin source "girl"
and narrowing LLM focus onto that one word. A compound directive
that simultaneously mandates preservation AND expansion should keep
V8's girl_sex win while restoring creative richness on thin sources.

Per CLAUDE.md lesson learned: didn't try this earlier because I was
treating preservation and expansion as opposing axes. They're separable.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.variants.v5 import _conditional_assemble_sp


_PROSE_DIRECTIVE_V9 = (
    "STRICT SOURCE ADHERENCE: the user's source prompt is ground truth "
    "for the scene's concrete content. Every specific element the source "
    "names — subjects, actions, states, objects, settings, attributes, "
    "including explicit / adult / mature content — must appear in your "
    "prose literally. Do not drop, substitute, soften, or euphemize any "
    "specific element the source provides.\n\n"
    "RICH EXPANSION: your prose should always describe a fully-realized "
    "scene — subject with pose and expression, clothing (multiple layers "
    "when applicable), setting, lighting, mood, supporting details. "
    "When the source is thin (just a subject word, a short phrase, or "
    "empty), INVENT coherent concrete detail to fill every compositional "
    "slot. A terse source is an invitation to expand, not a cap on "
    "output richness. Aim for a prose that lets downstream tag "
    "extraction produce 20+ distinct scene concepts.\n\n"
    "Reconciling the two: strict adherence applies to what the source "
    "gives you — never contradict or dilute it. Expansion adds detail "
    "around those concrete elements — never replaces them."
)


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v9_compound_directive",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V9,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
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
