"""Variant V13 — V11 + Pass 2 grounding (real root-cause fix).

Between Pass 1 (prose) and Pass 2 (tag-extract LLM call), retrieve the
top-K general-category Danbooru tags semantically matching the prose
(FAISS) and inject them into Pass 2's system prompt as a "prefer these"
candidate list.

Hypothesis: Pass 2 LLM's worst behaviors (user-reported weird tags like
sparse_leg_hair on outdoor scenes, overhead_lights in daylight, simple_fish
in gardens) come from the LLM *inventing* Danbooru-style compound tokens
from its training priors. Constraining it to select from a pre-retrieved
scene-relevant vocabulary removes the invention surface.

Rating target: V13 should preserve V11's wins (Random Era decade
diversity) while adding coherence on terse sources where the weird-tag
bias manifests. Does NOT target girl_ra_re specifically; the primary
win should be on prompts where Pass 2 currently emits niche off-theme
tags.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.steps import v11_source as s11
from experiments.steps import v13_grounding as s13
from experiments.variants.v5 import _conditional_assemble_sp
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v13_pass2_grounding",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("resolve_deferred_sources", s11.resolve_deferred_sources, {}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),

            Step("multi_sample_prose", s8.multi_sample_prose, {
                "n_samples": p.get("n_samples", 3),
                "temperature": p.get("temperature", 0.8),
                "num_predict": p.get("num_predict", 1024),
                "model": p.get("model"),
            }),

            # NEW in V13: build Pass 2 candidate list from prose BEFORE
            # tag_extract. tag_extract reads pass2_grounding_fragment from
            # state and prepends it to tag_sp.
            Step("build_pass2_candidates", s13.build_pass2_candidates, {
                "k":              p.get("pass2_grounding_k", 60),
                "min_post_count": p.get("pass2_grounding_min_pc", 100),
            }),

            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
            Step("source_inject",     s11.source_inject,   {}),
        ],
    )
