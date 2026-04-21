"""Variant V10 — V8 + niche-tag gating and meta-tag filter.

TWO FACTORS from V8 (bundled, will split if results are mixed):
  (a) general_min_post_count=100 on TagValidator — exact/alias hits
      on CAT_GENERAL tags must clear a popularity floor. Addresses
      niche variants like "weapon_across_shoulders" (65 posts),
      "holding_waist" (61), "bent" (78) slipping through and
      polluting output.
  (b) reject_meta_non_whitelist=True — CAT_META tags are dropped
      unless they're in the format whitelist (score_N, masterpiece).
      Addresses "fixed" (meta) and similar appearing in general tag
      output where they have no meaning for image gen.

Hypothesis: output tag lists will have higher average popularity,
reflecting tags a user would actually want. Groundedness dimension
should measurably improve. Possible risk: legitimate rare-but-valid
tags dropped — watch for coverage regression on specialized prompts.

Per CLAUDE.md: these two fixes are bundled because they address the
same symptom (niche-tag pollution) observed on a real Forge output.
If V10 wins overall, we'll know the bundle works. Separating into
V10a/V10b for attribution would only be worth it if V10 flops.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.variants.v5 import _conditional_assemble_sp
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def _validate_v10(state, params):
    """V8 validation flow but with V10 gates on the validator.

    Rebuilds the TagValidator with general_min_post_count + meta
    filter enabled, calls tagger.tag_from_draft pointing at that
    gated validator.
    """
    from experiments.steps.common import get_stack, pe
    from anima_tagger.validator import TagValidator
    from anima_tagger.tagger import AnimaTagger

    stack = get_stack()
    # Build a one-off validator with the V10 gates ON
    gated = TagValidator(
        db=stack.db, index=stack.index, embedder=stack.embedder,
        semantic_threshold=0.7,
        semantic_min_post_count=50,
        entity_min_post_count=500,
        general_min_post_count=params.get("general_min_post_count", 100),
        reject_meta_non_whitelist=params.get("reject_meta_non_whitelist", True),
    )
    gated_tagger = AnimaTagger(
        validator=gated,
        db=stack.db,
        cooccurrence=stack.cooccurrence,
    )
    safety = pe._anima_safety_from_modifiers(state.get("mods", []), state.get("source", ""))
    tags_list = gated_tagger.tag_from_draft(
        state["draft"],
        safety=safety,
        shortlist=state.get("shortlist"),
        compound_split=True,
    )
    return {**state, "tags_after_validate": ", ".join(tags_list), "stats": {"total": len(tags_list)}, "anima_safety": safety}


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v10_popularity_floor_meta_filter",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
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
            Step("validate",          _validate_v10,       {
                "general_min_post_count": p.get("general_min_post_count", 100),
                "reject_meta_non_whitelist": p.get("reject_meta_non_whitelist", True),
            }),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
