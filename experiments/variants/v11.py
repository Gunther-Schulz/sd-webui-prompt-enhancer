"""Variant V11 — V8 + source: mechanism + post-inject + ◆◇ on Random Artist.

This is NOT a fresh experiment — it's the rating target for the
cumulative effect of this session's post-V8 changes:

    1. Data-driven ◆ randoms (Random Era + 6 new modifiers) pick concrete
       Danbooru tags before the LLM runs, eliminating LLM collapse bias.
    2. ◆◇ on Random Artist + Random Franchise pre-picks a real artist
       via FAISS retrieval, steering prose style AND acting as post-fill
       safety net.
    3. _inject_source_picks post-validation pass guarantees every
       source:-picked tag survives into the final output.

Pipeline shape is V8 (multi-sample prose + picker) with two new steps:
deferred source resolution (after shortlist build), and source inject
(after slot_fill). Both mirror the shipped handler logic via the pe
bootstrap — no drift.

Comparison: V11 per-prompt means vs V8's stored ratings in
.ai/experiments/v8/_ratings.json. Hypothesis: the source mechanism
improves groundedness + coverage on modifier-heavy prompts
(girl_random_artist_random_era, empty_random_artist_random_setting)
without regressing on prompts that don't use ◆ modifiers
(girl_sex, miku_cake).
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.steps import v11_source as s11
from experiments.variants.v5 import _conditional_assemble_sp
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v11_source_mechanism",
        steps=[
            Step("assemble_sp",       _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            # NEW: resolve db_retrieve sources now that shortlist build
            # has warmed the stack. Must run AFTER build_shortlist (so
            # the stack is up) and BEFORE inject_shortlist (which bakes
            # the SP). Rebuild of style_str happens inside pe's resolver
            # so mods reflect the picked value for downstream prose.
            Step("resolve_deferred_sources", s11.resolve_deferred_sources, {}),
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
            # NEW: post-inject source-picked tags that didn't survive
            # validation/slot_fill. Mirrors the shipped handler logic.
            Step("source_inject",     s11.source_inject,   {}),
        ],
    )
