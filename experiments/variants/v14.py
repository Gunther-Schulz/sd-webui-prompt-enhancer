"""Variant V14 — structural-only Hybrid output.

V14 = V11 pipeline with two terminal steps appended:
    strip_to_structural     — drop CAT_GENERAL and CAT_META tags
    assemble_hybrid_output  — final_tags = structural_tags + "\\n\\n" + prose

Hypothesis: the Anima Qwen3 encoder handles prose natively, so general-category
tags (pose/clothing/body/setting/lighting/atmosphere) contribute nothing the
prose doesn't already carry — but they ARE the exclusive origin of "sparse leg
hair / simple fish / overhead lights" RAG mis-retrievals. Dropping them should
eliminate the weird-tag failure class while preserving style/quality/identity
conditioning (quality, safety, subject-count, @artist, character, series).

Single-factor change vs V11: terminal filter + Hybrid-format combine step.
Pipeline body through source_inject is byte-identical to V11.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s
from experiments.steps import v8_multisample as s8
from experiments.steps import v11_source as s11
from experiments.steps import v14_structural as s14
from experiments.variants.v4 import _PROSE_DIRECTIVE_V4
from experiments.variants.v5 import _conditional_assemble_sp


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v14_structural_hybrid",
        steps=[
            Step("assemble_sp", _conditional_assemble_sp, {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V4,
            }),
            Step("build_shortlist", s.build_shortlist, {
                "query_expansion": p.get("query_expansion", True),
                "model": p.get("model"),
            }),
            Step("resolve_deferred_sources", s11.resolve_deferred_sources, {}),
            Step("inject_shortlist", s.inject_shortlist, {}),
            Step("multi_sample_prose", s8.multi_sample_prose, {
                "n_samples": p.get("n_samples", 3),
                "temperature": p.get("temperature", 0.8),
                "num_predict": p.get("num_predict", 1024),
                "model": p.get("model"),
            }),
            Step("tag_extract", s.tag_extract, {
                "tag_fmt": p.get("tag_fmt", "Anima"),
                "temperature": p.get("temperature", 0.8),
                "num_predict": p.get("num_predict", 1024),
                "model": p.get("model"),
            }),
            Step("validate", s.validate, {}),
            Step("slot_fill", s.slot_fill, {"enabled": p.get("slot_fill", True)}),
            Step("source_inject", s11.source_inject, {}),
            # V14 additions: drop general tags, emit hybrid output
            Step("strip_to_structural", s14.strip_to_structural, {}),
            Step("assemble_hybrid_output", s14.assemble_hybrid_output, {}),
        ],
    )
