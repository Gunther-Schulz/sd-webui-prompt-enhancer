"""Variant V3 — retrieval-first curator.

STRUCTURAL change from V1: the tag_extract step is replaced with a
3-part sub-sequence (concept_enumerate → canonicalize_concepts →
curate). The LLM no longer emits tag-shaped tokens. It emits concept
phrases, those get canonicalized to real DB tags by the embedder,
then the LLM curates from the candidate pool. Invention is impossible
by construction.

Hypothesis: retrieval-first-curator reduces LLM-variance because the
candidate set is deterministic per prose — only the picking is
LLM-dependent, and picking tends to be lower-variance than free
generation of tag names.

Kept identical to V1 (to isolate the structural change):
  - prose step (Detailed base, shortlist injection)
  - validate + rule_layer + slot_fill

Changed:
  - tag_extract replaced by (concept_enumerate + canonicalize_concepts
    + curate)
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s1
from experiments.steps import v3_curator as s3


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v3_retrieval_curator",
        steps=[
            Step("assemble_sp",       s1.assemble_sp,       {"base_name": p.get("base_name", "Detailed")}),
            Step("build_shortlist",   s1.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s1.inject_shortlist,  {}),
            Step("prose",             s1.prose,             {"temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),

            # --- Structural change: replace tag_extract with 3-step curator flow ---
            Step("concept_enumerate", s3.concept_enumerate, {"temperature": p.get("enum_temp", 0.6), "num_predict": 512, "model": p.get("model")}),
            Step("canonicalize",      s3.canonicalize_concepts, {"min_score": p.get("min_score", 0.35), "min_post_count": 50, "top_k": p.get("canon_top_k", 5)}),
            Step("curate",            s3.curate,            {"temperature": p.get("curate_temp", 0.5), "num_predict": 512, "model": p.get("model")}),
            # --------------------------------------------------------------------

            Step("validate",          s1.validate,          {}),
            Step("slot_fill",         s1.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
