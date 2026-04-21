"""Variant V1 — baseline: the extension's current Hybrid/Tags pipeline
wrapped in the experiment runner.

Purpose: establish a rated-baseline measurement of the current system
before designing V2. Any V2 that scores worse than V1 on the rubric
is rejected regardless of structural appeal.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v1_baseline",
        steps=[
            Step("assemble_sp",       s.assemble_sp,       {"base_name": p.get("base_name", "Detailed")}),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
            Step("prose",             s.prose,             {"temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
