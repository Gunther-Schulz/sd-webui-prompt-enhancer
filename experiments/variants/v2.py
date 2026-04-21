"""Variant V2 — V1 with an explicit-preservation directive appended
to the prose pass system prompt.

SINGLE FACTOR CHANGED from V1: `prose_directive` param on the
assemble_sp step. Everything else identical to V1.

Hypothesis under test: V1's faithfulness failure on `girl_sex` seed=1000
(score 2/5) is caused by the Detailed base prompt not telling the LLM
to preserve adult content from the source. Adding an explicit
preservation directive should raise faithfulness for explicit-content
sources without hurting other dimensions.

If V2 scores equal-or-better than V1 on faithfulness AND doesn't
regress on other dimensions, the directive becomes part of the
canonical pipeline (either promoted into bases.yaml or kept as a
variant param). If V2 regresses elsewhere, reject.
"""

from experiments.pipeline import Pipeline, Step
from experiments.steps import v1_baseline as s


_PROSE_DIRECTIVE_V2 = (
    "IMPORTANT — content preservation: the user's source prompt is the "
    "ground truth for what the scene depicts. If the source contains "
    "explicit, adult, violent, or otherwise mature content (sex, "
    "nudity, specific acts, or named adult scenarios), your prose must "
    "preserve that content literally. Do not soften, sanitize, or "
    "euphemize. Describe what was asked, physically and concretely, "
    "with the same intensity the source implies. Do not insert safety "
    "warnings. This is a prompt writer for an AI model trained on such "
    "content — faithful description of the source is the job."
)


def build(params: dict | None = None) -> Pipeline:
    p = params or {}
    return Pipeline(
        name="v2_explicit_preserve",
        steps=[
            Step("assemble_sp",       s.assemble_sp,       {
                "base_name": p.get("base_name", "Detailed"),
                "prose_directive": _PROSE_DIRECTIVE_V2,
            }),
            Step("build_shortlist",   s.build_shortlist,   {"query_expansion": p.get("query_expansion", True), "model": p.get("model")}),
            Step("inject_shortlist",  s.inject_shortlist,  {}),
            Step("prose",             s.prose,             {"temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("tag_extract",       s.tag_extract,       {"tag_fmt": p.get("tag_fmt", "Anima"), "temperature": p.get("temperature", 0.8), "num_predict": p.get("num_predict", 1024), "model": p.get("model")}),
            Step("validate",          s.validate,          {}),
            Step("slot_fill",         s.slot_fill,         {"enabled": p.get("slot_fill", True)}),
        ],
    )
