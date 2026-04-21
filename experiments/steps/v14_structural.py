"""V14 steps — structural-only tag output + prose body for Hybrid mode.

Rationale from the Anima model card (circlestone-labs/Anima): the model was
trained on Danbooru tags, natural language captions, AND combinations.
Qwen3 0.6B as text encoder handles prose natively.

General-category tags (pose, clothing, body parts, setting, lighting,
atmosphere) add no encoder signal that the prose doesn't already carry,
AND are the exclusive origin of the "sparse leg hair / simple fish /
overhead lights" mis-retrievals from the FAISS validator — when the
draft has no good DB match, the validator force-maps to the nearest
neighbor which is often a low-popularity niche tag.

Structural tags the model DOES need as tags (not representable in prose):
  - Quality (masterpiece, score_N)   — format whitelist token
  - Safety (safe/sensitive/nsfw/explicit) — format whitelist token
  - Subject count (1girl, 1boy, 2girls) — regex match, structural
  - Artist (@name)                    — style conditioning via @ prefix
  - Character (named entity embedding)
  - Series/copyright                  — anchors character identity

Everything else → prose carries it.

Output shape: final_tags = "{structural_tags}\\n\\n{prose}" — the exact
string fed to Anima.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from anima_tagger import config as anima_config
from experiments.steps.common import get_stack, pe


_SUBJECT_COUNT_RE = re.compile(r"^\d+(girl|boy|other)s?$")

_STRUCTURAL_CATEGORIES = {
    anima_config.CAT_ARTIST,
    anima_config.CAT_CHARACTER,
    anima_config.CAT_COPYRIGHT,
}


def _is_structural(tag: str, stack, whitelist: set) -> bool:
    """Classify a tag as structural (keep) or general/meta/noise (drop).

    Structural = quality/safety/subject-count/artist/character/series.
    Normalization mirrors pe._tags_have_category (strip @, spaces→underscores).
    """
    norm = tag.strip().lstrip("@").lower().replace(" ", "_").replace("-", "_")
    if not norm:
        return False
    if norm in whitelist:
        return True
    if _SUBJECT_COUNT_RE.match(norm):
        return True
    if stack is None or stack.db is None:
        return False
    rec = stack.db.get_by_name(norm)
    if not rec:
        return False
    return rec.get("category") in _STRUCTURAL_CATEGORIES


def strip_to_structural(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: final_tags (after source_inject), tag_fmt.
    Write: structural_tags (str) — only structural-category tags retained.

    Drops CAT_GENERAL and CAT_META outright. Subject-count tokens (1girl etc.)
    survive via regex — they live in CAT_GENERAL in the DB but function as
    structural signal the model expects.
    """
    tags_str = state.get("final_tags") or state.get("tags_after_validate", "")
    tag_fmt = state.get("tag_fmt", "Anima")
    fmt = pe._tag_formats.get(tag_fmt, {})
    whitelist: set = set()
    whitelist |= {str(t).lower() for t in (fmt.get("quality_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt.get("leading_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt.get("rating_tags") or set())}
    stack = get_stack()

    dropped = []
    kept = []
    for t in tags_str.split(","):
        t = t.strip()
        if not t:
            continue
        if _is_structural(t, stack, whitelist):
            kept.append(t)
        else:
            dropped.append(t)
    return {
        **state,
        "structural_tags": ", ".join(kept),
        "v14_dropped_general": dropped,
    }


def assemble_hybrid_output(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: structural_tags, prose.
    Write: final_tags (str) — "{structural_tags}\\n\\n{prose}".

    Overwrites final_tags with the combined string that goes to Anima.
    The rater judges this exact string (image model input).
    """
    structural = state.get("structural_tags", "") or ""
    prose = state.get("prose", "") or ""
    if structural and prose:
        combined = f"{structural}\n\n{prose}"
    elif structural:
        combined = structural
    else:
        combined = prose
    return {
        **state,
        "final_tags": combined,
        "final_structural_tags": structural,
        "final_prose": prose,
    }
