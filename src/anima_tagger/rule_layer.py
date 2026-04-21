"""Anima convention rules applied to validated tag records.

After the validator returns canonical DB tag records for each accepted
draft token, the rule layer enforces Anima's structural requirements:

  - quality tag prefix (masterpiece, best_quality, score_7)
  - exactly one safety tag (safe / sensitive / nsfw / explicit)
  - @ prefix for artist-category tags
  - character → series pairing via co-occurrence
  - lowercase with spaces (except score_N, year_YYYY, source_*)
  - de-duplicate, enforce ordering: quality → safety → subjects →
    character → series → artist → general
"""

import re
from typing import Dict, Iterable, List, Optional

from . import config
from .cooccurrence import CoOccurrence


# Score_N / year_YYYY / source_* keep their underscores in space-form output.
_PRESERVE_UNDERSCORE_RE = re.compile(r"^(score_\d+(_up)?|year_\d{4}|source_[a-z]+)$")

# Default quality prefix when the LLM draft doesn't include its own.
_QUALITY_PREFIX = ("masterpiece", "best_quality", "score_7")

# Any of these in the draft are promoted to the quality slot (keeps
# the LLM's score_9 / best_quality / etc. intact instead of overwriting).
_QUALITY_TOKENS = {
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality",
    "score_9", "score_8", "score_7", "score_6", "score_5",
    "score_4", "score_3", "score_2", "score_1",
    "highres", "absurdres",
}

_VALID_SAFETY = {"safe", "sensitive", "nsfw", "explicit"}
_DEFAULT_SAFETY = "safe"

# Universal subject/count tags (same ordering slot across all booru formats)
_SUBJECT_TAGS = {
    "1girl", "1girls", "2girls", "3girls", "4girls", "5girls",
    "6+girls", "multiple_girls",
    "1boy", "1man", "2boys", "3boys", "4boys", "5boys",
    "6+boys", "multiple_boys",
    "1other", "solo", "no_humans", "male_focus", "female_focus",
}


def _format_out(name: str, use_underscores: bool) -> str:
    """Apply space/underscore convention with canonical exceptions."""
    if use_underscores:
        return name
    if _PRESERVE_UNDERSCORE_RE.match(name):
        return name
    return name.replace("_", " ")


def _bucket(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    """Sort tag records into Anima output order buckets."""
    buckets: Dict[str, List[Dict]] = {
        "subjects": [], "character": [], "series": [],
        "artist": [], "general": [],
    }
    for r in records:
        cat = r.get("category", config.CAT_GENERAL)
        if r["name"] in _SUBJECT_TAGS:
            buckets["subjects"].append(r)
        elif cat == config.CAT_CHARACTER:
            buckets["character"].append(r)
        elif cat == config.CAT_COPYRIGHT:
            buckets["series"].append(r)
        elif cat == config.CAT_ARTIST:
            buckets["artist"].append(r)
        else:
            buckets["general"].append(r)
    return buckets


def apply_anima_rules(records: List[Dict],
                      safety: str = _DEFAULT_SAFETY,
                      cooccurrence: Optional[CoOccurrence] = None,
                      use_underscores: bool = False,
                      include_quality_prefix: bool = True,
                      max_tags: int = 30) -> List[str]:
    """Turn a list of validated tag records into a final Anima tag list.

    `records` are DB dicts with at least `name` and `category`.
    """
    out: List[str] = []
    seen: set[str] = set()

    def _add(raw_name: str) -> bool:
        key = raw_name.lower()
        if key in seen:
            return False
        seen.add(key)
        out.append(_format_out(raw_name, use_underscores))
        return True

    # Quality prefix. Emit masterpiece + best_quality always, then
    # exactly ONE score_N tag:
    #   - If the LLM draft contains a score_N, use that (highest if multiple)
    #   - Otherwise use score_7 (the default)
    # Non-score quality tokens from the draft (masterpiece, best_quality,
    # highres, absurdres) are deduped via _add.
    import re as _re
    _SCORE_RE = _re.compile(r"^score_(\d+)(_up)?$")
    draft_quality = [r["name"] for r in records if r["name"] in _QUALITY_TOKENS]
    draft_scores = [q for q in draft_quality if _SCORE_RE.match(q)]
    draft_non_score = [q for q in draft_quality if not _SCORE_RE.match(q)]
    if include_quality_prefix:
        # Emit masterpiece + best_quality from the canonical prefix
        for q in _QUALITY_PREFIX:
            if _SCORE_RE.match(q):
                continue   # handled below — exactly one score tag
            _add(q)
        # Exactly one score tag: highest from draft, or default score_7
        if draft_scores:
            # Pick the highest score_N (score_9 > score_7 > score_1)
            chosen_score = max(
                draft_scores,
                key=lambda s: int(_SCORE_RE.match(s).group(1)),
            )
        else:
            chosen_score = next(
                (q for q in _QUALITY_PREFIX if _SCORE_RE.match(q)),
                "score_7",
            )
        _add(chosen_score)
        # Non-score draft quality tokens (masterpiece, absurdres, etc.)
        for q in draft_non_score:
            _add(q)

    # Safety — exactly one. If the LLM draft already included a safety
    # tag, trust it; otherwise inject the chosen default.
    safety_from_draft = next(
        (r["name"] for r in records if r["name"] in _VALID_SAFETY), None,
    )
    chosen_safety = safety_from_draft or (
        safety if safety in _VALID_SAFETY else _DEFAULT_SAFETY
    )
    _add(chosen_safety)

    # Remove quality + safety tags from records (already emitted)
    consumed = set(draft_quality) | _VALID_SAFETY | set(_QUALITY_PREFIX)
    filtered = [r for r in records if r["name"] not in consumed]

    buckets = _bucket(filtered)

    # Emit in Anima order: subjects → character → series → artist → general
    for bucket_name in ("subjects", "character", "series", "artist", "general"):
        for rec in buckets[bucket_name]:
            if len(out) >= max_tags:
                break
            name = rec["name"]
            # Artist gets @ prefix
            if rec.get("category") == config.CAT_ARTIST and not name.startswith("@"):
                name = "@" + name
            _add(name)
            # Pair character with series via co-occurrence
            if rec.get("category") == config.CAT_CHARACTER and cooccurrence:
                hits = cooccurrence.top_for(
                    rec["name"], category=config.CAT_COPYRIGHT,
                    top_k=1, min_prob=0.5,
                )
                if hits:
                    _add(hits[0]["tag"])
        if len(out) >= max_tags:
            break

    out = _dedup_redundant_subset_tags(out, use_underscores)
    return out[:max_tags]


def _dedup_redundant_subset_tags(tags: List[str], use_underscores: bool) -> List[str]:
    """Drop tags whose words are a strict subset of another tag's words,
    keeping the more specific one. Catches common LLM redundancy where
    it emits both the parent concept and a specific variant:

        cake / holding_cake       → keep holding_cake
        coat / trench_coat        → keep trench_coat
        floor / wooden_floor      → keep wooden_floor
        apron / black_apron       → keep black_apron
        boot / leather_boots      → keep leather_boots (plural match is OK)

    Does NOT drop @-prefixed tags (artists), score_N (quality), or
    subject-count tags — those are semantically atomic regardless of
    word overlap.
    """
    sep = "_" if use_underscores else " "
    PROTECT_PREFIXES = ("@",)
    def _words(t: str) -> set[str]:
        return set(t.replace("_", sep).split(sep))
    pairs_by_idx = [(i, t, _words(t)) for i, t in enumerate(tags)]
    drop_idx: set[int] = set()
    for i, ti, wi in pairs_by_idx:
        if ti.startswith(PROTECT_PREFIXES):
            continue
        if _PRESERVE_UNDERSCORE_RE.match(ti.replace(" ", "_")):
            continue
        if ti in _SUBJECT_TAGS or ti.replace(" ", "_") in _SUBJECT_TAGS:
            continue
        for j, tj, wj in pairs_by_idx:
            if i == j or j in drop_idx:
                continue
            # ti's words are strict subset of tj's → ti is less specific
            if wi and wi < wj:
                drop_idx.add(i)
                break
    return [t for i, t in enumerate(tags) if i not in drop_idx]
