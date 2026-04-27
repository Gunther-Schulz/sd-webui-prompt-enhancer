"""Tag validation and post-processing pipeline.

Pure-functional layer over `pe_data.tag_formats`. The caller passes
in the format config + tag DB + aliases; this module does the
correction / lookup / fuzzy-match / reorder / paren-escape work.

Pipeline (typical):

    tag_str → split-concatenated → underscore-normalize
            → validate (correct mistakes / dealias / fuzzy match / drop or keep)
            → reorder (quality → leading → subjects → rest → rating)
            → escape SD parens

Public API:
  TAG_CORRECTIONS         dict — common LLM tag mistakes → canonical
  SUBJECT_TAGS            set — universal Danbooru subject/count tags
  PRESERVE_UNDERSCORE_RE  compiled regex — tokens that keep underscores
                          regardless of format (score_N, year_YYYY, source_*)

  find_closest_tag(tag, valid_tags, max_distance=3) → (corrected, None)
                          or (None, original)
  format_tag_out(tag, use_underscores) → tag in format-correct form
  validate(tags_str, fmt_config, valid_tags, aliases, *, mode="Fuzzy",
           clean_tag=...) → (corrected_csv, stats_dict)
  reorder(tags, fmt_config, universal_quality_tags) → list[str]
  postprocess(tag_str, fmt_config, valid_tags, aliases, *,
              validation_mode, clean_tag=..., split_concat=...,
              universal_quality_tags=...) → (str, stats|None)

All functions take format / db data as arguments — no module-level
state, no Forge dependency. The text-helper functions (`clean_tag`,
`split_concat`) are passed in to avoid a forced dependency on
pe_text_utils (the caller in prompt_enhancer.py already imports them
under the historical names).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


# rapidfuzz imports are deferred to find_closest_tag so this module
# stays importable for static checks even if rapidfuzz isn't installed.
# In the Forge install rapidfuzz is always present (install.py installs it).


# ── Constants ───────────────────────────────────────────────────────────

# Tokens that must retain underscores even when the tag-format prefers
# spaces (score_N, year_YYYY, source_*). Used by format_tag_out.
PRESERVE_UNDERSCORE_RE = re.compile(
    r"^(score_\d+(_up)?|year_\d{4}|source_[a-z]+)$"
)

# Common LLM tag mistakes → canonical Danbooru form. Applied before
# any other lookup so e.g. "1woman" doesn't get fuzzy-matched to
# something weird.
TAG_CORRECTIONS: Dict[str, str] = {
    "1man": "1boy",
    "1woman": "1girl",
    "1female": "1girl",
    "1male": "1boy",
    "man": "male_focus",
    "woman": "1girl",
    "female": "1girl",
    "male": "1boy",
    "girl": "1girl",
    "a_girl": "1girl",
    "boy": "1boy",
    "a_boy": "1boy",
    "2girl": "2girls",
    "2boy": "2boys",
    "2men": "2boys",
    "2women": "2girls",
    "3girl": "3girls",
    "3boy": "3boys",
}

# Universal Danbooru subject/count tags (same across Illustrious / NoobAI /
# Pony / Anima). Used by reorder() to bucket tags into the standard
# Danbooru order.
SUBJECT_TAGS: Set[str] = {
    "1girl", "1girls", "2girls", "3girls", "4girls", "5girls",
    "6+girls", "multiple_girls",
    "1boy", "1man", "2boys", "3boys", "4boys", "5boys",
    "6+boys", "multiple_boys",
    "1other", "solo", "no_humans", "male_focus", "female_focus",
    "1woman", "man", "woman", "girl", "boy",
}


# ── Building blocks ─────────────────────────────────────────────────────


def find_closest_tag(
    tag: str,
    valid_tags: Iterable[str],
    max_distance: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """Find the closest valid tag for an invalid one.

    Priority:
      1. Prefix substring match — our tag is a prefix of a valid tag
         (e.g. "highres" → "highres_(imageboard)"). Only for tags 5+
         chars and where valid_len is in [tag_len, tag_len*2].
      2. Levenshtein distance via rapidfuzz (C-accelerated; ~100x
         faster than pure Python on a 142k-tag DB). Skipped for tags
         under 5 chars where edit distance 1-3 produces nonsense.

    Returns (corrected_tag, None) on hit, or (None, original_tag)
    on miss.
    """
    tag_len = len(tag)

    if tag_len >= 5:
        best_prefix = None
        best_prefix_len = 999
        for valid in valid_tags:
            valid_len = len(valid)
            if valid_len > 0 and tag_len / valid_len >= 0.5 and valid.startswith(tag):
                if valid_len < best_prefix_len:
                    best_prefix = valid
                    best_prefix_len = valid_len
        if best_prefix:
            return best_prefix, None

    if tag_len < 5:
        return None, tag

    from rapidfuzz import distance as _rf_distance
    from rapidfuzz.process import extractOne as _rf_extract_one
    result = _rf_extract_one(
        tag, valid_tags,
        scorer=_rf_distance.Levenshtein.distance,
        score_cutoff=max_distance,
    )
    if result is not None:
        return result[0], None

    return None, tag


def format_tag_out(tag: str, use_underscores: bool) -> str:
    """Apply the tag-format's underscore/space convention, preserving
    underscores for canonical tokens like score_7 that conventionally
    keep an underscore regardless of format."""
    if use_underscores:
        return tag
    if PRESERVE_UNDERSCORE_RE.match(tag):
        return tag
    return tag.replace("_", " ")


def _escape_sd_parens(tag: str) -> str:
    """Escape parentheses in disambig-suffixed tags so SD doesn't read
    them as emphasis/weight syntax. `artist_(style)` → `artist_\\(style\\)`."""
    if "(" in tag and "_(" in tag:
        return tag.replace("(", r"\(").replace(")", r"\)")
    return tag


# ── validate ────────────────────────────────────────────────────────────


def validate(
    tags_str: str,
    fmt_config: Dict[str, Any],
    valid_tags: Iterable[str],
    aliases: Dict[str, str],
    *,
    mode: str = "Fuzzy",
    clean_tag: Callable[[str], str] = lambda t: t.strip(),
) -> Tuple[str, Dict[str, Any]]:
    """Validate and correct tags against the format DB.

    Modes:
      Fuzzy        — exact + alias + fuzzy correction, keep unrecognized
      Fuzzy Strict — exact + alias + fuzzy correction, drop unrecognized

    `clean_tag` is the per-tag pre-cleaner (LLM-meta stripper); pass in
    pe_text_utils.clean_tag at the callsite. Default is a strip-only
    no-op so this module doesn't force a pe_text_utils import.

    Returns (corrected_csv_string, stats_dict).
    """
    valid_set = (
        valid_tags if isinstance(valid_tags, (set, frozenset))
        else set(valid_tags)
    )
    if not valid_set:
        return tags_str, {"error": "No tag database available"}

    db_filename = fmt_config.get("tag_db", "")
    use_underscores = fmt_config.get("use_underscores", False)
    use_fuzzy = mode in ("Fuzzy", "Fuzzy Strict")
    drop_invalid = mode == "Fuzzy Strict"
    whitelist = fmt_config.get("whitelist_set", set())

    raw_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    result_tags: List[str] = []
    corrected = 0
    dropped = 0
    kept = 0

    for tag in raw_tags:
        tag = clean_tag(tag)
        if not tag:
            continue

        # DB + corrections both keyed by underscored form.
        # use_underscores controls OUTPUT, not lookup.
        lookup = tag.replace(" ", "_")

        # 1. Common LLM mistakes
        if lookup in TAG_CORRECTIONS:
            result_tags.append(format_tag_out(TAG_CORRECTIONS[lookup], use_underscores))
            corrected += 1
            continue

        # 2. Whitelist (per-format quality + leading + rating)
        if lookup in whitelist:
            result_tags.append(format_tag_out(lookup, use_underscores))
            continue

        # 3. Exact match
        if lookup in valid_set:
            result_tags.append(tag)
            continue

        # 4. Alias match
        if lookup in aliases:
            result_tags.append(format_tag_out(aliases[lookup], use_underscores))
            corrected += 1
            continue

        # 5. Prefix-disambig match: "artist_name" → "artist_name_(style)"
        # Only multi-word lookups (avoid "high" → "high_(hgih)").
        if "_" in lookup:
            prefix = lookup + "_("
            prefix_matches = [v for v in valid_set if v.startswith(prefix)]
            if len(prefix_matches) == 1:
                result_tags.append(format_tag_out(prefix_matches[0], use_underscores))
                corrected += 1
                continue

        # 6. Fuzzy match (only if requested)
        if use_fuzzy:
            match, _ = find_closest_tag(lookup, valid_set)
            if match:
                result_tags.append(format_tag_out(match, use_underscores))
                corrected += 1
                continue

        # 7. Unrecognized
        if drop_invalid:
            dropped += 1
        else:
            result_tags.append(tag)
            kept += 1

    # 8. Reorder + escape SD parens.
    result_tags = reorder(
        result_tags, fmt_config,
        universal_quality_tags=fmt_config.get("quality_tags", set()),
    )
    result_tags = [_escape_sd_parens(t) for t in result_tags]

    stats = {
        "corrected": corrected,
        "dropped": dropped,
        "kept_invalid": kept,
        "total": len(result_tags),
    }
    return ", ".join(result_tags), stats


# ── reorder ─────────────────────────────────────────────────────────────


def reorder(
    tags: List[str],
    fmt_config: Dict[str, Any],
    universal_quality_tags: Set[str],
) -> List[str]:
    """Reorder tags into standard Danbooru convention.

    Order: quality → leading → subjects → rest → rating.
    The quality / leading / rating buckets come from the tag-format
    config. Deduplicates within the input. If multiple rating tags
    appear, keeps the LAST one (the LLM most often corrects itself).
    If `no_humans` appears alongside numbered-subject tags, drops the
    numbered ones (the model emitted both because it's confused).
    """
    quality_set = fmt_config.get("quality_tags", universal_quality_tags)
    leading_set = fmt_config.get("leading_tags", set())
    rating_set = fmt_config.get("rating_tags", set())

    seen = set()
    quality, leading, subjects, rating, rest = [], [], [], [], []

    for tag in tags:
        lookup = tag.replace(" ", "_")
        if lookup in seen:
            continue
        seen.add(lookup)

        if lookup in quality_set:
            quality.append(tag)
        elif lookup in leading_set:
            leading.append(tag)
        elif lookup in SUBJECT_TAGS:
            subjects.append(tag)
        elif lookup in rating_set:
            rating.append(tag)
        else:
            rest.append(tag)

    if len(rating) > 1:
        rating = [rating[-1]]

    subject_lookups = {t.replace(" ", "_") for t in subjects}
    if "no_humans" in subject_lookups:
        subjects = [t for t in subjects if t.replace(" ", "_") == "no_humans"]

    return quality + leading + subjects + rest + rating


# ── postprocess (top-level pipeline) ────────────────────────────────────


def postprocess(
    tag_str: str,
    fmt_config: Dict[str, Any],
    valid_tags: Iterable[str],
    aliases: Dict[str, str],
    *,
    validation_mode: str,
    clean_tag: Callable[[str], str] = lambda t: t.strip(),
    split_concat: Callable[[str], str] = lambda t: t,
    universal_quality_tags: Set[str] = frozenset(),
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Apply the full tag post-processing pipeline.

    Steps: split-concatenated → space/underscore normalize → validate
    (which also reorders + escapes parens). Validation is skipped
    when validation_mode is "Off" (returns processed-but-unvalidated
    tags + None stats).

    `clean_tag` and `split_concat` are passed in (typically
    pe_text_utils.clean_tag and pe_text_utils.split_concatenated_tag)
    so this module doesn't force a pe_text_utils import.

    `universal_quality_tags` is the cross-format quality token set,
    used as the fallback when the tag-format config doesn't supply
    its own quality_tags entry.
    """
    use_underscores = fmt_config.get("use_underscores", False)
    tag_str = ", ".join(
        split_concat(t.strip()).replace(" ", "_") if use_underscores
        else split_concat(t.strip())
        for t in tag_str.split(",") if t.strip()
    )
    if validation_mode != "Off":
        return validate(
            tag_str, fmt_config, valid_tags, aliases,
            mode=validation_mode, clean_tag=clean_tag,
        )
    return tag_str, None
