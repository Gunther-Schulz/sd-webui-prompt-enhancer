"""Tag filtering helpers driven by the anima DB.

  tags_have_category(tag_csv, stack, category) → bool
      Quick membership check — does any tag in `tag_csv` resolve to
      a DB record of the given category? Used by mode handlers to
      decide whether the slot-fill pass needs to fire.

  filter_to_structural_tags(tag_csv, stack, fmt_config) → str
      V14 Hybrid filter + V19 carve-outs: keep structural tags
      (artist / character / copyright + format whitelist + subject-
      count tags), with carefully-picked CAT_GENERAL exceptions:
        - the format's `general_tags_allowlist` (rendering-control
          and booru-trained-convention tokens)
        - DB co-occurrence top-K for surviving structural tags
          (character signatures: hatsune_miku → leek)

      Drops everything else (CAT_META + uncovered CAT_GENERAL),
      because the Hybrid prose carries those concepts and Anima's
      Qwen3 text encoder handles natural language natively.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from . import stack as _stack

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.filters")


SUBJECT_COUNT_RE = re.compile(r"^\d+(girl|boy|other)s?$")

# Structural Danbooru categories (artist/character/copyright). CAT_GENERAL=0
# and CAT_META=5 are dropped by V14 — that's where "sparse leg hair" /
# "simple fish" / "overhead lights" live after FAISS retrieval. The
# carve-out passes (general_tags_allowlist + cooccurrence) selectively
# rescue some CAT_GENERAL tokens.
STRUCTURAL_CATEGORIES: set = {1, 3, 4}


def tags_have_category(tag_csv: str, stack: Any, category: int) -> bool:
    """True if any tag in the comma-separated list resolves to a DB
    record of the given category. Trims `@` prefix since artist tokens
    carry it."""
    if not tag_csv or not stack or not stack.db:
        return False
    for t in tag_csv.split(","):
        norm = t.strip().lstrip("@").lower().replace(" ", "_").replace("-", "_")
        if not norm:
            continue
        rec = stack.db.get_by_name(norm)
        if rec and rec.get("category") == category:
            return True
    return False


def filter_to_structural_tags(
    tag_csv: str,
    stack: Any,
    fmt_config: Dict[str, Any],
) -> str:
    """V14 Hybrid filter + V19 carve-outs: keep structural tags + a
    small curated set of general tags that carry tag-level conditioning.

    V14 baseline: drop CAT_GENERAL and CAT_META. Keep only format-
    whitelist (quality/safety/rating), subject-count (1girl/1boy/2girls/…),
    and Danbooru categories {artist, character, copyright}. The Hybrid
    prose carries everything else since Anima's Qwen3 text encoder
    handles natural language natively.

    V19 carve-outs on top of V14:
      (A) Co-occurrence — for each surviving structural tag, retain
          CAT_GENERAL tags the DB records as highly co-occurring
          (p_given_src ≥ min_prob). Fixes character-signature objects
          (e.g. hatsune_miku → leek retained).
      (B) Format allowlist — retain CAT_GENERAL tags listed in the
          format's `general_tags_allowlist` yaml field. These are
          rendering-control / booru-trained-convention tokens
          (framing, named poses, clothing archetypes, anime-specific
          visual tokens, rendering techniques) that carry tag-level
          conditioning prose cannot fully replicate.

    Must be called with a stack that has an active DB (Anima format +
    RAG). Co-occurrence carve-out is a no-op if stack.cooccurrence is
    unavailable.
    """
    if not tag_csv:
        return tag_csv
    whitelist: set = set()
    whitelist |= {str(t).lower() for t in (fmt_config.get("quality_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt_config.get("leading_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt_config.get("rating_tags") or set())}
    general_allow: set = set(fmt_config.get("general_tags_allowlist") or set())

    # Parse once, normalize, classify per tag
    parsed = []
    for t in tag_csv.split(","):
        t = t.strip()
        if not t:
            continue
        norm = t.lstrip("@").lower().replace(" ", "_").replace("-", "_")
        if norm:
            parsed.append((t, norm))

    # Pass 1: keep structural tags; collect their names for cooccurrence lookup
    kept_indices: set = set()
    structural_names: list = []
    for i, (t, norm) in enumerate(parsed):
        if norm in whitelist or SUBJECT_COUNT_RE.match(norm):
            kept_indices.add(i)
            continue
        if stack is not None and stack.db is not None:
            rec = stack.db.get_by_name(norm)
            if rec and rec.get("category") in STRUCTURAL_CATEGORIES:
                kept_indices.add(i)
                structural_names.append(norm)

    # Pass 2a: build cooccurrence allowlist from surviving structural tags
    cooc_allow: set = set()
    if (stack is not None and getattr(stack, "cooccurrence", None) is not None
            and structural_names):
        top_k = int(_stack.opt("anima_tagger_v19_cooc_top_k", 20))
        min_prob = float(_stack.opt("anima_tagger_v19_cooc_min_prob", 0.3))
        for sname in structural_names:
            try:
                hits = stack.cooccurrence.top_for(
                    sname, category=0, top_k=top_k, min_prob=min_prob,
                )
            except Exception:
                hits = []
            for hit in hits:
                cooc_allow.add(hit.get("tag", ""))

    # Pass 2b: carve-outs — retain CAT_GENERAL tags in format allowlist or cooc allowlist
    if general_allow or cooc_allow:
        for i, (t, norm) in enumerate(parsed):
            if i in kept_indices:
                continue
            if norm not in general_allow and norm not in cooc_allow:
                continue
            # Sanity: verify it's actually CAT_GENERAL (not meta etc.)
            if stack is not None and stack.db is not None:
                rec = stack.db.get_by_name(norm)
                if rec is None or rec.get("category") != 0:
                    continue
            kept_indices.add(i)

    return ", ".join(parsed[i][0] for i in sorted(kept_indices))
