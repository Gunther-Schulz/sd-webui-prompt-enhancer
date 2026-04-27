"""Anima-pipeline functions called by mode handlers.

These functions wrap anima_tagger features but expect the stack to
already be loaded (callers use `pe_anima_glue.stack.get_stack()`).

  make_query_expander(call_llm, temperature=0.3, seed=-1) → callable | None
      Builds a callable that converts a source prompt into a tag-style
      concept list for richer FAISS-shortlist retrieval. Returns None
      when query expansion is disabled in settings or anima_tagger
      isn't importable. The LLM call function is passed in (typically
      pe_llm_layer.call_llm) so this module doesn't depend on the LLM
      layer directly.

  tag_from_draft(stack, draft_str, *, safety="safe", use_underscores=False,
                 shortlist=None) → (csv, stats)
      Validate + rule-layer an LLM tag draft via anima_tagger. Drop-in
      replacement for the rapidfuzz-path postprocess when RAG mode is
      active. Passes the shortlist through to the validator for
      category-aware alias resolution.

  general_tag_candidates(stack, prose, k=60, min_post_count=100) → list[str]
      Retrieve top-K general-category Danbooru tags semantically
      matching the prose. Used by the V13 grounded Pass 2 SP — the
      LLM is told to prefer these candidate tags rather than invent
      novel compounds.

  candidate_fragment_for_tag_sp(candidates) → str
      Format the candidate list as a Pass-2-system-prompt fragment.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from . import stack as _stack

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.pipeline")


# ── Query expander ──────────────────────────────────────────────────────


def make_query_expander(
    call_llm: Callable[..., str],
    temperature: float = 0.3,
    seed: int = -1,
) -> Optional[Callable[..., str]]:
    """Build a query-expander callable backed by `call_llm` for shortlist
    retrieval. Returns None when:
      - query expansion is disabled in settings, OR
      - anima_tagger.query_expansion isn't importable.

    Expansion is a cheap LLM pass (~1 s on 9b) that converts a raw
    source prompt into a tag-style concept list. The expanded form
    embeds into a denser, thematically-aligned FAISS query than the
    raw prompt would produce — name-collision risk drops, and the
    shortlist surfaces actually-relevant artists/characters.

    `call_llm` is the LLM call function (typically
    pe_llm_layer.call_llm). Passed in so this module doesn't have a
    pe_llm_layer dependency.
    """
    if not bool(_stack.opt("anima_tagger_query_expansion", True)):
        return None
    try:
        from anima_tagger.query_expansion import expand_query    # type: ignore
    except ImportError:
        return None

    def _expander(source: str, modifier_keywords: Optional[str]) -> str:
        def _oneshot(sys_prompt: str, user_msg: str) -> str:
            # call_llm RETURNS a string (not a generator). A previous
            # version of this bridge iterated `for chunk in call_llm(...)`,
            # which iterated character-by-character over the return
            # string and silently fed only the LAST character into FAISS
            # as the shortlist query. That corrupted every retrieval and
            # produced user-visible weird tags (sparse_leg_hair,
            # simple_fish, overhead_lights, etc.) because FAISS on a
            # single char pulls near-random entries.
            try:
                return call_llm(user_msg, sys_prompt, temperature, timeout=30, seed=seed) or ""
            except Exception:
                return ""

        return expand_query(source, _oneshot, modifier_keywords=modifier_keywords)

    return _expander


# ── Tag-from-draft (Anima validate path) ────────────────────────────────


def tag_from_draft(
    stack: Any,
    draft_str: str,
    *,
    safety: str = "safe",
    use_underscores: bool = False,
    shortlist: Any = None,
) -> Tuple[str, dict]:
    """Validate + rule-layer an LLM tag draft via anima_tagger.

    Drop-in replacement for the rapidfuzz-path postprocess when RAG
    mode is active. Passes the shortlist (when present) to the
    validator for category-aware alias resolution.

    Returns (corrected_csv, stats_dict).
    """
    compound_split = bool(_stack.opt("anima_tagger_compound_split", True))
    draft_tokens_preview = [t.strip() for t in draft_str.split(",") if t.strip()]
    print(f"[PromptEnhancer] Anima validate: draft_tokens={len(draft_tokens_preview)}, "
          f"compound_split={compound_split}, shortlist="
          f"{len(shortlist.artists) if shortlist else 0}a/"
          f"{len(shortlist.characters) if shortlist else 0}c/"
          f"{len(shortlist.series) if shortlist else 0}s")
    if len(draft_tokens_preview) <= 25:
        print(f"[PromptEnhancer]   draft: {draft_tokens_preview}")
    tags_list = stack.tagger.tag_from_draft(
        draft_str, safety=safety, use_underscores=use_underscores,
        shortlist=shortlist, compound_split=compound_split,
    )
    draft_token_count = len(draft_tokens_preview)
    stats = {
        "corrected": 0,
        "dropped": max(0, draft_token_count - len(tags_list)),
        "kept_invalid": 0,
        "total": len(tags_list),
    }
    print(f"[PromptEnhancer]   → {len(tags_list)} kept, raw dropped={stats['dropped']}")
    return ", ".join(tags_list), stats


# ── General-tag candidates (V13 Pass-2 grounding) ───────────────────────


def general_tag_candidates(
    stack: Any,
    prose: str,
    k: int = 60,
    min_post_count: int = 100,
) -> List[str]:
    """Retrieve top-k general-category Danbooru tags semantically matching
    `prose`. Used by Pass 2 grounding to constrain the LLM's tag output
    to scene-relevant real Danbooru vocabulary.

    Root-cause fix for the weird-tag problem observed in production:
    Pass 2 LLM freely invents tokens like `sparse_leg_hair`,
    `overhead_lights`, `simple_fish` that ARE real Danbooru tags but
    don't belong in the scene. Grounding Pass 2 in FAISS-retrieved
    candidates lets it SELECT from scene-relevant vocabulary instead.
    Returns underscore-form names (DB canonical); caller converts to
    space form for display when needed.

    Implementation: bypasses stack.retriever.retrieve() because that
    path's reranker + default retrieve_k=300 tends to return only a
    handful of general tags when the top-300 dense hits are dominated
    by thematic artist/character/series entries. We need broad
    general-tag coverage: dense search with retrieve_k=3000, filter
    to category=0 + min_post_count, sort by dense score, take top-k.
    No reranker (its query-pair scoring is designed for shorter
    queries than a multi-sentence prose).
    """
    if stack is None or getattr(stack, "retriever", None) is None:
        return []
    try:
        r = stack.retriever
        q_vec = r.embedder.encode_one(prose)
        scores, ids = r.index.search(q_vec, 3000)
        dense_ids = [int(i) for i in ids[0] if i >= 0]
        tags = r.db.get_by_ids(dense_ids)
        score_by_id = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}
        seen: set = set()
        pool: list = []
        for t in tags:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            if t["category"] != 0:
                continue
            if t["post_count"] < min_post_count:
                continue
            pool.append(t)
        pool.sort(key=lambda t: (score_by_id.get(t["id"], 0.0), t["post_count"]), reverse=True)
        return [t["name"] for t in pool[:k]]
    except Exception as e:
        logger.warning(f"general_tag_candidates: retrieval failed: {e}")
        return []


def candidate_fragment_for_tag_sp(candidates: List[str]) -> str:
    """Format the candidate tag list as a Pass 2 system-prompt fragment.

    Prose-friendly phrasing — tells the LLM the list is a pre-curated
    scene-relevant vocabulary and how it may safely deviate (named
    subjects, required format tokens).
    """
    if not candidates:
        return ""
    display = ", ".join(n.replace("_", " ") for n in candidates)
    return (
        "\nCANDIDATE TAGS FOR THIS SCENE (pre-retrieved from the Danbooru "
        "DB against the prose). Strongly prefer tags from this list — "
        "they are scene-relevant and real:\n\n"
        f"  {display}\n\n"
        "You may emit tags OUTSIDE this list ONLY when:\n"
        "  (a) the tag is required by the format (masterpiece, best quality, "
        "score_N, safe/sensitive/nsfw/explicit, 1girl/1boy/1other);\n"
        "  (b) the prose explicitly names a character, series, or artist not "
        "in the list;\n"
        "  (c) a concept is literally named in the prose but happens to be "
        "missing from the list.\n"
        "For any other concept, pick the closest match from the candidate "
        "list instead of inventing a novel compound tag. If no candidate "
        "fits a concept, omit it rather than inventing.\n"
    )
