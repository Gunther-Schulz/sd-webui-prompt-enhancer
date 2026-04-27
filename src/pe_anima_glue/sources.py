"""Source-pick + slot-retrieval functions.

  resolve_source(source_spec, seed, *, stack=None, query="") → dict | None
      Seed-pick a real Danbooru tag from the DB matching the
      source_spec. Two mechanisms (exactly one per modifier):
        db_pattern  — regex matched against category=general names.
                     Fast, no models. Used by Random Era / Flower /
                     Food / Animal / Constellation / Tarot / Symbol.
        db_retrieve — FAISS retrieval against a specific Danbooru
                     category (1=artist, 3=copyright, 4=character).
                     Requires the loaded anima stack. Used by ◆ on
                     Random Artist / Random Franchise / Random
                     Character. Returns None without a stack so the
                     caller can defer + re-resolve later.

  resolve_deferred_sources(mods, seed, stack, query) → int
      Second-pass resolver for entries that needed the loaded stack
      (db_retrieve). Mutates `mods` in place, returns count resolved.

  retrieve_prose_slot(stack, prose, slot, seed=0) → str | None
      Pick ONE DB tag for `slot` from the retriever's top-K matches
      to the prose. Used by the slot-fill safety net for
      target_slot:-flagged modifiers (Random Artist → artist,
      Random Franchise → copyright). Seed-driven so reproducible.
"""

from __future__ import annotations

import logging
import os
import random as _random
import re as _re
import sqlite3 as _sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.sources")


# Map of target_slot → (category_code, min_post_count, prefer_popularity).
# Matches the category IDs in anima_tagger.config. Slots without an
# entry here aren't eligible for retrieval fill (e.g. general-concept
# slots like "pose" or "setting" — those live in CAT_GENERAL and are
# already covered by the LLM draft + compound_split).
#
# prefer_popularity=True means "among semantically-relevant candidates,
# pick the most-popular one" — useful for copyright/franchise where we
# usually want the mainstream series name (vocaloid, pokemon), not a
# niche long-title reranker favorite.
SLOT_TO_CATEGORY: Dict[str, Dict[str, Any]] = {
    "artist":    {"category": 1, "min_post": 500,  "prefer_popularity": False},
    "copyright": {"category": 3, "min_post": 500,  "prefer_popularity": True},
}


# ── Single source pick ──────────────────────────────────────────────────


def resolve_source(
    source_spec: dict,
    seed: int,
    *,
    stack: Any = None,
    query: str = "",
) -> Optional[Dict[str, Any]]:
    """Seed-pick a real Danbooru tag from the DB matching `source_spec`.

    YAML schemas:
        source: { db_pattern: "^\\d{4}s_\\(style\\)$", min_post_count: 50,
                  template: "Set scene in {display}." }
        source: { db_retrieve: { category: 1, min_post_count: 500,
                                 final_k: 10 },
                  template: "Render in the style of {display}." }

    Returns {name, display, behavioral, keywords, post_count,
    pool_size} or None on failure. db_retrieve returns None when
    `stack` is missing (caller re-resolves after models load).
    """
    try:
        from anima_tagger.config import TAG_DB_PATH as _TAG_DB_PATH    # type: ignore
    except Exception:
        return None
    template = source_spec.get("template") or "Apply {display}."
    rng = _random.Random(int(seed) if seed not in (None, -1) else _random.randint(0, 2**31 - 1))

    picked_name: Optional[str] = None
    picked_pc: int = 0
    pool_size: int = 0

    if "db_pattern" in source_spec:
        pattern = source_spec["db_pattern"]
        if not os.path.isfile(_TAG_DB_PATH):
            return None
        min_pc = int(source_spec.get("min_post_count", 50))
        try:
            rx = _re.compile(pattern)
        except _re.error as e:
            logger.warning(f"resolve_source: bad regex {pattern!r}: {e}")
            return None
        try:
            conn = _sqlite3.connect(_TAG_DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "SELECT name, post_count FROM tags WHERE category=0 AND post_count >= ?",
                (min_pc,),
            )
            pool = [(n, pc) for n, pc in cur.fetchall() if rx.search(n)]
            conn.close()
        except Exception as e:
            logger.warning(f"resolve_source: DB query failed: {e}")
            return None
        if not pool:
            return None
        picked_name, picked_pc = rng.choice(pool)
        pool_size = len(pool)

    elif "db_retrieve" in source_spec:
        spec = source_spec["db_retrieve"]
        if stack is None or getattr(stack, "retriever", None) is None:
            return None
        category = spec.get("category")
        min_pc = int(spec.get("min_post_count", 500))
        final_k = int(spec.get("final_k", 10))
        prefer_pop = bool(spec.get("prefer_popularity", False))
        q = (query or "").strip() or "image"
        try:
            cands = stack.retriever.retrieve(
                q, retrieve_k=200, final_k=final_k,
                category=category, min_post_count=min_pc,
            )
        except Exception as e:
            logger.warning(f"resolve_source: db_retrieve failed: {e}")
            return None
        if not cands:
            return None
        if prefer_pop:
            chosen = max(cands, key=lambda c: getattr(c, "post_count", 0))
        else:
            chosen = rng.choice(cands)
        picked_name = chosen.name
        picked_pc = getattr(chosen, "post_count", 0)
        pool_size = len(cands)

    else:
        return None

    # display = human form: underscores→spaces, strip Danbooru disambiguator suffix.
    display = _re.sub(r"_\([^)]+\)$", "", picked_name).replace("_", " ")
    keywords = picked_name.replace("_", " ")
    try:
        behavioral = template.format(name=picked_name, display=display)
    except Exception as e:
        logger.warning(f"resolve_source: bad template {template!r}: {e}")
        behavioral = f"Apply {display}."
    return {
        "name": picked_name,
        "display": display,
        "behavioral": behavioral,
        "keywords": keywords,
        "post_count": picked_pc,
        "pool_size": pool_size,
    }


# ── Deferred-sources resolver ───────────────────────────────────────────


def resolve_deferred_sources(
    mods: List[Tuple[str, Dict[str, Any]]],
    seed: int,
    stack: Any,
    query: str,
) -> int:
    """Second-pass source resolution for entries that needed the loaded
    stack (db_retrieve). Mutates `mods` in place. Called by the mode
    handlers after the RAG models are loaded. Returns count of entries
    newly resolved.

    `mods` is the (name, entry) list from _collect_modifiers. Entries
    with `source.db_retrieve` and no `_resolved_from_source` yet are
    candidates — the pre-models pass couldn't fill them because the
    retriever wasn't available. Now it is.
    """
    resolved_count = 0
    for i, (name, entry) in enumerate(mods or []):
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if not source:
            continue
        if entry.get("_resolved_from_source"):
            continue  # already resolved via db_pattern
        if "db_retrieve" not in source:
            continue
        picked = resolve_source(source, seed, stack=stack, query=query)
        if not picked:
            continue
        updated = dict(entry)
        updated["behavioral"] = picked["behavioral"]
        updated["keywords"] = picked["keywords"]
        updated["_resolved_from_source"] = picked["name"]
        mods[i] = (name, updated)
        resolved_count += 1
        print(f"[PromptEnhancer] Random pick ({name}): "
              f"{picked['name']} (pool={picked['pool_size']}, "
              f"post_count={picked['post_count']})")
    return resolved_count


# ── Prose-slot retrieval ────────────────────────────────────────────────


def retrieve_prose_slot(
    stack: Any,
    prose: str,
    slot: str,
    seed: int = 0,
) -> Optional[str]:
    """Pick ONE DB tag for `slot` from semantically-closest candidates
    to the prose. Requires an open `stack.models()` context.

    Determinism: retrieval is always deterministic for a given prose.
    For artist (and similar random-X slots), we pull top-K and pick
    one using the user's seed — same seed gives the same artist
    (reproducible), different seeds give variation. Without this,
    `@e.o.` dominates every NSFW prompt because it's always the top
    reranker pick.

    For slots flagged `prefer_popularity` (copyright/franchise),
    return the most-popular candidate instead of a seed pick — users
    usually want `vocaloid` over a niche reranker-favorite series.
    """
    entry = SLOT_TO_CATEGORY.get(slot)
    if not entry or stack is None or getattr(stack, "retriever", None) is None:
        return None
    category = entry["category"]
    min_post = entry["min_post"]
    prefer_pop = entry.get("prefer_popularity", False)
    try:
        final_k = 10
        cands = stack.retriever.retrieve(
            prose, retrieve_k=200, final_k=final_k,
            category=category, min_post_count=min_post,
        )
        if not cands:
            return None
        if prefer_pop:
            return max(cands, key=lambda c: c.post_count).name
        rng = _random.Random(int(seed) if seed not in (None, -1) else _random.randint(0, 2**31 - 1))
        return rng.choice(cands).name
    except Exception as e:
        logger.warning(f"prose slot retrieval failed ({slot}): {e}")
        return None
