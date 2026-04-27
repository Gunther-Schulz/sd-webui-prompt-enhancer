import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request   # for tag DB download (the LLM path uses pe_llm_layer)

from rapidfuzz import distance as _rf_distance
from rapidfuzz.process import extractOne as _rf_extract_one

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory ─────────────────────────────────────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODIFIERS_DIR = os.path.join(_EXT_DIR, "modifiers")

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"
_TAGS_DIR = os.path.join(_EXT_DIR, "tags")
_TAG_FORMATS_DIR = os.path.join(_EXT_DIR, "tag-formats")

_tag_formats = {}    # format_name -> {system_prompt, use_underscores, tag_db, tag_db_url}
_tag_databases = {}  # db_filename -> set of valid tags


# ── anima_tagger (retrieval-augmented Anima pipeline) ─────────────────────
# Lazy-loaded on first Anima use. Falls back to the rapidfuzz path when:
#   - the data artefacts aren't built yet (user hasn't run build_index.py),
#   - tag_fmt is not "Anima",
#   - or the "Anima Tagger" Forge setting is disabled.
# Forge's script loader (modules/scripts.py) snapshots sys.path before
# loading each extension script and restores it afterward, so a
# module-level sys.path.insert here would be wiped out before the lazy
# import runs. The path is inserted inside _get_anima_stack() instead.
_ANIMA_SRC_PATH = os.path.join(_EXT_DIR, "src")

_anima_stack = None                # cached AnimaStack (db + index open)
_anima_load_attempted = False      # set True once we've tried to avoid repeat failures


# ── pe_llm_layer (in-process LLM via llama-cpp-python) ────────────────────
# Wraps src/llm_runner. Replaces the old Ollama HTTP path. Imported eagerly:
# unlike anima_tagger (which has a rapidfuzz fallback), the LLM is required
# for every generation mode, so failing to import is a hard error.
if _ANIMA_SRC_PATH not in sys.path:
    sys.path.insert(0, _ANIMA_SRC_PATH)
from pe_llm_layer import (
    call_llm as _call_llm,
    stream_llm as _call_llm_progress,
    multi_sample_prose as _multi_sample_prose,
    get_llm_status as _get_llm_status,
    TruncatedOutput as _TruncatedError,
)
from pe_text_utils import (
    clean_output as _clean_output,
    clean_tag as _clean_tag,
    split_concatenated_tag as _split_concatenated_tag,
    split_positive_negative as _split_positive_negative,
)
from pe_data import bases as _pe_bases, prompts as _pe_prompts
from pe_data._util import load_yaml_or_json as _load_file, get_local_dirs as _get_local_dirs


def _anima_opt(key: str, default):
    """Read an anima_tagger_* option from shared.opts, fallback to default."""
    try:
        from modules import shared
        return shared.opts.data.get(key, default)
    except Exception:
        return default


_anima_load_error: str | None = None


def _get_anima_stack():
    """Return a loaded AnimaStack or None if artefacts missing / disabled.

    DB + faiss index open on first call and stay cached. Models (bge-m3
    + reranker) still load on-demand via `stack.models()`.
    """
    global _anima_stack, _anima_load_attempted, _anima_load_error
    if _anima_stack is not None:
        return _anima_stack
    if _anima_load_attempted:
        return None
    _anima_load_attempted = True
    try:
        if _ANIMA_SRC_PATH not in sys.path:
            sys.path.insert(0, _ANIMA_SRC_PATH)
        from anima_tagger import load_all
        _anima_stack = load_all(
            semantic_threshold=float(_anima_opt("anima_tagger_semantic_threshold", 0.80)),
            semantic_min_post_count=int(_anima_opt("anima_tagger_semantic_min_post_count", 100)),
            enable_reranker=bool(_anima_opt("anima_tagger_enable_reranker", True)),
            enable_cooccurrence=bool(_anima_opt("anima_tagger_enable_cooccurrence", True)),
            device=str(_anima_opt("anima_tagger_device", "auto")),
        )
        logger.info("anima_tagger loaded (DB + FAISS ready)")
        return _anima_stack
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _anima_load_error = f"{type(e).__name__}: {e}"
        # Print the full traceback prominently so the user can diagnose
        # instead of seeing a generic "artefacts not loaded" message.
        print("\n" + "!" * 72, file=sys.stderr)
        print("  sd-webui-prompt-enhancer — anima_tagger load failed", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        for line in tb.rstrip().splitlines():
            print(f"  {line}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
        logger.error(f"anima_tagger failed to load: {_anima_load_error}")
        return None


def _rag_available_for(tag_fmt: str) -> tuple[bool, str]:
    """True + empty-string reason iff RAG pipeline can run for tag_fmt.
    Returns (False, reason_string) when unavailable — callers must
    ABORT (not fall back) when the user picked RAG and it's missing.
    Selecting RAG is an explicit user choice; silently swapping in a
    different validation mode hides real problems."""
    if tag_fmt != "Anima":
        return False, (
            f"RAG is currently Anima-only. Select the Anima tag format "
            f"or pick a different Tag Validation mode."
        )
    stack = _get_anima_stack()
    if stack is None:
        reason_path = os.path.join(_EXT_DIR, "data", ".rag_unavailable")
        detail = ""
        if os.path.exists(reason_path):
            try:
                with open(reason_path) as f:
                    detail = f" ({f.read().strip()})"
            except Exception:
                pass
        return False, (
            f"RAG artefacts not loaded{detail}. "
            f"Restart Forge to retry the HuggingFace download, or pick a "
            f"different Tag Validation mode."
        )
    return True, ""


def _use_anima_pipeline(tag_fmt: str, validation_mode: str = "RAG") -> bool:
    """True iff the user picked RAG AND it's available for this format."""
    if validation_mode != "RAG":
        return False
    ok, _ = _rag_available_for(tag_fmt)
    return ok


def _make_anima_query_expander(temperature=0.3, seed=-1):
    """Build a query-expander callable backed by _call_llm for shortlist retrieval.

    Expansion is a cheap LLM pass (~1 s on 9b) that converts the raw
    source prompt into a tag-style concept list, so the shortlist
    retriever embeds a denser, thematically-aligned query instead of
    a sparse source with name-collision risk.

    Returns None if disabled via setting.
    """
    if not bool(_anima_opt("anima_tagger_query_expansion", True)):
        return None
    try:
        from anima_tagger.query_expansion import expand_query, EXPANSION_SYSTEM_PROMPT
    except ImportError:
        return None

    def _expander(source: str, modifier_keywords: str | None) -> str:
        def _oneshot(sys_prompt: str, user_msg: str) -> str:
            # _call_llm RETURNS a string (not a generator). A previous
            # version of this bridge iterated `for chunk in _call_llm(...)`
            # which iterated character-by-character over the return string
            # — it silently fed only the LAST CHARACTER of the LLM's
            # expansion into FAISS as the shortlist query. That corrupted
            # every shortlist retrieval and produced user-visible weird
            # tags (sparse_leg_hair, simple_fish, overhead_lights, etc.)
            # because FAISS on a single char pulls near-random entries.
            try:
                return _call_llm(user_msg, sys_prompt, temperature, timeout=30, seed=seed) or ""
            except Exception:
                return ""

        return expand_query(source, _oneshot, modifier_keywords=modifier_keywords)

    return _expander


# Tier precedence — most permissive first. A modifier YAML entry declares
# its safety tier via a `safety_tier` field with one of these values.
# The highest-tier active modifier wins.
_SAFETY_TIER_ORDER = ("explicit", "nsfw", "sensitive", "safe")


def _anima_safety_from_modifiers(mod_list, source: str = "") -> str:
    """Resolve the safety tag from active modifiers via the `safety_tier`
    field declared on each modifier's YAML entry.

    Precedence: most permissive tier among active modifiers wins. If no
    active modifier declares a tier, defaults to `safe`. Rating detection
    from raw source text is deferred to the LLM (see V16 tag-extract SP).

    `source` argument is retained (unused) for call-site compatibility.

    Knowledge lives in data, not Python: the previous name-to-tier dict
    and production-style name set have been removed. A modifier's tier is
    declared on the modifier itself via YAML; adding a new modifier with
    a safety implication never requires a Python edit.
    """
    del source  # rating signal comes from modifiers' own declarations
    winning_index = len(_SAFETY_TIER_ORDER)  # higher = less permissive
    for _name, entry in mod_list:
        tier = entry.get("safety_tier") if isinstance(entry, dict) else None
        if tier in _SAFETY_TIER_ORDER:
            idx = _SAFETY_TIER_ORDER.index(tier)
            if idx < winning_index:
                winning_index = idx
    return _SAFETY_TIER_ORDER[winning_index] if winning_index < len(_SAFETY_TIER_ORDER) else "safe"


def _anima_tag_from_draft(stack, draft_str: str, safety: str = "safe",
                          use_underscores: bool = False,
                          shortlist=None) -> tuple[str, dict]:
    """Validate + rule-layer an LLM draft via anima_tagger.

    Drop-in replacement for _postprocess_tags when anima_tagger is
    active. Passes the shortlist (if any) to the validator for
    category-aware alias resolution.
    """
    compound_split = bool(_anima_opt("anima_tagger_compound_split", True))
    # Diagnostic: surface draft and validator outcome so integration
    # issues (short prose, validator dropping too much, compound_split
    # not helping) are visible in the Forge console. Cheap.
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


# Map of target_slot → (category_code, min_post_count, prefer_popularity).
# Matches the category IDs in anima_tagger.config. Slots without a
# category entry here aren't eligible for retrieval fill (e.g. general-
# concept slots like "pose" or "setting" — those live in CAT_GENERAL
# and are already covered by the LLM draft + compound_split).
#
# prefer_popularity = True means "among semantically-relevant candidates,
# pick the most-popular one" — useful for copyright/franchise where we
# usually want the mainstream series name (vocaloid, pokemon), not a
# niche long-title reranker favorite.
_SLOT_TO_CATEGORY = {
    "artist":    {"category": 1, "min_post": 500,  "prefer_popularity": False},
    "copyright": {"category": 3, "min_post": 500,  "prefer_popularity": True},
}


def _retrieve_prose_slot(stack, prose: str, slot: str, seed: int = 0):
    """Pick ONE DB tag for the given slot from semantically-closest
    candidates to the prose. Requires an open models() context.

    Determinism: retrieval is always deterministic for a given prose.
    For artist (and similar random-X slots), we pull top-K candidates
    and pick one using the user's seed — so the same seed gives the
    same artist (reproducible runs), but different seeds give genuine
    variation. Without this step, `@e.o.` dominates every NSFW prompt
    because it's always the top reranker pick.

    For slots flagged `prefer_popularity` (copyright/franchise), we
    prefer the mainstream option rather than the random pick — users
    usually want `vocaloid` over a niche reranker-favorite series.
    """
    entry = _SLOT_TO_CATEGORY.get(slot)
    if not entry or stack.retriever is None:
        return None
    category = entry["category"]
    min_post = entry["min_post"]
    prefer_pop = entry.get("prefer_popularity", False)
    try:
        # Pull a wider pool for variety; we'll pick one from it.
        final_k = 10
        cands = stack.retriever.retrieve(
            prose, retrieve_k=200, final_k=final_k,
            category=category, min_post_count=min_post,
        )
        if not cands:
            return None
        if prefer_pop:
            return max(cands, key=lambda c: c.post_count).name
        # Seed-driven pick from top-K — same seed reproducible,
        # different seeds yield different artists.
        import random as _random
        rng = _random.Random(int(seed) if seed not in (None, -1) else _random.randint(0, 2**31 - 1))
        return rng.choice(cands).name
    except Exception as e:
        logger.warning(f"prose slot retrieval failed ({slot}): {e}")
        return None


# UI badges for data-driven random mechanisms. Defined here (before
# _build_dropdown_data and _collect_modifiers use them at module load).
_BADGE_SOURCE = "\u25c6"      # ◆ filled diamond: source: pre-pick
_BADGE_TARGET_SLOT = "\u25c7" # ◇ hollow diamond: target_slot: post-fill


def _resolve_source(source_spec: dict, seed: int,
                    stack=None, query: str = "") -> dict | None:
    """Seed-pick a real Danbooru tag from the DB based on `source_spec`.

    Two mechanisms supported (exactly one per modifier):

        db_pattern:  regex matched against category=general names. Fast,
                     no models required. Used by Random Era, Random Flower,
                     Random Food, Random Animal, Random Constellation,
                     Random Tarot Card, Random Symbol.

        db_retrieve: FAISS retrieval against a specific Danbooru category
                     (1=artist, 3=copyright, 4=character). Requires the
                     anima_tagger stack with models loaded. Query is the
                     user's source prompt + modifier keywords. Seed-picks
                     from top-K scene-relevant candidates. Used by ◆ on
                     Random Artist / Random Franchise / Random Character.

    YAML schemas:
        source: { db_pattern: "^\\d{4}s_\\(style\\)$", min_post_count: 50,
                  template: "Set scene in {display}." }
        source: { db_retrieve: { category: 1, min_post_count: 500,
                                 final_k: 10 },
                  template: "Render in the style of {display}." }

    Returns dict {name, display, behavioral, keywords, post_count,
    pool_size} or None on failure. `db_retrieve` returns None when
    called without a stack (caller re-resolves after models load).
    """
    import re as _re
    import sqlite3 as _sqlite3
    try:
        from anima_tagger.config import TAG_DB_PATH as _TAG_DB_PATH
    except Exception:
        return None
    template = source_spec.get("template") or "Apply {display}."
    import random as _random
    rng = _random.Random(int(seed) if seed not in (None, -1) else _random.randint(0, 2**31 - 1))

    picked_name = None
    picked_pc = 0
    pool_size = 0

    if "db_pattern" in source_spec:
        pattern = source_spec["db_pattern"]
        if not os.path.isfile(_TAG_DB_PATH):
            return None
        min_pc = int(source_spec.get("min_post_count", 50))
        try:
            rx = _re.compile(pattern)
        except _re.error as e:
            logger.warning(f"_resolve_source: bad regex {pattern!r}: {e}")
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
            logger.warning(f"_resolve_source: DB query failed: {e}")
            return None
        if not pool:
            return None
        picked_name, picked_pc = rng.choice(pool)
        pool_size = len(pool)

    elif "db_retrieve" in source_spec:
        spec = source_spec["db_retrieve"]
        # db_retrieve requires the anima_tagger stack with models loaded.
        # When stack is absent (e.g. pre-RAG-load phase), return None so
        # the caller can re-resolve later.
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
            logger.warning(f"_resolve_source: db_retrieve failed: {e}")
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

    # display = human form: underscores→spaces, strip Danbooru _(X) disambiguator
    display = _re.sub(r"_\([^)]+\)$", "", picked_name).replace("_", " ")
    keywords = picked_name.replace("_", " ")
    try:
        behavioral = template.format(name=picked_name, display=display)
    except Exception as e:
        logger.warning(f"_resolve_source: bad template {template!r}: {e}")
        behavioral = f"Apply {display}."
    return {
        "name": picked_name,
        "display": display,
        "behavioral": behavioral,
        "keywords": keywords,
        "post_count": picked_pc,
        "pool_size": pool_size,
    }


def _general_tag_candidates(stack, prose: str, k: int = 60,
                            min_post_count: int = 100) -> list[str]:
    """Retrieve top-k general-category Danbooru tags semantically matching
    the prose. Used by Pass 2 grounding to constrain the LLM's tag output
    to scene-relevant real Danbooru vocabulary.

    Root-cause fix for the weird-tag problem observed in production:
    Pass 2 LLM freely invents tokens like `sparse_leg_hair`,
    `overhead_lights`, `simple_fish` that ARE real Danbooru tags but
    don't belong in the scene. Grounding Pass 2 in FAISS-retrieved
    candidates lets it SELECT from scene-relevant vocabulary instead of
    emitting arbitrary compounds. Returns underscore-form names (DB
    canonical), caller converts to space form for display.

    Implementation note: bypasses the default stack.retriever.retrieve()
    path because that path's reranker + default retrieve_k=300 tends to
    return only a handful of general tags when the top-300 dense hits
    are dominated by thematic artist/character/series entries. For V13
    grounding we want BROAD general-tag coverage: dense search with
    large retrieve_k, filter to category=0 + min_post_count, sort by
    dense score, take top-k. No reranker (its query-pair scoring is
    designed for shorter queries than a multi-sentence prose).
    """
    if stack is None or getattr(stack, "retriever", None) is None:
        return []
    try:
        r = stack.retriever
        q_vec = r.embedder.encode_one(prose)
        # retrieve_k=3000 gives plenty of headroom — FAISS is fast, and
        # we need enough hits that the category=0 filter leaves ≥k.
        scores, ids = r.index.search(q_vec, 3000)
        dense_ids = [int(i) for i in ids[0] if i >= 0]
        tags = r.db.get_by_ids(dense_ids)
        score_by_id = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}
        # Filter to general (cat=0) + popularity floor + dedupe
        seen: set[int] = set()
        pool: list[dict] = []
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
        logger.warning(f"_general_tag_candidates: retrieval failed: {e}")
        return []


def _candidate_fragment_for_tag_sp(candidates: list[str]) -> str:
    """Format the candidate tag list as a fragment appended to the Pass 2
    system prompt. Prose-friendly phrasing — tells the LLM the list is
    a pre-curated scene-relevant vocabulary and how to deviate safely
    (named subjects, required format tokens).
    """
    if not candidates:
        return ""
    # Space-form for readability matching anima convention
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


def _resolve_deferred_sources(mods, seed: int, stack, query: str):
    """Second-pass source resolution for entries that needed the anima
    stack (db_retrieve). Mutates `mods` in place. Called by handlers after
    the RAG models are loaded. Returns count of entries resolved.

    `mods` is the (name, entry) list from _collect_modifiers. Entries
    with `source.db_retrieve` and no `_resolved_from_source` yet are
    candidates — the pre-models _collect_modifiers pass couldn't fill
    them because the retriever wasn't available. Now it is.
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
        picked = _resolve_source(source, seed, stack=stack, query=query)
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


def _inject_source_picks(tags_csv: str, mods, stats: dict | None = None):
    """Post-fill safety net for `source:`-driven modifiers.

    After tag validation + slot_fill, ensure every `source:`-picked tag
    actually survives in the final tag list. The pre-pick already shapes
    the prose (the LLM writes a 1970s scene if we said 1970s), but the
    tag-extraction pass can fail to emit the literal decade tag if the
    LLM described the era in concrete terms without naming it.

    Mirrors `target_slot:`'s category-coverage pass — the pick was made
    with a seed for reproducibility; this just makes sure the picked
    value survives into the output. Together the two passes give
    source+target_slot modifiers a strictly-stronger guarantee than
    either alone (hence the ◆◇ double-badge convention).

    Tags are converted underscore→space to match the anima output form
    (other formats apply their own transforms downstream).
    """
    tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
    existing = {t.lower().replace("_", " ").strip() for t in tags}
    added = 0
    for name, entry in mods or []:
        if not isinstance(entry, dict):
            continue
        pick = entry.get("_resolved_from_source")
        if not pick:
            continue
        pick_spaced = pick.replace("_", " ")
        if pick_spaced.lower() in existing:
            continue
        tags.append(pick_spaced)
        existing.add(pick_spaced.lower())
        added += 1
        print(f"[PromptEnhancer] Source inject ({name}): appended '{pick_spaced}'")
    if added and stats is not None:
        stats["total"] = stats.get("total", 0) + added
        stats["source_injected"] = stats.get("source_injected", 0) + added
    return ", ".join(tags), (stats or {})


def _active_target_slots(mods) -> list[str]:
    """Collect the set of target_slot values from active modifiers.

    `mods` is the list of (name, entry) pairs produced by
    _collect_modifiers. Reads `target_slot` from each entry. Dedupes
    while preserving discovery order so injection is stable.
    """
    seen: set[str] = set()
    out: list[str] = []
    for _name, entry in mods or []:
        if not isinstance(entry, dict):
            continue
        slot = entry.get("target_slot")
        if slot and slot not in seen:
            seen.add(slot)
            out.append(slot)
    return out


def _tags_have_category(tag_csv: str, stack, category: int) -> bool:
    """True if any tag in the comma-separated list resolves to a DB record
    of the given category. Trims '@' prefix since artist tokens carry it."""
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


_SUBJECT_COUNT_RE = re.compile(r"^\d+(girl|boy|other)s?$")

# Structural Danbooru categories (artist/character/copyright). CAT_GENERAL=0
# and CAT_META=5 are the ones V14 drops — that's where "sparse leg hair" /
# "simple fish" / "overhead lights" live after FAISS retrieval.
_STRUCTURAL_CATEGORIES = {1, 3, 4}


def _filter_to_structural_tags(tag_csv: str, stack, tag_fmt: str) -> str:
    """V14 Hybrid filter + V19 carve-outs: keep structural tags, plus a
    small curated set of general tags that carry tag-level conditioning.

    V14 baseline: drop CAT_GENERAL and CAT_META. Keep only format whitelist
    (quality/safety/rating), subject-count (1girl/1boy/2girls/…), and
    Danbooru categories {artist, character, copyright}. The Hybrid prose
    carries everything else since Anima's Qwen3 text encoder handles
    natural language natively.

    V19 carve-outs on top of V14:
      (A) Co-occurrence — for each surviving structural tag (character /
          series / artist), retain CAT_GENERAL tags the DB records as
          highly co-occurring (p_given_src ≥ min_prob). Fixes character-
          signature objects (e.g. hatsune_miku → leek retained).
      (B) Format allowlist — retain CAT_GENERAL tags in the format's
          `general_tags_allowlist` yaml field. These are rendering-
          control / booru-trained-convention tokens (framing, named
          poses, clothing archetypes, anime-specific visual tokens,
          rendering techniques) that carry tag-level conditioning prose
          cannot fully replicate.

    Must be called with a stack that has an active DB (Anima format + RAG).
    Co-occurrence carve-out is a no-op if stack.cooccurrence is unavailable.
    """
    if not tag_csv:
        return tag_csv
    fmt = _tag_formats.get(tag_fmt, {})
    whitelist: set = set()
    whitelist |= {str(t).lower() for t in (fmt.get("quality_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt.get("leading_tags") or set())}
    whitelist |= {str(t).lower() for t in (fmt.get("rating_tags") or set())}
    general_allow: set = set(fmt.get("general_tags_allowlist") or set())

    # Parse once, normalize, classify per tag
    parsed = []
    for t in tag_csv.split(","):
        t = t.strip()
        if not t:
            continue
        norm = t.lstrip("@").lower().replace(" ", "_").replace("-", "_")
        if norm:
            parsed.append((t, norm))

    # Pass 1: keep structural tags; collect their names for co-occurrence lookup
    kept_indices = set()
    structural_names: list = []
    for i, (t, norm) in enumerate(parsed):
        if norm in whitelist or _SUBJECT_COUNT_RE.match(norm):
            kept_indices.add(i)
            continue
        if stack is not None and stack.db is not None:
            rec = stack.db.get_by_name(norm)
            if rec and rec.get("category") in _STRUCTURAL_CATEGORIES:
                kept_indices.add(i)
                structural_names.append(norm)

    # Pass 2a: build co-occurrence allowlist from surviving structural tags
    cooc_allow: set = set()
    if (stack is not None and getattr(stack, "cooccurrence", None) is not None
            and structural_names):
        top_k = int(_anima_opt("anima_tagger_v19_cooc_top_k", 20))
        min_prob = float(_anima_opt("anima_tagger_v19_cooc_min_prob", 0.3))
        for sname in structural_names:
            try:
                hits = stack.cooccurrence.top_for(
                    sname, category=0, top_k=top_k, min_prob=min_prob,
                )
            except Exception:
                hits = []
            for hit in hits:
                cooc_allow.add(hit.get("tag", ""))

    # Pass 2b: carve-outs — retain CAT_GENERAL tags present in the format
    # allowlist or the co-occurrence allowlist
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


def _load_tag_formats():
    """Load tag format definitions from tag-formats/ directory."""
    global _tag_formats
    _tag_formats = {}
    if not os.path.isdir(_TAG_FORMATS_DIR):
        return
    for name in sorted(os.listdir(_TAG_FORMATS_DIR)):
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(_TAG_FORMATS_DIR, name))
        if data and "system_prompt" in data:
            label = os.path.splitext(name)[0].replace("-", " ").replace("_", " ").title()
            quality = {t.replace(" ", "_") for t in (data.get("quality_tags") or [])}
            leading = {t.replace(" ", "_") for t in (data.get("leading_tags") or [])}
            rating = {t.replace(" ", "_") for t in (data.get("rating_tags") or [])}
            # V19: CAT_GENERAL tags that pass the V14 structural filter.
            # Stored underscored (DB-lookup form) for O(1) membership checks.
            general_allow = {
                str(t).lower().replace(" ", "_").replace("-", "_")
                for t in (data.get("general_tags_allowlist") or [])
            }
            quality_merged = _UNIVERSAL_QUALITY_TAGS | quality
            _tag_formats[label] = {
                "system_prompt": data["system_prompt"].strip(),
                "use_underscores": data.get("use_underscores", False),
                "negative_quality_tags": data.get("negative_quality_tags", []),
                "tag_db": data.get("tag_db", ""),
                "tag_db_url": data.get("tag_db_url", ""),
                "quality_tags": quality_merged,
                "leading_tags": leading,
                "rating_tags": rating,
                "general_tags_allowlist": general_allow,
                "whitelist_set": quality_merged | leading | rating,
            }



# ── File loading ─────────────────────────────────────────────────────────────

def _scan_modifier_files(directory):
    """Scan a directory for modifier YAML/JSON files.

    Returns dict: dropdown_name -> {category: {modifier: keywords}}.
    Skips _bases.* files. Uses _label field from YAML if present,
    otherwise derives label from filename.
    """
    result = {}
    if not directory or not os.path.isdir(directory):
        return result
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem == BASES_FILENAME:
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(directory, name))
        if data:
            # Require _label in YAML for dropdown name
            label = data.pop("_label", None)
            if not label:
                print(f"[PromptEnhancer] WARNING: Skipping {os.path.join(directory, name)}: missing '_label' field. Add '_label: Your Label' to the YAML file.")
                continue
            result[label] = data
    return result


def _merge_modifier_dicts(base, override):
    """Merge two dropdown-level dicts. Same dropdown name -> merge categories."""
    merged = {}
    for label, categories in base.items():
        merged[label] = {}
        for cat, items in categories.items():
            if isinstance(items, dict):
                merged[label][cat] = dict(items)
    for label, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if label not in merged:
            merged[label] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[label]:
                merged[label][cat] = {}
            merged[label][cat].update(items)
    return merged


def _normalize_modifier(entry):
    """Normalize a modifier entry to dict form with keys:
      behavioral: prose instruction for Prose/Hybrid modes. The prose
                  itself tells the LLM whether to apply a named style or
                  pick one ("The scene is lit by golden hour..." vs
                  "Choose a specific lighting condition..."). No separate
                  type flag — the behavioral text carries the meaning.
      keywords:   comma-separated keyword string for Tags mode and
                  direct-paste button. May be empty for random-choice
                  entries where the LLM synthesizes its own tags.

    Accepts:
      - str: legacy format. Treated as keywords; behavioral is synthesized.
      - dict: new format.
    """
    if isinstance(entry, str):
        kw = entry.strip()
        return {
            "behavioral": f"Apply this style to the scene — describe the qualities through prose, do not list them as keywords: {kw}.",
            "keywords": kw,
        }
    if not isinstance(entry, dict):
        return None
    norm = {
        "behavioral": (entry.get("behavioral") or "").strip(),
        "keywords": (entry.get("keywords") or "").strip(),
    }
    # Optional: target_slot lets the Anima slot-coverage pass (see
    # _active_target_slots) force-fill a category tag from prose when
    # the LLM didn't produce one. Currently used by 🎲 Random Artist
    # (target_slot: artist) and 🎲 Random Franchise (target_slot: copyright).
    if isinstance(entry.get("target_slot"), str) and entry["target_slot"].strip():
        norm["target_slot"] = entry["target_slot"].strip()
    # Optional: source lets _collect_modifiers resolve a concrete Danbooru
    # tag BEFORE the LLM runs, eliminating LLM-collapse bias on "random"
    # wildcards (see _resolve_source). Combines with target_slot for
    # strongest guarantee — pre-pick + post-fill safety net.
    if isinstance(entry.get("source"), dict):
        norm["source"] = entry["source"]
    if not norm["behavioral"] and norm["keywords"]:
        norm["behavioral"] = f"Apply this style to the scene — describe the qualities through prose, do not list them as keywords: {norm['keywords']}."
    # Entries with `source:` legitimately ship with both behavioral AND
    # keywords empty — _collect_modifiers fills them at runtime from the
    # picked DB tag. Skipping the empty-reject below is what lets these
    # survive the normalization pass. (target_slot alone without source
    # still needs a behavioral directive for the LLM, so it doesn't get
    # the same bypass.)
    if norm.get("source"):
        return norm
    if not norm["behavioral"] and not norm["keywords"]:
        return None
    return norm


def _build_dropdown_data(categories_dict):
    """Build flat lookup and choice list from a single dropdown's categories.

    Values in the returned flat dict are normalized modifier dicts (see
    _normalize_modifier) — callers select behavioral vs keywords based on mode.

    Display labels carry mechanism badges so the user can see at a glance
    which randoms have DB guarantees:
        ◆  — `source:`       pre-picked from DB, LLM renders chosen value
        ◇  — `target_slot:`  post-fills a category tag if LLM dropped it
        ◆◇ — both            strongest (pre-pick + post-fill safety net)
    Badges are appended to choice labels only; the flat lookup stays
    keyed on the raw YAML name. _collect_modifiers strips badges before
    looking up.
    """
    flat = {}
    choices = []
    for cat_name, items in categories_dict.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {cat_name.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, entry in items.items():
            norm = _normalize_modifier(entry)
            if norm is None:
                continue
            flat[name] = norm
            badge = ""
            if norm.get("source"):
                badge += _BADGE_SOURCE
            if norm.get("target_slot"):
                badge += _BADGE_TARGET_SLOT
            choices.append(f"{name} {badge}" if badge else name)
    return flat, choices


# ── Tag database ─────────────────────────────────────────────────────────────

def _download_tag_db(fmt_config):
    """Download tag database if not cached. Returns True if available."""
    filename = fmt_config.get("tag_db", "")
    url = fmt_config.get("tag_db_url", "")
    if not filename or not url:
        return False
    os.makedirs(_TAGS_DIR, exist_ok=True)
    local_path = os.path.join(_TAGS_DIR, filename)
    if os.path.isfile(local_path):
        return True
    try:
        logger.info(f"Downloading tag database: {filename}")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        logger.info(f"Tag database saved: {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download tag database: {e}")
        return False


def _load_tag_db(tag_format):
    """Load tag database into memory. Returns set of valid tag strings."""
    fmt_config = _tag_formats.get(tag_format, {})
    filename = fmt_config.get("tag_db", "")
    if not filename:
        return set()

    # Cache by filename (multiple formats can share a DB)
    if filename in _tag_databases:
        return _tag_databases[filename]

    if not _download_tag_db(fmt_config):
        return set()

    local_path = os.path.join(_TAGS_DIR, filename)
    tags = set()
    aliases = {}
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 3)
                if len(parts) >= 1:
                    # Normalize hyphens to underscores. Some CSVs (e.g.
                    # noobai.csv) store multi-word tags as "long-hair";
                    # others (illustrious.csv) as "long_hair". Validation
                    # looks up via underscore form, so unify at load time
                    # to avoid every space-form tag missing exact match
                    # and falling through to the (slower) fuzzy path.
                    tag = parts[0].strip().replace("-", "_")
                    if tag:
                        tags.add(tag)
                    # Parse aliases (field 4, quoted comma-separated)
                    if len(parts) >= 4 and parts[3]:
                        alias_str = parts[3].strip().strip('"')
                        for alias in alias_str.split(","):
                            alias = alias.strip().replace("-", "_")
                            if alias:
                                aliases[alias] = tag
    except Exception as e:
        logger.error(f"Failed to load tag database: {e}")
        return set()

    _tag_databases[filename] = tags
    _tag_databases[f"{filename}_aliases"] = aliases
    logger.info(f"Loaded {len(tags)} tags + {len(aliases)} aliases from {filename}")
    return tags


def _find_closest_tag(tag, valid_tags, max_distance=3):
    """Find the closest valid tag for an invalid one.

    Priority: prefix substring match > Levenshtein distance (via
    rapidfuzz). Returns (corrected_tag, None) on hit, or
    (None, original_tag) on miss. Aliases are checked earlier in
    _validate_tags; this function is only reached for tags that
    failed exact + alias lookup.
    """
    tag_len = len(tag)

    # Prefix substring: our tag is a prefix of a valid tag
    # (e.g. "highres" is a prefix of "highres_(imageboard)").
    # Only for tags 5+ chars and where valid_len is in [tag_len, tag_len*2].
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

    # Levenshtein distance via rapidfuzz (C-accelerated; ~100x faster
    # than the pure-Python loop on a 142k-tag DB). Skip for short tags
    # where edit distance 1-3 produces nonsensical matches.
    if tag_len < 5:
        return None, tag

    result = _rf_extract_one(
        tag, valid_tags,
        scorer=_rf_distance.Levenshtein.distance,
        score_cutoff=max_distance,
    )
    if result is not None:
        return result[0], None

    return None, tag  # unmatched


# Universal Danbooru-adjacent quality tokens every booru format accepts.
# Format-specific quality/leading/rating tokens live in each tag-format yaml.
_UNIVERSAL_QUALITY_TAGS = {
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality", "absurdres", "highres",
}

# Tags that must retain underscores even when the tag-format emits spaces
# (score_N, year_YYYY, source_*). Used by _format_tag_out().
_PRESERVE_UNDERSCORE_RE = re.compile(r"^(score_\d+(_up)?|year_\d{4}|source_[a-z]+)$")


def _format_tag_out(tag, use_underscores):
    """Apply the tag-format's underscore/space convention, preserving
    underscores for canonical tokens like score_7 that are conventionally
    written with an underscore regardless of format.
    """
    if use_underscores:
        return tag
    if _PRESERVE_UNDERSCORE_RE.match(tag):
        return tag
    return tag.replace("_", " ")

# Common LLM mistakes -> correct danbooru tag
_TAG_CORRECTIONS = {
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

def _validate_tags(tags_str, tag_format, mode="Fuzzy"):
    """Validate and correct tags against the database.

    Modes:
      Fuzzy        — exact + alias + fuzzy correction, keep unrecognized
      Fuzzy Strict — exact + alias + fuzzy correction, drop unrecognized

    Returns (corrected_tags_str, stats_dict).
    """
    valid_tags = _load_tag_db(tag_format)
    if not valid_tags:
        return tags_str, {"error": "No tag database available"}

    fmt_config = _tag_formats.get(tag_format, {})
    db_filename = fmt_config.get("tag_db", "")
    aliases = _tag_databases.get(f"{db_filename}_aliases", {})
    use_underscores = fmt_config.get("use_underscores", False)
    use_fuzzy = mode in ("Fuzzy", "Fuzzy Strict")
    drop_invalid = mode == "Fuzzy Strict"

    raw_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    result_tags = []
    corrected = 0
    dropped = 0
    kept = 0

    for tag in raw_tags:
        # Strip LLM meta-annotations: [style: X], (artist: X), etc.
        tag = _clean_tag(tag)
        if not tag:
            continue

        # Always use underscores for lookup (DB and corrections use underscores).
        # use_underscores only controls output format, not lookup.
        lookup = tag.replace(" ", "_")

        # Common LLM mistakes — correct before any other check
        if lookup in _TAG_CORRECTIONS:
            result_tags.append(_format_tag_out(_TAG_CORRECTIONS[lookup], use_underscores))
            corrected += 1
            continue

        # Whitelisted tags always pass (per-format quality + leading + rating)
        if lookup in fmt_config.get("whitelist_set", set()):
            result_tags.append(_format_tag_out(lookup, use_underscores))
            continue

        # Exact match
        if lookup in valid_tags:
            result_tags.append(tag)
            continue

        # Alias match (always applied in all modes)
        if lookup in aliases:
            result_tags.append(_format_tag_out(aliases[lookup], use_underscores))
            corrected += 1
            continue

        # Prefix match: "artist_name" -> "artist_name_(style)" etc.
        # Catches LLM omitting danbooru disambiguation suffixes.
        # Only for multi-word tags (contain underscore) to avoid matching
        # common words like "high" -> "high_(hgih)" or "dusty" -> "dusty_(gravity_daze)".
        if "_" in lookup:
            prefix = lookup + "_("
            prefix_matches = [v for v in valid_tags if v.startswith(prefix)]
            if len(prefix_matches) == 1:
                result_tags.append(_format_tag_out(prefix_matches[0], use_underscores))
                corrected += 1
                continue

        # Fuzzy match (only in Fuzzy mode)
        if use_fuzzy:
            match, _ = _find_closest_tag(lookup, valid_tags)
            if match:
                result_tags.append(_format_tag_out(match, use_underscores))
                corrected += 1
                continue

        # Unrecognized tag
        if drop_invalid:
            dropped += 1
        else:
            result_tags.append(tag)
            kept += 1

    # Reorder tags into standard danbooru order
    result_tags = _reorder_tags(result_tags, tag_format)

    # Escape parentheses for SD — danbooru tags like artist_(style) need
    # \( \) so SD doesn't interpret them as emphasis/weight syntax.
    def _escape_parens(tag):
        if "(" in tag and "_(" in tag:
            return tag.replace("(", r"\(").replace(")", r"\)")
        return tag
    result_tags = [_escape_parens(t) for t in result_tags]

    stats = {"corrected": corrected, "dropped": dropped, "kept_invalid": kept, "total": len(result_tags)}
    return ", ".join(result_tags), stats


# Universal Danbooru subject/count tags (same across Illustrious/NoobAI/Pony/Anima).
_SUBJECT_TAGS = {
    "1girl", "1girls", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "1boy", "1man", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "1other", "solo", "no_humans", "male_focus", "female_focus",
    "1woman", "man", "woman", "girl", "boy",
}


def _reorder_tags(tags, tag_format):
    """Reorder tags per the selected format's convention.

    Order: quality -> leading -> subjects -> rest -> rating.
    The quality / leading / rating buckets come from the tag-format yaml.
    Deduplicates and enforces single rating tag (keeps last).
    """
    fmt_config = _tag_formats.get(tag_format, {})
    quality_set = fmt_config.get("quality_tags", _UNIVERSAL_QUALITY_TAGS)
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
        elif lookup in _SUBJECT_TAGS:
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


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}          # flat: name -> keywords (for lookup across all dropdowns)
_dropdown_order = []         # list of dropdown labels in display order
_dropdown_choices = {}       # label -> [choice_list with separators]
_prompts = {}                # operational prompts loaded from prompts.yaml


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _dropdown_order, _dropdown_choices, _prompts

    local_dirs = _get_local_dirs(local_dir_path)

    # Bases (YAML, with local overrides) — loader lives in pe_data.bases.
    _bases = _pe_bases.load(_EXT_DIR, local_dirs)

    # Modifiers: scan extension modifiers/ dir + all local dirs, merge
    all_mods = _scan_modifier_files(_MODIFIERS_DIR)
    for local_dir in local_dirs:
        local_mods = _scan_modifier_files(local_dir)
        all_mods = _merge_modifier_dicts(all_mods, local_mods)

    _all_modifiers = {}
    _dropdown_order = []
    _dropdown_choices = {}
    for label in sorted(all_mods.keys()):
        flat, choices = _build_dropdown_data(all_mods[label])
        if choices:
            _dropdown_order.append(label)
            _dropdown_choices[label] = choices
            _all_modifiers.update(flat)

    # Prompts (YAML, with local overrides) — loader lives in pe_data.prompts.
    _prompts = _pe_prompts.load(_EXT_DIR, local_dirs)


_reload_all()


_load_tag_formats()

# Max tokens per forge preset (from backend/diffusion_engine/*.py)
_PRESET_MAX_TOKENS = {
    "sd": 75, "xl": 75, "flux": 255, "klein": 255,
    "qwen": 512, "lumina": 512, "zit": 999, "wan": 512,
    "anima": 512, "ernie": 512,
}

# Tag counts per detail level (model-independent)
_TAG_COUNTS = {0: None, 1: 8, 2: 12, 3: 18, 4: 25, 5: 35, 6: 45, 7: 55, 8: 65, 9: 75, 10: 90}

# Detail level descriptions
_DETAIL_LABELS = {
    0: None,
    1: "very short, minimal",
    2: "short, concise",
    3: "brief but complete",
    4: "moderate",
    5: "moderately detailed",
    6: "detailed",
    7: "detailed, vivid",
    8: "highly detailed",
    9: "very detailed, comprehensive",
    10: "extensive, exhaustive",
}


def _get_word_target(detail, preset="sd"):
    """Calculate word target based on detail level and forge preset."""
    if detail == 0:
        return None
    max_tokens = _PRESET_MAX_TOKENS.get(preset, 75)
    # Rough: 1 token ≈ 0.75 words, scale detail 1-10 across 20%-100% of max
    max_words = int(max_tokens * 0.75)
    fraction = 0.1 + (detail / 10) * 0.9  # detail 1 = 20%, detail 10 = 100%
    return max(20, int(max_words * fraction))


def _build_detail_instruction(detail, mode="enhance", preset="sd"):
    """Build detail instruction for enhance or tags.

    Returns a style-descriptor only ("detailed, vivid") — no word or
    tag count targets. Historical versions injected "Aim for around N
    words" / "around N tags"; those are removed since they made the
    LLM stingy and suppressed real coverage.
    """
    if detail == 0:
        return None
    label = _DETAIL_LABELS.get(detail, "moderate")
    if mode == "tags":
        return f"Generate a {label} set of tags. Cover every distinct concept."
    return f"Write a {label} description."



# ── Helpers ──────────────────────────────────────────────────────────────────

# Thin wrappers around pe_data.bases (imported as _pe_bases above) so
# the rest of this file keeps the historical _base_* identifier shape.
def _base_body(entry):
    return _pe_bases.body(entry)


def _base_meta(name):
    return _pe_bases.meta(_bases, name)


def _base_names():
    return _pe_bases.names(_bases)


def _strip_mechanism_badges(name: str) -> str:
    """Drop any trailing ◆/◇ badge chars + whitespace that the UI appended
    to the display label. Keeps the YAML-side name canonical for lookup.
    """
    if not isinstance(name, str):
        return name
    return name.rstrip(f" {_BADGE_SOURCE}{_BADGE_TARGET_SLOT}")


def _collect_modifiers(dropdown_selections, seed: int | None = None):
    """Collect all selected modifiers into a list of (name, normalized_entry) tuples.

    Each normalized_entry is a dict with 'behavioral', 'keywords' and
    optional 'source' / 'target_slot' fields. The caller decides which
    fields to use based on the mode.

    When a modifier has a `source:` entry AND a seed is provided, the
    pick is resolved HERE (before the LLM sees any system prompt): the
    returned entry has `behavioral` and `keywords` rewritten from the
    picked real Danbooru tag. This is the `source:` mechanism — see
    _resolve_source for how the pool is built.

    Names may arrive with UI-appended ◆/◇ badges (see _build_dropdown_data);
    strip those before looking up the canonical YAML entry.
    """
    result = []
    for selections in dropdown_selections:
        for raw_name in (selections or []):
            name = _strip_mechanism_badges(raw_name)
            entry = _all_modifiers.get(name)
            if not entry:
                continue
            source = entry.get("source") if isinstance(entry, dict) else None
            if source and seed is not None:
                # db_retrieve needs the anima stack (retriever + models),
                # which isn't available at _collect_modifiers time. Defer
                # those to _resolve_deferred_sources in the handler after
                # models load. db_pattern resolves immediately here.
                is_deferred = "db_retrieve" in source and "db_pattern" not in source
                if is_deferred:
                    # Leave entry as-is; handler will resolve later and
                    # update behavioral/keywords/_resolved_from_source.
                    pass
                else:
                    picked = _resolve_source(source, seed)
                    if picked:
                        # Materialize a per-run entry with picked values baked
                        # in. Preserve target_slot / other keys so the post-fill
                        # safety net still fires when combined with source.
                        resolved = dict(entry)
                        resolved["behavioral"] = picked["behavioral"]
                        resolved["keywords"] = picked["keywords"]
                        resolved["_resolved_from_source"] = picked["name"]
                        print(f"[PromptEnhancer] Random pick ({name}): "
                              f"{picked['name']} (pool={picked['pool_size']}, "
                              f"post_count={picked['post_count']})")
                        entry = resolved
                    else:
                        print(f"[PromptEnhancer] Random pick ({name}): "
                              f"pool empty, falling back to LLM behavioral")
            result.append((name, entry))
    return result


def _build_style_string(mod_list, mode="prose"):
    """Build the style block for the user message.

    mode:
      "prose"   — Prose/Hybrid. Uses behavioral field. Emits a
                  comma-separated list of style directives. The active
                  base prompt governs HOW these get applied (voice,
                  structure); we just name the styles.
      "tags"    — Tags mode. Uses keywords field. Classic "Apply these
                  styles: kw1, kw2" keyword-echoing directive.
    """
    if not mod_list:
        return ""
    if mode == "tags":
        parts = []
        for name, entry in mod_list:
            # Strip 🎲 emoji marker from random-entry names
            clean_name = name.replace("\U0001F3B2", "").strip()
            kw = entry.get("keywords") or ""
            if not kw:
                # Dice entry (empty keywords). In Tags mode, any instruction
                # text we inject is parsed by the LLM as tag tokens — even
                # bracketed directives and "pick a surprising X" fallbacks
                # leak words like "surprising", "location", "artist",
                # "detailed_background" into the output. So skip these
                # entirely in Tags mode. Dice entries remain fully functional
                # in Prose/Hybrid mode where the LLM reads context correctly.
                continue
            if clean_name.lower() not in kw.lower():
                kw = f"{clean_name.lower()}, {kw}"
            parts.append(kw)
        return f"Apply these styles: {', '.join(parts)}." if parts else ""
    # Prose/Hybrid: short behaviorals concatenate into a compact directive.
    # Qwen recognizes style/mood/setting concepts directly — we don't need
    # to teach it, just name them. Base prompt handles voice.
    behaviorals = []
    for name, entry in mod_list:
        text = (entry.get("behavioral") or "").strip()
        if text:
            # Strip trailing punctuation so items join cleanly with commas.
            text = text.rstrip(".!?: ")
            if text:
                behaviorals.append(text)
    if not behaviorals:
        return ""
    return f"Apply these styles to the scene: {', '.join(behaviorals)}."


# ── Ollama ───────────────────────────────────────────────────────────────────


# Placeholder used in the user message when Source Prompt is empty (dice roll).
# Styles and wildcards still flow through normally; this only replaces the
# "SOURCE PROMPT: {source}" line so the LLM knows to invent rather than expand.




# ── Core logic ───────────────────────────────────────────────────────────────

def _has_inline_wildcards(text):
    return bool(re.search(r"\{[^}]+\?\}", text))


    # Skip known valid patterns (rating:, score_, source_)
    if ":" in tag:
        return tag
    # Try to insert underscores at likely word boundaries
    # Look for lowercase->uppercase transitions (camelCase)
    result = re.sub(r"([a-z])([A-Z])", r"\1_\2", tag).lower()
    if result != tag.lower():
        return result
    # All lowercase concatenated — can't reliably split without a dictionary
    # Return as-is, validation will handle it
    return tag



def _postprocess_tags(tag_str, tag_fmt, validation_mode):
    """Apply full tag post-processing pipeline.

    Handles: concatenation split, underscore formatting, validation,
    reordering, paren escaping. Returns (processed_tags, stats_or_None).
    """
    fmt_config = _tag_formats.get(tag_fmt, {})
    use_underscores = fmt_config.get("use_underscores", False)
    tag_str = ", ".join(
        _split_concatenated_tag(t.strip()).replace(" ", "_") if use_underscores
        else _split_concatenated_tag(t.strip())
        for t in tag_str.split(",") if t.strip()
    )
    if validation_mode != "Off":
        tag_str, stats = _validate_tags(tag_str, tag_fmt, mode=validation_mode)
        return tag_str, stats
    return tag_str, None




# _cancel_flag is owned by pe_llm_layer so the LLM stream and the
# cancel button operate on the same Event. (Renamed import preserves
# the historical _cancel_flag identifier used throughout this file.)
from pe_llm_layer import cancel_flag as _cancel_flag
_last_seed = -1
# Which PE button produced the currently-staged prompt. Set by each
# handler entry point, consumed by process() to write PE Mode into image
# metadata and shown in every status line via _MODE_* prefixes below.
_last_pe_mode: str | None = None

# Status line prefix per mode — matches the button glyphs so the status
# message is self-identifying at a glance.
_MODE_PROSE  = "\u270d Prose"   # ✍ Prose
_MODE_HYBRID = "\u2728 Hybrid"  # ✨ Hybrid
_MODE_TAGS   = "\U0001F3F7 Tags"  # 🏷 Tags
_MODE_REMIX  = "\U0001F500 Remix" # 🔀 Remix



# ── V5 conditional adherence directive ──────────────────────────────
# V5 prose-adherence directive and V15 prose-v14-coverage directive are
# defined in prompts.yaml (keys: `prose_adherence`, `prose_v14_coverage`).
# Appended to the prose SP by the Hybrid handler when active. The V8
# picker SP is also in prompts.yaml (key: `picker`). See prompts.yaml
# for the full text; keeping all LLM-facing strings in yaml ensures no
# "hidden" prompt influence lives in Python.






def _build_inline_wildcard_text(source):
    """Return the inline-wildcard directive if source contains {name?} placeholders.

    The previous selected-wildcards system was folded into the modifier
    system (🎲 Random X entries). This helper only handles the remaining
    inline {name?} syntax in the source prompt itself.
    """
    if source and _has_inline_wildcards(source):
        return _prompts.get("inline_wildcard", "")
    return ""


def _assemble_system_prompt(base_name, custom_system_prompt, detail=3):
    """Assemble the system prompt (base + detail, no modifiers/wildcards).

    For non-Custom bases, wraps the per-base body with shared _preamble
    (prepended) and _format (appended) blocks from bases.yaml when
    present. Custom bases are used as-is.
    """
    if base_name == "Custom":
        system_prompt = (custom_system_prompt or "").strip()
    else:
        body = _base_body(_bases.get(base_name))
        if not body:
            return None
        parts = [
            _base_body(_bases.get("_preamble")).strip(),
            body.strip(),
            _base_body(_bases.get("_format")).strip(),
        ]
        system_prompt = "\n\n".join(p for p in parts if p)
    if not system_prompt:
        return None

    detail = int(detail) if detail else 0
    try:
        from modules import shared
        preset = getattr(shared.opts, "forge_preset", "sd")
    except Exception:
        preset = "sd"
    instruction = _build_detail_instruction(detail, "enhance", preset)
    if instruction:
        system_prompt = f"{system_prompt}\n\n{instruction}"

    return system_prompt





# ── UI ───────────────────────────────────────────────────────────────────────

class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        # The old Ollama-discovery block (poll /api/tags for available
        # models) is gone — the in-process runner reads model/quant/
        # compute from shared.opts.
        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt ──
            source_prompt = gr.Textbox(
                label="Source Prompt", lines=3,
                placeholder="Type your prompt here, or leave empty to roll the dice. Use {name?} for inline wildcards.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(value="\u270d Prose", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_enhance_btn")
                hybrid_btn = gr.Button(value="\u2728 Hybrid", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_hybrid_btn")
                tags_btn = gr.Button(value="\U0001f3f7 Tags", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_tags_btn")
                refine_btn = gr.Button(value="\U0001f500 Remix", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_refine_btn")
                cancel_btn = gr.Button(value="\u274c Cancel", scale=0, min_width=80, elem_id=f"{tab}_pe_cancel_btn")
                prepend_source = gr.Checkbox(label="Prepend", value=False, scale=0, min_width=60)
                prepend_source.do_not_save_to_config = True
                negative_prompt_cb = gr.Checkbox(label="+ Negative", value=False, scale=0, min_width=110)
                negative_prompt_cb.do_not_save_to_config = True
                motion_cb = gr.Checkbox(label="+ Motion and Audio", value=False, scale=0, min_width=150)
                motion_cb.do_not_save_to_config = True
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            # ── Base + Tag Format + Validation ──
            with gr.Row():
                base = gr.Dropdown(label="Base", choices=_base_names(), value="Default", scale=1, info="Prose voice — matches the image model family.")
                _tf_names = list(_tag_formats.keys())
                tag_format = gr.Dropdown(label="Tag Format", choices=_tf_names, value=_tf_names[0] if _tf_names else "", scale=1, info="Tag conventions for booru-trained fine-tunes.")
                tag_validation = gr.Radio(
                    label="Tag Validation",
                    choices=["RAG", "Fuzzy Strict", "Fuzzy", "Off"],
                    value="RAG", scale=2,
                    info="RAG=retrieval + embedding validator (Anima) | Fuzzy Strict=guess+drop | Fuzzy=guess+keep | Off=raw",
                )
                tag_validation.do_not_save_to_config = True

            def _base_description_html(name):
                if name == "Custom":
                    return "<div style='color:#888; font-size:0.9em; margin-top:-8px; padding-left:4px'>User-supplied system prompt (below). Bypasses the shared preamble and format.</div>"
                meta = _base_meta(name)
                desc = meta.get("description", "")
                if not desc:
                    return ""
                return f"<div style='color:#888; font-size:0.9em; margin-top:-8px; padding-left:4px'>{desc}</div>"

            base_description = gr.HTML(value=_base_description_html("Default"))

            # ── Auto-generated modifier dropdowns (one per file) ──
            dd_components = []
            dd_labels = list(_dropdown_order)

            # Layout: 3 dropdowns per row, pad incomplete rows
            for i in range(0, len(dd_labels), 3):
                row_labels = dd_labels[i:i+3]
                with gr.Row():
                    for label in row_labels:
                        d = gr.Dropdown(
                            label=label,
                            choices=_dropdown_choices.get(label, []),
                            value=[], multiselect=True, scale=1,
                        )
                        d.do_not_save_to_config = True
                        dd_components.append(d)
                    # Pad incomplete rows so dropdowns don't stretch
                    for _ in range(3 - len(row_labels)):
                        gr.HTML(value="", visible=True, scale=1)

            # ── Temperature + Think + Seed ──
            # detail_level is kept as a hidden-valued component (always 0)
            # so input positions in .click handlers stay stable without
            # requiring all four generator signatures to be refactored.
            # detail=0 routes through _build_detail_instruction → None,
            # so no word/tag count is ever injected into system prompts.
            detail_level = gr.Number(value=0, visible=False)
            detail_level.do_not_save_to_config = True
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.8, step=0.05, scale=2, info="0 = deterministic, 2 = creative")
                seed = gr.Number(label="Seed", value=-1, minimum=-1, step=1, scale=1, info="-1 = random", precision=0, elem_id=f"{tab}_pe_seed")
                seed.do_not_save_to_config = True
                seed_random_btn = ToolButton(value="\U0001f3b2", elem_id=f"{tab}_pe_seed_random")
                seed_reuse_btn = ToolButton(value="\u267b", elem_id=f"{tab}_pe_seed_reuse")
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)
                think.do_not_save_to_config = True
                seed_random_btn.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
                seed_reuse_btn.click(fn=lambda: _last_seed, inputs=[], outputs=[seed], show_progress=False)

            # ── Custom system prompt ──
            custom_system_prompt = gr.Textbox(label="Custom System Prompt", lines=4, visible=False, placeholder="Enter your custom system prompt...")
            base.change(fn=lambda b: gr.update(visible=(b == "Custom")), inputs=[base], outputs=[custom_system_prompt], show_progress=False)
            base.change(fn=_base_description_html, inputs=[base], outputs=[base_description], show_progress=False)

            # ── LLM model controls ──
            # The in-process llm_runner reads its model + compute config
            # from shared.opts (see Forge settings, "Prompt Enhancer LLM"
            # section). No per-tab API URL or model dropdown — the runner
            # is process-global.
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_dir_path = gr.Textbox(
                    label="Local Overrides",
                    placeholder=f"Using: {_env_local}" if _env_local else "Comma-separated dirs (refreshes content only, restart for new dropdowns)",
                    scale=3,
                )
                local_dir_path.do_not_save_to_config = True
                reload_btn = gr.Button(value="\U0001f504 Reload", scale=0, min_width=100)
                llm_status_btn = gr.Button(value="\U0001f504 LLM", scale=0, min_width=100)
            llm_status = gr.HTML(value=_get_llm_status())

            # ── Reload wiring ──
            # Note: reload rebuilds dropdowns but can't add/remove them dynamically.
            # New files require a Forge restart. Existing dropdown contents are refreshed.
            def _do_refresh(current_base, *args):
                # Last arg is local_dir_path
                local_path = args[-1]
                dd_vals = args[:-1]

                _reload_all(local_path)
                results = [gr.update(choices=_base_names(), value=current_base if current_base in _bases else "Default")]
                for i, label in enumerate(dd_labels):
                    choices = _dropdown_choices.get(label, [])
                    old_val = dd_vals[i] if i < len(dd_vals) else []
                    results.append(gr.update(choices=choices, value=[v for v in (old_val or []) if v in _all_modifiers]))
                msg = (f"<span style='color:#6c6'>Reloaded: {len(_bases)} bases, "
                       f"{len(_dropdown_order)} modifier groups, "
                       f"{len(_all_modifiers)} modifiers, "
                       f"{len(_prompts)} prompts</span>")
                results.append(msg)
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base] + dd_components + [local_dir_path],
                outputs=[base] + dd_components + [status],
                show_progress=False,
            )
            llm_status_btn.click(fn=_get_llm_status, inputs=[], outputs=[llm_status], show_progress=False)

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")
            negative_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_in")
            negative_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_out")

            # ── Prose ──
            def _enhance(source, base_name, custom_sp, *args):
                global _last_pe_mode
                _last_pe_mode = "Prose"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()
                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                # Build user message with modifiers + inline wildcards
                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                initial_status = "\U0001F3B2 Rolling dice (prose)..." if not source else "Generating prose..."
                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {initial_status}</span>"

                print(f"[PromptEnhancer] Prose: model={model}, think={th}, mods={len(mods)}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw)
                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""
                    if prepend and source:
                        result = f"{source}\n\n{result}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield result, negative, f"<span style='color:#6c6'>{_MODE_PROSE}: OK - {len(result.split())} words, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_PROSE}: Cancelled - {len(partial.split())} words (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: Cancelled</span>"
                except _TruncatedError as e:
                    result = _clean_output(str(e))
                    yield result, "", f"<span style='color:#ca6'>{_MODE_PROSE}: Truncated - {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: {msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: {msg}</span>"

            prose_event = enhance_btn.click(
                fn=_enhance,
                inputs=[source_prompt, base, custom_system_prompt]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Hybrid (two-pass: prose → extract tags + NL) ──
            def _hybrid(source, base_name, custom_sp, tag_fmt, validation_mode,
                        *args):
                global _last_pe_mode
                _last_pe_mode = "Hybrid"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]
                t0 = time.monotonic()

                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                # RAG path when the user picked it on the Tag Validation radio.
                # If picked but unavailable (non-Anima format, or artefacts
                # missing), surface the reason so the user sees why we fell
                # back to rapidfuzz.
                # User-chosen RAG path. No fallback — if RAG is selected
                # but unavailable, abort with a clear error. Other radio
                # modes (Fuzzy Strict / Fuzzy / Off) are separate code
                # paths; they do NOT run here as a substitute.
                _anima = None
                _anima_cm = None
                _anima_shortlist = None
                if validation_mode == "RAG":
                    ok, reason = _rag_available_for(tag_fmt)
                    if not ok:
                        logger.error(f"RAG unavailable: {reason}")
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: RAG unavailable — {reason}</span>"
                        return
                    _s = _get_anima_stack()
                    try:
                        _anima_cm = _s.models()
                        _anima_cm.__enter__()
                        _anima = _s
                        # Resolve any db_retrieve source: picks (e.g. ◆ on
                        # Random Artist) now that the stack is up. If any
                        # fire, rebuild style_str + user_msg so the picked
                        # value flows into the prompt AND the shortlist.
                        if _resolve_deferred_sources(mods, int(sd), _anima, query=source):
                            style_str = _build_style_string(mods)
                            user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                            if style_str:
                                user_msg = f"{user_msg}\n\n{style_str}"
                            if inline_text:
                                user_msg = f"{user_msg}\n\n{inline_text}"
                        _expander = _make_anima_query_expander(temperature=0.3,
                            think=False, seed=int(sd),
                        )
                        _anima_shortlist = _s.build_shortlist(
                            source_prompt=source,
                            modifier_keywords=style_str,
                            query_expander=_expander,
                        )
                        _sl_frag = _anima_shortlist.as_system_prompt_fragment()
                        if _sl_frag:
                            sp = f"{sp}\n\n{_sl_frag}"
                        # V5 conditional adherence directive — only when
                        # source is non-empty, so dice-roll creativity
                        # stays free.
                        if source:
                            sp = f"{sp}\n\n{_prompts.get('prose_adherence', '')}"
                        # V15: on V14 Hybrid path (Anima+RAG), the prose is
                        # the primary image input — no general tags will
                        # survive the structural filter to cover pose /
                        # clothing / setting / lighting. Tell the LLM so it
                        # produces self-sufficient prose.
                        sp = f"{sp}\n\n{_prompts.get('prose_v14_coverage', '')}"
                        print(f"[PromptEnhancer] RAG shortlist: "
                              f"{len(_anima_shortlist.artists)} artists, "
                              f"{len(_anima_shortlist.characters)} characters, "
                              f"{len(_anima_shortlist.series)} series")
                    except Exception as e:
                        logger.error(f"RAG setup failed: {e}")
                        if _anima_cm:
                            try: _anima_cm.__exit__(None, None, None)
                            except Exception: pass
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: RAG setup failed: {type(e).__name__}: {e}</span>"
                        return

                if not source:
                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: \U0001F3B2 Rolling dice (1/3 prose)...</span>"
                print(f"[PromptEnhancer] Hybrid pass 1/3 (prose): model={model}, think={th}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    # Pass 1: generate prose. V8 multi-sample mode when
                    # anima_tagger_prose_samples > 1 (RAG path only).
                    n_samples = int(_anima_opt("anima_tagger_prose_samples", 3)) if _anima is not None else 1
                    prose_raw = None
                    if n_samples > 1:
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose (multi-sample {n_samples})...</span>"
                        prose_raw, _samples_all, _picker_choice = _multi_sample_prose(user_msg, sp, temp, seed=int(sd), n_samples=n_samples, num_predict=1024, picker_system_prompt=_prompts.get("picker", ""))
                    else:
                        for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose: {p['elapsed']:.1f}s...</span>"
                            else:
                                prose_raw = chunk
                    prose_raw = _clean_output(prose_raw)
                    if not prose_raw:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Prose generation returned empty.</span>"
                        return

                    # Split negative from prose before passes 2 & 3
                    if neg_cb:
                        prose, negative = _split_positive_negative(prose_raw)
                    else:
                        prose, negative = prose_raw, ""

                    print(f"[PromptEnhancer] Hybrid pass 2/3 (tags): {len(prose.split())} words → tags")

                    # Pass 2: extract tags (tag format prompt + style context)
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    tag_sp = fmt_config.get("system_prompt", "")
                    if not tag_sp:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: No tag format configured.</span>"
                        return
                    if style_str:
                        tag_sp = f"{tag_sp}\n\n{_prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
                    # V17: shortlist-aware tag_extract. The RAG shortlist
                    # (retrieved artists / characters / series candidates)
                    # was previously injected only into the prose SP. The
                    # tag_extract LLM saw it indirectly via whatever the
                    # prose mentioned. V16 refocused tag_extract to emit
                    # ONLY structural tags (artist / character / series
                    # among them); giving it the shortlist directly
                    # sharpens precision on those categories.
                    if _anima_shortlist is not None:
                        _sl_frag_tag = _anima_shortlist.as_system_prompt_fragment()
                        if _sl_frag_tag:
                            tag_sp = f"{tag_sp}\n\n{_sl_frag_tag}"
                    # V13: Pass 2 grounding — prepend a scene-relevant candidate
                    # tag list retrieved from the DB against the prose. Constrains
                    # the LLM to pick from real scene-matching vocabulary instead
                    # of inventing niche compounds (sparse_leg_hair,
                    # overhead_lights, simple_fish). Only active with RAG + Anima.
                    if _anima is not None and bool(_anima_opt("anima_tagger_pass2_grounding", False)):
                        _cand = _general_tag_candidates(_anima, prose, k=int(_anima_opt("anima_tagger_pass2_grounding_k", 60)), min_post_count=int(_anima_opt("anima_tagger_pass2_grounding_min_pc", 100)))
                        _frag = _candidate_fragment_for_tag_sp(_cand)
                        if _frag:
                            tag_sp = f"{tag_sp}\n{_frag}"
                            print(f"[PromptEnhancer] Pass 2 grounding: {len(_cand)} candidate tags injected")
                    tags_raw = None
                    for chunk in _call_llm_progress(prose, tag_sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 2/3 tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 2/3 tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            tags_raw = chunk
                    tags_raw = _clean_output(tags_raw, strip_underscores=False)

                    # V14 path: on Anima + RAG we skip Pass 3 summarize and use
                    # full prose as the Hybrid body. The structural-tag filter
                    # applied below drops general-category tags (where FAISS
                    # mis-retrievals live), leaving only quality/safety/subject-
                    # count/@artist/character/series in the tag prefix. Anima's
                    # Qwen3 text encoder handles the prose natively, so the
                    # concepts that used to be general tags (pose, clothing,
                    # body, setting, lighting, atmosphere) are carried by prose.
                    v14_path = (_anima is not None and tag_fmt == "Anima"
                                and validation_mode == "RAG")
                    if v14_path:
                        nl_supplement = None
                    else:
                        print(f"[PromptEnhancer] Hybrid pass 3/3 (summarize): → NL supplement")
                        # Pass 3: summarize prose to 1-2 compositional sentences
                        summarize_sp = _prompts.get("summarize", "")
                        style_str = _build_style_string(mods)
                        if style_str:
                            summarize_sp = f"{summarize_sp}\n\nThe following styles were applied: {style_str} Ensure these stylistic choices are reflected in the compositional summary."
                        nl_supplement = None
                        for chunk in _call_llm_progress(prose, summarize_sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 3/3 summarize: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 3/3 summarize: {p['elapsed']:.1f}s...</span>"
                            else:
                                nl_supplement = chunk
                        nl_supplement = _clean_output(nl_supplement)

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: Validating tags ({validation_mode})...</span>"

                    # Route safety tag from any active NSFW modifier selection
                    _anima_safety = _anima_safety_from_modifiers(mods, source)

                    # Post-process negative tags through tag pipeline when applicable
                    if neg_cb and negative:
                        if _anima is not None:
                            negative, _ = _anima_tag_from_draft(
                                _anima, negative, safety=_anima_safety,
                                shortlist=_anima_shortlist,
                            )
                        else:
                            negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    # Run tags through full post-processing pipeline.
                    # When Anima retrieval is active, the bge-m3 validator
                    # + rule layer replace the rapidfuzz path entirely.
                    if _anima is not None:
                        tags_raw, stats = _anima_tag_from_draft(
                            _anima, tags_raw, safety=_anima_safety,
                            shortlist=_anima_shortlist,
                        )
                        # Slot-coverage pass: for each active modifier with a
                        # target_slot (e.g. Random Artist → artist), if no
                        # tag of that category survived validation, retrieve
                        # a top-1 from prose and inject it. Keeps "🎲 Random
                        # X" promises actually reflected in the output.
                        if bool(_anima_opt("anima_tagger_slot_fill", True)):
                            slots = _active_target_slots(mods)
                            for slot in slots:
                                cat_info = _SLOT_TO_CATEGORY.get(slot)
                                if not cat_info:
                                    continue
                                cat_code = cat_info["category"]
                                if _tags_have_category(tags_raw, _anima, cat_code):
                                    continue
                                picked = _retrieve_prose_slot(_anima, prose, slot, seed=int(sd))
                                if not picked:
                                    continue
                                # Format artist picks with @ prefix (Anima convention);
                                # other categories go in bare.
                                tag_out = picked.replace("_", " ")
                                if slot == "artist":
                                    tag_out = "@" + tag_out
                                tags_raw = f"{tags_raw}, {tag_out}" if tags_raw else tag_out
                                print(f"[PromptEnhancer] Slot fill ({slot}): injected '{tag_out}' from prose")
                                if stats:
                                    stats["total"] = stats.get("total", 0) + 1
                    else:
                        tags_raw, stats = _postprocess_tags(tags_raw, tag_fmt, validation_mode)
                    # Source post-inject: ensure every source:-picked tag
                    # survives the validator+slot_fill path. The pre-pick
                    # shaped the prose; this makes sure the picked tag
                    # actually appears in the output list.
                    tags_raw, stats = _inject_source_picks(tags_raw, mods, stats)

                    # V14: drop general+meta tags, emit {structural_tags}\n\n{prose}.
                    # The FAISS validator force-maps every draft tag to a nearest
                    # neighbor; on terse/mundane sources the neighbors are weird
                    # ("sparse leg hair" for "loose hair", "simple fish" for a
                    # garden scene, "overhead lights" for "warm light"). Anima's
                    # Qwen3 text encoder handles those concepts as prose natively,
                    # so we drop general tags by construction and let prose carry
                    # them. See experiments/variants/v14.py for rating data.
                    if v14_path:
                        tags_raw = _filter_to_structural_tags(tags_raw, _anima, tag_fmt)
                        tag_count = len([t for t in tags_raw.split(",") if t.strip()])
                        status_parts = [f"{tag_count} tags + prose"]
                    else:
                        tag_count = stats.get("total", 0) if stats else len([t for t in tags_raw.split(",") if t.strip()])
                        status_parts = [f"{tag_count} tags + NL"]
                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")

                    if v14_path:
                        final = f"{tags_raw}\n\n{prose}" if prose else tags_raw
                    else:
                        final = f"{tags_raw}\n\n{nl_supplement}" if nl_supplement else tags_raw
                    if prepend and source:
                        final = f"{source}\n\n{final}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield final, negative, f"<span style='color:#6c6'>{_MODE_HYBRID}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_HYBRID}: Cancelled (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Cancelled</span>"
                except _TruncatedError:
                    # Fail loud — truncated tag output is a reduced result
                    # that looks like success. Empty textbox + red status.
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: {msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: {msg}</span>"
                finally:
                    # Release bge-m3 + reranker VRAM for image gen
                    if _anima_cm is not None:
                        try:
                            _anima_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error: {_e}")

            hybrid_event = hybrid_btn.click(
                fn=_hybrid,
                inputs=[source_prompt, base, custom_system_prompt,
                        tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Remix ──
            def _detect_format(text):
                """Detect prompt format: 'prose', 'tags', or 'hybrid'."""
                if not text:
                    return "prose"
                paragraphs = text.split("\n\n")
                first_para = paragraphs[0]
                parts = [p.strip() for p in first_para.split(",")]
                if len(parts) < 3:
                    return "prose"
                avg_len = sum(len(p) for p in parts) / len(parts)
                if avg_len >= 30 or len(parts) < 5:
                    return "prose"
                # First paragraph is tags — check if there's an NL supplement after
                if len(paragraphs) > 1 and paragraphs[1].strip():
                    return "hybrid"
                return "tags"

            def _refine(existing, existing_neg, source, tag_fmt, validation_mode,
                        *args):
                global _last_pe_mode
                _last_pe_mode = "Remix"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, th = args[-6], args[-5], args[-4]
                dd_vals = args[:-6]

                _cancel_flag.clear()
                t0 = time.monotonic()

                existing = (existing or "").strip()
                existing_neg = (existing_neg or "").strip()
                print(f"[PromptEnhancer] Remix: existing_len={len(existing)}, source_len={len((source or '').strip())}, neg={neg_cb}")
                if not existing:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: No prompt to remix. Generate one first with Prose or Tags.</span>"
                    return

                source = (source or "").strip()
                mods = _collect_modifiers(dd_vals, seed=int(sd))
                print(f"[PromptEnhancer] Remix: mods={len(mods)}, source={'yes' if source else 'no'}")

                if not mods and not source:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Select modifiers or update source prompt.</span>"
                    return

                fmt = _detect_format(existing)
                print(f"[PromptEnhancer] Remix: detected={fmt}")

                if fmt == "hybrid":
                    sp = _prompts.get("remix_hybrid", "")
                elif fmt == "tags":
                    sp = _prompts.get("remix_tags", "")
                else:
                    sp = _prompts.get("remix_prose", "")

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                if source:
                    sp = f"{sp}\n\nInstruction:\n{source}"
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"

                # Build user message — include current negative if checkbox is on
                user_msg = existing
                if neg_cb and existing_neg:
                    user_msg = f"{user_msg}\n\nCurrent negative prompt:\n{existing_neg}"

                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw, strip_underscores=(fmt == "prose"))

                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""

                    if fmt == "prose":
                        if prepend and source:
                            result = f"{source}\n\n{result}"
                        elapsed = f"{time.monotonic() - t0:.1f}s"
                        yield result, negative, f"<span style='color:#6c6'>{_MODE_REMIX}: OK - remixed to {len(result.split())} words, {elapsed}</span>"
                        return

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: Validating tags ({validation_mode})...</span>"

                    # RAG routing for remix — only when user picked RAG on
                    # User-chosen RAG path. No fallback — abort on unavailable.
                    # No shortlist here — remix doesn't re-query.
                    _anima_r = None
                    _anima_r_cm = None
                    if validation_mode == "RAG":
                        ok, reason = _rag_available_for(tag_fmt)
                        if not ok:
                            logger.error(f"RAG unavailable (remix): {reason}")
                            yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: RAG unavailable — {reason}</span>"
                            return
                        _anima_r = _get_anima_stack()
                        try:
                            _anima_r_cm = _anima_r.models()
                            _anima_r_cm.__enter__()
                        except Exception as _e:
                            logger.error(f"RAG setup failed in remix: {_e}")
                            yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: RAG setup failed: {type(_e).__name__}: {_e}</span>"
                            return
                        # Resolve any db_retrieve source: picks now. In Remix
                        # the LLM already ran with an unresolved directive
                        # (stack wasn't up yet) — this late resolution at
                        # least ensures _inject_source_picks can land the
                        # picked tag. Pre-pick prose-shaping is a Remix
                        # limitation; ◆◇ still post-fills correctly.
                        _resolve_deferred_sources(mods, int(sd), _anima_r, query=(source or existing))
                    _anima_r_safety = _anima_safety_from_modifiers(mods, source)

                    def _validate_tag_str(tag_str: str) -> tuple[str, dict | None]:
                        if _anima_r is not None:
                            return _anima_tag_from_draft(
                                _anima_r, tag_str, safety=_anima_r_safety,
                            )
                        return _postprocess_tags(tag_str, tag_fmt, validation_mode)

                    if fmt == "hybrid":
                        # Split tags and NL, post-process tags only
                        parts = result.split("\n\n", 1)
                        tag_str = parts[0].strip()
                        nl_supplement = parts[1].strip() if len(parts) > 1 else ""
                        tag_str, stats = _validate_tag_str(tag_str)
                        tag_str, stats = _inject_source_picks(tag_str, mods, stats)
                        tag_count = stats.get("total", 0) if stats else len([t for t in tag_str.split(",") if t.strip()])
                        final = f"{tag_str}\n\n{nl_supplement}" if nl_supplement else tag_str
                        status_parts = [f"remixed {tag_count} tags + NL"]
                    else:
                        # Pure tags
                        result, stats = _validate_tag_str(result)
                        result, stats = _inject_source_picks(result, mods, stats)
                        tag_count = stats.get("total", 0) if stats else len([t for t in result.split(",") if t.strip()])
                        final = result
                        status_parts = [f"remixed {tag_count} tags"]

                    # Post-process negative tags
                    if neg_cb and negative:
                        negative, _ = _validate_tag_str(negative)

                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")

                    if prepend and source:
                        final = f"{source}\n\n{final}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield final, negative, f"<span style='color:#6c6'>{_MODE_REMIX}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Cancelled</span>"
                except _TruncatedError as e:
                    if fmt == "prose":
                        # Prose truncation → truncated prose is still prose,
                        # surface it so the user can use/edit it.
                        yield _clean_output(str(e), strip_underscores=True), "", f"<span style='color:#ca6'>{_MODE_REMIX}: Truncated</span>"
                    else:
                        # Tag-mode truncation → fail loud, no silent partial.
                        yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: {type(e).__name__}: {e}</span>"
                finally:
                    # Release Anima models if loaded (remix path)
                    _r_cm = locals().get("_anima_r_cm")
                    if _r_cm is not None:
                        try:
                            _r_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error (remix): {_e}")

            remix_event = refine_btn.click(
                fn=lambda x, y: (_cancel_flag.clear(), x, y)[1:],
                _js=f"""function(x, y) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    var neg = document.querySelector('#{tab}_neg_prompt textarea');
                    return [ta ? ta.value : x, neg ? neg.value : y];
                }}""",
                inputs=[prompt_in, negative_in], outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_refine,
                inputs=[prompt_in, negative_in, source_prompt, tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Tags ──
            # Tags is Hybrid minus the NL-summary pass. Same prose → tag-extract
            # flow, same validator + slot-fill, same RAG shortlist injection.
            # Only the Hybrid 3/3 summarize step is skipped and the output is
            # tags alone (no "\n\n{nl_supplement}" suffix).
            def _tags(source, base_name, custom_sp, tag_fmt, validation_mode,
                      *args):
                global _last_pe_mode
                _last_pe_mode = "Tags"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()

                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                # User-chosen RAG path. No fallback — abort with a clear
                # error when RAG is picked but unavailable.
                _anima_t = None
                _anima_t_cm = None
                _anima_t_shortlist = None
                if validation_mode == "RAG":
                    ok, reason = _rag_available_for(tag_fmt)
                    if not ok:
                        logger.error(f"RAG unavailable: {reason}")
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: RAG unavailable — {reason}</span>"
                        return
                    _s = _get_anima_stack()
                    try:
                        _anima_t_cm = _s.models()
                        _anima_t_cm.__enter__()
                        _anima_t = _s
                        # Resolve any db_retrieve source: picks now that
                        # the stack is up (see _hybrid for the rationale).
                        if _resolve_deferred_sources(mods, int(sd), _anima_t, query=source):
                            style_str = _build_style_string(mods)
                            user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                            if style_str:
                                user_msg = f"{user_msg}\n\n{style_str}"
                            if inline_text:
                                user_msg = f"{user_msg}\n\n{inline_text}"
                        _expander_t = _make_anima_query_expander(temperature=0.3,
                            think=False, seed=int(sd),
                        )
                        _anima_t_shortlist = _s.build_shortlist(
                            source_prompt=source, modifier_keywords=style_str,
                            query_expander=_expander_t,
                        )
                        _frag = _anima_t_shortlist.as_system_prompt_fragment()
                        if _frag:
                            sp = f"{sp}\n\n{_frag}"
                        # V5 conditional adherence directive — only when
                        # source is non-empty.
                        if source:
                            sp = f"{sp}\n\n{_prompts.get('prose_adherence', '')}"
                        print(f"[PromptEnhancer] RAG shortlist: "
                              f"{len(_anima_t_shortlist.artists)} artists, "
                              f"{len(_anima_t_shortlist.characters)} characters, "
                              f"{len(_anima_t_shortlist.series)} series")
                    except Exception as _e:
                        logger.error(f"RAG setup failed: {_e}")
                        if _anima_t_cm:
                            try: _anima_t_cm.__exit__(None, None, None)
                            except Exception: pass
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: RAG setup failed: {type(_e).__name__}: {_e}</span>"
                        return

                if not source:
                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: \U0001F3B2 Rolling dice (1/2 prose)...</span>"
                print(f"[PromptEnhancer] Tags pass 1/2 (prose): model={model}, think={th}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    # Pass 1: generate prose (same as Hybrid). V8
                    # multi-sample mode when anima_tagger_prose_samples > 1.
                    n_samples = int(_anima_opt("anima_tagger_prose_samples", 3)) if _anima_t is not None else 1
                    prose_raw = None
                    if n_samples > 1:
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose (multi-sample {n_samples})...</span>"
                        prose_raw, _samples_all, _picker_choice = _multi_sample_prose(user_msg, sp, temp, seed=int(sd), n_samples=n_samples, num_predict=1024, picker_system_prompt=_prompts.get("picker", ""))
                    else:
                        for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose: {p['elapsed']:.1f}s...</span>"
                            else:
                                prose_raw = chunk
                    prose_raw = _clean_output(prose_raw)
                    if not prose_raw:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Prose generation returned empty.</span>"
                        return

                    if neg_cb:
                        prose, negative = _split_positive_negative(prose_raw)
                    else:
                        prose, negative = prose_raw, ""

                    print(f"[PromptEnhancer] Tags pass 2/2 (tags): {len(prose.split())} words → tags")

                    # Pass 2: extract tags from prose
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    tag_sp = fmt_config.get("system_prompt", "")
                    if not tag_sp:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: No tag format configured.</span>"
                        return
                    if style_str:
                        tag_sp = f"{tag_sp}\n\n{_prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
                    # V17: shortlist-aware tag_extract. See _hybrid for rationale.
                    if _anima_t_shortlist is not None:
                        _sl_frag_tag = _anima_t_shortlist.as_system_prompt_fragment()
                        if _sl_frag_tag:
                            tag_sp = f"{tag_sp}\n\n{_sl_frag_tag}"
                    # V13: Pass 2 grounding — see _hybrid for rationale.
                    if _anima_t is not None and bool(_anima_opt("anima_tagger_pass2_grounding", False)):
                        _cand = _general_tag_candidates(_anima_t, prose, k=int(_anima_opt("anima_tagger_pass2_grounding_k", 60)), min_post_count=int(_anima_opt("anima_tagger_pass2_grounding_min_pc", 100)))
                        _frag = _candidate_fragment_for_tag_sp(_cand)
                        if _frag:
                            tag_sp = f"{tag_sp}\n{_frag}"
                            print(f"[PromptEnhancer] Pass 2 grounding: {len(_cand)} candidate tags injected")
                    tags_raw = None
                    for chunk in _call_llm_progress(prose, tag_sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 2/2 tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 2/2 tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            tags_raw = chunk
                    tags_raw = _clean_output(tags_raw, strip_underscores=False)

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: Validating tags ({validation_mode})...</span>"

                    _anima_t_safety = _anima_safety_from_modifiers(mods, source)

                    # Post-process negative tags through tag pipeline
                    if neg_cb and negative:
                        if _anima_t is not None:
                            negative, _ = _anima_tag_from_draft(
                                _anima_t, negative, safety=_anima_t_safety,
                                shortlist=_anima_t_shortlist,
                            )
                        else:
                            negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    # Run tags through validator + rule layer
                    if _anima_t is not None:
                        tags_raw, stats = _anima_tag_from_draft(
                            _anima_t, tags_raw, safety=_anima_t_safety,
                            shortlist=_anima_t_shortlist,
                        )
                        # Slot-fill — same as Hybrid
                        if bool(_anima_opt("anima_tagger_slot_fill", True)):
                            slots = _active_target_slots(mods)
                            for slot in slots:
                                cat_info = _SLOT_TO_CATEGORY.get(slot)
                                if not cat_info:
                                    continue
                                cat_code = cat_info["category"]
                                if _tags_have_category(tags_raw, _anima_t, cat_code):
                                    continue
                                picked = _retrieve_prose_slot(_anima_t, prose, slot, seed=int(sd))
                                if not picked:
                                    continue
                                tag_out = picked.replace("_", " ")
                                if slot == "artist":
                                    tag_out = "@" + tag_out
                                tags_raw = f"{tags_raw}, {tag_out}" if tags_raw else tag_out
                                print(f"[PromptEnhancer] Slot fill ({slot}): injected '{tag_out}' from prose")
                                if stats:
                                    stats["total"] = stats.get("total", 0) + 1
                    else:
                        tags_raw, stats = _postprocess_tags(tags_raw, tag_fmt, validation_mode)

                    # Source post-inject: ensure every source:-picked tag
                    # survives through to the final output.
                    tags_raw, stats = _inject_source_picks(tags_raw, mods, stats)
                    tag_count = stats.get("total", 0) if stats else len([t for t in tags_raw.split(",") if t.strip()])
                    status_parts = [f"{tag_count} tags"]
                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")
                        if stats.get("error"):
                            status_parts.append(stats["error"])

                    if prepend and source:
                        tags_raw = f"{source}\n\n{tags_raw}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield tags_raw, negative, f"<span style='color:#6c6'>{_MODE_TAGS}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_TAGS}: Cancelled (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Cancelled</span>"
                except _TruncatedError:
                    # Fail loud. A truncated partial — even post-validation —
                    # is a reduced result that looks like success to the user.
                    # Empty textbox + red status makes the failure visible so
                    # the user retries instead of accepting degraded output.
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: {type(e).__name__}: {e}</span>"
                finally:
                    if _anima_t_cm is not None:
                        try:
                            _anima_t_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error (tags): {_e}")

            tags_event = tags_btn.click(
                fn=_tags,
                inputs=[source_prompt, base, custom_system_prompt,
                        tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Cancel ──
            # Only sets the threading flag. The generation function detects
            # it via InterruptedError and returns the "Cancelled" status
            # through Gradio's normal .then() output delivery.
            # - No cancels=: that kills the asyncio task, orphaning the return value
            # - No _js: DOM manipulation can desync Svelte component state
            # - No outputs: avoids racing with the generation function's status output
            # - trigger_mode="multiple": default "once" silently drops repeat clicks
            cancel_btn.click(
                fn=lambda: _cancel_flag.set(),
                inputs=[], outputs=[],
                queue=False,
                show_progress=False,
                trigger_mode="multiple",
            )

            # ── Write to main prompt textarea ──
            prompt_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[prompt_out], outputs=[prompt_out], show_progress=False,
            )

            # ── Write to negative prompt textarea ──
            negative_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_neg_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[negative_out], outputs=[negative_out], show_progress=False,
            )

        # ── Metadata ──
        def _parse_modifiers(params):
            """Parse PE Modifiers string into a set of names."""
            raw = params.get("PE Modifiers", "")
            return {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

        def _make_dd_restore(dd_label):
            """Create a restore function for a specific dropdown."""
            dd_choices = _dropdown_choices.get(dd_label, [])
            def restore(params):
                saved = _parse_modifiers(params)
                return [m for m in saved if m in dd_choices and m in _all_modifiers]
            return restore

        def _restore_tag_format(params):
            val = params.get("PE Tag Format", "")
            return val if val in _tag_formats else gr.update()

        def _restore_tag_validation(params):
            val = params.get("PE Tag Validation", "")
            return val if val in ("RAG", "Fuzzy Strict", "Fuzzy", "Off") else gr.update()

        def _restore_temperature(params):
            raw = params.get("PE Temperature", "")
            if not raw:
                return gr.update()
            try:
                return max(0.0, min(2.0, float(raw)))
            except (TypeError, ValueError):
                return gr.update()

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (detail_level, lambda params: min(10, max(0, int(params.get("PE Detail", 0)))) if params.get("PE Detail") else 0),
            (think, "PE Think"),
            (seed, lambda params: int(params.get("PE Seed", -1)) if params.get("PE Seed") else -1),
            (tag_format, _restore_tag_format),
            (tag_validation, _restore_tag_validation),
            (temperature, _restore_temperature),
            (prepend_source, lambda params: params.get("PE Prepend", "").lower() == "true"),
            (motion_cb, lambda params: params.get("PE Motion", "").lower() == "true"),
        ]
        # Add each modifier dropdown
        for i, label in enumerate(dd_labels):
            self.infotext_fields.append((dd_components[i], _make_dd_restore(label)))

        self.paste_field_names = [
            "PE Source", "PE Base", "PE Detail", "PE Modifiers",
            "PE Think", "PE Seed", "PE Tag Format", "PE Tag Validation",
            "PE Temperature", "PE Prepend", "PE Motion", "PE Mode",
        ]

        return [source_prompt, base, custom_system_prompt,
                *dd_components, prepend_source, seed, detail_level, think, temperature,
                negative_prompt_cb, tag_format, tag_validation, motion_cb]

    def process(self, p, source_prompt, base, custom_system_prompt,
                *args):
        # args = *dd_values, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, tag_format, tag_validation, motion_cb
        motion = args[-1]
        tag_validation = args[-2]
        tag_format = args[-3]
        neg_cb = args[-4]
        temp = args[-5]
        think = args[-6]
        detail_level = args[-7]
        pe_seed = args[-8]
        prepend = args[-9]
        dd_vals = args[:-9]

        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if detail_level and int(detail_level) != 0:
            p.extra_generation_params["PE Detail"] = int(detail_level)

        all_mod_names = []
        for dd_val in dd_vals:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)
        if think:
            p.extra_generation_params["PE Think"] = True
        if neg_cb:
            p.extra_generation_params["PE Negative"] = True
        if _last_seed >= 0:
            p.extra_generation_params["PE Seed"] = _last_seed
        if tag_format:
            p.extra_generation_params["PE Tag Format"] = tag_format
        if tag_validation:
            p.extra_generation_params["PE Tag Validation"] = tag_validation
        if temp is not None:
            p.extra_generation_params["PE Temperature"] = round(float(temp), 3)
        if prepend:
            p.extra_generation_params["PE Prepend"] = True
        if motion:
            p.extra_generation_params["PE Motion"] = True
        if _last_pe_mode:
            p.extra_generation_params["PE Mode"] = _last_pe_mode


# ── Settings panel registration ──────────────────────────────────────────

def _on_ui_settings():
    """Register options under Settings → Prompt Enhancer LLM and
    Settings → Anima Tagger."""
    try:
        from modules import shared
        from modules.shared import OptionInfo
    except ImportError:
        return

    try:
        import gradio as _gr_settings
    except ImportError:
        _gr_settings = None

    # ── Prompt Enhancer LLM (in-process via llama-cpp-python) ────────────
    llm_section = ("pe_llm", "Prompt Enhancer LLM")
    shared.opts.add_option(
        "pe_llm_repo_id",
        OptionInfo(
            "mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF",
            "GGUF repo (HuggingFace)",
            section=llm_section,
        ).info(
            "HuggingFace repo to download the GGUF from. Default is the "
            "Huihui Qwen 3.5 9B abliterated quants. Override with any "
            "GGUF-quants repo on HF. Ignored when 'Custom GGUF path' is set."
        ),
    )
    shared.opts.add_option(
        "pe_llm_quant",
        OptionInfo(
            "Q4_K_M",
            "Quantization",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "GGUF quant to download from the repo. Q4_K_M is ~5.6 GB "
            "for a 9B (8 GB VRAM friendly with offload). Q5_K_M is "
            "~6.5 GB, slightly higher quality. Q6_K and Q8_0 are larger."
        ),
    )
    shared.opts.add_option(
        "pe_llm_model_path",
        OptionInfo(
            "",
            "Custom GGUF path (overrides repo + quant)",
            section=llm_section,
        ).info(
            "Absolute path to a local GGUF file. When set, the repo + "
            "quant settings are ignored — useful for BYO models."
        ),
    )
    shared.opts.add_option(
        "pe_llm_n_ctx",
        OptionInfo(
            4096,
            "Context size (tokens)",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": [2048, 4096, 8192, 16384]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "Max context window. Larger = more memory used. 4096 is "
            "ample for prose + multi-pass pipelines."
        ),
    )
    shared.opts.add_option(
        "pe_llm_compute",
        OptionInfo(
            "gpu",
            "Compute target",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["gpu", "cpu", "shared"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "gpu = all layers on GPU (fastest; needs VRAM). "
            "cpu = no GPU layers (slow but no VRAM cost). "
            "shared = split, controlled by 'n_gpu_layers' below."
        ),
    )
    shared.opts.add_option(
        "pe_llm_n_gpu_layers",
        OptionInfo(
            -1,
            "GPU layers when compute = shared",
            _gr_settings.Slider if _gr_settings else None,
            {"minimum": 0, "maximum": 100, "step": 1} if _gr_settings else None,
            section=llm_section,
        ).info(
            "Number of transformer layers offloaded to GPU when "
            "compute is 'shared'. Ignored otherwise. -1 means all."
        ),
    )
    shared.opts.add_option(
        "pe_llm_lifecycle",
        OptionInfo(
            "keep_loaded",
            "Memory persistence",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["keep_loaded", "unload_after_call"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "keep_loaded = model stays in VRAM/RAM after calls (fast, "
            "greedy with VRAM). unload_after_call = release after each "
            "generation (frees VRAM for image generation, ~30 s reload "
            "on next call)."
        ),
    )
    shared.opts.add_option(
        "pe_llm_use_low_level_dry",
        OptionInfo(
            True,
            "Use low-level DRY sampler (loop suppression)",
            section=llm_section,
        ).info(
            "Drops to llama_cpp's ctypes API to build a sampler chain "
            "with the DRY (Don't Repeat Yourself) sampler. Best loop "
            "prevention for Qwen models. Disable to use the high-level "
            "samplers only (no DRY) — useful if your llama-cpp-python "
            "version doesn't expose the DRY binding."
        ),
    )

    # ── Anima Tagger (RAG / retrieval pipeline) ──────────────────────────
    section = ("anima_tagger", "Anima Tagger")

    try:
        import gradio as _gr
    except ImportError:
        _gr = None
    # RAG enable/disable lives on the main Tag Validation radio, not here.
    # Settings below are tuning knobs for power users.
    shared.opts.add_option(
        "anima_tagger_semantic_threshold",
        OptionInfo(
            0.70,
            "Semantic match threshold",
            _gr.Slider if _gr else None,
            {"minimum": 0.50, "maximum": 0.99, "step": 0.01} if _gr else None,
            section=section,
        ).info(
            "Minimum cosine similarity to accept a semantic tag substitution. "
            "Higher = stricter (more drops, fewer wrong substitutions). "
            "Default 0.70 tuned against bge-m3 behaviour on multi-word LLM drafts."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_semantic_min_post_count",
        OptionInfo(
            50,
            "Minimum post_count for semantic matches",
            _gr.Slider if _gr else None,
            {"minimum": 0, "maximum": 10000, "step": 10} if _gr else None,
            section=section,
        ).info(
            "Niche tags below this popularity can't win semantic ties "
            "(kills noise like cozy_glow matching 'cozy')."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_compound_split",
        OptionInfo(
            True,
            "Split multi-word LLM drafts into sub-tag hits",
            section=section,
        ).info(
            "If the LLM emits a phrase like 'long silver hair' that isn't "
            "itself a tag, try 2- and 1-word sub-spans (long_hair, silver_hair) "
            "before falling back to semantic match. Roughly triples usable "
            "tag output on free-text drafts."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_prose_samples",
        OptionInfo(
            3,
            "Prose samples per generation (multi-sample picker)",
            _gr.Slider if _gr else None,
            {"minimum": 1, "maximum": 5, "step": 1} if _gr else None,
            section=section,
        ).info(
            "Number of prose candidates to generate per click. A small "
            "LLM picker selects the one that best preserves source "
            "intent. 1 = off (single-sample, fastest). 3 = default, "
            "validated in experiments to give +0.5 mean on explicit-"
            "content prompts vs single-sample without regressing other "
            "prompts. Higher = slower but more variance reduction."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_slot_fill",
        OptionInfo(
            True,
            "Slot-fill: retrieve a category tag from prose for 🎲 target_slot modifiers",
            section=section,
        ).info(
            "When a 🎲 modifier declares a target_slot (e.g. Random Artist → "
            "artist, Random Franchise → copyright) and the LLM output contains "
            "no tag of that Danbooru category, retrieve the best-matching real "
            "tag from the prose and inject it. Fixes the 'Random Artist produces "
            "no artist' failure; extensible via target_slot in modifier YAML."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding",
        OptionInfo(
            False,
            "Pass 2 grounding: inject DB-retrieved candidate tags into Pass 2 system prompt (V13 — not yet rated)",
            section=section,
        ).info(
            "Root-cause fix for weird-tag generation (sparse_leg_hair, "
            "overhead_lights, simple_fish on unrelated scenes). Before Pass 2 "
            "tag-extraction, FAISS-retrieve the top-K general-category "
            "Danbooru tags semantically matching the prose and inject as a "
            "'prefer these' candidate list. Constrains the LLM to pick from "
            "real scene-relevant vocabulary instead of inventing niche "
            "compounds. Increases Pass 2 system prompt by ~600 tokens."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding_k",
        OptionInfo(
            60,
            "Pass 2 grounding: candidate pool size (top-K)",
            _gr.Slider if _gr else None,
            {"minimum": 20, "maximum": 150, "step": 10} if _gr else None,
            section=section,
        ).info(
            "Number of FAISS-retrieved candidate tags injected into Pass 2 "
            "system prompt. Smaller = tighter constraint (may miss concepts); "
            "larger = more vocabulary (more room for noise). 60 is a starting "
            "value; titrate variant V13 to measure."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding_min_pc",
        OptionInfo(
            100,
            "Pass 2 grounding: minimum post_count for candidates",
            _gr.Slider if _gr else None,
            {"minimum": 0, "maximum": 1000, "step": 50} if _gr else None,
            section=section,
        ).info(
            "Floor on popularity for candidate tags. 100 excludes ultra-niche "
            "tags like overhead_lights (pc=59). Higher = stricter."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_enable_reranker",
        OptionInfo(
            True,
            "Enable cross-encoder reranker",
            section=section,
        ).info(
            "bge-reranker-v2-m3 re-scores top candidates. Adds ~100 ms "
            "per call on GPU; improves shortlist quality."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_enable_cooccurrence",
        OptionInfo(
            True,
            "Enable character → series pairing",
            section=section,
        ).info(
            "Auto-adds the originating series tag when a character tag "
            "fires (e.g. hatsune_miku → vocaloid)."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_query_expansion",
        OptionInfo(
            True,
            "Expand source to tag concepts before shortlist retrieval",
            section=section,
        ).info(
            "Short LLM pass converts vague source prompts into richer "
            "tag-style queries so the retriever surfaces thematically-"
            "fitting artists/characters instead of name-overlap matches. "
            "Adds ~1 s per click. Disable for sparse-source workflows or "
            "when LLM latency matters more than shortlist quality."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_device",
        OptionInfo(
            "auto",
            "RAG device (bge-m3 + reranker)",
            _gr.Radio if _gr else None,
            {"choices": ["auto", "cuda", "cpu"]} if _gr else None,
            section=section,
        ).info(
            "Where to load the embedder + reranker. 'auto' picks GPU when "
            "CUDA is available, else CPU. 'cpu' saves ~2 GB VRAM for image "
            "generation but adds ~3–5 s per Anima click (CPU encoding is "
            "noticeably slower than GPU for bge-m3). Takes effect on next "
            "load — disable/re-enable the extension or restart Forge."
        ),
    )


try:
    from modules import script_callbacks
    script_callbacks.on_ui_settings(_on_ui_settings)
except ImportError:
    pass
