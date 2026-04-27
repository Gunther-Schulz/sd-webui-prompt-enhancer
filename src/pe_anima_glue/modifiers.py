"""Modifier-driven Anima behaviors.

  collect_modifiers(dropdown_selections, all_modifiers, seed=None)
                                    → list[(name, entry)]
      Resolve the user's Gradio dropdown selections into normalized
      (name, entry) tuples. For modifiers with a `source:` entry +
      seed, performs the db_pattern source pick HERE so the picked
      value flows into the LLM system prompt. db_retrieve is
      deferred (needs the loaded stack — handler resolves later via
      pe_anima_glue.sources.resolve_deferred_sources).

  safety_from_modifiers(mod_list)   → str
      Resolves the safety tag (`safe`/`sensitive`/`nsfw`/`explicit`)
      from each active modifier's `safety_tier` YAML field. Most
      permissive tier wins.
  active_target_slots(mod_list)     → list[str]
      Collects unique `target_slot` values from active modifiers
      (preserving discovery order). Used by the slot-fill pass to
      decide which DB categories to retrieve from prose.
  inject_source_picks(tags_csv, mod_list, stats=None)
                                    → (tags_csv, stats)
      Post-fill safety net for `source:`-driven modifiers — appends
      any `_resolved_from_source` value that didn't make it into the
      final tag list, so source+target_slot modifiers have the
      strictly-stronger ◆◇ guarantee.

These operate on the (name, entry) tuples produced by
collect_modifiers. The first three need only modifier metadata;
collect_modifiers calls into sources.resolve_source for db_pattern
picks (which doesn't need the loaded stack — only the DB file).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.modifiers")


# Tier precedence — most permissive first. A modifier YAML entry
# declares its safety tier via a `safety_tier` field with one of
# these values. The most-permissive (lowest-index) active tier wins.
SAFETY_TIER_ORDER = ("explicit", "nsfw", "sensitive", "safe")


# ── Modifier collection (Gradio selections → resolved entries) ──────────


def collect_modifiers(
    dropdown_selections: Iterable[Iterable[str]],
    all_modifiers: dict,
    seed: Optional[int] = None,
) -> List[Tuple[str, Any]]:
    """Resolve Gradio dropdown selections into (name, entry) tuples.

    `dropdown_selections` is an iterable of iterables — one inner list
    per dropdown, each holding raw selection strings (which may carry
    UI-appended ◆/◇ badges from pe_data.modifiers.build_dropdown_data).
    `all_modifiers` is the flat lookup loaded by
    pe_data.modifiers.load_all (canonical YAML names → normalized entries).

    For modifiers with a `source:` block AND a non-None seed, the pick
    is resolved here:
      - db_pattern: resolved immediately (pure DB lookup, no models).
      - db_retrieve: deferred — the loaded stack isn't available yet
        and the caller (mode handler) re-resolves later.

    The resolved entry has `behavioral` / `keywords` overwritten from
    the picked Danbooru tag and gets a `_resolved_from_source`
    breadcrumb so post-fill safety nets can detect "I made this pick,
    make sure it survives validation".

    Returns the list of (name, entry) tuples in selection order.
    """
    # Lazy import to avoid pe_anima_glue.modifiers ↔ pe_anima_glue.sources
    # cycle issues (sources imports nothing from this module, but keep
    # the dep direction explicit anyway).
    from . import sources as _sources
    from pe_data.modifiers import strip_badges

    result: List[Tuple[str, Any]] = []
    for selections in dropdown_selections:
        for raw_name in (selections or []):
            name = strip_badges(raw_name)
            entry = all_modifiers.get(name)
            if not entry:
                continue
            source = entry.get("source") if isinstance(entry, dict) else None
            if source and seed is not None:
                # db_retrieve needs the loaded stack; defer to the handler.
                # db_pattern is a pure DB lookup; resolve immediately.
                is_deferred = "db_retrieve" in source and "db_pattern" not in source
                if is_deferred:
                    pass  # leave entry as-is; handler resolves later
                else:
                    picked = _sources.resolve_source(source, seed)
                    if picked:
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


def safety_from_modifiers(
    mod_list: Iterable[Tuple[str, Any]],
    source: str = "",
) -> str:
    """Resolve the safety tag from active modifiers via the `safety_tier`
    YAML field. Returns "safe" when no modifier declares a tier.

    Precedence: most permissive tier among active modifiers wins.
    Rating detection from raw `source` text is deferred to the LLM
    (the V16 tag-extract system prompt handles that). The `source`
    arg is retained for callsite compatibility but is unused.

    Knowledge lives in YAML, not Python: the tier is declared on the
    modifier itself; adding a new modifier with a safety implication
    never requires a code edit.
    """
    del source  # rating signal comes from modifiers' own declarations
    winning_index = len(SAFETY_TIER_ORDER)
    for _name, entry in mod_list:
        tier = entry.get("safety_tier") if isinstance(entry, dict) else None
        if tier in SAFETY_TIER_ORDER:
            idx = SAFETY_TIER_ORDER.index(tier)
            if idx < winning_index:
                winning_index = idx
    return SAFETY_TIER_ORDER[winning_index] if winning_index < len(SAFETY_TIER_ORDER) else "safe"


def active_target_slots(mod_list: Iterable[Tuple[str, Any]]) -> List[str]:
    """Collect the unique target_slot values declared by active modifiers.

    Reads `target_slot` from each entry; preserves discovery order so
    the downstream injection is stable across runs.
    """
    seen: set = set()
    out: List[str] = []
    for _name, entry in mod_list or []:
        if not isinstance(entry, dict):
            continue
        slot = entry.get("target_slot")
        if slot and slot not in seen:
            seen.add(slot)
            out.append(slot)
    return out


def inject_source_picks(
    tags_csv: str,
    mod_list: Iterable[Tuple[str, Any]],
    stats: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Post-fill safety net for `source:`-driven modifiers.

    After tag validation + slot_fill, ensure every `source:`-picked
    tag actually survives in the final tag list. The pre-pick already
    shapes the prose (LLM writes a 1970s scene if we said 1970s), but
    the tag-extraction pass can fail to emit the literal decade tag
    if the LLM described the era in concrete terms without naming it.

    Mirrors `target_slot:`'s category-coverage pass — the pick was
    made with a seed for reproducibility; this just makes sure it
    survives into the output. Together the two passes give
    source+target_slot modifiers the strictly-stronger ◆◇ guarantee.

    Tags are converted underscore→space to match the anima output form;
    other formats apply their own transforms downstream. Stats dict
    is mutated in place when entries are added.
    """
    tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
    existing = {t.lower().replace("_", " ").strip() for t in tags}
    added = 0
    for name, entry in mod_list or []:
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
        print(f"[PromptEnhancer] Source inject ({name}): appended {pick_spaced!r}")
    if added and stats is not None:
        stats["total"] = stats.get("total", 0) + added
        stats["source_injected"] = stats.get("source_injected", 0) + added
    return ", ".join(tags), (stats or {})
