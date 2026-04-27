"""Modifier-driven Anima behaviors.

Small, self-contained operations that read the active-modifier list
to drive Anima-pipeline behaviors:

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

All three are pure operations on the (name, entry) tuples produced
by `_collect_modifiers` in scripts/prompt_enhancer.py. No anima_tagger
dependency — they look at modifier metadata only.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.modifiers")


# Tier precedence — most permissive first. A modifier YAML entry
# declares its safety tier via a `safety_tier` field with one of
# these values. The most-permissive (lowest-index) active tier wins.
SAFETY_TIER_ORDER = ("explicit", "nsfw", "sensitive", "safe")


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
