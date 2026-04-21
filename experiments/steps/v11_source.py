"""Step functions for source-mechanism support (◆ / ◆◇ randoms).

Two steps introduced for V11 to test the cumulative effect of this
session's post-V8 changes:

    resolve_deferred_sources — mirrors pe._resolve_deferred_sources.
        Runs after the stack is loaded. Resolves db_retrieve source:
        picks (e.g. ◆◇ Random Artist) that couldn't resolve at
        _collect_modifiers time because the FAISS retriever wasn't up.

    source_inject — mirrors pe._inject_source_picks.
        Runs after slot_fill. Ensures every source:-picked tag actually
        appears in the final tag list (post-fill safety net).

Neither step re-implements the logic: both dispatch to the shipped
functions via the pe bootstrap, so any divergence is a real contract
violation, not test drift.
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.steps.common import get_stack, pe


def resolve_deferred_sources(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: mods, source, seed.
    Write: mods (mutated in place — deferred db_retrieve sources resolved).

    Calls get_stack() to ensure models are loaded (free if already up).
    Then asks pe to resolve any entries that _collect_modifiers left
    pending because they declared db_retrieve. Returns updated mods.
    """
    mods = state.get("mods", [])
    source = state.get("source", "")
    seed = state.get("seed", -1)
    stack = get_stack()
    resolved = pe._resolve_deferred_sources(mods, int(seed), stack, query=source)
    return {**state, "mods": mods, "deferred_resolved_count": resolved}


def source_inject(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: final_tags (from slot_fill), mods.
    Write: final_tags (with every source-picked tag appended if missing).

    The pre-pick already shaped the prose (LLM saw the directive); this
    guarantees the literal picked tag lands in the output regardless of
    whether Pass 2 happened to emit it verbatim.
    """
    tags_csv = state.get("final_tags", "")
    mods = state.get("mods", [])
    new_tags, stats = pe._inject_source_picks(tags_csv, mods, stats={})
    return {
        **state,
        "final_tags": new_tags,
        "source_inject_stats": stats,
    }
