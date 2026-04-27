"""Anima-stack singleton lifecycle.

Provides:
  opt(key, default)        — read an anima_tagger_* option from
                             shared.opts (Forge settings); return
                             default outside Forge or on read failure.
  get_stack()              — return the cached anima stack or attempt
                             load on first call. Returns None if
                             artefacts missing / load failed; the
                             error is logged + recorded in
                             load_error.
  available_for(tag_fmt)   → (bool, reason). True iff RAG can run
                             for the given format; reason explains
                             unavailability for the user.
  use_pipeline(tag_fmt, validation_mode="RAG")
                           → bool. Combined check for the mode
                             handlers — True iff user picked RAG
                             AND it's available.

State (module-level):
  _stack            cached AnimaStack or None
  _load_attempted   set True on the first load attempt so we don't
                    retry per call when the artefacts are missing
  load_error        last error message (str) when load fails

Per CLAUDE.md "no silent fallbacks" — when the user explicitly picks
RAG but it's unavailable, available_for returns False with a reason
so the caller can ABORT rather than silently downgrade.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("prompt_enhancer.pe_anima_glue.stack")


# Resolve the extension root + the src/ path that hosts the
# `anima_tagger` package. pe_anima_glue lives at
# <ext_root>/src/pe_anima_glue/stack.py — go up three levels to reach
# the extension root.
_THIS = Path(__file__).resolve()
_EXT_DIR = str(_THIS.parent.parent.parent)
_ANIMA_SRC_PATH = str(_THIS.parent.parent)


# ── State ───────────────────────────────────────────────────────────────

_stack: Optional[Any] = None
_load_attempted: bool = False
load_error: Optional[str] = None


# ── Settings access ─────────────────────────────────────────────────────


def opt(key: str, default: Any) -> Any:
    """Read an anima_tagger_* option from shared.opts. Falls back to
    default outside a Forge runtime or on read failure."""
    try:
        from modules import shared  # type: ignore
        return shared.opts.data.get(key, default)
    except Exception:
        return default


# ── Lifecycle ───────────────────────────────────────────────────────────


def get_stack() -> Optional[Any]:
    """Return the loaded AnimaStack or None.

    First call attempts the load; subsequent calls return the cached
    result. If the load fails, _load_attempted blocks future retries
    in the same process — restart Forge to retry (after fixing the
    underlying issue).
    """
    global _stack, _load_attempted, load_error
    if _stack is not None:
        return _stack
    if _load_attempted:
        return None
    _load_attempted = True
    try:
        if _ANIMA_SRC_PATH not in sys.path:
            sys.path.insert(0, _ANIMA_SRC_PATH)
        from anima_tagger import load_all   # type: ignore
        _stack = load_all(
            semantic_threshold=float(opt("anima_tagger_semantic_threshold", 0.80)),
            semantic_min_post_count=int(opt("anima_tagger_semantic_min_post_count", 100)),
            enable_reranker=bool(opt("anima_tagger_enable_reranker", True)),
            enable_cooccurrence=bool(opt("anima_tagger_enable_cooccurrence", True)),
            device=str(opt("anima_tagger_device", "auto")),
        )
        logger.info("anima_tagger loaded (DB + FAISS ready)")
        return _stack
    except Exception as e:
        tb = traceback.format_exc()
        load_error = f"{type(e).__name__}: {e}"
        # Print the full traceback prominently to stderr so the user
        # can diagnose instead of seeing a generic "artefacts not
        # loaded" message in the UI.
        print("\n" + "!" * 72, file=sys.stderr)
        print("  sd-webui-prompt-enhancer — anima_tagger load failed", file=sys.stderr)
        print(f"  {load_error}", file=sys.stderr)
        for line in tb.rstrip().splitlines():
            print(f"  {line}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
        logger.error(f"anima_tagger failed to load: {load_error}")
        return None


# ── Availability checks ─────────────────────────────────────────────────


def available_for(tag_fmt: str) -> Tuple[bool, str]:
    """True + empty reason iff the RAG pipeline can run for `tag_fmt`.
    Returns (False, reason_string) when unavailable.

    Callers must ABORT (not silently fall back) when the user picked
    RAG and it's missing — selecting RAG is an explicit choice;
    swapping in a different validation mode hides real problems.
    """
    if tag_fmt != "Anima":
        return False, (
            "RAG is currently Anima-only. Select the Anima tag format "
            "or pick a different Tag Validation mode."
        )
    if get_stack() is None:
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


def use_pipeline(tag_fmt: str, validation_mode: str = "RAG") -> bool:
    """True iff user picked RAG AND it's available for this format.
    Convenience for mode handlers."""
    if validation_mode != "RAG":
        return False
    ok, _ = available_for(tag_fmt)
    return ok
