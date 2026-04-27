"""Shared helpers for the prompt-enhancer mode handlers.

Pure formatters + small builders + the HandlerCtx bundle that the
four mode handlers (Prose, Hybrid, Tags, Remix) all use. Extracting
them removes ~80-100 LOC of duplicated boilerplate from
scripts/prompt_enhancer.py and gives each handler a clean
single-arg state surface.

Public API:
  status_html / progress_html / build_user_msg / apply_negative_hint
                              — see docstrings on each.
  HandlerCtx                  — frozen dataclass bundling the state
                                each handler needs (prompts dict,
                                tag_formats dict, cancel_flag, logger,
                                helper callables for collect_modifiers,
                                assemble_system_prompt, etc., plus
                                Anima-glue functions for RAG paths).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict


def status_html(mode_label: str, message: str, color: str = "#6c6") -> str:
    """Format `mode_label: message` as a colored <span>."""
    return f"<span style='color:{color}'>{mode_label}: {message}</span>"


def progress_html(mode_label: str, progress: Dict[str, Any]) -> str:
    """Render an LLM progress chunk (dict from pe_llm_layer.stream_llm)
    as the in-progress status HTML. Color is fixed grey ("#aaa") since
    progress is always mid-generation. Switches between the "X words,
    Y.Ys (Z.Z tok/s)" form (post-first-token) and the "Y.Ys..." form
    (pre-first-token, after the model has loaded but before output)."""
    if progress["tokens"] > 0:
        body = (
            f"{progress['words']} words, "
            f"{progress['elapsed']:.1f}s "
            f"({progress['tps']:.1f} tok/s)"
        )
    else:
        body = f"{progress['elapsed']:.1f}s..."
    return status_html(mode_label, body, color="#aaa")


def build_user_msg(
    source: str,
    style_str: str,
    inline_text: str,
    empty_source_signal: str = "",
) -> str:
    """Compose the user message sent to the LLM.

    Pieces (each appended only when non-empty):
      "SOURCE PROMPT: <source>"           (or `empty_source_signal`)
      "Apply these styles to the scene…"  (the active modifiers)
      "<inline wildcard directive>"       (when {name?} placeholders present)

    Empty pieces are skipped — there's never a stray double-newline
    where a section was empty.
    """
    parts = []
    head = f"SOURCE PROMPT: {source}" if source else (empty_source_signal or "")
    if head:
        parts.append(head)
    if style_str:
        parts.append(style_str)
    if inline_text:
        parts.append(inline_text)
    return "\n\n".join(parts)


def apply_negative_hint(
    system_prompt: str,
    fmt_config: Dict[str, Any],
    negative_directive: str,
) -> str:
    """Append the negative directive + per-format negative-quality hint
    to the system prompt.

    `negative_directive` is the prompts.yaml `negative` text. The
    hint comes from `fmt_config["negative_quality_tags"]` and points
    the LLM at the canonical worst-quality tokens for that format.
    Both pieces are appended only when present.
    """
    neg_quality = fmt_config.get("negative_quality_tags") or []
    hint = (
        f"\nAlways start negative tags with: {', '.join(neg_quality)}"
        if neg_quality else ""
    )
    return f"{system_prompt}\n\n{negative_directive or ''}{hint}"


@dataclass(frozen=True)
class HandlerCtx:
    """State bundle each mode handler needs.

    Constructed once by prompt_enhancer.py after _reload_all and
    passed into every handler.run() call. Frozen so handlers can't
    accidentally mutate the shared state — they call into the
    bundled functions instead.

    Data:
      prompts        — loaded prompts.yaml dict (mode/picker/remix_*/…)
      tag_formats    — loaded tag-format configs by name
      cancel_flag    — shared threading.Event for the Cancel button
      logger         — extension's logging.Logger

    Helpers from prompt_enhancer.py (wrap module globals + src/
    package functions):
      collect_modifiers           — fn(dropdown_selections, seed) → mods
      assemble_system_prompt      — fn(base_name, custom_sp, detail) → str
      build_style_string          — fn(mods) → str
      build_inline_wildcard_text  — fn(source) → str

    Anima glue functions (used by Hybrid / Tags / Remix RAG paths):
      rag_available_for           — fn(tag_fmt) → (bool, reason)
      get_anima_stack             — fn() → AnimaStack | None
      resolve_deferred_sources    — fn(mods, seed, stack, query) → int
      anima_safety_from_modifiers — fn(mods, source) → str
      anima_tag_from_draft        — fn(stack, draft, *, ...) → (str, dict)
      inject_source_picks         — fn(tags_csv, mods, stats) → (str, stats)

    Tag post-processing (non-RAG fallback):
      postprocess_tags            — fn(tag_str, fmt, mode) → (str, stats|None)

    Optional (only used by Hybrid / Tags grounded Pass-2):
      make_query_expander         — fn(temperature, seed) → callable | None
      general_tag_candidates      — fn(stack, prose, k) → list[str]
      candidate_fragment_for_tag_sp — fn(candidates) → str
      retrieve_prose_slot         — fn(stack, prose, slot, seed) → str|None
      active_target_slots         — fn(mods) → list[str]
      filter_to_structural_tags   — fn(tags_csv, stack, fmt) → str
      tags_have_category          — fn(tags_csv, stack, cat) → bool
    """

    prompts: Dict[str, Any]
    tag_formats: Dict[str, Any]
    cancel_flag: threading.Event
    logger: logging.Logger

    collect_modifiers: Callable[..., list]
    assemble_system_prompt: Callable[..., Any]
    build_style_string: Callable[..., str]
    build_inline_wildcard_text: Callable[..., str]

    rag_available_for: Callable[[str], tuple]
    get_anima_stack: Callable[[], Any]
    resolve_deferred_sources: Callable[..., int]
    anima_safety_from_modifiers: Callable[..., str]
    anima_tag_from_draft: Callable[..., tuple]
    inject_source_picks: Callable[..., tuple]

    postprocess_tags: Callable[..., tuple]

    make_query_expander: Callable[..., Any]
    general_tag_candidates: Callable[..., list]
    candidate_fragment_for_tag_sp: Callable[..., str]
    retrieve_prose_slot: Callable[..., Any]
    active_target_slots: Callable[..., list]
    filter_to_structural_tags: Callable[..., str]
    tags_have_category: Callable[..., bool]
