"""Shared helpers for the prompt-enhancer mode handlers.

Pure formatters + small builders that the four mode handlers (Prose,
Hybrid, Tags, Remix) all use. Extracting them removes ~80-100 LOC
of duplicated boilerplate from scripts/prompt_enhancer.py without
restructuring the handlers themselves.

Public API:
  status_html(mode_label, message, color="#6c6") → str
      Wraps `mode_label: message` in a colored <span>. Default color
      is the success green; pass "#c66" for error, "#ca6" for warning,
      "#aaa" for in-progress.

  progress_html(mode_label, progress_dict) → str
      Format the per-chunk LLM streaming progress dict (returned by
      pe_llm_layer.stream_llm) as the status line shown during
      generation. Picks a token-rate string when tokens > 0 else the
      "X.Xs..." pre-token form.

  build_user_msg(source, style_str, inline_text, empty_source_signal)
                                           → str
      Compose the user message: "SOURCE PROMPT: …" (or empty-source
      signal when source is empty) + style block + inline wildcard
      directive. Each piece is appended only when non-empty.

  apply_negative_hint(sp, fmt_config, negative_directive) → str
      Append the negative directive + per-format negative-quality
      hint to the system prompt. Used by Hybrid / Tags handlers when
      the user toggles "+ Negative".
"""

from __future__ import annotations

from typing import Any, Dict


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
