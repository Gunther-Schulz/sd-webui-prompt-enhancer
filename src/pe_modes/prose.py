"""Prose mode handler.

Pure generator that yields (prompt, negative, status_html) tuples to
Gradio. Owned by pe_modes/, not nested inside scripts/prompt_enhancer.py.

All extension state (prompts dict, modifier lookup, base loader, …)
is passed in via keyword arguments. No module-level coupling to
prompt_enhancer.py — the click-handler wrapper in ui() constructs
the state once and calls run() for each invocation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

import gradio as gr

from pe_llm_layer import stream_llm, TruncatedOutput
from pe_text_utils import clean_output, split_positive_negative
from pe_modes._shared import status_html, progress_html, build_user_msg


MODE_LABEL = "✍ Prose"   # ✍ Prose


def run(
    source: str,
    base_name: str,
    custom_sp: str,
    dd_vals: Iterable[Iterable[str]],
    *,
    prepend: bool,
    seed: int,
    detail: int,
    temperature: float,
    neg_cb: bool,
    motion_cb: bool,
    prompts: Dict[str, Any],
    collect_modifiers: Callable[..., list],
    assemble_system_prompt: Callable[..., Optional[str]],
    build_style_string: Callable[..., str],
    build_inline_wildcard_text: Callable[..., str],
    cancel_flag: threading.Event,
    logger: logging.Logger,
) -> Iterator[Tuple[Any, str, str]]:
    """Prose-mode generator.

    Yields (prompt_text, negative_text, status_html) tuples. The
    prompt and negative slots receive `gr.update()` (no change) for
    in-progress yields and the actual text for the final yield.

    Inputs:
      source         — user's source prompt (may be empty)
      base_name      — selected Base name
      custom_sp      — custom system prompt body when base_name="Custom"
      dd_vals        — iterable of dropdown selection lists
      prepend        — checkbox: prepend source to result?
      seed           — int seed (-1 for random)
      detail         — detail-level slider (0-10)
      temperature    — LLM temperature
      neg_cb         — checkbox: + Negative
      motion_cb      — checkbox: + Motion and Audio

    State (kwargs):
      prompts                — loaded prompts.yaml dict
      collect_modifiers      — fn(dropdown_selections, seed) → mod list
      assemble_system_prompt — fn(base_name, custom_sp, detail) → str|None
      build_style_string     — fn(mods) → str
      build_inline_wildcard_text — fn(source) → str
      cancel_flag            — threading.Event for Cancel button
      logger                 — extension's logging.Logger
    """
    cancel_flag.clear()
    t0 = time.monotonic()
    source = (source or "").strip()

    mods = collect_modifiers(dd_vals, seed=int(seed))
    sp = assemble_system_prompt(base_name, custom_sp, detail)
    if not sp:
        yield "", "", status_html(MODE_LABEL, "No system prompt configured.", color="#c66")
        return

    if motion_cb:
        sp = f"{sp}\n\n{prompts.get('motion', '')}"
    if neg_cb:
        sp = f"{sp}\n\n{prompts.get('negative', '')}"

    style_str = build_style_string(mods)
    inline_text = build_inline_wildcard_text(source)
    user_msg = build_user_msg(source, style_str, inline_text, prompts.get("empty_source_signal", ""))

    initial = "\U0001F3B2 Rolling dice (prose)..." if not source else "Generating prose..."
    yield gr.update(), gr.update(), status_html(MODE_LABEL, initial, color="#aaa")

    print(f"[PromptEnhancer] Prose: mods={len(mods)}, seed={int(seed)}, neg={neg_cb}, dice={not source}")
    try:
        raw = None
        for chunk in stream_llm(user_msg, sp, temperature, seed=int(seed), cancel_flag=cancel_flag):
            if isinstance(chunk, dict):
                yield gr.update(), gr.update(), progress_html(MODE_LABEL, chunk)
            else:
                raw = chunk
        raw = clean_output(raw or "")

        if neg_cb:
            result, negative = split_positive_negative(raw)
        else:
            result, negative = raw, ""
        if prepend and source:
            result = f"{source}\n\n{result}"

        elapsed = f"{time.monotonic() - t0:.1f}s"
        yield result, negative, status_html(MODE_LABEL, f"OK - {len(result.split())} words, {elapsed}")

    except InterruptedError as e:
        partial = clean_output(str(e))
        if partial:
            yield partial, "", status_html(
                MODE_LABEL,
                f"Cancelled - {len(partial.split())} words (partial)",
                color="#c66",
            )
        else:
            yield "", "", status_html(MODE_LABEL, "Cancelled", color="#c66")

    except TruncatedOutput as e:
        result = clean_output(getattr(e, "partial", "") or str(e))
        yield result, "", status_html(
            MODE_LABEL, f"Truncated - {len(result.split())} words", color="#ca6"
        )

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(msg)
        yield "", "", status_html(MODE_LABEL, msg, color="#c66")
