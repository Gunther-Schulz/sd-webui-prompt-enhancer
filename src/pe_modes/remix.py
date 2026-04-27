"""Remix mode handler — modify an existing prompt with directives.

Reads the current prompt from Gradio's main textbox + the user's
modifier selections + an optional source-prompt directive, applies
the directives as a minimal patch (per the prompts.yaml `remix_*`
system prompts), and writes the result back.

Auto-detects whether the existing prompt is prose / tags / hybrid
via `_detect_format` (small heuristic on punctuation density) and
chooses the matching remix system prompt + post-processing path.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator, Tuple

import gradio as gr

from pe_llm_layer import stream_llm, TruncatedOutput
from pe_text_utils import clean_output, split_positive_negative
from pe_modes._shared import (
    HandlerCtx,
    status_html,
    progress_html,
)


MODE_LABEL = "🔀 Remix"


def detect_format(text: str) -> str:
    """Detect prompt format: 'prose', 'tags', or 'hybrid'.

    Heuristic: split into paragraphs, look at the first paragraph's
    comma-separated parts. Many short parts → tags. Few or long parts
    → prose. Tags first paragraph + non-empty NL paragraph after →
    hybrid.
    """
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
    if len(paragraphs) > 1 and paragraphs[1].strip():
        return "hybrid"
    return "tags"


def run(
    existing: str,
    existing_neg: str,
    source: str,
    tag_fmt: str,
    validation_mode: str,
    dd_vals: Iterable[Iterable[str]],
    *,
    prepend: bool,
    seed: int,
    temperature: float,
    neg_cb: bool,
    motion_cb: bool,
    ctx: HandlerCtx,
) -> Iterator[Tuple[Any, str, str]]:
    """Remix-mode generator. Yields (prompt, negative, status_html)
    tuples to Gradio. State is bundled in `ctx` (see HandlerCtx)."""
    ctx.cancel_flag.clear()
    t0 = time.monotonic()

    existing = (existing or "").strip()
    existing_neg = (existing_neg or "").strip()
    print(f"[PromptEnhancer] Remix: existing_len={len(existing)}, "
          f"source_len={len((source or '').strip())}, neg={neg_cb}")
    if not existing:
        yield "", "", status_html(
            MODE_LABEL,
            "No prompt to remix. Generate one first with Prose or Tags.",
            color="#c66",
        )
        return

    source = (source or "").strip()
    mods = ctx.collect_modifiers(dd_vals, seed=int(seed))
    print(f"[PromptEnhancer] Remix: mods={len(mods)}, source={'yes' if source else 'no'}")

    if not mods and not source:
        yield "", "", status_html(
            MODE_LABEL, "Select modifiers or update source prompt.", color="#c66",
        )
        return

    fmt = detect_format(existing)
    print(f"[PromptEnhancer] Remix: detected={fmt}")

    if fmt == "hybrid":
        sp = ctx.prompts.get("remix_hybrid", "")
    elif fmt == "tags":
        sp = ctx.prompts.get("remix_tags", "")
    else:
        sp = ctx.prompts.get("remix_prose", "")

    if motion_cb:
        sp = f"{sp}\n\n{ctx.prompts.get('motion', '')}"
    if neg_cb:
        sp = f"{sp}\n\n{ctx.prompts.get('negative', '')}"

    if source:
        sp = f"{sp}\n\nInstruction:\n{source}"
    style_str = ctx.build_style_string(mods)
    if style_str:
        sp = f"{sp}\n\n{style_str}"

    user_msg = existing
    if neg_cb and existing_neg:
        user_msg = f"{user_msg}\n\nCurrent negative prompt:\n{existing_neg}"

    anima_r = None
    anima_r_cm = None
    try:
        raw = None
        for chunk in stream_llm(user_msg, sp, temperature, seed=int(seed), cancel_flag=ctx.cancel_flag):
            if isinstance(chunk, dict):
                yield gr.update(), gr.update(), progress_html(MODE_LABEL, chunk)
            else:
                raw = chunk
        raw = clean_output(raw or "", strip_underscores=(fmt == "prose"))

        if neg_cb:
            result, negative = split_positive_negative(raw)
        else:
            result, negative = raw, ""

        # Prose path — short-circuit (no tag validation needed).
        if fmt == "prose":
            if prepend and source:
                result = f"{source}\n\n{result}"
            elapsed = f"{time.monotonic() - t0:.1f}s"
            yield result, negative, status_html(
                MODE_LABEL,
                f"OK - remixed to {len(result.split())} words, {elapsed}",
            )
            return

        # Tag / hybrid path — validate, optionally via RAG.
        if validation_mode != "Off":
            yield gr.update(), gr.update(), status_html(
                MODE_LABEL, f"Validating tags ({validation_mode})...", color="#aaa",
            )

        if validation_mode == "RAG":
            ok, reason = ctx.rag_available_for(tag_fmt)
            if not ok:
                ctx.logger.error(f"RAG unavailable (remix): {reason}")
                yield "", "", status_html(
                    MODE_LABEL, f"RAG unavailable — {reason}", color="#c66",
                )
                return
            anima_r = ctx.get_anima_stack()
            try:
                anima_r_cm = anima_r.models()
                anima_r_cm.__enter__()
            except Exception as e:
                ctx.logger.error(f"RAG setup failed in remix: {e}")
                yield "", "", status_html(
                    MODE_LABEL, f"RAG setup failed: {type(e).__name__}: {e}", color="#c66",
                )
                return
            # Resolve any db_retrieve source: picks now. In Remix the LLM
            # already ran with an unresolved directive (stack wasn't up
            # yet); this late resolution at least lets _inject_source_picks
            # land the picked tag. Pre-pick prose-shaping is a Remix
            # limitation; ◆◇ still post-fills correctly.
            ctx.resolve_deferred_sources(mods, int(seed), anima_r, query=(source or existing))
        anima_r_safety = ctx.anima_safety_from_modifiers(mods, source)

        def _validate_tag_str(tag_str):
            if anima_r is not None:
                return ctx.anima_tag_from_draft(anima_r, tag_str, safety=anima_r_safety)
            return ctx.postprocess_tags(tag_str, tag_fmt, validation_mode)

        if fmt == "hybrid":
            parts = result.split("\n\n", 1)
            tag_str = parts[0].strip()
            nl_supplement = parts[1].strip() if len(parts) > 1 else ""
            tag_str, stats = _validate_tag_str(tag_str)
            tag_str, stats = ctx.inject_source_picks(tag_str, mods, stats)
            tag_count = stats.get("total", 0) if stats else len([t for t in tag_str.split(",") if t.strip()])
            final = f"{tag_str}\n\n{nl_supplement}" if nl_supplement else tag_str
            status_parts = [f"remixed {tag_count} tags + NL"]
        else:
            result, stats = _validate_tag_str(result)
            result, stats = ctx.inject_source_picks(result, mods, stats)
            tag_count = stats.get("total", 0) if stats else len([t for t in result.split(",") if t.strip()])
            final = result
            status_parts = [f"remixed {tag_count} tags"]

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
        yield final, negative, status_html(MODE_LABEL, f"OK - {', '.join(status_parts)}, {elapsed}")

    except InterruptedError:
        yield "", "", status_html(MODE_LABEL, "Cancelled", color="#c66")
    except TruncatedOutput as e:
        if fmt == "prose":
            # Truncated prose is still prose — surface so the user can use/edit.
            partial = clean_output(getattr(e, "partial", "") or str(e), strip_underscores=True)
            yield partial, "", status_html(MODE_LABEL, "Truncated", color="#ca6")
        else:
            # Tag-mode truncation → fail loud, no silent partial.
            yield "", "", status_html(MODE_LABEL, "Truncated — no output (retry)", color="#c66")
    except Exception as e:
        yield "", "", status_html(MODE_LABEL, f"{type(e).__name__}: {e}", color="#c66")
    finally:
        if anima_r_cm is not None:
            try:
                anima_r_cm.__exit__(None, None, None)
            except Exception as e:
                ctx.logger.warning(f"anima models unload error (remix): {e}")
