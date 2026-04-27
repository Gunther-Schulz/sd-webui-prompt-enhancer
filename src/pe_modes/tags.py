"""Tags mode handler — two-pass: prose → tags.

Generates pure booru-style tag output by:
  Pass 1  prose   full LLM-generated description (same as Prose mode,
                  with all modifier directives applied)
  Pass 2  tags    extract tags from the prose using the tag-format
                  system prompt + RAG shortlist (when active)

Tags mode is structurally Hybrid minus Pass 3 (NL summary). The
output is just the tag CSV — no NL supplement, no V14 prose-as-body
path. The same RAG / V13 grounding / V17 shortlist injection /
slot-fill / source-pick safety nets all apply.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator, Tuple

import gradio as gr

from pe_llm_layer import stream_llm, multi_sample_prose, TruncatedOutput
from pe_text_utils import clean_output, split_positive_negative
from pe_anima_glue.sources import SLOT_TO_CATEGORY
from pe_anima_glue.stack import opt as anima_opt
from pe_modes._shared import (
    HandlerCtx,
    status_html,
    progress_html,
    build_user_msg,
    apply_negative_hint,
)


MODE_LABEL = "🏷 Tags"


def run(
    source: str,
    base_name: str,
    custom_sp: str,
    tag_fmt: str,
    validation_mode: str,
    dd_vals: Iterable[Iterable[str]],
    *,
    prepend: bool,
    seed: int,
    detail: int,
    temperature: float,
    neg_cb: bool,
    motion_cb: bool,
    ctx: HandlerCtx,
) -> Iterator[Tuple[Any, str, str]]:
    """Tags-mode generator. Yields (prompt, negative, status_html)
    tuples. State bundled in `ctx` (see HandlerCtx)."""
    ctx.cancel_flag.clear()
    t0 = time.monotonic()
    source = (source or "").strip()

    mods = ctx.collect_modifiers(dd_vals, seed=int(seed))
    sp = ctx.assemble_system_prompt(base_name, custom_sp, detail)
    if not sp:
        yield "", "", status_html(MODE_LABEL, "No system prompt configured.", color="#c66")
        return

    if motion_cb:
        sp = f"{sp}\n\n{ctx.prompts.get('motion', '')}"
    if neg_cb:
        sp = apply_negative_hint(sp, ctx.tag_formats.get(tag_fmt, {}), ctx.prompts.get('negative', ''))

    style_str = ctx.build_style_string(mods)
    inline_text = ctx.build_inline_wildcard_text(source)
    user_msg = build_user_msg(source, style_str, inline_text, ctx.prompts.get("empty_source_signal", ""))

    # ── RAG init ──
    anima_t = None
    anima_t_cm = None
    anima_t_shortlist = None
    if validation_mode == "RAG":
        ok, reason = ctx.rag_available_for(tag_fmt)
        if not ok:
            ctx.logger.error(f"RAG unavailable: {reason}")
            yield "", "", status_html(MODE_LABEL, f"RAG unavailable — {reason}", color="#c66")
            return
        s = ctx.get_anima_stack()
        try:
            anima_t_cm = s.models()
            anima_t_cm.__enter__()
            anima_t = s
            if ctx.resolve_deferred_sources(mods, int(seed), anima_t, query=source):
                style_str = ctx.build_style_string(mods)
                user_msg = build_user_msg(source, style_str, inline_text, ctx.prompts.get("empty_source_signal", ""))
            expander = ctx.make_query_expander(temperature=0.3, seed=int(seed))
            anima_t_shortlist = s.build_shortlist(
                source_prompt=source, modifier_keywords=style_str,
                query_expander=expander,
            )
            frag = anima_t_shortlist.as_system_prompt_fragment()
            if frag:
                sp = f"{sp}\n\n{frag}"
            if source:
                sp = f"{sp}\n\n{ctx.prompts.get('prose_adherence', '')}"
            print(f"[PromptEnhancer] RAG shortlist: "
                  f"{len(anima_t_shortlist.artists)} artists, "
                  f"{len(anima_t_shortlist.characters)} characters, "
                  f"{len(anima_t_shortlist.series)} series")
        except Exception as e:
            ctx.logger.error(f"RAG setup failed: {e}")
            if anima_t_cm:
                try: anima_t_cm.__exit__(None, None, None)
                except Exception: pass
            yield "", "", status_html(MODE_LABEL, f"RAG setup failed: {type(e).__name__}: {e}", color="#c66")
            return

    if not source:
        yield gr.update(), gr.update(), status_html(MODE_LABEL, "\U0001F3B2 Rolling dice (1/2 prose)...", color="#aaa")
    print(f"[PromptEnhancer] Tags pass 1/2 (prose): seed={int(seed)}, neg={neg_cb}, dice={not source}")

    try:
        # ── Pass 1: prose ──
        n_samples = int(anima_opt("anima_tagger_prose_samples", 3)) if anima_t is not None else 1
        prose_raw = None
        if n_samples > 1:
            yield gr.update(), gr.update(), status_html(MODE_LABEL, f"1/2 prose (multi-sample {n_samples})...", color="#aaa")
            prose_raw, _samples_all, _picker_choice = multi_sample_prose(
                user_msg, sp, temperature, seed=int(seed), n_samples=n_samples,
                num_predict=1024, picker_system_prompt=ctx.prompts.get("picker", ""),
            )
        else:
            for chunk in stream_llm(user_msg, sp, temperature, seed=int(seed), cancel_flag=ctx.cancel_flag):
                if isinstance(chunk, dict):
                    yield gr.update(), gr.update(), progress_html(f"{MODE_LABEL}: 1/2 prose", chunk)
                else:
                    prose_raw = chunk
        prose_raw = clean_output(prose_raw or "")
        if not prose_raw:
            yield "", "", status_html(MODE_LABEL, "Prose generation returned empty.", color="#c66")
            return

        if neg_cb:
            prose, negative = split_positive_negative(prose_raw)
        else:
            prose, negative = prose_raw, ""

        print(f"[PromptEnhancer] Tags pass 2/2 (tags): {len(prose.split())} words → tags")

        # ── Pass 2: tags ──
        fmt_config = ctx.tag_formats.get(tag_fmt, {})
        tag_sp = fmt_config.get("system_prompt", "")
        if not tag_sp:
            yield "", "", status_html(MODE_LABEL, "No tag format configured.", color="#c66")
            return
        if style_str:
            tag_sp = f"{tag_sp}\n\n{ctx.prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
        if anima_t_shortlist is not None:
            sl_frag_tag = anima_t_shortlist.as_system_prompt_fragment()
            if sl_frag_tag:
                tag_sp = f"{tag_sp}\n\n{sl_frag_tag}"
        if anima_t is not None and bool(anima_opt("anima_tagger_pass2_grounding", False)):
            cands = ctx.general_tag_candidates(
                anima_t, prose,
                k=int(anima_opt("anima_tagger_pass2_grounding_k", 60)),
                min_post_count=int(anima_opt("anima_tagger_pass2_grounding_min_pc", 100)),
            )
            frag = ctx.candidate_fragment_for_tag_sp(cands)
            if frag:
                tag_sp = f"{tag_sp}\n{frag}"
                print(f"[PromptEnhancer] Pass 2 grounding: {len(cands)} candidate tags injected")
        tags_raw = None
        for chunk in stream_llm(prose, tag_sp, temperature, seed=int(seed), cancel_flag=ctx.cancel_flag):
            if isinstance(chunk, dict):
                yield gr.update(), gr.update(), progress_html(f"{MODE_LABEL}: 2/2 tags", chunk)
            else:
                tags_raw = chunk
        tags_raw = clean_output(tags_raw or "", strip_underscores=False)

        if validation_mode != "Off":
            yield gr.update(), gr.update(), status_html(MODE_LABEL, f"Validating tags ({validation_mode})...", color="#aaa")

        anima_t_safety = ctx.anima_safety_from_modifiers(mods, source)

        if neg_cb and negative:
            if anima_t is not None:
                negative, _ = ctx.anima_tag_from_draft(anima_t, negative, safety=anima_t_safety, shortlist=anima_t_shortlist)
            else:
                negative, _ = ctx.postprocess_tags(negative, tag_fmt, validation_mode)

        if anima_t is not None:
            tags_raw, stats = ctx.anima_tag_from_draft(anima_t, tags_raw, safety=anima_t_safety, shortlist=anima_t_shortlist)
            # Slot-fill — same as Hybrid
            if bool(anima_opt("anima_tagger_slot_fill", True)):
                slots = ctx.active_target_slots(mods)
                for slot in slots:
                    cat_info = SLOT_TO_CATEGORY.get(slot)
                    if not cat_info:
                        continue
                    cat_code = cat_info["category"]
                    if ctx.tags_have_category(tags_raw, anima_t, cat_code):
                        continue
                    picked = ctx.retrieve_prose_slot(anima_t, prose, slot, seed=int(seed))
                    if not picked:
                        continue
                    tag_out = picked.replace("_", " ")
                    if slot == "artist":
                        tag_out = "@" + tag_out
                    tags_raw = f"{tags_raw}, {tag_out}" if tags_raw else tag_out
                    print(f"[PromptEnhancer] Slot fill ({slot}): injected {tag_out!r} from prose")
                    if stats:
                        stats["total"] = stats.get("total", 0) + 1
        else:
            tags_raw, stats = ctx.postprocess_tags(tags_raw, tag_fmt, validation_mode)

        # source: post-inject safety net
        tags_raw, stats = ctx.inject_source_picks(tags_raw, mods, stats)
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
        yield tags_raw, negative, status_html(MODE_LABEL, f"OK - {', '.join(status_parts)}, {elapsed}")

    except InterruptedError as e:
        partial = clean_output(str(e))
        if partial:
            yield partial, "", status_html(MODE_LABEL, "Cancelled (partial)", color="#c66")
        else:
            yield "", "", status_html(MODE_LABEL, "Cancelled", color="#c66")
    except TruncatedOutput:
        # Fail loud — truncated tag output looks like success but is reduced.
        yield "", "", status_html(MODE_LABEL, "Truncated — no output (retry)", color="#c66")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        ctx.logger.error(msg)
        yield "", "", status_html(MODE_LABEL, msg, color="#c66")
    finally:
        if anima_t_cm is not None:
            try:
                anima_t_cm.__exit__(None, None, None)
            except Exception as e:
                ctx.logger.warning(f"anima models unload error (tags): {e}")
