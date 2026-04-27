"""Hybrid mode handler — three-pass: prose → tags → NL summary.

Generates a hybrid prompt (booru-style tags + short NL paragraph) by:
  Pass 1  prose          full LLM-generated description with all
                         modifier directives applied
  Pass 2  tags            extract tags from the prose using the
                         tag-format system prompt + RAG shortlist
                         (when active)
  Pass 3  NL summary      condense prose into 1-2 compositional
                         sentences (skipped on V14 Anima+RAG path)

V14 (Anima+RAG): full prose is used as the Hybrid body instead of
the summarized form, and general-category tags are filtered out
(Anima's Qwen3 text encoder handles those concepts via the prose).

Many feature flags layer on:
  - V8: multi-sample prose with picker
  - V13: Pass-2 grounding (candidate-tag injection into tag SP)
  - V17: shortlist injection into tag_extract SP
  - V19: cooccurrence carve-outs in the structural-tag filter
  - source: pre-pick + post-fill safety net
  - target_slot: slot-fill safety net
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


MODE_LABEL = "✨ Hybrid"


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
    """Hybrid-mode generator. Yields (prompt, negative, status_html)
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

    # ── RAG init (only when user picked it) ────────────────────────
    anima = None
    anima_cm = None
    anima_shortlist = None
    if validation_mode == "RAG":
        ok, reason = ctx.rag_available_for(tag_fmt)
        if not ok:
            ctx.logger.error(f"RAG unavailable: {reason}")
            yield "", "", status_html(MODE_LABEL, f"RAG unavailable — {reason}", color="#c66")
            return
        s = ctx.get_anima_stack()
        try:
            anima_cm = s.models()
            anima_cm.__enter__()
            anima = s
            # Resolve any db_retrieve source: picks now that the stack is up.
            if ctx.resolve_deferred_sources(mods, int(seed), anima, query=source):
                style_str = ctx.build_style_string(mods)
                user_msg = build_user_msg(source, style_str, inline_text, ctx.prompts.get("empty_source_signal", ""))
            expander = ctx.make_query_expander(temperature=0.3, seed=int(seed))
            anima_shortlist = s.build_shortlist(
                source_prompt=source,
                modifier_keywords=style_str,
                query_expander=expander,
            )
            sl_frag = anima_shortlist.as_system_prompt_fragment()
            if sl_frag:
                sp = f"{sp}\n\n{sl_frag}"
            # V5 conditional adherence — only when source is non-empty
            # so dice-roll creativity stays free.
            if source:
                sp = f"{sp}\n\n{ctx.prompts.get('prose_adherence', '')}"
            # V15: prose is the primary input on V14 path — tell the LLM
            # to produce self-sufficient prose since no general tags
            # will survive the structural filter.
            sp = f"{sp}\n\n{ctx.prompts.get('prose_v14_coverage', '')}"
            print(f"[PromptEnhancer] RAG shortlist: "
                  f"{len(anima_shortlist.artists)} artists, "
                  f"{len(anima_shortlist.characters)} characters, "
                  f"{len(anima_shortlist.series)} series")
        except Exception as e:
            ctx.logger.error(f"RAG setup failed: {e}")
            if anima_cm:
                try: anima_cm.__exit__(None, None, None)
                except Exception: pass
            yield "", "", status_html(MODE_LABEL, f"RAG setup failed: {type(e).__name__}: {e}", color="#c66")
            return

    if not source:
        yield gr.update(), gr.update(), status_html(MODE_LABEL, "\U0001F3B2 Rolling dice (1/3 prose)...", color="#aaa")
    print(f"[PromptEnhancer] Hybrid pass 1/3 (prose): seed={int(seed)}, neg={neg_cb}, dice={not source}")

    try:
        # ── Pass 1: prose. V8 multi-sample on RAG path. ──
        n_samples = int(anima_opt("anima_tagger_prose_samples", 3)) if anima is not None else 1
        prose_raw = None
        if n_samples > 1:
            yield gr.update(), gr.update(), status_html(MODE_LABEL, f"1/3 prose (multi-sample {n_samples})...", color="#aaa")
            prose_raw, _samples_all, _picker_choice = multi_sample_prose(
                user_msg, sp, temperature, seed=int(seed), n_samples=n_samples,
                num_predict=1024, picker_system_prompt=ctx.prompts.get("picker", ""),
            )
        else:
            for chunk in stream_llm(user_msg, sp, temperature, seed=int(seed), cancel_flag=ctx.cancel_flag):
                if isinstance(chunk, dict):
                    yield gr.update(), gr.update(), progress_html(f"{MODE_LABEL}: 1/3 prose", chunk)
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

        print(f"[PromptEnhancer] Hybrid pass 2/3 (tags): {len(prose.split())} words → tags")

        # ── Pass 2: extract tags ──
        fmt_config = ctx.tag_formats.get(tag_fmt, {})
        tag_sp = fmt_config.get("system_prompt", "")
        if not tag_sp:
            yield "", "", status_html(MODE_LABEL, "No tag format configured.", color="#c66")
            return
        if style_str:
            tag_sp = f"{tag_sp}\n\n{ctx.prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
        # V17: shortlist injection into tag_extract SP
        if anima_shortlist is not None:
            sl_frag_tag = anima_shortlist.as_system_prompt_fragment()
            if sl_frag_tag:
                tag_sp = f"{tag_sp}\n\n{sl_frag_tag}"
        # V13: Pass 2 grounding — DB candidate-tag injection
        if anima is not None and bool(anima_opt("anima_tagger_pass2_grounding", False)):
            cands = ctx.general_tag_candidates(
                anima, prose,
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
                yield gr.update(), gr.update(), progress_html(f"{MODE_LABEL}: 2/3 tags", chunk)
            else:
                tags_raw = chunk
        tags_raw = clean_output(tags_raw or "", strip_underscores=False)

        # V14 path: skip Pass 3, use prose as Hybrid body
        v14_path = (anima is not None and tag_fmt == "Anima" and validation_mode == "RAG")
        if v14_path:
            nl_supplement = None
        else:
            print(f"[PromptEnhancer] Hybrid pass 3/3 (summarize): → NL supplement")
            summarize_sp = ctx.prompts.get("summarize", "")
            if style_str:
                summarize_sp = (f"{summarize_sp}\n\nThe following styles were applied: {style_str} "
                                "Ensure these stylistic choices are reflected in the compositional summary.")
            nl_supplement = None
            for chunk in stream_llm(prose, summarize_sp, temperature, seed=int(seed), cancel_flag=ctx.cancel_flag):
                if isinstance(chunk, dict):
                    yield gr.update(), gr.update(), progress_html(f"{MODE_LABEL}: 3/3 summarize", chunk)
                else:
                    nl_supplement = chunk
            nl_supplement = clean_output(nl_supplement or "")

        if validation_mode != "Off":
            yield gr.update(), gr.update(), status_html(MODE_LABEL, f"Validating tags ({validation_mode})...", color="#aaa")

        anima_safety = ctx.anima_safety_from_modifiers(mods, source)

        # Negative tags through tag pipeline when applicable
        if neg_cb and negative:
            if anima is not None:
                negative, _ = ctx.anima_tag_from_draft(anima, negative, safety=anima_safety, shortlist=anima_shortlist)
            else:
                negative, _ = ctx.postprocess_tags(negative, tag_fmt, validation_mode)

        # Tags through full pipeline
        if anima is not None:
            tags_raw, stats = ctx.anima_tag_from_draft(anima, tags_raw, safety=anima_safety, shortlist=anima_shortlist)
            # Slot-coverage pass
            if bool(anima_opt("anima_tagger_slot_fill", True)):
                slots = ctx.active_target_slots(mods)
                for slot in slots:
                    cat_info = SLOT_TO_CATEGORY.get(slot)
                    if not cat_info:
                        continue
                    cat_code = cat_info["category"]
                    if ctx.tags_have_category(tags_raw, anima, cat_code):
                        continue
                    picked = ctx.retrieve_prose_slot(anima, prose, slot, seed=int(seed))
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

        # V14: drop general+meta tags, prose carries those concepts
        if v14_path:
            tags_raw = ctx.filter_to_structural_tags(tags_raw, anima, tag_fmt)
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
        yield final, negative, status_html(MODE_LABEL, f"OK - {', '.join(status_parts)}, {elapsed}")

    except InterruptedError as e:
        partial = clean_output(str(e))
        if partial:
            yield partial, "", status_html(MODE_LABEL, "Cancelled (partial)", color="#c66")
        else:
            yield "", "", status_html(MODE_LABEL, "Cancelled", color="#c66")
    except TruncatedOutput:
        # Fail loud — truncated tag output is reduced result that looks
        # like success. Empty textbox + red status.
        yield "", "", status_html(MODE_LABEL, "Truncated — no output (retry)", color="#c66")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        ctx.logger.error(msg)
        yield "", "", status_html(MODE_LABEL, msg, color="#c66")
    finally:
        if anima_cm is not None:
            try:
                anima_cm.__exit__(None, None, None)
            except Exception as e:
                ctx.logger.warning(f"anima models unload error: {e}")
