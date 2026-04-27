import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request   # for tag DB download (the LLM path uses pe_llm_layer)

from rapidfuzz import distance as _rf_distance
from rapidfuzz.process import extractOne as _rf_extract_one

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory ─────────────────────────────────────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODIFIERS_DIR = os.path.join(_EXT_DIR, "modifiers")

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"
_TAGS_DIR = os.path.join(_EXT_DIR, "tags")
_TAG_FORMATS_DIR = os.path.join(_EXT_DIR, "tag-formats")

_tag_formats = {}    # format_name -> {system_prompt, use_underscores, tag_db, tag_db_url}
_tag_databases = {}  # db_filename -> set of valid tags


# ── Path setup for src/ packages ─────────────────────────────────────────
# pe_llm_layer / pe_data / pe_tags / pe_anima_glue / pe_text_utils etc all
# live under src/. Add it to sys.path before importing those packages.
# The anima_tagger package also lives under src/ but is imported lazily
# inside pe_anima_glue.stack on first RAG use.
_PE_SRC_PATH = os.path.join(_EXT_DIR, "src")
if _PE_SRC_PATH not in sys.path:
    sys.path.insert(0, _PE_SRC_PATH)


# ── pe_llm_layer (in-process LLM via llama-cpp-python) ────────────────────
# Wraps src/llm_runner. Replaces the old Ollama HTTP path. Imported eagerly:
# unlike anima_tagger (which has a rapidfuzz fallback), the LLM is required
# for every generation mode, so failing to import is a hard error.
from pe_llm_layer import (
    call_llm as _call_llm,
    stream_llm as _call_llm_progress,
    multi_sample_prose as _multi_sample_prose,
    get_llm_status as _get_llm_status,
    TruncatedOutput as _TruncatedError,
)
from pe_text_utils import (
    clean_output as _clean_output,
    clean_tag as _clean_tag,
    split_concatenated_tag as _split_concatenated_tag,
    split_positive_negative as _split_positive_negative,
)
from pe_style import (
    has_inline_wildcards as _has_inline_wildcards,
    build_inline_wildcard_text as _pe_build_inline_wildcard_text,
    build_style_string as _pe_build_style_string,
)
from pe_detail import (
    PRESET_MAX_TOKENS as _PRESET_MAX_TOKENS,
    TAG_COUNTS as _TAG_COUNTS,
    DETAIL_LABELS as _DETAIL_LABELS,
    get_word_target as _get_word_target,
    build_instruction as _build_detail_instruction,
)
from pe_data import bases as _pe_bases, prompts as _pe_prompts, modifiers as _pe_mods, tag_formats as _pe_tf
from pe_data._util import load_yaml_or_json as _load_file, get_local_dirs as _get_local_dirs
from pe_anima_glue.stack import (
    opt as _anima_opt,
    get_stack as _get_anima_stack,
    available_for as _rag_available_for,
    use_pipeline as _use_anima_pipeline,
)
from pe_anima_glue.modifiers import (
    SAFETY_TIER_ORDER as _SAFETY_TIER_ORDER,
    safety_from_modifiers as _anima_safety_from_modifiers,
    active_target_slots as _active_target_slots,
    inject_source_picks as _inject_source_picks,
    collect_modifiers as _pe_collect_modifiers,
)
from pe_anima_glue.pipeline import (
    make_query_expander as _pe_make_query_expander,
    tag_from_draft as _anima_tag_from_draft,
    general_tag_candidates as _general_tag_candidates,
    candidate_fragment_for_tag_sp as _candidate_fragment_for_tag_sp,
)
from pe_anima_glue.sources import (
    SLOT_TO_CATEGORY as _SLOT_TO_CATEGORY,
    resolve_source as _resolve_source,
    resolve_deferred_sources as _resolve_deferred_sources,
    retrieve_prose_slot as _retrieve_prose_slot,
)
from pe_anima_glue.filters import (
    SUBJECT_COUNT_RE as _SUBJECT_COUNT_RE,
    STRUCTURAL_CATEGORIES as _STRUCTURAL_CATEGORIES,
    tags_have_category as _tags_have_category,
    filter_to_structural_tags as _pe_filter_to_structural_tags,
)
from pe_anima_glue import stack as _pe_anima_stack
from pe_modes._shared import (
    status_html as _status_html,
    progress_html as _progress_html,
    build_user_msg as _build_user_msg,
    apply_negative_hint as _apply_negative_hint,
)
from pe_modes import prose as _pe_prose, remix as _pe_remix, hybrid as _pe_hybrid, tags as _pe_tags
from pe_modes._shared import HandlerCtx as _HandlerCtx
import pe_metadata as _pe_metadata
from types import SimpleNamespace as _SimpleNamespace
from pe_tags import (
    validate as _pe_tags_validate,
    reorder as _pe_tags_reorder,
    postprocess as _pe_tags_postprocess,
    format_tag_out as _format_tag_out,
    find_closest_tag as _find_closest_tag,
    TAG_CORRECTIONS as _TAG_CORRECTIONS,
    SUBJECT_TAGS as _SUBJECT_TAGS,
    PRESERVE_UNDERSCORE_RE as _PRESERVE_UNDERSCORE_RE,
)

# Aliases for the historical underscore-prefixed names used at callsites.
_BADGE_SOURCE = _pe_mods.BADGE_SOURCE
_BADGE_TARGET_SLOT = _pe_mods.BADGE_TARGET_SLOT






def _make_anima_query_expander(temperature=0.3, seed=-1):
    """Thin wrapper around pe_anima_glue.pipeline.make_query_expander.
    Threads _call_llm in so the new module stays LLM-layer-agnostic."""
    return _pe_make_query_expander(_call_llm, temperature=temperature, seed=seed)






    category = entry["category"]
    min_post = entry["min_post"]
    prefer_pop = entry.get("prefer_popularity", False)
    try:
        # Pull a wider pool for variety; we'll pick one from it.
        final_k = 10
        cands = stack.retriever.retrieve(
            prose, retrieve_k=200, final_k=final_k,
            category=category, min_post_count=min_post,
        )
        if not cands:
            return None
        if prefer_pop:
            return max(cands, key=lambda c: c.post_count).name
        # Seed-driven pick from top-K — same seed reproducible,
        # different seeds yield different artists.
        import random as _random
        rng = _random.Random(int(seed) if seed not in (None, -1) else _random.randint(0, 2**31 - 1))
        return rng.choice(cands).name
    except Exception as e:
        logger.warning(f"prose slot retrieval failed ({slot}): {e}")
        return None





    try:
        r = stack.retriever
        q_vec = r.embedder.encode_one(prose)
        # retrieve_k=3000 gives plenty of headroom — FAISS is fast, and
        # we need enough hits that the category=0 filter leaves ≥k.
        scores, ids = r.index.search(q_vec, 3000)
        dense_ids = [int(i) for i in ids[0] if i >= 0]
        tags = r.db.get_by_ids(dense_ids)
        score_by_id = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}
        # Filter to general (cat=0) + popularity floor + dedupe
        seen: set[int] = set()
        pool: list[dict] = []
        for t in tags:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            if t["category"] != 0:
                continue
            if t["post_count"] < min_post_count:
                continue
            pool.append(t)
        pool.sort(key=lambda t: (score_by_id.get(t["id"], 0.0), t["post_count"]), reverse=True)
        return [t["name"] for t in pool[:k]]
    except Exception as e:
        logger.warning(f"_general_tag_candidates: retrieval failed: {e}")
        return []










    for t in tag_csv.split(","):
        norm = t.strip().lstrip("@").lower().replace(" ", "_").replace("-", "_")
        if not norm:
            continue
        rec = stack.db.get_by_name(norm)
        if rec and rec.get("category") == category:
            return True
    return False


def _filter_to_structural_tags(tag_csv, stack, tag_fmt):
    """Thin wrapper threading the format-config from _tag_formats."""
    fmt_config = _tag_formats.get(tag_fmt, {})
    return _pe_filter_to_structural_tags(tag_csv, stack, fmt_config)


def _load_tag_formats():
    """Populate _tag_formats from the tag-formats/ directory.
    Implementation lives in pe_data.tag_formats."""
    global _tag_formats
    _tag_formats = _pe_tf.load(_TAG_FORMATS_DIR)



# ── File loading ─────────────────────────────────────────────────────────────

# ── Tag database ─────────────────────────────────────────────────────────────

def _download_tag_db(fmt_config):
    """Ensure tag DB is on disk. Delegates to pe_data.tag_formats."""
    return _pe_tf.download_db(fmt_config, _TAGS_DIR, logger=logger)


def _load_tag_db(tag_format):
    """Load tag DB into memory, returning the set of valid tags.
    Caches into _tag_databases keyed by filename. Delegates the
    actual file read to pe_data.tag_formats."""
    fmt_config = _tag_formats.get(tag_format, {})
    return _pe_tf.load_db(fmt_config, _TAGS_DIR, _tag_databases, logger=logger)



# Universal quality tokens — alias to pe_data.tag_formats.UNIVERSAL_QUALITY_TAGS.
_UNIVERSAL_QUALITY_TAGS = _pe_tf.UNIVERSAL_QUALITY_TAGS



def _validate_tags(tags_str, tag_format, mode="Fuzzy"):
    """Thin wrapper around pe_tags.validate that threads the
    extension's tag-format config + DB + aliases through. Modes:
      Fuzzy        — exact + alias + fuzzy correction, keep unrecognized
      Fuzzy Strict — exact + alias + fuzzy correction, drop unrecognized"""
    valid_tags = _load_tag_db(tag_format)
    if not valid_tags:
        return tags_str, {"error": "No tag database available"}
    fmt_config = _tag_formats.get(tag_format, {})
    db_filename = fmt_config.get("tag_db", "")
    aliases = _tag_databases.get(f"{db_filename}_aliases", {})
    return _pe_tags_validate(
        tags_str, fmt_config, valid_tags, aliases,
        mode=mode, clean_tag=_clean_tag,
    )




def _reorder_tags(tags, tag_format):
    """Thin wrapper around pe_tags.reorder. Order: quality →
    leading → subjects → rest → rating."""
    fmt_config = _tag_formats.get(tag_format, {})
    return _pe_tags_reorder(tags, fmt_config, _UNIVERSAL_QUALITY_TAGS)


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}          # flat: name -> keywords (for lookup across all dropdowns)
_dropdown_order = []         # list of dropdown labels in display order
_dropdown_choices = {}       # label -> [choice_list with separators]
_prompts = {}                # operational prompts loaded from prompts.yaml


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _dropdown_order, _dropdown_choices, _prompts

    local_dirs = _get_local_dirs(local_dir_path)

    # Bases (YAML, with local overrides) — loader lives in pe_data.bases.
    _bases = _pe_bases.load(_EXT_DIR, local_dirs)

    # Modifiers — loader lives in pe_data.modifiers.
    _all_modifiers, _dropdown_order, _dropdown_choices = _pe_mods.load_all(
        _MODIFIERS_DIR, local_dirs,
    )

    # Prompts (YAML, with local overrides) — loader lives in pe_data.prompts.
    _prompts = _pe_prompts.load(_EXT_DIR, local_dirs)


_reload_all()


_load_tag_formats()




# ── Helpers ──────────────────────────────────────────────────────────────────

# Thin wrappers around pe_data.bases (imported as _pe_bases above) so
# the rest of this file keeps the historical _base_* identifier shape.
def _base_body(entry):
    return _pe_bases.body(entry)


def _base_meta(name):
    return _pe_bases.meta(_bases, name)


def _base_names():
    return _pe_bases.names(_bases)


_strip_mechanism_badges = _pe_mods.strip_badges


def _collect_modifiers(dropdown_selections, seed=None):
    """Thin wrapper that threads the _all_modifiers global to
    pe_anima_glue.modifiers.collect_modifiers."""
    return _pe_collect_modifiers(dropdown_selections, _all_modifiers, seed=seed)


_build_style_string = _pe_build_style_string


# ── Ollama ───────────────────────────────────────────────────────────────────


# Placeholder used in the user message when Source Prompt is empty (dice roll).
# Styles and wildcards still flow through normally; this only replaces the
# "SOURCE PROMPT: {source}" line so the LLM knows to invent rather than expand.




# ── Core logic ───────────────────────────────────────────────────────────────


def _postprocess_tags(tag_str, tag_fmt, validation_mode):
    """Thin wrapper around pe_tags.postprocess — full tag pipeline:
    split-concatenated → normalize → validate → reorder → escape
    parens. Returns (processed_tags, stats_or_None)."""
    fmt_config = _tag_formats.get(tag_fmt, {})
    valid_tags = _load_tag_db(tag_fmt)
    db_filename = fmt_config.get("tag_db", "")
    aliases = _tag_databases.get(f"{db_filename}_aliases", {})
    return _pe_tags_postprocess(
        tag_str, fmt_config, valid_tags, aliases,
        validation_mode=validation_mode,
        clean_tag=_clean_tag,
        split_concat=_split_concatenated_tag,
        universal_quality_tags=_UNIVERSAL_QUALITY_TAGS,
    )




# _cancel_flag is owned by pe_llm_layer so the LLM stream and the
# cancel button operate on the same Event. (Renamed import preserves
# the historical _cancel_flag identifier used throughout this file.)
from pe_llm_layer import cancel_flag as _cancel_flag
_last_seed = -1
# Which PE button produced the currently-staged prompt. Set by each
# handler entry point, consumed by process() to write PE Mode into image
# metadata and shown in every status line via _MODE_* prefixes below.
_last_pe_mode: str | None = None

# Status line prefix per mode — matches the button glyphs so the status
# message is self-identifying at a glance.
_MODE_PROSE  = "\u270d Prose"   # ✍ Prose
_MODE_HYBRID = "\u2728 Hybrid"  # ✨ Hybrid
_MODE_TAGS   = "\U0001F3F7 Tags"  # 🏷 Tags
_MODE_REMIX  = "\U0001F500 Remix" # 🔀 Remix



# ── V5 conditional adherence directive ──────────────────────────────
# V5 prose-adherence directive and V15 prose-v14-coverage directive are
# defined in prompts.yaml (keys: `prose_adherence`, `prose_v14_coverage`).
# Appended to the prose SP by the Hybrid handler when active. The V8
# picker SP is also in prompts.yaml (key: `picker`). See prompts.yaml
# for the full text; keeping all LLM-facing strings in yaml ensures no
# "hidden" prompt influence lives in Python.






def _build_inline_wildcard_text(source):
    """Thin wrapper threading _prompts (loaded YAML) to pe_style."""
    return _pe_build_inline_wildcard_text(source, _prompts)


def _assemble_system_prompt(base_name, custom_system_prompt, detail=3):
    """Thin wrapper over pe_data.bases.assemble_system_prompt — reads
    forge_preset from shared.opts at call time and delegates the actual
    assembly. Returns None (not '') when the base body is empty so
    existing callsites that check for falsy stay correct."""
    try:
        from modules import shared
        preset = getattr(shared.opts, "forge_preset", "sd")
    except Exception:
        preset = "sd"
    sp = _pe_bases.assemble_system_prompt(
        _bases, base_name, custom_system_prompt, detail=detail, preset=preset,
    )
    return sp or None


def _build_handler_ctx():
    """Construct the HandlerCtx bundle that mode handlers consume.
    Reads from this module's loaded state and the imported helper functions.
    Cheap (just packs references); called once per handler invocation."""
    return _HandlerCtx(
        prompts=_prompts,
        tag_formats=_tag_formats,
        cancel_flag=_cancel_flag,
        logger=logger,
        collect_modifiers=_collect_modifiers,
        assemble_system_prompt=_assemble_system_prompt,
        build_style_string=_build_style_string,
        build_inline_wildcard_text=_build_inline_wildcard_text,
        rag_available_for=_rag_available_for,
        get_anima_stack=_get_anima_stack,
        resolve_deferred_sources=_resolve_deferred_sources,
        anima_safety_from_modifiers=_anima_safety_from_modifiers,
        anima_tag_from_draft=_anima_tag_from_draft,
        inject_source_picks=_inject_source_picks,
        postprocess_tags=_postprocess_tags,
        make_query_expander=_make_anima_query_expander,
        general_tag_candidates=_general_tag_candidates,
        candidate_fragment_for_tag_sp=_candidate_fragment_for_tag_sp,
        retrieve_prose_slot=_retrieve_prose_slot,
        active_target_slots=_active_target_slots,
        filter_to_structural_tags=_filter_to_structural_tags,
        tags_have_category=_tags_have_category,
    )





# ── UI ───────────────────────────────────────────────────────────────────────

class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        # The old Ollama-discovery block (poll /api/tags for available
        # models) is gone — the in-process runner reads model/quant/
        # compute from shared.opts.
        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt ──
            source_prompt = gr.Textbox(
                label="Source Prompt", lines=3,
                placeholder="Type your prompt here, or leave empty to roll the dice. Use {name?} for inline wildcards.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(value="\u270d Prose", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_enhance_btn")
                hybrid_btn = gr.Button(value="\u2728 Hybrid", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_hybrid_btn")
                tags_btn = gr.Button(value="\U0001f3f7 Tags", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_tags_btn")
                refine_btn = gr.Button(value="\U0001f500 Remix", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_refine_btn")
                cancel_btn = gr.Button(value="\u274c Cancel", scale=0, min_width=80, elem_id=f"{tab}_pe_cancel_btn")
                prepend_source = gr.Checkbox(label="Prepend", value=False, scale=0, min_width=60)
                prepend_source.do_not_save_to_config = True
                negative_prompt_cb = gr.Checkbox(label="+ Negative", value=False, scale=0, min_width=110)
                negative_prompt_cb.do_not_save_to_config = True
                motion_cb = gr.Checkbox(label="+ Motion and Audio", value=False, scale=0, min_width=150)
                motion_cb.do_not_save_to_config = True
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            # ── Base + Tag Format + Validation ──
            with gr.Row():
                base = gr.Dropdown(label="Base", choices=_base_names(), value="Default", scale=1, info="Prose voice — matches the image model family.")
                _tf_names = list(_tag_formats.keys())
                tag_format = gr.Dropdown(label="Tag Format", choices=_tf_names, value=_tf_names[0] if _tf_names else "", scale=1, info="Tag conventions for booru-trained fine-tunes.")
                tag_validation = gr.Radio(
                    label="Tag Validation",
                    choices=["RAG", "Fuzzy Strict", "Fuzzy", "Off"],
                    value="RAG", scale=2,
                    info="RAG=retrieval + embedding validator (Anima) | Fuzzy Strict=guess+drop | Fuzzy=guess+keep | Off=raw",
                )
                tag_validation.do_not_save_to_config = True

            def _base_description_html(name):
                if name == "Custom":
                    return "<div style='color:#888; font-size:0.9em; margin-top:-8px; padding-left:4px'>User-supplied system prompt (below). Bypasses the shared preamble and format.</div>"
                meta = _base_meta(name)
                desc = meta.get("description", "")
                if not desc:
                    return ""
                return f"<div style='color:#888; font-size:0.9em; margin-top:-8px; padding-left:4px'>{desc}</div>"

            base_description = gr.HTML(value=_base_description_html("Default"))

            # ── Auto-generated modifier dropdowns (one per file) ──
            dd_components = []
            dd_labels = list(_dropdown_order)

            # Layout: 3 dropdowns per row, pad incomplete rows
            for i in range(0, len(dd_labels), 3):
                row_labels = dd_labels[i:i+3]
                with gr.Row():
                    for label in row_labels:
                        d = gr.Dropdown(
                            label=label,
                            choices=_dropdown_choices.get(label, []),
                            value=[], multiselect=True, scale=1,
                        )
                        d.do_not_save_to_config = True
                        dd_components.append(d)
                    # Pad incomplete rows so dropdowns don't stretch
                    for _ in range(3 - len(row_labels)):
                        gr.HTML(value="", visible=True, scale=1)

            # ── Temperature + Think + Seed ──
            # detail_level is kept as a hidden-valued component (always 0)
            # so input positions in .click handlers stay stable without
            # requiring all four generator signatures to be refactored.
            # detail=0 routes through _build_detail_instruction → None,
            # so no word/tag count is ever injected into system prompts.
            detail_level = gr.Number(value=0, visible=False)
            detail_level.do_not_save_to_config = True
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.8, step=0.05, scale=2, info="0 = deterministic, 2 = creative")
                seed = gr.Number(label="Seed", value=-1, minimum=-1, step=1, scale=1, info="-1 = random", precision=0, elem_id=f"{tab}_pe_seed")
                seed.do_not_save_to_config = True
                seed_random_btn = ToolButton(value="\U0001f3b2", elem_id=f"{tab}_pe_seed_random")
                seed_reuse_btn = ToolButton(value="\u267b", elem_id=f"{tab}_pe_seed_reuse")
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)
                think.do_not_save_to_config = True
                seed_random_btn.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
                seed_reuse_btn.click(fn=lambda: _last_seed, inputs=[], outputs=[seed], show_progress=False)

            # ── Custom system prompt ──
            custom_system_prompt = gr.Textbox(label="Custom System Prompt", lines=4, visible=False, placeholder="Enter your custom system prompt...")
            base.change(fn=lambda b: gr.update(visible=(b == "Custom")), inputs=[base], outputs=[custom_system_prompt], show_progress=False)
            base.change(fn=_base_description_html, inputs=[base], outputs=[base_description], show_progress=False)

            # ── LLM model controls ──
            # The in-process llm_runner reads its model + compute config
            # from shared.opts (see Forge settings, "Prompt Enhancer LLM"
            # section). No per-tab API URL or model dropdown — the runner
            # is process-global.
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_dir_path = gr.Textbox(
                    label="Local Overrides",
                    placeholder=f"Using: {_env_local}" if _env_local else "Comma-separated dirs (refreshes content only, restart for new dropdowns)",
                    scale=3,
                )
                local_dir_path.do_not_save_to_config = True
                reload_btn = gr.Button(value="\U0001f504 Reload", scale=0, min_width=100)
                llm_status_btn = gr.Button(value="\U0001f504 LLM", scale=0, min_width=100)
            llm_status = gr.HTML(value=_get_llm_status())

            # ── Reload wiring ──
            # Note: reload rebuilds dropdowns but can't add/remove them dynamically.
            # New files require a Forge restart. Existing dropdown contents are refreshed.
            def _do_refresh(current_base, *args):
                # Last arg is local_dir_path
                local_path = args[-1]
                dd_vals = args[:-1]

                _reload_all(local_path)
                results = [gr.update(choices=_base_names(), value=current_base if current_base in _bases else "Default")]
                for i, label in enumerate(dd_labels):
                    choices = _dropdown_choices.get(label, [])
                    old_val = dd_vals[i] if i < len(dd_vals) else []
                    results.append(gr.update(choices=choices, value=[v for v in (old_val or []) if v in _all_modifiers]))
                msg = (f"<span style='color:#6c6'>Reloaded: {len(_bases)} bases, "
                       f"{len(_dropdown_order)} modifier groups, "
                       f"{len(_all_modifiers)} modifiers, "
                       f"{len(_prompts)} prompts</span>")
                results.append(msg)
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base] + dd_components + [local_dir_path],
                outputs=[base] + dd_components + [status],
                show_progress=False,
            )
            llm_status_btn.click(fn=_get_llm_status, inputs=[], outputs=[llm_status], show_progress=False)

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")
            negative_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_in")
            negative_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_out")

            # ── Prose ──
            def _enhance(source, base_name, custom_sp, *args):
                # *args = *dd_vals, prepend, seed, detail, think, temperature, neg_cb, motion_cb
                global _last_pe_mode
                _last_pe_mode = "Prose"
                motion_cb = args[-1]
                neg_cb = args[-2]
                temp = args[-3]
                # th = args[-4]  # think — kept in inputs for compat, ignored by runner
                dl = args[-5]
                sd = args[-6]
                prepend = args[-7]
                dd_vals = args[:-7]
                yield from _pe_prose.run(
                    source, base_name, custom_sp, dd_vals,
                    prepend=prepend, seed=sd, detail=dl,
                    temperature=temp, neg_cb=neg_cb, motion_cb=motion_cb,
                    ctx=_build_handler_ctx(),
                )

            prose_event = enhance_btn.click(
                fn=_enhance,
                inputs=[source_prompt, base, custom_system_prompt]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Hybrid (three-pass: prose → tags → NL summary) ──
            def _hybrid(source, base_name, custom_sp, tag_fmt, validation_mode, *args):
                # *args = *dd_vals, prepend, seed, detail, think, temperature, neg_cb, motion_cb
                global _last_pe_mode
                _last_pe_mode = "Hybrid"
                motion_cb = args[-1]
                neg_cb = args[-2]
                temp = args[-3]
                # think arg ignored
                dl = args[-5]
                sd = args[-6]
                prepend = args[-7]
                dd_vals = args[:-7]
                yield from _pe_hybrid.run(
                    source, base_name, custom_sp, tag_fmt, validation_mode, dd_vals,
                    prepend=prepend, seed=sd, detail=dl,
                    temperature=temp, neg_cb=neg_cb, motion_cb=motion_cb,
                    ctx=_build_handler_ctx(),
                )

            hybrid_event = hybrid_btn.click(
                fn=_hybrid,
                inputs=[source_prompt, base, custom_system_prompt,
                        tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Remix ──
            def _refine(existing, existing_neg, source, tag_fmt, validation_mode, *args):
                # *args = *dd_vals, prepend, seed, detail, think, temperature, neg_cb, motion_cb
                global _last_pe_mode
                _last_pe_mode = "Remix"
                motion_cb = args[-1]
                neg_cb = args[-2]
                temp = args[-3]
                # think arg ignored (in-process runner controls thinking)
                # detail unused by Remix; consumed positionally
                sd = args[-6]
                prepend = args[-7]
                dd_vals = args[:-7]
                yield from _pe_remix.run(
                    existing, existing_neg, source, tag_fmt, validation_mode, dd_vals,
                    prepend=prepend, seed=sd, temperature=temp,
                    neg_cb=neg_cb, motion_cb=motion_cb,
                    ctx=_build_handler_ctx(),
                )

            remix_event = refine_btn.click(
                fn=lambda x, y: (_cancel_flag.clear(), x, y)[1:],
                _js=f"""function(x, y) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    var neg = document.querySelector('#{tab}_neg_prompt textarea');
                    return [ta ? ta.value : '', neg ? neg.value : ''];
                }}""",
                inputs=[prompt_in, negative_in],
                outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_refine,
                inputs=[prompt_in, negative_in, source_prompt, tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Tags (two-pass: prose → extract tags) ──
            def _tags(source, base_name, custom_sp, tag_fmt, validation_mode, *args):
                # *args = *dd_vals, prepend, seed, detail, think, temperature, neg_cb, motion_cb
                global _last_pe_mode
                _last_pe_mode = "Tags"
                motion_cb = args[-1]
                neg_cb = args[-2]
                temp = args[-3]
                # think arg ignored
                dl = args[-5]
                sd = args[-6]
                prepend = args[-7]
                dd_vals = args[:-7]
                yield from _pe_tags.run(
                    source, base_name, custom_sp, tag_fmt, validation_mode, dd_vals,
                    prepend=prepend, seed=sd, detail=dl,
                    temperature=temp, neg_cb=neg_cb, motion_cb=motion_cb,
                    ctx=_build_handler_ctx(),
                )

            tags_event = tags_btn.click(
                fn=_tags,
                inputs=[source_prompt, base, custom_system_prompt,
                        tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Cancel ──
            # Only sets the threading flag. The generation function detects
            # it via InterruptedError and returns the "Cancelled" status
            # through Gradio's normal .then() output delivery.
            # - No cancels=: that kills the asyncio task, orphaning the return value
            # - No _js: DOM manipulation can desync Svelte component state
            # - No outputs: avoids racing with the generation function's status output
            # - trigger_mode="multiple": default "once" silently drops repeat clicks
            cancel_btn.click(
                fn=lambda: _cancel_flag.set(),
                inputs=[], outputs=[],
                queue=False,
                show_progress=False,
                trigger_mode="multiple",
            )

            # ── Write to main prompt textarea ──
            prompt_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[prompt_out], outputs=[prompt_out], show_progress=False,
            )

            # ── Write to negative prompt textarea ──
            negative_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_neg_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[negative_out], outputs=[negative_out], show_progress=False,
            )

        # ── Metadata round-trip (extracted to pe_metadata) ──
        _pe_components = _SimpleNamespace(
            source_prompt=source_prompt,
            base=base,
            detail_level=detail_level,
            think=think,
            seed=seed,
            tag_format=tag_format,
            tag_validation=tag_validation,
            temperature=temperature,
            prepend_source=prepend_source,
            motion_cb=motion_cb,
            dd_components=dd_components,
        )
        _pe_restore = _pe_metadata.build_restore_funcs(
            _tag_formats, _all_modifiers, _dropdown_choices, gr,
        )
        self.infotext_fields = _pe_metadata.build_infotext_fields(
            _pe_components, _pe_restore, dd_labels,
        )
        self.paste_field_names = list(_pe_metadata.PASTE_FIELD_NAMES)

        return [source_prompt, base, custom_system_prompt,
                *dd_components, prepend_source, seed, detail_level, think, temperature,
                negative_prompt_cb, tag_format, tag_validation, motion_cb]

    def process(self, p, source_prompt, base, custom_system_prompt, *args):
        """Forge image-generation hook — write our params into the
        saved PNG's metadata via pe_metadata."""
        # args = *dd_values, prepend, seed, detail, think, temperature, neg_cb, tag_format, tag_validation, motion
        motion = args[-1]
        tag_validation = args[-2]
        tag_format = args[-3]
        neg_cb = args[-4]
        temp = args[-5]
        think = args[-6]
        detail_level = args[-7]
        pe_seed = args[-8]
        prepend = args[-9]
        dd_vals = args[:-9]
        _pe_metadata.apply_to_extra_generation_params(
            p,
            source_prompt=source_prompt, base=base,
            detail_level=detail_level, dd_vals=dd_vals,
            think=think, neg_cb=neg_cb, last_seed=_last_seed,
            tag_format=tag_format, tag_validation=tag_validation,
            temperature=temp, prepend=prepend, motion=motion,
            last_pe_mode=_last_pe_mode,
        )


# ── Settings panel registration ──────────────────────────────────────────

def _on_ui_settings():
    """Register Forge settings — implementation lives in pe_settings."""
    try:
        from modules import shared
        from modules.shared import OptionInfo
    except ImportError:
        return
    try:
        import gradio as _gr_settings
    except ImportError:
        _gr_settings = None
    from pe_settings import register as _pe_register_settings
    _pe_register_settings(shared, OptionInfo, _gr_settings)


try:
    from modules import script_callbacks
    script_callbacks.on_ui_settings(_on_ui_settings)
except ImportError:
    pass
