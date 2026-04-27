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


def _collect_modifiers(dropdown_selections, seed: int | None = None):
    """Collect all selected modifiers into a list of (name, normalized_entry) tuples.

    Each normalized_entry is a dict with 'behavioral', 'keywords' and
    optional 'source' / 'target_slot' fields. The caller decides which
    fields to use based on the mode.

    When a modifier has a `source:` entry AND a seed is provided, the
    pick is resolved HERE (before the LLM sees any system prompt): the
    returned entry has `behavioral` and `keywords` rewritten from the
    picked real Danbooru tag. This is the `source:` mechanism — see
    _resolve_source for how the pool is built.

    Names may arrive with UI-appended ◆/◇ badges (added by
    pe_data.modifiers.build_dropdown_data); strip those before looking
    up the canonical YAML entry.
    """
    result = []
    for selections in dropdown_selections:
        for raw_name in (selections or []):
            name = _strip_mechanism_badges(raw_name)
            entry = _all_modifiers.get(name)
            if not entry:
                continue
            source = entry.get("source") if isinstance(entry, dict) else None
            if source and seed is not None:
                # db_retrieve needs the anima stack (retriever + models),
                # which isn't available at _collect_modifiers time. Defer
                # those to _resolve_deferred_sources in the handler after
                # models load. db_pattern resolves immediately here.
                is_deferred = "db_retrieve" in source and "db_pattern" not in source
                if is_deferred:
                    # Leave entry as-is; handler will resolve later and
                    # update behavioral/keywords/_resolved_from_source.
                    pass
                else:
                    picked = _resolve_source(source, seed)
                    if picked:
                        # Materialize a per-run entry with picked values baked
                        # in. Preserve target_slot / other keys so the post-fill
                        # safety net still fires when combined with source.
                        resolved = dict(entry)
                        resolved["behavioral"] = picked["behavioral"]
                        resolved["keywords"] = picked["keywords"]
                        resolved["_resolved_from_source"] = picked["name"]
                        print(f"[PromptEnhancer] Random pick ({name}): "
                              f"{picked['name']} (pool={picked['pool_size']}, "
                              f"post_count={picked['post_count']})")
                        entry = resolved
                    else:
                        print(f"[PromptEnhancer] Random pick ({name}): "
                              f"pool empty, falling back to LLM behavioral")
            result.append((name, entry))
    return result


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
                global _last_pe_mode
                _last_pe_mode = "Prose"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()
                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                # Build user message with modifiers + inline wildcards
                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                initial_status = "\U0001F3B2 Rolling dice (prose)..." if not source else "Generating prose..."
                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {initial_status}</span>"

                print(f"[PromptEnhancer] Prose: think={th}, mods={len(mods)}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_PROSE}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw)
                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""
                    if prepend and source:
                        result = f"{source}\n\n{result}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield result, negative, f"<span style='color:#6c6'>{_MODE_PROSE}: OK - {len(result.split())} words, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_PROSE}: Cancelled - {len(partial.split())} words (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: Cancelled</span>"
                except _TruncatedError as e:
                    result = _clean_output(str(e))
                    yield result, "", f"<span style='color:#ca6'>{_MODE_PROSE}: Truncated - {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: {msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_PROSE}: {msg}</span>"

            prose_event = enhance_btn.click(
                fn=_enhance,
                inputs=[source_prompt, base, custom_system_prompt]
                       + dd_components
                       + [prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Hybrid (two-pass: prose → extract tags + NL) ──
            def _hybrid(source, base_name, custom_sp, tag_fmt, validation_mode,
                        *args):
                global _last_pe_mode
                _last_pe_mode = "Hybrid"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]
                t0 = time.monotonic()

                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                # RAG path when the user picked it on the Tag Validation radio.
                # If picked but unavailable (non-Anima format, or artefacts
                # missing), surface the reason so the user sees why we fell
                # back to rapidfuzz.
                # User-chosen RAG path. No fallback — if RAG is selected
                # but unavailable, abort with a clear error. Other radio
                # modes (Fuzzy Strict / Fuzzy / Off) are separate code
                # paths; they do NOT run here as a substitute.
                _anima = None
                _anima_cm = None
                _anima_shortlist = None
                if validation_mode == "RAG":
                    ok, reason = _rag_available_for(tag_fmt)
                    if not ok:
                        logger.error(f"RAG unavailable: {reason}")
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: RAG unavailable — {reason}</span>"
                        return
                    _s = _get_anima_stack()
                    try:
                        _anima_cm = _s.models()
                        _anima_cm.__enter__()
                        _anima = _s
                        # Resolve any db_retrieve source: picks (e.g. ◆ on
                        # Random Artist) now that the stack is up. If any
                        # fire, rebuild style_str + user_msg so the picked
                        # value flows into the prompt AND the shortlist.
                        if _resolve_deferred_sources(mods, int(sd), _anima, query=source):
                            style_str = _build_style_string(mods)
                            user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                            if style_str:
                                user_msg = f"{user_msg}\n\n{style_str}"
                            if inline_text:
                                user_msg = f"{user_msg}\n\n{inline_text}"
                        _expander = _make_anima_query_expander(temperature=0.3, seed=int(sd))
                        _anima_shortlist = _s.build_shortlist(
                            source_prompt=source,
                            modifier_keywords=style_str,
                            query_expander=_expander,
                        )
                        _sl_frag = _anima_shortlist.as_system_prompt_fragment()
                        if _sl_frag:
                            sp = f"{sp}\n\n{_sl_frag}"
                        # V5 conditional adherence directive — only when
                        # source is non-empty, so dice-roll creativity
                        # stays free.
                        if source:
                            sp = f"{sp}\n\n{_prompts.get('prose_adherence', '')}"
                        # V15: on V14 Hybrid path (Anima+RAG), the prose is
                        # the primary image input — no general tags will
                        # survive the structural filter to cover pose /
                        # clothing / setting / lighting. Tell the LLM so it
                        # produces self-sufficient prose.
                        sp = f"{sp}\n\n{_prompts.get('prose_v14_coverage', '')}"
                        print(f"[PromptEnhancer] RAG shortlist: "
                              f"{len(_anima_shortlist.artists)} artists, "
                              f"{len(_anima_shortlist.characters)} characters, "
                              f"{len(_anima_shortlist.series)} series")
                    except Exception as e:
                        logger.error(f"RAG setup failed: {e}")
                        if _anima_cm:
                            try: _anima_cm.__exit__(None, None, None)
                            except Exception: pass
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: RAG setup failed: {type(e).__name__}: {e}</span>"
                        return

                if not source:
                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: \U0001F3B2 Rolling dice (1/3 prose)...</span>"
                print(f"[PromptEnhancer] Hybrid pass 1/3 (prose): think={th}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    # Pass 1: generate prose. V8 multi-sample mode when
                    # anima_tagger_prose_samples > 1 (RAG path only).
                    n_samples = int(_anima_opt("anima_tagger_prose_samples", 3)) if _anima is not None else 1
                    prose_raw = None
                    if n_samples > 1:
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose (multi-sample {n_samples})...</span>"
                        prose_raw, _samples_all, _picker_choice = _multi_sample_prose(user_msg, sp, temp, seed=int(sd), n_samples=n_samples, num_predict=1024, picker_system_prompt=_prompts.get("picker", ""))
                    else:
                        for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 1/3 prose: {p['elapsed']:.1f}s...</span>"
                            else:
                                prose_raw = chunk
                    prose_raw = _clean_output(prose_raw)
                    if not prose_raw:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Prose generation returned empty.</span>"
                        return

                    # Split negative from prose before passes 2 & 3
                    if neg_cb:
                        prose, negative = _split_positive_negative(prose_raw)
                    else:
                        prose, negative = prose_raw, ""

                    print(f"[PromptEnhancer] Hybrid pass 2/3 (tags): {len(prose.split())} words → tags")

                    # Pass 2: extract tags (tag format prompt + style context)
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    tag_sp = fmt_config.get("system_prompt", "")
                    if not tag_sp:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: No tag format configured.</span>"
                        return
                    if style_str:
                        tag_sp = f"{tag_sp}\n\n{_prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
                    # V17: shortlist-aware tag_extract. The RAG shortlist
                    # (retrieved artists / characters / series candidates)
                    # was previously injected only into the prose SP. The
                    # tag_extract LLM saw it indirectly via whatever the
                    # prose mentioned. V16 refocused tag_extract to emit
                    # ONLY structural tags (artist / character / series
                    # among them); giving it the shortlist directly
                    # sharpens precision on those categories.
                    if _anima_shortlist is not None:
                        _sl_frag_tag = _anima_shortlist.as_system_prompt_fragment()
                        if _sl_frag_tag:
                            tag_sp = f"{tag_sp}\n\n{_sl_frag_tag}"
                    # V13: Pass 2 grounding — prepend a scene-relevant candidate
                    # tag list retrieved from the DB against the prose. Constrains
                    # the LLM to pick from real scene-matching vocabulary instead
                    # of inventing niche compounds (sparse_leg_hair,
                    # overhead_lights, simple_fish). Only active with RAG + Anima.
                    if _anima is not None and bool(_anima_opt("anima_tagger_pass2_grounding", False)):
                        _cand = _general_tag_candidates(_anima, prose, k=int(_anima_opt("anima_tagger_pass2_grounding_k", 60)), min_post_count=int(_anima_opt("anima_tagger_pass2_grounding_min_pc", 100)))
                        _frag = _candidate_fragment_for_tag_sp(_cand)
                        if _frag:
                            tag_sp = f"{tag_sp}\n{_frag}"
                            print(f"[PromptEnhancer] Pass 2 grounding: {len(_cand)} candidate tags injected")
                    tags_raw = None
                    for chunk in _call_llm_progress(prose, tag_sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 2/3 tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 2/3 tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            tags_raw = chunk
                    tags_raw = _clean_output(tags_raw, strip_underscores=False)

                    # V14 path: on Anima + RAG we skip Pass 3 summarize and use
                    # full prose as the Hybrid body. The structural-tag filter
                    # applied below drops general-category tags (where FAISS
                    # mis-retrievals live), leaving only quality/safety/subject-
                    # count/@artist/character/series in the tag prefix. Anima's
                    # Qwen3 text encoder handles the prose natively, so the
                    # concepts that used to be general tags (pose, clothing,
                    # body, setting, lighting, atmosphere) are carried by prose.
                    v14_path = (_anima is not None and tag_fmt == "Anima"
                                and validation_mode == "RAG")
                    if v14_path:
                        nl_supplement = None
                    else:
                        print(f"[PromptEnhancer] Hybrid pass 3/3 (summarize): → NL supplement")
                        # Pass 3: summarize prose to 1-2 compositional sentences
                        summarize_sp = _prompts.get("summarize", "")
                        style_str = _build_style_string(mods)
                        if style_str:
                            summarize_sp = f"{summarize_sp}\n\nThe following styles were applied: {style_str} Ensure these stylistic choices are reflected in the compositional summary."
                        nl_supplement = None
                        for chunk in _call_llm_progress(prose, summarize_sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 3/3 summarize: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: 3/3 summarize: {p['elapsed']:.1f}s...</span>"
                            else:
                                nl_supplement = chunk
                        nl_supplement = _clean_output(nl_supplement)

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_HYBRID}: Validating tags ({validation_mode})...</span>"

                    # Route safety tag from any active NSFW modifier selection
                    _anima_safety = _anima_safety_from_modifiers(mods, source)

                    # Post-process negative tags through tag pipeline when applicable
                    if neg_cb and negative:
                        if _anima is not None:
                            negative, _ = _anima_tag_from_draft(
                                _anima, negative, safety=_anima_safety,
                                shortlist=_anima_shortlist,
                            )
                        else:
                            negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    # Run tags through full post-processing pipeline.
                    # When Anima retrieval is active, the bge-m3 validator
                    # + rule layer replace the rapidfuzz path entirely.
                    if _anima is not None:
                        tags_raw, stats = _anima_tag_from_draft(
                            _anima, tags_raw, safety=_anima_safety,
                            shortlist=_anima_shortlist,
                        )
                        # Slot-coverage pass: for each active modifier with a
                        # target_slot (e.g. Random Artist → artist), if no
                        # tag of that category survived validation, retrieve
                        # a top-1 from prose and inject it. Keeps "🎲 Random
                        # X" promises actually reflected in the output.
                        if bool(_anima_opt("anima_tagger_slot_fill", True)):
                            slots = _active_target_slots(mods)
                            for slot in slots:
                                cat_info = _SLOT_TO_CATEGORY.get(slot)
                                if not cat_info:
                                    continue
                                cat_code = cat_info["category"]
                                if _tags_have_category(tags_raw, _anima, cat_code):
                                    continue
                                picked = _retrieve_prose_slot(_anima, prose, slot, seed=int(sd))
                                if not picked:
                                    continue
                                # Format artist picks with @ prefix (Anima convention);
                                # other categories go in bare.
                                tag_out = picked.replace("_", " ")
                                if slot == "artist":
                                    tag_out = "@" + tag_out
                                tags_raw = f"{tags_raw}, {tag_out}" if tags_raw else tag_out
                                print(f"[PromptEnhancer] Slot fill ({slot}): injected '{tag_out}' from prose")
                                if stats:
                                    stats["total"] = stats.get("total", 0) + 1
                    else:
                        tags_raw, stats = _postprocess_tags(tags_raw, tag_fmt, validation_mode)
                    # Source post-inject: ensure every source:-picked tag
                    # survives the validator+slot_fill path. The pre-pick
                    # shaped the prose; this makes sure the picked tag
                    # actually appears in the output list.
                    tags_raw, stats = _inject_source_picks(tags_raw, mods, stats)

                    # V14: drop general+meta tags, emit {structural_tags}\n\n{prose}.
                    # The FAISS validator force-maps every draft tag to a nearest
                    # neighbor; on terse/mundane sources the neighbors are weird
                    # ("sparse leg hair" for "loose hair", "simple fish" for a
                    # garden scene, "overhead lights" for "warm light"). Anima's
                    # Qwen3 text encoder handles those concepts as prose natively,
                    # so we drop general tags by construction and let prose carry
                    # them. See experiments/variants/v14.py for rating data.
                    if v14_path:
                        tags_raw = _filter_to_structural_tags(tags_raw, _anima, tag_fmt)
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
                    yield final, negative, f"<span style='color:#6c6'>{_MODE_HYBRID}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_HYBRID}: Cancelled (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Cancelled</span>"
                except _TruncatedError:
                    # Fail loud — truncated tag output is a reduced result
                    # that looks like success. Empty textbox + red status.
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: {msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{_MODE_HYBRID}: {msg}</span>"
                finally:
                    # Release bge-m3 + reranker VRAM for image gen
                    if _anima_cm is not None:
                        try:
                            _anima_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error: {_e}")

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
            def _detect_format(text):
                """Detect prompt format: 'prose', 'tags', or 'hybrid'."""
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
                # First paragraph is tags — check if there's an NL supplement after
                if len(paragraphs) > 1 and paragraphs[1].strip():
                    return "hybrid"
                return "tags"

            def _refine(existing, existing_neg, source, tag_fmt, validation_mode,
                        *args):
                global _last_pe_mode
                _last_pe_mode = "Remix"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, th = args[-6], args[-5], args[-4]
                dd_vals = args[:-6]

                _cancel_flag.clear()
                t0 = time.monotonic()

                existing = (existing or "").strip()
                existing_neg = (existing_neg or "").strip()
                print(f"[PromptEnhancer] Remix: existing_len={len(existing)}, source_len={len((source or '').strip())}, neg={neg_cb}")
                if not existing:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: No prompt to remix. Generate one first with Prose or Tags.</span>"
                    return

                source = (source or "").strip()
                mods = _collect_modifiers(dd_vals, seed=int(sd))
                print(f"[PromptEnhancer] Remix: mods={len(mods)}, source={'yes' if source else 'no'}")

                if not mods and not source:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Select modifiers or update source prompt.</span>"
                    return

                fmt = _detect_format(existing)
                print(f"[PromptEnhancer] Remix: detected={fmt}")

                if fmt == "hybrid":
                    sp = _prompts.get("remix_hybrid", "")
                elif fmt == "tags":
                    sp = _prompts.get("remix_tags", "")
                else:
                    sp = _prompts.get("remix_prose", "")

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                if source:
                    sp = f"{sp}\n\nInstruction:\n{source}"
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"

                # Build user message — include current negative if checkbox is on
                user_msg = existing
                if neg_cb and existing_neg:
                    user_msg = f"{user_msg}\n\nCurrent negative prompt:\n{existing_neg}"

                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw, strip_underscores=(fmt == "prose"))

                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""

                    if fmt == "prose":
                        if prepend and source:
                            result = f"{source}\n\n{result}"
                        elapsed = f"{time.monotonic() - t0:.1f}s"
                        yield result, negative, f"<span style='color:#6c6'>{_MODE_REMIX}: OK - remixed to {len(result.split())} words, {elapsed}</span>"
                        return

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_REMIX}: Validating tags ({validation_mode})...</span>"

                    # RAG routing for remix — only when user picked RAG on
                    # User-chosen RAG path. No fallback — abort on unavailable.
                    # No shortlist here — remix doesn't re-query.
                    _anima_r = None
                    _anima_r_cm = None
                    if validation_mode == "RAG":
                        ok, reason = _rag_available_for(tag_fmt)
                        if not ok:
                            logger.error(f"RAG unavailable (remix): {reason}")
                            yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: RAG unavailable — {reason}</span>"
                            return
                        _anima_r = _get_anima_stack()
                        try:
                            _anima_r_cm = _anima_r.models()
                            _anima_r_cm.__enter__()
                        except Exception as _e:
                            logger.error(f"RAG setup failed in remix: {_e}")
                            yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: RAG setup failed: {type(_e).__name__}: {_e}</span>"
                            return
                        # Resolve any db_retrieve source: picks now. In Remix
                        # the LLM already ran with an unresolved directive
                        # (stack wasn't up yet) — this late resolution at
                        # least ensures _inject_source_picks can land the
                        # picked tag. Pre-pick prose-shaping is a Remix
                        # limitation; ◆◇ still post-fills correctly.
                        _resolve_deferred_sources(mods, int(sd), _anima_r, query=(source or existing))
                    _anima_r_safety = _anima_safety_from_modifiers(mods, source)

                    def _validate_tag_str(tag_str: str) -> tuple[str, dict | None]:
                        if _anima_r is not None:
                            return _anima_tag_from_draft(
                                _anima_r, tag_str, safety=_anima_r_safety,
                            )
                        return _postprocess_tags(tag_str, tag_fmt, validation_mode)

                    if fmt == "hybrid":
                        # Split tags and NL, post-process tags only
                        parts = result.split("\n\n", 1)
                        tag_str = parts[0].strip()
                        nl_supplement = parts[1].strip() if len(parts) > 1 else ""
                        tag_str, stats = _validate_tag_str(tag_str)
                        tag_str, stats = _inject_source_picks(tag_str, mods, stats)
                        tag_count = stats.get("total", 0) if stats else len([t for t in tag_str.split(",") if t.strip()])
                        final = f"{tag_str}\n\n{nl_supplement}" if nl_supplement else tag_str
                        status_parts = [f"remixed {tag_count} tags + NL"]
                    else:
                        # Pure tags
                        result, stats = _validate_tag_str(result)
                        result, stats = _inject_source_picks(result, mods, stats)
                        tag_count = stats.get("total", 0) if stats else len([t for t in result.split(",") if t.strip()])
                        final = result
                        status_parts = [f"remixed {tag_count} tags"]

                    # Post-process negative tags
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
                    yield final, negative, f"<span style='color:#6c6'>{_MODE_REMIX}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Cancelled</span>"
                except _TruncatedError as e:
                    if fmt == "prose":
                        # Prose truncation → truncated prose is still prose,
                        # surface it so the user can use/edit it.
                        yield _clean_output(str(e), strip_underscores=True), "", f"<span style='color:#ca6'>{_MODE_REMIX}: Truncated</span>"
                    else:
                        # Tag-mode truncation → fail loud, no silent partial.
                        yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_REMIX}: {type(e).__name__}: {e}</span>"
                finally:
                    # Release Anima models if loaded (remix path)
                    _r_cm = locals().get("_anima_r_cm")
                    if _r_cm is not None:
                        try:
                            _r_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error (remix): {_e}")

            remix_event = refine_btn.click(
                fn=lambda x, y: (_cancel_flag.clear(), x, y)[1:],
                _js=f"""function(x, y) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    var neg = document.querySelector('#{tab}_neg_prompt textarea');
                    return [ta ? ta.value : x, neg ? neg.value : y];
                }}""",
                inputs=[prompt_in, negative_in], outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_refine,
                inputs=[prompt_in, negative_in, source_prompt, tag_format, tag_validation]
                       + dd_components
                       + [prepend_source, seed, think, temperature, negative_prompt_cb, motion_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Tags ──
            # Tags is Hybrid minus the NL-summary pass. Same prose → tag-extract
            # flow, same validator + slot-fill, same RAG shortlist injection.
            # Only the Hybrid 3/3 summarize step is skipped and the output is
            # tags alone (no "\n\n{nl_supplement}" suffix).
            def _tags(source, base_name, custom_sp, tag_fmt, validation_mode,
                      *args):
                global _last_pe_mode
                _last_pe_mode = "Tags"
                motion_cb = args[-1]
                neg_cb, temp = args[-2], args[-3]
                prepend, sd, dl, th = args[-7], args[-6], args[-5], args[-4]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()

                source = (source or "").strip()

                mods = _collect_modifiers(dd_vals, seed=int(sd))
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: No system prompt configured.</span>"
                    return

                if motion_cb:
                    sp = f"{sp}\n\n{_prompts.get('motion', '')}"
                if neg_cb:
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                inline_text = _build_inline_wildcard_text(source)
                if inline_text:
                    user_msg = f"{user_msg}\n\n{inline_text}"

                # User-chosen RAG path. No fallback — abort with a clear
                # error when RAG is picked but unavailable.
                _anima_t = None
                _anima_t_cm = None
                _anima_t_shortlist = None
                if validation_mode == "RAG":
                    ok, reason = _rag_available_for(tag_fmt)
                    if not ok:
                        logger.error(f"RAG unavailable: {reason}")
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: RAG unavailable — {reason}</span>"
                        return
                    _s = _get_anima_stack()
                    try:
                        _anima_t_cm = _s.models()
                        _anima_t_cm.__enter__()
                        _anima_t = _s
                        # Resolve any db_retrieve source: picks now that
                        # the stack is up (see _hybrid for the rationale).
                        if _resolve_deferred_sources(mods, int(sd), _anima_t, query=source):
                            style_str = _build_style_string(mods)
                            user_msg = f"SOURCE PROMPT: {source}" if source else _prompts.get("empty_source_signal", "")
                            if style_str:
                                user_msg = f"{user_msg}\n\n{style_str}"
                            if inline_text:
                                user_msg = f"{user_msg}\n\n{inline_text}"
                        _expander_t = _make_anima_query_expander(temperature=0.3, seed=int(sd))
                        _anima_t_shortlist = _s.build_shortlist(
                            source_prompt=source, modifier_keywords=style_str,
                            query_expander=_expander_t,
                        )
                        _frag = _anima_t_shortlist.as_system_prompt_fragment()
                        if _frag:
                            sp = f"{sp}\n\n{_frag}"
                        # V5 conditional adherence directive — only when
                        # source is non-empty.
                        if source:
                            sp = f"{sp}\n\n{_prompts.get('prose_adherence', '')}"
                        print(f"[PromptEnhancer] RAG shortlist: "
                              f"{len(_anima_t_shortlist.artists)} artists, "
                              f"{len(_anima_t_shortlist.characters)} characters, "
                              f"{len(_anima_t_shortlist.series)} series")
                    except Exception as _e:
                        logger.error(f"RAG setup failed: {_e}")
                        if _anima_t_cm:
                            try: _anima_t_cm.__exit__(None, None, None)
                            except Exception: pass
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: RAG setup failed: {type(_e).__name__}: {_e}</span>"
                        return

                if not source:
                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: \U0001F3B2 Rolling dice (1/2 prose)...</span>"
                print(f"[PromptEnhancer] Tags pass 1/2 (prose): think={th}, seed={int(sd)}, neg={neg_cb}, dice={not source}")
                try:
                    # Pass 1: generate prose (same as Hybrid). V8
                    # multi-sample mode when anima_tagger_prose_samples > 1.
                    n_samples = int(_anima_opt("anima_tagger_prose_samples", 3)) if _anima_t is not None else 1
                    prose_raw = None
                    if n_samples > 1:
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose (multi-sample {n_samples})...</span>"
                        prose_raw, _samples_all, _picker_choice = _multi_sample_prose(user_msg, sp, temp, seed=int(sd), n_samples=n_samples, num_predict=1024, picker_system_prompt=_prompts.get("picker", ""))
                    else:
                        for chunk in _call_llm_progress(user_msg, sp, temp, seed=int(sd)):
                            if isinstance(chunk, dict):
                                p = chunk
                                if p["tokens"] > 0:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                                else:
                                    yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 1/2 prose: {p['elapsed']:.1f}s...</span>"
                            else:
                                prose_raw = chunk
                    prose_raw = _clean_output(prose_raw)
                    if not prose_raw:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Prose generation returned empty.</span>"
                        return

                    if neg_cb:
                        prose, negative = _split_positive_negative(prose_raw)
                    else:
                        prose, negative = prose_raw, ""

                    print(f"[PromptEnhancer] Tags pass 2/2 (tags): {len(prose.split())} words → tags")

                    # Pass 2: extract tags from prose
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    tag_sp = fmt_config.get("system_prompt", "")
                    if not tag_sp:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: No tag format configured.</span>"
                        return
                    if style_str:
                        tag_sp = f"{tag_sp}\n\n{_prompts.get('tag_extract_style_preamble', '')}\n{style_str}"
                    # V17: shortlist-aware tag_extract. See _hybrid for rationale.
                    if _anima_t_shortlist is not None:
                        _sl_frag_tag = _anima_t_shortlist.as_system_prompt_fragment()
                        if _sl_frag_tag:
                            tag_sp = f"{tag_sp}\n\n{_sl_frag_tag}"
                    # V13: Pass 2 grounding — see _hybrid for rationale.
                    if _anima_t is not None and bool(_anima_opt("anima_tagger_pass2_grounding", False)):
                        _cand = _general_tag_candidates(_anima_t, prose, k=int(_anima_opt("anima_tagger_pass2_grounding_k", 60)), min_post_count=int(_anima_opt("anima_tagger_pass2_grounding_min_pc", 100)))
                        _frag = _candidate_fragment_for_tag_sp(_cand)
                        if _frag:
                            tag_sp = f"{tag_sp}\n{_frag}"
                            print(f"[PromptEnhancer] Pass 2 grounding: {len(_cand)} candidate tags injected")
                    tags_raw = None
                    for chunk in _call_llm_progress(prose, tag_sp, temp, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 2/2 tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: 2/2 tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            tags_raw = chunk
                    tags_raw = _clean_output(tags_raw, strip_underscores=False)

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>{_MODE_TAGS}: Validating tags ({validation_mode})...</span>"

                    _anima_t_safety = _anima_safety_from_modifiers(mods, source)

                    # Post-process negative tags through tag pipeline
                    if neg_cb and negative:
                        if _anima_t is not None:
                            negative, _ = _anima_tag_from_draft(
                                _anima_t, negative, safety=_anima_t_safety,
                                shortlist=_anima_t_shortlist,
                            )
                        else:
                            negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    # Run tags through validator + rule layer
                    if _anima_t is not None:
                        tags_raw, stats = _anima_tag_from_draft(
                            _anima_t, tags_raw, safety=_anima_t_safety,
                            shortlist=_anima_t_shortlist,
                        )
                        # Slot-fill — same as Hybrid
                        if bool(_anima_opt("anima_tagger_slot_fill", True)):
                            slots = _active_target_slots(mods)
                            for slot in slots:
                                cat_info = _SLOT_TO_CATEGORY.get(slot)
                                if not cat_info:
                                    continue
                                cat_code = cat_info["category"]
                                if _tags_have_category(tags_raw, _anima_t, cat_code):
                                    continue
                                picked = _retrieve_prose_slot(_anima_t, prose, slot, seed=int(sd))
                                if not picked:
                                    continue
                                tag_out = picked.replace("_", " ")
                                if slot == "artist":
                                    tag_out = "@" + tag_out
                                tags_raw = f"{tags_raw}, {tag_out}" if tags_raw else tag_out
                                print(f"[PromptEnhancer] Slot fill ({slot}): injected '{tag_out}' from prose")
                                if stats:
                                    stats["total"] = stats.get("total", 0) + 1
                    else:
                        tags_raw, stats = _postprocess_tags(tags_raw, tag_fmt, validation_mode)

                    # Source post-inject: ensure every source:-picked tag
                    # survives through to the final output.
                    tags_raw, stats = _inject_source_picks(tags_raw, mods, stats)
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
                    yield tags_raw, negative, f"<span style='color:#6c6'>{_MODE_TAGS}: OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>{_MODE_TAGS}: Cancelled (partial)</span>"
                    else:
                        yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Cancelled</span>"
                except _TruncatedError:
                    # Fail loud. A truncated partial — even post-validation —
                    # is a reduced result that looks like success to the user.
                    # Empty textbox + red status makes the failure visible so
                    # the user retries instead of accepting degraded output.
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Truncated — no output (retry)</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{_MODE_TAGS}: {type(e).__name__}: {e}</span>"
                finally:
                    if _anima_t_cm is not None:
                        try:
                            _anima_t_cm.__exit__(None, None, None)
                        except Exception as _e:
                            logger.warning(f"anima models unload error (tags): {_e}")

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

        # ── Metadata ──
        def _parse_modifiers(params):
            """Parse PE Modifiers string into a set of names."""
            raw = params.get("PE Modifiers", "")
            return {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

        def _make_dd_restore(dd_label):
            """Create a restore function for a specific dropdown."""
            dd_choices = _dropdown_choices.get(dd_label, [])
            def restore(params):
                saved = _parse_modifiers(params)
                return [m for m in saved if m in dd_choices and m in _all_modifiers]
            return restore

        def _restore_tag_format(params):
            val = params.get("PE Tag Format", "")
            return val if val in _tag_formats else gr.update()

        def _restore_tag_validation(params):
            val = params.get("PE Tag Validation", "")
            return val if val in ("RAG", "Fuzzy Strict", "Fuzzy", "Off") else gr.update()

        def _restore_temperature(params):
            raw = params.get("PE Temperature", "")
            if not raw:
                return gr.update()
            try:
                return max(0.0, min(2.0, float(raw)))
            except (TypeError, ValueError):
                return gr.update()

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (detail_level, lambda params: min(10, max(0, int(params.get("PE Detail", 0)))) if params.get("PE Detail") else 0),
            (think, "PE Think"),
            (seed, lambda params: int(params.get("PE Seed", -1)) if params.get("PE Seed") else -1),
            (tag_format, _restore_tag_format),
            (tag_validation, _restore_tag_validation),
            (temperature, _restore_temperature),
            (prepend_source, lambda params: params.get("PE Prepend", "").lower() == "true"),
            (motion_cb, lambda params: params.get("PE Motion", "").lower() == "true"),
        ]
        # Add each modifier dropdown
        for i, label in enumerate(dd_labels):
            self.infotext_fields.append((dd_components[i], _make_dd_restore(label)))

        self.paste_field_names = [
            "PE Source", "PE Base", "PE Detail", "PE Modifiers",
            "PE Think", "PE Seed", "PE Tag Format", "PE Tag Validation",
            "PE Temperature", "PE Prepend", "PE Motion", "PE Mode",
        ]

        return [source_prompt, base, custom_system_prompt,
                *dd_components, prepend_source, seed, detail_level, think, temperature,
                negative_prompt_cb, tag_format, tag_validation, motion_cb]

    def process(self, p, source_prompt, base, custom_system_prompt,
                *args):
        # args = *dd_values, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb, tag_format, tag_validation, motion_cb
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

        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if detail_level and int(detail_level) != 0:
            p.extra_generation_params["PE Detail"] = int(detail_level)

        all_mod_names = []
        for dd_val in dd_vals:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)
        if think:
            p.extra_generation_params["PE Think"] = True
        if neg_cb:
            p.extra_generation_params["PE Negative"] = True
        if _last_seed >= 0:
            p.extra_generation_params["PE Seed"] = _last_seed
        if tag_format:
            p.extra_generation_params["PE Tag Format"] = tag_format
        if tag_validation:
            p.extra_generation_params["PE Tag Validation"] = tag_validation
        if temp is not None:
            p.extra_generation_params["PE Temperature"] = round(float(temp), 3)
        if prepend:
            p.extra_generation_params["PE Prepend"] = True
        if motion:
            p.extra_generation_params["PE Motion"] = True
        if _last_pe_mode:
            p.extra_generation_params["PE Mode"] = _last_pe_mode


# ── Settings panel registration ──────────────────────────────────────────

def _on_ui_settings():
    """Register options under Settings → Prompt Enhancer LLM and
    Settings → Anima Tagger."""
    try:
        from modules import shared
        from modules.shared import OptionInfo
    except ImportError:
        return

    try:
        import gradio as _gr_settings
    except ImportError:
        _gr_settings = None

    # ── Prompt Enhancer LLM (in-process via llama-cpp-python) ────────────
    llm_section = ("pe_llm", "Prompt Enhancer LLM")
    shared.opts.add_option(
        "pe_llm_repo_id",
        OptionInfo(
            "mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF",
            "GGUF repo (HuggingFace)",
            section=llm_section,
        ).info(
            "HuggingFace repo to download the GGUF from. Default is the "
            "Huihui Qwen 3.5 9B abliterated quants. Override with any "
            "GGUF-quants repo on HF. Ignored when 'Custom GGUF path' is set."
        ),
    )
    shared.opts.add_option(
        "pe_llm_quant",
        OptionInfo(
            "Q4_K_M",
            "Quantization",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "GGUF quant to download from the repo. Q4_K_M is ~5.6 GB "
            "for a 9B (8 GB VRAM friendly with offload). Q5_K_M is "
            "~6.5 GB, slightly higher quality. Q6_K and Q8_0 are larger."
        ),
    )
    shared.opts.add_option(
        "pe_llm_model_path",
        OptionInfo(
            "",
            "Custom GGUF path (overrides repo + quant)",
            section=llm_section,
        ).info(
            "Absolute path to a local GGUF file. When set, the repo + "
            "quant settings are ignored — useful for BYO models."
        ),
    )
    shared.opts.add_option(
        "pe_llm_n_ctx",
        OptionInfo(
            4096,
            "Context size (tokens)",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": [2048, 4096, 8192, 16384]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "Max context window. Larger = more memory used. 4096 is "
            "ample for prose + multi-pass pipelines."
        ),
    )
    shared.opts.add_option(
        "pe_llm_compute",
        OptionInfo(
            "gpu",
            "Compute target",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["gpu", "cpu", "shared"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "gpu = all layers on GPU (fastest; needs VRAM). "
            "cpu = no GPU layers (slow but no VRAM cost). "
            "shared = split, controlled by 'n_gpu_layers' below."
        ),
    )
    shared.opts.add_option(
        "pe_llm_n_gpu_layers",
        OptionInfo(
            -1,
            "GPU layers when compute = shared",
            _gr_settings.Slider if _gr_settings else None,
            {"minimum": 0, "maximum": 100, "step": 1} if _gr_settings else None,
            section=llm_section,
        ).info(
            "Number of transformer layers offloaded to GPU when "
            "compute is 'shared'. Ignored otherwise. -1 means all."
        ),
    )
    shared.opts.add_option(
        "pe_llm_lifecycle",
        OptionInfo(
            "keep_loaded",
            "Memory persistence",
            _gr_settings.Radio if _gr_settings else None,
            {"choices": ["keep_loaded", "unload_after_call"]} if _gr_settings else None,
            section=llm_section,
        ).info(
            "keep_loaded = model stays in VRAM/RAM after calls (fast, "
            "greedy with VRAM). unload_after_call = release after each "
            "generation (frees VRAM for image generation, ~30 s reload "
            "on next call)."
        ),
    )
    shared.opts.add_option(
        "pe_llm_use_low_level_dry",
        OptionInfo(
            True,
            "Use low-level DRY sampler (loop suppression)",
            section=llm_section,
        ).info(
            "Drops to llama_cpp's ctypes API to build a sampler chain "
            "with the DRY (Don't Repeat Yourself) sampler. Best loop "
            "prevention for Qwen models. Disable to use the high-level "
            "samplers only (no DRY) — useful if your llama-cpp-python "
            "version doesn't expose the DRY binding."
        ),
    )

    # ── Anima Tagger (RAG / retrieval pipeline) ──────────────────────────
    section = ("anima_tagger", "Anima Tagger")

    try:
        import gradio as _gr
    except ImportError:
        _gr = None
    # RAG enable/disable lives on the main Tag Validation radio, not here.
    # Settings below are tuning knobs for power users.
    shared.opts.add_option(
        "anima_tagger_semantic_threshold",
        OptionInfo(
            0.70,
            "Semantic match threshold",
            _gr.Slider if _gr else None,
            {"minimum": 0.50, "maximum": 0.99, "step": 0.01} if _gr else None,
            section=section,
        ).info(
            "Minimum cosine similarity to accept a semantic tag substitution. "
            "Higher = stricter (more drops, fewer wrong substitutions). "
            "Default 0.70 tuned against bge-m3 behaviour on multi-word LLM drafts."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_semantic_min_post_count",
        OptionInfo(
            50,
            "Minimum post_count for semantic matches",
            _gr.Slider if _gr else None,
            {"minimum": 0, "maximum": 10000, "step": 10} if _gr else None,
            section=section,
        ).info(
            "Niche tags below this popularity can't win semantic ties "
            "(kills noise like cozy_glow matching 'cozy')."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_compound_split",
        OptionInfo(
            True,
            "Split multi-word LLM drafts into sub-tag hits",
            section=section,
        ).info(
            "If the LLM emits a phrase like 'long silver hair' that isn't "
            "itself a tag, try 2- and 1-word sub-spans (long_hair, silver_hair) "
            "before falling back to semantic match. Roughly triples usable "
            "tag output on free-text drafts."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_prose_samples",
        OptionInfo(
            3,
            "Prose samples per generation (multi-sample picker)",
            _gr.Slider if _gr else None,
            {"minimum": 1, "maximum": 5, "step": 1} if _gr else None,
            section=section,
        ).info(
            "Number of prose candidates to generate per click. A small "
            "LLM picker selects the one that best preserves source "
            "intent. 1 = off (single-sample, fastest). 3 = default, "
            "validated in experiments to give +0.5 mean on explicit-"
            "content prompts vs single-sample without regressing other "
            "prompts. Higher = slower but more variance reduction."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_slot_fill",
        OptionInfo(
            True,
            "Slot-fill: retrieve a category tag from prose for 🎲 target_slot modifiers",
            section=section,
        ).info(
            "When a 🎲 modifier declares a target_slot (e.g. Random Artist → "
            "artist, Random Franchise → copyright) and the LLM output contains "
            "no tag of that Danbooru category, retrieve the best-matching real "
            "tag from the prose and inject it. Fixes the 'Random Artist produces "
            "no artist' failure; extensible via target_slot in modifier YAML."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding",
        OptionInfo(
            False,
            "Pass 2 grounding: inject DB-retrieved candidate tags into Pass 2 system prompt (V13 — not yet rated)",
            section=section,
        ).info(
            "Root-cause fix for weird-tag generation (sparse_leg_hair, "
            "overhead_lights, simple_fish on unrelated scenes). Before Pass 2 "
            "tag-extraction, FAISS-retrieve the top-K general-category "
            "Danbooru tags semantically matching the prose and inject as a "
            "'prefer these' candidate list. Constrains the LLM to pick from "
            "real scene-relevant vocabulary instead of inventing niche "
            "compounds. Increases Pass 2 system prompt by ~600 tokens."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding_k",
        OptionInfo(
            60,
            "Pass 2 grounding: candidate pool size (top-K)",
            _gr.Slider if _gr else None,
            {"minimum": 20, "maximum": 150, "step": 10} if _gr else None,
            section=section,
        ).info(
            "Number of FAISS-retrieved candidate tags injected into Pass 2 "
            "system prompt. Smaller = tighter constraint (may miss concepts); "
            "larger = more vocabulary (more room for noise). 60 is a starting "
            "value; titrate variant V13 to measure."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_pass2_grounding_min_pc",
        OptionInfo(
            100,
            "Pass 2 grounding: minimum post_count for candidates",
            _gr.Slider if _gr else None,
            {"minimum": 0, "maximum": 1000, "step": 50} if _gr else None,
            section=section,
        ).info(
            "Floor on popularity for candidate tags. 100 excludes ultra-niche "
            "tags like overhead_lights (pc=59). Higher = stricter."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_enable_reranker",
        OptionInfo(
            True,
            "Enable cross-encoder reranker",
            section=section,
        ).info(
            "bge-reranker-v2-m3 re-scores top candidates. Adds ~100 ms "
            "per call on GPU; improves shortlist quality."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_enable_cooccurrence",
        OptionInfo(
            True,
            "Enable character → series pairing",
            section=section,
        ).info(
            "Auto-adds the originating series tag when a character tag "
            "fires (e.g. hatsune_miku → vocaloid)."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_query_expansion",
        OptionInfo(
            True,
            "Expand source to tag concepts before shortlist retrieval",
            section=section,
        ).info(
            "Short LLM pass converts vague source prompts into richer "
            "tag-style queries so the retriever surfaces thematically-"
            "fitting artists/characters instead of name-overlap matches. "
            "Adds ~1 s per click. Disable for sparse-source workflows or "
            "when LLM latency matters more than shortlist quality."
        ),
    )
    shared.opts.add_option(
        "anima_tagger_device",
        OptionInfo(
            "auto",
            "RAG device (bge-m3 + reranker)",
            _gr.Radio if _gr else None,
            {"choices": ["auto", "cuda", "cpu"]} if _gr else None,
            section=section,
        ).info(
            "Where to load the embedder + reranker. 'auto' picks GPU when "
            "CUDA is available, else CPU. 'cpu' saves ~2 GB VRAM for image "
            "generation but adds ~3–5 s per Anima click (CPU encoding is "
            "noticeably slower than GPU for bge-m3). Takes effect on next "
            "load — disable/re-enable the extension or restart Forge."
        ),
    )


try:
    from modules import script_callbacks
    script_callbacks.on_ui_settings(_on_ui_settings)
except ImportError:
    pass
