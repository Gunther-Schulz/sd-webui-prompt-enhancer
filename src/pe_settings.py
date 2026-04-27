"""Forge settings registration for the prompt-enhancer extension.

Two settings sections live under Forge's Settings tab:

  Prompt Enhancer LLM   (id: pe_llm)
      Configures the in-process llama-cpp-python runner: GGUF repo,
      quant, custom path, context size, compute target, GPU layers,
      lifecycle, low-level DRY toggle.

  Anima Tagger          (id: anima_tagger)
      Tuning knobs for the RAG pipeline: semantic threshold, post-
      count floor, compound split, multi-sample count, slot-fill,
      Pass-2 grounding (V13), reranker / cooccurrence / query-
      expansion toggles, RAG device.

The `register(shared, OptionInfo, gr_settings)` function below is
called by `_on_ui_settings` in scripts/prompt_enhancer.py — that
hook gives us the live shared/OptionInfo references; gr_settings is
the gradio module (or None outside Forge).

Pure declarative — no logic, no state. Easy to maintain because the
section is alphabetized within itself but otherwise just the 8 LLM
options + 11 Anima options as you scroll.
"""

from __future__ import annotations

from typing import Any


def register(shared: Any, OptionInfo: Any, gr_settings: Any) -> None:
    """Register the extension's two Forge settings sections."""
    _register_llm_section(shared, OptionInfo, gr_settings)
    _register_anima_tagger_section(shared, OptionInfo, gr_settings)


# ── Prompt Enhancer LLM ─────────────────────────────────────────────────


def _register_llm_section(shared, OptionInfo, gr_settings):
    section = ("pe_llm", "Prompt Enhancer LLM")
    shared.opts.add_option(
        "pe_llm_repo_id",
        OptionInfo(
            "mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF",
            "GGUF repo (HuggingFace)",
            section=section,
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
            gr_settings.Radio if gr_settings else None,
            {"choices": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]} if gr_settings else None,
            section=section,
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
            section=section,
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
            gr_settings.Radio if gr_settings else None,
            {"choices": [2048, 4096, 8192, 16384]} if gr_settings else None,
            section=section,
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
            gr_settings.Radio if gr_settings else None,
            {"choices": ["gpu", "cpu", "shared"]} if gr_settings else None,
            section=section,
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
            gr_settings.Slider if gr_settings else None,
            {"minimum": 0, "maximum": 100, "step": 1} if gr_settings else None,
            section=section,
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
            gr_settings.Radio if gr_settings else None,
            {"choices": ["keep_loaded", "unload_after_call"]} if gr_settings else None,
            section=section,
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
            section=section,
        ).info(
            "Drops to llama_cpp's ctypes API to build a sampler chain "
            "with the DRY (Don't Repeat Yourself) sampler. Best loop "
            "prevention for Qwen models. Disable to use the high-level "
            "samplers only (no DRY) — useful if your llama-cpp-python "
            "version doesn't expose the DRY binding."
        ),
    )


# ── Anima Tagger (RAG / retrieval pipeline) ─────────────────────────────


def _register_anima_tagger_section(shared, OptionInfo, gr_settings):
    section = ("anima_tagger", "Anima Tagger")
    # RAG enable/disable lives on the main Tag Validation radio, not here.
    # Settings below are tuning knobs for power users.
    shared.opts.add_option(
        "anima_tagger_semantic_threshold",
        OptionInfo(
            0.70,
            "Semantic match threshold",
            gr_settings.Slider if gr_settings else None,
            {"minimum": 0.50, "maximum": 0.99, "step": 0.01} if gr_settings else None,
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
            gr_settings.Slider if gr_settings else None,
            {"minimum": 0, "maximum": 10000, "step": 10} if gr_settings else None,
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
            gr_settings.Slider if gr_settings else None,
            {"minimum": 1, "maximum": 5, "step": 1} if gr_settings else None,
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
            gr_settings.Slider if gr_settings else None,
            {"minimum": 20, "maximum": 150, "step": 10} if gr_settings else None,
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
            gr_settings.Slider if gr_settings else None,
            {"minimum": 0, "maximum": 1000, "step": 50} if gr_settings else None,
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
            gr_settings.Radio if gr_settings else None,
            {"choices": ["auto", "cuda", "cpu"]} if gr_settings else None,
            section=section,
        ).info(
            "Where to load the embedder + reranker. 'auto' picks GPU when "
            "CUDA is available, else CPU. 'cpu' saves ~2 GB VRAM for image "
            "generation but adds ~3–5 s per Anima click (CPU encoding is "
            "noticeably slower than GPU for bge-m3). Takes effect on next "
            "load — disable/re-enable the extension or restart Forge."
        ),
    )
