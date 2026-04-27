"""Sampling presets and DRY-aware sampler chain construction.

The high-level llama-cpp-python API (`Llama.create_chat_completion`)
exposes top_k / top_p / min_p / temperature / repeat_penalty /
frequency_penalty / presence_penalty but NOT yet DRY (open issue
abetlen/llama-cpp-python#1813).

This module supports both paths:

  1. **High-level path** (config.use_low_level_dry == False)
     — uses create_chat_completion's built-in samplers. DRY is absent;
     repetition is handled by repeat_penalty + frequency_penalty +
     presence_penalty + min_p. Already much better than Ollama's
     silently-dropped penalties for Qwen 3.5.

  2. **Low-level path** (config.use_low_level_dry == True; default)
     — drops to llama_cpp's ctypes-level sampler API and builds a custom
     chain that includes DRY. The runner uses Llama.eval() + custom
     sampling instead of create_completion. More fragile (depends on
     llama_cpp binding stability), but real DRY.

Sampler order is load-bearing. Daniel Han (Unsloth) found that putting
repetition penalties first causes loops on quantized Qwen models. The
Qwen-tuned order is: top_k → top_p → min_p → temperature → DRY → penalties → dist.

DRY config (Qwen-tuned, lower than the universal default):
  - dry_multiplier 0.5 (general default is 0.8; lower for Qwen)
  - dry_base 1.75
  - dry_allowed_length 2 (raise to 3-4 for structured output)
  - dry_penalty_last_n -1 (scan full context)
  - dry_sequence_breakers ['\\n', ':', '"', '*'] (default for prose;
    add ';', '{', '}' for code)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SamplerConfig:
    """Sampling parameters. Defaults are Qwen 9B abliterated tuned.

    repeat_penalty=1.1 is intentional. The previous extension setting
    of 1.5 was both (a) silently dropped by Ollama for Qwen 3.5 and
    (b) actually counterproductive even when running — Daniel Han's
    finding from QwQ-32B work. Mild repeat_penalty paired with DRY
    (or with frequency/presence) gives better results than aggressive
    repeat_penalty alone.
    """

    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.5
    presence_penalty: float = 0.5
    # Penalty window (last N tokens to penalize). 64 matches llama.cpp
    # default; -1 = full context.
    repeat_last_n: int = 64

    # DRY (used by the low-level path; ignored by the high-level path
    # until llama-cpp-python issue #1813 lands)
    dry_multiplier: float = 0.5
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1
    dry_sequence_breakers: Tuple[str, ...] = field(
        default_factory=lambda: ("\n", ":", "\"", "*")
    )


SAMPLER_PRESETS = {
    "qwen_default": SamplerConfig(),
    # Future presets added per format / use-case. Examples:
    #   "qwen_strict_json" — higher dry_allowed_length for structured output
    #   "qwen_creative"    — higher temperature, looser top_p
}


def get_preset(name: str) -> SamplerConfig:
    """Resolve a preset name. Fail-loud on unknown name."""
    if name not in SAMPLER_PRESETS:
        raise ValueError(
            f"unknown sampler preset {name!r}; "
            f"available: {sorted(SAMPLER_PRESETS)}"
        )
    return SAMPLER_PRESETS[name]


# ── Low-level DRY sampler chain construction ─────────────────────────────
#
# This wraps the llama_cpp ctypes-level sampler API. It's only used when
# config.use_low_level_dry is True — otherwise the high-level
# create_chat_completion path runs.
#
# References:
#   - llama.cpp llama.h — llama_sampler_init_dry signature
#   - https://llama-dry-docs.site/view/docs — Qwen-tuned config
#   - llama-cpp-python issue #1813 — high-level API tracking
#
# This will gracefully fail with a clear ImportError-style message if
# the installed llama-cpp-python doesn't expose the DRY binding (older
# versions don't have it).


class DryNotAvailable(ImportError):
    """The installed llama-cpp-python doesn't expose llama_sampler_init_dry.

    Mitigation: upgrade llama-cpp-python (newer wheels track newer
    llama.cpp), or set pe_llm_use_low_level_dry=False to use the
    high-level samplers (no DRY but no extra binding required).
    """


def build_low_level_sampler_chain(llm, sampler: SamplerConfig, seed: int):
    """Build a Qwen-tuned sampler chain via llama_cpp's ctypes API.

    Order is load-bearing — see Daniel Han's note in the unsloth docs:
    top_k → top_p → min_p → temperature → DRY → penalties → dist.

    Args:
        llm: an instantiated llama_cpp.Llama (we pull vocab + n_ctx_train).
        sampler: SamplerConfig with all sampling params.
        seed: RNG seed for the dist sampler. -1 = random.

    Returns:
        An opaque pointer (ctypes) to the constructed sampler chain.
        The caller is responsible for freeing it via
        llama_cpp.llama_sampler_free(chain) when done — or letting it
        leak per-call (acceptable; chains are tiny).

    Raises:
        DryNotAvailable: if the llama_cpp module doesn't expose the
            DRY sampler binding (older llama-cpp-python).
    """
    from llama_cpp import llama_cpp

    # Validate the DRY binding exists before building anything else
    if not hasattr(llama_cpp, "llama_sampler_init_dry"):
        raise DryNotAvailable(
            "The installed llama-cpp-python version doesn't expose "
            "llama_sampler_init_dry. Either:\n"
            "  - Upgrade llama-cpp-python (recent wheels include DRY)\n"
            "  - Set pe_llm_use_low_level_dry=False in Forge settings "
            "to use the high-level samplers without DRY."
        )

    # Initialize empty chain
    chain_params = llama_cpp.llama_sampler_chain_default_params()
    chain = llama_cpp.llama_sampler_chain_init(chain_params)

    # Get vocab handle + n_ctx_train from the llama instance
    model_handle = llm.model
    vocab = llama_cpp.llama_model_get_vocab(model_handle)
    n_ctx_train = llama_cpp.llama_model_n_ctx_train(model_handle)

    # 1. top_k
    if sampler.top_k > 0:
        llama_cpp.llama_sampler_chain_add(
            chain, llama_cpp.llama_sampler_init_top_k(sampler.top_k)
        )

    # 2. top_p
    if 0.0 < sampler.top_p < 1.0:
        llama_cpp.llama_sampler_chain_add(
            chain, llama_cpp.llama_sampler_init_top_p(sampler.top_p, 1)
        )

    # 3. min_p
    if sampler.min_p > 0.0:
        llama_cpp.llama_sampler_chain_add(
            chain, llama_cpp.llama_sampler_init_min_p(sampler.min_p, 1)
        )

    # 4. temperature
    if sampler.temperature > 0.0:
        llama_cpp.llama_sampler_chain_add(
            chain, llama_cpp.llama_sampler_init_temp(sampler.temperature)
        )

    # 5. DRY — placed AFTER temperature per Daniel Han's finding
    breakers = sampler.dry_sequence_breakers
    breakers_array_type = ctypes.c_char_p * len(breakers)
    breakers_array = breakers_array_type(*(b.encode("utf-8") for b in breakers))
    llama_cpp.llama_sampler_chain_add(
        chain,
        llama_cpp.llama_sampler_init_dry(
            vocab,
            n_ctx_train,
            ctypes.c_float(sampler.dry_multiplier),
            ctypes.c_float(sampler.dry_base),
            ctypes.c_int32(sampler.dry_allowed_length),
            ctypes.c_int32(sampler.dry_penalty_last_n),
            breakers_array,
            ctypes.c_size_t(len(breakers)),
        ),
    )

    # 6. Repetition / frequency / presence penalties (mild, paired with DRY)
    if sampler.repeat_penalty != 1.0 or sampler.frequency_penalty != 0.0 or sampler.presence_penalty != 0.0:
        llama_cpp.llama_sampler_chain_add(
            chain,
            llama_cpp.llama_sampler_init_penalties(
                ctypes.c_int32(sampler.repeat_last_n),
                ctypes.c_float(sampler.repeat_penalty),
                ctypes.c_float(sampler.frequency_penalty),
                ctypes.c_float(sampler.presence_penalty),
            ),
        )

    # 7. Distribution (final token selection from the constrained logits)
    llama_cpp.llama_sampler_chain_add(
        chain, llama_cpp.llama_sampler_init_dist(seed)
    )

    return chain


def free_sampler_chain(chain):
    """Free a chain built by build_low_level_sampler_chain. Safe to skip
    if leaking is acceptable (chains are tiny), but cleaner."""
    try:
        from llama_cpp import llama_cpp
        if chain is not None:
            llama_cpp.llama_sampler_free(chain)
    except Exception:
        pass
