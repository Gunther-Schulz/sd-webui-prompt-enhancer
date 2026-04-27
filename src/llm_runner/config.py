"""Runner configuration — settings keys and defaults.

Settings live in shared.opts (Forge's persistent options dict). Default
model is the GGUF equivalent of what the extension shipped with under
Ollama; user can override via the "Custom GGUF path" setting if they
want to BYO model.

Compute target:
  - "gpu"    → all layers on GPU (n_gpu_layers = -1)
  - "cpu"    → all on CPU (n_gpu_layers = 0); slow but works without VRAM
  - "shared" → user-specified split (n_gpu_layers = N); useful for
               low-VRAM cards that can hold ~half the model

Lifecycle:
  - "keep_loaded"        → model stays in VRAM/RAM after each call (fast,
                          greedy with VRAM)
  - "unload_after_call"  → model is unloaded after every call (slow, but
                          frees VRAM for image generation)

Quant maps to the filename pattern HF download searches for. Q4_K_M is
the default — ~5.6 GB for a 9B, fits 8 GB VRAM with offload, good
quality/speed balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_REPO_ID = "mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF"
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_N_CTX = 4096
DEFAULT_COMPUTE = "gpu"
DEFAULT_LIFECYCLE = "keep_loaded"
DEFAULT_SAMPLER_PRESET = "qwen_default"
DEFAULT_USE_LOW_LEVEL_DRY = True   # User explicitly requested DRY from day 1.

# Available choices for UI dropdowns (Forge settings)
QUANT_CHOICES = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
COMPUTE_CHOICES = ("gpu", "cpu", "shared")
LIFECYCLE_CHOICES = ("keep_loaded", "unload_after_call")
N_CTX_CHOICES = (2048, 4096, 8192, 16384)


# ── Settings keys (persistent in shared.opts) ────────────────────────────

OPT_REPO_ID = "pe_llm_repo_id"
OPT_QUANT = "pe_llm_quant"
OPT_MODEL_PATH = "pe_llm_model_path"
OPT_N_CTX = "pe_llm_n_ctx"
OPT_COMPUTE = "pe_llm_compute"
OPT_N_GPU_LAYERS = "pe_llm_n_gpu_layers"
OPT_LIFECYCLE = "pe_llm_lifecycle"
OPT_SAMPLER_PRESET = "pe_llm_sampler_preset"
OPT_USE_LOW_LEVEL_DRY = "pe_llm_use_low_level_dry"


@dataclass(frozen=True)
class RunnerConfig:
    """Resolved configuration for one runner instance.

    Frozen so we can use it as a cache key — if any field changes, the
    runner must be rebuilt with the new config.
    """

    repo_id: str
    quant: str
    model_path: str           # absolute path; if non-empty, overrides repo_id
    n_ctx: int
    compute: str              # "gpu" | "cpu" | "shared"
    n_gpu_layers: int         # only used when compute == "shared"
    lifecycle: str            # "keep_loaded" | "unload_after_call"
    sampler_preset: str
    use_low_level_dry: bool

    @property
    def filename_pattern(self) -> str:
        """Glob pattern passed to llama_cpp.Llama.from_pretrained's
        filename arg. mradermacher quantizations follow the
        '<model>.<QUANT>.gguf' convention, so '*Q4_K_M.gguf' selects
        the right file from the repo."""
        return f"*{self.quant}.gguf"

    @property
    def effective_n_gpu_layers(self) -> int:
        """Maps the high-level 'compute' choice to llama-cpp-python's
        n_gpu_layers parameter."""
        if self.compute == "cpu":
            return 0
        if self.compute == "gpu":
            return -1   # offload everything
        # "shared" — caller-specified
        return max(0, self.n_gpu_layers)

    @property
    def has_local_path(self) -> bool:
        """True if the user provided a local GGUF path that exists.
        When set, repo_id is ignored."""
        return bool(self.model_path) and Path(self.model_path).is_file()

    @property
    def model_ref(self) -> str:
        """Human-readable identifier — local path if set, else repo+filename."""
        if self.has_local_path:
            return self.model_path
        return f"{self.repo_id}:{self.filename_pattern}"


def _opt(key: str, default):
    """Read from shared.opts with a default. Standalone path (no Forge)
    just returns default."""
    try:
        from modules import shared  # type: ignore
        return shared.opts.data.get(key, default)
    except Exception:
        return default


def load_config() -> RunnerConfig:
    """Read the current config from shared.opts. Re-reads every call so
    settings changes take effect on next runner reset.

    Standalone callers (experiments harness, tests) get the defaults
    when shared.opts isn't available.
    """
    return RunnerConfig(
        repo_id=str(_opt(OPT_REPO_ID, DEFAULT_REPO_ID)),
        quant=str(_opt(OPT_QUANT, DEFAULT_QUANT)),
        model_path=str(_opt(OPT_MODEL_PATH, "")),
        n_ctx=int(_opt(OPT_N_CTX, DEFAULT_N_CTX)),
        compute=str(_opt(OPT_COMPUTE, DEFAULT_COMPUTE)),
        n_gpu_layers=int(_opt(OPT_N_GPU_LAYERS, -1)),
        lifecycle=str(_opt(OPT_LIFECYCLE, DEFAULT_LIFECYCLE)),
        sampler_preset=str(_opt(OPT_SAMPLER_PRESET, DEFAULT_SAMPLER_PRESET)),
        use_low_level_dry=bool(_opt(OPT_USE_LOW_LEVEL_DRY, DEFAULT_USE_LOW_LEVEL_DRY)),
    )


def config_override(**kwargs) -> RunnerConfig:
    """Build a RunnerConfig from defaults overridden by kwargs. Used by
    the experiments harness to construct configs without needing Forge."""
    base = dict(
        repo_id=DEFAULT_REPO_ID,
        quant=DEFAULT_QUANT,
        model_path="",
        n_ctx=DEFAULT_N_CTX,
        compute=DEFAULT_COMPUTE,
        n_gpu_layers=-1,
        lifecycle=DEFAULT_LIFECYCLE,
        sampler_preset=DEFAULT_SAMPLER_PRESET,
        use_low_level_dry=DEFAULT_USE_LOW_LEVEL_DRY,
    )
    base.update(kwargs)
    return RunnerConfig(**base)
