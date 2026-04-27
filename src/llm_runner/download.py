"""GGUF model download helper — wraps llama_cpp.Llama.from_pretrained.

The default flow uses HuggingFace's cache (`~/.cache/huggingface/hub/`).
Users can override via the `pe_llm_model_path` setting to point at a
local GGUF anywhere.

This module exists to:
  1. Add a visible progress callback (default HF download is silent).
  2. Centralize the "find the actual GGUF path" logic so the rest of
     the runner doesn't need to know the difference between HF and
     local paths.
  3. Fail loud with actionable error messages when something breaks
     (missing file, network error, repo not found).

Per CLAUDE.md "no silent fallbacks" — if the model isn't reachable,
we raise ModelLoadError with the specific reason, NOT silently fall
back to a smaller default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .config import RunnerConfig
from .errors import ModelLoadError


def resolve_model_path(config: RunnerConfig) -> str:
    """Return the absolute path to the GGUF for this config.

    If config.model_path is set, returns it (after verifying it exists).
    Otherwise downloads from HF if needed, returns the cached path.

    Raises ModelLoadError on any failure.
    """
    if config.has_local_path:
        path = Path(config.model_path)
        if not path.is_file():
            raise ModelLoadError(
                f"Custom GGUF path does not exist: {path}. "
                f"Either fix the 'pe_llm_model_path' setting or clear it "
                f"to fall back to the HF auto-download.",
                model_ref=str(path),
            )
        return str(path.resolve())

    # HF download path
    return _download_from_hf(config.repo_id, config.filename_pattern)


def _download_from_hf(repo_id: str, filename_pattern: str) -> str:
    """Download (or find in cache) a GGUF matching the filename pattern
    from the given HF repo. Returns the local cache path.

    Uses huggingface_hub directly so we get the cache logic without
    needing to instantiate Llama (which would load the model into VRAM
    just to look up the path).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise ModelLoadError(
            "huggingface_hub is required for GGUF auto-download. "
            "Install with: pip install 'huggingface_hub>=0.24'",
            model_ref=repo_id,
        ) from e

    # Resolve the filename pattern (e.g. "*Q4_K_M.gguf") to a concrete
    # filename by listing the repo and matching.
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        raise ModelLoadError(
            f"Could not list files in HF repo {repo_id!r}: {type(e).__name__}: {e}\n"
            f"Check: (a) repo id is correct, (b) network reaches "
            f"huggingface.co, (c) the repo exists and is public.",
            model_ref=repo_id,
        ) from e

    import fnmatch
    matches = [f for f in files if fnmatch.fnmatch(f, filename_pattern)]
    if not matches:
        # Fail-loud with discoverability hint
        gguf_files = [f for f in files if f.endswith(".gguf")]
        raise ModelLoadError(
            f"No GGUF in {repo_id!r} matched pattern {filename_pattern!r}.\n"
            f"Available GGUFs in this repo:\n  "
            + ("\n  ".join(gguf_files[:20]) if gguf_files
               else "(none — not a GGUF repo?)"),
            model_ref=f"{repo_id}:{filename_pattern}",
        )

    # If multiple match (e.g. 'Q4_K_M' could match an .imatrix variant
    # in some repos), pick the smallest filename — usually the canonical
    # file. Imatrix-quant files have longer names.
    chosen = sorted(matches, key=len)[0]

    print(f"[llm_runner] Downloading from HF: {repo_id} / {chosen}")
    print(f"[llm_runner] (cached at ~/.cache/huggingface/hub/ — first "
          f"download is large, subsequent runs are instant)")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=chosen,
            # No custom progress callback — huggingface_hub's built-in
            # tqdm output goes to stderr and is visible in the Forge
            # console. Wrapping with a custom callback is possible but
            # adds complexity for marginal UX win.
        )
    except Exception as e:
        raise ModelLoadError(
            f"HF download failed for {repo_id} / {chosen}: "
            f"{type(e).__name__}: {e}\n"
            f"Check: (a) network reaches huggingface.co, (b) sufficient "
            f"disk space (5-10 GB), (c) huggingface_hub package is "
            f"recent (>=0.24).",
            model_ref=f"{repo_id}:{chosen}",
        ) from e

    print(f"[llm_runner] Model ready: {path}")
    return path
