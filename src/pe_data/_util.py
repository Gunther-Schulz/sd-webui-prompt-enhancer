"""Shared utilities for the pe_data loaders.

load_yaml_or_json    — generic file loader that handles either format,
                       returns {} on missing/malformed file (with a
                       printed error so the user sees what failed).
get_local_dirs       — resolve user-supplied local-overrides paths
                       (UI textbox + PROMPT_ENHANCER_LOCAL env var).
                       Returns valid directories only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_yaml_or_json(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON file and return its top-level mapping.

    Returns {} on:
      - file not found (silent — caller decides if missing is an error)
      - parse error (prints to stderr-equivalent so the user sees the
        problem; loader doesn't itself raise)
      - top level isn't a mapping (prints + returns {}).

    Per CLAUDE.md "no silent fallbacks" we do print on error rather
    than swallow — but data-load errors are recoverable (extension
    keeps running with empty config), so we return {} rather than
    raise.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        print(f"[pe_data] ERROR: {path} must be a YAML/JSON mapping (dict), "
              f"got {type(data).__name__}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[pe_data] ERROR: Failed to load {path}: {e}")
    return {}


def get_local_dirs(ui_path: str = "", env_var: str = "PROMPT_ENHANCER_LOCAL") -> List[str]:
    """Resolve local-overrides directories.

    Comma-separated paths come from either the UI textbox (ui_path
    arg) or the PROMPT_ENHANCER_LOCAL env var. UI takes precedence.
    Returns valid existing directories only.
    """
    raw = (ui_path or "").strip() or os.environ.get(env_var, "")
    if not raw:
        return []
    dirs = []
    for p in raw.split(","):
        p = p.strip()
        if p and os.path.isdir(p):
            dirs.append(p)
    return dirs
