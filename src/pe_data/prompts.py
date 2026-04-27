"""Operational-prompt loading.

Loads `prompts.yaml` from the extension root (the file containing
modes' system prompts: remix_prose, summarize, picker, motion,
negative, etc.) and merges per-user overrides from
`_prompts.{yaml,yml,json}` in each local-overrides directory.

Keys with string values are right-stripped of surrounding whitespace
during load — convenient because YAML pipe-blocks ("|") preserve
leading/trailing newlines that would otherwise propagate into
system prompts.

Public API:
  load(ext_dir, local_dirs) → dict mapping prompt-name → text/value
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable

from ._util import load_yaml_or_json


PROMPTS_LOCAL_FILENAME = "_prompts"


def load(ext_dir: str, local_dirs: Iterable[str] = ()) -> Dict[str, Any]:
    """Load prompts from `ext_dir/prompts.{yaml,yml,json}` and merge
    local overrides from `_prompts.{yaml,yml,json}` in each local dir.

    Returns a dict with string values whitespace-stripped. Non-string
    values pass through unchanged so structured prompt configs (lists,
    dicts) keep working.
    """
    prompts: Dict[str, Any] = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(ext_dir, "prompts" + ext)
        if os.path.isfile(path):
            data = load_yaml_or_json(path) or {}
            prompts = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in data.items()
            }
            break

    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, PROMPTS_LOCAL_FILENAME + ext)
            if os.path.isfile(path):
                local_p = load_yaml_or_json(path) or {}
                for k, v in local_p.items():
                    if isinstance(v, str):
                        prompts[k] = v.strip()
                    else:
                        prompts[k] = v
    return prompts
