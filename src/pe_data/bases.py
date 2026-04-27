"""Base prompt loading + access.

A "base" is the system-prompt body for a generation mode (Default,
Detailed, Cinematic, Custom, etc.). Bases live in `bases.yaml` at
the extension root, with per-user overrides via `_bases.yaml` in
local-overrides directories.

Each base is either a plain string (legacy: body only) or a dict
with `body`, `target`, `description`, `label` (paren-text override).

Two special bases are loaded but hidden from the user-facing dropdown:
  _preamble   — prepended to every non-Custom base
  _format     — appended to every non-Custom base

Public API:
  load(ext_dir, local_dirs)      → returns the merged bases dict
  body(entry)                    → extract body from an entry (str|dict)
  meta(bases, name)              → metadata dict (target, description, …)
  names(bases, curated_order=…)  → ordered (label, value) pairs for
                                   the Base Gradio dropdown
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

from ._util import load_yaml_or_json


BASES_FILENAME = "_bases"   # local-override file stem

CURATED_ORDER: Tuple[str, ...] = (
    "Default", "Detailed", "Narrative", "Cinematic", "Creative",
)


# ── Loading ─────────────────────────────────────────────────────────────


def load(ext_dir: str, local_dirs: Iterable[str] = ()) -> Dict[str, Any]:
    """Load bases from `ext_dir/bases.{yaml,yml,json}` and merge with
    `_bases.{yaml,yml,json}` from each local override dir.

    Returns the merged dict. Local entries override main entries
    when keys collide (matches existing extension behavior).
    """
    bases: Dict[str, Any] = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(ext_dir, "bases" + ext)
        if os.path.isfile(path):
            bases = {
                k: v for k, v in load_yaml_or_json(path).items()
                if isinstance(v, (str, dict))
            }
            break
    bases.update(_load_local(local_dirs))
    return bases


def _load_local(local_dirs: Iterable[str]) -> Dict[str, Any]:
    """Load and merge `_bases.{yaml,yml,json}` from each local dir.
    Later dirs override earlier ones."""
    merged: Dict[str, Any] = {}
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, BASES_FILENAME + ext)
            if os.path.isfile(path):
                merged.update({
                    k: v for k, v in load_yaml_or_json(path).items()
                    if isinstance(v, (str, dict))
                })
    return merged


# ── Access ──────────────────────────────────────────────────────────────


def body(entry: Any) -> str:
    """Return the body string from a base entry. Entries may be either
    a plain string (legacy) or a dict with a `body` key."""
    if isinstance(entry, dict):
        return entry.get("body", "")
    return entry or ""


def meta(bases: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return the metadata dict (target / description / label / …) for
    a base, or {} when the entry is a plain-string body or missing."""
    entry = bases.get(name)
    if isinstance(entry, dict):
        return {k: v for k, v in entry.items() if k != "body"}
    return {}


def names(
    bases: Dict[str, Any],
    curated_order: Iterable[str] = CURATED_ORDER,
) -> List[Tuple[str, str]]:
    """Return ordered (label, value) tuples for the Base Gradio dropdown.

    Label text comes from each base's metadata: an explicit `label`
    string takes precedence over auto-derivation from the `target`
    list (first 3 entries joined). Curated bases appear first in the
    order given by curated_order; user-added bases follow in dict
    iteration order. Underscore-prefixed keys (_preamble, _format)
    are excluded from the dropdown. "Custom" is always last.
    """

    def _label(value: str) -> str:
        m = meta(bases, value)
        paren = m.get("label")
        if not paren:
            target = m.get("target", [])
            if target and isinstance(target, list):
                paren = ", ".join(str(t) for t in target[:3])
        return f"{value} ({paren})" if paren else value

    result: List[Tuple[str, str]] = []
    seen = set()
    for value in curated_order:
        if value in bases:
            result.append((_label(value), value))
            seen.add(value)
    for value in bases.keys():
        if value.startswith("_") or value in seen:
            continue
        if value == "Custom":
            continue
        result.append((_label(value), value))
        seen.add(value)
    if "Custom" in bases:
        m = meta(bases, "Custom")
        paren = m.get("label", "user-supplied system prompt")
        result.append((f"Custom ({paren})", "Custom"))
    return result
