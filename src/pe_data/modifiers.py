"""Modifier loading + dropdown-data building.

Modifiers are categorized YAML files under `modifiers/` (one file per
dropdown). Each file has an `_label` field that names the dropdown +
a series of categories, each holding modifier entries.

Each modifier entry is either:
  - str: legacy format, treated as keywords; behavioral synthesized
  - dict with `behavioral`, `keywords`, and optional `source` (DB-pre-pick)
    and `target_slot` (post-fill safety net for category coverage)

The runtime resolution layer (`_collect_modifiers` in
scripts/prompt_enhancer.py) consumes the loader output and applies
the source / target_slot mechanisms — it stays in prompt_enhancer.py
because it's tangled with anima_tagger (DB lookups, retrieval). This
module only handles the data layer.

Public API:
  load_all(modifiers_dir, local_dirs, skip_stems=...) →
      (all_modifiers, dropdown_order, dropdown_choices) tuple.

      all_modifiers     dict: name → normalized entry dict
      dropdown_order    list[str]: dropdown labels in display order
                        (only labels with at least one valid choice)
      dropdown_choices  dict: label → [choice_str_with_separator_or_badge]

  scan_files(directory, skip_stems=...) → low-level scanner
  merge_dicts(base, override) → merge two dropdown-keyed dicts
  normalize(entry) → normalized entry dict, or None to drop
  build_dropdown_data(categories_dict) → (flat_lookup, choices_list)
  strip_badges(name) → strip trailing ◆/◇ chars from a UI label

Display badges (added by build_dropdown_data, stripped by strip_badges):
  ◆  source: pre-picked from DB, LLM renders chosen value
  ◇  target_slot: post-fills a category tag if LLM dropped it
  ◆◇ both: strongest guarantee
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._util import load_yaml_or_json


# Display badges
BADGE_SOURCE = "◆"        # ◆ filled diamond: source: pre-pick
BADGE_TARGET_SLOT = "◇"   # ◇ hollow diamond: target_slot: post-fill

# File stems to skip when scanning a directory for modifier files.
# Local-overrides dirs commonly mix bases / prompts / modifier YAMLs;
# this prevents bases/prompts files from being misclassified as
# modifier dropdowns.
DEFAULT_SKIP_STEMS: Tuple[str, ...] = ("_bases", "_prompts")


# ── Scanning + merging (file layer) ─────────────────────────────────────


def scan_files(
    directory: str,
    skip_stems: Iterable[str] = DEFAULT_SKIP_STEMS,
) -> Dict[str, Dict[str, Any]]:
    """Scan a directory for modifier YAML/JSON files.

    Returns a dict: dropdown_label -> {category_name: {modifier_name: entry}}.

    Files whose stem is in skip_stems are ignored. Hidden files (starting
    with ".") are also ignored. Files without an _label field at the
    top level are skipped with a warning printed (so the user knows to
    fix their YAML rather than silently losing modifiers).
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not directory or not os.path.isdir(directory):
        return result
    skip_set = set(skip_stems)
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem in skip_set:
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        path = os.path.join(directory, name)
        data = load_yaml_or_json(path)
        if not data:
            continue
        label = data.pop("_label", None)
        if not label:
            print(f"[pe_data.modifiers] WARNING: Skipping {path}: missing "
                  f"'_label' field. Add '_label: Your Label' to the YAML.")
            continue
        result[label] = data
    return result


def merge_dicts(
    base: Dict[str, Dict[str, Any]],
    override: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge two dropdown-level dicts. Same dropdown name → categories
    are merged; same category name → modifier entries from override
    win over base."""
    merged: Dict[str, Dict[str, Any]] = {}
    for label, categories in base.items():
        merged[label] = {}
        for cat, items in categories.items():
            if isinstance(items, dict):
                merged[label][cat] = dict(items)
    for label, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if label not in merged:
            merged[label] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[label]:
                merged[label][cat] = {}
            merged[label][cat].update(items)
    return merged


# ── Normalization (per-entry layer) ─────────────────────────────────────


def normalize(entry: Any) -> Optional[Dict[str, Any]]:
    """Normalize a modifier entry to dict form with keys:

      behavioral: prose instruction for Prose/Hybrid modes
      keywords:   comma-separated keywords for Tags mode + direct-paste
      source:     (optional) DB-pre-pick config
      target_slot:(optional) post-fill category for safety-net coverage

    Returns None when the entry is unusable (not str/dict, or both
    behavioral and keywords are empty without a source).

    Entries with a `source:` block are kept even with empty
    behavioral+keywords because the runtime layer fills them in
    from the picked DB tag.
    """
    if isinstance(entry, str):
        kw = entry.strip()
        return {
            "behavioral": (
                "Apply this style to the scene — describe the qualities "
                f"through prose, do not list them as keywords: {kw}."
            ),
            "keywords": kw,
        }
    if not isinstance(entry, dict):
        return None
    norm: Dict[str, Any] = {
        "behavioral": (entry.get("behavioral") or "").strip(),
        "keywords": (entry.get("keywords") or "").strip(),
    }
    if isinstance(entry.get("target_slot"), str) and entry["target_slot"].strip():
        norm["target_slot"] = entry["target_slot"].strip()
    if isinstance(entry.get("source"), dict):
        norm["source"] = entry["source"]
    if not norm["behavioral"] and norm["keywords"]:
        norm["behavioral"] = (
            "Apply this style to the scene — describe the qualities "
            f"through prose, do not list them as keywords: {norm['keywords']}."
        )
    if norm.get("source"):
        return norm
    if not norm["behavioral"] and not norm["keywords"]:
        return None
    return norm


# ── Dropdown shaping (presentation layer) ───────────────────────────────


def build_dropdown_data(
    categories_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the flat lookup + Gradio-ready choice list for ONE dropdown.

    Returns (flat, choices):
      flat     name → normalized entry dict (callers select behavioral
               vs keywords based on mode)
      choices  list of choice strings as they appear in the Gradio
               dropdown — category separators ("───── Setting ─────")
               interleaved with modifier names that carry mechanism
               badges (◆/◇/◆◇) when applicable. Display labels carry
               badges; the flat lookup stays keyed on raw YAML names.
    """
    flat: Dict[str, Any] = {}
    choices: List[str] = []
    for cat_name, items in categories_dict.items():
        if not isinstance(items, dict):
            continue
        separator = (
            "───── "
            f"{cat_name.title()} "
            "─────"
        )
        choices.append(separator)
        for name, entry in items.items():
            norm = normalize(entry)
            if norm is None:
                continue
            flat[name] = norm
            badge = ""
            if norm.get("source"):
                badge += BADGE_SOURCE
            if norm.get("target_slot"):
                badge += BADGE_TARGET_SLOT
            choices.append(f"{name} {badge}" if badge else name)
    return flat, choices


def strip_badges(name: str) -> str:
    """Drop any trailing ◆/◇ + whitespace that the UI appended to the
    display label. Returns the canonical YAML-side modifier name for
    lookup. Idempotent on names that have no badge."""
    if not isinstance(name, str):
        return name
    return name.rstrip(f" {BADGE_SOURCE}{BADGE_TARGET_SLOT}")


# ── Top-level orchestrator ──────────────────────────────────────────────


def load_all(
    modifiers_dir: str,
    local_dirs: Iterable[str] = (),
    skip_stems: Iterable[str] = DEFAULT_SKIP_STEMS,
) -> Tuple[Dict[str, Any], List[str], Dict[str, List[str]]]:
    """Top-level entry point used by the extension's _reload_all.

    Scans the extension's modifiers/ dir + each local-overrides dir,
    merges them, normalizes entries, and builds Gradio-ready dropdown
    data. Returns:

      all_modifiers    flat dict: modifier_name → normalized entry
                       (used for runtime lookup in _collect_modifiers)
      dropdown_order   list of dropdown labels in display order
                       (alphabetical; only labels with valid choices)
      dropdown_choices dict: label → list of choice strings
                       (for the Gradio Dropdown's choices=)
    """
    all_mods = scan_files(modifiers_dir, skip_stems=skip_stems)
    for local_dir in local_dirs:
        local_mods = scan_files(local_dir, skip_stems=skip_stems)
        all_mods = merge_dicts(all_mods, local_mods)

    all_modifiers: Dict[str, Any] = {}
    dropdown_order: List[str] = []
    dropdown_choices: Dict[str, List[str]] = {}
    for label in sorted(all_mods.keys()):
        flat, choices = build_dropdown_data(all_mods[label])
        if choices:
            dropdown_order.append(label)
            dropdown_choices[label] = choices
            all_modifiers.update(flat)
    return all_modifiers, dropdown_order, dropdown_choices
