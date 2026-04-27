"""Style + inline-wildcard string builders.

Pure helpers for assembling the user-message extras the LLM sees:
  - the "Apply these styles: ..." directive built from selected
    modifiers
  - the inline-wildcard directive when the source contains {name?}
    placeholders

No state. No deps on extension globals — callers thread the
modifiers list / prompts dict / mode through.

Public API:
  build_style_string(mod_list, mode="prose") → str
  has_inline_wildcards(text) → bool
  build_inline_wildcard_text(source, prompts) → str
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Tuple


# Compile once — used on every Prose / Hybrid / Tags call.
_INLINE_WILDCARD_RE = re.compile(r"\{[^}]+\?\}")


def has_inline_wildcards(text: str) -> bool:
    """Does `text` contain any `{name?}` inline-wildcard markers?

    The previous selected-wildcards system was folded into the
    modifier dropdowns (🎲 Random X entries). This helper only flags
    the inline `{name?}` syntax that may still appear in the source
    prompt itself.
    """
    if not text:
        return False
    return bool(_INLINE_WILDCARD_RE.search(text))


def build_inline_wildcard_text(
    source: str,
    prompts: Mapping[str, Any],
) -> str:
    """Return the inline-wildcard directive when `source` contains a
    `{name?}` placeholder, otherwise empty string.

    `prompts` is the loaded prompts.yaml dict (`pe_data.prompts.load(...)`
    output). Reads the `inline_wildcard` key. Returns "" if the source
    has no placeholders or the directive isn't configured.
    """
    if has_inline_wildcards(source):
        return prompts.get("inline_wildcard", "") or ""
    return ""


def build_style_string(
    mod_list: Iterable[Tuple[str, Mapping[str, Any]]],
    mode: str = "prose",
) -> str:
    """Build the style block injected into the user message.

    mod_list is a list of (name, normalized_entry) tuples from the
    runtime modifier resolver.

    mode:
      "prose" — Prose / Hybrid. Uses the modifier's `behavioral` field.
                Emits a single comma-joined directive. The active base
                prompt governs HOW the styles get applied (voice,
                structure); we only name them here.
      "tags"  — Tags mode. Uses the modifier's `keywords` field.
                Classic "Apply these styles: kw1, kw2" keyword-echo
                directive. Dice modifiers (empty keywords) are dropped
                in this mode because their behavioral instructions
                would leak words like "surprising"/"location"/"detailed"
                into the LLM's tag output.
    """
    mod_list = list(mod_list)
    if not mod_list:
        return ""

    if mode == "tags":
        parts = []
        for name, entry in mod_list:
            # Strip the 🎲 emoji marker from random-entry names.
            clean_name = name.replace("\U0001F3B2", "").strip()
            kw = entry.get("keywords") or ""
            if not kw:
                # Dice entry; would leak instruction words into Tags
                # mode output (see docstring).
                continue
            if clean_name.lower() not in kw.lower():
                kw = f"{clean_name.lower()}, {kw}"
            parts.append(kw)
        return f"Apply these styles: {', '.join(parts)}." if parts else ""

    # Prose / Hybrid mode — concatenate behavioral directives.
    behaviorals = []
    for name, entry in mod_list:
        text = (entry.get("behavioral") or "").strip()
        if text:
            text = text.rstrip(".!?: ")    # trim so commas join cleanly
            if text:
                behaviorals.append(text)
    if not behaviorals:
        return ""
    return f"Apply these styles to the scene: {', '.join(behaviorals)}."
