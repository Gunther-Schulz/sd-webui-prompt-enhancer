"""Pure text-processing helpers used by the prompt-enhancer extension.

These functions have no module state and no dependencies on the rest
of the extension — they're pulled out of scripts/prompt_enhancer.py
so the main file shrinks and so future tests can cover them in
isolation.

Functions exported:
  clean_output           — strip stray Markdown emphasis (*** **) from
                           prose output. Optionally strip underscores
                           too (used when the output goes into a
                           prose-style prompt vs. tag-style).
  clean_tag              — strip LLM meta-annotations from a single
                           draft tag (square brackets, "Illustrator: X",
                           trailing " style", hyphens). Preserves
                           danbooru disambiguation suffixes like
                           _(style) and weight syntax (tag:1.2).
  split_concatenated_tag — heuristic split of camelCase or run-on
                           tag strings ("moonlitcatacombs" →
                           "moonlit_catacombs"). Conservative — only
                           applies to long unsplit tokens.
  split_positive_negative — separate POSITIVE: / NEGATIVE: blocks
                           from a single LLM output blob.
"""

from __future__ import annotations

import re


def clean_output(text: str, strip_underscores: bool = True) -> str:
    """Strip stray Markdown emphasis. With strip_underscores=True
    (default), also remove `_word_` / `__word__` / `___word___`."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    if strip_underscores:
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    return text.strip()


def clean_tag(tag: str) -> str:
    """Strip LLM meta-annotations from a single tag.

    Handles patterns like [Illustrator: X], [style: X], (artist: X),
    [artist:X] style, "setting: garden", etc. Preserves danbooru
    disambiguation suffixes like _(style) and weight syntax like
    (tag:1.2).
    """
    tag = tag.strip()
    # Strip all square brackets (never valid in danbooru tags)
    tag = tag.replace("[", "").replace("]", "")
    # Strip paren-wrapped meta: (illustrator: X) -> X, but keep _(suffix) intact
    # Only strip if tag starts with ( (not a suffix like artist_(style))
    if tag.startswith("(") and not tag.startswith("_("):
        tag = tag.lstrip("(").rstrip(")")
    # Strip "key: value" meta-annotation prefixes generically.
    # Matches "word: content" or "word word: content" where the key part
    # is 2+ letters and doesn't look like a danbooru tag (rating:, score_).
    meta_match = re.match(
        r"^(?!rating|score)([a-zA-Z][a-zA-Z_ ]{1,30})\s*:\s*(.+)$", tag
    )
    if meta_match:
        key = meta_match.group(1).strip().lower()
        # Only strip if the key looks like a meta-annotation, not a valid tag prefix
        if key not in ("1", "2", "3"):  # don't strip numeric prefixes
            tag = meta_match.group(2)
    # Strip trailing " style" when it follows a name
    tag = re.sub(r"\s+style$", "", tag, flags=re.IGNORECASE)
    # Convert hyphens to underscores (hyphens are never valid in danbooru tags)
    if "-" in tag and "_" not in tag and " " not in tag:
        tag = tag.replace("-", "_")
    return tag.strip()


def split_concatenated_tag(tag: str) -> str:
    """Split run-on words like 'moonlitcatacombs' into 'moonlit_catacombs'.

    Conservative — only applies to long (>15 char) tokens with no
    existing separators. Uses camelCase boundaries; falls back to
    no-op when no clean boundary exists (downstream validation will
    catch the unsplit case).
    """
    if "_" in tag or " " in tag or len(tag) <= 15:
        return tag
    # Skip known valid patterns (rating:, score_, source_)
    if ":" in tag:
        return tag
    # Look for lowercase->uppercase transitions (camelCase)
    result = re.sub(r"([a-z])([A-Z])", r"\1_\2", tag).lower()
    if result != tag.lower():
        return result
    # All lowercase concatenated — can't reliably split without a dictionary
    return tag


def split_positive_negative(text: str) -> tuple[str, str]:
    """Split LLM output at POSITIVE:/NEGATIVE: markers.

    Returns (positive, negative). If no NEGATIVE marker is found,
    returns (text, ""). If only NEGATIVE: appears (no explicit
    POSITIVE: marker), everything before NEGATIVE: is treated as
    positive.
    """
    pos_match = re.search(r"(?i)^POSITIVE:\s*\n?", text, re.MULTILINE)
    neg_match = re.search(r"(?i)^NEGATIVE:\s*\n?", text, re.MULTILINE)
    if not neg_match:
        return text.strip(), ""
    if pos_match:
        positive = text[pos_match.end():neg_match.start()].strip()
    else:
        positive = text[:neg_match.start()].strip()
    negative = text[neg_match.end():].strip()
    return positive, negative
