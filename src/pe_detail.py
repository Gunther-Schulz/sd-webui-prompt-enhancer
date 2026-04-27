"""Detail-level instruction builder.

The detail slider (0-10) maps to a verbal "how much elaboration"
descriptor that goes into the LLM system prompt. Pure data + small
helpers — no extension state, no deps on other modules.

Public API:
  PRESET_MAX_TOKENS   max-token hint per Forge preset (used by
                      get_word_target — kept for callers, though
                      build_instruction itself no longer injects
                      explicit word/tag count targets)
  TAG_COUNTS          rough tag-count target per detail level
                      (model-independent, kept for callers that
                      surface a UI hint)
  DETAIL_LABELS       detail (int 0-10) → human-readable phrase
                      ("moderate", "detailed, vivid", etc.)
  get_word_target(detail, preset)
                      → optional int word target (None for detail=0)
  build_instruction(detail, mode="enhance", preset="sd")
                      → optional system-prompt-fragment string

Historical versions injected "Aim for around N words" / "around N
tags" — those got removed because they made the LLM stingy and
suppressed real coverage. Detail now contributes a style-descriptor
only.
"""

from __future__ import annotations

from typing import Optional


# Max tokens per Forge preset (from backend/diffusion_engine/*.py)
PRESET_MAX_TOKENS = {
    "sd": 75, "xl": 75, "flux": 255, "klein": 255,
    "qwen": 512, "lumina": 512, "zit": 999, "wan": 512,
    "anima": 512, "ernie": 512,
}

# Tag counts per detail level (model-independent UI hint)
TAG_COUNTS = {
    0: None, 1: 8, 2: 12, 3: 18, 4: 25,
    5: 35, 6: 45, 7: 55, 8: 65, 9: 75, 10: 90,
}

# Verbal label per detail level. Goes into the LLM system prompt
# verbatim ("Write a {label} description.").
DETAIL_LABELS = {
    0: None,
    1: "very short, minimal",
    2: "short, concise",
    3: "brief but complete",
    4: "moderate",
    5: "moderately detailed",
    6: "detailed",
    7: "detailed, vivid",
    8: "highly detailed",
    9: "very detailed, comprehensive",
    10: "extensive, exhaustive",
}


def get_word_target(detail: int, preset: str = "sd") -> Optional[int]:
    """Calculate a word target based on detail level and Forge preset.

    Returns None when detail is 0 (auto / no constraint).
    """
    if detail == 0:
        return None
    max_tokens = PRESET_MAX_TOKENS.get(preset, 75)
    max_words = int(max_tokens * 0.75)              # ~0.75 words / token
    fraction = 0.1 + (detail / 10) * 0.9            # detail 1 → 20 %, detail 10 → 100 %
    return max(20, int(max_words * fraction))


def build_instruction(
    detail: int,
    mode: str = "enhance",
    preset: str = "sd",
) -> Optional[str]:
    """Build the detail-instruction system-prompt fragment.

    Returns a style descriptor only (no word/tag count targets — those
    were removed because they made the LLM stingy on real coverage).

    mode = "enhance" or "tags"; controls the verb in the returned phrase.
    Returns None at detail=0 so the caller can skip injection entirely.
    """
    if detail == 0:
        return None
    label = DETAIL_LABELS.get(detail, "moderate")
    if mode == "tags":
        return f"Generate a {label} set of tags. Cover every distinct concept."
    return f"Write a {label} description."
