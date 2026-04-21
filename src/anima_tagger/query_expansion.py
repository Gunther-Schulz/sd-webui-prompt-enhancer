"""LLM-driven query expansion for the shortlist retrieval step.

Problem: user source prompts like "a girl reading" are too sparse for
semantic retrieval. The embedder has too little signal to distinguish
thematically-fitting artists/characters from name-collision matches
(`ireading`, `tf cafe`, etc.).

Fix: a cheap LLM pre-pass expands the source into ~15 Danbooru-style
tag concepts that capture subject / setting / mood / activity. This
expanded query embeds more densely and retrieves thematically-aligned
entities.

Usage: caller provides an `llm_call_fn(system_prompt, user_prompt) -> str`
and hands it + the source to `expand_query`. The result is a
comma-separated tag concept list to pass to `build_shortlist`.
"""

from typing import Callable, Optional


EXPANSION_SYSTEM_PROMPT = """You convert a brief scene description into a flat \
comma-separated list of 10-15 Danbooru-style tag concepts, used for retrieval.

Rules:
- Output ONLY the comma-separated list. No prose, no labels, no preamble.
- All-lowercase. Use spaces in multi-word concepts (not underscores).
- Include concepts covering: subject count (1girl / 1boy / 1other),
  clothing hints, setting / location, mood, lighting, activity /
  pose, composition style.
- Do NOT name specific real-world people, anime characters, or series.
  Those come from the retriever — your job is just concept expansion.
- Do NOT include image-format tokens (4k, 1920x1080) or invented words.
- Be generous: aim for 10-15 concepts.

Example input: "a girl reading in a cafe"
Example output: 1girl, solo, reading, holding book, cafe, indoor, \
sitting, window, warm lighting, cozy, peaceful, slice of life, casual clothes"""


def expand_query(source: str,
                 llm_call: Callable[[str, str], str],
                 modifier_keywords: Optional[str] = None) -> str:
    """Return a tag-concept expansion of `source` (comma-separated).

    Falls back to the raw source on any LLM error — expansion is an
    optimization, not a correctness requirement.
    """
    if not source or not source.strip():
        return source or ""
    user_msg = source.strip()
    if modifier_keywords:
        user_msg = f"{user_msg}\n\nActive style directives: {modifier_keywords}"
    try:
        out = llm_call(EXPANSION_SYSTEM_PROMPT, user_msg) or ""
        # Keep only the first line that looks like a tag list; strip
        # any stray lead-in the LLM might add.
        for line in out.strip().splitlines():
            line = line.strip().strip(".").strip()
            if "," in line and not line.lower().startswith(("here", "output", "example", "tags:")):
                return line
        # Nothing looked like a tag list — return the cleaned raw output
        return out.strip().replace("\n", ", ")
    except Exception:
        return source
