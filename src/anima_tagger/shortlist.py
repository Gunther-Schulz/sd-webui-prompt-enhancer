"""RAG shortlist: preselect real artists/characters/series for the prose pass.

Used to prevent the prose LLM from hallucinating an artist name like
'@takashi_murowo'. We retrieve a compact list of real names that
semantically fit the user's source prompt + active modifiers, and
inject it into the prose system prompt as "if referencing a name, use
only these".
"""

from dataclasses import dataclass
from typing import List, Optional

from . import config
from .retriever import Retriever


@dataclass
class Shortlist:
    artists: List[str]
    characters: List[str]
    series: List[str]

    def as_system_prompt_fragment(self) -> str:
        """Render as an injectable instruction block for the prose LLM."""
        lines: List[str] = []
        if self.artists or self.characters or self.series:
            lines.append(
                "Available references — if your prose names a specific real "
                "person or fictional entity, choose ONLY from the lists "
                "below. Do not invent names not listed here."
            )
        if self.artists:
            lines.append("Artists: " + ", ".join(self.artists))
        if self.characters:
            lines.append("Characters: " + ", ".join(self.characters))
        if self.series:
            lines.append("Series: " + ", ".join(self.series))
        return "\n".join(lines)


def build_shortlist(retriever: Retriever,
                    source_prompt: str,
                    modifier_keywords: Optional[str] = None,
                    per_category_k: int = 6,
                    min_post_count: int = 50) -> Shortlist:
    """Query the retriever once per category and assemble the shortlist.

    Using the same query text (source + modifier hints) per category
    lets the cross-encoder pick the best-matching entities without us
    having to craft category-specific prompts.
    """
    query = source_prompt if not modifier_keywords else f"{source_prompt}. {modifier_keywords}"

    def _names(cat: int) -> List[str]:
        cands = retriever.retrieve(
            query,
            retrieve_k=config.DEFAULT_RETRIEVE_K,
            final_k=per_category_k,
            category=cat,
            min_post_count=min_post_count,
        )
        return [c.name.replace("_", " ") for c in cands]

    return Shortlist(
        artists=_names(config.CAT_ARTIST),
        characters=_names(config.CAT_CHARACTER),
        series=_names(config.CAT_COPYRIGHT),
    )
