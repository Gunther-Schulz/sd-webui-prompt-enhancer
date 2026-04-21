"""RAG shortlist: preselect real artists/characters/series for the prose pass.

Used to prevent the prose LLM from hallucinating an artist name like
'@takashi_murowo'. We retrieve a compact list of real names that
semantically fit the user's source prompt + active modifiers, and
inject it into the prose system prompt as "if referencing a name, use
only these".
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

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
                    min_post_count: int = 50,
                    query_expander: Optional[Callable[[str, Optional[str]], str]] = None) -> Shortlist:
    """Query the retriever per category and assemble the shortlist.

    Per-category query strategy:
      - Artists: use the LLM-expanded tag-concept query when available.
        Artist tags have ~0 wiki coverage in the DB, so literal-name
        overlap dominates without expansion; expansion gives the
        embedder thematic signal (mood, lighting, activity) to match.
      - Characters / Series: use the raw source. The user naming a
        character or franchise IS the literal-name signal we want;
        expansion dilutes it. Character tags also have ~0 wiki so
        the embedder can only see the name itself either way.
    """
    raw_query = (
        source_prompt if not modifier_keywords
        else f"{source_prompt}. {modifier_keywords}"
    )
    if query_expander is not None:
        try:
            expanded_query = query_expander(source_prompt, modifier_keywords)
        except Exception:
            expanded_query = raw_query
    else:
        expanded_query = raw_query

    def _names(query: str, cat: int) -> List[str]:
        cands = retriever.retrieve(
            query,
            retrieve_k=config.DEFAULT_RETRIEVE_K,
            final_k=per_category_k,
            category=cat,
            min_post_count=min_post_count,
        )
        return [c.name.replace("_", " ") for c in cands]

    return Shortlist(
        artists=_names(expanded_query, config.CAT_ARTIST),
        characters=_names(raw_query, config.CAT_CHARACTER),
        series=_names(raw_query, config.CAT_COPYRIGHT),
    )
