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


def _extract_exact_name_hits(source: str, db, categories: List[int]) -> List[dict]:
    """Find tokens in the source that exactly match a DB tag name.

    Scans contiguous word-groups up to 4 words long. E.g. "hatsune miku"
    matches the tag `hatsune_miku`. Longest match wins per starting
    position so we don't split `"hatsune miku"` into `hatsune` + `miku`.

    Filters to the requested categories so exact-name pinning respects
    the category bucket being filled.
    """
    if not source:
        return []
    words = [w.strip().lower() for w in source.split() if w.strip()]
    hits: dict[int, dict] = {}   # tag_id → rec
    i = 0
    while i < len(words):
        found = None
        # Try longest first
        for span in range(min(4, len(words) - i), 0, -1):
            candidate = "_".join(words[i:i + span])
            candidate = candidate.replace("-", "_")
            rec = db.get_by_name(candidate)
            if rec and rec["category"] in categories:
                found = rec
                i += span
                break
        if found:
            hits[found["id"]] = found
        else:
            i += 1
    return list(hits.values())


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

    Pinned exact-name hits: tokens in the source that match DB tag
    names for the target category are forcibly injected at the top of
    the shortlist, regardless of retrieval scores. This guarantees that
    explicit entity mentions ("hatsune miku in maid uniform") surface
    the tag even if semantic retrieval would have missed it.
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
        # Retrieval candidates (leaves room for pinned additions at top)
        cands = retriever.retrieve(
            query,
            retrieve_k=config.DEFAULT_RETRIEVE_K,
            final_k=per_category_k,
            category=cat,
            min_post_count=min_post_count,
        )
        names = [c.name.replace("_", " ") for c in cands]
        # Pin exact-name source matches at the top
        pinned = _extract_exact_name_hits(source_prompt, retriever.db, [cat])
        pinned_names = [r["name"].replace("_", " ") for r in pinned]
        # Merge (pinned first, dedup, cap)
        seen: set[str] = set()
        out: List[str] = []
        for n in pinned_names + names:
            if n not in seen:
                seen.add(n)
                out.append(n)
            if len(out) >= per_category_k:
                break
        return out

    return Shortlist(
        artists=_names(expanded_query, config.CAT_ARTIST),
        characters=_names(raw_query, config.CAT_CHARACTER),
        series=_names(raw_query, config.CAT_COPYRIGHT),
    )
