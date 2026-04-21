"""High-level entry point: LLM-drafted tokens → final Anima tag list.

Takes the raw tag draft the LLM produced (Fuzzy-mode output, possibly
containing hallucinations like 'animedia' or phrase-shaped padding
like 'detailed_background'), runs each token through the embedding
validator against the Danbooru tag DB, then applies the Anima rule
layer (quality prefix, safety, @ prefix, character→series pairing).

Drop-in replacement for the rapidfuzz-based Fuzzy Strict path.
"""

from typing import List, Optional

from .cooccurrence import CoOccurrence
from .db import TagDB
from .rule_layer import apply_anima_rules
from .validator import TagValidator


def _split_tokens(draft: str) -> List[str]:
    """Split an LLM tag-draft string into individual tokens."""
    tokens: List[str] = []
    for t in draft.replace("\n", ",").split(","):
        t = t.strip()
        if t:
            tokens.append(t)
    return tokens


class AnimaTagger:
    """Validate an LLM tag draft and emit a compliant Anima tag list."""

    def __init__(self,
                 validator: TagValidator,
                 db: TagDB,
                 cooccurrence: Optional[CoOccurrence] = None):
        self.validator = validator
        self.db = db
        self.cooccurrence = cooccurrence

    def tag_from_draft(self,
                       draft: str | List[str],
                       safety: str = "safe",
                       use_underscores: bool = False,
                       max_tags: int = 30) -> List[str]:
        """Validate + transform LLM draft into final Anima tag list.

        `draft` may be the raw LLM string (comma-separated tags) or a
        pre-split list of tokens. Returns the final ordered tag list.
        """
        tokens = _split_tokens(draft) if isinstance(draft, str) else list(draft)

        results = self.validator.validate(tokens)

        # Fetch full DB records for each canonical hit so the rule
        # layer has category info for bucketing. Whitelist tokens
        # (Anima-convention like masterpiece/score_7/safe) aren't in
        # the DB — synthesize a minimal meta record so the rule layer
        # can still route them.
        from . import config
        records: list[dict] = []
        seen: set[str] = set()
        for r in results:
            if not r.canonical or r.canonical in seen:
                continue
            seen.add(r.canonical)
            if r.match_type == "whitelist":
                records.append({
                    "name": r.canonical,
                    "category": config.CAT_META,
                    "post_count": 0,
                    "aliases": "",
                    "wiki": "",
                })
                continue
            rec = self.db.get_by_name(r.canonical)
            if rec:
                records.append(rec)

        return apply_anima_rules(
            records,
            safety=safety,
            cooccurrence=self.cooccurrence,
            use_underscores=use_underscores,
            max_tags=max_tags,
        )
