"""High-level entry point: LLM-drafted tokens → final Anima tag list.

Takes the raw tag draft the LLM produced, runs each token through the
embedding validator (with optional shortlist context for category-aware
alias resolution + popularity gating), then applies the Anima rule
layer (quality prefix, safety, @ prefix, character→series pairing).

Drop-in replacement for the rapidfuzz-based Fuzzy Strict path.
"""

from typing import List, Optional

from .cooccurrence import CoOccurrence
from .db import TagDB
from .rule_layer import apply_anima_rules
from .shortlist import Shortlist
from .validator import TagValidator, ValidationContext


def _split_tokens(draft: str) -> List[str]:
    tokens: List[str] = []
    for t in draft.replace("\n", ",").split(","):
        t = t.strip()
        if t:
            tokens.append(t)
    return tokens


def _context_from_shortlist(sl: Optional[Shortlist], db: TagDB) -> Optional[ValidationContext]:
    """Build a ValidationContext from a shortlist for category-aware
    alias resolution and popularity gating. Returns None when sl is None
    or empty (validator then uses no-context defaults)."""
    if sl is None:
        return None
    names: set[str] = set()
    cats: set[int] = set()
    # Shortlist entries are in space-form ("hatsune miku"); DB uses
    # underscore form. Normalize both.
    for name_list in (sl.artists, sl.characters, sl.series):
        for n in name_list:
            canonical = n.replace(" ", "_").replace("-", "_")
            rec = db.get_by_name(canonical)
            if rec:
                names.add(rec["name"])
                cats.add(rec["category"])
    if not names and not cats:
        return None
    return ValidationContext(shortlist_names=names, shortlist_categories=cats)


class AnimaTagger:
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
                       max_tags: int = 30,
                       shortlist: Optional[Shortlist] = None,
                       compound_split: bool = True) -> List[str]:
        """Validate + transform LLM draft into final Anima tag list.

        `shortlist`, if provided, gives the validator category context
        to disambiguate aliases (e.g. `rococo` → character vs artist).
        `compound_split` (default True) lets multi-word drafts like
        'long silver hair' recover as {'long_hair', 'silver_hair'}
        instead of getting dropped when no direct tag matches.
        """
        tokens = _split_tokens(draft) if isinstance(draft, str) else list(draft)

        ctx = _context_from_shortlist(shortlist, self.db)
        if compound_split:
            results = self.validator.validate_with_compound_split(tokens, context=ctx)
        else:
            results = self.validator.validate(tokens, context=ctx)

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
