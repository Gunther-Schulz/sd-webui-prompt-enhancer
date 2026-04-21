"""Embedding-based tag validator with context-aware disambiguation.

For each draft token the LLM emits:

  1. Whitelist pass — Anima-convention tokens (masterpiece, score_7,
     safe, …) aren't in the Danbooru DB but must be kept verbatim.
  2. Exact DB match → keep canonical. If the match is a low-popularity
     artist or copyright tag AND not in the provided shortlist context,
     require it to clear a popularity floor (avoids niche-name collisions
     like LLM output "animedia" hitting an obscure real tag).
  3. Alias match → pick best canonical. When multiple tags share the
     alias (e.g. `rococo` → both `toeri_(rococo)` artist and
     `rococo_(girl_cafe_gun)` character), prefer a candidate whose
     category is in the shortlist context; fall back to highest
     post_count.
  4. Semantic match via bge-m3 → substitute if similarity ≥ threshold
     AND matched tag has enough popularity.
  5. Otherwise drop.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

from . import config
from .db import TagDB
from .embedder import Embedder
from .index import VectorIndex


# Anima-specific convention tokens. Not in the Danbooru DB but recognized
# by the Anima model; validator should pass them through as-is instead
# of dropping as unknown.
ANIMA_WHITELIST: Set[str] = {
    # Quality (human score system)
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality",
    # Quality (PonyV7 score system)
    "score_9", "score_8", "score_7", "score_6", "score_5",
    "score_4", "score_3", "score_2", "score_1",
    # Safety tags
    "safe", "sensitive", "nsfw", "explicit",
    # Time-period tags
    "newest", "recent", "mid", "early", "old",
    # Other meta that Anima recognizes
    "absurdres", "highres",
}

# Categories that are "named entities" — niche name collisions are
# dangerous here. Without popularity gating, a user-typed common phrase
# can accidentally pin to an obscure entity:
#   "dark shadow" (darkness concept) → dark_shadow character (96 posts)
#   "rococo"      (art style)        → toeri_(rococo) artist (niche)
#   "animedia"    (generic)          → magazine tag
# For these categories we require either shortlist membership or a
# popularity floor on exact matches.
_ENTITY_CATEGORIES = {
    config.CAT_ARTIST,
    config.CAT_COPYRIGHT,
    config.CAT_CHARACTER,
}


@dataclass
class ValidationContext:
    """Hints passed from the shortlist / caller to improve disambiguation.

    shortlist_names: canonical tag names the retriever surfaced for this
        source prompt (across all categories).
    shortlist_categories: set of categories present in the shortlist
        (used to break ties when an alias resolves to multiple tags).
    """
    shortlist_names: Set[str]
    shortlist_categories: Set[int]


@dataclass
class ValidationResult:
    original: str
    canonical: Optional[str]
    confidence: float
    match_type: str   # "whitelist" | "exact" | "alias" | "shortlist_stem" | "semantic" | "drop"
    drop_reason: Optional[str] = None
    # Populated only when match_type == "drop". One of:
    #   "too_short"            — token shorter than min_len after normalization
    #   "exact_entity_gate"    — direct DB hit but artist/copyright with low post_count
    #   "alias_entity_gate"    — alias resolved but target has low post_count
    #   "semantic_below_thr"   — top cosine score under semantic_threshold
    #   "semantic_low_popularity" — semantic match but matched tag under semantic_min_post_count
    #   "semantic_entity_gate" — semantic match but entity popularity gate failed
    #   "no_candidate"         — FAISS returned no result (top_id < 0)


class TagValidator:
    def __init__(self,
                 db: TagDB,
                 index: VectorIndex,
                 embedder: Embedder,
                 whitelist: Optional[Set[str]] = None,
                 semantic_threshold: float = 0.80,
                 semantic_min_post_count: int = 100,
                 entity_min_post_count: int = 1000,
                 min_len: int = 3):
        self.db = db
        self.index = index
        self.embedder = embedder
        self.whitelist = whitelist if whitelist is not None else ANIMA_WHITELIST
        self.semantic_threshold = semantic_threshold
        self.semantic_min_post_count = semantic_min_post_count
        self.entity_min_post_count = entity_min_post_count
        self.min_len = min_len
        # Build reverse alias maps once. `alias_multi` gives every
        # (canonical, category, post_count) mapping for ambiguity
        # resolution; `alias_single` is the legacy first-canonical
        # fallback when no context is available.
        self._alias_multi = db.build_alias_lookup_multi()
        self._alias_single = db.build_alias_lookup()

    @staticmethod
    def _normalize(token: str) -> str:
        t = token.strip().lower()
        if t.startswith("@"):
            t = t[1:].strip()
        t = t.replace(" ", "_").replace("-", "_")
        while "__" in t:
            t = t.replace("__", "_")
        return t.strip("_")

    def _resolve_alias(self, norm: str, ctx: Optional[ValidationContext]) -> Optional[dict]:
        """Pick the best alias candidate given optional shortlist context.

        Preference order:
          1. Candidate whose canonical name is in shortlist_names.
          2. Candidate whose category is in shortlist_categories.
          3. Highest post_count (already sorted).
        """
        cands = self._alias_multi.get(norm)
        if not cands:
            return None
        if not ctx:
            return cands[0]
        # Priority 1: canonical name in shortlist
        for c in cands:
            if c["name"] in ctx.shortlist_names:
                return c
        # Priority 2: category in shortlist's categories
        if ctx.shortlist_categories:
            for c in cands:
                if c["category"] in ctx.shortlist_categories:
                    return c
        return cands[0]

    def _is_trusted_entity(self, rec: dict, norm: str,
                           ctx: Optional[ValidationContext]) -> bool:
        """Popularity gate for niche-name collisions in artist/copyright.

        Exact match is trusted when:
          - category is not artist/copyright (non-entity, no collision risk), OR
          - popularity ≥ entity_min_post_count, OR
          - the canonical name is in the shortlist (user context implies it).
        """
        if rec.get("category") not in _ENTITY_CATEGORIES:
            return True
        if rec.get("post_count", 0) >= self.entity_min_post_count:
            return True
        if ctx and rec["name"] in ctx.shortlist_names:
            return True
        return False

    def validate(self, tokens: List[str],
                 context: Optional[ValidationContext] = None) -> List[ValidationResult]:
        results: List[Optional[ValidationResult]] = [None] * len(tokens)
        need_embed: list[tuple[int, str, str]] = []

        for i, tok in enumerate(tokens):
            norm = self._normalize(tok)
            if not norm or len(norm) < self.min_len:
                results[i] = ValidationResult(tok, None, 0.0, "drop", "too_short")
                continue
            if norm in self.whitelist:
                results[i] = ValidationResult(tok, norm, 1.0, "whitelist")
                continue
            direct = self.db.get_by_name(norm)
            if direct:
                if self._is_trusted_entity(direct, norm, context):
                    results[i] = ValidationResult(tok, direct["name"], 1.0, "exact")
                    continue
                # Niche entity collision — drop silently
                results[i] = ValidationResult(tok, None, 0.0, "drop", "exact_entity_gate")
                continue
            # Shortlist stem match: Danbooru encodes characters as
            #   character_(series)
            # LLM drafts often emit just the character name (`rococo`).
            # If the shortlist has `rococo_(girl_cafe_gun)`, treat a bare
            # `rococo` as meaning that shortlisted tag.
            if context and context.shortlist_names:
                stem_hit = None
                for sl_name in context.shortlist_names:
                    if sl_name.split("_(")[0] == norm:
                        stem_hit = sl_name
                        break
                if stem_hit:
                    rec = self.db.get_by_name(stem_hit)
                    if rec:
                        results[i] = ValidationResult(
                            tok, rec["name"], 1.0, "shortlist_stem",
                        )
                        continue
            aliased = self._resolve_alias(norm, context)
            if aliased:
                if self._is_trusted_entity(aliased, norm, context):
                    results[i] = ValidationResult(tok, aliased["name"], 1.0, "alias")
                    continue
                results[i] = ValidationResult(tok, None, 0.0, "drop", "alias_entity_gate")
                continue
            need_embed.append((i, tok, norm))

        if need_embed:
            queries = [norm.replace("_", " ") for _, _, norm in need_embed]
            vecs = self.embedder.encode(queries)
            scores, ids = self.index.search(vecs, 1)
            for (i, orig, norm), row_scores, row_ids in zip(need_embed, scores, ids):
                top_id = int(row_ids[0])
                top_score = float(row_scores[0])
                if top_id < 0:
                    results[i] = ValidationResult(orig, None, top_score, "drop", "no_candidate")
                    continue
                if top_score < self.semantic_threshold:
                    results[i] = ValidationResult(orig, None, top_score, "drop", "semantic_below_thr")
                    continue
                match = self.db.get_by_id(top_id)
                if not match or match["post_count"] < self.semantic_min_post_count:
                    results[i] = ValidationResult(orig, None, top_score, "drop", "semantic_low_popularity")
                    continue
                # Apply the same entity trust gate to semantic matches
                if not self._is_trusted_entity(match, norm, context):
                    results[i] = ValidationResult(orig, None, top_score, "drop", "semantic_entity_gate")
                    continue
                results[i] = ValidationResult(
                    orig, match["name"], top_score, "semantic",
                )

        return [r for r in results if r is not None]

    def validate_one(self, token: str,
                     context: Optional[ValidationContext] = None) -> ValidationResult:
        return self.validate([token], context=context)[0]

    def _sub_spans(self, token: str, min_parts: int = 1, max_parts: int = 3) -> List[str]:
        """Return longest-first contiguous sub-spans of a normalized token.
        'long_silver_hair' → ['long_silver', 'silver_hair', 'long', 'silver', 'hair']
        (The full phrase is excluded — caller already tried exact match on it.)
        """
        norm = self._normalize(token)
        parts = norm.split("_")
        if len(parts) < 2:
            return []
        out: List[str] = []
        # Longest → shortest; skip the full phrase length
        for length in range(min(len(parts) - 1, max_parts), min_parts - 1, -1):
            for i in range(len(parts) - length + 1):
                out.append("_".join(parts[i:i + length]))
        return out

    def validate_with_compound_split(
        self,
        tokens: List[str],
        context: Optional[ValidationContext] = None,
        max_splits_per_token: int = 3,
    ) -> List[ValidationResult]:
        """Variant of validate() that, when a draft token fails to match
        exact/alias/stem/semantic, retries with 2- and 1-word sub-spans
        and keeps any that DB-resolve. Lets 'long silver hair' recover
        as {'long_hair', 'silver_hair'} instead of getting dropped.

        One source token may produce multiple results when splits hit.
        """
        base = self.validate(tokens, context=context)
        expanded: List[ValidationResult] = []
        for r in base:
            if r.match_type != "drop":
                expanded.append(r)
                continue
            subs = self._sub_spans(r.original)
            if not subs:
                expanded.append(r)
                continue
            # Dedup while preserving longest-first order
            seen_sub: set[str] = set()
            ordered: List[str] = []
            for s in subs:
                if s not in seen_sub:
                    seen_sub.add(s)
                    ordered.append(s)
            sub_results = self.validate(ordered, context=context)
            # Cover every character position of the original with the first
            # match that claims it; keep up to max_splits_per_token hits.
            # Simpler: just take any non-drop hits, in longest-first order,
            # capped.
            hits = [sr for sr in sub_results
                    if sr.match_type in ("exact", "alias", "shortlist_stem")]
            if not hits:
                expanded.append(r)
                continue
            seen_canon: set[str] = set()
            kept = 0
            for sr in hits:
                if sr.canonical in seen_canon:
                    continue
                seen_canon.add(sr.canonical)
                expanded.append(ValidationResult(
                    original=r.original,
                    canonical=sr.canonical,
                    confidence=0.9,
                    match_type="compound_split",
                ))
                kept += 1
                if kept >= max_splits_per_token:
                    break
            if kept == 0:
                expanded.append(r)
        return expanded
