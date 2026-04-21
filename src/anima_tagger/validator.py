"""Embedding-based tag validator.

For each draft token the LLM emits:

  1. Whitelist pass — Anima-convention tokens (masterpiece, score_7,
     safe, …) aren't in the Danbooru DB but must be kept verbatim.
  2. Exact DB match → keep canonical.
  3. Alias match → substitute canonical (Danbooru alias table covers
     clouds → cloud, twintail → twintails, etc.).
  4. Semantic match via bge-m3 → substitute if similarity ≥ threshold
     AND the matched tag has enough popularity (avoids obscure niche
     tags winning semantic ties).
  5. Otherwise drop.

Thresholds chosen from empirical tests (see scripts/verify.py output):
correct semantic matches score ≥0.78, wrong matches fall in 0.70–0.77.
A 0.80 cutoff is tighter than strictly needed for correct matches but
comfortably excludes the noise.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

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


@dataclass
class ValidationResult:
    original: str
    canonical: Optional[str]
    confidence: float
    match_type: str   # "whitelist" | "exact" | "alias" | "semantic" | "drop"


class TagValidator:
    def __init__(self,
                 db: TagDB,
                 index: VectorIndex,
                 embedder: Embedder,
                 whitelist: Optional[Set[str]] = None,
                 semantic_threshold: float = 0.80,
                 semantic_min_post_count: int = 100,
                 min_len: int = 3):
        self.db = db
        self.index = index
        self.embedder = embedder
        self.whitelist = whitelist if whitelist is not None else ANIMA_WHITELIST
        self.semantic_threshold = semantic_threshold
        self.semantic_min_post_count = semantic_min_post_count
        self.min_len = min_len
        # Build reverse alias map once
        self._alias_lookup = db.build_alias_lookup()

    @staticmethod
    def _normalize(token: str) -> str:
        t = token.strip().lower()
        # Strip @ prefix for lookup — artist prefix re-applied by rule layer
        if t.startswith("@"):
            t = t[1:].strip()
        t = t.replace(" ", "_").replace("-", "_")
        while "__" in t:
            t = t.replace("__", "_")
        return t.strip("_")

    def validate(self, tokens: List[str]) -> List[ValidationResult]:
        """Validate many tokens; embedding calls are batched for speed."""
        results: List[Optional[ValidationResult]] = [None] * len(tokens)
        need_embed: list[tuple[int, str, str]] = []   # (idx, original, norm)

        # Fast path: whitelist / exact / alias
        for i, tok in enumerate(tokens):
            norm = self._normalize(tok)
            if not norm or len(norm) < self.min_len:
                results[i] = ValidationResult(tok, None, 0.0, "drop")
                continue
            if norm in self.whitelist:
                results[i] = ValidationResult(tok, norm, 1.0, "whitelist")
                continue
            direct = self.db.get_by_name(norm)
            if direct:
                results[i] = ValidationResult(tok, direct["name"], 1.0, "exact")
                continue
            aliased = self._alias_lookup.get(norm)
            if aliased:
                results[i] = ValidationResult(tok, aliased, 1.0, "alias")
                continue
            need_embed.append((i, tok, norm))

        # Slow path: semantic via bge-m3 (batched)
        if need_embed:
            queries = [norm.replace("_", " ") for _, _, norm in need_embed]
            vecs = self.embedder.encode(queries)
            scores, ids = self.index.search(vecs, 1)
            for (i, orig, _), row_scores, row_ids in zip(need_embed, scores, ids):
                top_id = int(row_ids[0])
                top_score = float(row_scores[0])
                if top_id < 0 or top_score < self.semantic_threshold:
                    results[i] = ValidationResult(orig, None, top_score, "drop")
                    continue
                match = self.db.get_by_id(top_id)
                # Require popularity floor for semantic matches — niche
                # tags (post_count < threshold) shouldn't win fuzzy
                # ties over what the user probably meant.
                if not match or match["post_count"] < self.semantic_min_post_count:
                    results[i] = ValidationResult(orig, None, top_score, "drop")
                    continue
                results[i] = ValidationResult(
                    orig, match["name"], top_score, "semantic",
                )

        return [r for r in results if r is not None]

    def validate_one(self, token: str) -> ValidationResult:
        return self.validate([token])[0]
