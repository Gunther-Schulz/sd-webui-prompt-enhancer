"""Hybrid retriever: dense + exact-name match + rerank.

Pipeline per query:
  1. Exact/near-exact name match against the tag vocabulary via rapidfuzz.
     Guarantees that a user-typed name like "hatsune miku" surfaces its
     canonical tag even if dense retrieval ranks it lower.
  2. Dense retrieval via bge-m3 + faiss-gpu top-K.
  3. Candidate union: name-matches + dense, de-duplicated.
  4. Cross-encoder rerank (bge-reranker-v2-m3) to top-N.
  5. Optional category filter + popularity floor.

Returns structured candidate dicts with category / post_count so the
caller (tagger / shortlist / rule-layer) can bucket and format as
needed.
"""

from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import process as rf_process
from rapidfuzz.distance import Levenshtein as rf_levenshtein

from . import config
from .db import TagDB
from .embedder import Embedder
from .index import VectorIndex
from .reranker import Reranker


@dataclass
class Candidate:
    id: int
    name: str
    category: int
    post_count: int
    aliases: str
    wiki: str
    score: float


def _format_for_rerank(tag: dict) -> str:
    """Compact text passed to the cross-encoder for (query, tag) scoring."""
    parts = [f"{tag['name']} [cat={tag['category']}]"]
    if tag.get("aliases"):
        parts.append(f"aliases: {tag['aliases']}")
    if tag.get("wiki"):
        # Keep wiki short — reranker context window is limited
        parts.append(tag["wiki"][:500])
    return " | ".join(parts)


class Retriever:
    def __init__(self,
                 embedder: Embedder,
                 index: VectorIndex,
                 db: TagDB,
                 reranker: Optional[Reranker] = None):
        self.embedder = embedder
        self.index = index
        self.db = db
        self.reranker = reranker
        # Cache the full vocab list for exact-match pass. Small (141k strings).
        self._vocab = db.all_names()

    def retrieve(self,
                 query: str,
                 retrieve_k: int = config.DEFAULT_RETRIEVE_K,
                 final_k: int = config.DEFAULT_FINAL_K,
                 category: Optional[int] = None,
                 min_post_count: int = config.MIN_POST_COUNT) -> List[Candidate]:
        """Run the full hybrid retrieval pipeline for one query string."""
        # --- 1. Exact / near-exact name match
        name_hits: list[dict] = []
        # Normalize: underscore form matches DB canonical form
        q_norm = query.strip().lower().replace(" ", "_")
        # Quick direct hit
        direct = self.db.get_by_name(q_norm)
        if direct:
            name_hits.append(direct)
        # Fuzzy top-3 for typo-tolerant name matching
        fuzzy = rf_process.extract(
            q_norm, self._vocab, scorer=rf_levenshtein.distance,
            score_cutoff=3, limit=3,
        )
        for match_name, _dist, _ in fuzzy:
            t = self.db.get_by_name(match_name)
            if t and t not in name_hits:
                name_hits.append(t)

        # --- 2. Dense retrieval
        q_vec = self.embedder.encode_one(query)
        scores, ids = self.index.search(q_vec, retrieve_k)
        dense_ids = [int(i) for i in ids[0] if i >= 0]
        dense_tags = self.db.get_by_ids(dense_ids)
        dense_score_by_id = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}

        # --- 3. Union (preserve order: name hits first, then dense)
        seen_ids: set[int] = set()
        pool: list[dict] = []
        for t in name_hits + dense_tags:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            pool.append(t)

        # Popularity floor + category filter
        pool = [t for t in pool
                if t["post_count"] >= min_post_count
                and (category is None or t["category"] == category)]

        if not pool:
            return []

        # --- 4. Rerank (or fall back to dense score order)
        if self.reranker:
            pairs = [(t["id"], _format_for_rerank(t)) for t in pool]
            ranked = self.reranker.rerank(query, pairs, top_k=final_k)
            id_to_tag = {t["id"]: t for t in pool}
            return [
                Candidate(
                    id=cid,
                    name=id_to_tag[cid]["name"],
                    category=id_to_tag[cid]["category"],
                    post_count=id_to_tag[cid]["post_count"],
                    aliases=id_to_tag[cid]["aliases"],
                    wiki=id_to_tag[cid]["wiki"],
                    score=score,
                )
                for cid, score in ranked
            ]

        # No reranker: sort pool by dense score, fall back to post_count
        def _score(t: dict) -> float:
            return dense_score_by_id.get(t["id"], 0.0)
        pool_sorted = sorted(pool, key=lambda t: (_score(t), t["post_count"]), reverse=True)
        return [
            Candidate(
                id=t["id"], name=t["name"], category=t["category"],
                post_count=t["post_count"], aliases=t["aliases"], wiki=t["wiki"],
                score=_score(t),
            )
            for t in pool_sorted[:final_k]
        ]
