"""A/B test for #1: hybrid dense + sparse retrieval.

Question: does adding bge-m3 sparse scores alongside dense improve
retrieval quality for our task, and if so at what alpha?

  final_score = alpha * dense_cosine + (1 - alpha) * sparse_lexical

alpha=1.0 is our current state (dense only).
alpha=0.0 is purely sparse / BM25-like lexical.
Between: combined signal.

Setup (cached to data/sparse_index/):
  1. Load bge-m3 via FlagEmbedding (outputs dense + sparse per call).
  2. Re-embed all 273k tags. Cache dense (for parity) and sparse
     (new) vectors to disk.
  3. For each alpha in the sweep, retrieve top-20 for a panel of
     source prompts and print the top-6 per category.

Run:
    python src/anima_tagger/scripts/ab_hybrid_retrieval.py

Output is a diff table eyeballed by the maintainer. Concerns we're
looking for:
  - Does alpha < 1.0 bring back the name-overlap dominance we just
    escaped via E_2 (artist signatures)?
  - Does it surface any obviously-better matches when the user types
    a literal entity name?
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import config
from anima_tagger.db import TagDB

SPARSE_CACHE_DIR = os.path.join(config.DATA_DIR, "sparse_index")
SPARSE_VECS_PATH = os.path.join(SPARSE_CACHE_DIR, "tag_sparse.pkl")
DENSE_VECS_PATH = os.path.join(SPARSE_CACHE_DIR, "tag_dense.npy")
META_PATH = os.path.join(SPARSE_CACHE_DIR, "ids.npy")


def _format_for_embedding(rec: dict) -> str:
    """Mirror the format used at index-build time so query/doc align."""
    from anima_tagger.scripts.build_index import (
        _format_for_embedding as _fmt,
    )
    return _fmt(rec)


def _load_all_tag_records(db: TagDB) -> list[dict]:
    return list(db.iter_ordered())


def _build_sparse_index(force: bool = False) -> tuple[list, np.ndarray, np.ndarray]:
    """Encode all tags with bge-m3, caching sparse + dense outputs.

    Returns (sparse_list, dense_matrix, id_array).
    Sparse entries are dict[token_id -> weight] as produced by
    FlagEmbedding.BGEM3FlagModel.encode(return_sparse=True).
    """
    os.makedirs(SPARSE_CACHE_DIR, exist_ok=True)
    if not force and os.path.exists(SPARSE_VECS_PATH) and os.path.exists(DENSE_VECS_PATH):
        print(f"Loading cached sparse + dense vectors from {SPARSE_CACHE_DIR}")
        with open(SPARSE_VECS_PATH, "rb") as f:
            sparse = pickle.load(f)
        dense = np.load(DENSE_VECS_PATH)
        ids = np.load(META_PATH)
        print(f"  {len(sparse):,} sparse rows, dense shape={dense.shape}")
        return sparse, dense, ids

    print("Building sparse + dense vectors from scratch …")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    db = TagDB(config.TAG_DB_PATH, create=False)
    records = _load_all_tag_records(db)
    # Same embedding text formatting as the main index
    texts = [_format_for_embedding(r) for r in records]
    ids = np.array([r["id"] for r in records], dtype=np.int64)
    print(f"  encoding {len(texts):,} tags …")

    t0 = time.perf_counter()
    # BGE-M3 sparse is a dict of {token_id: weight} per input
    out = model.encode(
        texts, batch_size=64,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
        max_length=512,
    )
    print(f"  encoded in {time.perf_counter()-t0:.0f}s")

    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    # Normalize dense (same as production)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(norms, 1e-8, None)
    sparse = out["lexical_weights"]

    with open(SPARSE_VECS_PATH, "wb") as f:
        pickle.dump(sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(DENSE_VECS_PATH, dense)
    np.save(META_PATH, ids)
    print(f"  cached to {SPARSE_CACHE_DIR}")
    return sparse, dense, ids


def _sparse_score(q_sparse: dict, doc_sparse: dict) -> float:
    """Inner product of two sparse weight dicts (BGE-M3 convention)."""
    if not q_sparse or not doc_sparse:
        return 0.0
    if len(q_sparse) > len(doc_sparse):
        q_sparse, doc_sparse = doc_sparse, q_sparse
    s = 0.0
    for k, v in q_sparse.items():
        if k in doc_sparse:
            s += v * doc_sparse[k]
    return float(s)


def _encode_query(model, text: str) -> tuple[np.ndarray, dict]:
    out = model.encode(
        [text], batch_size=1,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
        max_length=256,
    )
    dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
    dense /= max(1e-8, float(np.linalg.norm(dense)))
    sparse = out["lexical_weights"][0]
    return dense, sparse


def _hybrid_topk(q_dense: np.ndarray, q_sparse: dict,
                 dense_mat: np.ndarray, sparse_list: list,
                 ids: np.ndarray, alpha: float, k: int) -> list[tuple[int, float]]:
    """Top-k tag ids by (alpha * dense + (1-alpha) * sparse)."""
    # Dense similarity vector: (n,)
    dense_scores = dense_mat @ q_dense
    if alpha < 1.0:
        sparse_scores = np.fromiter(
            (_sparse_score(q_sparse, s) for s in sparse_list),
            dtype=np.float32, count=len(sparse_list),
        )
    else:
        sparse_scores = 0.0
    # Normalize sparse to roughly same dynamic range as dense before mixing:
    # cosine on normalized bge-m3 dense is [0, 1]; sparse inner products
    # can reach much higher. Rescale via max to keep alpha interpretable.
    if isinstance(sparse_scores, np.ndarray) and sparse_scores.size:
        m = float(sparse_scores.max())
        if m > 0:
            sparse_scores = sparse_scores / m
    combined = alpha * dense_scores + (1.0 - alpha) * (sparse_scores if isinstance(sparse_scores, np.ndarray) else 0.0)
    # Top-k
    top_idx = np.argpartition(-combined, k)[:k]
    top_idx = top_idx[np.argsort(-combined[top_idx])]
    return [(int(ids[i]), float(combined[i])) for i in top_idx]


# Test panel
SOURCES = [
    ("girl-reading-cafe", "a girl reading in a cafe"),
    ("miku-maid",         "hatsune miku in a maid uniform"),
    ("samurai-dusk",      "a lone samurai at dusk with a sword"),
    ("dragon-tower",      "a dragon perched on a ruined tower at sunset"),
]
ALPHAS = [1.0, 0.9, 0.7, 0.5, 0.3]


def main() -> int:
    sparse, dense, ids = _build_sparse_index(force=False)

    db = TagDB(config.TAG_DB_PATH, create=False)
    # Build id → (name, category, post_count) for fast lookup
    rec_by_id = {r["id"]: r for r in _load_all_tag_records(db)}

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    for label, src in SOURCES:
        print("\n" + "=" * 80)
        print(f"SOURCE: {src!r}")
        q_dense, q_sparse = _encode_query(model, src)

        for alpha in ALPHAS:
            top = _hybrid_topk(q_dense, q_sparse, dense, sparse, ids,
                                alpha=alpha, k=30)
            # Split by category, show top-6 per
            by_cat: dict[int, list[str]] = {1: [], 4: [], 3: []}
            for tag_id, _score in top:
                rec = rec_by_id.get(tag_id)
                if not rec:
                    continue
                cat = rec["category"]
                if cat in by_cat and len(by_cat[cat]) < 6:
                    by_cat[cat].append(rec["name"].replace("_", " "))
            print(f"\n  α = {alpha:>4.1f}   ({'dense only' if alpha == 1.0 else 'mixed'})")
            print(f"    artists:    {by_cat[1]}")
            print(f"    characters: {by_cat[4]}")
            print(f"    series:     {by_cat[3]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
