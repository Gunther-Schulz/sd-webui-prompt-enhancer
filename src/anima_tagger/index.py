"""faiss-gpu vector index wrapper.

IDs are caller-provided (integer row ids matching TagDB.id) so that
retrieval results can be joined back to metadata without a secondary
lookup table. We use an IDMap on top of a flat inner-product index —
IP on unit-normalized vectors is equivalent to cosine similarity.

FlatIP is brute-force (O(n) per query). At 141k × 1024-dim that's
~1ms on GPU, so no approximate index is needed.
"""

import os
from typing import Tuple

import numpy as np


class VectorIndex:
    """FlatIP faiss wrapper.

    Defaults to CPU. The public faiss-gpu-cu12 wheels (v1.14.1) ship
    kernels only for sm_70..sm_89; Blackwell GPUs (sm_120, RTX 50xx)
    fail with 'no kernel image' at runtime. CPU brute-force over
    ~273k × 1024-dim vectors is ~30 ms per query — imperceptible
    against the surrounding embedder/reranker/LLM work. Pass
    use_gpu=True on supported hardware to override.
    """

    def __init__(self, dim: int, use_gpu: bool = False):
        import faiss
        self._faiss = faiss

        flat = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(flat)
        self.dim = dim

        if use_gpu and faiss.get_num_gpus() > 0:
            self._res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self._res, 0, self.index)
            self.on_gpu = True
        else:
            self._res = None
            self.on_gpu = False

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert vectors.dtype == np.float32
        assert ids.dtype == np.int64
        assert vectors.shape[0] == ids.shape[0]
        self.index.add_with_ids(vectors, ids)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids) arrays of shape (n_queries, k)."""
        if query.dtype != np.float32:
            query = query.astype("float32")
        scores, ids = self.index.search(query, k)
        return scores, ids

    def size(self) -> int:
        return int(self.index.ntotal)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        idx = self._faiss.index_gpu_to_cpu(self.index) if self.on_gpu else self.index
        self._faiss.write_index(idx, path)

    @classmethod
    def load(cls, path: str, dim: int, use_gpu: bool = False) -> "VectorIndex":
        import faiss
        cpu_idx = faiss.read_index(path)
        obj = cls(dim, use_gpu=False)  # don't allocate a new index; replace below
        if use_gpu and faiss.get_num_gpus() > 0:
            obj._res = faiss.StandardGpuResources()
            obj.index = faiss.index_cpu_to_gpu(obj._res, 0, cpu_idx)
            obj.on_gpu = True
        else:
            obj.index = cpu_idx
            obj.on_gpu = False
        return obj
