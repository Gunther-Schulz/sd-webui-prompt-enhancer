"""bge-reranker-v2-m3 cross-encoder wrapper.

Takes the top-K candidates from the faiss retriever and re-scores each
(query, candidate) pair jointly. Adds ~100 ms per query on GPU but
sharpens top-of-list relevance meaningfully.
"""

from typing import List, Sequence, Tuple

from . import config


class Reranker:
    """bge-reranker-v2-m3 in fp16 (halves VRAM, negligible quality hit)."""

    def __init__(self, model_name: str = config.RERANK_MODEL,
                 device: str = config.DEVICE,
                 dtype: str = "float16"):
        import torch
        from sentence_transformers import CrossEncoder
        torch_dtype = getattr(torch, dtype) if device.startswith("cuda") else torch.float32
        self.model = CrossEncoder(
            model_name, device=device,
            model_kwargs={"torch_dtype": torch_dtype},
        )

    def rerank(self,
               query: str,
               candidates: Sequence[Tuple[int, str]],
               top_k: int) -> List[Tuple[int, float]]:
        """Re-rank (id, text) pairs by cross-encoder score.

        Returns a list of (id, score) sorted descending, truncated to top_k.
        """
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            ((cid, float(s)) for (cid, _), s in zip(candidates, scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
