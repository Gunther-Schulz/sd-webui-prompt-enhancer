"""bge-m3 embedder wrapper.

Loads a single sentence-transformers instance on the configured device
and exposes normalized dense encoding (inner product == cosine when
vectors are unit-normalized, which is how we build the faiss index).
"""

from typing import List

import numpy as np

from . import config


class Embedder:
    """Lazy-loaded bge-m3 dense embedder on GPU by default."""

    def __init__(self, model_name: str = config.EMBED_MODEL,
                 device: str = config.DEVICE):
        # Import here so importing this module doesn't drag torch into
        # environments that only need the metadata DB.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of strings into unit-normalized float32 vectors."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 512,
        )
        return vecs.astype("float32", copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single query; returns a (1, dim) float32 array."""
        return self.encode([text])
