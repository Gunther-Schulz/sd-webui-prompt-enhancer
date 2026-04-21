"""anima_tagger — LLM-draft + retrieval-validate pipeline for Anima tags.

Public entry:
    stack = load_all()              # cheap: opens DB + reads faiss index
    with stack.models():            # loads bge-m3 + reranker onto GPU
        tags = stack.tagger.tag_from_draft(draft, safety="safe")
        sl   = stack.build_shortlist(source_prompt)
    # models unloaded here; VRAM reclaimed for image gen

Artefacts must be built first via scripts/build_index.py (or downloaded
via scripts/download_artifacts.py once that's set up). Missing artefacts
raise FileNotFoundError with the path to the missing file.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from . import config
from .cooccurrence import CoOccurrence
from .db import TagDB
from .embedder import Embedder
from .index import VectorIndex
from .reranker import Reranker
from .retriever import Retriever
from .rule_layer import apply_anima_rules
from .shortlist import Shortlist, build_shortlist
from .tagger import AnimaTagger
from .validator import TagValidator, ValidationResult


def _require(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found at {path}. "
            f"Run: python src/anima_tagger/scripts/build_index.py"
        )


@dataclass
class AnimaStack:
    """Persistent: DB + index + cooccurrence. Models (embedder + reranker)
    are loaded on-demand via `stack.models()` context manager so VRAM is
    reclaimed between calls (matches your OLLAMA_KEEP_ALIVE=0 pattern).

    The tagger / retriever / validator are only set during a `models()`
    block. Accessing them outside the block raises AttributeError.
    """
    db: TagDB
    index: VectorIndex
    cooccurrence: Optional[CoOccurrence]
    semantic_threshold: float = 0.80
    semantic_min_post_count: int = 100
    enable_reranker: bool = True

    # Populated only inside .models():
    embedder: Optional[Embedder] = field(default=None, init=False)
    reranker: Optional[Reranker] = field(default=None, init=False)
    validator: Optional[TagValidator] = field(default=None, init=False)
    retriever: Optional[Retriever] = field(default=None, init=False)
    tagger: Optional[AnimaTagger] = field(default=None, init=False)

    @contextmanager
    def models(self) -> Iterator["AnimaStack"]:
        """Load bge-m3 (+ reranker) onto GPU; free on exit."""
        try:
            self.embedder = Embedder()
            self.reranker = Reranker() if self.enable_reranker else None
            self.validator = TagValidator(
                db=self.db, index=self.index, embedder=self.embedder,
                semantic_threshold=self.semantic_threshold,
                semantic_min_post_count=self.semantic_min_post_count,
            )
            self.retriever = Retriever(
                embedder=self.embedder, index=self.index, db=self.db,
                reranker=self.reranker,
            )
            self.tagger = AnimaTagger(
                validator=self.validator, db=self.db,
                cooccurrence=self.cooccurrence,
            )
            yield self
        finally:
            self.embedder = None
            self.reranker = None
            self.validator = None
            self.retriever = None
            self.tagger = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def build_shortlist(self, source_prompt: str, modifier_keywords: str = "") -> Shortlist:
        """Must be called inside a `models()` block."""
        if self.retriever is None:
            raise RuntimeError(
                "build_shortlist requires models loaded — use `with stack.models(): ...`"
            )
        return build_shortlist(
            self.retriever, source_prompt=source_prompt,
            modifier_keywords=modifier_keywords,
        )


def load_all(semantic_threshold: float = 0.80,
             semantic_min_post_count: int = 100,
             enable_reranker: bool = True,
             enable_cooccurrence: bool = True) -> AnimaStack:
    """Open DB + index + cooccurrence (cheap). Models load on .models()."""
    _require(config.TAG_DB_PATH, "Tag DB")
    _require(config.FAISS_INDEX_PATH, "FAISS index")

    db = TagDB(config.TAG_DB_PATH, create=False)
    index = VectorIndex.load(config.FAISS_INDEX_PATH, dim=config.EMBED_DIM)
    cooc: Optional[CoOccurrence] = None
    if enable_cooccurrence and os.path.exists(config.COOCCURRENCE_PATH):
        cooc = CoOccurrence(config.COOCCURRENCE_PATH)
    return AnimaStack(
        db=db, index=index, cooccurrence=cooc,
        semantic_threshold=semantic_threshold,
        semantic_min_post_count=semantic_min_post_count,
        enable_reranker=enable_reranker,
    )


__all__ = [
    "AnimaStack",
    "AnimaTagger",
    "CoOccurrence",
    "Embedder",
    "Reranker",
    "Retriever",
    "Shortlist",
    "TagDB",
    "TagValidator",
    "ValidationResult",
    "VectorIndex",
    "apply_anima_rules",
    "build_shortlist",
    "config",
    "load_all",
]
