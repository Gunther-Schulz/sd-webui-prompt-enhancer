"""anima_tagger — LLM-draft + retrieval-validate pipeline for Anima tags.

Public entry points:
  load_all()   — builds the full stack (embedder, index, db, reranker,
                 validator, tagger, cooccurrence) once; returns an
                 AnimaStack dataclass you can use.

Typical use from the Forge-side code:

  from anima_tagger import load_all
  stack = load_all()
  final_tags = stack.tagger.tag_from_draft(raw_llm_output, safety="safe")
  shortlist = stack.build_shortlist(source_prompt)

Artefacts must be built first via scripts/build_index.py. If they're
missing, calls will raise FileNotFoundError with the missing path.
"""

import os
from dataclasses import dataclass
from typing import Optional

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
    """All the loaded models + data structures, sharing resources."""
    db: TagDB
    embedder: Embedder
    index: VectorIndex
    reranker: Optional[Reranker]
    validator: TagValidator
    retriever: Retriever
    tagger: AnimaTagger
    cooccurrence: Optional[CoOccurrence]

    def build_shortlist(self, source_prompt: str, modifier_keywords: str = ""):
        """Shortlist of real artists/characters/series for prose-pass RAG."""
        return build_shortlist(
            self.retriever, source_prompt=source_prompt,
            modifier_keywords=modifier_keywords,
        )


def load_all(enable_reranker: bool = True,
             enable_cooccurrence: bool = True,
             semantic_threshold: float = 0.80) -> AnimaStack:
    """Load the full tagging stack once. Returns a shared AnimaStack."""
    _require(config.TAG_DB_PATH, "Tag DB")
    _require(config.FAISS_INDEX_PATH, "FAISS index")

    db = TagDB(config.TAG_DB_PATH, create=False)
    embedder = Embedder()
    index = VectorIndex.load(config.FAISS_INDEX_PATH, dim=config.EMBED_DIM)
    reranker = Reranker() if enable_reranker else None

    cooc: Optional[CoOccurrence] = None
    if enable_cooccurrence and os.path.exists(config.COOCCURRENCE_PATH):
        cooc = CoOccurrence(config.COOCCURRENCE_PATH)

    validator = TagValidator(
        db=db, index=index, embedder=embedder,
        semantic_threshold=semantic_threshold,
    )
    retriever = Retriever(embedder=embedder, index=index, db=db, reranker=reranker)
    tagger = AnimaTagger(validator=validator, db=db, cooccurrence=cooc)

    return AnimaStack(
        db=db, embedder=embedder, index=index, reranker=reranker,
        validator=validator, retriever=retriever, tagger=tagger,
        cooccurrence=cooc,
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
