"""Anima-tagger integration glue for the prompt-enhancer extension.

The Anima retrieval pipeline (`src/anima_tagger/`) is the heavy
ML-shaped subsystem — bge-m3 embedder, FAISS index, reranker,
co-occurrence lookups. This package wraps it with the surface the
prompt-enhancer's mode handlers consume.

Submodules:
  stack       — singleton lifecycle (lazy load, error reporting,
                availability checks for RAG mode)

More submodules will be extracted from prompt_enhancer.py as the
refactor progresses (pipeline / sources / filters / safety …).
"""

from . import stack

# Convenience re-exports for stable callsite shape.
from .stack import (
    get_stack,
    available_for,
    use_pipeline,
    opt as anima_opt,
)

__all__ = [
    "stack",
    "get_stack",
    "available_for",
    "use_pipeline",
    "anima_opt",
]
