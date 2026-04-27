"""Anima-tagger integration glue for the prompt-enhancer extension.

The Anima retrieval pipeline (`src/anima_tagger/`) is the heavy
ML-shaped subsystem — bge-m3 embedder, FAISS index, reranker,
co-occurrence lookups. This package wraps it with the surface the
prompt-enhancer's mode handlers consume.

Submodules:
  stack       — singleton lifecycle (lazy load, error reporting,
                availability checks for RAG mode)
  modifiers   — modifier-driven Anima behaviors (safety tier,
                target_slot collection, source-pick post-fill)
  pipeline    — main pipeline functions (query expander,
                tag_from_draft, general-tag candidates, candidate
                fragment for the Pass-2 system prompt)
  sources     — source-pick + slot-retrieval (db_pattern /
                db_retrieve / deferred / prose-slot)
  filters     — tag filtering helpers driven by the anima DB
                (tags_have_category, filter_to_structural_tags)
"""

from . import stack, modifiers, pipeline, sources, filters

from .stack import (
    get_stack,
    available_for,
    use_pipeline,
    opt as anima_opt,
)

__all__ = [
    "stack", "modifiers", "pipeline", "sources", "filters",
    "get_stack", "available_for", "use_pipeline", "anima_opt",
]
