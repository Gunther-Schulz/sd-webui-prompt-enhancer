"""Configuration constants for anima_tagger.

All paths are relative to the extension root. Override via environment
variables when needed (e.g. ANIMA_TAGGER_DATA_DIR).
"""

import os

# Extension root (parent of src/)
_EXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data directory (indexes, SQLite db, HF cache artefacts). Gitignored.
DATA_DIR = os.environ.get(
    "ANIMA_TAGGER_DATA_DIR",
    os.path.join(_EXT_ROOT, "data"),
)

# Persisted artefacts
TAG_DB_PATH = os.path.join(DATA_DIR, "tags.sqlite")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "tags.faiss")
COOCCURRENCE_PATH = os.path.join(DATA_DIR, "cooccurrence.sqlite")

# Source Danbooru CSV (already in the extension, used as fallback source)
DANBOORU_CSV_PATH = os.path.join(
    _EXT_ROOT, "tags", "danbooru.csv",
)

# Models
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_DIM = 1024  # bge-m3 dense output dimension

# Runtime
DEVICE = os.environ.get("ANIMA_TAGGER_DEVICE", "cuda")

# Retrieval
DEFAULT_RETRIEVE_K = 100   # raw top-K before rerank
DEFAULT_FINAL_K = 30       # after rerank
DEFAULT_SHORTLIST_K = 20   # for prose-pass RAG injection

# Danbooru tag categories (standard)
CAT_GENERAL = 0
CAT_ARTIST = 1
CAT_COPYRIGHT = 3  # a.k.a. "series"
CAT_CHARACTER = 4
CAT_META = 5

# Popularity floor for retrieval candidates (post_count threshold).
# Very rare tags (under ~10 posts) are usually noise.
MIN_POST_COUNT = 10
