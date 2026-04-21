"""Forge install hook.

Installs deps on-demand the first time the extension is loaded.
Heavy one-time setup (Danbooru dataset download + faiss index build)
is NOT done here — it would add ~10 minutes and ~500 MB to first boot.
Users run that manually when they want Anima retrieval:

    python src/anima_tagger/scripts/download_data.py
    python src/anima_tagger/scripts/build_index.py

The extension still works without those artefacts — the anima_tagger
module only activates when Tag Format = Anima AND the index is built.
"""

import launch


_DEPS = [
    # (import name, pip spec, purpose)
    ("rapidfuzz",             "rapidfuzz>=3.0",
     "fast tag validation (all tag formats)"),
    ("sentence_transformers", "sentence-transformers>=5.0",
     "bge-m3 embedder + bge-reranker cross-encoder (Anima retrieval)"),
    ("faiss",                 "faiss-cpu>=1.8",
     "vector index for Anima tag retrieval"),
    ("huggingface_hub",       "huggingface_hub>=0.24",
     "Danbooru dataset download (Anima retrieval)"),
    ("datasets",              "datasets>=4.0",
     "Danbooru post stream for co-occurrence (Anima retrieval)"),
    ("pyarrow",               "pyarrow>=15.0",
     "parquet I/O for downloaded datasets"),
]


for import_name, pip_spec, purpose in _DEPS:
    if not launch.is_installed(import_name):
        launch.run_pip(f"install {pip_spec}",
                       f"{pip_spec.split('>=')[0]} for sd-webui-prompt-enhancer "
                       f"({purpose})")
