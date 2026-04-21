"""DEV-ONLY: package + upload pre-built artefacts to HuggingFace.

Users never run this. It's the maintainer's one-off chore after every
rebuild of data/tags.sqlite, data/tags.faiss, data/cooccurrence.sqlite.

Uploads to:
  https://huggingface.co/datasets/<HF_REPO>

End-user install.py pulls these files via huggingface_hub.hf_hub_download
and drops them into data/ on first Anima-Tag-Format use.

Auth:
  Set HF_TOKEN env var (or use `huggingface-cli login` once).

Run:
    # Set these if different from defaults:
    export HF_REPO=Gunther-Schulz/anima-tagger-artifacts
    export HF_TOKEN=<your token>
    python src/anima_tagger/scripts/package_artifacts.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import config

HF_REPO = os.environ.get("HF_REPO", "freedumb2000/anima-tagger-artifacts")


ARTEFACTS = [
    (config.TAG_DB_PATH,       "tags.sqlite",         True),
    (config.FAISS_INDEX_PATH,  "tags.faiss",          True),
    (config.COOCCURRENCE_PATH, "cooccurrence.sqlite", False),  # optional
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_dataset_card() -> str:
    """Return the Markdown README.md for the HF dataset repo."""
    return """---
license: cc0-1.0
task_categories:
  - text-retrieval
  - feature-extraction
language:
  - en
tags:
  - danbooru
  - anime
  - rag
  - tag-retrieval
  - anima
size_categories:
  - 100K<n<1M
---

# anima-tagger-artifacts

Pre-built retrieval artefacts for the **Anima** tag format of the
[`sd-webui-prompt-enhancer`](https://github.com/Gunther-Schulz/sd-webui-prompt-enhancer)
Stable Diffusion WebUI extension. Lets the extension's Anima pipeline
do real-time embedding-based tag validation and shortlist retrieval
without users needing to rebuild a 270k+ entry FAISS index locally.

## Contents

| File | Size | Description |
|---|---|---|
| `tags.sqlite` | ~30 MB | 273,025 Danbooru tags (name, category, post count, aliases, wiki). Post-count floor 10. |
| `tags.faiss` | ~1.1 GB | FAISS FlatIP index of bge-m3 embeddings (1024-dim) for every tag. Artist and character embeddings include co-occurrence signatures (top-12 general tags from their actual Danbooru posts). |
| `cooccurrence.sqlite` | ~10 MB | Pointwise-mutual-information table for character↔series, character↔artist, series↔character pairs. Enables automatic series-pairing (e.g. `hatsune_miku` → `vocaloid`) at query time. |
| `VERSION` | <1 kB | JSON manifest with per-file sha256 + size + build date. |

## Usage

Automatic — the extension's `install.py` downloads these on Forge
startup, verifies sha256 against `VERSION`, and re-downloads when the
upstream hash changes.

Manual (e.g. for other projects):

```python
from huggingface_hub import hf_hub_download

for fname in ("tags.sqlite", "tags.faiss", "cooccurrence.sqlite", "VERSION"):
    hf_hub_download(
        repo_id="freedumb2000/anima-tagger-artifacts",
        filename=fname, repo_type="dataset",
        local_dir="./data",
    )
```

## How it was built

1. Tag metadata + wiki from
   [`NSFW-API/DanBooruTagsAndWikiDumpSept2025`](https://huggingface.co/datasets/NSFW-API/DanBooruTagsAndWikiDumpSept2025)
   (1.59M tags; filtered to post_count ≥ 10 → 273,025 tags).
2. Post-level tag sets from
   [`isek-ai/danbooru-tags-2024`](https://huggingface.co/datasets/isek-ai/danbooru-tags-2024)
   (streamed first 500k posts).
3. For each artist/character, compute top-12 co-occurring general tags
   across their posts → "style signature".
4. Format each tag as `"<name> (<category>) | aliases: ... | <wiki excerpt>
   | associated with: <top co-occurring tags>"` and embed with
   [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (fp16 on GPU,
   normalized, 1024-dim).
5. FAISS FlatIP index over all 273k vectors.
6. PMI table over the same post sample for character↔series pairing.

Full rebuild: ~10 minutes on a modern GPU.

## License

The artefacts themselves are released under CC0-1.0. The upstream
Danbooru tag data is public-domain by convention. Refer to the
linked source datasets for their own attribution requirements.

The underlying embedding model ([`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3))
is MIT-licensed.
"""


def _build_version_file() -> str:
    """Return the text content of a VERSION file listing each artefact
    with size + sha256. Used by the end-user client to verify integrity
    and detect when a new version is available."""
    meta = {
        "build_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "format_version": 1,
        "files": {},
    }
    for local_path, hf_name, required in ARTEFACTS:
        if not os.path.exists(local_path):
            if required:
                raise FileNotFoundError(
                    f"required artefact {local_path} missing — run build_index.py first"
                )
            continue
        meta["files"][hf_name] = {
            "size": os.path.getsize(local_path),
            "sha256": _sha256(local_path),
        }
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"


def main() -> int:
    from huggingface_hub import HfApi, upload_file, create_repo

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        create_repo(
            repo_id=HF_REPO, repo_type="dataset", token=token,
            exist_ok=True, private=False,
        )
        print(f"Repo ready: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"  (create_repo: {e}) — continuing")

    version_text = _build_version_file()
    print("\nVERSION metadata:\n" + version_text)

    # Write VERSION + README locally so we can upload them
    version_path = os.path.join(config.DATA_DIR, "VERSION")
    with open(version_path, "w") as f:
        f.write(version_text)
    readme_path = os.path.join(config.DATA_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(_build_dataset_card())

    targets = [
        (readme_path,  "README.md"),
        (version_path, "VERSION"),
    ]
    for local_path, hf_name, required in ARTEFACTS:
        if not os.path.exists(local_path):
            if required:
                print(f"  ✗ missing required: {local_path}")
                return 1
            print(f"  (skip optional) {hf_name}")
            continue
        targets.append((local_path, hf_name))

    print(f"\nUploading {len(targets)} files to {HF_REPO} …")
    for local_path, hf_name in targets:
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  [{size_mb:>7.1f} MB] {local_path}  →  {hf_name}")
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_name,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Update {hf_name}",
        )
    print("\nAll done.")
    print(f"Public URL: https://huggingface.co/datasets/{HF_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
