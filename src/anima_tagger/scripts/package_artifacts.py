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

    # Write VERSION locally first so we can upload it as a file
    version_path = os.path.join(config.DATA_DIR, "VERSION")
    with open(version_path, "w") as f:
        f.write(version_text)

    targets = [(version_path, "VERSION")]
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
