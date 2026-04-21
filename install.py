"""Forge install hook.

Runs on extension load:
  1. Installs Python deps on first use (rapidfuzz, sentence-transformers, …).
  2. Downloads the pre-built anima_tagger artefacts from HuggingFace
     so the Anima retrieval pipeline just works out of the box —
     no separate manual scripts for the end user.

Everything under src/anima_tagger/scripts/ is DEV-only (maintainer
workflow: rebuild index, upload to HF). End users never touch it.
"""

import hashlib
import json
import os
import sys
import urllib.request

import launch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "data")
HF_REPO = "freedumb2000/anima-tagger-artifacts"


# ── 1. Python dependencies ────────────────────────────────────────────
_DEPS = [
    # (import name, pip spec, purpose)
    ("rapidfuzz",             "rapidfuzz>=3.0",
     "fast tag validation (all tag formats)"),
    ("sentence_transformers", "sentence-transformers>=5.0",
     "bge-m3 embedder + bge-reranker cross-encoder (Anima retrieval)"),
    ("faiss",                 "faiss-cpu>=1.8",
     "vector index for Anima tag retrieval"),
    ("huggingface_hub",       "huggingface_hub>=0.24",
     "Danbooru dataset + artefact download"),
    ("pyarrow",               "pyarrow>=15.0",
     "parquet I/O for downloaded datasets"),
]


def _install_deps():
    for import_name, pip_spec, purpose in _DEPS:
        if not launch.is_installed(import_name):
            launch.run_pip(
                f"install {pip_spec}",
                f"{pip_spec.split('>=')[0]} for sd-webui-prompt-enhancer ({purpose})",
            )


# ── 2. Pre-built artefacts auto-download ──────────────────────────────
# Remote layout:
#   https://huggingface.co/datasets/<HF_REPO>/resolve/main/<filename>
#
# Flow:
#   - Read /resolve/main/VERSION (small JSON with per-file sha256).
#   - For each artefact, check local sha256; skip if it matches.
#   - Otherwise download the file and verify.
#
# Gracefully no-ops when anything fails — the extension still works in
# rapidfuzz mode; Anima retrieval just stays unavailable until the
# next successful install.py run.

_ARTEFACTS = [
    "tags.sqlite",           # ~25 MB
    "tags.faiss",             # ~1.1 GB
    "cooccurrence.sqlite",   # ~3 MB
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest + ".part", "wb") as f:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (16 * 1024 * 1024) < (1 << 20):
                    pct = 100.0 * downloaded / total
                    print(f"    {downloaded/1024/1024:>6.0f} / "
                          f"{total/1024/1024:>6.0f} MB ({pct:.0f}%)", flush=True)
    os.replace(dest + ".part", dest)


def _warn(msg: str) -> None:
    """Print a prominently-formatted warning to stderr so it's visible
    in the Forge console log, not lost in a sea of startup messages."""
    bar = "*" * 72
    print("", file=sys.stderr)
    print(bar, file=sys.stderr)
    print(f"  sd-webui-prompt-enhancer — ANIMA RAG UNAVAILABLE", file=sys.stderr)
    for line in msg.rstrip().splitlines():
        print(f"  {line}", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("", file=sys.stderr)


def _fetch_artefacts():
    """Ensure data/ has the current pre-built artefacts from HF.

    Called on every extension load. Fast when files are already fresh
    (just reads VERSION to compare hashes). On failure: prints a
    visible multi-line warning AND writes a human-readable reason to
    data/.rag_unavailable so the runtime side can surface it in the UI.
    Extension continues to work — Anima tag format falls back to the
    rapidfuzz path, other formats are unaffected.
    """
    reason_path = os.path.join(_DATA_DIR, ".rag_unavailable")
    # Clear stale reason at start of each run
    if os.path.exists(reason_path):
        try:
            os.remove(reason_path)
        except Exception:
            pass

    def _record_failure(reason: str) -> None:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(reason_path, "w") as f:
                f.write(reason)
        except Exception:
            pass

    base = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
    ver_url = f"{base}/VERSION"
    try:
        with urllib.request.urlopen(ver_url, timeout=15) as r:
            manifest = json.loads(r.read())
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _warn(
            f"Could not reach HuggingFace to download the RAG index.\n"
            f"URL: {ver_url}\n"
            f"Error: {err}\n\n"
            f"The Anima tag format will fall back to the rapidfuzz\n"
            f"validation path — usable but without embedding-based\n"
            f"retrieval, shortlist injection, or co-occurrence pairing.\n\n"
            f"Other tag formats (Illustrious / NoobAI / Pony) are unaffected.\n\n"
            f"Fix: check network connectivity to huggingface.co and\n"
            f"restart Forge to retry."
        )
        _record_failure(f"HF unreachable at install time: {err}")
        return

    expected = manifest.get("files", {})
    os.makedirs(_DATA_DIR, exist_ok=True)
    needed = []
    for fname in _ARTEFACTS:
        info = expected.get(fname)
        if not info:
            continue
        local = os.path.join(_DATA_DIR, fname)
        if os.path.exists(local) and os.path.getsize(local) == info.get("size", 0):
            try:
                if _sha256(local) == info.get("sha256", ""):
                    continue  # already current
            except Exception:
                pass
        needed.append((fname, info))

    if not needed:
        return

    print(f"[sd-webui-prompt-enhancer] anima artefacts: downloading "
          f"{len(needed)} file(s) from HuggingFace ({HF_REPO}) …")
    for fname, info in needed:
        dest = os.path.join(_DATA_DIR, fname)
        size_mb = info.get("size", 0) / 1024 / 1024
        print(f"  [{size_mb:>7.1f} MB] {fname}")
        try:
            _download(f"{base}/{fname}", dest)
            if info.get("sha256"):
                got = _sha256(dest)
                if got != info["sha256"]:
                    os.remove(dest)
                    raise RuntimeError(f"sha256 mismatch on {fname}")
            print(f"    ✓ {fname}")
        except Exception as e:
            print(f"    ✗ {fname}: {type(e).__name__}: {e}")
            if os.path.exists(dest + ".part"):
                try:
                    os.remove(dest + ".part")
                except Exception:
                    pass
    print("[sd-webui-prompt-enhancer] anima artefacts done.")


# ── entry ────────────────────────────────────────────────────────────
_install_deps()
try:
    _fetch_artefacts()
except Exception as e:
    print(f"[sd-webui-prompt-enhancer] anima artefacts: unexpected error "
          f"({type(e).__name__}: {e}) — extension will still work in "
          f"rapidfuzz mode.")
