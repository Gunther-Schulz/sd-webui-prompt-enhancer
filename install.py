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
import time
import urllib.request

import launch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "data")
HF_REPO = "freedumb2000/anima-tagger-artifacts"

_TAG = "[sd-webui-prompt-enhancer]"


def _open_console():
    """The terminal, when our stdout is not it.

    Forge does not stream extension installers. It runs each install.py
    through modules/launch_utils.run() with live=False, which pipes BOTH
    stdout and stderr (launch_utils.py:69-70), collects the output, and
    prints it only after the process exits (launch_utils.py:171-173).
    So during a 1.1 GB download every print here — however often it is
    flushed, whichever stream it picks — sits in a pipe, and the user
    watches a console that has said nothing since "Version: neo 2.28".

    Writing to the controlling terminal escapes that pipe. It is a
    genuine second channel rather than a stream choice, which is why
    stderr is not the fix.

    Returns None when stdout is already the terminal (nothing to
    duplicate) or when there is no terminal to open — a service, a CI
    run, output redirected to a file. Both degrade to plain printing.
    """
    try:
        if sys.stdout.isatty():
            return None
    except Exception:
        return None
    try:
        return open("/dev/tty", "w")
    except Exception:
        return None


_CONSOLE = _open_console()


def _say(msg: str) -> None:
    """Every progress line from this hook goes through here.

    Written to BOTH channels when they differ: the terminal so the wait
    is visible while it happens, stdout so the line still lands in
    whatever collected it — Forge's post-install dump, a redirected log.
    That duplicates the block once in Forge's console at the end, which
    is the cheap side of the trade: the alternative is a silent hour.

    flush=True is not decoration either. Forge's launcher starts webui
    with `python -u`, but the bootstrap that runs install.py after a
    fresh clone does not, and a block-buffered "downloading 1.1 GB"
    banner reaches the console only once the download is already under
    way — the exact moment it existed to warn about.

    The tag goes on continuation lines too: extension installers run
    back to back with nothing naming them, so an untagged line during a
    long startup does not tell the reader who is busy.
    """
    line = f"{_TAG} {msg}"
    print(line, flush=True)
    if _CONSOLE is not None:
        try:
            _CONSOLE.write(line + "\n")
            _CONSOLE.flush()
        except Exception:
            pass


def _fmt_mb(nbytes: int) -> str:
    return f"{nbytes / 1024 / 1024:,.1f} MB"


# Hashing a file this large takes long enough that a silent stretch
# reads as a hang — the incident this reporting exists for (2026-08-21:
# startup sat with no output at all after "Version: neo 2.28", and the
# operator killed it believing Forge had wedged). Below the threshold
# the check is fast enough that announcing it is just noise.
_ANNOUNCE_HASH_BYTES = 64 * 1024 * 1024

# How much has to arrive before another progress line. The old code
# tested `downloaded % (16 MB) < 1 MB`, which silently depends on every
# read returning exactly 1 MB: one short read desynchronises the
# modulus for the rest of the file. That is why the 30 MB artefact
# reported once, at 54%, and never again.
_PROGRESS_EVERY_BYTES = 64 * 1024 * 1024


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


def _progress(done: int, total: int, t0: float) -> None:
    elapsed = max(time.monotonic() - t0, 1e-6)
    rate = done / elapsed
    pct = 100.0 * done / total
    if done >= total:
        tail = f"done in {elapsed:.0f}s"
    elif rate > 0:
        eta = (total - done) / rate
        tail = f"~{int(eta) // 60}m{int(eta) % 60:02d}s left"
    else:
        tail = "stalled"
    _say(f"    {done/1024/1024:>6.0f} / {total/1024/1024:>6.0f} MB "
         f"({pct:>3.0f}%)  {rate/1024/1024:>5.1f} MB/s  {tail}")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        reported = 0
        t0 = time.monotonic()
        with open(dest + ".part", "wb") as f:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded - reported >= _PROGRESS_EVERY_BYTES:
                    reported = downloaded
                    _progress(downloaded, total, t0)
        if total:
            _progress(downloaded, total, t0)
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
    _say(f"anima artefacts: checking for updates ({HF_REPO}) …")
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
        size = info.get("size", 0)
        if os.path.exists(local) and os.path.getsize(local) == size:
            loud = size >= _ANNOUNCE_HASH_BYTES
            if loud:
                _say(f"  verifying the local copy of {fname} "
                     f"({_fmt_mb(size)}) — this reads the whole file, "
                     f"so it takes a moment …")
            t0 = time.monotonic()
            try:
                if _sha256(local) == info.get("sha256", ""):
                    if loud:
                        _say(f"  {fname} is current "
                             f"({time.monotonic() - t0:.0f}s)")
                    continue  # already current
            except Exception:
                pass
            if loud:
                _say(f"  {fname} does not match the published checksum "
                     f"— it will be downloaded again")
        needed.append((fname, info))

    if not needed:
        _say("anima artefacts are up to date.")
        return

    total = sum(i.get("size", 0) for _, i in needed)
    _say(f"anima artefacts: downloading {len(needed)} file(s), "
         f"{_fmt_mb(total)} in total, from HuggingFace ({HF_REPO}) …")
    _say(f"  this runs once per artefact version, and Forge does not "
         f"finish starting until it is done — expect a wait.")
    started = time.monotonic()
    for fname, info in needed:
        dest = os.path.join(_DATA_DIR, fname)
        size = info.get("size", 0)
        _say(f"  [{_fmt_mb(size):>12}] {fname}")
        t0 = time.monotonic()
        try:
            _download(f"{base}/{fname}", dest)
            if info.get("sha256"):
                if size >= _ANNOUNCE_HASH_BYTES:
                    _say(f"    verifying the checksum of {_fmt_mb(size)} "
                         f"— another whole-file read …")
                got = _sha256(dest)
                if got != info["sha256"]:
                    os.remove(dest)
                    raise RuntimeError(f"sha256 mismatch on {fname}")
            _say(f"    ✓ {fname} ({time.monotonic() - t0:.0f}s)")
        except Exception as e:
            _say(f"    ✗ {fname}: {type(e).__name__}: {e}")
            if os.path.exists(dest + ".part"):
                try:
                    os.remove(dest + ".part")
                except Exception:
                    pass
    _say(f"anima artefacts done ({time.monotonic() - started:.0f}s).")


# ── entry ────────────────────────────────────────────────────────────
_install_deps()
try:
    _fetch_artefacts()
except Exception as e:
    _say(f"anima artefacts: unexpected error ({type(e).__name__}: {e}) "
         f"— extension will still work in rapidfuzz mode.")
