"""Shared helpers for all pipeline steps.

Centralizes:
  * Ollama call plumbing (single implementation, shared by every step
    so model/LLM swaps don't require per-step edits).
  * Access to the bootstrapped extension module (`pe`) so steps can
    use the real `_assemble_system_prompt`, `_build_style_string`,
    `_tag_formats`, etc. — NOT reimplement them. This is the
    no-island-tests contract.
  * The anima_tagger `stack` singleton accessor so every step that
    needs retrieval/embedder/reranker uses the SAME loaded stack,
    not a per-step reload.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from typing import Any, Dict, Optional


# ── Extension module bootstrap ───────────────────────────────────────

# Import the real prompt_enhancer.py (outside Forge) so every step can
# call the actual extension functions. The bootstrap stubs Forge's
# modules.* so prompt_enhancer loads cleanly without the UI.
_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.abspath(os.path.join(_EXP_DIR, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger.scripts._pe_bootstrap import pe  # noqa: E402

# Anima tagger is loaded lazily (RAG models take time); cached after
# first use. Every step that needs retrieval/embedder calls `get_stack()`.
_stack = None
_stack_cm = None
_stack_lock = threading.Lock()


def get_stack():
    """Return the loaded AnimaStack with models() already entered.

    Cached — first call loads DB + models (~10s); subsequent calls are
    free. Thread-safe via lock.

    Because models() is a context manager and we want the stack alive
    for the whole experiment session, we enter it once and never exit.
    The process dying is what releases GPU memory — acceptable for an
    offline experiment runner.
    """
    global _stack, _stack_cm
    with _stack_lock:
        if _stack is not None:
            return _stack
        from anima_tagger import load_all
        s = load_all()
        cm = s.models()
        cm.__enter__()
        _stack = s
        _stack_cm = cm
        return _stack


# ── Ollama client ────────────────────────────────────────────────────

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "huihui_ai/qwen3.5-abliterated:9b")


def call_llm(
    system_prompt: str,
    user_message: str,
    *,
    model: str = DEFAULT_MODEL,
    seed: int = -1,
    temperature: float = 0.8,
    num_predict: int = 1024,
    think: bool = False,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 300,
) -> str:
    """Single, canonical Ollama chat call.

    Returns stripped content string.

    Uses stream: True to match shipped prompt_enhancer.py's `_call_llm`.
    This matters — with stream: False, Ollama produced EMPTY content
    at seeds where stream: True gave valid expansion (discovered during
    the expander-bridge audit). The shipped code has always used stream
    mode. The harness used to use non-stream mode, which was an
    accidental divergence — it made some seeds fail silently in the
    harness only.

    Fail-loud: raises ValueError on empty response. Silent empty-string
    returns make downstream steps crash in confusing ways later.
    """
    # top_p matches shipped: 0.95 when think, else 0.8
    top_p = 0.95 if think else 0.8
    body = {
        "model": model,
        "stream": True,
        "think": bool(think),
        "keep_alive": "5m",
        "options": {
            "temperature": float(temperature),
            "seed": int(seed),
            "top_k": 20,
            "top_p": top_p,
            "repeat_penalty": 1.5,
            "presence_penalty": 1.5,
            "num_predict": int(num_predict),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think\n{user_message}" if not think else user_message},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    content_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = chunk.get("message", {}).get("content", "")
            if part:
                content_parts.append(part)
            if chunk.get("done"):
                break
    content = "".join(content_parts).strip()
    # Strip Qwen3 think blocks if they leaked through despite think=False,
    # matching shipped _strip_think_blocks behavior.
    import re as _re
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
    if not content:
        raise ValueError(
            f"Ollama returned empty content. model={model} seed={seed} "
            f"system_len={len(system_prompt)} user_len={len(user_message)}"
        )
    return content
