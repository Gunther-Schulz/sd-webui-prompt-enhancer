"""LLMRunner — the main interface for in-process LLM calls.

Lazy singleton. First .call() or .stream() triggers GGUF download + load.
Subsequent calls reuse the loaded model.

Two generation paths internally:
  1. **High-level** (config.use_low_level_dry == False): uses
     llama_cpp.Llama.create_chat_completion. Simple, well-tested,
     supports response_format for JSON. NO DRY (high-level API doesn't
     expose it yet).
  2. **Low-level** (config.use_low_level_dry == True; default): uses
     a custom token-by-token generation loop with a manually-built
     sampler chain that includes DRY. Supports JSON via grammar.

Both paths produce the same Chunk stream out of the streaming wrapper,
so callers don't care which is active.

Thread safety: a single Llama instance is NOT safe for concurrent use
across threads. We hold a re-entrant lock around every generation, so
multiple Forge tabs invoking the LLM serialize cleanly.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional

from .config import RunnerConfig, load_config
from .download import resolve_model_path
from .errors import LLMError, ModelLoadError, TruncatedOutput
from .samplers import (
    SamplerConfig,
    build_low_level_sampler_chain,
    free_sampler_chain,
    get_preset,
    DryNotAvailable,
)
from .streaming import Chunk, stream_chunks


# ── Singleton machinery ──────────────────────────────────────────────────

_runner_instance: Optional["LLMRunner"] = None
_runner_lock = threading.Lock()


def get_runner() -> "LLMRunner":
    """Return the process-wide LLMRunner singleton.

    First call constructs (but does NOT yet load — load happens lazily
    on first .stream() / .call() to avoid loading 6 GB on Forge startup
    if the user never invokes prompt-enhancer).
    """
    global _runner_instance
    with _runner_lock:
        if _runner_instance is None:
            _runner_instance = LLMRunner(load_config())
        return _runner_instance


def reset_runner() -> None:
    """Drop the singleton (unloads the model). Used after settings
    change, or by tests."""
    global _runner_instance
    with _runner_lock:
        if _runner_instance is not None:
            _runner_instance.unload()
            _runner_instance = None


# ── LLMRunner ────────────────────────────────────────────────────────────


class LLMRunner:
    """Owns the loaded llama-cpp-python Llama instance and handles
    generation. One instance per (config, process) — see get_runner()
    for the singleton entrypoint."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._llm = None                                  # llama_cpp.Llama, lazy
        self._gen_lock = threading.Lock()                 # serialize generation
        self._loaded_path: Optional[str] = None
        self._load_time_s: Optional[float] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """Resolve + load the model. Idempotent (no-op if already loaded
        with current config)."""
        if self.is_loaded():
            return

        path = resolve_model_path(self.config)

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ModelLoadError(
                "llama-cpp-python is not installed. "
                "Install via the extension's install.py (auto on Forge "
                "load), or manually: pip install llama-cpp-python",
                model_ref=path,
            ) from e

        print(f"[llm_runner] Loading model: {path}")
        print(f"[llm_runner]   compute={self.config.compute} "
              f"n_gpu_layers={self.config.effective_n_gpu_layers} "
              f"n_ctx={self.config.n_ctx}")
        t0 = time.monotonic()
        try:
            self._llm = Llama(
                model_path=path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.effective_n_gpu_layers,
                # We do our own chat templating — let llama-cpp use the
                # GGUF's embedded chat_template via the create_chat_*
                # path. For Qwen3, this handles thinking modes.
                verbose=False,
            )
        except Exception as e:
            self._llm = None
            raise ModelLoadError(
                f"Failed to load GGUF: {type(e).__name__}: {e}\n"
                f"Path: {path}\n"
                f"Likely causes: (a) corrupt download (re-run install.py "
                f"to refetch), (b) insufficient VRAM/RAM for the chosen "
                f"quant + n_ctx, (c) llama-cpp-python built without "
                f"matching CUDA support.",
                model_ref=path,
            ) from e
        self._load_time_s = time.monotonic() - t0
        self._loaded_path = path
        print(f"[llm_runner] Model loaded in {self._load_time_s:.1f}s")

    def unload(self) -> None:
        """Free model memory. Safe to call when not loaded."""
        if self._llm is not None:
            try:
                # llama-cpp-python's Llama supports __del__, but explicit
                # close() lets us free immediately rather than at GC.
                if hasattr(self._llm, "close"):
                    self._llm.close()
            except Exception:
                pass
        self._llm = None
        self._loaded_path = None
        self._load_time_s = None

    def info(self) -> Dict[str, Any]:
        """Diagnostic snapshot for status display."""
        return {
            "loaded": self.is_loaded(),
            "path": self._loaded_path,
            "load_time_s": self._load_time_s,
            "n_ctx": self.config.n_ctx,
            "compute": self.config.compute,
            "n_gpu_layers": self.config.effective_n_gpu_layers,
            "lifecycle": self.config.lifecycle,
            "low_level_dry": self.config.use_low_level_dry,
            "model_ref": self.config.model_ref,
        }

    # ── Generation: sync ──────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        system_prompt: str,
        *,
        temperature: Optional[float] = None,
        seed: int = -1,
        num_predict: int = 1024,
        json_schema: Optional[dict] = None,
        sampler_preset: Optional[str] = None,
        cancel_flag: Optional[threading.Event] = None,
        stall_timeout_s: float = 30.0,
    ) -> str:
        """Synchronous wrapper — collects all chunks into one string.

        Raises TruncatedOutput if the stream didn't complete naturally.
        Caller decides whether to keep e.partial or treat as error.
        """
        accumulated = []
        last_chunk = None
        for chunk in self.stream(
            prompt, system_prompt,
            temperature=temperature, seed=seed, num_predict=num_predict,
            json_schema=json_schema, sampler_preset=sampler_preset,
            cancel_flag=cancel_flag, stall_timeout_s=stall_timeout_s,
        ):
            accumulated.append(chunk.text)
            last_chunk = chunk

        text = "".join(accumulated)
        if last_chunk is None or not last_chunk.is_complete:
            raise TruncatedOutput(text, "stream ended before completion")
        return text

    # ── Generation: streaming ─────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        system_prompt: str,
        *,
        temperature: Optional[float] = None,
        seed: int = -1,
        num_predict: int = 1024,
        json_schema: Optional[dict] = None,
        sampler_preset: Optional[str] = None,
        cancel_flag: Optional[threading.Event] = None,
        stall_timeout_s: float = 30.0,
        on_thinking: Optional[Callable[[bool], None]] = None,
    ) -> Iterator[Chunk]:
        """Yield Chunks as the model generates.

        Auto-loads the model on first call. Auto-unloads after the call
        if config.lifecycle == 'unload_after_call'.

        json_schema, when set, constrains output via GBNF — guarantees
        structurally-valid JSON matching the schema. Truncation can still
        happen mid-object if num_predict caps; final chunk's is_complete
        will be False in that case.
        """
        # Lazy load
        if not self.is_loaded():
            self.load()

        sampler = get_preset(sampler_preset or self.config.sampler_preset)
        if temperature is not None:
            # Override per-call without mutating the preset
            sampler = SamplerConfig(**{**sampler.__dict__, "temperature": float(temperature)})

        # Serialize generation across calls (Llama is not thread-safe)
        with self._gen_lock:
            try:
                if self.config.use_low_level_dry:
                    raw_iter = self._stream_low_level(
                        prompt, system_prompt, sampler, seed, num_predict, json_schema
                    )
                else:
                    raw_iter = self._stream_high_level(
                        prompt, system_prompt, sampler, seed, num_predict, json_schema
                    )

                yield from stream_chunks(
                    raw_iter,
                    cancel_flag=cancel_flag,
                    stall_timeout_s=stall_timeout_s,
                    on_thinking=on_thinking,
                )
            finally:
                if self.config.lifecycle == "unload_after_call":
                    self.unload()

    # ── Generation paths ──────────────────────────────────────────────

    def _stream_high_level(
        self, prompt: str, system_prompt: str, sampler: SamplerConfig,
        seed: int, num_predict: int, json_schema: Optional[dict],
    ) -> Iterator[str]:
        """High-level path via Llama.create_chat_completion. No DRY."""
        kwargs = dict(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _maybe_no_think(prompt)},
            ],
            temperature=sampler.temperature,
            top_k=sampler.top_k,
            top_p=sampler.top_p,
            min_p=sampler.min_p,
            repeat_penalty=sampler.repeat_penalty,
            frequency_penalty=sampler.frequency_penalty,
            presence_penalty=sampler.presence_penalty,
            seed=seed,
            max_tokens=num_predict,
            stream=True,
        )
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }

        for piece in self._llm.create_chat_completion(**kwargs):
            choices = piece.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                yield content

    def _stream_low_level(
        self, prompt: str, system_prompt: str, sampler: SamplerConfig,
        seed: int, num_predict: int, json_schema: Optional[dict],
    ) -> Iterator[str]:
        """Low-level path: custom generation loop with DRY-enabled
        sampler chain. Falls back to high-level if DRY binding missing.

        Uses Llama's eval/sample low-level API:
          - tokenize prompt
          - eval prompt to fill KV cache
          - in a loop: sample a token via our chain, append to context,
                       eval the new token, decode token to text, yield
          - exit on EOS, num_predict cap, or grammar completion
        """
        try:
            chain = build_low_level_sampler_chain(self._llm, sampler, seed)
        except DryNotAvailable as e:
            # Visible warning, fall back to high-level for THIS call
            print(f"[llm_runner] DRY unavailable ({e}). "
                  f"Falling back to high-level samplers for this call.")
            yield from self._stream_high_level(
                prompt, system_prompt, sampler, seed, num_predict, json_schema
            )
            return

        try:
            from llama_cpp import llama_cpp

            # Reset llama state for fresh generation
            self._llm.reset()

            # Build the chat-formatted prompt manually (Qwen3 template).
            # The high-level path uses llama-cpp-python's create_chat_completion
            # which applies the GGUF's embedded template; here we mirror that
            # template directly so we can drop to eval/sample without losing
            # the chat structure. If your GGUF uses a non-standard template,
            # override _format_chat_qwen.
            prompt_text = _format_chat_qwen(system_prompt, _maybe_no_think(prompt))
            prompt_tokens = self._llm.tokenize(prompt_text.encode("utf-8"), special=True)

            # Optional JSON grammar
            grammar = None
            if json_schema is not None:
                from llama_cpp.llama_grammar import LlamaGrammar, json_schema_to_gbnf
                gbnf = json_schema_to_gbnf(json_schema)
                grammar = LlamaGrammar.from_string(gbnf, verbose=False)

            # Eval the prompt to populate KV cache
            self._llm.eval(prompt_tokens)

            # Generate token-by-token
            tokens_generated = 0
            eos_token_id = self._llm.token_eos()
            while tokens_generated < num_predict:
                # Sample via our chain
                # llama_sampler_sample takes the chain, the context,
                # and an index (-1 = last position).
                ctx = self._llm._ctx.ctx if hasattr(self._llm, "_ctx") else self._llm.ctx
                token_id = llama_cpp.llama_sampler_sample(chain, ctx, -1)

                if token_id == eos_token_id:
                    return  # natural completion

                # Apply grammar if active
                if grammar is not None:
                    llama_cpp.llama_sampler_accept(grammar.sampler, token_id)

                # Detokenize and yield
                token_bytes = self._llm.detokenize([token_id], special=False)
                if token_bytes:
                    try:
                        text = token_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        text = ""
                    if text:
                        yield text

                # Eval the new token to extend context
                self._llm.eval([token_id])
                tokens_generated += 1
        finally:
            free_sampler_chain(chain)


# ── Helpers ──────────────────────────────────────────────────────────────


def _maybe_no_think(prompt: str) -> str:
    """Prepend `/no_think` for Qwen3 models that ignore the chat-template
    think flag. Same trick the old Ollama code used. Idempotent."""
    if prompt.startswith("/no_think"):
        return prompt
    return f"/no_think\n{prompt}"


def _format_chat_qwen(system_prompt: str, user_prompt: str) -> str:
    """Manual qwen3 chat-template formatter for the low-level path.

    Mirrors the canonical chat_template from Qwen3 GGUFs (the one
    llama-cpp-python's high-level path applies automatically). Used
    only when bypassing create_chat_completion in favor of manual
    eval/sample for DRY support.

    Format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
