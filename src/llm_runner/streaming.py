"""Streaming wrapper — yields Chunk objects with progress + handles
cancel, stall detection, and thinking-tag suppression.

Replaces the inlined streaming loop in the old _call_llm. The state
machine is the same; the I/O changes from "read socket lines" to
"iterate llama-cpp-python's chunk generator".

A Chunk carries:
  - text:        the new tokens since the last chunk (UI appends)
  - words:       cumulative word count (for live UI status)
  - tokens:      cumulative token count
  - elapsed_s:   wall time since stream started
  - tps:         tokens per second
  - is_complete: True only on the FINAL chunk emitted; False on every
                 mid-stream chunk. Lets callers distinguish a graceful
                 finish from a cancel/truncation/stall break.

Thinking-tag suppression: Qwen3 models can emit `<think>...</think>`
blocks even when told not to (the existing `/no_think` prefix usually
works but we belt-and-suspender it). Tokens inside think blocks are
NOT yielded to the caller (their text isn't visible), but DO update
the stall timer (we're still making progress, just on hidden content).

Cancel: caller passes a threading.Event; the loop checks it between
chunks and breaks immediately.

Stall: time since last visible token exceeds stall_timeout_s → break,
mark incomplete. Caller decides retry/error.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass
class Chunk:
    """One streaming update emitted by LLMRunner.stream().

    text is the *delta* (new tokens this chunk only). The cumulative
    accumulation is the caller's job — but words/tokens/elapsed/tps are
    computed cumulatively here for convenience (UI status fields).
    """
    text: str
    words: int
    tokens: int
    elapsed_s: float
    tps: float
    is_complete: bool   # True only on the final chunk; False otherwise


class _ThinkingState:
    """State machine that tracks `<think>...</think>` boundaries in a
    streaming token sequence and suppresses thinking-tag content from
    the visible output stream.

    Tokens may arrive split mid-tag (e.g. a chunk like '<thi' followed by
    'nk>'), so we operate on cumulative buffer rather than per-chunk text.
    """

    def __init__(self, on_thinking: Optional[Callable[[bool], None]] = None):
        self.on_thinking = on_thinking
        self._thinking = False
        self._buffer = ""           # untransmitted suffix (may contain partial tag)
        self._notified = False      # has on_thinking been called yet?

    def filter(self, new_text: str) -> str:
        """Add new_text to the buffer, return any visible (non-think) text
        that's now safe to emit. Hidden text stays in the buffer until
        we know it's not a tag."""
        self._buffer += new_text
        out = []

        while self._buffer:
            if self._thinking:
                # Looking for </think>
                close_idx = self._buffer.find("</think>")
                if close_idx == -1:
                    # No close yet — but if buffer is short enough that a
                    # close tag could be partially present, hold it back.
                    if len(self._buffer) < len("</think>"):
                        break  # wait for more
                    # Otherwise drop everything up to last 8 chars
                    self._buffer = self._buffer[-len("</think>"):]
                    break
                # Found close — exit thinking, drop the tag
                self._buffer = self._buffer[close_idx + len("</think>"):]
                self._thinking = False
                if self.on_thinking and self._notified:
                    self.on_thinking(False)
            else:
                # Looking for <think>
                open_idx = self._buffer.find("<think>")
                if open_idx == -1:
                    # No open tag — emit everything except a possible
                    # partial tag at the end. Hold back len("<think>")-1
                    # chars in case a tag is mid-arrival.
                    keep = len("<think>") - 1
                    if len(self._buffer) > keep:
                        out.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    break
                # Found open — emit pre-tag content, enter thinking
                if open_idx > 0:
                    out.append(self._buffer[:open_idx])
                self._buffer = self._buffer[open_idx + len("<think>"):]
                self._thinking = True
                if self.on_thinking and not self._notified:
                    self.on_thinking(True)
                    self._notified = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any non-thinking remainder. Called once at end-of-stream."""
        if self._thinking:
            # Stream ended inside a think block — suppress the partial
            return ""
        # Emit whatever's in the buffer (we held back chars for partial-tag
        # detection; at end-of-stream there's no more incoming, so emit all)
        out = self._buffer
        self._buffer = ""
        return out


def stream_chunks(
    raw_token_iter: Iterator[str],
    *,
    cancel_flag: Optional[threading.Event] = None,
    stall_timeout_s: float = 30.0,
    max_total_time_s: float = 180.0,
    on_thinking: Optional[Callable[[bool], None]] = None,
    on_stall: Optional[Callable[[float], None]] = None,
) -> Iterator[Chunk]:
    """Wrap a raw token iterator with progress accounting, cancel/stall
    detection, and `<think>` suppression.

    Args:
        raw_token_iter: yields per-token text strings from the LLM.
            Empty strings are allowed and ignored. The iterator is
            considered EOF when it stops yielding.
        cancel_flag: when set, the loop exits ASAP. Caller-owned.
        stall_timeout_s: seconds since last *visible* token before
            considered stalled. Hidden (thinking) tokens reset the
            timer too — we're making progress, just not visibly.
        max_total_time_s: hard wall-clock cap regardless of progress.
            Catches "loops in thinking" that don't trip stall_timeout.
        on_thinking: called once with True when entering a think block,
            once with False when leaving.
        on_stall: called with elapsed seconds when stall is detected
            (right before the loop breaks). Lets caller log / surface
            UI state.

    Yields:
        Chunk objects. The final chunk has is_complete=True only when
        the stream ended naturally (raw_token_iter exhausted without
        cancel/stall/cap).
    """
    thinking = _ThinkingState(on_thinking=on_thinking)
    accumulated = []
    tokens_total = 0
    start_time = time.monotonic()
    last_token_time = start_time
    completed_naturally = False
    cancelled = False
    stalled = False
    timed_out = False

    try:
        for raw_token in raw_token_iter:
            now = time.monotonic()
            tokens_total += 1
            last_token_time = now

            # Check cancel BEFORE filtering
            if cancel_flag is not None and cancel_flag.is_set():
                cancelled = True
                break

            # Total wall time cap (catches stuck-in-thinking)
            if now - start_time > max_total_time_s:
                timed_out = True
                break

            # Filter through thinking-state machine
            if not raw_token:
                continue
            visible = thinking.filter(raw_token)

            if visible:
                accumulated.append(visible)
                cumulative = "".join(accumulated)
                elapsed = now - start_time
                yield Chunk(
                    text=visible,
                    words=len(cumulative.split()),
                    tokens=tokens_total,
                    elapsed_s=elapsed,
                    tps=(tokens_total / elapsed) if elapsed > 0 else 0.0,
                    is_complete=False,
                )

            # Stall check (after visible-emission, so stall timer refreshes
            # based on visible-token arrival, not raw-token arrival)
            if time.monotonic() - last_token_time > stall_timeout_s:
                stalled = True
                if on_stall:
                    on_stall(time.monotonic() - last_token_time)
                break
        else:
            # for-loop completed without break — natural EOF
            completed_naturally = True

    except Exception:
        # Re-raise after computing final state
        raise
    finally:
        # Final chunk — flush any held thinking-state buffer
        tail = thinking.flush()
        if tail:
            accumulated.append(tail)
        cumulative = "".join(accumulated)
        elapsed = time.monotonic() - start_time
        yield Chunk(
            text=tail,
            words=len(cumulative.split()),
            tokens=tokens_total,
            elapsed_s=elapsed,
            tps=(tokens_total / elapsed) if elapsed > 0 else 0.0,
            is_complete=completed_naturally,
        )
