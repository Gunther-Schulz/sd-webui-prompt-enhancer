# Cancel Button Bug

## Current State
The cancel button partially works but has a stuck state issue.

## What Works
- Clicking Cancel while generating DOES stop Ollama streaming (threading flag works)
- After cancel, you CAN click Prose/Tags/Remix again to start a new generation
- The generation itself works correctly after cancel

## What's Broken
1. **Status stuck on "Cancelling..."** — after clicking Cancel, the status HTML never updates to "Cancelled" or clears. It stays on "Cancelling..." forever.
2. **Cancel button stops working** — after the first cancel, clicking Cancel again has no effect. The button appears to be dead/unresponsive.
3. Subsequent generations work but the status from cancel lingers

## Architecture

### Cancel mechanism
- `_cancel_flag = threading.Event()` — global flag
- Cancel button: `fn=lambda: _cancel_flag.set()`, `queue=False`, no outputs, JS sets "Cancelling..." via DOM
- `_call_llm()`: clears flag at start, checks flag in streaming loop, raises `InterruptedError(partial_content)` when cancelled
- Socket timeout (2s) on HTTP response so blocking reads unblock for flag checks
- Callers catch `InterruptedError` and return `("", "Cancelled")` to status

### Event chain for generation (e.g. Tags)
```
tags_btn.click(
    fn=lambda: "Generating tags...",   # step 1: update status
    outputs=[status]
).then(
    fn=_tags,                          # step 2: call LLM, return (result, status)
    outputs=[prompt_out, status]
)
```

### Cancel button
```
cancel_btn.click(
    fn=lambda: _cancel_flag.set(),     # set flag
    _js="...",                         # JS: set status to "Cancelling..."
    inputs=[], outputs=[],             # no Gradio outputs
    queue=False                        # bypass Gradio queue
)
```

## Root Cause Theory
The streaming loop uses `resp.readline()` with a 2s socket timeout. When Ollama is sending tokens continuously, the cancel flag is checked on every token. When Ollama is silent (thinking), the flag is checked every 2s on socket timeout.

The `InterruptedError` propagates up to the caller which returns `("", "Cancelled")`. This SHOULD update `prompt_out` and `status` via the `.then()` chain.

But it seems like Gradio's `.then()` chain doesn't deliver the return value after the event has been in progress for a while, or the status component gets into a broken state after JS direct DOM manipulation.

The cancel button itself becoming unresponsive after first use is the most critical issue. `queue=False` should make it always responsive, but something prevents it from firing on subsequent clicks.

## Attempted Fixes (all failed)
1. Gradio `cancels` parameter — caused stuck UI state
2. `InterruptedError` for proper status return — status still not updating
3. `queue=False` on cancel button — first click works, subsequent don't
4. JS-only status update — "Cancelling..." shows but never clears
5. Socket timeout for periodic flag checks — cancel itself works but status stuck
6. No outputs on cancel button (only flag set) — same issue
7. Various combinations of the above

## Files
- `/home/g/dev/Gunther-Schulz/sd-webui-prompt-enhancer/scripts/prompt_enhancer.py`
  - `_cancel_flag` — line ~762
  - `_call_llm()` — line ~767
  - Cancel button — search for `cancel_btn`
  - Event wiring — search for `prose_event`, `remix_event`, `tags_event`

## Environment
- Forge Classic (Gradio 3.x bundled)
- Python 3.11
- Ollama with huihui_ai/qwen3.5-abliterated:9b
- Streaming mode with `/no_think` prepended
