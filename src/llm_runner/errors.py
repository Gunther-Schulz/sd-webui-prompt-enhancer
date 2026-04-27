"""Errors raised by the llm_runner module.

Conventions (follow CLAUDE.md "no silent fallbacks"):
- LLMError is the base — catch this if you want to handle ANY runner
  failure generically.
- ModelLoadError specifically signals download / load / OOM problems —
  the runner itself failed to come up. Different remediation than a
  generation-time failure.
- TruncatedOutput is special: the model produced output but hit num_predict
  before completing. Carries the partial text — many callers want it
  (e.g. Prose mode shows "Truncated" status with the partial). Distinct
  from a generation crash because there's salvageable content.
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base for all llm_runner errors."""


class ModelLoadError(LLMError):
    """Model failed to download, load, or initialize.

    Carries the resolved model identifier (repo or path) for diagnosis.
    """

    def __init__(self, message: str, model_ref: str = ""):
        super().__init__(message)
        self.model_ref = model_ref


class TruncatedOutput(LLMError):
    """Generation hit num_predict (or hard token cap) before completing.

    The partial text is recoverable via .partial. Callers may choose to:
      - Use the partial as-is (Prose mode does this with a "Truncated" badge)
      - Retry with a higher num_predict
      - Surface the truncation to the user as an error

    This replaces the old `_TruncatedError` from prompt_enhancer.py.
    """

    def __init__(self, partial_text: str, reason: str):
        super().__init__(f"truncated: {reason}")
        self.partial = partial_text
        self.reason = reason
