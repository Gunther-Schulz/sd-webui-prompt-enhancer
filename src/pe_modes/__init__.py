"""Mode handlers for the prompt-enhancer extension's UI buttons.

Will eventually contain one module per generation mode (prose, hybrid,
tags, remix). For now: shared helpers used by the still-inline
handlers in scripts/prompt_enhancer.py.

  pe_modes._shared      — small formatters + builders shared across
                          handlers (status HTML, user-message build,
                          negative-hint suffix, progress-chunk render).
"""

from . import _shared

__all__ = ["_shared"]
