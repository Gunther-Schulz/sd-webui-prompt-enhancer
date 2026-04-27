"""Public API for the in-process LLM runner.

Drops Ollama as the LLM backend; everything goes through llama-cpp-python
in this Forge process. See client.py for the LLMRunner class and its
load/unload/stream/call methods.

Module entry points:

    from llm_runner import get_runner
    runner = get_runner()
    text = runner.call(prompt, system_prompt, temperature=0.7)
    for chunk in runner.stream(prompt, system_prompt):
        ...

The runner is a lazy singleton — first call triggers the GGUF download
(~5-7 GB on first install) and model load. Subsequent calls reuse the
loaded model.
"""

from .client import get_runner, LLMRunner, reset_runner
from .errors import LLMError, ModelLoadError, TruncatedOutput
from .streaming import Chunk
from .samplers import SamplerConfig, SAMPLER_PRESETS, get_preset

__all__ = [
    "get_runner",
    "reset_runner",
    "LLMRunner",
    "LLMError",
    "ModelLoadError",
    "TruncatedOutput",
    "Chunk",
    "SamplerConfig",
    "SAMPLER_PRESETS",
    "get_preset",
]
