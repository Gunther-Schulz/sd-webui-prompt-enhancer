"""LLM-call layer for the prompt-enhancer extension.

Wraps src.llm_runner with the call/streaming signatures
prompt_enhancer.py uses internally. Lives separately so prompt_enhancer.py
doesn't need a giant LLM block in the middle of it — and so swapping
backends in the future is one focused module rather than scattered
edits.

Public API (matches the historical _call_llm / _call_llm_progress shape
so callsites in prompt_enhancer.py change minimally):

    call_llm(prompt, system_prompt, temperature, *, seed=-1,
             num_predict=1024, json_schema=None, cancel_flag=None) -> str
    stream_llm(prompt, system_prompt, temperature, *, seed=-1,
               num_predict=1024, cancel_flag=None) -> generator
    multi_sample_prose(user_msg, sp, temperature, seed, n_samples,
                       num_predict, picker_sp) -> (picked, samples, idx)
    get_llm_status() -> HTML string

Errors:
    TruncatedOutput — raised by call_llm when num_predict caps mid-output.
                      Carries .partial. Re-exported from llm_runner.
"""

from .calls import (
    call_llm,
    stream_llm,
    multi_sample_prose,
    get_llm_status,
    cancel_flag,
)
from .calls import TruncatedOutput, LLMError, ModelLoadError

__all__ = [
    "call_llm",
    "stream_llm",
    "multi_sample_prose",
    "get_llm_status",
    "cancel_flag",
    "TruncatedOutput",
    "LLMError",
    "ModelLoadError",
]
