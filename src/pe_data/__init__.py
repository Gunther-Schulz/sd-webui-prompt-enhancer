"""YAML/JSON config loaders for the prompt-enhancer extension.

Submodules cover one config concern each — bases here, more to come
as we extract them from scripts/prompt_enhancer.py:

  pe_data.bases    — base prompt definitions (Default / Detailed /
                     Cinematic / Custom / etc.)

The shared YAML/JSON loader lives in ._util so all submodules use the
same parser + error handling.
"""

from . import bases
from ._util import load_yaml_or_json, get_local_dirs

__all__ = ["bases", "load_yaml_or_json", "get_local_dirs"]
