"""Tag validation + post-processing — see validate.py for details.

prompt_enhancer.py imports `validate.py`'s functions and threads the
extension's tag-format config / DB / aliases through. Pure-functional
layer; no module state.
"""

from .validate import (
    TAG_CORRECTIONS,
    SUBJECT_TAGS,
    PRESERVE_UNDERSCORE_RE,
    find_closest_tag,
    format_tag_out,
    validate,
    reorder,
    postprocess,
)

__all__ = [
    "TAG_CORRECTIONS",
    "SUBJECT_TAGS",
    "PRESERVE_UNDERSCORE_RE",
    "find_closest_tag",
    "format_tag_out",
    "validate",
    "reorder",
    "postprocess",
]
