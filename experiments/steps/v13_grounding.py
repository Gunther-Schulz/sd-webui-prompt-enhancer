"""Pass 2 grounding step for V13.

Dispatches to pe._general_tag_candidates + pe._candidate_fragment_for_tag_sp
so the experiment pipeline grounds Pass 2 exactly the way the shipped
extension does. The tag_extract step reads the grounded fragment from
state and prepends it to the tag_sp before the LLM call.

Root-cause fix for user-reported weird tags (sparse_leg_hair,
overhead_lights, simple_fish on unrelated scenes): the LLM no longer
freely invents Danbooru-style compound tokens — it selects from a
pre-retrieved scene-relevant candidate list.
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.steps.common import get_stack, pe


def build_pass2_candidates(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: prose. Write: pass2_candidates (list[str]), pass2_grounding_fragment (str).

    FAISS-retrieves top-K general-category Danbooru tags matching the
    prose. The fragment is what tag_extract step prepends to its
    tag_sp. Passing k/min_post_count lets variants titrate them.
    """
    prose = state.get("prose", "") or ""
    if not prose:
        return {**state, "pass2_candidates": [], "pass2_grounding_fragment": ""}

    k = int(params.get("k", 60))
    min_pc = int(params.get("min_post_count", 100))
    stack = get_stack()
    candidates = pe._general_tag_candidates(stack, prose, k=k, min_post_count=min_pc)
    frag = pe._candidate_fragment_for_tag_sp(candidates)
    return {
        **state,
        "pass2_candidates": candidates,
        "pass2_grounding_fragment": frag,
    }
