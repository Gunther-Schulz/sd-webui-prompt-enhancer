"""Step functions that reproduce the extension's current (baseline)
Hybrid-style tagging pipeline.

Each step reads/writes specific keys in `state`. The docstring declares
the contract. V1 is the current pipeline shape wrapped into the
experiment runner so we can measure it before designing V2.

Pipeline shape reproduced here:

    source + modifiers + seed
        │
        ▼ assemble_sp          base → prose system prompt
        ▼ build_shortlist      retrieval (artist/character/series candidates)
        ▼ inject_shortlist     inject shortlist fragment into SP
        ▼ prose                LLM call → rich prose
        ▼ tag_extract          LLM call → tag draft (comma-separated)
        ▼ validate             validator + rule_layer + slot_fill
        ▼ → state['final_tags']

This matches the Forge extension's _hybrid and _tags handlers after
the Tags refactor in this session. Any behavior difference between
running V1 in the experiment runner and running Tags/Hybrid in Forge
is a contract violation that must be fixed.
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.steps.common import call_llm, get_stack, pe


# ── Step: assemble_sp ────────────────────────────────────────────────


def assemble_sp(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: base_name (str), custom_sp (Optional[str]).
    Write: sp (str).

    Uses the real `pe._assemble_system_prompt` so the SP we build here
    is bit-identical to what the extension builds with the same inputs.

    Optional `prose_directive` param: an additional instruction block
    appended AFTER the base's own body. Used by variants (e.g. V2) to
    change prose behavior without editing bases.yaml — keeps one-factor
    experiments clean.
    """
    base_name = params.get("base_name") or state.get("base_name") or "Detailed"
    custom_sp = params.get("custom_sp") or state.get("custom_sp")
    sp = pe._assemble_system_prompt(base_name, custom_sp, 0)
    if not sp:
        raise ValueError(f"No system prompt assembled for base={base_name!r}")
    prose_directive = params.get("prose_directive")
    if prose_directive:
        sp = f"{sp}\n\n{prose_directive}"
    return {**state, "sp": sp, "base_name": base_name}


# ── Step: build_shortlist ────────────────────────────────────────────


def build_shortlist(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: source (str), mods (list of (name,entry)), seed (int).
    Write: shortlist (Shortlist), style_prose (str).

    Uses the real retriever. Query expansion is on by default (toggle
    via params['query_expansion']=False). The expander LLM call uses
    temperature=0.3 to match extension behavior.
    """
    source = state["source"]
    mods = state.get("mods", [])
    seed = state.get("seed", -1)

    style_prose = pe._build_style_string(mods, mode="prose")
    expander = None
    if params.get("query_expansion", True):
        from anima_tagger.query_expansion import expand_query

        def _oneshot(sys_prompt: str, user_msg: str) -> str:
            return call_llm(
                sys_prompt, user_msg,
                seed=seed, temperature=0.3, num_predict=256,
                model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
            )

        def _expander(src, mk):
            return expand_query(src, _oneshot, modifier_keywords=mk)
        expander = _expander

    stack = get_stack()
    sl = stack.build_shortlist(
        source_prompt=source,
        modifier_keywords=style_prose,
        query_expander=expander,
    )
    return {**state, "shortlist": sl, "style_prose": style_prose}


# ── Step: inject_shortlist ───────────────────────────────────────────


def inject_shortlist(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: sp (str), shortlist.
    Write: sp (replaced — with shortlist fragment appended).

    Mirrors the extension's injection of `sl.as_system_prompt_fragment()`
    into the prose system prompt.
    """
    sp = state["sp"]
    sl = state["shortlist"]
    frag = sl.as_system_prompt_fragment()
    if frag:
        sp = f"{sp}\n\n{frag}"
    return {**state, "sp": sp}


# ── Step: prose ──────────────────────────────────────────────────────


def prose(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: sp (str), source (str), style_prose (str), seed (int).
    Write: prose (str).

    Composes user_msg same as extension: "SOURCE PROMPT: <source>" (or
    the empty-source signal) + style block if modifiers are active.
    """
    sp = state["sp"]
    source = state["source"]
    style_prose = state.get("style_prose", "")
    seed = state.get("seed", -1)

    user_msg = f"SOURCE PROMPT: {source}" if source else pe._EMPTY_SOURCE_SIGNAL
    if style_prose:
        user_msg = f"{user_msg}\n\n{style_prose}"

    prose_text = call_llm(
        sp, user_msg,
        seed=seed,
        temperature=params.get("temperature", 0.8),
        num_predict=params.get("num_predict", 1024),
        model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
    )
    return {**state, "prose": prose_text, "user_msg": user_msg}


# ── Step: tag_extract ────────────────────────────────────────────────


def tag_extract(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: prose (str), tag_fmt (str), style_prose (str), seed (int).
    Write: draft (str).

    Uses the anima.yaml system_prompt + style directive block, same
    assembly as the extension.
    """
    tag_fmt = params.get("tag_fmt") or state.get("tag_fmt") or "Anima"
    fmt_config = pe._tag_formats.get(tag_fmt)
    if not fmt_config:
        raise ValueError(f"Unknown tag_fmt={tag_fmt!r}. Available: {list(pe._tag_formats.keys())}")
    tag_sp = fmt_config.get("system_prompt", "")
    if not tag_sp:
        raise ValueError(f"tag-formats/{tag_fmt.lower()}.yaml has no system_prompt")

    style_prose = state.get("style_prose", "")
    if style_prose:
        tag_sp = (
            f"{tag_sp}\n\nThe following style directives were requested. "
            f"Ensure they are reflected in the tags:\n{style_prose}"
        )

    draft = call_llm(
        tag_sp, state["prose"],
        seed=state.get("seed", -1),
        temperature=params.get("temperature", 0.8),
        num_predict=params.get("num_predict", 1024),
        model=params.get("model") or "huihui_ai/qwen3.5-abliterated:9b",
    )
    return {**state, "draft": draft, "tag_fmt": tag_fmt}


# ── Step: validate ───────────────────────────────────────────────────


def validate(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: draft (str), shortlist, source (str), mods (list).
    Write: tags_after_validate (str), stats (dict).

    Calls the real `pe._anima_tag_from_draft` — bit-identical to the
    extension's validator+rule_layer output for the same inputs.
    """
    stack = get_stack()
    safety = pe._anima_safety_from_modifiers(state.get("mods", []), state.get("source", ""))
    tags_str, stats = pe._anima_tag_from_draft(
        stack,
        state["draft"],
        safety=safety,
        shortlist=state.get("shortlist"),
    )
    return {**state, "tags_after_validate": tags_str, "stats": stats, "anima_safety": safety}


# ── Step: slot_fill ──────────────────────────────────────────────────


def slot_fill(state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Read: tags_after_validate, mods, prose, seed.
    Write: final_tags (str).

    Reproduces the extension's slot-fill pass: for each active
    modifier with a target_slot (e.g. Random Artist → artist), if no
    tag of that category survived validation, retrieve top-1 from
    prose and inject. Uses seed to pick among top-K for variance.
    """
    if not params.get("enabled", True):
        return {**state, "final_tags": state["tags_after_validate"]}

    stack = get_stack()
    tags_str = state["tags_after_validate"]
    mods = state.get("mods", [])
    slots = pe._active_target_slots(mods)
    slot_fill_trace = {}
    for slot in slots:
        cat_info = pe._SLOT_TO_CATEGORY.get(slot)
        if not cat_info:
            slot_fill_trace[slot] = "no_cat_mapping"
            continue
        if pe._tags_have_category(tags_str, stack, cat_info["category"]):
            slot_fill_trace[slot] = "already_present"
            continue
        picked = pe._retrieve_prose_slot(
            stack, state["prose"], slot, seed=state.get("seed", -1),
        )
        if not picked:
            slot_fill_trace[slot] = "no_candidate"
            continue
        tag_out = picked.replace("_", " ")
        if slot == "artist":
            tag_out = "@" + tag_out
        tags_str = f"{tags_str}, {tag_out}" if tags_str else tag_out
        slot_fill_trace[slot] = picked

    return {**state, "final_tags": tags_str, "slot_fill_trace": slot_fill_trace}
