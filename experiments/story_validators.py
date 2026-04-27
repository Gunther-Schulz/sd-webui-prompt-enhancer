"""Schema validators for story-mode variant outputs.

Each validator is a pure function: takes raw LLM output text, returns a
canonical dict, or raises ValidationError with a specific reason.

CLAUDE.md: fail-loud, no silent fallbacks. A malformed plan is recorded
in the trace with the exact failure reason; we do NOT auto-coerce.

Canonical plan shape (the target after parsing, regardless of variant):

    {
      "roles": [{"id": str, "description": str}, ...],
      "style_anchor": str,
      "panels": [
        {
          "n": int,
          "mode": "t2i" | "edit",
          "roles_present": [role_id, ...],
          "ref_panels": [int, ...],
          "caption": str,
          # one of these depending on mode (absent for terse plans):
          "t2i_prompt": str?,
          "edit_prompt": str?,
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import yaml


class ValidationError(ValueError):
    """Raised when LLM output doesn't match the expected schema. The
    message describes the specific failure for the trace."""


# ── shared helpers ─────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Strip ```yaml / ```json / ``` fences if the LLM ignored the
    'no markdown fences' rule. Common 9B failure mode."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the first fence line
        first_nl = t.find("\n")
        if first_nl == -1:
            raise ValidationError("output is just a code fence with no content")
        t = t[first_nl + 1 :]
        # Drop trailing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    return t.strip()


def _check_role_id(role_id: Any, defined_ids: set, panel_n: int) -> None:
    if not isinstance(role_id, str):
        raise ValidationError(
            f"panel {panel_n}: roles_present contains non-string {role_id!r}"
        )
    if role_id not in defined_ids:
        raise ValidationError(
            f"panel {panel_n}: roles_present references undefined role "
            f"{role_id!r} (defined: {sorted(defined_ids)})"
        )


def _check_panel_common(panel: Dict[str, Any], idx: int, defined_role_ids: set,
                        total_panels: int) -> None:
    """Shared per-panel sanity checks. idx is 0-based; panel['n'] is 1-based."""
    expected_n = idx + 1
    n = panel.get("n")
    if not isinstance(n, int):
        raise ValidationError(f"panel index {idx}: 'n' must be int, got {n!r}")
    if n != expected_n:
        raise ValidationError(
            f"panel index {idx}: 'n' is {n} but should be {expected_n} "
            f"(panels must be in order, 1..N)"
        )
    mode = panel.get("mode")
    if mode not in ("t2i", "edit"):
        raise ValidationError(
            f"panel {n}: 'mode' must be 't2i' or 'edit', got {mode!r}"
        )
    if n == 1 and mode != "t2i":
        raise ValidationError(f"panel 1: must be mode 't2i', got {mode!r}")
    if n > 1 and mode != "edit":
        raise ValidationError(
            f"panel {n}: must be mode 'edit' (only panel 1 is t2i), got {mode!r}"
        )

    roles_present = panel.get("roles_present")
    if not isinstance(roles_present, list):
        raise ValidationError(
            f"panel {n}: 'roles_present' must be a list, got {type(roles_present).__name__}"
        )
    for rp in roles_present:
        _check_role_id(rp, defined_role_ids, n)

    ref_panels = panel.get("ref_panels")
    if not isinstance(ref_panels, list):
        raise ValidationError(
            f"panel {n}: 'ref_panels' must be a list, got {type(ref_panels).__name__}"
        )
    for rp in ref_panels:
        if not isinstance(rp, int):
            raise ValidationError(
                f"panel {n}: ref_panels contains non-int {rp!r}"
            )
        if rp < 1 or rp >= n:
            raise ValidationError(
                f"panel {n}: ref_panels {rp} out of range "
                f"(must be 1..{n-1}, you can only reference earlier panels)"
            )
    if n == 1 and ref_panels:
        raise ValidationError(
            f"panel 1: ref_panels must be empty (no earlier panels exist), got {ref_panels!r}"
        )

    caption = panel.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValidationError(f"panel {n}: 'caption' missing or empty")


def _check_roles_block(roles: Any) -> set:
    if not isinstance(roles, list):
        raise ValidationError(f"'roles' must be a list, got {type(roles).__name__}")
    ids: set = set()
    for i, role in enumerate(roles):
        if not isinstance(role, dict):
            raise ValidationError(f"roles[{i}] must be a dict")
        rid = role.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValidationError(f"roles[{i}]: 'id' missing or empty")
        if rid in ids:
            raise ValidationError(f"roles[{i}]: duplicate id {rid!r}")
        desc = role.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ValidationError(f"role {rid!r}: 'description' missing or empty")
        ids.add(rid)
    return ids


def _check_style_anchor(style_anchor: Any) -> None:
    if not isinstance(style_anchor, str) or not style_anchor.strip():
        raise ValidationError("'style_anchor' missing or empty")


# ── full plan parsers (V1 / V2 — one-pass, includes panel prompts) ──────


def _check_full_panel(panel: Dict[str, Any], idx: int, defined_role_ids: set,
                      total_panels: int) -> None:
    _check_panel_common(panel, idx, defined_role_ids, total_panels)
    n = panel["n"]
    mode = panel["mode"]
    if mode == "t2i":
        if "t2i_prompt" not in panel or not str(panel["t2i_prompt"]).strip():
            raise ValidationError(f"panel {n} (t2i): 't2i_prompt' missing or empty")
        if panel.get("edit_prompt"):
            raise ValidationError(f"panel {n} (t2i): 'edit_prompt' must not be set")
    else:  # edit
        if "edit_prompt" not in panel or not str(panel["edit_prompt"]).strip():
            raise ValidationError(f"panel {n} (edit): 'edit_prompt' missing or empty")
        if panel.get("t2i_prompt"):
            raise ValidationError(f"panel {n} (edit): 't2i_prompt' must not be set")


def _check_full_plan(data: Any, expected_panel_count: int) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError(f"top level must be a mapping/object, got {type(data).__name__}")
    role_ids = _check_roles_block(data.get("roles"))
    _check_style_anchor(data.get("style_anchor"))
    panels = data.get("panels")
    if not isinstance(panels, list):
        raise ValidationError(f"'panels' must be a list, got {type(panels).__name__}")
    if len(panels) != expected_panel_count:
        raise ValidationError(
            f"panel count mismatch: got {len(panels)}, expected {expected_panel_count}"
        )
    for i, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise ValidationError(f"panels[{i}] must be a dict")
        _check_full_panel(panel, i, role_ids, len(panels))
    return data


def parse_full_plan_yaml(text: str, expected_panel_count: int) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as e:
        raise ValidationError(f"YAML parse error: {e}") from e
    return _check_full_plan(data, expected_panel_count)


def parse_full_plan_json(text: str, expected_panel_count: int) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValidationError(f"JSON parse error: {e}") from e
    return _check_full_plan(data, expected_panel_count)


# ── plan-only parsers (V3a / V3b — pass 1, no prompts yet) ──────────────


def _check_terse_panel(panel: Dict[str, Any], idx: int, defined_role_ids: set,
                       total_panels: int) -> None:
    _check_panel_common(panel, idx, defined_role_ids, total_panels)
    # Terse plans must NOT have prompt fields — those come from pass 2 or
    # template assembly.
    if panel.get("t2i_prompt") or panel.get("edit_prompt"):
        raise ValidationError(
            f"panel {panel['n']}: terse plan must not contain prompt fields "
            f"(found t2i_prompt/edit_prompt)"
        )


def _check_terse_plan(data: Any, expected_panel_count: int) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError(f"top level must be a mapping/object, got {type(data).__name__}")
    role_ids = _check_roles_block(data.get("roles"))
    _check_style_anchor(data.get("style_anchor"))
    panels = data.get("panels")
    if not isinstance(panels, list):
        raise ValidationError(f"'panels' must be a list, got {type(panels).__name__}")
    if len(panels) != expected_panel_count:
        raise ValidationError(
            f"panel count mismatch: got {len(panels)}, expected {expected_panel_count}"
        )
    for i, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise ValidationError(f"panels[{i}] must be a dict")
        _check_terse_panel(panel, i, role_ids, len(panels))
    return data


def parse_plan_only_yaml(text: str, expected_panel_count: int) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as e:
        raise ValidationError(f"YAML parse error: {e}") from e
    return _check_terse_plan(data, expected_panel_count)


def parse_plan_only_json(text: str, expected_panel_count: int) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValidationError(f"JSON parse error: {e}") from e
    return _check_terse_plan(data, expected_panel_count)


# Alias for V5 — same schema as plan-only
parse_terse_plan_yaml = parse_plan_only_yaml


# ── per-panel prompt parser (Pass 2 of two-pass variants) ───────────────


def parse_panel_prompt_text(text: str, **_kwargs) -> str:
    """Pass 2 of two-pass variants outputs free text — the panel's full
    prompt. Validate only that it isn't empty and isn't obviously
    wrapped in commentary the model was told not to produce."""
    cleaned = _strip_fences(text).strip()
    if not cleaned:
        raise ValidationError("panel prompt is empty")

    # Catch common 9B-abliterated leakage patterns
    bad_prefixes = (
        "prompt:", "panel prompt:", "here is", "here's", "okay,",
        "sure,", "alright,", "the prompt is",
    )
    lower = cleaned.lower()
    for bad in bad_prefixes:
        if lower.startswith(bad):
            raise ValidationError(
                f"panel prompt starts with leaked label/preamble {bad!r}: "
                f"{cleaned[: len(bad) + 30]!r}"
            )
    if len(cleaned.split()) < 8:
        raise ValidationError(
            f"panel prompt suspiciously short ({len(cleaned.split())} words): {cleaned!r}"
        )
    return cleaned


# ── registry — maps validator names from variant YAML to functions ──────


VALIDATORS = {
    "parse_full_plan_yaml": parse_full_plan_yaml,
    "parse_full_plan_json": parse_full_plan_json,
    "parse_plan_only_yaml": parse_plan_only_yaml,
    "parse_plan_only_json": parse_plan_only_json,
    "parse_terse_plan_yaml": parse_terse_plan_yaml,
    "parse_panel_prompt_text": parse_panel_prompt_text,
}


def get_validator(name: str):
    if name not in VALIDATORS:
        raise SystemExit(
            f"Unknown validator {name!r}. "
            f"Available: {sorted(VALIDATORS.keys())}"
        )
    return VALIDATORS[name]
