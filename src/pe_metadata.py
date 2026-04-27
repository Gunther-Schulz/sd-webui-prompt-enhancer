"""Image-metadata round-trip for the prompt-enhancer extension.

When the user generates an image, Forge calls our `process` method
with the values of every UI control. We write `PE Source`, `PE Base`,
`PE Modifiers`, etc. into `p.extra_generation_params` so the values
land in the saved PNG's metadata.

When the user later loads that image back into the WebUI, Forge
reads the metadata + dispatches each `(component, restore_fn)` pair
in `infotext_fields` so each control gets its old value back. This
module centralizes the restore functions + the field list and the
process-side write logic.

Public API:
  PASTE_FIELD_NAMES                     param names we own
  build_restore_funcs(tag_formats, all_modifiers, dropdown_choices,
                      gradio_module)
                                        → SimpleNamespace with the
                                          per-control restore callbacks
  build_infotext_fields(components, restore_funcs)
                                        → [(component, restore_fn|key)]
                                          list for self.infotext_fields
  apply_to_extra_generation_params(p, source_prompt, base,
                                   custom_system_prompt, *args,
                                   last_seed, last_pe_mode)
                                        → writes PE_* keys onto
                                          p.extra_generation_params
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple


# Names of the metadata params this extension owns. Forge's metadata
# parser uses this list to decide which keys to dispatch through our
# infotext_fields restore callbacks.
PASTE_FIELD_NAMES: List[str] = [
    "PE Source", "PE Base", "PE Detail", "PE Modifiers",
    "PE Think", "PE Seed", "PE Tag Format", "PE Tag Validation",
    "PE Temperature", "PE Prepend", "PE Motion", "PE Mode",
]


# ── Restore functions ───────────────────────────────────────────────────


def build_restore_funcs(
    tag_formats: Dict[str, Any],
    all_modifiers: Dict[str, Any],
    dropdown_choices: Dict[str, list],
    gradio_module: Any,
) -> SimpleNamespace:
    """Construct the restore callbacks used by infotext_fields.

    Each callback takes the metadata dict and returns either the
    restored value or `gradio_module.update()` (no-change sentinel).
    """
    gr = gradio_module

    def parse_modifiers(params):
        raw = params.get("PE Modifiers", "")
        return {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

    def make_dd_restore(dd_label):
        dd_choices = dropdown_choices.get(dd_label, [])
        def restore(params):
            saved = parse_modifiers(params)
            return [m for m in saved if m in dd_choices and m in all_modifiers]
        return restore

    def restore_tag_format(params):
        val = params.get("PE Tag Format", "")
        return val if val in tag_formats else gr.update()

    def restore_tag_validation(params):
        val = params.get("PE Tag Validation", "")
        return val if val in ("RAG", "Fuzzy Strict", "Fuzzy", "Off") else gr.update()

    def restore_temperature(params):
        raw = params.get("PE Temperature", "")
        if not raw:
            return gr.update()
        try:
            return max(0.0, min(2.0, float(raw)))
        except (TypeError, ValueError):
            return gr.update()

    def restore_detail(params):
        return min(10, max(0, int(params.get("PE Detail", 0)))) if params.get("PE Detail") else 0

    def restore_seed(params):
        return int(params.get("PE Seed", -1)) if params.get("PE Seed") else -1

    def restore_bool(key):
        def restore(params):
            return params.get(key, "").lower() == "true"
        return restore

    return SimpleNamespace(
        parse_modifiers=parse_modifiers,
        make_dd_restore=make_dd_restore,
        tag_format=restore_tag_format,
        tag_validation=restore_tag_validation,
        temperature=restore_temperature,
        detail=restore_detail,
        seed=restore_seed,
        prepend=restore_bool("PE Prepend"),
        motion=restore_bool("PE Motion"),
    )


def build_infotext_fields(
    components: SimpleNamespace,
    restore: SimpleNamespace,
    dd_labels: Iterable[str],
) -> List[Tuple[Any, Any]]:
    """Construct the `self.infotext_fields` list.

    `components` carries the Gradio component variables (source_prompt,
    base, detail_level, …, dd_components). `restore` is the namespace
    returned by build_restore_funcs.

    The non-modifier controls match by string key (Forge looks up the
    metadata value directly) or by callable (Forge passes the whole
    metadata dict). Per-modifier dropdowns each get a custom restore
    that filters the saved CSV against that dropdown's valid choices.
    """
    fields: List[Tuple[Any, Any]] = [
        (components.source_prompt, "PE Source"),
        (components.base, "PE Base"),
        (components.detail_level, restore.detail),
        (components.think, "PE Think"),
        (components.seed, restore.seed),
        (components.tag_format, restore.tag_format),
        (components.tag_validation, restore.tag_validation),
        (components.temperature, restore.temperature),
        (components.prepend_source, restore.prepend),
        (components.motion_cb, restore.motion),
    ]
    for i, label in enumerate(dd_labels):
        fields.append((components.dd_components[i], restore.make_dd_restore(label)))
    return fields


# ── process() side: write to extra_generation_params ────────────────────


def apply_to_extra_generation_params(
    p: Any,
    *,
    source_prompt: str,
    base: str,
    detail_level: Any,
    dd_vals: Iterable[Iterable[str]],
    think: bool,
    neg_cb: bool,
    last_seed: int,
    tag_format: str,
    tag_validation: str,
    temperature: Any,
    prepend: bool,
    motion: bool,
    last_pe_mode: Any,
) -> None:
    """Write the PE_* keys onto p.extra_generation_params for the
    saved PNG's metadata. Each key only writes when the value is set
    (truthy) so unused fields stay out of the metadata."""
    if source_prompt:
        p.extra_generation_params["PE Source"] = source_prompt
    if base:
        p.extra_generation_params["PE Base"] = base
    if detail_level and int(detail_level) != 0:
        p.extra_generation_params["PE Detail"] = int(detail_level)

    all_mod_names = []
    for dd_val in dd_vals:
        if dd_val:
            all_mod_names.extend(dd_val)
    if all_mod_names:
        p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)

    if think:
        p.extra_generation_params["PE Think"] = True
    if neg_cb:
        p.extra_generation_params["PE Negative"] = True
    if last_seed >= 0:
        p.extra_generation_params["PE Seed"] = last_seed
    if tag_format:
        p.extra_generation_params["PE Tag Format"] = tag_format
    if tag_validation:
        p.extra_generation_params["PE Tag Validation"] = tag_validation
    if temperature is not None:
        p.extra_generation_params["PE Temperature"] = round(float(temperature), 3)
    if prepend:
        p.extra_generation_params["PE Prepend"] = True
    if motion:
        p.extra_generation_params["PE Motion"] = True
    if last_pe_mode:
        p.extra_generation_params["PE Mode"] = last_pe_mode
