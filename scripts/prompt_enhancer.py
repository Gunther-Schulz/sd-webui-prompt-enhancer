import json
import logging
import os
import re
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory ─────────────────────────────────────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"

# Mode keywords (hardcoded, rendered as checkboxes)
MODE_KEYWORDS = {
    "Still": "frozen moment, static composition, describe a single instant in time, no temporal language, no movement, captured pose",
    "Scene": "chronological action, present-progressive verbs, temporal flow, describe movement unfolding over time, use connectors like as then while, no timestamps or scene cuts",
    "Audio": "include audio descriptions integrated chronologically with the visuals, describe specific sounds not vague atmosphere, ambient sounds and sound effects, be concrete about what is heard and when",
}

# UI group definitions: group_key -> list of category names that belong in it
UI_GROUPS = {
    "subject": ["genre", "subject", "activity", "relationship"],
    "setting": ["setting", "time period", "aesthetic"],
    "lighting_mood": ["lighting", "mood", "atmosphere", "emotion"],
    "visual_style": ["color", "art style", "anime", "cinema style", "photography format", "vintage format"],
    "camera": ["perspective", "distance", "focus", "technique", "motion", "material"],
    "audio": ["audio"],
}

# Human-readable labels for UI groups
UI_GROUP_LABELS = {
    "subject": "Subject",
    "setting": "Setting",
    "lighting_mood": "Lighting & Mood",
    "visual_style": "Visual Style",
    "camera": "Camera",
    "audio": "Audio",
}


# ── File loading helpers ─────────────────────────────────────────────────────

def _load_file(path):
    """Load a JSON or YAML file and return its parsed content as a dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        logger.error(f"{path} must be a mapping")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return {}


def _get_local_dir(ui_path=""):
    """Resolve the local overrides directory."""
    path = (ui_path or "").strip()
    if not path:
        path = os.environ.get("PROMPT_ENHANCER_LOCAL", "").strip()
    if path and os.path.isdir(path):
        return path
    return None


def _load_local_bases(local_dir):
    """Load _bases.yaml from the local directory."""
    if not local_dir:
        return {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(local_dir, BASES_FILENAME + ext)
        if os.path.isfile(path):
            return {k: v for k, v in _load_file(path).items() if isinstance(v, str)}
    return {}


def _load_local_modifiers(local_dir):
    """Load all files except _bases.* from the local directory.

    Local files use the same three-level structure (group > category > modifiers).
    Returns merged three-level dict.
    """
    if not local_dir:
        return {}
    merged = {}
    for name in sorted(os.listdir(local_dir)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem == BASES_FILENAME:
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(local_dir, name))
        if data:
            # Merge at group level, then category level
            for group, categories in data.items():
                if not isinstance(categories, dict):
                    continue
                if group not in merged:
                    merged[group] = {}
                for cat, items in categories.items():
                    if not isinstance(items, dict):
                        continue
                    if cat not in merged[group]:
                        merged[group][cat] = {}
                    merged[group][cat].update(items)
    return merged


def _merge_three_level(base, override):
    """Merge two three-level dicts (group > category > modifiers)."""
    merged = {}
    for group, categories in base.items():
        if isinstance(categories, dict):
            merged[group] = {}
            for cat, items in categories.items():
                if isinstance(items, dict):
                    merged[group][cat] = dict(items)
    for group, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if group not in merged:
            merged[group] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[group]:
                merged[group][cat] = {}
            merged[group][cat].update(items)
    return merged


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}       # flat: name -> keywords (for lookup)
_group_choices = {}       # group_key -> [choice_list with separators]
_wildcards = {}
_wildcard_choices = []


def _build_group_choices(mod_data):
    """Build per-group choice lists from three-level modifier data.

    Also builds the flat lookup dict.
    Returns (flat_dict, group_choices_dict).
    """
    flat = {}
    group_choices = {gk: [] for gk in UI_GROUPS}

    # Build a reverse map: category_name -> group_key
    cat_to_group = {}
    for gk, cats in UI_GROUPS.items():
        for cat in cats:
            cat_to_group[cat] = gk

    # Walk the three-level data
    for group_name, categories in mod_data.items():
        if not isinstance(categories, dict):
            continue
        for cat_name, items in categories.items():
            if not isinstance(items, dict):
                continue
            # Determine which UI group this category belongs to
            target_group = cat_to_group.get(cat_name)
            if not target_group:
                # Unknown category from local override — find by group name
                target_group = group_name if group_name in UI_GROUPS else None
            if not target_group:
                # Still unknown — put in a catch-all
                if "other" not in group_choices:
                    group_choices["other"] = []
                target_group = "other"

            separator = f"\u2500\u2500\u2500\u2500\u2500 {cat_name.title()} \u2500\u2500\u2500\u2500\u2500"
            group_choices[target_group].append(separator)
            for name, keywords in items.items():
                if isinstance(keywords, str):
                    flat[name] = keywords
                    group_choices[target_group].append(name)

    # Remove empty groups
    group_choices = {k: v for k, v in group_choices.items() if v}

    return flat, group_choices


def _build_wildcard_choices(wc_data):
    """Build flat dict and choice list from two-level wildcard data."""
    flat = {}
    choices = []
    for category, items in wc_data.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {category.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, prompt in items.items():
            if isinstance(prompt, str):
                flat[name] = prompt
                choices.append(name)
    return flat, choices


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _group_choices, _wildcards, _wildcard_choices

    local_dir = _get_local_dir(local_dir_path)

    # Bases
    _bases = {k: v for k, v in _load_file(os.path.join(_EXT_DIR, "bases.json")).items() if isinstance(v, str)}
    _bases.update(_load_local_bases(local_dir))

    # Modifiers (three-level YAML)
    mod_data = _load_file(os.path.join(_EXT_DIR, "modifiers.yaml"))
    # Remove "mode" from modifiers — it's handled as checkboxes
    mod_data.pop("mode", None)
    local_mods = _load_local_modifiers(local_dir)
    if local_mods:
        mod_data = _merge_three_level(mod_data, local_mods)
    _all_modifiers, _group_choices = _build_group_choices(mod_data)

    # Wildcards (two-level)
    wc_data = _load_file(os.path.join(_EXT_DIR, "wildcards.json"))
    _wildcards, _wildcard_choices = _build_wildcard_choices(wc_data)


_reload_all()


# ── Constants ────────────────────────────────────────────────────────────────

REFINE_SYSTEM_PROMPT = (
    "You are a prompt editor. You are given an already-complete image/video generation prompt. "
    "Do NOT rewrite, expand, or re-enhance it. Your only job is to integrate the requested "
    "style changes into the existing text.\n\n"
    "Rules:\n"
    "- If a style adds something new (mood, lighting, color), weave it into the existing "
    "description naturally.\n"
    "- If a style conflicts with something already present (different location, different "
    "time of day, different mood), replace the old element with the new one.\n"
    "- If a wildcard asks you to choose something (location, wardrobe, etc.) and one already "
    "exists in the prompt, replace it with your creative choice.\n"
    "- Preserve the original structure, flow, and approximate length.\n"
    "- Include the style keywords verbatim in the output.\n"
    "- Output the modified prompt only. No commentary."
)

TAGS_SYSTEM_PROMPTS = {
    "Illustrious": (
        "You are a danbooru tag expert. Convert the user's scene description into danbooru-style tags.\n\n"
        "Rules:\n"
        "- Output a comma-separated list of tags, nothing else\n"
        "- Use spaces between words in multi-word tags (long hair, blue eyes)\n"
        "- Start with quality tags: masterpiece, best quality, absurdres\n"
        "- Then subject count (1girl, 1boy, 1other, no humans, etc.)\n"
        "- Then appearance, clothing, pose, expression, setting, composition\n"
        "- Use the user's key descriptive words as tags\n"
        "- Include rating tag at end (rating:general, rating:sensitive, rating:questionable, rating:explicit)\n"
        "- When style directions are provided, include relevant tags for them"
    ),
    "NoobAI": (
        "You are a booru tag expert. Convert the user's scene description into booru-style tags for NoobAI XL.\n\n"
        "Rules:\n"
        "- Output a comma-separated list of tags, nothing else\n"
        "- Use spaces in multi-word tags (long hair, blue eyes) - NO underscores\n"
        "- Start with quality tags: masterpiece, best quality, very awa\n"
        "- Then subject count (1girl, 1boy, 1other, no humans, etc.)\n"
        "- Then appearance, clothing, pose, expression, setting, composition\n"
        "- Use the user's key descriptive words as tags\n"
        "- Include rating tag at end (rating:general, rating:sensitive, rating:questionable, rating:explicit)\n"
        "- When style directions are provided, include relevant tags for them"
    ),
    "Pony": (
        "You are a booru tag expert. Convert the user's scene description into tags for Pony Diffusion V6 XL.\n\n"
        "Rules:\n"
        "- Output a comma-separated list of tags, nothing else\n"
        "- Use spaces between words in multi-word tags (long hair, blue eyes)\n"
        "- Start with score tags: score 9, score 8 up, score 7 up\n"
        "- Then source tag if applicable (source anime, source furry, source pony, source cartoon)\n"
        "- Then subject count (1girl, 1boy, 1other, no humans, etc.)\n"
        "- Then appearance, clothing, pose, expression, setting, composition\n"
        "- Use the user's key descriptive words as tags\n"
        "- Do NOT use quality words like masterpiece or best quality\n"
        "- Include rating tag at end (rating safe, rating questionable, rating explicit)\n"
        "- When style directions are provided, include relevant tags for them"
    ),
}

INLINE_WILDCARD_INSTRUCTION = (
    "The user's prompt contains placeholders in {name?} format. "
    "For each one, choose a specific, vivid option that creates a coherent scene. "
    "Replace the placeholder with your choice seamlessly."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_names():
    names = list(_bases.keys())
    names.append("Custom")
    return names


def _collect_all_modifiers(*dropdown_values, mode_still=False, mode_scene=False, mode_audio=False):
    """Collect all selected modifiers from mode checkboxes and all dropdowns into one list."""
    all_mods = []
    # Mode checkboxes
    if mode_still:
        all_mods.append(("Still", MODE_KEYWORDS["Still"]))
    if mode_scene:
        all_mods.append(("Scene", MODE_KEYWORDS["Scene"]))
    if mode_audio:
        all_mods.append(("Audio", MODE_KEYWORDS["Audio"]))
    # All dropdown selections
    for selections in dropdown_values:
        for name in (selections or []):
            keywords = _all_modifiers.get(name, "")
            if keywords:
                all_mods.append((name, keywords))
    return all_mods


def _build_style_string(mod_list):
    """Build the 'Apply these styles:' string from a list of (name, keywords) tuples."""
    style_parts = []
    for name, keywords in mod_list:
        if name.lower() not in keywords.lower():
            keywords = f"{name.lower()}, {keywords}"
        style_parts.append(keywords)
    if style_parts:
        return f"Apply these styles: {', '.join(style_parts)}."
    return ""


# ── Ollama ───────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _to_ollama_base(api_url):
    base = api_url
    for suffix in ("/v1/chat/completions", "/v1", "/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or DEFAULT_OLLAMA_BASE


def _fetch_ollama_models(api_url):
    try:
        base = _to_ollama_base(api_url)
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        models.sort()
        return models
    except Exception:
        return []


def _refresh_models(api_url, current_model):
    models = _fetch_ollama_models(api_url)
    if not models:
        return gr.update()
    value = current_model if current_model in models else models[0]
    return gr.update(choices=models, value=value)


# ── Core logic ───────────────────────────────────────────────────────────────

def _strip_think_blocks(text):
    return re.sub(r"<thi" + r"nk>[\s\S]*?</thi" + r"nk>", "", text).strip()


def _has_inline_wildcards(text):
    return bool(re.search(r"\{[^}]+\?\}", text))


def _clean_output(text):
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    return text.strip()


def _call_llm(prompt, api_url, model, system_prompt, temperature, think=False):
    base = _to_ollama_base(api_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": float(temperature)},
        "think": bool(think),
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base}/api/chat"
    last_err = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "")
            return _strip_think_blocks(content)
        except urllib.error.URLError as e:
            last_err = e
            if attempt == 0:
                import time
                time.sleep(2)
    raise last_err


def enhance_prompt(source, api_url, model, base_name, custom_system_prompt,
                   mode_still, mode_scene, mode_audio,
                   dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
                   wildcards, word_limit, think, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    if base_name == "Custom":
        system_prompt = (custom_system_prompt or "").strip()
    else:
        system_prompt = _bases.get(base_name, "")
    if not system_prompt:
        return "", "<span style='color:#c66'>No system prompt configured.</span>"

    all_mods = _collect_all_modifiers(
        dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
        mode_still=mode_still, mode_scene=mode_scene, mode_audio=mode_audio,
    )
    style_str = _build_style_string(all_mods)
    if style_str:
        system_prompt = f"{system_prompt}\n\n{style_str}"

    for wc_name in (wildcards or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    if _has_inline_wildcards(source):
        system_prompt = f"{system_prompt}\n\n{INLINE_WILDCARD_INSTRUCTION}"

    word_limit = int(word_limit)
    if word_limit > 0:
        system_prompt = f"{system_prompt}\n\nAim for around {word_limit} words."

    try:
        enhanced = _clean_output(_call_llm(source, api_url, model, system_prompt, temperature, think=think))
        word_count = len(enhanced.split())
        return enhanced, f"<span style='color:#6c6'>OK - enhanced to {word_count} words</span>"
    except urllib.error.URLError as e:
        msg = f"Connection failed: {e.reason} - is Ollama running?"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"


def refine_prompt(existing_prompt, source_prompt, api_url, model,
                  mode_still, mode_scene, mode_audio,
                  dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
                  wildcards, think, temperature):
    existing = (existing_prompt or "").strip()
    if not existing:
        return "", "<span style='color:#c66'>No prompt to refine. Generate one first with Enhance.</span>"

    source = (source_prompt or "").strip()
    all_mods = _collect_all_modifiers(
        dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
        mode_still=mode_still, mode_scene=mode_scene, mode_audio=mode_audio,
    )

    if not all_mods and not wildcards and not source:
        return "", "<span style='color:#c66'>Select modifiers/wildcards or update the source prompt.</span>"

    system_prompt = REFINE_SYSTEM_PROMPT

    if source:
        system_prompt = f"{system_prompt}\n\nThe user has provided updated subject/scene direction. Integrate these changes into the existing prompt, replacing conflicting elements:\n{source}"

    style_str = _build_style_string(all_mods)
    if style_str:
        system_prompt = f"{system_prompt}\n\n{style_str}"

    for wc_name in (wildcards or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    try:
        refined = _clean_output(_call_llm(existing, api_url, model, system_prompt, temperature, think=think))
        word_count = len(refined.split())
        return refined, f"<span style='color:#6c6'>OK - refined to {word_count} words</span>"
    except urllib.error.URLError as e:
        msg = f"Connection failed: {e.reason} - is Ollama running?"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"


def generate_tags(source, api_url, model, tag_format,
                  mode_still, mode_scene, mode_audio,
                  dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
                  wildcards, think, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    system_prompt = TAGS_SYSTEM_PROMPTS.get(tag_format, list(TAGS_SYSTEM_PROMPTS.values())[0])

    all_mods = _collect_all_modifiers(
        dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
        mode_still=mode_still, mode_scene=mode_scene, mode_audio=mode_audio,
    )
    style_str = _build_style_string(all_mods)
    if style_str:
        system_prompt = f"{system_prompt}\n\n{style_str}"

    for wc_name in (wildcards or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    try:
        tags = _clean_output(_call_llm(source, api_url, model, system_prompt, temperature, think=think))
        if tag_format in ("Illustrious", "Pony"):
            tags = ", ".join(t.strip().replace(" ", "_") for t in tags.split(",") if t.strip())
        tag_count = len([t for t in tags.split(",") if t.strip()])
        return tags, f"<span style='color:#6c6'>OK - {tag_count} tags</span>"
    except urllib.error.URLError as e:
        msg = f"Connection failed: {e.reason} - is Ollama running?"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"


# ── UI ───────────────────────────────────────────────────────────────────────

class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        initial_models = _fetch_ollama_models(DEFAULT_API_URL)
        if not initial_models:
            initial_models = [DEFAULT_MODEL]

        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt ──
            source_prompt = gr.Textbox(
                label="Source Prompt",
                lines=3,
                placeholder="Type your prompt here, then click Enhance... Use {name?} for inline wildcards.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(
                    value="\U0001f4a1 Enhance",
                    variant="primary",
                    scale=0, min_width=120,
                    elem_id=f"{tab}_pe_enhance_btn",
                )
                tags_btn = gr.Button(
                    value="\U0001f3f7 Tags",
                    variant="primary",
                    scale=0, min_width=100,
                    elem_id=f"{tab}_pe_tags_btn",
                )
                refine_btn = gr.Button(
                    value="\U0001f527 Refine",
                    scale=0, min_width=120,
                    elem_id=f"{tab}_pe_refine_btn",
                )
                grab_btn = gr.Button(
                    value="\u2b07 Grab",
                    scale=0, min_width=80,
                    elem_id=f"{tab}_pe_grab_btn",
                )
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            grab_btn.click(
                fn=lambda x: x,
                _js=f"""function(x) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    return ta ? [ta.value] : [x];
                }}""",
                inputs=[source_prompt],
                outputs=[source_prompt],
                show_progress=False,
            )

            # ── Mode checkboxes ──
            with gr.Row():
                mode_still = gr.Checkbox(label="Still", value=False, elem_id=f"{tab}_pe_mode_still")
                mode_scene = gr.Checkbox(label="Scene", value=False, elem_id=f"{tab}_pe_mode_scene")
                mode_audio = gr.Checkbox(label="Audio", value=False, elem_id=f"{tab}_pe_mode_audio")

            # ── Base + Tag Format ──
            with gr.Row():
                base = gr.Dropdown(
                    label="Base",
                    choices=_base_names(),
                    value="Default",
                    scale=2,
                )
                tag_format = gr.Dropdown(
                    label="Tag Format",
                    choices=list(TAGS_SYSTEM_PROMPTS.keys()),
                    value=list(TAGS_SYSTEM_PROMPTS.keys())[0],
                    scale=1,
                )

            # ── Modifier dropdowns (grouped) ──
            dd = {}
            with gr.Row():
                dd["subject"] = gr.Dropdown(
                    label="Subject",
                    choices=_group_choices.get("subject", []),
                    value=[], multiselect=True, scale=1,
                )
                dd["setting"] = gr.Dropdown(
                    label="Setting",
                    choices=_group_choices.get("setting", []),
                    value=[], multiselect=True, scale=1,
                )
                dd["lighting_mood"] = gr.Dropdown(
                    label="Lighting & Mood",
                    choices=_group_choices.get("lighting_mood", []),
                    value=[], multiselect=True, scale=1,
                )
            with gr.Row():
                dd["visual_style"] = gr.Dropdown(
                    label="Visual Style",
                    choices=_group_choices.get("visual_style", []),
                    value=[], multiselect=True, scale=1,
                )
                dd["camera"] = gr.Dropdown(
                    label="Camera",
                    choices=_group_choices.get("camera", []),
                    value=[], multiselect=True, scale=1,
                )
                dd["audio"] = gr.Dropdown(
                    label="Audio",
                    choices=_group_choices.get("audio", []),
                    value=[], multiselect=True, scale=1,
                )

            # ── Wildcards ──
            with gr.Row():
                wildcards = gr.Dropdown(
                    label="Wildcards",
                    choices=list(_wildcard_choices),
                    value=[], multiselect=True, scale=1,
                )

            # ── Word Limit + Temperature + Think ──
            with gr.Row():
                word_limit = gr.Slider(
                    label="Word Limit", minimum=20, maximum=500,
                    value=150, step=10, scale=1,
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=2.0,
                    value=0.7, step=0.05, scale=1,
                )
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)

            # ── Custom system prompt ──
            custom_system_prompt = gr.Textbox(
                label="Custom System Prompt",
                lines=4, visible=False,
                placeholder="Enter your custom system prompt...",
            )
            base.change(
                fn=lambda b: gr.update(visible=(b == "Custom")),
                inputs=[base], outputs=[custom_system_prompt],
                show_progress=False,
            )

            # ── API + reload ──
            with gr.Row():
                api_url = gr.Textbox(label="API URL", value=DEFAULT_API_URL, scale=3)
                model = gr.Dropdown(
                    label="Model",
                    choices=initial_models,
                    value=DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0],
                    allow_custom_value=True, scale=2,
                )
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_dir_path = gr.Textbox(
                    label="Local Overrides Directory",
                    placeholder=f"Using: {_env_local}" if _env_local else "Path to directory with local YAML/JSON files",
                    scale=3,
                )
                reload_btn = gr.Button(value="\U0001f504 Reload Config", scale=0, min_width=140)
                refresh_models_btn = gr.Button(value="\U0001f504 Models", scale=0, min_width=120)

            # ── Reload wiring ──
            def _do_refresh(current_base, s, se, lm, vs, ca, au, wc, local_path):
                _reload_all(local_path)
                bn = _base_names()
                wcc = list(_wildcard_choices)
                results = [
                    gr.update(choices=bn, value=current_base if current_base in bn else bn[0]),
                ]
                for gk in ["subject", "setting", "lighting_mood", "visual_style", "camera", "audio"]:
                    choices = _group_choices.get(gk, [])
                    results.append(gr.update(choices=choices, value=[]))
                results.append(gr.update(choices=wcc, value=[w for w in (wc or []) if w in _wildcards]))
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base, dd["subject"], dd["setting"], dd["lighting_mood"],
                        dd["visual_style"], dd["camera"], dd["audio"], wildcards, local_dir_path],
                outputs=[base, dd["subject"], dd["setting"], dd["lighting_mood"],
                         dd["visual_style"], dd["camera"], dd["audio"], wildcards],
                show_progress=False,
            )
            refresh_models_btn.click(
                fn=_refresh_models,
                inputs=[api_url, model], outputs=[model],
                show_progress=False,
            )

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")

            # Common input lists
            mode_inputs = [mode_still, mode_scene, mode_audio]
            dd_inputs = [dd["subject"], dd["setting"], dd["lighting_mood"],
                         dd["visual_style"], dd["camera"], dd["audio"]]

            # ── Enhance ──
            enhance_btn.click(
                fn=lambda: "<span style='color:#aaa'>Enhancing...</span>",
                inputs=[], outputs=[status], show_progress=False,
            ).then(
                fn=enhance_prompt,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt]
                       + mode_inputs + dd_inputs
                       + [wildcards, word_limit, think, temperature],
                outputs=[prompt_out, status],
            )

            # ── Refine ──
            refine_btn.click(
                fn=lambda x: x,
                _js=f"""function(x) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    return ta ? [ta.value] : [x];
                }}""",
                inputs=[prompt_in], outputs=[prompt_in], show_progress=False,
            ).then(
                fn=lambda: "<span style='color:#aaa'>Refining...</span>",
                inputs=[], outputs=[status], show_progress=False,
            ).then(
                fn=refine_prompt,
                inputs=[prompt_in, source_prompt, api_url, model]
                       + mode_inputs + dd_inputs
                       + [wildcards, think, temperature],
                outputs=[prompt_out, status],
            )

            # ── Tags ──
            tags_btn.click(
                fn=lambda: "<span style='color:#aaa'>Generating tags...</span>",
                inputs=[], outputs=[status], show_progress=False,
            ).then(
                fn=generate_tags,
                inputs=[source_prompt, api_url, model, tag_format]
                       + mode_inputs + dd_inputs
                       + [wildcards, think, temperature],
                outputs=[prompt_out, status],
            )

            # ── Write to main prompt textarea ──
            prompt_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[prompt_out], outputs=[prompt_out], show_progress=False,
            )

        # ── Metadata ──
        all_dd_components = [dd["subject"], dd["setting"], dd["lighting_mood"],
                             dd["visual_style"], dd["camera"], dd["audio"]]

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (word_limit, "PE WordLimit"),
            (wildcards, lambda params: [w.strip() for w in params.get("PE Wildcards", "").split(",") if w.strip()] if params.get("PE Wildcards") else []),
            (think, "PE Think"),
        ]
        self.paste_field_names = [
            "PE Source", "PE Base", "PE WordLimit", "PE Wildcards", "PE Think",
        ]
        return [source_prompt, api_url, model, base, custom_system_prompt,
                mode_still, mode_scene, mode_audio,
                *all_dd_components, wildcards, word_limit, think, temperature]

    def process(self, p, source_prompt, api_url, model, base, custom_system_prompt,
                mode_still, mode_scene, mode_audio,
                dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio,
                wildcards, word_limit, think, temperature):
        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if word_limit and int(word_limit) != 150:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)

        # Collect all modifiers for metadata
        all_mod_names = []
        if mode_still:
            all_mod_names.append("Still")
        if mode_scene:
            all_mod_names.append("Scene")
        if mode_audio:
            all_mod_names.append("Audio")
        for dd_val in [dd_subject, dd_setting, dd_lighting_mood, dd_visual_style, dd_camera, dd_audio]:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)

        if wildcards:
            p.extra_generation_params["PE Wildcards"] = ", ".join(wildcards)
        if think:
            p.extra_generation_params["PE Think"] = True
