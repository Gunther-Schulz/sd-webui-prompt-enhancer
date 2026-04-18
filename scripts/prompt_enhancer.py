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
_MODIFIERS_DIR = os.path.join(_EXT_DIR, "modifiers")

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"

# Mode keywords (rendered as checkboxes, not in files)
MODE_KEYWORDS = {
    "Still": "frozen moment, static composition, describe a single instant in time, no temporal language, no movement, captured pose",
    "Scene": "chronological action, present-progressive verbs, temporal flow, describe movement unfolding over time, use connectors like as then while, no timestamps or scene cuts",
    "Audio": "include audio descriptions integrated chronologically with the visuals, describe specific sounds not vague atmosphere, ambient sounds and sound effects, be concrete about what is heard and when",
}


# ── File loading ─────────────────────────────────────────────────────────────

def _load_file(path):
    """Load a JSON or YAML file and return parsed content."""
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


def _scan_modifier_files(directory):
    """Scan a directory for modifier YAML/JSON files.

    Returns dict: dropdown_name -> {category: {modifier: keywords}}.
    Skips _bases.* files.
    """
    result = {}
    if not directory or not os.path.isdir(directory):
        return result
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem == BASES_FILENAME:
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        # Dropdown label from filename: "visual-style.yaml" -> "Visual Style"
        label = stem.replace("-", " ").replace("_", " ").title()
        data = _load_file(os.path.join(directory, name))
        if data:
            result[label] = data
    return result


def _merge_modifier_dicts(base, override):
    """Merge two dropdown-level dicts. Same dropdown name -> merge categories."""
    merged = {}
    for label, categories in base.items():
        merged[label] = {}
        for cat, items in categories.items():
            if isinstance(items, dict):
                merged[label][cat] = dict(items)
    for label, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if label not in merged:
            merged[label] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[label]:
                merged[label][cat] = {}
            merged[label][cat].update(items)
    return merged


def _build_dropdown_data(categories_dict):
    """Build flat lookup and choice list from a single dropdown's categories."""
    flat = {}
    choices = []
    for cat_name, items in categories_dict.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {cat_name.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, keywords in items.items():
            if isinstance(keywords, str):
                flat[name] = keywords
                choices.append(name)
    return flat, choices


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}          # flat: name -> keywords (for lookup across all dropdowns)
_dropdown_order = []         # list of dropdown labels in display order
_dropdown_choices = {}       # label -> [choice_list with separators]
_wildcards = {}
_wildcard_choices = []


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _dropdown_order, _dropdown_choices, _wildcards, _wildcard_choices

    local_dir = _get_local_dir(local_dir_path)

    # Bases
    _bases = {k: v for k, v in _load_file(os.path.join(_EXT_DIR, "bases.json")).items() if isinstance(v, str)}
    _bases.update(_load_local_bases(local_dir))

    # Modifiers: scan extension modifiers/ dir + local dir, merge
    ext_mods = _scan_modifier_files(_MODIFIERS_DIR)
    local_mods = _scan_modifier_files(local_dir)
    all_mods = _merge_modifier_dicts(ext_mods, local_mods)

    _all_modifiers = {}
    _dropdown_order = []
    _dropdown_choices = {}
    for label in sorted(all_mods.keys()):
        flat, choices = _build_dropdown_data(all_mods[label])
        if choices:
            _dropdown_order.append(label)
            _dropdown_choices[label] = choices
            _all_modifiers.update(flat)

    # Wildcards
    wc_data = _load_file(os.path.join(_EXT_DIR, "wildcards.json"))
    _wildcards = {}
    _wildcard_choices = []
    for cat, items in wc_data.items():
        if not isinstance(items, dict):
            continue
        _wildcard_choices.append(f"\u2500\u2500\u2500\u2500\u2500 {cat.title()} \u2500\u2500\u2500\u2500\u2500")
        for name, prompt in items.items():
            if isinstance(prompt, str):
                _wildcards[name] = prompt
                _wildcard_choices.append(name)


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


def _collect_modifiers(mode_still, mode_scene, mode_audio, dropdown_selections):
    """Collect all selected modifiers into a list of (name, keywords) tuples."""
    result = []
    if mode_still:
        result.append(("Still", MODE_KEYWORDS["Still"]))
    if mode_scene:
        result.append(("Scene", MODE_KEYWORDS["Scene"]))
    if mode_audio:
        result.append(("Audio", MODE_KEYWORDS["Audio"]))
    for selections in dropdown_selections:
        for name in (selections or []):
            keywords = _all_modifiers.get(name, "")
            if keywords:
                result.append((name, keywords))
    return result


def _build_style_string(mod_list):
    """Build the 'Apply these styles:' string."""
    parts = []
    for name, keywords in mod_list:
        if name.lower() not in keywords.lower():
            keywords = f"{name.lower()}, {keywords}"
        parts.append(keywords)
    return f"Apply these styles: {', '.join(parts)}." if parts else ""


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


def _assemble_system_prompt(base_name, custom_system_prompt, mod_list, wildcards_list, source="", word_limit=0):
    """Assemble the complete system prompt from all layers."""
    if base_name == "Custom":
        system_prompt = (custom_system_prompt or "").strip()
    else:
        system_prompt = _bases.get(base_name, "")
    if not system_prompt:
        return None

    style_str = _build_style_string(mod_list)
    if style_str:
        system_prompt = f"{system_prompt}\n\n{style_str}"

    for wc_name in (wildcards_list or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    if source and _has_inline_wildcards(source):
        system_prompt = f"{system_prompt}\n\n{INLINE_WILDCARD_INSTRUCTION}"

    word_limit = int(word_limit) if word_limit else 0
    if word_limit > 0:
        system_prompt = f"{system_prompt}\n\nAim for around {word_limit} words."

    return system_prompt


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
                label="Source Prompt", lines=3,
                placeholder="Type your prompt here, then click Enhance... Use {name?} for inline wildcards.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(value="\U0001f4a1 Enhance", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_enhance_btn")
                tags_btn = gr.Button(value="\U0001f3f7 Tags", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_tags_btn")
                refine_btn = gr.Button(value="\U0001f527 Refine", scale=0, min_width=120, elem_id=f"{tab}_pe_refine_btn")
                grab_btn = gr.Button(value="\u2b07 Grab", scale=0, min_width=80, elem_id=f"{tab}_pe_grab_btn")
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            grab_btn.click(
                fn=lambda x: x,
                _js=f"""function(x) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    return ta ? [ta.value] : [x];
                }}""",
                inputs=[source_prompt], outputs=[source_prompt], show_progress=False,
            )

            # ── Mode checkboxes ──
            with gr.Row():
                mode_still = gr.Checkbox(label="Still", value=False, elem_id=f"{tab}_pe_mode_still")
                mode_scene = gr.Checkbox(label="Scene", value=False, elem_id=f"{tab}_pe_mode_scene")
                mode_audio = gr.Checkbox(label="Audio", value=False, elem_id=f"{tab}_pe_mode_audio")
                mode_still.do_not_save_to_config = True
                mode_scene.do_not_save_to_config = True
                mode_audio.do_not_save_to_config = True

            # ── Base + Tag Format ──
            with gr.Row():
                base = gr.Dropdown(label="Base", choices=_base_names(), value="Default", scale=2)
                tag_format = gr.Dropdown(label="Tag Format", choices=list(TAGS_SYSTEM_PROMPTS.keys()), value=list(TAGS_SYSTEM_PROMPTS.keys())[0], scale=1)

            # ── Auto-generated modifier dropdowns (one per file) ──
            dd_components = []
            dd_labels = list(_dropdown_order)

            # Layout: 3 dropdowns per row
            for i in range(0, len(dd_labels), 3):
                row_labels = dd_labels[i:i+3]
                with gr.Row():
                    for label in row_labels:
                        d = gr.Dropdown(
                            label=label,
                            choices=_dropdown_choices.get(label, []),
                            value=[], multiselect=True, scale=1,
                        )
                        d.do_not_save_to_config = True
                        dd_components.append(d)

            # ── Wildcards ──
            with gr.Row():
                wildcards = gr.Dropdown(label="Wildcards", choices=list(_wildcard_choices), value=[], multiselect=True, scale=1)
                wildcards.do_not_save_to_config = True

            # ── Word Limit + Temperature + Think ──
            with gr.Row():
                word_limit = gr.Slider(label="Word Limit", minimum=20, maximum=500, value=150, step=10, scale=1)
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.05, scale=1)
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)
                think.do_not_save_to_config = True

            # ── Custom system prompt ──
            custom_system_prompt = gr.Textbox(label="Custom System Prompt", lines=4, visible=False, placeholder="Enter your custom system prompt...")
            base.change(fn=lambda b: gr.update(visible=(b == "Custom")), inputs=[base], outputs=[custom_system_prompt], show_progress=False)

            # ── API + reload ──
            with gr.Row():
                api_url = gr.Textbox(label="API URL", value=DEFAULT_API_URL, scale=3)
                model = gr.Dropdown(label="Model", choices=initial_models, value=DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0], allow_custom_value=True, scale=2)
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_dir_path = gr.Textbox(label="Local Overrides Directory", placeholder=f"Using: {_env_local}" if _env_local else "Path to directory with local YAML/JSON files", scale=3)
                reload_btn = gr.Button(value="\U0001f504 Reload", scale=0, min_width=100)
                refresh_models_btn = gr.Button(value="\U0001f504 Models", scale=0, min_width=100)

            # ── Reload wiring ──
            # Note: reload rebuilds dropdowns but can't add/remove them dynamically.
            # New files require a Forge restart. Existing dropdown contents are refreshed.
            def _do_refresh(current_base, *args):
                # Last arg is local_dir_path, second-to-last is wildcards
                local_path = args[-1]
                wc_val = args[-2]
                dd_vals = args[:-2]

                _reload_all(local_path)
                results = [gr.update(choices=_base_names(), value=current_base if current_base in _bases else "Default")]
                for i, label in enumerate(dd_labels):
                    choices = _dropdown_choices.get(label, [])
                    old_val = dd_vals[i] if i < len(dd_vals) else []
                    results.append(gr.update(choices=choices, value=[v for v in (old_val or []) if v in _all_modifiers]))
                results.append(gr.update(choices=list(_wildcard_choices), value=[w for w in (wc_val or []) if w in _wildcards]))
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base] + dd_components + [wildcards, local_dir_path],
                outputs=[base] + dd_components + [wildcards],
                show_progress=False,
            )
            refresh_models_btn.click(fn=_refresh_models, inputs=[api_url, model], outputs=[model], show_progress=False)

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")

            mode_inputs = [mode_still, mode_scene, mode_audio]

            # ── Enhance ──
            def _enhance(source, api_url, model, base_name, custom_sp, m_still, m_scene, m_audio, *args):
                wl, th, temp = args[-3], args[-2], args[-1]
                wc = args[-4]
                dd_vals = args[:-4]

                source = (source or "").strip()
                if not source:
                    return "", "<span style='color:#c66'>Source prompt is empty.</span>"

                mods = _collect_modifiers(m_still, m_scene, m_audio, dd_vals)
                sp = _assemble_system_prompt(base_name, custom_sp, mods, wc, source, wl)
                if not sp:
                    return "", "<span style='color:#c66'>No system prompt configured.</span>"

                try:
                    result = _clean_output(_call_llm(source, api_url, model, sp, temp, think=th))
                    return result, f"<span style='color:#6c6'>OK - {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    return "", f"<span style='color:#c66'>{msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    return "", f"<span style='color:#c66'>{msg}</span>"

            enhance_btn.click(
                fn=lambda: "<span style='color:#aaa'>Enhancing...</span>",
                inputs=[], outputs=[status], show_progress=False,
            ).then(
                fn=_enhance,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt]
                       + mode_inputs + dd_components
                       + [wildcards, word_limit, think, temperature],
                outputs=[prompt_out, status],
            )

            # ── Refine ──
            def _refine(existing, source, api_url, model, m_still, m_scene, m_audio, *args):
                th, temp = args[-2], args[-1]
                wc = args[-3]
                dd_vals = args[:-3]

                existing = (existing or "").strip()
                if not existing:
                    return "", "<span style='color:#c66'>No prompt to refine.</span>"

                source = (source or "").strip()
                mods = _collect_modifiers(m_still, m_scene, m_audio, dd_vals)

                if not mods and not wc and not source:
                    return "", "<span style='color:#c66'>Select modifiers/wildcards or update source prompt.</span>"

                sp = REFINE_SYSTEM_PROMPT
                if source:
                    sp = f"{sp}\n\nThe user has provided updated direction. Integrate:\n{source}"
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"
                for wc_name in (wc or []):
                    wc_prompt = _wildcards.get(wc_name, "")
                    if wc_prompt:
                        sp = f"{sp}\n\n{wc_prompt}"

                try:
                    result = _clean_output(_call_llm(existing, api_url, model, sp, temp, think=th))
                    return result, f"<span style='color:#6c6'>OK - refined to {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    return "", f"<span style='color:#c66'>Connection failed: {e.reason}</span>"
                except Exception as e:
                    return "", f"<span style='color:#c66'>{type(e).__name__}: {e}</span>"

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
                fn=_refine,
                inputs=[prompt_in, source_prompt, api_url, model]
                       + mode_inputs + dd_components
                       + [wildcards, think, temperature],
                outputs=[prompt_out, status],
            )

            # ── Tags ──
            def _tags(source, api_url, model, tag_fmt, m_still, m_scene, m_audio, *args):
                th, temp = args[-2], args[-1]
                wc = args[-3]
                dd_vals = args[:-3]

                source = (source or "").strip()
                if not source:
                    return "", "<span style='color:#c66'>Source prompt is empty.</span>"

                sp = TAGS_SYSTEM_PROMPTS.get(tag_fmt, list(TAGS_SYSTEM_PROMPTS.values())[0])
                mods = _collect_modifiers(m_still, m_scene, m_audio, dd_vals)
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"
                for wc_name in (wc or []):
                    wc_prompt = _wildcards.get(wc_name, "")
                    if wc_prompt:
                        sp = f"{sp}\n\n{wc_prompt}"

                try:
                    tags = _clean_output(_call_llm(source, api_url, model, sp, temp, think=th))
                    if tag_fmt in ("Illustrious", "Pony"):
                        tags = ", ".join(t.strip().replace(" ", "_") for t in tags.split(",") if t.strip())
                    tag_count = len([t for t in tags.split(",") if t.strip()])
                    return tags, f"<span style='color:#6c6'>OK - {tag_count} tags</span>"
                except urllib.error.URLError as e:
                    return "", f"<span style='color:#c66'>Connection failed: {e.reason}</span>"
                except Exception as e:
                    return "", f"<span style='color:#c66'>{type(e).__name__}: {e}</span>"

            tags_btn.click(
                fn=lambda: "<span style='color:#aaa'>Generating tags...</span>",
                inputs=[], outputs=[status], show_progress=False,
            ).then(
                fn=_tags,
                inputs=[source_prompt, api_url, model, tag_format]
                       + mode_inputs + dd_components
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
        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (word_limit, "PE WordLimit"),
            (wildcards, lambda params: [w.strip() for w in params.get("PE Wildcards", "").split(",") if w.strip()] if params.get("PE Wildcards") else []),
            (think, "PE Think"),
        ]
        self.paste_field_names = ["PE Source", "PE Base", "PE WordLimit", "PE Wildcards", "PE Think"]

        # Store dropdown references for process()
        self._dd_components = dd_components
        self._mode_inputs = [mode_still, mode_scene, mode_audio]

        return [source_prompt, api_url, model, base, custom_system_prompt,
                mode_still, mode_scene, mode_audio,
                *dd_components, wildcards, word_limit, think, temperature]

    def process(self, p, source_prompt, api_url, model, base, custom_system_prompt,
                mode_still, mode_scene, mode_audio, *args):
        # args = *dd_values, wildcards, word_limit, think, temperature
        wildcards = args[-4]
        word_limit = args[-3]
        think = args[-2]
        dd_vals = args[:-4]

        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if word_limit and int(word_limit) != 150:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)

        all_mod_names = []
        if mode_still:
            all_mod_names.append("Still")
        if mode_scene:
            all_mod_names.append("Scene")
        if mode_audio:
            all_mod_names.append("Audio")
        for dd_val in dd_vals:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)
        if wildcards:
            p.extra_generation_params["PE Wildcards"] = ", ".join(wildcards)
        if think:
            p.extra_generation_params["PE Think"] = True
