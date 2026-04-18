import json
import logging
import os
import re
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory (where JSON config files live) ──────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"


def _load_file(path):
    """Load a JSON or YAML file and return its parsed content as a dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.error(f"{path} must be a mapping")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return {}


def _merge_categorized(base_data, override_data):
    """Merge override into base: new categories append, existing categories extend."""
    merged = {}
    for cat, items in base_data.items():
        if isinstance(items, dict):
            merged[cat] = dict(items)
    for cat, items in override_data.items():
        if not isinstance(items, dict):
            continue
        if cat in merged:
            merged[cat].update(items)
        else:
            merged[cat] = dict(items)
    return merged


def _categorized_to_flat_and_choices(data):
    """Convert categorized dict to (flat_dict, choice_list) for dropdown UI."""
    flat = {}
    choices = []
    for category, items in data.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {category.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, prompt in items.items():
            if isinstance(prompt, str):
                flat[name] = prompt
                choices.append(name)
    return flat, choices


def _get_local_dir(ui_path=""):
    """Resolve the local overrides directory.

    Priority: UI field > PROMPT_ENHANCER_LOCAL env var > None.
    """
    path = (ui_path or "").strip()
    if not path:
        path = os.environ.get("PROMPT_ENHANCER_LOCAL", "").strip()
    if path and os.path.isdir(path):
        return path
    return None


def _load_local_bases(local_dir):
    """Load _bases.yaml from the local directory (flat: name -> prompt)."""
    if not local_dir:
        return {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(local_dir, BASES_FILENAME + ext)
        if os.path.isfile(path):
            return {k: v for k, v in _load_file(path).items() if isinstance(v, str)}
    return {}


def _load_local_modifiers(local_dir):
    """Load all files except _bases.* from the local directory (categorized)."""
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
            merged = _merge_categorized(merged, data)
    return merged


# ── Load all config from JSON files ──────────────────────────────────────────

_bases = {}
_modifiers = {}
_modifier_choices = []
_wildcards = {}
_wildcard_choices = []


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _modifiers, _modifier_choices, _wildcards, _wildcard_choices

    local_dir = _get_local_dir(local_dir_path)

    # Bases: extension bases.json + local bases.yaml
    _bases = {k: v for k, v in _load_file(os.path.join(_EXT_DIR, "bases.json")).items() if isinstance(v, str)}
    _bases.update(_load_local_bases(local_dir))

    # Modifiers: extension modifiers.json + all local files (except bases.yaml)
    mod_data = _load_file(os.path.join(_EXT_DIR, "modifiers.json"))
    local_mods = _load_local_modifiers(local_dir)
    if local_mods:
        mod_data = _merge_categorized(mod_data, local_mods)
    _modifiers, _modifier_choices = _categorized_to_flat_and_choices(mod_data)

    # Wildcards: extension wildcards.json only
    wc_data = _load_file(os.path.join(_EXT_DIR, "wildcards.json"))
    _wildcards, _wildcard_choices = _categorized_to_flat_and_choices(wc_data)


_reload_all()

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
        "- Use underscores in multi-word tags (long_hair, blue_eyes)\n"
        "- Start with quality tags: masterpiece, best_quality, absurdres\n"
        "- Then subject count (1girl, 1boy, 1other, no_humans, etc.)\n"
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
        "- Use underscores in multi-word tags (long_hair, blue_eyes)\n"
        "- Start with score tags: score_9, score_8_up, score_7_up\n"
        "- Then source tag if applicable (source_anime, source_furry, source_pony, source_cartoon)\n"
        "- Then subject count (1girl, 1boy, 1other, no_humans, etc.)\n"
        "- Then appearance, clothing, pose, expression, setting, composition\n"
        "- Use the user's key descriptive words as tags\n"
        "- Do NOT use quality words like masterpiece or best_quality\n"
        "- Include rating tag at end (rating_safe, rating_questionable, rating_explicit)\n"
        "- When style directions are provided, include relevant tags for them"
    ),
}

INLINE_WILDCARD_INSTRUCTION = (
    "The user's prompt contains placeholders in {name?} format. "
    "For each one, choose a specific, vivid option that creates a coherent scene. "
    "Replace the placeholder with your choice seamlessly."
)


# ── Dropdown helpers ─────────────────────────────────────────────────────────

def _base_names():
    names = list(_bases.keys())
    names.append("Custom")
    return names


def _refresh_all(current_base, current_modifiers, current_wildcards, local_dir_path=""):
    """Reload all config files and update all dropdowns."""
    _reload_all(local_dir_path)
    base_names = _base_names()
    mod_choices = list(_modifier_choices)
    wc_choices = list(_wildcard_choices)
    return (
        gr.update(choices=base_names, value=current_base if current_base in base_names else base_names[0]),
        gr.update(choices=mod_choices, value=[m for m in (current_modifiers or []) if m in _modifiers]),
        gr.update(choices=wc_choices, value=[w for w in (current_wildcards or []) if w in _wildcards]),
    )


# ── Ollama model management ─────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _to_ollama_base(api_url):
    """Convert an OpenAI-compatible URL to the Ollama base URL."""
    base = api_url
    for suffix in ("/v1/chat/completions", "/v1", "/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or DEFAULT_OLLAMA_BASE


def _fetch_ollama_models(api_url):
    """Fetch available models from the Ollama API."""
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
    """Refresh the model dropdown choices from Ollama."""
    models = _fetch_ollama_models(api_url)
    if not models:
        return gr.update()
    value = current_model if current_model in models else models[0]
    return gr.update(choices=models, value=value)


# ── Core enhancement logic ───────────────────────────────────────────────────

def _strip_think_blocks(text):
    """Remove thinking blocks that Qwen3 and similar models emit."""
    return re.sub(r"<thi" + r"nk>[\s\S]*?</thi" + r"nk>", "", text).strip()


def _has_inline_wildcards(text):
    """Check if the source prompt contains {name?} placeholders."""
    return bool(re.search(r"\{[^}]+\?\}", text))


def _clean_output(text):
    """Strip markdown formatting and other artifacts from LLM output."""
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove underscore bold/italic
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
        "options": {
            "temperature": float(temperature),
        },
        "think": bool(think),
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base}/api/chat"

    # Retry once on connection failure (Ollama may be reloading the model)
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


def enhance_prompt(source, api_url, model, base, custom_system_prompt,
                   modifiers, wildcards, word_limit, think, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    # Layer 1: Base system prompt
    if base == "Custom":
        system_prompt = (custom_system_prompt or "").strip()
    else:
        system_prompt = _bases.get(base, "")

    if not system_prompt:
        return "", "<span style='color:#c66'>No system prompt configured.</span>"

    # Layer 2: Style keywords from modifiers
    style_parts = []
    for mod_name in (modifiers or []):
        keywords = _modifiers.get(mod_name, "")
        if keywords:
            # Prepend modifier name so the exact term is always in the output
            if mod_name.lower() not in keywords.lower():
                keywords = f"{mod_name.lower()}, {keywords}"
            style_parts.append(keywords)
    if style_parts:
        system_prompt = f"{system_prompt}\n\nApply these styles: {', '.join(style_parts)}."

    # Layer 3: Wildcards (creative delegation instructions)
    for wc_name in (wildcards or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    # Inline wildcards: detect {name?} in source prompt
    if _has_inline_wildcards(source):
        system_prompt = f"{system_prompt}\n\n{INLINE_WILDCARD_INSTRUCTION}"

    # Word limit
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
    except KeyError:
        logger.error("Unexpected API response format")
        return "", "<span style='color:#c66'>Unexpected API response format</span>"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error(msg)
        return "", f"<span style='color:#c66'>{msg}</span>"


def refine_prompt(existing_prompt, source_prompt, api_url, model, modifiers, wildcards, think, temperature):
    existing = (existing_prompt or "").strip()
    if not existing:
        return "", "<span style='color:#c66'>No prompt to refine. Generate one first with Enhance.</span>"

    source = (source_prompt or "").strip()
    has_modifiers = bool(modifiers or wildcards)
    has_source = bool(source)

    if not has_modifiers and not has_source:
        return "", "<span style='color:#c66'>Select modifiers/wildcards or update the source prompt.</span>"

    system_prompt = REFINE_SYSTEM_PROMPT

    # Source prompt changes
    if has_source:
        system_prompt = f"{system_prompt}\n\nThe user has provided updated subject/scene direction. Integrate these changes into the existing prompt, replacing conflicting elements:\n{source}"

    # Build style changes to apply
    style_parts = []
    for mod_name in (modifiers or []):
        keywords = _modifiers.get(mod_name, "")
        if keywords:
            if mod_name.lower() not in keywords.lower():
                keywords = f"{mod_name.lower()}, {keywords}"
            style_parts.append(keywords)
    if style_parts:
        system_prompt = f"{system_prompt}\n\nApply these styles: {', '.join(style_parts)}."

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


def generate_tags(source, api_url, model, tag_format, modifiers, wildcards, think, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    system_prompt = TAGS_SYSTEM_PROMPTS.get(tag_format, list(TAGS_SYSTEM_PROMPTS.values())[0])

    # Add modifier keywords as style directions
    style_parts = []
    for mod_name in (modifiers or []):
        keywords = _modifiers.get(mod_name, "")
        if keywords:
            if mod_name.lower() not in keywords.lower():
                keywords = f"{mod_name.lower()}, {keywords}"
            style_parts.append(keywords)
    if style_parts:
        system_prompt = f"{system_prompt}\n\nApply these styles: {', '.join(style_parts)}."

    # Wildcards
    for wc_name in (wildcards or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            system_prompt = f"{system_prompt}\n\n{wc_prompt}"

    try:
        tags = _clean_output(_call_llm(source, api_url, model, system_prompt, temperature, think=think))
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

            # ── Source prompt + enhance ──
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
                    scale=0,
                    min_width=120,
                    elem_id=f"{tab}_pe_enhance_btn",
                )
                refine_btn = gr.Button(
                    value="\U0001f527 Refine",
                    scale=0,
                    min_width=120,
                    elem_id=f"{tab}_pe_refine_btn",
                )
                tags_btn = gr.Button(
                    value="\U0001f3f7 Tags",
                    scale=0,
                    min_width=100,
                    elem_id=f"{tab}_pe_tags_btn",
                )
                grab_btn = gr.Button(
                    value="\u2b07 Grab from prompt box",
                    scale=0,
                    min_width=180,
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

            # ── Base + Tag Format ──
            with gr.Row():
                base = gr.Dropdown(
                    label="Base",
                    choices=_base_names(),
                    value="Still",
                    scale=2,
                    info="Still = image, Scene = video",
                )
                tag_format = gr.Dropdown(
                    label="Tag Format",
                    choices=list(TAGS_SYSTEM_PROMPTS.keys()),
                    value=list(TAGS_SYSTEM_PROMPTS.keys())[0],
                    scale=1,
                    info="For Tags button only",
                )

            # ── Modifiers + Wildcards ──
            with gr.Row():
                modifiers = gr.Dropdown(
                    label="Modifiers",
                    choices=list(_modifier_choices),
                    value=[],
                    multiselect=True,
                    scale=2,
                )
                wildcards = gr.Dropdown(
                    label="Wildcards",
                    choices=list(_wildcard_choices),
                    value=[],
                    multiselect=True,
                    scale=1,
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
                think = gr.Checkbox(
                    label="Think",
                    value=False,
                    scale=0, min_width=80,
                )

            # ── Custom system prompt (shown when Base = Custom) ──
            custom_system_prompt = gr.Textbox(
                label="Custom System Prompt",
                lines=4,
                visible=False,
                placeholder="Enter your custom system prompt...",
            )

            base.change(
                fn=lambda b: gr.update(visible=(b == "Custom")),
                inputs=[base],
                outputs=[custom_system_prompt],
                show_progress=False,
            )

            # ── API + reload ──
            with gr.Row():
                api_url = gr.Textbox(
                    label="API URL",
                    value=DEFAULT_API_URL,
                    scale=3,
                )
                model = gr.Dropdown(
                    label="Model",
                    choices=initial_models,
                    value=DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0],
                    allow_custom_value=True,
                    scale=2,
                )
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_modifiers_path = gr.Textbox(
                    label="Local Overrides Directory",
                    placeholder=f"Using: {_env_local}" if _env_local else "Path to directory with local YAML/JSON files",
                    scale=3,
                )
                reload_btn = gr.Button(
                    value="\U0001f504 Reload Config",
                    scale=0,
                    min_width=140,
                )
                refresh_models_btn = gr.Button(
                    value="\U0001f504 Models",
                    scale=0,
                    min_width=120,
                )

            reload_btn.click(
                fn=_refresh_all,
                inputs=[base, modifiers, wildcards, local_modifiers_path],
                outputs=[base, modifiers, wildcards],
                show_progress=False,
            )

            refresh_models_btn.click(
                fn=_refresh_models,
                inputs=[api_url, model],
                outputs=[model],
                show_progress=False,
            )

            # Hidden bridges for reading from / writing to the main prompt textarea
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")

            enhance_btn.click(
                fn=lambda: "<span style='color:#aaa'>Enhancing...</span>",
                inputs=[],
                outputs=[status],
                show_progress=False,
            ).then(
                fn=enhance_prompt,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt,
                        modifiers, wildcards, word_limit, think, temperature],
                outputs=[prompt_out, status],
            )

            # Refine: grab current prompt from main textarea, apply modifiers, write back
            refine_btn.click(
                fn=lambda x: x,
                _js=f"""function(x) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    return ta ? [ta.value] : [x];
                }}""",
                inputs=[prompt_in],
                outputs=[prompt_in],
                show_progress=False,
            ).then(
                fn=lambda: "<span style='color:#aaa'>Refining...</span>",
                inputs=[],
                outputs=[status],
                show_progress=False,
            ).then(
                fn=refine_prompt,
                inputs=[prompt_in, source_prompt, api_url, model, modifiers, wildcards, think, temperature],
                outputs=[prompt_out, status],
            )

            # Tags: convert source prompt + modifiers into booru tags
            tags_btn.click(
                fn=lambda: "<span style='color:#aaa'>Generating tags...</span>",
                inputs=[],
                outputs=[status],
                show_progress=False,
            ).then(
                fn=generate_tags,
                inputs=[source_prompt, api_url, model, tag_format, modifiers, wildcards, think, temperature],
                outputs=[prompt_out, status],
            )

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
                inputs=[prompt_out],
                outputs=[prompt_out],
                show_progress=False,
            )

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (word_limit, "PE WordLimit"),
            (modifiers, lambda params: [m.strip() for m in params.get("PE Modifiers", "").split(",") if m.strip()] if params.get("PE Modifiers") else []),
            (wildcards, lambda params: [w.strip() for w in params.get("PE Wildcards", "").split(",") if w.strip()] if params.get("PE Wildcards") else []),
            (think, "PE Think"),
        ]
        self.paste_field_names = [
            "PE Source", "PE Base", "PE WordLimit",
            "PE Modifiers", "PE Wildcards", "PE Think",
        ]
        return [source_prompt, api_url, model, base, custom_system_prompt,
                modifiers, wildcards, word_limit, think, temperature]

    def process(self, p, source_prompt, api_url, model, base, custom_system_prompt,
                modifiers, wildcards, word_limit, think, temperature):
        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if word_limit and int(word_limit) != 150:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)
        if modifiers:
            p.extra_generation_params["PE Modifiers"] = ", ".join(modifiers)
        if wildcards:
            p.extra_generation_params["PE Wildcards"] = ", ".join(wildcards)
        if think:
            p.extra_generation_params["PE Think"] = True
