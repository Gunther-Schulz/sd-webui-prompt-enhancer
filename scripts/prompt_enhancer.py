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


def _load_flat_json(filename):
    """Load a flat JSON file: {name: prompt_string, ...}."""
    path = os.path.join(_EXT_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, str)}
        logger.error(f"{path} must be a JSON object {{name: value}}")
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
    return {}


try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _load_categorized_file(path):
    """Load a categorized file (JSON or YAML) and return raw dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.error(f"{path} must be a mapping of categories")
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


def _load_override_dir(dir_path):
    """Load and merge all .yaml/.yml/.json files from a directory."""
    merged = {}
    if not os.path.isdir(dir_path):
        return merged
    for name in sorted(os.listdir(dir_path)):
        if name.startswith("."):
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_categorized_file(os.path.join(dir_path, name))
        if data:
            merged = _merge_categorized(merged, data)
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


def _load_categorized_json(filename, local_override_path=""):
    """Load a categorized JSON file, optionally merging local overrides.

    The local override can be a single file (.json/.yaml/.yml) or a
    directory containing multiple files. Files in a directory are loaded
    in alphabetical order.

    Resolution order for local override location:
    1. Explicit path passed in (from UI field)
    2. PROMPT_ENHANCER_LOCAL_MODIFIERS env var (for modifiers.json)
    3. Default: <ext_dir>/<filename>.local.json

    Returns (flat_dict, choice_list) where flat_dict maps name->keywords
    and choice_list has category separators for the dropdown UI.
    """
    base_data = _load_categorized_file(os.path.join(_EXT_DIR, filename))

    # Resolve local override path
    local_path = (local_override_path or "").strip()
    if not local_path and filename == "modifiers.json":
        local_path = os.environ.get("PROMPT_ENHANCER_LOCAL_MODIFIERS", "")
    if not local_path:
        local_path = os.path.join(_EXT_DIR, filename.replace(".json", ".local.json"))

    # Load from directory or single file
    if os.path.isdir(local_path):
        override_data = _load_override_dir(local_path)
        if override_data:
            base_data = _merge_categorized(base_data, override_data)
    elif os.path.isfile(local_path):
        override_data = _load_categorized_file(local_path)
        if override_data:
            base_data = _merge_categorized(base_data, override_data)

    return _categorized_to_flat_and_choices(base_data)


# ── Load all config from JSON files ──────────────────────────────────────────

_bases = {}
_modifiers = {}
_modifier_choices = []
_wildcards = {}
_wildcard_choices = []


def _reload_all(local_modifiers_path=""):
    """Reload all JSON config files from disk."""
    global _bases, _modifiers, _modifier_choices, _wildcards, _wildcard_choices
    _bases = _load_flat_json("bases.json")
    _modifiers, _modifier_choices = _load_categorized_json("modifiers.json", local_modifiers_path)
    _wildcards, _wildcard_choices = _load_categorized_json("wildcards.json")


_reload_all()

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


def _refresh_all(current_base, current_modifiers, current_wildcards, local_modifiers_path=""):
    """Reload all JSON files and update all dropdowns."""
    _reload_all(local_modifiers_path)
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
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    content = result.get("message", {}).get("content", "")
    return _strip_think_blocks(content)


def enhance_prompt(source, api_url, model, base, custom_system_prompt,
                   style_notes, modifiers, wildcards, word_limit, think, temperature):
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

    # Layer 2: Style keywords (modifiers + freeform notes)
    style_parts = []
    for mod_name in (modifiers or []):
        keywords = _modifiers.get(mod_name, "")
        if keywords:
            style_parts.append(keywords)
    notes = (style_notes or "").strip()
    if notes:
        style_parts.append(notes)
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
        enhanced = _call_llm(source, api_url, model, system_prompt, temperature, think=think)
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

            # ── Base + Style Notes ──
            with gr.Row():
                base = gr.Dropdown(
                    label="Base",
                    choices=_base_names(),
                    value="Still",
                    scale=1,
                    info="Still = image, Scene = video, Refine = minimal cleanup",
                )
                style_notes = gr.Textbox(
                    label="Style Notes",
                    placeholder="Freeform style directions, e.g. 'moody, 1970s, desaturated'",
                    scale=2,
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
                    info="Let the LLM make creative choices",
                )

            # ── Word Limit + Temperature + Think ──
            with gr.Row():
                word_limit = gr.Slider(
                    label="Word Limit", minimum=20, maximum=500,
                    value=150, step=10, scale=1,
                    info="Target length of enhanced prompt",
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=2.0,
                    value=0.7, step=0.05, scale=1,
                )
                think = gr.Checkbox(
                    label="Think",
                    value=False,
                    info="Let model reason before answering (slower)",
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
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL_MODIFIERS", "")
                local_modifiers_path = gr.Textbox(
                    label="Local Modifiers Override",
                    placeholder=f"Using: {_env_local}" if _env_local else "File or directory path (YAML/JSON)",
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

            # Hidden bridge for writing to the main prompt textarea
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")

            enhance_btn.click(
                fn=enhance_prompt,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt,
                        style_notes, modifiers, wildcards, word_limit, think, temperature],
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
            (style_notes, "PE Style"),
            (word_limit, "PE WordLimit"),
            (modifiers, lambda params: [m.strip() for m in params.get("PE Modifiers", "").split(",") if m.strip()] if params.get("PE Modifiers") else []),
            (wildcards, lambda params: [w.strip() for w in params.get("PE Wildcards", "").split(",") if w.strip()] if params.get("PE Wildcards") else []),
            (think, "PE Think"),
        ]
        self.paste_field_names = [
            "PE Source", "PE Base", "PE Style", "PE WordLimit",
            "PE Modifiers", "PE Wildcards", "PE Think",
        ]
        return [source_prompt, api_url, model, base, custom_system_prompt,
                style_notes, modifiers, wildcards, word_limit, think, temperature]

    def process(self, p, source_prompt, api_url, model, base, custom_system_prompt,
                style_notes, modifiers, wildcards, word_limit, think, temperature):
        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if style_notes:
            p.extra_generation_params["PE Style"] = style_notes
        if word_limit and int(word_limit) != 150:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)
        if modifiers:
            p.extra_generation_params["PE Modifiers"] = ", ".join(modifiers)
        if wildcards:
            p.extra_generation_params["PE Wildcards"] = ", ".join(wildcards)
        if think:
            p.extra_generation_params["PE Think"] = True
