import json
import logging
import os
import re
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts

logger = logging.getLogger("prompt_enhancer")

# ── Built-in presets (shipped with the extension) ─────────────────────────────

BUILTIN_PRESETS = {
    "Cinematic (Video)": (
        "You are an expert cinematic director with many award winning movies. "
        "When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes. "
        "Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. "
        "Start directly with the action, and keep descriptions literal and precise. "
        "Think like a cinematographer describing a shot list. "
        "Do not change the user input intent, just enhance it. "
        "For best results, build your prompts using this structure: "
        "Start with main action in a single sentence. "
        "Add specific details about movements and gestures. "
        "Describe character/object appearances precisely. "
        "Include background and environment details. "
        "Specify camera angles and movements. "
        "Describe lighting and colors. "
        "Note any changes or sudden events. "
        "Output the enhanced prompt only."
    ),
    "Z-Image (Photorealistic)": (
        "You are a photographer writing detailed shot descriptions for a photorealistic AI image generator "
        "that understands natural language. "
        "Write a single flowing paragraph describing the scene as if writing notes for a photo shoot. "
        "Include: the subject and their exact pose, body position, and expression. "
        "The physical setting and background details. "
        "Camera perspective (close-up, medium shot, full body, POV, low angle, etc.). "
        "Lens characteristics (shallow depth of field, wide angle, 85mm portrait lens, etc.). "
        "Lighting setup (natural window light, studio softbox, golden hour, rim lighting, etc.). "
        "Color palette and mood. Material textures (skin, fabric, metal, etc.). "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it into a detailed photographic description. "
        "Output the enhanced prompt only."
    ),
    "Visual (Image)": (
        "You are an expert visual artist and photographer with award-winning compositions. "
        "When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition. "
        "Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph. "
        "Start directly with the main subject, and keep descriptions literal and precise. "
        "Think like a photographer describing the perfect shot. "
        "Do not change the user input intent, just enhance it. "
        "For best results, build your prompts using this structure: "
        "Start with main subject and pose in a single sentence. "
        "Add specific details about expressions and positioning. "
        "Describe character/object appearances precisely. "
        "Include background and environment details. "
        "Specify framing, composition and perspective. "
        "Describe lighting, colors, and mood. "
        "Note any atmospheric or stylistic elements. "
        "Output the enhanced prompt only."
    ),
    "Minimalist": (
        "You are a concise prompt editor. "
        "Refine the user's prompt for clarity and impact without expanding it significantly. "
        "Fix awkward phrasing, improve word choices, and tighten the description. "
        "Do not add new concepts, scenes, or details that the user did not imply. "
        "Do not pad with filler adjectives or unnecessary atmosphere. "
        "Keep the enhanced prompt as short as possible while being clear and effective. "
        "If the prompt is already good, return it with minimal changes. "
        "Output the enhanced prompt only."
    ),
}

# ── External presets (loaded from JSON file) ──────────────────────────────────

DEFAULT_PRESETS_PATH = ""
_external_presets = {}


def _find_default_presets_path():
    """Find presets.json from the PROMPT_ENHANCER_PRESETS env var."""
    env_path = os.environ.get("PROMPT_ENHANCER_PRESETS", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    return ""


def _load_external_presets(path):
    """Load presets from a JSON file. Returns dict of name -> system prompt."""
    global _external_presets
    path = (path or "").strip()
    if not path:
        path = _find_default_presets_path()
    if not path or not os.path.isfile(path):
        _external_presets = {}
        return _external_presets
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("presets.json must be a JSON object {name: system_prompt}")
            _external_presets = {}
            return _external_presets
        _external_presets = {k: v for k, v in data.items() if isinstance(v, str)}
        return _external_presets
    except Exception as e:
        logger.error(f"Failed to load external presets from {path}: {e}")
        _external_presets = {}
        return _external_presets


def _get_all_preset_names():
    """Return merged list of preset names: built-in + external + Custom."""
    names = list(BUILTIN_PRESETS.keys())
    if _external_presets:
        names.append("───── External ─────")
        names.extend(_external_presets.keys())
    names.append("Custom")
    return names


def _get_preset_prompt(name):
    """Look up a system prompt by preset name."""
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    if name in _external_presets:
        return _external_presets[name]
    return ""


def _refresh_presets(presets_path, current_preset):
    """Reload external presets from disk and update the dropdown."""
    _load_external_presets(presets_path)
    names = _get_all_preset_names()
    value = current_preset if current_preset in names else names[0]
    return gr.update(choices=names, value=value)


# ── Content modifiers (loaded from modifiers.json next to presets.json) ──────

_content_modifiers = {}


def _find_modifiers_path(presets_path):
    """Find modifiers.json in the same directory as presets.json, or via env var."""
    env_path = os.environ.get("PROMPT_ENHANCER_MODIFIERS", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    presets_path = (presets_path or "").strip() or _find_default_presets_path()
    if presets_path:
        candidate = os.path.join(os.path.dirname(presets_path), "modifiers.json")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _load_content_modifiers(presets_path):
    """Load content modifiers from modifiers.json."""
    global _content_modifiers
    path = _find_modifiers_path(presets_path)
    if not path or not os.path.isfile(path):
        _content_modifiers = {}
        return _content_modifiers
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("modifiers.json must be a JSON object {name: system_prompt}")
            _content_modifiers = {}
            return _content_modifiers
        _content_modifiers = {k: v for k, v in data.items() if isinstance(v, str)}
        return _content_modifiers
    except Exception as e:
        logger.error(f"Failed to load content modifiers: {e}")
        _content_modifiers = {}
        return _content_modifiers


def _get_modifier_names():
    """Return list of modifier names with a 'None' option."""
    names = ["None"]
    names.extend(_content_modifiers.keys())
    return names


def _get_modifier_prompt(name):
    """Look up a content modifier prompt by name."""
    if name == "None" or not name:
        return ""
    return _content_modifiers.get(name, "")


def _refresh_modifiers(presets_path, current_modifier):
    """Reload content modifiers from disk and update the dropdown."""
    _load_content_modifiers(presets_path)
    names = _get_modifier_names()
    value = current_modifier if current_modifier in names else "None"
    return gr.update(choices=names, value=value)


# ── Ollama model management ──────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "huihui_ai/qwen3-abliterated:4b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _fetch_ollama_models(api_url):
    """Fetch available models from the Ollama API."""
    try:
        base = api_url
        for suffix in ("/v1/chat/completions", "/v1", "/"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if not base:
            base = DEFAULT_OLLAMA_BASE

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


# ── Intensity ─────────────────────────────────────────────────────────────────

INTENSITY_LABELS = {
    1: "very restrained and minimal",
    2: "subtle and understated",
    3: "moderate and balanced",
    4: "detailed and vivid",
    5: "intense and highly detailed",
    6: "very intense, graphic, and extreme",
}


def _apply_intensity(system_prompt, intensity):
    """Prepend an intensity instruction to the system prompt."""
    intensity = int(intensity)
    if intensity == 3:
        return system_prompt
    desc = INTENSITY_LABELS.get(intensity, INTENSITY_LABELS[3])
    return (
        f"IMPORTANT: Apply the following style at intensity {intensity}/6 - "
        f"meaning {desc}. "
        f"Scale ALL descriptions to this intensity level.\n\n"
        f"{system_prompt}"
    )


def _apply_word_limit(system_prompt, word_limit):
    """Append a word-limit instruction to the system prompt."""
    word_limit = int(word_limit)
    if word_limit <= 0:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"IMPORTANT: Aim for around {word_limit} words."
    )


# ── Core enhancement logic ───────────────────────────────────────────────────

def _strip_think_blocks(text):
    """Remove <think>...</think> blocks that Qwen3 and similar models emit."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _call_llm(prompt, api_url, model, system_prompt, max_tokens, temperature):
    # Prepend /no_think to disable Qwen3 thinking mode so tokens aren't
    # wasted on internal reasoning that eats into the max_tokens budget.
    user_content = f"/no_think\n{prompt}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    # Strip any thinking blocks that slipped through despite /no_think
    return _strip_think_blocks(content)


def enhance_prompt(source, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, max_tokens_override, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    system_prompt = custom_system_prompt if preset == "Custom" else _get_preset_prompt(preset)
    if not system_prompt:
        return "", "<span style='color:#c66'>No system prompt configured.</span>"

    # Append content modifier if selected
    modifier_prompt = _get_modifier_prompt(content_modifier)
    if modifier_prompt:
        system_prompt = f"{system_prompt}\n\n{modifier_prompt}"

    system_prompt = _apply_intensity(system_prompt, intensity)
    system_prompt = _apply_word_limit(system_prompt, word_limit)

    # Auto-scale tokens from word limit (~3 tokens per word).
    # Use override if set, otherwise auto-calculate.
    max_tokens_override = int(max_tokens_override or 0)
    max_tokens = max_tokens_override if max_tokens_override > 0 else int(word_limit) * 3

    try:
        enhanced = _call_llm(source, api_url, model, system_prompt, max_tokens, temperature)
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

        # Load models, external presets, and content modifiers at startup
        initial_models = _fetch_ollama_models(DEFAULT_API_URL)
        if not initial_models:
            initial_models = [DEFAULT_MODEL]
        _load_external_presets("")
        _load_content_modifiers("")

        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt + enhance ──
            source_prompt = gr.Textbox(
                label="Source Prompt",
                lines=3,
                placeholder="Type your prompt here, then click Enhance...",
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

            # Grab: pull main prompt textarea into source
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

            # ── Preset + generation settings ──
            with gr.Row():
                preset = gr.Dropdown(
                    label="Style Preset",
                    choices=_get_all_preset_names(),
                    value="Cinematic (Video)",
                    scale=2,
                )
                content_modifier = gr.Dropdown(
                    label="Content Modifier",
                    choices=_get_modifier_names(),
                    value="None",
                    scale=1,
                    info="Layered on top of preset",
                )
            with gr.Row():
                intensity = gr.Slider(
                    label="Intensity", minimum=1, maximum=6,
                    value=3, step=1, scale=1,
                    info="1=restrained  3=balanced  6=extreme",
                )
                word_limit = gr.Slider(
                    label="Word Limit", minimum=20, maximum=500,
                    value=300, step=10, scale=1,
                    info="Target length of enhanced prompt",
                )
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=2.0,
                    value=0.7, step=0.05, scale=1,
                )
                max_tokens_override = gr.Number(
                    label="Max Tokens Override",
                    value=0, precision=0, minimum=0,
                    scale=1,
                    info="0 = auto (word limit × 3)",
                )

            custom_system_prompt = gr.Textbox(
                label="Custom System Prompt",
                lines=4,
                visible=False,
                placeholder="Enter your custom system prompt...",
            )

            preset.change(
                fn=lambda p: gr.update(visible=(p == "Custom")),
                inputs=[preset],
                outputs=[custom_system_prompt],
                show_progress=False,
            )

            # ── API + external presets settings ──
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
                presets_path = gr.Textbox(
                    label="External Presets (JSON)",
                    value=_find_default_presets_path(),
                    placeholder="Path to presets.json (optional)",
                    scale=3,
                )
                refresh_presets_btn = gr.Button(
                    value="\U0001f504 Presets",
                    scale=0,
                    min_width=120,
                )
                refresh_models_btn = gr.Button(
                    value="\U0001f504 Models",
                    scale=0,
                    min_width=120,
                )

            refresh_presets_btn.click(
                fn=_refresh_presets,
                inputs=[presets_path, preset],
                outputs=[preset],
                show_progress=False,
            ).then(
                fn=_refresh_modifiers,
                inputs=[presets_path, content_modifier],
                outputs=[content_modifier],
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

            # Enhance: source_prompt -> LLM -> prompt_out + status
            enhance_btn.click(
                fn=enhance_prompt,
                inputs=[source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, max_tokens_override, temperature],
                outputs=[prompt_out, status],
            )

            # When prompt_out gets a value, inject it into the real prompt textarea
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
            (preset, "PE Preset"),
            (intensity, "PE Intensity"),
            (word_limit, "PE WordLimit"),
        ]
        self.paste_field_names = ["PE Source", "PE Preset", "PE Intensity", "PE WordLimit"]
        return [source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, max_tokens_override, temperature]

    def process(self, p, source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, max_tokens_override, temperature):
        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if preset:
            p.extra_generation_params["PE Preset"] = preset
        if intensity and int(intensity) != 3:
            p.extra_generation_params["PE Intensity"] = int(intensity)
        if word_limit and int(word_limit) != 150:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)
