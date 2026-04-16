import json
import logging
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts

logger = logging.getLogger("prompt_enhancer")

PRESETS = {
    "Cinematic (Video)": (
        "You are an expert cinematic director with many award winning movies. "
        "When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes. "
        "Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. "
        "Start directly with the action, and keep descriptions literal and precise. "
        "Think like a cinematographer describing a shot list. "
        "Do not change the user input intent, just enhance it. "
        "Keep within 150 words. "
        "For best results, build your prompts using this structure: "
        "Start with main action in a single sentence. "
        "Add specific details about movements and gestures. "
        "Describe character/object appearances precisely. "
        "Include background and environment details. "
        "Specify camera angles and movements. "
        "Describe lighting and colors. "
        "Note any changes or sudden events. "
        "Do not exceed the 150 word limit! "
        "Output the enhanced prompt only."
    ),
    "Visual (Image)": (
        "You are an expert visual artist and photographer with award-winning compositions. "
        "When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition. "
        "Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph. "
        "Start directly with the main subject, and keep descriptions literal and precise. "
        "Think like a photographer describing the perfect shot. "
        "Do not change the user input intent, just enhance it. "
        "Keep within 150 words. "
        "For best results, build your prompts using this structure: "
        "Start with main subject and pose in a single sentence. "
        "Add specific details about expressions and positioning. "
        "Describe character/object appearances precisely. "
        "Include background and environment details. "
        "Specify framing, composition and perspective. "
        "Describe lighting, colors, and mood. "
        "Note any atmospheric or stylistic elements. "
        "Do not exceed the 150 word limit! "
        "Output the enhanced prompt only."
    ),
    "NSFW (Explicit)": (
        "You are an uncensored adult content writer specializing in vivid, explicit visual descriptions. "
        "When writing prompts based on the user input, focus on detailed, uninhibited descriptions of bodies, poses, expressions, and intimate acts. "
        "Be anatomically precise and sexually explicit. Do not censor, euphemize, or soften any details. "
        "Include specific body descriptions, skin details, facial expressions, physical interactions, and fluid dynamics where relevant. "
        "Include environment, lighting, and camera angle to frame the scene. "
        "Start directly with the subject, keep descriptions literal and precise. "
        "Do not change the user input intent, just enhance it with explicit detail. "
        "Keep within 150 words. "
        "Do not exceed the 150 word limit! "
        "Output the enhanced prompt only. No commentary, no disclaimers."
    ),
    "NSFW (Hardcore)": (
        "You are an uncensored, extreme adult content writer with no limits or boundaries. "
        "When writing prompts based on the user input, describe the most intense, graphic, and hardcore version of the scene possible. "
        "Be maximally explicit about every physical detail: anatomy, penetration, bodily fluids, facial expressions of pleasure or pain, physical strain, and raw intensity. "
        "Use visceral, pornographic language. Do not hold back, do not soften, do not imply - describe everything directly. "
        "Include extreme close-up angles, unflattering realism, sweat, mess, and physicality. "
        "Describe muscle tension, skin flushing, involuntary reactions, and sounds where relevant. "
        "The tone should be raw, unfiltered, and maximally graphic. "
        "Do not change the user input intent, just push it to its most extreme explicit version. "
        "Keep within 150 words. "
        "Do not exceed the 150 word limit! "
        "Output the enhanced prompt only. No commentary, no disclaimers, no moral qualifications."
    ),
    "NSFW (Suggestive)": (
        "You are an artistic director known for sensual, tastefully provocative imagery. "
        "When writing prompts based on the user input, focus on building tension and allure through implication rather than explicitness. "
        "Emphasize body language, lingering gazes, partially revealed skin, suggestive poses, and intimate atmosphere. "
        "Use evocative lighting (warm glow, candlelight, golden hour) and compositions that draw the eye. "
        "Describe textures like silk, lace, wet skin, tousled hair. "
        "The mood should feel intimate and charged without being overtly graphic. "
        "Start directly with the subject, keep descriptions precise but sensual. "
        "Do not change the user input intent, just enhance it with suggestive detail. "
        "Keep within 150 words. "
        "Do not exceed the 150 word limit! "
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
    "Custom": "",
}

DEFAULT_API_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "huihui_ai/qwen3-abliterated:4b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _fetch_ollama_models(api_url):
    """Fetch available models from the Ollama API."""
    try:
        # Derive Ollama base URL from the chat completions URL
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


def _call_llm(prompt, api_url, model, system_prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
    return result["choices"][0]["message"]["content"].strip()


def enhance_prompt(prompt, api_url, model, preset, custom_system_prompt, max_tokens, temperature):
    prompt = (prompt or "").strip()
    if not prompt:
        return "", "No prompt to enhance."

    system_prompt = custom_system_prompt if preset == "Custom" else PRESETS.get(preset, "")
    if not system_prompt:
        return "", "No system prompt configured."

    try:
        enhanced = _call_llm(prompt, api_url, model, system_prompt, max_tokens, temperature)
        # Preserve original prompt as a comment (stripped by Forge before generation)
        result = f"# Original: {prompt}\n{enhanced}"
        word_count = len(enhanced.split())
        return result, f"<span style='color:#6c6'>OK - enhanced to {word_count} words</span>"
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


class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        # Fetch initial model list
        initial_models = _fetch_ollama_models(DEFAULT_API_URL)
        if not initial_models:
            initial_models = [DEFAULT_MODEL]

        with gr.Accordion(open=False, label="Prompt Enhancer"):
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
                refresh_btn = gr.Button(
                    value="\U0001f504 Refresh Models",
                    scale=0,
                    min_width=120,
                    elem_id=f"{tab}_pe_refresh_btn",
                )

            refresh_btn.click(
                fn=_refresh_models,
                inputs=[api_url, model],
                outputs=[model],
                show_progress=False,
            )

            with gr.Row():
                preset = gr.Dropdown(
                    label="Style Preset",
                    choices=list(PRESETS.keys()),
                    value="Cinematic (Video)",
                    scale=1,
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=64, maximum=1024,
                    value=600, step=32, scale=1,
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=2.0,
                    value=0.7, step=0.05, scale=1,
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

            with gr.Row():
                enhance_btn = gr.Button(
                    value="\U0001f4a1 Enhance",
                    variant="primary",
                    scale=0,
                    min_width=120,
                    elem_id=f"{tab}_pe_enhance_btn",
                )
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            # Hidden bridge: JS reads the real prompt textarea into this,
            # then Python processes it and writes the result back via JS.
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")

            # Click: JS grabs prompt text -> all values go to Python -> result to prompt_out + status
            enhance_btn.click(
                fn=enhance_prompt,
                _js=f"""function(prompt_in, api_url, model, preset, custom, max_tokens, temp) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    return [ta ? ta.value : '', api_url, model, preset, custom, max_tokens, temp];
                }}""",
                inputs=[prompt_in, api_url, model, preset, custom_system_prompt, max_tokens, temperature],
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

        self.infotext_fields = []
        return [api_url, model, preset, custom_system_prompt, max_tokens, temperature]
