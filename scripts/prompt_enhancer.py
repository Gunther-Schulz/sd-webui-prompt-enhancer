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
    "Z-Image (Amateur)": (
        "You are describing a photo taken by an amateur photographer for a photorealistic AI image generator "
        "that understands natural language. Write a single flowing paragraph. "
        "The image should feel like it was taken by someone learning photography — not bad, but imperfect. "
        "Include common amateur characteristics: slightly off composition (subject not perfectly centered or "
        "framed), mixed or available lighting (on-camera flash, harsh overhead light, uneven natural light), "
        "busy or cluttered backgrounds that a pro would avoid. "
        "The camera is a consumer DSLR or mirrorless — decent but not top-end. "
        "Depth of field is often too deep (everything in focus) or awkwardly shallow. "
        "Colors are as-shot, not color-graded — slightly flat or oversaturated from auto settings. "
        "Poses feel directed but stiff, or genuinely candid and slightly awkward. "
        "The charm is in the authenticity — real moments, real places, real imperfections. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Z-Image (Outdoor)": (
        "You are a photographer writing shot descriptions for outdoor imagery for a photorealistic AI image generator. "
        "Write a single flowing paragraph. The scene takes place outside — beach, park, rooftop, balcony, "
        "garden, street, forest trail, urban alley. Describe the natural environment in detail: sunlight direction "
        "and quality (harsh midday, golden hour, overcast diffusion), wind effects on hair and fabric, "
        "ground textures (sand, grass, concrete, gravel). Include natural shadows, how light interacts with the "
        "subject outdoors. Camera angles should use the environment — shooting from below against sky, framed by "
        "architecture or foliage, wide establishing shots that show the setting. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Z-Image (Mirror/Reflection)": (
        "You are a photographer specializing in mirror and reflection compositions, writing shot descriptions "
        "for a photorealistic AI image generator. Write a single flowing paragraph. The scene uses reflective "
        "surfaces as a core compositional element — vanity mirrors, bathroom mirrors, full-length mirrors, "
        "mirrored walls, reflective windows at night. Describe what the camera sees directly AND what's visible "
        "in the reflection, creating dual perspectives in one frame. Include the subject's relationship with "
        "their reflection — watching themselves, caught mid-glance, eyes meeting their own gaze, or looking at "
        "the camera while the mirror shows another angle. Describe the frame-within-a-frame composition: mirror "
        "edges, the room visible around and behind. Lighting should work across both planes — how light falls on "
        "the subject versus how it appears in the reflection. "
        "Write in flowing sentences, not tags. Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Z-Image (Retro/Vintage)": (
        "You are a photographer shooting in a retro analog style, writing shot descriptions for a photorealistic "
        "AI image generator. Write a single flowing paragraph. The image should feel like it was shot on film — "
        "specify the era and film stock aesthetic: 70s Kodachrome warmth, 80s Polaroid with washed highlights, "
        "90s disposable camera flash, faded Fujifilm greens. Include analog imperfections: grain, light leaks, "
        "slight color shifts, soft focus at edges, lens flare. Describe era-appropriate styling: hairstyles, "
        "clothing styles, furniture, wallpaper, decor of the period. Lighting should match the aesthetic — "
        "on-camera flash harsh shadows, tungsten indoor warmth, overexposed window light. Composition should feel "
        "period-authentic — snapshot framing, slightly off-center, the casual imperfection of amateur photography "
        "from that era. Write in flowing sentences, not tags. "
        "Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Z-Image (Editorial/Fashion)": (
        "You are a high-fashion photographer in the style of Helmut Newton or Mario Testino, writing shot "
        "descriptions for a photorealistic AI image generator. Write a single flowing paragraph. The scene should "
        "feel like an editorial spread — stylized, deliberate, powerful. Poses should be commanding and "
        "architectural, not casual. Include fashion elements: designer pieces, strategic accessories, shoes. "
        "Lighting should be dramatic and precise — hard key light with sharp shadows, beauty dish, controlled "
        "studio setups or striking location light. Describe the set design or location: minimalist interiors, "
        "urban architecture, luxury hotels, stark white studios. The tone is cold confidence — the subject "
        "commands the frame. Composition should be clean and graphic with strong lines. Skin should look flawless "
        "and editorial. Write in flowing sentences, not tags. "
        "Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Street Photography": (
        "You are a street photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. Capture a candid urban moment — the decisive instant. "
        "Use a 35mm or 50mm lens perspective with moderate depth of field. The subject is caught unposed "
        "in their environment: walking, waiting, working, reacting. Include the surrounding street context — "
        "pedestrians, signage, vehicles, architecture. Lighting is whatever the street gives you — harsh "
        "midday sun, overcast flat light, dappled shade, neon at night. Composition should feel spontaneous "
        "but purposeful — a glance, a gesture, a juxtaposition that tells a story. The tone is observational "
        "and honest. Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Product / Still Life": (
        "You are a product photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. The subject is an object or arrangement — isolated, controlled, "
        "and precisely lit. Use a clean background (seamless white, gradient, or contextual surface like "
        "marble or wood). Lighting is deliberate: softbox key light, rim light for edge definition, "
        "fill to control shadows. Include material qualities — reflections on glass, texture on fabric, "
        "sheen on metal, condensation on cold surfaces. Camera angle is typically slightly elevated or "
        "eye-level with shallow depth of field. Composition is clean and commercial — the object is the hero. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Landscape / Nature": (
        "You are a landscape photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. Capture the grandeur of a natural scene — mountains, coastlines, "
        "forests, deserts, plains. Use wide-angle perspective with deep depth of field (everything sharp). "
        "Include foreground interest (rocks, flowers, water) leading to a strong midground and background. "
        "Describe the sky in detail — cloud formations, light quality, color gradients. Lighting is key: "
        "golden hour warmth, blue hour cool tones, dramatic storm light, or flat overcast for saturated greens. "
        "Composition should follow rule of thirds or leading lines. Include the sense of scale and atmosphere. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Portrait (Studio)": (
        "You are a portrait photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. The subject is photographed in a controlled studio environment "
        "against a solid or gradient background. Use classic portrait lighting: Rembrandt (triangle shadow "
        "on cheek), butterfly (shadow under nose), split (half-lit face), or broad/short lighting. "
        "Include modifier details: softbox, beauty dish, reflector fill, hair light. Frame as headshot, "
        "head-and-shoulders, or three-quarter. Describe the subject's expression, eye direction, and posture. "
        "Depth of field is shallow — subject sharp, background smooth. Skin texture is visible but flattering. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Portrait (Environmental)": (
        "You are a portrait photographer writing environmental portrait descriptions for a photorealistic AI "
        "image generator. Write a single flowing paragraph. The subject is shown in their natural context — "
        "their workplace, home, studio, neighborhood, or a location that tells their story. "
        "The environment is as important as the subject: include specific details of the space (tools, "
        "furniture, decor, clutter, signage). Use available or supplemented natural light. The subject's "
        "pose should feel natural to the setting — leaning on a counter, sitting at a desk, standing in a "
        "doorway. Depth of field is moderate — subject sharp, environment readable but slightly soft. "
        "The image should feel like a magazine feature — revealing character through context. "
        "Write in flowing sentences, not tags. Do not change the user input intent, just enhance it. "
        "Output the enhanced prompt only."
    ),
    "Architectural": (
        "You are an architectural photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. Focus on buildings, interiors, or structural details. "
        "Use wide-angle or tilt-shift perspective with corrected verticals (no converging lines). "
        "Describe geometric patterns, repeating elements, symmetry, leading lines, and structural rhythm. "
        "Lighting is critical: how daylight enters through windows, how artificial light defines spaces, "
        "how shadows create depth on facades. Include material textures — concrete, glass, steel, brick, "
        "wood. The space should feel intentional and composed. Human figures, if present, provide scale. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Sports / Action": (
        "You are a sports photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. Capture the peak moment of athletic action — the apex of a jump, "
        "the moment of impact, the strain of maximum effort. Use a fast shutter speed to freeze motion sharply. "
        "Include a telephoto compression look with shallow depth of field — subject razor-sharp against a "
        "creamy bokeh crowd or field. Describe muscle tension, facial expression of exertion, sweat, dirt, "
        "or water spray. Lighting is typically harsh and directional — stadium lights, outdoor sun. "
        "Composition is tight — cropped to the action, often off-center with implied direction of movement. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Food Photography": (
        "You are a food photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. The subject is a dish, ingredient, or table spread. "
        "Use overhead flat-lay or 45-degree angles — the two classic food perspectives. "
        "Lighting is soft and directional — side window light or a single diffused source creating gentle "
        "shadows that give dimension. Include texture details: glossy sauces, crispy edges, steam rising, "
        "crumb scatter, condensation on glasses. Style the scene with complementary props — linen napkins, "
        "wooden boards, herbs, utensils. Colors should feel appetizing — warm, saturated, fresh. "
        "Depth of field is shallow, focusing on the hero element. "
        "Do not use comma-separated tags. Write in complete descriptive sentences. "
        "Do not change the user input intent, just enhance it. Output the enhanced prompt only."
    ),
    "Wildlife": (
        "You are a wildlife photographer writing shot descriptions for a photorealistic AI image generator. "
        "Write a single flowing paragraph. The subject is an animal in its natural habitat — observed, not "
        "staged. Use a telephoto lens perspective with significant background compression and creamy bokeh. "
        "The camera is at eye-level with the animal. Describe the animal's pose, behavior, and expression: "
        "alert ears, focused eyes, mid-stride, feeding, resting. Include the natural environment — "
        "undergrowth, water, branches, terrain. Lighting is natural: early morning gold, dappled forest "
        "light, overcast soft light. The image should convey patience and observation — a moment of "
        "connection with a wild subject. Do not use comma-separated tags. Write in complete descriptive "
        "sentences. Do not change the user input intent, just enhance it. Output the enhanced prompt only."
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

# ── External presets (loaded from JSON files in multiple directories) ──────────

DEFAULT_PRESET_DIRS = ""
_external_presets = {}


def _get_default_dirs():
    """Get default preset directories from PROMPT_ENHANCER_DIRS env var.

    Returns a colon-separated string of directory paths.
    """
    return os.environ.get("PROMPT_ENHANCER_DIRS", "")


def _parse_dirs(dirs_str):
    """Parse a comma-or-colon-separated string into a list of directory paths."""
    if not dirs_str:
        return []
    # Support both comma and colon as separators
    parts = re.split(r"[,:]", dirs_str)
    return [p.strip() for p in parts if p.strip()]


def _load_json_from_dirs(dirs_str, filename):
    """Load and merge a named JSON file from multiple directories.

    Later directories override earlier ones for duplicate keys.
    Returns merged dict of name -> value (strings only).
    """
    merged = {}
    for d in _parse_dirs(dirs_str):
        path = os.path.join(d, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.error(f"{path} must be a JSON object {{name: value}}")
                continue
            merged.update({k: v for k, v in data.items() if isinstance(v, str)})
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    return merged


def _load_external_presets(dirs_str):
    """Load and merge presets.json from all preset directories."""
    global _external_presets
    dirs_str = (dirs_str or "").strip() or _get_default_dirs()
    _external_presets = _load_json_from_dirs(dirs_str, "presets.json")
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


def _refresh_presets(preset_dirs, current_preset):
    """Reload external presets from disk and update the dropdown."""
    _load_external_presets(preset_dirs)
    names = _get_all_preset_names()
    value = current_preset if current_preset in names else names[0]
    return gr.update(choices=names, value=value)


# ── Content modifiers (built-in + external from modifiers.json) ──────────────

BUILTIN_MODIFIERS = {
    "Black & White": (
        "Additionally, describe the scene as a black and white photograph. "
        "Emphasize tonal contrast, shadows, highlights, and texture over color. "
        "Mention the absence of color explicitly. Think in terms of luminance, "
        "grain, and the interplay of light and dark."
    ),
    "Film Noir": (
        "Additionally, apply a film noir aesthetic. Use dramatic chiaroscuro lighting "
        "with deep shadows and sharp highlights. Include rain-slicked surfaces, venetian "
        "blind shadows, low-key lighting, and a moody, suspenseful atmosphere. "
        "The tone should feel dangerous and stylized."
    ),
    "Dreamy / Ethereal": (
        "Additionally, give the scene a dreamy, ethereal quality. Use soft focus, "
        "hazy light, lens flare, and pastel or desaturated tones. The atmosphere "
        "should feel weightless and otherworldly — like a half-remembered memory "
        "or a waking dream."
    ),
    "Gritty / Urban": (
        "Additionally, apply a gritty urban aesthetic. Include harsh fluorescent or "
        "sodium vapor lighting, concrete textures, graffiti, worn surfaces, and visual "
        "noise. The mood should feel raw, unpolished, and street-level. Think documentary "
        "photography in rough neighborhoods."
    ),
    "Horror / Dark": (
        "Additionally, shift the tone to horror. Use unsettling lighting — underlit faces, "
        "sickly color casts, deep impenetrable shadows. Include visual unease: slightly wrong "
        "proportions, uncanny stillness, implied threat. The atmosphere should feel dreadful "
        "and oppressive."
    ),
    "Fantasy / Painterly": (
        "Additionally, describe the scene as if it were a fantasy painting. Use rich, "
        "saturated colors, dramatic composition, and theatrical lighting. Include elements "
        "that feel mythic or storybook — golden light, impossible architecture, fabric "
        "that flows dramatically. The style should evoke concept art or classical oil painting."
    ),
    "Vintage Film": (
        "Additionally, apply a vintage analog film look. Include film grain, slight color "
        "shifts, warm highlights, and faded blacks. The image should feel like it was shot "
        "on expired film stock — nostalgic, imperfect, and textured."
    ),
    "Macro / Close-up": (
        "Additionally, describe the scene from an extreme close-up or macro perspective. "
        "Focus on fine details invisible at normal distance: skin pores, fabric weave, "
        "water droplets, individual hairs, dust particles. Use very shallow depth of field "
        "with most of the frame softly blurred."
    ),
    "Golden Hour": (
        "Additionally, set the scene during golden hour. Bathe everything in warm amber "
        "and honey-toned light. Include long dramatic shadows, sun flare, and that soft "
        "directional warmth that makes skin glow and edges shimmer. The sky should be "
        "rich with color — peach, gold, soft pink."
    ),
    "Neon / Cyberpunk": (
        "Additionally, apply a neon cyberpunk aesthetic. Include colored neon reflections "
        "on wet or glossy surfaces — blue, pink, purple, magenta. The environment should "
        "feel urban and nocturnal: rain-slicked streets, glowing signs, LED strips. "
        "Contrast deep shadows with vivid artificial color."
    ),
    "Candlelight": (
        "Additionally, light the scene entirely by candlelight. Use warm flickering light "
        "that creates deep, moving shadows and pools of golden illumination. Faces and "
        "surfaces are partially lit, partially lost in darkness. The mood is intimate, "
        "quiet, and warm."
    ),
    "Long Exposure": (
        "Additionally, describe the scene as a long exposure photograph. Include motion "
        "blur on moving elements — light trails, silky water, streaked clouds, ghosted "
        "figures. Stationary elements remain sharp. The contrast between frozen and "
        "blurred creates a sense of time passing within a single frame."
    ),
    "Tilt-Shift / Miniature": (
        "Additionally, apply a tilt-shift miniature effect. Use extreme selective focus "
        "with a very narrow band of sharpness, making the scene look like a tiny model "
        "or diorama. Top-down or elevated angles enhance the effect. Colors should be "
        "slightly oversaturated to reinforce the toy-like quality."
    ),
    "Double Exposure": (
        "Additionally, describe the scene as a double exposure photograph. Two images "
        "are blended into one — a portrait filled with a landscape, a silhouette "
        "containing a cityscape, or overlapping textures that create a ghostly composite. "
        "Describe both layers and how they interact: which elements show through, where "
        "they align, and where they contrast."
    ),
    "Foggy / Misty": (
        "Additionally, fill the scene with fog or mist. Reduce visibility and soften "
        "distant elements into vague silhouettes. Light diffuses through the haze, "
        "creating god rays or glowing halos. Foreground elements emerge sharply from "
        "the murk. The atmosphere should feel quiet, isolated, and mysterious."
    ),
    "Rainy": (
        "Additionally, set the scene in rain. Include wet surfaces with reflections, "
        "streaked windows, visible raindrops in the air or on skin, puddles mirroring "
        "lights and shapes. Describe the specific quality of rain light — overcast, "
        "diffused, with a cool blue-grey cast. Include the texture of wet fabric, "
        "matted hair, and glistening surfaces."
    ),
    "Dusty / Hazy": (
        "Additionally, fill the scene with dust, haze, or smoke. Include visible "
        "particles caught in light beams, diffused rays, and a warm atmospheric glow. "
        "The air itself should feel thick and textured — desert heat shimmer, smoky "
        "room haze, or construction dust catching afternoon sun."
    ),
    "Cinematic Color Grade": (
        "Additionally, apply a cinematic color grade. Use the classic teal-and-orange "
        "split tone, with crushed blacks, lifted shadows, and a slight desaturation "
        "in midtones. The image should feel like a still from a Hollywood film — "
        "polished, moody, and color-separated."
    ),
    "Soft / Pastel": (
        "Additionally, apply a soft pastel aesthetic. Use desaturated, airy colors — "
        "blush pink, powder blue, lavender, mint, cream. Highlights should be lifted "
        "and whites should feel bright and clean. The overall tone is gentle, light, "
        "and delicate — Wes Anderson-adjacent."
    ),
    "In Motion": (
        "Additionally, capture the subject in dynamic motion. Include motion blur, "
        "frozen action mid-movement, flying hair or fabric, splashing water, or "
        "airborne particles. The pose should feel caught between moments — unstable, "
        "energetic, full of kinetic force. A fast shutter freezes the peak of action."
    ),
    "Silhouette": (
        "Additionally, render the subject as a silhouette. The subject is backlit — "
        "a dark outline against a bright background (sunset, window, neon, fire). "
        "Internal detail is lost; the image is about shape, outline, and the contrast "
        "between the dark figure and the luminous background behind them."
    ),
    "First Person / POV": (
        "Additionally, describe the scene entirely from a first-person perspective. "
        "The camera IS the viewer's eyes. Describe what 'you' see directly in front of you. "
        "Include your own hands, arms, or body in frame where relevant. Use spatial language "
        "relative to the viewer: 'in front of you', 'looking down', 'at arm's length'. "
        "The scene should feel immersive and present-tense — you are there."
    ),
}

_external_modifiers = {}


def _load_content_modifiers(dirs_str):
    """Load and merge modifiers.json from all preset directories."""
    global _external_modifiers
    dirs_str = (dirs_str or "").strip() or _get_default_dirs()
    _external_modifiers = _load_json_from_dirs(dirs_str, "modifiers.json")
    return _external_modifiers


def _get_modifier_names():
    """Return merged list of modifier names: built-in + external."""
    names = list(BUILTIN_MODIFIERS.keys())
    if _external_modifiers:
        names.append("───── External ─────")
        names.extend(_external_modifiers.keys())
    return names


def _get_modifier_prompt(name):
    """Look up a content modifier prompt by name."""
    if not name:
        return ""
    if name in BUILTIN_MODIFIERS:
        return BUILTIN_MODIFIERS[name]
    if name in _external_modifiers:
        return _external_modifiers[name]
    return ""


def _refresh_modifiers(preset_dirs, current_modifiers):
    """Reload content modifiers from disk and update the dropdown."""
    _load_content_modifiers(preset_dirs)
    names = _get_modifier_names()
    # Keep only still-valid selections
    value = [m for m in (current_modifiers or []) if m in names]
    return gr.update(choices=names, value=value)


# ── Ollama model management ──────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


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


def _to_ollama_base(api_url):
    """Convert an OpenAI-compatible URL to the Ollama base URL."""
    base = api_url
    for suffix in ("/v1/chat/completions", "/v1", "/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or DEFAULT_OLLAMA_BASE


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
    # Safety net: strip any thinking blocks that slipped through
    return _strip_think_blocks(content)


def enhance_prompt(source, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, think, temperature):
    source = (source or "").strip()
    if not source:
        return "", "<span style='color:#c66'>Source prompt is empty.</span>"

    system_prompt = custom_system_prompt if preset == "Custom" else _get_preset_prompt(preset)
    if not system_prompt:
        return "", "<span style='color:#c66'>No system prompt configured.</span>"

    # Append all selected content modifiers
    for mod_name in (content_modifier or []):
        mod_prompt = _get_modifier_prompt(mod_name)
        if mod_prompt:
            system_prompt = f"{system_prompt}\n\n{mod_prompt}"

    system_prompt = _apply_intensity(system_prompt, intensity)
    system_prompt = _apply_word_limit(system_prompt, word_limit)

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
                    label="Content Modifiers",
                    choices=_get_modifier_names(),
                    value=[],
                    multiselect=True,
                    scale=2,
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
                think = gr.Checkbox(
                    label="Think",
                    value=False,
                    info="Let model reason before answering (slower)",
                    scale=0, min_width=80,
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
                preset_dirs = gr.Textbox(
                    label="Preset Directories",
                    value=_get_default_dirs(),
                    placeholder="Comma-separated paths to directories containing presets.json / modifiers.json",
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
                inputs=[preset_dirs, preset],
                outputs=[preset],
                show_progress=False,
            ).then(
                fn=_refresh_modifiers,
                inputs=[preset_dirs, content_modifier],
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
                inputs=[source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, think, temperature],
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
            (content_modifier, lambda params: [m.strip() for m in params.get("PE Modifiers", "").split(",") if m.strip()] if params.get("PE Modifiers") else []),
            (think, "PE Think"),
        ]
        self.paste_field_names = ["PE Source", "PE Preset", "PE Intensity", "PE WordLimit", "PE Modifiers", "PE Think"]
        return [source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, think, temperature]

    def process(self, p, source_prompt, api_url, model, preset, custom_system_prompt, content_modifier, intensity, word_limit, think, temperature):
        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if preset:
            p.extra_generation_params["PE Preset"] = preset
        if intensity and int(intensity) != 3:
            p.extra_generation_params["PE Intensity"] = int(intensity)
        if word_limit and int(word_limit) != 300:
            p.extra_generation_params["PE WordLimit"] = int(word_limit)
        if content_modifier:
            p.extra_generation_params["PE Modifiers"] = ", ".join(content_modifier)
        if think:
            p.extra_generation_params["PE Think"] = True
