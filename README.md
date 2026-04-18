# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that enhances your prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed description using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** - uses Ollama or any OpenAI-compatible API
- **Two base modes** - Still (image) and Scene (video with chronological flow)
- **120+ categorized modifiers** - genre, lighting, mood, emotion, activity, perspective, camera distance, art style, cinema style, audio cues, and more
- **Wildcards** - let the LLM make creative choices (surprise location, unexpected angle, narrative detail, etc.)
- **Refine button** - apply modifiers to an already-enhanced prompt without re-enhancing from scratch
- **Inline wildcards** - use `{name?}` placeholders in your prompt for the LLM to fill creatively
- **Local overrides** - extend with your own modifiers and base prompts via YAML/JSON files in a local directory
- **Keyword preservation** - style keywords are included verbatim in the output so the image generator recognizes them
- **Word limit slider** - target output length (20-500 words)
- **Auto-retry** - retries once on Ollama connection failure (model reload after keep-alive expiry)
- **Zero VRAM impact** - runs the LLM on CPU by default
- **Works in both txt2img and img2img** tabs
- **Metadata saved** - source prompt, base, modifiers, and wildcards stored in generated images

## Requirements

You need [Ollama](https://ollama.com/download) running locally:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull huihui_ai/qwen3.5-abliterated:9b

# Start Ollama (CPU-only, no VRAM used)
OLLAMA_NUM_GPU=0 OLLAMA_KEEP_ALIVE=0 ollama serve
```

### Recommended model

**`huihui_ai/qwen3.5-abliterated:9b`** (~6 GB) - best balance of quality and instruction following. This is the default.

The 4b variant is not recommended as it produces noticeably lower quality enhancements. Larger models (14b+) work well if you have the RAM but are slower on CPU.

"Abliterated" models have refusal behaviors removed, which is useful for unrestricted creative content. Standard models work fine for general use.

## Installation

### From URL (recommended)

1. Open Forge/A1111 WebUI
2. Go to **Extensions** > **Install from URL**
3. Paste: `https://github.com/Gunther-Schulz/sd-webui-prompt-enhancer.git`
4. Click **Install** and restart the WebUI

### Manual

```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/Gunther-Schulz/sd-webui-prompt-enhancer.git
```

Restart the WebUI.

## Usage

1. Open the **Prompt Enhancer** accordion in the txt2img or img2img tab
2. Type your prompt in the **Source Prompt** box (or click **Grab** to pull from the main prompt area)
3. Choose a **Base** - Still for images, Scene for video
4. Optionally select **Modifiers** (categorized: genre, lighting, mood, etc.)
5. Optionally select **Wildcards** for creative LLM choices
6. Adjust **Word Limit** as needed
7. Click **Enhance** - the enhanced prompt is written to the main prompt box

### Refining an existing prompt

Already have an enhanced prompt and want to tweak it?

1. Select new modifiers or wildcards
2. Click **Refine** instead of Enhance
3. The LLM reads the current prompt from the main textarea and integrates the changes without rewriting everything

This lets you iteratively adjust style, mood, location, etc. without starting over.

### Inline wildcards

Use `{name?}` placeholders in your source prompt for the LLM to fill creatively:

```
a woman sitting in a {location?} wearing {outfit?} during {time?}
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Base | Still | Still = image description, Scene = video with chronological flow |
| Modifiers | (none) | Categorized style keywords layered on top of the base prompt |
| Wildcards | (none) | Creative delegation - let the LLM make choices |
| Word Limit | 150 | Target output length in words |
| Temperature | 0.7 | Creativity (0 = deterministic, 2 = very creative) |
| Think | off | Let model reason before answering (slower, may improve quality) |
| API URL | `http://localhost:11434` | Ollama API endpoint |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | LLM model (auto-detected from Ollama) |

## Modifier categories

All modifiers are keyword strings organized by category:

| Category | Examples |
|----------|----------|
| Genre | Studio Portrait, Street, Landscape, Food, Editorial Fashion, ... |
| Lighting | Golden Hour, Candlelight, Neon Glow, Backlit, Volumetric Light |
| Mood | Film Noir, Dreamy, Horror, Gritty, Fantasy |
| Emotion | Melancholy, Joy, Serenity, Tension, Defiance, ... |
| Activity | Dancing, Running, Reading, Embracing, Performing, ... |
| Relationship | Intimate, Romantic, Confrontational, Protective, Eye Contact, ... |
| Setting | Urban, Rural, Underwater, Rooftop, Forest, Coastal, ... |
| Time Period | Medieval, Victorian, 1920s Art Deco, 1970s, Futuristic, ... |
| Subject | Solo, Couple, Group, Hands Detail, ... |
| Material | Leather, Silk, Lace, Metal, Glass, Wet Skin |
| Aesthetic | Japanese, Nordic, Mediterranean, Parisian, Brutalist, ... |
| Perspective | First Person, Low Angle, Bird's Eye, Dutch Angle, Over the Shoulder, ... |
| Technique | Long Exposure, Double Exposure, Tilt-Shift, Macro, Silhouette, ... |
| Color | Black and White, Vintage Film, Cinematic Grade, Pastel, Retro |
| Photography Format | Polaroid, 35mm Film, Medium Format, Disposable Camera, ... |
| Art Style | Oil Painting, Watercolor, Anime, Comic Book, 3D Render, Pixel Art, ... |
| Cinema Style | Blockbuster, Indie Film, Home Video, Wes Anderson, Kubrick, ... |
| Vintage Format | Daguerreotype, Kodachrome, VHS, Early Digital, ... |
| Audio | Add Audio Cues, Ambient City, Rain Sounds, Silence, ... |
| Atmosphere | Fog, Rain, Dust and Haze |
| Motion | Frozen Action |

## Local overrides

You can extend the extension with your own modifiers and base prompts via a local directory. Files in this directory are never committed to git.

### Setup

Point to your local directory via:
- The **Local Overrides Directory** field in the UI, or
- The `PROMPT_ENHANCER_LOCAL` environment variable

### Directory structure

```
/path/to/your/overrides/
  _bases.yaml       # adds to the Base dropdown (special, underscore prefix)
  my-styles.yaml    # adds to the Modifiers dropdown
  another.yaml      # adds to the Modifiers dropdown
  ...
```

- `_bases.yaml` is reserved for base prompt overrides (flat format: `name: prompt`)
- All other `.yaml`/`.yml`/`.json` files are loaded as modifier overrides (categorized format)
- Files are loaded in alphabetical order; later files override earlier ones for duplicate keys
- New categories appear in the dropdown; entries in existing categories are merged

### YAML format for modifiers

```yaml
# my-styles.yaml
my custom category:
  Cozy Autumn: autumn, warm tones, falling leaves, golden light, wood smoke
  Rainy Tokyo: tokyo streets, neon reflections, rain, umbrellas, night

# Extend an existing category
lighting:
  Studio Strobe: studio strobe, hard flash, sharp shadows, commercial lighting
```

See `modifiers.local.example.yaml` in the extension directory for more examples.

### YAML format for base prompts

```yaml
# _bases.yaml
My Custom Base: |
  You are a Creative Assistant. Given a user's raw input, expand it into
  a detailed prompt for...
  (full system prompt here)
```

## How it works

1. Your source prompt is sent to the local LLM with a system prompt assembled from: base + modifier keywords + wildcard instructions + word limit
2. The LLM returns an expanded, detailed version with style keywords preserved verbatim
3. The enhanced text is written to the main prompt textbox
4. Settings are saved to generated image metadata for reproducibility

## License

MIT
