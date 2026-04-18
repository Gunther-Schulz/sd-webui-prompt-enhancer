# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that enhances your prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed description using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** - uses Ollama or any OpenAI-compatible API
- **Three action buttons** - Prose (flowing prompt), Tags (booru tags), Remix (modify existing)
- **Mode checkboxes** - Still (frozen moment), Scene (action over time), Audio (sound cues)
- **130+ categorized modifiers** - organized into multiple dropdowns: Subject, Setting, Lighting & Mood, Visual Style, Camera, Audio
- **Tag generation** - booru-style tags for Illustrious, NoobAI, and Pony models with tag validation
- **Tag validation** - auto-downloaded danbooru tag databases, alias correction, fuzzy matching, 5 validation modes
- **Wildcards** - let the LLM make creative choices (surprise location, unexpected angle, narrative detail, etc.)
- **Inline wildcards** - use `{name?}` placeholders in your prompt for the LLM to fill creatively
- **Remix button** - apply modifiers to an already-enhanced prompt without re-enhancing from scratch
- **Local overrides** - extend with your own modifiers via YAML files; each file becomes a dropdown
- **Tag formats from YAML** - add support for new models by dropping a tag format YAML file
- **Keyword preservation** - style keywords are included verbatim so the image generator recognizes them
- **Auto-retry** - retries once on Ollama connection failure
- **Configurable timeouts** - per-operation timeouts via environment variables
- **Zero VRAM impact** - runs the LLM on CPU by default
- **Works in both txt2img and img2img** tabs
- **Metadata saved** - source prompt, modifiers, and wildcards stored in generated images

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

The 4b variant is not recommended as it produces noticeably lower quality output. Larger models (14b+) work well if you have the RAM but are slower on CPU.

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

1. Open the **Prompt Proser** accordion in the txt2img or img2img tab
2. Type your prompt in the **Source Prompt** box (or click **Grab** to pull from the main prompt area)
3. Optionally check **Still** (image), **Scene** (video), or **Audio** (sound cues)
4. Optionally select modifiers from the categorized dropdowns (Subject, Setting, Lighting & Mood, etc.)
5. Optionally select **Wildcards** for creative LLM choices
6. Click **Prose** for a flowing paragraph, or **Tags** for booru-style tags

### Tag generation

Click **Tags** to generate danbooru-style tags instead of a flowing paragraph:

1. Select a **Tag Format** - Illustrious, NoobAI, or Pony
2. Choose a **Tag Validation** mode (Check recommended)
3. Click **Tags**

Tag databases are automatically downloaded on first use (~2-3 MB per format). Tags are validated, corrected (aliases, common mistakes like `1man` → `1boy`), deduplicated, and reordered into standard danbooru convention.

**Validation modes:**
- **Off** - raw LLM output, no validation
- **Check** - exact match + alias correction, keep unrecognized (default, safe)
- **Fuzzy** - alias + fuzzy string matching, keep unrecognized
- **Strict** - alias only, drop unrecognized tags
- **Fuzzy Strict** - alias + fuzzy matching, drop unrecognized

### Remixing an existing prompt

Already have an enhanced prompt and want to tweak it?

1. Select new modifiers or wildcards, or update the source prompt
2. Click **Remix** instead of Prose
3. The LLM reads the current prompt from the main textarea and integrates the changes without rewriting everything

### Inline wildcards

Use `{name?}` placeholders in your source prompt for the LLM to fill creatively:

```
a woman sitting in a {location?} wearing {outfit?} during {time?}
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Mode | (none) | Checkboxes: Still (frozen moment), Scene (action over time), Audio (sound cues) |
| Base | Default | System prompt template (Default or Custom) |
| Tag Format | Illustrious | Tag output format: Illustrious, NoobAI, Pony (for Tags button) |
| Tag Validation | Check | How to validate generated tags against the database |
| Modifiers | (none) | Multiple categorized dropdowns auto-generated from YAML files |
| Wildcards | (none) | Creative delegation — let the LLM make choices |
| Detail | 0 (auto) | Output length: 0=auto, 1=minimal ... 10=extensive, scales to model |
| Temperature | 0.7 | Creativity (0 = deterministic, 2 = very creative) |
| Think | off | Let model reason before answering (slower) |
| API URL | `http://localhost:11434` | Ollama API endpoint |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | LLM model (auto-detected from Ollama) |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_ENHANCER_LOCAL` | (none) | Comma-separated directories for local modifier overrides |
| `PROMPT_ENHANCER_STALL_TIMEOUT` | 10 | Abort if no tokens received for this many seconds (streaming) |
| `PROMPT_ENHANCER_MAX_TOKENS` | 2000 | Hard cap on output tokens to prevent endless generation |

## Published modifiers

Modifiers are organized into YAML files in the `modifiers/` directory. Each file becomes a dropdown in the UI:

| Dropdown | Categories |
|----------|------------|
| **Subject** | genre, subject, activity, relationship |
| **Setting** | setting, time period, aesthetic |
| **Lighting & Mood** | lighting, mood, atmosphere, emotion |
| **Visual Style** | color, art style, anime (25 sub-styles), cinema style, photography format, vintage format |
| **Camera** | perspective, distance, focus, technique, motion, material |
| **Audio** | ambient types, silence |

## Tag formats

Tag format definitions live in `tag-formats/` as YAML files. Each file defines:

```yaml
system_prompt: |
  (LLM instructions for generating tags)
use_underscores: true
tag_db: illustrious.csv
tag_db_url: https://...
```

Add support for new models by dropping a YAML file — no code changes needed.

## Local overrides

Extend the extension with your own modifiers and base prompts. Each YAML file in a local directory becomes its own dropdown in the UI.

### Setup

Set the `PROMPT_ENHANCER_LOCAL` environment variable to one or more comma-separated directories:

```bash
PROMPT_ENHANCER_LOCAL="/home/user/my-modifiers, /home/user/experimental"
```

The **Local Overrides** field in the UI can refresh content of existing dropdowns. **New files require a Forge restart** to create new dropdowns.

### How it works

Each `.yaml` file becomes a dropdown. The filename determines the dropdown label:

```
/home/user/my-modifiers/
  _bases.yaml        # extends the Base dropdown (underscore prefix = special)
  nsfw.yaml           # creates "Nsfw" dropdown
  my-styles.yaml      # creates "My Styles" dropdown
```

Files with the same name as published ones (e.g., `subject.yaml`) merge their content into the existing dropdown.

### YAML format

All files use the same two-level format — categories containing named keyword strings:

```yaml
# my-styles.yaml — becomes "My Styles" dropdown
my category:
  Cozy Autumn: autumn, warm tones, falling leaves, golden light, wood smoke
  Rainy Tokyo: tokyo streets, neon reflections, rain, umbrellas, night
```

### Base prompt overrides

`_bases.yaml` uses a flat format — name and full system prompt:

```yaml
My Custom Base: |
  You are a Creative Assistant. Given a user's raw input, expand it into
  a detailed prompt for...
```

## How it works

1. **Prose**: source prompt + modifiers + wildcards are assembled into a system prompt, sent to the local LLM, which returns a detailed flowing paragraph
2. **Tags**: source prompt + modifiers are sent as a structured user message, LLM generates booru tags, which are then validated, corrected, deduplicated, and reordered
3. **Remix**: the current enhanced prompt is sent back to the LLM with new modifiers to integrate without rewriting
4. The output is written to the main prompt textbox and settings are saved to image metadata

## License

MIT
