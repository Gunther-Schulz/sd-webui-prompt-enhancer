# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that enhances your prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed description using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** - uses any OpenAI-compatible API (Ollama, LM Studio, llama.cpp server, etc.)
- **Style presets** - Cinematic (video), Z-Image (photorealistic), Visual (image), Minimalist, or fully custom system prompts
- **Content modifiers** - stackable modifiers (Film Noir, Dreamy, Gritty, etc.) layered on top of any preset
- **External presets & modifiers** - load your own presets and modifiers from local JSON files
- **Intensity slider** - control enhancement strength from restrained (1) to extreme (6)
- **Word limit slider** - target output length (20-500 words), with auto-scaled token budget
- **Model selector** - dropdown auto-populated from your Ollama instance, with manual refresh
- **Qwen3 thinking mode handling** - automatically disables thinking mode and strips thinking blocks
- **Zero VRAM impact** - runs the LLM on CPU by default; VRAM is freed immediately after enhancement
- **Works in both txt2img and img2img** tabs
- **Metadata saved** - source prompt, preset, and intensity stored in generated images

## Requirements

You need a running OpenAI-compatible LLM server. [Ollama](https://ollama.com/download) is the easiest option:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model - any Ollama model works
ollama pull huihui_ai/qwen3.5-abliterated:9b

# Start Ollama (CPU-only, no VRAM used)
OLLAMA_NUM_GPU=0 OLLAMA_KEEP_ALIVE=0 ollama serve
```

You can use **any model** available on Ollama - browse models at [ollama.com/library](https://ollama.com/library). The extension auto-detects all models pulled on your Ollama instance and populates the model dropdown. Just click the refresh button to reload the list.

### Recommended models

| Model | Size | Notes |
|-------|------|-------|
| `huihui_ai/qwen3.5-abliterated:9b` | ~6 GB | **Recommended.** Best balance of quality and instruction following |
| `huihui_ai/qwen3.5-abliterated:4B` | ~3 GB | Good quality, faster |
| `huihui_ai/qwen3-abliterated:14b` | ~9 GB | Higher quality, slower |
| `huihui_ai/qwen3-abliterated:4b` | ~3 GB | Lightweight, fast |

"Abliterated" models have refusal behaviors removed, which is useful for unrestricted creative content. Standard (non-abliterated) models work fine for general use.

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
2. Type your prompt in the **Source Prompt** box (or click **Grab from prompt box** to pull it from the main prompt area)
3. Choose a **Style Preset** (or select "Custom" to write your own system prompt)
4. Optionally select one or more **Content Modifiers** to layer on top
5. Adjust **Intensity** and **Word Limit** as needed
6. Click **Enhance**
7. The enhanced prompt is written to the main prompt box

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| API URL | `http://localhost:11434/v1/chat/completions` | OpenAI-compatible endpoint |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | LLM model to use (auto-detected from Ollama) |
| Style Preset | Cinematic (Video) | Base system prompt template |
| Content Modifiers | (none) | Stackable style/tone modifiers layered on top of preset |
| Intensity | 3 (balanced) | Enhancement strength: 1=restrained, 6=extreme |
| Word Limit | 300 | Target output length in words |
| Temperature | 0.7 | Creativity (0 = deterministic, 2 = very creative) |
| Max Tokens Override | 0 (auto) | Override auto-calculated token budget (word limit x 3) |

## External presets and modifiers

You can add your own presets and modifiers via JSON files placed in one or more directories:

1. Set the `PROMPT_ENHANCER_DIRS` environment variable to a colon-separated list of directories, or
2. Enter comma-separated directory paths in the **Preset Directories** field in the UI

Each directory may contain a `presets.json` and/or a `modifiers.json`. Files from all directories are merged — later directories override earlier ones for duplicate keys.

Both files use the same format - a JSON object mapping names to system prompt strings:

```json
{
  "My Custom Style": "You are a ... describe the scene as ...",
  "Another Style": "You are a ... focus on ..."
}
```

External presets appear under a separator in the Style Preset dropdown. External modifiers appear alongside the built-in modifiers and can be stacked together.

## How it works

1. Your source prompt is sent to the local LLM with a system prompt assembled from: style preset + content modifiers + intensity + word limit
2. The LLM returns an expanded, detailed version
3. The enhanced text is written to the main prompt textbox
4. Source prompt, preset, and intensity are saved to generated image metadata

## License

MIT
