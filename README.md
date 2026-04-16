# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that enhances your prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed, cinematic or visual description using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** - uses any OpenAI-compatible API (Ollama, LM Studio, llama.cpp server, etc.)
- **Style presets** - Cinematic (video), Visual (image), or fully custom system prompts
- **Model selector** - dropdown auto-populated from your Ollama instance, with manual refresh
- **Zero VRAM impact** - runs the LLM on CPU by default; VRAM is freed immediately after enhancement
- **Works in both txt2img and img2img** tabs

## Requirements

- A running OpenAI-compatible LLM server. [Ollama](https://ollama.com/download) is recommended.
- A pulled model, e.g.:
  ```bash
  ollama pull huihui_ai/qwen3-abliterated:4b
  ```

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
2. Select a model from the dropdown (click the refresh button to reload available models)
3. Choose a style preset or write a custom system prompt
4. Type your prompt in the normal prompt box
5. Click **Enhance Prompt**
6. The enhanced prompt replaces your original text - edit further or generate

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| API URL | `http://localhost:11434/v1/chat/completions` | OpenAI-compatible endpoint |
| Model | `huihui_ai/qwen3-abliterated:4b` | LLM model to use |
| Style Preset | Cinematic (Video) | System prompt template |
| Max Tokens | 300 | Maximum response length |
| Temperature | 0.7 | Creativity (0 = deterministic, 2 = very creative) |

## Ollama setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (abliterated = uncensored)
ollama pull huihui_ai/qwen3-abliterated:4b

# Start with CPU-only and immediate VRAM release
OLLAMA_NUM_GPU=0 OLLAMA_KEEP_ALIVE=0 ollama serve
```

### Recommended models

| Model | Size | Notes |
|-------|------|-------|
| `huihui_ai/qwen3-abliterated:4b` | 2.5 GB | Good balance of quality and speed |
| `huihui_ai/qwen3-abliterated:1.7b` | 1.1 GB | Faster, lighter |
| `huihui_ai/qwen3-abliterated:8b` | 5.0 GB | Higher quality |

## How it works

1. Reads your prompt from the prompt textbox
2. Sends it to the LLM with a system prompt that guides the enhancement style
3. The LLM returns an expanded, detailed version of your prompt
4. The enhanced text is written back to the prompt textbox

The system prompts are inspired by [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)'s prompt enhancer.

## License

MIT
