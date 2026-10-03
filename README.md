# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that builds prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed natural-language description using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** — uses [Ollama](https://ollama.com)
- **Enhance** — expands the source prompt into prose, written straight into the main prompt box
- **Remix** — applies an instruction or new modifiers to the prompt that is already in the main prompt box
- **Keep LoRAs** — `<lora:...>` tokens in the main prompt box survive a regeneration
- **Bases** — the system prompt that turns your source into prose. One ships, `Default`: it keeps everything you state and invents what you leave open. Add your own through Local Overrides
- **220+ categorized modifiers** — organized into auto-generated dropdowns, one per YAML file
- **Inline wildcards** — `{name?}` placeholders in your prompt
- **`@` / `@@`** — add to or replace the system prompt from inside the source prompt
- **Local overrides** — extend with your own YAML files; each file becomes a dropdown
- **Streaming** — token streaming with stall detection, thinking detection, repetition detection, and configurable safeguards
- **Cancel button** — abort any running generation
- **Ollama status** — shows version, loaded model, and GPU/CPU mode
- **Metadata** — settings saved to generated images and restored when loading
- **Works in both txt2img and img2img** tabs

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

**`huihui_ai/qwen3.5-abliterated:9b`** (~6 GB) — best balance of quality and instruction following. This is the default.

The 4b variant is not recommended as it produces noticeably lower quality output. Larger models (14b+) work well if you have the RAM but are slower on CPU.

"Abliterated" models have refusal behaviors removed, which is useful for unrestricted creative content. Standard models work fine for general use.

**Known limitation:** Certain combinations of source prompt and modifiers can cause Qwen to enter a repetition loop, generating garbage until the token limit is reached. This shows as a "Truncated" status. If this happens, try removing or changing a modifier — some combinations are simply too complex for a 9B model to synthesize coherently. Larger models handle complex combinations better.

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
2. Type your prompt in the **Source Prompt** box, or leave it empty to let the model invent a scene
3. Optionally pick a **Base** and select modifiers from the categorized dropdowns
4. Click **✨ Enhance**

The result is written into the main prompt box. Nothing is sent to the image model until you press Forge's own Generate button.

### Checkboxes

- **Prepend** — puts your source prompt, verbatim, in front of the generated prose
- **+ Negative** — also asks for a short negative prompt and writes it into the negative prompt box
- **+ Motion and Audio** — adds a short motion and sound tail for image-to-video workflows
- **Keep LoRAs** (on by default) — see below

### Keep LoRAs

Enhance and Remix overwrite the main prompt box. With **Keep LoRAs** on, every `<...>` token found in the box before the run — `<lora:name:0.8>`, `<hypernet:name:1>`, anything between angle brackets — is appended to the new prompt, so a LoRA you are testing stays put while the prose changes. A token the new prompt already contains is not added twice. The tokens go to the end: Forge takes them out of the prompt wherever they sit, so their position carries no meaning.

On Remix, which sends the box's text to the model, the tokens are taken out before the send and put back afterwards, so the model never sees them.

With the checkbox off nothing is carried over. Known limit: prose containing bare comparison brackets (`5 < 10 > 3`) is read as a token.

### Remixing an existing prompt

Already have a prompt in the main prompt box and want to tweak it?

1. Type an instruction in the **Source Prompt** box ("make it night", "add rain"), select modifiers, or both
2. Click **🔀 Remix** instead of Enhance
3. The model reads the current prompt from the main prompt box and patches it rather than rewriting it from scratch

### Random modifiers

Modifiers marked 🎲 ask the model to pick something itself (an artist, a time of day, a colour palette). They are instructions to the model, nothing more: a small model tends to return to the same few choices.

### Inline wildcards

Use `{name?}` placeholders in your source prompt for the LLM to fill creatively:

```
a woman sitting in a {location?} wearing {outfit?} during {time?}
```

### `@` and `@@` — instructions to the enhancer, typed into the source prompt

`@` adds extra instructions for this run, on top of the selected base:

```
a serious woman at a desk @ keep it under three sentences
```

`@@` replaces the whole system prompt — the base and the Motion and Negative blocks are all left out:

```
a woman opens a door @@ Output exactly 6 lines. Each line describes only visible motion.
```

Everything from the sigil on is taken out of the prompt the model is asked to enhance. `@@` is looked for first, and the first occurrence splits. Nothing after an empty `@` or `@@` changes anything, so a half-typed `prompt @@` behaves like `prompt`.

There is no way to escape a literal `@`: `a poster for @midnight` is split at the `@`. Both work on Enhance and on Remix; on Remix the sigil acts on the editor instructions, and your instruction text before the sigil is still applied.

### Cancel

Click **❌ Cancel** to abort any running generation. Works reliably across multiple clicks.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Base | Default | System prompt that sets the prose voice |
| Modifiers | (none) | Multiple categorized dropdowns auto-generated from YAML files |
| Prepend | off | Put the source prompt in front of the result |
| + Negative | off | Also produce a negative prompt |
| + Motion and Audio | off | Add a motion and sound tail |
| Keep LoRAs | on | Carry `<...>` tokens in the main prompt box across a run |
| Temperature | 0.8 | Creativity (0 = deterministic, 2 = creative) |
| Think | off | Let model reason before answering (slower) |
| Seed | -1 (random) | LLM seed. 🎲 resets it to random, ♻ reuses the seed of the last run |
| API URL | `http://localhost:11434` | Ollama API endpoint |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | LLM model (auto-detected from Ollama) |
| Local Overrides | (none) | Comma-separated directories with your own YAML files |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_ENHANCER_LOCAL` | (none) | Comma-separated directories for local modifier overrides |
| `PROMPT_ENHANCER_STALL_TIMEOUT` | 10 | Abort if no tokens received for this many seconds |
| `PROMPT_ENHANCER_MAX_TOKENS` | 4000 | Hard cap on output tokens |
| `PROMPT_ENHANCER_MAX_TIME` | 180 | Hard cap on total generation time in seconds |

## Published modifiers

Modifiers are organized into YAML files in the `modifiers/` directory. Each file becomes a dropdown in the UI:

| Dropdown | Categories |
|----------|------------|
| **Audio** | audio |
| **Camera** | perspective, distance, focus, technique, motion, material |
| **Focus** | focus |
| **Lighting & Mood** | lighting, mood, atmosphere |
| **Narrative** | narrative |
| **Setting** | setting, time period, aesthetic |
| **Subject** | genre, subject, character detail, activity, relationship |
| **Temporal Framing** | temporal framing |
| **Visual Style** | color, art style, style influence, anime, cinema style, photography format, vintage format |

## Local overrides

Extend the extension with your own modifiers and base prompts. Each YAML file in a local directory becomes its own dropdown in the UI.

### Setup

Set the `PROMPT_ENHANCER_LOCAL` environment variable to one or more comma-separated directories:

```bash
PROMPT_ENHANCER_LOCAL="/home/user/my-modifiers, /home/user/experimental"
```

The **Local Overrides** field in the UI can refresh content of existing dropdowns. **New files require a Forge restart** to create new dropdowns.

### How it works

Each `.yaml` file becomes a dropdown. The file's `_label` field is the dropdown label; a file without one is skipped with a console warning:

```
/home/user/my-modifiers/
  _bases.yaml        # extends the Base dropdown (underscore prefix = special)
  _prompts.yaml      # overrides operational prompts
  my-styles.yaml     # creates the dropdown named in its _label
```

A file whose `_label` matches a published dropdown (e.g. `Subject`) merges its content into that dropdown.

### YAML format

All modifier files use the same two-level format — categories containing named entries. An entry is either a keyword string or a mapping with a `behavioral` instruction:

```yaml
# my-styles.yaml
_label: My Styles
my category:
  Cozy Autumn: autumn, warm tones, falling leaves, golden light, wood smoke
  Rainy Tokyo:
    behavioral: "Rainy Tokyo street at night."
    keywords: "tokyo streets, neon reflections, rain, umbrellas, night"
```

An entry carrying a `source:` key (the tag-database lookup of earlier versions) is skipped, with a console line naming it.

### Authoring base prompts and operational prompts

`_bases.yaml` extends or replaces entries in the **Base** dropdown. `_prompts.yaml` overrides the operational prompts (`remix_prose`, `inline_wildcard`, `motion`, `negative`, `empty_source_signal`, `sigil_append`). Both merge with the published defaults; anything you don't override stays.

```yaml
# _bases.yaml
My Custom Base: |
  You are a prompt writer. Given a user's raw input, expand it into a
  detailed scene description...
```

#### How a base is assembled

For every base, the system prompt is:

```
_preamble                  # shared: input handling
your base body             # per-base: style and content rules
_format                    # shared: no headings, no line breaks, no commentary
```

Then these blocks are appended as they apply: `motion` when **+ Motion and Audio** is on, and `negative` (with its `POSITIVE:`/`NEGATIVE:` contract) when **+ Negative** is on. The user message carries `SOURCE PROMPT: ...`, an `Apply these styles to the scene: ...` line for selected modifiers, and the `inline_wildcard` instruction when the source contains `{name?}` placeholders.

An `@` suffix in the source prompt is appended right after `_format`, under the `sigil_append` line, before the blocks above. An `@@` suffix replaces all of it.

Override `_preamble` or `_format` in `_bases.yaml` to change shared behavior for all bases.

#### The label-mirror trap

LLMs heavily mirror the structure of their system prompt. If you describe the output shape using labeled bullets, the model will often echo those labels as section headers in its response:

```yaml
# BAD — the model emits "Patched Prompt:" / "Creative Choice:" in the output
My Base: |
  Output the patched prompt.

  Instruction blocks below may include:
  - Instruction: — free-form text...
  - Creative choice blocks — optional...
```

```yaml
# GOOD — prose rules, no template structure, explicit anti-label coda
My Base: |
  The text below contains free-form directives, style keywords, and
  optional wildcard prompts. Apply directives literally, weave style
  keywords naturally, and treat wildcards as optional.

  Output only the updated prompt as raw text. No section headers, no
  labels, no prefaces like "Patched:" or "Result:".
```

Rules of thumb:

- Describe the LLM's *job* in imperative prose, not the *input/output shape* via labeled templates.
- Avoid echo-bait nouns in your system prompt (words like `patched`, `block`, `section`). If the model is going to invent a header, it picks one it saw in your prompt.
- When output structure matters, end with an explicit anti-label clause naming the concrete bad prefixes — the model avoids exactly what you tell it to avoid.

#### Writing style for base bodies

- Short imperative sentences beat essays. *"Write short, direct sentences."* is more reliable than a paragraph describing desired voice.
- Include concrete contrasts, not abstract rules. *`"thin black leather choker with a small metal ring"` not `"elegant necklace"`* teaches the model more than *"be specific"*.
- Use the same vocabulary you want in the output. If you want `"floorboards creak"`, don't just say *"add ambient sound"* — show the tone.
- Avoid dramatic or marketing language in the rules themselves — the model picks up tone from your examples.

#### Testing your prompts

- Turn on the **Think** checkbox to see the model's reasoning — useful for diagnosing why it picked a particular structure.
- Try an empty source with 🎲 modifiers, conflicting modifiers, and instruction-style source prompts (*"make it darker"*) as edge cases.
- Watch for any words or phrases from your system prompt that appear verbatim in output — that's mirroring, and usually means you need to reframe that section as prose instead of labeled structure.

## How it works

1. **Enhance**: the source prompt is sent to the local LLM with the assembled system prompt, and the selected modifiers in the user message. The LLM returns a description in prose.
2. **Remix**: the current prompt is sent back to the LLM together with your instruction and modifiers, under an editor prompt that asks for a minimal patch.
3. **Streaming**: all LLM calls use streaming with stall detection and thinking mode detection. `/no_think` is prepended to prevent Qwen3 models from entering thinking mode.
4. The output is written to the main prompt textbox and the settings are saved to image metadata.

## License

MIT
