# sd-webui-prompt-enhancer

A Stable Diffusion WebUI extension that builds prompts using a local LLM. Works with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), Forge, and AUTOMATIC1111.

Takes your short prompt and expands it into a detailed description, booru-style tags, or a hybrid of both using a locally-running language model. No cloud APIs, no data leaves your machine.

## Features

- **Local LLM powered** — uses Ollama or any OpenAI-compatible API
- **Four generation modes** — Prose (flowing paragraph), Hybrid (tags + NL supplement), Tags (booru tags), Remix (modify existing)
- **Mode modifiers** — Still (frozen moment), Scene (action over time), Audio (sound cues) — available in the Mode dropdown alongside other modifiers
- **130+ categorized modifiers** — organized into auto-generated dropdowns: Subject, Setting, Lighting & Mood, Visual Style, Camera, Audio
- **Tag generation & validation** — Illustrious, NoobAI, Pony, and **Anima** (retrieval-augmented) formats with auto-downloaded danbooru databases, alias correction, fuzzy matching, deduplication, and standard tag ordering
- **Tag post-processing** — strips LLM meta-annotations, converts hyphens, escapes parentheses for SD, prefix-matches danbooru disambiguation suffixes
- **Wildcards** — creative LLM choices: surprise location, random artist, anime era, narrative detail, and more
- **Inline wildcards** — `{name?}` placeholders in your prompt
- **Local overrides** — extend with your own YAML files; each file becomes a dropdown
- **Extensible tag formats** — add new model support by dropping a YAML file
- **Detail slider** — scales output length to the active image model (SD/SDXL/Flux/Z-Image)
- **Streaming** — real-time token streaming with stall detection, thinking detection, and configurable safeguards
- **Cancel button** — abort any running generation
- **Ollama status** — shows version, loaded model, and GPU/CPU mode
- **Metadata** — all settings saved to generated images and restored when loading
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

**Known limitation:** Certain combinations of source prompt, modifiers, and wildcards can cause Qwen to enter a repetition loop, generating garbage until the token limit is reached. This shows as a "Truncated" status. If this happens, try removing or changing a modifier — some combinations are simply too complex for a 9B model to synthesize coherently. Larger models handle complex combinations better.

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
2. Type your prompt in the **Source Prompt** box
3. Optionally select modifiers from the categorized dropdowns (Mode, Subject, Setting, Lighting & Mood, etc.)
5. Optionally select **Wildcards** for creative LLM choices
6. Choose a generation mode:

### Generation modes

**Prose** — Click **✍ Prose** for a flowing paragraph. Best for Flux, SD3, and other natural-language models.

**Hybrid** — Click **✨ Hybrid** for danbooru-style tags followed by a short NL description. Three-pass pipeline: (1) generates rich prose with wildcards/modifiers, (2) extracts tags from that prose using the selected tag format, (3) summarizes the prose into 1-2 compositional sentences. Best for Illustrious, NoobAI, and Pony — follows the community-recommended "tags + NL" format where tags provide precise control and the NL supplement captures spatial relationships, lighting, and mood that tags alone cannot express.

**Tags** — Click **🏷 Tags** for pure booru-style tags. Two-pass pipeline: (1) generates rich prose (same as Prose mode, with wildcards/modifiers), (2) extracts tags from that prose using the selected tag format. The prose pass gives the LLM room to reason about the scene before compressing it to tags — producing richer, more coherent tag lists than asking for tags directly. Tags are post-processed through validation, correction, reordering, and paren escaping.

### Tag generation

Both **Hybrid** and **Tags** modes use the tag post-processing pipeline:

1. Select a **Tag Format** — Illustrious, NoobAI, Pony, or **Anima** (recommended)
2. Choose a **Tag Validation** mode (RAG recommended with Anima)

Tag databases are automatically downloaded on first use (~2-3 MB per format; ~1.1 GB for Anima's FAISS index). Tags are validated, corrected (aliases, common mistakes like `1man` → `1boy`), deduplicated, and reordered into standard danbooru convention. Parentheses in disambiguation suffixes (e.g., `artist_(style)`) are automatically escaped for SD.

**Validation modes:**
- **RAG** — retrieval + embedding validator (Anima format only) — shortlist of real artists/characters/series injected into system prompt + every drafted tag checked against FAISS index. Default.
- **Fuzzy Strict** — alias + fuzzy matching, drop unrecognized
- **Fuzzy** — alias + fuzzy string matching, keep unrecognized
- **Off** — raw LLM output, no validation

On truncation (Ollama hit the token or time budget), tag-mode outputs fail loud: empty textbox, red "Truncated — no output (retry)" status. A reduced partial result would look like success but silently missing content; the retry path is more honest.

### Remixing an existing prompt

Already have an enhanced prompt and want to tweak it?

1. Select new modifiers or wildcards, or update the source prompt
2. Click **🔀 Remix** instead of Prose
3. The LLM reads the current prompt from the main textarea and integrates the changes without rewriting everything
4. Remix auto-detects whether the existing prompt is prose, tags, or hybrid format and applies the appropriate system prompt and post-processing

### Inline wildcards

Use `{name?}` placeholders in your source prompt for the LLM to fill creatively:

```
a woman sitting in a {location?} wearing {outfit?} during {time?}
```

### Cancel

Click **❌ Cancel** to abort any running generation. Works reliably across multiple clicks.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Mode | (none) | Dropdown: Still (frozen moment), Scene (action over time), Audio (sound cues) |
| Base | Default | System prompt template (Default or Custom) |
| Tag Format | Illustrious | Tag output format: Illustrious, NoobAI, Pony, Anima (for Tags and Hybrid buttons) |
| Tag Validation | RAG | How to validate generated tags: RAG (Anima only), Fuzzy Strict, Fuzzy, Off |
| Modifiers | (none) | Multiple categorized dropdowns auto-generated from YAML files |
| Wildcards | (none) | Creative delegation — let the LLM make choices |
| Detail | 0 (auto) | Output length: 0=auto, 1=minimal ... 10=extensive, scales to model |
| Temperature | 0.8 | Creativity (0 = deterministic, 2 = creative) |
| Think | off | Let model reason before answering (slower) |
| Seed | -1 (random) | LLM seed for reproducibility. Fixed seed = same output for same input |
| API URL | `http://localhost:11434` | Ollama API endpoint |
| Model | `huihui_ai/qwen3.5-abliterated:9b` | LLM model (auto-detected from Ollama) |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_ENHANCER_LOCAL` | (none) | Comma-separated directories for local modifier overrides |
| `PROMPT_ENHANCER_STALL_TIMEOUT` | 10 | Abort if no tokens received for this many seconds |
| `PROMPT_ENHANCER_MAX_TOKENS` | 1000 | Hard cap on output tokens |
| `PROMPT_ENHANCER_MAX_TIME` | 60 | Hard cap on total generation time in seconds |

## Published modifiers

Modifiers are organized into YAML files in the `modifiers/` directory. Each file becomes a dropdown in the UI:

| Dropdown | Categories |
|----------|------------|
| **Mode** | mode (Still, Scene, Audio) |
| **Subject** | genre, subject, activity, relationship |
| **Setting** | setting, time period, aesthetic |
| **Lighting & Mood** | lighting, mood, atmosphere, emotion |
| **Visual Style** | color, art style, anime (25+ sub-styles), cinema style, photography format, vintage format |
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

### Anima — retrieval-augmented tag pipeline

The **Anima** tag format uses a richer pipeline than the rapidfuzz-based
validation of the other formats:

- **Shortlist retrieval**: before prose generation, real Danbooru
  artists / characters / series that match the source prompt are
  pre-retrieved and injected into the LLM's system prompt. Prevents
  hallucinated names like `@takashi_murowo` at the source.
- **Embedding-based validator**: every LLM-drafted tag is checked
  against a 273k-entry FAISS index using bge-m3 embeddings. Real
  Danbooru tags pass through; phrase-shape hallucinations (`4k`,
  `detailed_background`, `animedia`) are dropped.
- **Character → series pairing** via pre-computed co-occurrence table
  (e.g. `hatsune_miku` → `vocaloid` added automatically).
- **Artist/character signatures**: artist embeddings include their
  top co-occurring general tags from 500k real Danbooru posts, so the
  retriever can match on style/theme rather than just name.

Zero setup needed — the extension's `install.py` downloads ~1.1 GB of
pre-built index artefacts from HuggingFace on first load
([dataset](https://huggingface.co/datasets/freedumb2000/anima-tagger-artifacts)).
The bge-m3 embedder and bge-reranker models auto-download via
`sentence-transformers` (~3.4 GB total) on first Anima click.

Settings live under **Settings → Anima Tagger** — threshold, reranker
toggle, co-occurrence pairing toggle, query expansion toggle.

### Developer pipeline (not for end users)

Scripts under `src/anima_tagger/scripts/` are the maintainer workflow
for (re)building the artefacts when Danbooru data updates:

```bash
# 1. Pull latest Danbooru tag dump + post dataset from HF
python src/anima_tagger/scripts/download_data.py

# 2. Rebuild sqlite + FAISS index + co-occurrence (~10 min on GPU)
python src/anima_tagger/scripts/build_index.py

# 3. Upload fresh artefacts to HF (auth via `hf auth login` first)
python src/anima_tagger/scripts/package_artifacts.py

# 4. (Optional) verify retrieval quality
python src/anima_tagger/scripts/verify.py
python src/anima_tagger/scripts/full_pipeline_test.py
```

End users never run these.

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

### Authoring base prompts and operational prompts

`_bases.yaml` extends or replaces entries in the **Base** dropdown. `_prompts.yaml` overrides the operational prompts (Remix, summarize, wildcard preamble, negative contract). Both merge with the published defaults; anything you don't override stays.

```yaml
# _bases.yaml
My Custom Base: |
  You are a prompt writer. Given a user's raw input, expand it into a
  detailed scene description...
```

#### How a base is assembled

For every non-Custom base, the final system prompt is:

```
_preamble                  # shared: input handling
your base body             # per-base: style and content rules
_format                    # shared: no headings, no line breaks, no commentary
[detail instruction]       # added when Detail slider > 0
```

Then during generation these sections are appended in order: `SOURCE PROMPT: ...`, `Apply these styles: ...` for selected modifiers, `wildcard_preamble` plus each wildcard instruction, and (when the `+ Negative` checkbox is on) the `negative` block with its `POSITIVE:`/`NEGATIVE:` contract.

Override `_preamble` or `_format` in `_bases.yaml` to change shared behavior for all bases. Select the `Custom` base to bypass the wrapping entirely and supply a raw system prompt from the UI.

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
- Try empty-source + active wildcards, conflicting modifiers, and instruction-style source prompts (*"make it darker"*) as edge cases.
- Watch for any words or phrases from your system prompt that appear verbatim in output — that's mirroring, and usually means you need to reframe that section as prose instead of labeled structure.

## How it works

1. **Prose**: source prompt is sent to the local LLM with system prompt (base + modifiers + detail level) and wildcards in the user message. The LLM returns a detailed flowing paragraph.
2. **Hybrid**: three-pass pipeline. Pass 1 generates prose (same as Prose mode). Pass 2 extracts danbooru tags from that prose using the selected tag format's system prompt. Pass 3 summarizes the prose into 1-2 compositional sentences. Tags go through the full post-processing pipeline. All three passes use the same seed for consistency.
3. **Tags**: two-pass pipeline. Pass 1 generates prose (same as Prose mode). Pass 2 extracts danbooru tags from that prose using the selected tag format's system prompt. Tags go through the full post-processing pipeline. Both passes use the same seed for consistency. On Anima + RAG, the same shortlist + embedding validator as Hybrid applies; on truncation, tag mode fails loud (empty output + retry status) rather than delivering a partial list that looks complete.
4. **Remix**: the current prompt is sent back to the LLM with new modifiers/wildcards to integrate. Auto-detects prose, tags, or hybrid format and uses the appropriate refine prompt and post-processing.
5. **Streaming**: all LLM calls use streaming with real-time stall detection and thinking mode detection. `/no_think` is prepended to prevent Qwen3 models from entering thinking mode.
6. The output is written to the main prompt textbox and all settings are saved to image metadata for reproducibility.

## License

MIT
