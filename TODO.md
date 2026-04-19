# Prompt Enhancer — Next Steps

## Negative Prompt Feature (Decided, Ready to Build)

### Design
- New checkbox in UI: **"+ Negative"** (unchecked by default)
- When checked, all modes (Prose, Hybrid, Tags, Remix) output both POSITIVE and NEGATIVE sections
- Format: `POSITIVE:\n(content)\n\nNEGATIVE:\n(keywords)`
- Python parses the output, routes positive to main textarea, negative to Forge's negative textarea
- Remix intelligently updates negatives: adds relevant ones, removes contradicting ones (tested, works 5/5)

### What was tested
- Qwen reliably outputs POSITIVE/NEGATIVE format (6/6 in testing)
- Remix correctly handles style transitions (DSLR→painting removes "painting" from neg, adds "photorealistic")
- LLM understands inverse relationships ("make head big" → neg gets "small head, normal proportions")
- LLM decides when negatives need updating vs leaving unchanged

### Implementation needed
- System prompt additions for all modes to request POSITIVE/NEGATIVE format when checkbox is on
- Parsing logic to split output at NEGATIVE: marker
- JS bridge to write to Forge's negative prompt textarea (similar to prompt_out.change)
- Tag format YAMLs could include negative quality tag conventions (worst_quality, low_quality for Illustrious)
- Hybrid mode: negative tags go through tag post-processing pipeline
- Remix: reads BOTH textareas (positive + negative) and sends combined to LLM

### Open questions
- Should tag format YAMLs define default negative quality tags?
- Should the negative textarea be cleared when checkbox is unchecked, or left as-is?

## Other Ideas Discussed But Not Decided

### Modifier placement
- Modifiers moved from system prompt to user message (tested, better adherence with 4+ mods)
- Source prompt now prefixed with "SOURCE PROMPT:" for LLM distinction
- Remix keeps modifiers in system prompt (existing prompt is user message)

### Reinforced base prompt
- `_bases.yaml` has a "Reinforced" variant that boosts source prompt repetition
- Works (5.4 → 15.0 avg keyword count) but also boosts modifiers
- Prepend Source checkbox is the cleaner targeted solution
- Reinforced base is still useful as a local override for when you want everything amplified

### Things we decided against
- TIPO integration (can't read prose, random tag generation)
- Running Qwen in-process (Ollama is simpler)
- Priority modifier system (marginal improvement, added user-facing complexity)
- Hardcoded character detection for tags (maintenance burden)
- Ban tags (not needed with current workflow)

## Session Summary — What Was Built

### Cancel button fix
- `trigger_mode="multiple"`, no `_js`, no `cancels=`, no outputs
- Root cause: `_js` DOM manipulation desynced Svelte state, `trigger_mode="once"` blocked repeat clicks

### Hybrid mode (3-pass pipeline)
- Pass 1: Prose generation (base + modifiers + wildcards + detail)
- Pass 2: Tag extraction (tag format prompt + wildcard context for artist preservation)
- Pass 3: NL summary (summarize prompt + modifier context for style preservation)
- Tags go through full post-processing pipeline

### Seed control
- UI with dice/reuse ToolButtons
- Passed to Ollama for reproducible output
- Saved to image metadata as PE Seed

### Tag post-processing improvements
- Generic meta-annotation stripping (`[artist: X]`, `setting: garden`, etc.)
- Hyphen-to-underscore conversion
- Paren escaping for SD (`_(style)` → `_\(style\)`)
- Prefix matching for disambiguation suffixes
- Fixed lookup to always use underscores (was broken for Illustrious format)
- Skip Levenshtein for tags < 5 chars (prevented garbage matches)
- Prefix match requires underscore (prevented `high` → `high_(hgih)`)

### UI changes
- Added Hybrid button between Prose and Tags
- Moved Still/Scene/Audio from checkboxes to YAML-driven Temporal Framing dropdown
- Added Cinematic, POV, Timelapse, Loop modes
- Base + Tag Format + Tag Validation on one row
- `_label` field in YAML for dropdown names (no filename convention)
- Prepend Source checkbox
- Elapsed time in status messages
- Removed Grab button
- Seed ToolButtons (Forge-native styling)

### Architecture changes
- Modifiers moved from system prompt to user message (better adherence)
- Source prompt prefixed with "SOURCE PROMPT:" marker
- Remix detects three formats: prose, tags, hybrid
- REFINE_HYBRID_SYSTEM_PROMPT for hybrid remix
- Wildcard context injection to Hybrid pass 2 (artist/style preservation)
- Modifier context injection to Hybrid pass 3 (NL summary style preservation)
- _postprocess_tags() helper consolidates tag pipeline
- _detect_format() replaces _is_tag_format() with three-way detection

### Ollama config
- OLLAMA_FORCE_CPU option (CUDA_VISIBLE_DEVICES="")
- Default changed to GPU + immediate unload (KEEP_ALIVE=0, NUM_GPU=-1)
