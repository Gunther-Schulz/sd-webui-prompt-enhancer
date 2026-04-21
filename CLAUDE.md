# Agent instructions — sd-webui-prompt-enhancer

This file gives agents (Claude Code, etc.) the project-specific context
they need to work effectively on this extension. Human contributors
can skim it but it's primarily written for AI sessions.

## Start new sessions by reading the experiment log

`experiments/LOG.md` is the persistent digest of what's been tried.
Read it first if you're coming in fresh — it gives current status,
rejected variants (don't re-test), cross-variant findings, and
next-hypothesis candidates. Individual `_ratings.json` files under
`.ai/experiments/<variant>/` have per-run details.

## TL;DR — the load-bearing rules

Before writing code, check these. Full explanations follow below.

1. **Don't pick, test.** When a design question has multiple reasonable
   answers, don't ask the maintainer — implement variants, rate outputs,
   recommend with evidence.
2. **Quality = intent fulfillment.** Not tag count. Not drop rate. Read
   every output yourself and rate it against a rubric with concrete
   anchors. Revise the rubric when it stops distinguishing good from bad.
3. **Knowledge lives in data (YAML + DB), not Python.** No keyword lists,
   no tag-name constants, no modifier→behavior dicts in `.py` files.
4. **No silent fallbacks.** Empty LLM output, zero retrieval results —
   fail loud, make it visible in the trace. Don't paper over with defaults.
5. **No island tests.** Experiment and extension share ONE pipeline
   implementation. Promoted variant → same source/seed → same output.
6. **Don't anchor on existing code.** Throw it away if the shape is
   wrong. This project has no shipped users.
7. **One factor at a time** between variants (attribution discipline) —
   but the **space** of variants to consider should include radical
   redesigns, not just one-line tweaks.

## Project state

Heavy active development. Single maintainer. Not yet a widely-used
extension. The RAG pipeline (`src/anima_tagger/`) was just built and
integrated into the Tag Validation flow in April 2026.

## No backwards compatibility required

**During the current development phase, do NOT write migration code,
legacy-name fallbacks, or deprecation shims.** Breaking changes to
settings keys, radio button values, internal APIs, module paths,
saved-image metadata fields, and yaml schema are all fine — there are
no deployed users with stale state to protect.

When the extension ships to a wider audience we'll reintroduce compat
concerns; until then, prioritize clean code over transitional crust.

Examples of what this means in practice:
- Renaming a setting key: just rename it; don't read both old + new.
- Removing a radio choice (e.g. `Check` mode): just remove it; don't
  add a `_restore_*` branch that maps old values to something else.
- Changing the HF artefact schema: bump `format_version` in
  `package_artifacts.py` and have `install.py` re-download; don't
  bother converting existing files.
- Renaming a module: just rename (though see "Naming conventions"
  below for specific modules that stay as-is).

## Naming conventions that DO stay stable

A few names are load-bearing even in dev:

- **`anima_tagger`** module under `src/` — referenced in HF dataset
  repo id (`freedumb2000/anima-tagger-artifacts`), in the user-facing
  "Anima Tagger" Settings section, and in persistent option keys
  (`anima_tagger_*`). Extending scope to other formats is fine; keep
  the name.
- **Settings keys** under `anima_tagger_*` — persisted in users'
  `config.json` files locally. Changing them silently invalidates
  their preferences. Rename only if you have a reason worth the cost.
- **Tag format yaml filenames** (`anima.yaml`, `illustrious.yaml`,
  `noobai.yaml`, `pony.yaml`) — referenced in the Tag Format
  dropdown's user-visible labels.

## Testing expectations

The canonical test workflow lives in `experiments/` (see "Working
style" below). The experiment runner + rubric + test prompt set is
the primary way to evaluate changes to tagging behavior.

Legacy scripts under `src/anima_tagger/scripts/` (`verify.py`,
`ab_*.py`, `full_pipeline_test.py`) measured tag counts and drop
rates, not quality. They're retained for unit-level sanity checks
on retriever/validator components, but should NOT be used to judge
whether a pipeline change is an improvement — use the rubric-based
`experiments/` runner for that.

New A/B test ideas go in `experiments/` as parameterized variants,
not as standalone ab_*.py scripts (which duplicate pipeline code and
drift from the extension).

## Commit style

One logical change per commit. Imperative subject, detailed body
(what + why), `Co-Authored-By: Claude Opus 4.7 (1M context)
<noreply@anthropic.com>` trailer.

Good recent examples: `c0e627b`, `64572b9`, `aad2e2c`.

## Developer pipeline vs end-user install

- `install.py` at the extension root runs on every Forge extension
  load. pip-installs deps, auto-downloads RAG artefacts from HF,
  falls back to rapidfuzz path with visible warning on failure.
  **Never** require the user to run anything from `src/` manually.

- Everything under `src/anima_tagger/scripts/` is DEV-only — the
  maintainer rebuilds the index or uploads artefacts. End users
  don't touch it.

- HF artefact upload: `package_artifacts.py` uses `huggingface_hub`
  with the maintainer's logged-in token (`hf auth login`). Credentials
  never go through the codebase; the script reads from
  `~/.cache/huggingface/token`.

## Working style: experiment-driven, willing to discard code

This is a research-heavy system under active redesign. Default to
**designing experiments and letting data decide**, not picking an
architecture up front and patching it.

### Don't pick, test

When there's a design question with multiple reasonable answers
("curator prompt should do X or Y?", "retrieval flat or faceted?",
"concept enumeration separate call or inline?") — **do not ask the
maintainer to pick.** Implement both (or several) as variants in the
experiment runner, rate the outputs against the rubric, and recommend
the winner with evidence.

Questions like "should I do A or B?" offload thinking that the
experiment is supposed to answer. Pre-deciding without data is exactly
what the test framework exists to avoid.

### Don't get anchored by existing code

If the pipeline shape is wrong, replace it. Don't patch around it.
This project has no shipped users (see "No backwards compatibility
required") — throwing away code is cheap, and patches on top of a
wrong architecture accumulate into brittleness that's expensive to
untangle later. Be willing to write a fresh pipeline alongside the
existing one and compare.

Red flag: you're editing the 5th fallback condition inside
`_anima_tag_from_draft`. Step back — the surrounding shape may be the
problem, not the details.

### Quality = intent fulfillment, not a metric

**Tag count is not quality. Drop rate is not quality. Coverage
markers are not quality.** These are cheap proxies that are easy to
optimize without improving actual output. Easy to optimize ≠
meaningful.

The real question: does the output, fed to the image model, produce
what the source prompt + modifiers asked for? Answering that requires
**reading every output and rating it**, per a rubric with concrete
dimensions — e.g. subject correct, modifier honored, scene coherent,
no inventions, coverage appropriate.

### Rubric must have concrete anchors per dimension

"Score 1-5 for scene coherence" drifts — my definition of 3 today
might be 2 tomorrow. Each rubric dimension must carry concrete anchor
descriptions at minimum for scores 1 / 3 / 5: "1 = the output contains
tags that clearly contradict the source (e.g. 'airplane' in a
speakeasy scene). 3 = mostly coherent, 1-2 off-theme tags. 5 = every
tag traces back to a concept in the prose."

Without anchors, ratings become mood-of-the-day. With anchors, they
are reproducible across runs and sessions.

### Quality standards are coarse-to-fine and revisable

The rubric itself is an artifact under bildhauer discipline. Start
with a coarse first-pass definition of what "good" means across a
small number of dimensions. Apply it in real rating sessions. When a
rating feels forced, ambiguous, or doesn't capture a failure mode
that matters, that's a signal the rubric needs revision — not that
you should grit your teeth and score anyway.

Rules for revising mid-experiment:
- Keep notes during rating: "I wanted to score this 4 but the anchor
  forced 3", "this failure mode isn't captured by any dimension",
  "dimensions X and Y overlap".
- Between rounds, revise the rubric based on those notes. Explicitly
  document what changed and why.
- Re-rate prior outputs under the new rubric so rounds are
  comparable. Don't mix old and new rubric scores.
- The goal is a rubric that genuinely distinguishes "this variant
  does the job" from "this variant doesn't" — not a pretty-looking
  scorecard.

When claiming a variant "works": include at least a handful of raw
outputs with ratings and notes. Aggregate numbers alone are
insufficient evidence.

### Traceability is non-negotiable

Every experiment run produces a structured trace: every LLM call's
input+output, every retrieval's query+candidates+picks, every
decision's reason. Without traceability you can't tell which factor
caused an outcome, which means you can't do controlled experiments.

The `experiments/` directory is the canonical place for this
tooling. The older `src/anima_tagger/scripts/` harness measured
counts; the new runner rates behavior.

### Iterative variants with one-factor-at-a-time attribution

Design variants **iteratively from evidence**, not from a pre-committed
up-front list. Build V1 = baseline (existing pipeline wrapped). Rate it
against the rubric. Design V2 as a direct response to V1's
highest-impact rated failure. Rate V2. Repeat.

Between any two variants being compared head-to-head, change **one
factor**. If you swap the pipeline shape AND the LLM AND the retrieval
config between V1 and V2, a V2 win could come from any of them and you
learn nothing. One-factor-at-a-time is an attribution discipline — it
tells you WHAT caused a change.

This is NOT a constraint on the search space. See "Fundamental redesign
is always on the table" — radical restructurings are legitimate variants
to consider. The rule is: when you evaluate variant N+1 against N,
change one factor. Not: only propose one-line tweaks.

### Fail-loud, no silent defaults

In the experiment runner, every step fails loudly on unexpected input
— LLM empty output, retrieval returning zero candidates, curator
dropping all tags. Silent fallbacks (empty string → "safe", missing
result → default value) make it impossible to tell whether a variant
failed structurally or got bad LLM output, because the trace looks
normal. Raise + log + fail the run; investigate before patching.

Concrete example of the pattern to avoid: the `chosen_safety =
safety_from_draft or default` idiom in `rule_layer.apply_anima_rules`
silently substitutes a default when the draft contains no safety tag.
The trace shows "safety was safe" with no indication that the LLM
emitted nothing. Make the empty case explicit and visible — ideally
return a sentinel the trace can display, and let the caller decide
whether to fall back.

### Isolate one factor at a time

Pipeline outputs are influenced by: LLM model choice per step, base
prompt, tag-format system prompt, modifier behaviorals, query
expansion config, retrieval params, pipeline shape, curator prompt,
rule-layer options. When output is bad, identify which factor to
swap — then swap only that, not several at once. Otherwise a win can
come from any combination and you learn nothing.

### No island tests — the extension and the experiment share code

A/B results are only trustworthy if the winning variant, once
promoted into the Forge extension, produces the **same** output as
it did in the experiment. Island tests (a harness that uses
simplified prompts, mirrors of the real code, or its own copy of the
pipeline) give false confidence and have bitten this project before.

Contract: the pipeline implementation lives in ONE place
(`experiments/pipeline.py` or equivalent) as pure, importable
functions. The experiment runner calls it. The Forge event handlers
(Hybrid / Tags / Remix buttons) are thin wrappers that call the
exact same code with a named variant. Zero duplication.

When a variant is promoted to the extension, verify with the same
source + modifiers + seed: extension output must equal experiment
output (modulo Ollama-side nondeterminism). If they differ, the
shared-implementation contract is broken and must be fixed before
trusting the variant.

### Fundamental redesign is always on the table

The user has repeatedly signaled: if the current pipeline is the
reason we're failing, redesign it — multi-LLM stages, multi-RAG
calls, whatever it takes. Don't confine the search space to "minimal
changes to what exists." Include structurally different variants in
every round of experiments.

## Architectural principle: knowledge lives in data, not in Python

This is a RAG system. The Danbooru tag DB + FAISS index + LLM are the
source of truth. Python's job is to **orchestrate** — read config, call
the LLM, run retrieval, apply structural rules — NOT to encode domain
knowledge about what tags mean or which words are NSFW.

### Where different kinds of things belong

| Kind of thing | Belongs in | Examples |
|---|---|---|
| Tag format conventions | `tag-formats/<name>.yaml` | quality prefix, valid safety tags, subject-count tag set, non-DB whitelist tokens |
| Modifier metadata | `modifiers/**/*.yaml` entries | behavioral text, keywords, `target_slot`, safety tier implied by the modifier |
| Base prose style | `bases.yaml` | voice, structure, content rules (including "do/don't sanitize") |
| Danbooru category IDs | `anima_tagger.config` | `CAT_ARTIST = 1`, `CAT_COPYRIGHT = 3`, etc. (these ARE the schema) |
| Tunable thresholds | `shared.opts` settings | semantic threshold, popularity floor |
| Content classification | LLM call or retrieval | "is this NSFW?", "does this tag fit the scene?", "which artist for this prose?" |
| Structural algorithms | Python | compound-split, dedup, ordering, @-prefix, category bucketing |

### Red flags — do NOT add any of these to Python

- **Keyword lists of tags.** If you're writing `{"sex", "nude", "penetration"}` in a `.py` file, stop. The DB has those tags with categories and context. Use retrieval or DB lookups.
- **Tag name constants for a specific tag format.** `_QUALITY_PREFIX = ("masterpiece", ...)` in Python duplicates what's in `tag-formats/anima.yaml` — changes drift. Read from the YAML.
- **Modifier-name → behavior dicts** in Python. Every modifier attribute (safety tier, target_slot, whatever) should be declared on the modifier's YAML entry and read out. Adding a new modifier should never require a Python edit.
- **Magic threshold numbers** without a setting or a derivation. If you pick 500, it should be a named setting or computed from the data (e.g. percentile of retrieved candidates).
- **"Fallback" keyword checks** to patch around the LLM or retrieval. If the primary path is giving bad output, fix the prompt or the retrieval — don't add a keyword filter to paper over it.

### Red flags — do NOT add to the test harness either

- Word lists like `_POSE_HINTS = {"standing", "sitting", ...}` for metric computation. If you need to check whether an output "has a pose tag", derive that from the DB (tags of a known category, tags matching a known semantic neighborhood) or from the tag-format config, not a Python constant that decays.

### The failure pattern to avoid

When the LLM sanitizes explicit content, the fix is **the prose/extract prompts** (or the modifier/base config that feeds them), NOT a hardcoded keyword override in `_anima_safety_from_modifiers`. When the "Random Artist" modifier always picks the same artist, the fix is **seeded retrieval** or **shortlist presentation randomization**, NOT a hardcoded exclude-list. When a low-popularity tag wins an exact-match lookup and hijacks a general concept, the fix is **a popularity gate driven by DB signals**, NOT a hardcoded list of "dangerous" tag names.

Patching symptoms in Python produces a system that needs constant maintenance and drifts from the YAML/DB that supposedly defines behavior.

## What the agent should NOT do

- Don't commit `data/` — gitignored (holds ~1.1 GB of artefacts).
- Don't run Git credential helpers (note the `install.py` upload
  warning says this explicitly).
- Don't add third-party vector DBs (Chroma, Qdrant, pgvector) — the
  project uses `faiss-cpu` directly and that's sufficient at 273k
  vectors.
- Don't introduce `langchain` / `haystack` / similar abstractions —
  overkill for a single-purpose retriever.
- Don't create README.md, dev docs, or any new documentation files
  unless explicitly asked.
- Don't add telemetry, usage analytics, or any phone-home code.

## Ollama dependency

Extension currently requires Ollama running on
`127.0.0.1:11434` with a model pulled. Replacing this with an
in-process LLM runner (llama-cpp-python) is in `TODO.md`. Until
that lands, assume Ollama is the only LLM path.
