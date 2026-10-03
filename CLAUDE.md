# Agent instructions — sd-webui-prompt-enhancer

This file gives agents (Claude Code, etc.) the project-specific context
they need to work effectively on this extension. Human contributors
can skim it but it's primarily written for AI sessions.

## Start new sessions by reading the tuning record

`experiments/PROSE-TUNING.md` is the record of what was learned tuning
the enhancer's prose: what counts as a failure (the operator's own
grading), the method, each finding with its numbers, the dead ends, and
what is still open. Read it before changing a base, a directive, the
sampling options or the model. Its instrument is
`experiments/prose_fidelity.py`.

`experiments/LOG.md`, where present, is an older local log of the
removed tag pipeline (gitignored, machine-local).

## TL;DR — the load-bearing rules

Before writing code, check these. Full explanations follow below.

1. **Don't pick, test.** When a design question has multiple reasonable
   answers, don't ask the maintainer — implement variants, rate outputs,
   recommend with evidence.
2. **Quality = intent fulfillment.** Not a word count, not a keyword
   screen. Read every output yourself and rate it against a rubric with
   concrete anchors. Revise the rubric when it stops distinguishing good
   from bad.
3. **Knowledge lives in data (YAML), not Python.** No keyword lists,
   no modifier→behavior dicts, no prompt text in `.py` files.
4. **No silent fallbacks.** Empty LLM output — fail loud, make it
   visible. Don't paper over with defaults.
5. **No island tests.** Test tools and extension share ONE
   implementation. Same source/seed → same request to the model.
6. **Don't anchor on existing code.** Throw it away if the shape is
   wrong. This project has no shipped users.
7. **One factor at a time** between variants (attribution discipline) —
   but the **space** of variants to consider should include radical
   redesigns, not just one-line tweaks.

## Project state

Heavy active development. Single maintainer. Not yet a widely-used
extension. Prose only since 2026-10-03: the extension expands a source
prompt into natural-language prose (Enhance) and patches an existing
prompt (Remix). The earlier tag half — Hybrid and Tags modes, tag
formats, the Danbooru retrieval pipeline — was removed.

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
- Removing a choice (a mode, a base): just remove it; don't add a
  `_restore_*` branch that maps old values to something else.
- Renaming a module: just rename.

## Testing expectations

The test tools live in `experiments/`:

- `experiments/prose_fidelity.py` — does the enhancer keep what the
  source states? Fixed sources, read every output, keyword screens only
  point at where to look.
- `experiments/compare_prompt.py` — prints the exact system prompt and
  user message for a source (`--dry`), or runs it against Ollama.

Both import the real extension module through
`experiments/_pe_bootstrap.py`, so they exercise the functions Forge
runs. They need Ollama and the Forge Python environment.

## Commit style

One logical change per commit. Imperative subject, detailed body
(what + why), `Co-Authored-By: Claude Opus 4.7 (1M context)
<noreply@anthropic.com>` trailer.

Good recent examples: `c0e627b`, `64572b9`, `aad2e2c`.

## Working style: experiment-driven, willing to discard code

This is a research-heavy system under active redesign. Default to
**designing experiments and letting data decide**, not picking an
architecture up front and patching it.

### Don't pick, test

When there's a design question with multiple reasonable answers
("should the base prompt do X or Y?", "one call or two?") — **do not
ask the maintainer to pick.** Implement both (or several) as variants,
rate the outputs against the rubric, and recommend the winner with
evidence.

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

### Quality = intent fulfillment, not a metric

**Word count is not quality. A keyword screen is not quality.** These
are cheap proxies that are easy to optimize without improving actual
output. Easy to optimize ≠ meaningful.

The real question: does the output, fed to the image model, produce
what the source prompt + modifiers asked for? Answering that requires
**reading every output and rating it**, per a rubric with concrete
dimensions — e.g. subject correct, modifier honored, scene coherent,
no inventions, coverage appropriate.

### Rubric must have concrete anchors per dimension

"Score 1-5 for scene coherence" drifts — my definition of 3 today
might be 2 tomorrow. Each rubric dimension must carry concrete anchor
descriptions at minimum for scores 1 / 3 / 5: "1 = the output contains
details that clearly contradict the source (e.g. an airplane in a
speakeasy scene). 3 = mostly coherent, 1-2 off-theme details. 5 = every
detail traces back to a concept in the source."

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
input+output, every decision's reason. Without traceability you can't
tell which factor caused an outcome, which means you can't do
controlled experiments.

The `experiments/` directory is the canonical place for this tooling.

### Iterative variants with one-factor-at-a-time attribution

Design variants **iteratively from evidence**, not from a pre-committed
up-front list. Build V1 = baseline (existing pipeline wrapped). Rate it
against the rubric. Design V2 as a direct response to V1's
highest-impact rated failure. Rate V2. Repeat.

Between any two variants being compared head-to-head, change **one
factor**. If you swap the pipeline shape AND the LLM AND the base
prompt between V1 and V2, a V2 win could come from any of them and you
learn nothing. One-factor-at-a-time is an attribution discipline — it
tells you WHAT caused a change.

This is NOT a constraint on the search space. See "Fundamental redesign
is always on the table" — radical restructurings are legitimate variants
to consider. The rule is: when you evaluate variant N+1 against N,
change one factor. Not: only propose one-line tweaks.

### Fail-loud, no silent defaults

Every step fails loudly on unexpected input — LLM empty output, a
missing prompt key, a base that assembles to nothing. Silent fallbacks
(empty string → "safe", missing result → default value) make it
impossible to tell whether a variant failed structurally or got bad LLM
output, because the trace looks normal. Raise + log + fail the run;
investigate before patching.

### Isolate one factor at a time

Outputs are influenced by: LLM model choice, base prompt, the appended
directives (adherence, motion, negative), modifier behaviorals,
sampling options, pipeline shape. When output is bad, identify which
factor to swap — then swap only that, not several at once. Otherwise a
win can come from any combination and you learn nothing.

### No island tests — the extension and the experiment share code

Results are only trustworthy if the tested behaviour is the behaviour
Forge runs. Island tests (a harness that uses simplified prompts,
mirrors of the real code, or its own copy of the pipeline) give false
confidence and have bitten this project before.

Contract: the implementation lives in ONE place
(`scripts/prompt_enhancer.py`). The test tools import that module
through `experiments/_pe_bootstrap.py` and call its functions. Where a
tool has to rebuild a step the button handler does inline (the handler
is a closure inside the UI), it says so and stays a line-for-line
mirror of the handler.

When behaviour changes, verify with the same source + modifiers + seed
that the request the handler sends is what the tool sends. If they
differ, the shared-implementation contract is broken and must be fixed
before trusting a result.

### Fundamental redesign is always on the table

The user has repeatedly signaled: if the current pipeline is the
reason we're failing, redesign it — multi-LLM stages, whatever it
takes. Don't confine the search space to "minimal changes to what
exists." Include structurally different variants in every round of
experiments.

## Architectural principle: knowledge lives in data, not in Python

The YAML files and the LLM are the source of truth. Python's job is to
**orchestrate** — read config, call the LLM, assemble prompts — NOT to
encode domain knowledge about what a style means or which words are
NSFW.

### Where different kinds of things belong

| Kind of thing | Belongs in | Examples |
|---|---|---|
| Modifier metadata | `modifiers/**/*.yaml` entries | behavioral text, keywords |
| Base prose style | `bases.yaml` | voice, structure, content rules (including "do/don't sanitize") |
| Operational prompts | `prompts.yaml` | adherence, remix, motion, negative, inline wildcards |
| Content classification | LLM call | "is this NSFW?", "does this detail fit the scene?" |
| Structural algorithms | Python | prompt assembly, output cleanup, repetition detection |

### Red flags — do NOT add any of these to Python

- **Keyword lists.** If you're writing `{"sex", "nude", "penetration"}` in a `.py` file, stop.
- **LLM-facing text.** Every instruction the model reads lives in YAML, so no "hidden" prompt influence sits in Python.
- **Modifier-name → behavior dicts** in Python. Every modifier attribute should be declared on the modifier's YAML entry and read out. Adding a new modifier should never require a Python edit.
- **Magic threshold numbers** without a setting or a derivation.
- **"Fallback" keyword checks** to patch around the LLM. If the primary path is giving bad output, fix the prompt — don't add a keyword filter to paper over it.

### The failure pattern to avoid

When the LLM sanitizes explicit content, the fix is **the base and the operational prompts** (or the modifier config that feeds them), NOT a hardcoded keyword override in Python.

Patching symptoms in Python produces a system that needs constant maintenance and drifts from the YAML that supposedly defines behavior.

## What the agent should NOT do

- Don't create README.md, dev docs, or any new documentation files
  unless explicitly asked.
- Don't add telemetry, usage analytics, or any phone-home code.

## Ollama dependency

Extension currently requires Ollama running on
`127.0.0.1:11434` with a model pulled. Replacing this with an
in-process LLM runner (llama-cpp-python) is in `TODO.md`. Until
that lands, assume Ollama is the only LLM path.
