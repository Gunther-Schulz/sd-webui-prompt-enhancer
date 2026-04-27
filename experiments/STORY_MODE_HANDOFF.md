# Story Mode — Phase A handoff

This is a handoff doc to pick up the story-mode experiment work on your
local machine. Phase A (prompt-side only, no image generation) is what's
in scope here; Phase B+ (Forge orchestration) comes after we have rated
test data to point at.

Branch: `claude/identify-repo-9YyME`. All changes from this session are
on that branch, ready to commit + push.

## What was built (file inventory)

| File | Purpose |
|---|---|
| `experiments/story-modifiers/story-length.yaml` | New modifier dropdown — 3/4/6/8/12 panels. **Lives under experiments/ deliberately**, not modifiers/, so it does NOT auto-register as a Forge UI dropdown until story mode ships. Move to `modifiers/` when wiring up the production UI. |
| `experiments/story-modifiers/story-mode.yaml` | New modifier — Linear / Vignettes / Character Study / Before & After / Day in the Life / Two Viewpoints. Same "experiments/-only for now" placement. |
| `experiments/story-rubric.yaml` | 7-dimension rating rubric with 1/3/5 anchors per dim. v0.1 — revise during rating sessions when scores feel forced. |
| `experiments/story-seeds.yaml` | 10 test stories spanning genres / lengths / modes. Most flagged `placeholder: true` — replace with your typical use cases when convenient. The harness runs against whichever set is in this file. |
| `experiments/story-variants/V1_one_pass_yaml.yaml` | Single LLM call, full plan in YAML. Tests max-output-per-call extreme. |
| `experiments/story-variants/V2_one_pass_json.yaml` | Same as V1 but JSON output. Tests YAML-vs-JSON axis. |
| `experiments/story-variants/V3a_two_pass_yaml.yaml` | Pass 1 = plan (YAML, captions only); Pass 2 = per-panel prompts (text, run N times). Tests one-vs-multi-pass axis. |
| `experiments/story-variants/V3b_two_pass_json.yaml` | Same as V3a but Pass 1 in JSON. |
| `experiments/story-variants/V5_terse_yaml.yaml` | Single LLM call producing roles + captions only; image prompts are template-assembled at runtime, no LLM elaboration. Tests whether per-panel LLM elaboration is necessary or just nice-to-have. |
| `experiments/story_validators.py` | Pure-function schema validators. One per output shape. Fail-loud per CLAUDE.md. |
| `experiments/story_runner.py` | Test harness. Loads variant + seed, runs LLM passes against Ollama, validates outputs, saves trace JSON. |
| `experiments/story_rate.py` | Interactive rater. Walks traces, prompts for rubric scores, saves `_ratings.json` (human pass). |
| `experiments/story_summarize.py` | Cross-variant aggregator. Reads traces + ratings, prints headline + per-dim + per-length comparison tables. |
| `experiments/AGENT_RATER_PROMPT.md` | Paste-ready prompt that has Claude Code read traces directly and write `_ratings_agent.json` + qualitative `_agent_review.md`. The agent pass is a second opinion, NOT a replacement for human rating. |

What was **not** changed: `scripts/prompt_enhancer.py`, `prompts.yaml`,
`modifiers/`, `src/`. No production behavior changes. The extension
behaves identically before and after this branch is merged.

## Prerequisites before running

1. llama-cpp-python installed (auto-installed by `install.py` on Forge load,
   or manually: `pip install llama-cpp-python`)
2. ~6 GB free disk for the GGUF (auto-downloaded on first run via HF cache)
3. PyYAML installed (already a dep of the extension)

GPU optional. CPU mode works (slower). Configurable via `--compute cpu`.

## How to run

### Smoke test — one variant, one seed

```bash
cd /path/to/sd-webui-prompt-enhancer
python -m experiments.story_runner \
    --variant V1_one_pass_yaml \
    --only lighthouse_6_linear \
    --llm-seeds 42
```

First run downloads the default GGUF (~5.6 GB for Q4_K_M of
huihui-ai/Huihui-Qwen3.5-9B-abliterated). Subsequent runs reuse the HF
cache. Output goes to `.ai/experiments/story/V1_one_pass_yaml/<run_id>.json`.

### Override the model / compute / DRY

```bash
# Use a different quant
python -m experiments.story_runner --variant V1_one_pass_yaml \
    --llm-quant Q5_K_M

# CPU only (slow but no GPU needed)
python -m experiments.story_runner --variant V1_one_pass_yaml \
    --compute cpu

# Bring your own GGUF
python -m experiments.story_runner --variant V1_one_pass_yaml \
    --llm-model-path /path/to/your.gguf

# Disable the low-level DRY sampler (use high-level samplers only;
# useful to A/B whether DRY is helping)
python -m experiments.story_runner --variant V1_one_pass_yaml --no-dry
```

### Full A/B grid

```bash
for v in V1_one_pass_yaml V2_one_pass_json \
         V3a_two_pass_yaml V3b_two_pass_json \
         V5_terse_yaml; do
    python -m experiments.story_runner --variant $v --llm-seeds 42 137
done
```

5 variants × 10 seeds × 2 LLM seeds = 100 runs. The model loads once
per variant (kept resident across all seeds within a variant invocation),
then unloads at end of variant. Two-pass variants do ~N+1 LLM calls
per run, so a 6-panel V3a run is 7 calls = a few minutes per run on
GPU, longer on CPU. Realistic budget: a few hours on GPU, an evening
on CPU.

### Rating — agent first, human last

The flow puts the human LAST in the loop. The agent does heavy
per-trace reading; the human reviews and overrides.

**Step 1 — agent rates (Claude Code reads traces directly):**

Open a Claude Code session in this repo. Paste the first prompt
template from `experiments/AGENT_RATER_PROMPT.md` per variant. Claude
reads every trace, applies the rubric, writes `_ratings_agent.json`
plus a qualitative `_agent_review.md`.

**Step 2 — agent drafts the cross-variant decision doc:**

Once every variant has agent ratings, paste the second prompt template
from `experiments/AGENT_RATER_PROMPT.md`. Claude synthesizes all
ratings + reviews into a draft `experiments/STORY_RESULTS_v1_DRAFT.md`
with a ranking and Phase A.1 promotion recommendation.

**Step 3 — human reviews each variant (interactive override):**

```bash
python -m experiments.story_rate --variant V1_one_pass_yaml --review-agent
```

For each trace, the script shows the source / plan / per-panel prompts
plus the agent's score per rubric dimension as a default. Press Enter
to accept; type 1-5 to override; 's' to skip; 'q' to quit. The human
only types on disagreements — fast.

Failed-validation traces auto-score 1 on `schema_validity` and only ask
for notes. Ratings persist to `_ratings.json` after every rating, so
Ctrl-C is safe.

If you want maximum independence (no agent score visible), drop the
`--review-agent` flag and rate from scratch. Useful on a sample to
calibrate, then switch to `--review-agent` for the rest.

**Step 4 — human writes final results doc:**

If human + agent ratings agree closely (overall within 0.5 across
variants), copy `STORY_RESULTS_v1_DRAFT.md` to `STORY_RESULTS_v1.md`
and ship. If they diverge by 1+ on overall for any variant, the
human writes `STORY_RESULTS_v1.md` from scratch (referencing the
draft) — that divergence is data about either the rubric or the
specific failure mode the agent missed.

Final ship/no-ship is in `STORY_RESULTS_v1.md`, written and approved
by the human.

### Summary across variants

After rating (one or both passes complete):

```bash
# Headline + per-dim tables, terminal output
python -m experiments.story_summarize

# Markdown tables (paste into commit message or design doc)
python -m experiments.story_summarize --md

# Per-length parse-rate breakdown — shows where each variant collapses
python -m experiments.story_summarize --by-length
```

The headline table shows: parse rate, mean wall time, mean tokens,
mean LLM-call count, mean overall rating per rater. The per-dimension
table shows means per rubric dim per variant. Disagreements between
human and agent overall means by 1+ flag a deeper look.

### Cross-variant judgment (the actual decision document)

After per-trace ratings exist, the second half of
`experiments/AGENT_RATER_PROMPT.md` has a separate prompt that asks
Claude Code to read all `_ratings_agent.json` + `_agent_review.md`
files and produce `experiments/STORY_RESULTS_v1.md` — a per-axis
comparison and a recommendation for which variant ships in Phase A.1.

That document is the artifact that drives Phase A.1 promotion (lifting
prompts into `prompts.yaml`, moving modifier YAMLs into `modifiers/`).

## What to look at while rating / interpreting

1. **Schema-validity pass rate per variant.** Binary signal. If V1
   fails to parse 70% of the time and V3a passes 95%, that's the answer
   to one-vs-multi-pass.
2. **Pass rate × length** (`--by-length`). Tells you whether single-
   pass collapses at length=12 (likely) and whether multi-pass costs
   are worth it for short stories.
3. **Quality dimensions on traces that DID parse.** Did multi-pass
   produce richer per-panel prompts (`scene_specificity` higher)? Did
   V5 terse + template-assembly produce mechanical prompts vs. V3a/V3b's
   LLM-elaborated ones?
4. **YAML vs JSON.** Compare V1 vs V2, V3a vs V3b. If JSON parses more
   reliably, prefer JSON regardless of the pass-count answer.
5. **Token cost vs. quality.** V3a/V3b will use ~3x the tokens of V1.
   If V1 has 60% parse rate and V3a has 95% but produces only marginally
   better prompts on the parsed runs, the trade-off may favor V1 +
   retry-on-parse-failure.

## Decisions to settle from the data

In rough priority order:

1. **Pass count.** One-pass (V1/V2/V5) vs two-pass (V3a/V3b). Expectation
   from CLAUDE.md and Qwen-9B's known limits: multi-pass wins at long
   stories, but V5 (terse + template) might win on simplicity if its
   output quality is acceptable.
2. **Output format.** YAML vs JSON for the structured pass.
3. **Per-panel elaboration necessity.** Does V5 (no per-panel LLM
   call) produce usable prompts, or do we need the V3a/V3b second
   pass to get scene-rich prompts?
4. **Length threshold.** Where does each variant break down? Useful
   to know for graceful UX (e.g. "you're asking for 12 panels with
   variant X, expect failures").

## What comes next (the rough plan, not committed)

After ratings are in:

- **Phase A.1 — pick winners.** Promote the winning variant's prompts
  into `prompts.yaml` (real production prompts, not experiments).
  Promote the modifier YAMLs from `experiments/story-modifiers/` to
  `modifiers/`.
- **Phase A.2 — wire the UI.** New `📖 Story` button in the prompt-
  enhancer accordion alongside Prose/Hybrid/Tags/Remix. Opens a plan
  editor pane where the user reviews/edits the generated plan. (No
  image generation yet — Phase A is review-only output.) Remix can
  operate on the plan before generation, matching the Prose pattern.
- **Phase B — Forge orchestration.** Wire up the actual edit-chain via
  Qwen-Image-Edit-2511. New module `src/story_orchestrator/`. Runner
  per backend (`runners/qwen_image_edit.py`, future
  `runners/z_image_omni.py`). Gallery UI showing panels as they
  generate. Per-panel regenerate buttons.
- **Phase C — Wan2GP metadata export** (opt-in toggle). Sidecar JSON
  per panel with motion-prompt + duration + ref-frame path + cut
  transition hints. Enables longer consistent videos by letting Wan2GP
  generate one cut per panel.

## Risks / things to watch for during testing

- **Endless-loop generation** (your README flags this). If a run hangs
  for hours, kill it. The runner's per-call timeout defaults to 600s;
  any single LLM call exceeding that hits Ollama's timeout and the
  runner records an error. If loops are a frequent failure mode,
  consider lowering `num_predict` in the variant configs.
- **Format leakage.** Qwen 9B abliterated sometimes wraps output in
  ```` ``` ```` fences despite being told not to. The validators strip
  these (`_strip_fences`); if a trace fails with "YAML/JSON parse
  error" but the raw output looks valid, check whether the strip
  worked.
- **Empty `roles_present` for character scenes.** The model may
  accidentally treat a character scene as an establishing shot and
  produce `roles_present: []` even when the caption mentions Alice.
  Validator catches this only indirectly (via the `roles_present`
  type/membership check). Worth eyeballing during rating.

## Commit + push

When done with this batch of work:

```bash
git add experiments/story-modifiers/ experiments/story-rubric.yaml \
        experiments/story-seeds.yaml experiments/story-variants/ \
        experiments/story_runner.py experiments/story_validators.py \
        experiments/story_rate.py experiments/story_summarize.py \
        experiments/AGENT_RATER_PROMPT.md \
        experiments/STORY_MODE_HANDOFF.md
git commit -m "Add story-mode Phase A test harness + variants

Phase A scope: prompt-side only, no image generation. Tests how reliably
Qwen 9B abliterated produces structured story plans across:
  - 1-pass vs 2-pass (V1/V2/V5 vs V3a/V3b)
  - YAML vs JSON (V1/V3a vs V2/V3b)
  - Full prompts vs caption-only (V1/V2/V3a/V3b vs V5)

Deliverables under experiments/:
  - story-modifiers/    new dropdowns (length, mode), staged here so
                        they don't appear in the Forge UI yet
  - story-variants/     5 variants spanning the design axes
  - story-rubric.yaml   7-dim rubric with 1/3/5 anchors
  - story-seeds.yaml    10 test stories (most marked placeholder until
                        replaced with real use cases)
  - story_runner.py     test harness, calls Ollama, saves traces
  - story_validators.py pure-function schema validators
  - story_rate.py       interactive rater
  - STORY_MODE_HANDOFF.md  this doc

No production code changes. scripts/prompt_enhancer.py and prompts.yaml
untouched. Promotion to production happens after testing picks winners.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push -u origin claude/identify-repo-9YyME
```

## When you come back

Read this doc first, then `experiments/story-rubric.yaml`. Look at one
seed (`experiments/story-seeds.yaml` → `lighthouse_6_linear`) and one
variant (`experiments/story-variants/V1_one_pass_yaml.yaml`) to refresh
on the data shapes. Run the smoke test from "How to run" above. Rate it.
That'll tell you whether the harness works end-to-end on your machine
before you commit to the full grid.
