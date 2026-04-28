# Prompt Enhancer — TODO

Forward-looking work. Everything previously in this file has shipped.

---

## Session conventions (read first if you're a fresh agent)

- **Never rename `anima_tagger`.** The module name is load-bearing —
  settings keys (`anima_tagger_*`), HF repo
  (`freedumb2000/anima-tagger-artifacts`), commit message conventions,
  user-facing "Anima Tagger" Forge settings section. If you extend its
  scope, do it under the existing name; docstring comments may say
  "retrieval pipeline" but don't rename modules or option keys.
- **Per-format config lives in YAML**, not Python. `tag-formats/*.yaml`
  is the source of truth for anything format-specific. When you find a
  hardcoded Anima constant in code, lift it to the yaml and read it
  back through the tag-format loader.
- **Prefer extending existing code over creating siblings.** Don't
  spawn `tagger_rag/` alongside `anima_tagger/` — just make
  `anima_tagger/` format-aware.
- **FAISS index + cooccurrence + tags.sqlite are Danbooru-wide.**
  Extending to new formats does NOT require rebuilding or re-uploading
  them. Only rebuild when upstream Danbooru data changes.
- **Commit style**: one logical change per commit. Detailed body
  (what + why), imperative subject, `Co-Authored-By: Claude Opus 4.7
  (1M context) <noreply@anthropic.com>` trailer. See recent commits
  (e.g. `c0e627b`, `64572b9`) for the pattern.
- **Test before committing.** For retriever/validator changes run
  `src/anima_tagger/scripts/verify.py`; for end-to-end changes run
  `src/anima_tagger/scripts/full_pipeline_test.py` against Ollama.
  New A/B test ideas go under `src/anima_tagger/scripts/` as
  `ab_*.py`, not `test_*.py` (gitignored prefix).
- **Don't touch Remix/Tags/Hybrid handler signatures.** They're
  Gradio-wired; changes must be internal to each `try:` body.

**Prior-art commits to read before significant work:**
- `d8f6742` original `anima_tagger` module
- `a7160da` Forge-side integration (how the stack is lazy-loaded)
- `aad2e2c` validator's category-aware + shortlist-stem alias path
- `c0e627b` build_index's E₂ signature embedding
- `64572b9` Remix integration + safety routing + pinned exact names
- `a6a4499` end-user HF auto-download flow

---

## New feature

- [ ] "Add modifiers to prompt" button — appends the selected modifiers'
      `keywords` (not `behavioral`) directly to the main prompt textbox,
      no LLM call. Useful for tag-based workflows where the user wants
      keyword-style additions without prose generation.

---

## anima_tagger: extend RAG pipeline to SDXL booru formats

Today the `anima_tagger` RAG pipeline (shortlist injection, bge-m3
validator, co-occurrence pairing, artist signatures) runs ONLY for
the Anima tag format. Illustrious / NoobAI / Pony still use the
rapidfuzz Fuzzy Strict path. They'd benefit from the same pipeline
with per-format tweaks.

### How to approach it

1. Add a `rag_enabled: true|false` flag to each `tag-formats/*.yaml`
   (default false so legacy behavior is preserved for formats not
   explicitly opted in).
2. Add a `FormatConfig` dataclass loaded by `_load_tag_formats`
   (in `scripts/prompt_enhancer.py`) that surfaces per-format:
     - `quality_prefix: list[str]` — tokens prepended by rule layer
     - `valid_safety: set[str]` — accepted safety tokens
     - `rating_vocab: set[str]` — e.g. `rating:*` for Illustrious,
       `rating_*` for Pony
     - `artist_prefix: str` — `@` for Anima, empty for others
     - `use_underscores: bool` — already present; keep
     - `whitelist_tokens: set[str]` — replaces hardcoded `ANIMA_WHITELIST`
3. Thread the `FormatConfig` into `TagValidator` and
   `rule_layer.apply_anima_rules` so they consult it instead of
   hardcoded constants. (Rename `apply_anima_rules` → just keep; it's
   called by the AnimaTagger entry point.)
4. Gate `_use_anima_pipeline(tag_fmt)` on `FormatConfig.rag_enabled`
   — currently it hardcodes `tag_fmt == "Anima"`; swap to the yaml
   flag.
5. Shortlist logic and validator are format-agnostic in their retrieval
   calls; only the post-retrieval rule layer needs format awareness.
6. Settings UI: the existing "Anima Tagger" section keeps its name
   (don't break persistent option keys). Add a NEW setting
   `anima_tagger_enabled_formats` (multi-select) if opt-in visibility
   is desired; default ["Anima"].

### Per-format details

- [ ] **Illustrious** — `use_underscores: true`,
      `rating_vocab: {rating:general, rating:sensitive, rating:questionable, rating:explicit}`,
      `artist_prefix: ""` (no @), `quality_prefix: [masterpiece, best_quality, absurdres, highres]`.
- [ ] **NoobAI** — identical to Illustrious but `use_underscores: false`
      and `quality_prefix` includes `very_awa`.
- [ ] **Pony Diffusion v6** — `use_underscores: true`,
      `rating_vocab: {rating_safe, rating_questionable, rating_explicit}`
      (note: underscore not colon), `artist_prefix: ""`,
      `quality_prefix: [score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up]`.

### No rebuild needed

FAISS index + cooccurrence table + tags.sqlite are Danbooru-wide —
each file applies unchanged to all booru formats. No HF re-upload
required when extending format support. Only thing to update on HF:
the dataset card wording (remove "Anima"-only framing).

### Risks

- **Pony's quality prefix is ENTIRELY score-based** — `masterpiece`
  isn't in Pony's expected tokens. Rule layer default for Anima is
  `masterpiece, best_quality, score_7`; for Pony it's the full
  `score_N_up` cascade. Make the rule layer read from FormatConfig.
- **Rating tag placement** — rule layer currently doesn't emit rating
  tokens (Anima uses safety tags, not `rating:*`). For Illustrious/
  NoobAI/Pony, rating tags should go at the END of the tag list per
  booru convention (look at how `tags/` CSVs rank them).
- **Artist signatures (E₂)** — the embedding-text enrichment applies
  regardless of consumer; no format-specific rebuild needed.

---

## Potentials to A/B test (not rigorously decided)

These three were **deferred**, not rejected — I (prior session) judged
them low ROI without running the tests. If you want the pipeline to
be fully evidence-based, A/B each one before concluding.

- [ ] **F / #9 Draft-context cross-validation** — when multiple alias
      candidates exist for a draft token, check which candidate
      co-occurs with other already-validated tags in the draft.
      Example: draft has `1girl, maid, chibi, rococo` — `rococo` →
      character candidate `rococo_(girl_cafe_gun)` vs artist
      `toeri_(rococo)`; check which has higher PMI with `maid`,
      `chibi` → prefer it.
      **Effort:** medium (needs per-pair cooccurrence lookup per
      ambiguous token). **Expected gain over current A:** small —
      shortlist already disambiguates most cases. **Failure mode:**
      legitimately unusual tag combos (crossover scenes) get penalized.
      **A/B:** compare existing Rococo-style scenarios; add
      deliberately-crossover drafts (e.g. miku tagged with samurai
      theme) and see if F hurts them.

- [ ] **J Per-subject sub-shortlists** — split the source prompt into
      subjects (e.g. "a girl reading" + "in a cafe" = 2 subjects),
      run shortlist per subject, merge by subject tag. Goal: surface
      MORE diverse thematic characters/artists for multi-subject
      scenes. **Effort:** low-medium (subject extraction could be
      regex or a tiny LLM pass). **Expected gain:** marginal — H
      (query expansion) already gives the retriever a richer query;
      this doubles retrieval calls. **Failure mode:** over-segmentation
      on verbose sources.
      **A/B:** compose scenarios with 2+ clearly distinct subjects
      (e.g. "a cat watching a wizard cast a spell") and compare
      shortlist breadth vs current single-query.

- [ ] **E₁ General-tag co-occurrence enrichment** — same treatment as
      E₂ but for general tags (`1girl`, `reading`, etc.) — embed each
      with its top-N co-occurring tags. **Effort:** medium — requires
      general↔general PMI (much larger than the current char↔series
      table). **Expected gain:** small — general tags mostly get
      exact-match hits; enrichment only helps typo/paraphrase cases.
      **Failure mode:** common tags like `1girl` get signatures that
      are basically "average Danbooru post" → hurts their differentiation.
      **A/B:** collect paraphrased draft tokens (e.g. `young girl`,
      `feminine figure`, `small female`) and measure whether E₁
      substitutes to canonical (`1girl`) vs current behavior (drop).

---

## Tested and rejected (A/B done, decision final)

- **Hybrid dense + sparse retrieval** (bge-m3 sparse) — tested at
  α ∈ {1.0, 0.9, 0.7, 0.5, 0.3}. Lower α regresses E₂'s thematic
  retrieval gain; name-overlap returns. One case it helps (literal
  entity names) is already covered by pinned exact-name matches and
  shortlist stem lookup. Data at
  `src/anima_tagger/scripts/ab_hybrid_retrieval.py`.
- **Character appearance auto-tags** — would require persisting per-
  character signature tables, plus contradiction detection. Contrary
  to Anima's tag-dropout training philosophy: the character tag
  already carries appearance implicitly, and adding it redundantly
  slightly over-emphasizes canonical features (hurts variants the
  user asked for).
- **Multi-query characters** (raw + expanded, merged) — no new hits
  in 4/5 scenarios on A/B; expansion strips literal name signal that
  character retrieval relies on.
- **Stem dedup on shortlist** — collapses legitimately-distinct
  vtubers sharing a stem (`dragoon (selen tatsuki)` vs `dragoon
  (dokibird)` vs `dragoon (sekaiju)` are different people).

---

## Outstanding from 2026-04-22 session (queued, not started)

Context: V14–V22 commits shipped on `main` (`6032757`). A separate
branch `claude/identify-repo-9YyME` was created in a different agent
session and contains:
  - the Ollama → llama-cpp-python in-process cutover (`src/llm_runner/`)
  - a major refactor splitting `scripts/prompt_enhancer.py` into
    `src/pe_data/`, `src/pe_modes/`, `src/pe_tags/`, `src/pe_anima_glue/`,
    `src/pe_settings.py`, `src/pe_metadata.py`, `src/pe_llm_layer/`
  - a story-mode experiment harness (`experiments/story_*`) with
    grammar-constrained JSON output and a 38-seed corpus
  - `experiments/STORY_MODE_HANDOFF.md` documenting how to run that
The branch is **unmerged**. `main` Anima Hybrid path still uses the
V14–V22 prompt-and-validate cascade for tag extraction.

### The actual RAG/tags strategy (the V34 root fix, half done)

Llama-cpp infra is on the branch. The REMAINING work (none of which
has been started):

1. Build a `_build_tag_grammar()` function that, per run, assembles
   a GBNF grammar from:
     - active tag format's whitelist (quality, safety, rating)
     - the shortlist's artists / characters / series candidates
     - format's `general_tags_allowlist` (V19)
     - subject-count productions enforcing non-contradiction
     - any source:-mechanism pre-picks (Random Artist etc.) baked in
2. Pass the grammar to the Hybrid Pass-2 (and Tags Pass-2) tag-extract
   LLM call.
3. Audit which V14–V22 layers become structurally unnecessary once the
   LLM physically cannot emit out-of-vocab tokens. Expected outcomes:
     - V31 (subject-count discipline) → obsolete
     - V32 (whitelist enforcement) → obsolete
     - V14 filter → still useful for policy, simpler internals
     - V19 carve-outs → still useful for character signature retention
     - RAG validator (`_anima_tag_from_draft`) → near-obsolete; keep
       only thin alias-normalization
4. Remove or simplify the layers that became unnecessary.

This is the principled root fix discussed in the prior session. It
obviates several queued bandaid Vs and simplifies downstream code.

### Bandaids (likely obsoleted by the V34 root fix)

Listed for awareness — do NOT implement before the grammar work above.
Each is the patch that becomes unnecessary if grammar lands.

- [ ] **V23** — NSFW co-occurrence carve-out in
      `_filter_to_structural_tags`. Query
      `cooc.top_for("nsfw", category=0)` and `top_for("explicit", ...)`
      to retain CAT_GENERAL tags that co-occur with NSFW rating in the
      DB; addresses V14's −0.6 girl_sex regression. Two settings:
      `anima_tagger_v23_nsfw_cooc_top_k` (default 15),
      `anima_tagger_v23_nsfw_cooc_min_prob` (default 0.3). Requires
      cooccurrence DB to have rating-keyed entries — verify or rebuild.
      **Better long-term:** rebuild cooccurrence DB with rating-keyed
      source tags (root data fix, not filter patch).
- [ ] **V31** — subject-count discipline. Tag-extract LLM emits both
      `1girl` and `2girls` for "woman kissing a man"; V14 filter keeps
      both. Fix path: (a) prompt — "emit one count tag per subject
      TYPE present; never both `1girl` and `2girls`"; (b) filter —
      if both N-girls and M-girls present, drop the smaller.
      Obsoleted by grammar (one production per type).
- [ ] **V32** — artist/character/series whitelist enforcement at
      filter time. Tag-extract sometimes emits `@man (man room)`
      (a real Danbooru artist tag, FAISS nearest-neighbor of bogus
      `@man` token). Drop any CAT_ARTIST / CAT_CHARACTER / CAT_COPYRIGHT
      tag whose normalized form isn't in the per-run shortlist OR a
      source:-mechanism pre-pick. Obsoleted by grammar.

### Sanitization track (prose-side; orthogonal to grammar)

Grammar constrains shape, not content. These remain relevant after V34.

- [ ] **V24** — prose self-check retry. After prose generation, a
      small LLM classifier asks "did this prose preserve the source's
      explicit content?" If no, retry with different seed (or stronger
      prompt variant). Costs one extra LLM call per run; more on
      sanitization hits.
- [ ] **V25** — larger / less RLHF'd LLM. With llama-cpp cutover this
      is just a GGUF swap. Try abliterated Qwen 14B+ if the 9B keeps
      sanitizing.
- [ ] **V26** — strengthen the V8 multi-sample picker SP to explicitly
      reject sanitized samples (not just "pick most source-faithful").

### Confidence-gating track (the "kissing problem" class)

Pipeline produces specific named entities even when source asks for
generic. Same meta-pattern as V32 but at retrieval boundaries.

- [ ] **V27** — suppress character shortlist on generic sources. If
      no character candidate exceeds a similarity floor, inject empty
      character shortlist into prose SP. Prevents
      `kisara (engage kiss)` from leaking in for "girl kissing a man".
- [ ] **V28** — seed-based shortlist diversification. Keep top-K fixed
      but pick seed-driven subset for injection. Solves determinism
      without solving bias direction.
- [ ] **V29** — similarity floor on retrieval. Only include shortlist
      candidates above threshold. Tight floor → usually-empty
      shortlists on generic sources.
- [ ] **V30** — confidence-gate audit + telemetry. Sweep every
      "soft signal → committed output" boundary in the pipeline (RAG
      retrieval, LLM character/artist commitment, safety-tag
      determination, slot-fill, source-inject) and add confidence
      gates. Pair with per-run telemetry log showing what was
      injected at what confidence. Makes future bias bugs
      deliberate-find vs accidental-find.

### UI / cleanup / test plan

- [ ] **V33** — UI icon marker for randoms that directly affect tag
      choice (Random Artist/Character/Era/Franchise inject structural
      tags) vs prose-only ones (Random Setting, Random Intensity,
      Random Production Style). Existing ◆ / ◆◇ system signals source
      mechanism; V33 adds visibility of the tag-output effect, perhaps
      a 🏷️ glyph next to dropdown options.
- [ ] **Band-aid audit (continued).** Sweep `src/` (post-refactor
      branch) for remaining hardcoded Python constants — magic
      thresholds, name-to-behavior dicts, keyword lists. Lift to yaml
      or make data-driven per CLAUDE.md "knowledge lives in data."
      V18b got `_ANIMA_NSFW_INTENSITY_TO_SAFETY` and
      `_ANIMA_NSFW_PRODUCTION_NAMES`; expect more in `src/pe_tags/`,
      `src/pe_anima_glue/`, `src/pe_settings.py`.
- [ ] **Dropped-tag reduction — design only, DO NOT run.** Investigate
      why Hybrid drops many draft tags (`9 dropped` observed in one
      run). Leverage points: tag-extract prompt tuning, fuzzy-fallback
      on drops, popularity floor, retrieval threshold. Inverse
      question: are dropped tags ones we'd want to keep, or is the
      filter doing its job? Test plan first; don't run until designed.

### Pipeline-level reconsideration (after grammar lands)

These are architectural alternatives evaluated in the 2026-04-22
discussion. Listed as live options to consider once V34 ships, NOT
queued for immediate work:

- **Drop the two-pass architecture.** A single grammar-constrained
  call produces both prose AND structured tags. No "extract from
  prose" step. Eliminates a class of inter-pass drift.
- **Similar-posts retrieval** instead of entity-shortlist. Embed the
  source, retrieve top-N similar Danbooru POSTS, aggregate their
  tags by category and popularity. Different mental model from the
  current shortlist injection.
- **Use Qwen3 0.6B (Anima's text encoder) for retrieval** instead of
  bge-m3. Aligns retrieval space with the image model's semantic
  understanding. Requires re-embedding the DB (one-time offline).
- **Rating-aware cooccurrence DB.** Rebuild the cooccurrence table
  keyed by rating context. Replaces V23 as a root data fix rather
  than a filter patch.
- **Fine-tuned tag-prediction LoRA** (or a dedicated tag-prediction
  small model). Domain-specific beats general-LLM-with-prompting.
  Larger effort, possibly bigger win than grammar alone.
- **Prose-first product pivot.** Maintainer observation: Prose mode
  is the only mode producing good Anima images. Make Prose the
  primary path; Hybrid becomes minimal-tag-prefix + prose; Tags
  mode arguably deprecated for Anima. Strategic reframe rather
  than tech change.

### Observed bugs from live use (2026-04-22) — to verify post-reload

V20/V21/V22 shipped fixes for several of these via prompts.yaml.
Verify the fixes took after Forge restart picks up the yaml changes.

- **Subject-count contradiction.** `2girls, 1boy` and `2girls, 1girl`
  emitted for "woman kissing a man." Sometimes `1girl` alone (omits
  the man). → V31.
- **Bogus `@artist` tags.** `@man (man room)` (real Danbooru artist
  tag, FAISS nearest-neighbor of generic `man`) emitted when source
  said only "a man." → V32.
- **Character determinism on generic sources.** `kisara (engage kiss)`
  injected consistently for "girl kissing a man" via name-token
  proximity in FAISS. → V27.
- **Prose echoes source as tag-style header.** "1girl, 1boy, kissing,
  a girl with..." even when source isn't tag-shaped. V20/V21
  prompts.yaml fix shipped — verify.
- **Prose describes drawing style.** "in the style of X, dense black
  ink washes, heavy cross-hatching" — tag territory leaking into
  prose. V21/V22 prompts.yaml fix shipped — verify.

### Process discipline note (from prior-session memory)

Before resuming any of the above project work, the prior session ranked
**bildhauer skill expansion** as the #1 priority. Reason: V14–V22
exposed a recurring failure pattern (agent shallow-enumerates options
even when explicitly asked for fundamental reconsideration) that
produces compounding bandaids. Fix is structural — expand bildhauer's
PROCEDURE.md to enforce option-space enumeration with externally-
verifiable evidence. Without that fix, this session's work is at risk
of the same plaster-on-plaster pattern. See
`/home/g/.claude/projects/-home-g-dev-Gunther-Schulz-sd-webui-prompt-enhancer/memory/project_next_session_skill_design.md`
for the full design brief.

---

## Investigate: replacing Ollama with an internal runner

**STATUS:** implemented on the unmerged branch
`claude/identify-repo-9YyME` (cutover + src/ refactor done).
Grammar-constrained decoding is wired for the story-mode experiment
paths only (JSON-schema → GBNF). The Anima Hybrid tag-extract path on
`main` still uses Ollama prompt-and-validate. Section retained below
for design context.

Currently the extension requires Ollama running externally (HTTP on
`127.0.0.1:11434`). Benefits of replacing it with an in-process runner
would be: no separate daemon, automatic model lifecycle management
coordinated with image generation, no HTTP overhead, and the ability
to use **grammar-constrained decoding** (which would solve the last
class of LLM-drafted hallucinations — tokens that only match real
Danbooru tags can be emitted).

### Candidate runners

- **llama-cpp-python** — Python bindings for llama.cpp. Loads GGUF
  models directly from HF. Supports GBNF grammar constraints (big
  unlock — we could directly enforce "only real Danbooru tags" at
  generation time). CPU + GPU (CUDA) builds available. Lightweight.
- **vllm** — high-throughput server. Overkill for single-user workflow.
- **transformers + accelerate** — slowest option; no grammar support.
- **MLC-LLM** — compiles models for specific hardware. Setup-heavy.

### Likely architecture

- Ship a "LLM Runner" setting: `ollama` (current) | `internal (llama-cpp)`.
- For internal runner: auto-download a recommended GGUF (e.g. a
  quantized version of the abliterated qwen3.5-9b) from HF via
  `huggingface_hub`.
- Load on-demand in a context manager similar to how `anima_tagger`
  handles bge-m3 (per-call load, auto-unload to free VRAM for image
  generation).
- Reuse existing `_call_llm` interface so the rest of the codebase
  doesn't care which runner is active.

### Considerations

- **Users with existing Ollama setups.** Keep Ollama as a supported
  runner; don't force-switch. Recommend internal for new installs.
- **GGUF licensing.** Model licenses must permit redistribution for
  auto-download to work cleanly. Abliterated qwen variants are
  typically Apache-2.0 — should be fine.
- **VRAM coordination.** With bge-m3 + reranker + Anima DiT + the LLM
  all potentially co-resident, need careful unload sequencing.
  Internal runner makes this easier (we own the lifecycle).
- **Grammar decoding benefit.** With GBNF constraints generating only
  valid Danbooru tags, we could drop the rapidfuzz validator entirely
  and rely on the retrieval validator for semantic correction only —
  simpler pipeline.

### Risks

- **Token throughput.** llama-cpp GGUF inference on RTX 4090/5090 is
  ~50-80 tok/s for 9B models; Ollama is similar. No regression expected.
- **Setup complexity for users.** Ollama "just works" on most OSes.
  llama-cpp-python requires CUDA toolkit matching torch. Wheels
  should cover most setups; fallback to CPU.
- **Context-length mismatches.** Current prompts are well under 4k
  tokens; both runners handle this fine.

---

## Session lessons worth preserving

- **Don't teach the LLM what it already knows** — for *concepts*
  (golden hour, Wes Anderson). Qwen-9B recognizes these; just name
  them. But *do* scaffold output characteristics (thoroughness,
  sentence variety, character completeness) — tested 2026-04-20,
  removing those rules cuts output to ~50% length.
- **Let the base prompt govern voice.** Modifier delivery should name
  the styles; the base handles HOW they're applied. Adding "apply as
  prose" to the delivery duplicates or contradicts the base.
- **Label echo in output is often useful.** Style tokens like "golden
  hour" or "oil painting" are recognized by image models. Fighting them
  out of output was wrong. Only section-header form ("Patched:",
  "Creative Choice:") was the original bug.
- **Existing format creates anchoring bias.** When a format has a
  problem, incremental rewrites perpetuate the bias. First ask "what's
  the minimum the downstream consumer needs?"
- **Concrete examples in system prompts leak ~20% of the time on 9B
  abliterated.** Prefer abstract descriptions over concrete
  lists — e.g. "available time-period tags" beats "e.g. year 2025,
  newest, recent". Verified 2026-04-20 on the Anima tag-format
  system prompt.
- **Artists / characters / series tags in Danbooru dumps have ~0 wiki
  coverage** (0 / 1 / 177 out of 618k / 344k / 60k respectively as of
  Sept 2025 dump). Dense retrieval on just-the-name falls back to
  literal overlap; E₂ co-occurrence signatures are what unlock
  thematic matching. If a new embedding feature only improves
  name matching, it will probably regress E₂.
- **Bake signatures into embeddings, not into output.** Anima is
  tag-dropout-trained, so emitting redundant appearance tags for named
  characters hurts variants. Different downstream consumers (SDXL
  without character training, Pony, etc.) may want the opposite.
  Decision should be per-format.
