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

## Investigate: replacing Ollama with an internal runner

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
