# Prompt Enhancer — TODO

Forward-looking work. Everything previously in this file has shipped.

## New feature

- [ ] "Add modifiers to prompt" button — appends the selected modifiers'
      `keywords` (not `behavioral`) directly to the main prompt textbox,
      no LLM call. Useful for tag-based workflows where the user wants
      keyword-style additions without prose generation.

## anima_tagger: extend RAG pipeline to SDXL booru formats

Today the anima_tagger RAG pipeline (shortlist injection, bge-m3
validator, co-occurrence pairing, artist signatures) runs ONLY for
the Anima tag format. Illustrious / NoobAI / Pony still use the
rapidfuzz Fuzzy Strict path. They'd benefit from the same pipeline
with per-format tweaks.

Scope outline per format:

- [ ] **Illustrious** — underscores, `rating:*` tokens, no `@artist`
      convention. Needs:
        - per-format validator whitelist (quality tokens,
          `rating:general|sensitive|questionable|explicit`)
        - rule layer with underscores preserved (no space conversion)
        - no `@` prefix on artist tags
        - illustrious-appropriate quality prefix (e.g. `masterpiece,
          best_quality, absurdres, highres` — no `score_N`)
- [ ] **NoobAI** — same as Illustrious at the schema level but uses
      spaces in tags and includes `very_awa` quality token. Mostly
      reuse Illustrious config with `use_underscores: false`.
- [ ] **Pony Diffusion v6** — requires `score_9, score_8_up, …,
      score_4_up` at the prompt start and `rating_safe|rating_explicit`
      tokens (note: no `rating:` colon). Pony-specific quality prefix,
      pony-specific safety vocabulary.

Shared infra work:

- [ ] Generalize `anima_tagger/rule_layer.py` — pull the Anima-specific
      constants (`_QUALITY_PREFIX`, `_VALID_SAFETY`, `_ENTITY_CATEGORIES`,
      `@`-prefix logic) into a `FormatConfig` dataclass loaded per tag
      format from `tag-formats/*.yaml`. Rename to `tagger_rag/` since it
      no longer only serves Anima.
- [ ] Generalize `TagValidator` — the ANIMA_WHITELIST is format-specific;
      take it from the active FormatConfig instead of hardcoded.
- [ ] Shortlist logic stays mostly format-agnostic. Decide whether to
      skip `@artist` in shortlist fragment for non-Anima formats
      (currently fragment is neutral; should be fine).
- [ ] Wire `_use_anima_pipeline` → `_use_rag_pipeline(tag_fmt)` in
      `prompt_enhancer.py`, gated by a per-format capability flag in the
      YAML (`rag_enabled: true`) so maintainers opt individual formats in.
- [ ] Same FAISS index works for all formats (tags are Danbooru-wide).
      Co-occurrence table likewise. No rebuild needed.
- [ ] Update `HF_REPO` dataset card to note it applies to all booru
      formats, not just Anima.
- [ ] Settings UI: rename "Anima Tagger" section → "Retrieval Tagger"
      and broaden descriptions.

Risks to monitor:

- Pony's quality prefix is "everything score_*", not "masterpiece". The
  rule layer needs to consult the active format rather than assuming
  Anima's `masterpiece, best_quality, score_7` default.
- Illustrious' character→series pairing via co-occurrence likely works
  identically since the PMI table is extracted from Danbooru posts
  regardless of which model consumes them downstream.
- Artist signatures (E₂) were built to help Anima's retrieval where
  artists have no wiki; the same data should transparently help
  Illustrious/NoobAI/Pony shortlists.

## Tested and rejected

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
