# Prompt Enhancer — TODO

Forward-looking work. Everything previously in this file has shipped.

## Wildcard unification

Fold the current `wildcards.yaml` system into modifier categories. Every
category gets a `🎲 Random X` entry; orphans that don't fit an existing
category get new files.

- [ ] Migrate category-mapped wildcards into their modifier files as
      random-choice entries:
    - Random Hair / Eyes / Outfit / Expression / Pose / Accessory → `subject.yaml`
    - Surprise Location / Background / Weather / Time → `setting.yaml`
    - Surprise Palette → `visual-style.yaml`
    - Random Artist / Series / Art Movement / Medium + all Anime-style
      wildcards → `visual-style.yaml`
    - Light Source → `lighting-mood.yaml`
    - Unexpected Angle → `camera.yaml`
- [ ] Create new modifier files for orphan wildcards:
    - `narrative.yaml` — Story Moment, Narrative Detail
    - `focus.yaml` — Imperfection, Texture Focus, Contrast Pairing
    - Era (wildcard) → goes to `setting.yaml` under time period
- [ ] Delete `wildcards.yaml` and remove the wildcard code path
      (`_build_wildcard_text`, `_load_wildcards`, separate dropdown, etc.).
      The unified modifier system handles everything.

## New feature

- [ ] "Add modifiers to prompt" button — appends the selected modifiers'
      `keywords` (not `behavioral`) directly to the main prompt textbox,
      no LLM call. Useful for tag-based workflows where the user wants
      keyword-style additions without prose generation.

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
