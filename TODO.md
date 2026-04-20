# Prompt Enhancer — TODO

Forward-looking work. Everything previously in this file has shipped.

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
