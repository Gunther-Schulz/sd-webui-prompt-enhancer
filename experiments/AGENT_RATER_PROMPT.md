# Agent rating step (Claude Code judges + compares)

The story-mode test flow has two rating passes:

1. **Human** rating via `experiments.story_rate` — interactive, your judgment.
2. **Agent** rating via Claude Code reading the traces directly — independent
   second opinion. Cross-checks human ratings; cheap to run after the human
   pass; sometimes spots failure modes a human would miss when fatigued.

Both write rubric scores to disk (`_ratings.json` for human,
`_ratings_agent.json` for agent). `experiments.story_summarize` reads both
and prints them side-by-side so disagreements surface immediately.

CLAUDE.md: "Quality = intent fulfillment. Read every output yourself and
rate it." The human pass is the primary signal. The agent pass is a
sanity check, not a replacement.

---

## How to invoke the agent pass

Open a Claude Code session in this repo. Paste the block below, **with
the variant id filled in.** Claude reads the traces, rates per the rubric,
and writes the ratings file. No further interaction needed beyond
confirming the file landed.

```
Read every trace JSON under .ai/experiments/story/<VARIANT_ID>/ (skip
files starting with underscore — those are summaries/ratings, not
traces). For each trace, apply the rubric in
experiments/story-rubric.yaml.

Output format: write a JSON file at
.ai/experiments/story/<VARIANT_ID>/_ratings_agent.json with this
schema (matching the human-rater schema so story_summarize.py can
read both):

{
  "<run_id>": {
    "run_id": "<run_id>",
    "seed_id": "<from trace>",
    "llm_seed": <int from trace>,
    "scores": {
      "schema_validity":      1-5,
      "role_specificity":     1-5,
      "scene_specificity":    1-5,
      "narrative_coherence":  1-5,
      "style_anchor_quality": 1-5,
      "ref_panel_logic":      1-5,
      "no_inventions":        1-5,
      "overall":              1-5
    },
    "notes": "<1-3 sentences: what stood out, why this score>"
  },
  ...
}

Rules for the agent pass:
- If a trace's outcome is "failed" (schema validation rejected the
  output), set schema_validity=1 and OMIT the other dimensions for that
  trace (set scores to just {"schema_validity": 1, "overall": 1}). Note
  field should describe the failure briefly.
- If a trace's outcome is "ok", rate every dimension. Use the rubric's
  1/3/5 anchors as concrete reference points. Don't average toward 3 —
  use 1 and 5 when warranted.
- For "no_inventions", check whether the plan introduces characters,
  major plot elements, or settings the source idea didn't ask for. A
  small atmospheric detail (a cat, a street vendor) is fine; a new
  named character is not.
- For "ref_panel_logic", check that establishing/character-free panels
  have ref_panels: [], that character scenes use [1] (identity anchor),
  [n-1] (continuity), or [1, n-1] (both), and that the choice fits the
  scene.

After writing the JSON file, ALSO write a short qualitative review at
.ai/experiments/story/<VARIANT_ID>/_agent_review.md with:
- 2-3 paragraphs: what this variant does well, what it does badly,
  failure modes you noticed across runs, where the rubric felt
  forced (so we know what to revise next iteration).
- One-line verdict: would you ship this variant as-is, with edits, or
  not at all?

Confirm both files exist when done. No further action needed.
```

---

## After all variants are agent-rated

```bash
# Summary across all rated variants — human + agent ratings side-by-side
python -m experiments.story_summarize

# Markdown for pasting into a commit message or design doc
python -m experiments.story_summarize --md

# See where each variant breaks at length
python -m experiments.story_summarize --by-length
```

If human and agent ratings agree within ~0.5 on the overall, the signal
is solid and you can pick a winner. If they diverge by 1+ on overall
for any variant, dig into the per-dim scores and the agent's
`_agent_review.md` to figure out which rater missed something —
usually a rubric gap, occasionally a real disagreement worth resolving.

---

## Cross-variant judgment (separate from per-trace rating)

After per-trace rating is complete, ask Claude Code one more question:

```
Read .ai/experiments/story/*/_ratings_agent.json and *_agent_review.md.
Produce a cross-variant comparison at experiments/STORY_RESULTS_v1.md
covering:

1. Headline ranking (which variant ships, which doesn't, why)
2. Per-axis findings:
   - 1-pass vs 2-pass — which won, at which lengths
   - YAML vs JSON — which parsed more reliably; quality difference
   - V5 (template-assembled) vs V3a/V3b (LLM-elaborated) — does the
     extra LLM pass produce noticeably better per-panel prompts?
3. Recommended Phase A.1 promotion: which variant's prompts go into
   prompts.yaml, which modifier YAMLs move from
   experiments/story-modifiers/ to modifiers/.
4. Open questions for Phase B (image-side testing on Qwen-Image-Edit).
```

That document becomes the artifact that drives Phase A.1 promotion to
production. It's the "decision with evidence" deliverable per CLAUDE.md.
