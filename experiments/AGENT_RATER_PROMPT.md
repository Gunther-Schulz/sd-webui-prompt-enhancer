# Agent rating step (Claude Code judges + compares — first pass)

The story-mode test flow puts the human LAST in the loop:

1. **Agent rating pass** (this doc): Claude Code reads every trace,
   applies the rubric, writes `_ratings_agent.json` and a qualitative
   `_agent_review.md` per variant. The agent does the tedious per-trace
   reading work first.
2. **Agent draft of cross-variant judgment**: separate Claude Code prompt
   below — produces `experiments/STORY_RESULTS_v1_DRAFT.md` synthesizing
   all variants' ratings + reviews into a ranking and ship/no-ship
   recommendation.
3. **Human review pass**: `python -m experiments.story_rate --variant <V>
   --review-agent`. Walks each trace showing the agent's scores as
   defaults; Enter accepts, 1-5 overrides. Fast — the human only types
   on disagreements. Saves to `_ratings.json`.
4. **Human finalization**: human reads the agent's draft results doc,
   override-rates surface in the summary, and the human writes (or
   approves) the final `experiments/STORY_RESULTS_v1.md` — the ship
   decision.

`experiments.story_summarize` reads both rating files and prints them
side-by-side so agent → human disagreements surface immediately.

Final ship/no-ship is the human's call. The agent does heavy lifting;
the human signs off. Per CLAUDE.md "read every output yourself and rate
it" — the human review pass is still reading every output, just with
the agent's pre-work as the starting point rather than a blank slate.

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

## Cross-variant draft judgment (still agent's job)

After every variant has agent-rated traces + an `_agent_review.md`,
ask Claude Code one more question to draft the cross-variant decision
doc. The human will review and finalize this draft, but the agent
synthesizes the data first:

```
Read .ai/experiments/story/*/_ratings_agent.json and *_agent_review.md.
Read also any .ai/experiments/story/*/_index.json files for run
metadata.

Produce a DRAFT cross-variant comparison at
experiments/STORY_RESULTS_v1_DRAFT.md. The human will review and either
accept it as STORY_RESULTS_v1.md or edit/override.

Cover:

1. Headline ranking — which variant looks shippable, which doesn't, why.
2. Per-axis findings:
   - 1-pass vs 2-pass — which performed better, at which story lengths
   - YAML vs JSON — which parsed more reliably; quality difference
   - V5 (template-assembled prompts) vs V3a/V3b (LLM-elaborated prompts)
     — does the extra LLM pass produce noticeably better per-panel
     prompts, or is template-assembly good enough?
3. Recommended Phase A.1 promotion: which variant's prompts go into
   prompts.yaml, which modifier YAMLs move from
   experiments/story-modifiers/ to modifiers/.
4. Open questions for Phase B (image-side testing on Qwen-Image-Edit).
5. Confidence: where is your draft most uncertain? What would change
   your recommendation if the human disagrees on a specific dimension?

Mark this clearly as a DRAFT awaiting human review. The human's
override-ratings (in _ratings.json) and final judgment in
STORY_RESULTS_v1.md are the canonical decision artifact.
```

---

## After human review (the final step)

The human runs:

```bash
# Per-variant review (agent's scores shown as defaults; Enter accepts)
python -m experiments.story_rate --variant V1_one_pass_yaml --review-agent
# ... repeat per variant ...

# Compare both rating sets side-by-side
python -m experiments.story_summarize
python -m experiments.story_summarize --by-length
```

If human and agent ratings agree within ~0.5 on the overall, the
agent's draft results doc is probably accurate; the human can copy
STORY_RESULTS_v1_DRAFT.md to STORY_RESULTS_v1.md, edit lightly, and
ship the recommendation.

If they diverge by 1+ on overall for any variant, the human writes
STORY_RESULTS_v1.md from scratch (referencing the draft) — that
divergence is the signal that the agent's analysis missed something
the human caught (or vice versa, in which case the rubric needs
revising before the next round).

Final ship/no-ship is in STORY_RESULTS_v1.md, written and approved
by the human.
