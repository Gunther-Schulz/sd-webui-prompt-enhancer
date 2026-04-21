# Experiment log

Short digest + narrative. Ratings source of truth is per-variant
`_ratings.json` under `.ai/experiments/<variant>/`. This file adds
what no JSON has: cross-variant findings, current status, open
questions, and a new-session entry point.

## Current status

**Best so far:** V8 (3.875 overall, +0.2 vs V1, +0.175 vs V5). First
variant with meaningful overall improvement. Big wins on girl_sex
(+0.5 vs V5) and empty_ra_rs (+0.2 vs V5). No regressions vs V5.

**Decision point:** V8 is the first real improvement after 7 prior
variants. Options:
1. **Promote V8 to the extension** — wire the multi-sample-prose step
   into Forge's Hybrid/Tags handlers. Then expand test prompt set
   to catch new failure modes (NSFW intensity modifiers, Hybrid NL
   output, less-famous characters, etc.).
2. **Keep iterating** — try V9 (different base), V10 (larger LLM).
   V8's remaining regression vs V1 on girl_ra_re (-0.4) is small
   but real; could chase it.

Leaning toward option 1 — V8 is good-enough and we've spent 8
variants on one failure axis. Time to validate V8 in the extension
and move to new axes.

## Variants tried (one-liners — full data in JSON)

| Variant | Delta from V1 | Verdict | Overall Δ | Ratings |
|---|---|---|---|---|
| V1 | baseline | accepted as baseline | 3.675 (10-seed) | `.ai/experiments/v1/_ratings_10seed.json` |
| V2 | NSFW prose directive | REJECTED | -0.25 (2-seed) | `.ai/experiments/v2/_ratings.json` |
| V3 | retrieval-first curator (structural) | REJECTED | -0.75 (2-seed) | `.ai/experiments/v3/_ratings.json` |
| V4 | generic source adherence directive | REJECTED net, partial win on girl_sex | -0.2 | `.ai/experiments/v4/_ratings.json` |
| V5 | V4 directive gated on non-empty source | ACCEPTED as current best | +0.025 | `.ai/experiments/v5/_ratings.json` |
| V6 | softer self-conditioning directive | REJECTED | 0.0 (surrendered V5's girl_sex win) | `.ai/experiments/v6/_ratings.json` |
| V7 | V5 + prose-pass temp 0.3 (vs 0.8) | REJECTED | -0.325 (converged on bad trajectories) | `.ai/experiments/v7/_ratings.json` |
| V8 | V5 + multi-sample prose (3 + picker) | **ACCEPTED — new best** | +0.2 (+0.175 vs V5) | `.ai/experiments/v8/_ratings.json` |

## Findings that persist across variants

These hold across V1/V2/V3 and don't need re-testing.

1. **LLM variance is at the prose step.** Same prompt, `girl_sex`
   seed=1000 consistently sanitizes across V1/V2/V3; seed=1001
   consistently preserves. No downstream (extract/canonicalize/curate)
   change will fix this — the prose itself is what varies.

2. **Embedder canonicalization of concept phrases surfaces rare
   Danbooru variants** (V3 finding). `twintails` → `twintails with
   hair base`. V1's compound-split is actually cleaner for this.
   Don't try another retrieval-first-curator variant without a
   different canonicalization strategy.

3. **2 seeds is too few for variance conclusions.** Moving to
   SEEDS_10 (`experiments/seeds.py`) — curated non-sequential set
   so across-variant comparisons stay apples-to-apples. V1 10-seed
   revealed girl_sex stdev of 1.49 (vs 2-seed estimate suggesting
   just occasional failure).

4. **Adherence directives have non-uniform effects by source type**
   (V4 finding). Generic source-adherence directive helps when
   source is non-empty with concrete content (girl_sex: variance
   halved, full-sanitize rate 20%→0%) but hurts when source is
   empty and LLM needs creative freedom (empty_ra_rs: -0.9 mean,
   incoherent mashups). Any directive targeting prose behavior
   needs to condition on source state.

5. **Strictness of the directive is what delivers the win**
   (V6 finding). Softening the V4/V5 directive's "every concrete
   element must appear" to a softer "if source has explicit
   content, preserve" surrendered the girl_sex gain (-0.5 vs V5).
   The LLM uses self-conditional language as license to decide the
   prompt doesn't warrant strict adherence. Don't optimize by
   making directives softer — find a different axis entirely.

6. **Lower temperature is NOT the variance fix** (V7 finding).
   Dropping prose temp 0.8 → 0.3 made things worse (-0.325
   overall). Lower temp converges on deterministic FAILURE
   trajectories for some seeds (seeds 42 and 1000003 both produced
   the weird 'female titan, acfun girl, high score girl' character-
   tag hallucination at temp 0.3, clearly reproducibly). Temperature
   reduces randomness but also removes escape paths from bad
   attractors. Don't try V7-style variants again.

7. **Multi-sample + picker IS the variance fix** (V8 finding).
   Keep temp 0.8 for variety, sample 3 proses, LLM picker selects
   the best for source fidelity. Works because when failure rate is
   < 1 (≈0.3 on girl_sex), P(all N samples fail) shrinks fast.
   Picker doesn't need to be brilliant — just needs to recognize
   obvious sanitization vs preservation. First variant to meaningfully
   beat baseline (+0.2 overall, +0.9 on girl_sex, +0.3 on empty+modifiers).

## Rubric notes (rubric.yaml v0)

Known friction, to revise after V4:
- Structure 4 vs 5 ambiguous when artist ordering is off.
- Coverage 3 vs 4 anchors too close.
- No dimension flags semantically-weak solo tags (`bent`, `pose`,
  `limbs`) — real DB tags but low meaning.

Revision happens between rounds; re-rate prior runs under new rubric
to keep comparisons valid.

## How to pick up in a new session

1. Read this file top-to-bottom — 3 min.
2. Check "Current status" for what's next.
3. Check "Findings that persist" before designing a new variant —
   don't re-test settled conclusions.
4. Drill into `_ratings.json` for the variant that matters to your
   next decision.
5. Session-level stuff (TaskList, ad-hoc prints) is lost on restart;
   this log is the ground truth.
