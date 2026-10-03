# Tuning the enhancer's prose: method, findings, dead ends

What was learned on 2026-10-03 while fixing the enhancer, written for
whoever tunes it next (a later session here, or the same work on the
Wan2GP enhancer). The instrument is `experiments/prose_fidelity.py`.
Model throughout: `huihui_ai/qwen3.5-abliterated:9b`, temperature 0.8,
the extension's own `_call_llm`.

## What "good" means (operator's grading, first-hand)

The enhancer has two jobs and they are graded differently.

**Keep what the source states.** Failures, most serious first:

1. **Hard class — never dropped, never changed:** how many people, each
   one's gender and age; who does what to whom; the sex act or position
   when one is named; each person's state of dress; the pose; the medium
   or look (amateur phone photo, polaroid, CCTV); camera angle and
   framing when given.
2. **Contradiction:** an addition that cannot be true beside what is
   stated (a film stock on a phone snapshot, a DSLR on a CCTV view,
   "prone" for "on her back").
3. **Softened words:** an explicit term replaced by a euphemism.
4. **Lesser misses:** a dropped adjective or small prop.

**Fill what it leaves open.** Plausible invention is the enhancer's
purpose and is never a failure. Bad invention is: hedged ("perhaps a
shirt or a jacket"), vague ("clothing suited to the place"), not visible
(smells, sounds), or instructions instead of description.

Age, specifically: the form given stays. A number stays that number
(digits or words, either is fine); "middle-aged" stays "middle-aged" and
is not turned into an invented number. Fitting visible age cues may be
ADDED beside it (useful for models that ignore the number); a
replacement or a cue that does not fit is a failure. "She" after the
first mention is fine.

A bare generic term ("couple having sex") is kept AND made specific:
the word stays, the unspecified part (which act, where) is invented.

Camera angle or framing the source does not give: not invented. Asked
for in the source ("pick an unusual camera angle"), it is.

Target the operator named: 80% clean would be a success; 100% is not
expected.

## Method

- **Small rounds.** One change, all sources, one seed is enough to see a
  large effect (about one call per source). Confirm a candidate on three
  seeds only when it looks right. The first attempt here launched 105
  and then 135 calls at once and both were stopped.
- **One factor per round**, each with its own number. When two changes
  target different failure classes they can share a round as separate
  arms.
- **Test a candidate without editing the repo:** `--base-file`,
  `--append-file`, `--repeat-penalty`, `--presence-penalty`.
- **Read the outputs.** Every score in the tool is a keyword check. Two
  of them were wrong in ways only reading showed: `hand` did not match
  "one of her hands" (three false misses), and "sex" does not match
  "sexual intercourse".
- **Use the extension's own call.** The first harness left out the
  `/no_think` prefix production sends and so did not reproduce the
  extension's output. With the real `_call_llm`, source + seed reproduced
  the operator's output through the first sentence and a half.
- **Same-seed output is not reproducible enough to diff.** Narrative
  reproduced byte for byte across runs; Default did not. A same-seed
  comparison is a verifier only after the unchanged code has been shown
  to reproduce itself. To prove a refactor left the prompts alone,
  compare the REQUEST (system prompt, user message, options), not the
  output.

## Findings, in the order they mattered

1. **`repeat_penalty` 1.5 was the main defect.** The penalty applies to
   tokens in the recent context, and the source prompt is the recent
   context, so the model was pushed off the user's own words: handrail
   -> escalator / shopping cart handle, "two" -> "exactly three
   figures", on her back -> prone, kitchen -> galley-style eatery, plus
   garbled run-on grammar. At 1.1 (Ollama's default) all of that went.
   Clean outputs on eight stated-content sources, seed 42: 3/8 -> 8/8
   with the base edit below. The tell was in the outputs: errors that
   were swaps of the source's own words, and grammar missing its
   articles. Eight earlier variants had changed prompts, pipeline shape
   and temperature and never the sampling options.
2. **`presence_penalty` 1.5 did the same thing more quietly.** Removing
   it raised exact explicit terms kept (Default 20 -> 23 of 30,
   Narrative 19 -> 27 of 30) and kept phrasing closer to the source. No
   repetition loop and no truncation followed in about 600 outputs;
   `repeat_penalty` 1.1 is the guard that remains. The 1.5 values came
   from a "Qwen-recommended" commit aimed at tag-list loops, a mode that
   no longer exists.
3. **A base that demands something adds it everywhere.** The old
   Narrative base required a camera, lens and film stock on every
   prompt, so a source naming its own look got a contradicting one
   stacked on top.
4. **Worked examples and quoted phrases leak.** Narrative's example
   phrases came back near-verbatim (Hasselblad X2D 100c, warm amber
   color grading, intimate documentary feel). Default's example sentence
   ("an elderly fisherman with sun-worn wrinkled skin sits...") appears
   to have fixed the opening sentence's shape, which had no slot for an
   age: the long mall source lost "25" in the same sentence across
   several rule changes.
5. **A list of words to avoid plants those words.** Naming the
   euphemisms to avoid ("lovemaking, member, shaft, coupling, pleasure")
   inside the base made the model use exactly those on a bare source.
   The same list appended at the END of the system prompt reduced
   softeners instead, so placement matters, but the safe form is the
   positive one: "the same plain register, the direct anatomical and
   everyday words".
6. **A separate "source is ground truth" directive makes thin sources
   timid.** Appended to the base, it kept explicit words but produced
   placeholders on short sources ("hair and build suited to the bustle":
   vague filler in 5 of 15 short outputs against 1 of 15 without it).
   Folding the keep-rule into the base, next to an explicit instruction
   to invent, removed the trade-off.
7. **Rules about wording get echoed.** "The exact age as the same
   number" plus a "commit to one version" line produced "her exact age
   twenty-five years fixed upon her frame". A rule the model can quote
   will sometimes be quoted. Fewer, plainer rules did better than more.
8. **Instruction voice.** A base written as "do X to achieve Y" for
   camera matters leaks as imperatives in the output ("Apply a slightly
   grainy Kodak Portra 400 film stock"). "Describe the picture, never
   give instructions" did not cure Narrative; removing the camera
   section did.
9. **Hedging went away with concreteness, not with a ban.** "No 'or',
   'perhaps', 'maybe'" changed nothing measurable. "Invented details are
   concrete and named: the actual garment and its colour" took
   "perhaps/maybe" from 8 of 45 outputs to 0 of 45.
10. **Steering in the source works.** "describe surroundings" made
    outputs 25-30% longer with the extra on the surroundings; "pick an
    unusual camera angle" produced one in 3 of 3.
11. **`@` is a weak nudge on this model, `@@` is obeyed** (measured by
    the agent that built the sigils: "exactly one short sentence" gave
    two to four).

## The base that came out of it

`bases.yaml`, `Default`, written from scratch around findings 3-9: KEEP
(everything stated, same words, hard class named) then FILL (invent
concretely what is open, generic terms made specific), no worked
examples, no word lists, no camera unless the source has one. Three
seeds, eighteen sources, against the tuned old Default:

| | old Default, tuned | new base |
|---|---|---|
| stated elements kept (ten long sources) | 230/237 | 232-237/237 across rounds |
| age / gender / count kept | 37/39 | 39/39 in every round |
| "perhaps / maybe" | 8 of 45 | 0-1 of 54 |
| vague placeholder | 3 of 45 | 1 of 54 |
| camera mentioned with no angle given | 6 of 18 | 2 of 24 |
| "blowjob" kept, "sex" kept on short sources | 2/3, 0/3 | 3/3, 6/6 |

Still imperfect: smells and sounds slip in (about 8 of 54), and one
long explicit source dropped a detail in 2 of 3 seeds.

## Variety between seeds: what was tried, and what is shelved

The model alone does NOT give varied scenes. Eight seeds of "a man and
a woman having sex": a dimly lit bedroom or dim room in all eight, the
man dark-haired and broad-shouldered almost every time, man on top with
her legs around his waist in 7 of 11. Eight seeds of "a man and a woman
dancing": a dim room, a charcoal suit, an emerald gown. Temperature 1.1
changed little. The shipped "Random" modifiers only tell the model to
"be surprising", which is the same biased sampler. Variety has to come
from a pick the CODE makes.

**Two-pass prototype** (`experiments/variety_two_pass.py`, shelved by
operator decision 2026-10-03): the model lists several setups for the
source, the code picks one by seed, Enhance writes the scene around it.
Four versions:

1. Separate option lists per aspect, one pick each. The lists differ a
   lot between seeds (30 distinct locations over three lists, none
   shared), so a static list is not needed for variety. But independent
   picks do not fit together ("crowded train car, dim candle glow,
   curled around the pillow"), "unusual" options were impossible
   ("floating weightlessly"), and two lists for a sexual source
   contained underage-sounding entries ("young freckled girl",
   "youthful energetic teens"). Any list prompt MUST state that every
   person is an adult.
2. Ten whole setups in one line each. Coherent places and light, but the
   act stayed vague, ages were absent, about two in ten were far-fetched,
   and the poetic tone of the setup pulled euphemisms back into the
   scene.
3. Plain setups naming the act, ages and looks, each self-rated by the
   model as workable or far-fetched. Good variety (kitchen, sofa,
   bathroom, park bench, car, hallway; ages 20 to 60; for a bare
   "woman": a grocery aisle, a mirror, a kitchen chair, a garden). The
   self-rating is useless: 96 of 96 marked workable, garbled ones
   included. About one setup in four garbled.
4. Five separate fields per setup, temperature 0.8. People, place,
   clothing and light come out plain, varied and believable. An
   "outdoors" steer was mostly respected (one "inside a wooden cabin"
   in about twenty setups). Age-range steers were not reached.

**The limit that shelved it: the model cannot arrange two bodies.**
About 4 in 10 sexual setups were physically wrong ("vaginal intercourse
while lying flat on their backs", "standing up facing each other, in an
office chair") or drifted off the source ("cuddling", "interlocking
fingers"). The same limit shows in plain Enhance whenever the source
leaves the arrangement open: of eleven outputs from "a man and a woman
having sex" / "couple having sex", about four were physically
inconsistent (one gives him two pairs of hands). Where the source NAMES
the arrangement, Enhance keeps it. So the arrangement must come from
outside the model: stated in the source, or a named position picked by
code from a list. Physical inconsistency is the worst invention failure
(it becomes body horror in the image) and no keyword check sees it; it
is graded by reading.

Direction chosen instead: positions (and anything else) as modifier
lists, with a code-side random pick by seed. Not built yet.

## Not tested

Other models. Remix. The Motion and Negative blocks with the new base.
The modifier dropdowns. Image results: every number above is about the
text; whether the images follow is the operator's test.
