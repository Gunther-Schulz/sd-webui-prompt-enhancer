# Backlog

Work that is decided but not built. An entry is READY when a fresh context
could execute it without making a design decision; PARKED entries name the
evidence they wait on. Closed entries move to `## Done` with their commit ref.

## Ready

_(none)_

## Parked

### Generate and the enhancer can run at the same time, and together they run out of memory

**PARKED 2026-10-03, operator report** ("a shim to block the generate button while
the prompt enhancer runs (forge) because they get in each other's way (will get
OOM)"; ranked by the operator as not very important).

The enhancer's LLM (Ollama) and Forge's image model share one machine. Nothing
stops a Generate click while an enhancer call is in flight, or an enhancer click
while an image is generating. The OOM is the operator's observation, not measured
in this repo.

**Waiting on** (design is open; each item is one short measurement or read):
- Which memory runs out — GPU or host — and whether the LLM is still loaded after
  the enhancer call returns (`curl 127.0.0.1:11434/api/ps` right after a call).
  If it stays loaded, unloading it on completion may matter more than any button
  lock, and a lock alone would not help a Generate clicked one second later.
- Which direction needs the lock: Generate blocked during enhancement (the
  request), the enhancer buttons blocked during generation, or both.
- Where a lock can live. `javascript/` is empty today, so there is no existing
  button handling to extend; the candidates are a small script disabling Forge's
  Generate button while the enhancer's status is in progress, or a server-side
  check. Neither has been read against Forge's UI code.

## Done

### Cut this extension down to prose only, with LoRA survival

**DONE 2026-10-03** — `47389de` (the cut), `f77c9cb` (prose_adherence wording), `9d2a6c3` (Enhance rename, README and CLAUDE.md), `c868a38` (Keep LoRAs). Built by a dispatched agent, verified at the desk: every tracked .py compiles, no `anima_tagger` reference remains under scripts/ or experiments/, both YAMLs parse, `experiments/prose_fidelity.py` runs against the result; the agent's handler probe showed system prompt, user message and options identical before and after the cut in 11 of 11 cases. Deviations accepted: `install.py` and `requirements.txt` deleted outright (nothing tag-free remained in them); Keep LoRAs done in Python through the `prompt_in` pull, not in the write-back JS. The Custom base went too (operator decision the same day: `@@` replaces it); Narrative and Default are the remaining bases. NOT verified: loading the extension inside Forge and clicking the buttons — first real test is the operator's. The two questions this entry waited on (the three checkboxes, which bases survive) stay with the operator's re-test and are not blockers.

**Waiting on:** two operator decisions, both asked 2026-10-03 and both waiting
on the operator trying things in Forge after that day's fidelity fix (`83ba634`,
`253f227`): (1) which of the `Prepend` / `+ Negative` / `+ Motion and Audio`
checkboxes survive — never used so far, desk recommendation cut all three;
(2) which bases survive — the operator wants ONE base if one serves Anima, Krea 2
and Z-Image, plus `Custom` and Local Overrides for experiments; only images can
decide that, so the cut ships `Default`, `Narrative` and `Custom` and the
comparison picks between the first two.

Decided 2026-10-03 (operator, first-hand; supersedes the 2026-08-21 questions):
- **Cut IN PLACE in this repo.** No second "lite" extension, no rename, no change
  to the clone line in `wan2gp/forge-config.sh`. Reason given: only prose is
  used now (Anima, Krea 2 — natural-language models); the tag half is unwanted.
- **Modifier dropdowns STAY for now.** They were tried and found weak, possibly
  because of the defects fixed that day; the operator re-tests.
- **Remix STAYS for now**, same reason, same re-test. This reverses 2026-08-21's
  "Remix is CUT", and it changes the Keep-LoRAs design below: Remix reads the
  main prompt field as LLM input, so `<...>` tokens must be stripped before send
  on that path.
- **Tag Format and Tag Validation controls are cut** with the tag half.
- **No streaming text display.** The status line stays as it is.

Requested 2026-08-21: a simple prompt enhancer with no Danbooru/RAG and no style
presets.

**Cut:** `src/anima_tagger/` entire, tag formats, tag post-processing, the Hybrid
and Tags modes with their handlers, `experiments/` except `compare_prompt.py`
(the prose harness the 2026-10-03 fix was measured with), the `detail_level`
stub, and `install.py`'s artefact download (all five
`_DEPS` entries are tagging-half; a prose-only `_DEPS` is empty, which also makes
`_open_console`/`_say` dead code — worth re-porting if a slow install step ever
returns).

**Keep:** Prose mode, the Base dropdown including `Custom`, Source Prompt textbox,
the single generate button (the `✍ Prose` button renamed — proposed `✨ Enhance`,
since Forge's own Generate button already owns the word "Generate"; the
four-button row collapses to one button plus Cancel), the status line,
Temperature, Seed + its
dice/reuse buttons, Think, API URL, Model dropdown, Local Overrides, Reload,
Models, the Ollama status line, and the whole `_call_llm` streaming client.

**Entanglement is low.** Every `anima_tagger` import is lazy and inside a function
body; none is reachable from `_enhance`. Prose's real dependencies are exactly
`bases.yaml`, `prompts.yaml` and the Ollama client. The one hidden thread is
`_collect_modifiers` → `_resolve_source` (`:346-351`), which imports
`anima_tagger.config` and queries the Danbooru SQLite for `source:`-tagged
modifiers — cutting the modifier dropdowns removes it for free. `_load_tag_formats()`
runs unconditionally at import (`:1336`) and needs deleting from UI construction,
not from any Prose-reachable function.

**New feature — Keep LoRAs (checkbox, default on).** Any `<...>` token in Forge's
main prompt field survives a regeneration, appended to the enhanced result, so a
LoRA under test stays put while the prose churns. Operator's rule taken literally:
any string starting with `<` and ending with `>`, so `<hypernet:...>` rides along
without a code change. Position is not preserved — Forge's parser strips these
wherever they sit, so placement carries no meaning.

Nothing has to be un-built first: no code in `scripts/prompt_enhancer.py`,
`src/anima_tagger/` or `experiments/` detects, strips, preserves or special-cases
`<...>` tokens anywhere — not in `source_prompt` handling, not in Remix's pull or
push, not in `_clean_output` (which only strips `*`/`_` markdown emphasis,
`:1640-1644`). Established by repo-wide search on 2026-08-21 with a positive
control (`has_inline_wildcards` → 2 hits) proving the instrument fires, so the
zero is a true absence.

**The whole change fits inside the existing write-back JS** (`:3250-3264`), which
already holds the textarea: read `ta.value` BEFORE overwriting it, pull the
`<...>` tokens out of the old value, and append any of them not already in the
new value. No Python-side plumbing and no reuse of Remix's `prompt_in` pull
bridge — an earlier draft of this entry specified that bridge and it was
unnecessary work.

There is no strip-before-send half to build either: nothing sends the main prompt
field to the model once Remix is cut. `_enhance`'s inputs are the Source Prompt
box plus settings (`:2415-2422`) — the main field is write-only, so the tokens
never reach the LLM and only need carrying across the overwrite. (Remix was the
one mode that read that field as LLM input, which would have needed the tokens
stripped before send; cutting it removes that case.)

Gate it on the Keep LoRAs checkbox by passing the checkbox as a second `inputs=`
entry to the same `.change()` handler — a `_js=` function takes multiple
arguments. Remix has the inverse bridge that pulls the live field first (`prompt_in`,
`:2335`, `:2959-2963`); the build is wiring that pull onto the Prose path, then
extract-before-send and re-append-after. Extract with `<[^<>]+>` — refusing nested
brackets is what stops one match swallowing a whole prompt containing two — collapse
the whitespace left behind, and skip on re-append any token already present so the
path is idempotent. Known limitation, accepted: prose containing bare comparison
brackets ("5 < 10 > 3") yields a spurious token; the stricter alternatives start
dropping LoRA names containing spaces.

**Decided 2026-10-03: not wanted.** There is no live streaming TEXT display today. The
status line updates ~1/s with word count / elapsed / tok-per-sec (`:2217`,
`:2379-2390`); the result textboxes are hidden and receive `gr.update()` no-ops
mid-stream, real text only on the final yield. A visibly-streaming output is new
work, small but not a port.

*Write boundary:* this repo — `scripts/prompt_enhancer.py`, `install.py`,
`bases.yaml`, `prompts.yaml`, `README.md`, and the deletions listed under Cut.

*Verifier:* the extension loads in Forge with no `anima_tagger` import reachable
(`command grep -rn "anima_tagger" scripts/` → 0 hits, with a positive control
proving the pattern fires); Prose generates end to end against Ollama; Cancel
interrupts mid-stream; a LoRA token in the main prompt field survives a
regeneration and is not doubled on a second one.

*Done-criterion:* the same source + seed + base produces the same Prose output
after the cut as before it (commit `83ba634` as the reference), and Remix still
runs end to end.

### `@` / `@@` inline system-prompt sigils in the source prompt

**DONE 2026-10-03** — `dd6d399`. Both sigils built as specified, on Enhance and Remix. `@@` replaces the whole system prompt, so the adherence directive and the + Motion / + Negative blocks are not appended on such a run. Measured by the building agent: `@@` is obeyed; `@` is a weak nudge on qwen3.5-abliterated:9b (asked for one sentence, got two to four). The check script lived in the agent's scratch and is not in the repo.

**Waiting on:** the operator testing whether the existing `Custom` base covers the
need. It discriminates cleanly. If `Custom` is enough, only `@` (append) remains
worth building and `@@` is dropped as duplicate; if reaching for the Custom box
mid-session turns out to be the friction, both ship.

Ported from WanGP, which is the definition, not a starting point:
`Wan2GP/docs/PROMPTS.md:610-657` (behaviour) and
`Wan2GP/shared/prompt_enhancer/prompt_enhance_utils.py:162-187` (reference
implementation). Requested 2026-08-21 as "the same feature we have in Wan2GP".

**Semantics** — the user types the sigil into the Source Prompt box:

- `prompt @ extra instructions` — the suffix is appended to the assembled system
  prompt under a fixed joining line, verbatim from upstream:
  `Follow these additional user instructions with higher priority if they conflict with the guidance above:`
- `prompt @@ replacement` — the suffix REPLACES the assembled system prompt.
- `@@` is tested BEFORE `@`, split on first occurrence, both halves stripped. The
  sigil and everything after it never reach the user message.
- An empty suffix changes nothing, replace or not, so a half-typed `foo @@`
  degrades to normal behaviour rather than sending the model no instructions.
- Deliberately NOT ported: upstream folds a thinking super-system-prompt into the
  same merge. `Think` here is request-side only — `payload["think"]`, a `top_p`
  swap, and a `/no_think\n` user-content prefix (`scripts/prompt_enhancer.py:1937-1952`)
  — so the two compose independently.
- No escape for a literal `@`, matching upstream. `a poster for @midnight` will
  split; the failure is visible in the output rather than silent.

**Why `@` is the half that matters.** `@@` duplicates what the `Custom` base
already does — `_assemble_system_prompt:2116-2117` takes `custom_system_prompt`
verbatim and skips the `_preamble`/`_format` wrapping, and with `detail_level`
pinned to 0 (`:2278`, `_build_detail_instruction:1383-1384` returns None) nothing
is appended afterwards, so `Custom` is a TOTAL replacement today. `@` has no
equivalent anywhere: extending the selected base with one extra instruction
currently means pasting the whole base body into the Custom box and editing it,
which loses the base as a base.

**Design (decided).** Two pure functions plus one seam. `split_sigil(prompt)` →
`(body, suffix, replace)`; `merge_system_prompt(sp, suffix, replace)` → the
effective system prompt. Hook them in `_enhance` between `sp =
_assemble_system_prompt(...)` (`:2354`) and the `user_msg = f"SOURCE PROMPT:
{source}"` construction (`:2364`) — `body` feeds the wrapper, the merged system
prompt feeds `_call_llm`. Applying the sigil to the RESULT of
`_assemble_system_prompt` rather than inside it is what makes `@@` a true total
override. Reference implementation, already written and mutation-tested:

    def split_sigil(prompt):
        prompt = str(prompt or "").strip()
        body, separator, suffix = prompt.partition("@@")
        if separator == "@@":
            return body.strip(), suffix.strip(), True
        body, separator, suffix = prompt.partition("@")
        if separator == "":
            return prompt, "", False
        return body.strip(), suffix.strip(), False

    def merge_system_prompt(system_prompt, suffix, replace=False):
        system_prompt = str(system_prompt or "").rstrip()
        suffix = str(suffix or "").strip()
        if not suffix:
            return system_prompt
        if replace:
            return suffix
        return f"{system_prompt}\n{APPEND_JOINER}\n{suffix}"

`merge_system_prompt` must tolerate an empty/None base — a `Custom` base with an
empty textbox assembles to `None` and `@@` has to work there.

*Note on `@` in this codebase:* `@` already prefixes ARTIST tags throughout the
tag pipeline (`:616`, `:620`, `:2686`, `:3174`; `src/anima_tagger/rule_layer.py:159-193`;
`validator.py:135`) — that is the "@-prefix" CLAUDE.md:303 refers to. Different
field, output side, no live collision: the only character-level parser on raw
`source` today is the `{name?}` wildcard detector (`:1636`). But any doc written
for this feature must disambiguate the two.

*Write boundary:* `scripts/prompt_enhancer.py` (the `_enhance` seam), plus a new
`tests/check_sigils.py`. One seam only — the lite extension keeps a single mode,
so the awkward case is gone: Remix folded `source` into the SYSTEM prompt
(`:2825`) rather than the user message, where a sigil would have interacted
differently. If this lands in the CURRENT extension instead of the lite one,
Hybrid and Tags need their own hooks at `:2453` and `:3009`.

*Verifier:* a mutation-driven test, already written and proven. Baseline green
first, then each mutant red — six mutants for six distinct defect classes, since a
red on one certifies that class only: `@` tested before `@@` (caught by 10 checks),
no stripping (6), empty suffix blanking the system prompt (1). Expectations derive
from `PROMPTS.md`, never from the implementation they grade. Wire it through
`src/anima_tagger/scripts/_pe_bootstrap.py` like `experiments/compare_prompt.py`
does, NOT as a standalone reimplementation — `tests/check_tags_pipeline.py` is the
island-test shape CLAUDE.md rule 5 warns against and would not catch a sigil
regression.

*Done-criterion:* `experiments/compare_prompt.py --dry --source "x @ be terse"`
shows the joiner and suffix in the assembled system prompt and a clean body in the
user message; the mutation test passes clean and goes red on each of the six
mutants.

### install.py's progress channel has no test, and the bug it fixes is invisible without one

**DROPPED 2026-10-03** — the subject no longer exists: `install.py` was deleted in `47389de` with the tag half it installed for.

**Found 2026-08-21**, the hard way. The Anima artefact download left the Forge
console silent after `Version: neo 2.28` and read as a hang. The cause was not
in this repo's printing at all: Forge runs each extension `install.py` through
`modules/launch_utils.run()` with `live=False`, which pipes **both** stdout and
stderr (`modules/launch_utils.py:69-70`) and prints the collected output only
after the process exits (`:171-173`). So no print from here reached the console
while the 1.1 GB download ran, however often it was flushed and whichever
stream it chose.

Fixed in `a4184d4` by writing progress to the controlling terminal via
`/dev/tty` as well as stdout. **Nothing guards that.** `_open_console()` looks
like a defensive nicety rather than the entire point, so a later tidy-up that
drops it — or that "simplifies" `_say` back to a plain `print` — restores the
original bug exactly, and restores it silently: the output still appears, just
an hour late, which is indistinguishable from working unless someone is
watching a real Forge start.

The first fix attempt missed this because it was verified by running
`install.py` directly. That harness could not see the defect: the capture lives
in the caller, not in the script. Reproducing the caller is the whole method,
and it is the part worth freezing into a test.

**The design:** `tests/check_install_progress.py`, run the way
`tests/check_tags_pipeline.py` is. It spawns `install.py` exactly as Forge does
— `subprocess.Popen(..., stdout=PIPE, stderr=PIPE)` — against a local fixture
served over `file://` so no network and no real artefacts are involved, and
asserts the discriminating **pair**:

- **without** a controlling terminal, the child's own stdout yields nothing
  until exit — this reproduces the defect and proves the harness can see it;
- **under a pty** (`pty.openpty()`, or the run wrapped in
  `script -qec ... /dev/null`), progress lines appear on the terminal *while*
  the child is still running.

The second half is the assertion that fails if `/dev/tty` is dropped. The first
half is what stops the test passing vacuously on a harness that could never
have observed a difference — without it, a test that always reports "live" is
byte-identical to a correct one.

*Write boundary:* `tests/check_install_progress.py` (new), and a line in
`CLAUDE.md`'s verify section naming it. `install.py` is NOT touched — the test
grades it, so deriving the expectation from it would move with the mutant.

*Verifier:* the test itself, proven red first by reverting `_say` to a plain
`print` and confirming the pty half goes red while the no-tty half stays green
— a red on both halves means the harness broke, not the fix.

*Done-criterion:* the test passes on the current tree, goes red on a `_say`
reverted to plain `print`, and needs neither network nor the real 1.1 GB
artefacts to run.

_(none yet)_
