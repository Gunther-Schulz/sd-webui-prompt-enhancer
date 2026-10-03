# Backlog

Work that is decided but not built. An entry is READY when a fresh context
could execute it without making a design decision; PARKED entries name the
evidence they wait on. Closed entries move to `## Done` with their commit ref.

## Ready

### install.py's progress channel has no test, and the bug it fixes is invisible without one

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

_(none yet)_
