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

_(none)_

## Done

_(none yet)_
