# Agent instructions — sd-webui-prompt-enhancer

This file gives agents (Claude Code, etc.) the project-specific context
they need to work effectively on this extension. Human contributors
can skim it but it's primarily written for AI sessions.

## Project state

Heavy active development. Single maintainer. Not yet a widely-used
extension. The RAG pipeline (`src/anima_tagger/`) was just built and
integrated into the Tag Validation flow in April 2026.

## No backwards compatibility required

**During the current development phase, do NOT write migration code,
legacy-name fallbacks, or deprecation shims.** Breaking changes to
settings keys, radio button values, internal APIs, module paths,
saved-image metadata fields, and yaml schema are all fine — there are
no deployed users with stale state to protect.

When the extension ships to a wider audience we'll reintroduce compat
concerns; until then, prioritize clean code over transitional crust.

Examples of what this means in practice:
- Renaming a setting key: just rename it; don't read both old + new.
- Removing a radio choice (e.g. `Check` mode): just remove it; don't
  add a `_restore_*` branch that maps old values to something else.
- Changing the HF artefact schema: bump `format_version` in
  `package_artifacts.py` and have `install.py` re-download; don't
  bother converting existing files.
- Renaming a module: just rename (though see "Naming conventions"
  below for specific modules that stay as-is).

## Naming conventions that DO stay stable

A few names are load-bearing even in dev:

- **`anima_tagger`** module under `src/` — referenced in HF dataset
  repo id (`freedumb2000/anima-tagger-artifacts`), in the user-facing
  "Anima Tagger" Settings section, and in persistent option keys
  (`anima_tagger_*`). Extending scope to other formats is fine; keep
  the name.
- **Settings keys** under `anima_tagger_*` — persisted in users'
  `config.json` files locally. Changing them silently invalidates
  their preferences. Rename only if you have a reason worth the cost.
- **Tag format yaml filenames** (`anima.yaml`, `illustrious.yaml`,
  `noobai.yaml`, `pony.yaml`) — referenced in the Tag Format
  dropdown's user-visible labels.

## Testing expectations

- For retriever/validator changes: run `src/anima_tagger/scripts/verify.py`.
- For end-to-end changes: `src/anima_tagger/scripts/full_pipeline_test.py`
  (requires Ollama running).
- New A/B test ideas go under `src/anima_tagger/scripts/` as
  `ab_*.py` (not `test_*.py` — that prefix is gitignored).
- The panels in verify.py and ab_*.py cover generic scenes, named
  characters, explicit artist references, and dragon/tower (non-human
  1other subject). Add new scenarios liberally.

## Commit style

One logical change per commit. Imperative subject, detailed body
(what + why), `Co-Authored-By: Claude Opus 4.7 (1M context)
<noreply@anthropic.com>` trailer.

Good recent examples: `c0e627b`, `64572b9`, `aad2e2c`.

## Developer pipeline vs end-user install

- `install.py` at the extension root runs on every Forge extension
  load. pip-installs deps, auto-downloads RAG artefacts from HF,
  falls back to rapidfuzz path with visible warning on failure.
  **Never** require the user to run anything from `src/` manually.

- Everything under `src/anima_tagger/scripts/` is DEV-only — the
  maintainer rebuilds the index or uploads artefacts. End users
  don't touch it.

- HF artefact upload: `package_artifacts.py` uses `huggingface_hub`
  with the maintainer's logged-in token (`hf auth login`). Credentials
  never go through the codebase; the script reads from
  `~/.cache/huggingface/token`.

## What the agent should NOT do

- Don't commit `data/` — gitignored (holds ~1.1 GB of artefacts).
- Don't run Git credential helpers (note the `install.py` upload
  warning says this explicitly).
- Don't add third-party vector DBs (Chroma, Qdrant, pgvector) — the
  project uses `faiss-cpu` directly and that's sufficient at 273k
  vectors.
- Don't introduce `langchain` / `haystack` / similar abstractions —
  overkill for a single-purpose retriever.
- Don't create README.md, dev docs, or any new documentation files
  unless explicitly asked.
- Don't add telemetry, usage analytics, or any phone-home code.

## Ollama dependency

Extension currently requires Ollama running on
`127.0.0.1:11434` with a model pulled. Replacing this with an
in-process LLM runner (llama-cpp-python) is in `TODO.md`. Until
that lands, assume Ollama is the only LLM path.
