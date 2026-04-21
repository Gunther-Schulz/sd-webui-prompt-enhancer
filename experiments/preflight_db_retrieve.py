"""Pre-flight sanity check for ◆◇ Random Artist (db_retrieve).

Shows which Danbooru artist pe._resolve_source picks for each test
prompt × seed. Reveals whether the pool is scene-relevant before
committing to the full V11 rating.

Run from repo root:
    python experiments/preflight_db_retrieve.py

Requires: faiss, rapidfuzz, and the anima_tagger artefacts (already
downloaded by install.py). No Ollama needed — this only tests the
retrieval + seed-pick path, not any LLM call.
"""

from __future__ import annotations

import sys
import os
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from anima_tagger.scripts._pe_bootstrap import pe  # noqa: E402
from anima_tagger import load_all  # noqa: E402


SEEDS_10 = [42, 137, 1729, 7919, 10001, 65537, 99991, 524287, 1000003, 2147483000]

# Mirrors the prompts in experiments/prompts/v1_full.yaml that use Random Artist
TESTS = [
    ("girl_random_artist",                "girl"),
    ("samurai_random_artist",             "a lone samurai at dusk with a sword"),
    ("girl_random_artist_random_era",     "girl"),
    ("empty_random_artist_random_setting", "image"),  # empty source → use fallback query
]

ARTIST_SPEC = {
    "db_retrieve": {
        "category": 1,
        "min_post_count": 500,
        "final_k": 10,
    },
    "template": "Render scene in the style of @{display}.",
}


def main() -> None:
    if not hasattr(pe, "_resolve_source"):
        raise SystemExit("pe._resolve_source unavailable — bootstrap failure.")
    print("Loading anima stack + models (~10s on first run)...")
    stack = load_all()
    cm = stack.models()
    cm.__enter__()
    print("Loaded. Running pre-flight.\n")

    print("db_retrieve artist picks (Random Artist, category=1, min_pc=500):\n")
    hdr = f"{'prompt':<40} {'seed':>12}  {'picked @artist':<30}  pool"
    print(hdr)
    print("-" * len(hdr))

    per_prompt = {}
    for prompt_id, source_text in TESTS:
        per_prompt[prompt_id] = Counter()
        for seed in SEEDS_10:
            picked = pe._resolve_source(
                ARTIST_SPEC, seed,
                stack=stack, query=source_text,
            )
            if picked:
                per_prompt[prompt_id][picked["name"]] += 1
                print(f"{prompt_id:<40} {seed:>12}  @{picked['display']:<29}  pool={picked['pool_size']}")
            else:
                print(f"{prompt_id:<40} {seed:>12}  (EMPTY)")
        print()

    print("Summary — unique artists per prompt across 10 seeds:")
    for pid, counter in per_prompt.items():
        top3 = counter.most_common(3)
        print(f"  {pid:<42} {len(counter)} unique  top3={top3}")

    # Sanity: is the pool scene-relevant? A rough proxy — are picks
    # varied enough that scene-different prompts give different artists?
    all_sets = {pid: set(c) for pid, c in per_prompt.items()}
    pids = list(all_sets.keys())
    print("\nOverlap across prompts (Jaccard):")
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = all_sets[pids[i]], all_sets[pids[j]]
            if a or b:
                jac = len(a & b) / len(a | b)
                print(f"  {pids[i]:<38} ∩ {pids[j]:<38} {jac:.2f}")


if __name__ == "__main__":
    main()
