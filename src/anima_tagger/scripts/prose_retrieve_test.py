"""Alternative-architecture test: retrieve tags from prose directly.

Instead of asking the LLM to extract tags from the prose (which
hallucinates), run the prose through the existing Retriever and keep
top-K candidates. Every result is a guaranteed real Danbooru tag.

Compares three paths against cached drafts + proses:

  A. LLM draft → validate (baseline — existing pipeline)
  B. LLM draft → validate + compound_split (current best)
  C. Prose → retrieve top-K general tags (no LLM tag pass)
  D. Union of B and C, deduped

For each, reports composition-tag count (excluding always-prepended
quality/safety defaults) and which specific tags it produced.

Run:
    python src/anima_tagger/scripts/prose_retrieve_test.py
"""

import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import load_all, config
from anima_tagger.validator import TagValidator
from anima_tagger.shortlist import Shortlist
from anima_tagger.tagger import _context_from_shortlist


EXT_DIR = os.path.abspath(os.path.join(_SRC, ".."))
DRAFT_CACHE = os.path.join(EXT_DIR, ".ai", "drafts.json")

DEFAULTS = {
    "masterpiece", "best_quality", "score_7", "score_8", "score_9",
    "safe", "sensitive", "nsfw", "explicit",
}


def tokens_of(draft: str) -> list[str]:
    out = []
    for t in draft.replace("\n", ",").split(","):
        t = t.strip()
        if t:
            out.append(t)
    return out


def llm_validated(stack, entry: dict, *, compound_split: bool,
                  semantic_threshold: float = 0.70,
                  semantic_min_post_count: int = 50) -> set[str]:
    """Path A/B: run LLM draft tokens through validator."""
    tokens = tokens_of(entry["draft"])
    sl = Shortlist(
        artists=entry["shortlist"]["artists"],
        characters=entry["shortlist"]["characters"],
        series=entry["shortlist"]["series"],
    )
    ctx = _context_from_shortlist(sl, stack.db)
    v = TagValidator(
        db=stack.db, index=stack.index, embedder=stack.embedder,
        semantic_threshold=semantic_threshold,
        semantic_min_post_count=semantic_min_post_count,
        entity_min_post_count=1000,
    )
    results = (v.validate_with_compound_split(tokens, ctx)
               if compound_split else v.validate(tokens, ctx))
    return {r.canonical for r in results if r.canonical}


def prose_retrieved(stack, entry: dict,
                    final_k: int = 30,
                    retrieve_k: int = 120,
                    min_post_count: int = 100) -> set[str]:
    """Path C: retrieve tags directly from prose via Retriever."""
    # Generic (all-category) retrieval — then we'll post-filter to keep
    # broadly-useful general tags and let entities come from shortlist.
    cands = stack.retriever.retrieve(
        entry["prose"], retrieve_k=retrieve_k, final_k=final_k,
        min_post_count=min_post_count,
    )
    return {c.name for c in cands}


def prose_retrieved_by_category(stack, entry: dict,
                                general_k: int = 20,
                                meta_k: int = 5,
                                retrieve_k: int = 150,
                                min_post_count: int = 100) -> set[str]:
    """Path C2: category-aware retrieval — generals + a few meta tags.
    Entities (artist/character/series) come from the shortlist, not from
    free retrieval, to avoid name-collision with prose concepts."""
    out: set[str] = set()
    generals = stack.retriever.retrieve(
        entry["prose"], retrieve_k=retrieve_k, final_k=general_k,
        category=config.CAT_GENERAL, min_post_count=min_post_count,
    )
    out.update(c.name for c in generals)
    metas = stack.retriever.retrieve(
        entry["prose"], retrieve_k=retrieve_k, final_k=meta_k,
        category=config.CAT_META, min_post_count=min_post_count,
    )
    out.update(c.name for c in metas)
    return out


def prose_retrieved_artist(stack, entry: dict, top_k: int = 1,
                            retrieve_k: int = 200,
                            min_post_count: int = 500) -> set[str]:
    """Retrieve the single best-matching artist for the prose. Used
    when the user asked for a 'random artist' modifier — we want exactly
    one real, popular artist whose visual style matches the scene."""
    cands = stack.retriever.retrieve(
        entry["prose"], retrieve_k=retrieve_k, final_k=top_k,
        category=config.CAT_ARTIST, min_post_count=min_post_count,
    )
    return {c.name for c in cands}


def report(label: str, entry: dict, tags: set[str]) -> None:
    comp = sorted(t for t in tags if t not in DEFAULTS)
    print(f"  {label:25s} {len(comp):3d} comp tags")
    # Sample for eyeball
    print(f"    {', '.join(comp[:18])}{'…' if len(comp) > 18 else ''}")


def main() -> int:
    print("Loading cache …")
    with open(DRAFT_CACHE) as f:
        entries = json.load(f)["entries"]
    print(f"  {len(entries)} prompts")

    stack = load_all()
    print("Loading models …")
    t = time.perf_counter()
    with stack.models():
        print(f"  up in {time.perf_counter()-t:.1f}s")

        agg = {"A_baseline": [], "B_split": [], "C2_prose_by_cat": [],
               "D_union_B_C2": [], "D_plus_artist": []}
        for entry in entries:
            modifier = entry.get("modifier")
            print(f"\n« {entry['prompt']} »  (modifier={modifier})")
            A = llm_validated(stack, entry, compound_split=False,
                              semantic_threshold=0.80, semantic_min_post_count=100)
            B = llm_validated(stack, entry, compound_split=True,
                              semantic_threshold=0.70, semantic_min_post_count=50)
            t = time.perf_counter()
            C2 = prose_retrieved_by_category(stack, entry)
            c2_time = time.perf_counter() - t
            D = B | C2
            D_plus = set(D)
            if modifier == "random_artist":
                t = time.perf_counter()
                artist = prose_retrieved_artist(stack, entry, top_k=1)
                artist_time = time.perf_counter() - t
                D_plus |= artist
                print(f"    [artist retrieval {artist_time:.2f}s: {artist or 'none'}]")

            report("A baseline validator", entry, A)
            report("B split+looser", entry, B)
            report(f"C2 prose-by-cat ({c2_time:.2f}s)", entry, C2)
            report("D union(B,C2)", entry, D)
            report("D' +retrieved artist", entry, D_plus)

            agg["A_baseline"].append(len([t for t in A if t not in DEFAULTS]))
            agg["B_split"].append(len([t for t in B if t not in DEFAULTS]))
            agg["C2_prose_by_cat"].append(len([t for t in C2 if t not in DEFAULTS]))
            agg["D_union_B_C2"].append(len([t for t in D if t not in DEFAULTS]))
            agg["D_plus_artist"].append(len([t for t in D_plus if t not in DEFAULTS]))

        print("\n" + "=" * 70)
        print("AVERAGE COMPOSITION TAGS PER PROMPT")
        for k, v in agg.items():
            print(f"  {k:20s} {sum(v)/len(v):5.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
