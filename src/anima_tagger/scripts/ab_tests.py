"""A/B tests for candidate retrieval/tagger improvements.

Each test toggles ONE feature on/off and prints the resulting shortlist
and final tag list for several scenarios. Eyeball the diffs to decide
whether the feature helps, hurts, or is neutral.

Tests:
  #2 multi-query characters (raw + expanded, merged)
  #6 character appearance auto-tags (from signature)
  #7 repeat-artist dedup (collapse by stem)

#1 (hybrid dense + sparse) needs the FlagEmbedding library; separate
script: ab_test_hybrid_retrieval.py.

Run:
    python src/anima_tagger/scripts/ab_tests.py
"""

import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import load_all, config
from anima_tagger.query_expansion import expand_query
from anima_tagger.shortlist import Shortlist


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "huihui_ai/qwen3.5-abliterated:9b"


def call_ollama(sp: str, up: str) -> str:
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": "5m",
        "options": {"temperature": 0.3, "num_predict": 200, "top_p": 0.8},
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": f"/no_think\n{up}"},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["message"]["content"].strip()


def make_expander():
    def _ex(src, mk):
        return expand_query(src, call_ollama, modifier_keywords=mk)
    return _ex


# ── #2 Multi-query characters (raw + expanded, merged) ─────────────

def characters_multiquery(stack, source: str, expander, per_cat_k: int = 6) -> list[str]:
    """Retrieve characters via BOTH raw source and expanded query, merge."""
    raw_cands = stack.retriever.retrieve(
        source, retrieve_k=config.DEFAULT_RETRIEVE_K,
        final_k=per_cat_k, category=config.CAT_CHARACTER, min_post_count=50,
    )
    expanded = expander(source, None)
    exp_cands = stack.retriever.retrieve(
        expanded, retrieve_k=config.DEFAULT_RETRIEVE_K,
        final_k=per_cat_k, category=config.CAT_CHARACTER, min_post_count=50,
    )
    seen: set[str] = set()
    merged: list[str] = []
    # Interleave: raw first, then expanded additions
    for src_list in (raw_cands, exp_cands):
        for c in src_list:
            n = c.name.replace("_", " ")
            if n not in seen:
                seen.add(n)
                merged.append(n)
            if len(merged) >= per_cat_k:
                return merged
    return merged


# ── #6 Character appearance auto-tags ─────────────────────────────

def character_autotags_for(stack, character_name: str, max_tags: int = 5) -> list[str]:
    """Look up character's co-occurrence signature and return top
    appearance-ish tags. Signature isn't stored on disk — we approximate
    by re-computing top cooc from posts at call time, which is costly.
    For A/B test, we parse the embedding text string stored in the DB
    wiki (if signatures were built that way) or fall back to the
    cooccurrence table if available.
    """
    # Without signature persistence, fall back to using the cooccurrence
    # table entries for this character (general category) if any.
    if not stack.cooccurrence:
        return []
    # Heuristic: take top co-occurring tags of ANY category, filter to
    # appearance-domain keywords.
    APPEARANCE_KEYWORDS = (
        "hair", "eyes", "skin", "ears", "tail", "horn", "wings", "fang",
        "glasses", "mask", "hat", "ribbon", "tattoo", "blush",
    )
    results = stack.cooccurrence.top_for(character_name, top_k=40, min_prob=0.1)
    out = []
    for r in results:
        tag = r["tag"]
        if any(kw in tag for kw in APPEARANCE_KEYWORDS):
            out.append(tag.replace("_", " "))
            if len(out) >= max_tags:
                break
    return out


# ── #7 Repeat-artist / character dedup by stem ─────────────────────

def dedup_by_stem(names: list[str]) -> list[str]:
    """Collapse names sharing a stem (part before '(' or an obvious
    disambiguation suffix). Keeps the shortest variant per stem."""
    by_stem: dict[str, str] = {}
    order: list[str] = []
    for n in names:
        # Strip parenthesized suffix: "rococo (girl cafe gun)" → "rococo"
        stem = n.split(" (")[0].strip()
        # Collapse repeated words at end ("hatsune miku (nt)" → "hatsune miku")
        if stem not in by_stem or len(n) < len(by_stem[stem]):
            if stem not in by_stem:
                order.append(stem)
            by_stem[stem] = n
    return [by_stem[s] for s in order]


# ── Test scenarios ─────────────────────────────────────────────────

SCENARIOS = [
    ("reading-no-entity",  "a girl reading in a cafe"),
    ("miku-explicit",      "hatsune miku in a maid uniform, chibi"),
    ("miku-studio",        "hatsune miku in a photo studio"),
    ("samurai-dusk",       "a lone samurai at dusk with a sword"),
    ("dragon-tower",       "a dragon perched on a ruined tower at sunset"),
]


def main() -> int:
    print("Loading stack …")
    stack = load_all()
    expander = make_expander()

    with stack.models():
        for label, source in SCENARIOS:
            print("\n" + "=" * 80)
            print(f"SCENARIO: {label}  —  {source!r}")

            # Baseline shortlist (current production pipeline)
            base_sl = stack.build_shortlist(source, query_expander=expander)
            print(f"\nBASELINE characters: {base_sl.characters}")
            print(f"BASELINE artists:    {base_sl.artists}")

            # #2 Multi-query characters
            mq_chars = characters_multiquery(stack, source, expander)
            diff_mq = set(mq_chars) - set(base_sl.characters)
            print(f"\n#2 multi-query characters: {mq_chars}")
            print(f"   new vs baseline: {sorted(diff_mq) if diff_mq else '(none)'}")

            # #6 Character autotags (apply to the top-1 character if present)
            print(f"\n#6 character autotags:")
            for char in base_sl.characters[:1]:
                # Pass through the underscore form used by cooccurrence
                key = char.replace(" ", "_")
                autos = character_autotags_for(stack, key)
                print(f"   {char!r}: {autos}")

            # #7 Stem dedup
            dd_chars = dedup_by_stem(base_sl.characters)
            dd_arts = dedup_by_stem(base_sl.artists)
            print(f"\n#7 stem-dedup characters: {dd_chars}")
            print(f"#7 stem-dedup artists:    {dd_arts}")
            drop_chars = [c for c in base_sl.characters if c not in dd_chars]
            drop_arts = [a for a in base_sl.artists if a not in dd_arts]
            if drop_chars or drop_arts:
                print(f"   dropped: chars={drop_chars}, arts={drop_arts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
