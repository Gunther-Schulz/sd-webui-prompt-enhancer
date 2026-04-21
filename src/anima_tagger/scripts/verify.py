"""Sanity check for the LLM-draft + retrieval-validate pipeline.

For each (source prompt, simulated LLM draft) pair, prints:
  - shortlist of real artists/characters/series the prose pass would
    see (RAG injection)
  - validator results per draft token (exact / semantic / drop)
  - final Anima tag list after rule layer

The simulated drafts intentionally include hallucinations we've seen
in real use (animedia, 4k, detailed_background, style_of_X) so we can
verify they get dropped.

Run:
    python src/anima_tagger/scripts/verify.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import load_all


# (source, simulated LLM draft, safety)
CASES = [
    (
        "a girl reading in a rainy cafe, cozy",
        # Mimics 9b abliterated output: mostly right, with a few typical leaks.
        "masterpiece, 1girl, solo, reading, book, cafe, rain, indoors, "
        "window, cozy, long hair, blue eyes, casual clothes, "
        "detailed_background, 4k, animedia",
        "safe",
    ),
    (
        "hatsune miku in a maid uniform, chibi",
        "1girl, hatsune miku, maid, maid uniform, chibi, apron, "
        "holding cake, twin tails, teal hair, aqua eyes, "
        "cute, Hatsune Miku, detailed_eyes",
        "safe",
    ),
    (
        "a lone samurai on a cliffside at dusk, sword in hand, by makoto shinkai",
        "masterpiece, 1boy, solo, samurai, katana, dusk, cliff, "
        "holding sword, long hair, traditional clothes, "
        "@makoto shinkai, style_of_makoto_shinkai, 1920x1080, "
        "wearing_traditional_armor",
        "safe",
    ),
    (
        "a dragon perched on a ruined stone tower at sunset",
        "dragon, tower, sunset, ruined, stone, perched, fantasy, "
        "no humans, outdoors, sky, clouds, animedia, detailed_scales",
        "safe",
    ),
]


def main() -> int:
    print("Loading anima_tagger stack …")
    t0 = time.perf_counter()
    stack = load_all()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s (cooccur={'yes' if stack.cooccurrence else 'no'})")

    for src, draft, safety in CASES:
        print("\n" + "=" * 72)
        print(f"SOURCE: {src!r}")
        print(f"DRAFT:  {draft}")

        # Shortlist (for prose-pass RAG)
        t = time.perf_counter()
        sl = stack.build_shortlist(src)
        print(f"\nSHORTLIST ({time.perf_counter()-t:.2f}s)")
        print(f"  artists:    {sl.artists}")
        print(f"  characters: {sl.characters}")
        print(f"  series:     {sl.series}")

        # Validator trace (per-token)
        from anima_tagger.tagger import _split_tokens
        tokens = _split_tokens(draft)
        t = time.perf_counter()
        results = stack.validator.validate(tokens)
        print(f"\nVALIDATOR ({time.perf_counter()-t:.2f}s)")
        for r in results:
            mark = "✓" if r.canonical else "✗"
            print(f"  {mark} {r.original!r:40} → {r.canonical!r:30} "
                  f"({r.match_type}, conf={r.confidence:.2f})")

        # Final pipeline output
        t = time.perf_counter()
        final = stack.tagger.tag_from_draft(draft, safety=safety)
        print(f"\nFINAL TAGS ({time.perf_counter()-t:.2f}s)")
        print("  " + ", ".join(final))

    return 0


if __name__ == "__main__":
    sys.exit(main())
