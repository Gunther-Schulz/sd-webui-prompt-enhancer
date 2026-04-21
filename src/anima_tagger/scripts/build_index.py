"""One-time: populate the tag DB, embed every tag, build the faiss index.

Reads the artefacts produced by download_data.py:
  - data/danbooru_tag_map.json   — tag metadata + wiki (NSFW-API Sept 2025 dump)
  - data/danbooru_posts.parquet  — 500k posts for co-occurrence stats

Filters tags by post_count >= MIN_POST_COUNT (drops ~1.3M low-signal
tags, leaving ~270k). Embeds each tag's (name + category + aliases +
wiki excerpt) with bge-m3 on GPU. Builds a faiss-gpu FlatIP index.

For co-occurrence: computes P(series|character) and P(character|series)
from the posts sample. Enables automatic series-pairing at query time
(e.g. hatsune_miku → vocaloid without the LLM's help).

Re-run any time the data files change. Full rebuild ~5-10 min on GPU.

Run:
    python src/anima_tagger/scripts/build_index.py
"""

import json
import os
import sys
from collections import defaultdict
from typing import Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import config
from anima_tagger.cooccurrence import CoOccurrence
from anima_tagger.db import TagDB
from anima_tagger.embedder import Embedder
from anima_tagger.index import VectorIndex


TAG_MAP_PATH = os.path.join(config.DATA_DIR, "danbooru_tag_map.json")
POSTS_PATH = os.path.join(config.DATA_DIR, "danbooru_posts.parquet")


_CATEGORY_STR_TO_INT = {
    "General":   config.CAT_GENERAL,
    "Artist":    config.CAT_ARTIST,
    "Copyright": config.CAT_COPYRIGHT,
    "Character": config.CAT_CHARACTER,
    "Meta":      config.CAT_META,
}
_CATEGORY_INT_TO_STR = {v: k for k, v in _CATEGORY_STR_TO_INT.items()}


def _load_and_filter_tags() -> list[dict]:
    """Read tag map JSON, filter by post_count, normalize into records."""
    print(f"Reading {TAG_MAP_PATH} …")
    with open(TAG_MAP_PATH) as f:
        doc = json.load(f)
    tags_raw = doc["tags"]
    print(f"  {len(tags_raw):,} total tags in dump")

    records: list[dict] = []
    for name, t in tags_raw.items():
        post_count = t.get("post_count", 0)
        if post_count < config.MIN_POST_COUNT:
            continue
        cat = _CATEGORY_STR_TO_INT.get(t.get("category"), config.CAT_GENERAL)
        aliases = t.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        wiki_body = ""
        wiki = t.get("wiki")
        if isinstance(wiki, dict):
            wiki_body = (wiki.get("body") or "").strip()
        # Normalize: replace any hyphens with underscores to match our
        # lookup convention (csv.py already does this for the existing
        # CSV; apply consistently here).
        clean_name = name.replace("-", "_")
        clean_aliases = ", ".join(a.replace("-", "_") for a in aliases if a)
        records.append({
            "name": clean_name,
            "category": cat,
            "post_count": int(post_count),
            "aliases": clean_aliases,
            "wiki": wiki_body[:1500],
        })
    print(f"  {len(records):,} tags after post_count >= {config.MIN_POST_COUNT} filter")
    with_wiki = sum(1 for r in records if r["wiki"])
    print(f"  {with_wiki:,} have wiki enrichment")
    return records


def _format_for_embedding(rec: dict) -> str:
    cat_name = _CATEGORY_INT_TO_STR.get(rec["category"], "General").lower()
    parts = [f"{rec['name'].replace('_', ' ')} ({cat_name})"]
    if rec["aliases"]:
        parts.append(f"aliases: {rec['aliases'].replace('_', ' ')}")
    if rec["wiki"]:
        parts.append(rec["wiki"][:800])
    return " | ".join(parts)


def _build_tag_db(records: list[dict]) -> None:
    if os.path.exists(config.TAG_DB_PATH):
        os.remove(config.TAG_DB_PATH)
    db = TagDB(config.TAG_DB_PATH, create=True)
    for i, r in enumerate(records):
        db.upsert(i, r["name"], r["category"], r["post_count"],
                  r["aliases"], r["wiki"])
        if (i + 1) % 50_000 == 0:
            db.commit()
    db.commit()
    print(f"  DB rows: {db.count():,} → {config.TAG_DB_PATH}")
    db.close()


def _build_faiss_index(records: list[dict]) -> None:
    print("Loading bge-m3 on GPU …")
    embedder = Embedder()
    texts = [_format_for_embedding(r) for r in records]

    print(f"Embedding {len(texts):,} tags … (this may take several minutes)")
    vecs = embedder.encode(texts, batch_size=128)
    print(f"  vectors: shape={vecs.shape} dtype={vecs.dtype}")

    # Release bge-m3 from GPU before building the index (free VRAM for
    # faiss GPU workspace).
    import gc, torch
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    print("Building faiss (CPU) IP index …")
    index = VectorIndex(config.EMBED_DIM, use_gpu=False)
    ids = np.arange(len(records), dtype=np.int64)
    index.add(vecs, ids)
    print(f"  index.ntotal = {index.size():,}")
    index.save(config.FAISS_INDEX_PATH)
    print(f"  → {config.FAISS_INDEX_PATH}")


def _build_cooccurrence(records: list[dict]) -> None:
    if not os.path.exists(POSTS_PATH):
        print("No posts parquet — skipping co-occurrence build")
        return

    # We need name→category lookup so we can target character→series pairs
    print(f"Reading {POSTS_PATH} for co-occurrence …")
    import pyarrow.parquet as pq
    table = pq.read_table(POSTS_PATH)

    chars_col = table["character"].to_pylist()
    copys_col = table["copyright"].to_pylist()
    artists_col = table["artist"].to_pylist()
    n_posts = len(chars_col)
    print(f"  {n_posts:,} posts")

    def _split(raw: str) -> list[str]:
        if not raw:
            return []
        return [t.strip().replace(" ", "_").replace("-", "_")
                for t in raw.split(",") if t.strip()]

    # Count character-wise aggregates: P(series | character), P(artist | character)
    # Also character-level total counts
    char_total: dict[str, int] = defaultdict(int)
    char_series: dict[tuple[str, str], int] = defaultdict(int)
    char_artist: dict[tuple[str, str], int] = defaultdict(int)
    # Series-wise: P(character | series) — useful for "show me miku if series is vocaloid"
    series_total: dict[str, int] = defaultdict(int)
    series_char: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(n_posts):
        chars = _split(chars_col[i])
        copys = _split(copys_col[i])
        artists = _split(artists_col[i])
        for c in chars:
            char_total[c] += 1
            for s in copys:
                char_series[(c, s)] += 1
            for a in artists:
                char_artist[(c, a)] += 1
        for s in copys:
            series_total[s] += 1
            for c in chars:
                series_char[(s, c)] += 1
        if (i + 1) % 100_000 == 0:
            print(f"    processed {i+1:,} posts …")

    # Lookup category by name (from records we just built)
    name_to_category = {r["name"]: r["category"] for r in records}

    def _dst_category(dst: str) -> int:
        return name_to_category.get(dst, config.CAT_GENERAL)

    print("Writing PMI rows …")
    if os.path.exists(config.COOCCURRENCE_PATH):
        os.remove(config.COOCCURRENCE_PATH)
    cooc = CoOccurrence(config.COOCCURRENCE_PATH, create=True)

    MIN_SRC_COUNT = 5      # character/series must appear in ≥5 posts
    MIN_PROB = 0.2         # keep only reasonably-strong associations
    rows_written = 0

    for (src, dst), count in char_series.items():
        total = char_total.get(src, 0)
        if total < MIN_SRC_COUNT:
            continue
        p = count / total
        if p < MIN_PROB:
            continue
        cooc.upsert(src, dst, _dst_category(dst), p)
        rows_written += 1

    for (src, dst), count in series_char.items():
        total = series_total.get(src, 0)
        if total < MIN_SRC_COUNT:
            continue
        p = count / total
        if p < MIN_PROB:
            continue
        cooc.upsert(src, dst, _dst_category(dst), p)
        rows_written += 1

    # Only include character→artist if the artist is very dominant (>0.5)
    # to avoid noise.
    for (src, dst), count in char_artist.items():
        total = char_total.get(src, 0)
        if total < MIN_SRC_COUNT * 4:
            continue
        p = count / total
        if p < 0.5:
            continue
        cooc.upsert(src, dst, _dst_category(dst), p)
        rows_written += 1

    cooc.commit()
    cooc.close()
    print(f"  {rows_written:,} PMI rows → {config.COOCCURRENCE_PATH}")


def main() -> int:
    if not os.path.exists(TAG_MAP_PATH):
        print(f"Missing {TAG_MAP_PATH}. Run download_data.py first.")
        return 1

    records = _load_and_filter_tags()

    print("\n— Building tag DB —")
    _build_tag_db(records)

    print("\n— Building faiss index —")
    _build_faiss_index(records)

    print("\n— Building co-occurrence table —")
    _build_cooccurrence(records)

    print("\nDone. Next: python src/anima_tagger/scripts/verify.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
