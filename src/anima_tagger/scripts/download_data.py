"""Download Danbooru tag metadata + wiki + post stats from HuggingFace.

Data sources (all freely available, cc0 / public):

  1. NSFW-API/DanBooruTagsAndWikiDumpSept2025
     Single JSON file `danbooru_tag_map_with_wiki.json` (~400 MB) with
     every Danbooru tag including category, post_count, aliases, and
     wiki body. Cutoff date matches Anima's training cutoff (Sept 2025).
     Source of truth for tag metadata.

  2. isek-ai/danbooru-tags-2024
     Parquet dataset of actual Danbooru posts (~5M rows). Used only to
     compute character→series co-occurrence during build_index. Each
     post has tag lists which we aggregate into conditional probability
     P(series | character).

Outputs to DATA_DIR:
  - danbooru_tag_map.json    (renamed copy of the NSFW-API file)
  - danbooru_posts.parquet   (sampled slice of isek-ai; optional)

Run:
    python src/anima_tagger/scripts/download_data.py
"""

import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from anima_tagger import config


# Primary tag metadata + wiki
TAG_MAP_REPO = "NSFW-API/DanBooruTagsAndWikiDumpSept2025"
TAG_MAP_FILE = "danbooru_tag_map_with_wiki.json"

# Post-level data for co-occurrence. Large — only a sample is used.
POSTS_REPO = "isek-ai/danbooru-tags-2024"
POSTS_SAMPLE_ROWS = 500_000  # enough for robust character→series stats


def _download_tag_map() -> bool:
    from huggingface_hub import hf_hub_download

    out_path = os.path.join(config.DATA_DIR, "danbooru_tag_map.json")
    if os.path.exists(out_path):
        print(f"  already present: {out_path}")
        return True
    try:
        print(f"  downloading {TAG_MAP_FILE} from {TAG_MAP_REPO} …")
        cached = hf_hub_download(
            repo_id=TAG_MAP_REPO,
            filename=TAG_MAP_FILE,
            repo_type="dataset",
        )
        shutil.copy(cached, out_path)
        print(f"  → {out_path} ({os.path.getsize(out_path)//1024//1024} MB)")
        return True
    except Exception as e:
        print(f"  ✗ failed: {type(e).__name__}: {e}")
        return False


def _download_posts_sample() -> bool:
    from datasets import load_dataset

    out_path = os.path.join(config.DATA_DIR, "danbooru_posts.parquet")
    if os.path.exists(out_path):
        print(f"  already present: {out_path}")
        return True
    try:
        print(f"  streaming {POSTS_REPO} and taking {POSTS_SAMPLE_ROWS:,} rows …")
        ds = load_dataset(POSTS_REPO, split="train", streaming=True)
        rows: list[dict] = []
        for i, row in enumerate(ds):
            # Keep only columns we need for co-occurrence
            rows.append({
                "general": row.get("general", ""),
                "character": row.get("character", ""),
                "copyright": row.get("copyright", ""),
                "artist": row.get("artist", ""),
            })
            if len(rows) >= POSTS_SAMPLE_ROWS:
                break
            if (i + 1) % 50000 == 0:
                print(f"    {i+1:,} rows …")
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, out_path)
        print(f"  → {out_path} ({os.path.getsize(out_path)//1024//1024} MB, {len(rows):,} rows)")
        return True
    except Exception as e:
        print(f"  ✗ failed: {type(e).__name__}: {e}")
        return False


def main() -> int:
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("1/2 Tag metadata + wiki (NSFW-API Sept 2025 dump) …")
    _download_tag_map()

    print("\n2/2 Danbooru posts sample (for co-occurrence stats) …")
    _download_posts_sample()

    print("\nDone. Run build_index.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
