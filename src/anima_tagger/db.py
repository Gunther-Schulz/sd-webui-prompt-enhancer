"""SQLite tag metadata store.

Schema mirrors what we need at retrieval time:
  - tag name (canonical, underscore form)
  - category (0 general, 1 artist, 3 copyright, 4 character, 5 meta)
  - post_count (popularity)
  - aliases (comma-separated)
  - wiki (enrichment text, may be empty)

The faiss index uses row `id` as the vector ID, so the id column is the
join key between index results and metadata. Do not reorder or delete
rows after building — append only if you want the index to stay aligned.
"""

import os
import sqlite3
from typing import Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS tags (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    category   INTEGER NOT NULL DEFAULT 0,
    post_count INTEGER NOT NULL DEFAULT 0,
    aliases    TEXT NOT NULL DEFAULT '',
    wiki       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tags_name     ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
CREATE INDEX IF NOT EXISTS idx_tags_popular  ON tags(post_count DESC);
"""


class TagDB:
    """Thin wrapper around the SQLite tag store."""

    def __init__(self, path: str, create: bool = True):
        if create:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        if create:
            self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def upsert(self, tag_id: int, name: str, category: int,
               post_count: int, aliases: str = "", wiki: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tags VALUES (?, ?, ?, ?, ?, ?)",
            (tag_id, name, category, post_count, aliases, wiki),
        )

    def commit(self) -> None:
        self.conn.commit()

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM tags")
        return cur.fetchone()[0]

    def get_by_id(self, tag_id: int) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM tags WHERE id = ?", (tag_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_by_ids(self, tag_ids: Iterable[int]) -> list[dict]:
        ids = list(tag_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self.conn.execute(
            f"SELECT * FROM tags WHERE id IN ({placeholders})", ids,
        )
        by_id = {row["id"]: dict(row) for row in cur.fetchall()}
        # Preserve the caller-requested order
        return [by_id[i] for i in ids if i in by_id]

    def get_by_name(self, name: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM tags WHERE name = ?", (name,))
        row = cur.fetchone()
        return dict(row) if row else None

    def all_names(self) -> list[str]:
        cur = self.conn.execute("SELECT name FROM tags")
        return [row[0] for row in cur.fetchall()]

    def build_alias_lookup(self) -> dict[str, str]:
        """Return {alias_token: canonical_name}. Built once per process."""
        lookup: dict[str, str] = {}
        cur = self.conn.execute(
            "SELECT name, aliases FROM tags WHERE aliases != ''"
        )
        for name, aliases_str in cur:
            for a in aliases_str.split(","):
                a = a.strip().replace("-", "_").replace(" ", "_").lower()
                if a and a != name and a not in lookup:
                    lookup[a] = name
        return lookup

    def iter_ordered(self) -> Iterable[dict]:
        """Yield all tags in ascending id order (index-alignment contract)."""
        cur = self.conn.execute("SELECT * FROM tags ORDER BY id ASC")
        for row in cur:
            yield dict(row)
