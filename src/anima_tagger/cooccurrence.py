"""Tag co-occurrence table.

Precomputed pointwise mutual information (PMI) over a Danbooru post
dump. Given a tag, quickly returns the tags that most reliably follow
it in real posts (e.g. hatsune_miku → vocaloid ~95% of the time).

Storage is a small SQLite table keyed by (src_tag, dst_tag) with the
conditional probability P(dst | src). At query time we pick the top-N
by probability optionally restricted to a tag category.

If the table isn't built (data not downloaded), the class returns
empty results gracefully — the rest of the pipeline still works, just
without the automatic series-pairing lift.
"""

import os
import sqlite3
from typing import List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS cooccur (
    src_tag      TEXT NOT NULL,
    dst_tag      TEXT NOT NULL,
    dst_category INTEGER NOT NULL DEFAULT 0,
    p_given_src  REAL NOT NULL,
    PRIMARY KEY (src_tag, dst_tag)
);
CREATE INDEX IF NOT EXISTS idx_cooccur_src ON cooccur(src_tag, p_given_src DESC);
"""


class CoOccurrence:
    def __init__(self, path: str, create: bool = False):
        self.path = path
        self.available = os.path.exists(path) or create
        if not self.available:
            self.conn = None
            return
        if create:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # check_same_thread=False: shared across Gradio worker threads
        # via the cached AnimaStack; runtime access is read-only (writes
        # happen offline in scripts/build_cooccurrence.py).
        self.conn = sqlite3.connect(path, check_same_thread=False)
        if create:
            self.conn.executescript(SCHEMA)

    def close(self):
        if self.conn:
            self.conn.close()

    def upsert(self, src: str, dst: str, dst_category: int,
               p_given_src: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cooccur VALUES (?, ?, ?, ?)",
            (src, dst, dst_category, p_given_src),
        )

    def commit(self) -> None:
        if self.conn:
            self.conn.commit()

    def top_for(self, src_tag: str, category: Optional[int] = None,
                top_k: int = 5, min_prob: float = 0.3) -> List[dict]:
        """Return top-K most-likely dst tags given src_tag."""
        if not self.conn:
            return []
        sql = (
            "SELECT dst_tag, dst_category, p_given_src FROM cooccur "
            "WHERE src_tag = ? AND p_given_src >= ?"
        )
        params: list = [src_tag, min_prob]
        if category is not None:
            sql += " AND dst_category = ?"
            params.append(category)
        sql += " ORDER BY p_given_src DESC LIMIT ?"
        params.append(top_k)
        cur = self.conn.execute(sql, params)
        return [
            {"tag": row[0], "category": row[1], "p": row[2]}
            for row in cur.fetchall()
        ]
