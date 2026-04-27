"""Tag-format loader + tag database access.

A tag format describes one model's tagging convention (Illustrious /
NoobAI / Pony / Anima). Each lives in `tag-formats/<name>.yaml` with:

  system_prompt           the LLM system prompt for tag generation
  use_underscores         whether the format wants underscored tags
                          (long_hair) vs spaced (long hair)
  tag_db                  filename of the CSV in tags/ for validation
  tag_db_url              download URL for the CSV
  quality_tags            quality tokens specific to this format
                          (merged with UNIVERSAL_QUALITY_TAGS)
  leading_tags            tags that appear at the start of the output
  rating_tags             rating-related tokens (rating:safe, etc.)
  general_tags_allowlist  CAT_GENERAL tokens explicitly allowed past
                          the structural filter
  negative_quality_tags   tokens for the negative prompt

Public API:
  UNIVERSAL_QUALITY_TAGS  set of cross-format quality tokens
  load(tag_formats_dir, universal_quality_tags=UNIVERSAL_QUALITY_TAGS)
       → dict: format_name → format_config
  download_db(fmt_config, tags_dir, logger=None) → bool (db reachable)
  load_db(fmt_config, tags_dir, cache, logger=None)
       → tags_set (also populates cache[filename] and cache[filename + "_aliases"])

Tag DB caching is delegated to the caller via the `cache` dict, so
this module owns no cross-call state.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any, Dict, Optional, Set, Tuple

from ._util import load_yaml_or_json


# Universal Danbooru-adjacent quality tokens every booru format accepts.
# Format-specific quality / leading / rating tokens live in each
# tag-format yaml. Callers pass this in (or use the default) to load().
UNIVERSAL_QUALITY_TAGS: Set[str] = {
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality",
    "absurdres", "highres",
}


def _underscored(items) -> Set[str]:
    return {str(t).replace(" ", "_") for t in (items or [])}


def _general_allow_set(items) -> Set[str]:
    """Normalize tokens for the general-tags allowlist. Lowercase
    underscored form for O(1) DB-style membership checks."""
    return {
        str(t).lower().replace(" ", "_").replace("-", "_")
        for t in (items or [])
    }


def load(
    tag_formats_dir: str,
    universal_quality_tags: Set[str] = UNIVERSAL_QUALITY_TAGS,
) -> Dict[str, Dict[str, Any]]:
    """Load tag format definitions from `tag_formats_dir`. Each `.yaml/
    .yml/.json` file with a `system_prompt` field becomes one format.
    Format name is derived from the filename (kebab/snake → Title Case).

    Returns dict: format_name → format_config.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(tag_formats_dir):
        return out
    for name in sorted(os.listdir(tag_formats_dir)):
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = load_yaml_or_json(os.path.join(tag_formats_dir, name))
        if not data or "system_prompt" not in data:
            continue
        label = (
            os.path.splitext(name)[0]
            .replace("-", " ").replace("_", " ").title()
        )
        quality = _underscored(data.get("quality_tags"))
        leading = _underscored(data.get("leading_tags"))
        rating = _underscored(data.get("rating_tags"))
        general_allow = _general_allow_set(data.get("general_tags_allowlist"))
        quality_merged = universal_quality_tags | quality
        out[label] = {
            "system_prompt": data["system_prompt"].strip(),
            "use_underscores": data.get("use_underscores", False),
            "negative_quality_tags": data.get("negative_quality_tags", []),
            "tag_db": data.get("tag_db", ""),
            "tag_db_url": data.get("tag_db_url", ""),
            "quality_tags": quality_merged,
            "leading_tags": leading,
            "rating_tags": rating,
            "general_tags_allowlist": general_allow,
            "whitelist_set": quality_merged | leading | rating,
        }
    return out


# ── Tag database (CSV) download + load ──────────────────────────────────


def download_db(
    fmt_config: Dict[str, Any],
    tags_dir: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Ensure the tag-database CSV is on disk. Downloads from
    fmt_config['tag_db_url'] to `tags_dir/<filename>` if not present.
    Returns True iff the file is now available locally.

    Network errors are logged and return False rather than raising —
    the caller decides whether to fall back (e.g. to fuzzy validation).
    """
    filename = fmt_config.get("tag_db", "")
    url = fmt_config.get("tag_db_url", "")
    if not filename or not url:
        return False
    os.makedirs(tags_dir, exist_ok=True)
    local_path = os.path.join(tags_dir, filename)
    if os.path.isfile(local_path):
        return True
    try:
        if logger:
            logger.info(f"Downloading tag database: {filename}")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        if logger:
            logger.info(f"Tag database saved: {local_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to download tag database: {e}")
        return False


def load_db(
    fmt_config: Dict[str, Any],
    tags_dir: str,
    cache: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Set[str]:
    """Load tag database (CSV) into memory. Returns the set of valid
    tag strings (also populates cache[filename] and cache[filename +
    "_aliases"] with the parsed data).

    The CSV format follows Danbooru's: tag,category,post_count,aliases.
    Tags + aliases are normalized to underscore form (some CSVs ship
    "long-hair", others "long_hair"; validation looks up by underscore
    form, so we unify at load time).

    Caches by filename so multiple formats sharing the same DB only
    parse it once. Returns empty set on download failure / parse error
    (logs the error if a logger is provided).
    """
    filename = fmt_config.get("tag_db", "")
    if not filename:
        return set()

    if filename in cache:
        return cache[filename]

    if not download_db(fmt_config, tags_dir, logger=logger):
        return set()

    local_path = os.path.join(tags_dir, filename)
    tags: Set[str] = set()
    aliases: Dict[str, str] = {}
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 3)
                if len(parts) >= 1:
                    tag = parts[0].strip().replace("-", "_")
                    if tag:
                        tags.add(tag)
                    if len(parts) >= 4 and parts[3]:
                        alias_str = parts[3].strip().strip('"')
                        for alias in alias_str.split(","):
                            alias = alias.strip().replace("-", "_")
                            if alias:
                                aliases[alias] = tag
    except Exception as e:
        if logger:
            logger.error(f"Failed to load tag database: {e}")
        return set()

    cache[filename] = tags
    cache[f"{filename}_aliases"] = aliases
    if logger:
        logger.info(
            f"Loaded {len(tags)} tags + {len(aliases)} aliases from {filename}"
        )
    return tags
