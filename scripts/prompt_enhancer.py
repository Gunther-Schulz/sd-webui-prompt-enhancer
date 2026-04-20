import json
import logging
import os
import re
import socket
import threading
import time
import urllib.request
import urllib.error

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

logger = logging.getLogger("prompt_enhancer")

# ── Extension root directory ─────────────────────────────────────────────────
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODIFIERS_DIR = os.path.join(_EXT_DIR, "modifiers")

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASES_FILENAME = "_bases"
_TAGS_DIR = os.path.join(_EXT_DIR, "tags")
_TAG_FORMATS_DIR = os.path.join(_EXT_DIR, "tag-formats")

_tag_formats = {}    # format_name -> {system_prompt, use_underscores, tag_db, tag_db_url}
_tag_databases = {}  # db_filename -> set of valid tags

def _load_tag_formats():
    """Load tag format definitions from tag-formats/ directory."""
    global _tag_formats
    _tag_formats = {}
    if not os.path.isdir(_TAG_FORMATS_DIR):
        return
    for name in sorted(os.listdir(_TAG_FORMATS_DIR)):
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(_TAG_FORMATS_DIR, name))
        if data and "system_prompt" in data:
            label = os.path.splitext(name)[0].replace("-", " ").replace("_", " ").title()
            _tag_formats[label] = {
                "system_prompt": data["system_prompt"].strip(),
                "use_underscores": data.get("use_underscores", False),
                "negative_quality_tags": data.get("negative_quality_tags", []),
                "tag_db": data.get("tag_db", ""),
                "tag_db_url": data.get("tag_db_url", ""),
            }



# ── File loading ─────────────────────────────────────────────────────────────

def _load_file(path):
    """Load a JSON or YAML file and return parsed content."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        print(f"[PromptEnhancer] ERROR: {path} must be a YAML/JSON mapping (dict), got {type(data).__name__}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[PromptEnhancer] ERROR: Failed to load {path}: {e}")
    return {}


def _get_local_dirs(ui_path=""):
    """Resolve local overrides directories.

    Supports comma-separated paths in UI field or env var.
    Returns list of valid directory paths.
    """
    raw = (ui_path or "").strip()
    if not raw:
        raw = os.environ.get("PROMPT_ENHANCER_LOCAL", "").strip()
    if not raw:
        return []
    dirs = []
    for p in raw.split(","):
        p = p.strip()
        if p and os.path.isdir(p):
            dirs.append(p)
    return dirs


def _load_local_bases(local_dirs):
    """Load _bases.yaml from all local directories."""
    merged = {}
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, BASES_FILENAME + ext)
            if os.path.isfile(path):
                merged.update({k: v for k, v in _load_file(path).items() if isinstance(v, str)})
    return merged


def _scan_modifier_files(directory):
    """Scan a directory for modifier YAML/JSON files.

    Returns dict: dropdown_name -> {category: {modifier: keywords}}.
    Skips _bases.* files. Uses _label field from YAML if present,
    otherwise derives label from filename.
    """
    result = {}
    if not directory or not os.path.isdir(directory):
        return result
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        stem = os.path.splitext(name)[0]
        if stem == BASES_FILENAME:
            continue
        if not name.endswith((".yaml", ".yml", ".json")):
            continue
        data = _load_file(os.path.join(directory, name))
        if data:
            # Require _label in YAML for dropdown name
            label = data.pop("_label", None)
            if not label:
                print(f"[PromptEnhancer] WARNING: Skipping {os.path.join(directory, name)}: missing '_label' field. Add '_label: Your Label' to the YAML file.")
                continue
            result[label] = data
    return result


def _merge_modifier_dicts(base, override):
    """Merge two dropdown-level dicts. Same dropdown name -> merge categories."""
    merged = {}
    for label, categories in base.items():
        merged[label] = {}
        for cat, items in categories.items():
            if isinstance(items, dict):
                merged[label][cat] = dict(items)
    for label, categories in override.items():
        if not isinstance(categories, dict):
            continue
        if label not in merged:
            merged[label] = {}
        for cat, items in categories.items():
            if not isinstance(items, dict):
                continue
            if cat not in merged[label]:
                merged[label][cat] = {}
            merged[label][cat].update(items)
    return merged


def _build_dropdown_data(categories_dict):
    """Build flat lookup and choice list from a single dropdown's categories."""
    flat = {}
    choices = []
    for cat_name, items in categories_dict.items():
        if not isinstance(items, dict):
            continue
        separator = f"\u2500\u2500\u2500\u2500\u2500 {cat_name.title()} \u2500\u2500\u2500\u2500\u2500"
        choices.append(separator)
        for name, keywords in items.items():
            if isinstance(keywords, str):
                flat[name] = keywords
                choices.append(name)
    return flat, choices


# ── Tag database ─────────────────────────────────────────────────────────────

def _download_tag_db(fmt_config):
    """Download tag database if not cached. Returns True if available."""
    filename = fmt_config.get("tag_db", "")
    url = fmt_config.get("tag_db_url", "")
    if not filename or not url:
        return False
    os.makedirs(_TAGS_DIR, exist_ok=True)
    local_path = os.path.join(_TAGS_DIR, filename)
    if os.path.isfile(local_path):
        return True
    try:
        logger.info(f"Downloading tag database: {filename}")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        logger.info(f"Tag database saved: {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download tag database: {e}")
        return False


def _load_tag_db(tag_format):
    """Load tag database into memory. Returns set of valid tag strings."""
    fmt_config = _tag_formats.get(tag_format, {})
    filename = fmt_config.get("tag_db", "")
    if not filename:
        return set()

    # Cache by filename (multiple formats can share a DB)
    if filename in _tag_databases:
        return _tag_databases[filename]

    if not _download_tag_db(fmt_config):
        return set()

    local_path = os.path.join(_TAGS_DIR, filename)
    tags = set()
    aliases = {}
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 3)
                if len(parts) >= 1:
                    tag = parts[0].strip()
                    if tag:
                        tags.add(tag)
                    # Parse aliases (field 4, quoted comma-separated)
                    if len(parts) >= 4 and parts[3]:
                        alias_str = parts[3].strip().strip('"')
                        for alias in alias_str.split(","):
                            alias = alias.strip()
                            if alias:
                                aliases[alias] = tag
    except Exception as e:
        logger.error(f"Failed to load tag database: {e}")
        return set()

    _tag_databases[filename] = tags
    _tag_databases[f"{filename}_aliases"] = aliases
    logger.info(f"Loaded {len(tags)} tags + {len(aliases)} aliases from {filename}")
    return tags


def _find_closest_tag(tag, valid_tags, aliases, max_distance=3):
    """Find the closest valid tag for an invalid one.

    Priority: exact alias match > substring match > Levenshtein distance.
    Returns (corrected_tag, None) or (None, original_tag) if no match.
    """
    # Check aliases first
    if tag in aliases:
        return aliases[tag], None

    # Prefix substring: match if our tag is a prefix of a valid tag
    # (e.g., "highres" starts with "high"). Only for tags 5+ chars to
    # avoid short-word false positives. Length ratio prevents wild mismatches.
    tag_len = len(tag)
    if tag_len >= 5:
        best_prefix = None
        best_prefix_len = 999
        for valid in valid_tags:
            valid_len = len(valid)
            len_ratio = tag_len / valid_len if valid_len > 0 else 0
            if len_ratio >= 0.5 and valid.startswith(tag):
                if valid_len < best_prefix_len:
                    best_prefix = valid
                    best_prefix_len = valid_len
        if best_prefix:
            return best_prefix, None

    # Levenshtein distance — skip for short tags (< 5 chars) where
    # edit distance 1-3 produces nonsensical matches (low→cow, red→rend)
    if tag_len < 5:
        return None, tag
    best_match = None
    best_dist = max_distance + 1
    for valid in valid_tags:
        # Quick length filter
        if abs(len(valid) - len(tag)) > max_distance:
            continue
        # Calculate distance
        dist = _levenshtein(tag, valid)
        if dist < best_dist:
            best_dist = dist
            best_match = valid
    if best_match and best_dist <= max_distance:
        return best_match, None

    return None, tag  # unmatched


def _levenshtein(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# Model-specific tags that aren't in the danbooru database but are valid triggers
_WHITELISTED_TAGS = {
    # Quality tags
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality", "absurdres", "highres",
    "very_awa",
    # Score tags (Pony)
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up",
    # Source tags (Pony)
    "source_anime", "source_furry", "source_pony", "source_cartoon",
}

# Common LLM mistakes -> correct danbooru tag
_TAG_CORRECTIONS = {
    "1man": "1boy",
    "1woman": "1girl",
    "1female": "1girl",
    "1male": "1boy",
    "man": "male_focus",
    "woman": "1girl",
    "female": "1girl",
    "male": "1boy",
    "girl": "1girl",
    "a_girl": "1girl",
    "boy": "1boy",
    "a_boy": "1boy",
    "2girl": "2girls",
    "2boy": "2boys",
    "2men": "2boys",
    "2women": "2girls",
    "3girl": "3girls",
    "3boy": "3boys",
}

# Valid rating tags (explicit list, not prefix matching)
_VALID_RATINGS = {
    # Danbooru style (Illustrious, NoobAI)
    "rating:general", "rating:sensitive", "rating:questionable", "rating:explicit",
    # Pony style
    "rating_safe", "rating_questionable", "rating_explicit",
}


def _validate_tags(tags_str, tag_format, mode="Check"):
    """Validate and correct tags against the database.

    Modes:
      Check  — exact + alias match only, keep unrecognized
      Fuzzy  — exact + alias + fuzzy correction, keep unrecognized
      Strict — exact + alias only, drop unrecognized

    Returns (corrected_tags_str, stats_dict).
    """
    valid_tags = _load_tag_db(tag_format)
    if not valid_tags:
        return tags_str, {"error": "No tag database available"}

    fmt_config = _tag_formats.get(tag_format, {})
    db_filename = fmt_config.get("tag_db", "")
    aliases = _tag_databases.get(f"{db_filename}_aliases", {})
    use_underscores = fmt_config.get("use_underscores", False)
    use_fuzzy = mode in ("Fuzzy", "Fuzzy Strict")
    drop_invalid = mode in ("Strict", "Fuzzy Strict")

    raw_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    result_tags = []
    corrected = 0
    dropped = 0
    kept = 0

    for tag in raw_tags:
        # Strip LLM meta-annotations: [style: X], (artist: X), etc.
        tag = _clean_tag(tag)
        if not tag:
            continue

        # Always use underscores for lookup (DB and corrections use underscores).
        # use_underscores only controls output format, not lookup.
        lookup = tag.replace(" ", "_")

        # Common LLM mistakes — correct before any other check
        if lookup in _TAG_CORRECTIONS:
            corrected_tag = _TAG_CORRECTIONS[lookup]
            result_tags.append(corrected_tag if use_underscores else corrected_tag.replace("_", " "))
            corrected += 1
            continue

        # Whitelisted tags always pass
        if lookup in _WHITELISTED_TAGS or lookup in _VALID_RATINGS:
            result_tags.append(tag)
            continue

        # Exact match
        if lookup in valid_tags:
            result_tags.append(tag)
            continue

        # Alias match (always applied in all modes)
        if lookup in aliases:
            match = aliases[lookup]
            result_tags.append(match if use_underscores else match.replace("_", " "))
            corrected += 1
            continue

        # Prefix match: "artist_name" -> "artist_name_(style)" etc.
        # Catches LLM omitting danbooru disambiguation suffixes.
        # Only for multi-word tags (contain underscore) to avoid matching
        # common words like "high" -> "high_(hgih)" or "dusty" -> "dusty_(gravity_daze)".
        if "_" in lookup:
            prefix = lookup + "_("
            prefix_matches = [v for v in valid_tags if v.startswith(prefix)]
            if len(prefix_matches) == 1:
                match = prefix_matches[0]
                result_tags.append(match if use_underscores else match.replace("_", " "))
                corrected += 1
                continue

        # Fuzzy match (only in Fuzzy mode)
        if use_fuzzy:
            match, _ = _find_closest_tag(lookup, valid_tags, {})  # skip aliases, already checked
            if match:
                result_tags.append(match if use_underscores else match.replace("_", " "))
                corrected += 1
                continue

        # Unrecognized tag
        if drop_invalid:
            dropped += 1
        else:
            result_tags.append(tag)
            kept += 1

    # Reorder tags into standard danbooru order
    result_tags = _reorder_tags(result_tags)

    # Escape parentheses for SD — danbooru tags like artist_(style) need
    # \( \) so SD doesn't interpret them as emphasis/weight syntax.
    def _escape_parens(tag):
        if "(" in tag and "_(" in tag:
            return tag.replace("(", r"\(").replace(")", r"\)")
        return tag
    result_tags = [_escape_parens(t) for t in result_tags]

    stats = {"corrected": corrected, "dropped": dropped, "kept_invalid": kept, "total": len(result_tags)}
    return ", ".join(result_tags), stats


# Known tags for ordering buckets
_QUALITY_TAGS = {
    "masterpiece", "best_quality", "amazing_quality", "good_quality",
    "normal_quality", "low_quality", "worst_quality", "absurdres", "highres",
    "very_awa",
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up",
}
_SOURCE_TAGS = {"source_anime", "source_furry", "source_pony", "source_cartoon"}
_SUBJECT_TAGS = {
    "1girl", "1girls", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "1boy", "1man", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "1other", "solo", "no_humans", "male_focus", "female_focus",
    "1woman", "man", "woman", "girl", "boy",
}
_RATING_TAGS = {
    "rating:general", "rating:sensitive", "rating:questionable", "rating:explicit",
    "rating_safe", "rating_questionable", "rating_explicit",
}


def _reorder_tags(tags):
    """Reorder tags into standard danbooru convention.

    Order: quality -> source -> subject count -> everything else -> rating
    Also deduplicates and enforces single rating tag (keeps last).
    """
    seen = set()
    quality = []
    source_tags = []
    subjects = []
    rating = []
    rest = []

    for tag in tags:
        # Deduplicate
        lookup = tag.replace(" ", "_")
        if lookup in seen:
            continue
        seen.add(lookup)

        if lookup in _QUALITY_TAGS:
            quality.append(tag)
        elif lookup in _SOURCE_TAGS:
            source_tags.append(tag)
        elif lookup in _SUBJECT_TAGS:
            subjects.append(tag)
        elif lookup in _RATING_TAGS:
            rating.append(tag)
        else:
            rest.append(tag)

    # Enforce single rating (keep last if multiple)
    if len(rating) > 1:
        rating = [rating[-1]]

    # Enforce no_humans: remove contradicting character count tags
    subject_lookups = {t.replace(" ", "_") for t in subjects}
    if "no_humans" in subject_lookups:
        subjects = [t for t in subjects if t.replace(" ", "_") == "no_humans"]

    return quality + source_tags + subjects + rest + rating


# ── Config state ─────────────────────────────────────────────────────────────

_bases = {}
_all_modifiers = {}          # flat: name -> keywords (for lookup across all dropdowns)
_dropdown_order = []         # list of dropdown labels in display order
_dropdown_choices = {}       # label -> [choice_list with separators]
_wildcards = {}
_wildcard_choices = []
_prompts = {}                # operational prompts loaded from prompts.yaml


def _load_wildcards(wc_data):
    """Parse wildcard data dict into flat lookup and UI choice list."""
    wildcards = {}
    choices = []
    for cat, items in wc_data.items():
        if not isinstance(items, dict):
            continue
        choices.append(f"\u2500\u2500\u2500\u2500\u2500 {cat.title()} \u2500\u2500\u2500\u2500\u2500")
        for name, prompt in items.items():
            if isinstance(prompt, str):
                wildcards[name] = prompt
                choices.append(name)
    return wildcards, choices


def _reload_all(local_dir_path=""):
    """Reload all config files from disk."""
    global _bases, _all_modifiers, _dropdown_order, _dropdown_choices, _wildcards, _wildcard_choices, _prompts

    local_dirs = _get_local_dirs(local_dir_path)

    # Bases (YAML, with local overrides)
    _bases = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(_EXT_DIR, "bases" + ext)
        if os.path.isfile(path):
            _bases = {k: v for k, v in _load_file(path).items() if isinstance(v, str)}
            break
    _bases.update(_load_local_bases(local_dirs))

    # Modifiers: scan extension modifiers/ dir + all local dirs, merge
    all_mods = _scan_modifier_files(_MODIFIERS_DIR)
    for local_dir in local_dirs:
        local_mods = _scan_modifier_files(local_dir)
        all_mods = _merge_modifier_dicts(all_mods, local_mods)

    _all_modifiers = {}
    _dropdown_order = []
    _dropdown_choices = {}
    for label in sorted(all_mods.keys()):
        flat, choices = _build_dropdown_data(all_mods[label])
        if choices:
            _dropdown_order.append(label)
            _dropdown_choices[label] = choices
            _all_modifiers.update(flat)

    # Wildcards (YAML, with local overrides)
    wc_data = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(_EXT_DIR, "wildcards" + ext)
        if os.path.isfile(path):
            wc_data = _load_file(path) or {}
            break
    # Merge local wildcards
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, "_wildcards" + ext)
            if os.path.isfile(path):
                local_wc = _load_file(path) or {}
                for cat, items in local_wc.items():
                    if isinstance(items, dict):
                        wc_data.setdefault(cat, {}).update(items)
    _wildcards, _wildcard_choices = _load_wildcards(wc_data)

    # Prompts (YAML, with local overrides)
    _prompts = {}
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(_EXT_DIR, "prompts" + ext)
        if os.path.isfile(path):
            data = _load_file(path) or {}
            _prompts = {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}
            break
    # Merge local prompt overrides
    for local_dir in local_dirs:
        for ext in (".yaml", ".yml", ".json"):
            path = os.path.join(local_dir, "_prompts" + ext)
            if os.path.isfile(path):
                local_p = _load_file(path) or {}
                for k, v in local_p.items():
                    if isinstance(v, str):
                        _prompts[k] = v.strip()


_reload_all()


_load_tag_formats()

# Max tokens per forge preset (from backend/diffusion_engine/*.py)
_PRESET_MAX_TOKENS = {
    "sd": 75, "xl": 75, "flux": 255, "klein": 255,
    "qwen": 512, "lumina": 512, "zit": 999, "wan": 512,
    "anima": 512, "ernie": 512,
}

# Tag counts per detail level (model-independent)
_TAG_COUNTS = {0: None, 1: 8, 2: 12, 3: 18, 4: 25, 5: 35, 6: 45, 7: 55, 8: 65, 9: 75, 10: 90}

# Detail level descriptions
_DETAIL_LABELS = {
    0: None,
    1: "very short, minimal",
    2: "short, concise",
    3: "brief but complete",
    4: "moderate",
    5: "moderately detailed",
    6: "detailed",
    7: "detailed, vivid",
    8: "highly detailed",
    9: "very detailed, comprehensive",
    10: "extensive, exhaustive",
}


def _get_word_target(detail, preset="sd"):
    """Calculate word target based on detail level and forge preset."""
    if detail == 0:
        return None
    max_tokens = _PRESET_MAX_TOKENS.get(preset, 75)
    # Rough: 1 token ≈ 0.75 words, scale detail 1-10 across 20%-100% of max
    max_words = int(max_tokens * 0.75)
    fraction = 0.1 + (detail / 10) * 0.9  # detail 1 = 20%, detail 10 = 100%
    return max(20, int(max_words * fraction))


def _build_detail_instruction(detail, mode="enhance", preset="sd"):
    """Build detail instruction for enhance or tags."""
    if detail == 0:
        if mode == "tags":
            return "Generate a good set of tags with reasonable coverage."
        return "Write a moderately detailed description."
    label = _DETAIL_LABELS.get(detail, "moderate")
    if mode == "tags":
        count = _TAG_COUNTS.get(detail, 20)
        return f"Generate a {label} set of tags. Aim for around {count} tags."
    else:
        words = _get_word_target(detail, preset)
        return f"Write a {label} description. Aim for around {words} words."



# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_names():
    names = list(_bases.keys())
    names.append("Custom")
    return names


def _collect_modifiers(dropdown_selections):
    """Collect all selected modifiers into a list of (name, keywords) tuples."""
    result = []
    for selections in dropdown_selections:
        for name in (selections or []):
            keywords = _all_modifiers.get(name, "")
            if keywords:
                result.append((name, keywords))
    return result


def _build_style_string(mod_list):
    """Build the 'Apply these styles:' string."""
    parts = []
    for name, keywords in mod_list:
        if name.lower() not in keywords.lower():
            keywords = f"{name.lower()}, {keywords}"
        parts.append(keywords)
    return f"Apply these styles: {', '.join(parts)}." if parts else ""


# ── Ollama ───────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:11434"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _to_ollama_base(api_url):
    base = api_url
    for suffix in ("/v1/chat/completions", "/v1", "/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or DEFAULT_OLLAMA_BASE


def _fetch_ollama_models(api_url):
    try:
        base = _to_ollama_base(api_url)
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        models.sort()
        return models
    except Exception:
        return []


def _refresh_models(api_url, current_model):
    models = _fetch_ollama_models(api_url)
    if not models:
        return gr.update()
    value = current_model if current_model in models else models[0]
    return gr.update(choices=models, value=value)


def _get_ollama_status(api_url):
    """Get Ollama status info: running, model loaded, GPU/CPU."""
    try:
        base = _to_ollama_base(api_url)
        req = urllib.request.Request(f"{base}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            version_data = json.loads(resp.read().decode("utf-8"))
        version = version_data.get("version", "?")

        # Check loaded models
        req = urllib.request.Request(f"{base}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ps_data = json.loads(resp.read().decode("utf-8"))

        models = ps_data.get("models", [])
        if models:
            m = models[0]
            name = m.get("name", "?")
            vram = m.get("size_vram", 0)
            total = m.get("size", 0)
            if vram > 0 and total > 0:
                gpu_pct = int(vram / total * 100)
                mode = f"GPU ({gpu_pct}%)" if gpu_pct > 50 else f"CPU ({100-gpu_pct}% offloaded)"
            elif vram > 0:
                mode = "GPU"
            else:
                mode = "CPU"
            return f"<span style='color:#6c6'>Ollama v{version} \u2022 {name} \u2022 {mode}</span>"
        else:
            return f"<span style='color:#6c6'>Ollama v{version} \u2022 connected</span>"
    except Exception:
        return "<span style='color:#c66'>Ollama not running</span>"


# ── Core logic ───────────────────────────────────────────────────────────────

def _strip_think_blocks(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _has_inline_wildcards(text):
    return bool(re.search(r"\{[^}]+\?\}", text))


def _clean_output(text, strip_underscores=True):
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    if strip_underscores:
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    return text.strip()


def _clean_tag(tag):
    """Strip LLM meta-annotations from a single tag.

    Handles patterns like [Illustrator: X], [style: X], (artist: X),
    [artist:X] style, "setting: garden", etc. Preserves danbooru
    disambiguation suffixes like _(style) and weight syntax like (tag:1.2).
    """
    tag = tag.strip()
    # Strip all square brackets (never valid in danbooru tags)
    tag = tag.replace("[", "").replace("]", "")
    # Strip paren-wrapped meta: (illustrator: X) -> X, but keep _(suffix) intact
    # Only strip if tag starts with ( (not a suffix like artist_(style))
    if tag.startswith("(") and not tag.startswith("_("):
        tag = tag.lstrip("(").rstrip(")")
    # Strip "key: value" meta-annotation prefixes generically.
    # Matches "word: content" or "word word: content" where the key part
    # is 2+ letters and doesn't look like a danbooru tag (rating:, score_).
    meta_match = re.match(
        r"^(?!rating|score)([a-zA-Z][a-zA-Z_ ]{1,30})\s*:\s*(.+)$", tag
    )
    if meta_match:
        key = meta_match.group(1).strip().lower()
        # Only strip if the key looks like a meta-annotation, not a valid tag prefix
        if key not in ("1", "2", "3"):  # don't strip numeric prefixes
            tag = meta_match.group(2)
    # Strip trailing " style" when it follows a name: "Kazuhiro Fujita style" -> "Kazuhiro Fujita"
    tag = re.sub(r"\s+style$", "", tag, flags=re.IGNORECASE)
    # Convert hyphens to underscores (hyphens are never valid in danbooru tags)
    if "-" in tag and "_" not in tag and " " not in tag:
        tag = tag.replace("-", "_")
    return tag.strip()


def _split_concatenated_tag(tag):
    """Split concatenated words like 'moonlitcatacombs' into 'moonlit_catacombs'.

    Only applies to tags longer than 15 chars with no existing separators.
    Uses camelCase boundaries and common word boundary heuristics.
    """
    # Skip if already has separators or is short enough to be a real tag
    if "_" in tag or " " in tag or len(tag) <= 15:
        return tag
    # Skip known valid patterns (rating:, score_, source_)
    if ":" in tag:
        return tag
    # Try to insert underscores at likely word boundaries
    # Look for lowercase->uppercase transitions (camelCase)
    result = re.sub(r"([a-z])([A-Z])", r"\1_\2", tag).lower()
    if result != tag.lower():
        return result
    # All lowercase concatenated — can't reliably split without a dictionary
    # Return as-is, validation will handle it
    return tag


def _split_positive_negative(text):
    """Split LLM output at POSITIVE:/NEGATIVE: markers.

    Returns (positive, negative).  If no markers found, returns (text, "").
    """
    # Case-insensitive search for markers
    pos_match = re.search(r"(?i)^POSITIVE:\s*\n?", text, re.MULTILINE)
    neg_match = re.search(r"(?i)^NEGATIVE:\s*\n?", text, re.MULTILINE)
    if not neg_match:
        # No NEGATIVE marker — treat entire text as positive
        return text.strip(), ""
    if pos_match:
        positive = text[pos_match.end():neg_match.start()].strip()
    else:
        # NEGATIVE marker but no POSITIVE marker — everything before is positive
        positive = text[:neg_match.start()].strip()
    negative = text[neg_match.end():].strip()
    return positive, negative


def _postprocess_tags(tag_str, tag_fmt, validation_mode):
    """Apply full tag post-processing pipeline.

    Handles: concatenation split, underscore formatting, validation,
    reordering, paren escaping. Returns (processed_tags, stats_or_None).
    """
    fmt_config = _tag_formats.get(tag_fmt, {})
    use_underscores = fmt_config.get("use_underscores", False)
    tag_str = ", ".join(
        _split_concatenated_tag(t.strip()).replace(" ", "_") if use_underscores
        else _split_concatenated_tag(t.strip())
        for t in tag_str.split(",") if t.strip()
    )
    if validation_mode != "Off":
        tag_str, stats = _validate_tags(tag_str, tag_fmt, mode=validation_mode)
        return tag_str, stats
    return tag_str, None


_STALL_TIMEOUT = int(os.environ.get("PROMPT_ENHANCER_STALL_TIMEOUT", "10"))
_MAX_TOKENS = int(os.environ.get("PROMPT_ENHANCER_MAX_TOKENS", "1000"))
_MAX_TIME = int(os.environ.get("PROMPT_ENHANCER_MAX_TIME", "60"))


def _detect_repetition(text):
    """Detect repetitive output by tracking unique content ratio.

    Splits text into comma/period-separated segments. If the last 20 segments
    contain fewer than 30% unique values (compared to what appeared earlier),
    the output is looping. Returns trimmed text if detected, None otherwise.
    """
    # Split on commas and periods to get segments
    segments = [s.strip().lower() for s in re.split(r"[,.\n]+", text) if s.strip()]
    if len(segments) < 30:
        return None
    # Check: of the last 20 segments, how many are duplicates of earlier segments?
    early = set(segments[:-20])
    recent = segments[-20:]
    if not early:
        return None
    repeated = sum(1 for s in recent if s in early)
    ratio = repeated / len(recent)
    if ratio >= 0.7:
        # 70%+ of recent segments already appeared — it's looping
        # Trim to the point where content was still fresh
        # Walk backwards to find where repetition started
        seen = set()
        trim_idx = len(segments)
        dup_streak = 0
        for i in range(len(segments) - 1, -1, -1):
            if segments[i] in seen:
                dup_streak += 1
            else:
                if dup_streak > 5:
                    trim_idx = i + 1
                    break
                dup_streak = 0
            seen.add(segments[i])
        # Rebuild text from non-repeated segments
        # Find character position of the trim point
        parts = re.split(r"([,.\n]+)", text)
        seg_count = 0
        char_pos = 0
        for part in parts:
            if part.strip() and not re.match(r"^[,.\n]+$", part):
                seg_count += 1
                if seg_count > trim_idx:
                    break
            char_pos += len(part)
        trimmed = text[:char_pos].rstrip(" ,.\n")
        print(f"[PromptEnhancer] Aborting: repetition detected ({ratio:.0%} of last 20 segments are repeats)")
        return trimmed
    return None
_cancel_flag = threading.Event()
_last_seed = -1


class _TruncatedError(Exception):
    """LLM output was truncated (stall, max tokens, or max time)."""
    pass


def _call_llm(prompt, api_url, model, system_prompt, temperature, think=False, timeout=None, seed=-1, _progress=None):
    global _last_seed
    import random as _random
    if seed == -1:
        seed = _random.randint(0, 2**31 - 1)
    _last_seed = seed

    base = _to_ollama_base(api_url)
    # Prepend /no_think to user message for Qwen3 models that ignore think:false
    user_content = prompt if think else f"/no_think\n{prompt}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": float(temperature),
            "seed": int(seed),
            "top_k": 20,
            "top_p": 0.95 if think else 0.8,
            "repeat_penalty": 1.5,
            "presence_penalty": 1.5,
        },
        "think": bool(think),
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base}/api/chat"
    stall_timeout = timeout or _STALL_TIMEOUT
    print(f"[PromptEnhancer] LLM call: model={model}, think={think}, temp={temperature}, stall={stall_timeout}s, prompt_len={len(prompt)}, system_len={len(system_prompt)}")

    last_err = None
    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"[PromptEnhancer] Ollama retry attempt {attempt + 1}")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Initial connection timeout
            resp = urllib.request.urlopen(req, timeout=30)
            # Set socket timeout so reads unblock periodically for cancel checks.
            # Uses CPython internals — wrapped for safety.
            try:
                resp.fp.raw._sock.settimeout(2.0)
            except (AttributeError, OSError):
                pass
            content_parts = []
            completed = False
            start_time = time.monotonic()
            last_token_time = start_time
            thinking_detected = False

            try:
                while True:
                    try:
                        line = resp.readline()
                    except (socket.timeout, TimeoutError):
                        # Socket timeout — no data yet, check cancel and continue
                        if _cancel_flag.is_set():
                            print(f"[PromptEnhancer] Cancelled by user")
                            break
                        continue
                    if not line:
                        break  # EOF
                    # Check cancel flag
                    if _cancel_flag.is_set():
                        print(f"[PromptEnhancer] Cancelled by user")
                        break
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check for thinking tokens (Qwen3 ignoring think:false)
                    thinking_text = chunk.get("message", {}).get("thinking", "")
                    if thinking_text and not think:
                        if not thinking_detected:
                            thinking_detected = True
                            print(f"[PromptEnhancer] WARNING: Model is thinking despite think=false")
                        # Don't reset timer for thinking tokens — let it stall out
                        if time.monotonic() - last_token_time > stall_timeout:
                            print(f"[PromptEnhancer] Aborting: thinking exceeded {stall_timeout}s")
                            break
                        continue

                    # Content tokens
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        content_parts.append(token)
                        last_token_time = time.monotonic()
                        # Repetition detection: check every 20 tokens
                        if len(content_parts) % 20 == 0 and len(content_parts) >= 40:
                            text = "".join(content_parts)
                            repetition = _detect_repetition(text)
                            if repetition:
                                content_parts = [repetition]
                                break
                        # Update shared progress for live status
                        if _progress is not None:
                            elapsed = last_token_time - start_time
                            _progress["words"] = len("".join(content_parts).split())
                            _progress["tokens"] = len(content_parts)
                            _progress["elapsed"] = elapsed
                            _progress["tps"] = len(content_parts) / elapsed if elapsed > 0 else 0
                        # Hard cap on total tokens
                        if len(content_parts) > _MAX_TOKENS:
                            print(f"[PromptEnhancer] Aborting: exceeded {_MAX_TOKENS} tokens")
                            break

                    # Check for completion
                    if chunk.get("done", False):
                        completed = True
                        break

                    # Check stall (no content token for too long)
                    if time.monotonic() - last_token_time > stall_timeout:
                        print(f"[PromptEnhancer] Aborting: no tokens for {stall_timeout}s")
                        break

                    # Total time cap (catches thinking disguised as content)
                    if time.monotonic() - start_time > _MAX_TIME:
                        print(f"[PromptEnhancer] Aborting: exceeded {_MAX_TIME}s total time")
                        break
            finally:
                try:
                    resp.close()
                except OSError:
                    pass

            content = "".join(content_parts)
            was_cancelled = _cancel_flag.is_set()
            print(f"[PromptEnhancer] Done: {len(content.split())} words, thinking={'yes' if thinking_detected else 'no'}, cancelled={'yes' if was_cancelled else 'no'}")
            result = _strip_think_blocks(content)
            if was_cancelled:
                raise InterruptedError(result)
            if not completed and result:
                raise _TruncatedError(result)
            return result

        except urllib.error.URLError as e:
            last_err = e
            print(f"[PromptEnhancer] Ollama connection failed (attempt {attempt + 1}): {e.reason}")
            if attempt == 0:
                time.sleep(2)
        except TimeoutError as e:
            last_err = urllib.error.URLError(str(e))
            print(f"[PromptEnhancer] Timeout (attempt {attempt + 1}): {e}")
            if attempt == 0:
                time.sleep(2)

    raise last_err


def _build_wildcard_text(wildcards_list, source=""):
    """Build wildcard instructions for the user message."""
    parts = []
    has_inline = source and _has_inline_wildcards(source)
    has_wildcards = any(_wildcards.get(wc, "") for wc in (wildcards_list or []))
    if has_wildcards or has_inline:
        parts.append(_prompts.get("wildcard_preamble", ""))
    for wc_name in (wildcards_list or []):
        wc_prompt = _wildcards.get(wc_name, "")
        if wc_prompt:
            parts.append(wc_prompt)
    if has_inline:
        parts.append(_prompts.get("inline_wildcard", ""))
    return "\n\n".join(parts) if parts else ""


def _assemble_system_prompt(base_name, custom_system_prompt, detail=3):
    """Assemble the system prompt (base + detail, no modifiers/wildcards)."""
    if base_name == "Custom":
        system_prompt = (custom_system_prompt or "").strip()
    else:
        system_prompt = _bases.get(base_name, "")
    if not system_prompt:
        return None

    detail = int(detail) if detail else 0
    try:
        from modules import shared
        preset = getattr(shared.opts, "forge_preset", "sd")
    except Exception:
        preset = "sd"
    instruction = _build_detail_instruction(detail, "enhance", preset)
    if instruction:
        system_prompt = f"{system_prompt}\n\n{instruction}"

    return system_prompt



# ── Streaming progress ──────────────────────────────────────────────────────

def _call_llm_progress(prompt, api_url, model, system_prompt, temperature,
                       think=False, timeout=None, seed=-1):
    """Run _call_llm in a thread, yielding progress dicts every ~1s.

    Final yield is the result string. Exceptions propagate normally.
    Must be iterated from a generator function wired to a Gradio .click() handler.
    """
    progress = {"words": 0, "tokens": 0, "elapsed": 0.0, "tps": 0.0}
    result_box = [None]
    error_box = [None]

    def _worker():
        try:
            result_box[0] = _call_llm(prompt, api_url, model, system_prompt,
                                      temperature, think=think, timeout=timeout,
                                      seed=seed, _progress=progress)
        except Exception as e:
            error_box[0] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=1.0)
        if thread.is_alive():
            yield dict(progress)

    if error_box[0] is not None:
        raise error_box[0]
    yield result_box[0]


# ── UI ───────────────────────────────────────────────────────────────────────

class PromptEnhancer(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        initial_models = _fetch_ollama_models(DEFAULT_API_URL)
        if not initial_models:
            initial_models = [DEFAULT_MODEL]

        with gr.Accordion(open=False, label="Prompt Enhancer"):

            # ── Source prompt ──
            source_prompt = gr.Textbox(
                label="Source Prompt", lines=3,
                placeholder="Type your prompt here, then click Prose or Tags... Use {name?} for inline wildcards.",
                elem_id=f"{tab}_pe_source",
            )
            with gr.Row():
                enhance_btn = gr.Button(value="\u270d Prose", variant="primary", scale=0, min_width=120, elem_id=f"{tab}_pe_enhance_btn")
                hybrid_btn = gr.Button(value="\u2728 Hybrid", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_hybrid_btn")
                tags_btn = gr.Button(value="\U0001f3f7 Tags", variant="primary", scale=0, min_width=100, elem_id=f"{tab}_pe_tags_btn")
                refine_btn = gr.Button(value="\U0001f500 Remix", scale=0, min_width=120, elem_id=f"{tab}_pe_refine_btn")
                cancel_btn = gr.Button(value="\u274c Cancel", scale=0, min_width=80, elem_id=f"{tab}_pe_cancel_btn")
                prepend_source = gr.Checkbox(label="Prepend", value=False, scale=0, min_width=60)
                prepend_source.do_not_save_to_config = True
                negative_prompt_cb = gr.Checkbox(label="+ Negative", value=False, scale=0, min_width=60)
                negative_prompt_cb.do_not_save_to_config = True
                status = gr.HTML(value="", elem_id=f"{tab}_pe_status")

            # ── Base + Tag Format + Validation ──
            with gr.Row():
                base = gr.Dropdown(label="Base", choices=_base_names(), value="Default", scale=1, info="System prompt for Prose/Hybrid")
                _tf_names = list(_tag_formats.keys())
                tag_format = gr.Dropdown(label="Tag Format", choices=_tf_names, value=_tf_names[0] if _tf_names else "", scale=1, info="For Hybrid and Tags modes")
                tag_validation = gr.Radio(
                    label="Tag Validation",
                    choices=["Off", "Check", "Fuzzy", "Strict", "Fuzzy Strict"],
                    value="Check", scale=2,
                    info="Off=raw | Check=alias | Fuzzy=alias+guess | Strict=alias+drop | Fuzzy Strict=guess+drop",
                )
                tag_validation.do_not_save_to_config = True

            # ── Auto-generated modifier dropdowns (one per file) ──
            dd_components = []
            dd_labels = list(_dropdown_order)

            # Layout: 3 dropdowns per row, pad incomplete rows
            for i in range(0, len(dd_labels), 3):
                row_labels = dd_labels[i:i+3]
                with gr.Row():
                    for label in row_labels:
                        d = gr.Dropdown(
                            label=label,
                            choices=_dropdown_choices.get(label, []),
                            value=[], multiselect=True, scale=1,
                        )
                        d.do_not_save_to_config = True
                        dd_components.append(d)
                    # Pad incomplete rows so dropdowns don't stretch
                    for _ in range(3 - len(row_labels)):
                        gr.HTML(value="", visible=True, scale=1)

            # ── Wildcards ──
            with gr.Row():
                wildcards = gr.Dropdown(label="Wildcards", choices=list(_wildcard_choices), value=[], multiselect=True, scale=1)
                wildcards.do_not_save_to_config = True

            # ── Detail Level + Temperature + Think + Seed ──
            with gr.Row():
                detail_level = gr.Slider(label="Detail", minimum=0, maximum=10, value=0, step=1, scale=2, info="0=auto, 1=minimal ... 10=extensive, scales to model")
                detail_level.do_not_save_to_config = True
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.8, step=0.05, scale=2, info="0 = deterministic, 2 = creative")
                seed = gr.Number(label="Seed", value=-1, minimum=-1, step=1, scale=1, info="-1 = random", precision=0, elem_id=f"{tab}_pe_seed")
                seed.do_not_save_to_config = True
                seed_random_btn = ToolButton(value="\U0001f3b2", elem_id=f"{tab}_pe_seed_random")
                seed_reuse_btn = ToolButton(value="\u267b", elem_id=f"{tab}_pe_seed_reuse")
                think = gr.Checkbox(label="Think", value=False, scale=0, min_width=80)
                think.do_not_save_to_config = True
                seed_random_btn.click(fn=lambda: -1, inputs=[], outputs=[seed], show_progress=False)
                seed_reuse_btn.click(fn=lambda: _last_seed, inputs=[], outputs=[seed], show_progress=False)

            # ── Custom system prompt ──
            custom_system_prompt = gr.Textbox(label="Custom System Prompt", lines=4, visible=False, placeholder="Enter your custom system prompt...")
            base.change(fn=lambda b: gr.update(visible=(b == "Custom")), inputs=[base], outputs=[custom_system_prompt], show_progress=False)

            # ── API + reload ──
            with gr.Row():
                api_url = gr.Textbox(label="API URL", value=DEFAULT_API_URL, scale=3)
                model = gr.Dropdown(label="Model", choices=initial_models, value=DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0], allow_custom_value=True, scale=2)
            with gr.Row():
                _env_local = os.environ.get("PROMPT_ENHANCER_LOCAL", "")
                local_dir_path = gr.Textbox(
                    label="Local Overrides",
                    placeholder=f"Using: {_env_local}" if _env_local else "Comma-separated dirs (refreshes content only, restart for new dropdowns)",
                    scale=3,
                )
                local_dir_path.do_not_save_to_config = True
                reload_btn = gr.Button(value="\U0001f504 Reload", scale=0, min_width=100)
                refresh_models_btn = gr.Button(value="\U0001f504 Models", scale=0, min_width=100)
            ollama_status = gr.HTML(value=_get_ollama_status(DEFAULT_API_URL))

            # ── Reload wiring ──
            # Note: reload rebuilds dropdowns but can't add/remove them dynamically.
            # New files require a Forge restart. Existing dropdown contents are refreshed.
            def _do_refresh(current_base, *args):
                # Last arg is local_dir_path, second-to-last is wildcards
                local_path = args[-1]
                wc_val = args[-2]
                dd_vals = args[:-2]

                _reload_all(local_path)
                results = [gr.update(choices=_base_names(), value=current_base if current_base in _bases else "Default")]
                for i, label in enumerate(dd_labels):
                    choices = _dropdown_choices.get(label, [])
                    old_val = dd_vals[i] if i < len(dd_vals) else []
                    results.append(gr.update(choices=choices, value=[v for v in (old_val or []) if v in _all_modifiers]))
                results.append(gr.update(choices=list(_wildcard_choices), value=[w for w in (wc_val or []) if w in _wildcards]))
                return results

            reload_btn.click(
                fn=_do_refresh,
                inputs=[base] + dd_components + [wildcards, local_dir_path],
                outputs=[base] + dd_components + [wildcards],
                show_progress=False,
            )
            def _refresh_models_and_status(api_url, current_model):
                return _refresh_models(api_url, current_model), _get_ollama_status(api_url)
            refresh_models_btn.click(fn=_refresh_models_and_status, inputs=[api_url, model], outputs=[model, ollama_status], show_progress=False)

            # ── Hidden bridges ──
            prompt_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_in")
            prompt_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_out")
            negative_in = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_in")
            negative_out = gr.Textbox(visible=False, elem_id=f"{tab}_pe_neg_out")

            # ── Prose ──
            def _enhance(source, api_url, model, base_name, custom_sp, *args):
                neg_cb, temp = args[-1], args[-2]
                prepend, sd, dl, th = args[-6], args[-5], args[-4], args[-3]
                wc = args[-7]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()
                source = (source or "").strip()
                if not source:
                    yield "", "", "<span style='color:#c66'>Source prompt is empty.</span>"
                    return

                mods = _collect_modifiers(dd_vals)
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", "<span style='color:#c66'>No system prompt configured.</span>"
                    return

                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                # Build user message with modifiers + wildcards
                user_msg = f"SOURCE PROMPT: {source}"
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                wc_text = _build_wildcard_text(wc, source)
                if wc_text:
                    user_msg = f"{user_msg}\n\n{wc_text}"

                yield gr.update(), gr.update(), "<span style='color:#aaa'>Generating prose...</span>"

                print(f"[PromptEnhancer] Prose: model={model}, think={th}, mods={len(mods)}, wc={len(wc or [])}, seed={int(sd)}, neg={neg_cb}")
                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Prose: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw)
                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""
                    if prepend and source:
                        result = f"{source}\n\n{result}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield result, negative, f"<span style='color:#6c6'>OK - {len(result.split())} words, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>Cancelled - {len(partial.split())} words (partial)</span>"
                    else:
                        yield "", "", "<span style='color:#c66'>Cancelled</span>"
                except _TruncatedError as e:
                    result = _clean_output(str(e))
                    yield result, "", f"<span style='color:#ca6'>Truncated - {len(result.split())} words</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{msg}</span>"

            prose_event = enhance_btn.click(
                fn=_enhance,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt]
                       + dd_components
                       + [wildcards, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Hybrid (two-pass: prose → extract tags + NL) ──
            def _hybrid(source, api_url, model, base_name, custom_sp, tag_fmt, validation_mode,
                        *args):
                neg_cb, temp = args[-1], args[-2]
                prepend, sd, dl, th = args[-6], args[-5], args[-4], args[-3]
                wc = args[-7]
                dd_vals = args[:-7]
                t0 = time.monotonic()

                source = (source or "").strip()
                if not source:
                    yield "", "", "<span style='color:#c66'>Source prompt is empty.</span>"
                    return

                mods = _collect_modifiers(dd_vals)
                sp = _assemble_system_prompt(base_name, custom_sp, dl)
                if not sp:
                    yield "", "", "<span style='color:#c66'>No system prompt configured.</span>"
                    return

                if neg_cb:
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                user_msg = f"SOURCE PROMPT: {source}"
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                wc_text = _build_wildcard_text(wc, source)
                if wc_text:
                    user_msg = f"{user_msg}\n\n{wc_text}"

                print(f"[PromptEnhancer] Hybrid pass 1/3 (prose): model={model}, think={th}, seed={int(sd)}, neg={neg_cb}")
                try:
                    # Pass 1: generate prose (uses base + modifiers + detail level)
                    prose_raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 1/3 prose: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 1/3 prose: {p['elapsed']:.1f}s...</span>"
                        else:
                            prose_raw = chunk
                    prose_raw = _clean_output(prose_raw)
                    if not prose_raw:
                        yield "", "", "<span style='color:#c66'>Prose generation returned empty.</span>"
                        return

                    # Split negative from prose before passes 2 & 3
                    if neg_cb:
                        prose, negative = _split_positive_negative(prose_raw)
                    else:
                        prose, negative = prose_raw, ""

                    print(f"[PromptEnhancer] Hybrid pass 2/3 (tags): {len(prose.split())} words → tags")

                    # Pass 2: extract tags (tag format prompt + wildcard context)
                    fmt_config = _tag_formats.get(tag_fmt, {})
                    tag_sp = fmt_config.get("system_prompt", "")
                    if not tag_sp:
                        yield "", "", "<span style='color:#c66'>No tag format configured.</span>"
                        return
                    if wc_text:
                        tag_sp = f"{tag_sp}\n\nThe following creative choices were requested. Ensure they are reflected in the tags:\n{wc_text}"
                    tags_raw = None
                    for chunk in _call_llm_progress(prose, api_url, model, tag_sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 2/3 tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 2/3 tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            tags_raw = chunk
                    tags_raw = _clean_output(tags_raw, strip_underscores=False)
                    print(f"[PromptEnhancer] Hybrid pass 3/3 (summarize): → NL supplement")

                    # Pass 3: summarize prose to 1-2 compositional sentences
                    summarize_sp = _prompts.get("summarize", "")
                    style_str = _build_style_string(mods)
                    if style_str:
                        summarize_sp = f"{summarize_sp}\n\nThe following styles were applied: {style_str} Ensure these stylistic choices are reflected in the compositional summary."
                    nl_supplement = None
                    for chunk in _call_llm_progress(prose, api_url, model, summarize_sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 3/3 summarize: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Hybrid 3/3 summarize: {p['elapsed']:.1f}s...</span>"
                        else:
                            nl_supplement = chunk
                    nl_supplement = _clean_output(nl_supplement)

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>Validating tags ({validation_mode})...</span>"

                    # Post-process negative tags through tag pipeline when applicable
                    if neg_cb and negative:
                        negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    # Run tags through full post-processing pipeline
                    tags_raw, stats = _postprocess_tags(tags_raw, tag_fmt, validation_mode)
                    tag_count = stats.get("total", 0) if stats else len([t for t in tags_raw.split(",") if t.strip()])
                    status_parts = [f"{tag_count} tags + NL"]
                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")

                    final = f"{tags_raw}\n\n{nl_supplement}" if nl_supplement else tags_raw
                    if prepend and source:
                        final = f"{source}\n\n{final}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield final, negative, f"<span style='color:#6c6'>OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError as e:
                    partial = _clean_output(str(e))
                    if partial:
                        yield partial, "", f"<span style='color:#c66'>Cancelled (partial)</span>"
                    else:
                        yield "", "", "<span style='color:#c66'>Cancelled</span>"
                except _TruncatedError as e:
                    yield _clean_output(str(e), strip_underscores=False), "", "<span style='color:#ca6'>Truncated</span>"
                except urllib.error.URLError as e:
                    msg = f"Connection failed: {e.reason} - is Ollama running?"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{msg}</span>"
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error(msg)
                    yield "", "", f"<span style='color:#c66'>{msg}</span>"

            hybrid_event = hybrid_btn.click(
                fn=_hybrid,
                inputs=[source_prompt, api_url, model, base, custom_system_prompt,
                        tag_format, tag_validation]
                       + dd_components
                       + [wildcards, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Remix ──
            def _detect_format(text):
                """Detect prompt format: 'prose', 'tags', or 'hybrid'."""
                if not text:
                    return "prose"
                paragraphs = text.split("\n\n")
                first_para = paragraphs[0]
                parts = [p.strip() for p in first_para.split(",")]
                if len(parts) < 3:
                    return "prose"
                avg_len = sum(len(p) for p in parts) / len(parts)
                if avg_len >= 30 or len(parts) < 5:
                    return "prose"
                # First paragraph is tags — check if there's an NL supplement after
                if len(paragraphs) > 1 and paragraphs[1].strip():
                    return "hybrid"
                return "tags"

            def _refine(existing, existing_neg, source, api_url, model, tag_fmt, validation_mode,
                        *args):
                neg_cb, temp = args[-1], args[-2]
                prepend, sd, th = args[-5], args[-4], args[-3]
                wc = args[-6]
                dd_vals = args[:-6]

                _cancel_flag.clear()
                t0 = time.monotonic()

                existing = (existing or "").strip()
                existing_neg = (existing_neg or "").strip()
                print(f"[PromptEnhancer] Remix: existing_len={len(existing)}, source_len={len((source or '').strip())}, neg={neg_cb}")
                if not existing:
                    yield "", "", "<span style='color:#c66'>No prompt to remix. Generate one first with Prose or Tags.</span>"
                    return

                source = (source or "").strip()
                mods = _collect_modifiers(dd_vals)
                print(f"[PromptEnhancer] Remix: mods={len(mods)}, wc={len(wc or [])}, source={'yes' if source else 'no'}")

                if not mods and not wc and not source:
                    yield "", "", "<span style='color:#c66'>Select modifiers/wildcards or update source prompt.</span>"
                    return

                fmt = _detect_format(existing)
                print(f"[PromptEnhancer] Remix: detected={fmt}")

                if fmt == "hybrid":
                    sp = _prompts.get("remix_hybrid", "")
                elif fmt == "tags":
                    sp = _prompts.get("remix_tags", "")
                else:
                    sp = _prompts.get("remix_prose", "")

                if neg_cb:
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}"

                if source:
                    sp = f"{sp}\n\nThe user has provided updated direction. Integrate:\n{source}"
                style_str = _build_style_string(mods)
                if style_str:
                    sp = f"{sp}\n\n{style_str}"
                if any(_wildcards.get(w, "") for w in (wc or [])):
                    sp = f"{sp}\n\n{_prompts.get('wildcard_preamble', '')}"
                for wc_name in (wc or []):
                    wc_prompt = _wildcards.get(wc_name, "")
                    if wc_prompt:
                        sp = f"{sp}\n\n{wc_prompt}"

                # Build user message — include current negative if checkbox is on
                user_msg = existing
                if neg_cb and existing_neg:
                    user_msg = f"{user_msg}\n\nCurrent negative prompt:\n{existing_neg}"

                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Remix: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Remix: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw, strip_underscores=(fmt == "prose"))

                    if neg_cb:
                        result, negative = _split_positive_negative(raw)
                    else:
                        result, negative = raw, ""

                    if fmt == "prose":
                        if prepend and source:
                            result = f"{source}\n\n{result}"
                        elapsed = f"{time.monotonic() - t0:.1f}s"
                        yield result, negative, f"<span style='color:#6c6'>OK - remixed to {len(result.split())} words, {elapsed}</span>"
                        return

                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>Validating tags ({validation_mode})...</span>"

                    if fmt == "hybrid":
                        # Split tags and NL, post-process tags only
                        parts = result.split("\n\n", 1)
                        tag_str = parts[0].strip()
                        nl_supplement = parts[1].strip() if len(parts) > 1 else ""
                        tag_str, stats = _postprocess_tags(tag_str, tag_fmt, validation_mode)
                        tag_count = stats.get("total", 0) if stats else len([t for t in tag_str.split(",") if t.strip()])
                        final = f"{tag_str}\n\n{nl_supplement}" if nl_supplement else tag_str
                        status_parts = [f"remixed {tag_count} tags + NL"]
                    else:
                        # Pure tags
                        result, stats = _postprocess_tags(result, tag_fmt, validation_mode)
                        tag_count = stats.get("total", 0) if stats else len([t for t in result.split(",") if t.strip()])
                        final = result
                        status_parts = [f"remixed {tag_count} tags"]

                    # Post-process negative tags
                    if neg_cb and negative:
                        negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)

                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")

                    if prepend and source:
                        final = f"{source}\n\n{final}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield final, negative, f"<span style='color:#6c6'>OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError:
                    yield "", "", "<span style='color:#c66'>Cancelled</span>"
                except _TruncatedError as e:
                    yield _clean_output(str(e), strip_underscores=(fmt == "prose")), "", "<span style='color:#ca6'>Truncated</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{type(e).__name__}: {e}</span>"

            remix_event = refine_btn.click(
                fn=lambda x, y: (_cancel_flag.clear(), x, y)[1:],
                _js=f"""function(x, y) {{
                    var ta = document.querySelector('#{tab}_prompt textarea');
                    var neg = document.querySelector('#{tab}_neg_prompt textarea');
                    return [ta ? ta.value : x, neg ? neg.value : y];
                }}""",
                inputs=[prompt_in, negative_in], outputs=[prompt_in, negative_in], show_progress=False,
            ).then(
                fn=_refine,
                inputs=[prompt_in, negative_in, source_prompt, api_url, model, tag_format, tag_validation]
                       + dd_components
                       + [wildcards, prepend_source, seed, think, temperature, negative_prompt_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Tags ──
            def _tags(source, api_url, model, tag_fmt, validation_mode, *args):
                neg_cb, temp = args[-1], args[-2]
                prepend, sd, dl, th = args[-6], args[-5], args[-4], args[-3]
                wc = args[-7]
                dd_vals = args[:-7]

                _cancel_flag.clear()
                t0 = time.monotonic()

                source = (source or "").strip()
                if not source:
                    yield "", "", "<span style='color:#c66'>Source prompt is empty.</span>"
                    return

                fmt_config = _tag_formats.get(tag_fmt, {})
                sp = fmt_config.get("system_prompt", "")
                if not sp:
                    yield "", "", "<span style='color:#c66'>No tag format configured.</span>"
                    return

                if neg_cb:
                    neg_quality = fmt_config.get("negative_quality_tags", [])
                    neg_hint = ""
                    if neg_quality:
                        neg_hint = f"\nAlways start negative tags with: {', '.join(neg_quality)}"
                    sp = f"{sp}\n\n{_prompts.get('negative', '')}{neg_hint}"

                # Build user message with everything explicit
                mods = _collect_modifiers(dd_vals)
                user_msg = f"SOURCE PROMPT: {source}"
                style_str = _build_style_string(mods)
                if style_str:
                    user_msg = f"{user_msg}\n\n{style_str}"
                if any(_wildcards.get(w, "") for w in (wc or [])):
                    user_msg = f"{user_msg}\n\n{_prompts.get('wildcard_preamble', '')}"
                for wc_name in (wc or []):
                    wc_prompt = _wildcards.get(wc_name, "")
                    if wc_prompt:
                        user_msg = f"{user_msg}\n\n{wc_prompt}"
                detail = int(dl) if dl else 0
                tag_instruction = _build_detail_instruction(detail, "tags")
                if tag_instruction:
                    user_msg = f"{user_msg}\n\n{tag_instruction} Every tag MUST be consistent with the scene and styles above. Do not contradict any detail."
                else:
                    user_msg = f"{user_msg}\n\nGenerate tags. Every tag MUST be consistent with the scene and styles above. Do not contradict any detail."

                print(f"[PromptEnhancer] Tags: model={model}, think={th}, seed={int(sd)}, neg={neg_cb}")
                try:
                    raw = None
                    for chunk in _call_llm_progress(user_msg, api_url, model, sp, temp, think=th, seed=int(sd)):
                        if isinstance(chunk, dict):
                            p = chunk
                            if p["tokens"] > 0:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Tags: {p['words']} words, {p['elapsed']:.1f}s ({p['tps']:.1f} tok/s)</span>"
                            else:
                                yield gr.update(), gr.update(), f"<span style='color:#aaa'>Tags: {p['elapsed']:.1f}s...</span>"
                        else:
                            raw = chunk
                    raw = _clean_output(raw, strip_underscores=False)
                    if validation_mode != "Off":
                        yield gr.update(), gr.update(), f"<span style='color:#aaa'>Validating tags ({validation_mode})...</span>"
                    if neg_cb:
                        tags, negative = _split_positive_negative(raw)
                        negative, _ = _postprocess_tags(negative, tag_fmt, validation_mode)
                    else:
                        tags, negative = raw, ""
                    tags, stats = _postprocess_tags(tags, tag_fmt, validation_mode)
                    tag_count = stats.get("total", 0) if stats else len([t for t in tags.split(",") if t.strip()])
                    status_parts = [f"{tag_count} tags"]
                    if stats:
                        if stats.get("corrected"):
                            status_parts.append(f"{stats['corrected']} corrected")
                        if stats.get("dropped"):
                            status_parts.append(f"{stats['dropped']} dropped")
                        if stats.get("kept_invalid"):
                            status_parts.append(f"{stats['kept_invalid']} unverified")
                        if stats.get("error"):
                            status_parts.append(stats["error"])

                    if prepend and source:
                        tags = f"{source}\n\n{tags}"
                    elapsed = f"{time.monotonic() - t0:.1f}s"
                    yield tags, negative, f"<span style='color:#6c6'>OK - {', '.join(status_parts)}, {elapsed}</span>"
                except InterruptedError:
                    yield "", "", "<span style='color:#c66'>Cancelled</span>"
                except _TruncatedError as e:
                    yield _clean_output(str(e), strip_underscores=False), "", "<span style='color:#ca6'>Truncated</span>"
                except urllib.error.URLError as e:
                    yield "", "", f"<span style='color:#c66'>Connection failed: {e.reason}</span>"
                except Exception as e:
                    yield "", "", f"<span style='color:#c66'>{type(e).__name__}: {e}</span>"

            tags_event = tags_btn.click(
                fn=_tags,
                inputs=[source_prompt, api_url, model, tag_format, tag_validation]
                       + dd_components
                       + [wildcards, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb],
                outputs=[prompt_out, negative_out, status],
                show_progress=False,
            )

            # ── Cancel ──
            # Only sets the threading flag. The generation function detects
            # it via InterruptedError and returns the "Cancelled" status
            # through Gradio's normal .then() output delivery.
            # - No cancels=: that kills the asyncio task, orphaning the return value
            # - No _js: DOM manipulation can desync Svelte component state
            # - No outputs: avoids racing with the generation function's status output
            # - trigger_mode="multiple": default "once" silently drops repeat clicks
            cancel_btn.click(
                fn=lambda: _cancel_flag.set(),
                inputs=[], outputs=[],
                queue=False,
                show_progress=False,
                trigger_mode="multiple",
            )

            # ── Write to main prompt textarea ──
            prompt_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[prompt_out], outputs=[prompt_out], show_progress=False,
            )

            # ── Write to negative prompt textarea ──
            negative_out.change(
                fn=None,
                _js=f"""function(v) {{
                    if (v) {{
                        var ta = document.querySelector('#{tab}_neg_prompt textarea');
                        if (ta) {{
                            ta.value = v;
                            ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                        }}
                    }}
                    return v;
                }}""",
                inputs=[negative_out], outputs=[negative_out], show_progress=False,
            )

        # ── Metadata ──
        def _parse_modifiers(params):
            """Parse PE Modifiers string into a set of names."""
            raw = params.get("PE Modifiers", "")
            return {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

        def _make_dd_restore(dd_label):
            """Create a restore function for a specific dropdown."""
            dd_choices = _dropdown_choices.get(dd_label, [])
            def restore(params):
                saved = _parse_modifiers(params)
                return [m for m in saved if m in dd_choices and m in _all_modifiers]
            return restore

        self.infotext_fields = [
            (source_prompt, "PE Source"),
            (base, "PE Base"),
            (detail_level, lambda params: min(10, max(0, int(params.get("PE Detail", 0)))) if params.get("PE Detail") else 0),
            # Wildcards
            (wildcards, lambda params: [w.strip() for w in params.get("PE Wildcards", "").split(",") if w.strip()] if params.get("PE Wildcards") else []),
            (think, "PE Think"),
            (seed, lambda params: int(params.get("PE Seed", -1)) if params.get("PE Seed") else -1),
        ]
        # Add each modifier dropdown
        for i, label in enumerate(dd_labels):
            self.infotext_fields.append((dd_components[i], _make_dd_restore(label)))

        self.paste_field_names = ["PE Source", "PE Base", "PE Detail", "PE Modifiers", "PE Wildcards", "PE Think", "PE Seed"]

        return [source_prompt, api_url, model, base, custom_system_prompt,
                *dd_components, wildcards, prepend_source, seed, detail_level, think, temperature,
                negative_prompt_cb]

    def process(self, p, source_prompt, api_url, model, base, custom_system_prompt,
                *args):
        # args = *dd_values, wildcards, prepend_source, seed, detail_level, think, temperature, negative_prompt_cb
        neg_cb = args[-1]
        temp = args[-2]
        think = args[-3]
        detail_level = args[-4]
        pe_seed = args[-5]
        prepend = args[-6]
        wildcards = args[-7]
        dd_vals = args[:-7]

        if source_prompt:
            p.extra_generation_params["PE Source"] = source_prompt
        if base:
            p.extra_generation_params["PE Base"] = base
        if detail_level and int(detail_level) != 0:
            p.extra_generation_params["PE Detail"] = int(detail_level)

        all_mod_names = []
        for dd_val in dd_vals:
            if dd_val:
                all_mod_names.extend(dd_val)
        if all_mod_names:
            p.extra_generation_params["PE Modifiers"] = ", ".join(all_mod_names)
        if wildcards:
            p.extra_generation_params["PE Wildcards"] = ", ".join(wildcards)
        if think:
            p.extra_generation_params["PE Think"] = True
        if neg_cb:
            p.extra_generation_params["PE Negative"] = True
        if _last_seed >= 0:
            p.extra_generation_params["PE Seed"] = _last_seed
