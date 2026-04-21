"""End-to-end evaluation harness that calls the REAL prompt_enhancer.py
functions (via _pe_bootstrap), not simplified mirrors. This closes the
drift that hid integration bugs in earlier rounds of testing.

Pipeline mirrors Forge's Hybrid flow bit-identically:

  source, modifiers, seed, llm →
    build_shortlist(source, modifier_keywords, query_expander)
    prose pass:  sp = pe._assemble_system_prompt(...) + shortlist fragment;
                 user_msg = "SOURCE PROMPT: " + source + style_block
    tag extract: tag_sp from tag-formats/anima.yaml + style directives
    validate:    pe._anima_tag_from_draft(stack, draft, ..., shortlist=sl)
                 — which internally uses compound_split + reads opts
    slot-fill:   pe._retrieve_prose_slot for each pe._active_target_slots
                 when no CAT-tag of that slot survived validation

Metrics go well beyond tag count: subject-count coverage, pose/lighting/
setting markers, artist-category presence, artist-variance across seeds
(the "always @e.o." bug), expected-coverage violations per prompt.

Run:
    python src/anima_tagger/scripts/e2e_eval.py --seeds 3 [--only mod_ra_girl]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Bootstrap the real prompt_enhancer module first (before importing
# anything from it) so its module-level code runs with Forge stubs.
from anima_tagger.scripts._pe_bootstrap import pe

from anima_tagger import load_all, config as anima_config


EXT_DIR = os.path.abspath(os.path.join(_SRC, ".."))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_LLM = "huihui_ai/qwen3.5-abliterated:9b"

DEFAULTS_LOWER = {
    "masterpiece", "best_quality", "score_7", "score_8", "score_9",
    "safe", "sensitive", "nsfw", "explicit", "highres", "absurdres",
}


# ── Prompt suite (20+ scenarios, labeled with expectations) ───────────

@dataclass
class PromptSpec:
    id: str
    source: str
    modifiers: list[str] = field(default_factory=list)
    expected_character: bool = False
    expected_series: bool = False
    expected_artist: bool = False
    expected_subject_count: bool = True
    note: str = ""


SUITE: list[PromptSpec] = [
    # Named entities — must surface the character/series tag
    PromptSpec("char_miku",        "hatsune miku holding a cake, chibi",
               expected_character=True, expected_series=True),
    PromptSpec("char_rem",         "rem from re:zero in maid outfit",
               expected_character=True, expected_series=True),
    PromptSpec("char_fox",         "a fox girl with sword, traditional clothing"),
    # Unnamed subjects
    PromptSpec("terse_girl",       "1girl, long hair, cafe"),
    PromptSpec("woman_tree",       "a woman reading under a tree"),
    PromptSpec("wizard",           "an old wizard casting a spell"),
    PromptSpec("cyberpunk",        "a cyberpunk street scene with holograms",
               expected_subject_count=False),
    # Portrait vs action
    PromptSpec("portrait_knight",  "portrait of a knight"),
    PromptSpec("dancer",           "dancer mid-spin in a ballroom"),
    # Style-loaded
    PromptSpec("rain_melancholy",  "anime girl in the rain, melancholic"),
    PromptSpec("chibi_cat",        "colorful chibi sticker of a cat",
               expected_subject_count=False),
    # Terse + Random Artist — the "always @e.o." bug hunt
    PromptSpec("mod_ra_girl",      "girl",
               modifiers=["🎲 Random Artist"], expected_artist=True,
               note="the 'always @e.o.' case"),
    PromptSpec("user_exact_girl",  "girl",
               modifiers=["🎲 Random Artist", "🎲 Random Era"],
               expected_artist=True,
               note="user's exact reproducer: girl + Random Artist + Random Era"),
    PromptSpec("mod_ra_woman",     "woman",
               modifiers=["🎲 Random Artist"], expected_artist=True),
    PromptSpec("mod_rf_girl",      "girl",
               modifiers=["🎲 Random Franchise"], expected_series=True),
    PromptSpec("mod_ra_samurai",   "samurai",
               modifiers=["🎲 Random Artist"], expected_artist=True),
    PromptSpec("mod_ra_cat",       "cat",
               modifiers=["🎲 Random Artist"], expected_artist=True,
               expected_subject_count=False),
    PromptSpec("mod_ra_empty",     "",
               modifiers=["🎲 Random Artist", "🎲 Random Setting"],
               expected_artist=True, expected_subject_count=False),
    # Control: "random artist" in source (user confirmed this works)
    PromptSpec("source_ra_hint",   "girl, random artist",
               expected_artist=True,
               note="control: LLM CAN emit artist when told via source"),
    # Edge cases
    PromptSpec("empty_plain",      "", expected_subject_count=False,
               note="dice roll without modifiers"),
    PromptSpec("already_tagged",   "1girl, masterpiece, best quality, score_7, safe, hatsune miku, holding cake, twintails",
               expected_character=True),
    PromptSpec("rich_scene",       "a complicated scene with multiple characters, guns, explosions, neon lighting",
               expected_subject_count=False),
]


# ── LLM call (same payload shape as prompt_enhancer.py) ───────────────

def call_llm(model: str, sp: str, up: str, seed: int, temperature: float = 0.8,
             num_predict: int = 1024) -> str:
    body = {
        "model": model, "stream": False, "think": False, "keep_alive": "5m",
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": f"/no_think\n{up}"},
        ],
        "options": {
            "temperature": temperature, "seed": int(seed),
            "top_k": 20, "top_p": 0.8,
            "repeat_penalty": 1.5, "presence_penalty": 1.5,
            "num_predict": int(num_predict),
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


# ── One Hybrid run (bit-identical to Forge) ───────────────────────────

@dataclass
class RunResult:
    prompt_id: str
    source: str
    modifiers: list[str]
    seed: int
    llm: str
    assembled_sp: str
    user_msg: str
    shortlist_artists: list[str]
    shortlist_characters: list[str]
    shortlist_series: list[str]
    prose: str
    draft: str
    tags_pre_slotfill: str
    final_tags: str
    slot_fill: dict
    elapsed: float


def _coherence_filter(stack, prose: str, tags_csv: str,
                      threshold: float = 0.15) -> tuple[str, list[tuple[str, float]]]:
    """Score each tag against the prose via the cross-encoder reranker.
    Drop tags scoring below `threshold` as probable hallucinations.

    Defaults to 0.15 (catches egregious mismatches only — e.g. airplane
    in a speakeasy scene) without hurting edge cases.

    Preserves: canonical defaults (quality/safety/score_N) + whitelisted
    anima tokens + @artist tags (cross-encoder poor at tags that are
    names rather than concepts).

    Returns (filtered_csv, list_of_dropped_with_scores).
    """
    if stack.reranker is None or not tags_csv:
        return tags_csv, []
    raw_tokens = [t.strip() for t in tags_csv.split(",") if t.strip()]
    SKIP = DEFAULTS_LOWER | {"1girl", "1boy", "1other", "solo", "multiple_girls"}
    to_score = []
    to_score_idx = []
    for i, t in enumerate(raw_tokens):
        norm = t.lstrip("@").strip().lower().replace(" ", "_")
        # Skip defaults, subject-count, score tags, and @artist tags
        if norm in SKIP or t.startswith("@") or _SCORE_RE.match(norm):
            continue
        to_score.append(t.replace("_", " "))
        to_score_idx.append(i)
    if not to_score:
        return tags_csv, []
    scores = stack.reranker.score_pairs(prose, to_score)
    keep = set(range(len(raw_tokens)))
    dropped: list[tuple[str, float]] = []
    for idx, score in zip(to_score_idx, scores):
        if score < threshold:
            keep.discard(idx)
            dropped.append((raw_tokens[idx], score))
    filtered = [raw_tokens[i] for i in range(len(raw_tokens)) if i in keep]
    return ", ".join(filtered), dropped


def run_one(stack, spec: PromptSpec, seed: int, llm: str,
            temperature: float, coherence_threshold: float | None = None) -> RunResult:
    t0 = time.perf_counter()

    # 1. Assemble mods in the same dict-pair shape _collect_modifiers returns
    mods = []
    for mod_name in spec.modifiers:
        entry = pe._all_modifiers.get(mod_name)
        if entry:
            mods.append((mod_name, entry))

    # 2. REAL system prompt — pe._assemble_system_prompt, detail=0 (slider removed)
    sp = pe._assemble_system_prompt("Detailed", None, 0)

    # 3. Shortlist via the real anima_tagger stack (query expander LLM-backed)
    style_prose_str = pe._build_style_string(mods, mode="prose")
    def _expander(src, mk):
        from anima_tagger.query_expansion import expand_query
        def _oneshot(sp_, up_):
            return call_llm(llm, sp_, up_, seed=seed, temperature=0.3, num_predict=256)
        return expand_query(src, _oneshot, modifier_keywords=mk)

    sl = stack.build_shortlist(
        source_prompt=spec.source,
        modifier_keywords=style_prose_str,
        query_expander=_expander,
    )

    # 4. Inject shortlist fragment into SP — exactly as Hybrid does
    frag = sl.as_system_prompt_fragment()
    if frag:
        sp = f"{sp}\n\n{frag}"

    # 5. User message for prose pass: source + style block
    user_msg = f"SOURCE PROMPT: {spec.source}" if spec.source else pe._EMPTY_SOURCE_SIGNAL
    if style_prose_str:
        user_msg = f"{user_msg}\n\n{style_prose_str}"

    # 6. Prose pass
    prose = call_llm(llm, sp, user_msg, seed=seed, temperature=temperature)

    # 7. Tag extract pass — tag_sp from anima format + style directives
    fmt_config = pe._tag_formats.get("Anima", {})
    tag_sp = fmt_config.get("system_prompt", "")
    tag_style_str = pe._build_style_string(mods, mode="tags")
    if tag_style_str:
        tag_sp = f"{tag_sp}\n\nThe following style directives were requested. Ensure they are reflected in the tags:\n{tag_style_str}"

    draft = call_llm(llm, tag_sp, prose, seed=seed, temperature=temperature)

    # 8. Validate + rule-layer (real _anima_tag_from_draft — reads opts,
    #    applies compound_split, rule_layer defaults, etc.)
    safety = pe._anima_safety_from_modifiers(mods) if hasattr(pe, "_anima_safety_from_modifiers") else "safe"
    tags_str, stats = pe._anima_tag_from_draft(stack, draft, safety=safety, shortlist=sl)
    tags_pre_slotfill = tags_str

    # 9. Slot-fill pass — only fires for modifiers with target_slot
    slot_fill: dict = {}
    slots = pe._active_target_slots(mods) if hasattr(pe, "_active_target_slots") else []
    for slot in slots:
        cat_info = pe._SLOT_TO_CATEGORY.get(slot)
        if not cat_info:
            continue
        if pe._tags_have_category(tags_str, stack, cat_info["category"]):
            slot_fill[slot] = "already-present"
            continue
        picked = pe._retrieve_prose_slot(stack, prose, slot)
        if picked:
            tag_out = picked.replace("_", " ")
            if slot == "artist":
                tag_out = "@" + tag_out
            tags_str = f"{tags_str}, {tag_out}"
            slot_fill[slot] = picked
        else:
            slot_fill[slot] = None

    # Optional coherence filter — drops tags that score very low against
    # the prose via the cross-encoder reranker. A/B'd to see if it kills
    # scene-incoherent hallucinations without hurting legitimate tags.
    coherence_dropped: list = []
    tags_pre_coherence = tags_str
    if coherence_threshold is not None:
        tags_str, coherence_dropped = _coherence_filter(
            stack, prose, tags_str, threshold=coherence_threshold,
        )

    return RunResult(
        prompt_id=spec.id, source=spec.source, modifiers=list(spec.modifiers),
        seed=seed, llm=llm,
        assembled_sp=sp, user_msg=user_msg,
        shortlist_artists=sl.artists, shortlist_characters=sl.characters,
        shortlist_series=sl.series,
        prose=prose, draft=draft,
        tags_pre_slotfill=tags_pre_slotfill, final_tags=tags_str,
        slot_fill=slot_fill,
        elapsed=time.perf_counter() - t0,
    ), coherence_dropped


# ── Quality metrics ──────────────────────────────────────────────────

_SUBJECT_MARKERS = {"1girl", "1boy", "1other", "solo", "multiple_girls",
                    "multiple_boys", "no_humans", "2girls", "3girls"}
_POSE_HINTS = {"standing", "sitting", "walking", "running", "leaning",
               "kneeling", "crouching", "lying", "looking_at_viewer",
               "looking_back", "looking_down", "arms_up", "hands_on_hips",
               "crossed_arms", "holding", "holding_book", "a_pose",
               "contrapposto", "hand_on_hip", "hand_on_own_hip"}
_LIGHTING_HINTS = {"sunlight", "backlighting", "rim_lighting", "golden_hour",
                   "sunset", "dramatic_lighting", "soft_lighting", "overhead_lights",
                   "cinematic_lighting", "warm_lighting", "neon_lights",
                   "candlelight", "moonlight", "twilight", "warm_light",
                   "natural_light", "soft_light", "dim_lighting"}
_SETTING_HINTS = {"outdoors", "indoors", "cafe", "library", "forest", "city",
                  "street", "bedroom", "classroom", "beach", "mountain",
                  "ruins", "tower", "shrine", "park", "desert", "office",
                  "garden", "meadow", "rooftop", "alley"}


def _norm(tag: str) -> str:
    return tag.strip().lstrip("@").lower().replace(" ", "_").replace("-", "_")


def tag_category(stack, tag: str):
    n = _norm(tag)
    if not n:
        return None
    rec = stack.db.get_by_name(n)
    return rec["category"] if rec else None


import re as _re_qm
_MALFORMED_ARTIST_RE = _re_qm.compile(r"^@[^@]*(,|\s\s|$)")
_SCORE_RE = _re_qm.compile(r"^score_\d+(_up)?$")


def _quality_issues(stack, tag_csv: str) -> dict:
    """Detect quality issues the user flagged: malformed @artist, duplicate
    score tags, tags not found in DB (hallucinations that slipped through),
    redundancy between shadow / dark_shadow / long_shadow kind of overlaps."""
    raw_tokens = [t.strip() for t in tag_csv.split(",") if t.strip()]
    normed = [_norm(t) for t in raw_tokens]

    # Duplicate score_N
    scores = [n for n in normed if _SCORE_RE.match(n)]
    duplicate_scores = scores if len(scores) > 1 else []

    # Malformed @artist tokens (contain whitespace after the name,
    # or colon patterns that aren't real DB artists)
    malformed_artist = []
    for t in raw_tokens:
        if not t.startswith("@"):
            continue
        bare = t[1:].strip().lower().replace(" ", "_").replace("-", "_")
        rec = stack.db.get_by_name(bare)
        # If not in DB OR not an artist — flag it
        if rec is None:
            malformed_artist.append(t)
        elif rec.get("category") != 1:  # CAT_ARTIST
            malformed_artist.append(f"{t} (cat={rec['category']})")

    # Hallucinated tokens (non-whitelist, non-default, not in DB)
    whitelist_ok = {"masterpiece", "best_quality", "score_7", "score_8", "score_9",
                    "safe", "sensitive", "nsfw", "explicit", "highres", "absurdres"}
    hallucinated = []
    for n in normed:
        if n in whitelist_ok:
            continue
        if _SCORE_RE.match(n):
            continue  # any score_N is OK
        rec = stack.db.get_by_name(n)
        if rec is None:
            hallucinated.append(n)

    # Redundancy clusters — tokens where one contains another as a stem
    redundancy_pairs = []
    for i, a in enumerate(normed):
        for b in normed[i + 1:]:
            if a == b:
                continue
            # a is a subset word of b OR vice versa, both are non-trivial words
            if len(a) >= 4 and len(b) >= 4:
                if a in b.split("_") and b != a:
                    redundancy_pairs.append((a, b))
                elif b in a.split("_") and a != b:
                    redundancy_pairs.append((b, a))

    return {
        "duplicate_scores": duplicate_scores,
        "malformed_artist": malformed_artist,
        "hallucinated": hallucinated,
        "redundancy_pairs": redundancy_pairs,
        "num_issues": (
            (1 if len(duplicate_scores) > 1 else 0)
            + len(malformed_artist)
            + len(hallucinated)
            + len(redundancy_pairs)
        ),
    }


def score(run: RunResult, spec: PromptSpec, stack) -> dict:
    tags = [_norm(t) for t in run.final_tags.split(",") if t.strip()]
    comp = [t for t in tags if t not in DEFAULTS_LOWER]
    artist_tokens = [t.strip() for t in run.final_tags.split(",") if t.strip().startswith("@")]
    qi = _quality_issues(stack, run.final_tags)

    has_character = any(tag_category(stack, t) == anima_config.CAT_CHARACTER
                        for t in run.final_tags.split(","))
    has_series = any(tag_category(stack, t) == anima_config.CAT_COPYRIGHT
                     for t in run.final_tags.split(","))
    has_artist = bool(artist_tokens) or any(
        tag_category(stack, t) == anima_config.CAT_ARTIST
        for t in run.final_tags.split(","))
    has_subject = bool(set(tags) & _SUBJECT_MARKERS)
    has_pose = any(h in tags for h in _POSE_HINTS)
    has_lighting = any(h in tags for h in _LIGHTING_HINTS)
    has_setting = any(h in tags for h in _SETTING_HINTS)

    violations = []
    if spec.expected_character and not has_character:
        violations.append("missing_character")
    if spec.expected_series and not has_series:
        violations.append("missing_series")
    if spec.expected_artist and not has_artist:
        violations.append("missing_artist")
    if spec.expected_subject_count and not has_subject:
        violations.append("missing_subject_count")

    draft_count = len([t for t in run.draft.split(",") if t.strip()])
    # Drop rate: fraction of draft tokens that didn't survive validation.
    # (Negative → compound_split expanded some tokens into multiple sub-tags;
    # we floor at 0 for reporting.)
    drop_rate = max(0.0, (draft_count - len(tags)) / draft_count) if draft_count else 0.0
    return {
        "comp_count": len(comp),
        "total_count": len(tags),
        "draft_tokens": draft_count,
        "drop_rate": round(drop_rate, 2),
        "has_character": has_character,
        "has_series": has_series,
        "has_artist": has_artist,
        "artist_picks": artist_tokens,
        "has_subject_count": has_subject,
        "has_pose": has_pose,
        "has_lighting": has_lighting,
        "has_setting": has_setting,
        "coverage_score": sum([has_subject, has_pose, has_lighting, has_setting]),
        "violations": violations,
        "slot_fill": run.slot_fill,
        "elapsed": round(run.elapsed, 1),
        "prose_words": len(run.prose.split()),
        # Quality-issue fields (user-flagged):
        "qi_duplicate_scores": qi["duplicate_scores"],
        "qi_malformed_artist": qi["malformed_artist"],
        "qi_hallucinated": qi["hallucinated"],
        "qi_redundancy_pairs": qi["redundancy_pairs"],
        "qi_num_issues": qi["num_issues"],
    }


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--llm", default=DEFAULT_LLM)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--only", default="", help="comma-separated prompt IDs")
    parser.add_argument("--coherence", type=float, default=None,
                        help="Coherence filter threshold (e.g. 0.15). Omit → filter off.")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    prompts = [p for p in SUITE if not only or p.id in only]
    seeds = list(range(1000, 1000 + args.seeds))

    print(f"Running {len(prompts)} prompts × {len(seeds)} seeds "
          f"= {len(prompts)*len(seeds)} runs  |  llm={args.llm}  temp={args.temperature}")

    stack = load_all()
    print("Loading models …")
    t = time.perf_counter()
    with stack.models():
        print(f"  up in {time.perf_counter()-t:.1f}s")

        results: list[tuple[PromptSpec, RunResult, dict]] = []
        for spec in prompts:
            print(f"\n── {spec.id} ── source={spec.source!r} mods={spec.modifiers}")
            for seed in seeds:
                try:
                    run, coh_dropped = run_one(
                        stack, spec, seed, args.llm, args.temperature,
                        coherence_threshold=args.coherence,
                    )
                except Exception as e:
                    print(f"  seed={seed}: FAILED {type(e).__name__}: {e}")
                    continue
                sc = score(run, spec, stack)
                if coh_dropped:
                    sc["coherence_dropped"] = coh_dropped
                violated = "; ".join(sc["violations"]) or "OK"
                qi_flags = []
                if sc["qi_duplicate_scores"]:
                    qi_flags.append(f"dup_scores={sc['qi_duplicate_scores']}")
                if sc["qi_malformed_artist"]:
                    qi_flags.append(f"bad_artist={sc['qi_malformed_artist']}")
                if sc["qi_hallucinated"]:
                    qi_flags.append(f"hallu={sc['qi_hallucinated']}")
                if sc["qi_redundancy_pairs"]:
                    qi_flags.append(f"redund={sc['qi_redundancy_pairs']}")
                qi_str = f"  ⚠ {'  '.join(qi_flags)}" if qi_flags else ""
                print(f"  seed={seed}  prose={sc['prose_words']}w "
                      f"draft={sc['draft_tokens']}t comp={sc['comp_count']:2d} "
                      f"drop={sc['drop_rate']*100:.0f}% "
                      f"cov={sc['coverage_score']}/4 artist={sc['artist_picks'] or '—'}  "
                      f"[{violated}]  {sc['elapsed']}s{qi_str}")
                results.append((spec, run, sc))

        # Aggregate
        print("\n" + "=" * 78)
        print(f"SUMMARY — llm={args.llm}  temp={args.temperature}  runs={len(results)}")
        print("=" * 78)

        # Artist variance per prompt (the "always @e.o." test)
        print("\nARTIST VARIANCE (expected_artist prompts only):")
        by_pid = defaultdict(list)
        for spec, run, sc in results:
            by_pid[spec.id].append((run, sc))
        for pid, rs in by_pid.items():
            spec = next(p for p in prompts if p.id == pid)
            if not spec.expected_artist:
                continue
            picks = [tuple(sc["artist_picks"]) for _, sc in rs]
            uniq = len({p for p in picks if p})
            print(f"  {pid:20s} {uniq}/{len(picks)} unique — {picks}")

        # Violations
        vios = Counter()
        for _, _, sc in results:
            for v in sc["violations"]:
                vios[v] += 1
        total = max(1, len(results))
        print("\nVIOLATIONS:")
        for v, n in vios.most_common():
            print(f"  {v:30s} {n}/{total} ({100*n/total:.0f}%)")

        avg_comp = sum(sc["comp_count"] for _, _, sc in results) / total
        avg_cov = sum(sc["coverage_score"] for _, _, sc in results) / total
        avg_prose = sum(sc["prose_words"] for _, _, sc in results) / total
        avg_draft = sum(sc["draft_tokens"] for _, _, sc in results) / total
        avg_drop = sum(sc["drop_rate"] for _, _, sc in results) / total
        print(f"\nAVG comp_tags={avg_comp:.1f} drop={avg_drop*100:.0f}% cov={avg_cov:.2f}/4 "
              f"prose={avg_prose:.0f}w draft={avg_draft:.0f}t")

        # Quality issue totals
        qi_counts = Counter()
        for _, _, sc in results:
            if sc["qi_duplicate_scores"]:
                qi_counts["duplicate_scores"] += 1
            if sc["qi_malformed_artist"]:
                qi_counts["malformed_artist"] += 1
            if sc["qi_hallucinated"]:
                qi_counts["hallucinated"] += 1
            if sc["qi_redundancy_pairs"]:
                qi_counts["redundancy_pairs"] += 1
        print("\nQUALITY ISSUES (runs affected):")
        for k, n in qi_counts.most_common():
            print(f"  {k:25s} {n}/{total} ({100*n/total:.0f}%)")
        total_hallucinated = sum(len(sc["qi_hallucinated"]) for _, _, sc in results)
        total_bad_artists = sum(len(sc["qi_malformed_artist"]) for _, _, sc in results)
        total_redund = sum(len(sc["qi_redundancy_pairs"]) for _, _, sc in results)
        print(f"  (cumulative) hallucinated tokens: {total_hallucinated}")
        print(f"  (cumulative) malformed artists:   {total_bad_artists}")
        print(f"  (cumulative) redundancy pairs:    {total_redund}")

        # Persist
        out_path = os.path.join(EXT_DIR, ".ai",
                                f"e2e_{args.llm.replace(':', '_').replace('/', '_')}"
                                f"_t{args.temperature}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "llm": args.llm, "temperature": args.temperature,
                "runs": [
                    {"spec_id": spec.id, "source": spec.source,
                     "modifiers": spec.modifiers, "seed": run.seed,
                     "assembled_sp_preview": run.assembled_sp[:500],
                     "shortlist": {"artists": run.shortlist_artists,
                                   "characters": run.shortlist_characters,
                                   "series": run.shortlist_series},
                     "prose": run.prose, "draft": run.draft,
                     "tags_pre_slotfill": run.tags_pre_slotfill,
                     "final_tags": run.final_tags,
                     "slot_fill": run.slot_fill,
                     "score": sc}
                    for spec, run, sc in results
                ],
            }, f, indent=2)
        print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
