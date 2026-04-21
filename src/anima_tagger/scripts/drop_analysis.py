"""Drop-rate analysis harness for RAG validation.

Generates LLM drafts for N prompts (cached to disk), then re-runs the
validator against each cached draft with multiple configs. Reports:

  - overall drop rate per config
  - breakdown by drop_reason
  - per-prompt sample of dropped tokens (with reason)

Run:
    python src/anima_tagger/scripts/drop_analysis.py

Configurable via env:
    OLLAMA_URL       (default http://127.0.0.1:11434/api/chat)
    OLLAMA_MODEL     (default huihui_ai/qwen3.5-abliterated:9b)
    DRAFT_CACHE      (default .ai/drafts.json under extension root)
    FORCE_REDRAFT    set to 1 to ignore cache and re-call the LLM
"""

import json
import os
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import yaml

from anima_tagger import load_all
from anima_tagger.validator import TagValidator, ValidationContext
from anima_tagger.tagger import _context_from_shortlist


EXT_DIR = os.path.abspath(os.path.join(_SRC, ".."))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "huihui_ai/qwen3.5-abliterated:9b")
DRAFT_CACHE = os.environ.get(
    "DRAFT_CACHE", os.path.join(EXT_DIR, ".ai", "drafts.json"),
)
FORCE_REDRAFT = os.environ.get("FORCE_REDRAFT") == "1"


# Representative prompts: mix of characters, free-text descriptions,
# niche vs popular, stylized vs plain. Some carry a modifier flavor
# (prefix notes so the draft pass simulates the "Random Artist" path
# where the prose-pass system prompt would normally add an artist ask).
PROMPTS = [
    {"prompt": "hatsune miku holding a cake, chibi", "modifier": None},
    {"prompt": "a girl reading in a cafe",             "modifier": None},
    {"prompt": "girl",                                   "modifier": "random_artist"},
    {"prompt": "a lone samurai at dusk with a sword",   "modifier": "random_artist"},
    {"prompt": "a dragon perched on a tower at sunset", "modifier": None},
    {"prompt": "portrait of a young woman with long silver hair and golden eyes, ornate jewelry", "modifier": None},
    {"prompt": "1girl, long hair, cherry blossoms, kimono, spring", "modifier": None},
    {"prompt": "anime girl with glasses, sweater, library", "modifier": "random_artist"},
]


RANDOM_ARTIST_DIRECTIVE = (
    "Apply these styles to the scene: pick one real illustrator from "
    "the artist shortlist and render the scene in their recognizable "
    "visual style."
)


def call_llm(system_prompt: str, user_msg: str, max_tokens: int = 600) -> str:
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": "5m",
        "options": {
            "temperature": 0.6, "num_predict": max_tokens,
            "top_k": 20, "top_p": 0.8,
            "repeat_penalty": 1.5, "presence_penalty": 1.5,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think\n{user_msg}"},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


def load_yaml(rel: str) -> dict:
    with open(os.path.join(EXT_DIR, rel)) as f:
        return yaml.safe_load(f)


def assemble_base_prompt(bases: dict, name: str) -> str:
    preamble = bases.get("_preamble", {}).get("body", "")
    fmt = bases.get("_format", {}).get("body", "")
    entry = bases[name]
    body = entry["body"] if isinstance(entry, dict) else entry
    return "\n\n".join(p.strip() for p in [preamble, body, fmt] if p and p.strip())


def tokens_of(draft: str) -> list[str]:
    out = []
    for t in draft.replace("\n", ",").split(","):
        t = t.strip()
        if t:
            out.append(t)
    return out


def run_validator(stack, entries: list[dict], *,
                  semantic_threshold: float,
                  semantic_min_post_count: int,
                  entity_min_post_count: int,
                  use_compound_split: bool = False) -> dict:
    """Re-run validation over cached drafts with given thresholds.
    Returns aggregated stats + per-prompt breakdowns."""
    from anima_tagger.shortlist import Shortlist

    validator = TagValidator(
        db=stack.db, index=stack.index, embedder=stack.embedder,
        semantic_threshold=semantic_threshold,
        semantic_min_post_count=semantic_min_post_count,
        entity_min_post_count=entity_min_post_count,
    )

    per_prompt = []
    all_reasons = Counter()
    all_match_types = Counter()
    total_tokens = 0
    total_kept = 0

    for e in entries:
        tokens = tokens_of(e["draft"])
        sl = Shortlist(
            artists=e["shortlist"]["artists"],
            characters=e["shortlist"]["characters"],
            series=e["shortlist"]["series"],
        )
        ctx = _context_from_shortlist(sl, stack.db)
        if use_compound_split:
            results = validator.validate_with_compound_split(tokens, context=ctx)
        else:
            results = validator.validate(tokens, context=ctx)

        reasons = Counter()
        match_types = Counter()
        dropped_samples = []
        for r in results:
            match_types[r.match_type] += 1
            if r.match_type == "drop":
                reasons[r.drop_reason or "unknown"] += 1
                dropped_samples.append({
                    "token": r.original,
                    "reason": r.drop_reason,
                    "score": round(r.confidence, 3),
                })

        kept_results = [r for r in results if r.match_type != "drop"]
        unique_canonicals = {r.canonical for r in kept_results if r.canonical}
        kept = len(unique_canonicals)
        # Composition-relevant = kept minus defaults rule layer always adds
        # (masterpiece + best_quality + score_7 + one safety tag).
        DEFAULTS = {"masterpiece", "best_quality", "score_7", "score_8", "score_9",
                    "safe", "sensitive", "nsfw", "explicit"}
        comp_tags = [c for c in unique_canonicals if c not in DEFAULTS]
        per_prompt.append({
            "prompt": e["prompt"],
            "n_tokens": len(tokens),
            "kept": kept,
            "composition_tags": len(comp_tags),
            "drop_rate": round(sum(reasons.values()) / max(1, len(tokens)), 3),
            "reasons": dict(reasons),
            "match_types": dict(match_types),
            "samples": dropped_samples,
            "comp_tag_sample": sorted(comp_tags)[:15],
        })
        total_tokens += len(tokens)
        total_kept += kept
        all_reasons.update(reasons)
        all_match_types.update(match_types)

    total_comp = sum(pp["composition_tags"] for pp in per_prompt)
    avg_comp = round(total_comp / max(1, len(per_prompt)), 1)
    return {
        "config": {
            "semantic_threshold": semantic_threshold,
            "semantic_min_post_count": semantic_min_post_count,
            "entity_min_post_count": entity_min_post_count,
            "use_compound_split": use_compound_split,
        },
        "total_tokens": total_tokens,
        "total_kept": total_kept,
        "total_composition_tags": total_comp,
        "avg_composition_tags_per_prompt": avg_comp,
        "overall_drop_rate": round(1 - total_kept / max(1, total_tokens), 3),
        "reasons": dict(all_reasons),
        "match_types": dict(all_match_types),
        "per_prompt": per_prompt,
    }


def print_summary(label: str, stats: dict) -> None:
    cfg = stats["config"]
    print(f"\n{'='*80}")
    print(f"CONFIG: {label}")
    print(f"  semantic_threshold={cfg['semantic_threshold']} "
          f"semantic_min_post_count={cfg['semantic_min_post_count']} "
          f"entity_min_post_count={cfg['entity_min_post_count']} "
          f"compound_split={cfg.get('use_compound_split', False)}")
    print(f"OVERALL: {stats['total_kept']}/{stats['total_tokens']} kept "
          f"({stats['overall_drop_rate']*100:.1f}% drop rate)")
    print(f"  composition tags: {stats['total_composition_tags']} total "
          f"(avg {stats['avg_composition_tags_per_prompt']}/prompt)")
    print(f"DROP REASONS:")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason:30s} {count}")
    print(f"MATCH TYPES:")
    for mt, count in sorted(stats["match_types"].items(), key=lambda x: -x[1]):
        print(f"  {mt:30s} {count}")


def main() -> int:
    print("Opening anima_tagger stack …")
    t = time.perf_counter()
    stack = load_all()
    print(f"  opened in {time.perf_counter()-t:.1f}s")

    print("\nLoading models …")
    t = time.perf_counter()
    # Stay inside models() context for both draft generation (one-time) and
    # validator re-runs (multiple configs reusing the same embedder).
    with stack.models():
        print(f"  models up in {time.perf_counter()-t:.1f}s")

        # Draft generation: we're already inside models(); generate_drafts
        # opens its own nested models() but that's harmless (re-entrant).
        entries_need_llm = FORCE_REDRAFT or not os.path.exists(DRAFT_CACHE)

        if entries_need_llm:
            print("\nGenerating drafts via Ollama …")
            entries = []
            bases = load_yaml("bases.yaml")
            tf_anima = load_yaml("tag-formats/anima.yaml")
            detailed_sp = assemble_base_prompt(bases, "Detailed")
            anima_tag_sp = tf_anima["system_prompt"]

            for i, spec in enumerate(PROMPTS, 1):
                prompt = spec["prompt"]
                modifier = spec["modifier"]
                print(f"\n[{i}/{len(PROMPTS)}] {prompt}  (modifier={modifier})")
                t = time.perf_counter()
                sl = stack.build_shortlist(prompt)
                print(f"  shortlist: a={len(sl.artists)} c={len(sl.characters)} s={len(sl.series)} ({time.perf_counter()-t:.1f}s)")
                if sl.artists[:3]:
                    print(f"    artists sample: {sl.artists[:3]}")
                sp = detailed_sp
                frag = sl.as_system_prompt_fragment()
                if frag:
                    sp = f"{sp}\n\n{frag}"
                user_msg = f"SOURCE PROMPT: {prompt}"
                if modifier == "random_artist":
                    user_msg = f"{user_msg}\n\n{RANDOM_ARTIST_DIRECTIVE}"
                t = time.perf_counter()
                prose = call_llm(sp, user_msg, max_tokens=400)
                print(f"  prose: {len(prose)} chars ({time.perf_counter()-t:.1f}s)")
                t = time.perf_counter()
                draft = call_llm(anima_tag_sp, prose, max_tokens=400)
                print(f"  draft: {draft[:160]}{'...' if len(draft) > 160 else ''} ({time.perf_counter()-t:.1f}s)")
                entries.append({
                    "prompt": prompt,
                    "modifier": modifier,
                    "shortlist": {"artists": sl.artists, "characters": sl.characters, "series": sl.series},
                    "prose": prose,
                    "draft": draft,
                })
            os.makedirs(os.path.dirname(DRAFT_CACHE), exist_ok=True)
            with open(DRAFT_CACHE, "w") as f:
                json.dump({"model": MODEL, "entries": entries}, f, indent=2)
            print(f"\n[cache] Saved {len(entries)} drafts to {DRAFT_CACHE}")
        else:
            print(f"\n[cache] Loading drafts from {DRAFT_CACHE}")
            with open(DRAFT_CACHE) as f:
                entries = json.load(f)["entries"]

        # Configs to compare.
        configs = [
            ("baseline", dict(semantic_threshold=0.80, semantic_min_post_count=100, entity_min_post_count=1000)),
            ("split_only",                   dict(semantic_threshold=0.80, semantic_min_post_count=100, entity_min_post_count=1000, use_compound_split=True)),
            ("split+thr_0.75",               dict(semantic_threshold=0.75, semantic_min_post_count=100, entity_min_post_count=1000, use_compound_split=True)),
            ("split+thr_0.70",               dict(semantic_threshold=0.70, semantic_min_post_count=100, entity_min_post_count=1000, use_compound_split=True)),
            ("split+thr_0.70+pop_50",         dict(semantic_threshold=0.70, semantic_min_post_count=50, entity_min_post_count=500, use_compound_split=True)),
            ("split+permissive",             dict(semantic_threshold=0.65, semantic_min_post_count=25, entity_min_post_count=250, use_compound_split=True)),
        ]

        all_stats = []
        for label, cfg in configs:
            stats = run_validator(stack, entries, **cfg)
            print_summary(label, stats)
            all_stats.append((label, stats))

        # Detailed per-prompt baseline view so we can eyeball what's being dropped.
        print("\n\n" + "="*80)
        print("BASELINE per-prompt drops (for eyeballing):")
        print("="*80)
        base_stats = all_stats[0][1]
        for pp in base_stats["per_prompt"]:
            print(f"\n« {pp['prompt']} »  drop_rate={pp['drop_rate']*100:.0f}% "
                  f"({pp['n_tokens']-pp['kept']}/{pp['n_tokens']})")
            for s in pp["samples"]:
                print(f"    {s['reason']:25s} score={s['score']:5.2f}  '{s['token']}'")

        # Save full results for later analysis.
        out_path = os.path.join(EXT_DIR, ".ai", "drop_analysis_results.json")
        with open(out_path, "w") as f:
            json.dump([{"label": l, "stats": s} for l, s in all_stats], f, indent=2)
        print(f"\n[out] Full results written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
