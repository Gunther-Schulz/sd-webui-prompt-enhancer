"""Prose fidelity test: does the enhancer keep what the source states,
and does it invent well where the source is silent?

The instrument the 2026-10-03 tuning was done with. The method, the
findings and the dead ends are in experiments/PROSE-TUNING.md — read it
before changing a base, a directive, the sampling options or the model.
The sources and the grading are the reusable part (also for testing
another enhancer, e.g. Wan2GP's); the model call is the part to swap.

Quick use (Forge's Python environment, Ollama running):

  # what the extension does today, one seed, every source
  python -m experiments.prose_fidelity --out /tmp/now.jsonl

  # a candidate base text from a file, without touching bases.yaml
  python -m experiments.prose_fidelity --out /tmp/cand.jsonl \\
      --label cand --base-file /tmp/candidate_base.txt

  # confirm a candidate on three seeds
  python -m experiments.prose_fidelity --out /tmp/cand.jsonl \\
      --label cand --base-file /tmp/candidate_base.txt --seeds 42,7919,137

  # variety between seeds on the bare sources
  python -m experiments.prose_fidelity --out /tmp/var.jsonl --label var \\
      --sources var_sex,var_dance --seeds 42,137,1729,7919,10001,65537,1000003,2147483000

The call is the extension's own `_call_llm` and the system prompt is
assembled the way the Enhance handler does it, so a result here is a
result in Forge. The printed scores are keyword checks: they point at
where to read and they miss anything they do not name. The verdict is
reading the outputs.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments._pe_bootstrap import pe  # noqa: E402

# ── Sources ──────────────────────────────────────────────────────────────
# `check` lists what the source STATES, as name -> regex over the output.
# The names age / gender / count are the hard class (operator: never
# dropped, never skewed). `no_angle` marks sources that give no camera
# angle; `sexual` the ones where softeners are counted.
SOURCES = {
    "mall": {"text": "woman, 25, light tanned skin, red silk blouse, tight black skirt, leaning back against a metal handrail in a crowded mall, one hand on the rail, low angle three quarter rear view from thighs to head, amateur smartphone photo, soft diffused light",
             "check": {"age": r"\b25\b|twenty-five", "gender": r"\bwoman\b", "tanned": "tanned", "red silk blouse": "red silk blouse", "black skirt": "black skirt", "handrail": "handrail", "mall": r"\bmall\b", "hand on rail": r"\bhands?\b", "low angle": r"low[- ]angle", "three quarter rear": r"three[- ]quarter rear", "thighs to head": "thighs", "smartphone": "smartphone|phone", "diffused": "diffused"}},
    "mall_short": {"no_angle": True, "text": "woman, 25, standing in a crowded mall",
                   "check": {"age": r"\b25\b|twenty-five", "gender": r"\bwoman\b", "standing": "stand", "mall": r"\bmall\b", "crowded": "crowd|bustl|throng"}},
    "mall_steer": {"no_angle": True, "text": "woman, 25, standing in a crowded mall, describe surroundings",
                   "check": {"age": r"\b25\b|twenty-five", "gender": r"\bwoman\b", "mall": r"\bmall\b"}},
    "angle_steer": {"text": "woman, 25, standing in a crowded mall, pick an unusual camera angle",
                    "check": {"age": r"\b25\b|twenty-five", "gender": r"\bwoman\b", "an angle": r"angle|from above|from below|overhead|bird's|dutch|worm's"}},
    "cctv": {"text": "woman reading at a kitchen table at night, harsh fluorescent ceiling light, cctv camera view from the ceiling corner",
             "check": {"gender": r"\bwoman\b", "reading": "read|book", "kitchen": "kitchen", "table": r"\btable\b", "night": "night", "fluorescent": "fluorescent", "cctv": "cctv|security|surveillance", "corner": "corner"}},
    "bed": {"text": "amateur phone photo with flash, nude woman, 30, lying on her back on an unmade bed, messy bedroom, looking at the camera",
            "check": {"age": r"\b30\b|thirty", "gender": r"\bwoman\b", "nude": "nude|naked", "on her back": "on her back|supine", "unmade bed": "unmade|rumpled", "messy": "mess|clutter|dishevel", "flash": "flash", "looks at camera": "camera|lens", "phone": "phone"}},
    "polaroid": {"text": "polaroid photo of an old man fishing from a rowboat on a foggy lake at dawn",
                 "check": {"gender": r"\bman\b", "old": r"\bold\b|elderly", "fishing": "fish", "rowboat": "row ?boat", "fog": "fog|mist", "lake": "lake", "dawn": "dawn|sunrise|morning", "polaroid": "polaroid|instant"}},
    "woman40": {"no_angle": True, "text": "middle-aged woman gardening",
                "check": {"age": "middle-aged", "gender": r"\bwoman\b", "gardening": "garden"}},
    "fishing": {"no_angle": True, "text": "man fishing", "check": {"gender": r"\bman\b", "fishing": "fish"}},
    "bar": {"no_angle": True, "text": "two women and a man at a bar",
            "check": {"count": r"two women and (a|one) man", "bar": r"\bbar\b"}},
    "trio": {"no_angle": True, "text": "two women and one man sitting at a bar counter, the man in the middle, the woman on the left has short blonde hair, the woman on the right has long black hair, all three holding cocktails",
             "check": {"count": r"two women and (one|a) man", "man in the middle": "middle|cent(er|re|ral)|between", "left: short blonde": r"left[^.;]*short blonde|short blonde[^.;]*left", "right: long black": r"right[^.;]*long (dark )?black|long black[^.;]*right", "cocktails": "cocktail", "bar": r"\bbar\b"}},
    "couple": {"no_angle": True, "text": "one woman and one man dancing in a living room, the woman leads, the woman wears a green dress and has red hair, the man wears a grey suit and is bald",
               "check": {"count": r"(one|a) woman and (one|a) man|woman[^.]*\bman\b", "woman leads": "lead", "green dress": r"green[^.;]{0,20}dress", "red hair": "red hair|redhead", "grey suit": r"gr[ea]y[^.;]{0,20}suit", "bald": "bald", "living room": "living room"}},
    "couple_x": {"sexual": True, "text": "amateur phone photo, one woman and one man having sex on a couch, the woman is on top and fully nude, the man lies underneath and still wears a white shirt",
                 "check": {"count": r"(one|a) woman and (one|a) man|woman[^.]*\bman\b", "sex": r"\bsex\b", "couch": "couch|sofa", "woman on top": "on top|atop|astride", "nude": "nude|naked", "man underneath": "underneath|beneath|under her", "white shirt": r"white[^.;]{0,20}shirt", "phone": "phone"}},
    "trio_x": {"sexual": True, "text": "amateur phone photo, threesome on a bed, two women and one man, all nude, the blonde woman is kissing the man while the brunette woman lies beside them watching",
               "check": {"count": r"two (nude |naked )?women and (one|a) (nude |naked )?man", "threesome": "threesome", "bed": r"\bbed", "nude": "nude|naked", "blonde kisses man": r"blonde[^.]*(kiss|lips)", "brunette beside, watching": r"brunette[^.]*(watch|observ|gaz)", "phone": "phone"}},
    "sex": {"sexual": True, "text": "amateur phone photo, a woman and a man having sex on a bed, missionary position, penetration, his penis inside her, both nude",
            "check": {"sex": r"\bsex\b", "missionary": "missionary", "penetration": "penetrat", "penis": "penis", "nude": "nude|naked", "bed": r"\bbed", "phone": "phone", "gender": r"woman[^.]*\bman\b|\bman\b[^.]*woman"}},
    "oral": {"sexual": True, "text": "male pov, a woman kneeling and giving a blowjob, penis in her mouth, she looks up at the camera, bedroom",
             "check": {"blowjob": "blowjob|blow job", "penis": "penis", "mouth": "mouth", "kneeling": "kneel", "gender": r"\bwoman\b", "pov": "pov|point of view|perspective"}},
    "sex_steer": {"sexual": True, "text": "couple having sex, describe their faces", "check": {"sex": r"\bsex\b"}},
    "sex_short": {"sexual": True, "no_angle": True, "text": "couple having sex", "check": {"sex": r"\bsex\b"}},
    # variety probes: run on many seeds, read how much the scenes differ
    "var_sex": {"sexual": True, "no_angle": True, "text": "a man and a woman having sex", "check": {"sex": r"\bsex\b", "count": r"man and a woman"}},
    "var_dance": {"no_angle": True, "text": "a man and a woman dancing", "check": {"dancing": "danc", "count": r"man and a woman"}},
}
HARD = ("age", "gender", "count")
DEFAULT_SET = [k for k in SOURCES if not k.startswith("var_")]

# ── Screens: each names ONE failure class found on 2026-10-03 ────────────
SCREENS = {
    "age-skew word beside a stated age": (re.compile(r"\b(mature|elderly|older|teen|teenage|girl|lady|youthful)\b", re.I), lambda s: "age" in SOURCES[s]["check"] and s != "woman40"),
    "softener for a sexual term": (re.compile(r"making love|lovemaking|intimate embrace|intimacy|coupling|\bmember\b|\bshaft\b|oral pleasure|passionate embrace|intercourse", re.I), lambda s: SOURCES[s].get("sexual")),
    "sexual word in a non-sexual source": (re.compile(r"penis|vagina|\bnude\b|naked|\bsex\b", re.I), lambda s: not SOURCES[s].get("sexual") and s != "bed"),
    "hedge (perhaps / maybe)": (re.compile(r"\b(perhaps|maybe|possibly|presumably)\b", re.I), lambda s: True),
    "'X or Y' alternative": (re.compile(r"\b\w+ or \w+\b"), lambda s: True),
    "vague placeholder": (re.compile(r"suitable for|suited (to|for)|practical (clothing|attire)|casual attire|simple attire|whatever|some kind of|appropriate for|a quiet spot|general setting", re.I), lambda s: True),
    "smell or sound": (re.compile(r"\b(scent|scents|smell|smells|aroma|hum|hums|humming|chatter|sound|sounds|noise|echo|echoing|clatter|clink|clinking)\b", re.I), lambda s: True),
    "camera gear or grading": (re.compile(r"Kodak|Fujifilm|Canon|Hasselblad|Sony|Portra|Ektachrome|\bf/\d|\d+mm\b|film stock|color grading", re.I), lambda s: True),
    "camera mentioned though the source gives no angle": (re.compile(r"camera|\blens\b|low[- ]angle|high[- ]angle|eye level|close-up|wide shot|medium shot", re.I), lambda s: SOURCES[s].get("no_angle")),
    "instruction voice": (re.compile(r"(^|[.;] )(Apply|Frame|Use|Shoot|Render|Add|Set)\b|\bTo (capture|ground|render|complete|achieve)\b", re.M), lambda s: True),
    "the rule or the request echoed": (re.compile(r"the source|as requested|exact age|one version|made specific|describ(e|ing) (the |their )?(surroundings|faces)", re.I), lambda s: True),
}


def report(rows, label):
    tot = hit = hard_t = hard_h = 0
    missed = collections.Counter()
    for r in rows:
        for name, pat in SOURCES[r["source"]]["check"].items():
            ok = bool(re.search(pat, r["out"], re.I))
            tot += 1; hit += ok
            if name in HARD:
                hard_t += 1; hard_h += ok
            if not ok:
                missed[f"{r['source']}:{name}"] += 1
    print("\n" + "=" * 78)
    print(f"ARM {label!r}: {len(rows)} outputs")
    print(f"  stated elements kept: {hit}/{tot}" + (f" ({100 * hit // tot}%)" if tot else ""))
    print(f"  age / gender / count kept: {hard_h}/{hard_t}   <- the hard class")
    if missed:
        print("  missed: " + ", ".join(f"{k} x{v}" for k, v in missed.most_common(16)))
    for name, (pat, applies) in SCREENS.items():
        scope = [r for r in rows if applies(r["source"])]
        if scope:
            print(f"  {name}: {sum(bool(pat.search(r['out'])) for r in scope)}/{len(scope)}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    for s, rs in by.items():
        if len(rs) >= 4:
            opens = collections.Counter(" ".join(r["out"].lower().split()[:9]) for r in rs)
            print(f"  variety, {s}: {len(opens)} distinct openings in {len(rs)} seeds; most common x{opens.most_common(1)[0][1]}")
    words = [len(r["out"].split()) for r in rows]
    if words:
        print(f"  words: mean {sum(words) // len(words)}, max {max(words)}")
    print("  (keyword checks; a check can be wrong in either direction — read the outputs)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prose fidelity test (see module docstring).")
    ap.add_argument("--out", required=True, help="JSONL to append to; finished cases are skipped on a re-run")
    ap.add_argument("--label", default="current", help="name of this arm, stored with each result")
    ap.add_argument("--bases", default="Default")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sources", default=",".join(DEFAULT_SET), help="comma-separated; available: " + ", ".join(SOURCES))
    ap.add_argument("--model", default="huihui_ai/qwen3.5-abliterated:9b")
    ap.add_argument("--api-url", default="http://127.0.0.1:11434")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--base-file", default=None, help="use this file's text as the base body instead of bases.yaml's (candidate testing)")
    ap.add_argument("--append-file", default=None, help="append this file's text to the system prompt")
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--quiet", action="store_true", help="scores only, do not print the outputs")
    args = ap.parse_args()

    overrides = {k: v for k, v in (("repeat_penalty", args.repeat_penalty), ("presence_penalty", args.presence_penalty)) if v is not None}
    if overrides:
        _request = pe.urllib.request.Request

        def _patched(url, data=None, **kw):
            if data:
                body = json.loads(data)
                if "options" in body:
                    body["options"].update(overrides)
                    data = json.dumps(body).encode("utf-8")
            return _request(url, data=data, **kw)

        pe.urllib.request.Request = _patched

    names = [s for s in args.sources.split(",") if s]
    unknown = [s for s in names if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}")

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["label"], r["base"], r["source"], r["seed"]))

    with out_path.open("a") as fh:
        for base in args.bases.split(","):
            if args.base_file:
                pe._bases[base] = {"body": Path(args.base_file).read_text().strip()}
            # mirrors the Enhance handler: the assembled base, nothing else for a plain source
            sp = pe._assemble_system_prompt(base)
            if not sp:
                raise SystemExit(f"no system prompt for base {base!r}")
            if args.append_file:
                sp = f"{sp}\n\n{Path(args.append_file).read_text().strip()}"
            for name in names:
                for seed in (int(s) for s in args.seeds.split(",")):
                    if (args.label, base, name, seed) in done:
                        continue
                    text = pe._clean_output(pe._call_llm(
                        f"SOURCE PROMPT: {SOURCES[name]['text']}", args.api_url, args.model, sp,
                        args.temp, think=False, seed=seed))
                    if not text:
                        raise RuntimeError(f"empty LLM output: {base} {name} {seed}")
                    fh.write(json.dumps({"label": args.label, "base": base, "source": name,
                                         "seed": seed, "out": text}, ensure_ascii=False) + "\n")
                    fh.flush()

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    rows = [r for r in rows if r["label"] == args.label and r["source"] in SOURCES]
    if not args.quiet:
        for r in rows:
            print(f"\n## {r['base']} · {r['source']} · seed {r['seed']}\n{r['out']}")
    report(rows, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
