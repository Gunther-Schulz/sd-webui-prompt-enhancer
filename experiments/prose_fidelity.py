"""Prose fidelity test: does the enhancer keep what the source states?

The procedure the 2026-10-03 fidelity fix was found and measured with.
Use it before and after any change to a base, a prompt directive, the
sampling options or the model — and as the template for testing another
enhancer (the Wan2GP one): the sources and the grading are the reusable
part, the call is the part to swap.

WHAT COUNTS AS A FAILURE (operator's ranking, and the grading rule):
  1. CHANGED INPUT — a stated element comes out different: person count,
     gender, who does what, an object, a pose, the medium.
  2. CONTRADICTION — an addition that cannot be true beside what the
     source states (a film stock on a phone snapshot, a DSLR on a CCTV
     view).
  3. EUPHEMISM — an explicit source term replaced by a softer one.
  Plausible invented fill-in is NOT a failure; it is the enhancer's job.

HOW TO RUN A ROUND (small rounds first, one change per round):
  1. Baseline:  python -m experiments.prose_fidelity --out /tmp/a.jsonl
  2. Make ONE change (a base edit, or --repeat-penalty / --no-adherence
     to flip a setting without editing code).
  3. Same command with a new --out. One seed and all sources is enough
     to see a large effect (~1 call per source per base).
  4. READ the outputs (printed in full) and count clean ones per the
     rule above. The printed screens only point at where to look: they
     are keyword checks and miss anything they do not name.
  5. Only when a change looks right, confirm with --seeds 42,7919,137.

The call is the extension's own `_call_llm`, and the system prompt is
assembled the way the Enhance handler (`_enhance`) does it, so a result
here is a result in Forge. Needs Ollama running and the Forge Python
environment (the system one lacks dependencies).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments._pe_bootstrap import pe  # noqa: E402

# Each source fixes things a swap would show up against. "look" marks
# sources that name their own medium, where added camera gear is a
# contradiction; "terms" are explicit words that must survive.
SOURCES = {
    "mall": {"look": True, "text": "woman, 25, light tanned skin, red silk blouse, tight black skirt, leaning back against a metal handrail in a crowded mall, one hand on the rail, low angle three quarter rear view from thighs to head, amateur smartphone photo, soft diffused light"},
    "polaroid": {"look": True, "text": "polaroid photo of an old man fishing from a rowboat on a foggy lake at dawn"},
    "meadow": {"text": "girl in a meadow"},
    "cctv": {"look": True, "text": "woman reading at a kitchen table at night, harsh fluorescent ceiling light, cctv camera view from the ceiling corner"},
    "bed": {"look": True, "text": "amateur phone photo with flash, nude woman, 30, lying on her back on an unmade bed, messy bedroom, looking at the camera"},
    "trio": {"text": "two women and one man sitting at a bar counter, the man in the middle, the woman on the left has short blonde hair, the woman on the right has long black hair, all three holding cocktails"},
    "couple": {"text": "one woman and one man dancing in a living room, the woman leads, the woman wears a green dress and has red hair, the man wears a grey suit and is bald"},
    "couple_x": {"look": True, "text": "amateur phone photo, one woman and one man having sex on a couch, the woman is on top and fully nude, the man lies underneath and still wears a white shirt"},
    "trio_x": {"look": True, "text": "amateur phone photo, threesome on a bed, two women and one man, all nude, the blonde woman is kissing the man while the brunette woman lies beside them watching"},
    "sex": {"look": True, "terms": ["sex", "penetrat", "penis", "missionary"], "text": "amateur phone photo, a woman and a man having sex on a bed, missionary position, penetration, his penis inside her, both nude"},
    "oral": {"terms": ["blowjob", "penis", "mouth", "kneel"], "text": "male pov, a woman kneeling and giving a blowjob, penis in her mouth, she looks up at the camera, bedroom"},
}

_GEAR = re.compile(r"Kodak|Fujifilm|Canon|Hasselblad|Sony|Portra|Ektachrome|Ektar|tilt-shift|film stock|color grading", re.I)
_COUNT = re.compile(r"exactly (\w+) (figures|people)", re.I)
_LEAK = re.compile(r"the source|source prompt|texturing term|describe it|\bapply a\b|as requested|unrelated medium", re.I)


def system_prompt(base: str, source: str, adherence: bool) -> str:
    """Mirror of the Enhance handler's assembly (scripts/prompt_enhancer.py, _enhance)."""
    sp = pe._assemble_system_prompt(base)
    if not sp:
        raise SystemExit(f"no system prompt for base {base!r}")
    if adherence and source:
        sp = f"{sp}\n\n{pe._prompts.get('prose_adherence', '')}"
    return sp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="JSONL file to append results to (resumes: finished cases are skipped)")
    ap.add_argument("--bases", default="Default,Narrative")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated subset of: " + ", ".join(SOURCES))
    ap.add_argument("--model", default="huihui_ai/qwen3.5-abliterated:9b")
    ap.add_argument("--api-url", default="http://127.0.0.1:11434")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--label", default="current", help="name for this arm, stored with each result")
    ap.add_argument("--repeat-penalty", type=float, default=None, help="override the extension's repeat_penalty for this run")
    ap.add_argument("--no-adherence", action="store_true", help="leave out the prose_adherence directive")
    args = ap.parse_args()

    if args.repeat_penalty is not None:
        _request = pe.urllib.request.Request

        def _patched(url, data=None, **kw):
            if data:
                body = json.loads(data)
                if "options" in body:
                    body["options"]["repeat_penalty"] = args.repeat_penalty
                    data = json.dumps(body).encode("utf-8")
            return _request(url, data=data, **kw)

        pe.urllib.request.Request = _patched

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["label"], r["base"], r["source"], r["seed"]))

    names = [s for s in args.sources.split(",") if s]
    unknown = [s for s in names if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}")

    with out_path.open("a") as fh:
        for base in args.bases.split(","):
            for name in names:
                source = SOURCES[name]["text"]
                sp = system_prompt(base, source, not args.no_adherence)
                for seed in (int(s) for s in args.seeds.split(",")):
                    if (args.label, base, name, seed) in done:
                        continue
                    text = pe._clean_output(pe._call_llm(
                        f"SOURCE PROMPT: {source}", args.api_url, args.model, sp,
                        args.temp, think=False, seed=seed))
                    if not text:
                        raise RuntimeError(f"empty LLM output: {base} {name} {seed}")
                    fh.write(json.dumps({"label": args.label, "base": base, "source": name,
                                         "seed": seed, "out": text}, ensure_ascii=False) + "\n")
                    fh.flush()

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    rows = [r for r in rows if r["label"] == args.label]
    for r in rows:
        print(f"\n## {r['base']} · {r['source']} · seed {r['seed']}\n{r['out']}")

    print("\n" + "=" * 78)
    print(f"SCREENS for arm {args.label!r} — pointers only; the verdict is the read above")
    for base in sorted({r["base"] for r in rows}):
        rs = [r for r in rows if r["base"] == base]
        look = [r for r in rs if SOURCES[r["source"]].get("look")]
        gear = [r for r in look if _GEAR.search(r["out"])]
        kept = total = 0
        for r in rs:
            terms = SOURCES[r["source"]].get("terms", [])
            total += len(terms)
            kept += sum(bool(re.search(t, r["out"], re.I)) for t in terms)
        print(f"{base}: {len(rs)} outputs"
              f" | camera gear or grading on a source that names its medium: {len(gear)}/{len(look)}"
              f" | explicit source terms kept: {kept}/{total}" if total else
              f"{base}: {len(rs)} outputs"
              f" | camera gear or grading on a source that names its medium: {len(gear)}/{len(look)}"
              f" | explicit source terms: no such source in this run")
        for r in rs:
            m = _COUNT.search(r["out"])
            if m:
                print(f"  count phrase, check it: {r['source']} seed {r['seed']}: {m.group(0)!r}")
            m = _LEAK.search(r["out"])
            if m:
                print(f"  instruction wording in output: {r['source']} seed {r['seed']}: …{r['out'][max(0, m.start() - 40):m.end() + 40]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
