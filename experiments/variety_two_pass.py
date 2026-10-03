"""SHELVED PROTOTYPE (2026-10-03) — two-pass scene variety. Not wired into the
extension. Kept so the idea can be picked up again; read the "Variety"
section of experiments/PROSE-TUNING.md first, it says what this showed
and why it was shelved.

Pass 1 asks the model for several whole scene setups for a source (five
fields each); the code picks one with the seed; pass 2 is the normal
Enhance call with the picked setup added to the user message.

    python -m experiments.variety_two_pass /tmp/out.jsonl
"""
import json, random, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments._pe_bootstrap import pe
MODEL = "huihui_ai/qwen3.5-abliterated:9b"
FIELDS = ("people", "doing", "wearing", "place", "light")
LIST_SP = """You prepare scene setups for an image prompt. You get a short source prompt.
Write 6 setups that differ from each other in the place, in what the people do, and in the people themselves.
Each setup has five short fields, a few plain words each:
- people: every person the source names, each with an age as a number and two visible traits
- doing: the specific act or activity and the position, named directly
- wearing: what each person wears
- place: where it is
- light: the time and the light
Rules: stay inside everything the source states (people, counts, ages or age ranges, places); situations that can physically happen; every person is an adult; plain direct words, no poetic language. If the source is sexual, name the sexual act in plain words.
A field the source already settles repeats the source's wording.
Answer with JSON only: {"setups": [{"people": "", "doing": "", "wearing": "", "place": "", "light": ""}]}"""
def list_setups(source, seed):
    body = {"model": MODEL, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0.8, "seed": seed, "top_k": 40, "top_p": 0.95, "repeat_penalty": 1.1, "num_predict": 900},
            "messages": [{"role": "system", "content": LIST_SP}, {"role": "user", "content": f"/no_think\nSOURCE PROMPT: {source}"}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(json.loads(r.read())["message"]["content"])
    out = []
    for x in d.get("setups") or []:
        if isinstance(x, dict) and all(str(x.get(f, "")).strip() for f in FIELDS):
            out.append({f: str(x[f]).strip() for f in FIELDS})
    return out
out = open(sys.argv[1], "a")
sp = pe._assemble_system_prompt("Default")
SOURCES = (("var_sex", "a man and a woman having sex"), ("sex_outdoors", "a man and a woman having sex, outdoors"),
           ("woman40", "middle-aged woman gardening"), ("twenties", "woman in her twenties"),
           ("var_dance", "a man and a woman dancing"), ("mall_short", "woman, 25, standing in a crowded mall"))
for name, source in SOURCES:
    for seed in (42, 137, 1729, 7919, 10001):
        setups = list_setups(source, seed)
        if not setups:
            out.write(json.dumps({"source": name, "seed": seed, "setups": [], "error": "no complete setup"}) + "\n"); out.flush(); continue
        pick = random.Random(seed).choice(setups)
        setup_txt = "; ".join(f"{f}: {pick[f]}" for f in FIELDS)
        user = f"SOURCE PROMPT: {source}\n\nSetup for what the source leaves open (it is part of the scene, use it as given): {setup_txt}"
        text = pe._clean_output(pe._call_llm(user, "http://127.0.0.1:11434", MODEL, sp, 0.8, think=False, seed=seed))
        out.write(json.dumps({"source": name, "seed": seed, "setups": setups, "pick": pick, "out": text}, ensure_ascii=False) + "\n"); out.flush()
print("DONE")
