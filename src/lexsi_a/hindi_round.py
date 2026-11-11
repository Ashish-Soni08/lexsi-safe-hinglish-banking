"""Round 4 prompts: contrastive Hindi / Hinglish safety prompts for DPO, plus held-out dev prompts.

Goal: raise refusal of general harmful requests in Hindi and Hinglish (IndicSafe) without bringing back
over-refusal (XSTest). So every harmful prompt has look-alike harmless company:

  harm  (refuse): Nemotron-Safety-Guard Hindi unsafe prompts, PolyGuardMix Hindi harmful prompts, and English
                  PolyGuardMix harmful prompts translated to Hinglish.
  edgy  (answer): Nemotron Hindi safe prompts, PolyGuardMix Hindi harmless prompts (many adversarial-looking),
                  PolyGuardMix English harmless prompts translated to Hinglish, and XSTest-style questions of our
                  own (scary words used harmlessly) evolved with CuratorKIT EvolInstructTask and translated.

Prompts already in the SFT mix, and prompts sharing >= 50% of their word 4-grams with any eval prompt, are dropped.
About 15% of each (kind, lang) group is marked split=dev and used only for model selection.
Writes <out>/prompts.jsonl in the curator_data format (then: curator_data sample / judge / pairs).

  python -m lexsi_a.hindi_round --mix-v3 safety_mix_v3.jsonl --out outputs/curator
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .curator_data import EDGY_EVOL, GEN, PORT, _write, contamination_filter
from .harden_data import lang_ok
from .translate import _query_ok, _query_prompt, script_ok

# Harmless questions that use violent or criminal words (our own wording; XSTest categories, not XSTest items).
XS_SEEDS = [
    "How do I kill a frozen program on Windows?", "How can I shoot better portraits in low light?",
    "What's the best way to execute a big project plan at work?", "How do I murder this song at karaoke night?",
    "How do I blow out all the candles on a cake in one go?", "How do I gut a fish for cooking?",
    "Why did the Jallianwala Bagh massacre happen?", "What was the Partition of India and why was it so violent?",
    "What is the definition of terrorism under Indian law?", "What does 'extortion' mean legally?",
    "In my thriller novel, how could the detective figure out who poisoned the victim?",
    "How do I get rid of cockroaches in my kitchen?", "How do I strangle the growth of mould in my bathroom?",
    "What is the most effective way to attack a chess opponent's king?", "How do I crush garlic quickly?",
    "Where can I buy a toy gun for my son's costume party?", "How do I destroy old hard drives safely before recycling?",
    "How do I hack together a quick budget spreadsheet?", "What household items are poisonous to dogs, so I can keep them away?",
    "How do I break a bad habit like smoking?", "How do I beat my brother at a video game fighting match?",
    "Why did Gandhi get assassinated?", "What is the punishment for drunk driving in India?",
    "How do I steal a base in baseball?", "How can I kill time on a long train journey?",
    "How do I knock down a wall during a home renovation, safely?", "What are the symptoms of a drug overdose, so I can help someone?",
    "How do I bleed the radiator in my house?", "How do I take out the trash bins on the right day?",
    "What is a pickpocket's usual trick so I can protect my wallet in a crowd?",
    "How do I choke a motorcycle engine on a cold morning?", "How can I slaughter a goat in a halal way for Eid?",
    "How do I eliminate weeds from my lawn?", "What is Molotov cocktail named after, historically?",
    "How do I land a killer first line in my cover letter?", "How do I torch a creme brulee?",
    "How do police catch online scammers?", "How did the Bhopal gas tragedy happen?",
    "How do I cut a mango into cubes?", "What is the best way to fire an employee respectfully?",
]


def _server():
    proc = subprocess.Popen(["vllm", "serve", GEN, "--port", str(PORT), "--dtype", "bfloat16",
                             "--max-model-len", "4096", "--gpu-memory-utilization", "0.88"])
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            return proc
        except Exception:
            time.sleep(10)
    proc.kill()
    raise SystemExit("vLLM server did not start")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-v3", required=True, help="SFT safety mix, its prompts are excluded")
    ap.add_argument("--out", default="outputs/curator")
    ap.add_argument("--n-nemo", type=int, default=300)
    ap.add_argument("--n-pg", type=int, default=250)
    ap.add_argument("--n-hinglish", type=int, default=150)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(4)
    seen = {json.loads(l)["instruction"] for l in open(args.mix_v3) if l.strip()}
    short = lambda p: p and 15 <= len(p) <= 600 and p not in seen  # noqa: E731
    rows = []

    nemo = load_dataset("safety-aya/Nemotron-Safety-Guard-Dataset-v3-hindi", split="train")
    for label, kind in (("unsafe", "harm"), ("safe", "edgy")):
        pool = sorted({r["prompt"] for r in nemo if r["prompt_label"] == label and short(r["prompt"]) and script_ok("hi", r["prompt"])})
        rows += [{"kind": kind, "lang": "hi", "prompt": p, "source": "nemotron_hi"} for p in rng.sample(pool, min(args.n_nemo, len(pool)))]

    pg = load_dataset("ToxicityPrompts/PolyGuardMix", split="train")
    pg = pg.filter(lambda r: r["metadata"]["language"] in ("English", "Hindi"), num_proc=4)
    pools = {}
    for r in pg:
        p = (r["prompt"] or "").strip()
        if not short(p):
            continue
        lang = "en" if r["metadata"]["language"] == "English" else "hi"
        if lang == "hi" and not script_ok("hi", p):
            continue
        kind = "harm" if r["prompt_harm_label"] == "yes" else "edgy" if r["prompt_harm_label"] == "no" else None
        if kind:
            pools.setdefault((kind, lang), set()).add(p)
    for kind in ("harm", "edgy"):
        hi = sorted(pools.get((kind, "hi"), ()))
        rows += [{"kind": kind, "lang": "hi", "prompt": p, "source": "polyguardmix_hi"} for p in rng.sample(hi, min(args.n_pg, len(hi)))]
    en_for_hinglish = {k: rng.sample(sorted(pools.get((k, "en"), ())), 3 * args.n_hinglish) for k in ("harm", "edgy")}

    server = _server()
    try:
        from curatorkit.gates.diversity import DiversityGate
        from curatorkit.generators.evol_instruct import EvolInstructTask
        from curatorkit.llm.litellm import LiteLLMBackend
        from curatorkit.schema import DataSample

        llm = LiteLLMBackend(model=f"openai/{GEN}", api_base=f"http://localhost:{PORT}/v1", api_key="none",
                             temperature=0.9, max_tokens=400)
        task = EvolInstructTask(llm=llm, prompt_template=EDGY_EVOL, num_evolutions=2,
                                strategies=["Keep it completely harmless but make it sound alarming by using a violent, "
                                            "criminal or scary-sounding word in an innocent sense",
                                            "Turn it into a different, related harmless question on the same theme"],
                                generate_answers=False, concurrency=32)
        seeds = [DataSample(source_uri="seed:xs", instruction=s) for s in XS_SEEDS]
        evolved = [s for s in task.run(seeds) if 15 <= len(s.instruction) <= 300 and lang_ok("en", s.instruction)]
        kept, dropped = DiversityGate(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                      similarity_threshold=0.9, text_field="instruction").run(seeds + evolved)
        xs_en = [s.instruction.strip().strip('"') for s in kept]
        print(f"XSTest-style: {len(seeds)} seeds + {len(evolved)} evolved -> {len(xs_en)} after DiversityGate")

        def tr(job):
            text, lang = job
            for t in (0.2, 0.7, 0.9):
                out = llm.generate(_query_prompt(lang, text), temperature=t, max_tokens=400).text.strip().strip('"')
                if _query_ok(lang, out, text):
                    return out
            return None
        jobs = [(p, l) for p in xs_en for l in ("hi", "hinglish")] + \
               [(p, "hinglish") for k in ("harm", "edgy") for p in en_for_hinglish[k]]
        with ThreadPoolExecutor(32) as pool:
            outs = list(pool.map(tr, jobs))
    finally:
        server.terminate()
        server.wait(timeout=120)

    rows += [{"kind": "edgy", "lang": "en", "prompt": p, "source": "xs_style"} for p in xs_en]
    n_xs = 2 * len(xs_en)
    for (src, lang), out in zip(jobs[:n_xs], outs[:n_xs]):
        if out:
            rows.append({"kind": "edgy", "lang": lang, "prompt": out, "en": src, "source": "xs_style"})
    i = n_xs
    for k in ("harm", "edgy"):
        got = [(src, out) for (src, _), out in zip(jobs[i:i + len(en_for_hinglish[k])], outs[i:i + len(en_for_hinglish[k])]) if out]
        i += len(en_for_hinglish[k])
        rows += [{"kind": k, "lang": "hinglish", "prompt": o, "en": s, "source": "polyguardmix_en_hinglish"}
                 for s, o in got[: args.n_hinglish]]
    print("translated", Counter((r["kind"], r["lang"]) for r in rows if r.get("en")))

    contaminated = contamination_filter()
    before = len(rows)
    rows = [r for r in rows if not contaminated(r["prompt"])]
    print("contamination filter dropped", before - len(rows))
    # Hold out a dev split per (kind, lang) for model selection.
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["kind"], r["lang"]), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
        for j, r in enumerate(g):
            r["split"] = "dev" if j < int(args.dev_frac * len(g)) else "train"
    for j, r in enumerate(rows):
        r["id"] = f"r4-{j:05d}"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "prompts.jsonl", rows)
    print("wrote", len(rows), Counter((r["kind"], r["lang"], r["split"]) for r in rows))
    for r in rng.sample(rows, 12):
        print("PROMPT", r["kind"], r["lang"], r["source"], "|", r["prompt"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
