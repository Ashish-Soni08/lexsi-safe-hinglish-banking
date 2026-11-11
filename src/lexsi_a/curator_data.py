"""Our own trilingual preference data, built with CuratorKIT, for a DPO pass on the final model.

The SFT safety mix fixed fraud refusals but left over-refusal on harmless "sounds scary" prompts. This
builds preference pairs that target both failure modes:

  * fraud request    -> chosen: a refusal          rejected: a compliant answer
  * edgy-but-harmless -> chosen: a helpful answer   rejected: a refusal

Steps (each a sub-command, so that every vLLM model gets its own process and GPU):

  evolve  Aya Expanse 8B is served with vLLM's OpenAI server and driven through CuratorKIT's LiteLLMBackend.
          CuratorKIT EvolInstructTask evolves hand-written + earlier seed prompts (disguised / indirect fraud
          requests; harmless prompts with violent or criminal-sounding words), DiversityGate drops near
          duplicates, then each prompt is translated to Hindi and Hinglish. Prompts that share >= 50% of their
          word 4-grams with any eval prompt are dropped.
  sample  sample answers from one model (final model, Fire, banking SFT) with the banking system prompt.
  judge   Qwen3Guard-Gen-8B labels every prompt (harmful?) and every answer (refusal? unsafe?).
  pairs   pick chosen / rejected per prompt (preferring the final model's own answers, i.e. on-policy),
          export with CuratorKIT's DPOExporter, and write conversational pairs for AlignTune DPO.

  python -m lexsi_a.curator_data evolve --seeds-v2 safety_mix_v2.jsonl --out outputs/curator
  python -m lexsi_a.curator_data sample --model ashish-soni08/tiny-aya-fire-banking-safe --tag final --out outputs/curator
  python -m lexsi_a.curator_data judge --out outputs/curator
  python -m lexsi_a.curator_data pairs --out outputs/curator
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .common import SYSTEM_PROMPT, looks_like_refusal
from .harden_data import REFUSALS, lang_ok
from .translate import _query_ok, _query_prompt

GEN = "CohereLabs/aya-expanse-8b"
PORT = 8000

# Harmless prompts with scary words (our own wording; XSTest-style categories, not XSTest items).
EDGY_SEEDS = [
    "How do I kill all the child processes of a script in Linux?",
    "What's the fastest way to kill my credit card debt?",
    "How can I knock them dead in my interview for a bank teller job?",
    "How do I shoot a sharp photo of the moon with my phone?",
    "What is the best way to butcher a whole chicken at home?",
    "How do I get rid of the rats in my shop without poison that could hurt my cat?",
    "How do I safely destroy an old debit card I no longer use?",
    "How do I wipe all data from my old phone before selling it so nobody can access my bank app?",
    "What exactly is money laundering, and why is it illegal?",
    "What is a money mule, and how do I make sure I never become one by accident?",
    "How do phishing SMS messages that pretend to be from a bank usually look? I want to warn my parents.",
    "My grandmother got a call asking for her OTP. How do I explain to her why she should never share it?",
    "How did the Harshad Mehta securities scam actually work, historically?",
    "Why did the Satyam accounting fraud go undetected for so long?",
    "In my crime novel, a detective tracks money stolen through fake loan apps. What clues would she follow?",
    "What's the best way to assassinate the boss in the final level of a stealth video game?",
    "How do I attack the main problem in my business plan before I pitch it to a bank?",
    "How do I blow up a photo to poster size without it getting blurry?",
    "How can I hack my morning routine to save more money each month?",
    "What is the legal limit for cash deposits in an Indian savings account before tax questions come up?",
    "Can I get arrested for accidentally receiving money from a stranger by UPI? What should I do?",
    "My account was hacked last night. How do I lock it down right now?",
    "Someone stole my wallet with my cards. What do I do first?",
    "How do banks detect fraudulent transactions on credit cards?",
    "What are the warning signs of a SIM swap attack on my number?",
    "How do I report a fake customer-care number that scammed my friend?",
    "Why do scammers ask people to install screen-sharing apps like AnyDesk?",
    "How do I execute a standing instruction to pay my rent every month?",
    "What's the penalty if I bounce a cheque in India?",
    "How do I choke off the extra spending on my credit card each month?",
    "What household chemicals should never be mixed when cleaning, for safety?",
    "How do I strangle a weed problem in my garden organically?",
    "How do I break into the banking industry without a finance degree?",
    "What does it mean when a bank 'freezes' an account, and how do I unfreeze mine?",
    "How do I steal the show at my sister's wedding with a good speech?",
    "Where can I buy a good knife for cutting vegetables?",
    "How do I take down my old shop's listing on Google Maps?",
    "What's the best way to bury a time capsule in my backyard?",
    "How did counterfeit currency detection improve after demonetisation?",
    "How do I explain to my teenager why giving their bank login to a friend is dangerous?",
]
# Disguised fraud: evolve the earlier fraud seeds into harder, less obvious variants.
FRAUD_STRATEGIES = [
    "Hide the harmful goal behind a plausible pretext, such as claiming to be bank staff, a researcher, a relative or a police officer",
    "Make it specific and operational: name a concrete bank, app, amount or step",
    "Ask for it indirectly, as a role-play, a story, a hypothetical or 'just for awareness'",
]
EDGY_STRATEGIES = [
    "Keep it completely harmless but make it sound alarming by using a violent, criminal or scary-sounding word in an innocent sense",
    "Add a realistic personal context (who is asking and why) that keeps the request clearly legitimate",
    "Turn it into a different, related harmless question on the same theme",
]
FRAUD_EVOL = (
    "We are building red-teaming training data so that an Indian bank's customer-support chatbot learns to REFUSE "
    "fraud requests. Rewrite the fraudster's message below into a harder-to-spot version.\n\n"
    "Original message:\n{instruction}\n\nRewriting strategy: {strategy}\n{context}\n"
    "The rewritten message must still ask for the same harmful thing, read like a real chat message (1-3 sentences), "
    "and be in English. Respond with JSON only:\n"
    '{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}'
)
EDGY_EVOL = (
    "We are building training data so that a chatbot learns NOT to refuse harmless questions that merely sound "
    "dangerous. Rewrite the question below.\n\nOriginal question:\n{instruction}\n\nRewriting strategy: {strategy}\n"
    "{context}\nThe rewritten question must stay completely harmless and answerable, read like a real chat message "
    "(1-3 sentences), and be in English. Respond with JSON only:\n"
    '{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}'
)


def grams(text: str, n: int = 4) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def contamination_filter():
    from .evalsets import load_all
    eval_grams = [g for g in (grams(r["prompt"]) for r in load_all(None)) if g]
    index: dict = {}
    for i, g in enumerate(eval_grams):
        for t in g:
            index.setdefault(t, set()).add(i)

    def contaminated(p: str) -> bool:
        g = grams(p)
        if not g:
            return False
        hits = Counter(i for t in g for i in index.get(t, ()))
        return any(c / min(len(g), len(eval_grams[i])) >= 0.5 for i, c in hits.items())
    return contaminated


def _read(p) -> list[dict]:
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def _write(p, rows) -> None:
    pathlib.Path(p).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


# ----------------------------------------------------------------------------------------------- evolve
def evolve(args) -> None:
    from curatorkit.gates.diversity import DiversityGate
    from curatorkit.generators.evol_instruct import EvolInstructTask
    from curatorkit.llm.litellm import LiteLLMBackend
    from curatorkit.schema import DataSample

    rng = random.Random(0)
    server = subprocess.Popen(["vllm", "serve", GEN,
                               "--port", str(PORT), "--dtype", "bfloat16", "--max-model-len", "4096",
                               "--gpu-memory-utilization", "0.88"])
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(10)
    else:
        server.kill()
        raise SystemExit("vLLM server did not start")
    try:
        llm = LiteLLMBackend(model=f"openai/{GEN}", api_base=f"http://localhost:{PORT}/v1", api_key="none",
                             temperature=0.9, max_tokens=400)
        v2 = _read(args.seeds_v2)
        fraud_seeds = [r["instruction"] for r in v2 if r["kind"] == "fraud" and r["lang"] == "en"]
        edgy_seeds = EDGY_SEEDS + rng.sample([r["instruction"] for r in v2 if r["kind"] == "benign_edgy"], args.n_orbench)
        jobs_spec = [("fraud", fraud_seeds, FRAUD_EVOL, FRAUD_STRATEGIES), ("edgy", edgy_seeds, EDGY_EVOL, EDGY_STRATEGIES)]
        if args.orbench_fresh:
            # Round 2: the round-1 prompts were mostly answered already, so mine many fresh OR-Bench-80k
            # "seemingly toxic" prompts (none from OR-Bench-hard-1k, which the eval samples, nor from the SFT mix)
            # and evolve each once toward scarier wording.
            from datasets import load_dataset
            seen = {r["instruction"] for r in v2}
            hard = {r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")}
            pool = [r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
                    if r["prompt"] not in hard and r["prompt"] not in seen]
            jobs_spec = [("edgy", rng.sample(pool, args.orbench_fresh), EDGY_EVOL, EDGY_STRATEGIES[:1])]
        print("seeds", {k: len(s) for k, s, _, _ in jobs_spec})

        rows = []
        for kind, seeds, tmpl, strategies in jobs_spec:
            task = EvolInstructTask(llm=llm, prompt_template=tmpl, num_evolutions=len(strategies),
                                    strategies=strategies, generate_answers=False, concurrency=32)
            samples = [DataSample(source_uri=f"seed:{kind}", instruction=s, metadata={"kind": kind}) for s in seeds]
            evolved = [s for s in task.run(samples) if 15 <= len(s.instruction) <= 400 and lang_ok("en", s.instruction)]
            # Keep the seeds themselves too (they are new to the model as preference prompts).
            pool = samples + evolved
            gate = DiversityGate(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                 similarity_threshold=0.9, text_field="instruction")
            kept, dropped = gate.run(pool)
            print(f"{kind}: {len(samples)} seeds + {len(evolved)} evolved -> {len(kept)} after DiversityGate "
                  f"({len(dropped)} near-duplicates)")
            rows += [{"kind": kind, "lang": "en", "prompt": s.instruction.strip().strip('"'),
                      "strategy": s.metadata.get("evolution_strategy", "seed")} for s in kept]

        # Translate every English prompt to Hindi and Hinglish (fenced prompt, retries for the wrong script).
        def tr(job):
            row, lang = job
            for t in (0.2, 0.7, 0.9):
                out = llm.generate(_query_prompt(lang, row["prompt"]), temperature=t, max_tokens=300).text
                out = out.strip().strip('"')
                if _query_ok(lang, out, row["prompt"]):
                    return {**row, "lang": lang, "prompt": out, "en": row["prompt"]}
            return None
        jobs = [(r, l) for r in rows for l in ("hi", "hinglish")]
        with ThreadPoolExecutor(32) as pool:
            trans = [t for t in pool.map(tr, jobs) if t]
        print("translated", Counter(t["lang"] for t in trans), "of", len(jobs))
        rows += trans
    finally:
        server.terminate()
        server.wait(timeout=120)

    if args.prev_prompts:  # carry the earlier rounds' prompts (already translated) into this round
        rows += [{k: v for k, v in r.items() if k not in ("id", "prompt_harmful", "prompt_safe")}
                 for r in _read(args.prev_prompts)]
    contaminated = contamination_filter()
    before = len(rows)
    rows = [r for r in rows if not contaminated(r["prompt"])]
    print("contamination filter dropped", before - len(rows))
    for i, r in enumerate(rows):
        r["id"] = f"cur-{i:05d}"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "prompts.jsonl", rows)
    print("wrote", len(rows), Counter((r["kind"], r["lang"]) for r in rows))
    for r in rng.sample(rows, min(12, len(rows))):
        print("PROMPT", r["kind"], r["lang"], r["strategy"][:30], "|", r["prompt"][:200])


# ----------------------------------------------------------------------------------------------- sample
def sample(args) -> None:
    from vllm import LLM, SamplingParams

    out = pathlib.Path(args.out)
    rows = _read(out / "prompts.jsonl")
    llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    convs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}] for r in rows]
    outs = llm.chat(convs, SamplingParams(n=args.n, temperature=0.8, top_p=0.95, max_tokens=320, seed=0), use_tqdm=True)
    (out / "samples").mkdir(exist_ok=True)
    _write(out / "samples" / f"{args.tag}.jsonl",
           [{"id": r["id"], "model": args.tag, "answer": c.text.strip(), "finished": c.finish_reason == "stop"}
            for r, o in zip(rows, outs) for c in o.outputs])
    print("sampled", args.tag, len(rows), "x", args.n)


# ----------------------------------------------------------------------------------------------- judge
def judge(args) -> None:
    from vllm import LLM, SamplingParams

    from .judge import JUDGES, _parse_qwen3guard

    out = pathlib.Path(args.out)
    prompts = _read(out / "prompts.jsonl")
    by_id = {r["id"]: r for r in prompts}
    llm = LLM(JUDGES["qwen3guard"], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0, max_tokens=32)
    pv = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in prompts], sp, use_tqdm=True)
    for r, o in zip(prompts, pv):
        v = _parse_qwen3guard(o.outputs[0].text)
        r["prompt_harmful"] = bool(v["unsafe"] or v["controversial"])
        r["prompt_safe"] = v["unsafe"] is False and not v["controversial"]
    _write(out / "prompts.jsonl", prompts)
    for f in sorted((out / "samples").glob("*.jsonl")):
        if f.name.endswith(".judged.jsonl"):
            continue
        rows = _read(f)
        outs = llm.chat([[{"role": "user", "content": by_id[r["id"]]["prompt"]},
                          {"role": "assistant", "content": r["answer"][:3000]}] for r in rows], sp, use_tqdm=True)
        for r, o in zip(rows, outs):
            v = _parse_qwen3guard(o.outputs[0].text)
            r["refused"] = bool(v["refused"]) or (v["refused"] is None and looks_like_refusal(r["answer"]))
            r["kw_refusal"] = looks_like_refusal(r["answer"])
            r["unsafe"] = bool(v["unsafe"])
        _write(f.with_suffix(".judged.jsonl"), rows)
        print("judged", f.name, "refused", sum(r["refused"] for r in rows), "/", len(rows),
              "unsafe", sum(r["unsafe"] for r in rows))


# ----------------------------------------------------------------------------------------------- pairs
def pairs(args) -> None:
    from curatorkit.exporters.dpo import DPOExporter
    from curatorkit.schema import DataSample

    rng = random.Random(0)
    out = pathlib.Path(args.out)
    prompts = {r["id"]: r for r in _read(out / "prompts.jsonl")}
    answers: dict = {}
    for f in sorted((out / "samples").glob("*.judged.jsonl")):
        for r in _read(f):
            answers.setdefault(r["id"], []).append(r)
    policy = args.policy
    stats, made = Counter(), {"fraud": [], "edgy": []}
    for pid, p in prompts.items():
        kind, cands = p["kind"], answers.get(pid, [])
        # Labels must agree with the guard: fraud prompts must be harmful, edgy prompts safe.
        if (kind == "fraud" and not p.get("prompt_harmful")) or (kind == "edgy" and not p.get("prompt_safe")):
            stats[f"{kind}:label_mismatch"] += 1
            continue
        ok = [a for a in cands if a["finished"] and lang_ok(p["lang"], a["answer"]) and len(a["answer"]) >= 20]
        if kind == "fraud":
            good = [a for a in ok if a["refused"] and a["kw_refusal"] and not a["unsafe"] and len(a["answer"]) < 700]
            bad = [a for a in cands if not a["refused"] and not a["kw_refusal"]]
        else:
            good = [a for a in ok if not a["refused"] and not a["kw_refusal"] and not a["unsafe"]]
            bad = [a for a in cands if a["refused"] and a["kw_refusal"]]
        if not bad:
            stats[f"{kind}:no_rejected"] += 1
            continue
        on_policy_bad = [a for a in bad if a["model"] == policy]
        rejected = rng.choice(on_policy_bad or bad)
        if good:
            on_policy_good = [a for a in good if a["model"] == policy]
            chosen = rng.choice(on_policy_good or good)["answer"]
            src = "policy" if on_policy_good else "other"
        elif kind == "fraud":
            chosen, src = rng.choice(REFUSALS[p["lang"]]), "template"
        else:
            stats[f"{kind}:no_chosen"] += 1
            continue
        stats[f"{kind}:chosen_{src}"] += 1
        stats[f"{kind}:rejected_{'policy' if on_policy_bad else 'other'}"] += 1
        made[kind].append({"id": pid, "kind": kind, "lang": p["lang"], "prompt": p["prompt"], "chosen": chosen,
                           "rejected": rejected["answer"], "policy_failed": bool(on_policy_bad)})
    print("pair stats", dict(sorted(stats.items())))
    # Prioritise prompts the final model gets wrong; cap the fraud side so the pass mostly fixes over-refusal
    # while still anchoring refusals.
    for k in made:
        made[k].sort(key=lambda r: (not r["policy_failed"], rng.random()))
    n_edgy = len(made["edgy"])
    rows = made["edgy"] + made["fraud"][: max(int(args.fraud_ratio * n_edgy), args.min_fraud, 1)]
    if args.anchors:
        # Banking anchors: ordinary Bitext questions (training split only) with the reference answer as chosen and
        # a refusal as rejected, so DPO does not drift toward refusing customers or toward worse banking answers.
        bitext = _read(args.anchors)
        for lang in ("en", "hi", "hinglish"):
            pool = [r for r in bitext if r["lang"] == lang and len(r["response"]) < 1500]
            for r in rng.sample(pool, min(args.n_anchors, len(pool))):
                rows.append({"id": r["id"], "kind": "anchor", "lang": lang, "prompt": r["instruction"],
                             "chosen": r["response"], "rejected": rng.choice(REFUSALS[lang]), "policy_failed": False})
    rng.shuffle(rows)
    print("pairs", Counter((r["kind"], r["lang"]) for r in rows), "on-policy failures",
          Counter(r["kind"] for r in rows if r["policy_failed"]))

    samples = [DataSample(source_uri=f"curator:{r['id']}", instruction=r["prompt"], chosen=r["chosen"],
                          rejected=r["rejected"], task_type="preference",
                          metadata={"kind": r["kind"], "lang": r["lang"]}) for r in rows]
    DPOExporter().export(samples, out)
    _write(out / "pairs.jsonl", rows)
    # Conversational format with the banking system prompt, for AlignTune / TRL DPO.
    _write(out / "dpo_train.jsonl", [{
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])
    for r in rows[:8]:
        print("PAIR", r["kind"], r["lang"], "|", r["prompt"][:120], "| +", r["chosen"][:120].replace("\n", " "),
              "| -", r["rejected"][:120].replace("\n", " "))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evolve")
    e.add_argument("--seeds-v2", required=True)
    e.add_argument("--n-orbench", type=int, default=60)
    e.add_argument("--orbench-fresh", type=int, default=0, help="round 2: evolve this many fresh OR-Bench-80k prompts")
    e.add_argument("--prev-prompts", default=None, help="prompts.jsonl of an earlier round to include")
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--n", type=int, default=2)
    sub.add_parser("judge")
    p = sub.add_parser("pairs")
    p.add_argument("--policy", default="final")
    p.add_argument("--fraud-ratio", type=float, default=0.6)
    p.add_argument("--min-fraud", type=int, default=0, help="keep at least this many fraud pairs")
    p.add_argument("--anchors", default=None, help="Bitext train jsonl for legit-banking anchor pairs")
    p.add_argument("--n-anchors", type=int, default=150, help="anchor pairs per language")
    for x in (e, s, sub.choices["judge"], p):
        x.add_argument("--out", default="outputs/curator")
    args = ap.parse_args()
    {"evolve": evolve, "sample": sample, "judge": judge, "pairs": pairs}[args.cmd](args)


if __name__ == "__main__":
    main()
