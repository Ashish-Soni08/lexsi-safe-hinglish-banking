"""Safety mix v3: swap most templated refusals for natural ones and add many more 'sounds risky but is fine'
examples with helpful answers, in English, Hindi and Hinglish.

Sources
  * v2 mix (outputs/harden/safety_mix.jsonl from run harden2): keeps banking `fraud` refusals, `victim`
    and `benign_edgy` rows; drops the templated `aya_redteam` refusals.
  * ToxicityPrompts/PolyGuardMix (CC-BY-4.0; WildGuardMix translated, labels for prompt harm / response
    refusal / response harm), English + Hindi:
      - `pg_refuse`: harmful prompt, response refuses and is not harmful
      - `pg_comply`: harmless prompt (many of them adversarial-looking), response complies and is not harmful
  * safety-aya/Nemotron-Safety-Guard-Dataset-v3-hindi: `nemo_comply`, safe Hindi prompt with a safe answer.
  * Hinglish: a slice of the English PolyGuardMix rows translated with Aya Expanse 8B (`*_hinglish`).

Any candidate sharing >= 50% of its word 4-grams with an eval prompt (all suites) is dropped.

  python -m lexsi_a.harden_data_v3 --v2 /tmp/prev/harden2/outputs/harden/safety_mix.jsonl --out outputs/harden
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from collections import Counter

from .evalsets import load_all
from .translate import STYLE, _query_ok, _query_prompt, script_ok


def grams(text: str, n: int = 4) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", default="outputs/harden")
    ap.add_argument("--n-refuse", type=int, default=250, help="PolyGuardMix refusals per language (en, hi)")
    ap.add_argument("--n-comply", type=int, default=700, help="PolyGuardMix compliant rows per language (en, hi)")
    ap.add_argument("--n-nemo", type=int, default=300)
    ap.add_argument("--n-hinglish", type=int, default=400, help="English rows to translate to Hinglish")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rng = random.Random(0)
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

    def ok(p: str, r: str) -> bool:
        return 10 <= len(p) <= 1500 and 10 <= len(r) <= 2500 and not contaminated(p)

    v2 = [json.loads(l) for l in open(args.v2) if l.strip()]
    mix = [r for r in v2 if r["kind"] != "aya_redteam"]
    print("kept from v2", Counter(r["kind"] for r in mix))

    pg = load_dataset("ToxicityPrompts/PolyGuardMix", split="train")
    pg = pg.filter(lambda r: r["metadata"]["language"] in ("English", "Hindi"), num_proc=4)
    buckets: dict = {}
    for r in pg:
        lang = "en" if r["metadata"]["language"] == "English" else "hi"
        if r["prompt_harm_label"] == "yes" and r["response_refusal_label"] == "yes" and r["response_harm_label"] == "no":
            kind = "pg_refuse"
        elif r["prompt_harm_label"] == "no" and r["response_refusal_label"] == "no" and r["response_harm_label"] == "no":
            kind = "pg_comply"
        else:
            continue
        p, a = (r["prompt"] or "").strip(), (r["response"] or "").strip()
        if lang == "hi" and not (script_ok("hi", p) and script_ok("hi", a)):
            continue
        buckets.setdefault((kind, lang), []).append((p, a))
    print("PolyGuardMix pools", {k: len(v) for k, v in buckets.items()})
    english_pool = {}
    for (kind, lang), rows in buckets.items():
        rng.shuffle(rows)
        n = args.n_refuse if kind == "pg_refuse" else args.n_comply
        picked = [(p, a) for p, a in rows if ok(p, a)][: n + (args.n_hinglish if lang == "en" else 0)]
        mix += [{"lang": lang, "kind": kind, "instruction": p, "response": a} for p, a in picked[:n]]
        if lang == "en":
            english_pool[kind] = picked[n:]

    nemo = load_dataset("safety-aya/Nemotron-Safety-Guard-Dataset-v3-hindi", split="train")
    nemo = [r for r in nemo if r["prompt_label"] == "safe" and r["response_label"] == "safe" and r["response"]]
    rng.shuffle(nemo)
    nemo = [r for r in nemo if ok(r["prompt"], r["response"]) and script_ok("hi", r["response"])][: args.n_nemo]
    mix += [{"lang": "hi", "kind": "nemo_comply", "instruction": r["prompt"], "response": r["response"]} for r in nemo]

    # Hinglish: translate English rows (prompt with the fenced query prompt, answer with the normal style prompt).
    n_ref = args.n_hinglish // 3
    to_tr = [("pg_refuse", p, a) for p, a in english_pool.get("pg_refuse", [])[:n_ref]] + \
            [("pg_comply", p, a) for p, a in english_pool.get("pg_comply", [])[: args.n_hinglish - n_ref]]
    llm = LLM("CohereLabs/aya-expanse-8b", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.2, max_tokens=1024)
    resp_conv = lambda t: [{"role": "system", "content": STYLE["hinglish"] + " Output only the translation, nothing else."},
                           {"role": "user", "content": t}]
    qs = llm.chat([_query_prompt("hinglish", p) for _, p, _ in to_tr], sp, use_tqdm=True)
    rs = llm.chat([resp_conv(a) for _, _, a in to_tr], sp, use_tqdm=True)
    n_h = 0
    for (kind, p, a), q, r in zip(to_tr, qs, rs):
        q, r = q.outputs[0].text.strip().strip('"'), r.outputs[0].text.strip()
        if _query_ok("hinglish", q, p) and script_ok("hinglish", r):
            mix.append({"lang": "hinglish", "kind": kind + "_hinglish", "instruction": q, "response": r})
            n_h += 1
    print("hinglish translated", n_h, "of", len(to_tr))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "safety_mix.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mix) + "\n")
    print("wrote", len(mix), Counter((r["kind"], r["lang"]) for r in mix))
    for r in rng.sample(mix, 10):
        print("SAMPLE", r["kind"], r["lang"], "|", r["instruction"][:110].replace("\n", " "), "->", r["response"][:130].replace("\n", " "))


if __name__ == "__main__":
    main()
