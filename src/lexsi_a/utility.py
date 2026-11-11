"""Banking task utility: answer held-out Bitext questions (en / hi / hinglish) and score
against the reference answer with chrF++, plus the share of answers that are not refusals.

  python -m lexsi_a.utility --model CohereLabs/tiny-aya-fire --tag fire --test outputs/sft/test_trilingual.jsonl
Writes results/utility/<tag>.jsonl and results/utility/<tag>.summary.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

from .common import SYSTEM_PROMPT, looks_like_refusal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--out", default="results/utility")
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()

    import sacrebleu
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in pathlib.Path(args.test).read_text().splitlines() if l.strip()]
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    rows = [r for lang in sorted(by_lang) for r in by_lang[lang][: args.per_lang]]

    llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    convs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["instruction"]}] for r in rows]
    outs = llm.chat(convs, SamplingParams(temperature=0, max_tokens=args.max_tokens), use_tqdm=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored = []
    for r, o in zip(rows, outs):
        ans = o.outputs[0].text
        chrf = sacrebleu.sentence_chrf(ans, [r["response"]], word_order=2).score
        scored.append({**r, "model": args.tag, "answer": ans, "chrf": round(chrf, 2),
                       "refused": bool(looks_like_refusal(ans))})
    (out / f"{args.tag}.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in scored) + "\n")

    summary = {}
    for lang in sorted(by_lang) + ["all"]:
        rs = [s for s in scored if lang == "all" or s["lang"] == lang]
        if rs:
            summary[lang] = {"n": len(rs), "chrf": round(sum(s["chrf"] for s in rs) / len(rs), 2),
                             "refusal_rate": round(sum(s["refused"] for s in rs) / len(rs), 4)}
    (out / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print("utility", args.tag, json.dumps(summary))


if __name__ == "__main__":
    main()
