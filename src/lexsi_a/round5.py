"""Round 5 prompts: mine what the round-3 final model gets wrong, with a dev split that can see over-refusal.

Round 4 picked its variant on a dev set whose harmless English prompts were all easy, so it could not see the
over-refusal it brought back (OR-Bench-hard 29% -> 58%). Round 5 fixes the dev set and the pair balance:

  edgy (answer): fresh OR-Bench-80k "seemingly toxic" prompts (never OR-Bench-hard-1k, which the test samples, and
                 never an earlier round's prompt), translated to Hindi and Hinglish, plus round 4's harmless prompts.
  harm (refuse): round 4's Hindi / Hinglish harmful prompts.
About 12% of the fresh prompts per language are split=dev; round 4's dev rows stay dev. The pairs step then keeps only
prompts the round-3 model gets wrong, at 1 harm : 1 edgy (see jobs/stage_dpo5.sh).

  python -m lexsi_a.round5 --r4 r4_prompts.jsonl --seen pairs_dpo3.jsonl --out outputs/curator
"""
from __future__ import annotations

import argparse
import pathlib
import random
from collections import Counter

from .curator_data import GEN, _read, _write, contamination_filter
from .translate import _query_ok, _query_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r4", required=True, help="round-4 prompts.jsonl (Hindi/Hinglish harm + edgy, with splits)")
    ap.add_argument("--seen", nargs="*", default=[], help="jsonl files whose prompts must not be reused")
    ap.add_argument("--out", default="outputs/curator")
    ap.add_argument("--n-en", type=int, default=2400, help="fresh English OR-Bench-80k prompts")
    ap.add_argument("--n-tr", type=int, default=700, help="of those, translated to each of hi and hinglish")
    ap.add_argument("--dev-frac", type=float, default=0.12)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rng = random.Random(5)
    r4 = _read(args.r4)
    seen = {r["prompt"] for r in r4}
    for f in args.seen:
        seen |= {r.get("prompt") or r.get("instruction") for r in _read(f)}
    hard = {r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")}
    pool = sorted({r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
                   if r["prompt"] not in hard and r["prompt"] not in seen and 15 <= len(r["prompt"]) <= 400})
    en = rng.sample(pool, args.n_en)
    rows = [{"kind": "edgy", "lang": "en", "prompt": p, "source": "orbench80k"} for p in en]

    llm = LLM(GEN, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    for lang in ("hi", "hinglish"):
        src = en[: args.n_tr] if lang == "hi" else en[args.n_tr: 2 * args.n_tr]
        todo, got = list(src), {}
        for t in (0.2, 0.7, 0.9):  # retry the wrong-script / answered-instead-of-translated ones
            outs = llm.chat([_query_prompt(lang, p) for p in todo], SamplingParams(temperature=t, max_tokens=300),
                            use_tqdm=False)
            for p, o in zip(todo, outs):
                text = o.outputs[0].text.strip().strip('"').strip()
                if _query_ok(lang, text, p):
                    got[p] = text
            todo = [p for p in todo if p not in got]
        rows += [{"kind": "edgy", "lang": lang, "prompt": got[p], "en": p, "source": "orbench80k"} for p in src if p in got]
        print("translated", lang, len(got), "of", len(src))

    contaminated = contamination_filter()
    before = len(rows)
    rows = [r for r in rows if not contaminated(r["prompt"])]
    print("contamination filter dropped", before - len(rows))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["lang"], []).append(r)
    for g in groups.values():
        rng.shuffle(g)
        for j, r in enumerate(g):
            r["split"] = "dev" if j < int(args.dev_frac * len(g)) else "train"
    rows += [{k: v for k, v in r.items() if k not in ("id", "prompt_harmful", "prompt_safe")} for r in r4]
    for j, r in enumerate(rows):
        r["id"] = f"r5-{j:05d}"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "prompts.jsonl", rows)
    print("wrote", len(rows), sorted(Counter((r["kind"], r["lang"], r["split"]) for r in rows).items()))
    for r in rng.sample(rows, 10):
        print("PROMPT", r["kind"], r["lang"], "|", r["prompt"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
