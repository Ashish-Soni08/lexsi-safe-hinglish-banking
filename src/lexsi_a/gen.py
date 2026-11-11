"""Generate answers for every eval row with vLLM.

  python -m lexsi_a.gen --model CohereLabs/tiny-aya-fire --tag fire --out results/gens
  python -m lexsi_a.gen --model CohereLabs/tiny-aya-fire --lora outputs/sft_lora --tag sft --out results/gens
Writes <out>/<tag>.jsonl with the eval row plus 'answer'.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .common import SYSTEM_PROMPT
from .evalsets import load_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--out", default="results/gens")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-system", action="store_true", help="omit the banking system prompt")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_all(args.suites)
    llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enable_lora=bool(args.lora), max_lora_rank=64)
    sys_msg = [] if args.no_system else [{"role": "system", "content": SYSTEM_PROMPT}]
    convs = [sys_msg + [{"role": "user", "content": r["prompt"]}] for r in rows]
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    outs = llm.chat(convs, SamplingParams(temperature=0, max_tokens=args.max_tokens), lora_request=lora, use_tqdm=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}.jsonl"
    with path.open("w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({**r, "model": args.tag, "answer": o.outputs[0].text}, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "->", path)


if __name__ == "__main__":
    main()
