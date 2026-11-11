"""Check the labels of the generated safety mix with Qwen3Guard-Gen-8B (prompt moderation).

Aya Expanse sometimes writes a harmless or victim-style message when asked for a fraud request (and the
reverse), which would teach the model to refuse legitimate customers. Keep a refusal example only if
Qwen3Guard rates the request Unsafe or Controversial, and a helpful example only if it rates it Safe.

  python -m lexsi_a.harden_filter outputs/harden/safety_mix.jsonl
Rewrites the file in place and writes <file>.dropped.jsonl.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

from .judge import JUDGES, _parse_qwen3guard


def main() -> None:
    from vllm import LLM, SamplingParams

    path = pathlib.Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    llm = LLM(JUDGES["qwen3guard"], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    outs = llm.chat([[{"role": "user", "content": r["instruction"]}] for r in rows],
                    SamplingParams(temperature=0, max_tokens=32), use_tqdm=True)
    keep, drop = [], []
    for r, o in zip(rows, outs):
        v = _parse_qwen3guard(o.outputs[0].text)
        harmful = bool(v["unsafe"] or v["controversial"])
        ok = harmful if r["kind"] in ("fraud", "aya_redteam") else (v["unsafe"] is False and not v["controversial"])
        (keep if ok else drop).append({**r, "guard": o.outputs[0].text.strip()[:80]})
    path.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "guard"}, ensure_ascii=False) for r in keep) + "\n")
    path.with_suffix(".dropped.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in drop) + "\n")
    print("kept", Counter((r["kind"], r["lang"]) for r in keep))
    print("dropped", Counter((r["kind"], r["lang"]) for r in drop))
    for r in drop[:6]:
        print("DROPPED", r["kind"], r["lang"], "|", r["instruction"][:140], "|", r["guard"])


if __name__ == "__main__":
    main()
