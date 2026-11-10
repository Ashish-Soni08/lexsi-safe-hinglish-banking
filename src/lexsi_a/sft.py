"""LoRA SFT of Tiny Aya Fire on the trilingual Bitext banking mix, through AlignTune.

  python -m lexsi_a.sft --train data/sft/train_trilingual.jsonl --out outputs/sft_lora
The banking system prompt is part of every training conversation, matching evaluation.
Hyper-parameters follow Lexsi's CuratorKIT paper (LoRA r=16, alpha=32).
"""
from __future__ import annotations

import argparse
import json
import pathlib

from . import FIRE
from .common import SYSTEM_PROMPT


def to_messages(src: str, dst: pathlib.Path) -> pathlib.Path:
    rows = [json.loads(l) for l in pathlib.Path(src).read_text().splitlines() if l.strip()]
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["response"]}]}, ensure_ascii=False) + "\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="outputs/sft_lora")
    ap.add_argument("--model", default=FIRE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--merged-out", default=None, help="also save the merged bf16 model here")
    ap.add_argument("--push", default=None, help="push the merged model to this private HF repo")
    args = ap.parse_args()

    from aligntune.core.backend_factory import create_sft_trainer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = to_messages(args.train, out / "train_messages.jsonl")
    trainer = create_sft_trainer(
        model_name=args.model, dataset_name=str(data), backend="trl", output_dir=str(out),
        num_epochs=args.epochs, max_steps=args.max_steps, batch_size=8, gradient_accumulation_steps=2,
        learning_rate=args.lr, max_seq_length=1024, use_peft=True, lora_r=16, lora_alpha=32,
        lora_dropout=0.05, lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bf16=True, seed=42,
    )
    trainer.train()
    trainer.save_model(str(out))
    print("saved adapter ->", out)
    if args.merged_out or args.push:
        merge(args.model, out, args.merged_out or "outputs/sft_merged", args.push)


def merge(model: str, adapter: pathlib.Path, dst: str, push: str | None) -> None:
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(model)
    merged.save_pretrained(dst)
    tok.save_pretrained(dst)
    print("saved merged ->", dst)
    if push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(push, private=True, exist_ok=True)
        api.upload_folder(folder_path=dst, repo_id=push, commit_message="SFT merged model")
        print("pushed ->", push)


if __name__ == "__main__":
    main()
