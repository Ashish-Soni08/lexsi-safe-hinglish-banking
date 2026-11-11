"""LoRA DPO on the CuratorKIT preference pairs, through AlignTune (TRL backend).

  python -m lexsi_a.dpo --train outputs/curator/dpo_train.jsonl --model ashish-soni08/tiny-aya-fire-banking-safe \
      --out outputs/dpo_lora --merged-out /tmp/dpo
The pairs are conversational (system + user prompt, assistant chosen / rejected), so the chat template and the
banking system prompt match evaluation. With LoRA the reference model is the adapter-free policy.
"""
from __future__ import annotations

import argparse
import pathlib

from .sft import merge

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/dpo_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--merged-out", default=None)
    ap.add_argument("--push", default=None)
    args = ap.parse_args()

    from aligntune.core.backend_factory import create_rl_trainer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer, fallback = None, False
    try:
        trainer = create_rl_trainer(
            model_name=args.model, dataset_name=args.train, algorithm="dpo", backend="trl", output_dir=str(out),
            num_epochs=args.epochs, max_steps=args.max_steps, batch_size=2, gradient_accumulation_steps=8,
            use_gradient_checkpointing=True, learning_rate=args.lr, beta=args.beta, max_seq_length=1024,
            use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_TARGETS,
            precision="bf16", split="train", seed=42, logging_steps=5, save_steps=100000,
        )
        result = trainer.train()
        print("DPO result (AlignTune)", result)
        trainer.save_model(str(out))
    except Exception as e:  # keep the run alive if AlignTune's data layer rejects the conversational pairs
        import traceback
        traceback.print_exc()
        print("AlignTune DPO failed, falling back to plain TRL:", repr(e)[:300])
        fallback = True
    if fallback:  # outside the except block so the failed trainer's frames (and GPU memory) are released
        import gc

        import torch
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        _trl_dpo(args, out)
    if args.merged_out or args.push:
        merge(args.model, out, args.merged_out or "outputs/dpo_merged", args.push)


def _trl_dpo(args, out: pathlib.Path) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    ds = load_dataset("json", data_files=args.train, split="train")
    cfg = DPOConfig(output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=True,
                    learning_rate=args.lr, beta=args.beta, bf16=True, logging_steps=5, save_strategy="no",
                    max_length=1024, seed=42, report_to=[])
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft)
    trainer.train()
    trainer.save_model(str(out))


if __name__ == "__main__":
    main()
