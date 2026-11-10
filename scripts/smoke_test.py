"""Day-1 smoke test: do SafeTune (ReSta, SafeMerge, C-ΔΘ, Steer), CircuitKIT and
AlignTune all run on Tiny Aya Fire? Each stage is isolated so one failure does not
hide the others. Writes results/smoke_test.json.

Run on one GPU with >=40 GB (L40S / A100):  python scripts/smoke_test.py
"""
from __future__ import annotations

import copy
import gc
import json
import pathlib
import sys
import time
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexsi_a import BASE, FIRE  # noqa: E402
from lexsi_a.common import ROOT, generate, load_slice, refusal_rates  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
REPORT: dict = {"stages": {}}
SAMPLES: dict = {}

# Small probe set: 8 harmful + 8 benign, mixed languages.
rows = load_slice()
probe = [r for r in rows if r["label"] == "harmful"][::9][:8] + [r for r in rows if r["label"] == "benign"][::9][:8]
probe_prompts = [r["prompt"] for r in probe]


def stage(name):
    def wrap(fn):
        t0 = time.time()
        try:
            res = fn() or {}
            REPORT["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), **res}
        except Exception as e:  # noqa: BLE001 - smoke test records every failure
            REPORT["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        print(name, json.dumps({k: v for k, v in REPORT["stages"][name].items() if k != "trace"}, ensure_ascii=False)[:600], flush=True)
        (OUT / "smoke_test.json").write_text(json.dumps(REPORT, indent=1, ensure_ascii=False))
        (OUT / "smoke_samples.json").write_text(json.dumps(SAMPLES, indent=1, ensure_ascii=False))
        torch.cuda.empty_cache()
        return fn
    return wrap


def probe_model(tag, model):
    answers = generate(model, tok, probe_prompts, max_new_tokens=96)
    SAMPLES[tag] = [{"id": r["id"], "prompt": r["prompt"], "answer": a} for r, a in zip(probe, answers)]
    return refusal_rates(probe, answers)


tok = AutoTokenizer.from_pretrained(FIRE)
load = lambda name: AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
fire = base = None


@stage("load_and_generate")
def _():
    global fire, base
    fire, base = load(FIRE), load(BASE)
    return {"fire": probe_model("fire", fire), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}


circuit_scores: dict = {}


@stage("circuitkit_discover")
def _():
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    import circuitkit.quick as ck

    harmful = [{"prompt": r["prompt"]} for r in load_slice(label="harmful")]
    ds = SafetyPromptAdapter().adapt(harmful, name="banking_hinglish_refusal", source="local",
                                     max_records=len(harmful), pairing_mode="answer_contrastive")
    register_task(NormalizedTaskSpec(ds, name="banking_hinglish_refusal"))
    hooked = ck.load_model(FIRE, algorithm="eap-ig")
    out = OUT / "circuit_banking.pt"
    circ = ck.discover(hooked, "banking_hinglish_refusal", algorithm="eap-ig", level="node",
                       n_examples=32, batch_size=4, sparsity=0.3, output_path=str(out))
    side = sorted(OUT.glob("circuit_banking*_scores.json"))
    if side:
        circuit_scores.update(json.loads(side[0].read_text()))
    del hooked
    gc.collect()
    ls = circuit_scores.get("layer_suggestions", {})
    return {"n_nodes": len(getattr(circ, "nodes", []) or []), "scores_file": [p.name for p in side],
            "layer_subset": ls.get("layer_subset"), "target_modules": ls.get("target_modules")}


@stage("safetune_resta")
def _():
    from safetune.recover import apply_resta
    m = apply_resta(copy.deepcopy(fire), base=base, aligned=fire, alpha=0.5)
    r = probe_model("resta", m); del m
    return r


@stage("safetune_safemerge")
def _():
    from safetune.recover import apply_safemerge
    if apply_safemerge is None:
        raise RuntimeError("apply_safemerge import failed inside safetune")
    m = apply_safemerge(copy.deepcopy(fire), base=base, aligned=fire)
    r = probe_model("safemerge", m); del m
    return r


@stage("safetune_ctheta")
def _():
    from safetune.recover import apply_ctheta
    ls = circuit_scores.get("layer_suggestions") or {"layer_subset": list(range(8, 20)), "target_modules": ["o_proj", "down_proj"]}
    info = SimpleNamespace(layer_suggestions=SimpleNamespace(**{k: ls.get(k) for k in ("layer_subset", "target_modules")}))
    m = apply_ctheta(copy.deepcopy(fire), positive=fire, negative=base, circuit_info=info, strength=0.5)
    r = probe_model("ctheta", m); del m
    return {**r, "used_circuit": bool(circuit_scores), "layer_subset": ls.get("layer_subset")}


@stage("safetune_steer")
def _():
    from safetune.steer import RefusalDirectionModel
    from safetune.interventions.steer.refusal_direction import extract_refusal_direction, RefusalDirectionConfig
    harm = [r["prompt"] for r in load_slice(label="harmful")]
    ben = [r["prompt"] for r in load_slice(label="benign")]
    cfg = RefusalDirectionConfig(select_directions=False)
    direction, layer, _ = extract_refusal_direction(fire, tok, harm, ben, cfg)
    wrapped = RefusalDirectionModel(fire, direction=direction, mode="steer", strength=4.0)
    wrapped.install()
    try:
        r = probe_model("steer", fire)
    finally:
        wrapped.remove()
    return {**r, "layer": int(layer)}


@stage("aligntune_sft_tiny")
def _():
    from datasets import load_dataset
    from aligntune.core.backend_factory import create_sft_trainer
    ds = load_dataset("sohamb37lexsi/bitext-retail-banking-llm-chatbot-splits", split="train[:256]")
    ds = ds.map(lambda r: {"messages": [{"role": "user", "content": r["instruction"]},
                                        {"role": "assistant", "content": r["response"]}]},
                remove_columns=ds.column_names)
    local = OUT / "sft_tiny"
    ds.to_json(str(local) + ".jsonl")
    trainer = create_sft_trainer(model_name=FIRE, dataset_name="json", data_files=str(local) + ".jsonl",
                                 backend="trl", output_dir=str(local), num_epochs=1, max_steps=20,
                                 batch_size=4, learning_rate=2e-4, max_seq_length=512, use_peft=True,
                                 lora_r=16, lora_alpha=32, bf16=True)
    trainer.train()
    return {"trainer": type(trainer).__name__}


@stage("vllm_load")
def _():
    global fire, base
    del fire, base
    gc.collect(); torch.cuda.empty_cache()
    from vllm import LLM, SamplingParams
    llm = LLM(FIRE, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.6)
    msgs = [[{"role": "user", "content": p}] for p in probe_prompts[:4]]
    outs = llm.chat(msgs, SamplingParams(max_tokens=48, temperature=0))
    return {"sample": outs[0].outputs[0].text[:200]}


REPORT["summary"] = {k: v["ok"] for k, v in REPORT["stages"].items()}
(OUT / "smoke_test.json").write_text(json.dumps(REPORT, indent=1, ensure_ascii=False))
print("SUMMARY", json.dumps(REPORT["summary"]))
