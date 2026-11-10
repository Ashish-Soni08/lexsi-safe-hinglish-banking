"""Repair the SFT banking model with SafeTune, guided by a CircuitKIT refusal circuit.

Two sub-commands, each run in its own process (vLLM and HF models do not share a GPU well):

  python scripts/repair.py build --sft <repo-or-dir> --circuit <scores.json> --out /tmp/rep
      For every method, sweep its strength, score each candidate on the DEV half of the banking
      slice (keyword refusal judge, 25 harmful + 25 benign items x 3 languages), keep the best
      candidate per method, and save it to <out>/<method>. Writes results/repair_sweep.json.
      Selection score = refusal on harmful - refusal on benign (dev only, never the test half).

  python scripts/repair.py steer --sft <repo-or-dir> --sweep results/repair_sweep.json
      Runtime refusal-direction steering cannot be saved as weights, so this generates the full
      eval suites with plain transformers + hooks for the best steering setting.
      Writes results/gens/steer.jsonl.

Methods: ReSta (alpha), SafeMerge, C-ΔΘ on all layers the circuit touches, C-ΔΘ on the top-8
circuit layers only (MLP weights, which carry ~all of the circuit's node mass), and Steer.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import pathlib
import re
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexsi_a import BASE, FIRE  # noqa: E402
from lexsi_a.common import ROOT, generate, load_slice, looks_like_refusal  # noqa: E402

RESULTS = ROOT / "results"
DEV_MAX_INDEX = 25  # banking items 000-024 are the dev half, 025-044 the test half


def is_dev(row: dict) -> bool:
    return int(row["id"].split("-")[1]) < DEV_MAX_INDEX


def load(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()


def dev_score(model, tok, dev) -> dict:
    answers = generate(model, tok, [r["prompt"] for r in dev], max_new_tokens=80, batch_size=32)
    harm = [looks_like_refusal(a) for r, a in zip(dev, answers) if r["label"] == "harmful"]
    ben = [looks_like_refusal(a) for r, a in zip(dev, answers) if r["label"] == "benign"]
    h, b = sum(harm) / len(harm), sum(ben) / len(ben)
    return {"refusal_harmful": round(h, 3), "refusal_benign": round(b, 3), "score": round(h - b, 3)}


def circuit_layers(scores_path: str, top_k: int = 8) -> tuple[list[int], list[int]]:
    d = json.loads(pathlib.Path(scores_path).read_text())
    per_layer: dict[int, float] = defaultdict(float)
    for node, s in d["node_scores"].items():
        m = re.match(r"[AM](\d+)", node)
        if m:
            per_layer[int(m.group(1))] += abs(float(s))
    ranked = sorted(per_layer, key=per_layer.get, reverse=True)
    return d["layer_suggestions"]["layer_subset"], sorted(ranked[:top_k])


def build(args) -> None:
    from safetune.recover import apply_ctheta, apply_resta, apply_safemerge

    tok = AutoTokenizer.from_pretrained(FIRE)
    dev = [r for r in load_slice() if is_dev(r)]
    base, fire, sft = load(BASE), load(FIRE), load(args.sft)
    all_layers, top_layers = circuit_layers(args.circuit)
    mlp = ["gate_proj", "up_proj", "down_proj"]
    info = SimpleNamespace(layer_suggestions=None)

    candidates = {
        "resta": [(f"alpha={a}", lambda m, a=a: apply_resta(m, base=base, aligned=fire, alpha=a)) for a in (0.25, 0.5, 1.0)],
        "safemerge": [("default", lambda m: apply_safemerge(m, base=base, aligned=fire))],
        "ctheta_all": [(f"strength={s}", lambda m, s=s: apply_ctheta(m, positive=fire, negative=base, circuit_info=info,
                                                                       strength=s, layer_subset=all_layers))
                       for s in (0.25, 0.5, 1.0)],
        "ctheta_top8": [(f"strength={s}", lambda m, s=s: apply_ctheta(m, positive=fire, negative=base, circuit_info=info,
                                                                        strength=s, layer_subset=top_layers, target_modules=mlp))
                        for s in (0.5, 1.0, 2.0)],
    }
    report = {"dev_n": len(dev), "circuit": {"all_layers": all_layers, "top8_layers": top_layers},
              "reference": {"fire": dev_score(fire, tok, dev), "sft": dev_score(sft, tok, dev)}, "methods": {}}
    print("reference", report["reference"], flush=True)
    out = pathlib.Path(args.out)
    for method, cands in candidates.items():
        rows, best = [], None
        for label, fn in cands:
            m = fn(copy.deepcopy(sft))
            sc = dev_score(m, tok, dev)
            rows.append({"setting": label, **sc})
            print(method, label, sc, flush=True)
            if best is None or sc["score"] > best[1]["score"]:
                if best is not None:
                    del best[2]
                best = (label, sc, m)
            else:
                del m
            gc.collect(); torch.cuda.empty_cache()
        label, sc, m = best
        m.save_pretrained(out / method)
        tok.save_pretrained(out / method)
        report["methods"][method] = {"chosen": label, "chosen_dev": sc, "sweep": rows}
        del m, best
        gc.collect(); torch.cuda.empty_cache()
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "repair_sweep.json").write_text(json.dumps(report, indent=1))

    # Steering sweep (runtime hooks on the SFT model).
    from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig, extract_refusal_direction
    from safetune.steer import RefusalDirectionModel
    harm = [r["prompt"] for r in dev if r["label"] == "harmful"]
    ben = [r["prompt"] for r in dev if r["label"] == "benign"]
    direction, layer, _ = extract_refusal_direction(sft, tok, harm, ben, RefusalDirectionConfig(select_directions=False))
    torch.save({"direction": direction.cpu(), "layer": int(layer)}, out / "steer_direction.pt")
    rows = []
    for layers, strengths in (([int(layer)], (4.0, 8.0, 16.0, 32.0)), (None, (0.25, 0.5, 1.0))):
        for s in strengths:
            h = RefusalDirectionModel(sft, direction=direction, mode="steer", strength=s, layers=layers).install()
            try:
                sc = dev_score(sft, tok, dev)
            finally:
                h.remove()
            rows.append({"setting": f"layers={'L' + str(layer) if layers else 'all'},strength={s}",
                         "layers": layers, "strength": s, **sc})
            print("steer", rows[-1], flush=True)
    best = max(rows, key=lambda r: r["score"])
    report["methods"]["steer"] = {"chosen": best["setting"], "chosen_dev": best, "sweep": rows, "layer": int(layer)}
    (RESULTS / "repair_sweep.json").write_text(json.dumps(report, indent=1))


def steer(args) -> None:
    from safetune.steer import RefusalDirectionModel
    from lexsi_a.evalsets import load_all

    sweep = json.loads(pathlib.Path(args.sweep).read_text())["methods"]["steer"]["chosen_dev"]
    saved = torch.load(pathlib.Path(args.out) / "steer_direction.pt")
    tok = AutoTokenizer.from_pretrained(FIRE)
    sft = load(args.sft)
    rows = load_all(None)
    h = RefusalDirectionModel(sft, direction=saved["direction"].cuda(), mode="steer",
                              strength=sweep["strength"], layers=sweep["layers"]).install()
    answers = generate(sft, tok, [r["prompt"] for r in rows], max_new_tokens=256, batch_size=48)
    h.remove()
    gens = RESULTS / "gens"
    gens.mkdir(parents=True, exist_ok=True)
    with (gens / "steer.jsonl").open("w") as f:
        for r, a in zip(rows, answers):
            f.write(json.dumps({**r, "model": "steer", "answer": a}, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "steer gens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "steer"])
    ap.add_argument("--sft", required=True)
    ap.add_argument("--circuit", default=None)
    ap.add_argument("--sweep", default=str(RESULTS / "repair_sweep.json"))
    ap.add_argument("--out", default="/tmp/rep")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else steer(a)
