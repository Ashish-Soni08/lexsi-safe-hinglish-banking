"""Refusal circuits for the dashboard: run CircuitKIT EAP-IG on several models with the same task and settings,
then render each circuit with CircuitKIT's D3 graph and compare them with its ComparisonDashboard.

  python scripts/circuits.py fire=CohereLabs/tiny-aya-fire sft=ashish-soni08/tiny-aya-fire-banking-sft \
      final=ashish-soni08/tiny-aya-fire-banking-safe --out results/circuits
Writes <out>/<tag>_scores.json, <out>/<tag>_circuit.html, <out>/comparison.html and <out>/layer_mass.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexsi_a.common import load_slice  # noqa: E402

TASK = "banking_hinglish_refusal"
BASE = "CohereLabs/tiny-aya-fire"


def register() -> None:
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task

    # Harmful banking prompts contrasted with benign banking prompts that use the same scary words.
    harmful = [{"prompt": r["prompt"]} for r in load_slice(label="harmful")]
    benign = [r["prompt"] for r in load_slice(label="benign")]
    ds = SafetyPromptAdapter().adapt(harmful, name=TASK, source="local", max_records=len(harmful),
                                     pairing_mode="harmful_vs_benign", benign_controls=benign)
    register_task(NormalizedTaskSpec(ds, name=TASK))


def layer_mass(scores: dict) -> dict:
    mass: dict = {}
    for node, v in scores.items():
        m = re.match(r"([AM])(\d+)", node)
        if m:
            kind = "attn" if m.group(1) == "A" else "mlp"
            mass.setdefault(int(m.group(2)), {"attn": 0.0, "mlp": 0.0})[kind] += abs(float(v))
    return {str(k): mass[k] for k in sorted(mass)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="tag=model")
    ap.add_argument("--out", default="results/circuits")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import circuitkit.quick as ck
    from circuitkit.artifacts.scores import CircuitScores
    from circuitkit.visualize import CircuitGraphVisualizer, ComparisonDashboard

    register()
    circuits = {}
    for spec in args.models:
        tag, model = spec.split("=", 1)
        if (out / f"{tag}_scores.json").exists():
            print("reusing", tag)
            circuits[tag] = json.loads((out / f"{tag}_scores.json").read_text())["node_scores"]
            continue
        torch.manual_seed(0)
        if model == BASE:
            hooked = ck.load_model(model, algorithm="eap-ig")
        else:
            # TransformerLens only knows the registered tiny-aya repo IDs, so load the Fire config and
            # hand it the fine-tuned weights (same Cohere2 architecture).
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
            hooked = ck.load_model(BASE, algorithm="eap-ig", hf_model=hf)
            del hf
        ck.discover(hooked, TASK, algorithm="eap-ig", level="node", n_examples=32, batch_size=4, sparsity=0.3,
                    output_path=str(out / f"{tag}.pt"))
        del hooked
        gc.collect()
        torch.cuda.empty_cache()
        side = sorted(out.glob(f"{tag}*_scores.json"))
        if not side:
            print("no scores for", tag)
            continue
        if side[0].name != f"{tag}_scores.json":
            side[0].rename(out / f"{tag}_scores.json")
        path = out / f"{tag}_scores.json"
        scores = CircuitScores.from_json(path)
        CircuitGraphVisualizer({"nodes": [], "edges": []}, scores).to_html(
            str(out / f"{tag}_circuit.html"), title=f"Refusal circuit: {tag}")
        circuits[tag] = json.loads(path.read_text())["node_scores"]
        print("circuit", tag, "nodes", len(circuits[tag]))
        (out / f"{tag}.pt").unlink(missing_ok=True)

    (out / "layer_mass.json").write_text(json.dumps({t: layer_mass(s) for t, s in circuits.items()}, indent=1))
    if len(circuits) >= 2:
        ComparisonDashboard(circuits, comparison_type="stability").export_to_html(str(out / "comparison.html"))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
