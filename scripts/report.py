"""Collect every stage's metrics into one results table (markdown + JSON).

  python scripts/report.py <runs-dir>      # a local copy of the lexsi-a-runs dataset
Writes results/summary.json and results/RESULTS.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = [  # tag, run that holds its metrics, description
    ("fire", "repair", "Tiny Aya Fire (aligned starting point)"),
    ("sft", "repair", "LoRA SFT on trilingual Bitext banking data"),
    ("resta", "repair", "SFT + ReSta (alpha tuned on dev)"),
    ("safemerge", "repair", "SFT + SafeMerge"),
    ("ctheta_all", "repair", "SFT + C-ΔΘ (Fire − Base) on all circuit layers"),
    ("ctheta_top8", "repair", "SFT + C-ΔΘ (Fire − Base) on the top-8 circuit layers, MLP"),
    ("steer", "repair", "SFT + refusal-direction steering"),
    ("revert_top8", "repair2", "SFT reverted toward Fire on the top-8 circuit layers"),
    ("revert_rand8", "repair2", "SFT reverted toward Fire on 8 random layers (control)"),
    ("revert_all", "repair2", "SFT reverted toward Fire on all layers (WiSE 0.5)"),
    ("harden", "harden", "SFT on Bitext + safety mix v1"),
    ("harden_wise", "harden", "harden v1, 50/50 merge with Fire"),
    ("harden2", "harden2", "SFT on Bitext + safety mix v2 (adds edgy-but-safe examples)"),
    ("harden2_wise", "harden2", "harden v2, 50/50 merge with Fire"),
    ("harden2_wise3", "harden2", "harden v2, 70/30 merge with Fire"),
    ("harden3", "harden3", "SFT on Bitext + safety mix v3 (PolyGuardMix, Nemotron Hindi)"),
    ("harden3_wise3", "harden3", "harden v3, 70/30 merge with Fire"),
    ("dpo", "curator", "harden2_wise3 + DPO on CuratorKIT pairs, round 1 (113 pairs)"),
    ("dpo2", "curator2", "harden2_wise3 + DPO on CuratorKIT pairs, round 2 (932 pairs)"),
    ("dpo2_mix05", "dpomix", "DPO round 2, 50/50 merge with harden2_wise3"),
    ("dpo3", "dpo3e", "harden3_wise3 + DPO on CuratorKIT pairs + banking anchors, round 3"),
    ("dpo3_mix05", "dpo3e", "DPO round 3, 50/50 merge with harden3_wise3"),
]
COLS = [("banking_test/all", "Banking test"), ("banking/hinglish", "Banking Hinglish"),
        ("indicsafe_hi/all", "IndicSafe Hindi"), ("xstest/all", "XSTest"), ("orbench/all", "OR-Bench")]


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}"


def main() -> None:
    runs = pathlib.Path(sys.argv[1])
    metrics = {}  # (run, tag) -> metrics; a tag is taken from the run named in MODELS
    for f in runs.glob("*/results/metrics*.json"):
        for tag, m in json.loads(f.read_text()).items():
            metrics[(f.parts[-3], tag)] = m
    summary, lines = {}, []
    head = "| Model | " + " | ".join(f"{n} refuse↑ / comply↓ / over-refuse↓" for _, n in COLS) + " | Banking chrF++ |"
    lines += [head, "|" + "---|" * (len(COLS) + 2)]
    for tag, run, desc in MODELS:
        m = metrics.get((run, tag))
        u = runs / run / "results" / "utility" / f"{tag}.summary.json"
        if u.exists() is False:
            u = next(iter(runs.glob(f"*/results/utility/{tag}.summary.json")), None)
        util = json.loads(u.read_text())["all"]["chrf"] if u else None
        if not m:
            continue
        row = {"description": desc, "chrf": util}
        cells = []
        for key, _ in COLS:
            v = m.get(key, {})
            r, c, o = (v.get(k, {}).get("rate") for k in ("refusal_on_harmful", "harmful_compliance", "over_refusal"))
            row[key] = {"refusal_on_harmful": r, "harmful_compliance": c, "over_refusal": o}
            cells.append(f"{pct(r)} / {pct(c)} / {pct(o)}")
        summary[tag] = row
        lines.append(f"| `{tag}` {desc} | " + " | ".join(cells) + f" | {util if util is not None else '–'} |")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
