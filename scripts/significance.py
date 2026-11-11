"""Paired significance tests between models with AuditKIT's paired bootstrap.

Rows are paired by eval id (every model answers the same prompts). For each suite we compare
  refusal on harmful prompts, harmful compliance, and over-refusal on benign prompts.

  python scripts/significance.py <runs-dir> fire:repair sft:repair harden2_wise3:harden2 ...
The first model is the baseline for every comparison listed in PAIRS below.
Writes results/significance.json and results/SIGNIFICANCE.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

from auditkit._bootstrap import paired_bootstrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexsi_a.judge import aggregate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITES = ["banking_test", "banking", "indicsafe_hi", "xstest", "orbench"]


def suite_of(r: dict) -> list[str]:
    out = [r["suite"]]
    if r["suite"] == "banking" and int(r["id"].split("-")[1]) >= 25:
        out.append("banking_test")
    if r["suite"] == "banking" and r["lang"] == "hinglish":
        out.append("banking_hinglish")
    return out


def main() -> None:
    runs = pathlib.Path(sys.argv[1])
    specs = [a.split(":") for a in sys.argv[2:]]
    rows = {tag: {r["id"]: r for r in aggregate(str(runs / run / "results" / "gens" / f"{tag}.jsonl"))} for tag, run in specs}
    tags = [t for t, _ in specs]
    pairs = [(tags[0], t) for t in tags[1:]] + ([(tags[1], t) for t in tags[2:]] if len(tags) > 2 else [])
    res, md = [], ["| Baseline → candidate | Suite | Metric | Baseline | Candidate | Δ | p |", "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        for suite in SUITES + ["banking_hinglish"]:
            for metric, expected, field in (("refusal_on_harmful", "refuse", "refused"),
                                            ("harmful_compliance", "refuse", "unsafe"),
                                            ("over_refusal", "comply", "refused")):
                ids = [i for i, r in rows[a].items() if suite in suite_of(r) and r["expected"] == expected and i in rows[b]]
                if len(ids) < 2:
                    continue
                pr = [(float(rows[a][i][field]), float(rows[b][i][field])) for i in ids]
                t = paired_bootstrap(pr, n_resamples=5000)
                res.append({"baseline": a, "candidate": b, "suite": suite, "metric": metric, **t})
                star = " **" if t["p_value"] < 0.05 else ""
                md.append(f"| {a} → {b} | {suite} | {metric} | {100 * t['mean_baseline']:.0f}% | {100 * t['mean_candidate']:.0f}% "
                          f"| {100 * t['delta']:+.0f} | {t['p_value']:.3f}{star} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "significance.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results" / "SIGNIFICANCE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
