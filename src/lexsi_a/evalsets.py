"""Unified safety eval suites. Every row: id, suite, lang, expected ('refuse'|'comply'|'ambiguous'), prompt.

Suites
  banking      our 150-prompt English / Hindi / Hinglish banking slice
  indicsafe_hi IndicSafe Hindi (500): harmless controls -> comply, tricky -> ambiguous, rest -> refuse
  xstest       XSTest v2 (450): 'contrast_*' -> refuse, others -> comply (over-refusal probe)
  orbench      OR-Bench: 200 from hard-1k (seemingly toxic but safe -> comply) + 200 from toxic (-> refuse)
"""
from __future__ import annotations

import random

from datasets import load_dataset

from .common import load_slice


def banking() -> list[dict]:
    return [{"id": r["id"], "suite": "banking", "lang": r["lang"], "expected": r["expected"],
             "category": r["category"], "prompt": r["prompt"]} for r in load_slice()]


def indicsafe_hi() -> list[dict]:
    ds = load_dataset("ppattnay/IndicSafe", split="benchmark").filter(lambda r: r["Language"] == "Hindi")
    rows = []
    for i, r in enumerate(ds):
        cat = r["Category"]
        exp = "comply" if cat.startswith("Harmless Control") else "ambiguous" if cat.startswith("Tricky") else "refuse"
        rows.append({"id": f"indicsafe-hi-{i}", "suite": "indicsafe_hi", "lang": "hi", "expected": exp,
                     "category": cat, "prompt": r["Translated_Prompt"]})
    return rows


def xstest() -> list[dict]:
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    return [{"id": f"xstest-{r['id']}", "suite": "xstest", "lang": "en",
             "expected": "refuse" if r["type"].startswith("contrast_") else "comply",
             "category": r["type"], "prompt": r["prompt"]} for r in ds]


def orbench(n: int = 200, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for cfg, exp in (("or-bench-hard-1k", "comply"), ("or-bench-toxic", "refuse")):
        ds = list(load_dataset("bench-llm/or-bench", cfg, split="train"))
        for i, r in enumerate(rng.sample(ds, n)):
            rows.append({"id": f"{cfg}-{i}", "suite": "orbench", "lang": "en", "expected": exp,
                         "category": r["category"], "prompt": r["prompt"]})
    return rows


SUITES = {"banking": banking, "indicsafe_hi": indicsafe_hi, "xstest": xstest, "orbench": orbench}


def load_all(names: list[str] | None = None) -> list[dict]:
    rows = []
    for name in names or SUITES:
        rows += SUITES[name]()
    return rows
