# Safe Hinglish banking assistant on Tiny Aya Fire

Lexsi AI Safety & Alignment Hackathon, Track 2b (safety correction).

**Question.** When you fine-tune a small multilingual model into a Hindi / Hinglish
retail-banking assistant, how much refusal behaviour does it lose (helping with OTP
scams, phishing, mule accounts), and which repair restores it without making the bot
refuse ordinary banking questions?

## Pipeline

| Stage | Lexsi repo | What it does |
|---|---|---|
| Fine-tune | AlignTune | LoRA SFT of `CohereLabs/tiny-aya-fire` on Bitext retail banking (English + Hindi + Hinglish) |
| Locate | CircuitKIT | EAP-IG refusal circuit on Hindi / Hinglish banking harm prompts |
| Repair | SafeTune | ReSta, SafeMerge, C-ΔΘ (circuit-restricted weight arithmetic), refusal-direction steering |
| Measure | AuditKIT | before / after comparison with significance tests and a multi-judge panel |

## Evaluation suites

- `data/banking_safety_slice.jsonl`: 25 harmful + 25 benign banking requests, each in English,
  Devanagari Hindi and romanised Hinglish (150 prompts). Benign items deliberately use scary
  words (OTP, CVV, fraud, block, kill) to measure over-refusal.
- IndicSafe (Hindi, 500), XSTest v2 (English over-refusal), OR-Bench hard / toxic samples.
- Utility: Bitext held-out test split (Lexsi's splits), English and translated.

## Layout

```
data/      eval slice builder + data
src/lexsi_a/  shared code
scripts/   runnable entry points (smoke_test.py first)
results/   metrics and samples written by runs
```

Status: work in progress.
