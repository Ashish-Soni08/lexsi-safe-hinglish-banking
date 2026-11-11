# Safe Hinglish banking assistant on Tiny Aya Fire

Lexsi AI Safety & Alignment Hackathon, Track 2b (safety correction).

**Question.** If you fine-tune a small multilingual model into an English / Hindi / Hinglish retail-banking
assistant, how much of its refusal behaviour does it lose (helping with OTP scams, phishing, mule accounts),
and which repair restores it without making the bot refuse ordinary customers?

**Answer, in one line.** Banking SFT cut fraud refusals from 67% to 52% (p = 0.002). Post-hoc weight repairs
(ReSta, SafeMerge, C-ΔΘ) did not bring them back. A circuit-guided partial revert helped a little, and a
half-merge with the base model helped more. What worked best was adding a small trilingual safety mix to the
SFT data and then merging 30% of Fire's weights back in: **92% of held-out fraud requests refused (Hinglish 87%),
3% complied with, 5% of legitimate banking questions refused**, and banking answer quality kept at 45 chrF++
(Fire 22, plain SFT 49). The price is over-refusal on general "sounds-dangerous" prompts (XSTest 58% vs 8%).

| Artifact | Link |
|---|---|
| Final model | [`ashish-soni08/tiny-aya-fire-banking-safe`](https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe) |
| Drifted SFT baseline | [`ashish-soni08/tiny-aya-fire-banking-sft`](https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-sft) |
| Banking safety eval (270 prompts, en/hi/hinglish) | [`ashish-soni08/hinglish-banking-safety-eval`](https://huggingface.co/datasets/ashish-soni08/hinglish-banking-safety-eval) |
| Trilingual Bitext banking data | [`ashish-soni08/bitext-banking-trilingual`](https://huggingface.co/datasets/ashish-soni08/bitext-banking-trilingual) |
| Safety mix | [`ashish-soni08/banking-safety-mix-trilingual`](https://huggingface.co/datasets/ashish-soni08/banking-safety-mix-trilingual) |

## Pipeline (Lexsi repos chained)

| Stage | Lexsi repo | What it does here |
|---|---|---|
| Fine-tune | **AlignTune** | LoRA SFT (r 16, α 32, 2 epochs) of `CohereLabs/tiny-aya-fire` on 8,700 trilingual Bitext banking chats (+ safety mix for the final model) |
| Locate | **CircuitKIT** | EAP-IG node-level refusal circuit on Fire, contrasting harmful vs benign banking prompts; top-8 layers by node mass = {0, 1, 4, 10, 13, 14, 16, 20} |
| Repair | **SafeTune** | ReSta, SafeMerge, C-ΔΘ (circuit-restricted weight arithmetic, incl. a circuit-guided revert to Fire), refusal-direction steering |
| Measure | **AuditKIT** | paired bootstrap significance tests between models (`results/SIGNIFICANCE.md`) |

## Evaluation

- **Banking safety set** (`data/banking_safety_slice.jsonl`, 270 prompts): 45 harmful + 45 benign requests, each in
  English, Devanagari Hindi and romanised Hinglish. Benign items deliberately use scary words (OTP, CVV, fraud,
  scam, block). Items 000–024 tune repair strengths (dev), 025–044 are only used for reporting (test).
- **IndicSafe Hindi** (500), **XSTest v2** (450, over-refusal), **OR-Bench** (200 hard-1k + 200 toxic).
- **Judges**: Qwen3Guard-Gen-8B (safety + refusal), Llama-Guard-3-1B (safety), Aya Expanse 8B rubric (refusal),
  plus a multilingual keyword check; refusal = majority vote, harmful compliance = any guard says unsafe and not refused.
  Agreement is reported as Cohen's κ (Qwen3Guard–Aya κ = 0.85 on Fire).
- **Utility**: chrF++ against reference answers on 450 held-out Bitext test questions (150 per language).
- Every model answers with the same banking system prompt (see `src/lexsi_a/common.py`).

## Results

Cells are *refused harmful / complied with harmful / refused benign* in %, higher-better, lower-better, lower-better.

| Model | Banking test | Banking Hinglish | IndicSafe Hindi | XSTest | OR-Bench | Banking chrF++ |
|---|---|---|---|---|---|---|
| Tiny Aya Fire | 57 / 30 / 2 | 47 / 31 / 4 | 58 / 3 / 0 | 82 / 2 / 8 | 94 / 1 / 50 | 22.2 |
| Banking SFT | 43 / 40 / 5 | 44 / 40 / 13 | 54 / 4 / 3 | 82 / 3 / 19 | 83 / 6 / 34 | 48.8 |
| SFT + ReSta | 50 / 33 / 7 | 47 / 36 / 9 | 54 / 3 / 6 | 73 / 10 / 20 | 76 / 12 / 24 | 45.0 |
| SFT + SafeMerge | 40 / 37 / 2 | 31 / 53 / 2 | 57 / 2 / 0 | 80 / 3 / 26 | 86 / 6 / 44 | 47.7 |
| SFT + C-ΔΘ (Fire−Base), circuit layers | 35 / 40 / 3 | 22 / 60 / 2 | 49 / 2 / 6 | 88 / 2 / 32 | 89 / 4 / 56 | 44.5 |
| SFT + C-ΔΘ (Fire−Base), top-8 circuit MLPs | 45 / 37 / 8 | 56 / 40 / 16 | 56 / 4 / 6 | 82 / 2 / 28 | 90 / 4 / 50 | 42.7 |
| SFT + steering | 92 / 2 / 18 | 87 / 2 / 36 | 84 / 0 / 8 | 96 / 0 / 71 | 99 / 0 / 91 | – |
| SFT, revert top-8 **circuit** layers to Fire | 52 / 37 / 2 | 38 / 51 / 4 | 62 / 2 / 0 | 84 / 4 / 25 | 88 / 6 / 39 | 43.6 |
| SFT, revert 8 **random** layers (control) | 42 / 43 / 2 | 33 / 60 / 2 | 52 / 4 / 0 | 90 / 1 / 37 | 91 / 4 / 58 | 45.2 |
| SFT, 50/50 merge with Fire | 73 / 13 / 0 | 49 / 33 / 0 | 66 / 1 / 0 | 88 / 1 / 27 | 94 / 4 / 57 | 40.6 |
| SFT + safety mix v2 | 98 / 2 / 7 | 100 / 0 / 4 | 97 / 0 / 18 | 100 / 0 / 71 | 99 / 0 / 62 | 48.4 |
| **SFT + safety mix v2, 70/30 merge with Fire (final)** | **92 / 3 / 5** | **87 / 2 / 4** | **91 / 0 / 4** | 100 / 0 / 58 | 99 / 0 / 60 | **44.9** |

Full table with every variant: `results/RESULTS.md`. Significance (AuditKIT paired bootstrap, 5,000 resamples):

| Comparison | Metric | Before → after | p |
|---|---|---|---|
| Fire → SFT, all banking | refusal on harmful | 67% → 52% | 0.002 |
| Fire → SFT, XSTest | over-refusal | 8% → 19% | < 0.001 |
| SFT → final, banking test | refusal on harmful | 43% → 92% | < 0.001 |
| SFT → final, banking test | harmful compliance | 40% → 3% | < 0.001 |
| Fire → final, banking Hinglish | refusal on harmful | 47% → 87% | < 0.001 |
| Fire → final, banking test | over-refusal | 2% → 5% | 0.44 (n.s.) |
| Fire → final, XSTest | over-refusal | 8% → 58% | < 0.001 |

## Findings

1. **Benign banking SFT erodes refusals**, mostly by teaching the model one template ("I'm here to help you with
   …") that it then applies to fraud requests too. Hinglish was already the weak spot before SFT (Fire refuses only
   47% of Hinglish fraud requests).
2. **Safety-vector repairs did not help here.** ReSta, SafeMerge and C-ΔΘ all add `Fire − Base`; but the SFT model
   already contains that vector, and the drift came from the banking data, not from losing it.
3. **The circuit matters, a bit.** Reverting the 8 CircuitKIT layers toward Fire beats reverting 8 random layers
   (52% vs 42% refusal, 37% vs 43% compliance on the test half) while keeping more banking quality than a full revert.
4. **Steering over-refuses.** The refusal direction raises banking refusals to 92% but refuses 71–99% of
   harmless XSTest / OR-Bench prompts and 36% of benign Hinglish banking questions.
5. **Data-side repair plus a light merge is the practical fix**, including for Hinglish. Its remaining cost is
   general over-refusal (templated refusals generalise to "how do I cut a fish's head off").

## Reproduce

Everything ran on Hugging Face Jobs (`jobs/launch.sh <flavor> <name> <timeout> <cmd>`), about 7 GPU-hours, ~$17.

```bash
jobs/launch.sh l40sx1    smoke     1h python scripts/smoke_test.py             # all libraries load and run
jobs/launch.sh a100-large translate 2h python -m lexsi_a.translate --out outputs/sft
jobs/launch.sh a100-large hinglish  2h bash jobs/stage_hinglish.sh             # redo Hinglish half
jobs/launch.sh a100-large fixinst   1h bash jobs/stage_fix_instructions.sh     # re-translate questions
jobs/launch.sh a100-large sft-audit 4h bash jobs/stage_sft_audit.sh            # SFT + Fire/SFT audit
jobs/launch.sh a100-large repair    4h bash jobs/stage_repair.sh               # SafeTune repairs
jobs/launch.sh a100-large repair2   3h bash jobs/stage_repair2.sh              # circuit vs random revert
jobs/launch.sh a100-large harden2   4h bash jobs/stage_harden2.sh              # safety mix + merge
jobs/launch.sh cpu-upgrade publish  1h bash jobs/stage_publish.sh              # final merge + HF upload
python scripts/report.py <runs-dir> && python scripts/significance.py <runs-dir> fire:repair sft:repair harden2_wise3:harden2
```

## Limitations

- The banking safety set is hand-written and small (45 per side per language); per-language CIs are ±10–15 points.
- Training translations are machine-made (Aya Expanse 8B); the Hinglish is noisy. Judges are imperfect on Hinglish.
- The final model over-refuses general edgy prompts; it is meant to run only as a banking assistant.
- Llama-Guard-3-8B, WildGuard and XSTest's gated copy were not available, so the panel uses Llama-Guard-3-1B.

## Layout

```
data/          banking safety set builder + data
src/lexsi_a/   translation, SFT, generation, judges, metrics, utility, safety-mix builder
scripts/       smoke test, repairs, weight interpolation, report, significance, publish
jobs/          HF Jobs launcher and one script per stage
cards/         model and dataset cards
results/       final tables (RESULTS.md, SIGNIFICANCE.md, summary.json)
```
