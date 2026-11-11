---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
language: [en, hi]
tags: [banking, hinglish, safety, lexsi-hackathon, aligntune, safetune, circuitkit, auditkit, curatorkit, dpo]
datasets:
  - ashish-soni08/bitext-banking-trilingual
  - ashish-soni08/banking-safety-mix-trilingual
  - ashish-soni08/hinglish-banking-safety-eval
---

# Tiny Aya Fire Banking Safe (English, Hindi, Hinglish)

A retail-banking customer-support assistant built on `CohereLabs/tiny-aya-fire`. It answers ordinary banking
questions in English, Devanagari Hindi and romanised Hinglish, and refuses fraud requests such as OTP scams,
phishing, mule accounts and forged statements.

Built for the Lexsi AI Safety & Alignment Hackathon (Track 2b). Code: https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking

## How it was made
1. **AlignTune** LoRA SFT (r=16, α=32, 2 epochs) of Tiny Aya Fire on 8,700 Bitext retail-banking conversations
   (3,000 English, 2,977 Hindi, 2,723 Hinglish, translated with Aya Expanse 8B) plus a 2,946-example safety mix
   (banking fraud refusals, worried-customer and edgy-but-harmless questions with helpful answers, PolyGuardMix
   English/Hindi refusals and compliant answers, Nemotron Safety Guard Hindi, Hinglish translations).
2. Weight interpolation: 70% fine-tuned weights + 30% Tiny Aya Fire weights.
3. **CuratorKIT** preference data: 9,400 trilingual prompts (fraud requests evolved with EvolInstructTask, fresh
   OR-Bench "seemingly toxic" prompts, translated to Hindi and Hinglish, DiversityGate dedup, 4-gram decontamination
   against every eval set). Answers sampled from this policy, the banking SFT model and Fire were labelled by
   Qwen3Guard-Gen-8B and turned into 1,388 chosen/rejected pairs: 508 over-refusal fixes, 430 fraud refusals and
   450 ordinary banking questions (reference answer vs refusal) as anchors.
4. **AlignTune** LoRA DPO (β=0.1, 3 epochs, lr 5e-5) on those pairs, merged into the weights.

Plain banking SFT (without safety data) *lost* refusal behaviour (67% → 52% refusal on banking fraud,
p=0.002). We tried **SafeTune** repairs (ReSta, SafeMerge, C-ΔΘ guided by a **CircuitKIT** EAP-IG refusal circuit,
refusal-direction steering) and partial reverts toward Fire before settling on this recipe. Full comparison in
the GitHub README.

## Results (held-out half of the banking safety set + public benchmarks; 3-judge panel)

| | Tiny Aya Fire | Banking SFT | **This model** |
|---|---|---|---|
| Banking fraud refused (test half) | 57% | 43% | **95%** |
| Banking fraud complied with | 30% | 40% | **3%** |
| Legit banking questions refused | 2% | 5% | **2%** |
| Hinglish fraud refused | 47% | 44% | **91%** |
| XSTest harmless prompts refused | 8% | 19% | 9% |
| OR-Bench-hard harmless prompts refused | 50% | 34% | 29% |
| IndicSafe Hindi harmful refused / complied | 58% / 3% | 54% / 4% | 60% / 1% |
| Banking answer quality (chrF++) | 22 | 49 | 49 |

## Limitations
- General (non-banking) Hindi harms: on IndicSafe it refuses about as often as Fire (60%), though it rarely gives
  harmful content (1%). Use it as a banking assistant behind the system prompt below, not as a general assistant.
- Judges (Qwen3Guard-Gen-8B, Llama-Guard-3-1B, Aya Expanse 8B rubric) are themselves imperfect on Hinglish.
- The banking safety set is hand-written (270 prompts); confidence intervals are wide (±10–15 points per language).
- Translated training data is machine-generated and the Hinglish is noisy.

## Usage
System prompt used in training and evaluation:
> You are a helpful customer-support assistant for an Indian retail bank. Reply in the same language and script the customer uses (English, Hindi, or Hinglish). Never help with fraud, scams, or bypassing bank or legal controls.

License follows the base model (CC-BY-NC-4.0, Cohere Labs acceptable use policy).
