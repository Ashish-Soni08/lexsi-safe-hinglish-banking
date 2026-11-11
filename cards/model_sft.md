---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
language: [en, hi]
tags: [banking, hinglish, lexsi-hackathon, aligntune, research-artifact]
datasets: [ashish-soni08/bitext-banking-trilingual]
---

# Tiny Aya Fire Banking SFT (drifted baseline, research artifact)

LoRA SFT (AlignTune, r=16, α=32) of `CohereLabs/tiny-aya-fire` on 8,700 trilingual Bitext banking conversations,
merged. **This is the "before" model in a safety-drift study and is less safe than its base**: refusal on banking
fraud requests fell from 67% to 52% (p=0.002) and Hinglish fraud compliance rose to 40%. Use
[`ashish-soni08/tiny-aya-fire-banking-safe`](https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe) instead.

Code and full results: https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking
