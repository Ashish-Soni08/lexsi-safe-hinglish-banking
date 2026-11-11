---
license: cc-by-nc-4.0
language: [en, hi]
tags: [safety, refusal, banking, hinglish, lexsi-hackathon]
task_categories: [text-generation]
---

# Banking Safety Mix, Trilingual

1,222 supervised examples mixed into banking SFT to keep a Tiny Aya banking assistant safe:
- `fraud`: synthetic banking-fraud requests (Aya Expanse 8B) → templated refusal in the same language
- `aya_redteam`: prompts from `CohereLabs/aya_redteaming` (English, Hindi, plus Hinglish translations) → templated refusal
- `victim`: worried or scammed customers asking for help → helpful answer (Aya Expanse 8B)
- `benign_edgy`: OR-Bench-80k "seemingly toxic but safe" prompts (OR-Bench-hard-1k excluded) → helpful answer

Every request was checked with Qwen3Guard-Gen-8B: refusal rows kept only if rated Unsafe/Controversial,
helpful rows only if rated Safe. Synthetic requests too close to the eval set (chrF ≥ 50) were dropped.

Contains harmful *requests* (from red-teaming data) paired with refusals; no harmful completions.
Built for https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking
