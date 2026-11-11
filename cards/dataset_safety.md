---
license: cc-by-nc-4.0
language: [en, hi]
tags: [safety, refusal, banking, hinglish, lexsi-hackathon, dpo, preference, curatorkit]
task_categories: [text-generation]
configs:
  - config_name: sft_mix_v3
    data_files: data/train.jsonl
  - config_name: dpo_pairs
    data_files: data/dpo_pairs.jsonl
  - config_name: sft_mix_v2
    data_files: data/v2.jsonl
---

# Banking Safety Mix, Trilingual (SFT mix + DPO preference pairs)

Safety data used to build [`tiny-aya-fire-banking-safe`](https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe),
a Tiny Aya Fire banking assistant for English, Devanagari Hindi and romanised Hinglish.

## `sft_mix_v3` (2,946 rows, used in the final model's SFT)
Fields: `lang`, `kind`, `instruction`, `response`.
- `fraud` (synthetic banking-fraud requests by Aya Expanse 8B → refusal), `victim` (worried customers → helpful
  answer), `benign_edgy` (OR-Bench-80k "seemingly toxic but safe" prompts, OR-Bench-hard-1k excluded → helpful answer),
  kept from v2 after a Qwen3Guard-Gen-8B label check.
- `pg_refuse` / `pg_comply`: English and Hindi rows from ToxicityPrompts/PolyGuardMix (CC-BY-4.0): harmful prompts
  with non-harmful refusals, and harmless (often adversarial-looking) prompts with non-harmful compliant answers.
- `nemo_comply`: safe Hindi prompts with safe answers from safety-aya/Nemotron-Safety-Guard-Dataset-v3-hindi.
- `*_hinglish`: PolyGuardMix English rows translated to Hinglish with Aya Expanse 8B.
Any row sharing ≥ 50% of its word 4-grams with an evaluation prompt was dropped.

## `dpo_pairs` (1,388 pairs, used for the final model's DPO pass)
Fields: `kind`, `lang`, `prompt`, `chosen`, `rejected`, `policy_failed`.
Built with Lexsi's **CuratorKIT**: EvolInstructTask (Aya Expanse 8B) evolved banking-fraud requests and harmless
"sounds scary" prompts (plus fresh OR-Bench-80k prompts), DiversityGate removed near duplicates, Aya translated them
to Hindi and Hinglish, and prompts overlapping any eval set were dropped. Answers were sampled from the policy, the
banking SFT model and Tiny Aya Fire, labelled by Qwen3Guard-Gen-8B, and paired:
- `fraud` (430): chosen = refusal, rejected = compliant answer
- `edgy` (508): chosen = helpful answer, rejected = refusal
- `anchor` (450): ordinary Bitext banking questions, chosen = reference answer, rejected = refusal

**Content warning:** `rejected` answers in `fraud` pairs are small-model completions that comply with fraud requests
(e.g. phishing-message drafts). They are kept as negative examples for preference training only.

## `sft_mix_v2` (1,222 rows, the previous mix, kept for reproducibility)
Templated refusals for `fraud` and `aya_redteam` (CohereLabs/aya_redteaming) requests plus `victim` and `benign_edgy` answers.

Code: https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking
