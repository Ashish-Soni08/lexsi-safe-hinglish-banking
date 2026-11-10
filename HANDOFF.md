# Handoff: idea A build (updated 2026-11-10, second build thread)

## Status
Branch `claude/project-thread-6p77a6` (continues `build/idea-a`). All GPU work runs on HF Jobs (HF credits only;
Ashish: "just use HF", no Modal). Each stage uploads `results/` + `outputs/` to the private dataset
`ashish-soni08/lexsi-a-runs` under its run name.

| Run name | What | State |
|---|---|---|
| smoke2 | smoke test, all 8 stages pass; `results/circuit_banking_scores.json` = CircuitKIT EAP-IG refusal circuit on Fire | done |
| fixinst | final translated Bitext: `outputs/sft/train_trilingual.jsonl` (8,700) + `test_trilingual.jsonl` (872) | done |
| sft-audit | SFT (pushed to private `ashish-soni08/tiny-aya-fire-banking-sft`) + Fire/SFT gens, 3 judges, utility | done |
| repair | ReSta, SafeMerge, C-ΔΘ (fire−base, all / top-8 circuit layers), steering; sweeps on banking dev half | done |
| repair2 | partial revert toward Fire: circuit top-8 vs random-8 layers vs all layers (WiSE 0.5); steering 1.5–3 | done |
| harden | trilingual safety data (Aya Expanse + templated refusals, Qwen3Guard-checked) + SFT on Bitext+safety, WiSE | running |

## Results so far (banking test half: refusal on harmful / harmful compliance / over-refusal; utility chrF++)
Fire 57/30/2 (22) · SFT 43/40/5 (49) · WiSE-0.5 SFT↔Fire 73/13/0 (41) · circuit-top8 revert 52/37/2 (44) ·
random-8 revert 42/43/2 (45). ReSta/SafeMerge/C-ΔΘ(fire−base) ≈ no gain. Steering over-refuses (XSTest 71–99%).
Hinglish is the weakest language for every model.

## Next
1. Read harden results; pick the final model (best safety/utility trade-off on banking_test).
2. Publish: final model(s) + datasets (banking safety slice, trilingual Bitext, safety mix) in an HF collection,
   model cards, README results table, HF article.

## Gotchas
- `hf jobs wait` returns early; poll `hf jobs inspect`. `hf jobs logs` only returns the tail of long logs.
- `hf download` ignores `--include` once explicit filenames are given; use one form per call.
- python:3.12 image has no nvcc: `VLLM_USE_FLASHINFER_SAMPLER=0` (set in entry.sh).
- SFT at bs8 OOMs on A100-80GB (262k vocab logits): bs4 × accum4 + gradient checkpointing.
- Aya Expanse: writes Devanagari when asked for Hinglish; answers short questions instead of translating them
  (fixed with few-shot + fenced prompts); complies with many fraud requests (hence templated refusals).
- Keyword refusal heuristic counts Bitext's "I'm sorry to hear..." as refusal; repair sweeps use a strict decline regex.
