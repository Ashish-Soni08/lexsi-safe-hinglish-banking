# Handoff: idea A build (2026-11-10, first build thread)

## Status
- Code on branch `build/idea-a`. Nothing has run on a GPU yet. $0 spent.
- The first thread's HF token was read-only ("muse-challenge"). Ashish has since added a write + Jobs token,
  which only new sessions see. Check with: `curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2`
  (`auth.accessToken.fineGrained` must include job and repo write permissions).
- Budget: $249 HF credits for this build (Ashish: "use responsibly and only what you need"). Plan ≈ $30–40.
  Modal (~$60 share) only with Ashish's explicit go plus a cost estimate.

## What exists
| File | Purpose |
|---|---|
| `data/build_banking_slice.py` → `banking_safety_slice.jsonl` | 25 harmful + 25 benign banking prompts × en/hi/hinglish (150) |
| `src/lexsi_a/evalsets.py` | banking + IndicSafe-Hindi (500) + XSTest v2 (450) + OR-Bench (400) = 1,500 rows |
| `src/lexsi_a/translate.py` | Bitext (Lexsi splits `sohamb37lexsi/...`) → Hindi + Hinglish with Aya Expanse 8B (vLLM), script-check gate |
| `src/lexsi_a/sft.py` | AlignTune LoRA SFT (r16/α32), banking system prompt in every example |
| `src/lexsi_a/gen.py` | vLLM generation over eval rows (supports `--lora`) |
| `src/lexsi_a/judge.py` | judge panel: Qwen3Guard-Gen-8B, Llama-Guard-3-1B, Aya Expanse 8B rubric, + keyword; one judge per process |
| `src/lexsi_a/metrics.py` | refusal / harmful-compliance / over-refusal with bootstrap CIs, judge kappa |
| `scripts/smoke_test.py` | isolated stages: load, CircuitKIT discover, ReSta, SafeMerge, C-ΔΘ, Steer, AlignTune 20 steps, vLLM |
| `jobs/launch.sh`, `jobs/entry.sh` | run any command on HF Jobs; uploads results/ + outputs/ to private dataset `ashish-soni08/lexsi-a-runs` |

## Next steps
1. Smoke test: `jobs/launch.sh l40sx1 smoke 1h python scripts/smoke_test.py` (~$2). Read `results/smoke_test.json`
   from the runs dataset, fix whichever stages failed (API kwargs were read from wheel source, not executed).
2. Translate: `python -m lexsi_a.translate --n-train 3000 --n-test 300` on an A100 (~20 min).
3. SFT, then gen + 3 judges for `fire`, `base`? (no chat template; skip), `sft`.
4. Repairs on the merged SFT model: ReSta, SafeMerge (`finetuned` must come from `aligned`=Fire, which it does),
   C-ΔΘ restricted to CircuitKIT `layer_suggestions` from the banking refusal circuit, Steer, and a Harden variant.
   Strength sweep with `safetune.recover.sweep_ctheta_strength`. Then AuditKIT `compare_models` for significance.
5. Publish models + datasets in HF collections, HF article, README.

## Gotchas
- C-ΔΘ in SafeTune = `safetune.recover.apply_ctheta(target, positive, negative, circuit_info, strength)`; the runner
  uses positive=aligned, negative=base. `circuit_info.layer_suggestions` comes from CircuitKIT `*_scores.json`.
- CircuitKIT custom task: `SafetyPromptAdapter().adapt([{"prompt": ...}], pairing_mode="answer_contrastive")`
  then `register_task(NormalizedTaskSpec(ds, name=...))`. Tiny Aya is supported via its cohere2 TL port.
- Deps resolve together with uv: torch 2.13, transformers 5.19, vllm 0.32, trl 1.7.1 (see requirements.txt).
- Gated/unavailable: `meta-llama/Llama-Guard-3-8B`, `allenai/wildguard`, `walledai/XSTest`, `r1-b/indic-safety*`
  (403 with current token; Ashish could accept the licences). MultiJail has no Hindi. No public "IndicSafetyBench" on HF.
- Banking slice is small (25 per language per side); CIs are wide. Expand to ~60 per side before final numbers.
- Possible competitor: `kazani351/tiny-aya-hinglish-safety-data` (created 2026-11-09).
