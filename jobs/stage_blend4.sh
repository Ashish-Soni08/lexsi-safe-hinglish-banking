#!/usr/bin/env bash
# Round 4 blend: round 4 (Am) raised Hindi harm refusal but brought back over-refusal. One pre-chosen 50/50 weight
# average of the round-3 final and round-4 Am, scored once on the full test suites (no selection on test).
#   jobs/launch.sh a100-large blend4 3h bash jobs/stage_blend4.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/test_trilingual.jsonl \
  dpo3/outputs/dpo_lora/adapter_config.json dpo3/outputs/dpo_lora/adapter_model.safetensors \
  dpo4/outputs/dpo4a_lora/adapter_config.json dpo4/outputs/dpo4a_lora/adapter_model.safetensors \
  --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy
rm -rf /tmp/harden3
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo3/outputs/dpo_lora'), '/tmp/R3', None)"
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo4/outputs/dpo4a_lora'), '/tmp/A', None)"
python scripts/interp.py --a /tmp/A --b /tmp/policy --s 0.3 --out /tmp/Am
rm -rf /tmp/A
python scripts/interp.py --a /tmp/R3 --b /tmp/Am --s 0.5 --out /tmp/blend
rm -rf /tmp/R3 /tmp/Am

python -m lexsi_a.gen --model /tmp/blend --tag blend4
python -m lexsi_a.utility --model /tmp/blend --tag blend4 --test outputs/sft/test_trilingual.jsonl
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" results/gens/blend4.jsonl || echo "judge $j failed"; done
python -m lexsi_a.metrics results/gens/blend4.jsonl > results/metrics_blend4.json
