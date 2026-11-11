#!/usr/bin/env bash
# Rebuild and audit DPO round 3 from its saved LoRA adapter (the first run trained fine but could not push the
# merged model: private storage limit). No push here.
#   jobs/launch.sh a100-large dpo3 3h bash jobs/stage_dpo3_eval.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/test_trilingual.jsonl --local-dir /tmp/runs --quiet
hf download --repo-type dataset "$RUNS" --include "dpo3/outputs/dpo_lora/adapter_*" --include "dpo3/outputs/dpo_lora/*.json" \
  --include "dpo3/outputs/curator/pairs.jsonl" --local-dir /tmp/runs --quiet
mkdir -p outputs/sft outputs/curator && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
cp /tmp/runs/dpo3/outputs/curator/pairs.jsonl outputs/curator/
hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo3/outputs/dpo_lora'), '/tmp/dpo', None)"
python scripts/interp.py --a /tmp/dpo --b /tmp/policy --s 0.5 --out /tmp/dpo_mix
for pair in "dpo3:/tmp/dpo" "dpo3_mix05:/tmp/dpo_mix"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/dpo3.jsonl results/gens/dpo3_mix05.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_dpo3.json
