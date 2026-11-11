#!/usr/bin/env bash
# Interpolate the DPO round-2 model with the current final model (s = share of the final model), then audit.
#   jobs/launch.sh a100-large dpomix 3h bash jobs/stage_dpo_mix.sh 0.5
set -euo pipefail
S=${1:-0.5}; TAG=dpo2_mix${S/./}
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/test_trilingual.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download ashish-soni08/tiny-aya-fire-banking-dpo2 --local-dir /tmp/dpo2 --quiet
hf download ashish-soni08/tiny-aya-fire-banking-safe --local-dir /tmp/final --quiet
rm -rf /tmp/dpo2/.cache /tmp/final/.cache
python scripts/interp.py --a /tmp/dpo2 --b /tmp/final --s "$S" --out /tmp/mix
python -m lexsi_a.gen --model /tmp/mix --tag $TAG
python -m lexsi_a.utility --model /tmp/mix --tag $TAG --test outputs/sft/test_trilingual.jsonl
N="results/gens/$TAG.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_$TAG.json
