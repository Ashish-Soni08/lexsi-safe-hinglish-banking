#!/usr/bin/env bash
# Stage 3b: circuit-guided partial revert (+ random-layer control) and stronger steering.
#   jobs/launch.sh a100-large repair2 3h bash jobs/stage_repair2.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
SFT=${SFT_REPO:-ashish-soni08/tiny-aya-fire-banking-sft}
hf download --repo-type dataset "$RUNS" smoke2/results/circuit_banking_scores.json fixinst/outputs/sft/test_trilingual.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download "$SFT" --local-dir /tmp/sft --quiet

python scripts/repair.py build --round 2 --sft /tmp/sft --circuit /tmp/runs/smoke2/results/circuit_banking_scores.json --out /tmp/rep
python scripts/repair.py steer --round 2 --sft /tmp/sft --out /tmp/rep
for tag in revert_top8 revert_rand8 revert_all; do
  python -m lexsi_a.gen --model /tmp/rep/$tag --tag "$tag"
  python -m lexsi_a.utility --model /tmp/rep/$tag --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/revert_top8.jsonl results/gens/revert_rand8.jsonl results/gens/revert_all.jsonl results/gens/steer_r2.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_r2.json
