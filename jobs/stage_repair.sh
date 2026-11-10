#!/usr/bin/env bash
# Stage 3: build repaired variants of the SFT model, generate, judge, and compare against Fire and SFT.
#   jobs/launch.sh a100-large repair 4h bash jobs/stage_repair.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
SFT=${SFT_REPO:-ashish-soni08/tiny-aya-fire-banking-sft}
hf download --repo-type dataset "$RUNS" --include "smoke2/results/circuit_banking_scores.json" "fixinst/outputs/sft/test_trilingual.jsonl" --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download "$SFT" --local-dir /tmp/sft --quiet
# Reuse Fire / SFT generations, utility scores and judge votes from the sft-audit run.
hf download --repo-type dataset "$RUNS" --include "sft-audit/results/gens/*" "sft-audit/results/utility/*" --local-dir /tmp/prev --quiet
mkdir -p results/gens results/utility && cp /tmp/prev/sft-audit/results/gens/* results/gens/ && cp /tmp/prev/sft-audit/results/utility/* results/utility/

python scripts/repair.py build --sft /tmp/sft --circuit /tmp/runs/smoke2/results/circuit_banking_scores.json --out /tmp/rep
python scripts/repair.py steer --sft /tmp/sft --out /tmp/rep
for tag in resta safemerge ctheta_all ctheta_top8; do
  model=/tmp/rep/$tag
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/resta.jsonl results/gens/safemerge.jsonl results/gens/ctheta_all.jsonl results/gens/ctheta_top8.jsonl results/gens/steer.jsonl"
G="results/gens/fire.jsonl results/gens/sft.jsonl results/gens/resta.jsonl results/gens/safemerge.jsonl results/gens/ctheta_all.jsonl results/gens/ctheta_top8.jsonl results/gens/steer.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $G > results/metrics.json
cp -r /tmp/rep/steer_direction.pt outputs/ 2>/dev/null || true
