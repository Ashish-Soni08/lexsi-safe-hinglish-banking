#!/usr/bin/env bash
# Stage 4c: safety mix v3 (PolyGuardMix + Nemotron-Hindi + Hinglish translations), retrain, merge, audit.
#   jobs/launch.sh a100-large harden3 5h bash jobs/stage_harden3.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/train_trilingual.jsonl fixinst/outputs/sft/test_trilingual.jsonl harden2/outputs/harden/safety_mix.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/*.jsonl outputs/sft/

python -m lexsi_a.harden_data_v3 --v2 /tmp/runs/harden2/outputs/harden/safety_mix.jsonl --out outputs/harden
cat outputs/sft/train_trilingual.jsonl outputs/harden/safety_mix.jsonl > /tmp/train_harden.jsonl
python -m lexsi_a.sft --train /tmp/train_harden.jsonl --out outputs/harden_lora --merged-out /tmp/harden \
  --push ashish-soni08/tiny-aya-fire-banking-harden3
rm -f outputs/harden_lora/train_messages.jsonl; rm -rf outputs/harden_lora/checkpoint-*
python scripts/interp.py --a /tmp/harden --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/harden_wise3

for pair in "harden3:/tmp/harden" "harden3_wise3:/tmp/harden_wise3"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/harden3.jsonl results/gens/harden3_wise3.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_harden3.json
