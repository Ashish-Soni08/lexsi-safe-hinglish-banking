#!/usr/bin/env bash
# Stage 4: data-side repair. Generate trilingual refusal + victim-help data, retrain SFT on Bitext + safety mix,
# then audit the hardened model and its 50/50 merge with Fire.
#   jobs/launch.sh a100-large harden 4h bash jobs/stage_harden.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/train_trilingual.jsonl fixinst/outputs/sft/test_trilingual.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/*.jsonl outputs/sft/

python -m lexsi_a.harden_data --out outputs/harden
python -m lexsi_a.harden_filter outputs/harden/safety_mix.jsonl
cat outputs/sft/train_trilingual.jsonl outputs/harden/safety_mix.jsonl > /tmp/train_harden.jsonl
python -m lexsi_a.sft --train /tmp/train_harden.jsonl --out outputs/harden_lora --merged-out /tmp/harden \
  --push ashish-soni08/tiny-aya-fire-banking-harden
rm -f outputs/harden_lora/train_messages.jsonl; rm -rf outputs/harden_lora/checkpoint-*
python scripts/interp.py --a /tmp/harden --b CohereLabs/tiny-aya-fire --s 0.5 --out /tmp/harden_wise

for pair in "harden:/tmp/harden" "harden_wise:/tmp/harden_wise"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/harden.jsonl results/gens/harden_wise.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_harden.json
