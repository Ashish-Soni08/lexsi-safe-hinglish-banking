#!/usr/bin/env bash
# Stage 4: data-side repair. Generate trilingual refusal + victim-help data, retrain SFT on Bitext + safety mix,
# then audit the hardened model and its 50/50 merge with Fire.
#   jobs/launch.sh a100-large harden2 4h bash jobs/stage_harden2.sh  (v2: fewer generic refusals + OR-Bench-80k helpful examples)
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/train_trilingual.jsonl fixinst/outputs/sft/test_trilingual.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/*.jsonl outputs/sft/

python -m lexsi_a.harden_data --out outputs/harden --n-aya 250 --n-benign 900
python -m lexsi_a.harden_filter outputs/harden/safety_mix.jsonl
cat outputs/sft/train_trilingual.jsonl outputs/harden/safety_mix.jsonl > /tmp/train_harden.jsonl
python -m lexsi_a.sft --train /tmp/train_harden.jsonl --out outputs/harden_lora --merged-out /tmp/harden \
  --push ashish-soni08/tiny-aya-fire-banking-harden2
rm -f outputs/harden_lora/train_messages.jsonl; rm -rf outputs/harden_lora/checkpoint-*
python scripts/interp.py --a /tmp/harden --b CohereLabs/tiny-aya-fire --s 0.5 --out /tmp/harden_wise
python scripts/interp.py --a /tmp/harden --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/harden_wise3

for pair in "harden2:/tmp/harden" "harden2_wise:/tmp/harden_wise" "harden2_wise3:/tmp/harden_wise3"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/harden2.jsonl results/gens/harden2_wise.jsonl results/gens/harden2_wise3.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_harden2.json
