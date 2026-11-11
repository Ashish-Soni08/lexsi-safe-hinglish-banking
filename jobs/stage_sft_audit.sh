#!/usr/bin/env bash
# Stage 2 on one GPU: SFT on the translated Bitext mix, then audit Fire vs SFT.
#   jobs/launch.sh a100-large sft-audit 4h bash jobs/stage_sft_audit.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
SFT_REPO=${SFT_REPO:-ashish-soni08/tiny-aya-fire-banking-sft}
FIRE=CohereLabs/tiny-aya-fire

hf download --repo-type dataset "$RUNS" --include "${DATA_RUN:-fixinst}/outputs/sft/*" --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/${DATA_RUN:-fixinst}/outputs/sft/*.jsonl outputs/sft/
wc -l outputs/sft/*.jsonl

python -m lexsi_a.sft --train outputs/sft/train_trilingual.jsonl --out outputs/sft_lora \
  --merged-out /tmp/sft_merged --push "$SFT_REPO"
rm -f outputs/sft_lora/train_messages.jsonl; rm -rf outputs/sft_lora/checkpoint-*

for pair in "fire:$FIRE" "sft:/tmp/sft_merged"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
for j in qwen3guard llamaguard aya; do
  python -m lexsi_a.judge --judge "$j" results/gens/fire.jsonl results/gens/sft.jsonl || echo "judge $j failed"
done
python -m lexsi_a.metrics results/gens/fire.jsonl results/gens/sft.jsonl > results/metrics.json
cat results/metrics.json | head -80
