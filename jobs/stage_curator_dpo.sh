#!/usr/bin/env bash
# Stage 5: our own preference data with CuratorKIT, then a DPO pass with AlignTune on the final model, then audit.
#   jobs/launch.sh a100-large curator 5h bash jobs/stage_curator_dpo.sh [policy-model-repo]
# Round 2 (mine fresh OR-Bench prompts the policy over-refuses, stronger DPO):
#   jobs/launch.sh a100-large curator2 5h env ROUND=2 bash jobs/stage_curator_dpo.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
POLICY=${1:-ashish-soni08/tiny-aya-fire-banking-safe}
uv pip install --system -q "curatorkit[generation,embedding]==0.1.24" -c requirements.txt
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/test_trilingual.jsonl harden2/outputs/harden/safety_mix.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/

C=outputs/curator; TAG=dpo; EVOLVE=""; DPO="--epochs 2 --lr 1e-5"
if [ "${ROUND:-1}" = 2 ]; then
  hf download --repo-type dataset "$RUNS" curator/outputs/curator/prompts.jsonl --local-dir /tmp/runs --quiet
  TAG=dpo2; EVOLVE="--orbench-fresh ${N_FRESH:-1500} --prev-prompts /tmp/runs/curator/outputs/curator/prompts.jsonl"
  DPO="--epochs 3 --lr 5e-5"
fi
python -m lexsi_a.curator_data evolve --seeds-v2 /tmp/runs/harden2/outputs/harden/safety_mix.jsonl --out $C $EVOLVE
python -m lexsi_a.curator_data sample --model "$POLICY" --tag final --n 3 --out $C
python -m lexsi_a.curator_data sample --model ashish-soni08/tiny-aya-fire-banking-sft --tag sft --n 2 --out $C
python -m lexsi_a.curator_data sample --model CohereLabs/tiny-aya-fire --tag fire --n 2 --out $C
python -m lexsi_a.curator_data judge --out $C
python -m lexsi_a.curator_data pairs --policy final --out $C

python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model "$POLICY" --out outputs/dpo_lora --merged-out /tmp/dpo \
  --push ashish-soni08/tiny-aya-fire-banking-$TAG $DPO
rm -rf outputs/dpo_lora/checkpoint-*

python -m lexsi_a.gen --model /tmp/dpo --tag $TAG
python -m lexsi_a.utility --model /tmp/dpo --tag $TAG --test outputs/sft/test_trilingual.jsonl
N="results/gens/$TAG.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_$TAG.json
