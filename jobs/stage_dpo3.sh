#!/usr/bin/env bash
# Stage 5c: DPO round 3. Policy = safety mix v3 model merged 70/30 with Fire (low over-refusal, weaker on fraud).
# Reuses the round-2 CuratorKIT prompts and SFT/Fire samples, samples the new policy on-policy, adds Bitext banking
# anchor pairs, trains DPO with AlignTune, and audits the DPO model and its 50/50 merge with the policy.
#   jobs/launch.sh a100-large dpo3 5h bash jobs/stage_dpo3.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
uv pip install --system -q "curatorkit[generation,embedding]==0.1.24" -c requirements.txt
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/train_trilingual.jsonl fixinst/outputs/sft/test_trilingual.jsonl \
  curator2/outputs/curator/prompts.jsonl curator2/outputs/curator/samples/sft.judged.jsonl \
  curator2/outputs/curator/samples/fire.judged.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
C=outputs/curator; mkdir -p $C/samples
cp /tmp/runs/curator2/outputs/curator/prompts.jsonl $C/
cp /tmp/runs/curator2/outputs/curator/samples/*.judged.jsonl $C/samples/

hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy

python -m lexsi_a.curator_data sample --model /tmp/policy --tag final --n 3 --out $C
python -m lexsi_a.curator_data judge --out $C
python -m lexsi_a.curator_data pairs --policy final --fraud-ratio 1.0 --min-fraud 300 \
  --anchors /tmp/runs/fixinst/outputs/sft/train_trilingual.jsonl --n-anchors 150 --out $C

python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model /tmp/policy --out outputs/dpo_lora --merged-out /tmp/dpo \
  --push ashish-soni08/tiny-aya-fire-banking-dpo3 --epochs 3 --lr 5e-5
rm -rf outputs/dpo_lora/checkpoint-*
python scripts/interp.py --a /tmp/dpo --b /tmp/policy --s 0.5 --out /tmp/dpo_mix

for pair in "dpo3:/tmp/dpo" "dpo3_mix05:/tmp/dpo_mix"; do
  tag=${pair%%:*}; model=${pair#*:}
  python -m lexsi_a.gen --model "$model" --tag "$tag"
  python -m lexsi_a.utility --model "$model" --tag "$tag" --test outputs/sft/test_trilingual.jsonl
done
N="results/gens/dpo3.jsonl results/gens/dpo3_mix05.jsonl"
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" $N || echo "judge $j failed"; done
python -m lexsi_a.metrics $N > results/metrics_dpo3.json
