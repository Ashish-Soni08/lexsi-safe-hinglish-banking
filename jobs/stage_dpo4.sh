#!/usr/bin/env bash
# Stage 5d: DPO round 4. Contrastive Hindi/Hinglish safety pairs (harmful -> refuse, look-alike harmless -> answer)
# added to the round-3 pairs, two DPO strengths x two merge ratios, picked on DEV prompts only, then the chosen
# variant is scored once on the full test suites.
#   jobs/launch.sh a100-large dpo4 7h bash jobs/stage_dpo4.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
uv pip install --system -q "curatorkit[generation,embedding]==0.1.24" -c requirements.txt
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/test_trilingual.jsonl harden3/outputs/harden/safety_mix.jsonl \
  dpo3/outputs/curator/pairs.jsonl dpo3/outputs/dpo_lora/adapter_config.json dpo3/outputs/dpo_lora/adapter_model.safetensors \
  --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy
rm -rf /tmp/harden3

C=outputs/curator
python -m lexsi_a.hindi_round --mix-v3 /tmp/runs/harden3/outputs/harden/safety_mix.jsonl --out $C
python -m lexsi_a.curator_data sample --model /tmp/policy --tag final --n 3 --out $C
python -m lexsi_a.curator_data sample --model ashish-soni08/tiny-aya-fire-banking-sft --tag sft --n 2 --out $C
python -m lexsi_a.curator_data sample --model CohereLabs/tiny-aya-fire --tag fire --n 2 --out $C
python -m lexsi_a.curator_data judge --out $C
python -m lexsi_a.curator_data pairs --policy final --fraud-ratio 0 --extra-pairs /tmp/runs/dpo3/outputs/curator/pairs.jsonl --out $C

# Two DPO strengths (beta 0.1 as in round 3; beta 0.25 stays closer to the policy).
python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model /tmp/policy --out outputs/dpo4a_lora --merged-out /tmp/A --epochs 3 --lr 5e-5 --beta 0.1
python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model /tmp/policy --out outputs/dpo4b_lora --merged-out /tmp/B --epochs 3 --lr 5e-5 --beta 0.25
rm -rf outputs/dpo4*_lora/checkpoint-*
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo3/outputs/dpo_lora'), '/tmp/R3', None)"

# Dev selection: round-3 final (reference), A, B, and each merged 70/30 back toward the policy.
for v in R3 A B; do python -m lexsi_a.dev_select gen --model /tmp/$v --tag $v --dev $C/dev.jsonl; done
for v in A B; do
  python scripts/interp.py --a /tmp/$v --b /tmp/policy --s 0.3 --out /tmp/${v}m
  python -m lexsi_a.dev_select gen --model /tmp/${v}m --tag ${v}m --dev $C/dev.jsonl
  rm -rf /tmp/${v}m
done
python -m lexsi_a.dev_select score
BEST=$(cat results/dev_best.txt)
echo "DEV BEST: $BEST"
case $BEST in Am|Bm) python scripts/interp.py --a /tmp/${BEST%m} --b /tmp/policy --s 0.3 --out /tmp/$BEST ;; esac

# Score the dev-selected variant once on the full test suites.
python -m lexsi_a.gen --model /tmp/$BEST --tag dpo4
python -m lexsi_a.utility --model /tmp/$BEST --tag dpo4 --test outputs/sft/test_trilingual.jsonl
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" results/gens/dpo4.jsonl || echo "judge $j failed"; done
python -m lexsi_a.metrics results/gens/dpo4.jsonl > results/metrics_dpo4.json
echo "$BEST" > results/dpo4_variant.txt
