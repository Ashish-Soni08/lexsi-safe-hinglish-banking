#!/usr/bin/env bash
# Stage 5e: DPO round 5, from the round-3 final model. Prompts it gets wrong only (harmless prompts it refuses,
# Hindi/Hinglish harmful prompts it answers), 1:1 harm:edgy, chosen answers gated by an Aya quality rating, plus a
# slice of round-3 pairs and banking anchors to hold what already works. Picked on a dev set that includes unseen
# hard harmless prompts in en/hi/hinglish, with a no-worse-over-refusal rule; scored once on the test suites.
#   jobs/launch.sh a100-large dpo5 7h bash jobs/stage_dpo5.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
uv pip install --system -q "curatorkit[generation,embedding]==0.1.24" -c requirements.txt
hf download --repo-type dataset "$RUNS" fixinst/outputs/sft/train_trilingual.jsonl fixinst/outputs/sft/test_trilingual.jsonl \
  dpo3/outputs/curator/pairs.jsonl dpo3/outputs/dpo_lora/adapter_config.json dpo3/outputs/dpo_lora/adapter_model.safetensors \
  dpo4/outputs/curator/prompts.jsonl curator2/outputs/curator/prompts.jsonl --local-dir /tmp/runs --quiet
mkdir -p outputs/sft && cp /tmp/runs/fixinst/outputs/sft/test_trilingual.jsonl outputs/sft/
hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy
rm -rf /tmp/harden3
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo3/outputs/dpo_lora'), '/tmp/R3', None)"
rm -rf /tmp/policy

C=outputs/curator
python -m lexsi_a.round5 --r4 /tmp/runs/dpo4/outputs/curator/prompts.jsonl \
  --seen /tmp/runs/dpo3/outputs/curator/pairs.jsonl /tmp/runs/curator2/outputs/curator/prompts.jsonl --out $C
python -m lexsi_a.curator_data sample --model /tmp/R3 --tag r3 --n 4 --out $C
python -m lexsi_a.curator_data sample --model CohereLabs/tiny-aya-fire --tag fire --n 2 --out $C
python -m lexsi_a.curator_data sample --model CohereLabs/aya-expanse-8b --tag aya --n 2 --out $C
python -m lexsi_a.curator_data judge --out $C
python -m lexsi_a.curator_data rate --out $C
python -m lexsi_a.curator_data pairs --policy r3 --failures-only --harm-ratio 1.0 --fraud-ratio 0 --min-quality 4 \
  --extra-pairs /tmp/runs/dpo3/outputs/curator/pairs.jsonl --n-extra 400 \
  --anchors /tmp/runs/fixinst/outputs/sft/train_trilingual.jsonl --n-anchors 100 --out $C

# Two gentle DPO passes from round 3 (higher beta = stays closer to it).
python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model /tmp/R3 --out outputs/dpo5c_lora --merged-out /tmp/C --epochs 2 --lr 2e-5 --beta 0.25
python -m lexsi_a.dpo --train $C/dpo_train.jsonl --model /tmp/R3 --out outputs/dpo5d_lora --merged-out /tmp/D --epochs 2 --lr 3e-5 --beta 0.5
rm -rf outputs/dpo5*_lora/checkpoint-*

for v in R3 C D; do python -m lexsi_a.dev_select gen --model /tmp/$v --tag $v --dev $C/dev.jsonl; done
for v in C D; do
  python scripts/interp.py --a /tmp/$v --b /tmp/R3 --s 0.5 --out /tmp/${v}m
  python -m lexsi_a.dev_select gen --model /tmp/${v}m --tag ${v}m --dev $C/dev.jsonl
  rm -rf /tmp/${v}m
done
python -m lexsi_a.dev_select score
# Ship rule on dev: best score among variants whose over-refusal (banking, Hindi, English look-alikes) is no more
# than 2 points above round 3's; otherwise keep round 3.
BEST=$(python - <<'PY'
import json
t = json.load(open("results/dev_scores.json")); r3 = t["R3"]
ok = [v for v in t if v != "R3" and all((t[v][k] or 0) <= (r3[k] or 0) + 0.02 for k in ("banking_over", "hindi_over", "general_en_over"))]
print(max(ok, key=lambda v: t[v]["score"]) if ok and max(t[v]["score"] for v in ok) > r3["score"] else "R3")
PY
)
echo "DEV PICK: $BEST"
echo "$BEST" > results/dpo5_variant.txt
[ "$BEST" = R3 ] && { echo "round 3 kept; no test eval"; exit 0; }
case $BEST in Cm|Dm) python scripts/interp.py --a /tmp/${BEST%m} --b /tmp/R3 --s 0.5 --out /tmp/$BEST ;; esac

python -m lexsi_a.gen --model /tmp/$BEST --tag dpo5
python -m lexsi_a.utility --model /tmp/$BEST --tag dpo5 --test outputs/sft/test_trilingual.jsonl
for j in qwen3guard llamaguard aya; do python -m lexsi_a.judge --judge "$j" results/gens/dpo5.jsonl || echo "judge $j failed"; done
python -m lexsi_a.metrics results/gens/dpo5.jsonl > results/metrics_dpo5.json
