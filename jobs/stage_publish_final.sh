#!/usr/bin/env bash
# Rebuild the final model (DPO round 3) and publish it PUBLIC as ashish-soni08/tiny-aya-fire-banking-safe.
# final = harden3 (SFT + safety mix v3) 70/30 with Fire, + DPO round-3 LoRA merged in.
#   jobs/launch.sh a100-large publish-final 2h bash jobs/stage_publish_final.sh
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" dpo3/outputs/dpo_lora/adapter_config.json dpo3/outputs/dpo_lora/adapter_model.safetensors \
  --local-dir /tmp/runs --quiet
hf download ashish-soni08/tiny-aya-fire-banking-harden3 --local-dir /tmp/harden3 --quiet && rm -rf /tmp/harden3/.cache
python scripts/interp.py --a /tmp/harden3 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/policy
python -c "from lexsi_a.sft import merge; import pathlib; merge('/tmp/policy', pathlib.Path('/tmp/runs/dpo3/outputs/dpo_lora'), '/tmp/final', None)"
python - <<'PY'
from huggingface_hub import HfApi
api, rid = HfApi(), "ashish-soni08/tiny-aya-fire-banking-safe"
api.update_repo_settings(rid, private=False)   # public first: the private storage quota is full
api.upload_folder(repo_id=rid, folder_path="/tmp/final", ignore_patterns=[".cache/*", "README.md"],
                  commit_message="Final model: safety mix v3 + 70/30 Fire merge + CuratorKIT/AlignTune DPO (round 3)")
api.upload_file(path_or_fileobj="cards/model_safe.md", path_in_repo="README.md", repo_id=rid,
                commit_message="Model card for the DPO round-3 final model")
print("published", rid, "private =", api.model_info(rid).private)
PY
