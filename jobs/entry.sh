#!/usr/bin/env bash
# Runs inside an HF Job: clone this repo at $GIT_REF, install pinned deps, run "$@",
# then upload results/ and outputs/ to the private dataset repo $RUNS_REPO under $RUN_NAME/.
set -euo pipefail
: "${GIT_REF:=build/idea-a}" "${RUNS_REPO:=ashish-soni08/lexsi-a-runs}" "${RUN_NAME:=run-$(date +%s)}"
apt-get update -qq && apt-get install -y -qq git >/dev/null
pip install -q uv
git clone -q --depth 1 --branch "$GIT_REF" https://github.com/Ashish-Soni08/lexsi-safe-hinglish-banking /work
cd /work
uv pip install --system -q -r requirements.txt "huggingface_hub[cli]"
export PYTHONPATH=/work/src HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results outputs
status=0
"$@" 2>&1 | tee results/job.log || status=$?
hf upload --repo-type dataset --private "$RUNS_REPO" results "$RUN_NAME/results" --quiet || true
[ -n "$(ls -A outputs)" ] && hf upload --repo-type dataset --private "$RUNS_REPO" outputs "$RUN_NAME/outputs" --quiet || true
exit $status
