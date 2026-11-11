#!/usr/bin/env bash
# Build the final model (70% harden2 + 30% Fire) and publish models, datasets and the collection.
#   jobs/launch.sh cpu-upgrade publish 1h bash jobs/stage_publish.sh [--public]
set -euo pipefail
RUNS=ashish-soni08/lexsi-a-runs
hf download --repo-type dataset "$RUNS" --include "fixinst/outputs/sft/*" --local-dir /tmp/runs --quiet
hf download --repo-type dataset "$RUNS" --include "harden2/outputs/harden/*" --local-dir /tmp/runs --quiet
hf download ashish-soni08/tiny-aya-fire-banking-harden2 --local-dir /tmp/harden2 --quiet
hf download ashish-soni08/tiny-aya-fire-banking-sft --local-dir /tmp/sft --quiet
rm -rf /tmp/harden2/.cache /tmp/sft/.cache
python scripts/interp.py --a /tmp/harden2 --b CohereLabs/tiny-aya-fire --s 0.3 --out /tmp/final
python scripts/publish.py --runs /tmp/runs "$@"
