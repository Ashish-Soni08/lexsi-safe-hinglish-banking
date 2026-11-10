#!/usr/bin/env bash
# Launch a step on HF Jobs.   jobs/launch.sh <flavor> <run-name> <timeout> <command...>
# e.g. jobs/launch.sh l40sx1 smoke 1h python scripts/smoke_test.py
set -euo pipefail
flavor=$1; name=$2; timeout=$3; shift 3
ref=$(git rev-parse --abbrev-ref HEAD)
hf jobs run --detach --flavor "$flavor" --timeout "$timeout" --name "lexsi-a-$name" \
  --secrets HF_TOKEN -e GIT_REF="$ref" -e RUN_NAME="$name" \
  python:3.12 bash -c "curl -fsSL https://raw.githubusercontent.com/Ashish-Soni08/lexsi-safe-hinglish-banking/$ref/jobs/entry.sh | bash -s -- $*"
