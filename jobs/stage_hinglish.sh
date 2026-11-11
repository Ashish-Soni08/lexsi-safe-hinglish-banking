#!/usr/bin/env bash
# Redo only the Hinglish half of the translation, keeping the earlier en/hi rows.
#   jobs/launch.sh a100-large hinglish 2h bash jobs/stage_hinglish.sh
set -euo pipefail
hf download --repo-type dataset ashish-soni08/lexsi-a-runs --include "translate/outputs/sft/*" --local-dir /tmp/prev --quiet
python -m lexsi_a.translate --n-train 3000 --n-test 300 --out outputs/sft --hinglish-only /tmp/prev/translate/outputs/sft
