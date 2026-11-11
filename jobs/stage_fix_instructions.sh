#!/usr/bin/env bash
# Re-translate hi/hinglish instructions that Aya answered instead of translating.
set -euo pipefail
hf download --repo-type dataset ashish-soni08/lexsi-a-runs --include "hinglish/outputs/sft/*" --local-dir /tmp/prev --quiet
python -m lexsi_a.translate --out outputs/sft --fix-instructions /tmp/prev/hinglish/outputs/sft
