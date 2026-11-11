#!/usr/bin/env bash
# Refusal circuits for the dashboard (Fire, banking SFT, final), CircuitKIT EAP-IG + its HTML views.
#   jobs/launch.sh a100-large circuits 2h bash jobs/stage_circuits.sh [extra tag=model ...]
set -euo pipefail
python scripts/circuits.py fire=CohereLabs/tiny-aya-fire sft=ashish-soni08/tiny-aya-fire-banking-sft \
  final=ashish-soni08/tiny-aya-fire-banking-safe "$@" --out results/circuits
