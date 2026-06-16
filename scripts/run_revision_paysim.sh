#!/usr/bin/env bash
set -euo pipefail
# Example full PaySim revision run. Adjust --data to your local path.
PYTHONPATH=src python run_experiment.py \
  --dataset paysim \
  --data data/raw/paysim.csv \
  --outdir outputs/paysim_revision \
  --max-rows 200000 \
  --time-bin 7D \
  --label-mode both \
  --epochs-pre 10 \
  --epochs-sup 20 \
  --K 2 \
  --dk 128 \
  --seeds 42 43 44 \
  --profile
