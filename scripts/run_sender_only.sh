#!/usr/bin/env bash
set -euo pipefail
# Reviewer-1 label-noise check: sender-only PaySim node labels.
PYTHONPATH=src python run_experiment.py \
  --dataset paysim \
  --data data/raw/paysim.csv \
  --outdir outputs/paysim_sender_only \
  --max-rows 200000 \
  --time-bin 7D \
  --label-mode sender_only \
  --epochs-pre 10 \
  --epochs-sup 20 \
  --K 2 \
  --dk 128 \
  --seeds 42 43 44 \
  --profile
