# Checked runs before packaging

The following checks were run in the sandbox before packaging this revision repo.

## Unit / smoke tests

```bash
cd STC-MixHop-IJDSA-revision
PYTHONPATH=src pytest -q
```

Result: `3 passed`.

## PaySim quick run

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiment.py \
  --dataset paysim \
  --data /mnt/data/paysim.zip \
  --outdir /mnt/data/v3_test_paysim \
  --max-rows 300 \
  --quick \
  --time-bin 2D \
  --seeds 1 \
  --no-graph-baselines \
  --no-temporal-baselines \
  --no-tabular
```

Result: completed and wrote output tables/figures.

## PaySim sender-only quick run

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiment.py \
  --dataset paysim \
  --data /mnt/data/paysim.zip \
  --outdir /mnt/data/v3_test_sender \
  --max-rows 500 \
  --quick \
  --time-bin 2D \
  --label-mode sender_only \
  --seeds 1 \
  --no-graph-baselines \
  --no-temporal-baselines \
  --no-tabular
```

Result: completed and wrote output tables/figures.

## Porto Seguro quick run

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiment.py \
  --dataset porto \
  --data /mnt/data/porto-seguro-safe-driver-prediction.zip \
  --outdir /mnt/data/v3_test_porto \
  --max-rows 300 \
  --quick \
  --n-snapshots 3 \
  --seeds 1 \
  --no-graph-baselines \
  --no-temporal-baselines \
  --no-tabular
```

Result: completed and wrote output tables/figures.

## FEMA policy-only audit

```bash
PYTHONPATH=src python scripts/check_fema_policy_file.py \
  --data /mnt/data/fema_head_10k.csv \
  --nrows 1000 \
  --out /mnt/data/v3_fema_audit/fema_policy_audit.md
```

Result: no supervised target was found. The repo therefore follows Route B: remove FEMA quantitative claims from the main manuscript and mention FEMA/NFIP only as a limitation/future extension requiring labeled claims or policy-claims joined data.

## FEMA policy-only runner behavior

```bash
PYTHONPATH=src python run_experiment.py \
  --dataset fema \
  --data /mnt/data/fema_head_10k.csv \
  --outdir /mnt/data/v3_test_fema \
  --max-rows 1000 \
  --quick
```

Result: intentionally fails with a clear `ValueError` explaining that policy-only FEMA cannot support supervised ROC-AUC/PR-AUC experiments.

## PaySim full-baseline quick run

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiment.py \
  --dataset paysim \
  --data /mnt/data/paysim.zip \
  --outdir /mnt/data/v3_test_paysim_fullquick \
  --max-rows 500 \
  --quick \
  --time-bin 1D \
  --seeds 1 \
  --no-tabular
```

Result: completed and exercised GCN, GraphSAGE, GAT, TemporalGCN-GRU, DySAT-lite, EvolveGCN-lite, supervised-only, NT-Xent, DGI, and STC-MixHop code paths.
