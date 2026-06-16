# STC-MixHop IJDSA Revision Reproducibility Package

This repository contains a cleaned implementation for the IJDSA major revision of **STC-MixHop**, a stability-focused spatio-temporal graph learning framework for fraud/risk detection under chronological evaluation.

The revision strategy is intentionally conservative:

- **PaySim** remains the primary chronological transaction-network experiment.
- **Porto Seguro** is retained as a cross-domain, attribute-dominant stress test.
- **FEMA/NFIP quantitative results are removed from the main manuscript** unless a labeled claims or policy-claims joined table is available. The inspected policy-only FEMA file does not contain a valid claim/loss target, so the code refuses to fabricate labels.

The package addresses the reviewers' reproducibility and analysis concerns with executable scripts, explicit preprocessing, chronological splitting, validation-threshold selection, temporal baselines, sender-only label variants, loss logs, plots, statistical tests, and profiling hooks.

## 1. Repository structure

```text
STC-MixHop-IJDSA-revision/
  run_experiment.py                  # unified CLI runner for PaySim/Porto and labeled FEMA only
  FEMA_REMOVAL_NOTE.md               # manuscript-facing FEMA decision note
  FEMA_DATA_NOTE.md                  # data-validity note for FEMA/NFIP
  REVISION_CHECKLIST.md              # reviewer-to-action mapping
  src/stcmixhop/
    data_ingest.py                   # PaySim / Porto / conservative FEMA loaders
    graph_build.py                   # dynamic graph construction
    models.py                        # STC-MixHop and baseline models
    train.py                         # pretraining, fine-tuning, evaluation, profiling
    baselines.py                     # LogReg, RF, MLP
    plotting.py                      # publication-readable figures
    stats.py                         # mean±std, ranking, paired tests
  scripts/
    run_revision_paysim.sh           # example PaySim run
    run_sender_only.sh               # reviewer label-mode check
    check_fema_policy_file.py        # verifies whether FEMA has a real supervised target
  tests/
    test_smoke.py                    # tiny synthetic smoke tests
```

Large datasets are intentionally not committed. Place local datasets under `data/raw/` or pass absolute paths with `--data`.

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you use Anaconda, run `conda deactivate` before activating `.venv`, so `python` and `pytest` come from the venv (not `D:\anaconda3`).

The code uses common dependencies: NumPy, pandas, scikit-learn, SciPy, matplotlib, PyTorch, pyarrow, and pytest.

## 3. PaySim main experiment

Input may be the Kaggle PaySim CSV or a zip containing `paysim.csv`.

```bash
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
```

Main outputs:

- `Table_overall_all_seeds.csv`
- `Table_multiseed_mean_std.csv`
- `Table_average_ranking.csv`
- `Stats_STC_vs_*.csv`
- `Figure_overall_mean.png`
- per-seed loss curves, prediction violin plots, embedding PCA plots, logs, and graph metadata.

## 4. Sender-only label variant

This addresses the review concern that marking both senders and receivers as positive can conflate fraudster and victim roles.

```bash
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
```

Supported label modes:

- `both`: sender and receiver are positive when a transaction is fraudulent;
- `sender_only`: only fraudulent transaction originators are positive;
- `receiver_only`: only fraudulent transaction recipients are positive.

## 5. Porto Seguro cross-domain stress test

```bash
PYTHONPATH=src python run_experiment.py \
  --dataset porto \
  --data data/raw/porto-seguro-safe-driver-prediction.zip \
  --outdir outputs/porto_revision \
  --max-rows 50000 \
  --n-snapshots 5 \
  --epochs-pre 10 \
  --epochs-sup 20 \
  --seeds 42 43 44 \
  --profile
```

Porto Seguro has no public timestamp. The runner uses a synthetic chronological order by row/id and constructs entity-similarity graphs from low-cardinality attributes. In the revised manuscript, Porto should be framed as a cross-domain stress test, not as a true transaction-network deployment simulation.

## 6. FEMA/NFIP: removed from quantitative claims

For the fast major revision, FEMA/NFIP quantitative results should be removed from the main manuscript.

The inspected FEMA/NFIP policy file contains policy, premium/coverage, date, geography, and identifier fields, but no verified claim-count or loss target. Therefore, supervised ROC-AUC/PR-AUC, ablation, or sensitivity results on that policy-only file would be misleading.

Use the audit script to document this decision:

```bash
PYTHONPATH=src python scripts/check_fema_policy_file.py \
  --data data/raw/FimaNfipPoliciesV2.parquet \
  --nrows 10000 \
  --out outputs/fema_policy_audit.md
```

The main runner supports `--dataset fema` **only if** the provided file contains a real supervised target such as `totalNumberOfClaims`, `numberOfClaims`, or `totalLossAmount`. If the file is policy-only, it raises a clear error instead of creating random labels.

Recommended manuscript action:

> Remove FEMA quantitative tables and figures from the main text. Mention FEMA/NFIP only as a limitation/future extension requiring a labeled claims or policy-claims joined table.

## 7. Baselines included

The runner includes:

- Tabular: Logistic Regression, Random Forest, MLP.
- Static graph: GCN, GraphSAGE, GAT-lite.
- Temporal graph baselines: TemporalGCN-GRU, DySAT-lite, EvolveGCN-lite.
- Self-supervised graph variants: NT-Xent and DGI.
- Proposed: STC-MixHop.

The temporal baselines are dependency-light implementations intended to make the revision reproducible without PyTorch Geometric or specialized temporal-graph libraries. If the manuscript claims comparison to official EvolveGCN/TGAT/TGN/DySAT implementations, those official implementations should be run separately or the text should explicitly describe these as lightweight temporal baselines.

## 8. Reviewer-oriented outputs

| Reviewer concern | Code support |
|---|---|
| Reproducibility | CLI runner, config JSON, graph metadata, logs, output tables |
| Hyperparameters | All CLI arguments saved in `config.json` |
| Sender/receiver label noise | `--label-mode sender_only` |
| Temporal graph baselines | `TemporalGCN-GRU`, `DySAT-lite`, `EvolveGCN-lite` |
| Statistical tests | `Stats_STC_vs_*.csv`, `Table_average_ranking.csv` |
| Training/validation curves | `logs_*.csv`, `loss_curve_*.png` |
| Prediction distribution | `violin_predictions_*.png` |
| Embedding visualization | `embedding_pca_*.png` |
| Efficiency/deployment metrics | `--profile` adds inference time and CUDA memory columns |
| FEMA near-chance concern | FEMA quantitative claims removed; policy-only data audit included |

## 9. Smoke test

```bash
python -m pytest tests/test_smoke.py -v
```

Use `python -m pytest` (not bare `pytest`) so the venv's pytest runs instead of Anaconda's.

A quick manual run on a real local PaySim zip can be done with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_experiment.py \
  --dataset paysim \
  --data /path/to/paysim.zip \
  --outdir outputs/paysim_quick \
  --max-rows 300 \
  --quick \
  --time-bin 2D \
  --seeds 1 \
  --no-graph-baselines \
  --no-temporal-baselines \
  --no-tabular
```

## 10. Manuscript-positioning note

The revised framing should avoid claiming universal superiority over strong tabular baselines in attribute-dominant datasets. Recommended framing:

> STC-MixHop is a stability-focused framework for evaluating when multi-hop structure, temporal smoothing, and auxiliary contrastive regularization help in non-stationary risk detection. Its value is conditional on whether graph construction captures meaningful relational signal.
