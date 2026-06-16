# Data (not committed)

Place local datasets under `data/raw/` and pass paths to `run_experiment.py --data`.

| Dataset | Typical path | Source |
|---------|--------------|--------|
| PaySim | `data/raw/paysim.csv` or `paysim.zip` | [Kaggle PaySim](https://www.kaggle.com/datasets/ealaxi/paysim-fraud-detection) |
| Porto Seguro | `data/raw/porto-seguro-safe-driver-prediction.zip` | [Kaggle Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) |
| FEMA/NFIP (audit only) | `data/raw/fema_head_10k.parquet` | FEMA OpenFEMA; policy-only files lack supervised claim targets |

Raw files are excluded by `.gitignore` due to size and licensing.
