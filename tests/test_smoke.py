import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from stcmixhop.data_ingest import load_fema
from stcmixhop.graph_build import build_paysim_graph


def _tiny_paysim_df(n=240, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": np.arange(n) % 96,
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n),
        "amount": rng.lognormal(5, 1, size=n),
        "nameOrig": [f"C{i%80}" for i in range(n)],
        "oldbalanceOrg": rng.lognormal(6, 1, size=n),
        "newbalanceOrig": rng.lognormal(6, 1, size=n),
        "nameDest": [f"M{i%90}" for i in range(n)],
        "oldbalanceDest": rng.lognormal(6, 1, size=n),
        "newbalanceDest": rng.lognormal(6, 1, size=n),
        "isFraud": (rng.random(n) < 0.08).astype(int),
    })


def test_paysim_smoke(tmp_path):
    df = _tiny_paysim_df()
    data = tmp_path / "paysim_tiny.csv"
    df.to_csv(data, index=False)
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--dataset", "paysim",
        "--data", str(data),
        "--outdir", str(outdir),
        "--max-rows", "240",
        "--quick",
        "--time-bin", "12h",
        "--seeds", "3",
        "--no-graph-baselines",
        "--no-temporal-baselines",
        "--no-tabular",
    ]
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    assert (outdir / "Table_overall_all_seeds.csv").exists()
    assert (outdir / "Figure_overall_mean.png").exists()


def test_sender_only_changes_positive_receiver_labeling():
    df = pd.DataFrame({
        "from": ["a", "b"],
        "to": ["victim", "c"],
        "label": [1, 0],
        "timestamp": pd.to_datetime(["2000-01-01", "2000-01-01"]),
        "amount": [10.0, 1.0],
    })
    both = build_paysim_graph(df, time_bin="1D", label_mode="both", use_type_stats=False)
    sender = build_paysim_graph(df, time_bin="1D", label_mode="sender_only", use_type_stats=False)
    assert both.snapshots[0].y.sum() == 2
    assert sender.snapshots[0].y.sum() == 1


def test_fema_policy_only_raises_clear_error(tmp_path):
    df = pd.DataFrame({
        "policyEffectiveDate": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "reportedZipCode": ["10001", "10002"],
        "totalInsurancePremiumOfThePolicy": [1000, 1200],
        "propertyState": ["NY", "NY"],
    })
    path = tmp_path / "fema_policy_only.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Could not infer a FEMA supervised claim/loss target"):
        load_fema(str(path), max_rows=2)
