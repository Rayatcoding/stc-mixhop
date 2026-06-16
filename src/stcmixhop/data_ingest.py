from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _first_existing_zip_member(zip_path: str | os.PathLike, candidates: Iterable[str]) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for c in candidates:
            if c in names:
                return c
        raise FileNotFoundError(f"None of {list(candidates)} found in {zip_path}. Members={sorted(names)[:20]}")


def _read_csv_maybe_zip(path: str | os.PathLike, member: str | None = None, **read_csv_kwargs) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".zip"):
        if member is None:
            with zipfile.ZipFile(path) as zf:
                csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise FileNotFoundError(f"No CSV files found inside {path}")
                member = csvs[0]
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as f:
                return pd.read_csv(f, **read_csv_kwargs)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, **read_csv_kwargs)
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path, **read_csv_kwargs)
    return pd.read_csv(path, **read_csv_kwargs)


def _time_stratified_sample(df: pd.DataFrame, time_col: str, max_rows: int, seed: int | None) -> pd.DataFrame:
    """Sample across time buckets instead of taking an initial file prefix."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df.sort_values(time_col).reset_index(drop=True)
    work = df.copy()
    if not np.issubdtype(work[time_col].dtype, np.number):
        t = pd.to_datetime(work[time_col], errors="coerce").astype("int64")
    else:
        t = pd.to_numeric(work[time_col], errors="coerce").fillna(0).astype(float)
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    span = max(tmax - tmin, 1.0)
    buckets = int(max(10, min(200, math.ceil(max_rows / 2000))))
    work["__bucket__"] = np.floor((t - tmin) * buckets / span).astype(int).clip(0, buckets - 1)
    per = max(1, max_rows // buckets)
    parts = []
    rng = np.random.default_rng(seed)
    for _, g in work.groupby("__bucket__", sort=True):
        take = min(per, len(g))
        if take > 0:
            parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts, axis=0, ignore_index=False) if parts else work.sample(n=max_rows, random_state=seed)
    if len(out) < max_rows:
        rest = work.drop(index=out.index, errors="ignore")
        if len(rest) > 0:
            out = pd.concat([out, rest.sample(n=min(max_rows - len(out), len(rest)), random_state=seed)], axis=0)
    return out.drop(columns=["__bucket__"], errors="ignore").sort_values(time_col).reset_index(drop=True)



def _sample_paysim_csv(path: str, max_rows: int, seed: int | None, member: str | None = None) -> pd.DataFrame:
    """Chunked time-spread sampling for large PaySim CSV/zip files."""
    rng = np.random.default_rng(seed)
    chunks = []
    # PaySim is naturally sorted by step; chunk sampling preserves time coverage without loading all rows.
    per_chunk = max(50, int(math.ceil(max_rows / 30)))
    if str(path).lower().endswith(".zip"):
        if member is None:
            member = "paysim.csv"
        zf = zipfile.ZipFile(path)
        fh = zf.open(member)
        close = lambda: (fh.close(), zf.close())
    else:
        fh = path
        close = lambda: None
    try:
        for chunk in pd.read_csv(fh, chunksize=200_000):
            if len(chunk) == 0:
                continue
            take = min(per_chunk, len(chunk))
            chunks.append(chunk.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    finally:
        close()
    if not chunks:
        raise ValueError("No rows read from PaySim file")
    out = pd.concat(chunks, axis=0, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out

def load_paysim(path: str, max_rows: int | None = None, seed: int | None = 42) -> pd.DataFrame:
    """Load PaySim/Kaggle paysim1 and standardize column names.

    Output columns include from, to, label, timestamp, step_raw, and raw transaction features.
    """
    raw = _sample_paysim_csv(path, int(max_rows), seed, member="paysim.csv" if str(path).lower().endswith(".zip") else None) if max_rows is not None else _read_csv_maybe_zip(path, member="paysim.csv" if str(path).lower().endswith(".zip") else None)
    required = {"step", "nameOrig", "nameDest", "isFraud"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Expected PaySim columns {required}; found {list(raw.columns)[:20]}")
    keep = [
        "step", "nameOrig", "nameDest", "isFraud", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    ]
    out = raw[[c for c in keep if c in raw.columns]].copy()
    out = out.rename(columns={
        "step": "step_raw", "nameOrig": "from", "nameDest": "to", "isFraud": "label", "type": "tx_type"
    })
    out["step_raw"] = pd.to_numeric(out["step_raw"], errors="coerce").fillna(0).astype(int)
    if max_rows is not None and len(out) > max_rows:
        out = _time_stratified_sample(out, "step_raw", int(max_rows), seed)
    base = pd.Timestamp("2000-01-01 00:00:00")
    out["timestamp"] = base + pd.to_timedelta(out["step_raw"], unit="h")
    out["from"] = out["from"].astype(str).str.lower().str.strip()
    out["to"] = out["to"].astype(str).str.lower().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "tx_type" in out.columns:
        out["tx_type"] = out["tx_type"].astype(str).str.upper().str.strip()
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "oldbalanceOrg" in out.columns and "newbalanceOrig" in out.columns:
        out["org_delta"] = (out["oldbalanceOrg"] - out["newbalanceOrig"]).astype(float)
    if "oldbalanceDest" in out.columns and "newbalanceDest" in out.columns:
        out["dest_delta"] = (out["newbalanceDest"] - out["oldbalanceDest"]).astype(float)
    return out.reset_index(drop=True)


def load_porto(path: str, max_rows: int | None = None, seed: int | None = 42) -> pd.DataFrame:
    """Load Porto Seguro train.csv from a CSV or Kaggle zip.

    A synthetic chronological index is used because the public dataset is not timestamped.
    This is reported as a cross-domain stress test, not as a true temporal transaction log.
    """
    member = _first_existing_zip_member(path, ["train.csv"]) if str(path).lower().endswith(".zip") else None
    df = _read_csv_maybe_zip(path, member=member)
    if "target" not in df.columns:
        raise ValueError("Porto Seguro loader expects a 'target' column from train.csv")
    df = df.sort_values("id" if "id" in df.columns else df.columns[0]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        # preserve the synthetic ordering but sample across the full index range
        df["__order__"] = np.arange(len(df))
        df = _time_stratified_sample(df, "__order__", int(max_rows), seed).drop(columns="__order__")
    df = df.reset_index(drop=True)
    df["timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(df)), unit="h")
    df["label"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["entity_id"] = df["id"].astype(str) if "id" in df.columns else df.index.astype(str)
    return df


def load_fema(path: str, max_rows: int | None = None, seed: int | None = 42) -> pd.DataFrame:
    """Load FEMA NFIP-like CSV/parquet and infer a binary claim target.

    The loader is intentionally conservative: it excludes obvious payment/loss columns from features
    unless they are used only to construct the target. The manuscript should describe FEMA as an
    exploratory/boundary setting if discrimination is near chance.
    """
    kwargs = {}
    if str(path).lower().endswith(".csv") and max_rows is not None:
        kwargs["nrows"] = max_rows
    df = _read_csv_maybe_zip(path, **kwargs)
    label_cands = ["totalNumberOfClaims", "numberOfClaims", "totalLossAmount", "reportedClaims", "claims"]
    label_col = next((c for c in df.columns if c.lower() in [x.lower() for x in label_cands]), None)
    if label_col is None:
        raise ValueError(
            "Could not infer a FEMA supervised claim/loss target column. "
            f"Expected one of {label_cands}. Policy-only NFIP files cannot support "
            "supervised ROC-AUC/PR-AUC experiments; provide a claims or policy-claims "
            "joined file, or remove FEMA quantitative results from the manuscript."
        )
    df["label"] = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0).astype(int)
    date_col = next((c for c in df.columns if ("date" in c.lower() and "effective" in c.lower())), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is not None:
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(df)), unit="h")
    if max_rows is not None and len(df) > max_rows:
        df = _time_stratified_sample(df, "timestamp", int(max_rows), seed)
    df["entity_id"] = df.index.astype(str)
    return df.reset_index(drop=True)


def load_dataset(dataset: str, path: str, max_rows: int | None = None, seed: int | None = 42) -> pd.DataFrame:
    ds = dataset.lower().strip()
    if ds == "paysim":
        return load_paysim(path, max_rows=max_rows, seed=seed)
    if ds == "porto":
        return load_porto(path, max_rows=max_rows, seed=seed)
    if ds == "fema":
        return load_fema(path, max_rows=max_rows, seed=seed)
    raise ValueError(f"Unknown dataset '{dataset}'. Use paysim, porto, or fema.")
