#!/usr/bin/env python
"""Inspect a FEMA/NFIP file and report whether it can support supervised evaluation.

This script is intentionally lightweight. It reads only a small preview from CSV/parquet files,
checks for common claim/loss target columns, and writes a short markdown report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_CANDIDATES = [
    "totalNumberOfClaims",
    "numberOfClaims",
    "totalLossAmount",
    "reportedClaims",
    "claims",
]


def read_preview(path: Path, nrows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=nrows))
        return batch.to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported FEMA file type: {path.suffix}. Use .csv or .parquet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to FEMA/NFIP CSV or parquet file")
    p.add_argument("--nrows", type=int, default=10000)
    p.add_argument("--out", default="fema_policy_audit.md")
    args = p.parse_args()

    path = Path(args.data)
    df = read_preview(path, args.nrows)
    lower_to_actual = {c.lower(): c for c in df.columns}
    target = next((lower_to_actual[c.lower()] for c in TARGET_CANDIDATES if c.lower() in lower_to_actual), None)

    date_cols = [c for c in df.columns if "date" in c.lower()]
    geo_cols = [c for c in df.columns if any(k in c.lower() for k in ["zip", "state", "county", "community", "latitude", "longitude"])]

    lines = [
        "# FEMA/NFIP policy-file audit",
        "",
        f"File: `{path}`",
        f"Preview rows inspected: {len(df)}, columns: {len(df.columns)}",
        "",
        "## Supervised target check",
    ]
    if target is None:
        lines += [
            "",
            "No recognized claim/loss target column was found.",
            "",
            "This file should **not** be used for supervised ROC-AUC/PR-AUC experiments, ablations, or sensitivity analysis.",
            "Use it only as an unlabeled policy table unless a labeled claims table or policy-claims joined table is available.",
        ]
    else:
        rate = (pd.to_numeric(df[target], errors="coerce").fillna(0) > 0).mean()
        lines += ["", f"Found candidate target column: `{target}`", f"Positive rate in preview: {rate:.6f}"]

    lines += [
        "",
        "## Date-like columns",
        "",
        "- " + "\n- ".join(date_cols[:30]) if date_cols else "None found.",
        "",
        "## Geography-like columns",
        "",
        "- " + "\n- ".join(geo_cols[:30]) if geo_cols else "None found.",
        "",
        "## First columns",
        "",
        "- " + "\n- ".join(list(df.columns)[:80]),
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    if target is None:
        print("No supervised target found; remove FEMA quantitative results or provide a labeled claims table.")


if __name__ == "__main__":
    main()
