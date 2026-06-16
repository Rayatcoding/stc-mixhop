from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def summarize_multiseed(df: pd.DataFrame, group_col: str = "Model", metrics=None) -> pd.DataFrame:
    if metrics is None:
        metrics = ["AUC", "PR_AUC", "Fbeta", "F1", "Precision", "Recall", "Accuracy"]
    rows = []
    for model, g in df.groupby(group_col):
        row = {group_col: model, "n": len(g)}
        for m in metrics:
            if m in g.columns:
                vals = pd.to_numeric(g[m], errors="coerce").dropna()
                row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def average_ranking(df: pd.DataFrame, group_col: str = "Model", metrics=None, higher_is_better=None) -> pd.DataFrame:
    if metrics is None:
        metrics = ["AUC", "PR_AUC", "Fbeta", "F1", "Precision", "Recall", "Accuracy"]
    if higher_is_better is None:
        higher_is_better = {m: True for m in metrics}
    means = df.groupby(group_col)[metrics].mean(numeric_only=True)
    ranks = []
    for m in metrics:
        if m not in means.columns:
            continue
        ranks.append(means[m].rank(ascending=not higher_is_better.get(m, True)).rename(m))
    if not ranks:
        return pd.DataFrame()
    r = pd.concat(ranks, axis=1)
    out = r.assign(avg_rank=r.mean(axis=1)).reset_index().sort_values("avg_rank")
    return out


def paired_tests(df: pd.DataFrame, baseline: str, challenger: str, group_col: str = "Model", seed_col: str = "Seed", metrics=None) -> pd.DataFrame:
    if metrics is None:
        metrics = ["AUC", "PR_AUC", "Fbeta", "F1", "Precision", "Recall"]
    rows = []
    b = df[df[group_col] == baseline].set_index(seed_col)
    c = df[df[group_col] == challenger].set_index(seed_col)
    common = b.index.intersection(c.index)
    for m in metrics:
        if m not in b.columns or m not in c.columns or len(common) < 2:
            continue
        x = pd.to_numeric(b.loc[common, m], errors="coerce")
        y = pd.to_numeric(c.loc[common, m], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask].to_numpy(float); y = y[mask].to_numpy(float)
        if len(x) < 2:
            continue
        try:
            t_p = float(ttest_rel(y, x, nan_policy="omit").pvalue)
        except Exception:
            t_p = np.nan
        try:
            w_p = float(wilcoxon(y, x).pvalue) if np.any(np.abs(y - x) > 1e-12) else 1.0
        except Exception:
            w_p = np.nan
        rows.append({"baseline": baseline, "challenger": challenger, "metric": m, "n_pairs": len(x), "mean_delta": float(np.mean(y - x)), "paired_t_p": t_p, "wilcoxon_p": w_p})
    return pd.DataFrame(rows)
