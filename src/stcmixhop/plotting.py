from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def save_table_csv(rows, path: str) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    ensure_dir(path)
    df.to_csv(path, index=False)
    return df


def _rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in {"auc", "auc-roc", "roc_auc", "roc-auc"}: cmap[c] = "ROC-AUC"
        elif cl in {"pr_auc", "pr-auc", "prauc"}: cmap[c] = "PR-AUC"
        elif cl in {"fbeta", "f_beta", "fβ", "f"}: cmap[c] = "Fbeta"
        elif cl == "model" or cl == "method": cmap[c] = "Model"
        elif cl == "variant": cmap[c] = "Variant"
    return df.rename(columns=cmap)


def plot_overall_figure(table_df: pd.DataFrame, out_png: str, beta: float = 0.5) -> None:
    """Readable overall comparison figure for revision: horizontal bars avoid overlapping labels."""
    df = _rename_metrics(table_df.copy())
    metrics = [("ROC-AUC", "ROC-AUC"), ("PR-AUC", "PR-AUC"), ("Fbeta", rf"F$_{{{beta:g}}}$"), ("Recall", "Recall")]
    missing = [m for m, _ in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; have {list(df.columns)}")
    methods = df["Model"].astype(str).tolist()
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 4, figsize=(17, max(4.8, 0.45 * len(methods))), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        vals = np.array([as_float(v) for v in df[col]])
        ax.barh(y, vals, edgecolor="black", linewidth=0.7)
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(min(v + 0.015, 1.02), i, f"{v:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods, fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_ablation_figure(table_df: pd.DataFrame, out_png: str) -> None:
    df = _rename_metrics(table_df.copy())
    if "Variant" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "Variant"})
    metrics = ["ROC-AUC", "PR-AUC", "F1", "Recall"]
    missing = [m for m in ["Variant"] + metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; have {list(df.columns)}")
    variants = df["Variant"].astype(str).tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(variants), 1)
    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    for j, vname in enumerate(variants):
        vals = [as_float(df.loc[j, m]) for m in metrics]
        ax.bar(x + (j - (len(variants) - 1) / 2.0) * width, vals, width=width, label=vname, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Component ablation study", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_sensitivity(df: pd.DataFrame, xcol: str, out_png: str, x_label: str) -> None:
    df = _rename_metrics(df.copy()).sort_values(xcol)
    metrics = [("ROC-AUC", "ROC-AUC"), ("PR-AUC", "PR-AUC"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1")]
    xs = pd.to_numeric(df[xcol]).tolist()
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for col, label in metrics:
        if col in df.columns:
            ax.plot(xs, [as_float(v) for v in df[col]], marker="o", label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_loss_curve(log_df: pd.DataFrame, out_png: str) -> None:
    if log_df.empty or "loss" not in log_df.columns:
        return
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for stage, g in log_df.groupby("stage"):
        ax.plot(g["epoch"], g["loss"], marker="o", label=str(stage))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_prediction_violin(pred_df: pd.DataFrame, out_png: str) -> None:
    if pred_df.empty:
        return
    raw_groups = [pred_df.loc[pred_df["y_true"] == 0, "prob"].astype(float).values, pred_df.loc[pred_df["y_true"] == 1, "prob"].astype(float).values]
    groups = []
    labels = []
    for arr, lab in zip(raw_groups, ["Negative", "Positive"]):
        if len(arr) > 0:
            groups.append(arr)
            labels.append(lab)
    if not groups:
        return
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.violinplot(groups, showmeans=True, showmedians=True)
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Predicted fraud probability")
    ax.set_title("Prediction score distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_embedding_pca(embeddings: np.ndarray, labels: np.ndarray, out_png: str) -> None:
    if embeddings is None or embeddings.size == 0 or embeddings.shape[0] < 3:
        return
    z = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels = labels.astype(int)
    for lab, name in [(0, "Negative"), (1, "Positive")]:
        mask = labels == lab
        if mask.any():
            ax.scatter(z[mask, 0], z[mask, 1], s=8, alpha=0.45, label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Test embedding PCA")
    ax.legend(frameon=True)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
