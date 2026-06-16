from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Snapshot:
    t_idx: int
    node_ids: np.ndarray          # global node ids present in this snapshot
    edges_src: np.ndarray         # local source indices
    edges_dst: np.ndarray         # local destination indices
    x: np.ndarray                 # [n, F]
    y: np.ndarray                 # [n]
    global_to_local: dict


@dataclass
class DynamicGraph:
    snapshots: list[Snapshot]
    num_nodes_global: int
    feat_dim: int
    meta: dict


def _safe_mean(sum_v: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    return sum_v / np.maximum(cnt, 1.0)


def build_paysim_graph(
    tx: pd.DataFrame,
    time_bin: str = "7D",
    use_type_stats: bool = True,
    label_mode: str = "both",
) -> DynamicGraph:
    """Build time-binned directed entity graph for PaySim.

    label_mode:
      - both: sender and receiver are positive if either participates in a fraudulent tx
      - sender_only: only fraudulent transaction originators are positive
      - receiver_only: only fraudulent transaction recipients are positive
    """
    if label_mode not in {"both", "sender_only", "receiver_only"}:
        raise ValueError("label_mode must be one of: both, sender_only, receiver_only")
    tx = tx.copy()
    if "timestamp" not in tx.columns:
        raise ValueError("Expected timestamp column. Use data_ingest.load_paysim first.")
    tx["bin"] = tx["timestamp"].dt.to_period(time_bin).dt.start_time

    addrs = pd.Index(pd.concat([tx["from"], tx["to"]], axis=0).unique())
    addr_to_gid = {a: i for i, a in enumerate(addrs)}
    has_amount = "amount" in tx.columns
    has_org_delta = "org_delta" in tx.columns
    has_dest_delta = "dest_delta" in tx.columns
    has_type = use_type_stats and ("tx_type" in tx.columns)
    type_vocab = sorted([t for t in tx.get("tx_type", pd.Series(dtype=str)).dropna().unique().tolist() if str(t).strip()]) if has_type else []
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}

    feat_dim = 3 + (6 if has_amount else 0) + (2 if has_org_delta else 0) + (2 if has_dest_delta else 0) + (2 * len(type_vocab) if has_type else 0)
    snapshots: list[Snapshot] = []

    for t_idx, (_, g) in enumerate(tx.groupby("bin", sort=True)):
        nodes = pd.Index(pd.concat([g["from"], g["to"]], axis=0).unique())
        node_gids = np.array([addr_to_gid[a] for a in nodes], dtype=np.int64)
        gid_to_local = {int(gid): j for j, gid in enumerate(node_gids)}
        src_g = g["from"].map(addr_to_gid).to_numpy(np.int64)
        dst_g = g["to"].map(addr_to_gid).to_numpy(np.int64)
        src_l = np.array([gid_to_local[int(s)] for s in src_g], dtype=np.int64)
        dst_l = np.array([gid_to_local[int(d)] for d in dst_g], dtype=np.int64)
        n = len(node_gids)

        in_deg = np.zeros(n, dtype=np.float32); out_deg = np.zeros(n, dtype=np.float32)
        out_amt_sum = np.zeros(n, dtype=np.float32); out_amt_max = np.zeros(n, dtype=np.float32); out_amt_cnt = np.zeros(n, dtype=np.float32)
        in_amt_sum = np.zeros(n, dtype=np.float32); in_amt_max = np.zeros(n, dtype=np.float32); in_amt_cnt = np.zeros(n, dtype=np.float32)
        out_org_d_sum = np.zeros(n, dtype=np.float32); out_org_d_cnt = np.zeros(n, dtype=np.float32)
        in_dst_d_sum = np.zeros(n, dtype=np.float32); in_dst_d_cnt = np.zeros(n, dtype=np.float32)
        out_type = np.zeros((n, len(type_vocab)), dtype=np.float32) if has_type and type_vocab else None
        in_type = np.zeros((n, len(type_vocab)), dtype=np.float32) if has_type and type_vocab else None
        y = np.zeros(n, dtype=np.int64)

        labels = g["label"].to_numpy(np.int64)
        amounts = g["amount"].to_numpy(np.float32) if has_amount else None
        org_d = g["org_delta"].to_numpy(np.float32) if has_org_delta else None
        dst_d = g["dest_delta"].to_numpy(np.float32) if has_dest_delta else None
        types = g["tx_type"].astype(str).str.upper().to_numpy() if has_type else None

        for i, (s, d, lab) in enumerate(zip(src_l, dst_l, labels)):
            out_deg[s] += 1.0; in_deg[d] += 1.0
            if lab == 1:
                if label_mode in {"both", "sender_only"}:
                    y[s] = 1
                if label_mode in {"both", "receiver_only"}:
                    y[d] = 1
            if has_amount:
                a = float(amounts[i])
                out_amt_sum[s] += a; out_amt_cnt[s] += 1.0; out_amt_max[s] = max(out_amt_max[s], a)
                in_amt_sum[d] += a; in_amt_cnt[d] += 1.0; in_amt_max[d] = max(in_amt_max[d], a)
            if has_org_delta:
                out_org_d_sum[s] += float(org_d[i]); out_org_d_cnt[s] += 1.0
            if has_dest_delta:
                in_dst_d_sum[d] += float(dst_d[i]); in_dst_d_cnt[d] += 1.0
            if has_type and type_vocab:
                j = type_to_idx.get(types[i])
                if j is not None:
                    out_type[s, j] += 1.0; in_type[d, j] += 1.0

        feats = [in_deg, out_deg, in_deg + out_deg]
        if has_amount:
            feats += [out_amt_sum, _safe_mean(out_amt_sum, out_amt_cnt), out_amt_max, in_amt_sum, _safe_mean(in_amt_sum, in_amt_cnt), in_amt_max]
        if has_org_delta:
            feats += [out_org_d_sum, _safe_mean(out_org_d_sum, out_org_d_cnt)]
        if has_dest_delta:
            feats += [in_dst_d_sum, _safe_mean(in_dst_d_sum, in_dst_d_cnt)]
        x = np.stack(feats, axis=1).astype(np.float32)
        if has_type and type_vocab:
            x = np.concatenate([x, out_type, in_type], axis=1).astype(np.float32)
        snapshots.append(Snapshot(t_idx, node_gids, src_l, dst_l, x, y, gid_to_local))

    return DynamicGraph(snapshots, len(addrs), feat_dim, {"dataset": "paysim", "time_bin": time_bin, "label_mode": label_mode, "type_vocab": type_vocab})


def _numeric_feature_columns(df: pd.DataFrame, exclude: Sequence[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _candidate_graph_columns(df: pd.DataFrame, exclude: Sequence[str], max_cols: int = 4) -> list[str]:
    cats = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        nunique = s.nunique(dropna=True)
        if (str(c).endswith("_cat") or pd.api.types.is_object_dtype(s) or nunique <= 30) and 2 <= nunique <= 100:
            cats.append(c)
    return cats[:max_cols]


def build_tabular_similarity_graph(
    df: pd.DataFrame,
    dataset_name: str,
    label_col: str = "label",
    time_col: str = "timestamp",
    entity_col: str = "entity_id",
    n_snapshots: int = 5,
    graph_cols: Sequence[str] | None = None,
    max_group_edges: int = 2,
    max_features: int = 64,
) -> DynamicGraph:
    """Build a lightweight entity-similarity graph for non-transaction tabular datasets.

    The graph connects rows/entities sharing selected low-cardinality attributes. This is intended
    for cross-domain stress testing and should not be over-interpreted if AUC is near chance.
    """
    work = df.copy().reset_index(drop=True)
    if label_col not in work.columns:
        raise ValueError(f"Missing label column {label_col}")
    if entity_col not in work.columns:
        work[entity_col] = work.index.astype(str)
    if time_col in work.columns:
        work = work.sort_values(time_col).reset_index(drop=True)
    work["__snap__"] = pd.qcut(np.arange(len(work)), q=min(n_snapshots, len(work)), labels=False, duplicates="drop")
    exclude = {label_col, "target", time_col, entity_col, "from", "to", "timestamp", "__snap__"}
    feat_cols = _numeric_feature_columns(work, exclude)
    leak_keys = ["loss", "claim", "payment", "premium", "fee", "amount", "cost", "surcharge"]
    if dataset_name.lower() == "fema":
        feat_cols = [c for c in feat_cols if not any(k in c.lower() for k in leak_keys)]
    feat_cols = feat_cols[:max_features]
    if not feat_cols:
        raise ValueError("No numeric features available after leakage filtering.")
    X_all = work[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    X_all = StandardScaler().fit_transform(X_all).astype(np.float32)

    if graph_cols is None:
        graph_cols = _candidate_graph_columns(work, exclude, max_cols=4)
    graph_cols = list(graph_cols)

    all_entities = pd.Index(work[entity_col].astype(str).unique())
    entity_to_gid = {e: i for i, e in enumerate(all_entities)}
    snapshots: list[Snapshot] = []
    for t_idx, (_, g) in enumerate(work.groupby("__snap__", sort=True)):
        idx_global_rows = g.index.to_numpy()
        node_gids = np.array([entity_to_gid[str(e)] for e in g[entity_col].astype(str)], dtype=np.int64)
        gid_to_local = {int(gid): i for i, gid in enumerate(node_gids)}
        x = X_all[idx_global_rows].astype(np.float32)
        y = pd.to_numeric(g[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy(np.int64)
        src: list[int] = []; dst: list[int] = []
        for col in graph_cols:
            if col not in g.columns:
                continue
            for _, gg in g.groupby(col, dropna=True, sort=False):
                locs = list(range(len(gg)))
                if len(locs) <= 1:
                    continue
                # connect local chain / short-range neighborhood to avoid dense cliques
                for p in range(len(locs)):
                    for q in range(p + 1, min(p + 1 + max_group_edges, len(locs))):
                        src.extend([locs[p], locs[q]])
                        dst.extend([locs[q], locs[p]])
        snapshots.append(Snapshot(t_idx, node_gids, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), x, y, gid_to_local))
    return DynamicGraph(snapshots, len(all_entities), X_all.shape[1], {"dataset": dataset_name, "graph_cols": graph_cols, "feature_cols": feat_cols, "n_snapshots": n_snapshots})


def build_graph_for_dataset(dataset: str, df: pd.DataFrame, **kwargs) -> DynamicGraph:
    ds = dataset.lower().strip()
    if ds == "paysim":
        return build_paysim_graph(df, **kwargs)
    if ds in {"porto", "fema"}:
        allowed = {k: v for k, v in kwargs.items() if k in {"n_snapshots", "graph_cols", "max_group_edges", "max_features"}}
        return build_tabular_similarity_graph(df, dataset_name=ds, **allowed)
    raise ValueError(f"Unknown dataset {dataset}")
