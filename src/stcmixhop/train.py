from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

from .models import build_sparse_adj, dgi_loss, info_nce_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_split_snapshots(dyn_graph, train_ratio=0.70, val_ratio=0.15):
    T = len(dyn_graph.snapshots)
    t_train = max(1, int(train_ratio * T))
    t_val = max(t_train + 1, int((train_ratio + val_ratio) * T))
    train = dyn_graph.snapshots[:t_train]
    val = dyn_graph.snapshots[t_train:t_val]
    test = dyn_graph.snapshots[t_val:]
    if len(val) == 0 and len(test) > 0:
        val = test[:1]
        test = test[1:] if len(test) > 1 else test
    if len(test) == 0:
        test = val
    return train, val, test


def pick_threshold_by_fbeta(y_true, y_prob, beta: float = 0.5):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    qs = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_f, best_th = -1.0, 0.5
    for th in qs:
        y_pred = (y_prob >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, average="binary", zero_division=0)
        if f > best_f:
            best_f, best_th = float(f), float(th)
    return best_th


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float, beta: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= th).astype(int)
    p, r, fbeta, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, average="binary", zero_division=0)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        pr_auc = float("nan")
    return {
        "AUC": float(auc), "PR_AUC": float(pr_auc), "Fbeta": float(fbeta), "F1": float(f1),
        "Precision": float(p), "Recall": float(r), "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Threshold": float(th),
    }


def _build_z_hist(snap, z_t, prev_cache, prev_nodes, device, use_temporal=True):
    if (not use_temporal) or (prev_cache is None) or (prev_nodes is None):
        return torch.stack([z_t, z_t], dim=1)
    prev_map = {int(g): i for i, g in enumerate(prev_nodes.tolist())}
    idx_prev = [prev_map.get(int(gid), -1) for gid in snap.node_ids.tolist()]
    idx_prev = torch.tensor(idx_prev, device=device)
    z_prev = torch.where(idx_prev[:, None] >= 0, prev_cache[idx_prev.clamp(min=0)], z_t)
    return torch.stack([z_t, z_prev], dim=1)


def collect_probs(model, snaps, device, use_temporal=True, return_embeddings=False):
    model.eval()
    prev_cache = None; prev_nodes = None
    ys, ps, embs, attns = [], [], [], []
    with torch.no_grad():
        for snap in snaps:
            X = torch.tensor(snap.x, dtype=torch.float32, device=device)
            n = X.size(0)
            src = torch.tensor(snap.edges_src, dtype=torch.long, device=device)
            dst = torch.tensor(snap.edges_dst, dtype=torch.long, device=device)
            A = build_sparse_adj(n, src, dst, edge_keep_prob=1.0)
            z_t = model.encode(X, A)
            z_hist = _build_z_hist(snap, z_t, prev_cache, prev_nodes, device, use_temporal=use_temporal)
            logits, z = model(z_t, z_hist)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            ys.append(snap.y.astype(np.int64)); ps.append(prob)
            if return_embeddings:
                embs.append(z.detach().cpu().numpy())
                if hasattr(model, "temp_attn") and getattr(model.temp_attn, "last_attention", None) is not None:
                    attns.append(model.temp_attn.last_attention.detach().cpu().numpy())
            prev_cache = z_t.detach()
            prev_nodes = torch.tensor(snap.node_ids, dtype=torch.long, device=device)
    y = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.int64)
    p = np.concatenate(ps, axis=0) if ps else np.array([], dtype=np.float32)
    out = {"y": y, "prob": p}
    if return_embeddings:
        out["embeddings"] = np.concatenate(embs, axis=0) if embs else np.empty((0, 0))
        out["attention"] = np.concatenate(attns, axis=0) if attns else np.empty((0, 0))
    return out


def pretrain(model, dyn_graph, device, epochs=10, lr=1e-3, p_x=0.9, p_a=0.9, tau0=0.2, beta_temp=0.3, mode: str = "stc", seed: int = 42, log: list[dict[str, Any]] | None = None):
    """Pretrain encoder: nt_xent, stc, or dgi."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        prev_z_global = None; prev_nodes = None
        t0 = time.perf_counter()
        for snap in dyn_graph.snapshots:
            X = torch.tensor(snap.x, dtype=torch.float32, device=device)
            n = X.size(0)
            src = torch.tensor(snap.edges_src, dtype=torch.long, device=device)
            dst = torch.tensor(snap.edges_dst, dtype=torch.long, device=device)
            X1 = X * (torch.rand_like(X) < p_x).float()
            X2 = X * (torch.rand_like(X) < p_x).float()
            A1 = build_sparse_adj(n, src, dst, edge_keep_prob=p_a)
            A2 = build_sparse_adj(n, src, dst, edge_keep_prob=p_a)
            z1 = model.encode(X1, A1); z2 = model.encode(X2, A2)
            if mode == "dgi":
                loss = dgi_loss(z1, z2[torch.randperm(z2.size(0), device=device)])
            else:
                loss_intra = info_nce_loss(z1, z2, temperature=tau0)
                loss_temp = torch.tensor(0.0, device=device)
                if mode == "stc" and prev_z_global is not None:
                    common = np.intersect1d(snap.node_ids, prev_nodes, assume_unique=False)
                    if len(common) >= 2:
                        cur_map = snap.global_to_local
                        prev_map = {int(gid): i for i, gid in enumerate(prev_nodes.tolist())}
                        cur_idx = torch.tensor([cur_map[int(g)] for g in common], device=device)
                        prev_idx = torch.tensor([prev_map[int(g)] for g in common], device=device)
                        tau = tau0 * (1.0 + beta_temp)
                        loss_temp = info_nce_loss(z1[cur_idx], prev_z_global[prev_idx], temperature=tau)
                loss = loss_intra + loss_temp
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach().cpu()); count += 1
            prev_z_global = z1.detach(); prev_nodes = torch.tensor(snap.node_ids, dtype=torch.long)
        row = {"stage": f"pretrain_{mode}", "epoch": ep, "loss": total / max(count, 1), "seconds": time.perf_counter() - t0}
        if log is not None:
            log.append(row)
        print(f"[pretrain-{mode}] epoch={ep} loss={row['loss']:.4f}", flush=True)
    return log


def finetune_and_eval(model, dyn_graph, device, epochs=10, lr=1e-3, beta=0.5, use_temporal_hist=True, seed: int = 42, log: list[dict[str, Any]] | None = None, return_artifacts: bool = False, pos_weight_scale: float = 1.0):
    set_seed(seed)
    train_snaps, val_snaps, test_snaps = time_split_snapshots(dyn_graph, 0.70, 0.15)
    y_train = np.concatenate([s.y for s in train_snaps], axis=0)
    pos = float((y_train == 1).sum()); neg = float((y_train == 0).sum())
    pos_weight = (neg / max(pos, 1.0)) * float(pos_weight_scale)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        prev_cache = None; prev_nodes = None
        t0 = time.perf_counter()
        for snap in train_snaps:
            X = torch.tensor(snap.x, dtype=torch.float32, device=device)
            y = torch.tensor(snap.y, dtype=torch.float32, device=device)
            n = X.size(0)
            src = torch.tensor(snap.edges_src, dtype=torch.long, device=device)
            dst = torch.tensor(snap.edges_dst, dtype=torch.long, device=device)
            A = build_sparse_adj(n, src, dst, edge_keep_prob=1.0)
            z_t = model.encode(X, A)
            z_hist = _build_z_hist(snap, z_t, prev_cache, prev_nodes, device, use_temporal=use_temporal_hist)
            logits, _ = model(z_t, z_hist)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach().cpu()); count += 1
            prev_cache = z_t.detach(); prev_nodes = torch.tensor(snap.node_ids, dtype=torch.long, device=device)
        val_probs = collect_probs(model, val_snaps, device, use_temporal=use_temporal_hist)
        th_tmp = pick_threshold_by_fbeta(val_probs["y"], val_probs["prob"], beta=beta)
        val_metrics = evaluate_at_threshold(val_probs["y"], val_probs["prob"], th_tmp, beta=beta)
        row = {"stage": "finetune", "epoch": ep, "loss": total / max(count, 1), "seconds": time.perf_counter() - t0, **{f"val_{k}": v for k, v in val_metrics.items()}}
        if log is not None:
            log.append(row)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[finetune] epoch={ep} loss={row['loss']:.4f} val_PR_AUC={val_metrics['PR_AUC']:.4f}", flush=True)
    val_probs = collect_probs(model, val_snaps, device, use_temporal=use_temporal_hist)
    th = pick_threshold_by_fbeta(val_probs["y"], val_probs["prob"], beta=beta)
    test_probs = collect_probs(model, test_snaps, device, use_temporal=use_temporal_hist, return_embeddings=return_artifacts)
    metrics = evaluate_at_threshold(test_probs["y"], test_probs["prob"], th, beta=beta)
    if return_artifacts:
        return metrics, {"val": val_probs, "test": test_probs, "threshold": th}
    return metrics


def profile_inference(model, dyn_graph, device, repeats: int = 3, use_temporal_hist: bool = True) -> dict[str, float]:
    _, _, test_snaps = time_split_snapshots(dyn_graph, 0.70, 0.15)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = collect_probs(model, test_snaps, device, use_temporal=use_temporal_hist)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    n_nodes = int(sum(len(s.y) for s in test_snaps))
    peak_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() and str(device).startswith("cuda") else float("nan")
    return {"test_nodes": n_nodes, "inference_seconds_mean": float(np.mean(times)), "inference_ms_per_1k_nodes": float(np.mean(times) * 1000.0 / max(n_nodes / 1000.0, 1e-9)), "cuda_peak_memory_mb": peak_mb}
