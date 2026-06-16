from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .train import evaluate_at_threshold, pick_threshold_by_fbeta, time_split_snapshots


def _collect_xy(snaps):
    X = np.concatenate([s.x for s in snaps], axis=0)
    y = np.concatenate([s.y for s in snaps], axis=0)
    return X, y


def _fit_predict_proba(clf, Xtr, ytr, Xva, Xte):
    clf.fit(Xtr, ytr)
    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    return pva, pte


def run_tabular_baselines(dyn_graph, beta: float = 0.5, seed: int = 42):
    train, val, test = time_split_snapshots(dyn_graph, 0.70, 0.15)
    Xtr, ytr = _collect_xy(train); Xva, yva = _collect_xy(val); Xte, yte = _collect_xy(test)
    rows = []
    models = [
        ("LogReg", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])),
        ("RF", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)),
        ("MLP", Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, random_state=seed))])),
    ]
    for name, clf in models:
        pva, pte = _fit_predict_proba(clf, Xtr, ytr, Xva, Xte)
        th = pick_threshold_by_fbeta(yva, pva, beta=beta)
        rows.append({"Model": name, **evaluate_at_threshold(yte, pte, th, beta=beta)})
    return rows
