#!/usr/bin/env python
"""Unified experiment runner for IJDSA revision.

Examples
--------
PaySim quick smoke test:
  python run_experiment.py --dataset paysim --data /path/paysim.zip --max-rows 5000 --quick --outdir outputs/paysim_quick

Fuller PaySim run:
  python run_experiment.py --dataset paysim --data /path/paysim.csv --max-rows 200000 --epochs-pre 10 --epochs-sup 20 --time-bin 7D --K 2 --dk 128 --seeds 42 43 44

Porto:
  python run_experiment.py --dataset porto --data /path/porto-seguro-safe-driver-prediction.zip --max-rows 50000 --n-snapshots 5

FEMA:
  python run_experiment.py --dataset fema --data /path/FimaNfipPoliciesV2.parquet --max-rows 20000 --n-snapshots 5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stcmixhop.baselines import run_tabular_baselines
from stcmixhop.data_ingest import load_dataset
from stcmixhop.graph_build import build_graph_for_dataset
from stcmixhop.models import (
    DySATLite,
    EvolveGCNLite,
    GATEncoderLite,
    GCNEncoder,
    GraphClassifier,
    SAGEEncoder,
    STCMixHop,
    TemporalGCNGRU,
    count_parameters,
)
from stcmixhop.plotting import (
    plot_ablation_figure,
    plot_embedding_pca,
    plot_loss_curve,
    plot_overall_figure,
    plot_prediction_violin,
    plot_sensitivity,
    save_table_csv,
)
from stcmixhop.stats import average_ranking, paired_tests, summarize_multiseed
from stcmixhop.train import finetune_and_eval, pretrain, profile_inference, set_seed


def ensure_dir(p: str | os.PathLike):
    Path(p).mkdir(parents=True, exist_ok=True)


def build_model(name: str, in_dim: int, K: int, dk: int, dropout: float = 0.2):
    emb_dim = 64
    if name == "GCN":
        return GraphClassifier(GCNEncoder(in_dim, emb_dim, dropout=dropout), emb_dim, d_k=dk, dropout=dropout, temporal="attention")
    if name == "GraphSAGE":
        return GraphClassifier(SAGEEncoder(in_dim, emb_dim, dropout=dropout), emb_dim, d_k=dk, dropout=dropout, temporal="attention")
    if name == "GAT":
        return GraphClassifier(GATEncoderLite(in_dim, emb_dim, dropout=dropout), emb_dim, d_k=dk, dropout=dropout, temporal="attention")
    if name == "TemporalGCN-GRU":
        return TemporalGCNGRU(in_dim, emb_dim=emb_dim, dropout=dropout)
    if name == "DySAT-lite":
        return DySATLite(in_dim, emb_dim=emb_dim, d_k=dk, dropout=dropout)
    if name == "EvolveGCN-lite":
        return EvolveGCNLite(in_dim, emb_dim=emb_dim, dropout=dropout)
    if name in {"Supervised-only GNN", "Self-sup. (NT-Xent)", "Self-sup. (DGI)", "STC-MixHop (ours)", "Full (STC-MixHop)", "w/o contrastive learning"}:
        return STCMixHop(in_dim, K=K, d_k=dk, dropout=dropout, use_temporal_attention=True)
    if name == "w/o same-step structure":
        return STCMixHop(in_dim, K=0, d_k=dk, dropout=dropout, use_temporal_attention=True)
    if name == "w/o temporal attention":
        return STCMixHop(in_dim, K=K, d_k=dk, dropout=dropout, use_temporal_attention=False)
    if name == "w/o time-decay weighting":
        return STCMixHop(in_dim, K=K, d_k=dk, dropout=dropout, use_temporal_attention=True)
    raise ValueError(f"Unknown model {name}")


def run_torch_variant(name, dyn, device, args, seed, outdir, return_artifacts=False):
    log = []
    model = build_model(name, dyn.feat_dim, args.K, args.dk, dropout=args.dropout).to(device)
    pre_mode = None
    beta_temp = args.beta_temp
    if name == "Self-sup. (NT-Xent)":
        pre_mode = "nt_xent"
    elif name == "Self-sup. (DGI)":
        pre_mode = "dgi"
    elif name in {"STC-MixHop (ours)", "Full (STC-MixHop)", "w/o same-step structure", "w/o temporal attention", "w/o time-decay weighting"}:
        pre_mode = "stc"
    if name == "w/o contrastive learning":
        pre_mode = None
    if name == "w/o time-decay weighting":
        beta_temp = 0.0
    if pre_mode is not None and args.epochs_pre > 0:
        pretrain(model, dyn, device, epochs=args.epochs_pre, lr=args.lr, p_x=args.px, p_a=args.pa, tau0=args.tau0, beta_temp=beta_temp, mode=pre_mode, seed=seed, log=log)
    use_temporal = name != "w/o temporal attention"
    result = finetune_and_eval(model, dyn, device, epochs=args.epochs_sup, lr=args.lr, beta=args.beta, use_temporal_hist=use_temporal, seed=seed, log=log, return_artifacts=return_artifacts, pos_weight_scale=args.pos_weight_scale)
    if return_artifacts:
        metrics, artifacts = result
    else:
        metrics, artifacts = result, None
    prof = profile_inference(model, dyn, device, repeats=args.profile_repeats, use_temporal_hist=use_temporal) if args.profile else {}
    row = {"Model": name, "Seed": seed, "Params": count_parameters(model), **metrics, **prof}
    log_df = pd.DataFrame(log)
    log_path = Path(outdir) / f"logs_{name.replace('/', '_').replace(' ', '_')}_seed{seed}.csv"
    save_table_csv(log_df, str(log_path))
    return row, model, artifacts, log_df


def run_one_seed(df, seed, args, outdir):
    set_seed(seed)
    build_kwargs = {}
    if args.dataset == "paysim":
        build_kwargs = {"time_bin": args.time_bin, "label_mode": args.label_mode, "use_type_stats": not args.no_type_stats}
    else:
        build_kwargs = {"n_snapshots": args.n_snapshots, "max_group_edges": args.max_group_edges, "max_features": args.max_features}
        if args.graph_cols:
            build_kwargs["graph_cols"] = args.graph_cols
    dyn = build_graph_for_dataset(args.dataset, df, **build_kwargs)
    meta = {"num_snapshots": len(dyn.snapshots), "num_nodes_global": dyn.num_nodes_global, "feat_dim": dyn.feat_dim, **dyn.meta}
    with open(Path(outdir) / f"graph_meta_seed{seed}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if len(dyn.snapshots) < 3:
        raise RuntimeError(f"Too few snapshots ({len(dyn.snapshots)}). Reduce time-bin or increase max-rows.")

    rows = []
    if not args.no_tabular:
        for r in run_tabular_baselines(dyn, beta=args.beta, seed=seed):
            rows.append({"Seed": seed, **r})

    model_names = []
    if not args.no_graph_baselines:
        model_names += ["GCN", "GraphSAGE", "GAT"]
    if not args.no_temporal_baselines:
        model_names += ["TemporalGCN-GRU", "EvolveGCN-lite"]
    model_names += ["Supervised-only GNN", "Self-sup. (NT-Xent)", "Self-sup. (DGI)", "STC-MixHop (ours)"]

    artifacts_for_visuals = None
    log_for_visuals = None
    for name in model_names:
        print(f"\n=== seed={seed} model={name} ===", flush=True)
        row, model, artifacts, log_df = run_torch_variant(name, dyn, torch.device(args.device), args, seed, outdir, return_artifacts=(name == "STC-MixHop (ours)"))
        rows.append(row)
        if name == "STC-MixHop (ours)":
            artifacts_for_visuals = artifacts
            log_for_visuals = log_df
            if artifacts is not None:
                pred_df = pd.DataFrame({"y_true": artifacts["test"]["y"], "prob": artifacts["test"]["prob"]})
                save_table_csv(pred_df, str(Path(outdir) / f"test_predictions_stcmixhop_seed{seed}.csv"))
                plot_prediction_violin(pred_df, str(Path(outdir) / f"violin_predictions_seed{seed}.png"))
                plot_embedding_pca(artifacts["test"].get("embeddings"), artifacts["test"]["y"], str(Path(outdir) / f"embedding_pca_seed{seed}.png"))
            if log_df is not None:
                plot_loss_curve(log_df, str(Path(outdir) / f"loss_curve_stcmixhop_seed{seed}.png"))

    if not args.skip_ablation:
        ab_names = ["Full (STC-MixHop)", "w/o same-step structure", "w/o time-decay weighting", "w/o temporal attention", "w/o contrastive learning"]
        ab_rows = []
        for name in ab_names:
            print(f"\n=== seed={seed} ablation={name} ===", flush=True)
            row, _, _, _ = run_torch_variant(name, dyn, torch.device(args.device), args, seed, outdir, return_artifacts=False)
            row = {"Variant": row.pop("Model"), **row}
            ab_rows.append(row)
        ab_df = pd.DataFrame(ab_rows)
        save_table_csv(ab_df, str(Path(outdir) / f"ablation_seed{seed}.csv"))
        plot_ablation_figure(ab_df, str(Path(outdir) / f"Figure_ablation_seed{seed}.png"))

    if not args.skip_sensitivity:
        for sweep_name, values in [("K", args.K_sweep), ("dk", args.dk_sweep), ("tau0", args.tau_sweep), ("pos_weight_scale", args.pos_weight_sweep)]:
            sens_rows = []
            for val in values:
                oldK, olddk, oldtau, oldpws = args.K, args.dk, args.tau0, args.pos_weight_scale
                if sweep_name == "K": args.K = int(val)
                elif sweep_name == "dk": args.dk = int(val)
                elif sweep_name == "tau0": args.tau0 = float(val)
                elif sweep_name == "pos_weight_scale": args.pos_weight_scale = float(val)
                row, _, _, _ = run_torch_variant("STC-MixHop (ours)", dyn, torch.device(args.device), args, seed, outdir, return_artifacts=False)
                sens_rows.append({sweep_name: val, **row})
                args.K, args.dk, args.tau0, args.pos_weight_scale = oldK, olddk, oldtau, oldpws
            sens_df = pd.DataFrame(sens_rows)
            save_table_csv(sens_df, str(Path(outdir) / f"sensitivity_{sweep_name}_seed{seed}.csv"))
            plot_sensitivity(sens_df, sweep_name, str(Path(outdir) / f"Figure_sensitivity_{sweep_name}_seed{seed}.png"), sweep_name)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["paysim", "porto", "fema"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="outputs/revision_run")
    p.add_argument("--max-rows", type=int, default=200000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="Small smoke-test settings.")
    p.add_argument("--time-bin", default="7D")
    p.add_argument("--label-mode", choices=["both", "sender_only", "receiver_only"], default="both")
    p.add_argument("--no-type-stats", action="store_true")
    p.add_argument("--n-snapshots", type=int, default=5)
    p.add_argument("--graph-cols", nargs="*", default=None)
    p.add_argument("--max-group-edges", type=int, default=2)
    p.add_argument("--max-features", type=int, default=64)
    p.add_argument("--epochs-pre", type=int, default=10)
    p.add_argument("--epochs-sup", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--px", type=float, default=0.9)
    p.add_argument("--pa", type=float, default=0.9)
    p.add_argument("--tau0", type=float, default=0.2)
    p.add_argument("--beta-temp", type=float, default=0.3)
    p.add_argument("--no-graph-baselines", action="store_true", help="Disable GCN/GraphSAGE/GAT rows.")
    p.add_argument("--no-temporal-baselines", action="store_true", help="Disable temporal baseline rows.")
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-repeats", type=int, default=3)
    p.add_argument("--K-sweep", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--dk-sweep", type=int, nargs="+", default=[32, 64, 96, 128, 192, 256])
    p.add_argument("--tau-sweep", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    p.add_argument("--pos-weight-scale", type=float, default=1.0)
    p.add_argument("--pos-weight-sweep", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = p.parse_args()
    # Deterministic and substantially faster on small sparse CPU workloads.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    if args.quick:
        args.max_rows = min(args.max_rows, 3000)
        args.epochs_pre = min(args.epochs_pre, 1)
        args.epochs_sup = min(args.epochs_sup, 1)
        args.K_sweep = [1, 2]
        args.dk_sweep = [32, 64]
        args.tau_sweep = [0.2]
        args.skip_sensitivity = True
        args.skip_ablation = True

    ensure_dir(args.outdir)
    with open(Path(args.outdir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    df = load_dataset(args.dataset, args.data, max_rows=args.max_rows, seed=args.seeds[0])
    all_rows = []
    for seed in args.seeds:
        seed_out = Path(args.outdir) / f"seed_{seed}"
        ensure_dir(seed_out)
        all_rows.append(run_one_seed(df.copy(), seed, args, seed_out))
    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    save_table_csv(all_df, str(Path(args.outdir) / "Table_overall_all_seeds.csv"))
    plot_overall_figure(all_df.groupby("Model", as_index=False).mean(numeric_only=True), str(Path(args.outdir) / "Figure_overall_mean.png"), beta=args.beta)
    summary = summarize_multiseed(all_df)
    save_table_csv(summary, str(Path(args.outdir) / "Table_multiseed_mean_std.csv"))
    ranks = average_ranking(all_df)
    save_table_csv(ranks, str(Path(args.outdir) / "Table_average_ranking.csv"))
    if "STC-MixHop (ours)" in set(all_df["Model"]):
        for base in ["MLP", "GAT", "EvolveGCN-lite"]:
            if base in set(all_df["Model"]):
                tests = paired_tests(all_df, baseline=base, challenger="STC-MixHop (ours)")
                save_table_csv(tests, str(Path(args.outdir) / f"Stats_STC_vs_{base.replace(' ', '_')}.csv"))
    print(f"\nDone. Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
