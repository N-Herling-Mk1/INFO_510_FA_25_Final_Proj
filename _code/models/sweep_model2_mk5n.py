#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sweep_model2_mk5n.py
──────────────────────────────────────────────────────────
Full-Isolated Sweep System (with RRM Integration + Early Stopping Added)
2025-12-01

ONLY MODIFICATION FROM mk5m:
    ✔ EarlyStopping callback added inside train_one_fold()

NO OTHER LOGIC WAS MODIFIED.
"""

###############################################################################
# GLOBAL LOG / WARNING SUPPRESSION
###############################################################################
import sys
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

###############################################################################
# FIX PYTHONPATH SO "_code" IMPORTS ALWAYS WORK
###############################################################################
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# STANDARD IMPORTS
###############################################################################
import argparse
import json
import shutil
from datetime import datetime
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import BayesianRidge

from tensorflow import keras
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

###############################################################################
# LOCAL IMPORTS
###############################################################################
from _code.data.loaders_model2 import make_dataset_model2
from _code.models.model2_fusion_mk2 import build_model2_fusion

console = Console()

def safe_import(package_path, name):
    try:
        exec(f"from {package_path} import {name}", globals())
    except Exception:
        root = Path(__file__).resolve()
        while root.name not in (".", "", "/") and "_code" not in root.name:
            root = root.parent
        sys.path.insert(0, str(root))
        exec(f"from {package_path} import {name}", globals())

safe_import("_code.data.loaders_model2", "make_dataset_model2")
safe_import("_code.models.model2_fusion_mk2", "build_model2_fusion")

###############################################################################
# Coloring helpers
###############################################################################
def c_green(x): return f"[green]{x}[/green]"
def c_red(x):   return f"[red]{x}[/red]"
def c_blue(x):  return f"[cyan]{x}[/cyan]"
def c_yellow(x):return f"[yellow]{x}[/yellow]"

###############################################################################
# DIRECTORY HELPERS
###############################################################################
def create_sweep_dirs(root_dir: str):
    root = Path(root_dir)
    root.mkdir(exist_ok=True, parents=True)
    tstamp = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")

    sweep_dir = root / tstamp
    sweep_dir.mkdir()

    runs_dir = sweep_dir / "runs"
    leaders_dir = sweep_dir / "leaders"

    runs_dir.mkdir()
    leaders_dir.mkdir()

    console.print(c_green(f"📁 Created sweep directory: {sweep_dir}"))
    return sweep_dir, runs_dir, leaders_dir

def create_run_dir(runs_dir: Path, cfg_id: str):
    stamp = datetime.now().strftime("%H%M%S%f")[:-3]
    run_dir = runs_dir / f"{cfg_id}_{stamp}"
    run_dir.mkdir()
    (run_dir / "plots").mkdir()
    (run_dir / "metrics").mkdir()
    return run_dir

###############################################################################
# LOAD + NORMALIZE DATA
###############################################################################
def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    console.print(c_blue(f"📥 Loaded CSV: {len(df):,} rows"))

    candidates = ["label", "genre", "class", "y"]
    label_col = next((c for c in candidates if c in df.columns), None)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    labels = sorted(df["label"].unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    skip = {"label", "filename", "path"}
    num_cols = [
        c for c in df.columns
        if c not in skip and np.issubdtype(df[c].dtype, np.number)
    ]
    for c in num_cols:
        mu, sigma = df[c].mean(), df[c].std() or 1.0
        df[c] = (df[c] - mu) / sigma

    console.print(c_green(f"✔ Normalized {len(num_cols)} numeric columns"))
    return df, label_to_idx, labels

###############################################################################
# HOLDOUT SPLIT
###############################################################################
def build_holdout_with_coverage(df, classes, ratio=0.15, cov_target=0.90, max_tries=12):
    for t in range(1, max_tries + 1):
        df_shuf = df.sample(frac=1.0, random_state=50 + t).reset_index(drop=True)
        n = int(len(df_shuf) * ratio)
        hold = df_shuf[:n]
        cov = len(hold["label"].unique()) / len(classes)
        if cov >= cov_target:
            console.print(c_green(f"✔ Holdout coverage {cov:.2f} (attempt {t})"))
            return hold, df_shuf.drop(hold.index), cov
        console.print(c_yellow(f"⚠ Coverage {cov:.2f} < {cov_target}, retry {t}"))
    return hold, df_shuf.drop(hold.index), cov

###############################################################################
# CONFIG SAMPLER  (refined hyperparameter windows from sweep insights)
###############################################################################
def random_configs(k: int):
    return [
        {
            "id": f"cfg_{i}",

            # Dropout: narrowed to high-performance band
            "dropout_backbone": float(np.random.uniform(0.10, 0.28)),
            "dropout_tab": float(np.random.uniform(0.10, 0.25)),

            # Dense units focused on best-performing clusters
            "dense_units_img": int(np.random.choice([256, 384, 512])),
            "dense_units_tab": int(np.random.choice([128, 192, 256])),

            # LR strongly constrained to winning region (≈1e-4 – 4.5e-4)
            "lr": float(10 ** np.random.uniform(-4.0, -3.35)),

            # Epochs remain the same cluster
            "epochs": int(np.random.choice([50, 75, 100, 125, 150])),

            # rmsprop removed (never competitive), weighted selection
            "optimizer": str(np.random.choice(["adam", "adamw"], p=[0.6, 0.4])),

            # Fusion: gated strongly preferred
            "fusion": str(np.random.choice(["gated", "concat"], p=[0.7, 0.3])),

            # Head type: reduce flipout probability
            "head_type": str(np.random.choice(
                ["det", "mc_dropout", "flipout"],
                p=[0.60, 0.30, 0.10]
            )),

            # Bayesian factors kept same but safe within optimal range
            "kl_factor": float(np.random.uniform(1.0, 1.5)),
            "prior_scale": float(np.random.uniform(0.7, 1.5)),
            "posterior_scale_init": float(np.random.uniform(0.005, 0.03)),
        }
        for i in range(k)
    ]

###############################################################################
# PLOTS
###############################################################################
def plot_training_curves(history, cfg_id, plots_dir):
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    ax1.plot(history.history.get("loss", []), label="Train Loss")
    ax1.plot(history.history.get("val_loss", []), label="Val Loss")
    ax2.plot(history.history.get("accuracy", []), "--", label="Train Acc")
    ax2.plot(history.history.get("val_accuracy", []), "--", label="Val Acc")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    out = Path(plots_dir) / f"{cfg_id}_metrics.png"
    plt.title(f"Training Curves — {cfg_id}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_curves(y_true, y_prob, labels, out_path):
    plt.figure(figsize=(8,6))
    n_classes = len(labels)
    for c in range(n_classes):
        y_bin = (np.array(y_true)==c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:,c])
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f"{labels[c]} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],"k--")
    plt.title("ROC — One-vs-Rest")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_latent_correlation(latents, out_path):
    corr = np.corrcoef(latents.T)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Latent Feature Correlation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_uncertainty_hist(sigma, out_path):
    plt.figure(figsize=(7,5))
    plt.hist(sigma, bins=25, color="slateblue", alpha=0.8)
    plt.title("Posterior Std Dev Distribution")
    plt.xlabel("σ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

###############################################################################
# METRICS
###############################################################################
def compute_all_metrics(y_true, y_pred, y_prob, labels, run_dir):
    plots = run_dir / "plots"
    metrics_dir = run_dir / "metrics"

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plot_confusion_matrix(cm, labels, plots/"confusion_matrix.png")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )

    df_genre = pd.DataFrame({
        "genre": labels,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": cm.diagonal()/cm.sum(axis=1),
    })
    df_genre.to_csv(metrics_dir/"per_genre_metrics.csv", index=False)

    plot_roc_curves(y_true, y_prob, labels, plots/"roc_curves.png")

    summary = {
        "overall_accuracy": float(np.mean(np.array(y_true)==np.array(y_pred))),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
    }
    with open(metrics_dir/"summary.json","w") as f:
        json.dump(summary,f,indent=2)

###############################################################################
# LATENT BAYESIAN REGRESSION
###############################################################################
def run_bayesian(latents, labels_onehot, cfg_id, run_dir):
    labels_idx = np.argmax(labels_onehot, axis=1)
    plots = run_dir/"plots"
    metrics_dir = run_dir/"metrics"

    br = BayesianRidge()
    br.fit(latents, labels_idx)

    coef = np.abs(br.coef_)
    idx_top = np.argsort(coef)[::-1][:25]

    posterior_var = np.diag(br.sigma_)[idx_top]
    norm_importance = coef[idx_top] / np.max(coef)
    sigma_all = np.sqrt(np.diag(br.sigma_))

    preds = br.predict(latents)
    mae = mean_absolute_error(labels_idx,preds)
    mse = mean_squared_error(labels_idx,preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(labels_idx,preds)

    out = pd.DataFrame({
        "feature_idx": idx_top,
        "abs_weight": coef[idx_top],
        "posterior_var": posterior_var,
        "norm_importance": norm_importance,
    })
    out.to_csv(metrics_dir/"bayesian_top_features.csv", index=False)

    reg_metrics = {
        "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
    }
    with open(metrics_dir/"bayes_regression_metrics.json","w") as f:
        json.dump(reg_metrics,f,indent=2)

    plt.figure(figsize=(8,4))
    plt.bar(out["feature_idx"].astype(str), out["abs_weight"],
            yerr=np.sqrt(out["posterior_var"]), capsize=3)
    plt.xticks(rotation=45)
    plt.title(f"Top Bayesian Latent Features — {cfg_id}")
    plt.tight_layout()
    plt.savefig(plots/"bayesian_top_features.png")
    plt.close()

    plot_latent_correlation(latents, plots/"latent_correlation.png")
    plot_uncertainty_hist(sigma_all, plots/"posterior_uncertainty_hist.png")

    return float(np.mean(sigma_all))

###############################################################################
# TRAIN ONE FOLD  (EARLY STOPPING INSERTED HERE)
###############################################################################
def train_one_fold(cfg, df_tr, df_va, args, label_to_idx, run_dir):
    plots = run_dir / "plots"

    tmp_tr = run_dir / "_tr.csv"
    tmp_va = run_dir / "_va.csv"
    df_tr.to_csv(tmp_tr,index=False)
    df_va.to_csv(tmp_va,index=False)

    tr_ds, _, _, stats = make_dataset_model2(
        repo_root=".",
        img_root=args.img_root,
        features_csv=str(tmp_tr),
        classes=list(label_to_idx.keys()),
        batch=args.batch_size,
        shuffle=True,
        img_size=(args.img_height,args.img_width),
        channels=args.channels
    )
    va_ds, _, _, _ = make_dataset_model2(
        repo_root=".",
        img_root=args.img_root,
        features_csv=str(tmp_va),
        classes=list(label_to_idx.keys()),
        batch=args.batch_size,
        shuffle=False,
        img_size=(args.img_height,args.img_width),
        channels=args.channels
    )
    
    n_tab_features = len(stats)

    model, _ = build_model2_fusion(
        img_shape=(args.img_height,args.img_width,args.channels),
        n_tab_features=n_tab_features,
        dense_units_img=cfg["dense_units_img"],
        dense_units_tab=cfg["dense_units_tab"],
        dropout_tab=cfg["dropout_tab"],
        dropout_backbone=cfg["dropout_backbone"],
        fusion=cfg["fusion"],
        fusion_units=cfg["dense_units_img"],
        num_classes=len(label_to_idx),
        kl_weight=cfg["kl_factor"],
        prior_scale=cfg["prior_scale"],
        posterior_scale_init=cfg["posterior_scale_init"],
        head_type=cfg["head_type"],
    )

    opt = keras.optimizers.Adam(cfg["lr"])
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    ###########################################################################
    # ✔ EARLY STOPPING — ONLY CHANGE MADE
    ###########################################################################
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    ###########################################################################

    history = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=cfg["epochs"],
        callbacks=[early_stop],   # <<---- INSERTED
        verbose=0
    )

    plot_training_curves(history, cfg["id"], plots)

    y_true, y_pred, y_prob = [], [], []
    for X,y in va_ds:
        y_true.extend(np.argmax(y.numpy(),axis=1))
        p = model.predict(X,verbose=0)
        y_pred.extend(np.argmax(p,axis=1))
        y_prob.append(p)
    y_prob = np.vstack(y_prob)

    val_acc = float(np.mean(np.array(y_true)==np.array(y_pred)))
    compute_all_metrics(y_true,y_pred,y_prob,list(label_to_idx.keys()),run_dir)

    tmp_tr.unlink(missing_ok=True)
    tmp_va.unlink(missing_ok=True)

    return val_acc, model

###############################################################################
# RRM
###############################################################################
def compute_rrm(mean_acc, fold_std, U_mean, sigma_max, U_max):
    if sigma_max == 0: sigma_max = 1e-8
    if U_max == 0: U_max = 1e-8

    v0 = 1.0 - mean_acc
    v1 = fold_std / sigma_max
    v2 = U_mean / U_max

    norm_v = math.sqrt(v0*v0 + v1*v1 + v2*v2)
    return 1.0 - norm_v, norm_v

###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--classes", required=True)

    parser.add_argument("--random_sweep", type=int, default=3)
    parser.add_argument("--k_folds", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--img_width", type=int, default=224)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--holdout_frac", type=float, default=0.20)
    parser.add_argument("--holdout_min_cov", type=float, default=0.90)

    parser.add_argument("--out_dir", default="_eda_outputs")
    parser.add_argument("--no_ui", action="store_true")

    args = parser.parse_args()

    df, label_to_idx, all_labels = load_and_prepare_data(args.features_csv)
    sweep_dir, runs_dir, leaders_dir = create_sweep_dirs(args.out_dir)

    with open(sweep_dir/"sweep_args.json","w") as f:
        json.dump(vars(args),f,indent=2)

    hold, remain, cov = build_holdout_with_coverage(
        df, all_labels,
        ratio=args.holdout_frac,
        cov_target=args.holdout_min_cov
    )
    hold.to_csv(sweep_dir/"holdout.csv",index=False)
    remain.to_csv(sweep_dir/"train_val.csv",index=False)

    cfgs = random_configs(args.random_sweep)
    leaderboard = []

    best_row = None
    best_run_dir = None

    global_sigma_max = 1e-8
    global_U_max = 1e-8

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        disable=args.no_ui
    ) as prog:

        task = prog.add_task("Sweeping Configurations", total=len(cfgs))

        for cfg in cfgs:
            run_dir = create_run_dir(runs_dir, cfg["id"])

            with open(run_dir/"config.json","w") as f:
                json.dump(cfg,f,indent=2)

            if not args.no_ui:
                t = Table(title=f"Config {cfg['id']}", header_style="bold magenta")
                for k,v in cfg.items(): t.add_row(k,str(v))
                console.print(t)

            kf = StratifiedKFold(n_splits=args.k_folds,shuffle=True,random_state=42)
            y_all = remain["label"].map(label_to_idx).values

            fold_accs = []
            last_model = None

            for fold,(tr_idx,va_idx) in enumerate(kf.split(remain,y_all),1):
                if not args.no_ui:
                    console.print(c_blue(f"🔹 Fold {fold}/{args.k_folds} — {cfg['id']}"))

                acc,model = train_one_fold(
                    cfg,
                    remain.iloc[tr_idx],
                    remain.iloc[va_idx],
                    args,
                    label_to_idx,
                    run_dir
                )
                fold_accs.append(acc)
                last_model = model

            mean_acc = float(np.mean(fold_accs))
            fold_std = float(np.std(fold_accs))

            ds_full, _, _, _ = make_dataset_model2(
                repo_root=".",
                img_root=args.img_root,
                features_csv=str(sweep_dir/"train_val.csv"),
                classes=list(label_to_idx.keys()),
                batch=args.batch_size,
                shuffle=False,
                img_size=(args.img_height,args.img_width),
                channels=args.channels
            )

            extractor = keras.Model(
                inputs=last_model.inputs,
                outputs=last_model.layers[-3].output
            )

            lat, lab = [], []
            for X,y in ds_full:
                lat.append(extractor(X,training=False).numpy())
                lab.append(y.numpy())
            lat = np.vstack(lat)
            lab = np.concatenate(lab)

            U_mean = run_bayesian(lat,lab,cfg["id"],run_dir)

            global_sigma_max = max(global_sigma_max,fold_std)
            global_U_max = max(global_U_max,U_mean)

            rrm, vnorm = compute_rrm(
                mean_acc,fold_std,U_mean,
                global_sigma_max,global_U_max
            )

            with open(run_dir/"metrics"/"fold_summary.json","w") as f:
                json.dump(
                    {
                        "fold_accuracies": fold_accs,
                        "mean_val_acc": mean_acc,
                        "fold_std": fold_std,
                        "posterior_sigma_mean": U_mean,
                        "v_norm": vnorm,
                        "RRM": rrm,
                    },
                    f, indent=2
                )

            leaderboard.append(
                {
                    "id": cfg["id"],
                    "mean_val_acc": mean_acc,
                    "fold_std": fold_std,
                    "posterior_sigma_mean": U_mean,
                    "RRM": rrm,
                    "run_dir": str(run_dir),
                }
            )
            leaderboard = sorted(leaderboard,key=lambda x: x["mean_val_acc"],reverse=True)

            if not args.no_ui:
                tbl = Table(title="Live Leaderboard",header_style="bold cyan")
                tbl.add_column("Rank")
                tbl.add_column("Config")
                tbl.add_column("Acc")
                tbl.add_column("σ")
                tbl.add_column("Umean")
                tbl.add_column("RRM")

                for i,row in enumerate(leaderboard,1):
                    tbl.add_row(
                        str(i),
                        row["id"],
                        f"{row['mean_val_acc']:.4f}",
                        f"{row['fold_std']:.4f}",
                        f"{row['posterior_sigma_mean']:.4f}",
                        f"{row['RRM']:.4f}",
                    )
                console.print(tbl)

            if best_row is None or mean_acc > best_row["mean_val_acc"]:
                best_row = leaderboard[0]
                best_run_dir = Path(best_row["run_dir"])

                stamp = datetime.now().strftime("%H%M%S")
                dest = leaders_dir / f"{best_row['id']}_{stamp}"
                shutil.copytree(best_run_dir,dest)

            prog.advance(task)

    leaderboard_csv = sweep_dir/"leaderboard.csv"
    leaderboard_json = sweep_dir/"leaderboard.json"

    pd.DataFrame(leaderboard).to_csv(leaderboard_csv,index=False)
    with open(leaderboard_json,"w") as f:
        json.dump(leaderboard,f,indent=2)

    console.print("")
    console.print(c_green("🏁 Sweep complete!"))
    console.print(c_green(f"📊 Leaderboard saved: {leaderboard_csv}"))
    console.print(c_green(f"🏆 Leaders folder: {leaders_dir}"))


if __name__ == "__main__":
    main()
