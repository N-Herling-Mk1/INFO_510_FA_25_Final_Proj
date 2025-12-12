#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sweep_model2_mk5l.py
Full Run-Isolated Sweep System (2025-11-26)
───────────────────────────────────────────────────────────────────────────────
• Each run gets its own folder.
• Each fold writes plots + metrics.
• Bayesian latent regression + uncertainty diagrams.
• Per-genre accuracy / precision / recall / F1.
• ROC curves (one-vs-rest) per class.
• Confusion matrix + correlation heatmap.
• MAE + MSE + RMSE + R² for latent→label regression.
• Live leaderboard in terminal.
• Final sweep leaderboard CSV + JSON.
• Leader duplicated into its own “leaders/” folder.
• SILENCES TF & sklearn warnings.
───────────────────────────────────────────────────────────────────────────────
"""

###############################################################################
# GLOBAL SETUP — Warning Suppression
###############################################################################
import warnings
warnings.filterwarnings("ignore")                     # sklearn, numpy, misc
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"              # TensorFlow internal logs
os.environ["PYTHONWARNINGS"] = "ignore"               # Python-level suppress
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

###############################################################################
# Imports
###############################################################################
import json, argparse, shutil
from pathlib import Path
from datetime import datetime

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

# Local modules
from _code.data.loaders_model2 import make_dataset_model2
from _code.models.model2_fusion_mk2 import build_model2_fusion

console = Console()

###############################################################################
# Formatting Helpers
###############################################################################
def c_green(x): return f"[green]{x}[/green]"
def c_yellow(x): return f"[yellow]{x}[/yellow]"
def c_blue(x):   return f"[cyan]{x}[/cyan]"
def c_red(x):    return f"[red]{x}[/red]"


###############################################################################
# OUTPUT DIRECTORY SYSTEM
###############################################################################
def create_sweep_dirs(root_dir: str):
    """
    sweep_dir/
        sweep_args.json
        holdout.csv
        train_val.csv
        runs/
            cfg_xxxxxx/
                plots/
                metrics/
        leaders/
    """
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
    """
    runs/cfg_<id>_<HHMMSSms>/
         metrics/
         plots/
    """
    stamp = datetime.now().strftime("%H%M%S%f")[:-3]
    run_dir = runs_dir / f"{cfg_id}_{stamp}"
    run_dir.mkdir()

    (run_dir / "plots").mkdir()
    (run_dir / "metrics").mkdir()

    return run_dir


###############################################################################
# DATA LOADING + NORMALIZATION
###############################################################################
def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    console.print(c_blue(f"📥 Loaded CSV: {len(df):,} rows"))

    # Identify label column
    candidates = ["label", "genre", "class", "y"]
    label_col = next((c for c in candidates if c in df.columns), None)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    labels = sorted(df["label"].unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Z-normalize numeric columns
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
# HOLDOUT SPLIT w/ COVERAGE
###############################################################################
def build_holdout_with_coverage(
    df, classes, ratio=0.15, cov_target=0.90, max_tries=12
):
    for t in range(1, max_tries + 1):
        df_shuf = df.sample(frac=1.0, random_state=50 + t).reset_index(drop=True)
        n = int(len(df_shuf) * ratio)
        hold = df_shuf[:n]
        cov = len(hold["label"].unique()) / len(classes)
        if cov >= cov_target:
            console.print(c_green(f"✔ Holdout coverage {cov:.2f} (attempt {t})"))
            return hold, df_shuf.drop(hold.index), cov
        console.print(c_yellow(f"⚠ Coverage {cov:.2f} < {cov_target}, retry {t}"))

    # fallback
    return hold, df_shuf.drop(hold.index), cov


###############################################################################
# CONFIG GENERATION
###############################################################################
def random_configs(k: int):
    """
    Updated hyperparameter ranges based on:
      • First sweep diagnostics (mk5l + mk5m)
      • RRM stability terms (σ_folds, U_mean)
      • Avoiding extremely unstable Bayesian heads
      • Slight LR upward shift since 1e-4.5–1e-3 performed best
      • Boosting tabular width slightly (benefits fusion stability)
    """

    return [
        {
            "id": f"cfg_{i}",

            # --- Dropout ---
            # Backbone: Slightly increased MIN dropout → more stable σ_folds, better RRM
            "dropout_backbone": float(np.random.uniform(0.18, 0.32)),
            # Tabular: Upper bound kept (0.40) but floor raised to 0.12 for improved fusion reliability
            "dropout_tab":      float(np.random.uniform(0.12, 0.40)),

            # --- Dense units ---
            # CNN top layers: all values seen working; 512–768 best balance.
            "dense_units_img":  int(np.random.choice([384, 512, 768])),
            # Tabular: up-weighted since 64 was underperforming in mk5l
            "dense_units_tab":  int(np.random.choice([96, 128, 192, 256])),

            # --- Learning Rate ---
            # mk5l/mk5m showed that too-low LR underfit; we tighten LR upward.
            "lr":               float(10 ** np.random.uniform(-3.8, -3.0)),

            # --- Epochs ---
            # 75 usually better, but 50 stays for exploration.
            "epochs":           int(np.random.choice([50, 75])),

            # --- Optimizer ---
            # adam and adamw performed consistently; rmsprop kept for diversity.
            "optimizer":        str(np.random.choice(["adam", "adamw", "rmsprop"])),

            # --- Fusion strategy ---
            "fusion":           str(np.random.choice(["concat", "gated"])),

            # --- Classification head ---
            # flipout and mc_dropout both allowed; det kept for control models.
            "head_type":        str(np.random.choice(["flipout", "mc_dropout", "det"])),

            # --- Bayesian Hyperparameters ---
            # KL term expanded upward (1.1–1.5 had best stability in mk5l)
            "kl_factor":        float(np.random.uniform(1.0, 1.6)),
            # Prior widened; low prior (<0.6) was unstable, so removed.
            "prior_scale":      float(np.random.uniform(0.8, 1.6)),
            # Posterior init tightened (previous upper bound 0.03 produced unstable σ)
            "posterior_scale_init": float(np.random.uniform(0.005, 0.02)),
        }
        for i in range(k)
    ]


###############################################################################
# PLOTTING UTILITIES (PER-RUN)
###############################################################################
def plot_training_curves(history, cfg_id, plots_dir):
    fig, ax1 = plt.subplots(figsize=(7, 5))
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
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curves(y_true, y_prob, labels, out_path):
    """
    y_true: integer labels
    y_prob: softmax probabilities (N × C)
    """
    plt.figure(figsize=(8, 6))

    n_classes = len(labels)
    for c in range(n_classes):
        y_bin = (np.array(y_true) == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{labels[c]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves — One-vs-Rest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latent_correlation(latents, out_path):
    corr = np.corrcoef(latents.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Latent Feature Correlation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_uncertainty_hist(sigma, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(sigma, bins=25, color="slateblue", alpha=0.8)
    plt.title("Posterior Std Dev Distribution")
    plt.xlabel("σ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


###############################################################################
# METRIC RECORDING
###############################################################################
def compute_all_metrics(y_true, y_pred, y_prob, labels, run_dir):
    plots = run_dir / "plots"
    metrics_dir = run_dir / "metrics"

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plot_confusion_matrix(cm, labels, plots / "confusion_matrix.png")

    # Per-genre metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )

    df_genre = pd.DataFrame({
        "genre": labels,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": cm.diagonal() / cm.sum(axis=1),
    })
    df_genre.to_csv(metrics_dir / "per_genre_metrics.csv", index=False)

    # ROC curves
    plot_roc_curves(y_true, y_prob, labels, plots / "roc_curves.png")

    # Summary JSON
    summary = {
        "overall_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
    }
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


###############################################################################
# BAYESIAN LATENT REGRESSION
###############################################################################
def run_bayesian(latents, labels_onehot, cfg_id, run_dir):
    labels_idx = np.argmax(labels_onehot, axis=1)
    plots = run_dir / "plots"
    metrics_dir = run_dir / "metrics"

    br = BayesianRidge()
    br.fit(latents, labels_idx)

    coef = np.abs(br.coef_)
    idx_top = np.argsort(coef)[::-1][:25]

    posterior_var = np.diag(br.sigma_)[idx_top]
    norm_importance = coef[idx_top] / np.max(coef)
    sigma_all = np.sqrt(np.diag(br.sigma_))

    # Regression metrics
    preds = br.predict(latents)
    mae = mean_absolute_error(labels_idx, preds)
    mse = mean_squared_error(labels_idx, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(labels_idx, preds)

    # Save summary
    out = pd.DataFrame({
        "feature_idx": idx_top,
        "abs_weight": coef[idx_top],
        "posterior_var": posterior_var,
        "norm_importance": norm_importance,
    })
    out.to_csv(metrics_dir / "bayesian_top_features.csv", index=False)

    reg_metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
    with open(metrics_dir / "bayes_regression_metrics.json", "w") as f:
        json.dump(reg_metrics, f, indent=2)

    # Feature bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(out["feature_idx"].astype(str), out["abs_weight"],
            yerr=np.sqrt(out["posterior_var"]), capsize=3)
    plt.xticks(rotation=45)
    plt.title(f"Top Bayesian Latent Features — {cfg_id}")
    plt.tight_layout()
    plt.savefig(plots / "bayesian_top_features.png")
    plt.close()

    # Correlation heatmap
    plot_latent_correlation(latents, plots / "latent_correlation.png")

    # Posterior uncertainty histogram
    plot_uncertainty_hist(sigma_all, plots / "posterior_uncertainty_hist.png")


###############################################################################
# TRAIN ONE FOLD
###############################################################################
def train_one_fold(cfg, df_tr, df_va, args, label_to_idx, run_dir):
    plots = run_dir / "plots"

    # Write temp CSVs (loader requirement)
    tmp_tr = run_dir / "_tr.csv"
    tmp_va = run_dir / "_va.csv"
    df_tr.to_csv(tmp_tr, index=False)
    df_va.to_csv(tmp_va, index=False)

    # Build datasets
    tr_ds, _, _, stats = make_dataset_model2(
        repo_root=".",
        img_root=args.img_root,
        features_csv=str(tmp_tr),
        classes=list(label_to_idx.keys()),
        batch=args.batch_size,
        shuffle=True,
        img_size=(args.img_height, args.img_width),
        channels=args.channels
    )
    va_ds, _, _, _ = make_dataset_model2(
        repo_root=".",
        img_root=args.img_root,
        features_csv=str(tmp_va),
        classes=list(label_to_idx.keys()),
        batch=args.batch_size,
        shuffle=False,
        img_size=(args.img_height, args.img_width),
        channels=args.channels
    )
    n_tab_features = len(stats)

    # Build model
    model, _ = build_model2_fusion(
        img_shape=(args.img_height, args.img_width, args.channels),
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

    # Train
    history = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=cfg["epochs"],
        verbose=0
    )

    plot_training_curves(history, cfg["id"], plots)

    # Predict
    y_true, y_pred, y_prob = [], [], []
    for X, y in va_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        p = model.predict(X, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_prob.append(p)
    y_prob = np.vstack(y_prob)

    val_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    # Compute metrics
    compute_all_metrics(y_true, y_pred, y_prob, list(label_to_idx.keys()), run_dir)

    # Clean temporary files
    tmp_tr.unlink(missing_ok=True)
    tmp_va.unlink(missing_ok=True)

    return val_acc, model


###############################################################################
# MAIN SWEEP LOOP
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

    # Save global args
    with open(sweep_dir / "sweep_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Holdout creation
    hold, remain, cov = build_holdout_with_coverage(
        df, all_labels,
        ratio=args.holdout_frac,
        cov_target=args.holdout_min_cov
    )
    hold.to_csv(sweep_dir / "holdout.csv", index=False)
    remain.to_csv(sweep_dir / "train_val.csv", index=False)

    # Sweep configs
    cfgs = random_configs(args.random_sweep)
    leaderboard = []
    best_row = None
    best_run_dir = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        disable=args.no_ui
    ) as prog:

        task = prog.add_task("Sweeping configs", total=len(cfgs))

        for cfg in cfgs:
            # Directory for this run
            run_dir = create_run_dir(runs_dir, cfg["id"])

            # Save config
            with open(run_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)

            # Display config
            if not args.no_ui:
                t = Table(title=f"Config {cfg['id']}", header_style="bold magenta")
                for k, v in cfg.items():
                    t.add_row(k, str(v))
                console.print(t)

            # Cross-validation
            kf = StratifiedKFold(
                n_splits=args.k_folds, shuffle=True, random_state=42
            )
            y_all = remain["label"].map(label_to_idx).values

            fold_accs = []
            last_model = None

            for fold, (tr_idx, va_idx) in enumerate(kf.split(remain, y_all), 1):
                if not args.no_ui:
                    console.print(
                        c_blue(f"🔹 Fold {fold}/{args.k_folds} — {cfg['id']}")
                    )

                acc, model = train_one_fold(
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

            # Save fold summary
            with open(run_dir / "metrics" / "fold_summary.json", "w") as f:
                json.dump(
                    {"fold_accuracies": fold_accs, "mean_val_acc": mean_acc},
                    f,
                    indent=2
                )

            # Extract latents + Bayesian diag
            if last_model is not None:
                ds_full, _, _, _ = make_dataset_model2(
                    repo_root=".",
                    img_root=args.img_root,
                    features_csv=str(sweep_dir / "train_val.csv"),
                    classes=list(label_to_idx.keys()),
                    batch=args.batch_size,
                    shuffle=False,
                    img_size=(args.img_height, args.img_width),
                    channels=args.channels
                )

                extractor = keras.Model(
                    inputs=last_model.inputs,
                    outputs=last_model.layers[-3].output
                )

                lat, lab = [], []
                for X, y in ds_full:
                    lat.append(extractor(X, training=False).numpy())
                    lab.append(y.numpy())
                lat = np.vstack(lat)
                lab = np.concatenate(lab)

                run_bayesian(lat, lab, cfg["id"], run_dir)

            # Update leaderboard
            leaderboard.append(
                {"id": cfg["id"], "mean_val_acc": mean_acc, "run_dir": str(run_dir)}
            )
            leaderboard = sorted(
                leaderboard, key=lambda x: x["mean_val_acc"], reverse=True
            )

            # Print live leaderboard
            if not args.no_ui:
                tbl = Table(title="Live Leaderboard", header_style="bold cyan")
                tbl.add_column("Rank")
                tbl.add_column("Config")
                tbl.add_column("Val Acc")
                for i, row in enumerate(leaderboard, 1):
                    tbl.add_row(str(i), row["id"], f"{row['mean_val_acc']:.4f}")
                console.print(tbl)

            # Best snapshot
            if best_row is None or mean_acc > best_row["mean_val_acc"]:
                best_row = leaderboard[0]
                best_run_dir = Path(best_row["run_dir"])

                # Copy to leaders/
                stamp = datetime.now().strftime("%H%M%S")
                dest = leaders_dir / f"{best_row['id']}_{stamp}"
                shutil.copytree(best_run_dir, dest)

            prog.advance(task)

    # Save final leaderboard
    leaderboard_csv = sweep_dir / "leaderboard.csv"
    leaderboard_json = sweep_dir / "leaderboard.json"
    pd.DataFrame(leaderboard).to_csv(leaderboard_csv, index=False)
    with open(leaderboard_json, "w") as f:
        json.dump(leaderboard, f, indent=2)

    console.print("")
    console.print(c_green("🏁 Sweep complete!"))
    console.print(f"📂 Sweep folder: {sweep_dir}")
    console.print(f"📊 Leaderboard CSV: {leaderboard_csv}")
    console.print(f"🏆 Leaders folder:  {leaders_dir}")


if __name__ == "__main__":
    main()

