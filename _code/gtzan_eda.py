#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTZAN EDA & Split Script
- Auto-detects repo root from this file location: <repo>/_code/gtzan_eda.py
- Counts audio & spectrograms, loads 3s/30s feature CSVs, basic QC, and stratified splits.
"""

from __future__ import annotations
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# Optional dependency
try:
    import librosa  # type: ignore
except Exception:
    librosa = None

# ---------------- CONFIG ---------------- #
# This file lives in <repo>/_code/gtzan_eda.py
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[1]                     # <repo>
DATA_DIR  = ROOT_DIR / "_data" / "gtzan_kaggle" / "Data"
OUTDIR    = ROOT_DIR / "_eda_outputs"

AUDIO_DIR     = DATA_DIR / "genres_original"
IMG_COLOR_DIR = DATA_DIR / "images_original"
IMG_GRAY_DIR  = DATA_DIR / "images_grey_scale"

FEATURES_30 = DATA_DIR / "features_30_sec.csv"
FEATURES_3  = DATA_DIR / "features_3_sec.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Ensure output dir
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------- HELPERS ---------------- #
def _barplot(counts: dict[str, int], title: str, outpath: Path) -> None:
    if not counts:
        print(f"⚠️  Skipping plot '{title}' — no data.")
        return
    plt.figure(figsize=(9.5, 4.5))
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.bar(keys, vals)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _count_leaf_dirs_by_ext(base: Path, exts: tuple[str, ...]) -> dict[str, int]:
    """
    Counts files per immediate leaf directory (genre) by extension.
    Example: <base>/<genre>/*.ext
    """
    if not base.exists():
        return {}
    counts = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        n = 0
        for ext in exts:
            n += sum(1 for _ in d.glob(f"*{ext}"))
        counts[d.name.lower()] = n
    return counts


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # filename / file -> base_id
    if "filename" in df.columns:
        src = "filename"
    elif "file" in df.columns:
        src = "file"
    else:
        src = None

    if src:
        df["base_id"] = (
            df[src]
            .astype(str)
            .apply(lambda x: Path(x).stem.split("-")[0])
        )

    # establish 'genre'
    if "label" in df.columns and "genre" not in df.columns:
        df["genre"] = df["label"].astype(str).str.lower()
    if "genre" not in df.columns:
        # fallback: infer from base_id if present
        if "base_id" in df.columns:
            df["genre"] = df["base_id"].astype(str).apply(lambda s: s.split(".")[0].lower())
        else:
            # leave missing; caller can handle
            df["genre"] = pd.NA
    return df


def _probe_durations(audio_root: Path, n: int = 10):
    if librosa is None:
        print("ℹ️  librosa not installed; skipping duration probe.")
        return None
    files = list(audio_root.rglob("*.wav"))
    if not files:
        print("ℹ️  No .wav files found for duration probe.")
        return None
    sample = random.sample(files, min(n, len(files)))
    durations = []
    for f in sample:
        try:
            y, sr = librosa.load(f, sr=None)
            durations.append(len(y) / float(sr))
        except Exception:
            pass
    if durations:
        return float(np.mean(durations)), float(np.std(durations))
    return None


# ---------------- MAIN EDA ---------------- #
def run_eda() -> None:
    # Sanity banner
    print(f"📂 ROOT: {ROOT_DIR}")
    print(f"📁 DATA_DIR: {DATA_DIR}  (exists={DATA_DIR.exists()})")
    print(f"📄 30s CSV: {FEATURES_30}  (exists={FEATURES_30.exists()})")
    print(f"📄 3s  CSV: {FEATURES_3}   (exists={FEATURES_3.exists()})")
    print(f"📦 OUTDIR: {OUTDIR}\n")

    # --- Inventory --- #
    print("🔍 Counting files by genre...")
    audio_counts = _count_leaf_dirs_by_ext(AUDIO_DIR, (".wav",))
    color_counts = _count_leaf_dirs_by_ext(IMG_COLOR_DIR, (".png", ".jpg", ".jpeg"))
    gray_counts  = _count_leaf_dirs_by_ext(IMG_GRAY_DIR,  (".png", ".jpg", ".jpeg"))

    print(f"Audio counts: {audio_counts}")
    print(f"Color spectrogram counts: {color_counts}")
    print(f"Gray  spectrogram counts: {gray_counts}")

    _barplot(audio_counts, "Audio Files per Genre", OUTDIR / "audio_balance.png")
    _barplot(color_counts, "Color Spectrograms per Genre", OUTDIR / "color_balance.png")
    _barplot(gray_counts,  "Gray Spectrograms per Genre",  OUTDIR / "gray_balance.png")

    # --- Feature CSVs --- #
    print("\n📄 Loading feature files...")
    try:
        df30 = _safe_read_csv(FEATURES_30)
        print(f"features_30_sec: {df30.shape}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        df30 = pd.DataFrame()

    try:
        df3 = _safe_read_csv(FEATURES_3)
        print(f"features_3_sec:  {df3.shape}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        df3 = pd.DataFrame()

    # Missing & duplicates (only if loaded)
    for name, df in (("30s", df30), ("3s", df3)):
        if df.empty:
            continue
        print(f"\n[{name}] Missing values (total): {int(df.isna().sum().sum())}")
        print(f"[{name}] Duplicated rows: {int(df.duplicated().sum())}")

        # class balance plots
        if "genre" in df.columns and df["genre"].notna().any():
            balance = df["genre"].astype(str).str.lower().value_counts().to_dict()
            _barplot(balance, f"{name} Feature Genres", OUTDIR / f"features_{name}_balance.png")
        else:
            print(f"[{name}] ⚠️  'genre' column missing or empty; skipping balance plot.")

    # --- Alignment Check (30s vs audio) --- #
    if not df30.empty and "base_id" in df30.columns:
        audio_bases = {p.stem.split(".")[0] for p in AUDIO_DIR.rglob("*.wav")}
        csv_bases   = set(df30["base_id"].dropna().unique())
        missing_in_csv = audio_bases - csv_bases
        print(f"\n⚖️  Missing feature rows for {len(missing_in_csv)} audio tracks.")
    else:
        print("\n⚖️  Skipping alignment check (30s features unavailable).")

    # --- Duration Probe --- #
    probe = _probe_durations(AUDIO_DIR, n=10)
    if probe:
        mu, sig = probe
        print(f"⏱️  Avg duration: {mu:.2f}s ± {sig:.2f}")

    # --- Stratified Splits (30s level) --- #
    print("\n🧩 Creating train/val/test splits...")
    if not df30.empty and {"base_id", "genre"}.issubset(df30.columns):
        groups = df30[["base_id", "genre"]].dropna().drop_duplicates()
        X, y = groups["base_id"].astype(str).values, groups["genre"].astype(str).values

        # test = 15%
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
        (train_idx, test_idx), = sss.split(X, y)
        train_df = groups.iloc[train_idx].reset_index(drop=True)
        test_df  = groups.iloc[test_idx ].reset_index(drop=True)

        # val = 15% of total -> 0.176 of remaining (~15/85)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_SEED)
        (tr_idx, val_idx), = sss2.split(train_df["base_id"], train_df["genre"])
        train_final = train_df.iloc[tr_idx].reset_index(drop=True)
        val_final   = train_df.iloc[val_idx].reset_index(drop=True)

        train_final.to_csv(OUTDIR / "train_split.csv", index=False)
        val_final.to_csv(OUTDIR / "val_split.csv", index=False)
        test_df.to_csv(OUTDIR / "test_split.csv", index=False)
        print(f"✅ Saved splits under {OUTDIR}")
        print(f"   - train_split.csv: {len(train_final)} rows")
        print(f"   - val_split.csv:   {len(val_final)} rows")
        print(f"   - test_split.csv:  {len(test_df)} rows")
    else:
        print("⚠️  Skipping splits — need non-empty 30s features with ['base_id','genre'].")


if __name__ == "__main__":
    run_eda()
