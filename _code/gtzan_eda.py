#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTZAN EDA & Split Script
Root directory: INFO_510_FA_25_Final_Proj
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import librosa
except ImportError:
    librosa = None

# ---------------- CONFIG ---------------- #
ROOT_DIR = Path(r"INFO_510_FA_25_Final_Proj")
DATA_DIR = ROOT_DIR / "_data" / "gtzan_kaggle" / "Data"

AUDIO_DIR = DATA_DIR / "genres_original"
IMG_COLOR_DIR = DATA_DIR / "images_original"
IMG_GRAY_DIR = DATA_DIR / "images_grey_scale"

FEATURES_30 = DATA_DIR / "features_30_sec.csv"
FEATURES_3 = DATA_DIR / "features_3_sec.csv"

OUTDIR = ROOT_DIR / "_eda_outputs"
OUTDIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42


# ---------------- HELPERS ---------------- #
def count_files_by_genre(base_path: Path, exts=(".wav", ".png", ".jpg", ".jpeg")):
    counts = Counter()
    for ext in exts:
        for f in base_path.rglob(f"*{ext}"):
            genre = f.parent.name.lower()
            counts[genre] += 1
    return dict(sorted(counts.items()))


def report_balance(counts: dict, title: str, outpath: Path):
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "filename" in df.columns:
        df["base_id"] = df["filename"].apply(lambda x: Path(x).stem.split("-")[0])
    elif "file" in df.columns:
        df["base_id"] = df["file"].apply(lambda x: Path(x).stem.split("-")[0])
    if "label" in df.columns:
        df["genre"] = df["label"].str.lower()
    elif "genre" not in df.columns:
        df["genre"] = df["base_id"].apply(lambda s: s.split(".")[0])
    return df


def probe_durations(n=10):
    if librosa is None:
        print("librosa not installed; skipping duration probe.")
        return None
    import random
    files = list(AUDIO_DIR.rglob("*.wav"))
    sample = random.sample(files, min(n, len(files)))
    durations = []
    for f in sample:
        try:
            y, sr = librosa.load(f, sr=None)
            durations.append(len(y) / sr)
        except Exception:
            pass
    if durations:
        return np.mean(durations), np.std(durations)
    return None


# ---------------- MAIN EDA ---------------- #
def run_eda():
    print(f"📂 Root: {ROOT_DIR}")
    print(f"📊 Output: {OUTDIR}\n")

    # --- Inventory --- #
    print("🔍 Counting files by genre...")
    audio_counts = count_files_by_genre(AUDIO_DIR, (".wav",))
    color_counts = count_files_by_genre(IMG_COLOR_DIR)
    gray_counts = count_files_by_genre(IMG_GRAY_DIR)

    print(f"Audio counts: {audio_counts}")
    print(f"Color spectrogram counts: {color_counts}")
    print(f"Gray spectrogram counts: {gray_counts}")

    report_balance(audio_counts, "Audio Files per Genre", OUTDIR / "audio_balance.png")
    report_balance(color_counts, "Color Spectrograms per Genre", OUTDIR / "color_balance.png")
    report_balance(gray_counts, "Gray Spectrograms per Genre", OUTDIR / "gray_balance.png")

    # --- Feature CSVs --- #
    print("\n📄 Loading feature files...")
    df30 = read_features(FEATURES_30)
    df3 = read_features(FEATURES_3)

    print(f"features_30_sec: {df30.shape}")
    print(f"features_3_sec: {df3.shape}")

    # Missing & duplicates
    for name, df in {"30s": df30, "3s": df3}.items():
        print(f"\n[{name}] Missing values: {df.isna().sum().sum()}")
        print(f"[{name}] Duplicated rows: {df.duplicated().sum()}")

    # --- Class balance --- #
    report_balance(df30["genre"].value_counts().to_dict(),
                   "30s Feature Genres", OUTDIR / "features_30_balance.png")
    report_balance(df3["genre"].value_counts().to_dict(),
                   "3s Feature Genres", OUTDIR / "features_3_balance.png")

    # --- Alignment Check --- #
    base30_audio = {Path(f).stem.split(".")[0] for f in AUDIO_DIR.rglob("*.wav")}
    base30_csv = set(df30["base_id"].unique())
    missing_in_csv = base30_audio - base30_csv
    print(f"\n⚖️ Missing feature rows for {len(missing_in_csv)} audio tracks.")

    # --- Duration Probe --- #
    probe = probe_durations(n=10)
    if probe:
        print(f"⏱️ Avg duration: {probe[0]:.2f}s ± {probe[1]:.2f}")

    # --- Stratified Splits (30s level) --- #
    print("\n🧩 Creating train/val/test splits...")
    groups = df30[["base_id", "genre"]].drop_duplicates()
    X, y = groups["base_id"], groups["genre"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    for train_idx, test_idx in sss.split(X, y):
        train_df = groups.iloc[train_idx]
        test_df = groups.iloc[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_SEED)
    for tr_idx, val_idx in sss2.split(train_df["base_id"], train_df["genre"]):
        train_final = train_df.iloc[tr_idx]
        val_final = train_df.iloc[val_idx]

    train_final.to_csv(OUTDIR / "train_split.csv", index=False)
    val_final.to_csv(OUTDIR / "val_split.csv", index=False)
    test_df.to_csv(OUTDIR / "test_split.csv", index=False)

    print("✅ Saved splits under", OUTDIR)


if __name__ == "__main__":
    run_eda()
