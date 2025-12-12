# _code/data/loaders_model2.py
# -*- coding: utf-8 -*-
"""
Model 2 dataset: fuse spectrogram PNGs (images) with 30s tabular features.
- Matches by normalized ID (e.g., 'blues00000') derived from image filename and CSV.
- Automatically picks numeric feature columns.
- Returns tf.data.Dataset yielding ((image, tab_vector), one_hot_label).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import tensorflow as tf


def _normalize_id(x: str) -> str:
    """Make IDs comparable: basename, drop extension, remove ._- and lower."""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    x = x.replace("\\", "/").split("/")[-1]              # basename
    x = re.sub(r"\.(wav|au|mp3|flac|ogg|png|jpg|jpeg)$", "", x, flags=re.I)
    x = x.replace(".", "").replace("-", "").replace("_", "")
    return x.lower()


def _load_image(path: tf.Tensor, img_size: Tuple[int, int], channels: int):
    img_bytes = tf.io.read_file(path)
    # spectrograms are grayscale PNGs in your repo; channels still configurable
    img = tf.io.decode_png(img_bytes, channels=channels)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def _load_features_csv(features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Read features CSV and pick numeric feature columns.
    Returns (df_with__norm_id, numeric_feature_columns).
    """
    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError(f"Features CSV is empty: {features_csv}")

    # Heuristics: look for a column that contains the filename
    # Common GTZAN variants: 'filename', 'slice_file_name', 'file', 'track'
    candidates = ["filename", "slice_file_name", "file", "track", "path", "name"]
    key_col = None
    for c in candidates:
        if c in df.columns:
            key_col = c
            break
    if key_col is None:
        # Fall-back: try the first column if it's object-like
        obj_cols = [c for c in df.columns if df[c].dtype == "O"]
        if not obj_cols:
            raise KeyError(
                f"Could not find a filename-like column in {features_csv}. "
                f"Tried {candidates}; object-like columns={obj_cols}"
            )
        key_col = obj_cols[0]

    df = df.copy()
    df["_norm_id"] = df[key_col].astype(str).map(_normalize_id)

    # Select numeric features (exclude obvious label-ish columns)
    banned = {"label", "genre", "class", key_col, "_norm_id", "length"}
    num_cols = []
    for c in df.columns:
        if c in banned:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            num_cols.append(c)
    if not num_cols:
        raise ValueError(
            f"No numeric feature columns detected in {features_csv}. "
            f"Columns: {list(df.columns)[:20]}..."
        )

    return df, num_cols


def _scan_image_paths(img_root: Path, classes: List[str]) -> tuple[list[str], list[int], list[str]]:
    """
    Discover image files under <img_root>/<genre>/*.png and build:
    - paths: list[str]
    - label_idxs: list[int]
    - norm_ids: list[str] (from filenames)
    """
    paths: list[str] = []
    labels: list[int] = []
    ids: list[str] = []

    label_map = {g: i for i, g in enumerate(classes)}
    for g in classes:
        genre_dir = img_root / g
        if not genre_dir.exists():
            continue
        for p in sorted(genre_dir.glob("*.png")):
            paths.append(str(p))
            labels.append(label_map[g])
            ids.append(_normalize_id(p.name))  # filename -> norm id

    return paths, labels, ids


def make_dataset_model2(
    *,
    repo_root: str,
    img_root: str,
    features_csv: str,
    classes: List[str],
    batch: int = 32,
    shuffle: bool = True,
    img_size: Tuple[int, int] = (224, 224),
    channels: int = 1,
):
    """
    Build a fused dataset of ((image, tabular_features), one_hot_label).

    Args:
        repo_root: project root
        img_root:  relative path to spectrogram roots (genre subdirs)
        features_csv: path to features_30_sec.csv (abs or relative to repo_root)
        classes: ordered class names
        batch, shuffle, img_size, channels: standard tf.data args

    Returns:
        ds: tf.data.Dataset yielding ((image, tab), y_one_hot)
        label_map: dict[str,int]
        n: number of matched samples
        used_feat_cols: list[str] of tabular features used
    """
    repo_root = Path(repo_root)
    img_root_path = (repo_root / img_root) if not Path(img_root).is_absolute() else Path(img_root)
    feat_csv_path = (repo_root / features_csv) if not Path(features_csv).is_absolute() else Path(features_csv)

    if not feat_csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found:\n  {feat_csv_path}\n"
            f"Update config to point to the correct 'features_30_sec.csv'."
        )

    # Load features + choose numeric cols
    df_feat, feat_cols = _load_features_csv(str(feat_csv_path))

    # Scan all images (respecting provided class list)
    img_paths, label_idxs, img_ids = _scan_image_paths(img_root_path, classes)
    label_map = {g: i for i, g in enumerate(classes)}

    # Align by _norm_id
    feat_index = df_feat.set_index("_norm_id")
    tab_rows = []
    keep_paths = []
    keep_labels = []
    miss_ids = 0

    for p, y, nid in zip(img_paths, label_idxs, img_ids):
        if nid in feat_index.index:
            row = feat_index.loc[nid, feat_cols]
            # If duplicates exist, pandas may return Series or DataFrame; take first if needed
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            tab_rows.append(row.values.astype("float32"))
            keep_paths.append(p)
            keep_labels.append(y)
        else:
            miss_ids += 1

    if not keep_paths:
        raise RuntimeError(
            "No aligned samples between images and features CSV.\n"
            f"  img_root: {img_root_path}\n"
            f"  features: {feat_csv_path}\n"
            "Check that IDs like 'blues00000' exist in the CSV (after normalization)."
        )

    # Stats printout (helps debug coverage)
    n_total = len(img_paths)
    n_keep = len(keep_paths)
    print(
        f"🧩 Alignment: kept {n_keep}/{n_total} spectrograms "
        f"(missing in CSV: {miss_ids})"
    )
    # Build tensors
    X_tab = np.stack(tab_rows, axis=0)  # [N, F]
    y_idx = np.array(keep_labels, dtype=np.int32)
    paths_np = np.array(keep_paths, dtype=np.string_)  # tf likes bytes for file paths

    num_classes = len(classes)

    ds_paths = tf.data.Dataset.from_tensor_slices(paths_np)
    ds_tab   = tf.data.Dataset.from_tensor_slices(X_tab)
    ds_lbl   = tf.data.Dataset.from_tensor_slices(y_idx)

    ds = tf.data.Dataset.zip((ds_paths, ds_tab, ds_lbl))

    if shuffle:
        ds = ds.shuffle(buffer_size=max(1024, n_keep), reshuffle_each_iteration=True)

    autotune = tf.data.AUTOTUNE

    def _encode(path, tab_vec, label_idx):
        img = _load_image(path, img_size, channels)
        y = tf.one_hot(label_idx, num_classes)
        return (img, tf.cast(tab_vec, tf.float32)), y

    ds = ds.map(_encode, num_parallel_calls=autotune)
    ds = ds.batch(batch).prefetch(autotune)

    return ds, label_map, n_keep, feat_cols


