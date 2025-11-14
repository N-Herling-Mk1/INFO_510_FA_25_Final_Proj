#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_3sec_clips.py

Model 3 helper script:
- INPUT:  30s grayscale spectrograms (one per original track)
- INPUT:  features_3_sec.csv (10 × 3s rows per track)
- OUTPUT: 3s spectrogram crops as PNGs, aligned to features_3_sec.csv

For each row in features_3_sec.csv, e.g.:
    filename = "blues.00000.3.wav"
    label    = "blues"

we:
  1. Find the corresponding 30s spectrogram image for "blues.00000"
  2. Slice its width into 10 equal (or near-equal) vertical strips
  3. Save the 3s crop as:
       images_grey_scale_3sec/<label>/blues.00000.3.png

Also writes a mapping CSV:
  _data/gtzan_kaggle/Data/images_3sec_map.csv
"""

from __future__ import annotations
import os
from pathlib import Path
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# .../_code/_supplemental_scripts/make_3sec_clips.py
REPO_ROOT = THIS_FILE.parents[2]   # -> INFO_510_FA_25_Final_Proj

IMG30_ROOT = REPO_ROOT / "_data" / "gtzan_kaggle" / "Data" / "images_grey_scale"
FEAT3_CSV  = REPO_ROOT / "_data" / "gtzan_kaggle" / "Data" / "features_3_sec.csv"
OUT_ROOT   = REPO_ROOT / "_data" / "gtzan_kaggle" / "Data" / "images_grey_scale_3sec"
MAP_CSV    = REPO_ROOT / "_data" / "gtzan_kaggle" / "Data" / "images_3sec_map.csv"


def _print_banner():
    print("-" * 60)
    print(" GTZAN → 3-Second Spectrogram Slice Generator (Model 3)")
    print("-" * 60)
    print(f"📂 30s images root:   {IMG30_ROOT}")
    print(f"📄 3s feature CSV:    {FEAT3_CSV}")
    print(f"📁 3s images out dir: {OUT_ROOT}")
    print()


def _find_30s_image(base_id: str) -> Path:
    """
    base_id example: 'blues.00000'

    We search under IMG30_ROOT for likely patterns:
      - 'blues.00000.png'
      - 'blues00000.png'
      - '*blues.00000*.png'
      - '*blues00000*.png'
    and return the first match.
    """
    patterns = [
        f"{base_id}.png",
        f"{base_id.replace('.', '')}.png",
        f"*{base_id}*.png",
        f"*{base_id.replace('.', '')}*.png",
    ]
    hits: list[Path] = []
    for pat in patterns:
        for p in IMG30_ROOT.rglob(pat):
            if p not in hits:
                hits.append(p)
        if hits:
            break

    if not hits:
        raise FileNotFoundError(
            f"No 30s spectrogram PNG found for base_id='{base_id}' "
            f"under {IMG30_ROOT}"
        )
    if len(hits) > 1:
        # Not fatal, but warn
        warnings.warn(
            f"Multiple candidates for base_id='{base_id}', using first: {hits[0]}"
        )
    return hits[0]


def main():
    _print_banner()

    # Basic checks
    if not IMG30_ROOT.exists():
        raise FileNotFoundError(f"30s grayscale image directory not found:\n  {IMG30_ROOT}")
    if not FEAT3_CSV.exists():
        raise FileNotFoundError(f"3s feature CSV not found:\n  {FEAT3_CSV}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load features_3_sec.csv
    df = pd.read_csv(FEAT3_CSV)
    if df.empty:
        raise ValueError(f"features_3_sec.csv appears to be empty: {FEAT3_CSV}")

    required_cols = {"filename", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"features_3_sec.csv missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    print(f"✅ Loaded features_3_sec.csv with {len(df)} rows\n")

    # Parse helper: from "blues.00000.3.wav" → base_id="blues.00000", seg_idx=3
    def parse_fname(fname: str):
        parts = fname.split(".")
        # expecting something like [genre, id, segment_index, 'wav']
        if len(parts) < 4:
            raise ValueError(f"Unexpected filename pattern in CSV: {fname}")
        base_id = ".".join(parts[:2])    # "blues.00000"
        try:
            seg_idx = int(parts[-2])     # "3"
        except ValueError:
            raise ValueError(f"Cannot parse segment index from filename: {fname}")
        return base_id, seg_idx

    # Cache loaded 30s images by base_id
    img_cache: dict[str, np.ndarray] = {}
    wh_cache: dict[str, tuple[int, int]] = {}

    # rows for mapping CSV
    map_rows = []

    total_rows = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        fname_3s = getattr(row, "filename")
        label    = getattr(row, "label")

        base_id, seg_idx = parse_fname(fname_3s)

        # Load base 30s spectrogram if not already cached
        if base_id not in img_cache:
            try:
                src_path = _find_30s_image(base_id)
            except FileNotFoundError as e:
                # 🔹 NEW LOGIC: skip rows whose 30s image is missing
                warnings.warn(
                    f"[row {i}/{total_rows}] {e} — skipping this 3s segment."
                )
                continue

            img = Image.open(src_path).convert("L")  # grayscale
            arr = np.array(img)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected grayscale 2D image for {src_path}, got shape {arr.shape}"
                )
            h, w = arr.shape
            img_cache[base_id] = arr
            wh_cache[base_id] = (w, h)
            print(f"📷 Loaded 30s image for {base_id}: {src_path} (h={h}, w={w})")

        arr = img_cache[base_id]
        w, h = wh_cache[base_id]

        # Compute segment width
        seg_w = w // 10
        if seg_w <= 0:
            raise ValueError(f"Computed non-positive seg_w for base_id={base_id}, w={w}")

        if not (0 <= seg_idx <= 9):
            raise ValueError(f"Segment index out of range [0,9] in filename: {fname_3s}")

        x0 = seg_idx * seg_w
        # To avoid losing any trailing pixels due to integer division, let
        # the last segment grab everything to the end.
        x1 = w if seg_idx == 9 else (seg_idx + 1) * seg_w

        # Safety clamp
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))

        crop = arr[:, x0:x1]
        out_img = Image.fromarray(crop.astype(np.uint8))

        # Build destination path: images_grey_scale_3sec/<label>/<name>.png
        genre_dir = OUT_ROOT / str(label)
        genre_dir.mkdir(parents=True, exist_ok=True)

        out_name = fname_3s.replace(".wav", ".png")
        out_path = genre_dir / out_name

        out_img.save(out_path)

        # Collect mapping info
        map_rows.append(
            {
                "orig_30s_id": base_id,
                "segment_index": seg_idx,
                "segment_filename_png": out_name,
                "segment_path": str(out_path.relative_to(REPO_ROOT)),
                "label": label,
                "source_feature_filename": fname_3s,
            }
        )

        if i % 500 == 0 or i == total_rows:
            print(f"  → Processed {i}/{total_rows} rows...")

    # Write mapping CSV
    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(MAP_CSV, index=False)
    print("\n✅ Done.")
    print(f"   3s spectrogram images written under: {OUT_ROOT}")
    print(f"   Mapping CSV: {MAP_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ Error:", e)
        sys.exit(1)
