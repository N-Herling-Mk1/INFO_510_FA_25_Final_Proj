# _code/data/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import re
import pandas as pd
import tensorflow as tf

def _load_image(path: tf.Tensor, img_size: Tuple[int,int], channels: int):
    img_bytes = tf.io.read_file(path)
    # Decode PNG/JPEG transparently
    img = tf.io.decode_image(img_bytes, channels=channels, expand_animations=False)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, tf.float32)   # [0,1]
    return img

def _encode_example(img_path: tf.Tensor, label_idx: tf.Tensor, img_size, channels, num_classes: int):
    img = _load_image(img_path, img_size, channels)
    y = tf.one_hot(label_idx, num_classes)
    return img, y

_NUM_RE = re.compile(r"(\d{1,6})$")  # capture trailing digits like 00081

def _resolve_image_path(
    base_dir: Path, genre: str, base_id: str, exts=(".png", ".jpg", ".jpeg")
) -> Optional[Path]:
    """
    Try common GTZAN spectrogram filename variants:
      genre.num.ext      e.g., blues.00081.png
      genre-num.ext      e.g., blues-00081.png
      genre{num}.ext     e.g., blues00081.png  <-- your case
    Fallbacks: if base_id already includes dots/hyphens, try that literal too.
    """
    # numeric part from base_id (e.g., 'blues.00081' -> '00081')
    m = _NUM_RE.search(base_id)
    num = m.group(1) if m else ""

    candidates = []
    if num:
        candidates += [
            f"{genre}.{num}",
            f"{genre}-{num}",
            f"{genre}{num}",
        ]
    # also try the literal base_id as-is (sometimes the csv stem matches the file)
    candidates.append(base_id)

    for stem in candidates:
        for ext in exts:
            p = base_dir / genre / (stem + ext)
            if p.exists():
                return p
    return None

def make_dataset(
    repo_root: str,
    split_csv: str,
    img_root: str,
    classes: List[str],
    batch: int = 32,
    shuffle: bool = True,
    img_size: Tuple[int,int] = (224,224),
    channels: int = 1
):
    """
    Map split CSV (columns: base_id, genre) to spectrogram images under:
      <repo_root>/<img_root>/<genre>/<filename>
    Returns: (tf.data.Dataset, label_map, n_samples)
    """
    root = Path(repo_root)
    df = pd.read_csv(root / split_csv)
    needed = {"base_id", "genre"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Split CSV must have columns {sorted(needed)}: {split_csv}")

    label_map = {g:i for i,g in enumerate(classes)}

    paths, labels = [], []
    missing, bad_label = 0, 0
    kept_per_genre = {g: 0 for g in classes}
    base_dir = root / img_root

    for _, r in df.iterrows():
        genre = str(r["genre"]).lower()
        base_id = str(r["base_id"])
        if genre not in label_map:
            bad_label += 1
            continue
        resolved = _resolve_image_path(base_dir, genre, base_id)
        if resolved is None:
            missing += 1
            continue
        paths.append(str(resolved))
        labels.append(label_map[genre])
        kept_per_genre[genre] += 1

    n = len(paths)
    # Friendly report
    kept_str = ", ".join([f"{g}: {kept_per_genre[g]}/{sum(df['genre'].str.lower()==g)}" for g in classes])
    print(f"\n📁 Image mapping: kept {n}/{len(df)} rows (dropped_missing={missing}, dropped_bad_label={bad_label})")
    print(f"   per-genre kept: {kept_str}")

    if n == 0:
        raise RuntimeError(
            "No samples resolved to existing image files. "
            "Verify filename pattern (e.g., blues00000.png vs blues.00000.png) and extension."
        )

    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((ds_paths, ds_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=max(1024, n))

    autotune = tf.data.AUTOTUNE
    num_classes = len(classes)
    ds = ds.map(lambda p, y: _encode_example(p, y, img_size, channels, num_classes), num_parallel_calls=autotune)
    ds = ds.batch(batch).prefetch(autotune)
    return ds, label_map, n
