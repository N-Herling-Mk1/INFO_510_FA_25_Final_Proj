"""
# - model_2_v2

"""
# 10_13_25_model_2.py
import os, math, json, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------
# Repro / small utilities
# -----------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def norm_key(x: str) -> str:
    """
    Normalize a filename to a robust join key:
    - lowercased
    - strip on both ends
    - remove common image suffixes left by exports
    - collapse spaces/underscores
    """
    s = str(x).strip().lower()
    s = s.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".bmp","")
    s = s.replace("_", " ").replace("-", " ").replace("  ", " ")
    return s.strip()

# -----------------------------
# Dataset: spectrogram image + tabular features (from CSV)
# -----------------------------
class SpectroTabDataset(Dataset):
    def __init__(
        self,
        img_root: str,
        csv_path: str,
        id_col: Optional[str] = None,
        label_col: Optional[str] = None,
        transform=None,
        feature_allowlist: Optional[List[str]] = None,
        debug: bool = False,
    ):
        self.img_root = Path(img_root)
        self.transform = transform
        self.debug = debug
        self.extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        # class folders (assume subfolders under img_root are class names)
        classes = []
        for p in sorted(self.img_root.iterdir()):
            if p.is_dir(): classes.append(p.name)

        self.class_names = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # map of normalized stem -> (path, class_idx, raw_stem)
        self.img_by_norm: Dict[str, Tuple[str, int, str]] = {}

        for c in self.class_names:
            cdir = self.img_root / c
            if not cdir.exists():
                continue
            for p in cdir.iterdir():
                if p.is_file() and p.suffix.lower() in self.extensions:
                    raw_stem = p.stem
                    key = norm_key(raw_stem)
                    self.img_by_norm[key] = (str(p), self.class_to_idx[c], raw_stem)

        # Load features CSV
        df = pd.read_csv(csv_path)

        # Resolve id_col if not provided/found
        if id_col not in df.columns:
            for cand in ["filename", "file", "track", "track_id", "slice_file_name", "stem", "id", "name", "path"]:
                if cand in df.columns:
                    id_col = cand
                    break
        if id_col not in df.columns:
            raise ValueError(
                f"Could not find an ID column. CSV columns: {list(df.columns)}. "
                f"Provide --id_col to match images (e.g., filename)."
            )

        # Normalize CSV IDs
        df["__norm__"] = df[id_col].astype(str).map(norm_key)

        # Resolve label column
        if label_col not in df.columns:
            for cand in ["label", "genre", "class", "target"]:
                if cand in df.columns:
                    label_col = cand
                    break
        if label_col not in df.columns:
            raise ValueError(
                f"Could not find a label column. CSV columns: {list(df.columns)}. "
                f"Provide --label_col (e.g., genre/class)."
            )

        # Feature columns (all numeric besides the ID/label)
        exlude_cols = {id_col, label_col, "__norm__"}
        num_cols = [c for c in df.columns if c not in exlude_cols and pd.api.types.is_numeric_dtype(df[c])]
        if feature_allowlist:
            feature_allowlist = [c for c in feature_allowlist if c in num_cols]
            self.feature_cols = feature_allowlist
        else:
            self.feature_cols = num_cols

        # join CSV rows to images by normalized key
        rows = []
        miss = 0
        for _, r in df.iterrows():
            k = r["__norm__"]
            if k not in self.img_by_norm:
                miss += 1
                continue
            img_path, class_idx, raw_stem = self.img_by_norm[k]
            feats = r[self.feature_cols].astype(float).to_numpy(dtype=np.float32) if len(self.feature_cols) else np.zeros((0,), dtype=np.float32)
            rows.append((img_path, feats, class_idx, raw_stem))

        if self.debug:
            print(f"[Dataset] unmatched rows in CSV: {miss}")

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        img_path, feats, class_idx, raw_stem = self.rows[idx]
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            x_img = self.transform(im)
        else:
            x_img = transforms.ToTensor()(im)
        x_tab = torch.from_numpy(feats)
        y = class_idx
        return x_img, x_tab, y, raw_stem

# -----------------------------
# Standardizer for tabular features
# -----------------------------
class Standardizer:
    def __init__(self, mean: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None, eps: float=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, X: torch.Tensor):
        Xn = X.detach().cpu().numpy()
        self.mean = Xn.mean(axis=0)
        self.std = Xn.std(axis=0) + self.eps

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        m = torch.from_numpy(self.mean).to(x.device, dtype=x.dtype)
        s = torch.from_numpy(self.std).to(x.device, dtype=x.dtype)
        return (x - m) / s

    def save(self, path: str, feature_cols: List[str]):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(), "features": feature_cols}, f, indent=2)

    @staticmethod
    def load(path: str) -> "Standardizer":
        with open(path, "r") as f:
            obj = json.load(f)
        return Standardizer(np.array(obj["mean"]), np.array(obj["std"]))


# -----------------------------
# Variational Linear (last-layer Bayesian)
# -----------------------------
class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=2.5, init_logsigma=-6.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_logsigma = nn.Parameter(torch.full((out_features, in_features), init_logsigma))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_logsigma = nn.Parameter(torch.full((out_features,), init_logsigma))
        self.prior_std = prior_std
        self.prior_var = prior_std ** 2

        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.b_mu, -bound, bound)

    def forward(self, x):
        w_sigma = torch.exp(self.w_logsigma)
        b_sigma = torch.exp(self.b_logsigma)
        W = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        B = self.b_mu + b_sigma * torch.randn_like(self.b_mu)
        return x @ W.t() + B

    def kl_div(self):
        w_sigma2 = torch.exp(2*self.w_logsigma)
        b_sigma2 = torch.exp(2*self.b_logsigma)
        w_mu2 = self.w_mu.pow(2)
        b_mu2 = self.b_mu.pow(2)
        # KL diag Gaussians vs N(0, prior_std^2)
        kl_w = 0.5 * torch.sum((w_mu2 + w_sigma2) / self.prior_var - 1 - 2*self.w_logsigma + 2*math.log(self.prior_std))
        kl_b = 0.5 * torch.sum((b_mu2 + b_sigma2) / self.prior_var - 1 - 2*self.b_logsigma + 2*math.log(self.prior_std))
        return kl_w + kl_b


# -----------------------------
# Conv backbone + fusion with tabular, then VariationalLinear
# -----------------------------
class FusionBayesNet(nn.Module):
    def __init__(self, n_classes: int, tab_dim: int, prior_std=2.5):
        super().__init__()
        # simple conv stem
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.img_proj = nn.Linear(64, 64)
        self.tab_proj = nn.Linear(max(1, tab_dim), 32) if tab_dim > 0 else None
        feat_dim = 64 + (32 if tab_dim > 0 else 0)

        # one deterministic hidden before Bayesian head
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, 128),
            nn.ReLU()
        )

        # Bayesian last layer
        self.bayes_head = VariationalLinear(128, n_classes, prior_std=prior_std)

    def forward(self, x_img, x_tab=None):
        z = self.conv(x_img).flatten(1)
        z = self.img_proj(z)
        if self.tab_proj is not None and x_tab is not None and x_tab.numel() > 0:
            zt = self.tab_proj(x_tab)
            z = torch.cat([z, zt], dim=1)
        z = self.hidden(z)
        logits = self.bayes_head(z)
        return logits, self.bayes_head.kl_div()

# -----------------------------
# Helpers
# -----------------------------
def discover_class_counts(ds: Dataset) -> Dict[int,int]:
    counts = {}
    for _, _, y, _ in ds:
        counts[int(y)] = counts.get(int(y), 0) + 1
    return counts

# -----------------------------
# ELBO train epoch
# -----------------------------
def elbo_train_epoch(
    model, loader, optimizer, device, dataset_size,
    kl_beta=1.0, scaler=None, label_smooth=0.05, stdz: Optional[Standardizer]=None
):
    model.train()
    ce_sum, kl_sum = 0.0, 0.0

    for x_img, x_tab, y, _stem in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        if stdz is not None:
            x_tab = stdz.transform(x_tab)

        optimizer.zero_grad(set_to_none=True)

        logits, kl = model(x_img, x_tab)
        nll = F.cross_entropy(logits, y, label_smoothing=label_smooth, reduction="mean")
        kl_weight = (x_img.size(0) / dataset_size) * kl_beta
        loss = nll + kl_weight * kl
        loss.backward()
        optimizer.step()

        ce_sum += nll.item() * x_img.size(0)
        kl_sum += kl.item() * x_img.size(0)

    return ce_sum / dataset_size, kl_sum / dataset_size


@torch.no_grad()
def evaluate(model, loader, device, mc_passes=5, label_smooth=0.05, stdz: Optional[Standardizer]=None):
    model.eval()
    correct, total = 0, 0
    ce_sum = 0.0

    for x_img, x_tab, y, _stem in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        if stdz is not None:
            x_tab = stdz.transform(x_tab)

        probs_accum = 0
        ce_accum = 0
        for _ in range(mc_passes):
            logits, _ = model(x_img, x_tab)
            ce_accum += F.cross_entropy(logits, y, label_smoothing=label_smooth, reduction="sum").item()
            probs_accum += F.softmax(logits, dim=-1)

        probs = probs_accum / mc_passes
        pred = probs.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ce_sum += ce_accum / mc_passes

    return ce_sum / total, (correct / total if total > 0 else 0.0)


@torch.no_grad()
def quick_train_accuracy(model, loader, device, stdz: Optional[Standardizer]=None):
    model.eval()
    correct, total = 0, 0
    for x_img, x_tab, y, _stem in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        if stdz is not None:
            x_tab = stdz.transform(x_tab)
        logits, _ = model(x_img, x_tab)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return (correct / total) if total > 0 else 0.0


# -----------------------------
# Stratified split
# -----------------------------
def stratified_split(labels: np.ndarray, train=0.8, val=0.1, test=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(labels))
    groups = {}
    for i, y in enumerate(labels):
        groups.setdefault(int(y), []).append(i)

    tr, va, te = [], [], []
    for _, arr in groups.items():
        arr = np.array(arr)
        rng.shuffle(arr)
        n = len(arr)
        n_tr = int(round(train * n))
        n_val = int(round(val * n))
        tr.extend(arr[:n_tr].tolist())
        va.extend(arr[n_tr:n_tr+n_val].tolist())
        te.extend(arr[n_tr+n_val:].tolist())
    return np.array(tr), np.array(va), np.array(te)

# -----------------------------
# Arg parsing
# -----------------------------
def build_args():
    import argparse
    ap = argparse.ArgumentParser("Bayesian CNN + Tabular Fusion (Variational Last Layer)")
    ap.add_argument("--data_root", type=str, required=True, help="root containing class folders with spectrogram images")
    ap.add_argument("--csv_path", type=str, required=True, help="CSV of features and labels")
    ap.add_argument("--id_col", type=str, default="", help="CSV column that matches image stems")
    ap.add_argument("--label_col", type=str, default="", help="CSV column for labels")
    ap.add_argument("--feature_cols", type=str, default="", help="comma list of feature columns to include (optional)")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--val_mc", type=int, default=5)
    ap.add_argument("--test_mc", type=int, default=25)
    ap.add_argument("--prior_std", type=float, default=2.5)

    ap.add_argument("--kl_beta_max", type=float, default=1.0)
    ap.add_argument("--kl_warmup_epochs", type=int, default=10)

    ap.add_argument("--scaler_json", type=str, default="tab_scaler.json")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = build_args()
    seed_all(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Optional allowlist from CLI
    allowlist = None
    if args.feature_cols.strip():
        allowlist = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    # Build dataset (loads CSV, matches to images)
    full_ds = SpectroTabDataset(
        img_root=args.data_root,
        csv_path=args.csv_path,
        id_col=args.id_col,
        label_col=args.label_col,
        transform=transform,
        feature_allowlist=allowlist,
        debug=args.debug,
    )
    n_classes = len(full_ds.class_names)
    tab_dim = len(full_ds.feature_cols)
    print(f"Classes ({n_classes}): {full_ds.class_names}")
    print(f"Matched samples: {len(full_ds)} | Tabular dim: {tab_dim}")

    # Build splits
    labels = np.array([y for _, _, y, _ in full_ds])
    tr_idx, va_idx, te_idx = stratified_split(labels, train=0.8, val=0.1, test=0.1, seed=123)
    train_ds = torch.utils.data.Subset(full_ds, tr_idx)
    val_ds   = torch.utils.data.Subset(full_ds, va_idx)
    test_ds  = torch.utils.data.Subset(full_ds, te_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    # Fit standardizer on TRAIN features only
    stdz = Standardizer()
    with torch.no_grad():
        all_train_tab = []
        for _, x_tab, _, _ in train_loader:
            all_train_tab.append(x_tab)
        all_train_tab = torch.cat(all_train_tab, dim=0)
        stdz.fit(all_train_tab)
        stdz.save(args.scaler_json, feature_cols=full_ds.feature_cols)
    print(f"Saved tabular scaler to {args.scaler_json}")

    # Model, optimizer, scaler
    model = FusionBayesNet(n_classes=n_classes, tab_dim=tab_dim, prior_std=args.prior_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = None

    # KL warmup + early stopping on val CE
    epochs = args.epochs
    dataset_size = len(train_ds)
    beta_max = max(0.0, min(1.0, args.kl_beta_max))
    warm = max(1, int(args.kl_warmup_epochs))

    # ---- training history for plots ----
    history = {
        "tr_loss": [],
        "val_loss": [],
        "tr_acc":  [],
        "val_acc": []
    }

    patience, bad = 8, 0
    best_val_ce = float("inf")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        kl_beta = min(beta_max, (epoch / warm) * beta_max)

        # ---- train epoch ----
        ce_tr, kl_tr = elbo_train_epoch(
            model, train_loader, optimizer, device, dataset_size,
            kl_beta=kl_beta, scaler=scaler, label_smooth=args.label_smoothing, stdz=stdz
        )

        # ---- validation (MC passes) ----
        ce_val, acc_val = evaluate(
            model, val_loader, device, mc_passes=args.val_mc,
            label_smooth=args.label_smoothing, stdz=stdz
        )

        # ---- quick train accuracy (no MC) ----
        acc_tr = quick_train_accuracy(model, train_loader, device, stdz=stdz)

        # ---- log ----
        history["tr_loss"].append(float(ce_tr))
        history["val_loss"].append(float(ce_val))
        history["tr_acc"].append(float(acc_tr))
        history["val_acc"].append(float(acc_val))

        print(f"Epoch {epoch:02d} | CE_tr {ce_tr:.4f} | CE_val {ce_val:.4f} | "
              f"Acc_tr {acc_tr:.3f} | Acc_val {acc_val:.3f} | KL {kl_tr:.2f} | beta {kl_beta:.3f}")

        # ---- early stopping on val CE ----
        improved = ce_val + 1e-6 < best_val_ce
        if improved:
            best_val_ce = ce_val
            best_val_acc = acc_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no val CE improvement in {patience} epochs).")
                break


    # ---- restore best ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best checkpoint (val CE: {best_val_ce:.4f}, val acc: {best_val_acc:.3f})")

    # ---- plot curves ----
    try:
        epochs_r = range(1, len(history["tr_loss"]) + 1)
        plt.figure(figsize=(10, 5.2), dpi=120)

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_r, history["tr_loss"], label="Train Loss")
        plt.plot(epochs_r, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title("Loss")
        plt.legend()
        plt.grid(alpha=0.25)

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_r, history["tr_acc"], label="Train Acc")
        plt.plot(epochs_r, history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig("training_curves.png", bbox_inches="tight")
        print("Saved training curves to training_curves.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # Final test
    ce_test, acc_test = evaluate(
        model, test_loader, device, mc_passes=args.test_mc,
        label_smooth=args.label_smoothing, stdz=stdz
    )
    print(f"\nTest CE: {ce_test:.4f} | Test Acc: {acc_test:.3f}")


if __name__ == "__main__":
    main()
