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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Helpers
# -----------------------------
def discover_classes(root: Path) -> List[str]:
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def norm_key(s: str) -> str:
    """
    Normalize a CSV or image filename to a robust join key:
    - lowercased
    - strip one extension
    - remove all non-alphanumerics
    E.g., "genres_original/blues/blues.00012.wav" -> "blues00012"
          "blues_00012.png"                     -> "blues00012"
    """
    stem = Path(str(s)).stem.lower()
    return "".join(ch for ch in stem if ch.isalnum())


# -----------------------------
# Dataset: Spectrogram + Tabular
# -----------------------------
class SpectroTabDataset(Dataset):
    def __init__(
        self,
        img_root: str,
        csv_path: str,
        id_col: Optional[str] = "filename",
        label_col: str = "label",
        drop_cols: Optional[List[str]] = None,
        transform=None,
        extensions=(".png", ".jpg", ".jpeg"),
        feature_allowlist: Optional[List[str]] = None,
        debug: bool = False,
    ):
        self.img_root = Path(img_root)
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)
        self.debug = debug

        # Classes from folder names (must match your 10 GTZAN genres)
        self.class_names = discover_classes(self.img_root)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Build image index by normalized stem key
        self.img_by_norm: Dict[str, Tuple[str, int, str]] = {}  # key -> (path, class_idx, raw_stem)
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
            # not fatal; we can fallback to folder label
            label_col = None

        # Feature columns (numeric, excluding id/label)
        if drop_cols is None:
            drop_cols = []
        reserved = set([id_col, "__norm__"] + ([label_col] if label_col else []))
        if feature_allowlist is not None:
            feature_cols = [c for c in feature_allowlist if c in df.columns]
        else:
            feature_cols = [
                c for c in df.columns
                if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not feature_cols:
            raise ValueError("No numeric feature columns found (after excluding ID/label).")

        # Match CSV rows to images
        samples = []
        not_found_examples = []
        label_mismatch_examples = []

        # precompute folder_map for lenient label matching
        folder_map = {"".join(ch for ch in k if ch.isalnum()): k for k in self.class_names}

        for _, r in df.iterrows():
            key = r["__norm__"]
            tup = self.img_by_norm.get(key)
            if tup is None:
                if len(not_found_examples) < 8:
                    not_found_examples.append(r[id_col])
                continue

            img_path, folder_cls_idx, raw_stem = tup

            # Choose class: prefer CSV label if it maps; else fallback to folder label
            if label_col:
                lbl = str(r[label_col]).strip().lower()
                if lbl in self.class_to_idx:
                    cls_idx = self.class_to_idx[lbl]
                else:
                    lbl_norm = "".join(ch for ch in lbl if ch.isalnum())
                    mapped = folder_map.get(lbl_norm, None)
                    if mapped is not None:
                        cls_idx = self.class_to_idx[mapped]
                    else:
                        cls_idx = folder_cls_idx
            else:
                cls_idx = folder_cls_idx

            if label_col:
                # report mismatch examples (not an error)
                lbl_folder = self.class_names[folder_cls_idx]
                lbl_csv = str(r[label_col]).strip()
                if self.class_names[cls_idx] != lbl_folder and len(label_mismatch_examples) < 8:
                    label_mismatch_examples.append((raw_stem, lbl_csv, lbl_folder))

            feat = r[feature_cols].to_numpy(dtype=np.float32, copy=True)
            samples.append((img_path, cls_idx, raw_stem, feat))

        if not samples:
            raise RuntimeError(
                "No CSV rows matched images. Likely ID pattern mismatch.\n"
                f"Tried id_col='{id_col}'. Example CSV values: {df[id_col].head().tolist()}\n"
                f"Make sure the normalized stems align with image stems."
            )

        self.samples = samples
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.label_col = label_col

        if self.debug:
            print(f"[DEBUG] Matched {len(self.samples)} rows.")
            print(f"[DEBUG] First 8 feature cols: {self.feature_cols[:8]}{' ...' if len(self.feature_cols)>8 else ''}")
            if not_found_examples:
                print(f"[DEBUG] Example CSV IDs not found (up to 8): {not_found_examples[:8]}")
            if label_mismatch_examples:
                print(f"[DEBUG] Example label mismatches (stem, csv_label, folder_label) (up to 8):")
                for e in label_mismatch_examples[:8]:
                    print("   ", e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx, stem, feat = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        feat = torch.from_numpy(feat)
        return img, feat, class_idx, stem


# -----------------------------
# Standardizer (fit on train only)
# -----------------------------
class Standardizer:
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, X: torch.Tensor):
        m = X.mean(dim=0)
        s = X.std(dim=0, unbiased=False)
        self.mean = m.cpu().numpy()
        self.std = (s.cpu().numpy() + self.eps)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        mean = torch.from_numpy(self.mean).to(X.device)
        std = torch.from_numpy(self.std).to(X.device)
        return (X - mean) / std

    def save(self, path: str, feature_cols: List[str]):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(), "cols": feature_cols}, f, indent=2)

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
        return F.linear(x, W, B)

    def kl_divergence(self):
        w_sigma = torch.exp(self.w_logsigma)
        b_sigma = torch.exp(self.b_logsigma)
        w_kl = (torch.log(self.prior_std / w_sigma)
                + (w_sigma**2 + self.w_mu**2) / (2 * self.prior_var) - 0.5).sum()
        b_kl = (torch.log(self.prior_std / b_sigma)
                + (b_sigma**2 + self.b_mu**2) / (2 * self.prior_var) - 0.5).sum()
        return w_kl + b_kl


# -----------------------------
# CNN backbone (images -> 256)
# -----------------------------
class CNNBackbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        z = self.net(x)         # (B,256,1,1)
        return z.flatten(1)     # (B,256)


# -----------------------------
# Tabular MLP (features -> 128)
# -----------------------------
class TabMLP(nn.Module):
    def __init__(self, in_dim: int, hid=128, out=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hid, out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Fusion model: (img->256) ⊕ (tab->128) -> 128 -> Bayesian head
# -----------------------------
class FusionBayesNet(nn.Module):
    def __init__(self, n_classes: int, tab_dim: int, prior_std=2.5):
        super().__init__()
        self.backbone = CNNBackbone(in_ch=1)
        self.tab = TabMLP(in_dim=tab_dim, hid=128, out=128, p=0.1)
        fused = 256 + 128
        self.fc1 = nn.Linear(fused, 128)
        self.var_out = VariationalLinear(128, n_classes, prior_std=prior_std)

    def forward(self, x_img, x_tab):
        h_img = self.backbone(x_img)           # (B,256)
        h_tab = self.tab(x_tab)                # (B,128)
        h = torch.cat([h_img, h_tab], dim=1)   # (B,384)
        h = F.relu(self.fc1(h))                # (B,128)
        logits = self.var_out(h)               # (B,C)
        kl = self.var_out.kl_divergence()
        return logits, kl


# -----------------------------
# Overfit signal helper
# -----------------------------
def overfit_signal(train_loss, val_loss, train_acc, val_acc,
                   loss_margin=0.10, acc_margin=0.05) -> str:
    """
    Simple heuristic:
      - If val_loss > train_loss + loss_margin AND (train_acc - val_acc) > acc_margin -> flag.
      - Otherwise, likely no strong signs of overfitting.
    """
    loss_gap = val_loss - train_loss
    acc_gap = train_acc - val_acc
    if (loss_gap > loss_margin) and (acc_gap > acc_margin):
        return "⚠︎ potential overtraining"
    return "✓ no overtraining indicated"


# -----------------------------
# Train / Eval
# -----------------------------
def elbo_train_epoch(model, loader, optimizer, device, dataset_size, kl_beta=1.0,
                     scaler=None, label_smooth=0.05, stdz: Optional[Standardizer]=None):
    model.train()
    ce_sum, kl_sum = 0.0, 0.0
    correct, total = 0, 0

    for x_img, x_tab, y, _stem in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        if stdz is not None:
            x_tab = stdz.transform(x_tab)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, kl = model(x_img, x_tab)
                nll = F.cross_entropy(logits, y, label_smoothing=label_smooth)
                kl_weight = (x_img.size(0) / dataset_size) * kl_beta
                loss = nll + kl_weight * kl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, kl = model(x_img, x_tab)
            nll = F.cross_entropy(logits, y, label_smoothing=label_smooth)
            kl_weight = (x_img.size(0) / dataset_size) * kl_beta
            loss = nll + kl_weight * kl
            loss.backward()
            optimizer.step()

        ce_sum += nll.item() * x_img.size(0)
        kl_sum += kl.item() * x_img.size(0)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    train_ce = ce_sum / dataset_size
    train_acc = (correct / total) if total > 0 else 0.0
    train_kl = kl_sum / dataset_size
    return train_ce, train_kl, train_acc


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


# -----------------------------
# Stratified split
# -----------------------------
def stratified_split(labels: np.ndarray, train=0.8, val=0.1, test=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(labels))
    train_idx, val_idx, test_idx = [], [], []
    for c in sorted(set(labels.tolist())):
        c_idx = idxs[labels == c]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(round(train * n))
        n_val = int(round(val * n))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - 1 - n_train)
        train_idx += c_idx[:n_train].tolist()
        val_idx   += c_idx[n_train:n_train + n_val].tolist()
        test_idx  += c_idx[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fusion CNN+Tabular with Bayesian output for GTZAN")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir with genre subfolders (grayscale spectrograms).")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with one row per 30s track and features.")
    parser.add_argument("--id_col", type=str, default="filename", help="Column in CSV to match image stems.")
    parser.add_argument("--label_col", type=str, default="label", help="Column with class labels.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows-safe
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prior_std", type=float, default=2.5)
    parser.add_argument("--val_mc", type=int, default=10)
    parser.add_argument("--test_mc", type=int, default=20)
    parser.add_argument("--kl_beta_max", type=float, default=0.30)
    parser.add_argument("--kl_warmup_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_json", type=str, default="")
    parser.add_argument("--scaler_json", type=str, default="tab_scaler.json")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print matcher diagnostics.")
    # Optional: comma-separated feature allowlist (if you want to restrict)
    parser.add_argument("--feature_cols", type=str, default="", help="Comma-separated feature columns to use (optional).")
    args = parser.parse_args()

    set_seed(args.seed)

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
    print(f"Using feature columns ({len(full_ds.feature_cols)}): {full_ds.feature_cols[:8]}{' ...' if len(full_ds.feature_cols)>8 else ''}")

    # Build split
    labels = np.array([y for _, _, y, _ in full_ds])
    if args.split_json and os.path.exists(args.split_json):
        with open(args.split_json, "r") as f:
            split = json.load(f)
        train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
        print(f"Loaded split from {args.split_json}")
    else:
        train_idx, val_idx, test_idx = stratified_split(labels, train=0.8, val=0.1, test=0.1, seed=args.seed)
        if args.split_json:
            with open(args.split_json, "w") as f:
                json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f)
            print(f"Saved split to {args.split_json}")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    # DataLoaders
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == "cuda") else None

    # Training loop with KL warmup + early stopping on val CE
    epochs = args.epochs
    dataset_size = len(train_ds)
    beta_max = max(0.0, min(1.0, args.kl_beta_max))
    warm = max(1, int(args.kl_warmup_epochs))

    patience, bad = 8, 0
    best_val_ce = float("inf")
    best_val_acc = 0.0
    best_state = None

    # track train metrics for the same epoch we keep as 'best'
    best_train_ce = None
    best_train_acc = None
    best_epoch = None

    # Column header once
    print("\nEpoch |  train_loss  val_loss  |  train_acc  val_acc  |  KLβ  | note")
    print("------+------------------------+----------------------+-------+-------------------------")

    for epoch in range(1, epochs + 1):
        kl_beta = min(beta_max, (epoch / warm) * beta_max)

        ce_tr, kl_tr, acc_tr = elbo_train_epoch(
            model, train_loader, optimizer, device, dataset_size,
            kl_beta=kl_beta, scaler=scaler, label_smooth=args.label_smoothing, stdz=stdz
        )
        ce_val, acc_val = evaluate(
            model, val_loader, device, mc_passes=args.val_mc,
            label_smooth=args.label_smoothing, stdz=stdz
        )

        note = overfit_signal(ce_tr, ce_val, acc_tr, acc_val)
        print(f"{epoch:5d} |  {ce_tr:10.4f}  {ce_val:8.4f}  |   {acc_tr:8.3f}  {acc_val:7.3f}  |  {kl_beta:4.2f} | {note}")

        improved = ce_val + 1e-6 < best_val_ce
        if improved:
            best_val_ce = ce_val
            best_val_acc = acc_val
            best_train_ce = ce_tr
            best_train_acc = acc_tr
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no val CE improvement in {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best checkpoint from epoch {best_epoch} "
              f"(val_loss: {best_val_ce:.4f}, val_acc: {best_val_acc:.3f})")

    # Post-train summary re: overtraining at best epoch
    if best_train_ce is not None:
        summary_note = overfit_signal(best_train_ce, best_val_ce, best_train_acc, best_val_acc)
        print("\n=== Training Summary (best epoch) ===")
        print(f"epoch: {best_epoch}")
        print(f"training loss:   {best_train_ce:.4f}")
        print(f"validation loss: {best_val_ce:.4f}")
        print(f"training acc:    {best_train_acc:.3f}")
        print(f"validation acc:  {best_val_acc:.3f}")
        print(f"inference: {summary_note}")

    # Final test
    ce_test, acc_test = evaluate(
        model, test_loader, device, mc_passes=args.test_mc,
        label_smooth=args.label_smoothing, stdz=stdz
    )
    print(f"\nTest CE: {ce_test:.4f} | Test Acc: {acc_test:.3f}")


if __name__ == "__main__":
    main()
