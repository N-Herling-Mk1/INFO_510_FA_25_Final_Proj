# 10_13_25_model_1.py
import os
import math
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
    torch.backends.cudnn.deterministic = False  # allows benchmark to work
    torch.backends.cudnn.benchmark = True       # faster convs on GPU


# -----------------------------
# Dataset for grayscale 128x128 spectrograms
# -----------------------------
class SpectrogramFolder(Dataset):
    def __init__(self, root, class_names=None, transform=None, extensions=(".png", ".jpg", ".jpeg")):
        self.root = Path(root)
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)
        if class_names is None:
            class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples: List[Tuple[str, int]] = []

        for c in self.class_names:
            class_dir = self.root / c
            if not class_dir.exists():
                continue
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in self.extensions:
                    self.samples.append((str(p), self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")  # ensure grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# -----------------------------
# Variational Linear (Bayes by Backprop)
# -----------------------------
class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=2.0, init_logsigma=-5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mean & log-std parameters
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_logsigma = nn.Parameter(torch.full((out_features, in_features), init_logsigma))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_logsigma = nn.Parameter(torch.full((out_features,), init_logsigma))

        # Fixed isotropic Gaussian prior
        self.prior_std = prior_std
        self.prior_var = prior_std ** 2

        # He init on means helps training
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.b_mu, -bound, bound)

    def sample_weights(self):
        w_sigma = torch.exp(self.w_logsigma)
        b_sigma = torch.exp(self.b_logsigma)
        W = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        b = self.b_mu + b_sigma * torch.randn_like(self.b_mu)
        return W, b

    def kl_divergence(self):
        # KL(q||p) with p ~ N(0, prior_std^2), summed over all params
        w_sigma = torch.exp(self.w_logsigma)
        b_sigma = torch.exp(self.b_logsigma)
        w_kl = (torch.log(self.prior_std / w_sigma)
                + (w_sigma**2 + self.w_mu**2) / (2 * self.prior_var) - 0.5).sum()
        b_kl = (torch.log(self.prior_std / b_sigma)
                + (b_sigma**2 + self.b_mu**2) / (2 * self.prior_var) - 0.5).sum()
        return w_kl + b_kl

    def forward(self, x):
        W, b = self.sample_weights()
        return F.linear(x, W, b)


# -----------------------------
# Deterministic CNN Backbone
# Input: (B,1,128,128) -> Output: (B,256)
# -----------------------------
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),# 16x16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (B,256,1,1)
        )

    def forward(self, x):
        z = self.net(x)
        return z.flatten(1)


# -----------------------------
# Full Model: CNN + Variational Head
# -----------------------------
class BayesianCNN(nn.Module):
    def __init__(self, n_classes=10, prior_std=2.0):
        super().__init__()
        self.backbone = CNNBackbone()
        self.var1 = VariationalLinear(256, 128, prior_std=prior_std)
        self.var2 = VariationalLinear(128, n_classes, prior_std=prior_std)

    def forward(self, x):
        h = self.backbone(x)
        h = F.relu(self.var1(h))
        logits = self.var2(h)
        kl = self.var1.kl_divergence() + self.var2.kl_divergence()
        return logits, kl


# -----------------------------
# Training / Evaluation utils
# -----------------------------
def elbo_train_epoch(model, loader, optimizer, device, dataset_size, kl_beta=1.0, amp=False, scaler=None):
    model.train()
    ce_sum, kl_sum = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, kl = model(x)
                nll = F.cross_entropy(logits, y)
                kl_weight = (x.size(0) / dataset_size) * kl_beta
                loss = nll + kl_weight * kl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, kl = model(x)
            nll = F.cross_entropy(logits, y)
            kl_weight = (x.size(0) / dataset_size) * kl_beta
            loss = nll + kl_weight * kl
            loss.backward()
            optimizer.step()

        ce_sum += nll.item() * x.size(0)
        kl_sum += kl.item() * x.size(0)

    return ce_sum / dataset_size, kl_sum / dataset_size


@torch.no_grad()
def evaluate(model, loader, device, mc_passes=1):
    model.eval()
    correct, total = 0, 0
    ce_sum = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        probs_accum = 0
        ce_accum = 0
        for _ in range(mc_passes):
            logits, _ = model(x)
            ce_accum += F.cross_entropy(logits, y, reduction="sum").item()
            probs_accum += F.softmax(logits, dim=-1)

        probs = probs_accum / mc_passes
        pred = probs.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ce_sum += ce_accum / mc_passes

    return ce_sum / total, (correct / total if total > 0 else 0.0)


@torch.no_grad()
def mc_predict(model, x, T=20):
    model.eval()
    probs = []
    for _ in range(T):
        logits, _ = model(x)
        probs.append(F.softmax(logits, dim=-1))
    P = torch.stack(probs, dim=0)       # (T,B,C)
    mean_p = P.mean(dim=0)              # (B,C)
    var_p  = P.var(dim=0)               # (B,C)
    entropy = -(mean_p * (mean_p.clamp_min(1e-12)).log()).sum(dim=-1)  # (B,)
    return mean_p, var_p, entropy


# -----------------------------
# Utility: balanced split by class (track-level)
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
        # ensure all go somewhere
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - 1 - n_train)
        # remaining to test
        train_idx += c_idx[:n_train].tolist()
        val_idx   += c_idx[n_train:n_train + n_val].tolist()
        test_idx  += c_idx[n_train + n_val:].tolist()

    return train_idx, val_idx, test_idx


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian CNN for GTZAN spectrograms")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir with genre subfolders of grayscale spectrogram images.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0, help="Windows-safe default 0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prior_std", type=float, default=2.0, help="Gaussian prior std for variational layers.")
    parser.add_argument("--split_json", type=str, default="", help="Path to save/load split indices (JSON).")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on GPU.")
    parser.add_argument("--val_mc", type=int, default=5, help="MC passes for validation.")
    parser.add_argument("--test_mc", type=int, default=10, help="MC passes for test.")
    parser.add_argument("--seed", type=int, default=42)
    # ---- NEW: KL schedule flags ----
    parser.add_argument("--kl_beta_max", type=float, default=0.5, help="Max KL weight (<=1).")
    parser.add_argument("--kl_warmup_epochs", type=int, default=10, help="Epochs to reach beta_max.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ---- Transforms: ensure 1x128x128 & scale to [0,1] ----
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # grayscale -> (1,H,W) in [0,1]
    ])

    # ---- Dataset & labels ----
    full_ds = SpectrogramFolder(args.data_root, transform=transform)
    labels = np.array([y for _, y in full_ds])
    n_classes = len(full_ds.class_names)
    print(f"Classes ({n_classes}): {full_ds.class_names}")
    print(f"Total images: {len(full_ds)}")

    # ---- Split (load or create) ----
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

    # ---- Loaders (Windows-safe defaults) ----
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_mem)

    # ---- Model, Optim ----
    model = BayesianCNN(n_classes=n_classes, prior_std=args.prior_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ---- Train with configurable KL schedule ----
    epochs = args.epochs
    dataset_size = len(train_ds)

    best_val_acc, best_state = 0.0, None
    beta_max = max(0.0, min(1.0, args.kl_beta_max))
    warm = max(1, int(args.kl_warmup_epochs))

    for epoch in range(1, epochs + 1):
        # linear warmup to beta_max
        kl_beta = min(beta_max, (epoch / warm) * beta_max)

        ce_tr, kl_tr = elbo_train_epoch(
            model, train_loader, optimizer, device, dataset_size,
            kl_beta=kl_beta, amp=args.amp and device.type == "cuda", scaler=scaler
        )
        ce_val, acc_val = evaluate(model, val_loader, device, mc_passes=args.val_mc)

        print(f"Epoch {epoch:02d} | train CE: {ce_tr:.4f} KL:{kl_tr:.2f} | "
              f"val CE: {ce_val:.4f} acc:{acc_val:.3f} | beta:{kl_beta:.2f}")

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final test (MC averaging for robustness) ----
    ce_test, acc_test = evaluate(model, test_loader, device, mc_passes=args.test_mc)
    print(f"\nTest CE: {ce_test:.4f} | Test Acc: {acc_test:.3f}")


if __name__ == "__main__":
    main()
