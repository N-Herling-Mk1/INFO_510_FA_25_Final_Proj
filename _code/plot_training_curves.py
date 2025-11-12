from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config.load_hparams import load_hparams

ROOT = Path(__file__).resolve().parents[1]
CFG = load_hparams(ROOT / "_code" / "config" / "hparams.yaml")
OUTDIR = ROOT / CFG["train"]["checkpoints"]["dir"]

csv_hist_path = OUTDIR / "train_history.csv"
df = pd.read_csv(csv_hist_path)

# Accuracy plot
plt.figure()
plt.plot(df["epoch"] if "epoch" in df.columns else range(len(df)), df["accuracy"], label="train_acc")
plt.plot(df["epoch"] if "epoch" in df.columns else range(len(df)), df["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (train vs val)")
plt.legend(); plt.tight_layout()
acc_png = OUTDIR / "curve_accuracy.png"
plt.savefig(acc_png, dpi=200); plt.close()

# Loss plot
plt.figure()
plt.plot(df["epoch"] if "epoch" in df.columns else range(len(df)), df["loss"], label="train_loss")
plt.plot(df["epoch"] if "epoch" in df.columns else range(len(df)), df["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (train vs val)")
plt.legend(); plt.tight_layout()
loss_png = OUTDIR / "curve_loss.png"
plt.savefig(loss_png, dpi=200); plt.close()

print(f"✅ Saved: {acc_png}")
print(f"✅ Saved: {loss_png}")
