import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from config.load_hparams import load_hparams
from data.loaders import make_dataset

ROOT = Path(__file__).resolve().parents[1]
CFG  = load_hparams(ROOT / "_code" / "config" / "hparams.yaml")
OUT  = ROOT / CFG["train"]["checkpoints"]["dir"]

MODEL_PATH = OUT / CFG["train"]["checkpoints"]["filename_best"]
IMG_ROOT   = CFG["data"]["img_root"]
CLASSES    = CFG["data"]["classes"]
IMG_SIZE   = tuple(CFG["data"]["img_size"])
CHANNELS   = CFG["data"]["channels"]

# Load test dataset
test_csv = CFG["data"].get("splits", {}).get("test_csv", "_eda_outputs/test_split.csv")
test_ds, label_map, n_test = make_dataset(str(ROOT), test_csv, IMG_ROOT, CLASSES,
                                          batch=32, shuffle=False, img_size=IMG_SIZE, channels=CHANNELS)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Collect predictions and labels
y_true = []
y_pred = []
for x, y in test_ds:
    p = model(x, training=False).numpy()
    y_pred.extend(np.argmax(p, axis=1).tolist())
    y_true.extend(np.argmax(y.numpy(), axis=1).tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Metrics
acc = (y_true == y_pred).mean()
print(f"✅ Test accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (acc={acc:.3f})")
plt.tight_layout()

cm_path = OUT / "confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"✅ Saved: {cm_path}")

# Classification report
rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
rep_path = OUT / "classification_report.txt"
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(rep)
print(f"✅ Saved: {rep_path}")