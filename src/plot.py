"""
Brain2Hand — Chart Generator
Run AFTER train.py and eval.py to produce all publication-quality figures.

Output files (all saved to <repo_root>/results/):
  01_training_curves.png      — loss + val accuracy over epochs
  02_confusion_matrix.png     — normalised confusion matrix
  03_class_metrics.png        — precision / recall / F1 bar chart
  04_roc_curve.png            — ROC curve with AUC score
  05_eeg_sample.png           — raw EEG epoch sample (left vs right)
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support,
)
import config
from dataset import load_and_preprocess_data, make_loaders
from model import HybridBrainTransformer

RESULTS_DIR = config.RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

STYLE  = "seaborn-v0_8-whitegrid"
COLORS = {"left": "#3B82F6", "right": "#EF4444", "loss": "#6366F1", "acc": "#10B981"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history_path: str = None):
    if history_path is None:
        history_path = os.path.join(RESULTS_DIR, "history.json")
    with open(history_path) as f:
        h = json.load(f)

    epochs     = range(1, len(h["train_loss"]) + 1)
    train_loss = h["train_loss"]
    val_acc    = [v * 100 for v in h["val_acc"]]

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Loss
        ax1.plot(epochs, train_loss, color=COLORS["loss"], linewidth=2, marker="o", markersize=4)
        ax1.set_title("Training Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.fill_between(epochs, train_loss, alpha=0.1, color=COLORS["loss"])
        best_epoch = int(np.argmin(train_loss)) + 1
        ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch {best_epoch}")
        ax1.legend()

        # Val Accuracy
        ax2.plot(epochs, val_acc, color=COLORS["acc"], linewidth=2, marker="o", markersize=4)
        ax2.set_title("Validation Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(40, 105)
        ax2.fill_between(epochs, val_acc, alpha=0.1, color=COLORS["acc"])
        best_acc = max(val_acc)
        ax2.axhline(best_acc, color="gray", linestyle="--", linewidth=1, label=f"Best {best_acc:.1f}%")
        ax2.legend()

        fig.suptitle("Brain2Hand — Training Progress", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "01_training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3 & 4. Confusion Matrix + Class Metrics + ROC  (all need model inference)
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference():
    """
    Evaluates the trained model on the SAME held-out validation fold that
    train.py used (seed=42, val_split=0.2). The val fold contains only
    clean, non-augmented samples — no overlap with training data.
    """
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    X, y = load_and_preprocess_data(data_dir=config.DATA_DIR, subjects=config.SUBJECTS)

    # Reproduce train.py's exact split. augment_train=False because we don't
    # need the augmented train fold here — just the clean val fold.
    _, val_loader = make_loaders(X, y, batch_size=128, augment_train=False, seed=42)

    ckpt  = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model = HybridBrainTransformer(
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logits = model(X_b.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_pct],
            ["d", ".1f"],
            ["Counts", "Normalised (%)"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                xticklabels=["Pred: Left", "Pred: Right"],
                yticklabels=["True: Left", "True: Right"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")

        acc = (labels == preds).mean()
        fig.suptitle(f"Brain2Hand Classification Results  |  Overall Accuracy: {acc:.2%}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "02_confusion_matrix.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved {out}")


def plot_class_metrics(labels, preds):
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1])
    classes      = ["Left Hand", "Right Hand"]
    x            = np.arange(len(classes))
    width        = 0.25

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, p  * 100, width, label="Precision", color="#3B82F6", alpha=0.85)
        ax.bar(x,         r  * 100, width, label="Recall",    color="#10B981", alpha=0.85)
        ax.bar(x + width, f1 * 100, width, label="F1 Score",  color="#F59E0B", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=12)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Per-Class Metrics — Precision / Recall / F1", fontweight="bold")
        ax.legend(loc="lower right")

        for i, (pi, ri, fi) in enumerate(zip(p, r, f1)):
            ax.text(i - width, pi * 100 + 1.5, f"{pi:.0%}", ha="center", fontsize=9)
            ax.text(i,         ri * 100 + 1.5, f"{ri:.0%}", ha="center", fontsize=9)
            ax.text(i + width, fi * 100 + 1.5, f"{fi:.0%}", ha="center", fontsize=9)

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "03_class_metrics.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved {out}")


def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc     = auc(fpr, tpr)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#3B82F6", linewidth=2.5,
                label=f"Brain2Hand  (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#3B82F6")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Right Hand Motor Imagery", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "04_roc_curve.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EEG Signal Sample
# ─────────────────────────────────────────────────────────────────────────────

def plot_eeg_sample():
    X, y = load_and_preprocess_data(data_dir=config.DATA_DIR, subjects=[1])

    idx_left  = np.where(y == 0)[0][0]
    idx_right = np.where(y == 1)[0][0]

    sfreq      = 160
    times      = np.linspace(0, 4, X.shape[1])
    channels   = [0, 10, 20, 30]          # 4 representative channels
    chan_names = ["Fc5", "C3", "Cz", "C4"]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, len(channels), figsize=(14, 6), sharey="row")

        for col, (ch, name) in enumerate(zip(channels, chan_names)):
            for row, (idx, label, color) in enumerate([
                (idx_left,  "Left Hand Imagery",  COLORS["left"]),
                (idx_right, "Right Hand Imagery", COLORS["right"]),
            ]):
                ax = axes[row][col]
                ax.plot(times, X[idx, :, ch], color=color, linewidth=1)
                ax.set_title(name if row == 0 else "", fontweight="bold")
                if col == 0:
                    ax.set_ylabel(label, fontsize=10, fontweight="bold")
                if row == 1:
                    ax.set_xlabel("Time (s)")

        fig.suptitle("EEG Signal Epochs — 8–30 Hz Filtered (Mu/Beta Band)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "05_eeg_sample.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Brain2Hand Chart Generator ===\n")

    print("→ [1/5] Training curves...")
    plot_training_curves()

    print("→ [2-4/5] Running model inference for metrics...")
    labels, preds, probs = _run_inference()
    plot_confusion_matrix(labels, preds)
    plot_class_metrics(labels, preds)
    plot_roc_curve(labels, probs)

    print("→ [5/5] EEG signal sample...")
    plot_eeg_sample()

    print(f"\n=== All charts saved to {RESULTS_DIR} ===")
