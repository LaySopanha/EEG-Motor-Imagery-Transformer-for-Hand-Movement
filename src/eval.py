import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import load_and_preprocess_data, EEGDataset
from model import HybridBrainTransformer
from torch.utils.data import DataLoader


def evaluate():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("\n--- [Eval] Brain2Hand Evaluation ---")
    print(f"--- [Eval] Device : {device} ---")

    # ── 1. Load data (no augmentation — test on clean signals only) ───────────
    print(f"--- [Eval] Subjects: {config.SUBJECTS} ---")
    X, y = load_and_preprocess_data(data_dir=config.DATA_DIR, subjects=config.SUBJECTS, augment=False)
    dataset = EEGDataset(X, y)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"[ERROR] No checkpoint found at {config.CHECKPOINT_PATH}. Run train.py first.")
        return

    ckpt       = torch.load(config.CHECKPOINT_PATH, map_location=device)
    num_layers = ckpt.get("num_layers", 2)
    dropout    = ckpt.get("dropout",    0.3)

    model = HybridBrainTransformer(num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"--- [Eval] Loaded checkpoint (epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.2%}) ---")

    # ── 3. Inference ──────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    print(f"\n{'='*50}")
    print(f"   OVERALL ACCURACY: {acc:.2%}")
    print(f"{'='*50}")
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=["Left Hand", "Right Hand"]))

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: Left", "Pred: Right"],
                yticklabels=["True: Left", "True: Right"])
    plt.title(f"Brain2Hand Confusion Matrix  (Acc={acc:.2%})")
    plt.tight_layout()
    out_path = "eval_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[Viz] Confusion matrix saved → {out_path}")
    print("--- Evaluation complete ---\n")


if __name__ == "__main__":
    evaluate()
