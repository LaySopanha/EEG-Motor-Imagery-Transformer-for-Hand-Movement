import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import load_and_preprocess_data, make_loaders
from model import HybridBrainTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)
    return correct / total


def load_best_params():
    try:
        with open(config.BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"--- [AutoML] Loaded best_params.json: {params} ---")
        return params
    except Exception:
        print("--- [AutoML] best_params.json not found. Using config defaults. ---")
        return {}


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 50)
    print("   BRAIN2HAND TRAINING  (PyTorch)")
    print(f"   Device : {device}")
    print(f"   Mode   : {config.MODE}  ({len(config.SUBJECTS)} subjects)")
    print("=" * 50 + "\n")

    # ── 1. Hyperparameters ────────────────────────────────────────────────────
    params     = load_best_params()
    lr         = float(params.get("lr",         config.LEARNING_RATE))
    num_layers = int  (params.get("layers",     2))
    dropout    = float(params.get("dropout",    0.3))
    batch_size = int  (params.get("batch_size", config.BATCH_SIZE))
    epochs     = int  (params.get("epochs",     config.EPOCHS))

    # ── 2. Data ───────────────────────────────────────────────────────────────
    X, y = load_and_preprocess_data(data_dir=config.DATA_DIR, subjects=config.SUBJECTS, augment=True)
    train_loader, val_loader = make_loaders(X, y, batch_size=batch_size)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = HybridBrainTransformer(num_layers=num_layers, dropout=dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--- [Model] Parameters: {total_params:,} ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)

    best_val_acc = 0.0
    patience     = 5
    wait         = 0

    print(f"\n--- [Train] Starting {epochs} epochs ---\n")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_acc    = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        print(f"[Epoch {epoch:>3}/{epochs}]  loss={train_loss:.4f}  val_acc={val_acc:.2%}  ({elapsed:.1f}s)")

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc, "num_layers": num_layers, "dropout": dropout},
                       config.CHECKPOINT_PATH)
            print(f"   >>> New best: {val_acc:.2%} — checkpoint saved.")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n[STOP] Early stopping triggered (patience={patience}).")
                break

    print(f"\n{'='*50}")
    print(f"   TRAINING COMPLETE")
    print(f"   Best Val Accuracy : {best_val_acc:.2%}")
    print(f"   Checkpoint        : {config.CHECKPOINT_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    train()
