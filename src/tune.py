"""
Optuna hyperparameter search for HybridBrainTransformer.
Uses REAL validation accuracy (not mock) to guide the search.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import config
from dataset import load_and_preprocess_data, make_loaders
from model import HybridBrainTransformer

# ── Global data cache (loaded once before all trials) ─────────────────────────
_X = _y = None
_DEVICE = None


def _get_val_acc(model, val_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total


def objective(trial: optuna.Trial) -> float:
    lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
    num_layers = trial.suggest_int  ("layers",     1, 4)
    dropout    = trial.suggest_float("dropout",    0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs     = trial.suggest_int  ("epochs",     20, 60, step=20)

    print(f"\n[Trial {trial.number}]  lr={lr:.5f}  layers={num_layers}  "
          f"dropout={dropout:.2f}  batch={batch_size}  epochs={epochs}")

    train_loader, val_loader = make_loaders(_X, _y, batch_size=batch_size)

    model     = HybridBrainTransformer(num_layers=num_layers, dropout=dropout).to(_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(X_batch), y_batch).backward()
            optimizer.step()
        scheduler.step()

        val_acc = _get_val_acc(model, val_loader)
        trial.report(val_acc, epoch)

        if trial.should_prune():
            print(f"   [Pruned at epoch {epoch + 1}]")
            raise optuna.TrialPruned()

    print(f"   Final val_acc: {val_acc:.2%}")
    return val_acc


def main():
    global _X, _y, _DEVICE

    _DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("   OPTUNA HYPERPARAMETER SEARCH  (PyTorch)")
    print(f"   Device  : {_DEVICE}")
    print(f"   Subjects: {config.TUNE_SUBJECTS} (subset)")
    print("=" * 60)

    tune_subjects = config.SUBJECTS[: config.TUNE_SUBJECTS]
    _X, _y = load_and_preprocess_data(data_dir=config.DATA_DIR, subjects=tune_subjects, augment=True)
    print(f"--- [Data] {len(_X)} samples for tuning ---\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("\n" + "=" * 60)
    print(f"   BEST TRIAL  #{study.best_trial.number}")
    print(f"   Best Val Accuracy : {study.best_value:.2%}")
    print(f"   Best Params       : {study.best_params}")
    print("=" * 60)

    with open(config.BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n[SUCCESS] Saved → {config.BEST_PARAMS_PATH}")

    df = study.trials_dataframe().sort_values("value", ascending=False).head(5)
    print("\n--- Top 5 Trials ---")
    print(df[["number", "value", "params_lr", "params_layers", "params_batch_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
