# Results

This folder holds the training artifacts that reviewers can inspect directly on GitHub without re-running anything.

## What lives here

| File | Produced by | Description |
|---|---|---|
| `history.json` | `src/train.py` | Per-epoch `train_loss` and `val_acc` lists |
| `01_training_curves.png` | `src/plot.py` | Loss and validation accuracy over epochs |
| `02_confusion_matrix.png` | `src/plot.py` | Confusion matrix (counts + normalized) |
| `03_class_metrics.png` | `src/plot.py` | Per-class precision, recall, F1 |
| `04_roc_curve.png` | `src/plot.py` | ROC curve with AUC |
| `05_eeg_sample.png` | `src/plot.py` | Sample left vs right hand EEG epochs |

## How the workflow goes

1. Push code changes from local to GitHub.
2. On Lightning AI studio: `git pull`, then `python src/train.py && python src/plot.py`.
3. On Lightning AI: commit and push the updated `results/` back to GitHub.
4. On local: `git pull` to sync the fresh artifacts.

Only the files listed above should be committed — nothing else in this folder is tracked.

## Current status

The PNGs and `history.json` are regenerated on every full training run on Lightning AI. If this folder looks empty on GitHub, it means a fresh run is pending.
