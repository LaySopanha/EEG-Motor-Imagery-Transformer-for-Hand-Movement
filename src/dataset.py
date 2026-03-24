import numpy as np
import mne
from mne.datasets import eegbci
from torch.utils.data import Dataset, DataLoader


# ── PyTorch Dataset wrapper ───────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Main data loading function ────────────────────────────────────────────────

def load_and_preprocess_data(data_dir: str = "./data", subjects: list = [1, 2], augment: bool = True):
    """
    Loads PhysioNet EEG motor imagery data for the given subjects.

    Returns
    -------
    X : np.ndarray  shape (N, 641, 64)  — float32, z-score normalised
    y : np.ndarray  shape (N,)          — int64, 0=Left / 1=Right
    """
    print(f"--- [Data] Loading {len(subjects)} subject(s): {subjects} ---")

    all_X, all_y = [], []

    try:
        mne.set_config("MNE_DATASETS_EEGBCI_PATH", data_dir)

        for subject in subjects:
            try:
                runs = [4, 8, 12]   # Motor imagery: left/right fist
                fnames = eegbci.load_data(subject, runs, path=data_dir, update_path=False, verbose=False)

                raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
                for fname in fnames[1:]:
                    raw.append(mne.io.read_raw_edf(fname, preload=True, verbose=False))

                eegbci.standardize(raw)
                raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
                raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False)

                events, _ = mne.events_from_annotations(raw, verbose=False)
                epochs = mne.Epochs(
                    raw, events,
                    event_id=dict(T1=2, T2=3),
                    tmin=0, tmax=4.0,
                    proj=True, baseline=None, preload=True, verbose=False,
                )

                # (N, Channels, Time) → (N, Time, Channels)
                X_subj = epochs.get_data(copy=True).transpose(0, 2, 1).astype(np.float32)
                y_subj = (epochs.events[:, -1] - 2).astype(np.int64)  # 2→0 (Left), 3→1 (Right)

                all_X.append(X_subj)
                all_y.append(y_subj)
                print(f"   > Subject {subject:03d}: {X_subj.shape[0]} trials")

            except Exception as e:
                print(f"   > Subject {subject:03d}: SKIPPED ({e})")

        if not all_X:
            raise ValueError("No subjects loaded successfully.")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

    except Exception as e:
        print(f"--- [WARN] Data loading failed ({e}). Using MOCK data. ---")
        X = np.random.randn(60, 641, 64).astype(np.float32)
        y = np.random.randint(0, 2, size=(60,)).astype(np.int64)
        return X, y

    # ── Normalisation ─────────────────────────────────────────────────────────
    X = X * 1e6                              # Volts → microvolts
    X = (X - X.mean()) / (X.std() + 1e-8)   # Z-score

    print(f"--- [Data] Raw shape: {X.shape}  |  Classes: {np.bincount(y)} ---")

    if augment:
        X, y = _augment(X, y)
        print(f"--- [Data] Augmented shape: {X.shape} ---")

    return X, y


# ── Data augmentation ─────────────────────────────────────────────────────────

def _augment(X: np.ndarray, y: np.ndarray, noise_level: float = 0.05, shift_max: int = 10):
    """
    3× dataset expansion:
      1. Original sample
      2. Gaussian noise injection  — simulates sensor noise
      3. Temporal shift (roll)     — simulates imperfect trial locking
    """
    X_out, y_out = [X], [y]

    # Noise
    X_noise = X + np.random.normal(0, noise_level, X.shape).astype(np.float32)
    X_out.append(X_noise)
    y_out.append(y)

    # Temporal shift
    shifts = np.random.randint(-shift_max, shift_max, size=len(X))
    X_shift = np.stack([np.roll(X[i], s, axis=0) for i, s in enumerate(shifts)])
    X_out.append(X_shift.astype(np.float32))
    y_out.append(y)

    return np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 64, val_split: float = 0.2):
    """Returns (train_loader, val_loader)."""
    split = int(len(X) * (1 - val_split))
    train_ds = EEGDataset(X[:split], y[:split])
    val_ds   = EEGDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"--- [Data] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples ---")
    return train_loader, val_loader


if __name__ == "__main__":
    X, y = load_and_preprocess_data(subjects=[1])
    train_loader, val_loader = make_loaders(X, y)
    xb, yb = next(iter(train_loader))
    print(f"Batch X: {xb.shape}, y: {yb.shape}")
