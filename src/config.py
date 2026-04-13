# --- BRAIN2HAND CENTRAL CONFIGURATION ---

from pathlib import Path

# Repo root, computed from this file's location.
# All other paths are anchored here so scripts work regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. DATASET SCALING MODE
# Change this to switch between laptop and cloud training
MODE = "FULL"  # Options: "LAPTOP", "CLOUD", "FULL"

if MODE == "LAPTOP":
    SUBJECTS = [1, 2]               # Fast, for debugging
elif MODE == "CLOUD":
    SUBJECTS = list(range(1, 26))   # 25 subjects, ~90% accuracy target
elif MODE == "FULL":
    SUBJECTS = list(range(1, 110))  # All 109 PhysioNet subjects
else:
    SUBJECTS = [1, 2]

# How many subjects to use during hyperparameter tuning (subset of SUBJECTS)
TUNE_SUBJECTS = 10  # Lower = faster; higher = better params

# Data location
DATA_DIR = str(REPO_ROOT / "data")

# 2. HARDWARE
# "cpu" or "cuda" (Lightning AI GPU)
DEVICE = "cuda"

# 3. TRAINING HYPERPARAMETERS (defaults; overridden by best_params.json if available)
EPOCHS       = 50
BATCH_SIZE   = 256
LEARNING_RATE = 0.001

# 4. MISC
USE_MOCK_ON_FAIL = True  # Fall back to random data if download fails
CHECKPOINT_PATH  = str(REPO_ROOT / "checkpoints" / "brain2hand.pt")
BEST_PARAMS_PATH = str(REPO_ROOT / "best_params.json")
RESULTS_DIR      = str(REPO_ROOT / "results")
