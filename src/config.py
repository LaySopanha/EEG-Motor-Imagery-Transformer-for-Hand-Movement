# --- BRAIN2HAND CENTRAL CONFIGURATION ---

# 1. DATASET SCALING MODE
# Change this to switch between laptop and cloud training
MODE = "CLOUD"  # Options: "LAPTOP", "CLOUD", "FULL"

# Dynamic subject selection based on MODE
if MODE == "LAPTOP":
    SUBJECTS = [1, 2]  # Fast, for debugging (2 subjects)
elif MODE == "CLOUD":
    SUBJECTS = list(range(1, 26))  # Cloud training (25 subjects, ~90% accuracy)
elif MODE == "FULL":
    SUBJECTS = list(range(1, 110))  # Full PhysioNet (109 subjects, maximum accuracy)
else:
    SUBJECTS = [1, 2]  # Default fallback

# Tuning Configuration
# How many subjects to use for hyperparameter tuning (subset of SUBJECTS)
# Lower = Faster tuning, Higher = Better hyperparameters
TUNE_SUBJECTS = 10  # Options: 1 (fast), 5 (balanced), len(SUBJECTS) (best but slow)

# Data Locations
DATA_DIR = "./data"

# 2. HARDWARE CONFIGURATION
# Device: "CPU" or "GPU"
DEVICE = "GPU"  # Set to "GPU" for H200 acceleration

# 3. TRAINING HYPERPARAMETERS
EPOCHS = 50           # Sufficient for convergence
BATCH_SIZE = 64       # Increased for H200 GPU (141GB memory!)
LEARNING_RATE = 0.001 # Default start, overriden by best_params.json if available

# 4. MOCKING
# If True, generates fake data if internet is down. 
# Set to False for real training.
USE_MOCK_ON_FAIL = True
