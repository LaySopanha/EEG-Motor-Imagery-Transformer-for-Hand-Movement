
# Brain2Hand: EEG-Based Intention Decoding System

**Huawei ICT Competition 2025-2026 | Cloud Track | Team B2H**

Brain2Hand is a Brain-Computer Interface (BCI) system that decodes motor imagery intentions (Left Hand vs. Right Hand movements) from EEG signals using a Spatial-Temporal Transformer built with MindSpore.

## Features

- **MindSpore Transformer**: Custom Transformer encoder for EEG signal classification.
- **Real-Time Visualization**: Streamlit dashboard simulating real-time EEG processing.
- **PhysioNet Integration**: Automated downloading and preprocessing of the PhysioNet Motor Imagery Dataset.

## Installation

1.  **Clone the repository**
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Launch the Dashboard (Demo Mode)
To see the system in action immediately:
```bash
streamlit run src/app.py
```
*Note: The system will automatically generate synthetic data if the model or dataset is not found.*

### 2. Train the Model (Optional)
To train the Transformer on real PhysioNet data:
```bash
python src/train.py
```
This will download the dataset (~50MB), train the model, and save checkpoints to `checkpoints/`.

## 📂 Project Structure

- `src/model.py`: MindSpore Transformer architecture.
- `src/train.py`: Training loop and checkpoint management.
- `src/dataset.py`: Data loading and preprocessing pipeline (MNE + PhysioNet).
- `src/app.py`: Streamlit web interface for demonstration.
- `src/tune.py`: Hyperparameter optimization using Optuna.
- `src/eval.py`: Model evaluation on test set.
- `src/overnight_train.sh`: Script for overnight training pipeline.


## Tech Stack

- **Framework**: MindSpore
- **Frontend**: Streamlit
- **Data Processing**: MNE-Python, NumPy
=======
# EEG Motor Imagery Transformer for Hand Movement

