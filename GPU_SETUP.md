# Brain2Hand GPU Setup Guide

## H200 GPU Installation

### 1. Install CUDA Toolkit (if not already installed)
```bash
# Check CUDA version
nvidia-smi

# Should show CUDA 12.x for H200
```

### 2. Install MindSpore GPU Version
```bash
# Uninstall CPU version
pip uninstall mindspore

# Install GPU version (CUDA 12.x for H200)
pip install mindspore-gpu
```

### 3. Verify GPU Setup
```bash
python -c "import mindspore; mindspore.set_context(device_target='GPU'); print('GPU Ready!')"
```

## Configuration

Edit `src/config.py`:
```python
DEVICE = "GPU"          # Enable GPU
MODE = "FULL"           # Use all 109 subjects
BATCH_SIZE = 64         # Optimized for H200
TUNE_SUBJECTS = 25      # Use 25 subjects for tuning
```

## Expected Performance

| Task | CPU (8 cores) | H200 GPU | Speedup |
|------|---------------|----------|---------|
| Tuning (25 subjects) | 2-3 hours | 5-10 min | **20x** |
| Training (109 subjects) | 6-8 hours | 15-30 min | **30x** |
| Evaluation | 10 min | 30 sec | **20x** |

## Run Training
```bash
# Full pipeline with GPU
./overnight_train.sh
# Now completes in ~1 hour instead of 8!
```

## Troubleshooting

### Out of Memory Error
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 32  # Instead of 64
```

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# Verify MindSpore sees it
python -c "import mindspore; print(mindspore.get_context('device_target'))"
```
