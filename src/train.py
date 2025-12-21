import os
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Model, save_checkpoint
from mindspore.train.callback import Callback, LossMonitor

# Import your modules
from dataset import load_and_preprocess_data
from model import HybridBrainTransformer
import config 

# --- CONFIGURATION ---
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LR = config.LEARNING_RATE # Default, will be overriden by best_params

class TrainingLogger(Callback):
    """Custom Callback for professional logging and Early Stopping"""
    def __init__(self, per_print_times=1, patience=3):
        super(TrainingLogger, self).__init__()
        self.per_print_times = per_print_times
        self.best_loss = -1.0 # Initialize for Accuracy (Maximization)
        self.patience = patience
        self.wait = 0

    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"\n[INFO] Starting Epoch {cb_params.cur_epoch_num}/{cb_params.epoch_num}...")
        self.epoch_start = time.time()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        # Helper to get float from various MS return types
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], (mindspore.Tensor,)):
                loss = loss[0]
        if isinstance(loss, mindspore.Tensor):
            loss = loss.asnumpy()
        
        # 1. Logging Training Status
        epoch_seconds = (time.time() - self.epoch_start)
        
        # 2. Perform Validation (If available)
        val_acc_str = ""
        current_score = -loss # Default to negative loss (maximize)
        
        if hasattr(self, 'model') and hasattr(self, 'eval_dataset'):
            # Run evaluation
            metrics = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            acc = metrics.get('Accuracy', 0.0)
            val_acc_str = f"| Val Acc: {acc:.2%}"
            current_score = acc # Overwrite with Accuracy (maximize)
            
        print(f"[INFO] Epoch {cb_params.cur_epoch_num} Finished. Loss: {loss:.4f} {val_acc_str} | Time: {epoch_seconds:.2f}s")
        
        # 3. Early Stopping Logic (Based on ACCURACY if available, else Loss)
        # If we have accuracy, we want to maximize it. If loss, minimize.
        # Simple hack: if using acc, store best_score as -inf and check >
        # Current logic assumes MINIMIZING loss. Let's adapt.
        
        # If we are using Accuracy (Maximize)
        if hasattr(self, 'eval_dataset'):
             # Logic for Maximizing Accuracy
            if current_score > self.best_loss: # Reusing 'best_loss' variable name for 'best_score'
                self.best_loss = current_score
                self.wait = 0
                print(f"   >>> Accuracy Improved. Resetting patience.")
            else:
                self.wait += 1
                print(f"   >>> Accuracy did not improve. Patience: {self.wait}/{self.patience}")
        else:
            # Logic for Minimizing Loss
            if loss < self.best_loss:
                self.best_loss = loss
                self.wait = 0
                print(f"   >>> Loss Improved. Resetting patience.")
            else:
                self.wait += 1
                print(f"   >>> Loss did not improve. Patience: {self.wait}/{self.patience}")

        if self.wait >= self.patience:
            print(f"\n[STOP] Early Stopping triggered!")
            run_context.request_stop()

def create_dataset(X, y, batch_size=2):
    """
    Converts Numpy arrays into a MindSpore GeneratorDataset
    """
    def generator():
        for i in range(len(X)):
            yield (X[i], y[i])

    dataset = ds.GeneratorDataset(generator, column_names=["data", "label"])
    dataset = dataset.batch(batch_size)
    return dataset

def train():
    # 1. SETUP ENVIRONMENT
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target=config.DEVICE)
    print("\n" + "="*40)
    print("   BRAIN2HAND TRAINING SYSTEM v2.0")
    print("   Target: Hybrid CNN-Transformer")
    print(f"   Device: {config.DEVICE}")
    if config.DEVICE == "GPU":
        print("   GPU: H200 (141GB) - Optimized")
    print("="*40 + "\n")

    # 2. PREPARE DATA
    # Load 2 subjects to increase data diversity
    # Augment=True is default, but explicit for clarity
    print(f"--- [System] Using Subjects: {config.SUBJECTS} ---")
    X_all, y_all = load_and_preprocess_data(subjects=config.SUBJECTS, augment=True)
    
    # Validation Split (80/20)
    split_idx = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]
    
    print(f"--- [Data] Split: Train={len(X_train)} | Val={len(X_val)} ---")

    # 3. LOAD HYPERPARAMETERS (before creating datasets)
    # Initialize defaults first
    LEARNING_RATE = LR
    NUM_LAYERS = 2
    DROPOUT = 0.1
    BATCH_SIZE_TUNED = BATCH_SIZE
    EPOCHS_TUNED = EPOCHS
    
    # Try to load optimized parameters
    import json
    try:
        with open("best_params.json", "r") as f:
            content = f.read().replace("'", '"')
            best_params = json.loads(content)
            
        print(f"--- [AutoML] Loaded Best Params: {best_params} ---")
        LEARNING_RATE = best_params.get('lr', LR)
        NUM_LAYERS = int(best_params.get('layers', 2))
        DROPOUT = best_params.get('dropout', 0.1)
        BATCH_SIZE_TUNED = best_params.get('batch_size', BATCH_SIZE)
        EPOCHS_TUNED = best_params.get('epochs', EPOCHS)
        
        print(f"--- [AutoML] Using: LR={LEARNING_RATE}, Layers={NUM_LAYERS}, Batch={BATCH_SIZE_TUNED}, Epochs={EPOCHS_TUNED} ---")
        
    except Exception as e:
        print(f"--- [AutoML] Could not load best_params.json ({e}). Using Defaults. ---")

    # 4. CREATE DATASETS (after loading batch size)
    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE_TUNED)
    val_dataset = create_dataset(X_val, y_val, BATCH_SIZE_TUNED)
    
    steps_per_epoch = train_dataset.get_dataset_size()
    print(f"--- [Data] Batches per epoch: {steps_per_epoch} ---")

    # 5. SETUP MODEL
    net = HybridBrainTransformer(input_dim=64, num_layers=NUM_LAYERS) 
    print(f"--- [Model] Initialized HybridBrainTransformer (Layers={NUM_LAYERS}) ---")
    
    # Loss Function & Optimizer
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # Decay learning rate
    lr_schedule = nn.cosine_decay_lr(min_lr=0.00001, max_lr=LEARNING_RATE, total_step=EPOCHS_TUNED*steps_per_epoch, step_per_epoch=steps_per_epoch, decay_epoch=EPOCHS_TUNED)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr_schedule)

    # 4. TRAINING LOOP
    # We define 'Accuracy' as the metric to track
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'Accuracy': nn.Accuracy()})
    
    # Callbacks
    logger_cb = TrainingLogger(patience=5) # Increased patience slightly
    
    # Checkpoint Dir
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    print("\n--- [System] Commencing Training Loop ---")
    
    # Custom Loop to print Validation Accuracy
    # Note: Model.train allows callbacks, but usually prints training loss.
    # To see val accuracy per epoch, we can use a custom Callback or just loop manually.
    # For professional display, let's use the simplest MindSpore way: Model.eval in the callback?
    # Or simply:
    
    # Advanced: Pass Eval Dataset to the Callback? 
    # Let's Modify TrainingLogger to handle eval!
    logger_cb.model = model
    logger_cb.eval_dataset = val_dataset
    
    model.train(EPOCHS_TUNED, train_dataset, callbacks=[logger_cb], dataset_sink_mode=False)
    
    # Final Validation
    print("\n--- [Validation] Final Evaluation ---")
    acc = model.eval(val_dataset, dataset_sink_mode=False)
    print(f"Final Validation Accuracy: {acc}")

    # Save Final Model
    save_path = "./checkpoints/brain2hand-hybrid.ckpt"
    save_checkpoint(net, save_path)
    print(f"\n[SUCCESS] Model saved to {save_path}")
    print("="*40)

if __name__ == "__main__":
    train()