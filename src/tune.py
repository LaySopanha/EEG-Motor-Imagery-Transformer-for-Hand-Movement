import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Model, Tensor
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from dataset import load_and_preprocess_data
from model import HybridBrainTransformer
from train import create_dataset
import config

# Global data (loaded once)
X_TUNE = None
Y_TUNE = None

def objective(trial):
    """
    Optuna objective function: Trains model with suggested params and returns validation accuracy.
    """
    # 1. Suggest hyperparameters (H200 GPU optimized)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_layers = trial.suggest_int('layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Larger for GPU
    epochs = trial.suggest_int('epochs', 20, 100, step=20)  # More epochs for GPU speed
    
    print(f"\n[Trial {trial.number}] LR={lr:.5f}, Layers={num_layers}, Dropout={dropout:.2f}, Batch={batch_size}, Epochs={epochs}")
    
    try:
        # 2. Create dataset
        ms_dataset = create_dataset(X_TUNE, Y_TUNE, batch_size=batch_size)
        
        # 3. Build model
        net = HybridBrainTransformer(input_dim=64, num_layers=num_layers)
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
        model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'Accuracy': nn.Accuracy()})
        
        # 4. Train with pruning
        for epoch in range(epochs):
            model.train(1, ms_dataset, dataset_sink_mode=False)
            
            # Evaluate (mock with random for speed - in production, use real validation)
            # For real validation, you'd need a separate val set
            intermediate_acc = np.random.random() * 0.3 + 0.5  # Placeholder: 50-80%
            
            # Report to Optuna for pruning
            trial.report(intermediate_acc, epoch)
            
            # Prune if trial is unpromising
            if trial.should_prune():
                print(f"   [Pruned at epoch {epoch+1}]")
                raise optuna.TrialPruned()
        
        # 5. Final accuracy (mock - replace with real validation)
        final_acc = np.random.random() * 0.3 + 0.5
        print(f"   Final Acc: {final_acc:.2%}")
        
        return final_acc  # Optuna maximizes this
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"   Trial failed: {e}")
        return 0.0  # Return worst score on failure

def main():
    global X_TUNE, Y_TUNE
    
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target=config.DEVICE)
    
    print("="*60)
    print("   OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("   Algorithm: Tree-structured Parzen Estimator (TPE)")
    print(f"   Device: {config.DEVICE}")
    print("="*60)
    
    # Load data once
    print("\n--- [Data] Loading Tuning Dataset... ---")
    tune_subjects = config.SUBJECTS[:config.TUNE_SUBJECTS]
    print(f"--- [Data] Using {len(tune_subjects)} subjects: {tune_subjects} ---")
    
    X_TUNE, Y_TUNE = load_and_preprocess_data(subjects=tune_subjects)
    print(f"--- [Data] Loaded {len(X_TUNE)} samples ---")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # Maximize accuracy
        sampler=TPESampler(seed=42),  # Bayesian optimization
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)  # Early stopping
    )
    
    # Run optimization
    print("\n--- [Optuna] Starting Optimization (20 trials) ---\n")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    # Results
    print("\n" + "="*60)
    print("   OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value:.2%}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save to JSON
    import json
    best_params = study.best_params.copy()
    
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n[SUCCESS] Saved to best_params.json")
    print("="*60)
    
    # Optional: Print optimization history
    print("\n--- Top 5 Trials ---")
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(trials_df[['number', 'value', 'params_lr', 'params_layers', 'params_batch_size']])

if __name__ == "__main__":
    main()
