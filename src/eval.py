import numpy as np
import mindspore
from mindspore import Tensor, load_checkpoint, load_param_into_net
import mindspore.ops as ops
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_and_preprocess_data
from model import HybridBrainTransformer
import config

def evaluate():
    print("--- [Eval] Standard Evaluation Protocol 1.0 ---")
    
    # 1. Load Data (Same Subjects as Train for consistency, or new for cross-subject)
    # Using Subjects from Config ensures we evaluate what we defined
    # CRITICAL: Do NOT augment Eval data! We want to test on real signals, not noisy ones.
    print(f"--- [Eval] Using Subjects: {config.SUBJECTS} ---")
    X, y = load_and_preprocess_data(subjects=config.SUBJECTS, augment=False)
    
    # We will evaluate on the *entire* set to get a comprehensive report, 
    # or you could manually slice for the hold-out set if you fixed the seed.
    # For this script, evaluating the full dataset provides the "System Performance" overview.
    
    # 2. Load Model
    # Automagically detect the trained architecture
    import json
    try:
        with open("best_params.json", "r") as f:
            content = f.read().replace("'", '"')
            best_params = json.loads(content)
        NUM_LAYERS = int(best_params.get('layers', 2))
        print(f"--- [Eval] Detected Trained Architecture: Layers={NUM_LAYERS} ---")
    except:
        NUM_LAYERS = 2 # Default fallback
        print(f"--- [Eval] Warning: Could not load best_params.json. Defaulting to Layers=2 ---")

    net = HybridBrainTransformer(input_dim=64, num_layers=NUM_LAYERS) 
    ckpt_path = "./checkpoints/brain2hand-hybrid.ckpt"
    
    try:
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(net, param_dict)
        print(f"--- [Eval] Loaded Model: {ckpt_path} ---")
    except Exception as e:
        print(f"[Results] Error loading model: {e}")
        return

    net.set_train(False)

    print("--- [Eval] Running Inference... ---")
    
    # Batch processing to avoid memory issues if dataset is massive (optional for this size)
    # Simple loop for prototype:
    preds = []
    
    # Inference Loop
    for i in range(len(X)):
        input_tensor = Tensor(np.expand_dims(X[i], axis=0), mindspore.float32)
        logits = net(input_tensor)
        prob = ops.Softmax(axis=1)(logits).asnumpy()[0]
        pred_label = np.argmax(prob)
        preds.append(pred_label)
        
        if i % 50 == 0:
            print(f"   > Processed {i}/{len(X)} samples...")

    preds = np.array(preds)
    
    # 3. Metrics
    acc = accuracy_score(y, preds)
    print(f"\n========================================")
    print(f"   OVERALL ACCURACY: {acc:.2%}")
    print(f"========================================")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y, preds, target_names=['Left Hand', 'Right Hand']))
    
    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Left', 'Pred Right'], 
                yticklabels=['Actual Left', 'Actual Right'])
    plt.title('Brain2Hand Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_img = "eval_confusion_matrix.png"
    plt.savefig(save_img)
    print(f"\n[Viz] Confusion Matrix saved to: {save_img}")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    evaluate()
