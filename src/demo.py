import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from dataset import load_and_preprocess_data
from model import HybridBrainTransformer


def run_visual_demo():
    print("--- [Demo] Initializing Brain2Hand Simulation ---")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    X, y = load_and_preprocess_data(subjects=[1], augment=False)

    ckpt_path = config.CHECKPOINT_PATH
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        num_layers = ckpt.get("num_layers", 2)
        dropout = ckpt.get("dropout", 0.3)
        model = HybridBrainTransformer(num_layers=num_layers, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"--- [Demo] Model weights loaded from {ckpt_path} ---")
    else:
        model = HybridBrainTransformer(num_layers=2, dropout=0.3).to(device)
        print(f"Warning: No checkpoint at {ckpt_path}. Running with random weights.")

    model.eval()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Brain2Hand: Real-Time PyTorch Inference", fontsize=16, fontweight="bold")

    classes = ["Left Hand", "Right Hand"]

    print("--- [Demo] Starting Simulation Loop... (Press Ctrl+C to stop) ---")

    for i in range(len(X)):
        input_tensor = torch.from_numpy(X[i:i+1]).float().to(device)
        true_label = classes[int(y[i])]

        with torch.no_grad():
            logits = model(input_tensor)
            probs_np = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs_np))
        prediction = classes[pred_idx]
        confidence = probs_np[pred_idx] * 100

        ax1.clear()
        ax2.clear()

        raw_wave = X[i, :200, 0]
        ax1.plot(raw_wave, color="#0077BE")
        ax1.set_title(f"Live EEG Stream (Sensor C3) - User Intention: {true_label}", fontsize=12)
        ax1.set_ylim([-3, 3])
        ax1.set_ylabel("Amplitude (uV)")
        ax1.grid(True, alpha=0.3)

        bars = ax2.bar(classes, probs_np, color=["#FF6B6B", "#4ECDC4"])
        ax2.set_ylim([0, 1])
        ax2.set_title(f"Brain2Hand Prediction: {prediction.upper()} ({confidence:.1f}%)", fontsize=14)
        ax2.set_ylabel("Confidence")

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f"{height:.2f}", ha="center", va="bottom")

        if prediction == "Left Hand":
            fig.patch.set_facecolor("#fff5f5")
        else:
            fig.patch.set_facecolor("#f0fffa")

        plt.draw()
        plt.pause(2)

        print(f"Sample {i}: True={true_label}, Pred={prediction}, Conf={confidence:.1f}%")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual_demo()
