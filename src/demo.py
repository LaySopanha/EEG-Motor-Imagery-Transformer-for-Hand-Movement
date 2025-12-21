import mindspore
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net
import numpy as np
import matplotlib.pyplot as plt
import time

# Import your modules
from dataset import load_and_preprocess_data
from model import Brain2HandTransformer

# --- CONFIGURATION ---
# UPDATE THIS with the actual filename in your checkpoints folder!
CKPT_FILE = "./checkpoints/brain2hand-10_4.ckpt" 

def run_visual_demo():
    print("--- [Demo] Initializing Brain2Hand Simulation ---")
    
    # 1. Load the Data (Simulating a user thinking)
    # We take the first few samples to "replay"
    X, y = load_and_preprocess_data(subject=1)
    
    # 2. Load the Model
    net = Brain2HandTransformer(input_dim=64, num_layers=2)
    
    # Load trained weights
    try:
        param_dict = load_checkpoint(CKPT_FILE)
        load_param_into_net(net, param_dict)
        print(f"--- [Demo] Model weights loaded from {CKPT_FILE} ---")
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}). Running with random weights for demo.")

    net.set_train(False) # Evaluation mode

    # 3. SETUP VISUALIZATION
    plt.ion() # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Brain2Hand: Real-Time MindSpore Inference', fontsize=16, fontweight='bold')

    classes = ['Left Hand', 'Right Hand'] # Labels 0 and 1
    
    print("--- [Demo] Starting Simulation Loop... (Press Ctrl+C to stop) ---")

    # Loop through samples to simulate a continuous stream
    for i in range(len(X)):
        # Get one sample (1 trial)
        # Shape: (1, Time, Channels)
        input_tensor = Tensor(np.expand_dims(X[i], axis=0), mindspore.float32)
        true_label = classes[y[i]]

        # INFERENCE (The AI part)
        logits = net(input_tensor)
        probs =  mindspore.ops.Softmax(axis=1)(logits)
        probs_np = probs.asnumpy()[0] # e.g., [0.8, 0.2]
        
        # Determine Prediction
        pred_idx = np.argmax(probs_np)
        prediction = classes[pred_idx]
        confidence = probs_np[pred_idx] * 100

        # --- UPDATE PLOTS ---
        ax1.clear()
        ax2.clear()

        # Plot 1: The EEG Signal (Show just one channel, e.g., C3 sensor)
        # We plot the first 200 time steps to make it look like a wave
        raw_wave = X[i, :200, 0] 
        ax1.plot(raw_wave, color='#0077BE')
        ax1.set_title(f"Live EEG Stream (Sensor C3) - User Intention: {true_label}", fontsize=12)
        ax1.set_ylim([-3, 3]) # Fixed scale to stop jumping
        ax1.set_ylabel("Amplitude (uV)")
        ax1.grid(True, alpha=0.3)

        # Plot 2: The Model Decision (Bar Chart)
        bars = ax2.bar(classes, probs_np, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_ylim([0, 1])
        ax2.set_title(f"MindSpore Prediction: {prediction.upper()} ({confidence:.1f}%)", fontsize=14)
        ax2.set_ylabel("Confidence")
        
        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        # Highlight the decision
        if prediction == "Left Hand":
            fig.patch.set_facecolor('#fff5f5') # Reddish tint
        else:
            fig.patch.set_facecolor('#f0fffa') # Greenish tint

        plt.draw()
        plt.pause(2) # Pause for 2 seconds to let the viewer see the result
        
        print(f"Sample {i}: True={true_label}, Pred={prediction}, Conf={confidence:.1f}%")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visual_demo()