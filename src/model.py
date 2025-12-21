import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

class HybridBrainTransformer(nn.Cell):
    """
    State-of-the-Art Hybrid Architecture:
    1. Spatial-Temporal Convolution (CNN) for local feature extraction
    2. Transformer Encoder for long-range dependency capture
    """
    def __init__(self, input_dim=64, d_model=128, n_head=4, num_layers=2, num_classes=2):
        super(HybridBrainTransformer, self).__init__()
        
        # --- Stage 1: Convolutional Feature Extraction (The "Eye") ---
        # Input shape expected: (Batch, Time, Channels)
        # We process it as a 1D sequence initially
        
        self.conv_spatial = nn.SequentialCell([
            # Conv1d behaves like a temporal filter over channels if we transpose correctly
            # Input: (N, C_in, L) -> we will transpose input to this
            nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=15, 
                      stride=2, padding=0, pad_mode='same', has_bias=True),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        ])
        
        # --- Stage 2: Transformer Encoder (The "Brain") ---
        # Captures the "intent" over the simplified signal sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, 
                                                   dim_feedforward=512, dropout=0.3, # Higher dropout for regularization
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Stage 3: Classification Head (The "Hand") ---
        self.classifier = nn.SequentialCell([
            nn.Dense(d_model, 64),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Dense(64, num_classes)
        ])

    def construct(self, x):
        """
        x shape: (Batch, Time, Channels) e.g., (B, 641, 64)
        """
        # 1. Align for Conv1d: Needs (Batch, Channels, Time)
        x = ops.transpose(x, (0, 2, 1)) # -> (B, 64, 641)
        
        # 2. Extract Features
        x = self.conv_spatial(x) # -> (B, 128, Reduced_Time)
        
        # 3. Align for Transformer: Needs (Batch, Time, Channels/d_model)
        x = ops.transpose(x, (0, 2, 1)) # -> (B, Reduced_Time, 128)
        
        # 4. Global Context
        x = self.transformer(x) # -> (B, Reduced_Time, 128)
        
        # 5. Pooling (Global Average)
        x = x.mean(axis=1) # -> (B, 128)
        
        # 6. Classify
        logits = self.classifier(x) # -> (B, 2)
        return logits

# Backwards compatibility alias
Brain2HandTransformer = HybridBrainTransformer

if __name__ == "__main__":
    import mindspore.context as context
    from mindspore import Tensor
    import numpy as np

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    # Simulate dummy input
    # Batch=2, Time=641, Channels=64
    dummy_input = Tensor(np.random.randn(2, 641, 64), mindspore.float32)
    
    model = HybridBrainTransformer(input_dim=64)
    output = model(dummy_input)
    
    print("--- Hybrid Model Structure ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: 2, 2)")
    print("Success: Hybrid CNN-Transformer built!")