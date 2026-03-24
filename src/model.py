import torch
import torch.nn as nn


class HybridBrainTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for EEG motor imagery classification.

    Architecture:
        Stage 1 — Spatial-Temporal CNN  : extracts local features from raw EEG
        Stage 2 — Transformer Encoder   : captures long-range temporal dependencies
        Stage 3 — Classification Head   : produces left / right hand logits

    Input shape : (Batch, Time=641, Channels=64)
    Output shape: (Batch, num_classes=2)
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        n_head: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── Stage 1: Spatial-Temporal CNN ────────────────────────────────────
        # Input transposed to (B, C=64, T=641) before this block
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
        )

        # ── Stage 2: Transformer Encoder ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Stage 3: Classification Head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.permute(0, 2, 1)          # → (B, C, T)
        x = self.cnn(x)                  # → (B, d_model, T')
        x = x.permute(0, 2, 1)          # → (B, T', d_model)
        x = self.transformer(x)          # → (B, T', d_model)
        x = x.mean(dim=1)               # global average pool → (B, d_model)
        return self.classifier(x)        # → (B, num_classes)


# Alias for backward compatibility
Brain2HandTransformer = HybridBrainTransformer


if __name__ == "__main__":
    model = HybridBrainTransformer()
    dummy = torch.randn(4, 641, 64)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}  (expected: [4, 2])")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
