import torch
import torch.nn as nn
from .mamba import MambaBlock
from .kan import KANLayer


class MambaKANClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_hidden: int = 64, kan_terms: int = 64):
        super().__init__()
        self.mamba = MambaBlock(input_dim, seq_hidden)
        # pool across sequence (mean pooling) then pass to KAN head
        self.kan = KANLayer(in_dim=seq_hidden, num_terms=kan_terms, phi_hidden=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> logits (batch, 1)
        """
        seq_out = self.mamba(x)  # (batch, seq_len, hidden)
        pooled = seq_out.mean(dim=1)  # (batch, hidden)
        out = self.kan(pooled)  # (batch, 1)
        return out.squeeze(-1)

    def interpret(self):
        """Return interpretation artifacts from KAN layer.

        Returns projection weights and phi parameters.
        """
        return self.kan.interpret()
