import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    """Lightweight continuous-time state-space block (Mamba-like).

    Implements a simple linear state update with learnable decay (alpha)
    and input/output mappings. This is O(N) in sequence length and hidden dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input transform B: maps input -> hidden state increments
        self.B = nn.Linear(input_dim, hidden_dim)
        # Output transform C: maps hidden state -> output features
        self.C = nn.Linear(hidden_dim, hidden_dim)

        # Learnable decay (per hidden dimension), initialize near 0.9
        self.logit_alpha = nn.Parameter(torch.logit(torch.full((hidden_dim,), 0.9)))

    def forward(self, u: torch.Tensor):
        """Run the state-space block on input sequence u.

        Args:
            u: Tensor of shape (batch, seq_len, input_dim)

        Returns:
            y: Tensor of shape (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = u.shape
        device = u.device
        alpha = torch.sigmoid(self.logit_alpha).to(device)  # (hidden_dim,)

        h = torch.zeros(batch, self.hidden_dim, device=device)
        outputs = []
        Bu = self.B(u)  # (batch, seq_len, hidden_dim)
        for t in range(seq_len):
            # elementwise decay and input addition
            h = h * alpha + Bu[:, t, :]
            y = self.C(h)
            outputs.append(y.unsqueeze(1))

        return torch.cat(outputs, dim=1)
