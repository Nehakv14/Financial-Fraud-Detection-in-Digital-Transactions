from typing import Optional
import torch
import torch.nn as nn

class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network style layer.

    Approximates multivariate functions via sums of univariate functions of
    learned linear projections: f(x) = sum_j phi_j(w_j^T x) + b

    Each phi_j is implemented as a small 1D MLP (scalar input -> scalar output).
    """

    def __init__(self, in_dim: int, num_terms: int = 32, phi_hidden: int = 16, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_terms = num_terms

        # projection vectors w_j
        self.W = nn.Parameter(torch.randn(num_terms, in_dim) * 0.1)
        # small MLPs for phi_j implemented as a shared MLP applied to scalar projection
        self.phi = nn.Sequential(
            nn.Linear(1, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, 1),
        )

        # Final scaling per term
        self.term_scale = nn.Parameter(torch.ones(num_terms))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, features) or (batch, seq_len, features). Returns scalar output(s).
        """
        was_seq = False
        if x.dim() == 3:
            was_seq = True
            b, s, f = x.shape
            x = x.reshape(b * s, f)

        # projections: (batch*, num_terms)
        proj = x @ self.W.t()
        # apply phi elementwise by reshaping to vector of scalars
        out_terms = []
        for j in range(self.num_terms):
            z = proj[:, j : j + 1]  # (batch*,1)
            phi_z = self.phi(z) * self.term_scale[j]
            out_terms.append(phi_z)
        out = torch.sum(torch.cat(out_terms, dim=1), dim=1, keepdim=True)
        if self.bias is not None:
            out = out + self.bias

        if was_seq:
            out = out.view(b, s, 1)
        return out

    def interpret(self):
        """Return interpretable components: projection weights and phi params.

        The user can inspect `W` and `phi` weights for a simple explanation.
        """
        return {"W": self.W.detach().cpu().numpy(), "phi_weights": [p.detach().cpu().numpy() for p in self.phi.parameters()]}
