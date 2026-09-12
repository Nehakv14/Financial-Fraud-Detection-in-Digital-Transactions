import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNClassifier(nn.Module):
    """Multi-layer Leaky Integrate-and-Fire (LIF) SNN classifier.

    Designed for spike-encoded inputs: (T, batch, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        output_dim: int = 1,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_function: str = "atan",
        reset_mechanism: str = "subtract",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.beta = beta
        self.threshold = threshold

        # Choose surrogate gradient function
        if surrogate_function == "atan":
            self.surrogate = surrogate.atan()
        elif surrogate_function == "sigmoid":
            self.surrogate = surrogate.sigmoid()
        elif surrogate_function == "fast_sigmoid":
            self.surrogate = surrogate.fast_sigmoid()
        else:
            raise ValueError(f"Unknown surrogate: {surrogate_function}")

        # Build layers: Linear + LIF for each hidden layer
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:  # Hidden layers get LIF
                lif = snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    spike_grad=self.surrogate,
                    reset_mechanism=reset_mechanism,
                )
                layers.append(lif)
            else:  # Output layer: no LIF, just linear for logits
                pass
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, batch, input_dim) -> logits (batch, output_dim)
        """
        T, batch, _ = x.shape
        mem_rec = []  # Optional: record membrane potentials for analysis

        # Initialize membrane potentials for LIF layers
        mems = [torch.zeros(batch, dim, device=x.device) for dim in self.hidden_dims]

        # Process each time step
        for t in range(T):
            x_t = x[t]  # (batch, input_dim)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    x_t = layer(x_t)
                elif isinstance(layer, snn.Leaky):
                    # LIF layer: update membrane and get spike
                    spk, mem = layer(x_t, mems[i // 2])  # i//2 since Linear + LIF pairs
                    mems[i // 2] = mem
                    x_t = spk
                    if i == len(self.layers) - 2:  # Last LIF before output
                        mem_rec.append(mem.detach())
            # After all layers, accumulate output (sum over time for classification)
            if t == 0:
                out = x_t.unsqueeze(0)
            else:
                out = out + x_t.unsqueeze(0)

        # Average over time steps for final logit
        logits = out.mean(dim=0)  # (batch, output_dim)
        return logits.squeeze(-1) if self.output_dim == 1 else logits

    def get_spike_counts(self):
        """Optional: return spike counts from LIF layers (for analysis).
        """
        # This would require storing spike counts during forward; placeholder.
        return None