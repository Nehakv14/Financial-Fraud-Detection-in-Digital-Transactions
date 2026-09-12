import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mamba_kan_model import MambaKANClassifier


def plot_kan_weights(model, save_path: Path):
    """Plot KAN projection weights and phi parameter norms."""
    interp = model.interpret()
    W = interp["W"]  # (num_terms, in_dim)
    phi_weights = interp["phi_weights"]  # list of tensors

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Projection weights heatmap
    im = axes[0].imshow(W.T, aspect="auto", cmap="viridis")
    axes[0].set_title("KAN Projection Weights (W)")
    axes[0].set_xlabel("Terms")
    axes[0].set_ylabel("Features")
    plt.colorbar(im, ax=axes[0])

    # Phi parameter norms
    phi_norms = [np.linalg.norm(p.flatten()) for p in phi_weights]
    axes[1].bar(range(len(phi_norms)), phi_norms)
    axes[1].set_title("Phi Parameter Norms")
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Norm")

    plt.tight_layout()
    plt.savefig(save_path / "kan_importance.png")
    plt.close()


def main():
    model_path = "checkpoints/mamba_kan.pt"
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer input_dim from processed data
    data = np.load("processed/sequences.npz")
    input_dim = data["X"].shape[-1]

    # Load model
    checkpoint = torch.load(model_path)
    model = MambaKANClassifier(input_dim=input_dim, seq_hidden=64, kan_terms=64)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    plot_kan_weights(model, output_dir)
    print("KAN importance plot saved.")


if __name__ == "__main__":
    main()