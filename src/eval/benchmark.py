import sys
import time
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_device
from src.models.mamba_kan_model import MambaKANClassifier
from src.models.snn_model import SNNClassifier
from src.data.spike_encoding import rate_encode, to_snn_input_format


def load_model(model_path: str, model_type: str, input_dim: int = None):
    """Load a trained model."""
    device = get_device()
    if model_type == "mamba_kan":
        model = MambaKANClassifier(input_dim=input_dim, seq_hidden=64, kan_terms=64)
    elif model_type == "snn":
        model = SNNClassifier(input_dim=input_dim, hidden_dims=[128, 64])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def benchmark_inference_time(model, X_test, num_runs: int = 100):
    """Benchmark average inference time per sample."""
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    model.eval()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(X_test)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.time()
            times.append((end - start) / X_test.size(0))  # per sample

    avg_time_us = np.mean(times) * 1e6  # microseconds
    return avg_time_us


def estimate_ops(model, X_sample):
    """Rough estimate of FLOPs for Mamba-KAN or SOPs for SNN."""
    # Placeholder: count parameters and assume ops per param
    total_params = sum(p.numel() for p in model.parameters())
    if isinstance(model, SNNClassifier):
        # SOPs: synaptic operations (spikes * synapses)
        # Rough: assume average spike rate 0.1, T=20
        T = 20  # from config
        spike_rate = 0.1
        ops = total_params * spike_rate * T
        return ops, "SOPs"
    else:
        # FLOPs: floating point ops, rough estimate
        ops = total_params * 2  # multiply-add
        return ops, "FLOPs"


def evaluate_model(model, X_test, y_test, model_type: str):
    """Compute metrics and return dict."""
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    model.eval()

    with torch.no_grad():
        if model_type == "snn":
            # For SNN, encode to spikes
            spikes_tensor, _ = rate_encode(X_test.cpu().numpy(), T=20, rng=42)
            spikes_tensor = to_snn_input_format(spikes_tensor).to(device)
            logits = model(spikes_tensor)
        else:
            logits = model(X_test)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, preds)

    fpr, tpr, _ = roc_curve(y_test, probs)

    return {
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
    }


def plot_comparison(mamba_metrics, snn_metrics, save_path: Path):
    """Plot ROC curves and confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC
    axes[0].plot(mamba_metrics["roc_curve"][0], mamba_metrics["roc_curve"][1], label=f"Mamba-KAN (AUC={mamba_metrics['auc']:.3f})")
    axes[0].plot(snn_metrics["roc_curve"][0], snn_metrics["roc_curve"][1], label=f"SNN (AUC={snn_metrics['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    # Confusion matrices
    sns.heatmap(mamba_metrics["confusion_matrix"], annot=True, fmt="d", ax=axes[1], cmap="Blues")
    axes[1].set_title("Mamba-KAN Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    sns.heatmap(snn_metrics["confusion_matrix"], annot=True, fmt="d", ax=axes[2], cmap="Blues")
    axes[2].set_title("SNN Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path / "comparison_plots.png")
    plt.close()


def main():
    # Example usage: compare best models
    data_npz = "processed/sequences.npz"
    mamba_path = "checkpoints/mamba_kan.pt"
    snn_path = "checkpoints/snn.pt"
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    data = np.load(data_npz)
    X = data["X"]
    y = data["y"]
    # Simple split (use same as training)
    n = len(y)
    split = int(0.8 * n)
    X_test, y_test = X[split:], y[split:]

    X_test_t = torch.from_numpy(X_test).float()

    # Load models
    input_dim = X.shape[-1]
    mamba_model = load_model(mamba_path, "mamba_kan", input_dim)
    snn_model = load_model(snn_path, "snn", input_dim * 50)  # seq_len=50 flattened

    # Evaluate
    mamba_metrics = evaluate_model(mamba_model, X_test_t, y_test, "mamba_kan")
    snn_metrics = evaluate_model(snn_model, X_test_t, y_test, "snn")

    # Benchmark latency
    mamba_time = benchmark_inference_time(mamba_model, X_test_t[:32])  # small batch
    snn_time = benchmark_inference_time(snn_model, X_test_t[:32])

    # Estimate ops
    mamba_ops, mamba_unit = estimate_ops(mamba_model, X_test_t[:1])
    snn_ops, snn_unit = estimate_ops(snn_model, X_test_t[:1])

    # Save results
    results = {
        "mamba_kan": {**mamba_metrics, "inference_time_us": mamba_time, "ops": mamba_ops, "ops_unit": mamba_unit},
        "snn": {**snn_metrics, "inference_time_us": snn_time, "ops": snn_ops, "ops_unit": snn_unit},
    }

    import json
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)  # str for numpy arrays

    # Plot
    plot_comparison(mamba_metrics, snn_metrics, output_dir)

    print("Benchmarking complete. Results in benchmark_results/")