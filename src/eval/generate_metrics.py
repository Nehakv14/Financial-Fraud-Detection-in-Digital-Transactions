import sys
from pathlib import Path
import json
import numpy as np
import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark import load_model, evaluate_model, estimate_ops, benchmark_inference_time
from src.models.snn_model import SNNClassifier
from src.models.mamba_kan_model import MambaKANClassifier


def main():
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    data_npz = Path("processed/sequences.npz")
    if not data_npz.exists():
        print(f"Data file not found: {data_npz}")
        return 1

    data = np.load(data_npz)
    X = data["X"]
    y = data["y"]
    n = len(y)
    split = int(0.8 * n)
    X_test, y_test = X[split:], y[split:]

    X_test_t = torch.from_numpy(X_test).float()

    mamba_path = Path("checkpoints/mamba_kan.pt")
    snn_path = Path("checkpoints/snn.pt")

    if not mamba_path.exists() or not snn_path.exists():
        print("Missing model checkpoints. Expected: checkpoints/mamba_kan.pt and checkpoints/snn.pt")
        return 1

    input_dim = X.shape[-1]
    # Load Mamba-KAN using helper
    mamba_model = load_model(str(mamba_path), "mamba_kan", input_dim)
    # Construct SNN with hidden dims matching the trained checkpoint.
    # Determine SNN time dimension from the data (seq_len) so shapes align with checkpoints.
    seq_len = X.shape[1]
    T_snn = seq_len
    snn_input_dim = input_dim * T_snn
    snn_model = SNNClassifier(input_dim=snn_input_dim, hidden_dims=[256, 128])
    checkpoint = torch.load(str(snn_path), map_location=next(mamba_model.parameters()).device)
    snn_model.load_state_dict(checkpoint["model_state"])
    snn_model.to(next(mamba_model.parameters()).device)

    # Evaluate Mamba-KAN using the helper (expects raw X)
    mamba_metrics = evaluate_model(mamba_model, X_test_t, y_test, "mamba_kan")

    # For SNN, encode with the same T used in training and evaluate
    from src.data.spike_encoding import rate_encode, to_snn_input_format

    spikes_tensor, _ = rate_encode(X_test, T=T_snn, rng=42)
    spikes_tensor = to_snn_input_format(spikes_tensor).to(next(mamba_model.parameters()).device)

    # Run SNN inference
    with torch.no_grad():
        logits = snn_model(spikes_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
    snn_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)

    snn_metrics = {"auc": snn_auc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm, "roc_curve": (fpr, tpr)}

    mamba_time = benchmark_inference_time(mamba_model, X_test_t[:32])

    # Benchmark SNN inference time on the pre-encoded spikes
    import time
    device = next(snn_model.parameters()).device
    spikes_tensor = spikes_tensor.to(device)
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = snn_model(spikes_tensor[:32])
            end = time.time()
            times.append((end - start) / spikes_tensor[:32].size(0))
    snn_time = float(np.mean(times) * 1e6)

    mamba_ops, mamba_unit = estimate_ops(mamba_model, X_test_t[:1])
    snn_ops, snn_unit = estimate_ops(snn_model, X_test_t[:1])

    results = {
        "mamba_kan": {**mamba_metrics, "inference_time_us": float(mamba_time), "ops": float(mamba_ops), "ops_unit": mamba_unit},
        "snn": {**snn_metrics, "inference_time_us": float(snn_time), "ops": float(snn_ops), "ops_unit": snn_unit},
    }

    # Convert numpy arrays to lists for JSON serialization
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    def make_serializable(d):
        if isinstance(d, dict):
            return {k: make_serializable(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [make_serializable(v) for v in d]
        return _serialize(d)

    serializable_results = make_serializable(results)

    out_path = output_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Wrote metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
