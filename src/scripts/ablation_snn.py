import argparse
import re
import sys
from pathlib import Path
import json
import subprocess
from typing import Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_npz", default="processed/sequences.npz")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default="ablation_results")
    return p.parse_args()


def run_config(config: Dict, output_dir: Path) -> Dict:
    """Run a single SNN training config and return metrics."""
    cmd = [
        sys.executable, "src/train/train_snn.py",
        "--data_npz", config["data_npz"],
        "--batch_size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--seed", str(config["seed"]),
        "--device", config["device"] or "",
        "--encoding", config["encoding"],
        "--T", str(config["T"]),
        "--beta", str(config["beta"]),
        "--surrogate", config["surrogate"],
        "--hidden_dims", *map(str, config["hidden_dims"]),
        "--save", str(output_dir / f"snn_{config['name']}.pt"),
    ]
    # Run subprocess and capture output (simplified: assume it prints metrics)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"train_snn failed: {result.stderr.strip() or result.stdout.strip()}")

    lines = [l.strip() for l in result.stdout.strip().split('\n') if 'Epoch' in l and 'val_auc' in l]
    if not lines:
        raise RuntimeError(f"No training metrics found. stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    last_line = lines[-1]
    metrics = {}
    for key, out_key in [("loss", "loss"), ("val_auc", "auc"), ("prec", "prec"), ("rec", "rec"), ("f1", "f1")]:
        match = re.search(rf"{key}=([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", last_line)
        if not match:
            raise RuntimeError(f"Missing metric '{key}' in line: {last_line}")
        metrics[out_key] = float(match.group(1))
    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define ablation grid
    encodings = ["rate", "latency"]
    Ts = [5, 10, 20, 50]
    surrogates = ["atan", "sigmoid"]
    betas = [0.8, 0.9, 0.95]
    hidden_dims_options = [[128, 64], [256, 128]]

    configs = []
    for enc in encodings:
        for T in Ts:
            for surr in surrogates:
                for beta in betas:
                    for hd in hidden_dims_options:
                        name = f"{enc}_T{T}_{surr}_b{beta}_hd{'_'.join(map(str, hd))}"
                        configs.append({
                            "name": name,
                            "data_npz": args.data_npz,
                            "batch_size": args.batch_size,
                            "epochs": args.epochs,
                            "lr": args.lr,
                            "seed": args.seed,
                            "device": args.device,
                            "encoding": enc,
                            "T": T,
                            "beta": beta,
                            "surrogate": surr,
                            "hidden_dims": hd,
                        })

    results = []
    for i, config in enumerate(configs):
        print(f"Running config {i+1}/{len(configs)}: {config['name']}")
        try:
            metrics = run_config(config, output_dir)
            result = {**config, **metrics}
            results.append(result)
        except Exception as e:
            print(f"Failed {config['name']}: {e}")
            results.append({**config, "error": str(e)})

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ablation complete. Results saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()