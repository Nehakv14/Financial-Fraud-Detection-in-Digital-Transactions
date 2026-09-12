import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_ablation_results(results_path: str, save_path: Path):
    """Plot ablation study results."""
    with open(results_path, "r") as f:
        results = json.load(f)

    # Filter successful runs
    data = [r for r in results if "error" not in r]

    df = pd.DataFrame(data)

    # Plot F1 vs T for different encodings
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # F1 vs T
    for enc in df["encoding"].unique():
        sub = df[df["encoding"] == enc]
        axes[0, 0].plot(sub["T"], sub["f1"], marker="o", label=enc)
    axes[0, 0].set_xlabel("T (time steps)")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 vs Time Steps")
    axes[0, 0].legend()

    # F1 vs beta
    for surr in df["surrogate"].unique():
        sub = df[df["surrogate"] == surr]
        axes[0, 1].plot(sub["beta"], sub["f1"], marker="o", label=surr)
    axes[0, 1].set_xlabel("Beta (decay)")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].set_title("F1 vs Beta")
    axes[0, 1].legend()

    # AUC vs T
    for enc in df["encoding"].unique():
        sub = df[df["encoding"] == enc]
        axes[1, 0].plot(sub["T"], sub["auc"], marker="o", label=enc)
    axes[1, 0].set_xlabel("T (time steps)")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_title("AUC vs Time Steps")
    axes[1, 0].legend()

    # Heatmap: F1 by T and beta (average over others)
    pivot = df.pivot_table(values="f1", index="T", columns="beta", aggfunc="mean")
    sns.heatmap(pivot, annot=True, ax=axes[1, 1], cmap="viridis")
    axes[1, 1].set_title("F1 Heatmap (T vs Beta)")

    plt.tight_layout()
    plt.savefig(save_path / "ablation_plots.png")
    plt.close()


def main():
    results_path = "ablation_results/ablation_results.json"
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation_results(results_path, output_dir)
    print("Ablation plots saved.")


if __name__ == "__main__":
    main()