# FFDS: Fraud Detection Research Stack

A comprehensive research stack comparing interpretable sequential models (**Mamba-KAN**) with energy-efficient spiking neural networks (**SNN**) for fraud detection on transaction sequences.

## Overview

This project preprocesses financial transaction data, trains two distinct models, and performs extensive ablation studies to understand trade-offs between:
- **Interpretability**: KAN (Kolmogorov-Arnold Network) terms provide per-feature importance
- **Efficiency**: SNNs (Spiking Neural Networks) with rate/latency encoding reduce computational cost
- **Performance**: AUC, precision, recall, and inference latency

## Quick Start

### 1. Setup Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Preprocess Data

Convert your CSV to aligned transaction sequences:

```powershell
python src/scripts/preprocess.py `
  --input data/transactions.csv `
  --out processed/sequences.npz `
  --id_col card1 --time_col TransactionDT --label_col isFraud `
  --seq_len 50
```

Outputs: `processed/sequences.npz` (X: batch × seq_len × features; y: labels)

### 3. Train Models

**Mamba-KAN** (interpretable sequential):

```powershell
python src/train/train_mamba.py `
  --data_npz processed/sequences.npz `
  --epochs 20 --batch_size 64 --lr 1e-3
```

Checkpoint: `checkpoints/mamba_kan.pt`

**SNN** (spiking neural network, example config):

```powershell
python src/train/train_snn.py `
  --data_npz processed/sequences.npz `
  --encoding rate --T 50 --surrogate atan --beta 0.8 `
  --hidden_dims 256 128 --epochs 10 --batch_size 64 --lr 0.001 `
  --save checkpoints/snn_best.pt
```

### 4. Run Ablation & Benchmark

Ablation study (tests 96 SNN configs):

```powershell
python src/scripts/ablation_snn.py `
  --data_npz processed/sequences.npz `
  --epochs 10 --batch_size 64 --output_dir ablation_results
```

Generate metrics & plots:

```powershell
python src/eval/generate_metrics.py
python src/eval/benchmark.py
python src/eval/plot_ablation.py --results ablation_results/ablation_results.json --out_dir benchmark_results
python src/eval/plot_kan_importance.py --model checkpoints/mamba_kan.pt --out_dir benchmark_results
```

### 5. Demo

Run the full demo script (opens plots, prints metrics):

```powershell
powershell -ExecutionPolicy Bypass -File .\demo\demo_script.ps1
```

Or run individual pieces:

```powershell
# Show plots
Start-Process benchmark_results\ablation_plots.png
Start-Process benchmark_results\kan_importance.png

# Print metrics
Get-Content benchmark_results\benchmark_results.json | Out-String -Stream
```

## Project Structure

```
FFDS/
├── src/
│   ├── data/
│   │   ├── loader.py              # TabularSequenceDataset
│   │   └── spike_encoding.py      # Rate & latency encodings
│   ├── models/
│   │   ├── kan.py                 # KAN layers
│   │   ├── mamba.py               # Mamba sequential encoder
│   │   ├── mamba_kan_model.py     # Full Mamba-KAN classifier
│   │   └── snn_model.py           # LIF-based SNN classifier
│   ├── train/
│   │   ├── train_mamba.py         # Mamba-KAN training
│   │   └── train_snn.py           # SNN training
│   ├── eval/
│   │   ├── benchmark.py           # Model comparison & plots
│   │   ├── plot_ablation.py       # Ablation heatmaps
│   │   ├── plot_kan_importance.py # Interpretability plots
│   │   └── generate_metrics.py    # Compute final metrics JSON
│   └── scripts/
│       ├── preprocess.py          # Data preprocessing
│       └── ablation_snn.py        # Full ablation harness
├── demo/
│   ├── slide_deck.md              # 6-slide presentation
│   ├── demo_README.md             # Quick reference
│   ├── demo_script.ps1            # Full demo script
│   └── final_report.md            # Detailed technical report
├── checkpoints/                   # Trained model checkpoints
├── processed/                     # Preprocessed sequences (NPZ)
├── benchmark_results/             # Plots & metrics JSON
├── ablation_results/              # Ablation configs & results
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Results Summary

**Example run metrics** (`benchmark_results/benchmark_results.json`):

| Model | AUC | Precision | Recall | F1 | Latency (ms/sample) | Ops |
|-------|-----|-----------|--------|----|--------------------|-----|
| Mamba-KAN | 1.00 | 1.00 | 1.00 | 1.00 | 1.83 | 18.5k FLOPs |
| SNN (best) | 1.00 | 0.33 | 1.00 | 0.50 | 0.70 | 348k SOPs |

*Note: Metrics are on a small validation split (example run). Use stratified cross-validation and larger hold-out sets for production claims.*

## Key Features

- **Data Pipeline**: CSV → sequences with windowing, imputation, scaling, one-hot encoding
- **Spike Encodings**: Rate coding (Bernoulli spikes) and latency coding (single spike per feature)
- **Mamba-KAN**: Sequential + attention + interpretable KAN layers for transparency
- **SNN**: Leaky Integrate-and-Fire neurons with surrogate gradients (atan, sigmoid, fast_sigmoid)
- **Ablation**: Systematic grid search over encodings, T, surrogates, betas, and hidden dimensions
- **Evaluation**: ROC curves, confusion matrices, inference time, operation counts, feature importance

## Configuration

All scripts support CLI arguments. Key parameters:

- `--data_npz`: path to preprocessed sequences
- `--encoding`: `rate` or `latency` for SNN
- `--T`: number of time steps for spike encoding
- `--surrogate`: surrogate gradient function (`atan`, `sigmoid`, `fast_sigmoid`)
- `--beta`: leak factor for LIF neurons (0.0–1.0)
- `--hidden_dims`: list of hidden layer sizes
- `--epochs`, `--batch_size`, `--lr`: training hyperparameters

## Demo & Presentation

For the 4-person team demo, assign roles:

- **Member 1 — Presenter & Overview**: Lead the presentation using `demo/slide_deck.md`. Orchestrate the flow and speak to motivation.
- **Member 2 — Data & Preprocessing**: Show `processed/sequences.npz` shape, explain spike encodings (`src/data/spike_encoding.py`), run a quick preprocess.
- **Member 3 — Models & Training**: Explain Mamba-KAN and SNN architectures (code in `src/models/`), show checkpoints, optionally re-run a short training.
- **Member 4 — Evaluation & Visuals**: Present `benchmark_results/ablation_plots.png`, `kan_importance.png`, and metrics from `benchmark_results/benchmark_results.json`.

**Demo files:**
- `demo/slide_deck.md` — 6-slide presentation outline
- `demo/demo_README.md` — Quick reference & commands
- `demo/demo_script.ps1` — Full automated demo (opens plots, prints metrics)
- `demo/final_report.md` — Detailed technical report

**Run the full demo:**
```powershell
powershell -ExecutionPolicy Bypass -File .\demo\demo_script.ps1
```

## Caveats & Next Steps

- **Small sample sizes**: Example runs use small validation splits; use stratified cross-validation and hold-out test sets for production.
- **Interpretability**: Load Mamba-KAN checkpoint and call `model.kan_layers()` to inspect term weights.
- **Future work**: Calibration, threshold optimization, larger-scale evaluation, FPGA deployment for SNNs.

## References

- **Kolmogorov-Arnold Networks (KAN)**: Explainable and efficient function approximation.
- **Spiking Neural Networks (SNNs)**: Event-driven computation for energy efficiency.
- **snntorch**: Surrogate gradient framework for training SNNs with backprop.
- **IEEE-CIS Fraud Detection**: Kaggle dataset used for testing.

## License

MIT License — see LICENSE file (if included).

## Contact

For questions, open an issue or contact the research team.

---

Last updated: 2026-09-12
