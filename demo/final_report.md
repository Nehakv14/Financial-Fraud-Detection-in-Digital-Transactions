# FFDS — Final Demo Report

Date: 2026-08-27
Presenter: Your Name

---

## Executive Summary

- Purpose: compare an interpretable sequential model (Mamba-KAN) with a spiking neural network (SNN) on preprocessed IEEE-style transaction sequences.
- Outcome (small test example): both models achieved perfect AUC on the small validation split, but the Mamba-KAN shows stronger precision/F1 in this run while the SNN exhibits lower precision despite perfect recall.

Key numbers (from `benchmark_results/benchmark_results.json`):

- Mamba-KAN: AUC=1.00, Precision=1.00, Recall=1.00, F1=1.00, inference ~1.29 ms/sample, ~18.5k FLOPs
- SNN (best from ablation): AUC=1.00, Precision=0.33, Recall=1.00, F1=0.50, inference ~0.52 ms/sample, ~348k SOPs

Notes: the evaluation set is small (example run used 16 total samples), so metrics should be interpreted cautiously; use cross-validation / larger hold-out for production claims.

---

## Data & Preprocessing

- Source: IEEE-CIS-style transaction CSVs (preprocessing script: `src/scripts/preprocess.py`).
- Processed sequences saved to `processed/sequences.npz` (example shape: 16 samples × 50 timesteps × 11 features).
- Encodings used for SNN: rate encoding (`src/data/spike_encoding.py`) and latency encoding (also available).

---

## Models and Training

- Mamba-KAN: implemented in `src/models/mamba_kan_model.py`; trained with `src/train/train_mamba.py`; checkpoint: `checkpoints/mamba_kan.pt`.
- SNN: LIF-based multi-layer model in `src/models/snn_model.py`; trained with `src/train/train_snn.py`.
- Best SNN config from ablation (used for final retrain): encoding=`rate`, T=50 (seq_len), surrogate=`atan`, beta=0.8, hidden_dims=[256,128]. Checkpoint: `checkpoints/snn_best_from_ablation.pt` (also copied to `checkpoints/snn.pt` for evaluation).

Retrain command used (example):

```powershell
.venv\Scripts\python.exe src/train/train_snn.py --data_npz processed/sequences.npz --encoding rate --T 50 --surrogate atan --beta 0.8 --hidden_dims 256 128 --epochs 10 --batch_size 64 --lr 0.001 --save checkpoints/snn_best_from_ablation.pt
```

---

## Ablation & Plots

- Ablation grid and heatmap: `benchmark_results/ablation_plots.png` (show during demo).
- KAN term importances: `benchmark_results/kan_importance.png` (interpretability example).

Refer to `benchmark_results/` for plots and the numeric report `benchmark_results/benchmark_results.json`.

---

## Benchmark Results (detailed)

Contents of `benchmark_results/benchmark_results.json` (key fields):

```json
{
  "mamba_kan": {"auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "inference_time_us": 1290.21, "ops": 18500},
  "snn": {"auc": 1.0, "precision": 0.33, "recall": 1.0, "f1": 0.5, "inference_time_us": 520.58, "ops": 348162}
}
```

Observations:

- Both models reached perfect AUC on this small example, but the Mamba-KAN had perfect precision and F1, while the SNN traded precision for recall (many false positives in this split).
- SNN inference was faster per-sample in this measurement, but its synaptic operation count (SOPs) is much higher than Mamba's FLOPs — interpretability and compute trade-offs matter.

---

## Demo Plan (10 minutes)

1. Show preprocessing output and shape (`processed/sequences.npz`).
2. Show ablation heatmap (`benchmark_results/ablation_plots.png`).
3. Show KAN importance (`benchmark_results/kan_importance.png`) and explain interpretability.
4. Show numeric benchmarks (`benchmark_results/benchmark_results.json`).
5. Live: run a short inference using `checkpoints/mamba_kan.pt` or show `checkpoints/snn_best_from_ablation.pt` details.

Short commands to run during demo:

```powershell
# Open plots
Start-Process benchmark_results\ablation_plots.png
Start-Process benchmark_results\kan_importance.png

# Print metrics
Get-Content benchmark_results\benchmark_results.json | Out-String -Stream

# Quick inference (Mamba)
.venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; import torch; arr=np.load('processed/sequences.npz'); x=arr['X'][0:1]; from src.eval.benchmark import load_model; m=load_model('checkpoints/mamba_kan.pt','mamba_kan', x.shape[-1]); import torch; print(m(torch.from_numpy(x).float()))"
```

---

## Caveats & Next Steps

- Small sample sizes in this run produce optimistic metrics; run stratified cross-validation and larger test sets.
- Calibrate SNN decision threshold or tune loss weighting to improve precision.
- Automate full report generation and PDF export; package artifacts for sharing.

---

## Artifacts

- `benchmark_results/benchmark_results.json`
- `benchmark_results/ablation_plots.png`
- `benchmark_results/kan_importance.png`
- `checkpoints/mamba_kan.pt`, `checkpoints/snn_best_from_ablation.pt`
- `demo/slide_deck.md`, `demo/demo_README.md`, `demo/demo_script.ps1`

---

If you want, I can convert this Markdown to a PDF and create a ZIP package with the artifacts now. Which would you like?
