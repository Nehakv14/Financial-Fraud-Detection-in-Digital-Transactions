Demo README — FFDS (one page)

Purpose
- Quick reference for the 10-minute demo comparing Mamba-KAN and SNN.

Files to show
- `processed/sequences.npz` — processed sequences
- `benchmark_results/ablation_plots.png` — ablation grid heatmap
- `benchmark_results/kan_importance.png` — KAN feature importance
- `benchmark_results/benchmark_results.json` — numeric metrics
- `checkpoints/snn_best_from_ablation.pt` and `checkpoints/mamba_kan.pt` — trained models

Key commands (PowerShell)
```powershell
# Open plots
Start-Process benchmark_results\ablation_plots.png
Start-Process benchmark_results\kan_importance.png

# Show metrics
Get-Content benchmark_results\benchmark_results.json | Out-String -Stream

# Re-run benchmark to reproduce comparison plots
.venv\Scripts\python.exe src/eval/benchmark.py

# Retrain best SNN (if you want to show training)
.venv\Scripts\python.exe src/train/train_snn.py --data_npz processed/sequences.npz --encoding rate --T 50 --surrogate atan --beta 0.8 --hidden_dims 256 128 --epochs 10 --batch_size 64 --lr 0.001 --save checkpoints/snn_best_from_ablation.pt
```

Talking points
- Explain preprocessing and encoding choices briefly.
- Highlight interpretability: `Mamba-KAN` gives term importances (show `kan_importance.png`).
- Performance: mention speed and ops differences; caution on small evaluation set.
- Next steps: cross-validation, larger test set, calibration.

Notes for presenter
- Use the slide deck `demo/slide_deck.md` as speaking notes.
- Keep live runs short (use precomputed plots and checkpoints).
