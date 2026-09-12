# PowerShell demo script — FFDS

# 1. Show ablation and KAN importance plots
Start-Process benchmark_results\ablation_plots.png
Start-Process benchmark_results\kan_importance.png

# 2. Print benchmark metrics
Write-Host "\n--- Benchmark results ---\n"
Get-Content benchmark_results\benchmark_results.json | Write-Host

# 3. Optional: show comparison plots (regenerate if needed)
.venv\Scripts\python.exe src/eval/benchmark.py
Start-Process benchmark_results\comparison_plots.png

# 4. Quick inference example (run a single-batch prediction)
.venv\Scripts\python.exe - <<'PY'
import torch, numpy as np
from pathlib import Path
from src.eval.benchmark import load_model

p = Path('processed/sequences.npz')
arr = np.load(p)
X = arr['X']
# take first sample
x = X[0:1]
# prepare for Mamba
x_t = torch.from_numpy(x).float()
model = load_model('checkpoints/mamba_kan.pt','mamba_kan', x.shape[-1])
with torch.no_grad():
    out = model(x_t)
print('Mamba logit:', out)
PY

Write-Host "Demo script finished."