# Slide 1 — Title

- **Project:** FFDS — Fraud FDS (IEEE-CIS experiments)
- **Presenter:** Your Name
- **Date:** Tomorrow

---

# Slide 2 — Dataset & Preprocessing

- **Dataset:** IEEE-CIS transaction data (processed sequences)
- **Preprocessing:** `src/scripts/preprocess.py` -> `processed/sequences.npz`
- **Sequence shape:** 16 samples × 50 timesteps × 11 features (example run)
- Key point: sequences prepared for SNN rate/latency encoding.

---

# Slide 3 — Models

- **Mamba-KAN:** interpretable sequential classifier (KAN terms)
- **SNN:** LIF-based Spiking Neural Network (rate-coded input)
- Training scripts: `src/train/train_mamba.py`, `src/train/train_snn.py`

---

# Slide 4 — Ablation & Best Config

- Ablation tested encodings, T, surrogate, beta, hidden dims.
- Top SNN configuration found: `rate`, `T=50`, `surrogate=atan`, `beta=0.8`, `hidden_dims=[256,128]`.
- Retrain command used:

```powershell
.venv\Scripts\python.exe src/train/train_snn.py --data_npz processed/sequences.npz --encoding rate --T 50 --surrogate atan --beta 0.8 --hidden_dims 256 128 --epochs 10 --batch_size 64 --lr 0.001 --save checkpoints/snn_best_from_ablation.pt
```

---

# Slide 5 — Benchmark Summary (key numbers)

- Source: `benchmark_results/benchmark_results.json`
- `Mamba-KAN`: AUC=1.00, Precision=1.00, Recall=1.00, F1=1.00
- `SNN` (best): AUC=1.00, Precision=0.33, Recall=1.00, F1=0.50
- Latency: Mamba ~1.29 ms/sample, SNN ~0.52 ms/sample (measured)
- Ops: Mamba ~18.5k FLOPs, SNN ~348k SOPs

---

# Slide 6 — Demo Flow & Talking Points

- 1) Show preprocessing output (`processed/sequences.npz`) and shapes.
- 2) Show ablation plot: `benchmark_results/ablation_plots.png`.
- 3) Show KAN importance: `benchmark_results/kan_importance.png`.
- 4) Show benchmark metrics (`benchmark_results/benchmark_results.json`) and `benchmark_results/comparison_plots.png`.
- 5) Live demo: run a quick inference or open `checkpoints/snn_best_from_ablation.pt` details.
- Talking points: interpretability of KAN, SNN speed vs ops, robustness caveats (small eval set), next steps.
