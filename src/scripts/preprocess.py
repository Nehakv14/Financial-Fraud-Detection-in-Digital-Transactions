import argparse
import os
import sys
from pathlib import Path
import joblib
import numpy as np

# Ensure project root is on sys.path so `from src...` imports work when running script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import set_seed
from src.data.loader import TabularSequenceDataset
from src.data.spike_encoding import rate_encode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to transactions CSV")
    p.add_argument("--id_col", default="account_id")
    p.add_argument("--time_col", default="transaction_time")
    p.add_argument("--label_col", default="is_fraud")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--categorical_cols", nargs="*", default=None, help="Comma-separated or space-separated categorical column names")
    p.add_argument("--smote", action="store_true")
    p.add_argument("--out_dir", default="processed")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categorical_cols = []
    if args.categorical_cols:
        for item in args.categorical_cols:
            categorical_cols.extend([col.strip() for col in item.split(",") if col.strip()])

    ds = TabularSequenceDataset(
        csv_path=args.csv,
        id_col=args.id_col,
        time_col=args.time_col,
        seq_len=args.seq_len,
        step=args.step,
        categorical_cols=categorical_cols,
        label_col=args.label_col,
        smote=args.smote,
        seed=args.seed,
    )

    # Save arrays
    X = ds.X_windows
    y = ds.y
    np.savez_compressed(out_dir / "sequences.npz", X=X, y=y)

    # Save transforms/scalers
    joblib.dump(ds.num_imputer, out_dir / "num_imputer.joblib")
    if ds.scaler is not None:
        joblib.dump(ds.scaler, out_dir / "scaler.joblib")
    if ds.ohe is not None:
        joblib.dump(ds.ohe, out_dir / "ohe.joblib")

    print(f"Saved sequences to {out_dir / 'sequences.npz'}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Generate a small example spike encoding (rate) for the first 32 sequences
    sample = X[: min(32, len(X))]
    spikes_tensor, spikes_np = rate_encode(sample, T=20, rng=args.seed)
    np.savez_compressed(out_dir / "spikes_rate_T20.npz", spikes=spikes_np)
    print(f"Saved sample rate-encoded spikes to {out_dir / 'spikes_rate_T20.npz'}")


if __name__ == "__main__":
    main()
