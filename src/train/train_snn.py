import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import set_seed, get_device
from src.models.snn_model import SNNClassifier
from src.data.spike_encoding import rate_encode, latency_encode, to_snn_input_format


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_npz", default="processed/sequences.npz")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--save", default="checkpoints/snn.pt")
    # SNN-specific
    p.add_argument("--encoding", choices=["rate", "latency"], default="rate")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--surrogate", choices=["atan", "sigmoid", "fast_sigmoid"], default="atan")
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 64])
    return p.parse_args()


def load_npz(path: str):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    return X, y


def encode_data(X: np.ndarray, encoding: str, T: int, seed: int):
    """Encode numpy sequences to spike tensors."""
    if encoding == "rate":
        _, spikes_np = rate_encode(X, T=T, rng=seed)
    elif encoding == "latency":
        _, spikes_np = latency_encode(X, T=T)
    spikes_tensor = to_snn_input_format(torch.from_numpy(spikes_np))
    return spikes_tensor


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    X, y = load_npz(args.data_npz)
    # Simple train/val split
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Encode to spikes in time-major format: (T, batch, input_dim)
    X_train_spikes = encode_data(X_train, args.encoding, args.T, args.seed)
    X_val_spikes = encode_data(X_val, args.encoding, args.T, args.seed)

    # Convert to batch-first tensors for TensorDataset: (batch, T, input_dim)
    X_train_spikes = X_train_spikes.permute(1, 0, 2)
    X_val_spikes = X_val_spikes.permute(1, 0, 2)

    y_train_t = torch.from_numpy(y_train).float()
    y_val_t = torch.from_numpy(y_val).float()

    train_ds = torch.utils.data.TensorDataset(X_train_spikes, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_spikes, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train_spikes.shape[-1]  # seq_len * features
    model = SNNClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        beta=args.beta,
        surrogate_function=args.surrogate,
    ).to(device)

    pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_acc = 0.0
        for xb, yb in train_loader:
            xb = xb.permute(1, 0, 2).to(device)  # (T, batch, input_dim)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            loss_acc += loss.item() * xb.size(1)  # batch size

        loss_acc /= len(train_loader.dataset)

        # Validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.permute(1, 0, 2).to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist())
                trues.extend(yb.cpu().numpy().tolist())

        preds = np.array(preds)
        trues = np.array(trues)
        auc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else 0.0
        yhat = (preds > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(trues, yhat, average="binary", zero_division=0)
        cm = confusion_matrix(trues, yhat)

        print(f"Epoch {epoch} loss={loss_acc:.4f} val_auc={auc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        print("Confusion matrix:\n", cm)

        if auc > best_auc:
            best_auc = auc
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.save)
            print(f"Saved best model to {args.save}")


if __name__ == "__main__":
    main()