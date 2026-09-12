"""Spike encoding utilities: Rate and Latency coding for SNN input.

Functions expect continuous inputs normalized to [0, 1].
Output shapes are documented per function.
"""
from typing import Tuple
import numpy as np
import torch


def rate_encode(
    X: np.ndarray,
    T: int = 20,
    rng: int = 42,
    return_type: str = "tensor",
) -> Tuple[torch.Tensor, np.ndarray]:
    """Convert normalized inputs to Bernoulli spike trains (Rate coding).

    Args:
        X: numpy array of shape (batch, seq_len, num_features) with values in [0,1]
        T: number of time steps
        rng: random seed
        return_type: 'tensor' or 'numpy'

    Returns:
        spikes: tensor of shape (batch, seq_len, T, num_features) of 0/1
    """
    np.random.seed(rng)
    batch, seq_len, num_features = X.shape
    probs = np.expand_dims(X, axis=2)  # (batch, seq_len, 1, features)
    probs = np.repeat(probs, T, axis=2)  # (batch, seq_len, T, features)
    rand = np.random.rand(batch, seq_len, T, num_features)
    spikes = (rand < probs).astype(np.float32)
    if return_type == "tensor":
        return torch.from_numpy(spikes), spikes
    return spikes


def latency_encode(
    X: np.ndarray,
    T: int = 20,
    jitter: float = 0.0,
    return_type: str = "tensor",
) -> Tuple[torch.Tensor, np.ndarray]:
    """Latency coding: single spike per feature per transaction, earlier for larger values.

    Args:
        X: shape (batch, seq_len, num_features) normalized to [0,1]
        T: number of timesteps
        jitter: fraction of time step to jitter spike time (adds randomness)

    Returns:
        spikes: (batch, seq_len, T, num_features)
    """
    batch, seq_len, num_features = X.shape
    spikes = np.zeros((batch, seq_len, T, num_features), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            vals = X[b, s]
            vals = np.clip(vals, 0.0, 1.0)
            # Map value 1.0 -> time 0 (earliest), 0.0 -> time T-1 (latest)
            times = np.floor((1.0 - vals) * (T - 1)).astype(int)
            times = np.clip(times, 0, T - 1)
            if jitter > 0.0:
                jitter_samples = np.random.randint(-int(jitter * T), int(jitter * T) + 1, size=times.shape)
                times = np.clip(times + jitter_samples, 0, T - 1)
            for f, t in enumerate(times):
                spikes[b, s, t, f] = 1.0

    if return_type == "tensor":
        return torch.from_numpy(spikes), spikes
    return spikes


def to_snn_input_format(spikes: torch.Tensor) -> torch.Tensor:
    """Convert spikes (batch, seq_len, T, features) to (T, batch, seq_len*features)

    Many SNN training loops prefer time-major tensors: (T, batch, -1).
    """
    # spikes: (batch, seq_len, T, features)
    spikes_np = spikes.numpy()
    batch, seq_len, T, features = spikes_np.shape
    spikes_perm = np.transpose(spikes_np, (2, 0, 1, 3))  # (T, batch, seq_len, features)
    spikes_flat = spikes_perm.reshape(T, batch, seq_len * features)
    return torch.from_numpy(spikes_flat)
