"""simulator.py — Synthetic vessel-audio simulator.

We don't have a real hydrophone recording of a fishing vessel in our
sandbox. To make the federated TinyML research reproducible, we generate
synthetic audio that mimics 5 classes of vessel sounds:

  0. silence  — low-energy noise
  1. normal   — engine idle + ambient water
  2. wind     — broadband noise with low-frequency roll-off
  3. net_haul — periodic transients (winch + line)
  4. line_tangle — broadband impulse train (the "tangled hooks" killer app)

The audio is generated as colored noise plus class-specific events.
The point isn't physical realism — it's to have a reproducible
benchmark that exercises the whole pipeline.

Each class has a "fingerprint" in the mel spectrogram domain. The
frozen backbone will project these into distinct clusters in the
64-dim embedding space. The on-device head learns to separate them.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


CLASSES = ["silence", "normal", "wind", "net_haul", "line_tangle"]


def generate_sample(class_id: int, duration_sec: float = 1.0,
                   sample_rate: int = 16000, seed: int = 0) -> np.ndarray:
    """Generate one synthetic audio sample for a given class."""
    rng = np.random.default_rng(seed * 31 + class_id * 7919)
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    audio = np.zeros(n, dtype=np.float32)
    if class_id == 0:  # silence
        audio = rng.standard_normal(n).astype(np.float32) * 0.005
    elif class_id == 1:  # normal — engine idle + ambient
        # Engine drone around 80 Hz and harmonics
        audio += 0.1 * np.sin(2 * np.pi * 80 * t)
        audio += 0.05 * np.sin(2 * np.pi * 160 * t)
        audio += 0.02 * np.sin(2 * np.pi * 240 * t)
        audio += rng.standard_normal(n).astype(np.float32) * 0.03
    elif class_id == 2:  # wind — broadband with LF roll-off
        # Low-pass filtered noise
        noise = rng.standard_normal(n).astype(np.float32)
        # Simple 1-pole LPF
        alpha = 0.05
        filtered = np.zeros_like(noise)
        filtered[0] = noise[0] * alpha
        for i in range(1, n):
            filtered[i] = filtered[i - 1] + alpha * (noise[i] - filtered[i - 1])
        audio = filtered * 0.15
    elif class_id == 3:  # net_haul — periodic transients
        # Winch/grinding every 0.2s, each transient is a short chirp
        transient_every = 0.2
        for k in range(int(duration_sec / transient_every)):
            start = int(k * transient_every * sample_rate)
            transient_dur = int(0.05 * sample_rate)  # 50ms transient
            if start + transient_dur > n:
                break
            chirp_t = np.arange(transient_dur) / sample_rate
            chirp = 0.2 * np.sin(2 * np.pi * (200 + 800 * chirp_t / 0.05) * chirp_t)
            audio[start:start + transient_dur] += chirp
        audio += rng.standard_normal(n).astype(np.float32) * 0.04
    elif class_id == 4:  # line_tangle — broadband impulse train
        # Random impulses at 50-150 Hz rate, each is a short burst
        impulse_rate = 80  # Hz
        n_impulses = int(duration_sec * impulse_rate)
        impulse_times = rng.uniform(0, duration_sec, n_impulses)
        for imp_t in impulse_times:
            start = int(imp_t * sample_rate)
            burst_dur = int(0.02 * sample_rate)  # 20ms
            if start + burst_dur > n:
                continue
            burst_freq = rng.uniform(400, 1200)  # broadband
            burst_t = np.arange(burst_dur) / sample_rate
            burst = 0.25 * np.sin(2 * np.pi * burst_freq * burst_t) * np.exp(-burst_t * 50)
            audio[start:start + burst_dur] += burst
        audio += rng.standard_normal(n).astype(np.float32) * 0.05
    return audio.astype(np.float32)


def generate_dataset(samples_per_class: int = 100,
                     duration_sec: float = 1.0,
                     sample_rate: int = 16000,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a balanced dataset. Returns (X, y) where X is (N, n) and y is (N,)."""
    Xs, ys = [], []
    for c in range(len(CLASSES)):
        for i in range(samples_per_class):
            Xs.append(generate_sample(c, duration_sec, sample_rate, seed=seed * 1000 + i))
            ys.append(c)
    return np.stack(Xs), np.array(ys, dtype=np.int64)


def split_train_test(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
                     seed: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                              Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


if __name__ == "__main__":
    X, y = generate_dataset(samples_per_class=20, duration_sec=0.5, seed=42)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("class distribution:", np.bincount(y))
    for c in range(len(CLASSES)):
        print(f"  class {c} ({CLASSES[c]}): sample mean={X[y == c].mean():.4f}, std={X[y == c].std():.4f}")
