"""robust_aggregator.py — F172: Robust aggregation under adversaries.

Implements:
- FedAvg (baseline)
- Trimmed Mean (drop top/bottom alpha)
- Coordinate-wise Median
- Krum (select nearest neighbor)

Tests with 1/10 adversary poisoning.
"""
from __future__ import annotations
import numpy as np
import sys
sys.path.insert(0, "/workspace/tinyml-federated")
from classifier_head import ClassifierHead, NUM_CLASSES


def fedavg(heads: list) -> ClassifierHead:
    """Standard FedAvg: simple average of all heads."""
    ref = heads[0]
    avg_w = np.mean([h.weights for h in heads], axis=0)
    avg_b = np.mean([h.biases for h in heads], axis=0)
    new_head = ClassifierHead(num_classes=ref.weights.shape[0], 
                               embedding_dim=ref.weights.shape[1], 
                               seed=0)
    new_head.weights = avg_w
    new_head.biases = avg_b
    return new_head


def trimmed_mean(heads: list, trim_ratio: float = 0.2) -> ClassifierHead:
    """Coordinate-wise trimmed mean: drop top and bottom trim_ratio fraction."""
    n = len(heads)
    k = int(n * trim_ratio)
    if k == 0:
        return fedavg(heads)
    
    Ws = np.stack([h.weights for h in heads], axis=0)  # (n, C, D)
    bs = np.stack([h.biases for h in heads], axis=0)   # (n, C)
    
    Ws_sorted = np.sort(Ws, axis=0)
    bs_sorted = np.sort(bs, axis=0)
    
    W_trimmed = Ws_sorted[k:n-k].mean(axis=0)
    b_trimmed = bs_sorted[k:n-k].mean(axis=0)
    
    new_head = ClassifierHead(num_classes=heads[0].weights.shape[0],
                               embedding_dim=heads[0].weights.shape[1],
                               seed=0)
    new_head.weights = W_trimmed
    new_head.biases = b_trimmed
    return new_head


def coordinate_median(heads: list) -> ClassifierHead:
    """Coordinate-wise median: take the median of each weight."""
    Ws = np.stack([h.weights for h in heads], axis=0)
    bs = np.stack([h.biases for h in heads], axis=0)
    W_med = np.median(Ws, axis=0)
    b_med = np.median(bs, axis=0)
    new_head = ClassifierHead(num_classes=heads[0].weights.shape[0],
                               embedding_dim=heads[0].weights.shape[1],
                               seed=0)
    new_head.weights = W_med
    new_head.biases = b_med
    return new_head


def krum(heads: list, f: int = 1) -> ClassifierHead:
    """Krum: select the single head closest to its neighbors."""
    n = len(heads)
    Ws = np.stack([h.weights for h in heads], axis=0).reshape(n, -1)
    bs = np.stack([h.biases for h in heads], axis=0).reshape(n, -1)
    flat = np.concatenate([Ws, bs], axis=1)
    
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(flat[i] - flat[j])
            distances[i, j] = d
            distances[j, i] = d
    
    # For each i, sum of distances to n-f-2 nearest neighbors
    scores = np.zeros(n)
    for i in range(n):
        sorted_d = np.sort(distances[i])
        # Sum the closest n-f-2 (skip self at index 0)
        scores[i] = sorted_d[1:n-f].sum()
    
    # Select the head with the smallest score
    selected = np.argmin(scores)
    return heads[selected]


if __name__ == "__main__":
    # Test all aggregators with a synthetic scenario
    print("Testing robust aggregators...")
    
    # Create 10 clean heads
    rng = np.random.default_rng(42)
    clean_heads = []
    for i in range(9):
        h = ClassifierHead(num_classes=5, embedding_dim=64, seed=i)
        # Add small random noise
        h.weights += rng.normal(0, 0.01, h.weights.shape)
        h.biases += rng.normal(0, 0.01, h.biases.shape)
        clean_heads.append(h)
    
    # 1 adversary: flip sign
    adv = ClassifierHead(num_classes=5, embedding_dim=64, seed=99)
    adv.weights = -adv.weights * 5
    adv.biases = -adv.biases * 5
    all_heads = clean_heads + [adv]
    
    # Test each aggregator
    print("\nFedAvg (baseline, should be poisoned):")
    head = fedavg(all_heads)
    head.weights += rng.normal(0, 0.01, head.weights.shape)  # simulate test
    # The avg should be pulled toward the adversary
    print(f"  W mean: {head.weights.mean():.3f} (should be near 0 for clean)")
    
    print("\nTrimmed Mean (alpha=0.2):")
    head = trimmed_mean(all_heads, trim_ratio=0.2)
    print(f"  W mean: {head.weights.mean():.3f}")
    
    print("\nCoordinate Median:")
    head = coordinate_median(all_heads)
    print(f"  W mean: {head.weights.mean():.3f}")
    
    print("\nKrum:")
    head = krum(all_heads, f=1)
    print(f"  W mean: {head.weights.mean():.3f}")
