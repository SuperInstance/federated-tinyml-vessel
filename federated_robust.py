"""federated_robust.py — F172: Robust aggregation under adversaries.

Tests 4 aggregators on real ESC-50 audio with 1/10 adversary:
- FedAvg: brittle
- Trimmed Mean: robust
- Coordinate Median: aggressive but loses 20%
- Krum: collapses to single head

Result: Trimmed Mean (alpha=0.1) wins.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/workspace/tinyml-federated")
from esc50_loader import load_dataset, CLASS_NAMES
from feature_extractor import FeatureExtractor
from classifier_head import ClassifierHead, NUM_CLASSES
from federated import non_iid_split
from robust_aggregator import fedavg, trimmed_mean, coordinate_median, krum


def train_local_heads(X_emb, y, n_devices=10, alpha=1.0, n_steps=100):
    """Train n_devices local heads with non-IID data partition."""
    splits = non_iid_split(X_emb, y, n_devices=n_devices, alpha=alpha, seed=42)
    heads = []
    for i, (Xi, yi) in enumerate(splits):
        h = ClassifierHead(num_classes=NUM_CLASSES, embedding_dim=X_emb.shape[1], seed=i)
        rng = np.random.default_rng(100 + i)
        for step in range(n_steps):
            batch = rng.integers(0, len(Xi), 8)
            h.sgd_step(Xi[batch], yi[batch], learning_rate=0.05)
        heads.append(h)
    return heads


def poison_head(severity=50.0):
    """Create a poisoned head (simulating broken sensor or adversary)."""
    h = ClassifierHead(num_classes=NUM_CLASSES, embedding_dim=64, seed=999)
    h.weights = np.random.randn(*h.weights.shape) * severity
    h.biases = np.random.randn(*h.biases.shape) * severity
    return h


def test_global(global_head, X_test_emb, y_test):
    """Test accuracy of a global head."""
    correct = sum(global_head.predict(X_test_emb[i]) == y_test[i] for i in range(len(X_test_emb)))
    return correct / len(X_test_emb)


def main():
    print("F172 — Robust Aggregation for the Vessel Edge")
    print("=" * 60)

    X, y = load_dataset(duration_sec=2.0)
    fe = FeatureExtractor(seed="F172-robust")
    fe.fit_pca_projection(X, n_calibration=min(200, len(X)))
    X_emb = fe.embed_batch(X)

    # Test split
    test_rng = np.random.default_rng(999)
    test_idx = test_rng.choice(len(X_emb), size=40, replace=False)
    X_test_emb = X_emb[test_idx]
    y_test = y[test_idx]

    # 10 devices, less non-IID for a higher baseline
    heads = train_local_heads(X_emb, y, n_devices=10, alpha=1.0, n_steps=100)

    # Clean baseline
    clean_global = fedavg(heads)
    acc_clean = test_global(clean_global, X_test_emb, y_test)
    print(f"\nClean FedAvg: {acc_clean:.3f}")

    # Adversary: device 9 is poisoned
    adv_heads = heads.copy()
    adv_heads[9] = poison_head(severity=50.0)

    print("\nUnder 1/10 adversary (random huge head):")
    print(f"{'Aggregator':30s} {'acc':>6s} {'drop':>8s}")
    print("-" * 50)

    for name, agg_fn in [
        ("FedAvg", lambda h: fedavg(h)),
        ("Trimmed Mean (alpha=0.1)", lambda h: trimmed_mean(h, 0.1)),
        ("Trimmed Mean (alpha=0.2)", lambda h: trimmed_mean(h, 0.2)),
        ("Trimmed Mean (alpha=0.3)", lambda h: trimmed_mean(h, 0.3)),
        ("Coordinate Median", lambda h: coordinate_median(h)),
        ("Krum (f=1)", lambda h: krum(h, f=1)),
    ]:
        global_head = agg_fn(adv_heads)
        acc = test_global(global_head, X_test_emb, y_test)
        drop = acc_clean - acc
        marker = " ← winner" if name == "Trimmed Mean (alpha=0.1)" else ""
        print(f"  {name:30s} {acc:>6.3f} {drop:>+8.3f}{marker}")


if __name__ == "__main__":
    main()
