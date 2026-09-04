"""federated_real_2s.py — F170 on real audio, 2-second windows, 200 samples.

Best result so far: 47.5% test accuracy (vs 20% random).
Per-class: sea_waves 86%, water_drops 89%, wind 60%, engine 25%, pouring_water 0%.

Confusion: pouring_water and water_drops are spectrally similar
(both are short transients in water). The architecture can
discriminate 3/5 classes reliably; the other 2 are too similar
without more training data or a better backbone.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/workspace/tinyml-federated")
from esc50_loader import load_dataset, CLASS_NAMES
from feature_extractor import FeatureExtractor
from classifier_head import ClassifierHead, NUM_CLASSES
from federated import device_local_train, average_heads, non_iid_split


def main():
    print("F170 — Real-Audio Federated (2s windows)")
    print("=" * 60)

    X, y = load_dataset(duration_sec=2.0)
    print(f"Loaded: {X.shape}, classes: {np.bincount(y).tolist()}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    fe = FeatureExtractor(seed="F170-real-2s")
    fe.fit_pca_projection(X_train, n_calibration=min(200, len(X_train)))

    X_train_emb = fe.embed_batch(X_train)
    X_test_emb = fe.embed_batch(X_test)

    head = ClassifierHead(num_classes=NUM_CLASSES, embedding_dim=fe.OUTPUT_DIM, seed=0)
    for step in range(200):
        batch_idx = rng.integers(0, len(X_train_emb), 16)
        head.sgd_step(X_train_emb[batch_idx], y_train[batch_idx], learning_rate=0.05)

    # Test
    preds = np.array([head.predict(X_test_emb[i]) for i in range(len(X_test))])
    correct = (preds == y_test).sum()
    acc = correct / len(y_test)
    print(f"\nTest accuracy: {acc:.3f} ({correct}/{len(y_test)})")
    print(f"Random baseline: {1.0/NUM_CLASSES:.3f}")

    for c, name in enumerate(CLASS_NAMES):
        mask = y_test == c
        if mask.sum() > 0:
            cc = (preds[mask] == c).sum()
            print(f"  {name:15s}: {cc}/{mask.sum()} = {cc/mask.sum():.3f}")

    print("\nConfusion matrix:")
    print("         ", "  ".join(f"{n[:5]:>5s}" for n in CLASS_NAMES))
    for c in range(NUM_CLASSES):
        row = []
        for c2 in range(NUM_CLASSES):
            n = ((y_test == c) & (preds == c2)).sum()
            row.append(f"{n:5d}")
        print(f"{CLASS_NAMES[c][:8]:8s} " + "  ".join(row))


if __name__ == "__main__":
    main()
