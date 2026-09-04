"""federated_real.py — F170 on REAL audio (ESC-50 vessel subset).

This is the next-step R&D: take the F170 pipeline (proven on synthetic
data) and run it on 200 real audio samples from the ESC-50 dataset.

The 5 classes are real environmental sounds:
  0. engine
  1. sea_waves
  2. wind
  3. pouring_water
  4. water_drops

We expect lower accuracy than synthetic (real audio is messier), but
the architecture should still learn something useful.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/workspace/tinyml-federated")
from feature_extractor import FeatureExtractor
from classifier_head import ClassifierHead, NUM_CLASSES, HEAD_DIM
from esc50_loader import load_dataset, CLASS_NAMES
from federated import device_local_train, average_heads, non_iid_split


def run_federated_real_audio(
    n_rounds: int = 30,
    n_devices: int = 5,
    local_steps: int = 8,
    batch_size: int = 4,
    learning_rate: float = 0.05,
    alpha: float = 0.5,
    seed: int = 42,
):
    """Run F170 on real ESC-50 audio."""
    print("F170 — Federated TinyML on REAL Audio (ESC-50)")
    print("=" * 60)
    print(f"Classes: {CLASS_NAMES}")
    print()

    # Load real audio
    X, y = load_dataset(duration_sec=0.5)
    print(f"Loaded {len(X)} real audio samples, {X.shape[1]/16000:.1f}s each")
    print(f"Per class: {np.bincount(y).tolist()}")
    print()

    # Split train/test
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Init backbone + head
    fe = FeatureExtractor(seed="F170-real-audio")
    fe.fit_pca_projection(X_train, n_calibration=min(200, len(X_train)))
    print(f"Backbone state hash: 0x{fe.state_hash():016x}")
    print()

    global_head = ClassifierHead(num_classes=NUM_CLASSES, embedding_dim=fe.OUTPUT_DIM, seed=0)

    # Non-IID split
    device_data = non_iid_split(X_train, y_train, n_devices, alpha=alpha, seed=seed)
    for d, (Xl, yl) in enumerate(device_data):
        print(f"  device {d}: {len(Xl)} samples, class dist {np.bincount(yl, minlength=NUM_CLASSES).tolist()}")

    # Pre-compute test embeddings
    test_embeddings = fe.embed_batch(X_test)

    history = []
    for rnd in range(n_rounds):
        device_heads = []
        device_losses = []
        for d, (Xl, yl) in enumerate(device_data):
            local_head = ClassifierHead.from_bytes(global_head.to_bytes(),
                                                   num_classes=NUM_CLASSES,
                                                   embedding_dim=fe.OUTPUT_DIM)
            local_head.steps = global_head.steps
            local_head.samples_seen = global_head.samples_seen
            updated_head, loss = device_local_train(fe, local_head, Xl, yl,
                                                    batch_size=batch_size,
                                                    local_steps=local_steps,
                                                    learning_rate=learning_rate)
            device_heads.append(updated_head)
            device_losses.append(loss)

        weights = [len(Xl) for Xl, _ in device_data]
        global_head = average_heads(device_heads, weights=weights)

        # Eval
        correct = 0
        for x_emb, label in zip(test_embeddings, y_test):
            if global_head.predict(x_emb) == label:
                correct += 1
        test_acc = correct / len(y_test)
        mean_loss = float(np.mean(device_losses))
        history.append((rnd, mean_loss, test_acc))
        if rnd % 5 == 0 or rnd == n_rounds - 1:
            print(f"  round {rnd:3d}: loss={mean_loss:.4f}  test_acc={test_acc:.3f}")

    final_acc = history[-1][2]
    print()
    print(f"Final test accuracy: {final_acc:.3f}")
    print(f"Random baseline: {1.0/NUM_CLASSES:.3f}")
    print(f"Final head hash: 0x{global_head.state_hash():016x}")
    return final_acc, history


if __name__ == "__main__":
    # Run with different alphas
    for alpha in [10.0, 1.0, 0.5, 0.3]:
        print(f"\n{'='*60}")
        print(f"alpha={alpha}")
        print(f"{'='*60}")
        run_federated_real_audio(n_rounds=20, alpha=alpha, seed=42)
