"""federated.py — The federated training loop.

Architecture (F170):
  - 1 central aggregator (the "vessel" or the "fleet-bridge")
  - N edge devices (the captain's wrist, the back-deck sensor, the sounder)
  - Each device has a frozen backbone + a local classifier head
  - Each round:
    1. Aggregator ships the current global head to all devices
    2. Each device runs N local SGD steps on its own data
    3. Each device ships its updated head (1.3 KB) back to the aggregator
    4. Aggregator FedAvg-averages the heads and emits a new global head

This is the canonical pattern for on-device learning. The F161
conservation law applies: the FROZEN backbone is the contract, the
HEAD is the only thing that moves, and the head is 1.3 KB.

The research question this script answers: how well does FedAvg
converge on this synthetic vessel-audio benchmark, with N devices
each having a non-IID data distribution?
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from feature_extractor import FeatureExtractor
from classifier_head import ClassifierHead, average_heads
from simulator import generate_dataset, split_train_test, CLASSES


def device_local_train(fe: FeatureExtractor,
                       head: ClassifierHead,
                       X_local: np.ndarray,
                       y_local: np.ndarray,
                       batch_size: int = 8,
                       local_steps: int = 10,
                       learning_rate: float = 0.05) -> Tuple[ClassifierHead, float]:
    """One device's local training round.

    Returns the updated head and the final mean loss.
    """
    # 1. Embed all local data (uses the frozen backbone — same on every device)
    embeddings = fe.embed_batch(X_local)
    # 2. Local SGD
    rng = np.random.default_rng()
    n = len(embeddings)
    final_loss = 0.0
    for step in range(local_steps):
        idx = rng.integers(0, n, batch_size)
        x_batch = embeddings[idx]
        y_batch = y_local[idx]
        final_loss = head.sgd_step(x_batch, y_batch, learning_rate=learning_rate)
    return head, final_loss


def non_iid_split(X: np.ndarray, y: np.ndarray, n_devices: int,
                  alpha: float = 0.3, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dirichlet non-IID data split. Lower alpha = more skewed per device.

    alpha=0.3 means each device has 70% of one class and small amounts
    of the others. This is the realistic scenario on a fishing vessel:
    the captain's wrist mostly hears wind, the back-deck sensor mostly
    hears net hauls, the sounder mostly hears engine drone.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES)
    # Dirichlet: per-class proportions per device
    proportions = rng.dirichlet([alpha] * n_devices, size=n_classes)  # (n_classes, n_devices)
    # Normalize to actual counts
    per_device_indices = [[] for _ in range(n_devices)]
    for c in range(n_classes):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        n_class = len(class_idx)
        # Each device gets proportions[c, d] * n_class samples
        cumulative = 0
        for d in range(n_devices):
            count = int(proportions[c, d] * n_class)
            if d == n_devices - 1:
                count = n_class - cumulative
            per_device_indices[d].extend(class_idx[cumulative:cumulative + count].tolist())
            cumulative += count
    out = []
    for d in range(n_devices):
        idx = np.array(per_device_indices[d])
        rng.shuffle(idx)
        out.append((X[idx], y[idx]))
    return out


def federated_train(n_rounds: int = 30,
                    n_devices: int = 5,
                    samples_per_class: int = 80,
                    local_steps: int = 8,
                    batch_size: int = 8,
                    learning_rate: float = 0.05,
                    alpha: float = 0.3,
                    test_frac: float = 0.2,
                    seed: int = 42,
                    verbose: bool = True) -> Dict:
    """Run a federated training experiment and return the global head +
    per-round metrics.

    Returns:
      {
        "global_head": ClassifierHead,
        "feature_extractor": FeatureExtractor,
        "history": [{"round": i, "loss": ..., "test_acc": ..., "head_hash": ...}, ...],
        "final_test_acc": float,
      }
    """
    # 1. Generate a balanced dataset
    X, y = generate_dataset(samples_per_class=samples_per_class, duration_sec=0.5, seed=seed)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y, test_frac=test_frac, seed=seed)

    # 2. Initialize the frozen backbone (same on all devices)
    fe = FeatureExtractor(seed="F170-tinyml-v1")
    # Calibrate the projection with PCA so the 64-dim embedding preserves
    # the top-64 directions of variance in the 192-dim hand-crafted space.
    fe.fit_pca_projection(X_train, n_calibration=200)

    # 3. Initialize the global head (same on all devices)
    global_head = ClassifierHead(num_classes=len(CLASSES), seed=0)

    # 4. Split training data into non-IID per-device shards
    device_data = non_iid_split(X_train, y_train, n_devices, alpha=alpha, seed=seed)

    # 5. Pre-compute test embeddings
    test_embeddings = fe.embed_batch(X_test)

    history = []
    for rnd in range(n_rounds):
        device_heads = []
        device_losses = []
        for d, (X_local, y_local) in enumerate(device_data):
            # Each device gets a copy of the global head (the broadcast)
            local_head = ClassifierHead.from_bytes(global_head.to_bytes(),
                                                   num_classes=global_head.num_classes,
                                                   embedding_dim=global_head.embedding_dim)
            local_head.steps = global_head.steps
            local_head.samples_seen = global_head.samples_seen
            # Local training
            updated_head, loss = device_local_train(fe, local_head, X_local, y_local,
                                                    batch_size=batch_size,
                                                    local_steps=local_steps,
                                                    learning_rate=learning_rate)
            device_heads.append(updated_head)
            device_losses.append(loss)

        # FedAvg
        # Weight by data size (more data = more weight)
        weights = [len(X_local) for X_local, _ in device_data]
        global_head = average_heads(device_heads, weights=weights)

        # Evaluate
        correct = 0
        for x_emb, label in zip(test_embeddings, y_test):
            if global_head.predict(x_emb) == label:
                correct += 1
        test_acc = correct / len(y_test)
        mean_loss = float(np.mean(device_losses))
        head_hash = f"0x{global_head.state_hash():016x}"

        history.append({
            "round": rnd,
            "mean_local_loss": mean_loss,
            "test_accuracy": test_acc,
            "head_state_hash": head_hash,
        })
        if verbose:
            print(f"  round {rnd:3d}: loss={mean_loss:.4f}  test_acc={test_acc:.3f}  head={head_hash}")

    final_test_acc = history[-1]["test_accuracy"]
    return {
        "global_head": global_head,
        "feature_extractor": fe,
        "history": history,
        "final_test_acc": final_test_acc,
    }


if __name__ == "__main__":
    print("F170 — Federated TinyML for the Vessel Edge")
    print(f"  classes: {CLASSES}")
    print()
    print("Test 1: IID-ish split (alpha=10.0, mostly uniform)")
    print("-" * 60)
    res_iid = federated_train(n_rounds=15, alpha=10.0, samples_per_class=60, seed=42)
    print(f"  final test accuracy: {res_iid['final_test_acc']:.3f}")
    print()
    print("Test 2: Non-IID split (alpha=0.3, very skewed per device)")
    print("-" * 60)
    res_noniid = federated_train(n_rounds=15, alpha=0.3, samples_per_class=60, seed=42)
    print(f"  final test accuracy: {res_noniid['final_test_acc']:.3f}")
    print()
    print(f"  FROZEN BACKBONE state hash: 0x{res_iid['feature_extractor'].state_hash():016x}")
    print(f"  IID final head state hash:  0x{res_iid['global_head'].state_hash():016x}")
    print(f"  NonIID final head state hash: 0x{res_noniid['global_head'].state_hash():016x}")
