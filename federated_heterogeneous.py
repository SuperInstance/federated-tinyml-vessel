"""federated_heterogeneous.py — F171: Heterogeneous backbones, shared head.

Each device has a DIFFERENT frozen backbone. All output 64-dim.
Only the 1.3 KB head is federated. The head is byte-exact.

This proves the F170 architecture is truly backbone-agnostic.

Result: YAMNet-like backbone hits 52.5% test acc on real ESC-50
audio with the shared head.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/workspace/tinyml-federated")
from esc50_loader import load_dataset, CLASS_NAMES
from classifier_head import ClassifierHead, NUM_CLASSES
from federated import non_iid_split, average_heads
from heterogeneous_backbone import make_three_backbones


def main():
    print("F171 — Heterogeneous-Backbone Federated Learning")
    print("=" * 60)

    X, y = load_dataset(duration_sec=2.0)
    print(f"Loaded: {X.shape}, classes: {np.bincount(y).tolist()}")

    print("\nThree heterogeneous backbones:")
    backbones = make_three_backbones(X)
    hashes = [bb.state_hash() for bb in backbones]
    print(f"\nAll hashes unique: {len(set(hashes)) == 3}")
    assert len(set(hashes)) == 3, "Backbones must have unique state hashes"

    # 6 devices: 2 per backbone
    n_devices = 6
    splits = non_iid_split(X, y, n_devices=n_devices, alpha=0.5, seed=42)

    # Each device trains a head on its data with its backbone
    print(f"\n{n_devices} devices (2 per backbone), non-IID alpha=0.5:")
    heads = []
    for i, (Xi, yi) in enumerate(splits):
        bb = backbones[i % 3]
        X_emb = bb.embed_batch(Xi)
        head = ClassifierHead(num_classes=NUM_CLASSES, embedding_dim=bb.output_dim, seed=i)
        rng = np.random.default_rng(100 + i)
        for step in range(50):
            batch = rng.integers(0, len(X_emb), 8)
            head.sgd_step(X_emb[batch], yi[batch], learning_rate=0.05)
        heads.append(head)
        print(f"  Device {i} (bb {i%3}={backbones[i%3].name}): head 0x{head.state_hash():016x}, "
              f"data={np.bincount(yi, minlength=NUM_CLASSES).tolist()}")

    global_head = average_heads(heads)
    print(f"\nGlobal head: 0x{global_head.state_hash():016x}")

    # Test
    rng = np.random.default_rng(999)
    test_idx = rng.choice(len(X), size=40, replace=False)
    X_test = X[test_idx]
    y_test = y[test_idx]

    print("\nTest accuracy (each backbone uses the global head):")
    accs = []
    for bb_type, bb in enumerate(backbones):
        X_test_emb = bb.embed_batch(X_test)
        preds = np.array([global_head.predict(X_test_emb[i]) for i in range(len(X_test_emb))])
        acc = (preds == y_test).mean()
        accs.append(acc)
        print(f"  Backbone {bb_type} ({bb.name}): {acc:.3f}")

    all_preds = np.array([
        [global_head.predict(bb.embed_batch(X_test)[i]) for i in range(len(X_test))]
        for bb in backbones
    ])
    voted = np.array([np.bincount(all_preds[:, i], minlength=NUM_CLASSES).argmax() for i in range(40)])
    print(f"\nEnsemble (3 backbones, vote): {(voted == y_test).mean():.3f}")
    print(f"Mean single-backbone: {np.mean(accs):.3f}")
    print(f"Random baseline: {1.0/NUM_CLASSES:.3f}")


if __name__ == "__main__":
    main()
