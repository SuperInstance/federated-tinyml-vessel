"""federated_tt.py — F173: Tensor-Train head, federated on real ESC-50.

The flat 64x5 head from F170 is replaced with a Tensor-Train (MPS) head:
- n_cores = 8, bond_dim = chi
- 2 values per core (binary interpolation)
- Output: 5 logits via a tiny class head (5 params)

Total params: chi=1: 32, chi=2: 64, chi=4: 208, chi=8: 768
Total bytes (fp32): chi=1: 128, chi=2: 256, chi=4: 832, chi=8: 3072

F170 baseline: 1300 bytes (325 params, dense 64x5).
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/workspace/tinyml-federated")
from esc50_loader import load_dataset, CLASS_NAMES
from feature_extractor import FeatureExtractor
from tensor_train_head import TensorTrainHead
from federated import non_iid_split


def train_local_heads_tt(X_emb, y, n_devices=6, n_cores=8, bond_dim=4,
                          n_steps=50, lr=0.05, alpha=0.5, seed=42):
    """Train n_devices local TT heads with non-IID data partition."""
    splits = non_iid_split(X_emb, y, n_devices=n_devices, alpha=alpha, seed=seed)
    heads = []
    for i, (Xi, yi) in enumerate(splits):
        h = TensorTrainHead(n_classes=5, input_dim=64, n_cores=n_cores,
                             bond_dim=bond_dim, seed=i*100)
        rng = np.random.default_rng(100 + i)
        for step in range(n_steps):
            idx = rng.integers(0, len(Xi))
            h.sgd_step(Xi[idx], yi[idx], lr=lr)
        heads.append(h)
    return heads


def average_tt_heads(heads):
    """Average cores across heads (simple FedAvg on cores)."""
    ref = heads[0]
    avg = TensorTrainHead(n_classes=ref.n_classes, input_dim=ref.input_dim,
                           n_cores=ref.n_cores, bond_dim=ref.bond_dim, seed=0)
    for l in range(avg.n_cores):
        stacked = np.stack([h.cores[l] for h in heads])
        avg.cores[l] = stacked.mean(axis=0)
    # Average the class heads too
    stacked_ch = np.stack([h._class_head for h in heads])
    avg._class_head = stacked_ch.mean(axis=0)
    return avg


def test_global_tt(global_head, X_test_emb, y_test):
    """Test accuracy of a global TT head."""
    correct = sum(global_head.predict(X_test_emb[i]) == y_test[i] for i in range(len(X_test_emb)))
    return correct / len(X_test_emb)


def main():
    print("F173 — Tensor-Train Head, Federated on Real ESC-50")
    print("=" * 60)

    X, y = load_dataset(duration_sec=2.0)
    fe = FeatureExtractor(seed="F173-tt")
    fe.fit_pca_projection(X, n_calibration=min(200, len(X)))
    X_emb = fe.embed_batch(X)

    test_rng = np.random.default_rng(999)
    test_idx = test_rng.choice(len(X_emb), size=40, replace=False)
    X_test = X_emb[test_idx]
    y_test = y[test_idx]

    print("\nBond dimension sweep (n_cores=8, 6 devices, 50 SGD steps, alpha=0.5):")
    print(f"{'chi':>3s} {'params':>6s} {'bytes':>5s} {'test_acc':>9s}")
    print("-" * 30)

    for chi in [1, 2, 4, 8]:
        heads = train_local_heads_tt(X_emb, y, n_devices=6, n_cores=8, 
                                      bond_dim=chi, n_steps=50)
        avg = average_tt_heads(heads)
        acc = test_global_tt(avg, X_test, y_test)
        print(f"{chi:>3d} {avg.total_params():>6d} {avg.total_bytes():>5d} {acc:>9.3f}")


if __name__ == "__main__":
    main()
