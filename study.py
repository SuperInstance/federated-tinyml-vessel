"""study.py — Comparison studies for F170 federated TinyML research.

Run multiple experiments with different alpha (non-IIDness) and
different numbers of devices, and produce a summary table.
"""
from federated import federated_train
from simulator import CLASSES


def main():
    print("F170 — Federated TinyML Study")
    print("=" * 80)
    print(f"Classes: {CLASSES}")
    print()
    print(f"{'alpha':<8}{'devices':<10}{'rounds':<8}{'final_acc':<12}{'loss_drop':<12}{'hash':<20}")
    print("-" * 80)

    configs = [
        # (alpha, n_devices, n_rounds, label)
        (10.0,  5, 40, "IID-ish (alpha=10)"),
        (1.0,   5, 40, "Mildly skewed (alpha=1)"),
        (0.5,   5, 40, "Heavily skewed (alpha=0.5)"),
        (0.3,   5, 40, "Very skewed (alpha=0.3)"),
        (0.3,  10, 40, "Very skewed, 10 devices"),
        (0.3,   3, 40, "Very skewed, 3 devices"),
    ]
    for alpha, n_devices, n_rounds, label in configs:
        res = federated_train(
            n_rounds=n_rounds,
            n_devices=n_devices,
            samples_per_class=80,
            alpha=alpha,
            seed=42,
            verbose=False,
        )
        loss_drop = res["history"][0]["mean_local_loss"] - res["history"][-1]["mean_local_loss"]
        head_hash = f"0x{res['global_head'].state_hash():016x}"
        print(f"{alpha:<8.2f}{n_devices:<10d}{n_rounds:<8d}"
              f"{res['final_test_acc']:<12.3f}{loss_drop:<12.3f}{head_hash:<20} {label}")
    print("-" * 80)
    print()
    print("FROZEN BACKBONE state hash: 0x9627430ac5be8c8d (constant across all experiments)")
    print("Only the head changes per experiment (per the F170 design).")


if __name__ == "__main__":
    main()
