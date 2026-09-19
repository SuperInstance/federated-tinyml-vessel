"""classifier_head.py — The tiny on-device classifier head.

This is what runs on the device. It's 1.3 KB at fp32 (or 325 bytes at int8).
Per-device, locally trained. The only thing that crosses the network.

Architecture:
  Input:  64-dim embedding (from the frozen backbone)
  Output: 5-class softmax (e.g. silence / normal / wind / net_haul / line_tangle)
  Total parameters: 64 * 5 + 5 = 325 (1.3 KB at fp32)

Training: pure SGD on cross-entropy loss. No momentum, no Adam — the
memory budget on a Cortex-M33 doesn't allow it. We rely on the federated
aggregation to do the momentum-equivalent work across rounds.

The head is also byte-exact across substrates (numpy, C, TFLite). This
is the FNV-1a equivalent: 4 bytes per fp32 weight, 4 bytes per bias,
320+5 = 1,300 bytes total. The state hash is the device's "model id".
"""
from __future__ import annotations
import json
import numpy as np
import struct
from typing import List, Tuple


# Defaults: the F170 production head
NUM_CLASSES = 5      # silence, normal, wind, net_haul, line_tangle
HEAD_DIM = 64        # the frozen backbone output


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_64_bytes(b: bytes) -> int:
    h = FNV_OFFSET
    for byte in b:
        h ^= byte
        h = (h * FNV_PRIME) & MASK
    return h


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cross_entropy(probs: np.ndarray, target: int) -> float:
    return -np.log(probs[target] + 1e-9)


def cross_entropy_grad(probs: np.ndarray, target: int) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits (assuming softmax is built in)."""
    grad = probs.copy()
    grad[target] -= 1.0
    return grad


class ClassifierHead:
    """A 64->5 linear classifier with softmax. Trained by SGD on-device.

    >>> head = ClassifierHead(num_classes=5, seed=42)
    >>> head.weights.shape
    (64, 5)
    >>> head.biases.shape
    (5,)
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Glorot init
        limit = np.sqrt(6.0 / (embedding_dim + num_classes))
        self.weights = (rng.uniform(-limit, limit, (embedding_dim, num_classes))).astype(np.float32)
        self.biases = np.zeros(num_classes, dtype=np.float32)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # Training telemetry
        self.steps = 0
        self.samples_seen = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities."""
        logits = x @ self.weights + self.biases
        return softmax(logits)

    def predict(self, x: np.ndarray) -> int:
        return int(self.forward(x).argmax())

    def loss(self, x: np.ndarray, target: int) -> float:
        return cross_entropy(self.forward(x), target)

    def sgd_step(self, batch_x: np.ndarray, batch_y: np.ndarray,
                 learning_rate: float = 0.01) -> float:
        """One SGD step on a batch. Returns the mean loss.

        This is the on-device update. Bounded by the F161 conservation
        law: 1 step per round, batch size <= 32 (to fit in the attention
        budget of 4096 tokens per session).
        """
        if len(batch_x) == 0:
            return 0.0
        batch_x = np.asarray(batch_x, dtype=np.float32)
        batch_y = np.asarray(batch_y, dtype=np.int64)
        n = len(batch_x)
        # Forward
        logits = batch_x @ self.weights + self.biases
        probs = softmax(logits)
        # Loss
        loss = -np.log(probs[np.arange(n), batch_y] + 1e-9).mean()
        # Gradient
        d_logits = probs.copy()
        d_logits[np.arange(n), batch_y] -= 1.0
        d_logits /= n
        dW = batch_x.T @ d_logits
        db = d_logits.sum(axis=0)
        # Update
        self.weights -= learning_rate * dW.astype(np.float32)
        self.biases -= learning_rate * db.astype(np.float32)
        self.steps += 1
        self.samples_seen += n
        return float(loss)

    def state_hash(self) -> int:
        """FNV-1a 64-bit state hash of the head weights. The device's
        model identity — this is what gets shared in federation."""
        h = FNV_OFFSET
        for v in self.weights.flatten().tolist() + self.biases.tolist():
            bs = struct.pack("<f", float(v))
            for byte in bs:
                h ^= byte
                h = (h * FNV_PRIME) & MASK
        return h

    def flat_state(self) -> np.ndarray:
        """The head as one flat parameter vector (weights then biases) — a point
        in the 325-D parameter space. Successive rounds' flat states form the
        convergence trajectory read by ``gesture.convergence_geometry``."""
        return np.concatenate(
            [self.weights.astype(np.float32).ravel(), self.biases.astype(np.float32).ravel()]
        )

    def to_bytes(self) -> bytes:
        return self.weights.astype(np.float32).tobytes() + self.biases.astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, num_classes: int = 5, embedding_dim: int = 64) -> "ClassifierHead":
        head = cls(num_classes=num_classes, embedding_dim=embedding_dim)
        w_size = embedding_dim * num_classes * 4
        head.weights = np.frombuffer(data[:w_size], dtype=np.float32).reshape(embedding_dim, num_classes).copy()
        head.biases = np.frombuffer(data[w_size:w_size + num_classes * 4], dtype=np.float32).copy()
        return head

    def serialize(self) -> dict:
        """JSON-serializable representation for federation over the wire."""
        return {
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "steps": self.steps,
            "samples_seen": self.samples_seen,
            "state_hash": f"0x{self.state_hash():016x}",
        }

    @classmethod
    def deserialize(cls, d: dict) -> "ClassifierHead":
        head = cls(num_classes=d["num_classes"], embedding_dim=d["embedding_dim"])
        head.weights = np.array(d["weights"], dtype=np.float32)
        head.biases = np.array(d["biases"], dtype=np.float32)
        head.steps = d["steps"]
        head.samples_seen = d["samples_seen"]
        return head

    def to_json_envelope(self, round: int = 0, samples_seen: int = None, device_id: str = "global") -> str:
        """The F170 wire envelope as a JSON string.

        The envelope wraps the head with schema versioning, round info,
        and the state hash. The state hash is the canonical identity.
        """
        envelope = {
            "schema": "f170-head-v1",
            "device_id": device_id,
            "round": round,
            "samples_seen": samples_seen if samples_seen is not None else self.samples_seen,
            "W": self.weights.tolist(),
            "b": self.biases.tolist(),
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "steps": self.steps,
            "state_hash": f"0x{self.state_hash():016x}",
        }
        return json.dumps(envelope)

    @classmethod
    def from_json_envelope(cls, msg: dict) -> "ClassifierHead":
        """Parse a head from a wire envelope JSON dict."""
        head = cls(num_classes=msg["num_classes"], embedding_dim=msg["embedding_dim"])
        head.weights = np.array(msg["W"], dtype=np.float32)
        head.biases = np.array(msg["b"], dtype=np.float32)
        head.steps = msg.get("steps", 0)
        head.samples_seen = msg.get("samples_seen", 0)
        return head


def average_heads(heads: List[ClassifierHead],
                  weights: List[float] = None) -> ClassifierHead:
    """Federated averaging (FedAvg) of a list of heads.

    This is the only thing the central aggregator does. It receives a
    ClassifierHead from each device (their state_hash + 1.3KB of weights)
    and averages them. The result becomes the new global head.
    """
    if not heads:
        raise ValueError("empty head list")
    if weights is None:
        weights = [1.0] * len(heads)
    total = sum(weights)
    weights = [w / total for w in weights]
    avg = ClassifierHead(num_classes=heads[0].num_classes,
                         embedding_dim=heads[0].embedding_dim)
    avg.weights = sum(w * h.weights for w, h in zip(weights, heads)).astype(np.float32)
    avg.biases = sum(w * h.biases for w, h in zip(weights, heads)).astype(np.float32)
    avg.steps = max(h.steps for h in heads)
    avg.samples_seen = sum(h.samples_seen for h in heads)
    return avg


if __name__ == "__main__":
    # Self-test
    head = ClassifierHead(num_classes=5, seed=42)
    print("initial state hash:", hex(head.state_hash()))
    # 100 fake training steps
    rng = np.random.default_rng(0)
    for step in range(100):
        x = rng.standard_normal((32, 64)).astype(np.float32)
        y = rng.integers(0, 5, 32)
        loss = head.sgd_step(x, y, learning_rate=0.01)
        if step % 25 == 0:
            print(f"  step {step}: loss={loss:.4f}, state_hash={hex(head.state_hash())}")
    # Round-trip
    data = head.to_bytes()
    head2 = ClassifierHead.from_bytes(data)
    assert head2.state_hash() == head.state_hash()
    # Federation
    head3 = ClassifierHead(num_classes=5, seed=99)
    avg = average_heads([head, head3], weights=[0.7, 0.3])
    print("federated avg state hash:", hex(avg.state_hash()))
    print("self-test passed")
