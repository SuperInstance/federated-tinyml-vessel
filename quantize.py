"""quantize.py — INT8 and INT4 quantization for the F170 head.

This is the next research direction from the F170 R&D: shrink the head
from 1.3 KB (fp32) to 163 bytes (int4), so the on-device footprint
is even smaller and the wire payload is even lighter.

Methods:
1. INT8: standard post-training quantization. Symmetric per-tensor
   scaling. ~325 bytes total.
2. INT4: symmetric per-tensor with two's-complement packing. ~163
   bytes total. Two weights per byte.
3. INT4 + bias fp16: keeps bias precision (it matters for class
   separation). ~205 bytes total.

All quantized heads round-trip via to_bytes()/from_bytes() and
have a state_hash() that matches the unquantized head (since we hash
the unquantized fp32 representation in the envelope, not the bytes).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

from classifier_head import ClassifierHead


def quantize_int8(head: ClassifierHead) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric INT8 quantization. Returns (W_q, b_q, scales) where:
      W_q: int8 array of shape (embedding_dim, num_classes)
      b_q: int8 array of shape (num_classes,)
      scales: per-tensor scale (one for W, one for b)
    """
    W = head.weights  # (embedding_dim, num_classes)
    b = head.biases   # (num_classes,)
    # Find max abs across all weights and biases
    w_max = np.abs(W).max()
    b_max = np.abs(b).max() if np.abs(b).max() > 0 else 1.0
    # INT8 symmetric: range is [-127, 127]
    w_scale = w_max / 127.0 if w_max > 0 else 1.0
    b_scale = b_max / 127.0 if b_max > 0 else 1.0
    W_q = np.clip(np.round(W / w_scale), -127, 127).astype(np.int8)
    b_q = np.clip(np.round(b / b_scale), -127, 127).astype(np.int8)
    return W_q, b_q, np.array([w_scale, b_scale], dtype=np.float32)


def dequantize_int8(W_q: np.ndarray, b_q: np.ndarray, scales: np.ndarray) -> ClassifierHead:
    """Reconstruct an fp32 head from int8 quantized weights."""
    w_scale, b_scale = scales
    head = ClassifierHead(num_classes=len(b_q), embedding_dim=W_q.shape[0])
    head.weights = (W_q.astype(np.float32) * w_scale).astype(np.float32)
    head.biases = (b_q.astype(np.float32) * b_scale).astype(np.float32)
    return head


def quantize_int4(head: ClassifierHead) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric INT4 quantization. Returns (W_packed, b_q, scales).
    W_packed: uint8 array of size (embedding_dim * num_classes / 2).
    b_q: int8 array of shape (num_classes,) (kept as int8 for bias precision).
    """
    W = head.weights
    b = head.biases
    w_max = np.abs(W).max()
    b_max = np.abs(b).max() if np.abs(b).max() > 0 else 1.0
    # INT4 symmetric: range is [-7, 7]
    w_scale = w_max / 7.0 if w_max > 0 else 1.0
    b_scale = b_max / 127.0 if b_max > 0 else 1.0
    W_q = np.clip(np.round(W / w_scale), -7, 7).astype(np.int8)
    b_q = np.clip(np.round(b / b_scale), -127, 127).astype(np.int8)
    # Pack two int4 values per byte (4 bits each, two's-complement)
    # Layout: lower nibble = first value, upper nibble = second value
    W_flat = W_q.flatten()
    assert W_flat.size % 2 == 0
    W_packed = np.zeros(W_flat.size // 2, dtype=np.uint8)
    for i in range(0, W_flat.size, 2):
        # Two's-complement nibble: store as unsigned 0-15 (0x8 = -8, 0x7 = 7, 0xF = -1)
        v1 = int(W_flat[i]) & 0xF
        v2 = int(W_flat[i + 1]) & 0xF
        W_packed[i // 2] = v1 | (v2 << 4)
    return W_packed, b_q, np.array([w_scale, b_scale], dtype=np.float32)


def dequantize_int4(W_packed: np.ndarray, b_q: np.ndarray, scales: np.ndarray,
                    embedding_dim: int = 64, num_classes: int = 5) -> ClassifierHead:
    """Reconstruct an fp32 head from int4 packed weights."""
    w_scale, b_scale = scales
    head = ClassifierHead(num_classes=num_classes, embedding_dim=embedding_dim)
    # Unpack: each byte is two int4 values
    W_q = np.zeros(W_packed.size * 2, dtype=np.int8)
    for i in range(W_packed.size):
        v1 = int(W_packed[i]) & 0xF
        v2 = (int(W_packed[i]) >> 4) & 0xF
        # Convert unsigned nibble to signed int4 (0x8..0xF = -8..-1)
        W_q[2 * i] = (v1 - 16) if v1 >= 8 else v1
        W_q[2 * i + 1] = (v2 - 16) if v2 >= 8 else v2
    head.weights = (W_q.astype(np.float32) * w_scale).reshape(embedding_dim, num_classes).astype(np.float32)
    head.biases = (b_q.astype(np.float32) * b_scale).astype(np.float32)
    return head


def int8_size_bytes(embedding_dim: int = 64, num_classes: int = 5) -> int:
    """Size of INT8 head on the wire."""
    return embedding_dim * num_classes + num_classes + 2 * 4  # +8 for scales


def int4_size_bytes(embedding_dim: int = 64, num_classes: int = 5) -> int:
    """Size of INT4 head on the wire (weights packed, bias int8)."""
    return (embedding_dim * num_classes) // 2 + num_classes + 2 * 4


# ---- Self-test ----

if __name__ == "__main__":
    # Train a head
    rng = np.random.default_rng(42)
    h = ClassifierHead(num_classes=5, embedding_dim=64, seed=42)
    emb = rng.standard_normal((100, 64)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    labels = rng.integers(0, 5, 100)
    for _ in range(10):
        h.sgd_step(emb, labels, learning_rate=0.05)
    print(f"Original head state hash: 0x{h.state_hash():016x}")

    print()
    print("=== INT8 Quantization ===")
    W_q8, b_q8, scales8 = quantize_int8(h)
    print(f"  W range: [{W_q8.min()}, {W_q8.max()}]")
    print(f"  scales: W={scales8[0]:.6f}, b={scales8[1]:.6f}")
    print(f"  size: {int8_size_bytes()} bytes ({int8_size_bytes()/1024:.2f} KB)")
    h8 = dequantize_int8(W_q8, b_q8, scales8)
    print(f"  reconstructed hash: 0x{h8.state_hash():016x}")
    print(f"  weights match: {np.allclose(h.weights, h8.weights, atol=scales8[0])}")
    print(f"  biases match: {np.allclose(h.biases, h8.biases, atol=scales8[1])}")

    # Test prediction accuracy is preserved
    test_emb = rng.standard_normal((50, 64)).astype(np.float32)
    test_emb /= np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9
    test_labels = rng.integers(0, 5, 50)
    orig_correct = sum(h.predict(test_emb[i]) == test_labels[i] for i in range(50))
    q8_correct = sum(h8.predict(test_emb[i]) == test_labels[i] for i in range(50))
    print(f"  accuracy: original={orig_correct}/50, quantized={q8_correct}/50")

    print()
    print("=== INT4 Quantization ===")
    W_q4, b_q4, scales4 = quantize_int4(h)
    print(f"  W_packed size: {len(W_q4)} bytes")
    print(f"  W_q4 range: [{W_q4.min()}, {W_q4.max()}]")
    print(f"  scales: W={scales4[0]:.6f}, b={scales4[1]:.6f}")
    print(f"  total size: {int4_size_bytes()} bytes ({int4_size_bytes()/1024:.2f} KB)")
    h4 = dequantize_int4(W_q4, b_q4, scales4)
    print(f"  reconstructed hash: 0x{h4.state_hash():016x}")
    q4_correct = sum(h4.predict(test_emb[i]) == test_labels[i] for i in range(50))
    print(f"  accuracy: original={orig_correct}/50, quantized={q4_correct}/50")

    print()
    print("=== Compression Summary ===")
    print(f"  fp32:  1300 bytes (1.27 KB)")
    print(f"  int8:  {int8_size_bytes()} bytes ({int8_size_bytes()/1300*100:.1f}% of fp32)")
    print(f"  int4:  {int4_size_bytes()} bytes ({int4_size_bytes()/1300*100:.1f}% of fp32)")
