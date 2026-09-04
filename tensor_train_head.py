"""tensor_train_head.py — F173: Tensor-Train (MPS) head with analytical gradient.

The flat 64x5 head from F170 is replaced with a Tensor-Train decomposition:
the head is a chain of L small 3-tensors (cores) connected by virtual bonds
of dimension chi.

For L=8 cores, chi=4: 208 params, 832 bytes fp32
For L=8 cores, chi=2: 64 params, 256 bytes fp32  (vs 1300 byte F170!)

The wire format is the full set of cores. State hash covers all cores.
Analytical gradient via backprop through the contraction.
"""
from __future__ import annotations
import numpy as np
import struct
import sys
sys.path.insert(0, "/workspace/tinyml-federated")
from classifier_head import FNV_OFFSET, FNV_PRIME, MASK


def fnv1a_64(data: bytes) -> int:
    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & MASK
    return h


class TensorTrainHead:
    """A Tensor-Train (MPS) head for n_classes classification."""

    def __init__(self, n_classes: int = 5, input_dim: int = 64, 
                 n_cores: int = 8, bond_dim: int = 4, seed: int = 0):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.n_cores = n_cores
        self.bond_dim = bond_dim

        # Each core handles 1 input dim (2 values: index 0 and 1)
        # For input_dim=64, n_cores=8: each core handles 8 input dims (averaged)
        # We'll use 2-dim cores with soft interpolation
        self.core_dims = [2] * n_cores

        rng = np.random.default_rng(seed)
        self.cores = []
        for l in range(n_cores):
            chi_l = 1 if l == 0 else bond_dim
            chi_r = 1 if l == n_cores - 1 else bond_dim
            # Output dim: use chi_r for last core as n_classes basis
            core = rng.standard_normal((chi_l, self.core_dims[l], chi_r)).astype(np.float32) * 0.1
            self.cores.append(core)

    def total_params(self) -> int:
        return sum(c.size for c in self.cores)

    def total_bytes(self) -> int:
        return self.total_params() * 4

    def state_hash(self) -> int:
        h = FNV_OFFSET
        for core in self.cores:
            for v in core.flatten().tolist():
                bs = struct.pack("<f", float(v))
                for byte in bs:
                    h ^= byte
                    h = (h * FNV_PRIME) & MASK
        return h

    def to_bytes(self) -> bytes:
        out = b""
        for core in self.cores:
            out += core.astype("<f").tobytes()
        return out

    @classmethod
    def from_bytes(cls, data: bytes, n_classes: int = 5, input_dim: int = 64,
                    n_cores: int = 8, bond_dim: int = 4):
        head = cls(n_classes=n_classes, input_dim=input_dim,
                   n_cores=n_cores, bond_dim=bond_dim, seed=0)
        offset = 0
        for i, core in enumerate(head.cores):
            size = core.nbytes
            head.cores[i] = np.frombuffer(data[offset:offset+size], dtype="<f").reshape(core.shape).astype(np.float32)
            offset += size
        return head

    def _contract_with_intermediates(self, x_chunks: np.ndarray):
        """Contract the TT, returning (logits, intermediates) for backprop.

        x_chunks: (n_cores,) values in [0, 1]
        Returns: logits (n_classes,), and list of intermediates for backprop
        """
        # Forward: compute left and right partial contractions
        # left[l] = contraction from cores 0..l-1, shape (chi_l,)
        # right[l] = contraction from cores l..L-1, shape (chi_{l+1},)
        L = self.n_cores
        chi = self.bond_dim
        
        left = [None] * (L + 1)
        left[0] = np.array([1.0])
        for l in range(L):
            core = self.cores[l]
            w0 = 1.0 - x_chunks[l]
            w1 = x_chunks[l]
            # Effective core: weighted sum of i=0 and i=1 slices
            eff = core[:, 0, :] * w0 + core[:, 1, :] * w1
            left[l+1] = left[l] @ eff
        
        # right is symmetric
        right = [None] * (L + 1)
        right[L] = np.array([1.0])
        for l in range(L-1, -1, -1):
            core = self.cores[l]
            w0 = 1.0 - x_chunks[l]
            w1 = x_chunks[l]
            eff = core[:, 0, :] * w0 + core[:, 1, :] * w1
            right[l] = eff @ right[l+1]
        
        # Logit = left[L] (which equals right[0])
        # But we need n_classes logits
        # For chi=1 at boundaries, left[L] is a single number
        logit_scalar = left[L].item() if left[L].size == 1 else left[L][0]
        # Replicate to n_classes with a small learned "class head"
        if not hasattr(self, '_class_head'):
            rng = np.random.default_rng(0)
            self._class_head = rng.standard_normal(self.n_classes).astype(np.float32) * 0.1
        logits = self._class_head * logit_scalar
        
        return logits, (left, right)

    def contract(self, x: np.ndarray) -> np.ndarray:
        """Contract TT with input x to get class logits."""
        chunk_size = max(1, len(x) // self.n_cores)
        x_chunks = x[:self.n_cores * chunk_size].reshape(self.n_cores, chunk_size).mean(axis=1)
        # Normalize
        rng_range = x_chunks.max() - x_chunks.min()
        if rng_range > 0:
            x_chunks = (x_chunks - x_chunks.min()) / rng_range
        else:
            x_chunks = np.zeros_like(x_chunks)
        logits, _ = self._contract_with_intermediates(x_chunks)
        return logits

    def predict(self, x: np.ndarray) -> int:
        logits = self.contract(x)
        return int(np.argmax(logits))

    def sgd_step(self, x: np.ndarray, label: int, lr: float = 0.05):
        """Single SGD step with analytical gradient on cores + class head."""
        chunk_size = max(1, len(x) // self.n_cores)
        x_chunks = x[:self.n_cores * chunk_size].reshape(self.n_cores, chunk_size).mean(axis=1)
        rng_range = x_chunks.max() - x_chunks.min()
        if rng_range > 0:
            x_chunks = (x_chunks - x_chunks.min()) / rng_range
        else:
            x_chunks = np.zeros_like(x_chunks)
        
        logits, (left, right) = self._contract_with_intermediates(x_chunks)
        # Softmax
        logits_exp = np.exp(logits - logits.max())
        probs = logits_exp / logits_exp.sum()
        # Gradient w.r.t. logits: p - y
        grad_logits = probs.copy()
        grad_logits[label] -= 1.0
        
        # Gradient w.r.t. logit_scalar = sum_c grad_logits[c] * class_head[c]
        grad_logit_scalar = (grad_logits * self._class_head).sum()
        
        # Gradient w.r.t. each core:
        # dL/dcore[l][j, i, k] = grad_logit_scalar * left[l][j] * w_i * right[l+1][k]
        L = self.n_cores
        for l in range(L):
            w0 = 1.0 - x_chunks[l]
            w1 = x_chunks[l]
            weights = np.array([w0, w1])
            # Outer product: (chi_l, chi_r)
            grad_core = np.outer(left[l], right[l+1])
            # Expand to (chi_l, 2, chi_r) with weights
            grad_core = grad_core[:, None, :] * weights[None, :, None]
            grad_core *= grad_logit_scalar
            self.cores[l] -= lr * grad_core.astype(np.float32)
        
        # Update class head
        self._class_head -= lr * grad_logit_scalar * grad_logits


if __name__ == "__main__":
    h = TensorTrainHead(n_classes=5, input_dim=64, n_cores=8, bond_dim=4, seed=42)
    print(f"Total params: {h.total_params()}")
    print(f"Total bytes (fp32): {h.total_bytes()}")
    print(f"State hash: 0x{h.state_hash():016x}")
    
    data = h.to_bytes()
    h2 = TensorTrainHead.from_bytes(data)
    print(f"Round-trip hash: 0x{h2.state_hash():016x}")
    assert h.state_hash() == h2.state_hash()
    print("Round-trip OK")
    
    x = np.random.randn(64)
    pred = h.predict(x)
    print(f"Prediction: {pred}")
    
    for step in range(50):
        x = np.random.randn(64)
        label = step % 5
        h.sgd_step(x, label, lr=0.05)
    print(f"After 50 SGD steps, hash: 0x{h.state_hash():016x}")
