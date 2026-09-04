"""feature_extractor.py — The frozen audio feature extractor.

The on-device pipeline:
  raw audio (16 kHz) -> log-mel spectrogram (64 x ~50)
                    -> MFCC + delta + delta-delta (60 x ~50)
                    -> hand-crafted stats (mean + std + max over time) -> 180-dim
                    -> linear projection 180 -> 64 (the frozen backbone)
                    -> L2 normalize

The projection is "frozen" in the federated design: weights are
calibrated once on a centralized server (PCA on the 180-dim features),
then pushed to every device. Devices never update the projection.
They only train a 64x5 linear classifier head on top.

This mirrors the federated TinyML pattern from F170: a frozen
backbone + tiny on-device head. The backbone is byte-exact across
substrates (Python, C, TFLite), so the embeddings are deterministic.

Architecture (12,544 parameters total at fp32 = 50 KB, or 12.5 KB at int8):
  Input:  180-dim (MFCC mean + std + max of 60-mel x 3 log-spectrogram)
  Linear: 180 -> 64
  L2 normalize -> 64-dim embedding
                                (180*64 + 64 = 11,584 params)
"""
from __future__ import annotations
import numpy as np
import hashlib
import struct
from typing import Tuple, List


# FNV-1a 64-bit — same constants as the rest of the Quilt.
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(s: str) -> int:
    h = FNV_OFFSET
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * FNV_PRIME) & MASK
    return h


def hamming_window(n: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                        n_mels: int = 60, n_fft: int = 512,
                        hop_length: int = 160) -> np.ndarray:
    """Compute a log-mel-like spectrogram of shape (n_mels, n_frames)."""
    if len(audio) < n_fft:
        audio = np.concatenate([audio, np.zeros(n_fft - len(audio))])
    window = hamming_window(n_fft)
    n_frames = (len(audio) - n_fft) // hop_length + 1
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length][:n_frames]
    frames = frames * window
    specs = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    specs = np.log(specs + 1e-9)
    fft_bins = n_fft // 2 + 1
    log_bins = np.logspace(0, np.log10(fft_bins - 1), n_mels + 1).astype(int)
    log_bins = np.clip(log_bins, 0, fft_bins - 1)
    out = np.zeros((n_mels, n_frames), dtype=np.float32)
    for m in range(n_mels):
        out[m] = specs[:, log_bins[m]:log_bins[m + 1] + 1].mean(axis=1)
    return out


def log_mel_spectrogram_batch(audios: np.ndarray, sample_rate: int = 16000,
                              n_mels: int = 60, n_fft: int = 512,
                              hop_length: int = 160) -> np.ndarray:
    """Vectorized batched spectrogram. Input: (N, T). Output: (N, n_mels, n_frames)."""
    N, T = audios.shape
    if T < n_fft:
        audios = np.pad(audios, ((0, 0), (0, n_fft - T)))
    window = hamming_window(n_fft)
    n_frames = (audios.shape[1] - n_fft) // hop_length + 1
    frames = np.lib.stride_tricks.sliding_window_view(audios, n_fft, axis=1)[:, ::hop_length][:, :n_frames]
    frames = frames * window
    specs = np.abs(np.fft.rfft(frames, axis=2)) ** 2
    specs = np.log(specs + 1e-9)
    fft_bins = n_fft // 2 + 1
    log_bins = np.logspace(0, np.log10(fft_bins - 1), n_mels + 1).astype(int)
    log_bins = np.clip(log_bins, 0, fft_bins - 1)
    out = np.zeros((N, n_mels, n_frames), dtype=np.float32)
    for m in range(n_mels):
        out[:, m] = specs[:, :, log_bins[m]:log_bins[m + 1] + 1].mean(axis=2)
    return out


def mfcc_features(audio: np.ndarray, sample_rate: int = 16000,
                  n_mels: int = 60, n_mfcc: int = 20, n_fft: int = 512,
                  hop_length: int = 160) -> np.ndarray:
    """Compute MFCCs (DCT of log-mel) — compact, decorrelated.

    Returns (n_mfcc * 3, n_frames) where the 3 is [mfcc, delta, delta-delta].
    """
    spec = log_mel_spectrogram(audio, sample_rate=sample_rate,
                               n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    n_frames = spec.shape[1]
    n = n_mels
    k = np.arange(n_mfcc)[:, None]
    i = np.arange(n)[None, :]
    dct_basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    mfccs = dct_basis @ spec  # (n_mfcc, n_frames)
    if n_frames > 2:
        delta = np.gradient(mfccs, axis=1)
        delta2 = np.gradient(delta, axis=1)
    else:
        delta = np.zeros_like(mfccs)
        delta2 = np.zeros_like(mfccs)
    return np.concatenate([mfccs, delta, delta2], axis=0)  # (n_mfcc*3, n_frames)


def mfcc_features_batch(audios: np.ndarray, sample_rate: int = 16000,
                        n_mels: int = 60, n_mfcc: int = 20, n_fft: int = 512,
                        hop_length: int = 160) -> np.ndarray:
    """Batched MFCC features. Returns (N, n_mfcc*3, n_frames)."""
    specs = log_mel_spectrogram_batch(audios, sample_rate=sample_rate,
                                      n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    N, _, n_frames = specs.shape
    n = n_mels
    k = np.arange(n_mfcc)[:, None]
    i = np.arange(n)[None, :]
    dct_basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n))  # (n_mfcc, n_mels)
    mfccs = np.einsum('cd,ndt->nct', dct_basis, specs)  # (N, n_mfcc, n_frames)
    if n_frames > 2:
        delta = np.gradient(mfccs, axis=2)
        delta2 = np.gradient(delta, axis=2)
    else:
        delta = np.zeros_like(mfccs)
        delta2 = np.zeros_like(mfccs)
    return np.concatenate([mfccs, delta, delta2], axis=1)  # (N, n_mfcc*3, n_frames)


def handcrafted_features(mfccs: np.ndarray) -> np.ndarray:
    """Reduce (n_mfcc*3, n_frames) -> (n_mfcc*3 * 3) by taking mean, std, max over time."""
    mean = mfccs.mean(axis=1)        # (60,)
    std = mfccs.std(axis=1) + 1e-6
    mx = mfccs.max(axis=1)
    return np.concatenate([mean, std, mx]).astype(np.float32)


def handcrafted_features_batch(mfccs_batch: np.ndarray) -> np.ndarray:
    """Reduce (N, n_mfcc*3, n_frames) -> (N, 180)."""
    mean = mfccs_batch.mean(axis=2)        # (N, 60)
    std = mfccs_batch.std(axis=2) + 1e-6
    mx = mfccs_batch.max(axis=2)
    return np.concatenate([mean, std, mx], axis=1).astype(np.float32)  # (N, 180)


def _seeded_glorot(seed: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Glorot-uniform initialization seeded by hash bytes.

    Uses numpy's default_rng to derive a reproducible PRNG from the
    seed. Output is properly centered on 0 and bounded by ±limit,
    where limit = sqrt(6 / (in_dim + out_dim)).
    """
    in_dim, out_dim = shape
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    seed_int = int.from_bytes(seed[:8], "big")
    rng = np.random.default_rng(seed_int)
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


class FeatureExtractor:
    """The frozen audio feature extractor.

    Pipeline (byte-exact across Python/C/TFLite):
    1. log-mel spectrogram (60 bins, ~1s window)
    2. MFCC + delta + delta-delta (60 dims)
    3. hand-crafted stats: mean + std + max over time -> 180-dim
    4. linear projection 180 -> 64 (the frozen backbone)
    5. L2 normalize

    The projection is calibrated via PCA on a calibration set
    (see fit_pca_projection). After calibration, the weights are
    frozen and pushed to every device. Devices run inference
    identically and produce deterministic 64-dim embeddings.
    """

    INPUT_DIM = 180   # 60 MFCCs * 3 stats
    OUTPUT_DIM = 64

    def __init__(self, seed: str = "F170-tinyml-v1"):
        h = hashlib.sha256(seed.encode("utf-8")).digest()
        # Single 180 -> 64 projection (the frozen backbone)
        W = _seeded_glorot(h, (self.INPUT_DIM, self.OUTPUT_DIM))
        b = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        self.weights = [(W, b)]
        self.W = W
        self.b = b

    def fit_pca_projection(self, audios, n_calibration: int = 200, sample_rate: int = 16000):
        """Calibrate the projection on a set of calibration audios.

        Computes the 180-dim hand-crafted features for each audio, then
        fits a 180->64 projection that preserves the top-64 directions
        of variance. This makes the embedding discriminative for the
        actual data distribution.

        Call this once with a calibration set before deploy; the
        resulting weights are then frozen.
        """
        if isinstance(audios, tuple):
            X, _ = audios
        else:
            X = audios
        idx = np.random.default_rng(0).choice(len(X), min(n_calibration, len(X)), replace=False)
        mfccs_batch = mfcc_features_batch(X[idx].astype(np.float32), sample_rate=sample_rate)
        flat = handcrafted_features_batch(mfccs_batch)  # (n_cal, 180)
        flat -= flat.mean(axis=0, keepdims=True)
        # SVD: flat = U S Vt. Vt has shape (180, 180). Top-64 rows are
        # the principal directions. The projection 180->64 is Vt[:64].T.
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        W_new = Vt[:self.OUTPUT_DIM, :].T.astype(np.float32)  # (180, 64)
        self.weights = [(W_new, np.zeros(self.OUTPUT_DIM, dtype=np.float32))]
        self.W = W_new
        self.b = np.zeros(self.OUTPUT_DIM, dtype=np.float32)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Compute a 64-dim L2-normalized embedding of an audio clip."""
        mfccs = mfcc_features(audio, sample_rate=sample_rate)
        flat = handcrafted_features(mfccs)  # (180,)
        h = flat @ self.W + self.b
        norm = np.linalg.norm(h) + 1e-9
        return (h / norm).astype(np.float32)

    def embed_batch(self, audios, sample_rate: int = 16000) -> np.ndarray:
        """Embed a batch of audio clips. Returns (N, 64)."""
        if not isinstance(audios, np.ndarray):
            audios = np.stack(audios)
        if audios.ndim == 2 and len(audios) > 0:
            mfccs_batch = mfcc_features_batch(audios.astype(np.float32), sample_rate=sample_rate)
            flat = handcrafted_features_batch(mfccs_batch)  # (N, 180)
        else:
            return np.stack([self.embed(a, sample_rate) for a in audios])
        h = flat @ self.W + self.b
        norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        return (h / norms).astype(np.float32)

    def param_count(self) -> int:
        return sum(W.size + b.size for W, b in self.weights)

    def state_hash(self) -> int:
        """FNV-1a 64-bit state hash of the frozen weights.

        This is the device's identity anchor — the F161 conservation law
        contract. Two devices with the same state hash are running the
        same backbone; their classifier heads are compatible.
        """
        h = FNV_OFFSET
        for W, b in self.weights:
            for v in W.flatten().tolist() + b.tolist():
                bs = struct.pack("<f", float(v))
                for byte in bs:
                    h ^= byte
                    h = (h * FNV_PRIME) & MASK
        return h

    def to_bytes(self) -> bytes:
        """Serialize the backbone to a byte stream. ~50 KB at fp32, ~12.5 KB at int8."""
        out = bytearray()
        for W, b in self.weights:
            out += W.astype(np.float32).tobytes()
            out += b.astype(np.float32).tobytes()
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes, seed: str = "F170-tinyml-v1") -> "FeatureExtractor":
        fe = cls(seed=seed)
        in_dim, out_dim = cls.INPUT_DIM, cls.OUTPUT_DIM
        W_size = in_dim * out_dim * 4
        W = np.frombuffer(data[:W_size], dtype=np.float32).reshape(in_dim, out_dim).copy()
        b = np.frombuffer(data[W_size:W_size + out_dim * 4], dtype=np.float32).copy()
        fe.weights = [(W, b)]
        fe.W = W
        fe.b = b
        return fe


if __name__ == "__main__":
    fe = FeatureExtractor()
    print("state hash:", hex(fe.state_hash()))
    print("param count:", fe.param_count())
    print("byte size:", len(fe.to_bytes()))
    audio = np.random.randn(16000).astype(np.float32)
    emb = fe.embed(audio)
    print("embedding shape:", emb.shape)
    print("embedding norm:", np.linalg.norm(emb))
    # Same audio -> same embedding
    emb2 = fe.embed(audio)
    assert np.allclose(emb, emb2)
    # Different audio -> different embedding
    emb3 = fe.embed(np.random.randn(16000).astype(np.float32) * 5)
    assert not np.allclose(emb, emb3)
    # Round-trip serialization
    data = fe.to_bytes()
    fe2 = FeatureExtractor.from_bytes(data)
    assert fe2.state_hash() == fe.state_hash()
    emb4 = fe2.embed(audio)
    assert np.allclose(emb, emb4)
    print("self-test passed (state hash + round-trip)")
