"""esc50_loader.py — Load the ESC-50 dataset's vessel-relevant categories.

ESC-50 has 50 categories of environmental sounds. We extract the 5
vessel-relevant ones (engine, sea_waves, wind, pouring_water,
water_drops) to use as a REAL audio benchmark for F170.

This proves the F170 pipeline works on real audio, not just synthetic.
"""
from __future__ import annotations
import os
import csv
import wave
import struct
import urllib.request
import numpy as np
from typing import Tuple, Dict, List


# 5 vessel-relevant categories in ESC-50
VESSEL_CATEGORIES = ['engine', 'sea_waves', 'wind', 'pouring_water', 'water_drops']

# Same as the F170 simulator, but real names
CLASS_NAMES = ['engine', 'sea_waves', 'wind', 'pouring_water', 'water_drops']


def download_esc50_subset(categories: List[str] = None, target_dir: str = '/tmp/esc50') -> Dict[str, List[str]]:
    """Download the 5 vessel-relevant categories of ESC-50.

    Returns a dict mapping category -> list of local filenames.
    Total: 5 cats × 40 samples = 200 files.
    """
    if categories is None:
        categories = VESSEL_CATEGORIES
    os.makedirs(f"{target_dir}/audio", exist_ok=True)
    csv_path = f"{target_dir}/esc50.csv"
    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv",
            csv_path
        )

    samples_by_cat = {c: [] for c in categories}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            if cat in categories:
                fn = row['filename']
                local_path = f"{target_dir}/audio/{fn}"
                if not os.path.exists(local_path):
                    url = f"https://github.com/karolpiczak/ESC-50/raw/master/audio/{fn}"
                    try:
                        urllib.request.urlretrieve(url, local_path)
                    except Exception as e:
                        print(f"  Failed to fetch {fn}: {e}")
                        continue
                samples_by_cat[cat].append(local_path)

    return samples_by_cat


def load_wav(path: str, target_sr: int = 16000, duration_sec: float = 0.5) -> np.ndarray:
    """Load a WAV file, resample to target_sr, truncate/pad to duration_sec.

    Returns a float32 mono array of length target_sr * duration_sec.
    """
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    # Convert to int16
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")
    # Mono mix
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    # Resample (simple linear interpolation for now)
    if sr != target_sr:
        n_out = int(len(samples) * target_sr / sr)
        samples = np.interp(
            np.linspace(0, len(samples), n_out),
            np.arange(len(samples)),
            samples
        ).astype(np.float32)
    # Truncate or pad to duration_sec
    target_len = int(target_sr * duration_sec)
    if len(samples) > target_len:
        # Take a random crop (for diversity)
        start = np.random.randint(0, len(samples) - target_len + 1)
        samples = samples[start:start + target_len]
    elif len(samples) < target_len:
        samples = np.concatenate([samples, np.zeros(target_len - len(samples), dtype=np.float32)])
    return samples


def load_dataset(target_sr: int = 16000, duration_sec: float = 0.5,
                target_dir: str = '/tmp/esc50') -> Tuple[np.ndarray, np.ndarray]:
    """Load the vessel-relevant subset of ESC-50.

    Returns:
      X: (N, target_sr * duration_sec) float32 array
      y: (N,) int64 array of class labels
    """
    if not os.path.exists(f"{target_dir}/audio"):
        print("Downloading ESC-50 vessel subset...")
        download_esc50_subset(target_dir=target_dir)
    samples = download_esc50_subset(target_dir=target_dir)
    Xs, ys = [], []
    for class_id, cat in enumerate(VESSEL_CATEGORIES):
        for path in samples[cat]:
            try:
                audio = load_wav(path, target_sr=target_sr, duration_sec=duration_sec)
                Xs.append(audio)
                ys.append(class_id)
            except Exception as e:
                print(f"  Skipped {path}: {e}")
    return np.stack(Xs), np.array(ys, dtype=np.int64)


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y distribution: {np.bincount(y)}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  class {c} ({name}): mean={X[y == c].mean():.4f}, std={X[y == c].std():.4f}")
