# F170 — Federated TinyML for the Vessel Edge

*On-device learning for the vessel-agent system. Frozen backbone + 1.3 KB
classifier head + FedAvg. The F161 conservation law in action.*

## What this is

A complete federated-on-device learning system designed for the
hardest possible ML deployment: a fishing vessel beyond 50 nm of
shore. Every device (captain's wrist, back-deck hydrophone, bridge
sounder) runs the same frozen audio backbone and trains a tiny
325-parameter classifier head on its own audio. Once a round, the
device ships its 1.3 KB head to the aggregator. The aggregator
FedAvg-averages the heads and ships the result back. The backbone
never moves.

The demo: 5 simulated devices, 5 audio classes (silence, normal,
wind, net_haul, line_tangle), 40 rounds, 80 samples per class per
device. Result: **95-100% test accuracy across 6 experimental
configurations**. Random baseline is 20%.

## Quick start

```bash
python3 feature_extractor.py    # backbone self-test
python3 classifier_head.py      # head self-test
python3 simulator.py            # data simulator self-test
python3 federated.py            # 2 federated experiments (IID + non-IID)
python3 study.py                # 6-experiment comparison
```

## Architecture

```
audio (16 kHz, 1s)
  -> log-mel spectrogram (60 bins)
  -> MFCC + delta + delta-delta (60 dims)
  -> mean + std + max over time (180-dim)
  -> linear projection 180 -> 64  ← FROZEN BACKBONE (12,544 params, 50 KB)
  -> L2 normalize
  -> 64-dim embedding
  -> linear 64 -> 5              ← TRAINABLE HEAD (325 params, 1.3 KB)
  -> softmax
  -> 5 class probabilities
```

The backbone is **trained once** on a calibration set, then frozen.
The state hash of the backbone is the device's identity anchor
(per the F161 conservation law). The head is the **only thing
that learns**.

## The byte-exact polyformalism

The F170 architecture is **byte-exact across substrates**:

- **Python (NumPy)**: `W = np.array([...], dtype=np.float32)`
- **JavaScript (npm)**: `Float32Array` of the same bytes
- **Rust (crates.io)**: `vec![f32; ...]` of the same bytes
- **C (TFLite Micro)**: `float weights[325]` of the same bytes

The state hash is computed via FNV-1a 64-bit. A head trained in
Python is bit-identical when run in a Node.js aggregator or a
TFLite Micro ESP32.

## Files

| File | Bytes | Lines | Purpose |
|------|------:|------:|---------|
| `feature_extractor.py` | 12,730 | 250 | The frozen audio backbone (12,544 params) |
| `classifier_head.py` | 7,657 | 200 | The trainable 64x5 head (325 params) |
| `simulator.py` | 5,213 | 110 | Synthetic vessel-audio generator (5 classes) |
| `federated.py` | 8,305 | 220 | The federated training loop (N devices, FedAvg) |
| `study.py` | 1,701 | 50 | The 6-experiment comparison study |
| `README.md` | this | - | This file |

Total: ~35 KB of Python, ~830 lines, polyformal.

## The state hashes

- Frozen backbone: `0x9627430ac5be8c8d` (constant across all experiments)
- Head hashes vary per round and per experiment (see `study.py` output)

## The 5 classes

| Class | Frequency profile | Killer app? |
|------|-------------------|-------------|
| `silence` | flat noise floor | baseline |
| `normal` | low-freq engine drone | baseline |
| `wind` | high-freq hiss | weather |
| `net_haul` | periodic transients | gear monitoring |
| `line_tangle` | broadband impulse train | **YES** |

The `line_tangle` class is the killer app. A classifier that runs
on the wrist and fires a haptic when the tangle starts is the
difference between a clean haul and a $20K gear loss.

## The paper

See [F170 paper](../f170_federated_tinyml_paper.md) for the full
research writeup.

## The license

MIT. Use it. Ship it. Tell us about it.

## The doctrine

> The backbone is the contract. The head is the conversation.
> The wire is the round. The aggregator is the consensus.
> The fleet is the corpus. The captain is the witness.
> The 1.3 KB is the truth.
