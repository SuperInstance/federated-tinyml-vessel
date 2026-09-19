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
| `gesture.py` | ~7,000 | 200 | Convergence geometry — the head's path read as a gesture |
| `study.py` | 1,701 | 50 | The 6-experiment comparison study |
| `README.md` | this | - | This file |

Total: ~35 KB of Python, ~830 lines, polyformal.

## The state hashes

- Frozen backbone: `0x9627430ac5be8c8d` (constant across all experiments)
- Head hashes vary per round and per experiment (see `study.py` output)

## Convergence geometry

The final accuracy is a *point*. But the global head **moved** to get there —
each round, FedAvg takes a step through the 325-D parameter space, and those steps
trace a path. [`gesture.py`](gesture.py) reads the shape of that path, so
`federated_train(...)` now returns a `convergence_geometry` block:

```python
r = federated_train(n_devices=5, n_rounds=12, alpha=0.3, verbose=False)
r["convergence_geometry"]
# {'rounds': 13, 'arc_length': ...,   # total distance the head travelled
#  'final_step': ...,                 # how far the last round moved (has it settled?)
#  'bending_energy': ..., 'mean_bending': ...,  # oscillation of the descent
#  'twist_energy': ..., 'planarity': ...}       # does convergence open new dimensions?
```

Read **order by order**: `arc_length`/`heading` (how far, and where it's going —
the `d_mu`), `bending_energy` (curvature — turning *within* a plane, i.e. an
aggregation that oscillates vs. one that descends smoothly), and `twist_energy`
(torsion — turning *out of* the plane, into a fresh direction of parameter
space). This is the SuperInstance fleet's shared "abstraction as gesture" reading
— a tensor approximates a function; we approximate the *shape of the motion* —
the same three orders that read notes ([musician-soul](https://github.com/SuperInstance/musician-soul)),
rooms ([elephant](https://github.com/SuperInstance/elephant)), conversations
([tensor-midi](https://github.com/SuperInstance/tensor-midi)) and cell state
([quilt](https://github.com/SuperInstance/quilt)). *The property is in the twist*
([twist-engine](https://github.com/SuperInstance/twist-engine)).

Honest note: whether twist tracks non-IID skew is an open hypothesis — a quick
seed sweep at moderate skew did not show a clear effect. Treat these as an
*observable* of the convergence path, not a validated non-IID detector. Run
`python gesture.py` for the self-test.

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

## F170 R&D v0.2: Quantization + Aggregator + Multi-substrate

This version adds three new R&D directions:

### 1. INT4 quantization (`quantize.py`)
The head can be quantized to 4-bit weights (2 per byte) and 8-bit biases:
- **fp32**: 1300 bytes (1.27 KB)
- **int8**: 333 bytes (0.33 KB) — 25.6% of fp32
- **int4**: 173 bytes (0.17 KB) — 13.3% of fp32, **7.5x smaller**

All three maintain 100% test accuracy on the synthetic data.

### 2. Real aggregator server (`aggregator.py`)
A working HTTP aggregator that:
- Accepts head uploads from devices via POST /head
- Runs FedAvg when 3+ devices have submitted
- Serves the current global head via GET /global
- Tracks history via GET /stats
- Supports reset via POST /reset

Run: `python3 aggregator.py --port 8765`
Demo: `python3 aggregator.py --demo`

### 3. On-device client (`device_client.py`)
A simulated device that:
- Fetches the current global head
- Runs local SGD on its own audio data
- Uploads the updated head
- Repeats

Run 5 devices in parallel: `for i in 0 1 2 3 4; do python3 device_client.py --device-id device-$i --preferred-class $i --aggregator http://localhost:8765 & done`

### 4. C port (`c_port/f170_head.c` and `c_port/f170_head_int4.c`)
Byte-exact TFLite-Micro-compatible C port. Both fp32 (1.3 KB) and int4 (0.17 KB) versions. All 4 substrates (Python/JS/C/Rust) compute the same FNV-1a 64-bit state hash for the same head bytes.

### 5. Rust crate (`quilt-rust/crates/federated-tinyml`)
11/11 tests pass. The Rust port with FNV-1a verified byte-exact.

## Multi-substrate polyformalism

| Substrate | State hash (test) |
|---|---|
| **Python** | `0x5fd69fcc4833d9fc` |
| **JavaScript** | `0x5fd69fcc4833d9fc` |
| **C** | `0x5fd69fcc4833d9fc` |
| **Rust** | `0x5fd69fcc4833d9fc` |

A head trained in any substrate is bit-identical when loaded in any other.
