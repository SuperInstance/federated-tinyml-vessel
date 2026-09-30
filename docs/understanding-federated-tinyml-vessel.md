# Understanding federated-tinyml-vessel

## 1. In one breath

A Python reference implementation of federated on-device learning in which every device shares one frozen audio-feature backbone, trains only a 325-parameter (1,300-byte) classifier head, and sends that head to an aggregator that averages the heads — with the head's byte layout and FNV-1a state hash defined so that Python, C, Rust and JavaScript can check they hold the same model.

## 2. Why it exists

The scenario in `README.md` is a fishing vessel more than 50 nm offshore: a wrist device, a hydrophone and a sounder each hear different audio, connectivity is poor, and the raw audio should not be shipped anywhere. Before this you had two options: ship audio to a central trainer (bandwidth and privacy cost), or train a whole network on each device (memory cost on a microcontroller).

The repo's answer is to split the model. The feature extractor is fixed once and never changes, so it needs no training or communication. What changes is a linear head small enough to send every round and to run on an ESP32-class chip (`c_port/f170_head.c` prints `sizeof head: 1308 bytes`). The second problem it addresses is "are two devices actually running the same model?" Floating-point models are easy to confuse across languages, so the head has a fixed byte layout and a 64-bit hash that any language can recompute.

The data is synthetic (`simulator.py`) unless you run the ESC-50 or Whisper scripts, which need downloads or an API. Treat results as a test of the pipeline, not of fishing-vessel acoustics.

## 3. The mental model

Five nouns.

**Backbone** (`feature_extractor.py`, class `FeatureExtractor`). Audio → log-mel → MFCC (+ delta, delta-delta) → mean/std/max over time (180 numbers) → a linear 180→64 projection → L2-normalize. The projection is the "frozen" part. At construction it is Glorot-random from a SHA-256 of a seed string; `fit_pca_projection` then replaces it with the top-64 principal directions of a calibration set. After that it is never trained again. It has 180·64+64 = 11,584 parameters and serializes to 46,336 bytes (`python3 feature_extractor.py`). The README says 12,544 params / 50 KB; the code says 11,584 / 46,336 — trust the code.

**Head** (`classifier_head.py`, `ClassifierHead`). A 64×5 weight matrix plus 5 biases, fp32, softmax on top. It is the only thing that learns, by plain SGD on cross-entropy (no momentum). Its wire form is `to_bytes()`: weights row-major, then biases, all little-endian float32 — 1,300 bytes.

**State hash** (`ClassifierHead.state_hash`). FNV-1a 64-bit over exactly those 1,300 bytes. It is the head's identity: same hash ⇔ same bytes. The backbone has its own `state_hash` over its own bytes.

**Round / FedAvg** (`federated.py`, `average_heads`). Broadcast the global head; each device copies it and does `local_steps` SGD steps on its own shard; the aggregator takes the average of the returned heads weighted by shard size. `non_iid_split` gives each device a Dirichlet-skewed slice (low `alpha` = each device sees mostly a few classes).

**Substrate**. A language/runtime that holds the head: NumPy, C (`c_port/`), Rust (`verify_rust/`), JS. Substrates do not need to share code, only the byte layout and hash.

How they relate: the backbone turns audio into a 64-vector everywhere identically; the head maps that vector to 5 probabilities; a round moves heads (never audio, never the backbone) between devices and the aggregator; the state hash is how you confirm a head that crossed a wire or a language boundary is the one you meant.

What this does *not* mean: only the serialized bytes and their hash are cross-checked in this repo (see the blueprint article). Training itself is not bit-reproducible between substrates or even between two Python runs — see §6.

## 4. Walkthrough

Install (only NumPy is needed; the sandbox did not have it):

```
$ pip install numpy
$ python3 feature_extractor.py
state hash: 0x9627430ac5be8c8d
param count: 11584
byte size: 46336
embedding shape: (64,)
embedding norm: 1.0
self-test passed (state hash + round-trip)

$ python3 classifier_head.py
initial state hash: 0x9bb0248700b52f77
  step 0: loss=5.9699, state_hash=0x42a0d107dea1d7c0
  step 25: loss=5.5193, state_hash=0xede63c14eef7da23
  step 50: loss=5.8898, state_hash=0x5ad433179459560c
  step 75: loss=6.1095, state_hash=0xe81086f8ae9a3c41
federated avg state hash: 0x4f9a64148b9993e8
self-test passed
```

(The loss there is on random noise labels, so it does not fall.) Now a federated run on the synthetic data, 15 rounds, 5 devices, 60 samples/class:

```
$ python3 federated.py
Test 2: Non-IID split (alpha=0.3, very skewed per device)
  round   0: loss=3.5907  test_acc=0.233  head=0x5b5929619fa7b3fa
  ...
  round  14: loss=3.1963  test_acc=0.717  head=0xe3cf44f3959e7bff
  final test accuracy: 0.717

  FROZEN BACKBONE state hash: 0xf130bacfaecc3b63
  IID final head state hash:  0xba9c61e33916fc6c
  NonIID final head state hash: 0xe3cf44f3959e7bff
```

Running it a second time printed the same backbone hash (`0xf130…`) but different head hashes (`0xca5aad92bb311a66`, `0x002777db359f7553`). That is expected; §6 explains it. The longer study:

```
$ python3 study.py          # ~32 s here
alpha   devices   rounds  final_acc   loss_drop   hash
10.00   5         40      1.000       0.925       0x0dbdaf04bca66e2b   IID-ish (alpha=10)
1.00    5         40      1.000       0.831       0x6c75f39a5f9b45c7   Mildly skewed (alpha=1)
0.50    5         40      1.000       0.893       0xbed9be9f3ac84eea   Heavily skewed (alpha=0.5)
0.30    5         40      1.000       0.853       0xac78627eb51cd6bf   Very skewed (alpha=0.3)
0.30    10        40      1.000       0.783       0xdc340903fe57ed6a   Very skewed, 10 devices
0.30    3         40      0.825       0.718       0x7e81f0e052fb1078   Very skewed, 3 devices
```

In that run five configurations reached 1.000 and the 3-device one 0.825. The synthetic classes are easy to separate, so 1.000 says the pipeline is wired correctly, not that a real hydrophone would score that. The head hashes will differ on your machine's run.

The C port self-test:

```
$ gcc -o f170 c_port/f170_head.c -lm && ./f170
sizeof head: 1308 bytes (1.3 KB at fp32)
Zero head state hash: 0x87098d0fbca42d35
...
Serialized 1300 bytes
Round-trip state hash match: 0x88006bf39919eca8
FNV-1a('hello') = 0xa430d84680aabd0b (expected 0xa430d84680aabd0b)

All tests passed.
```

## 5. The contract

**Inputs.** Audio as float32 arrays at 16 kHz; a head as 64×5 float32 + 5 float32 biases.

**Outputs.** A 64-d unit-norm embedding; 5 class probabilities (`silence, normal, wind, net_haul, line_tangle`, `simulator.CLASSES`); a 1,300-byte head; a 16-hex-digit state hash. Wire envelope: JSON with `schema: "f170-head-v1"`, `W`, `b`, `round`, `state_hash` (`ClassifierHead.to_json_envelope`, consumed by `aggregator.py`).

**Invariants.**
- Head bytes = float32 LE, weights row-major (`W[d*5+c]`) then biases; hash = FNV-1a 64 over those bytes, in that order.
- The backbone is identical on all devices in a run; only the head is exchanged.
- FedAvg weights = per-device sample counts; a device with zero samples skips the round (`federated.py`, commit 9d38566).

**Receipt (what I ran on this checkout).** The repo has no `tests/` directory and no pytest suite; verification is the self-tests in each module. All of these passed: `feature_extractor.py`, `classifier_head.py`, `c_port/f170_head.c` ("All tests passed."), `gesture.py` ("self-test OK"), `federated.py`, `study.py` (ran to completion). Cross-substrate hash agreement over two heads and four implementations is in the blueprint article. I did not run `aggregator.py`, `device_client.py`, `tt_int4.py`, the ESC-50 or Whisper scripts, or the JS/Rust crates named in the README that are not in this repo.

## 6. Failure modes / scars

- **Training is not deterministic.** `device_local_train` calls `np.random.default_rng()` with no seed, so batches differ each run. Two runs of `federated.py` gave different final head hashes (above). Fix if you need a reproducible run: pass a seed through (a code change; not done here). What *is* reproducible: the backbone hash, data generation (seeded), and any head you serialize once.
- **Two different "backbone hashes".** README/`study.py` quote `0x9627430ac5be8c8d`; that is the hash of the seed-initialized backbone before PCA calibration, and `study.py` prints it as a hardcoded string. `federated.py` calls `fit_pca_projection`, so the backbone it actually uses hashes to `0xf130bacfaecc3b63`. Compare hashes you compute, not ones in prose.
- **README multi-substrate table (`0x5fd69fcc4833d9fc`) is not reproducible from this repo**: no code here produces that head. The verification in the blueprint uses heads that the repo can regenerate.
- **Stale README numbers**: backbone size (above), and "95–100% across 6 configurations" — my 15-round `federated.py` run ended at 71.7%, `study.py` (40 rounds) at 82.5–100%.
- **Hard-coded paths**: `robust_aggregator.py`, `tensor_train_head.py`, `tt_int4.py`, `heterogeneous_backbone.py` do `sys.path.insert(0, "/workspace/tinyml-federated")`. They work from the repo root only because the same modules are found in the cwd; the extra path is dead weight or a wrong-copy hazard.
- **Extras have external needs**: `esc50_loader.py` downloads data; `whisper_backbone.py` calls a hosted API. Neither ran here.
- **Committed build output**: `__pycache__`, `verify_rust/target/`, `dist/` are tracked in git, so running anything dirties `git status`.
- **`setup.py` `py_modules`** lists only five modules, so an installed wheel lacks `gesture.py`, which `federated.py` imports.

## 7. How it composes

- **Aggregator over HTTP**: `aggregator.py` (POST `/head`, GET `/global`, `/stats`, `/reset`) and `device_client.py` speak the JSON envelope above.
- **Smaller heads**: `quantize.py` (int8 333 B, int4 173 B; its self-test shows accuracy 9/50 → 8/50 for int4 on its own random head), `tensor_train_head.py` / `tt_int4.py` (TT-cores head), C ports in `c_port/` and `c_port_int4/`.
- **Robustness and variants**: `robust_aggregator.py` (trimmed mean, median, Krum), `mode_wise_federation.py`, `heterogeneous_backbone.py` (different backbones, same 64-d head interface), `federated_real*.py`/`federated_whisper.py` for real audio.
- **Convergence read-out**: `gesture.py` adds `convergence_geometry` (arc length, bending, twist) to every `federated_train` result. The README itself says the non-IID link is an unvalidated hypothesis.
- **Downstream**: `DOWNSTREAM-SCRAPCRAFT-EDU.md` maps this design to a browser-game engine that uses determinism as its version of the state hash.

## 8. Where to look next

- `docs/blueprint-byte-exact-across-substrates.md` — reproduce and verify a head across Python/C/Rust/Node.
- `classifier_head.py` and `federated.py` — the whole learning loop is ~470 lines.
- `c_port/f170_head.c` — the on-device side, including its self-test.
