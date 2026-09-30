# Blueprint: a byte-exact head across Python, C, Rust and Node

## 1. In one breath

Serialize a trained 64×5 classifier head to its 1,300-byte little-endian fp32 form, then have four independent implementations (NumPy, C, Rust, Node) compute the FNV-1a 64-bit hash of those bytes and confirm all four print the same value.

## 2. Why it exists

Before this you had to trust that "the model on the device is the model we trained". Floats survive JSON round-trips, endianness changes, `float64` upcasts and truncated transfers in ways that look fine and are not. The hash is a cheap, language-independent check: it is 16 hex digits, needs no library (an FNV-1a loop is five lines in any language), and changes if a single bit of the head changes. Use it when a head crosses a boundary — trainer → aggregator → flash image on a microcontroller — and you want a pass/fail on "same bytes".

What it checks: byte identity of a serialized head. What it does not check: that training gives the same weights on different substrates (it does not — see §6), or that forward passes agree numerically (I did not test that; `c_port/f170_head.c` uses `expf`, NumPy uses its own `exp`).

## 3. The mental model

- **Head bytes**: `ClassifierHead.to_bytes()` (`classifier_head.py`) = weights `(64,5)` row-major float32 LE, then 5 biases float32 LE. 320+5 = 325 floats = 1,300 bytes. This file is the artifact everything else is judged against.
- **State hash**: FNV-1a 64 (offset `0xCBF29CE484222325`, prime `0x100000001B3`) over the head bytes. `ClassifierHead.state_hash()` feeds `struct.pack("<f", v)` for each value; that yields the same byte stream as `to_bytes()`, so hashing the *file* equals the state hash.
- **Verifiers**: one per substrate, each reading the same file — `c_port/verify_py_compat.c` (hashes W then b as floats), `verify_rust/src/main.rs` (hashes raw file bytes), and a Node snippet (below; the repo has no JS source, so I wrote one inline).
- **Receipt**: the set of printed hashes. Pass = all equal to the Python one. A `sha256sum` of the file lets you also confirm the *file* is unchanged between machines.

Relationship: Python is the producer and reference; the other three are consumers that recompute the hash independently. Agreement ⇒ they read the same 1,300 bytes the same way (same order, width and endianness).

## 4. Walkthrough

Requirements: Python 3 + NumPy, gcc, cargo (offline build is fine; `verify_rust` has no dependencies), Node. From the repo root:

```bash
set -e
W=$(mktemp -d)
python3 - "$W/head.bin" <<'PY'
import sys, numpy as np
from classifier_head import ClassifierHead
h = ClassifierHead(seed=42); rng = np.random.default_rng(0)
for _ in range(100):
    h.sgd_step(rng.standard_normal((32, 64)).astype(np.float32), rng.integers(0, 5, 32), 0.01)
open(sys.argv[1], "wb").write(h.to_bytes())
print("Python state hash: 0x%016x  (%d bytes)" % (h.state_hash(), len(h.to_bytes())))
PY
gcc -O2 -w -o "$W/verify_c" c_port/verify_py_compat.c
"$W/verify_c" "$W/head.bin"
(cd verify_rust && cargo build --release --offline -q) && verify_rust/target/release/verify_rust "$W/head.bin"
node -e 'const b=require("fs").readFileSync(process.argv[1]);let h=0xcbf29ce484222325n;for(const x of b){h^=BigInt(x);h=(h*0x100000001b3n)&0xffffffffffffffffn}console.log("Node state hash: 0x"+h.toString(16).padStart(16,"0"))' "$W/head.bin"
```

Actual output on this checkout:

```
Python state hash: 0x21c95a084480bdc9  (1300 bytes)
C state hash: 0x21c95a084480bdc9
Rust state hash: 0x21c95a084480bdc9
Node state hash: 0x21c95a084480bdc9
```

The head is "trained" on seeded random data with a seeded RNG, so it is deterministic: I generated it twice in separate processes and `cmp` reported the files identical (sha256 `14db769000aaf95a90c7966fbcc3fd49379d7e87604d71b57fc785a6eefdc320`). For an untrained head, `ClassifierHead(seed=42)` gives `0x9bb0248700b52f77`, and C, Rust and Node all printed that too.

**Note:** `cargo build` rewrites files under `verify_rust/target/`, which are tracked in git. Run `git checkout -- verify_rust/target` afterwards, or build with `CARGO_TARGET_DIR=$W/target`.

**Reproducing a head, not just checking one.** To reproduce byte-for-byte on another machine, ship either the `.bin` (compare with `sha256sum`) or regenerate it with the script above — same NumPy `default_rng` streams gave identical output between my two runs, but I have not tested across NumPy versions or CPUs. If you regenerate, prefer comparing the hash on both sides rather than assuming.

**The negative controls** (all on the trained head above; the mismatch is the point):

```
one bit flipped in byte 0     -> C 0x51c7e6d3b7b9810c   Node 0x51c7e6d3b7b9810c   (differs from 0x21c95a084480bdc9)
float64 instead of float32    -> 2600 bytes, Node 0xe4b82acb115a8ffc   (on the seed-42 head; expected 0x9bb0248700b52f77)
big-endian float32            -> Node 0x15843778b64cccfb               (same head; expected 0x9bb0248700b52f77)
```

## 5. The contract

**Inputs.** A file of exactly 1,300 bytes: 320 float32 weights (row-major, index `d*5+c`), then 5 float32 biases, little-endian.

**Outputs.** One line per verifier, `<Substrate> state hash: 0x<16 hex>`.

**Invariants.**
1. Hash = FNV-1a 64 over the bytes in that order; multiplication wraps mod 2^64 (Rust uses `wrapping_mul`, Node masks a BigInt, C uses `uint64_t`, Python masks).
2. Equal hashes ⇒ equal bytes with overwhelming but not cryptographic certainty. FNV-1a is not collision-resistant against an adversary; use `sha256sum` when tampering matters.
3. Length matters: any file that is not 1,300 bytes is not a head.

**Receipt.** Two heads (untrained seed 42: `0x9bb0248700b52f77`; trained: `0x21c95a084480bdc9`) × four implementations = 8/8 hashes matching the Python value, three negative controls that changed the hash, and the C self-test (`gcc c_port/f170_head.c -lm`) printing "All tests passed." including the FNV-1a("hello") = `0xa430d84680aabd0b` known-answer check. That known-answer check is a good first thing to run on any new port.

## 6. Failure modes / scars

- **Wrong dtype or endianness** changes the hash (see controls). Cause: `weights.astype(np.float64)`, or `>f4`. Fix: always go through `to_bytes()`, which forces `float32`, little-endian on the machines tested (x86-64).
- **Truncated or oversized file.** `verify_py_compat.c` ignores `fread`'s return value (compiler warns; I built with `-w`). A short file leaves the tail of its buffers uninitialized, so it prints a hash that may look plausible. I truncated to 1,296 bytes: Node and C printed different hashes from each other and from the original. Fix: check `wc -c` = 1300 first. The Rust and Node verifiers hash whatever length they get, so they also do not enforce 1,300.
- **Different "hash" for the same idea.** `feature_extractor.py`'s `state_hash` covers the backbone (11,584 floats), `ClassifierHead.state_hash` the head. The tensor-train and int4 variants hash their own packed bytes (`tt_int4.py`, `quantize.py` reconstructed heads hash differently from the fp32 head — `quantize.py` prints `0xc2022efb0625d7ab` for the original and different values for int8/int4 reconstructions). Compare like with like.
- **You cannot reproduce a hash by re-running federated training.** `federated.py` draws minibatches from an unseeded RNG, so its final head hashes differ per run (two runs gave `0xba9c61e33916fc6c` and `0xca5aad92bb311a66` for the same IID config). Export the head you care about and verify *that file*. Training reproducibility would need a seed threaded into `device_local_train` (code change; out of scope here).
- **README's table** (`0x5fd69fcc4833d9fc` on all four substrates) can't be regenerated from files in this repo. It should be read as an example, not a fixture.
- **Signed NaNs / -0.0.** Hashing is over bit patterns, so `-0.0` and `0.0` hash differently even though they compare equal. I did not test this; it follows from the byte-level definition.

## 7. How it composes

- Put the check at every hand-off: after `aggregator.py` emits a global head (its envelope carries `state_hash`), after flashing `f170_head_t` on a device (`f170_head_from_bytes` then `f170_head_state_hash`), and in CI for any new port.
- `to_json_envelope()` / `from_json_envelope()` carry the hash alongside `W`/`b`; recompute from `W`/`b` and compare rather than trusting the field. JSON goes through decimal text, so it is the place a float can drift — I did not test that path.
- The int4 C port (`c_port/f170_head_int4.c`, `c_port_int4/f175_tt_int4.c`) has its own packed layout; the same recipe applies over the packed bytes but the verifier here does not cover it.

## 8. Where to look next

- `classifier_head.py` (`to_bytes`, `state_hash`) and `c_port/verify_py_compat.c` — the two sides of the contract in ~60 lines.
- `verify_rust/src/main.rs` — the shortest verifier (25 lines); copy it to start a new substrate.
- `docs/understanding-federated-tinyml-vessel.md` — what the head is and why it is the only thing shipped.
