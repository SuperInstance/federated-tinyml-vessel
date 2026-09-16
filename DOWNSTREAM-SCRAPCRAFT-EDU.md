# Downstream — F170 in a classroom

*This design has a fielded educational descendant. Recording it here so the
lineage is legible and a future integrator has a bridge. Not a fork, not a
reimplementation of the vessel system — a toy-scale engine built to the same
architecture, so that children learn federated learning by doing the real move.*

## The descendant

[Scrapcraft](https://github.com/SuperInstance/Scrapcraft) — a browser game where
middle-schoolers build robots and program them — ships a deterministic
**Fleet Learning engine** (`src/maker/FleetLearning.js`) that is F170's shape at
toy scale. The vessel system teaches a wrist, a hydrophone, and a sounder to
share a classifier without sharing their audio; the game teaches a room of kids'
robots to share a classifier without sharing their play. Same move. Same
discipline (a frozen projection; only a tiny head learns; heads combine by
sample-weighted average). Smaller everything, and a child at every node.

## The mapping (F170 → the engine)

| F170 (this repo) | Scrapcraft Fleet Learning engine |
|---|---|
| **Frozen backbone** — log-mel → MFCC → 180-dim → linear 180→64, trained once then frozen | `makeFeatures(sample)` — a **fixed, non-trained** projection from a small raw sensor window to a small feature vector. The "frozen backbone" idea, minus the audio: the point a child needs is *some features are given, not learned.* |
| **Trainable head** — linear 64→5, 325 params, 1.3 KB, the *only* thing that learns | `trainHead(samples, opts)` — a tiny logistic head trained by plain gradient descent over the fixed features. Deterministic given a seed. |
| **Inference** — softmax → 5 class probabilities | `predict(weights, sample)` |
| **FedAvg** — average the heads, ship back | `fedAvg(headsWithCounts)` — the sample-count-**weighted** average (equal counts → plain mean; unequal → weighted). |
| **A round** — each device trains locally, aggregator averages | `fedRound(perRobotData, globalWeights, opts)` |
| **Byte-exact polyformalism** (Python/JS/Rust/C agree bit-for-bit) | **Determinism** (seeded, no `Math.random`/`Date`): same seed + data → byte-identical head weights. The classroom version of "the state hash is the identity anchor." |

The five vessel classes (`silence, normal, wind, net_haul, line_tangle`) become a
classroom analog — e.g. a robot's sensor window classified as *stuck vs. cruising
vs. about-to-crash*. The task is a toy; the pipeline is the real one.

## What is deliberately the same, and what is not

**The same** — because it's the part worth teaching: the frozen/trainable split,
the tiny head as the only learner, sample-weighted FedAvg, determinism as the
guarantee that a result can be reproduced and therefore trusted. A ten-year-old
who runs one honest FedAvg round has done the same thing F170 does beyond 50 nm
of shore.

**Not the same** — and the note is honest about it so nobody over-reads the
kinship: the game's backbone is a numeric sensor projection, not a log-mel audio
model; there is no ESC-50, no Whisper, no int4 tensor-train, no C/Rust device
port; scale is dozens of parameters and a toy separable task, not 325 params on
real vessel audio. The engine proves the *shape* of the idea, unit-tested for
determinism, learning-above-chance, exact FedAvg weighting, and multi-round
convergence. It is the on-ramp, not the vessel.

## Why record it

F161's conservation law and F170's frozen-backbone federation are the kind of
research that earns its keep twice: once by working on the boat, and once by
being simple enough at the core that a child can be handed the core and grow up
knowing what "the head learns, the backbone holds, the average is fair" means in
their hands. The vessel is where it's load-bearing. The classroom is where it
becomes native. Same law, two altitudes.

---

*— filed by the builder of the downstream engine. The engine
(`src/maker/FleetLearning.js`) and its tests run in the
[Scrapcraft](https://github.com/SuperInstance/Scrapcraft) repository; this note
claims nothing that isn't in code.*
