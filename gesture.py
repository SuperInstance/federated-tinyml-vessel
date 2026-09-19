"""gesture.py — the geometry of federated convergence, read as a gesture.

A federated model does not sit at a point; it *moves*. Each round, FedAvg
averages the devices' heads and the global head takes a step through parameter
space. Over a run, the head traces a **path** — a gesture — and that gesture
holds information the final accuracy number cannot:

* ``arc_length`` — how far the global head travelled to arrive (a long, wandering
  road vs. a short, direct one),
* ``bending_energy`` — curvature: how much the aggregation direction keeps
  *turning within a plane* (a model that oscillates as it converges vs. one that
  descends smoothly),
* ``twist_energy`` — torsion: how much the path turns *out of* its plane, into a
  fresh direction of parameter space — i.e. whether the convergence keeps opening
  genuinely new dimensions round after round, or refines within one plane.

A hypothesis worth probing (not yet confirmed here): that heavier non-IID skew
shows up as higher twist, as devices pull the global head in conflicting
directions. A quick sweep across a few seeds at moderate skew did **not** show a
clear effect, so treat twist as an honest *observable* of the convergence path,
not a validated non-IID detector. The value is the lens; the reading is yours.

This is the fleet's shared "abstraction as gesture" reading — a tensor
approximates a function; we approximate the *abstraction*, the shape of the
motion — carried to the vessel edge. The same three orders read notes
(musician-soul's ``AbstractionSpline``), rooms (elephant's ``VibeTrajectory``),
conversations (tensor-midi's ``Clip``), and cell state (quilt's ``Gesture``);
here they read a federated model's convergence. *The property is in the twist*
(SuperInstance/twist-engine): new structure is the turning that leaves the plane.

Pure numpy (the repo's only runtime dep). Never raises on empty/degenerate input.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

VectorLike = Union[np.ndarray, Sequence[float]]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Gesture:
    """The path a vector traces through a state space over successive readings.

    Args:
        points: an ordered sequence of vectors (oldest first) — e.g. the flattened
            global-head weights at each federated round.
    """

    def __init__(self, points: Sequence[VectorLike]):
        rows = [np.asarray(p, dtype=float).ravel() for p in points]
        self.points = np.vstack(rows) if rows else np.empty((0, 0))

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1]) if self.points.ndim == 2 and self.points.shape[0] else 0

    def steps(self) -> np.ndarray:
        """The (T-1, D) array of consecutive step vectors (the discrete velocity)."""
        if len(self) < 2:
            return np.empty((0, self.dim))
        return np.diff(self.points, axis=0)

    def arc_length(self) -> float:
        """Total distance travelled through parameter space. 0.0 for < 2 readings."""
        s = self.steps()
        return float(np.linalg.norm(s, axis=1).sum()) if s.shape[0] else 0.0

    def speed(self) -> float:
        """Magnitude of the latest step — how fast the model is still moving."""
        s = self.steps()
        return float(np.linalg.norm(s[-1])) if s.shape[0] else 0.0

    def heading(self) -> np.ndarray:
        """Unit direction of the latest step — the convergence **d_mu**. Zero
        vector for < 2 readings or a model that has stopped moving."""
        s = self.steps()
        return _unit(s[-1]) if s.shape[0] else np.zeros(self.dim)

    def bending_energy(self) -> float:
        """Curvature — total turning *within a plane*: summed ``1 - cos`` between
        consecutive step directions. 0.0 for a straight descent at any speed;
        large for an aggregation that oscillates as it converges."""
        s = self.steps()
        if s.shape[0] < 2:
            return 0.0
        energy = 0.0
        for i in range(1, s.shape[0]):
            if np.linalg.norm(s[i - 1]) > 1e-12 and np.linalg.norm(s[i]) > 1e-12:
                energy += 1.0 - _cos(s[i - 1], s[i])
        return float(energy)

    def twist_energy(self) -> float:
        """Torsion — total **twist**: the turning that leaves the osculating plane.
        Per interior vertex, ``sin θ`` where θ is the angle by which the next step
        leaves the plane of the previous two, so each vertex is in ``[0, 1]`` and a
        straight or planar path contributes 0. Needs ≥4 readings.

        In federated terms: 0 when the global head refines within one plane;
        positive when convergence keeps opening genuinely new dimensions of
        parameter space. Whether that tracks non-IID skew is an open question
        (see module docstring) — this reports the geometry, not a verdict.
        *The property is in the twist.*"""
        s = self.steps()
        if s.shape[0] < 3:
            return 0.0
        energy = 0.0
        for i in range(1, s.shape[0] - 1):
            s1, s2, s3 = s[i - 1], s[i], s[i + 1]
            n1 = float(np.linalg.norm(s1))
            if n1 < 1e-12:
                continue
            e1 = s1 / n1
            perp = s2 - np.dot(s2, e1) * e1  # s2 ⟂ e1
            npn = float(np.linalg.norm(perp))
            if npn < 1e-12:
                continue  # s1 ∥ s2: no plane to leave
            e2 = perp / npn
            n3 = float(np.linalg.norm(s3))
            if n3 < 1e-12:
                continue
            d3 = s3 / n3
            out = d3 - np.dot(d3, e1) * e1 - np.dot(d3, e2) * e2
            energy += min(float(np.linalg.norm(out)), 1.0)
        return energy

    def planarity(self) -> float:
        """How flat the path stays, in ``[0, 1]``: 1.0 for a convergence confined
        to one plane (all curvature, no twist), toward 0.0 as more turning leaves
        the plane. 1.0 for a path too short to twist. Scale-free inverse of
        ``twist_energy``."""
        s = self.steps()
        vertices = max(0, s.shape[0] - 2)
        if vertices == 0:
            return 1.0
        return float(np.clip(1.0 - self.twist_energy() / vertices, 0.0, 1.0))


def convergence_geometry(trajectory: Sequence[VectorLike]) -> Dict[str, float]:
    """Summarize a federated run's convergence as gesture geometry.

    Give it the ordered per-round global-head vectors (see
    ``ClassifierHead.flat_state``); get back the shape of how the model got where
    it did:

    * ``arc_length`` — total distance travelled through parameter space,
    * ``final_step`` — how far the last round moved (has it settled?),
    * ``bending_energy`` / ``mean_bending`` — oscillation of the descent,
    * ``twist_energy`` / ``planarity`` — is convergence opening new dimensions of
      parameter space, or refining within a plane? (See the module docstring on
      the unconfirmed non-IID hypothesis.)
    """
    g = Gesture(trajectory)
    rounds = len(g)
    turns = max(0, rounds - 2)
    return {
        "rounds": float(rounds),
        "arc_length": g.arc_length(),
        "final_step": g.speed(),
        "bending_energy": g.bending_energy(),
        "mean_bending": g.bending_energy() / turns if turns else 0.0,
        "twist_energy": g.twist_energy(),
        "planarity": g.planarity(),
    }


if __name__ == "__main__":
    # Self-test: a planar descent has no twist; a run recruiting a third axis does.
    rng = np.random.default_rng(0)
    planar = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 4.0, 8)]
    helix = [np.array([np.cos(a), np.sin(a), 0.4 * a]) for a in np.linspace(0, 4.0, 8)]
    gp = Gesture(planar)
    gh = Gesture(helix)
    assert gp.twist_energy() < 1e-6, gp.twist_energy()
    assert gh.twist_energy() > gp.twist_energy()
    assert 0.0 <= gp.planarity() <= 1.0
    assert Gesture([]).arc_length() == 0.0
    assert Gesture([rng.standard_normal(325)]).twist_energy() == 0.0
    geo = convergence_geometry([rng.standard_normal(325) for _ in range(6)])
    assert geo["rounds"] == 6.0 and geo["arc_length"] > 0.0
    print("gesture.py self-test OK")
    print("  planar twist:", round(gp.twist_energy(), 4), "| helix twist:", round(gh.twist_energy(), 4))
    print("  sample convergence geometry:", {k: round(v, 3) for k, v in geo.items()})
