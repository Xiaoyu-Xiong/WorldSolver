"""XPBD finite-constraint solver for the suspended elastic bunny pendulum.

Physical model
--------------
The bunny is a closed triangulated surface that discretizes the soft
volumetric solid as a thin-shell / pressurized-membrane continuum:

* stretch: distance constraints on every mesh edge (a triangulated membrane
  with per-edge stiffness ``k = 1 / c_stretch``),
* bending: distance constraints between the opposite vertices of the two
  triangles adjacent to each edge (discrete shell bending),
* volumetric stiffness: one global enclosed-volume constraint, which gives
  the closed body its incompressible-solid response (axial stretching is
  coupled to lateral contraction, like a Poisson effect).

Time integration uses XPBD (extended position-based dynamics): every
substep solves the compliance form of an implicit Euler step for the
constrained system, so the scheme is unconditionally stable and realizes
the target stiffnesses independent of the step size.  Gravity enters
through the explicit prediction step, exactly as in implicit Euler.

All computations are deterministic pure-NumPy array operations; no
randomness, no keyframes, no tabulated motion.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

SUBSTEPS_PER_SAMPLE = 20
SOLVER_ITERATIONS = 2
JACOBI_RELAXATION = 0.35

# Compliances (inverse stiffnesses) of the constitutive model.
STRETCH_COMPLIANCE = 1.0 / 75.0  # m/N  -> ~25% strain at the suspension ring
BEND_COMPLIANCE = 0.08  # m/N    -> shape-preserving shell bending
VOLUME_COMPLIANCE = 1e-11  # stiff incompressibility of the closed solid

_LENGTH_EPS = 1e-12


def _asset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "bunny.obj"


def _load_faces(path: Path) -> np.ndarray:
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("f "):
                tokens = line.split()[1:4]
                faces.append([int(tok.split("/")[0]) - 1 for tok in tokens])
    if not faces:
        raise ValueError(f"bunny asset has no faces: {path}")
    return np.asarray(faces, dtype=np.int64)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    cand = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    cand = np.sort(cand, axis=1)
    return np.unique(cand, axis=0)


def _opposite_pairs(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opposite vertices of the two triangles sharing each interior edge."""
    opposite: dict[tuple[int, int], list[int]] = defaultdict(list)
    for a, b, c in faces.tolist():
        opposite[(a, b) if a < b else (b, a)].append(c)
        opposite[(b, c) if b < c else (c, b)].append(a)
        opposite[(c, a) if c < a else (a, c)].append(b)
    us: list[int] = []
    vs: list[int] = []
    for (_, _), opps in sorted(opposite.items()):
        if len(opps) == 2:
            us.append(opps[0])
            vs.append(opps[1])
    return np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64)


class BenchmarkSolver:
    """Swinging suspended elastic bunny integrated with XPBD substeps."""

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        metadata = scene.metadata()
        self.gravity = np.asarray(
            metadata.get("gravity", (0.0, 0.0, -9.81)), dtype=np.float64
        )
        times = scene.sample_times()
        sample_dt = float(times[1] - times[0]) if len(times) > 1 else 0.02
        self.h = sample_dt / SUBSTEPS_PER_SAMPLE

        self.x0 = np.array(frame0.position, dtype=np.float64)
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x)
        self.m = np.array(frame0.mass, dtype=np.float64)
        fixed_flag = np.asarray(frame0.attributes["fixed"], dtype=np.float64) > 0.5
        self.fixed = fixed_flag
        self.fixed_idx = np.nonzero(fixed_flag)[0]
        self.free_col = (~fixed_flag)[:, None]
        self.N = self.x0.shape[0]
        self.w = np.where(fixed_flag, 0.0, 1.0 / self.m)
        self.steps = 0

        faces = _load_faces(_asset_path())
        self.f0 = faces[:, 0].astype(np.int64)
        self.f1 = faces[:, 1].astype(np.int64)
        self.f2 = faces[:, 2].astype(np.int64)
        self.n_faces = faces.shape[0]

        # Combined distance constraints: mesh edges (stretch) plus
        # opposite-vertex pairs (bending), deterministically ordered.
        edges = _unique_edges(faces)
        bend_u, bend_v = _opposite_pairs(faces)
        ia = np.concatenate([edges[:, 0], bend_u]).astype(np.int64)
        ib = np.concatenate([edges[:, 1], bend_v]).astype(np.int64)
        rest = np.concatenate(
            [
                np.linalg.norm(self.x0[edges[:, 0]] - self.x0[edges[:, 1]], axis=1),
                np.linalg.norm(self.x0[bend_u] - self.x0[bend_v], axis=1),
            ]
        )
        compliance = np.concatenate(
            [
                np.full(len(edges), STRETCH_COMPLIANCE),
                np.full(len(bend_u), BEND_COMPLIANCE),
            ]
        )
        order = np.lexsort((ib, ia))
        self.ca = ia[order]
        self.cb = ib[order]
        self.rest = rest[order]
        self.alpha = compliance[order] / (self.h * self.h)
        self.wa = self.w[self.ca]
        self.wb = self.w[self.cb]
        self.n_constraints = self.ca.shape[0]
        # Constraints whose first/second endpoint lies on the fixed crown.
        self.fix_a = np.nonzero(fixed_flag[self.ca])[0]
        self.fix_b = np.nonzero(fixed_flag[self.cb])[0]

        _, volume0 = self._volume_gradient(self.x0)
        self.V0 = volume0

        total_mass = float(self.m.sum())
        com = (self.m[:, None] * self.x0).sum(axis=0) / total_mass
        attachment = self.x0[self.fixed_idx].mean(axis=0)
        length = float(np.linalg.norm(com - attachment))
        self.reference_momentum = float(total_mass * np.sqrt(9.81 * max(length, 0.1)))
        self.total_mass = total_mass
        self.m_fixed = float(self.m[self.fixed_idx].sum())
        self.domain_weights = np.vstack(
            [np.ones(self.N), fixed_flag.astype(np.float64)]
        )

        # Momentum-budget accumulators.
        self.cum_ext = np.zeros((2, 3))
        self.last_force = None

    # ------------------------------------------------------------------ #
    # continuum helpers
    # ------------------------------------------------------------------ #
    def _volume_gradient(self, points: np.ndarray) -> tuple[np.ndarray, float]:
        """Enclosed volume and its per-vertex gradient (divergence theorem)."""
        a = points[self.f0]
        b = points[self.f1]
        c = points[self.f2]
        ga = np.cross(b, c)
        gb = np.cross(c, a)
        gc = np.cross(a, b)
        volume = float(np.einsum("ij,ij->i", a, ga).sum() / 6.0)
        grad = np.zeros((self.N, 3))
        for k in range(3):
            grad[:, k] = (
                np.bincount(self.f0, ga[:, k], minlength=self.N)
                + np.bincount(self.f1, gb[:, k], minlength=self.N)
                + np.bincount(self.f2, gc[:, k], minlength=self.N)
            ) / 6.0
        return grad, volume

    def _deformation_gradients(self) -> np.ndarray:
        """Per-face 3x3 deformation gradient (material frame = initial pose)."""
        xi, xd = self.x0, self.x
        a0, b0, c0 = xi[self.f0], xi[self.f1], xi[self.f2]
        e1, e2 = b0 - a0, c0 - a0
        n0 = np.cross(e1, e2)
        n0 = n0 / (np.linalg.norm(n0, axis=1, keepdims=True) + _LENGTH_EPS)
        material = np.stack([e1, e2, n0], axis=-1)

        a1, b1, c1 = xd[self.f0], xd[self.f1], xd[self.f2]
        d1, d2 = b1 - a1, c1 - a1
        nd = np.cross(d1, d2)
        nd = nd / (np.linalg.norm(nd, axis=1, keepdims=True) + _LENGTH_EPS)
        deformed = np.stack([d1, d2, nd], axis=-1)

        grad = deformed @ np.linalg.inv(material)
        bad = ~np.isfinite(grad).all(axis=(1, 2))
        if bad.any():
            grad[bad] = np.eye(3, dtype=np.float64)
        return grad.astype(np.float32)

    # ------------------------------------------------------------------ #
    # time stepping
    # ------------------------------------------------------------------ #
    def _substep(self) -> None:
        h = self.h
        n = self.N

        # Implicit-Euler style prediction: gravity, then constraint solve.
        v_pred = np.where(self.free_col, self.v + h * self.gravity, 0.0)
        xp = self.x + h * v_pred

        lam = np.zeros(self.n_constraints)
        # Sum over fixed crown nodes of constraint multiplier * gradient
        # (equals physical constraint impulse on the crown times h).
        crown_constraint_impulse_h = np.zeros(3)

        wa, wb, alpha, rest = self.wa, self.wb, self.alpha, self.rest
        for _ in range(SOLVER_ITERATIONS):
            diff = xp[self.ca] - xp[self.cb]
            length = np.sqrt((diff * diff).sum(axis=1)) + _LENGTH_EPS
            direction = diff / length[:, None]
            residual = length - rest
            dlam = -(residual + alpha * lam) / (wa + wb + alpha)
            odlam = JACOBI_RELAXATION * dlam
            lam += odlam
            disp_a = (odlam * wa)[:, None] * direction
            disp_b = (odlam * wb)[:, None] * direction
            dx = np.empty((n, 3))
            for k in range(3):
                dx[:, k] = np.bincount(
                    self.ca, disp_a[:, k], minlength=n
                ) - np.bincount(self.cb, disp_b[:, k], minlength=n)
            if self.fix_a.size:
                crown_constraint_impulse_h += (
                    odlam[self.fix_a][:, None] * direction[self.fix_a]
                ).sum(axis=0)
            if self.fix_b.size:
                crown_constraint_impulse_h -= (
                    odlam[self.fix_b][:, None] * direction[self.fix_b]
                ).sum(axis=0)
            xp = xp + dx

        # Global enclosed-volume constraint (single XPBD/Newton step).
        grad_v, volume = self._volume_gradient(xp)
        residual = volume - self.V0
        denom = float((self.w * (grad_v * grad_v).sum(axis=1)).sum()) + (
            VOLUME_COMPLIANCE / (h * h)
        )
        dlam_v = -residual / denom
        xp = xp + (dlam_v * self.w)[:, None] * grad_v
        crown_constraint_impulse_h += dlam_v * grad_v[self.fixed_idx].sum(axis=0)

        # Finalize the substep.
        v_new = (xp - self.x) / h
        v_new[self.fixed_idx] = 0.0
        self.x = xp
        self.v = v_new
        self.steps += 1

        # ---------------- momentum budget bookkeeping ---------------- #
        gravity_impulse_all = self.total_mass * h * self.gravity
        gravity_impulse_crown = self.m_fixed * h * self.gravity
        constraint_impulse_crown = crown_constraint_impulse_h / h
        reaction_impulse_crown = -(
            gravity_impulse_crown + constraint_impulse_crown
        )
        row_global = gravity_impulse_all + reaction_impulse_crown
        row_crown = (
            gravity_impulse_crown
            + reaction_impulse_crown
            + constraint_impulse_crown
        )
        self.cum_ext[0] += row_global
        self.cum_ext[1] += row_crown
        zero3 = np.zeros(3)
        self.last_force = np.array(
            [
                [gravity_impulse_all / h, reaction_impulse_crown / h, zero3],
                [
                    gravity_impulse_crown / h,
                    reaction_impulse_crown / h,
                    constraint_impulse_crown / h,
                ],
            ]
        )

    # ------------------------------------------------------------------ #
    # contract entry points
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.h))
        while self.steps < target:
            self._substep()
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        if self.last_force is None:
            gravity_all = self.total_mass * self.gravity
            gravity_crown = self.m_fixed * self.gravity
            reaction = -gravity_crown
            zero3 = np.zeros(3)
            force = np.array(
                [
                    [gravity_all, reaction, zero3],
                    [gravity_crown, reaction, zero3],
                ]
            )
        else:
            force = self.last_force

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "fixed": self.fixed.astype(np.float64),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_ext.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((2, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((2, 3)),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": force.copy(),
                "cumulative_mass_source": np.zeros((1,)),
                "element_deformation_gradient": self._deformation_gradients(),
            },
            scalars={"reference_momentum": self.reference_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "xpbd_membrane_shell_volume",
            "substeps": self.steps,
            "substep_h": self.h,
            "iterations_per_substep": SOLVER_ITERATIONS,
            "nodes": int(self.N),
            "distance_constraints": int(self.n_constraints),
            "faces": int(self.n_faces),
            "rest_volume_m3": float(self.V0),
            "total_mass_kg": self.total_mass,
            "reference_momentum_kg_m_s": self.reference_momentum,
        }
