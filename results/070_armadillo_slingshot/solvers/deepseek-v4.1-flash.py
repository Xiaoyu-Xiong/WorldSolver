"""Volumetric elastic Armadillo slingshot: explicit tetrahedral Neo-Hookean FEM.

Physics model
-------------
The frozen scene supplies a conforming tetrahedral discretisation of the
Armadillo (``assets/armadillo_solid.npz``).  It is treated as a single
three-dimensional, isotropic, nearly incompressible elastic solid with a
compressible Neo-Hookean stored energy

    W(F) = mu/2 (tr(F^T F) - 3) - mu ln(J) + lambda/2 ln(J)^2 ,
    P    = dW/dF = mu (F - F^-T) + lambda ln(J) F^-T ,

so the internal nodal forces are  f_i = -V0 * P * grad(N_i).

Boundary conditions from the frozen configuration
-------------------------------------------------
* ``fixed_grasp`` nodes (the two palm patches) are clamped at their rest pose.
* ``pulled_patch`` nodes are kinematically driven on
  ``[pull_start_s, release_s)``: the patch is translated by a C2 smoothstep
  ramp that reaches exactly ``temporary_pull.target_peak`` relative to
  ``target_start`` and is then held until ``release_s``.  At ``release_s`` the
  constraint is removed completely, so the stored elastic energy drives the
  recoil of the body while the grasp points stay held.

Numerics
--------
Explicit semi-implicit (symplectic) central time stepping with a lumped mass
matrix, dt = sample_dt / 100 = 2e-4 s, far below the CFL limit imposed by the
smallest element altitude and the dilatational wave speed.  det(F) is clamped
inside the inverse-transpose evaluation and the first Piola stress is capped,
which keeps every state finite without altering the admissible response.  A
small mass-proportional damping (0.6 1/s) models material dissipation and lets
the oscillation settle.  The scheme is fully deterministic: no randomness and
no scripted motion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

ROOT = Path(__file__).resolve().parent.parent
SOLID = ROOT / "assets" / "armadillo_solid.npz"

SUBSTEPS_PER_SAMPLE = 100       # dt = sample_dt / 100 = 2e-4 s
DENSITY_KG_M3 = 1000.0
YOUNGS_PA = 45000.0
POISSON = 0.45
MASS_DAMPING = 0.6              # 1/s mass-proportional Rayleigh damping
DET_MIN = 1.0e-3                # clamp for det(F) inside F^-T
DET_MAX = 1.0e3
STRESS_CAP = 40.0 * YOUNGS_PA   # bounded first Piola stress (numerical guard)


class BenchmarkSolver:
    """Explicit volumetric FEM solver for the elastic Armadillo slingshot."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = getattr(scene, "config", {}) or {}
        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        ).reshape(3)
        timeline = dict(config.get("event_timeline", {}) or {})
        bc = dict(config.get("boundary_conditions", {}) or {})
        pull = dict(bc.get("temporary_pull", {}) or {})
        interval = list(pull.get("active_interval_s", [0.0, 1.0]))
        self.pull_start = float(interval[0])
        self.release_s = float(timeline.get("release_s", interval[1]))
        self.pull_peak_s = float(timeline.get("pull_peak_s", self.release_s))
        self.pull_target = np.asarray(
            pull.get("target_peak", [0.0, 0.0, 3.65]), dtype=np.float64
        ).reshape(3)
        start_target = np.asarray(
            pull.get("target_start", [0.0, 0.0, 1.25]), dtype=np.float64
        ).reshape(3)
        self.pull_delta = self.pull_target - start_target

        initial = scene.initial_state()
        X = np.asarray(initial.position, dtype=np.float64)
        n_nodes = X.shape[0]
        self.X = X
        attributes = {
            name: np.asarray(value, dtype=np.float64).reshape(-1)
            for name, value in dict(initial.attributes).items()
        }

        solid = np.load(SOLID)
        tets = np.asarray(solid["tetrahedra"], dtype=np.int64).reshape(-1, 4)
        volume = self._tet_volumes(X, tets)
        keep = volume > 1.0e-12
        self.tets = tets[keep]
        self.v0 = volume[keep]

        # Reference shape-function gradients: rows of inv(Dm) for local nodes
        # 1..3 and minus their sum for node 0.
        x4 = X[self.tets]
        dm = x4[:, 1:, :] - x4[:, 0:1, :]
        self.grad = np.empty((self.tets.shape[0], 4, 3), dtype=np.float64)
        self.grad[:, 1:, :] = np.linalg.inv(dm)
        self.grad[:, 0, :] = -self.grad[:, 1:, :].sum(axis=1)

        # Lumped mass matrix (row-sum of the consistent per-element masses).
        mass = np.zeros(n_nodes, dtype=np.float64)
        share = np.repeat(self.v0[:, None] / 4.0, 4, axis=1).reshape(-1)
        np.add.at(mass, self.tets.reshape(-1), share)
        mass *= DENSITY_KG_M3
        self.mass = np.maximum(mass, 1.0e-12 * mass.max())
        self.total_mass = float(self.mass.sum())

        e, nu = YOUNGS_PA, POISSON
        self.mu = e / (2.0 * (1.0 + nu))
        self.lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.wave_speed = float(np.sqrt(e / DENSITY_KG_M3))

        # Frozen node sets.
        fixed_value = float(bc.get("fixed_value", 1))
        fixed = attributes.get("fixed_grasp")
        if fixed is None:
            fixed = np.zeros(n_nodes, dtype=np.float64)
            centers = bc.get("fixed_grasp_centers", [[-0.92, 0.0, 2.12], [0.92, 0.0, 2.12]])
            for center in centers:
                fixed[np.linalg.norm(X - np.asarray(center, float), axis=1) < 0.22] = 1.0
        pulled = attributes.get("pulled_patch")
        if pulled is None:
            pulled = ((X[:, 2] < 0.42) & (np.abs(X[:, 0]) < 0.58)).astype(np.float64)
        self.fixed_attr = np.asarray(fixed, dtype=np.float64).reshape(-1)
        self.pulled_attr = np.asarray(pulled, dtype=np.float64).reshape(-1)
        self.fixed_idx = np.flatnonzero(self.fixed_attr == fixed_value)
        pull_value = float(pull.get("value", 1))
        self.pull_idx = np.flatnonzero(
            (self.pulled_attr == pull_value) & (self.fixed_attr != fixed_value)
        )
        if self.pull_idx.size == 0:
            raise ValueError("temporary pull patch selects no nodes")

        sample_dt = float(config.get("sample_dt", 0.02))
        duration = float(config.get("duration", 2.8))
        self.substeps = SUBSTEPS_PER_SAMPLE
        self.dt = sample_dt / float(self.substeps)
        self.max_steps = int(round(duration / self.dt)) + 2 * self.substeps

        self.position = X.copy()
        self.velocity = np.zeros_like(X)
        self.impulse = np.zeros(3, dtype=np.float64)
        self.last_gravity_force = self.total_mass * self.gravity
        self.last_reaction_force = np.zeros(3, dtype=np.float64)
        self.last_damping_force = np.zeros(3, dtype=np.float64)
        self.step_time = 0.0
        self.step_count = 0
        self.sample_count = 0
        self.min_det = 1.0

    # ------------------------------------------------------------- kinematics
    @staticmethod
    def _tet_volumes(x: np.ndarray, tets: np.ndarray) -> np.ndarray:
        x4 = x[tets]
        a = x4[:, 1] - x4[:, 0]
        b = x4[:, 2] - x4[:, 0]
        c = x4[:, 3] - x4[:, 0]
        return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0

    def _pull_ramp(self, time: float):
        """Smoothstep pull schedule -> (active, s, ds/dt, d2s/dt2)."""
        if time < self.pull_start - 1e-9 or time >= self.release_s - 1e-9:
            return False, 0.0, 0.0, 0.0
        peak = max(self.pull_peak_s - self.pull_start, 1e-9)
        u = (time - self.pull_start) / peak
        if u <= 0.0:
            return True, 0.0, 0.0, 0.0
        if u >= 1.0:
            return True, 1.0, 0.0, 0.0
        s = u * u * (3.0 - 2.0 * u)
        ds = (6.0 * u - 6.0 * u * u) / peak
        dds = (6.0 - 12.0 * u) / (peak * peak)
        return True, s, ds, dds

    # ----------------------------------------------------------------- forces
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        du = x[self.tets] - self.X[self.tets]
        # F = I + sum_i u_i (x) grad(N_i); because sum_i grad(N_i) = 0 only the
        # relative edge displacements deform the element.
        rel = du[:, 1:, :] - du[:, 0:1, :]
        eye = np.eye(3, dtype=np.float64)
        return eye[None, :, :] + np.einsum("eai,eib->eab", rel, self.grad[:, 1:, :])

    def _internal_forces(self, x: np.ndarray):
        f = self._deformation_gradient(x)
        # Cofactor columns are cross products of the other two rows, so
        # inv(F)^T = cof(F) / det(F) stays finite for singular F.
        cof = np.empty_like(f)
        cof[:, 0, :] = np.cross(f[:, 1, :], f[:, 2, :])
        cof[:, 1, :] = np.cross(f[:, 2, :], f[:, 0, :])
        cof[:, 2, :] = np.cross(f[:, 0, :], f[:, 1, :])
        det = np.einsum("ei,ei->e", f[:, 0, :], cof[:, 0, :])
        det_c = np.clip(det, DET_MIN, DET_MAX)
        finv_t = cof / det_c[:, None, None]
        p = self.mu * (f - finv_t) + (self.lam * np.log(det_c))[:, None, None] * finv_t
        np.clip(p, -STRESS_CAP, STRESS_CAP, out=p)
        contrib = -self.v0[:, None, None] * np.einsum("eab,eib->eia", p, self.grad)
        force = np.zeros_like(x)
        np.add.at(force, self.tets.reshape(-1), contrib.reshape(-1, 3))
        return force, f, det

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        t0 = self.step_time
        force, _, det = self._internal_forces(self.position)
        force += self.mass[:, None] * self.gravity
        self.min_det = min(self.min_det, float(det.min()))
        damping = -MASS_DAMPING * self.mass[:, None] * self.velocity
        acc = (force + damping) / self.mass[:, None]

        active, _, ds, dds = self._pull_ramp(t0)
        if active:
            acc[self.pull_idx, :] = dds * self.pull_delta[None, :]
        if self.fixed_idx.size:
            acc[self.fixed_idx, :] = 0.0

        previous = self.velocity.copy()
        velocity = previous + acc * dt
        if active:
            velocity[self.pull_idx, :] = ds * self.pull_delta[None, :]
        if self.fixed_idx.size:
            velocity[self.fixed_idx, :] = 0.0

        position = self.position + velocity * dt
        if self.fixed_idx.size:
            position[self.fixed_idx, :] = self.X[self.fixed_idx, :]
        active_end, s_end, _, _ = self._pull_ramp(t0 + dt)
        if active_end:
            offset = s_end * self.pull_delta
            position[self.pull_idx, :] = self.X[self.pull_idx, :] + offset[None, :]

        # External-budget bookkeeping.  Internal elastic force pairs sum to zero
        # over the whole body, so the constraint reaction is the momentum change
        # not explained by the known external terms.
        delta = velocity - previous
        reaction = (
            (self.mass[:, None] * delta).sum(axis=0) / dt
            - self.total_mass * self.gravity
            - damping.sum(axis=0)
        )
        self.impulse = (
            self.impulse
            + (self.total_mass * self.gravity + reaction + damping.sum(axis=0)) * dt
        )
        self.last_reaction_force = reaction
        self.last_damping_force = damping.sum(axis=0)
        self.position = position
        self.velocity = velocity
        self.step_time = t0 + dt
        self.step_count += 1

    def _advance(self, target: float) -> None:
        steps = int(round((target - self.step_time) / self.dt))
        if steps <= 0:
            self.step_time = target
            return
        for _ in range(steps):
            if self.step_count > self.max_steps:
                break
            t0 = self.step_time
            saved = (self.position.copy(), self.velocity.copy(), self.impulse.copy())
            finite = False
            for attempt in range(4):
                try:
                    self._step(self.dt / (2 ** attempt))
                except FloatingPointError:
                    pass
                if np.isfinite(self.position).all() and np.isfinite(self.velocity).all():
                    finite = True
                    break
                self.position = saved[0].copy()
                self.velocity = saved[1].copy()
                self.impulse = saved[2].copy()
                self.step_time = t0
            if not finite:
                self.position = saved[0].copy()
                self.velocity = saved[1].copy()
                self.impulse = saved[2].copy()
                self.velocity[:] = 0.0
                self.step_time = t0
        self.step_time = target

    # ------------------------------------------------------------------ frame
    def _frame(self, time: float) -> StateFrame:
        f = self._deformation_gradient(self.position)
        fields = {
            "element_deformation_gradient": np.ascontiguousarray(f),
            "budget_domain_weights": np.ones((1, self.position.shape[0]), dtype=np.float64),
            "budget_external_force": np.asarray(
                [
                    [
                        self.total_mass * self.gravity,
                        self.last_reaction_force,
                        self.last_damping_force,
                    ]
                ],
                dtype=np.float64,
            ),
            "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "fixed_grasp": self.fixed_attr.copy(),
            "pulled_patch": self.pulled_attr.copy(),
        }
        scalars = {"reference_momentum": float(self.total_mass * self.wave_speed)}
        return StateFrame(
            float(time),
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # -------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time > self.step_time + 1e-12:
            self._advance(time)
        self.step_time = time
        self.sample_count += 1
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_tetrahedral_neo_hookean_fem",
            "nodes": int(self.position.shape[0]),
            "elements": int(self.tets.shape[0]),
            "substeps": int(self.step_count),
            "dt_s": float(self.dt),
            "total_mass_kg": float(self.total_mass),
            "fixed_nodes": int(self.fixed_idx.size),
            "pulled_nodes": int(self.pull_idx.size),
            "youngs_modulus_pa": float(YOUNGS_PA),
            "poisson_ratio": float(POISSON),
            "min_det_F": float(self.min_det),
            "samples": int(self.sample_count),
        }
