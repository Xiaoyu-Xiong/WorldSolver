"""Elastic FEM solver for the Armadillo slingshot scene.

A stable explicit St. Venant-Kirchhoff solid is integrated with sub-stepping.
Fixed grasp nodes are held and a temporary kinematic pull drags the lower
body upward until release; the body then recoils under elastic forces.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from worldsolver.contracts import StateFrame

ROOT = Path(__file__).resolve().parent.parent
SOLID = ROOT / "assets" / "armadillo_solid.npz"


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.X = np.asarray(frame0.position, dtype=np.float64).copy()
        self.x = self.X.copy()
        self.v = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.m = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.n = self.X.shape[0]

        self.fixed = np.asarray(frame0.attributes["fixed_grasp"], dtype=bool)
        self.pulled = np.asarray(frame0.attributes["pulled_patch"], dtype=bool)

        # Tetrahedral topology and reference shape matrices.
        data = np.load(SOLID, allow_pickle=False)
        self.tets = np.asarray(data["tetrahedra"], dtype=np.int64)
        i, j, k, l = self.tets.T
        Dm = np.stack(
            [self.X[j] - self.X[i], self.X[k] - self.X[i], self.X[l] - self.X[i]],
            axis=-1,
        )
        self.Dm_inv = np.linalg.inv(Dm)
        self.rest_volume = np.abs(np.linalg.det(Dm)) / 6.0
        self.F_cache = np.tile(np.eye(3, dtype=np.float64), (len(self.tets), 1, 1))

        # Material parameters (Young = 45 kPa, nu = 0.45).
        E = 45000.0
        nu = 0.45
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Pull schedule.
        self.pull_start = 0.0
        self.pull_peak = 0.9
        self.release = 1.0
        self.target_start = np.array([0.0, 0.0, 1.25], dtype=np.float64)
        self.target_peak = np.array([0.0, 0.0, 3.65], dtype=np.float64)

        # Light implicit velocity damping (per second).
        self.damping = 1.0

        # Time and budget accumulators.
        self.current_time = 0.0
        self.cum_external_impulse = np.zeros((3, 3), dtype=np.float64)
        self.cum_inflow = np.zeros((3, 3), dtype=np.float64)
        self.cum_outflow = np.zeros((3, 3), dtype=np.float64)
        self.budget_external_force = np.zeros((3, 3, 3), dtype=np.float64)

        # Domain weights: global, fixed, pulled.
        global_w = np.ones(self.n, dtype=np.float64)
        fixed_w = self.fixed.astype(np.float64)
        pulled_w = self.pulled.astype(np.float64)
        self.domain_weights = np.stack([global_w, fixed_w, pulled_w], axis=0)

        self.ref_momentum = float(self.m.sum()) * 2.0

        # Prime deformation gradient cache at rest.
        self._compute_forces(self.X)

    def _pull_target(self, t: float):
        """Return (position, velocity) of the pull target, or (None, None) if inactive."""
        if t < self.pull_start or t >= self.release:
            return None, None
        ramp = self.target_peak - self.target_start
        duration = self.pull_peak - self.pull_start
        if t <= self.pull_peak:
            s = (t - self.pull_start) / duration
            p = self.target_start + s * ramp
            v = ramp / duration
        else:
            p = self.target_peak
            v = np.zeros(3, dtype=np.float64)
        return p, v

    def _compute_forces(self, x: np.ndarray) -> np.ndarray:
        """Compute nodal elastic forces for positions x and cache F per element."""
        i, j, k, l = self.tets.T
        Ds = np.stack(
            [x[j] - x[i], x[k] - x[i], x[l] - x[i]],
            axis=-1,
        )
        F = Ds @ self.Dm_inv
        self.F_cache = F

        # St. Venant-Kirchhoff: P = F * (2*mu*E + lambda*tr(E)*I), E = 0.5*(F^T F - I).
        FtF = F.transpose(0, 2, 1) @ F
        I = np.eye(3, dtype=np.float64)
        E_green = 0.5 * (FtF - I)
        trace_E = np.trace(E_green, axis1=-2, axis2=-1)
        S = 2.0 * self.mu * E_green + self.lam * trace_E[:, None, None] * I
        P = F @ S

        H = P @ self.Dm_inv.transpose(0, 2, 1)
        H = H * self.rest_volume[:, None, None]

        f = np.zeros((self.n, 3), dtype=np.float64)
        np.add.at(f, (j, slice(None)), H[:, :, 0])
        np.add.at(f, (k, slice(None)), H[:, :, 1])
        np.add.at(f, (l, slice(None)), H[:, :, 2])
        f[i] -= H[:, :, 0] + H[:, :, 1] + H[:, :, 2]
        return f

    def _compute_instantaneous_budget(self, x: np.ndarray, v: np.ndarray, t: float) -> None:
        """Compute instantaneous budget terms at state (x, v, t)."""
        f_el = self._compute_forces(x)
        f_grav = self.m[:, None] * self.g[None, :]
        p_target, v_target = self._pull_target(t)

        f_pull = np.zeros((self.n, 3), dtype=np.float64)
        f_fixed = np.zeros((self.n, 3), dtype=np.float64)

        # Fixed nodes are held at rest.
        if self.fixed.any():
            f_fixed[self.fixed] = -(f_el[self.fixed] + f_grav[self.fixed])

        # Pulled nodes follow the target velocity.
        if p_target is not None and self.pulled.any():
            # Treat the change over one simulation sub-step as the instantaneous force.
            sample_dt = 0.02
            f_pull[self.pulled] = (
                self.m[self.pulled, None] * (v_target[None, :] - v[self.pulled]) / sample_dt
            )

        f_total = np.stack([f_grav, f_pull, f_fixed], axis=0)
        self.budget_external_force = np.einsum("gn,kni->kgi", self.domain_weights, f_total).transpose(
            1, 0, 2
        )

    def _step(self, dt: float, t0: float) -> float:
        """Advance one sub-step and return new time."""
        t1 = t0 + dt
        t_mid = 0.5 * (t0 + t1)
        p_target, v_target = self._pull_target(t_mid)

        f_el = self._compute_forces(self.x)
        f_grav = self.m[:, None] * self.g[None, :]
        f_uncon = f_el + f_grav

        # Symplectic velocity update with implicit damping.
        v_temp = self.v + dt * (f_uncon / self.m[:, None])
        v_temp = v_temp / (1.0 + dt * self.damping)

        f_pull = np.zeros((self.n, 3), dtype=np.float64)
        f_fixed = np.zeros((self.n, 3), dtype=np.float64)

        # Enforce fixed constraint: velocity becomes zero.
        if self.fixed.any():
            f_fixed[self.fixed] = -self.m[self.fixed, None] * v_temp[self.fixed] / dt
            v_temp[self.fixed] = 0.0

        # Enforce pull constraint: velocity follows the target.
        if p_target is not None and self.pulled.any():
            v_new_pulled = np.broadcast_to(v_target, (self.pulled.sum(), 3))
            f_pull[self.pulled] = (
                self.m[self.pulled, None] * (v_new_pulled - v_temp[self.pulled]) / dt
            )
            v_temp[self.pulled] = v_new_pulled

        self.v = v_temp
        self.x = self.x + dt * self.v

        # Project fixed nodes back to reference and pulled nodes to target position.
        if self.fixed.any():
            self.x[self.fixed] = self.X[self.fixed]
        if p_target is not None and self.pulled.any():
            p1, _ = self._pull_target(t1)
            if p1 is not None:
                self.x[self.pulled] = p1[None, :]

        # Update budget.
        f_total = np.stack([f_grav, f_pull, f_fixed], axis=0)
        self.budget_external_force = np.einsum(
            "gn,kni->kgi", self.domain_weights, f_total
        ).transpose(1, 0, 2)

        self.cum_external_impulse += dt * np.sum(self.budget_external_force, axis=1)
        # No mass flow in this scene.
        self.cum_inflow += 0.0
        self.cum_outflow += 0.0

        return t1

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.current_time:
            # Time moved backward; reset to rest.
            self.x = self.X.copy()
            self.v = np.zeros_like(self.v)
            self.current_time = 0.0
            self.cum_external_impulse.fill(0.0)
            self.cum_inflow.fill(0.0)
            self.cum_outflow.fill(0.0)

        dt = time - self.current_time
        if dt > 0.0:
            max_dt = 2.0e-4
            n_sub = max(1, int(np.ceil(dt / max_dt)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                self.current_time = self._step(sub_dt, self.current_time)

        # Ensure F cache matches the final state.
        self._compute_forces(self.x)

        if time <= 0.0:
            self._compute_instantaneous_budget(self.x, self.v, time)

        return StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "fixed_grasp": self.fixed.astype(np.float64),
                "pulled_patch": self.pulled.astype(np.float64),
                "mass_quantity": self.m[:, None].copy(),
            },
            fields={
                "element_deformation_gradient": self.F_cache.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.budget_external_force.copy(),
            },
            scalars={
                "density_kg_m3": 1000.0,
                "reference_momentum": self.ref_momentum,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "nodes": self.n,
            "elements": len(self.tets),
            "total_mass_kg": float(self.m.sum()),
            "current_time": self.current_time,
        }
