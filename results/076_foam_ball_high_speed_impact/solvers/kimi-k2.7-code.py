"""Explicit finite-element solver for the high-speed foam ball impact.

The ball is modelled as a Neo-Hookean hyperelastic solid discretised with a
regular tetrahedral mesh built from the scene's particle samples.  Time
integration uses a symplectic Euler sub-stepping scheme with a fixed sub-step
chosen well below the elastic and contact stability limits.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay  # noqa: F401  # kept available, not used for final mesh

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically step the foam-ball impact and emit StateFrame samples."""

    def initialize(self, scene, seed: int) -> None:
        # The scaffold scene exposes its config and the initial state.
        self.config = getattr(scene, "config", {})
        frame0 = scene.initial_state()

        # Particle state.
        self.x = np.asarray(frame0.position, dtype=np.float64).copy()
        self.v = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.m = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.rest = np.asarray(frame0.attributes["rest_position"], dtype=np.float64).copy()
        self.n = self.x.shape[0]

        # Material parameters (SI).
        ball_cfg = self.config["foam_ball"]
        E = float(ball_cfg["youngs_modulus"])
        nu = float(ball_cfg["poisson_ratio"])
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Gravity and plate data.
        self.gravity = np.asarray(
            self.config["initial_conditions"]["gravity"], dtype=np.float64
        )
        self.plate_x = float(
            self.config.get("boundary_conditions", {}).get("plate_contact_surface_x", 0.0)
        )

        # Build a regular tetrahedral mesh from the particle samples.
        self.tets = self._build_regular_tetrahedral_mesh()
        self.n_tets = self.tets.shape[0]
        self._precompute_tet_bases()

        # Time-stepping control.
        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        # Fixed sub-step count.  100 sub-steps per sample frame keeps the
        # explicit elastic update well below the CFL limit and gives the
        # penalty contact spring a stable integration.
        self.substeps = 100
        self.dt = self.sample_dt / self.substeps
        self.t = 0.0

        # Contact penalty parameters (per point).
        m_avg = float(np.mean(self.m))
        self.k_contact = 2.0e7  # N/m
        self.c_contact = 2.0 * np.sqrt(self.k_contact * m_avg)

        # Budget / trajectory accumulators.
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.last_budget_external_force = np.zeros((1, 2, 3), dtype=np.float64)
        self.reference_momentum = float(frame0.scalars.get("total_mass", 1.0))
        self.cumulative_mass_source = np.zeros((1,), dtype=np.float64)

        # Pre-allocated buffers.
        self._f = np.zeros((self.n, 3), dtype=np.float64)
        self._F = np.zeros((self.n_tets, 3, 3), dtype=np.float64)

    # ------------------------------------------------------------------ mesh
    def _build_regular_tetrahedral_mesh(self) -> np.ndarray:
        """Return tet indices using a regular grid split inside the sphere."""
        ball = self.config["foam_ball"]
        radius = float(ball["radius"])
        center = np.asarray(ball["initial_center"], dtype=np.float64)
        spacing = radius / 5.0
        axis = np.arange(-radius, radius + spacing * 0.5, spacing)
        m = len(axis)

        # Map each particle to its (i,j,k) grid coordinate.
        rel = self.rest - center
        ijk = np.rint((rel + radius) / spacing).astype(np.int64)
        key_to_local: dict[tuple[int, int, int], int] = {}
        for idx, key in enumerate(ijk):
            key_to_local[tuple(key.tolist())] = idx

        # Inside-sphere mask on the regular grid (matching scene.py).
        r2 = (radius / spacing) ** 2
        mask = np.zeros((m, m, m), dtype=bool)
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    d2 = i * i + j * j + k * k
                    if d2 <= r2 + 1e-9:
                        mask[i, j, k] = True

        # 6-tet decomposition of every fully interior cube.
        tets = []
        for i in range(m - 1):
            for j in range(m - 1):
                for k in range(m - 1):
                    # Cube corner keys.
                    c000 = (i, j, k)
                    c100 = (i + 1, j, k)
                    c010 = (i, j + 1, k)
                    c001 = (i, j, k + 1)
                    c110 = (i + 1, j + 1, k)
                    c101 = (i + 1, j, k + 1)
                    c011 = (i, j + 1, k + 1)
                    c111 = (i + 1, j + 1, k + 1)
                    cube = [c000, c100, c010, c001, c110, c101, c011, c111]
                    if not all(mask[key] for key in cube):
                        continue
                    # Six tets that fill the cube using the main diagonal.
                    tets.append([c000, c100, c110, c111])
                    tets.append([c000, c010, c110, c111])
                    tets.append([c000, c001, c101, c111])
                    tets.append([c000, c001, c011, c111])
                    tets.append([c000, c010, c011, c111])
                    tets.append([c000, c100, c101, c111])

        if not tets:
            # Fallback to a Delaunay tetrahedralization if the regular mesh
            # somehow has no elements (e.g. for an exotic sample set).
            dela = Delaunay(self.rest)
            tets = dela.simplices.copy()
            return self._orient_tets(np.asarray(tets, dtype=np.int64))

        tets_local = np.array(
            [[key_to_local[k] for k in tet] for tet in tets], dtype=np.int64
        )
        return self._orient_tets(tets_local)

    def _orient_tets(self, tets: np.ndarray) -> np.ndarray:
        """Ensure positive rest volumes by swapping vertex pairs if needed."""
        x0 = self.rest[tets[:, 0]]
        x1 = self.rest[tets[:, 1]]
        x2 = self.rest[tets[:, 2]]
        x3 = self.rest[tets[:, 3]]
        Dm = np.stack([x0 - x3, x1 - x3, x2 - x3], axis=2)  # (Ne,3,3)
        vol6 = np.linalg.det(Dm)
        tets = tets.copy()
        flip = vol6 < 0.0
        tets[flip, 0], tets[flip, 1] = tets[flip, 1].copy(), tets[flip, 0].copy()
        return tets

    def _precompute_tet_bases(self) -> None:
        x0 = self.rest[self.tets[:, 0]]
        x1 = self.rest[self.tets[:, 1]]
        x2 = self.rest[self.tets[:, 2]]
        x3 = self.rest[self.tets[:, 3]]
        Dm = np.stack([x0 - x3, x1 - x3, x2 - x3], axis=2)  # (Ne,3,3)
        self.Dm_inv = np.linalg.inv(Dm)
        self.Dm_inv_T = np.transpose(self.Dm_inv, (0, 2, 1))
        self.V0 = np.abs(np.linalg.det(Dm)) / 6.0

        # Filter out any degenerate elements.
        keep = self.V0 > 1e-15
        self.tets = self.tets[keep]
        self.Dm_inv = self.Dm_inv[keep]
        self.Dm_inv_T = self.Dm_inv_T[keep]
        self.V0 = self.V0[keep]
        self.n_tets = self.tets.shape[0]
        self._F = np.zeros((self.n_tets, 3, 3), dtype=np.float64)

    # --------------------------------------------------------------- physics
    def _compute_internal_forces(self) -> np.ndarray:
        """Compute nodal elastic forces from the Neo-Hookean model."""
        x_tet = self.x[self.tets]  # (Ne,4,3)
        # Edge vectors from the fourth vertex.
        e0 = x_tet[:, 0, :] - x_tet[:, 3, :]
        e1 = x_tet[:, 1, :] - x_tet[:, 3, :]
        e2 = x_tet[:, 2, :] - x_tet[:, 3, :]
        Ds = np.stack([e0, e1, e2], axis=2)  # (Ne,3,3)

        # Deformation gradient F = Ds * Dm^{-1}.
        F = Ds @ self.Dm_inv
        self._F[:] = F

        # Stable Neo-Hookean first Piola stress.
        # W = mu/2 (I1 - 3) + lambda/2 (J - 1)^2 - mu log(J)
        # P = mu (F - F^-T) + lambda J (J - 1) F^-T
        J = np.linalg.det(F)
        # Clamp tiny / negative J to avoid division by zero; this is a mild
        # numerical guard and keeps the explicit step finite.
        J = np.where(J < 1e-6, 1e-6, J)
        F_inv = np.linalg.inv(F)
        F_inv_T = np.transpose(F_inv, (0, 2, 1))
        P = self.mu * (F - F_inv_T) + self.lam * J[:, None, None] * (J - 1.0)[:, None, None] * F_inv_T

        # Nodal forces: f_i = -V0 * P * grad_X N_i.
        H = P @ self.Dm_inv_T  # (Ne,3,3)
        f0 = -self.V0[:, None] * H[:, :, 0]
        f1 = -self.V0[:, None] * H[:, :, 1]
        f2 = -self.V0[:, None] * H[:, :, 2]
        f3 = -self.V0[:, None] * (H[:, :, 0] + H[:, :, 1] + H[:, :, 2])

        f = np.zeros((self.n, 3), dtype=np.float64)
        np.add.at(f, self.tets[:, 0], f0)
        np.add.at(f, self.tets[:, 1], f1)
        np.add.at(f, self.tets[:, 2], f2)
        np.add.at(f, self.tets[:, 3], f3)
        return f

    def _contact_force(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
        """Return per-sample contact force and its vector sum for the ball."""
        f = np.zeros_like(v)
        dx = self.plate_x - x[:, 0]
        mask = dx > 0.0
        if not np.any(mask):
            return f, 0.0
        # Damping only opposes motion into the plate.
        v_into = np.minimum(v[mask, 0], 0.0)
        f_x = self.k_contact * dx[mask] - self.c_contact * v_into
        f[mask, 0] = f_x
        total = f.sum(axis=0)
        return f, total

    def _substep(self, dt: float) -> None:
        f = self._compute_internal_forces()
        f += self.m[:, None] * self.gravity

        # Contact force and budget contribution.
        f_contact, contact_total = self.contact_force(self.x, self.v)
        f += f_contact

        # Symplectic Euler update.
        a = f / self.m[:, None]
        self.v += a * dt
        self.x += self.v * dt

        # Budget updates (single domain = the ball; two force terms).
        gravity_total = (self.m[:, None] * self.gravity).sum(axis=0)
        self.last_budget_external_force[0, 0, :] = gravity_total
        self.last_budget_external_force[0, 1, :] = contact_total
        self.cumulative_external_impulse[0, :] += (gravity_total + contact_total) * dt

    # ----------------------------------------------------------- public API
    def advance_to(self, time: float) -> StateFrame:
        while self.t < time - 0.5 * self.dt:
            self._substep(self.dt)
            self.t += self.dt

        # Build the frame exactly at the requested sample time using the
        # current integrated state.
        self._compute_internal_forces()  # fills self._F for output

        # Domain weights: single domain covering all ball particles.
        domain_weights = np.ones((1, self.n), dtype=np.float64)

        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "mass_quantity": self.m.copy()[:, None],
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": domain_weights,
                "budget_external_force": self.last_budget_external_force.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "element_deformation_gradient": self._F.copy(),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_neo_hookean_tet_fem",
            "particles": self.n,
            "tetrahedra": self.n_tets,
            "substep_dt": self.dt,
            "mu": self.mu,
            "lambda": self.lam,
        }
