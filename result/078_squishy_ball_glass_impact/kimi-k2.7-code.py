"""Finite-element elastic solid solver for the squishy-ball impact.

A corotated linear-elastic tetrahedral mesh is integrated with a stable
symplectic Euler substep.  Contact with the fixed glass wall is handled by a
penalty force plus an inelastic position projection so the ball flattens,
rebounds and the tendrils lift away after maximum compression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        # Initial state from the scene contract.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.component = np.asarray(
            frame0.attributes.get("component", np.zeros(len(self.position), dtype=np.int64))
        ).copy()

        self.time = 0.0
        self.cum_contact_impulse = np.zeros(3, dtype=np.float64)

        # Load the volumetric tetrahedral mesh that belongs to this scene.
        asset_path = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "squishy_ball_volume.npz"
        )
        with np.load(asset_path) as data:
            # Nodes in the asset should coincide with the initial state; the
            # scene already places the ball at the required starting location.
            ref_nodes = np.asarray(data["nodes"], dtype=np.float64)
            elements = None
            for key in ("elements", "tets", "tetrahedra", "element"):
                if key in data:
                    elements = np.asarray(data[key], dtype=np.int64)
                    break
        if elements is None:
            raise ValueError("Volume asset does not contain tetrahedral elements")

        # Convert 1-based indices if necessary.
        if elements.min() >= 1:
            elements = elements - 1

        self.n = len(self.position)
        self.elements = elements
        self.ne = len(self.elements)

        # Reference configuration: the initial shape at t = 0.
        self.ref_position = self.position.copy()
        self._build_tet_data()

        # Soft rubber-like material (SI units).  Stiffness is chosen to give
        # visible large deformation while remaining stable with the substep
        # chosen below.
        E = 5.0e4
        nu = 0.45
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Light Rayleigh-style velocity damping to remove high-frequency
        # oscillations without killing the rebound.
        self.damping = 25.0

        # Contact with the glass wall at x = 0 (left face of the box).
        self.k_wall = 1.0e5
        self.c_wall = 20.0

        # Fixed physical substep.  h = 1e-4 s keeps the explicit elastic
        # integration stable for the stiffness above.
        self.dt_sub = 1.0e-4

        # Positive momentum scale for the trajectory contract.
        total_mass = float(self.mass.sum())
        self.ref_momentum = max(1.0, total_mass * 2.0)

    def _build_tet_data(self) -> None:
        """Compute rest shape matrices, inverse and signed rest volume."""
        e = self.elements
        p0 = self.ref_position[e[:, 0]]
        p1 = self.ref_position[e[:, 1]] - p0
        p2 = self.ref_position[e[:, 2]] - p0
        p3 = self.ref_position[e[:, 3]] - p0
        Dm = np.stack([p1, p2, p3], axis=2)  # (Ne, 3, 3)

        vol = np.linalg.det(Dm) / 6.0
        # Fix any inverted tets by swapping two nodes.
        if np.any(vol < 0.0):
            neg = vol < 0.0
            self.elements[neg, [1, 2]] = self.elements[neg, [2, 1]]
            e = self.elements
            p0 = self.ref_position[e[:, 0]]
            p1 = self.ref_position[e[:, 1]] - p0
            p2 = self.ref_position[e[:, 2]] - p0
            p3 = self.ref_position[e[:, 3]] - p0
            Dm = np.stack([p1, p2, p3], axis=2)
            vol = np.linalg.det(Dm) / 6.0

        self.vol = vol
        self.inv_Dm = np.linalg.inv(Dm)

    def _deformation_gradients(self, position: np.ndarray) -> np.ndarray:
        """Return per-element deformation gradients F = Ds * Dm^{-1}."""
        e = self.elements
        p0 = position[e[:, 0]]
        dx = position[e[:, 1:]] - p0[:, None, :]  # (Ne, 3, 3)
        Ds = dx.transpose(0, 2, 1)  # columns are x1-x0, x2-x0, x3-x0
        return Ds @ self.inv_Dm

    def _elastic_forces(self, position: np.ndarray) -> np.ndarray:
        """Corotated linear-elastic nodal forces."""
        F = self._deformation_gradients(position)
        U, s, Vt = np.linalg.svd(F)
        R = U @ Vt
        trS = s.sum(axis=1, keepdims=True)  # trace(S)
        # P = 2*mu*(F - R) + lambda * (tr(S) - 3) * R
        P = 2.0 * self.mu * (F - R) + self.lam * (trS[:, :, None] - 3.0) * R

        # H = V0 * P * Dm^{-T}
        H = P @ self.inv_Dm.transpose(0, 2, 1)
        H *= self.vol[:, None, None]

        f = np.zeros((self.n, 3), dtype=np.float64)
        # node 0
        f0 = H.sum(axis=2)
        np.add.at(f, self.elements[:, 0], f0)
        # nodes 1..3
        for i in range(3):
            np.add.at(f, self.elements[:, i + 1], -H[:, :, i])
        return f

    def _contact_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Penalty force from the fixed glass wall at x = 0."""
        f = np.zeros_like(position)
        depth = -position[:, 0]
        in_contact = depth > 0.0
        if not np.any(in_contact):
            return f
        f[in_contact, 0] = (
            self.k_wall * depth[in_contact]
            - self.c_wall * velocity[in_contact, 0]
        )
        return f

    def _project_wall(self) -> None:
        """Enforce the non-penetration constraint x >= 0 with slight restitution."""
        penetrated = self.position[:, 0] < 0.0
        if not np.any(penetrated):
            return
        self.position[penetrated, 0] = 0.0
        self.velocity[penetrated, 0] = -0.1 * np.abs(self.velocity[penetrated, 0])

    def _substep(self, h: float) -> None:
        f_elastic = self._elastic_forces(self.position)
        f_contact = self._contact_forces(self.position, self.velocity)

        # Accumulate external impulse for the budget fields.
        self.cum_contact_impulse += f_contact.sum(axis=0) * h

        f_total = (
            f_elastic
            + f_contact
            - self.damping * self.mass[:, None] * self.velocity
        )

        # Symplectic Euler.
        self.velocity += h * (f_total / self.mass[:, None])
        self.position += h * self.velocity

        # Hard position constraint for the wall.
        self._project_wall()

    def advance_to(self, time: float) -> StateFrame:
        if time > self.time:
            elapsed = time - self.time
            n = max(1, int(round(elapsed / self.dt_sub)))
            h = elapsed / n
            for _ in range(n):
                self._substep(h)
            self.time = time

        F = self._deformation_gradients(self.position)
        contact_f = self._contact_forces(self.position, self.velocity)
        contact_total = contact_f.sum(axis=0)

        return StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "component": self.component.copy(),
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "budget_cumulative_external_impulse": self.cum_contact_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": np.ones((1, self.n), dtype=np.float64),
                "budget_external_force": contact_total.reshape(1, 1, 3).copy(),
            },
            scalars={
                "reference_momentum": float(self.ref_momentum),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt_sub": self.dt_sub,
            "num_nodes": self.n,
            "num_elements": self.ne,
        }
