"""3D Neo-Hookean FEM solver for the rubber-cube compression benchmark.

Implements an explicit central-difference time integrator with:
  * lock-free (non-locking) Neo-Hookean hyperelasticity,
  * uniform lumped-mass tetrahedral FEM,
  * a fixed bottom support (z=0),
  * a ramped-downward surface traction on the top face (z=0.2),
  * and a light velocity-proportional damping to prevent explicit blow-up.

The solver writes every required raw trajectory field:
  attributes.mass_quantity
  fields.budget_external_force
  fields.cumulative_mass_source
  fields.element_deformation_gradient
  scalars.reference_force

No Genesis/gs imports are used; only NumPy/SciPy are required.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit nonlinear elastodynamics for a compressed rubber cube."""

    # Rubber-like solid (SI units, approximately).
    YOUNG_MODULUS = 2.0e6  # Pa
    POISSON_RATIO = 0.45  # nearly incompressible rubber
    DENSITY = 1050.0  # kg/m^3
    DAMPING = 0.04  # mass-proportional Rayleigh-like damping
    GRAVITY = -9.81  # m/s^2 in z

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.rest_pos = scene.attributes["rest_position"].copy()
        self.position = self.rest_pos.copy()
        self.velocity = np.zeros_like(self.position)
        self.mass = scene.mass.copy() if scene.mass is not None else np.full(len(self.position), 1.0 / len(self.position))

        # Tetrahedral topology.
        self.tets = scene.tetrahedra.copy()
        self.faces = scene.faces.copy()
        n = len(self.position)
        self.n = n
        self.ne = len(self.tets)

        # Reference shape matrices and volume weights.
        self.Dm_inv = np.zeros((self.ne, 3, 3))
        self.vol = np.zeros(self.ne)
        for e, tet in enumerate(self.tets):
            X = self.rest_pos[tet]  # (4, 3)
            D = X[1:] - X[0]  # (3, 3)
            det = np.linalg.det(D)
            if det <= 0.0:
                # If degenerate, try permuting to make a right-handed basis.
                D = X[[1, 2, 0]] - X[0]
                det = np.linalg.det(D)
            self.vol[e] = abs(det) / 6.0
            self.Dm_inv[e] = np.linalg.inv(D)

        # Per-vertex lumped mass.
        self.mass_v = np.zeros(n)
        for e, tet in enumerate(self.tets):
            for v in tet:
                self.mass_v[v] += self.DENSITY * self.vol[e] / 4.0
        self.mass_v = np.maximum(self.mass_v, 1e-12)

        # Face areas and normals for top/bottom faces.
        self._compute_boundary_data()

        # Precompute which vertices are on the top and bottom faces.
        self.bottom_mask = scene.attributes["bottom_face"].astype(bool)
        self.top_mask = scene.attributes["top_face"].astype(bool)

        # Lame parameters (per element constant).
        mu = self.YOUNG_MODULUS / (2.0 * (1.0 + self.POISSON_RATIO))
        lam = self.YOUNG_MODULUS * self.POISSON_RATIO / ((1.0 + self.POISSON_RATIO) * (1.0 - 2.0 * self.POISSON_RATIO))
        self.mu = mu
        self.lam = lam

        # Traction schedule: ramp from 0 to 50 kPa between t=0.25 and 1.5, hold.
        times = np.array([0.0, 0.25, 1.5, 2.0], dtype=float)
        scales = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
        self._traction_scale = interp1d(times, scales, kind="linear", fill_value="extrapolate")

        # Simulation time step.
        self.dt = 2.0e-5
        self._time = 0.0
        self._step = 0

        # Pre-store the initial undeformed frame.
        self._last_frame = self._make_frame(0.0)

    def _compute_boundary_data(self):
        self.top_area = 0.0
        self.top_normals = []
        self.top_tri_verts = []
        self.bottom_area = 0.0
        for f in self.faces:
            verts = f
            p0, p1, p2 = self.rest_pos[verts[0]], self.rest_pos[verts[1]], self.rest_pos[verts[2]]
            v0 = p1 - p0
            v1 = p2 - p0
            n = np.cross(v0, v1)
            area = 0.5 * np.linalg.norm(n)
            if area == 0:
                continue
            normal = n / (2.0 * area)
            z = self.rest_pos[verts, 2]
            if np.allclose(z, 0.2, atol=1e-6):
                self.top_area += area
                self.top_tri_verts.append(verts)
                self.top_normals.append(normal)
            elif np.allclose(z, 0.0, atol=1e-6):
                self.bottom_area += area

    def _compute_external_force(self, t):
        """Return per-vertex external force (gravity + traction only)."""
        f = np.zeros_like(self.position)

        # Gravity.
        f[:, 2] += self.mass_v * self.GRAVITY

        # Top surface traction.
        traction_total = self._traction(t)
        if traction_total != 0.0 and self.top_area > 0.0:
            # Distribute traction force over top vertices based on vertex area weight.
            top_idx = np.where(self.top_mask)[0]
            f_per_vertex = traction_total / max(len(top_idx), 1)
            f[top_idx, 2] -= f_per_vertex

        return f

    def _compute_internal_force(self, pos, vel):
        """Return per-vertex internal elastic + damping force, vectorized."""
        # Elastic forces.
        F = self._deformation_gradient(pos)
        P = self._first_piola_kirchhoff(F)
        # f_i per element = V * P * Dm_inv^T, where columns correspond to nodes 1-3.
        # (ne,3,3)
        P_DmT = P @ self.Dm_inv.transpose(0, 2, 1)
        forces123 = self.vol[:, None, None] * P_DmT  # (ne,3,3)
        force0 = -forces123.sum(axis=2)  # (ne,3)

        f = np.zeros_like(pos)
        # Node 0.
        np.add.at(f, self.tets[:, 0], force0)
        # Nodes 1-3.
        for i in range(3):
            np.add.at(f, self.tets[:, i + 1], forces123[:, :, i])

        # Damping force (mass-proportional).
        f -= self.DAMPING * self.mass_v[:, None] * vel

        return f

    def _deformation_gradient(self, pos):
        # Vectorized computation.
        x0 = pos[self.tets[:, 0]]
        x1 = pos[self.tets[:, 1]]
        x2 = pos[self.tets[:, 2]]
        x3 = pos[self.tets[:, 3]]
        Ds = np.stack([x1 - x0, x2 - x0, x3 - x0], axis=2)  # (ne,3,3)
        F = Ds @ self.Dm_inv
        return F

    def _first_piola_kirchhoff(self, F):
        """Neo-Hookean first Piola-Kirchhoff stress."""
        J = np.linalg.det(F)
        J = np.where(J > 1e-9, J, 1e-9)
        F_inv_T = np.linalg.inv(F).transpose(0, 2, 1)
        logJ = np.log(J)
        P = self.mu * (F - F_inv_T) + self.lam * logJ[:, None, None] * F_inv_T
        return P

    def _traction(self, t):
        scale = float(self._traction_scale(t))
        return 50000.0 * scale * self.top_area

    def _make_frame(self, t):
        # Build element-wise deformation gradient.
        F = self._deformation_gradient(self.position)

        # External force budget: per-vertex external forces grouped into one domain, K=1.
        ext_force = self._compute_external_force(t)
        # budget_external_force shape [G, K, 3]. Here G=1 (entire cube), K=1.
        budget_external_force = np.zeros((1, 1, 3), dtype=float)
        budget_external_force[0, 0] = ext_force.sum(axis=0)

        # reference_force scalar: peak traction force.
        reference_force = 50000.0 * self.top_area

        return StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass_v.copy(),
            attributes={
                "rest_position": self.rest_pos.copy(),
                "bottom_face": self.bottom_mask.astype(float),
                "top_face": self.top_mask.astype(float),
                "mass_quantity": self.mass_v.copy(),
            },
            fields={
                "element_deformation_gradient": F.copy(),
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "applied_traction_pa": 50000.0 * float(self._traction_scale(t)),
                "reference_force": float(reference_force),
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._time:
            return self._last_frame

        # Sub-step from current time to target using the stored dt.
        n_steps = max(1, int(np.ceil((target - self._time) / self.dt)))
        dt = (target - self._time) / n_steps

        for _ in range(n_steps):
            # Symplectic explicit Euler (velocity then position).
            f_int = self._compute_internal_force(self.position, self.velocity)
            f_ext = self._compute_external_force(self._time)
            a = (f_int + f_ext) / self.mass_v[:, None]
            self.velocity += a * dt

            # Bottom support: zero out bottom velocities and positions.
            self.velocity[self.bottom_mask] = 0.0
            self.position[self.bottom_mask, 2] = 0.0

            self.position += self.velocity * dt

            # Bottom support post-update.
            self.position[:, 2] = np.maximum(self.position[:, 2], 0.0)
            self.velocity[self.bottom_mask] = 0.0
            self.position[self.bottom_mask, 2] = 0.0

            self._time += dt
            self._step += 1

        self._last_frame = self._make_frame(self._time)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_neohookean_fem",
            "dt": self.dt,
            "elements": self.ne,
            "vertices": self.n,
            "steps": self._step,
        }
