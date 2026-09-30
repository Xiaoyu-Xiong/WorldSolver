"""Deterministic elastic-solid solver for the octopus cylinder squash scene.

Implements a small-strain-stable explicit FEM with a stable Neo-Hookean
hyperelastic model on a one-point tetrahedralization of the input surface,
plus kinematic cylinder contact via a velocity projection / penalty mixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Scene / geometry helpers
# ---------------------------------------------------------------------------

def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            # OBJ uses 1-based indices
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _quaternion_to_matrix(qxyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = qxyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# BenchmarkSolver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        # Configuration is stored on the BenchmarkScene instance.
        self.config: dict = getattr(scene, "config", {})

        # Gravity
        g = self.config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(g, dtype=float)

        # Initial scene state
        init_frame = scene.initial_state()
        x0_surf = np.asarray(init_frame.position, dtype=float)  # [N_surf, 3]
        v0_surf = np.asarray(init_frame.velocity, dtype=float)
        mass_surf = np.asarray(init_frame.mass, dtype=float)
        n_surf = x0_surf.shape[0]

        # Add an interior node to tetrahedralize the closed-ish visual surface.
        interior = x0_surf.mean(axis=0)
        self._x = np.vstack([x0_surf, interior])
        self.v = np.vstack([v0_surf, np.zeros((1, 3), dtype=float)])
        self.N = self._x.shape[0]

        # Alias for clarity
        self.x0 = self._x.copy()

        # Total mass from the scene is 12 kg; distribute among all nodes.
        total_mass = float(mass_surf.sum())
        self.mass = np.full(self.N, total_mass / self.N, dtype=float)

        # Parse the visual mesh and build tetrahedra (face + interior node).
        obj_path = Path(__file__).resolve().parent.parent / "assets" / "octopus.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(obj_path)
        vertices, faces = _parse_obj(obj_path)
        if not np.allclose(vertices, x0_surf):
            # If the scene has already reordered vertices, fall back to the
            # provided surface positions.
            vertices = x0_surf.copy()

        self.faces = faces
        interior_idx = self.N - 1
        tets: list[list[int]] = []
        rest_data: list[tuple[np.ndarray, float]] = []
        for a, b, c in faces:
            tet = [int(a), int(b), int(c), int(interior_idx)]
            d1 = self.x0[tet[0]] - self.x0[tet[3]]
            d2 = self.x0[tet[1]] - self.x0[tet[3]]
            d3 = self.x0[tet[2]] - self.x0[tet[3]]
            D = np.column_stack([d1, d2, d3])
            det = np.linalg.det(D)
            if det < 0.0:
                # Swap two vertices to obtain a positive reference volume
                tet[0], tet[1] = tet[1], tet[0]
                d1 = self.x0[tet[0]] - self.x0[tet[3]]
                d2 = self.x0[tet[1]] - self.x0[tet[3]]
                d3 = self.x0[tet[2]] - self.x0[tet[3]]
                D = np.column_stack([d1, d2, d3])
                det = np.linalg.det(D)
            B = np.linalg.inv(D)  # D_X^{-1}
            V0 = abs(det) / 6.0
            tets.append(tet)
            rest_data.append((B, V0))

        self.tets = np.asarray(tets, dtype=int)
        self.tet_B = [B for B, _ in rest_data]
        self.tet_V0 = np.array([V0 for _, V0 in rest_data], dtype=float)
        self.Ne = self.tets.shape[0]

        # Kinematic cylinder motion (hardcoded fallback + config override)
        self._upper_keyframes = self._load_upper_keyframes()

        # Static cylinders: list of (center, radius, half_height)
        self.cylinders = [
            (np.array([-0.72, 0.0, 0.72]), 0.58, 3.0),
            (np.array([0.72, 0.0, 0.72]), 0.58, 3.0),
            (np.array([0.0, 0.0, 1.72]), 0.58, 3.0),
        ]

        # Elastic parameters (Lamé)
        self.mu = 8.0e3      # Shear modulus [Pa]
        self.lam = 4.0e4    # First Lamé parameter [Pa]

        # Time integration
        self.sub_dt = 2.0e-4  # 100 substeps per sample_dt=0.02
        self.current_time = 0.0

        # Damping
        self.damping_rate = 2.0  # s^-1

        # Contact
        self.k_pen = 3.0e4   # penalty stiffness [N/m]
        self.k_damp = 400.0  # contact damping [N*s/m]

        # Budget bookkeeping (single global domain for the octopus)
        self.G = 1
        self.K = 1  # only gravity as external force term (contact handled implicitly)
        self.cum_external_impulse = np.zeros(3, dtype=float)
        self.cum_inflow = np.zeros(3, dtype=float)
        self.cum_outflow = np.zeros(3, dtype=float)

        # Reference momentum scale
        char_speed = np.sqrt(max(0.0, -self.gravity[2]))  # sqrt(g)
        self.reference_momentum = max(1.0, total_mass * char_speed)

        self._diagnostics: dict[str, object] = {"initialized": True}

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._x = value

    def _load_upper_keyframes(self) -> list[tuple[float, float]]:
        motion = self.config.get("upper_cylinder_motion", {})
        keyframes = motion.get("keyframes", None)
        if keyframes is None:
            return [(0.0, 4.2), (2.4, 4.2), (3.8, 2.38), (4.5, 2.38), (5.5, 4.2), (8.0, 4.2)]
        parsed = [(float(k["time"]), float(k["translation"][2])) for k in keyframes]
        parsed.sort(key=lambda x: x[0])
        return parsed

    def _upper_z(self, t: float) -> tuple[float, float]:
        """Return (z_position, z_velocity) for the upper cylinder at time t."""
        kf = self._upper_keyframes
        if t <= kf[0][0]:
            return kf[0][1], 0.0
        if t >= kf[-1][0]:
            return kf[-1][1], 0.0
        for i in range(len(kf) - 1):
            t0, z0 = kf[i]
            t1, z1 = kf[i + 1]
            if t0 <= t <= t1:
                if abs(t1 - t0) < 1e-12:
                    return z0, 0.0
                alpha = (t - t0) / (t1 - t0)
                z = z0 + alpha * (z1 - z0)
                v = (z1 - z0) / (t1 - t0)
                return z, v
        return kf[-1][1], 0.0

    # -----------------------------------------------------------------------
    # Physics kernels
    # -----------------------------------------------------------------------

    def _deformation_gradient(self, tet: np.ndarray, B: np.ndarray) -> np.ndarray:
        """F = D_x * B where D_x = [x1-x4, x2-x4, x3-x4]."""
        x = self.x
        d1 = x[tet[0]] - x[tet[3]]
        d2 = x[tet[1]] - x[tet[3]]
        d3 = x[tet[2]] - x[tet[3]]
        D = np.column_stack([d1, d2, d3])
        return D @ B

    def _first_piola(self, F: np.ndarray) -> np.ndarray:
        """Stable Neo-Hookean first Piola stress, regularised by SVD clamping."""
        try:
            U, s, Vt = np.linalg.svd(F)
        except np.linalg.LinAlgError:
            return np.zeros((3, 3), dtype=float)
        # Avoid collapse / inversion during extreme compression.
        s = np.maximum(s, 1e-4)
        J = np.prod(s)
        F_bar = U @ np.diag(s) @ Vt
        try:
            FinvT = np.linalg.inv(F_bar).T
        except np.linalg.LinAlgError:
            FinvT = np.zeros_like(F_bar)
        logJ = np.log(max(J, 1e-6))
        P = self.mu * (F_bar - FinvT) + self.lam * logJ * FinvT
        return P

    def _cylinder_contact_data(
        self, p: np.ndarray, center: np.ndarray, radius: float, half_height: float
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (normal, signed_distance) for a finite cylinder along y."""
        d = p - center
        # Cylinder axis is y; half_height is along y.
        if abs(d[1]) > half_height:
            return None, None
        rho = np.hypot(d[0], d[2])
        if rho >= radius:
            return None, None
        if rho < 1e-9:
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            n = np.array([d[0] / rho, 0.0, d[2] / rho], dtype=float)
        return n, radius - rho

    def _integrate(self, dt: float, t: float) -> None:
        """Advance positions and velocities by one substep."""
        n = self.N
        f = np.zeros((n, 3), dtype=float)

        # Gravity
        f += self.mass[:, None] * self.gravity[None, :]

        # Elastic forces from each tetrahedron
        for idx in range(self.Ne):
            tet = self.tets[idx]
            B = self.tet_B[idx]
            V0 = self.tet_V0[idx]
            F = self._deformation_gradient(tet, B)
            P = self._first_piola(F)
            # Nodal gradients in material coordinates
            g1 = B.T[:, 0]
            g2 = B.T[:, 1]
            g3 = B.T[:, 2]
            g4 = -(g1 + g2 + g3)
            f[tet[0]] -= V0 * (P @ g1)
            f[tet[1]] -= V0 * (P @ g2)
            f[tet[2]] -= V0 * (P @ g3)
            f[tet[3]] -= V0 * (P @ g4)

        # Upper cylinder position/velocity at this instant
        upper_z, upper_vz = self._upper_z(t)
        upper_center = np.array([0.0, 0.0, upper_z], dtype=float)
        upper_vel = np.array([0.0, 0.0, upper_vz], dtype=float)

        # Contact penalty against all cylinders (also used for budget)
        contact_force_total = np.zeros(3, dtype=float)
        for i in range(n):
            pi = self.x[i]
            vi = self.v[i]
            # static cylinders
            for center, radius, half_height in self.cylinders:
                n_c, sd = self._cylinder_contact_data(pi, center, radius, half_height)
                if n_c is not None and sd > 0.0:
                    v_surf = np.zeros(3, dtype=float)
                    rel = vi - v_surf
                    f_pen = self.k_pen * sd * n_c - self.k_damp * rel
                    f[i] += f_pen
                    contact_force_total += f_pen
            # upper kinematic cylinder
            n_c, sd = self._cylinder_contact_data(pi, upper_center, 0.62, 3.0)
            if n_c is not None and sd > 0.0:
                rel = vi - upper_vel
                f_pen = self.k_pen * sd * n_c - self.k_damp * rel
                f[i] += f_pen
                contact_force_total += f_pen

        # Symplectic Euler velocity update
        self.v += dt * f / self.mass[:, None]

        # Global velocity damping
        self.v *= (1.0 - self.damping_rate * dt)

        # Hard normal velocity projection (static + kinematic cylinders)
        for i in range(n):
            pi = self.x[i]
            vi = self.v[i]
            # static
            for center, radius, half_height in self.cylinders:
                n_c, sd = self._cylinder_contact_data(pi, center, radius, half_height)
                if n_c is not None:
                    self.v[i] = vi - np.dot(vi, n_c) * n_c
            # upper
            n_c, sd = self._cylinder_contact_data(pi, upper_center, 0.62, 3.0)
            if n_c is not None:
                rel_n = np.dot(vi - upper_vel, n_c)
                self.v[i] = vi - rel_n * n_c

        # Position update
        self.x += dt * self.v

        # Position projection to remove any residual penetration
        for i in range(n):
            pi = self.x[i]
            # static
            for center, radius, half_height in self.cylinders:
                n_c, sd = self._cylinder_contact_data(pi, center, radius, half_height)
                if n_c is not None and sd > 0.0:
                    self.x[i] += sd * n_c
            # upper
            n_c, sd = self._cylinder_contact_data(pi, upper_center, 0.62, 3.0)
            if n_c is not None and sd > 0.0:
                self.x[i] += sd * n_c

        # Budget updates (gravity + explicit contact penalty)
        gravity_total = self.mass.sum() * self.gravity
        self.cum_external_impulse += (gravity_total + contact_force_total) * dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            return self._build_frame(self.current_time)
        while self.current_time < time - 1e-12:
            dt = min(self.sub_dt, time - self.current_time)
            self._integrate(dt, self.current_time + dt / 2.0)
            self.current_time += dt
        return self._build_frame(self.current_time)

    def _build_frame(self, time: float) -> StateFrame:
        # Compute current element deformation gradients
        F_array = np.empty((self.Ne, 3, 3), dtype=float)
        for idx in range(self.Ne):
            F_array[idx] = self._deformation_gradient(self.tets[idx], self.tet_B[idx])

        # External force budget (gravity + current contact penalty)
        contact_total = np.zeros(3, dtype=float)
        upper_z, _ = self._upper_z(time)
        upper_center = np.array([0.0, 0.0, upper_z], dtype=float)
        for i in range(self.N):
            pi = self.x[i]
            # static
            for center, radius, half_height in self.cylinders:
                n_c, sd = self._cylinder_contact_data(pi, center, radius, half_height)
                if n_c is not None and sd > 0.0:
                    f_pen = self.k_pen * sd * n_c - self.k_damp * self.v[i]
                    contact_total += f_pen
            # upper
            n_c, sd = self._cylinder_contact_data(pi, upper_center, 0.62, 3.0)
            if n_c is not None and sd > 0.0:
                f_pen = self.k_pen * sd * n_c - self.k_damp * (self.v[i] - np.array([0.0, 0.0, upper_z], dtype=float))
                contact_total += f_pen

        gravity_total = self.mass.sum() * self.gravity

        # budget_external_force: one domain, two force terms (gravity, contact)
        budget_external_force = np.zeros((self.G, 2, 3), dtype=float)
        budget_external_force[0, 0, :] = gravity_total
        budget_external_force[0, 1, :] = contact_total

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "rest_position": self.x0.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "element_deformation_gradient": F_array,
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy()[None, :],
                "budget_cumulative_inflow_momentum": np.zeros((self.G, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((self.G, 3), dtype=float),
                "budget_domain_weights": np.ones((self.G, self.N), dtype=float),
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": float(self.gravity[2]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "num_nodes": int(self.N),
            "num_elements": int(self.Ne),
            "sub_dt": float(self.sub_dt),
            "mu": float(self.mu),
            "lambda": float(self.lam),
            **self._diagnostics,
        }
