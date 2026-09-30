"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        np.random.seed(seed)

        init = scene.initial_state()
        self.x = init.position.copy()
        self.v = init.velocity.copy()
        self.mass = init.mass.copy()
        self.n = self.x.shape[0]
        self.rest_x = self.x.copy()
        self.time = 0.0

        # Build a tetrahedral discretisation of the subject.
        self._build_mesh()

        # Material parameters (stable Neo-Hookean model).
        self.mu = float(self.cfg.get("stiffness", 28.0))
        self.lam = self.mu * 0.5
        self.damping = float(self.cfg.get("damping", 0.42))
        self.g = np.asarray(self.cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # Stretch / pulse script parameters.
        self.stretch_ratio = float(self.cfg.get("stretch_ratio", 2.15))
        self.stretch_end = float(self.cfg.get("stretch_end", 1.25))
        self.pulse_time = float(self.cfg.get("pulse_time", 1.55))
        self.pulse_width = 0.1
        self.pulse_amp = 0.05

        # Boundary (grip) particles.
        self._init_boundary()

        # Choose a stable sub-step.
        rho = float(self.cfg.get("subject_mass", 1.3)) / max(np.sum(self.rest_volume), 1e-12)
        c = np.sqrt((self.lam + 2.0 * self.mu) / max(rho, 1e-12))
        h = np.power(self.rest_volume, 1.0 / 3.0)
        dt_cfl = 0.1 * np.min(h / c) if h.size else 1e-3
        self.dt = max(1e-4, min(1e-3, float(dt_cfl)))

    def _build_mesh(self) -> None:
        asset = Path(self.cfg.get("subject_asset", "assets/subject.npz"))
        elems = None
        try:
            with np.load(asset) as data:
                for key in ("tets", "elements", "tetrahedra"):
                    if key in data:
                        elems = np.asarray(data[key], dtype=int)
                        break
        except Exception:
            elems = None

        if elems is None or elems.ndim != 2 or elems.shape[1] != 4:
            from scipy.spatial import Delaunay

            tri = Delaunay(self.rest_x)
            elems = tri.simplices

        elems = np.asarray(elems, dtype=int)
        if elems.max() >= self.n:
            elems = elems - 1
        self.elems = elems

        # Precompute the rest shape matrix and volume of each tet.
        self._recompute_shape_quantities()

    def _recompute_shape_quantities(self) -> None:
        X = self.rest_x[self.elems]
        Dm = np.empty((X.shape[0], 3, 3))
        Dm[:, :, 0] = X[:, 1] - X[:, 0]
        Dm[:, :, 1] = X[:, 2] - X[:, 0]
        Dm[:, :, 2] = X[:, 3] - X[:, 0]

        det = np.linalg.det(Dm)
        self.rest_volume = np.abs(det) / 6.0
        valid = self.rest_volume > 1e-12
        if not valid.all():
            self.elems = self.elems[valid]
            Dm = Dm[valid]
            self.rest_volume = self.rest_volume[valid]

        self.Dm_inv = np.linalg.inv(Dm)

        # Distribute tet mass to vertices so total mass equals the prescribed value.
        total_mass = float(self.cfg.get("subject_mass", 1.3))
        total_volume = np.sum(self.rest_volume)
        density = total_mass / max(total_volume, 1e-12)
        tet_mass = density * self.rest_volume
        self.mass = np.zeros(self.n)
        np.add.at(self.mass, self.elems.ravel(), np.repeat(tet_mass / 4.0, 4))
        self.mass = np.maximum(self.mass, 1e-12)
        self.T = self.elems.shape[0]

    def _init_boundary(self) -> None:
        x = self.rest_x[:, 0]
        y = self.rest_x[:, 1]
        z = self.rest_x[:, 2]
        min_x = float(x.min())
        max_x = float(x.max())
        tol = 0.05 * max(max_x - min_x, 1e-6)

        # The scene geometry defines the grip y/z extents.
        gy_min, gy_max = -0.55, 0.55
        gz_min, gz_max = 0.31, 1.41

        left = (
            (x < min_x + tol)
            & (y >= gy_min)
            & (y <= gy_max)
            & (z >= gz_min)
            & (z <= gz_max)
        )
        right = (
            (x > max_x - tol)
            & (y >= gy_min)
            & (y <= gy_max)
            & (z >= gz_min)
            & (z <= gz_max)
        )

        if not left.any():
            left = x < min_x + tol
        if not right.any():
            right = x > max_x - tol

        self.left_idx = np.where(left)[0]
        self.right_idx = np.where(right)[0]
        self.left_face = min_x
        self.right_face = max_x
        self.left_off = self.rest_x[self.left_idx, 0] - self.left_face
        self.right_off = self.rest_x[self.right_idx, 0] - self.right_face

    def _ratio(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            r = 1.0
        elif t < self.stretch_end:
            r = 1.0 + (self.stretch_ratio - 1.0) * (t / self.stretch_end)
        else:
            r = self.stretch_ratio

        pt = self.pulse_time
        pw = self.pulse_width
        if pt <= t < pt + pw:
            r *= 1.0 - self.pulse_amp * np.sin(np.pi * (t - pt) / pw)
        return r

    def _elastic_forces(self) -> np.ndarray:
        x = self.x[self.elems]
        Ds = np.empty((x.shape[0], 3, 3))
        Ds[:, :, 0] = x[:, 1] - x[:, 0]
        Ds[:, :, 1] = x[:, 2] - x[:, 0]
        Ds[:, :, 2] = x[:, 3] - x[:, 0]

        F = Ds @ self.Dm_inv
        J = np.linalg.det(F)

        # Clamp near-singular determinants.
        J = np.where(np.abs(J) < 1e-10, np.sign(J + 1e-12) * 1e-10, J)

        try:
            FinvT = np.linalg.inv(F).transpose(0, 2, 1)
        except np.linalg.LinAlgError:
            FinvT = np.empty_like(F)
            for i in range(F.shape[0]):
                try:
                    FinvT[i] = np.linalg.inv(F[i]).T
                except np.linalg.LinAlgError:
                    FinvT[i] = np.linalg.pinv(F[i]).T

        # Stable Neo-Hookean strain energy derivative: psi = ||F||^2 - 3 - ln(J).
        # First Piola-Kirchhoff stress:
        #   P = mu * F + (lam * (J - 1) - mu) * J * F^{-T}.
        Jm1 = J - 1.0
        P = self.mu * F + (self.lam * Jm1[:, None, None] - self.mu) * (J[:, None, None] * FinvT)

        H = -self.rest_volume[:, None, None] * (P @ self.Dm_inv.transpose(0, 2, 1))

        f = np.zeros((self.n, 3))
        f0 = -(H[:, :, 0] + H[:, :, 1] + H[:, :, 2])
        np.add.at(f, self.elems[:, 0], f0)
        np.add.at(f, self.elems[:, 1], H[:, :, 0])
        np.add.at(f, self.elems[:, 2], H[:, :, 1])
        np.add.at(f, self.elems[:, 3], H[:, :, 2])
        return f

    def _apply_boundary(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        r0 = self._ratio(t0)
        r1 = self._ratio(t1)

        left_x0 = self.left_face * r0 + self.left_off
        right_x0 = self.right_face * r0 + self.right_off
        left_x1 = self.left_face * r1 + self.left_off
        right_x1 = self.right_face * r1 + self.right_off

        self.x[self.left_idx, 0] = left_x1
        self.x[self.right_idx, 0] = right_x1

        if dt > 0.0:
            self.v[self.left_idx, 0] = (left_x1 - left_x0) / dt
            self.v[self.right_idx, 0] = (right_x1 - right_x0) / dt
        else:
            self.v[self.left_idx, 0] = 0.0
            self.v[self.right_idx, 0] = 0.0
        self.v[self.left_idx, 1:] = 0.0
        self.v[self.right_idx, 1:] = 0.0

    def _step(self, dt: float) -> None:
        t0 = self.time
        t1 = t0 + dt

        f = self._elastic_forces()
        # Symplectic Euler with global velocity damping.
        self.v = (self.v + dt * f / self.mass[:, None]) / (1.0 + self.damping * dt)
        self.x = self.x + dt * self.v

        self._apply_boundary(t0, t1)
        self.time = t1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target:
            dt = min(self.dt, target - self.time)
            self._step(dt)
        return self._frame(target)

    def _vertex_deformation_gradient(self) -> np.ndarray:
        x = self.x[self.elems]
        Ds = np.empty((x.shape[0], 3, 3))
        Ds[:, :, 0] = x[:, 1] - x[:, 0]
        Ds[:, :, 1] = x[:, 2] - x[:, 0]
        Ds[:, :, 2] = x[:, 3] - x[:, 0]
        F_tet = Ds @ self.Dm_inv

        F_vert = np.zeros((self.n, 3, 3))
        weights = np.zeros(self.n)
        idx = self.elems.ravel()
        F_rep = np.repeat(F_tet, 4, axis=0)
        w_rep = np.repeat(self.rest_volume, 4)
        np.add.at(F_vert, idx, F_rep * w_rep[:, None, None])
        np.add.at(weights, idx, w_rep)
        weights = weights[:, None, None]
        F_vert = F_vert / np.maximum(weights, 1e-12)
        return F_vert

    @staticmethod
    def _sd_box(p: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
        q = np.abs(p - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _contact_signed_distance(self) -> np.ndarray:
        p = self.x
        # Ground slab.
        d = self._sd_box(p, np.array([0.0, 0.0, -0.04]), np.array([2.3, 1.7, 0.04]))
        # Left and right grip boxes.
        d = np.minimum(
            d,
            self._sd_box(
                p, np.array([-0.72, 0.0, 0.86]), np.array([0.12, 0.55, 0.55])
            ),
        )
        d = np.minimum(
            d,
            self._sd_box(
                p, np.array([0.72, 0.0, 0.86]), np.array([0.12, 0.55, 0.55])
            ),
        )
        return d

    def _frame(self, time: float) -> StateFrame:
        F = self._vertex_deformation_gradient()
        contact = self._contact_signed_distance()
        external = self.mass[:, None] * self.g

        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
                "contact_signed_distance": contact,
            },
            scalars={
                "contact_length_scale": float(self.cfg.get("contact_length_scale", 0.04)),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": getattr(self, "dt", None),
            "elements": getattr(self, "T", 0),
            "vertices": getattr(self, "n", 0),
        }
