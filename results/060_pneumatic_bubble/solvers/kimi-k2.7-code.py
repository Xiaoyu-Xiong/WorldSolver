"""Solver Agent-owned implementation for the pneumatic bubble chamber.

Implements a neo-Hookean finite-element solver on the tetrahedralization
implied by the chamber wall's closed surface.  The wall is treated as a
thin-ish volumetric shell; the mesh is a stack of two concentric vertex
rings (outer then inner), each extruded around a circle.  We add a simple
fan tetrahedralization between adjacent rings and the two layers, consistent
with the surface provided by ``scene.py``.

The material model is a nearly-incompressible neo-Hookean solid.  Internal
pressure is applied as a uniform traction on the inner surface triangles,
ramped from 0 to the configured pressure.

Fixed vertices (base and rim rings) have zero velocity and position pinned
to their rest values.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.config: dict | None = None
        self.scene = None
        self.seed = 0

        self.X: NDArray[np.float64] | None = None
        self.fixed: NDArray[np.float64] | None = None
        self.faces: NDArray[np.int32] | None = None
        self.body_id: NDArray[np.int32] | None = None
        self.mass: NDArray[np.float64] | None = None
        self.surface_faces: NDArray[np.int32] | None = None

        self.tets: NDArray[np.int32] | None = None
        self.tet_volume0: NDArray[np.float64] | None = None
        self.tet_Dm_inv: NDArray[np.float64] | None = None

        self.q: NDArray[np.float64] | None = None
        self.v: NDArray[np.float64] | None = None
        self.t: float = 0.0

        self.E: float = 1.0e5
        self.nu: float = 0.499
        self.rho: float = 1070.0
        self.mu: float = 0.0
        self.lam: float = 0.0

        self.pressure: float = 0.0
        self.ramp_start: float = 0.25
        self.ramp_end: float = 2.5
        self.hold_until: float = 4.0

        self.dt: float = 1.0e-4
        self.damping: float = 2.0

        self.volume0: NDArray[np.float64] | None = None
        self.element_F: NDArray[np.float64] | None = None

        self.reference_force: float = 1.0
        self._inner_face_mask: NDArray[np.bool_] | None = None

    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config
        self.scene = scene
        self.seed = seed

        frame = scene.initial_state()
        self.q = np.asarray(frame.position, dtype=np.float64).copy()
        self.v = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.X = self.q.copy()
        self.t = float(frame.time)
        self.fixed = np.asarray(frame.attributes["fixed"], dtype=np.float64)
        self.faces = np.asarray(frame.fields["surface_faces"], dtype=np.int32)
        self.body_id = np.asarray(frame.body_id, dtype=np.int32)
        self.mass = np.asarray(frame.mass, dtype=np.float64)
        self.surface_faces = self.faces.copy()

        n = self.q.shape[0]

        mat = self.config.get("material_reference", {})
        self.E = float(mat.get("youngs_modulus_pa", 1.0e5))
        self.nu = float(mat.get("poisson_ratio", 0.499))
        self.rho = float(mat.get("density_kg_m3", 1070.0))

        loading = self.config.get("loading", {})
        self.pressure = float(loading.get("pressure_pa", 4000.0))
        self.ramp_start = float(loading.get("ramp_start_s", 0.25))
        self.ramp_end = float(loading.get("ramp_end_s", 2.5))
        self.hold_until = float(loading.get("hold_until_s", 4.0))

        l2m = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.lam = self.nu * l2m
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        self.reference_force = self.pressure * 1.0e-3

        self.tets = self._build_tets(n)
        if self.tets.shape[0] == 0:
            raise RuntimeError("No valid tetrahedral elements were generated")

        self.tet_volume0 = np.zeros(self.tets.shape[0], dtype=np.float64)
        self.tet_Dm_inv = np.zeros((self.tets.shape[0], 3, 3), dtype=np.float64)
        for e, tet in enumerate(self.tets):
            x0, x1, x2, x3 = self.X[tet]
            Dm = np.column_stack((x1 - x0, x2 - x0, x3 - x0))
            self.tet_volume0[e] = np.linalg.det(Dm) / 6.0
            self.tet_Dm_inv[e] = np.linalg.inv(Dm)

        self.volume0 = self._per_vertex_volume(self.tets, self.tet_volume0, n)
        self.element_F = np.tile(np.eye(3, dtype=np.float64)[None, ...], (self.tets.shape[0], 1, 1))

        self._inner_face_mask = self._detect_inner_faces(self.X)

    def _detect_inner_faces(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        centroids = X[self.faces].mean(axis=1)
        radii = np.linalg.norm(centroids[:, :2], axis=1)
        inner_layer_radii = []
        for f in self.faces:
            r = np.linalg.norm(X[f, :2], axis=1)
            inner_layer_radii.append(np.mean(r))
        inner_layer_radii = np.asarray(inner_layer_radii)
        threshold = 0.06
        is_inner = (radii < threshold) & (inner_layer_radii < threshold)
        if not np.any(is_inner):
            is_inner = radii < np.median(radii)
        return is_inner

    def _build_tets(self, n: int) -> NDArray[np.int32]:
        if n % 14 != 0:
            raise ValueError(f"Vertex count {n} is not compatible with 2*nr*n_theta mesh")
        n_theta = n // 14
        nr = 7
        outer = np.arange(0, nr * n_theta, dtype=np.int32)
        inner = np.arange(nr * n_theta, 2 * nr * n_theta, dtype=np.int32)

        tets = []
        for j in range(nr - 1):
            for i in range(n_theta):
                i1 = (i + 1) % n_theta
                o0 = outer[j * n_theta + i]
                o1 = outer[j * n_theta + i1]
                o2 = outer[(j + 1) * n_theta + i1]
                o3 = outer[(j + 1) * n_theta + i]
                in0 = inner[j * n_theta + i]
                in1 = inner[j * n_theta + i1]
                in2 = inner[(j + 1) * n_theta + i1]
                in3 = inner[(j + 1) * n_theta + i]
                tets.append([o0, o1, o3, in0])
                tets.append([o1, o2, o3, in1])
                tets.append([in0, in1, in3, o0])
                tets.append([in1, in2, in3, o3])

        tets = np.asarray(tets, dtype=np.int32)
        valid = []
        for tet in tets:
            x0, x1, x2, x3 = self.X[tet]
            vol = np.linalg.det(np.column_stack((x1 - x0, x2 - x0, x3 - x0))) / 6.0
            if vol > 0.0:
                valid.append(tet)
        return np.asarray(valid, dtype=np.int32)

    def _per_vertex_volume(self, tets: NDArray[np.int32], tet_volume: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        volume = np.zeros(n, dtype=np.float64)
        for tet, vol in zip(tets, tet_volume):
            volume[tet] += vol / 4.0
        return volume

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.t:
            return self._current_frame()
        while self.t + self.dt < target:
            self._step(self.dt)
        if target > self.t:
            self._step(target - self.t)
        return self._current_frame()

    def _current_pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return 0.0
        if t >= self.ramp_end:
            return self.pressure
        s = (t - self.ramp_start) / (self.ramp_end - self.ramp_start)
        return self.pressure * s

    def _step(self, dt: float) -> None:
        q_old = self.q.copy()
        v_old = self.v.copy()

        f = self._compute_forces(q_old)
        f -= self.damping * self.v

        a = f / self.mass[:, None]
        a[self.fixed > 0.5] = 0.0

        v_new = v_old + dt * a
        q_new = q_old + dt * v_new
        v_new[self.fixed > 0.5] = 0.0
        q_new[self.fixed > 0.5] = self.X[self.fixed > 0.5]

        self.v = v_new
        self.q = q_new
        self.t += dt

    def _compute_forces(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        f = np.zeros_like(q)

        F, P = self._compute_stress(q)
        for e, tet in enumerate(self.tets):
            H = -self.tet_volume0[e] * P[e] @ self.tet_Dm_inv[e].T
            f[tet[1]] += H[:, 0]
            f[tet[2]] += H[:, 1]
            f[tet[3]] += H[:, 2]
            f[tet[0]] -= H[:, 0] + H[:, 1] + H[:, 2]

        p = self._current_pressure(self.t)
        if p > 0.0:
            for fi, fidx in enumerate(self.faces):
                if not self._inner_face_mask[fi]:
                    continue
                pts = q[fidx]
                nvec = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                area = 0.5 * np.linalg.norm(nvec)
                normal = nvec / (2.0 * area + 1e-20)
                force = p * area * normal
                for vi in fidx:
                    f[vi] += force / 3.0

        return f

    def _compute_stress(self, q: NDArray[np.float64]):
        F = np.zeros((self.tets.shape[0], 3, 3), dtype=np.float64)
        P = np.zeros((self.tets.shape[0], 3, 3), dtype=np.float64)
        for e, tet in enumerate(self.tets):
            x0, x1, x2, x3 = q[tet]
            Ds = np.column_stack((x1 - x0, x2 - x0, x3 - x0))
            Fe = Ds @ self.tet_Dm_inv[e]
            F[e] = Fe
            J = np.linalg.det(Fe)
            J = max(J, 0.1)
            FinvT = np.linalg.inv(Fe).T
            P[e] = self.mu * (Fe - FinvT) + self.lam * np.log(J) * FinvT
        self.element_F = F
        return F, P

    def _current_frame(self) -> StateFrame:
        n = self.q.shape[0]
        vol = self._per_vertex_volume(self.tets, self.tet_volume0, n)
        mass_quantity = self.rho * vol

        f_ext = np.zeros_like(self.q)
        p = self._current_pressure(self.t)
        if p > 0.0:
            for fi, fidx in enumerate(self.faces):
                if not self._inner_face_mask[fi]:
                    continue
                pts = self.q[fidx]
                nvec = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                area = 0.5 * np.linalg.norm(nvec)
                normal = nvec / (2.0 * area + 1e-20)
                force = p * area * normal
                for vi in fidx:
                    f_ext[vi] += force / 3.0

        budget_external_force = f_ext[None, None, :, :]
        cumulative_mass_source = np.zeros(1, dtype=np.float64)

        return StateFrame(
            time=self.t,
            position=self.q.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "fixed": self.fixed.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "surface_faces": self.surface_faces.copy(),
                "element_deformation_gradient": self.element_F.copy(),
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "internal_pressure_pa": self._current_pressure(self.t),
                "reference_force": self.reference_force,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "t": self.t,
            "n_vertices": self.q.shape[0] if self.q is not None else 0,
            "n_elements": self.tets.shape[0] if self.tets is not None else 0,
        }
