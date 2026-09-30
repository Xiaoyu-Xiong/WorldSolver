"""Mass-spring physics solver for the suspended elastic bunny pendulum.

The bunny surface mesh is treated as a soft elastic membrane/network.  Each
surface edge behaves as a damped linear spring.  Gravity drives the pendulum
swing and the fixed crown patch stays attached to the suspension point.  The
integrator is explicit (symplectic Euler) with a sub-step chosen for stability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.config import load_yaml
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self._scene = None
        self._seed = 0

        # state
        self._x: np.ndarray | None = None  # [N,3]
        self._v: np.ndarray | None = None  # [N,3]
        self._x0: np.ndarray | None = None  # rest/initial positions
        self._fixed: np.ndarray | None = None  # [N] bool
        self._mass: np.ndarray | None = None  # [N]

        # mesh
        self._faces: np.ndarray | None = None  # [Ne,3]
        self._edges: np.ndarray | None = None  # [E,2]
        self._edge_rest: np.ndarray | None = None  # [E]

        # parameters
        self._g = np.array([0.0, 0.0, -9.81], dtype=float)
        self._k = 1500.0  # edge stiffness (N/m)
        self._zeta = 0.02  # damping ratio
        self._dt = 1e-4  # sub-step
        self._sample_dt = 0.02
        self._time = 0.0

        # budget accounting
        self._total_gravity_force = np.zeros(3, dtype=float)
        self._cumulative_external_impulse = np.zeros(3, dtype=float)
        self._reference_momentum = 1.0

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _parse_obj(path: Path):
        vertices = []
        faces = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()
                    # support f v/vt/vn and f v//vn forms
                    face = []
                    for token in parts[1:4]:
                        face.append(int(token.split("/")[0]) - 1)
                    faces.append(face)
        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)

    def _build_edges(self):
        edges = set()
        face_list = []
        for f in self._faces:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            face_list.append([i, j, k])
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))
        self._faces = np.asarray(face_list, dtype=int)
        self._edges = np.asarray(sorted(edges), dtype=int)
        self._edge_rest = np.zeros(len(self._edges), dtype=float)

    def _compute_masses(self, rho: float, thickness: float):
        n = len(self._x0)
        mass = np.zeros(n, dtype=float)
        x0 = self._x0
        for f in self._faces:
            i, j, k = f
            v1 = x0[j] - x0[i]
            v2 = x0[k] - x0[i]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            contrib = rho * thickness * area / 3.0
            mass[i] += contrib
            mass[j] += contrib
            mass[k] += contrib
        mass[mass < 1e-9] = 1e-9
        self._mass = mass

    def _element_deformation_gradients(self) -> np.ndarray:
        """Return per-face 3x3 deformation gradient (F = [q1,q2,q3] [e1,e2,n0]^{-1})."""
        x = self._x
        x0 = self._x0
        faces = self._faces
        i0 = faces[:, 0]
        i1 = faces[:, 1]
        i2 = faces[:, 2]

        e1 = x0[i1] - x0[i0]  # [Ne,3]
        e2 = x0[i2] - x0[i0]
        n0 = np.cross(e1, e2)  # [Ne,3]

        q1 = x[i1] - x[i0]
        q2 = x[i2] - x[i0]
        n = np.cross(q1, q2)

        # A0 and A as [Ne,3,3]
        A0 = np.stack([e1, e2, n0], axis=2)  # last dim is column index
        A = np.stack([q1, q2, n], axis=2)
        # solve F * A0 = A  =>  F = A * inv(A0)
        inv_A0 = np.linalg.inv(A0)
        F = A @ inv_A0
        return F

    # ------------------------------------------------------------------ public
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        config = load_yaml(Path("config.yaml"))
        self._g = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._sample_dt = float(config.get("sample_dt", 0.02))
        duration = float(config.get("duration", 10.0))

        # material parameters
        mat = config.get("material_reference", {})
        E = float(mat.get("youngs_modulus_pa", 30000.0))
        nu = float(mat.get("poisson_ratio", 0.45))
        rho = float(mat.get("density_kg_m3", 1000.0))
        # treat the bunny as a soft solid shell of finite thickness
        thickness = 0.05
        self._k = E * thickness  # N/m

        # initial state from the scene (poses already match the source)
        init = scene.initial_state()
        self._x0 = np.asarray(init.position, dtype=float).copy()
        self._x = self._x0.copy()
        self._v = np.asarray(init.velocity, dtype=float).copy()
        self._fixed = np.asarray(init.attributes["fixed"], dtype=bool)

        # load surface mesh topology
        _, faces = self._parse_obj(Path("assets/bunny.obj"))
        self._faces = faces
        self._build_edges()

        # per-vertex mass from surface area
        self._compute_masses(rho, thickness)

        # choose stable sub-step.  Estimate highest spring frequency using the
        # smallest effective mass connected by the stiffest edge.
        m_i = self._mass[self._edges[:, 0]]
        m_j = self._mass[self._edges[:, 1]]
        m_eff = (m_i * m_j) / (m_i + m_j)
        omega_max = np.sqrt(self._k / m_eff.max())
        # safety factor of 0.2 on the explicit spring stability limit
        dt_max = 0.2 * 2.0 / omega_max
        self._dt = min(1e-4, dt_max)
        if self._dt < 1e-6:
            self._dt = 1e-6
        # ensure that the number of substeps for a sample period is an integer
        self._dt = self._sample_dt / max(1, int(round(self._sample_dt / self._dt)))

        # budget constants
        self._total_gravity_force = self._mass.sum() * self._g
        self._cumulative_external_impulse = np.zeros(3, dtype=float)
        self._reference_momentum = self._mass.sum() * np.sqrt(np.linalg.norm(self._g) * 1.0)
        if self._reference_momentum <= 0.0 or not np.isfinite(self._reference_momentum):
            self._reference_momentum = 1.0

        self._time = 0.0

    def _step(self, dt: float) -> None:
        x = self._x
        v = self._v
        edges = self._edges
        fixed = self._fixed

        i = edges[:, 0]
        j = edges[:, 1]

        dx = x[j] - x[i]  # [E,3]
        dist = np.linalg.norm(dx, axis=1)
        # avoid division by zero for degenerate edges
        dist[dist < 1e-12] = 1e-12
        n = dx / dist[:, None]

        # spring force magnitude (linear springs)
        delta = dist - self._edge_rest
        f_spring = self._k * delta

        # damping along the edge
        dv = v[j] - v[i]
        v_n = np.einsum("ij,ij->i", dv, n)  # [E]
        # reduced mass per edge
        m_i = self._mass[i]
        m_j = self._mass[j]
        m_eff = (m_i * m_j) / (m_i + m_j)
        c = 2.0 * self._zeta * np.sqrt(self._k * m_eff)
        f_damp = c * v_n

        # total force on j from each edge (positive pulls j toward i)
        f_edge = (f_spring + f_damp)[:, None] * n

        # accumulate to nodes
        f = np.zeros_like(x)
        np.add.at(f, i, -f_edge)
        np.add.at(f, j, f_edge)

        # gravity
        f += self._mass[:, None] * self._g

        # update velocity and position (symplectic Euler)
        v_next = v + dt * f / self._mass[:, None]
        x_next = x + dt * v_next

        # enforce fixed boundary condition
        x_next[fixed] = self._x0[fixed]
        v_next[fixed] = 0.0

        self._v = v_next
        self._x = x_next
        self._time += dt
        self._cumulative_external_impulse += self._total_gravity_force * dt

    def _build_frame(self) -> StateFrame:
        n = len(self._x)
        ne = len(self._faces)

        # single global domain
        domain_weights = np.ones((1, n), dtype=float)
        ext_force = self._total_gravity_force.reshape(1, 1, 3)
        zero_vec = np.zeros((1, 3), dtype=float)

        F = self._element_deformation_gradients()

        frame = StateFrame(
            time=self._time,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={
                "fixed": self._fixed.astype(float),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": self._cumulative_external_impulse.copy().reshape(1, 3),
                "budget_cumulative_inflow_momentum": zero_vec.copy(),
                "budget_cumulative_outflow_momentum": zero_vec.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": float(self._reference_momentum),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if abs(target - self._time) < 1e-12:
            return self._build_frame()
        if target < self._time:
            raise ValueError(f"advance_to target {target} is earlier than current time {self._time}")
        while self._time < target - 1e-12:
            dt = min(self._dt, target - self._time)
            self._step(dt)
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_elastic_bunny",
            "substep_dt": self._dt,
            "edge_count": len(self._edges) if self._edges is not None else 0,
            "face_count": len(self._faces) if self._faces is not None else 0,
            "sample_dt": self._sample_dt,
        }
