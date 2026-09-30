"""Cornstarch impact solver.

A lightweight FEM-style solver for a viscoplastic cornstarch blob.
It uses a particle representation with edge springs for elasticity/plasticity,
a simple penalty contact model with restitution/friction, and explicit
time stepping.  The implementation only depends on NumPy/SciPy; it does not
use Genesis or any external simulation packages.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic cornstarch-impact solver.

    The runner calls:
        solver.initialize(scene, seed)
        frames = [solver.advance_to(t) for t in scene.sample_times()]
    """

    def __init__(self):
        self.scene = None
        self.cfg = None
        self.time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        np.random.seed(seed)

        # Physics parameters
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.contact_length_scale = float(self.cfg["contact_length_scale"])
        self.subject_mass = float(self.cfg["subject_mass"])
        self.stiffness = float(self.cfg["stiffness"])
        self.damping = float(self.cfg["damping"])
        self.restitution = float(self.cfg["restitution"])
        self.friction = float(self.cfg["friction"])
        self.yield_time = float(self.cfg["yield_time"])

        # Initial state from the scene contract
        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float, copy=True)
        self.v = np.array(frame0.velocity, dtype=float, copy=True)
        self.m = np.array(frame0.mass, dtype=float, copy=True)
        self.body_id = np.array(frame0.body_id, dtype=np.int64, copy=True)
        self.n = self.x.shape[0]

        # External force (gravity only, excluding contact)
        self.external_force = self.m[:, None] * self.gravity[None, :]

        # Time-stepping parameters
        self.sub_dt = self.contact_length_scale * 0.05  # ~2e-3 for this scene
        if self.sub_dt <= 0.0:
            self.sub_dt = 0.001

        # Contact model parameters
        m_avg = float(np.mean(self.m)) if self.n > 0 else 1.0
        self.contact_k = 1000.0  # penalty stiffness (N/m)
        self.contact_c = 0.5 * np.sqrt(self.contact_k * m_avg)  # under-critical
        self.viscosity = 0.5  # mass-proportional damping after yield (1/s)
        self.plastic_rate = 2.0  # plastic rest-length adaptation rate (1/s)
        self.fluid_stiffness_factor = 0.05  # stiffness multiplier after yield

        # Build edge connectivity from the subject mesh asset
        self._build_connectivity()
        self._init_deformation_gradient_data()

        # Rest lengths for springs
        self.rest_len = np.linalg.norm(
            self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]], axis=1
        )

        self.time = 0.0

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.time
        if dt > 0.0:
            n_sub = max(1, int(np.ceil(dt / self.sub_dt)))
            h = dt / n_sub
            for _ in range(n_sub):
                self._substep(h)
            self.time = time
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": int(self.n),
            "num_edges": int(self.edges.shape[0]) if self.edges is not None else 0,
            "time": float(self.time),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_connectivity(self) -> None:
        """Create a spring network for the deformable body."""
        edges: set[tuple[int, int]] = set()

        # Try to read the asset for explicit topology
        asset_path = Path(__file__).resolve().parent.parent / self.cfg["subject_asset"]
        try:
            data = np.load(asset_path)
            if "tets" in data:
                faces = np.asarray(data["tets"], dtype=int)
                for f in faces:
                    for a, b in ((0, 1), (1, 2), (2, 0)):
                        i, j = int(f[a]), int(f[b])
                        if i == j:
                            continue
                        edges.add((min(i, j), max(i, j)))
            elif "faces" in data:
                faces = np.asarray(data["faces"], dtype=int)
                for f in faces:
                    for a, b in ((0, 1), (1, 2), (2, 0)):
                        i, j = int(f[a]), int(f[b])
                        if i == j:
                            continue
                        edges.add((min(i, j), max(i, j)))
        except Exception:
            pass

        # Radius-based neighbors guarantee a connected, volumetric network
        max_neigh = 12
        radius = 3.0 * self.contact_length_scale
        tree = cKDTree(self.x)
        k = min(self.n, max_neigh + 1)
        dist, idx = tree.query(self.x, k=k)
        for i in range(self.n):
            for dval, j in zip(dist[i], idx[i]):
                if j <= i:
                    continue
                if dval > radius:
                    continue
                edges.add((i, int(j)))

        if not edges:
            # Degenerate case: create a minimal empty edge set so the code runs.
            self.edges = np.zeros((0, 2), dtype=int)
        else:
            self.edges = np.asarray(sorted(edges), dtype=int)

    def _init_deformation_gradient_data(self) -> None:
        """Per-vertex neighbor lists and rest offsets for F computation."""
        self.neighbor_lists: list[np.ndarray] = [np.array([], dtype=int) for _ in range(self.n)]
        self.rest_offsets: list[np.ndarray] = [np.zeros((0, 3), dtype=float) for _ in range(self.n)]

        if self.edges.size == 0:
            return

        adjacency: list[list[int]] = [[] for _ in range(self.n)]
        for i, j in self.edges:
            adjacency[i].append(j)
            adjacency[j].append(i)

        for i in range(self.n):
            nbrs = np.asarray(adjacency[i], dtype=int)
            self.neighbor_lists[i] = nbrs
            if nbrs.size:
                self.rest_offsets[i] = self.x[nbrs] - self.x[i]

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Return per-vertex deformation gradient relative to the rest shape."""
        F = np.tile(np.eye(3, dtype=float)[None, ...], (self.n, 1, 1))
        for i in range(self.n):
            nbrs = self.neighbor_lists[i]
            if nbrs.size < 3:
                continue
            dX = self.rest_offsets[i]  # Kx3
            dx = self.x[nbrs] - self.x[i]  # Kx3
            # F * dX^T = dx^T  =>  F = dx^T * pinv(dX^T)
            F[i] = dx.T @ np.linalg.pinv(dX.T)
        return F

    def _substep(self, h: float) -> None:
        n = self.n
        x = self.x
        v = self.v
        m = self.m

        # Gravity
        f = self.external_force.copy()

        # Material state: solid before yield, soft/fluid after
        if self.time < self.yield_time:
            k = self.stiffness
            yielded = False
        else:
            k = self.stiffness * self.fluid_stiffness_factor
            yielded = True

        # Internal spring and damping forces
        if self.edges.shape[0] > 0:
            i0 = self.edges[:, 0]
            i1 = self.edges[:, 1]
            dx = x[i0] - x[i1]
            l = np.linalg.norm(dx, axis=1)
            l = np.where(l < 1e-12, 1.0, l)
            e_dir = dx / l[:, None]

            # Elastic force
            dl = l - self.rest_len
            fmag = -k * dl
            fvec = fmag[:, None] * e_dir
            np.add.at(f, i0, fvec)
            np.add.at(f, i1, -fvec)

            # Edge damping
            dv = v[i0] - v[i1]
            vrel = np.sum(dv * e_dir, axis=1)
            m_eff = 0.5 * (m[i0] + m[i1])
            c = self.damping * np.sqrt(self.stiffness * m_eff)
            fd = -(c * vrel)[:, None] * e_dir
            np.add.at(f, i0, fd)
            np.add.at(f, i1, -fd)

        # Ground contact penalty
        pen = self.ground_z - x[:, 2]
        contact = pen > 0.0
        if np.any(contact):
            # normal spring + damping
            f[contact, 2] += self.contact_k * pen[contact] - self.contact_c * v[contact, 2]

        # Global (mass-proportional) damping after yielding
        if yielded:
            f -= self.viscosity * m[:, None] * v

        # Semi-implicit Euler: update velocity, then position
        v_old = v.copy()
        v += h * f / m[:, None]

        # Contact impulse: restitution and friction
        if np.any(contact):
            # Normal restitution
            vz = v[contact, 2]
            moving_down = vz < 0.0
            vz[moving_down] = -self.restitution * vz[moving_down]
            v[contact, 2] = vz

            # Friction: limit tangential velocity using the normal impulse
            delta_vn = np.maximum(0.0, v[contact, 2] - v_old[contact, 2])
            m_contact = m[contact]
            max_tangential_impulse = self.friction * (m_contact * delta_vn)

            vxy = v[contact, :2]
            speed = np.linalg.norm(vxy, axis=1)
            reduce = np.zeros_like(speed)
            nonzero = speed > 1e-12
            reduce[nonzero] = np.maximum(
                0.0, 1.0 - max_tangential_impulse[nonzero] / (m_contact[nonzero] * speed[nonzero])
            )
            v[contact, :2] = (vxy.T * reduce).T

        # Position update
        x += h * v

        # Plastic flow: rest lengths adapt to current lengths after yield
        if yielded and self.edges.shape[0] > 0:
            dx = x[self.edges[:, 0]] - x[self.edges[:, 1]]
            l_now = np.linalg.norm(dx, axis=1)
            self.rest_len += h * self.plastic_rate * (l_now - self.rest_len)

        self.time += h

    def _make_frame(self) -> StateFrame:
        F = self._compute_deformation_gradient()
        contact_sd = self.x[:, 2] - self.ground_z
        frame = StateFrame(
            time=float(self.time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.external_force.copy(),
                "contact_signed_distance": contact_sd,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame
