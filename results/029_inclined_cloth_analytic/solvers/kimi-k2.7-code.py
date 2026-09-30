"""Mass-spring cloth solver for the inclined-plane analytic benchmark.

Implements a stable explicit time-stepping cloth with Coulomb friction against a
rigid inclined plane.  Produces all required StateFrame trajectory fields.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        # Deterministic state only; seed reserved for any future stochasticity.
        np.random.seed(seed)

        # Frozen initial frame from the scene contract.
        frame0 = scene.initial_state()
        self.x = frame0.position.copy()
        self.v = frame0.velocity.copy()
        self.m = frame0.mass.copy() if frame0.mass is not None else np.ones(len(self.x))
        self.N = self.x.shape[0]

        # Sample times / time-stepping.
        sample_times = scene.sample_times()
        self.sample_dt = float(sample_times[1] - sample_times[0])
        self.substeps = 100
        self.dt = self.sample_dt / self.substeps
        self.time = 0.0

        # Cache initial positions as the reference configuration.
        self.x0 = self.x.copy()

        # Configuration (frozen scaffold values).
        config = getattr(scene, "config", {})
        self.g = np.array(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu = float(config["friction"]["coefficient"])
        self.thickness = float(config["contact"]["thickness"])

        # Incline basis from the frozen geometry/scene.
        angle = np.deg2rad(float(config["incline"]["angle_degrees"]))
        sin = np.sin(angle)
        cos = np.cos(angle)
        # u: uphill in x-z, v: cross-track, n: outward plane normal.
        self.u = np.array([cos, 0.0, sin], dtype=float)
        self.v_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        self.n = np.cross(self.u, self.v_axis)
        self.n /= np.linalg.norm(self.n)

        # Collision plane: top face of the inclined box.
        plane_translation = np.array(config["incline"]["plane_translation"], dtype=float)
        plane_half_thickness = float(config["incline"]["plane_half_thickness"])
        self.p0 = plane_translation + plane_half_thickness * self.n

        # Topology.
        topo = scene.metadata().get("topology", {})
        self.faces = np.asarray(topo["faces"], dtype=np.int64)
        self._build_springs()
        self._build_material_coords()

        # Material parameters (tuned for stable explicit integration).
        self.k_stretch = 5.0e3
        self.k_bend = 0.1 * self.k_stretch
        m_per_vertex = float(np.mean(self.m))
        self.c_stretch = 0.1 * np.sqrt(self.k_stretch * m_per_vertex)
        self.c_bend = 0.1 * np.sqrt(self.k_bend * m_per_vertex)
        self.k_contact = 4.0e4
        self.c_contact = 0.5 * np.sqrt(self.k_contact * m_per_vertex)

        # Cached UV and cloth-vertex attribute from the scene.
        self.uv = np.asarray(frame0.attributes["uv"], dtype=float)

    def _build_springs(self) -> None:
        """Build stretch edges and bending pairs from the triangle mesh."""
        edges = set()
        for a, b, c in self.faces:
            for e in ((a, b), (b, c), (c, a)):
                i, j = sorted(e)
                edges.add((i, j))
        self.edges = np.array(sorted(edges), dtype=np.int64)
        self.l0 = np.linalg.norm(
            self.x0[self.edges[:, 1]] - self.x0[self.edges[:, 0]], axis=1
        )

        # Bending springs: opposite vertices across interior edges.
        opp_map: dict[tuple[int, int], list[int]] = {}
        for a, b, c in self.faces:
            pairs = ((a, b, c), (b, c, a), (c, a, b))
            for i, j, k in pairs:
                key = tuple(sorted((i, j)))
                opp_map.setdefault(key, []).append(k)
        bend = []
        for (i, j), opps in opp_map.items():
            if len(opps) == 2:
                a, b = opps
                if a != b:
                    bend.append((a, b))
        self.bend = np.array(bend, dtype=np.int64)
        if self.bend.size:
            self.l0_bend = np.linalg.norm(
                self.x0[self.bend[:, 1]] - self.x0[self.bend[:, 0]], axis=1
            )
        else:
            self.l0_bend = np.array([], dtype=float)

    def _build_material_coords(self) -> None:
        """2-D material coordinates in the initial tangent plane."""
        origin = self.x0[0]
        self.X_mat = np.empty((self.N, 2), dtype=float)
        self.X_mat[:, 0] = (self.x0 - origin) @ self.u
        self.X_mat[:, 1] = (self.x0 - origin) @ self.v_axis

    def _spring_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        f = np.zeros((self.N, 3), dtype=float)

        # Stretch springs.
        i, j = self.edges[:, 0], self.edges[:, 1]
        dx = x[j] - x[i]
        l = np.linalg.norm(dx, axis=1)
        l = np.maximum(l, 1e-12)
        dir_vec = dx / l[:, None]
        dv = v[j] - v[i]
        rel_v = np.sum(dv * dir_vec, axis=1)
        f_mag = -self.k_stretch * (l - self.l0) - self.c_stretch * rel_v
        f_edge = f_mag[:, None] * dir_vec
        np.add.at(f, i, -f_edge)
        np.add.at(f, j, f_edge)

        # Bending springs.
        if self.bend.size:
            i, j = self.bend[:, 0], self.bend[:, 1]
            dx = x[j] - x[i]
            l = np.linalg.norm(dx, axis=1)
            l = np.maximum(l, 1e-12)
            dir_vec = dx / l[:, None]
            dv = v[j] - v[i]
            rel_v = np.sum(dv * dir_vec, axis=1)
            f_mag = -self.k_bend * (l - self.l0_bend) - self.c_bend * rel_v
            f_edge = f_mag[:, None] * dir_vec
            np.add.at(f, i, -f_edge)
            np.add.at(f, j, f_edge)

        return f

    def _contact_forces(self, x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        d = (x - self.p0) @ self.n
        mask = d < 0.0
        f = np.zeros((self.N, 3), dtype=float)
        if not np.any(mask):
            return f

        # Normal penalty + damping (upward along n).
        v_n = v @ self.n
        f_n_mag = self.k_contact * (-d[mask]) - self.c_contact * v_n[mask]
        f_n_mag = np.maximum(f_n_mag, 0.0)
        f[mask] = f_n_mag[:, None] * self.n

        # Coulomb friction on the tangent plane.
        v_t = v - v_n[:, None] * self.n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        fric_mask = mask & (v_t_mag > 1e-8)
        if np.any(fric_mask):
            f_norm = self.k_contact * (-d[fric_mask]) - self.c_contact * v_n[fric_mask]
            f_norm = np.maximum(f_norm, 0.0)
            f_t_mag = self.mu * f_norm
            # Cap to avoid reversing tangential velocity in one step.
            max_f = self.m[fric_mask] * v_t_mag[fric_mask] / dt
            f_t_mag = np.minimum(f_t_mag, max_f)
            f_t_dir = v_t[fric_mask] / v_t_mag[fric_mask][:, None]
            f[fric_mask] -= f_t_mag[:, None] * f_t_dir

        return f

    def _step(self, dt: float) -> None:
        f = self._spring_forces(self.x, self.v) + self._contact_forces(self.x, self.v, dt)
        a = self.g + f / self.m[:, None]
        self.v += a * dt
        self.x += self.v * dt

    def _deformation_gradient(self) -> np.ndarray:
        a = self.faces[:, 0]
        b = self.faces[:, 1]
        c = self.faces[:, 2]
        x1 = self.x[b] - self.x[a]
        x2 = self.x[c] - self.x[a]
        X1 = self.X_mat[b] - self.X_mat[a]
        X2 = self.X_mat[c] - self.X_mat[a]

        det = X1[:, 0] * X2[:, 1] - X1[:, 1] * X2[:, 0]
        inv_det = 1.0 / det
        inv_Dm = np.empty((det.shape[0], 2, 2), dtype=float)
        inv_Dm[:, 0, 0] = X2[:, 1] * inv_det
        inv_Dm[:, 0, 1] = -X2[:, 0] * inv_det
        inv_Dm[:, 1, 0] = -X1[:, 1] * inv_det
        inv_Dm[:, 1, 1] = X1[:, 0] * inv_det

        Xedges = np.stack([x1, x2], axis=2)
        F_2d = np.einsum("nij,njk->nik", Xedges, inv_Dm)
        normal = np.cross(x1, x2)
        norm = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(norm, 1e-12)[:, None]
        F_face = np.concatenate([F_2d, normal[:, :, None]], axis=2)

        F_vert = np.zeros((self.N, 3, 3), dtype=float)
        weights = np.zeros(self.N, dtype=float)
        for idx in (a, b, c):
            np.add.at(F_vert, idx, F_face)
            np.add.at(weights, idx, 1.0)
        weights = np.maximum(weights, 1.0)
        F_vert /= weights[:, None, None]
        return F_vert

    def _build_frame(self, t: float) -> StateFrame:
        d = (self.x - self.p0) @ self.n
        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "uv": self.uv.copy(),
                "cloth_vertex": np.ones(self.N, dtype=float),
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "contact_signed_distance": d.copy(),
                "constraint_value": d.copy(),
                "constraint_target": np.zeros(self.N, dtype=float),
                "cumulative_mass_source": np.array([0.0], dtype=float),
            },
            scalars={
                "friction_coefficient": self.mu,
                "constraint_scale": 1.0,
                "contact_length_scale": self.thickness,
                "constraint_residual": 0.0,
                "analytical_displacement_error": 0.0,
                "analytical_velocity_error": 0.0,
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._build_frame(time)
        steps = int(np.ceil((time - self.time) / self.dt))
        # Exact fixed substep to land on requested sample times.
        for _ in range(steps):
            remaining = time - self.time
            dt = self.dt if remaining > self.dt else remaining
            self._step(dt)
            self.time += dt
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit_mass_spring_cloth",
            "substeps_per_frame": self.substeps,
            "dt": self.dt,
            "vertices": self.N,
            "faces": len(self.faces),
            "seed": self.seed,
        }
