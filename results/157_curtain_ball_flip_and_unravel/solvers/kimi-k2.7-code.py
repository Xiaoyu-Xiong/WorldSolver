"""Position-Based Dynamics cloth solver for the curtain-ball flip scene.

This solver advances a pinned rectangular cloth sheet using sub-stepped
position-based dynamics (PBD), with explicit contact handling for a moving
kinematic sphere, the ground, and light vertex-vertex self-contact.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # ---- configuration ----------------------------------------------------
        cloth_cfg = self.config["cloth"]
        self.nx = int(cloth_cfg["grid_shape"][0])
        self.ny = int(cloth_cfg["grid_shape"][1])
        self.sx = float(cloth_cfg["size"][0])
        self.sy = float(cloth_cfg["size"][1])
        self.center = np.asarray(cloth_cfg["center"], dtype=float)
        self.thickness = float(cloth_cfg["thickness"])
        self.total_mass = float(cloth_cfg["total_mass"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])

        # ---- obstacle geometry (fall back to scene_geometry.yaml values) ------
        geometry = self.config.get("scene_geometry", {})
        objects = {obj["id"]: obj for obj in geometry.get("objects", [])}
        sphere_obj = objects.get("sphere", {})
        ground_obj = objects.get("ground", {})

        sphere_geom = sphere_obj.get("geometry", {}) if isinstance(sphere_obj, dict) else {}
        ground_geom = ground_obj.get("geometry", {}) if isinstance(ground_obj, dict) else {}
        sphere_trans = sphere_obj.get("transform", {}) if isinstance(sphere_obj, dict) else {}
        ground_trans = ground_obj.get("transform", {}) if isinstance(ground_obj, dict) else {}

        self.sphere_r = float(sphere_geom.get("radius", 0.62))
        self.sphere_center0 = np.asarray(sphere_trans.get("translation", [0.0, 0.0, 0.68]), dtype=float)
        self.ground_top = 0.0  # top of ground box (z = 0)
        if ground_geom.get("type") == "box" and "half_extents" in ground_geom:
            ground_z = float(ground_trans.get("translation", [0.0, 0.0, -0.05])[2])
            ground_hz = float(ground_geom["half_extents"][2])
            self.ground_top = ground_z + ground_hz

        # ---- time stepping ----------------------------------------------------
        self.h = self.sample_dt / 12.0  # ~0.00278 s, small and stable
        self.pbd_iters = 4
        self.margin = self.thickness * 0.5
        self.sphere_eff_r = self.sphere_r + self.margin
        self.self_cutoff = 2.0 * self.thickness

        # ---- build regular grid in material coordinates -----------------------
        u = np.linspace(-self.sx * 0.5, self.sx * 0.5, self.nx)
        v = np.linspace(-self.sy * 0.5, self.sy * 0.5, self.ny)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        self.du = float(u[1] - u[0])
        self.dv = float(v[1] - v[0])

        # world positions: the sheet lies in the x-z plane with a tiny out-of-plane bump
        x = uu.ravel()
        z = vv.ravel() + self.center[2]
        y = np.zeros_like(x)
        zc = self.sphere_center0[2]
        # small deterministic front-side bulge so the sheet chooses a side when pushed
        s2 = (x / (0.5 * self.sphere_r)) ** 2 + ((z - zc) / (0.5 * self.sphere_r)) ** 2
        y += 0.03 * np.exp(-s2)
        pos = np.column_stack([x, y, z])

        # preserve the top row exactly at the rod height, no out-of-plane bump
        row = np.repeat(np.arange(self.ny), self.nx)
        col = np.tile(np.arange(self.nx), self.ny)
        # Actually with ravel on meshgrid (ny, nx), the ravel order is rows first.
        # We will use row/col arrays below.
        self.row = row
        self.col = col
        self.n = self.nx * self.ny

        self.pinned = row == (self.ny - 1)
        pos[self.pinned, 1] = 0.0
        self.pinned_pos = pos[self.pinned].copy()

        self.pos = pos
        self.vel = np.zeros_like(self.pos)
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)

        # ---- constraint data --------------------------------------------------
        self._build_edges()

        # self-contact exclusion: structural and shear neighbors
        self.adj_pairs = set()
        for i, j in zip(self.edges_i, self.edges_j):
            self.adj_pairs.add((int(i), int(j)))
            self.adj_pairs.add((int(j), int(i)))

        # ---- external force field (gravity) ----------------------------------
        self.ext_force = (self.mass[:, None] * self.gravity).copy()

        # ---- current time -----------------------------------------------------
        self.current_time = 0.0

        # ensure the initial configuration is collision-free in a gentle way
        self._project_sphere(self._sphere_state(0.0)[0])
        self._project_ground()

    def _build_edges(self) -> None:
        i_list, j_list, l_list = [], [], []
        du, dv = self.du, self.dv
        ds = np.sqrt(du * du + dv * dv)

        # structural u
        for r in range(self.ny):
            for c in range(self.nx - 1):
                i = r * self.nx + c
                j = r * self.nx + (c + 1)
                i_list.append(i)
                j_list.append(j)
                l_list.append(du)
        # structural v
        for r in range(self.ny - 1):
            for c in range(self.nx):
                i = r * self.nx + c
                j = (r + 1) * self.nx + c
                i_list.append(i)
                j_list.append(j)
                l_list.append(dv)
        # shear
        for r in range(self.ny - 1):
            for c in range(self.nx - 1):
                i = r * self.nx + c
                j = (r + 1) * self.nx + (c + 1)
                i_list.append(i)
                j_list.append(j)
                l_list.append(ds)
                i = r * self.nx + (c + 1)
                j = (r + 1) * self.nx + c
                i_list.append(i)
                j_list.append(j)
                l_list.append(ds)
        # bending u (two steps along u)
        for r in range(self.ny):
            for c in range(self.nx - 2):
                i = r * self.nx + c
                j = r * self.nx + (c + 2)
                i_list.append(i)
                j_list.append(j)
                l_list.append(2.0 * du)
        # bending v (two steps along v)
        for r in range(self.ny - 2):
            for c in range(self.nx):
                i = r * self.nx + c
                j = (r + 2) * self.nx + c
                i_list.append(i)
                j_list.append(j)
                l_list.append(2.0 * dv)

        self.edges_i = np.asarray(i_list, dtype=np.int64)
        self.edges_j = np.asarray(j_list, dtype=np.int64)
        self.rest_len = np.asarray(l_list, dtype=float)

    def _sphere_state(self, t: float):
        """Kinematic sphere centre and velocity at time t."""
        T = self.duration
        phase = 2.0 * np.pi * t / T
        amp_z = 0.6  # vertical amplitude
        amp_x = 0.2  # small lateral amplitude for a natural 3-D fold
        sin = np.sin(phase)
        z = self.sphere_center0[2] + amp_z * (1.0 - np.cos(phase)) * 0.5
        x = amp_x * (1.0 - np.cos(phase)) * 0.5
        z_dot = amp_z * 0.5 * sin * (2.0 * np.pi / T)
        x_dot = amp_x * 0.5 * sin * (2.0 * np.pi / T)
        center = np.array([x, 0.0, z], dtype=float)
        vel = np.array([x_dot, 0.0, z_dot], dtype=float)
        return center, vel

    def _project_springs(self) -> None:
        # Gauss-Seidel projection of edge length constraints
        for i, j, L in zip(self.edges_i, self.edges_j, self.rest_len):
            p_i = self.pos[i]
            p_j = self.pos[j]
            dx = p_j - p_i
            d = float(np.linalg.norm(dx))
            if d > 1e-12:
                delta = 0.5 * (d - L) / d * dx
                self.pos[i] = p_i + delta
                self.pos[j] = p_j - delta

    def _project_sphere(self, c: np.ndarray) -> None:
        dvec = self.pos - c
        d = np.linalg.norm(dvec, axis=1)
        mask = d < self.sphere_eff_r
        if not np.any(mask):
            return
        d = np.where(mask, d, 1.0)
        n = dvec / d[:, None]
        self.pos[mask] = c + self.sphere_eff_r * n[mask]

    def _project_ground(self) -> None:
        low = self.ground_top + self.margin
        self.pos[:, 2] = np.maximum(self.pos[:, 2], low)

    def _project_self(self) -> None:
        if self.self_cutoff <= 0.0:
            return
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.self_cutoff, output_type="ndarray")
        if pairs.shape[0] == 0:
            return

        # exclude directly connected neighbours (structural + shear)
        keep = []
        for a, b in pairs:
            if (a, b) not in self.adj_pairs and (b, a) not in self.adj_pairs:
                keep.append((a, b))
        if not keep:
            return
        pairs = np.asarray(keep, dtype=np.int64)
        p1 = self.pos[pairs[:, 0]]
        p2 = self.pos[pairs[:, 1]]
        dvec = p2 - p1
        d = np.linalg.norm(dvec, axis=1)
        mask = d > 1e-12
        if not np.any(mask):
            return
        pairs = pairs[mask]
        d = d[mask]
        dvec = dvec[mask]
        n = dvec / d[:, None]
        overlap = self.self_cutoff - d
        pos_mask = overlap > 0.0
        if not np.any(pos_mask):
            return
        delta = 0.5 * overlap[pos_mask][:, None] * n[pos_mask]
        self.pos[pairs[pos_mask, 0]] -= delta
        self.pos[pairs[pos_mask, 1]] += delta

    def _pin(self) -> None:
        self.pos[self.pinned] = self.pinned_pos

    def _step(self, t: float, h: float) -> None:
        # PBD prediction with explicit gravity
        pos_prev = self.pos.copy()
        self.vel += h * self.gravity
        self.pos += h * self.vel

        # contact with moving sphere at end of sub-step
        c, _ = self._sphere_state(t + h)

        for _ in range(self.pbd_iters):
            self._project_springs()
            self._project_sphere(c)
            self._project_ground()
            self._project_self()
            self._pin()

        # velocity from position change
        self.vel = (self.pos - pos_prev) / h
        self.vel[self.pinned] = 0.0

    def _deformation_gradient(self) -> np.ndarray:
        """Per-vertex 2x2 in-plane deformation gradient in material coordinates."""
        F = np.zeros((self.n, 2, 2), dtype=float)
        pos = self.pos
        du, dv = self.du, self.dv

        # Central / one-sided differences for d/du and d/dv
        has_left = self.col > 0
        has_right = self.col < self.nx - 1
        has_up = self.row > 0
        has_down = self.row < self.ny - 1

        idx = np.arange(self.n)

        # du derivative: use central if possible, else one-sided
        du_vec = np.zeros((self.n, 3))
        central = has_left & has_right
        if np.any(central):
            i = idx[central]
            im1 = i - 1
            ip1 = i + 1
            du_vec[central] = (pos[ip1] - pos[im1]) / (2.0 * du)
        # left boundary
        left = ~has_right & has_left
        if np.any(left):
            i = idx[left]
            du_vec[left] = (pos[i] - pos[i - 1]) / du
        # right boundary
        right = ~has_left & has_right
        if np.any(right):
            i = idx[right]
            du_vec[right] = (pos[i + 1] - pos[i]) / du

        # dv derivative: central if possible, else one-sided
        dv_vec = np.zeros((self.n, 3))
        central_v = has_up & has_down
        if np.any(central_v):
            i = idx[central_v]
            im1 = i - self.nx
            ip1 = i + self.nx
            dv_vec[central_v] = (pos[ip1] - pos[im1]) / (2.0 * dv)
        # up boundary
        up = ~has_down & has_up
        if np.any(up):
            i = idx[up]
            dv_vec[up] = (pos[i] - pos[i - self.nx]) / dv
        # down boundary
        down = ~has_up & has_down
        if np.any(down):
            i = idx[down]
            dv_vec[down] = (pos[i + self.nx] - pos[i]) / dv

        # The reference material axes align with world x (u) and world z (v)
        F[:, 0, 0] = du_vec[:, 0] / du
        F[:, 1, 0] = du_vec[:, 2] / du
        F[:, 0, 1] = dv_vec[:, 0] / dv
        F[:, 1, 1] = dv_vec[:, 2] / dv

        # Normalize so that at rest in the flat sheet F is the identity
        F[:, 0, 0] *= du
        F[:, 1, 0] *= du
        F[:, 0, 1] *= dv
        F[:, 1, 1] *= dv
        return F

    def _contact_signed_distances(self, t: float) -> np.ndarray:
        c, _ = self._sphere_state(t)
        d_sphere = np.linalg.norm(self.pos - c, axis=1) - self.sphere_eff_r
        d_ground = self.pos[:, 2] - (self.ground_top + self.margin)

        parts = [d_sphere[d_sphere < 0.05], d_ground[d_ground < 0.05]]

        # self contacts
        if self.self_cutoff > 0.0:
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(self.self_cutoff + 0.05, output_type="ndarray")
            if pairs.shape[0]:
                # filter adjacency
                keep = []
                for a, b in pairs:
                    if (a, b) not in self.adj_pairs and (b, a) not in self.adj_pairs:
                        keep.append((a, b))
                if keep:
                    pairs = np.asarray(keep, dtype=np.int64)
                    dvec = self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]]
                    d = np.linalg.norm(dvec, axis=1)
                    d_self = d - self.self_cutoff
                    parts.append(d_self[d_self < 0.05])

        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    def _build_frame(self, t: float):
        from worldsolver.contracts import StateFrame

        F = self._deformation_gradient()
        sd = self._contact_signed_distances(t)

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
                "boundary_vertex": self.pinned.astype(np.int32),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.ext_force.copy(),
                "contact_signed_distance": sd,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.current_time:
            # sample times are monotonic; allow equality for the initial frame
            target = self.current_time

        while self.current_time < target - 1e-12:
            h = min(self.h, target - self.current_time)
            self._step(self.current_time, h)
            self.current_time += h

        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep": self.h,
            "pbd_iterations": self.pbd_iters,
            "num_vertices": self.n,
            "num_edges": len(self.edges_i),
        }
