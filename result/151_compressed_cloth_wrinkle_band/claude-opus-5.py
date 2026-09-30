"""Compressed-cloth wrinkle-band solver (self-contained, NumPy only).

Physical model
--------------
* Thin sheet discretised on the frozen 17x17 grid with lumped vertex masses.
* Membrane response: near-inextensible position-based distance constraints on
  the structural and diagonal (shear) grid edges, solved with colored
  Gauss-Seidel sweeps.  In-plane compression is resisted, so the excess arc
  length created by the inward grip motion must buckle out of plane.
* Bending response: discrete curvature (second-difference) plate forces along
  both material directions.  The bending / membrane stiffness ratio selects the
  wrinkle wavelength and keeps folds smooth instead of mesh-locked facets.
* Loads and boundaries: gravity from the frozen config, two kinematic grip
  clamps that pin the two opposite cloth edges and translate inward, box
  signed-distance contact against the ground slab and both clamps, and
  vertex-vertex self-repulsion at the cloth thickness scale.
* Dissipation: momentum-conserving Laplacian (relative-velocity) internal
  damping, so the exported external force stays exactly the gravity load.
* Integration: symplectic-Euler prediction + constraint projection with a fixed
  sub-step dt = sample_dt / 12, so every requested sample time is reached with
  an integer number of steps and the trajectory is deterministic.
"""
from __future__ import annotations

import numpy as np


def _smoothstep(u: float) -> float:
    """C1 ramp on [0, 1] used for the prescribed grip motion."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return float(u * u * (3.0 - 2.0 * u))


def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact signed distance of points to an axis-aligned box (negative inside)."""
    d = np.abs(points - center) - half
    dp = np.maximum(d, 0.0)
    outside = np.sqrt(np.einsum("ij,ij->i", dp, dp))
    inside = np.minimum(d.max(axis=1), 0.0)
    return outside + inside


class BenchmarkSolver:
    """Explicitly stepped cloth solver for 151_compressed_cloth_wrinkle_band."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()
        self._frame_cls = type(frame0)

        cloth = self.config["cloth"]
        self.nx = int(cloth["grid_shape"][0])
        self.ny = int(cloth["grid_shape"][1])
        self.thickness = float(cloth.get("thickness", 0.01))
        self.gravity = np.asarray(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.sample_dt = float(self.config.get("sample_dt", 1.0 / 30.0))
        self.substeps = 12
        self.dt = self.sample_dt / self.substeps
        self.iterations = 8

        self.position = np.array(frame0.position, dtype=float)
        self.n = int(len(self.position))
        self.velocity = np.zeros_like(self.position)
        self.mass = np.array(frame0.mass, dtype=float).reshape(self.n)
        self.boundary_vertex = np.asarray(
            frame0.attributes.get("boundary_vertex", np.zeros(self.n)), dtype=np.int32
        )

        size_x = float(cloth["size"][0])
        size_y = float(cloth["size"][1])
        self.hx = size_x / max(self.nx - 1, 1)
        self.hy = size_y / max(self.ny - 1, 1)
        # Slight slack along the gripped direction: the rest length slightly
        # exceeds the initial clamp separation, so the flat sheet starts loose.
        self.slack = 1.008
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        self.gi = ii.ravel().astype(np.int64)
        self.gj = jj.ravel().astype(np.int64)
        self.material = np.column_stack(
            (self.gi * self.hx * self.slack, self.gj * self.hy)
        ).astype(float)

        self.grip_travel = 0.30       # metres of inward travel per clamp
        self.grip_start = 0.6         # settle first, then compress
        self.grip_end = 4.2
        self.bending_stiffness = 4.0e-4   # N*m, thin-cloth plate regime
        self.damping = 0.30              # Laplacian relative-velocity damping
        self.max_speed = 40.0            # safety clamp, keeps states finite

        self._build_obstacles()
        self._build_constraints()
        self._build_bending()
        self._build_triangles()

        rng = np.random.default_rng(int(seed))
        free = self.grip_owner < 0
        # Sub-thickness symmetry breaking; the buckling mode itself is selected
        # by the physics, not by this perturbation.
        self.position[free, 2] += rng.normal(0.0, 3.0e-4, int(free.sum()))
        self.grip_rest = self.position[self.grip_index].copy()

        inv_mass = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)
        inv_mass[self.grip_index] = 0.0
        self.inv_mass = inv_mass

        self.step_count = 0
        self.time = 0.0
        self._peak_speed = 0.0
        self._contact_events = 0
        self._sanitized = 0

    def _vid(self, i: int, j: int) -> int:
        return int(j * self.nx + i)

    # -------------------------------------------------------- frozen geometry
    def _build_obstacles(self) -> None:
        """Read the frozen scene geometry: ground slab and the two grip clamps."""
        geometry = self.config.get("scene_geometry") or {}
        static_boxes: list[tuple[np.ndarray, np.ndarray]] = []
        grip_boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for item in geometry.get("objects", []) or []:
            geo = item.get("geometry", {}) or {}
            if geo.get("type") != "box":
                continue
            if not item.get("collision_enabled", True):
                continue
            half = np.asarray(geo.get("half_extents", [0.1, 0.1, 0.1]), dtype=float)
            transform = item.get("transform", {}) or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            motion = str(item.get("motion", "static"))
            if motion == "kinematic":
                sign = -1.0 if center[0] >= 0.0 else 1.0
                grip_boxes.append((center, half, np.array([sign, 0.0, 0.0])))
            else:
                static_boxes.append((center, half))

        if not grip_boxes:  # frozen-config fallback
            for cx in (-1.5, 1.5):
                grip_boxes.append(
                    (
                        np.array([cx, 0.0, float(self.position[:, 2].mean())]),
                        np.array([0.12, 1.25, 0.12]),
                        np.array([-np.sign(cx), 0.0, 0.0]),
                    )
                )
        if not static_boxes:
            static_boxes.append((np.array([0.0, 0.0, -0.05]), np.array([2.4, 2.0, 0.05])))

        self.static_centers = np.array([c for c, _ in static_boxes], dtype=float)
        self.static_halfs = np.array([h for _, h in static_boxes], dtype=float)
        self.grip_centers0 = np.array([c for c, _, _ in grip_boxes], dtype=float)
        self.grip_halfs = np.array([h for _, h, _ in grip_boxes], dtype=float)
        self.grip_dirs = np.array([d for _, _, d in grip_boxes], dtype=float)

        # Cloth vertices clamped by each grip: those inside the clamp volume.
        owner = np.full(self.n, -1, dtype=np.int64)
        tol = 1.0e-6
        for k in range(len(self.grip_centers0)):
            inside = _box_sdf(self.position, self.grip_centers0[k], self.grip_halfs[k]) <= tol
            owner[inside & (owner < 0)] = k
        self.grip_owner = owner
        self.grip_index = np.nonzero(owner >= 0)[0]
        self.grip_index_owner = owner[self.grip_index]

    def _grip_offset(self, t: float) -> np.ndarray:
        """Prescribed inward displacement of every clamp at time ``t``."""
        u = (float(t) - self.grip_start) / max(self.grip_end - self.grip_start, 1e-9)
        return self.grip_dirs * (self.grip_travel * _smoothstep(u))

    # ------------------------------------------------- membrane / shear graph
    def _build_constraints(self) -> None:
        """Colored (vertex-disjoint) distance-constraint groups on grid edges."""
        nx, ny = self.nx, self.ny
        lx = self.hx * self.slack
        ly = self.hy
        ld = float(np.hypot(lx, ly))
        groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

        for parity in (0, 1):  # edges along x
            a, b = [], []
            for j in range(ny):
                for i in range(parity, nx - 1, 2):
                    a.append(self._vid(i, j))
                    b.append(self._vid(i + 1, j))
            if a:
                groups.append((np.array(a), np.array(b), np.full(len(a), lx), 1.0))

        for parity in (0, 1):  # edges along y
            a, b = [], []
            for j in range(parity, ny - 1, 2):
                for i in range(nx):
                    a.append(self._vid(i, j))
                    b.append(self._vid(i, j + 1))
            if a:
                groups.append((np.array(a), np.array(b), np.full(len(a), ly), 1.0))

        for pj in (0, 1):  # diagonals, 4 colors each family
            for pi in (0, 1):
                a1, b1, a2, b2 = [], [], [], []
                for j in range(pj, ny - 1, 2):
                    for i in range(pi, nx - 1, 2):
                        a1.append(self._vid(i, j))
                        b1.append(self._vid(i + 1, j + 1))
                        a2.append(self._vid(i + 1, j))
                        b2.append(self._vid(i, j + 1))
                if a1:
                    groups.append((np.array(a1), np.array(b1), np.full(len(a1), ld), 0.25))
                    groups.append((np.array(a2), np.array(b2), np.full(len(a2), ld), 0.25))

        self.groups = [
            (a.astype(np.int64), b.astype(np.int64), r.astype(float), float(s))
            for a, b, r, s in groups
        ]

        # structural neighbour edge list for momentum-conserving damping
        ea, eb = [], []
        for j in range(ny):
            for i in range(nx - 1):
                ea.append(self._vid(i, j))
                eb.append(self._vid(i + 1, j))
        for j in range(ny - 1):
            for i in range(nx):
                ea.append(self._vid(i, j))
                eb.append(self._vid(i, j + 1))
        self.edge_a = np.array(ea, dtype=np.int64)
        self.edge_b = np.array(eb, dtype=np.int64)
        degree = np.bincount(
            np.concatenate((self.edge_a, self.edge_b)), minlength=self.n
        ).astype(float)
        self.damp_coeff = float(self.damping / max(degree.max(), 1.0))

    # ----------------------------------------------------------- plate bending
    def _build_bending(self) -> None:
        """Three-vertex curvature stencils along both material directions."""
        nx, ny = self.nx, self.ny
        sx = [[], [], []]
        for j in range(ny):
            for i in range(1, nx - 1):
                sx[0].append(self._vid(i - 1, j))
                sx[1].append(self._vid(i, j))
                sx[2].append(self._vid(i + 1, j))
        sy = [[], [], []]
        for j in range(1, ny - 1):
            for i in range(nx):
                sy[0].append(self._vid(i, j - 1))
                sy[1].append(self._vid(i, j))
                sy[2].append(self._vid(i, j + 1))
        self.bend_x = tuple(np.array(s, dtype=np.int64) for s in sx)
        self.bend_y = tuple(np.array(s, dtype=np.int64) for s in sy)
        # k = B / h^2 from E_bend = 0.5*B*kappa^2 * area with kappa ~ d2x/h^2
        self.kb_x = self.bending_stiffness / (self.hx * self.hx)
        self.kb_y = self.bending_stiffness / (self.hy * self.hy)

    # ------------------------------------------- surface deformation gradient
    def _build_triangles(self) -> None:
        """Grid triangulation plus reference (material) shape-matrix inverses."""
        nx, ny = self.nx, self.ny
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v00 = self._vid(i, j)
                v10 = self._vid(i + 1, j)
                v01 = self._vid(i, j + 1)
                v11 = self._vid(i + 1, j + 1)
                tris.append((v00, v10, v11))
                tris.append((v00, v11, v01))
        self.tris = np.array(tris, dtype=np.int64)
        m = self.material
        e1 = m[self.tris[:, 1]] - m[self.tris[:, 0]]
        e2 = m[self.tris[:, 2]] - m[self.tris[:, 0]]
        dm = np.stack((e1, e2), axis=2)                     # [T,2,2]
        det = np.linalg.det(dm)
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        inv = np.empty_like(dm)
        inv[:, 0, 0] = dm[:, 1, 1] / det
        inv[:, 1, 1] = dm[:, 0, 0] / det
        inv[:, 0, 1] = -dm[:, 0, 1] / det
        inv[:, 1, 0] = -dm[:, 1, 0] / det
        self.dm_inv = inv
        self.tri_area = 0.5 * np.abs(det)
        weight = np.zeros(self.n)
        np.add.at(weight, self.tris.ravel(), np.repeat(self.tri_area, 3))
        self.vertex_tri_weight = np.where(weight > 0.0, weight, 1.0)

        # Self-repulsion only between samples that are far apart in the material
        # domain; immediate topological neighbours are governed by the membrane
        # and bending terms instead.
        di = np.abs(self.gi[:, None] - self.gi[None, :])
        dj = np.abs(self.gj[:, None] - self.gj[None, :])
        self.self_pair_mask = np.maximum(di, dj) > 2

    # ------------------------------------------------------------ internal ops
    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Discrete plate bending force from second differences along x and y."""
        force = np.zeros_like(x)
        for (a, b, c), k in ((self.bend_x, self.kb_x), (self.bend_y, self.kb_y)):
            lap = x[a] - 2.0 * x[b] + x[c]
            np.add.at(force, b, 2.0 * k * lap)
            np.add.at(force, a, -k * lap)
            np.add.at(force, c, -k * lap)
        return force

    def _damp_velocity(self, v: np.ndarray) -> None:
        """Momentum-conserving Laplacian damping of relative edge velocities."""
        dv = v[self.edge_b] - v[self.edge_a]
        acc = np.zeros_like(v)
        np.add.at(acc, self.edge_a, dv)
        np.add.at(acc, self.edge_b, -dv)
        v += self.damp_coeff * acc

    def _project(self, p: np.ndarray) -> None:
        """Colored Gauss-Seidel distance-constraint projection (membrane/shear)."""
        w = self.inv_mass
        for _ in range(self.iterations):
            for a, b, rest, stiff in self.groups:
                d = p[b] - p[a]
                length = np.sqrt(np.einsum("ij,ij->i", d, d))
                safe = np.maximum(length, 1e-12)
                wsum = w[a] + w[b]
                scale = np.zeros(len(a))
                ok = wsum > 0.0
                scale[ok] = stiff * (length[ok] - rest[ok]) / (safe[ok] * wsum[ok])
                corr = d * scale[:, None]
                p[a] += corr * w[a][:, None]
                p[b] -= corr * w[b][:, None]

    def _collide(self, p: np.ndarray, t: float) -> None:
        """Non-penetration against the ground slab and both moving grip clamps."""
        margin = 0.5 * self.thickness
        boxes: list[tuple[np.ndarray, np.ndarray, int]] = [
            (self.static_centers[k], self.static_halfs[k], -1)
            for k in range(len(self.static_centers))
        ]
        offsets = self._grip_offset(t)
        for k in range(len(self.grip_centers0)):
            boxes.append((self.grip_centers0[k] + offsets[k], self.grip_halfs[k], k))
        for center, half, owner in boxes:
            h = half + margin
            d = np.abs(p - center) - h
            inside = np.all(d < 0.0, axis=1)
            if owner >= 0:
                inside &= self.grip_owner != owner
            idx = np.nonzero(inside)[0]
            if idx.size == 0:
                continue
            axis = np.argmax(d[idx], axis=1)          # least-penetration axis
            sign = np.sign(p[idx, axis] - center[axis])
            sign = np.where(sign == 0.0, 1.0, sign)
            p[idx, axis] = center[axis] + sign * h[axis]
            self._contact_events += int(idx.size)

    def _self_collide(self, p: np.ndarray) -> None:
        """Vertex-vertex repulsion at the sheet thickness scale (layer contact)."""
        radius = 2.0 * self.thickness
        sq = np.einsum("ij,ij->i", p, p)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (p @ p.T)
        close = (d2 < radius * radius) & self.self_pair_mask
        ia, ib = np.nonzero(np.triu(close, 1))
        if ia.size == 0:
            return
        d = p[ib] - p[ia]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = np.maximum(dist, 1e-9)
        wa = self.inv_mass[ia]
        wb = self.inv_mass[ib]
        wsum = wa + wb
        scale = np.zeros(ia.size)
        ok = wsum > 0.0
        scale[ok] = (radius - dist[ok]) / (safe[ok] * wsum[ok])
        corr = d * scale[:, None]
        np.add.at(p, ia, -corr * wa[:, None])
        np.add.at(p, ib, corr * wb[:, None])
        self._contact_events += int(ia.size)

    def _substep(self, t0: float) -> None:
        dt = self.dt
        t1 = t0 + dt
        x = self.position
        v = self.velocity

        force = self.mass[:, None] * self.gravity[None, :] + self._bending_force(x)
        v = v + dt * force * self.inv_mass[:, None]
        self._damp_velocity(v)

        offset_now = self._grip_offset(t1)
        offset_prev = self._grip_offset(t0)
        grip_target = self.grip_rest + offset_now[self.grip_index_owner]
        grip_velocity = (
            offset_now[self.grip_index_owner] - offset_prev[self.grip_index_owner]
        ) / dt
        v[self.grip_index] = grip_velocity

        p = x + dt * v
        p[self.grip_index] = grip_target
        self._project(p)
        self._collide(p, t1)
        self._self_collide(p)
        p[self.grip_index] = grip_target

        v_new = (p - x) / dt
        v_new[self.grip_index] = grip_velocity

        bad = ~np.isfinite(p).all(axis=1)
        if bad.any():
            p[bad] = x[bad]
            v_new[bad] = 0.0
            self._sanitized += int(bad.sum())
        bad_v = ~np.isfinite(v_new).all(axis=1)
        if bad_v.any():
            v_new[bad_v] = 0.0
            self._sanitized += int(bad_v.sum())

        speed = np.sqrt(np.einsum("ij,ij->i", v_new, v_new))
        hot = speed > self.max_speed
        if hot.any():
            v_new[hot] *= (self.max_speed / speed[hot])[:, None]
        self._peak_speed = max(self._peak_speed, float(speed.max()))

        self.position = p
        self.velocity = v_new

    # ------------------------------------------------------- derived quantities
    def _deformation_gradient(self) -> np.ndarray:
        """Per-vertex in-plane surface deformation gradient, shape [N,2,2]."""
        x = self.position
        e1 = x[self.tris[:, 1]] - x[self.tris[:, 0]]
        e2 = x[self.tris[:, 2]] - x[self.tris[:, 0]]
        ds = np.stack((e1, e2), axis=2)                      # [T,3,2]
        f3 = ds @ self.dm_inv                                # [T,3,2]
        c = np.einsum("tki,tkj->tij", f3, f3)                # right Cauchy-Green
        c = 0.5 * (c + np.transpose(c, (0, 2, 1)))
        vals, vecs = np.linalg.eigh(c)
        vals = np.clip(vals, 0.0, None)
        root = np.sqrt(vals)
        f2 = np.einsum("tij,tj,tkj->tik", vecs, root, vecs)  # symmetric stretch
        out = np.zeros((self.n, 2, 2))
        weight = np.repeat(self.tri_area, 3)
        contrib = np.repeat(f2, 3, axis=0) * weight[:, None, None]
        np.add.at(out, self.tris.ravel(), contrib)
        out /= self.vertex_tri_weight[:, None, None]
        return np.where(np.isfinite(out), out, 0.0)

    def _contact_distance(self, t: float) -> np.ndarray:
        """Signed gap of every cloth sample to the nearest frozen collider."""
        p = self.position
        gap = np.full(self.n, np.inf)
        for k in range(len(self.static_centers)):
            gap = np.minimum(gap, _box_sdf(p, self.static_centers[k], self.static_halfs[k]))
        offsets = self._grip_offset(t)
        for k in range(len(self.grip_centers0)):
            d = _box_sdf(p, self.grip_centers0[k] + offsets[k], self.grip_halfs[k])
            own = self.grip_owner == k
            d = np.where(own, np.inf, d)
            gap = np.minimum(gap, d)
        gap = gap - 0.5 * self.thickness
        gap[self.grip_index] = 0.0     # clamped samples are in exact contact
        return np.where(np.isfinite(gap), gap, 1.0e3)

    def _frame(self, t: float):
        position = np.array(self.position, dtype=float)
        velocity = np.array(self.velocity, dtype=float)
        position = np.where(np.isfinite(position), position, 0.0)
        velocity = np.where(np.isfinite(velocity), velocity, 0.0)
        external_force = self.mass[:, None] * self.gravity[None, :]
        frame = self._frame_cls(
            float(t),
            position,
            velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "external_force": external_force,
                "contact_signed_distance": self._contact_distance(t),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = float(time)
        target_steps = int(round(target / self.dt))
        while self.step_count < target_steps:
            self._substep(self.step_count * self.dt)
            self.step_count += 1
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position-based cloth: inextensible membrane + shear "
            "constraints, discrete plate bending, box SDF contact, "
            "self-repulsion, symplectic-Euler substepping",
            "substep_dt": float(self.dt),
            "substeps_taken": int(self.step_count),
            "solver_iterations": int(self.iterations),
            "vertices": int(self.n),
            "distance_constraint_groups": int(len(self.groups)),
            "grip_travel_per_side_m": float(self.grip_travel),
            "compression_fraction": float(
                2.0 * self.grip_travel / max(float(self.config["cloth"]["size"][0]), 1e-9)
            ),
            "bending_stiffness_N_m": float(self.bending_stiffness),
            "peak_speed_m_s": float(self._peak_speed),
            "contact_corrections": int(self._contact_events),
            "sanitized_samples": int(self._sanitized),
            "final_time_s": float(self.time),
        }
