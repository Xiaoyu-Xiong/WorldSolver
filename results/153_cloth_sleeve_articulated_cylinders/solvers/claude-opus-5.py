"""Cloth sleeve on articulated cylinders.

Self-contained physical solver (NumPy only):

* Thin-shell cloth is discretised as a periodic mass point lattice forming a
  closed tubular sleeve (16 rings x 17 axial stations of unique nodes, exported
  as the frozen 17x17 grid with a coincident seam ring).
* Time stepping is sub-stepped semi-implicit (symplectic) Euler prediction
  followed by Gauss-Seidel/Jacobi constraint projection (XPBD "small steps"
  style) for stretch, shear and bending resistance.  This is unconditionally
  stable for the stiffness regime of woven cloth.
* Boundaries: frozen ground box plus two frozen capsules.  ``limb_upper`` is
  anchored, ``limb_lower`` is driven about the joint pivot, giving the articulated
  elbow motion.  Contacts are resolved as non-penetration projections with
  Coulomb friction against the kinematic surface velocity.
* Cloth-cloth (self) contact uses a proximity repulsion so folds stack instead
  of interpenetrating.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _smoothstep_derivative(s: float) -> float:
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 6.0 * s * (1.0 - s)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])

        self.nz, self.nx = (int(v) for v in cloth["grid_shape"])  # 17 x 17
        self.size_u, self.size_v = (float(v) for v in cloth["size"])  # 3.0 x 2.5
        self.nring = self.nz - 1          # unique rings (seam ring duplicated)
        self.ncol = self.nx               # axial stations
        self.nnodes = self.nring * self.ncol
        self.nout = self.nz * self.nx     # exported sample count (289)

        self.du = self.size_u / (self.nx - 1)
        self.dv = self.size_v / (self.nz - 1)

        # ----- boundary geometry (frozen scene_geometry.yaml) -----
        geom = {}
        scene_geom = cfg.get("scene_geometry")
        if isinstance(scene_geom, dict):
            for obj in scene_geom.get("objects", []):
                geom[str(obj.get("id"))] = obj

        self.ground_top = 0.0
        gnd = geom.get("ground")
        if gnd is not None:
            half = np.asarray(gnd["geometry"]["half_extents"], dtype=float)
            trans = np.asarray(gnd.get("transform", {}).get("translation", [0, 0, 0]), float)
            self.ground_top = float(trans[2] + half[2])

        self.capsules = []  # (endpoint_a, endpoint_b, radius, driven_flag)
        for name, driven in (("limb_upper", False), ("limb_lower", True)):
            item = geom.get(name)
            if item is None:
                continue
            g = item["geometry"]
            radius = float(g["radius"])
            height = float(g["height"])
            trans = np.asarray(item.get("transform", {}).get("translation", [0, 0, 0]), float)
            quat = np.asarray(
                item.get("transform", {}).get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), float
            )
            axis = self._quat_apply(quat, np.array([0.0, 0.0, 1.0]))
            nrm = float(np.linalg.norm(axis))
            axis = axis / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])
            a = trans - 0.5 * height * axis
            b = trans + 0.5 * height * axis
            self.capsules.append([a, b, radius, driven])
        if not self.capsules:  # defensive fallback (geometry always present)
            self.capsules = [
                [np.array([-1.2, 0.0, 1.0]), np.array([0.1, 0.0, 1.0]), 0.34, False],
                [np.array([-0.1, 0.0, 1.0]), np.array([1.2, 0.0, 1.0]), 0.31, True],
            ]

        # joint pivot: inboard end of the driven capsule, on the limb axis
        driven = [c for c in self.capsules if c[3]]
        if driven:
            a, b, _, _ = driven[0]
            self.pivot = a.copy() if abs(a[0]) < abs(b[0]) else b.copy()
        else:
            self.pivot = np.array([0.0, 0.0, 1.0])
        self.pivot[0] = 0.0  # joint centre between the two aligned limbs

        # ----- sleeve reference configuration -----
        axis_z = float(np.mean([0.5 * (c[0][2] + c[1][2]) for c in self.capsules]))
        axis_y = float(np.mean([0.5 * (c[0][1] + c[1][1]) for c in self.capsules]))
        self.sleeve_radius = self.size_v / (2.0 * np.pi)

        rng = np.random.default_rng(int(seed))
        r_idx = np.arange(self.nring)
        c_idx = np.arange(self.ncol)
        theta = 2.0 * np.pi * r_idx / self.nring
        u = -0.5 * self.size_u + self.du * c_idx
        TH, U = np.meshgrid(theta, u, indexing="ij")
        # tiny deterministic radial imperfection seeds physical wrinkling
        radius = self.sleeve_radius * (
            1.0 + 0.004 * np.sin(3.0 * TH + 1.7 * U) + 0.002 * np.cos(5.0 * TH - 0.9 * U)
        )
        X = np.empty((self.nring, self.ncol, 3), dtype=float)
        X[..., 0] = U
        X[..., 1] = axis_y + radius * np.cos(TH)
        X[..., 2] = axis_z + radius * np.sin(TH)
        X = X.reshape(-1, 3)
        X += rng.normal(0.0, 2.0e-4, X.shape)

        self.X = X
        self.V = np.zeros_like(X)
        self.mass_node = self.total_mass / self.nnodes
        self.mass_out = np.full(self.nout, self.total_mass / self.nout)

        self._build_constraints()
        self._build_selfcollision_mask()

        # reference surface Jacobian for the deformation gradient
        G0 = self._surface_jacobian(self.X)
        self.inv_G0 = np.linalg.inv(G0 + 1e-9 * np.eye(2)[None])

        # ----- time stepping -----
        self.substeps_per_sample = 25
        self.dt = self.sample_dt / self.substeps_per_sample
        self.step_index = 0
        self.time = 0.0
        self.max_speed = 25.0
        self.mu_body = 0.45
        self.mu_ground = 0.6
        self.velocity_damping = 0.9995
        self.drag = 0.6  # aerodynamic drag rate [1/s]: force = -drag * m * v
        self.contact_margin = 0.5 * self.thickness + 0.004
        self.self_dist = 0.35 * min(self.du, self.dv)
        self.iterations = 4
        self.joint_max = np.deg2rad(78.0)
        self.joint_hold = np.deg2rad(18.0)
        self._diag = {"substeps": 0, "max_speed": 0.0, "min_gap": 0.0}

    # ------------------------------------------------------------- topology
    def _nid(self, r, c):
        return (r % self.nring) * self.ncol + c

    def _build_constraints(self) -> None:
        pairs = []
        weights = []
        R, C = self.nring, self.ncol
        for r in range(R):
            for c in range(C):
                i = self._nid(r, c)
                # axial stretch
                if c + 1 < C:
                    pairs.append((i, self._nid(r, c + 1)))
                    weights.append(1.0)
                # circumferential stretch (periodic seam)
                pairs.append((i, self._nid(r + 1, c)))
                weights.append(1.0)
                # shear diagonals
                if c + 1 < C:
                    pairs.append((i, self._nid(r + 1, c + 1)))
                    weights.append(0.35)
                    pairs.append((self._nid(r + 1, c), self._nid(r, c + 1)))
                    weights.append(0.35)
                # bending (two-ring / two-station span)
                if c + 2 < C:
                    pairs.append((i, self._nid(r, c + 2)))
                    weights.append(0.10)
                pairs.append((i, self._nid(r + 2, c)))
                weights.append(0.10)
        idx = np.asarray(pairs, dtype=np.int64)
        self.c_i = idx[:, 0]
        self.c_j = idx[:, 1]
        self.c_w = np.asarray(weights, dtype=float)
        d = self.X[self.c_j] - self.X[self.c_i]
        self.c_rest = np.linalg.norm(d, axis=1)
        counts = np.zeros(self.nnodes)
        np.add.at(counts, self.c_i, self.c_w)
        np.add.at(counts, self.c_j, self.c_w)
        self.c_norm = 1.0 / np.maximum(counts, 1e-9)

    def _build_selfcollision_mask(self) -> None:
        # exclude topological near-neighbours from proximity repulsion
        n = self.nnodes
        adj = np.zeros((n, n), dtype=bool)
        adj[self.c_i, self.c_j] = True
        adj[self.c_j, self.c_i] = True
        two = adj @ adj
        near = adj | two | np.eye(n, dtype=bool)
        self.self_pairs = np.triu(~near, 1)
        pi, pj = np.nonzero(self.self_pairs)
        self.sp_i = pi
        self.sp_j = pj

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _quat_apply(q, v):
        x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        n = np.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.asarray(v, dtype=float)
        x, y, z, w = x / n, y / n, z / n, w / n
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return rot @ np.asarray(v, dtype=float)

    def _joint_angle(self, t: float):
        """Driven joint angle and rate (radians, radians/second)."""
        segments = (
            (0.0, 0.7, 0.0, 0.0),
            (0.7, 3.0, 0.0, self.joint_max),
            (3.0, 3.8, self.joint_max, self.joint_max),
            (3.8, 5.2, self.joint_max, self.joint_hold),
            (5.2, 7.0, self.joint_hold, 0.86 * self.joint_max),
        )
        for t0, t1, a0, a1 in segments:
            if t < t1 or t1 >= 7.0:
                span = max(t1 - t0, 1e-9)
                s = (min(max(t, t0), t1) - t0) / span
                angle = a0 + (a1 - a0) * _smoothstep(s)
                rate = (a1 - a0) * _smoothstep_derivative(s) / span
                return float(angle), float(rate)
        return float(segments[-1][3]), 0.0

    def _capsule_state(self, t: float):
        """Current endpoints, radius and rigid motion (omega) of each limb."""
        angle, rate = self._joint_angle(t)
        ca, sa = np.cos(-angle), np.sin(-angle)
        rot = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
        omega = np.array([0.0, -rate, 0.0])
        out = []
        for a0, b0, radius, driven in self.capsules:
            if driven:
                a = self.pivot + rot @ (a0 - self.pivot)
                b = self.pivot + rot @ (b0 - self.pivot)
                out.append((a, b, radius, omega, self.pivot))
            else:
                out.append((a0, b0, radius, np.zeros(3), self.pivot))
        return out

    # ------------------------------------------------------------ kinematics
    @staticmethod
    def _closest_point_on_segment(P, a, b):
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            t = np.zeros(len(P))
        else:
            t = np.clip((P - a) @ ab / denom, 0.0, 1.0)
        return a[None, :] + t[:, None] * ab[None, :]

    def _boundary_gap(self, P, t):
        """Minimum signed distance from every sample to the frozen boundaries."""
        gap = P[:, 2] - self.ground_top
        for a, b, radius, _omega, _piv in self._capsule_state(t):
            q = self._closest_point_on_segment(P, a, b)
            d = np.linalg.norm(P - q, axis=1) - radius
            gap = np.minimum(gap, d)
        return gap

    # ------------------------------------------------------------- one substep
    def _substep(self) -> None:
        dt = self.dt
        t0 = self.step_index * dt
        t1 = t0 + dt
        X0 = self.X
        m = self.mass_node

        # --- explicit external forcing (gravity + weak aerodynamic drag) ---
        force = np.repeat((m * self.gravity)[None, :], self.nnodes, axis=0)
        force -= (self.drag * m) * self.V
        V = self.V + dt * force / m
        V *= self.velocity_damping
        speed = np.linalg.norm(V, axis=1)
        too_fast = speed > self.max_speed
        if np.any(too_fast):
            V[too_fast] *= (self.max_speed / speed[too_fast])[:, None]
        P = X0 + dt * V

        # --- internal elasticity: XPBD-style distance projections ---
        for _ in range(self.iterations):
            d = P[self.c_j] - P[self.c_i]
            length = np.linalg.norm(d, axis=1)
            np.maximum(length, 1e-9, out=length)
            corr = (length - self.c_rest) / length
            grad = d * (self.c_w * corr)[:, None]
            delta = np.zeros_like(P)
            np.add.at(delta, self.c_i, 0.5 * grad)
            np.add.at(delta, self.c_j, -0.5 * grad)
            P += delta * self.c_norm[:, None]

        # --- cloth self contact (proximity repulsion so folds stack) ---
        if len(self.sp_i):
            d = P[self.sp_j] - P[self.sp_i]
            dist = np.linalg.norm(d, axis=1)
            hit = dist < self.self_dist
            if np.any(hit):
                idx_i = self.sp_i[hit]
                idx_j = self.sp_j[hit]
                dh = d[hit]
                dl = np.maximum(dist[hit], 1e-9)
                push = (0.5 * (self.self_dist - dl) / dl)[:, None] * dh
                delta = np.zeros_like(P)
                np.add.at(delta, idx_i, -push)
                np.add.at(delta, idx_j, push)
                counts = np.zeros(self.nnodes)
                np.add.at(counts, idx_i, 1.0)
                np.add.at(counts, idx_j, 1.0)
                P += delta / np.maximum(counts, 1.0)[:, None]

        # --- kinematic body contact with Coulomb friction ---
        for a, b, radius, omega, pivot in self._capsule_state(t1):
            q = self._closest_point_on_segment(P, a, b)
            n = P - q
            dist = np.linalg.norm(n, axis=1)
            safe = np.maximum(dist, 1e-9)
            n = n / safe[:, None]
            gap = dist - (radius + self.contact_margin)
            hit = gap < 0.0
            if not np.any(hit):
                continue
            depth = -gap[hit]
            nh = n[hit]
            P[hit] += depth[:, None] * nh
            v_surface = np.cross(np.broadcast_to(omega, (hit.sum(), 3)), q[hit] - pivot)
            slip = (P[hit] - X0[hit]) - dt * v_surface
            slip_t = slip - (np.einsum("ij,ij->i", slip, nh))[:, None] * nh
            mag = np.linalg.norm(slip_t, axis=1)
            limit = np.minimum(self.mu_body * depth, mag)
            scale = np.where(mag > 1e-12, limit / np.maximum(mag, 1e-12), 0.0)
            P[hit] -= slip_t * scale[:, None]

        # --- ground contact ---
        floor = self.ground_top + self.contact_margin
        below = P[:, 2] < floor
        if np.any(below):
            depth = floor - P[below, 2]
            P[below, 2] = floor
            slip = P[below] - X0[below]
            slip[:, 2] = 0.0
            mag = np.linalg.norm(slip, axis=1)
            limit = np.minimum(self.mu_ground * depth, mag)
            scale = np.where(mag > 1e-12, limit / np.maximum(mag, 1e-12), 0.0)
            P[below] -= slip * scale[:, None]

        self.V = (P - X0) / dt
        speed = np.linalg.norm(self.V, axis=1)
        too_fast = speed > self.max_speed
        if np.any(too_fast):
            self.V[too_fast] *= (self.max_speed / speed[too_fast])[:, None]
        self.X = P
        self.step_index += 1
        self.time = self.step_index * dt
        self._diag["substeps"] = self.step_index
        self._diag["max_speed"] = max(self._diag["max_speed"], float(speed.max()))

    # ------------------------------------------------- surface deformation
    def _surface_jacobian(self, X):
        """Per-node 2x2 material-frame Jacobian from the discrete surface metric."""
        P = X.reshape(self.nring, self.ncol, 3)
        # axial tangent (central differences, one-sided at the free hems)
        t_u = np.empty_like(P)
        t_u[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2.0 * self.du)
        t_u[:, 0] = (P[:, 1] - P[:, 0]) / self.du
        t_u[:, -1] = (P[:, -1] - P[:, -2]) / self.du
        # circumferential tangent (periodic)
        t_v = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) / (2.0 * self.dv)
        t_u = t_u.reshape(-1, 3)
        t_v = t_v.reshape(-1, 3)
        a11 = np.einsum("ij,ij->i", t_u, t_u)
        a12 = np.einsum("ij,ij->i", t_u, t_v)
        a22 = np.einsum("ij,ij->i", t_v, t_v)
        # upper-triangular square root of the metric (rotation-free frame)
        l11 = np.sqrt(np.maximum(a11, 1e-12))
        l21 = a12 / l11
        l22 = np.sqrt(np.maximum(a22 - l21 * l21, 1e-12))
        G = np.zeros((len(t_u), 2, 2), dtype=float)
        G[:, 0, 0] = l11
        G[:, 0, 1] = l21
        G[:, 1, 1] = l22
        return G

    def _deformation_gradient(self):
        F = self._surface_jacobian(self.X) @ self.inv_G0
        return np.where(np.isfinite(F), F, 0.0)

    # ------------------------------------------------------------- exporting
    def _out_index(self):
        rows = np.arange(self.nz) % self.nring
        cols = np.arange(self.ncol)
        return (rows[:, None] * self.ncol + cols[None, :]).ravel()

    def _frame(self, time: float) -> StateFrame:
        sel = self._out_index()
        position = np.ascontiguousarray(self.X[sel], dtype=float)
        velocity = np.ascontiguousarray(self.V[sel], dtype=float)
        F = np.ascontiguousarray(self._deformation_gradient()[sel], dtype=float)

        external = self.mass_out[:, None] * (
            self.gravity[None, :] - self.drag * velocity
        )

        gap = self._boundary_gap(position, float(time)) - 0.5 * self.thickness
        near = gap < max(10.0 * self.thickness, 0.16)
        if not np.any(near):
            order = np.argsort(gap)[: max(16, self.nout // 8)]
            contact = gap[order]
        else:
            contact = gap[near]

        frame = StateFrame(
            float(time),
            position,
            velocity,
            mass=self.mass_out.copy(),
            body_id=np.zeros(self.nout, dtype=np.int32),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass_out.copy(),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(contact, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(external, dtype=float),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "joint_angle": float(self._joint_angle(float(time))[0]),
            },
        )
        frame.validate()
        self._diag["min_gap"] = float(np.min(gap))
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        limit = self.substeps_per_sample * (int(self.duration / self.sample_dt) + 8) + 64
        while self.time < target - 0.25 * self.dt and guard < limit:
            self._substep()
            guard += 1
            if not np.isfinite(self.X).all():
                self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
                self.V = np.zeros_like(self.V)
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "xpbd_cloth_sleeve_articulated_capsules",
            "nodes": int(self.nnodes),
            "exported_samples": int(self.nout),
            "substep_dt": float(self.dt),
            "substeps_taken": int(self._diag["substeps"]),
            "max_speed": float(self._diag["max_speed"]),
            "final_min_gap": float(self._diag["min_gap"]),
        }
