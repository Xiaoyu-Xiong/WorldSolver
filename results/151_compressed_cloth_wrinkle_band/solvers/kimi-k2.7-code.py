"""Mass-spring cloth solver for the compressed wrinkle-band scene.

Uses an explicit time stepping loop with stretch/bend springs, viscous
damping, gravity, and penalty-plus-projection collision response against
the kinematic grips and the ground.  The left/right boundary vertices are
rigidly attached to inward-moving grips to generate the characteristic
compression-induced wrinkle band.
"""

from __future__ import annotations

import json
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        # ------------------------------------------------------------------
        # Pull the frozen initial state from the scaffold
        # ------------------------------------------------------------------
        frame0 = scene.initial_state()
        self.frame0 = frame0
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.n = len(self.position)
        self.body_id = (
            frame0.body_id.copy()
            if frame0.body_id is not None
            else np.zeros(self.n, dtype=np.int32)
        )

        # ------------------------------------------------------------------
        # Scene parameters
        # ------------------------------------------------------------------
        try:
            self.gravity = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        except Exception:
            self.gravity = np.array([0.0, 0.0, -9.81])

        try:
            self.thickness = float(scene.config["cloth"]["thickness"])
        except Exception:
            self.thickness = 0.015

        # ------------------------------------------------------------------
        # Mesh topology from the frozen asset
        # ------------------------------------------------------------------
        try:
            with open("assets/cloth_topology.json", "r") as f:
                topo = json.load(f)
            self.faces = np.asarray(topo["faces"], dtype=np.int32)
        except Exception:
            # Fallback regular grid if the asset cannot be read.
            nx, ny = 17, 17
            faces = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    v0 = i * ny + j
                    v1 = (i + 1) * ny + j
                    v2 = (i + 1) * ny + (j + 1)
                    v3 = i * ny + (j + 1)
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
            self.faces = np.asarray(faces, dtype=np.int32)

        # ------------------------------------------------------------------
        # Edges and stretch rest lengths
        # ------------------------------------------------------------------
        edge_set = set()
        for f in self.faces:
            i, j, k = f
            edge_set.add((min(i, j), max(i, j)))
            edge_set.add((min(j, k), max(j, k)))
            edge_set.add((min(k, i), max(k, i)))
        self.edges = np.asarray(list(edge_set), dtype=np.int32)

        self.rest_len = np.zeros(len(self.edges))
        for idx, (i, j) in enumerate(self.edges):
            self.rest_len[idx] = np.linalg.norm(self.position[i] - self.position[j])

        # ------------------------------------------------------------------
        # Bending hinges (interior edges -> spring between opposite vertices)
        # ------------------------------------------------------------------
        edge_to_opps: dict[tuple[int, int], list[int]] = {}
        for f in self.faces:
            i, j, k = f
            edge_to_opps.setdefault((min(i, j), max(i, j)), []).append(k)
            edge_to_opps.setdefault((min(j, k), max(j, k)), []).append(i)
            edge_to_opps.setdefault((min(k, i), max(k, i)), []).append(j)

        bend_pairs = []
        bend_rest = []
        for edge, opps in edge_to_opps.items():
            if len(opps) == 2:
                k, l = opps
                bend_pairs.append([k, l])
                bend_rest.append(np.linalg.norm(self.position[k] - self.position[l]))
        self.bend_pairs = np.asarray(bend_pairs, dtype=np.int32)
        self.bend_rest = np.asarray(bend_rest, dtype=float)

        # ------------------------------------------------------------------
        # Material (parametric) coordinates used for the deformation gradient
        # ------------------------------------------------------------------
        self.material_xy = self.position[:, :2].copy()

        self.rest_U_inv = np.zeros((len(self.faces), 2, 2))
        self.rest_area = np.zeros(len(self.faces))
        for idx, f in enumerate(self.faces):
            i, j, k = f
            u0 = self.material_xy[i]
            u1 = self.material_xy[j]
            u2 = self.material_xy[k]
            U = np.column_stack([u1 - u0, u2 - u0])
            det = np.linalg.det(U)
            self.rest_area[idx] = 0.5 * abs(det)
            self.rest_U_inv[idx] = np.linalg.inv(U)

        # ------------------------------------------------------------------
        # Boundary masks (left/right edges are gripped)
        # ------------------------------------------------------------------
        x = self.position[:, 0]
        x_min, x_max = x.min(), x.max()
        tol = 1e-4
        self.left_mask = np.abs(x - x_min) < tol
        self.right_mask = np.abs(x - x_max) < tol
        self.boundary_mask = self.left_mask | self.right_mask
        self.interior_mask = ~self.boundary_mask

        # ------------------------------------------------------------------
        # Small deterministic geometric imperfection to seed wrinkling while
        # keeping the t=0 frame flat.
        # ------------------------------------------------------------------
        rng = np.random.default_rng(self.seed)
        z_noise = np.zeros(self.n)
        z_noise[self.interior_mask] = rng.normal(0.0, 1e-4, size=int(self.interior_mask.sum()))
        self.position[:, 2] += z_noise

        # ------------------------------------------------------------------
        # Obstacle definitions (ground + moving grips)
        # ------------------------------------------------------------------
        self.ground_c = np.array([0.0, 0.0, -0.05])
        self.ground_h = np.array([2.4, 2.0, 0.05])
        self.grip_half = np.array([0.12, 1.25, 0.12])

        # ------------------------------------------------------------------
        # Physical parameters
        # ------------------------------------------------------------------
        self.k_stretch = 5000.0
        self.k_bend = 1.0
        self.damping = 0.5
        self.k_coll = 2000.0
        self.c_coll = 5.0
        self.coll_margin = 0.0
        self.substep_dt = 2.0e-4

        self.current_time = 0.0

    # ----------------------------------------------------------------------
    # Kinematic grip schedule
    # ----------------------------------------------------------------------
    def _grip_state(self, t: float):
        """Return (left_x, right_x, grip_speed) at time t."""
        start = 1.0
        duration = 3.0
        speed = 0.25
        s = min(max(t - start, 0.0), duration)
        left_x = -1.5 + speed * s
        right_x = 1.5 - speed * s
        grip_speed = speed if start < t < start + duration else 0.0
        return left_x, right_x, grip_speed

    def _apply_boundary(self, t: float) -> None:
        left_x, right_x, grip_speed = self._grip_state(t)

        # Position: left/right edges follow their grips; y,z stay fixed.
        self.position[self.left_mask, 0] = left_x
        self.position[self.right_mask, 0] = right_x
        self.position[self.boundary_mask, 1] = self.frame0.position[self.boundary_mask, 1]
        self.position[self.boundary_mask, 2] = self.frame0.position[self.boundary_mask, 2]

        # Velocity matches the grip motion.
        self.velocity[self.left_mask] = [grip_speed, 0.0, 0.0]
        self.velocity[self.right_mask] = [-grip_speed, 0.0, 0.0]

    # ----------------------------------------------------------------------
    # Signed distance to an axis-aligned box (batch)
    # ----------------------------------------------------------------------
    @staticmethod
    def _box_sdf_batch(pos: np.ndarray, c: np.ndarray, h: np.ndarray):
        """Return signed distances (N,) and outward normals (N,3) for a box."""
        d = pos - c
        inside = np.all(np.abs(d) <= h, axis=1)
        s = np.empty(pos.shape[0], dtype=float)
        n = np.zeros_like(pos)

        # Outside points
        outside = ~inside
        if np.any(outside):
            closest = c + np.clip(d[outside], -h, h)
            diff = pos[outside] - closest
            dist = np.linalg.norm(diff, axis=1)
            dist = np.maximum(dist, 1e-12)
            s[outside] = dist
            n[outside] = diff / dist[:, None]

        # Inside points
        if np.any(inside):
            dists = h - np.abs(d[inside])
            axis = np.argmin(dists, axis=1)
            s[inside] = -dists[np.arange(len(axis)), axis]
            n[inside, axis] = np.sign(d[inside, axis])
            # Guard against a zero normal component at the center.
            zero = n[inside, axis] == 0.0
            if np.any(zero):
                n[np.where(inside)[0][zero], axis[zero]] = 1.0

        return s, n

    def _obstacle_specs(self):
        left_x, right_x, _ = self._grip_state(self.current_time)
        return [
            (self.ground_c, self.ground_h),
            (np.array([left_x, 0.0, 2.15]), self.grip_half),
            (np.array([right_x, 0.0, 2.15]), self.grip_half),
        ]

    # ----------------------------------------------------------------------
    # Internal force computations
    # ----------------------------------------------------------------------
    def _stretch_force(self, pos: np.ndarray) -> np.ndarray:
        if len(self.edges) == 0:
            return np.zeros_like(pos)
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        e = pos[j] - pos[i]
        L = np.linalg.norm(e, axis=1)
        safe = L > 1e-12
        factor = np.zeros_like(L)
        factor[safe] = (L[safe] - self.rest_len[safe]) / L[safe]
        force = self.k_stretch * factor[:, None] * e

        f = np.zeros_like(pos)
        np.add.at(f, i, force)
        np.add.at(f, j, -force)
        return f

    def _bend_force(self, pos: np.ndarray) -> np.ndarray:
        if len(self.bend_pairs) == 0:
            return np.zeros_like(pos)
        k = self.bend_pairs[:, 0]
        l = self.bend_pairs[:, 1]
        e = pos[l] - pos[k]
        L = np.linalg.norm(e, axis=1)
        safe = L > 1e-12
        factor = np.zeros_like(L)
        factor[safe] = (L[safe] - self.bend_rest[safe]) / L[safe]
        force = self.k_bend * factor[:, None] * e

        f = np.zeros_like(pos)
        np.add.at(f, k, force)
        np.add.at(f, l, -force)
        return f

    def _collision_penalty(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        f = np.zeros_like(pos)
        for c, h in self._obstacle_specs():
            s, n = self._box_sdf_batch(pos, c, h)
            mask = self.interior_mask & (s < self.coll_margin)
            if not np.any(mask):
                continue
            pen = self.coll_margin - s[mask]
            v = vel[mask]
            vn = np.sum(v * n[mask], axis=1)
            f_pen = self.k_coll * pen[:, None] * n[mask] - self.c_coll * vn[:, None] * n[mask]
            f[mask] += f_pen
        return f

    def _project_collisions(self, pos: np.ndarray, vel: np.ndarray):
        for c, h in self._obstacle_specs():
            s, n = self._box_sdf_batch(pos, c, h)
            mask = self.interior_mask & (s < 0.0)
            if not np.any(mask):
                continue
            pos[mask] -= s[mask, None] * n[mask]
            vn = np.sum(vel[mask] * n[mask], axis=1)
            vel[mask] -= vn[:, None] * n[mask]
        return pos, vel

    # ----------------------------------------------------------------------
    # Trajectory helpers
    # ----------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]
        p0 = self.position[i]
        p1 = self.position[j]
        p2 = self.position[k]

        e1 = (p1 - p0)[:, :2]
        e2 = (p2 - p0)[:, :2]
        E = np.stack([e1, e2], axis=2)  # (F, 2, 2)
        F_tri = np.einsum("fij,fjk->fik", E, self.rest_U_inv)

        F = np.zeros((self.n, 2, 2))
        W = np.zeros((self.n, 2, 2))
        weight = self.rest_area[:, None, None]
        for v in range(3):
            idx = self.faces[:, v]
            np.add.at(F, idx, weight * F_tri)
            np.add.at(W, idx, weight)

        mask = W > 0.0
        F[mask] /= W[mask]
        return F

    def _signed_distance(self) -> np.ndarray:
        s = np.full(self.n, np.inf)
        for c, h in self._obstacle_specs():
            s_i, _ = self._box_sdf_batch(self.position, c, h)
            s = np.minimum(s, s_i)
        return s

    def _external_force(self) -> np.ndarray:
        f = self.mass[:, None] * self.gravity[None, :]
        for c, h in self._obstacle_specs():
            s, n = self._box_sdf_batch(self.position, c, h)
            mask = self.interior_mask & (s < self.coll_margin)
            if not np.any(mask):
                continue
            pen = self.coll_margin - s[mask]
            v = self.velocity[mask]
            vn = np.sum(v * n[mask], axis=1)
            f_pen = self.k_coll * pen[:, None] * n[mask] - self.c_coll * vn[:, None] * n[mask]
            f[mask] += f_pen
        return f

    def _build_frame(self, t: float) -> StateFrame:
        F = self._deformation_gradient()
        sd = self._signed_distance()
        ext = self._external_force()

        return StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": sd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )

    # ----------------------------------------------------------------------
    # Public solver interface
    # ----------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity

        f = np.zeros_like(pos)
        f += self.mass[:, None] * self.gravity[None, :]
        f += self._stretch_force(pos)
        f += self._bend_force(pos)
        f -= self.damping * vel
        f += self._collision_penalty(pos, vel)

        v_new = vel + dt * (f / self.mass[:, None])
        x_new = pos + dt * v_new

        x_new, v_new = self._project_collisions(x_new, v_new)

        self.position = x_new
        self.velocity = v_new
        self.current_time += dt
        self._apply_boundary(self.current_time)

    def advance_to(self, time: float) -> StateFrame:
        if time <= 0.0:
            # Return the exact scaffold initial state at t=0.
            return self.frame0

        # Integrate forward with fixed substeps.
        while self.current_time < time - 1e-12:
            dt = min(self.substep_dt, time - self.current_time)
            self._substep(dt)

        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_dt": self.substep_dt,
            "vertices": self.n,
            "faces": len(self.faces),
            "edges": len(self.edges),
            "bending_hinges": len(self.bend_pairs),
        }
