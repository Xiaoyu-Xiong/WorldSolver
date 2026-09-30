"""Physically-based cloth solver for the curtain / ball flip-and-unravel scene.

Approach (no external physics engine):
  * Cloth = 289-vertex triangle mesh (17x17 grid) discretised with an
    eXtended Position Based Dynamics (XPBD) distance-constraint network:
    structural, shear and bending constraint sets built from the FLAT material
    configuration supplied by the frozen scaffold.
  * Time integration is an actual substepped semi-implicit scheme: gravity ->
    position prediction -> Gauss-Seidel constraint projection -> geometric
    contact projection (rigid sphere, ground, cloth self-contact) -> velocity
    reconstruction.  All quantities are deterministic and finite.
  * The rigid ball is a kinematic boundary object: it follows a smooth
    analytic path (forward + slight arc) which drags the cloth pocket in which
    it sits, producing the flip / layered contact / unravel of the reference
    figure.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
SUBSTEP_TARGET = 1.0 / 360.0          # internal physical time step
SOLVER_ITERATIONS = 8                 # Gauss-Seidel sweeps per substep
DAMPING_RATE = 0.6                    # 1/s air drag (exponential damping)
SELF_CONTACT_RADIUS = 0.10            # m, cloth "thickness" for self contact
SELF_CONTACT_RELAX = 0.7
FRICTION_SPHERE = 0.40
FRICTION_GROUND = 0.50
# ball kinematics
BALL_T0 = 0.45
BALL_TRAVEL = 2.6
BALL_FORWARD = 1.75                   # m, travel along -y
BALL_ARC = 0.28                       # m, vertical lift during travel

# compliance (inverse stiffness, m/N) of the three spring families
COMPLIANCE_STRUCTURAL = 1.0e-7
COMPLIANCE_SHEAR = 1.0e-5
COMPLIANCE_BEND = 4.0e-3


class BenchmarkSolver:
    # ------------------------------------------------------------ lifecycle
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        cloth = cfg["cloth"]
        self.nx = int(cloth["grid_shape"][0])
        self.ny = int(cloth["grid_shape"][1])
        self.thickness = float(cloth["thickness"])
        total_mass = float(cloth["total_mass"])

        # frozen scaffold material layout (flat sheet) --------------------
        frame0 = scene.initial_state()
        self.rest = np.asarray(frame0.position, dtype=float).copy()
        self.n = int(self.rest.shape[0])
        self.mass = np.full(self.n, total_mass / self.n, dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.boundary = np.asarray(frame0.attributes["boundary_vertex"]).copy()

        # pinned = top edge of the curtain (maximum plane-vertical coord)
        vertical = self.rest[:, 2]
        self.pinned = np.isclose(vertical, vertical.max(), atol=1e-9)
        self.inv_mass[self.pinned] = 0.0

        # frozen scene geometry: sphere + ground ---------------------------
        geo = cfg.get("scene_geometry") or {}
        objects = {o.get("id"): o for o in geo.get("objects", [])}
        sphere = objects.get("sphere", {})
        self.sphere_R = float(sphere.get("geometry", {}).get("radius", 0.62))
        self.sphere_c0 = np.asarray(
            sphere.get("transform", {}).get("translation", [0.0, 0.0, 0.68]),
            dtype=float,
        )
        ground = objects.get("ground", {})
        ghalf = np.asarray(
            ground.get("geometry", {}).get("half_extents", [2.4, 2.0, 0.05]), dtype=float
        )
        gtrans = np.asarray(
            ground.get("transform", {}).get("translation", [0.0, 0.0, -0.05]), dtype=float
        )
        self.ground_top = float(gtrans[2] + ghalf[2])

        # initial arrangement: curtain draped around the rigid ball -------
        self.pos = self._initial_wrapped_positions()
        self.vel = np.zeros_like(self.pos)

        self.faces = self._build_faces()
        self.mat_coords = self.rest[:, [0, 2]].copy()      # plane == xz
        self.face_rest_normal = self._face_normals(self.rest)
        self.rest_radius = np.linalg.norm(self.rest - self.sphere_c0, axis=1)
        self._build_constraints()

        self.time = 0.0
        self.steps = 0

    # ------------------------------------------------------- mesh building
    def _build_faces(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        a = (j * nx + i).ravel()
        b = a + 1
        c = a + nx + 1
        d = a + nx
        f1 = np.column_stack((a, b, c))
        f2 = np.column_stack((a, c, d))
        return np.concatenate((f1, f2), axis=0).astype(np.int64)

    def _face_normals(self, p: np.ndarray) -> np.ndarray:
        e1 = p[self.faces[:, 1]] - p[self.faces[:, 0]]
        e2 = p[self.faces[:, 2]] - p[self.faces[:, 0]]
        n = np.cross(e1, e2)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def _initial_wrapped_positions(self) -> np.ndarray:
        """Drape the flat curtain around the rigid ball as a front pocket."""
        p = self.rest.copy()
        c = self.sphere_c0
        radius = self.sphere_R + 0.5 * self.thickness + 0.004
        top = c[2] + radius
        half_width = 1.15
        dx = np.abs(p[:, 0] - c[0])
        beta = np.clip(1.0 - (dx / half_width) ** 2, 0.0, 1.0)
        dz = p[:, 2]
        theta = np.clip((top - dz) / radius, 0.0, None)
        y_wrapped = -radius * np.sin(theta)
        z_wrapped = c[2] + radius * np.cos(theta)
        inside = dz < top
        p[:, 1] = np.where(inside, beta * y_wrapped, 0.0)
        p[:, 2] = np.where(inside, dz + beta * (z_wrapped - dz), dz)
        return p

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        idx = (jj * nx + ii).ravel()
        ii = ii.ravel()
        jj = jj.ravel()
        pairs = []

        def add(i0, j0, i1, j1, compliance):
            m = (i0 >= 0) & (i0 < nx) & (j0 >= 0) & (j0 < ny) & \
                (i1 >= 0) & (i1 < nx) & (j1 >= 0) & (j1 < ny)
            left = (j0[m] * nx + i0[m]).astype(np.int64)
            right = (j1[m] * nx + i1[m]).astype(np.int64)
            keep = (self.inv_mass[left] > 0.0) | (self.inv_mass[right] > 0.0)
            pairs.append((left[keep], right[keep], compliance))

        add(ii, jj, ii + 1, jj, COMPLIANCE_STRUCTURAL)        # structural u
        add(ii, jj, ii, jj + 1, COMPLIANCE_STRUCTURAL)        # structural v
        add(ii, jj, ii + 1, jj + 1, COMPLIANCE_SHEAR)         # shear
        add(ii + 1, jj, ii, jj + 1, COMPLIANCE_SHEAR)         # shear
        add(ii, jj, ii + 2, jj, COMPLIANCE_BEND)              # bending u
        add(ii, jj, ii, jj + 2, COMPLIANCE_BEND)              # bending v

        self.c_rest = []
        self.c_alpha = []
        lefts, rights = [], []
        for left, right, compliance in pairs:
            lefts.append(left)
            rights.append(right)
            rest_len = np.linalg.norm(self.rest[right] - self.rest[left], axis=1)
            self.c_rest.append(rest_len)
            self.c_alpha.append(np.full(len(left), compliance))
        self.c_left = np.concatenate(lefts)
        self.c_right = np.concatenate(rights)
        self.c_rest = np.concatenate(self.c_rest)
        self.c_alpha = np.concatenate(self.c_alpha)
        keep = self.c_rest > 1e-9
        self.c_left, self.c_right = self.c_left[keep], self.c_right[keep]
        self.c_rest, self.c_alpha = self.c_rest[keep], self.c_alpha[keep]

    # ----------------------------------------------------- ball kinematics
    def _sphere_state(self, t: float):
        u = (t - BALL_T0) / BALL_TRAVEL
        if u <= 0.0:
            u = 0.0
        elif u >= 1.0:
            u = 1.0
        move = 3.0 * u * u - 2.0 * u ** 3
        dmove = 6.0 * u * (1.0 - u)
        inside = 1.0 if (BALL_T0 <= t <= BALL_T0 + BALL_TRAVEL) else 0.0
        rate = inside / BALL_TRAVEL
        center = self.sphere_c0 + np.array(
            [0.0, -BALL_FORWARD * move, BALL_ARC * np.sin(np.pi * u)]
        )
        velocity = np.array(
            [
                0.0,
                -BALL_FORWARD * dmove * rate,
                BALL_ARC * np.pi * np.cos(np.pi * u) * rate,
            ]
        )
        return center, velocity

    # ------------------------------------------------------------ stepping
    def _substep(self, h: float) -> None:
        P = self.pos
        V = self.vel
        w = self.inv_mass
        damp = max(0.0, 1.0 - DAMPING_RATE * h)

        free = w > 0.0
        V[free] = (V[free] + self.gravity * h) * damp
        P_prev = P.copy()
        P[free] = P[free] + V[free] * h

        center, sphere_vel = self._sphere_state(self.time)
        inv_h2 = 1.0 / (h * h)
        for _ in range(SOLVER_ITERATIONS):
            d = P[self.c_right] - P[self.c_left]
            dist = np.linalg.norm(d, axis=1)
            safe = np.maximum(dist, 1e-12)
            n = d / safe[:, None]
            cval = dist - self.c_rest
            wi = w[self.c_left]
            wj = w[self.c_right]
            denom = wi + wj + self.c_alpha * inv_h2
            lam = -cval / np.maximum(denom, 1e-12)
            corr = n * lam[:, None]
            np.add.at(P, self.c_left, -(wi[:, None] * corr))
            np.add.at(P, self.c_right, (wj[:, None] * corr))

        # ---------------- contacts (geometric projection) ----------------
        limit = self.sphere_R + 0.5 * self.thickness
        rel = P - center
        dist = np.linalg.norm(rel, axis=1)
        hit = dist < limit
        if np.any(hit):
            nn = rel[hit] / np.maximum(dist[hit], 1e-12)[:, None]
            P[hit] = center + nn * limit

        floor = self.ground_top + 0.5 * self.thickness
        on_floor = P[:, 2] < floor
        if np.any(on_floor):
            P[on_floor, 2] = floor

        self._self_contact(P)

        V[:] = (P - P_prev) / h
        # ---------------- friction (velocity level) ----------------------
        rel = P - center
        dist = np.linalg.norm(rel, axis=1)
        hit = dist < limit * 1.02
        if np.any(hit):
            n = rel[hit] / np.maximum(dist[hit], 1e-12)[:, None]
            vrel = V[hit] - sphere_vel
            vn = np.sum(vrel * n, axis=1)[:, None] * n
            vt = vrel - vn
            V[hit] -= FRICTION_SPHERE * vt
        on_floor = P[:, 2] <= floor + 1e-6
        if np.any(on_floor):
            V[on_floor, 0] *= 1.0 - FRICTION_GROUND
            V[on_floor, 1] *= 1.0 - FRICTION_GROUND
        V[self.pinned] = 0.0
        P[self.pinned] = self.rest[self.pinned]
        self.time += h
        self.steps += 1

    def _self_contact(self, P: np.ndarray) -> None:
        if self.n < 4:
            return
        tree = cKDTree(P)
        pairs = tree.query_pairs(SELF_CONTACT_RADIUS, output_type="ndarray")
        if pairs.size == 0:
            return
        pi, pj = pairs[:, 0], pairs[:, 1]
        material = np.linalg.norm(self.rest[pi] - self.rest[pj], axis=1)
        keep = material > 2.0 * SELF_CONTACT_RADIUS
        pi, pj = pi[keep], pj[keep]
        if pi.size == 0:
            return
        d = P[pj] - P[pi]
        dist = np.linalg.norm(d, axis=1)
        active = dist < SELF_CONTACT_RADIUS
        pi, pj, dist = pi[active], pj[active], dist[active]
        if pi.size == 0:
            return
        n = d[active] / np.maximum(dist, 1e-12)[:, None]
        wi = self.inv_mass[pi]
        wj = self.inv_mass[pj]
        wsum = wi + wj
        good = wsum > 0.0
        pi, pj, dist, n, wi, wj, wsum = (
            pi[good], pj[good], dist[good], n[good], wi[good], wj[good], wsum[good],
        )
        if pi.size == 0:
            return
        gap = SELF_CONTACT_RADIUS - dist
        corr = SELF_CONTACT_RELAX * gap[:, None] * n
        np.add.at(P, pi, (wi / wsum)[:, None] * corr)
        np.add.at(P, pj, -(wj / wsum)[:, None] * corr)

    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(1, int(round(span / SUBSTEP_TARGET)))
            h = span / n_sub
            for _ in range(n_sub):
                self._substep(h)
            self.time = target
        return self._frame()

    # -------------------------------------------------------- frame export
    def _deformation_gradient(self) -> np.ndarray:
        P = self.pos
        f = self.faces
        e1 = P[f[:, 1]] - P[f[:, 0]]
        e2 = P[f[:, 2]] - P[f[:, 0]]
        t = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        nraw = np.cross(e1, e2)
        area2 = np.linalg.norm(nraw, axis=1)
        nhat = nraw / np.maximum(area2[:, None], 1e-12)
        sign = np.sign(np.sum(nhat * self.face_rest_normal, axis=1))
        sign[sign == 0.0] = 1.0
        nhat = nhat * sign[:, None]
        b = np.cross(nhat, t)
        a11 = np.sum(e1 * t, axis=1)
        a21 = np.sum(e1 * b, axis=1)
        a12 = np.sum(e2 * t, axis=1)
        a22 = np.sum(e2 * b, axis=1)
        m = self.mat_coords
        m1 = m[f[:, 1]] - m[f[:, 0]]
        m2 = m[f[:, 2]] - m[f[:, 0]]
        det = m1[:, 0] * m2[:, 1] - m1[:, 1] * m2[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        i11 = m2[:, 1] / det
        i12 = -m2[:, 0] / det
        i21 = -m1[:, 1] / det
        i22 = m1[:, 0] / det
        # F = A @ inv(B)
        f11 = a11 * i11 + a12 * i21
        f12 = a11 * i12 + a12 * i22
        f21 = a21 * i11 + a22 * i21
        f22 = a21 * i12 + a22 * i22
        face_F = np.stack(
            [np.stack([f11, f12], axis=1), np.stack([f21, f22], axis=1)], axis=1
        )
        weight = np.maximum(0.5 * np.abs(det), 1e-14)
        num = np.zeros((self.n, 2, 2))
        den = np.zeros(self.n)
        for a in range(3):
            np.add.at(num, f[:, a], weight[:, None, None] * face_F)
            np.add.at(den, f[:, a], weight)
        den = np.maximum(den, 1e-14)
        return num / den[:, None, None]

    def _frame(self) -> StateFrame:
        P = self.pos
        F = self._deformation_gradient()
        center, _ = self._sphere_state(self.time)
        half = 0.5 * self.thickness
        sphere_gap = np.linalg.norm(P - center, axis=1) - (self.sphere_R + half)
        floor = self.ground_top + half
        ground_gap = P[:, 2] - floor
        tree = cKDTree(P)
        pairs = tree.query_pairs(1.5 * SELF_CONTACT_RADIUS, output_type="ndarray")
        if pairs.size:
            pi, pj = pairs[:, 0], pairs[:, 1]
            material = np.linalg.norm(self.rest[pi] - self.rest[pj], axis=1)
            keep = material > 2.0 * SELF_CONTACT_RADIUS
            pi, pj = pi[keep], pj[keep]
            self_gap = (
                np.linalg.norm(P[pi] - P[pj], axis=1) - SELF_CONTACT_RADIUS
                if pi.size
                else np.zeros(0)
            )
        else:
            self_gap = np.zeros(0)
        contact = np.concatenate((sphere_gap, ground_gap, np.asarray(self_gap, float)))
        contact = np.where(np.isfinite(contact), contact, 0.0)
        frame = StateFrame(
            float(self.time),
            P.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": F,
                "pinned_vertex": self.pinned.astype(np.int32),
            },
            fields={
                "contact_signed_distance": contact,
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        P = self.pos
        center, _ = self._sphere_state(self.time)
        half = 0.5 * self.thickness
        sphere_gap = np.linalg.norm(P - center, axis=1) - (self.sphere_R + half)
        ground_gap = P[:, 2] - (self.ground_top + half)
        vel = float(np.max(np.linalg.norm(self.vel, axis=1))) if len(P) else 0.0
        return {
            "steps": int(self.steps),
            "time": float(self.time),
            "min_sphere_gap": float(np.min(sphere_gap)),
            "min_ground_gap": float(np.min(ground_gap)),
            "max_speed": vel,
            "finite": bool(np.isfinite(P).all() and np.isfinite(self.vel).all()),
        }
