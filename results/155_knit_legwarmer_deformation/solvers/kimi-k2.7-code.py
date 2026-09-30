"""Solver Agent-owned implementation for knitted legwarmer deformation.

Implements a mass-spring cloth solver with contact against an articulated
leg collider. The leg bends at the knee, compressing the inner side of the
cloth sleeve and stretching the outer side.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        frame0 = scene.initial_state()
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.body_id = frame0.body_id.copy()
        self.boundary = frame0.attributes.get(
            "boundary_vertex", np.zeros(len(self.mass), dtype=np.int32)
        )
        self.rest_pos = self.position.copy()

        # Cloth parameters from configuration
        cloth_cfg = self.config["cloth"]
        self.thickness = float(cloth_cfg["thickness"])
        self.gravity_vec = np.asarray(self.config["gravity"], dtype=float)
        self.n_axial, self.n_ang = map(int, cloth_cfg["grid_shape"])
        self.n_verts = len(self.mass)

        # Time-stepping
        self.substeps = 200
        self.dt = float(self.config["sample_dt"]) / self.substeps

        # Material / numerical parameters
        self.k_stretch = 8.0e3
        self.k_bend = 3.0e2
        self.k_contact = 3.0e4
        self.c_spring = 0.5
        self.c_damp = 0.1
        self.mu = 0.3
        self.contact_margin = 0.005

        self.time = 0.0

        # Build connectivity and precompute rest quantities
        self._build_mesh()
        self._compute_rest_basis()

    def _build_mesh(self):
        """Construct the tube mesh and precompute rest spring lengths."""
        n_a, n_g = self.n_axial, self.n_ang

        # Build triangle faces for a tube: axial rows are open, angular columns wrap.
        faces = []
        for i in range(n_a - 1):
            for j in range(n_g):
                v0 = i * n_g + j
                v1 = (i + 1) * n_g + j
                v2 = (i + 1) * n_g + (j + 1) % n_g
                v3 = i * n_g + (j + 1) % n_g
                faces.append((v0, v1, v2))
                faces.append((v0, v2, v3))
        self.faces = np.asarray(faces, dtype=np.int32)

        # Unique undirected edges
        edge_set = set()
        for f in self.faces:
            for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                a, b = min(e), max(e)
                edge_set.add((a, b))
        self.edges = np.asarray(list(edge_set), dtype=np.int32)

        # Rest lengths for stretch springs
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        self.rest_stretch_len = np.linalg.norm(
            self.rest_pos[i0] - self.rest_pos[i1], axis=1
        )

        # Build edge-to-opposite-vertex map for bending springs
        edge_to_tris: dict[tuple[int, int], list[int]] = {}
        for ti, f in enumerate(self.faces):
            for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                a, b = min(e), max(e)
                edge_to_tris.setdefault((a, b), []).append(ti)

        bend_a, bend_b = [], []
        for (a, b), tris in edge_to_tris.items():
            if len(tris) == 2:
                t1, t2 = self.faces[tris[0]], self.faces[tris[1]]
                opp1 = [v for v in t1 if v != a and v != b]
                opp2 = [v for v in t2 if v != a and v != b]
                if opp1 and opp2:
                    bend_a.append(opp1[0])
                    bend_b.append(opp2[0])
        self.bend_a = np.asarray(bend_a, dtype=np.int32)
        self.bend_b = np.asarray(bend_b, dtype=np.int32)
        self.rest_bend_len = np.linalg.norm(
            self.rest_pos[self.bend_a] - self.rest_pos[self.bend_b], axis=1
        )

    def _compute_rest_basis(self):
        """Precompute orthonormal rest tangent basis and material element sizes."""
        n_g = self.n_ang
        rest_u = np.zeros((self.n_verts, 3))
        rest_v = np.zeros((self.n_verts, 3))

        for idx in range(self.n_verts):
            i = idx // n_g
            j = idx % n_g

            # Axial tangent (one-sided at boundaries)
            if i == 0:
                du = self.rest_pos[idx + n_g] - self.rest_pos[idx]
            elif i == self.n_axial - 1:
                du = self.rest_pos[idx] - self.rest_pos[idx - n_g]
            else:
                du = 0.5 * (
                    self.rest_pos[idx + n_g] - self.rest_pos[idx - n_g]
                )

            # Circumferential tangent (wraps around)
            jm = (j - 1) % n_g
            jp = (j + 1) % n_g
            dv = 0.5 * (
                self.rest_pos[i * n_g + jp] - self.rest_pos[i * n_g + jm]
            )

            rest_u[idx] = du
            rest_v[idx] = dv

        rest_u_len = np.linalg.norm(rest_u, axis=1)
        rest_v_len = np.linalg.norm(rest_v, axis=1)
        rest_u_len[rest_u_len == 0] = 1.0
        rest_v_len[rest_v_len == 0] = 1.0

        # Orthonormalize: eu is unit, ev is unit and orthogonal to eu
        eu = rest_u / rest_u_len[:, None]
        proj = np.einsum("ij,ij->i", rest_v, eu)[:, None]
        ev = rest_v - proj * eu
        ev_norm = np.linalg.norm(ev, axis=1, keepdims=True)
        ev_norm[ev_norm == 0] = 1.0
        ev = ev / ev_norm

        self.rest_u_len = rest_u_len
        self.rest_v_len = rest_v_len
        self.basis_u = eu
        self.basis_v = ev

    # ------------------------------------------------------------------
    # Kinematic leg motion
    # ------------------------------------------------------------------
    def _leg_bend(self, t: float) -> tuple[float, float]:
        """Return knee bend angle (rad) and angular velocity (rad/s)."""
        duration = float(self.config["duration"])
        T = duration * 0.6
        theta_max = np.deg2rad(60.0)
        if t <= 0.0:
            return 0.0, 0.0
        if t >= T:
            return theta_max, 0.0
        s = t / T
        s2 = s * s
        s3 = s2 * s
        smooth = 3.0 * s2 - 2.0 * s3
        dsmooth = 6.0 * s / T - 6.0 * s2 / T
        return theta_max * smooth, theta_max * dsmooth

    def _rotate_y(self, p, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c])

    def _colliders(self, t: float):
        """Return a list of collider descriptors with local velocities."""
        theta, omega = self._leg_bend(t)
        colliders: list[dict] = []

        # Upper leg (static)
        colliders.append(
            {
                "type": "capsule",
                "a": np.array([-1.27, 0.0, 1.10]),
                "b": np.array([0.03, 0.0, 1.10]),
                "r": 0.34,
                "omega": np.zeros(3),
            }
        )

        # Lower leg (rotates around knee at origin)
        c_lo = self._rotate_y([0.62, 0.0, 1.10], theta)
        axis_lo = np.array(
            [np.sin(np.pi / 2.0 + theta), 0.0, np.cos(np.pi / 2.0 + theta)]
        )
        half_h = 1.30 * 0.5
        a_lo = c_lo - half_h * axis_lo
        b_lo = c_lo + half_h * axis_lo
        colliders.append(
            {
                "type": "capsule",
                "a": a_lo,
                "b": b_lo,
                "r": 0.31,
                "omega": np.array([0.0, omega, 0.0]),
            }
        )

        # Knee joint (static sphere)
        colliders.append(
            {
                "type": "sphere",
                "c": np.array([0.0, 0.0, 1.10]),
                "r": 0.36,
                "omega": np.zeros(3),
            }
        )

        # Foot (rotates with lower leg)
        c_ft = self._rotate_y([1.55, -0.1, 0.92], theta)
        axis_ft = np.array(
            [np.sin(np.pi / 2.0 + theta), 0.0, np.cos(np.pi / 2.0 + theta)]
        )
        half_h_ft = 0.72 * 0.5
        a_ft = c_ft - half_h_ft * axis_ft
        b_ft = c_ft + half_h_ft * axis_ft
        colliders.append(
            {
                "type": "capsule",
                "a": a_ft,
                "b": b_ft,
                "r": 0.25,
                "omega": np.array([0.0, omega, 0.0]),
            }
        )

        # Ground (axis aligned box)
        colliders.append(
            {
                "type": "box",
                "c": np.array([0.0, 0.0, -0.05]),
                "h": np.array([2.4, 2.0, 0.05]),
                "omega": np.zeros(3),
            }
        )

        return colliders

    # ------------------------------------------------------------------
    # Collision / contact helpers
    # ------------------------------------------------------------------
    def _sd_collider(self, p: np.ndarray, col: dict) -> np.ndarray:
        if col["type"] == "capsule":
            a, b, r = col["a"], col["b"], col["r"]
            ab = b - a
            denom = np.dot(ab, ab)
            if denom <= 0.0:
                t = np.zeros(len(p))
            else:
                t = np.clip(np.einsum("ij,j->i", p - a, ab) / denom, 0.0, 1.0)
            closest = a[None, :] + t[:, None] * ab[None, :]
            return np.linalg.norm(p - closest, axis=1) - r
        if col["type"] == "sphere":
            return np.linalg.norm(p - col["c"][None, :], axis=1) - col["r"]
        # box
        q = np.abs(p - col["c"][None, :]) - col["h"][None, :]
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0
        )

    def _signed_distances(self, p: np.ndarray, colliders: list[dict]) -> np.ndarray:
        sd = np.full(len(p), 1e6, dtype=float)
        for col in colliders:
            sd = np.minimum(sd, self._sd_collider(p, col))
        return sd

    def _contact_force_and_normal(
        self, p: np.ndarray, v: np.ndarray, colliders: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute contact force, signed distance and outward normal per vertex."""
        # Signed distance to the union of colliders and identify closest collider
        sd_per = np.array([self._sd_collider(p, col) for col in colliders])
        closest = np.argmin(sd_per, axis=0)
        sd = sd_per[closest, np.arange(len(p))]

        # Approximate normal by finite differences of the union SDF
        eps = 1e-5
        normals = np.zeros_like(p)
        for dim in range(3):
            dp = p.copy()
            dp[:, dim] += eps
            sdp = self._signed_distances(dp, colliders)
            normals[:, dim] = (sdp - sd) / eps
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm_len > 1e-12, normals / norm_len, 0.0)

        penetration = -sd
        active = penetration > 0.0
        f_contact = np.zeros_like(p)
        if not np.any(active):
            return f_contact, sd, normals

        n = normals[active]

        # Normal penalty force
        f_n_mag = self.k_contact * penetration[active]
        f_n = f_n_mag[:, None] * n

        # Tangential friction using the velocity of the closest collider
        active_idx = np.where(active)[0]
        v_coll = np.zeros((len(active_idx), 3))
        for k, idx in enumerate(active_idx):
            v_coll[k] = np.cross(colliders[closest[idx]]["omega"], p[idx])

        v_rel = v[active] - v_coll
        v_n = np.einsum("ij,ij->i", v_rel, n)[:, None] * n
        v_t = v_rel - v_n
        v_t_mag = np.linalg.norm(v_t, axis=1)

        f_t_mag = self.mu * f_n_mag
        f_t = np.zeros_like(v_t)
        mask = v_t_mag > 1e-9
        f_t[mask] = (
            -f_t_mag[mask][:, None]
            * (v_t[mask] / v_t_mag[mask][:, None])
        )

        f_contact[active] = f_n + f_t
        return f_contact, sd, normals

    # ------------------------------------------------------------------
    # Elastic forces
    # ------------------------------------------------------------------
    def _spring_forces(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        f = np.zeros_like(p)

        # Stretch springs
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        dx = p[i1] - p[i0]
        L = np.linalg.norm(dx, axis=1)
        L[L < 1e-12] = 1e-12
        dir_ = dx / L[:, None]
        f_mag = self.k_stretch * (L - self.rest_stretch_len)
        dv = v[i1] - v[i0]
        f_damp = self.c_spring * np.einsum("ij,ij->i", dv, dir_)[:, None] * dir_
        f_vec = f_mag[:, None] * dir_ - f_damp
        np.add.at(f, i0, -f_vec)
        np.add.at(f, i1, f_vec)

        # Bending springs across adjacent triangles
        if len(self.bend_a) > 0:
            a = self.bend_a
            b = self.bend_b
            dx = p[b] - p[a]
            L = np.linalg.norm(dx, axis=1)
            L[L < 1e-12] = 1e-12
            dir_ = dx / L[:, None]
            f_mag = self.k_bend * (L - self.rest_bend_len)
            dv = v[b] - v[a]
            f_damp = self.c_spring * np.einsum("ij,ij->i", dv, dir_)[:, None] * dir_
            f_vec = f_mag[:, None] * dir_ - f_damp
            np.add.at(f, a, -f_vec)
            np.add.at(f, b, f_vec)

        return f

    # ------------------------------------------------------------------
    # Deformation gradient per vertex (2x2 in-plane)
    # ------------------------------------------------------------------
    def _deformation_gradient(self, p: np.ndarray) -> np.ndarray:
        n_g = self.n_ang
        F = np.zeros((self.n_verts, 2, 2))
        for idx in range(self.n_verts):
            i = idx // n_g
            j = idx % n_g

            if i == 0:
                du = p[idx + n_g] - p[idx]
            elif i == self.n_axial - 1:
                du = p[idx] - p[idx - n_g]
            else:
                du = 0.5 * (p[idx + n_g] - p[idx - n_g])

            jm = (j - 1) % n_g
            jp = (j + 1) % n_g
            dv = 0.5 * (p[i * n_g + jp] - p[i * n_g + jm])

            deformed_u = du / self.rest_u_len[idx]
            deformed_v = dv / self.rest_v_len[idx]

            eu = self.basis_u[idx]
            ev = self.basis_v[idx]
            F[idx, 0, 0] = np.dot(deformed_u, eu)
            F[idx, 1, 0] = np.dot(deformed_u, ev)
            F[idx, 0, 1] = np.dot(deformed_v, eu)
            F[idx, 1, 1] = np.dot(deformed_v, ev)
        return F

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def _step(self, t: float) -> None:
        p = self.position
        v = self.velocity

        # External + internal forces
        f_grav = self.mass[:, None] * self.gravity_vec[None, :]
        f_spring = self._spring_forces(p, v)

        colliders = self._colliders(t)
        f_contact, _, _ = self._contact_force_and_normal(p, v, colliders)

        f_total = f_grav + f_spring + f_contact

        # Symplectic Euler
        accel = f_total / self.mass[:, None]
        v_new = v + self.dt * accel
        v_new *= 1.0 - self.c_damp * self.dt
        p_new = p + self.dt * v_new

        self.position = p_new
        self.velocity = v_new
        self.time += self.dt

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_steps = int(np.round((target - self.time) / self.dt))
        for _ in range(max(n_steps, 0)):
            # Use mid-step time for the moving colliders
            self._step(self.time + 0.5 * self.dt)

        # Avoid drift of the internal clock; keep it aligned with requested sample times.
        self.time = target

        # Assemble required trajectory fields
        colliders = self._colliders(self.time)
        f_contact, sd, _ = self._contact_force_and_normal(
            self.position, self.velocity, colliders
        )
        f_grav = self.mass[:, None] * self.gravity_vec[None, :]
        external_force = f_grav + f_contact
        F = self._deformation_gradient(self.position)

        frame = StateFrame(
            time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": sd,
                "cumulative_mass_source": np.zeros_like(self.mass),
                "external_force": external_force,
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "vertices": self.n_verts,
            "faces": len(self.faces),
        }
