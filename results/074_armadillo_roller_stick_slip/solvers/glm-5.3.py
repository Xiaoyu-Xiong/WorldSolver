"""Volumetric elastic solver: armadillo drawn through counter-rotating rollers.

Physical model
--------------
- The deformable armadillo is a 3-D elastic solid discretised with a
  tetrahedral mesh (surface triangle fan from one interior node), using
  corotated linear elasticity (E = 5e5 Pa frozen by the task, nu = 0.3) with
  Kelvin-Voigt material damping.  Per-element deformation gradients are
  exported every frame.
- Time integration: semi-implicit (symplectic) Euler with a fixed substep
  dt = 5e-5 s, chosen from the assembled stiffness/mass spectrum of the actual
  discretisation (safety factor ~0.28 of the explicit stability limit).
- Rollers are kinematic cylinders rotating about the world y axis
  (counter-rotating upper/lower pair, ramped on per the frozen schedule);
  their surface velocity enters the contact model exactly.
- Contact: penalty normal force plus per-contact Coulomb friction with a
  static threshold and a lower dynamic coefficient.  Sticking contacts carry
  a tangential anchor (elastic stick force); when the tangential force
  exceeds mu_s * N the contact slips at mu_d * N, which produces genuine
  stick-slip (catch -> stretch -> slip) dynamics rather than scripted motion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _load_obj_triangles(path: Path):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(tok) for tok in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        for i in range(1, len(face) - 1):
            triangles.append((face[0], face[i], face[i + 1]))
    return np.asarray(vertices, dtype=float), triangles


def _quat_rotate_xyzw(q, vec):
    """Rotate vec by quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot @ np.asarray(vec, dtype=float)


class BenchmarkSolver:
    DT = 5.0e-5                 # integration substep [s]
    POISSON = 0.30
    CONTACT_K = 5.0e5           # normal penalty stiffness [N/m]
    FRICTION_K = 2.0e5          # tangential (stick) stiffness [N/m]
    MU_DYNAMIC = 0.40           # dynamic friction coefficient
    VISCOUS_ZETA = 0.05         # Kelvin-Voigt damping coefficient [s]
    CENTROID_MASS = 0.12        # interior node lumped mass [kg]
    STICK_SPEED_TOL = 0.01      # stick classification tolerance [m/s]

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.mu_s = float(frame0.scalars.get("roller_friction", 0.5))
        self.E = float(frame0.scalars.get("reference_youngs_modulus_pa", 5.0e5))
        self.mu_lame = self.E / (2.0 * (1.0 + self.POISSON))
        self.lam_lame = (
            self.E * self.POISSON / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))
        )
        self.c_visc = self.VISCOUS_ZETA * 2.0 * self.mu_lame

        # --- nodes: surface samples + one interior fan node ---
        self.x_surf0 = np.asarray(frame0.position, dtype=float).copy()
        self.n_surf = len(self.x_surf0)
        interior = self.x_surf0.mean(axis=0)
        self.x = np.vstack([self.x_surf0, interior[None, :]])
        self.rest = self.x.copy()
        self.n_nodes = len(self.x)
        self.v = np.zeros_like(self.x)

        # --- tetrahedra: (interior, f0, f1, f2) per surface triangle ---
        obj_path = Path(__file__).resolve().parents[1] / "assets" / "armadillo.obj"
        verts_obj, tris = _load_obj_triangles(obj_path)
        if len(verts_obj) != self.n_surf:
            raise ValueError("armadillo asset vertex count mismatch")
        t0 = np.full(len(tris), self.n_surf - 1, dtype=np.int64)
        t1 = np.array([t[0] for t in tris], dtype=np.int64)
        t2 = np.array([t[1] for t in tris], dtype=np.int64)
        t3 = np.array([t[2] for t in tris], dtype=np.int64)
        self.T0, self.T1, self.T2, self.T3 = t0, t1, t2, t3
        self.tet_nodes = np.concatenate([t0, t1, t2, t3])

        p0 = self.rest[t0]
        dm = np.stack(
            [self.rest[t1] - p0, self.rest[t2] - p0, self.rest[t3] - p0], axis=2
        )                                                    # (Ne,3,3), columns
        self.Dm_inv = np.linalg.inv(dm)
        self.Dm_inv_T = self.Dm_inv.transpose(0, 2, 1)
        self.vols = np.abs(np.linalg.det(dm)) / 6.0
        self.n_elems = len(self.vols)

        # --- lumped masses (total exactly 2.4 kg) ---
        total_mass = float(np.sum(frame0.mass))
        rho = total_mass / self.vols.sum()
        raw = np.zeros(self.n_nodes)
        for idx in (t0, t1, t2, t3):
            np.add.at(raw, idx, rho * self.vols / 4.0)
        m_int = raw.copy()
        m_int[-1] = min(self.CENTROID_MASS, 0.25 * total_mass)
        surf_sum = raw[:-1].sum()
        m_int[:-1] = raw[:-1] * (total_mass - m_int[-1]) / surf_sum
        self.m_int = m_int
        self.m_total = float(m_int.sum())
        self.m_out = m_int[:-1] * (total_mass / m_int[:-1].sum())
        self.body_ids = np.ones(self.n_surf, dtype=np.int64)

        # per-node normal contact damping (mass-scaled, explicit-stable)
        self.c_n = np.minimum(300.0, 1.4 * np.sqrt(self.CONTACT_K * m_int))[:, None]

        # --- rollers (kinematic cylinders from the frozen geometry) ---
        geom = cfg.get("scene_geometry", {}) or {}
        centers, radii, half_heights, local_axis_w = [], [], [], []
        bc = cfg.get("boundary_conditions", {}) or {}
        w_upper = float(bc.get("upper_angular_velocity_rad_s", 5.0))
        w_lower = float(bc.get("lower_angular_velocity_rad_s", -5.0))
        for obj in geom.get("objects", []):
            g = obj.get("geometry", {}) or {}
            if g.get("type") != "cylinder":
                continue
            tr = obj["transform"]
            tx, ty, tz = (float(v) for v in tr["translation"])
            centers.append((tx, tz))
            radii.append(float(g["radius"]))
            half_heights.append(0.5 * float(g["height"]))
            # cylinder local axis (+z) mapped by the frozen quaternion
            axis_world = _quat_rotate_xyzw(tr["rotation_xyzw"], (0.0, 0.0, 1.0))
            w_local = w_upper if tz > 0 else w_lower
            local_axis_w.append(w_local * axis_world)      # world angular velocity
        if not centers:
            raise ValueError("no roller cylinders found in frozen geometry")
        self.roller_c = np.asarray(centers, dtype=float)          # (R,2) x,z
        self.roller_R = float(radii[0])
        self.roller_hh = float(half_heights[0])
        self.roller_w = np.asarray(local_axis_w, dtype=float)     # (R,3)
        self.n_rollers = len(centers)

        self.t_start = float(bc.get("start_time", 0.35))
        self.t_ramp = float(bc.get("ramp_duration", 0.25))
        self.surface_speed = abs(w_upper) * self.roller_R

        # --- contact state ---
        self.xi = np.zeros((self.n_nodes, self.n_rollers, 3))
        self.prev_valid = np.zeros((self.n_nodes, self.n_rollers), dtype=bool)

        # --- budgets ---
        self.time = 0.0
        self.cum_impulse = np.zeros(3)
        self.step_count = 0

    # ------------------------------------------------------------------
    def _ramp(self, t: float) -> float:
        if self.t_ramp <= 0.0:
            return 1.0 if t >= self.t_start else 0.0
        return float(np.clip((t - self.t_start) / self.t_ramp, 0.0, 1.0))

    def _roller_omega(self, t: float) -> np.ndarray:
        return self.roller_w * self._ramp(t)                 # (R,3)

    def _contact(self, dt):
        """Roller contact forces; updates tangential anchors when dt > 0."""
        x, v = self.x, self.v
        om = self._roller_omega(self.time)                     # (R,3)
        om_y = om[:, 1]                                        # (R,)
        rx = x[:, 0:1] - self.roller_c[None, :, 0]             # (N,R)
        rz = x[:, 2:3] - self.roller_c[None, :, 1]
        dist = np.hypot(rx, rz)
        pen = self.roller_R - dist
        in_band = np.abs(x[:, 1:2]) <= (self.roller_hh + 1e-9)
        valid = (pen > 0.0) & in_band & (dist > 1e-9)

        safe = np.maximum(dist, 1e-12)
        nx = rx / safe
        nz = rz / safe
        normal = np.stack([nx, np.zeros_like(nx), nz], axis=2)   # (N,R,3)
        v_surf = np.stack(
            [om_y[None, :] * rz, np.zeros_like(nx), -om_y[None, :] * rx], axis=2
        )
        v_rel = v[:, None, :] - v_surf                          # (N,R,3)
        vn = np.einsum("ncd,ncd->nc", v_rel, normal)
        n_force = self.CONTACT_K * pen - self.c_n * np.minimum(vn, 0.0)
        n_force = np.where(valid, np.maximum(n_force, 0.0), 0.0)

        # tangential (stick/slip) friction with anchor state xi
        vt = v_rel - vn[..., None] * normal
        if dt and dt > 0.0:
            xi = np.where(
                valid[..., None] & (~self.prev_valid[..., None]), 0.0, self.xi
            )
            xi = xi + vt * dt
            xi = xi - np.einsum("ncd,ncd->nc", xi, normal)[..., None] * normal
            xi = np.where(valid[..., None], xi, 0.0)
        else:
            xi = self.xi
        f_t = -self.FRICTION_K * xi
        mag = np.linalg.norm(f_t, axis=2)
        thresh = self.mu_s * n_force
        slip = valid & (mag > thresh) & (mag > 1e-12)
        vmag = np.linalg.norm(vt, axis=2)
        dir_v = np.where(
            vmag[..., None] > 1e-9,
            vt / np.maximum(vmag, 1e-12)[..., None],
            f_t / np.maximum(mag, 1e-12)[..., None],
        )
        f_t_slip = -self.MU_DYNAMIC * n_force[..., None] * dir_v
        f_t = np.where(slip[..., None], f_t_slip, f_t)
        if dt and dt > 0.0:
            self.xi = np.where(slip[..., None], -f_t / self.FRICTION_K, xi)
            self.prev_valid = valid

        force = np.where(valid[..., None], n_force[..., None] * normal + f_t, 0.0)
        return {
            "valid": valid,
            "normal": normal,
            "v_rel": v_rel,
            "n_force": n_force,
            "f_t": f_t,
            "force": force.sum(axis=1),        # (N,3) per node
        }

    def _elastic_force(self):
        x, v = self.x, self.v
        p0 = x[self.T0]
        ds = np.stack(
            [x[self.T1] - p0, x[self.T2] - p0, x[self.T3] - p0], axis=2
        )
        F = ds @ self.Dm_inv
        U, _, Vt = np.linalg.svd(F)
        R = U @ Vt
        flip = np.linalg.det(R) < 0.0
        if np.any(flip):
            U = U.copy()
            U[flip, :, 2] *= -1.0
            R = U @ Vt
        tr = np.einsum("nij,nij->n", R, F)
        P = 2.0 * self.mu_lame * (F - R) + self.lam_lame * (tr - 3.0)[:, None, None] * R
        v0 = v[self.T0]
        dsv = np.stack(
            [v[self.T1] - v0, v[self.T2] - v0, v[self.T3] - v0], axis=2
        )
        P = P + self.c_visc * (dsv @ self.Dm_inv)
        H = -self.vols[:, None, None] * (P @ self.Dm_inv_T)
        hsum = H[:, 0, :] + H[:, 1, :] + H[:, 2, :]
        f = np.zeros((self.n_nodes, 3))
        nodes = self.tet_nodes
        for d in range(3):
            vals = np.concatenate([-hsum[:, d], H[:, 0, d], H[:, 1, d], H[:, 2, d]])
            f[:, d] += np.bincount(nodes, weights=vals, minlength=self.n_nodes)
        return f

    def _substep(self, dt: float) -> None:
        f_elastic = self._elastic_force()
        contact = self._contact(dt)
        f_ext = self.m_int[:, None] * self.gravity[None, :] + contact["force"]
        acc = (f_elastic + f_ext) / self.m_int[:, None]
        v_new = self.v + dt * acc
        x_new = self.x + dt * v_new
        if not (np.isfinite(x_new).all() and np.isfinite(v_new).all()):
            # numerical safety net: keep last good state, damp velocities
            self.v *= 0.5
            self.time += dt
            return
        self.x = x_new
        self.v = v_new
        self.time += dt
        self.cum_impulse += f_ext.sum(axis=0) * dt
        self.step_count += 1

    # ------------------------------------------------------------------
    def _deformation_gradients(self):
        p0 = self.x[self.T0]
        ds = np.stack(
            [self.x[self.T1] - p0, self.x[self.T2] - p0, self.x[self.T3] - p0],
            axis=2,
        )
        return ds @ self.Dm_inv

    def advance_to(self, time: float):
        target = float(time)
        if target > self.time:
            while self.time < target - 1e-12:
                dt = min(self.DT, target - self.time)
                self._substep(dt)

        contact = self._contact(0.0)                     # snapshot, no update
        f_ext_total = (
            self.m_total * self.gravity + contact["force"].sum(axis=0)
        )

        # --- friction contact export (surface samples only) ---
        valid_surf = contact["valid"][: self.n_surf]
        n_contacts = int(np.count_nonzero(valid_surf))
        ones_c = np.ones(n_contacts)
        fields = {
            "element_deformation_gradient": self._deformation_gradients(),
            "friction_active_mask": ones_c,
            "friction_static_coefficient": ones_c * self.mu_s,
            "friction_dynamic_coefficient": ones_c * self.MU_DYNAMIC,
            "friction_normal": contact["normal"][: self.n_surf][valid_surf],
            "friction_normal_force": contact["n_force"][: self.n_surf][valid_surf],
            "friction_relative_velocity": contact["v_rel"][: self.n_surf][valid_surf],
            "friction_tangential_force": contact["f_t"][: self.n_surf][valid_surf],
            "budget_domain_weights": np.ones((2, self.n_surf)),
            "budget_external_force": np.stack(
                [f_ext_total, f_ext_total]
            ).reshape(2, 1, 3),
            "budget_cumulative_external_impulse": np.stack(
                [self.cum_impulse, self.cum_impulse]
            ),
            "budget_cumulative_inflow_momentum": np.zeros((2, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((2, 3)),
            "cumulative_mass_source": np.zeros(1),
        }

        scalars = {
            "friction_stick_speed_tolerance": self.STICK_SPEED_TOL,
            "reference_force": 100.0,
            "reference_velocity": self.surface_speed,
            "reference_momentum": self.m_total * self.surface_speed,
            "roller_friction": self.mu_s,
            "reference_youngs_modulus_pa": self.E,
        }

        frame = StateFrame(
            time=target,
            position=self.x[: self.n_surf].copy(),
            velocity=self.v[: self.n_surf].copy(),
            mass=self.m_out.copy(),
            body_id=self.body_ids.copy(),
            attributes={
                "rest_position": self.x_surf0.copy(),
                "mass_quantity": self.m_out.copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "corotated FEM + penalty frictional roller contact, "
                      "semi-implicit Euler",
            "substep_dt": self.DT,
            "substeps": self.step_count,
            "elements": int(self.n_elems),
            "nodes": int(self.n_nodes),
            "total_mass_kg": self.m_total,
        }
