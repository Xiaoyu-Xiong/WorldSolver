"""Physically-based cloth sleeve on two articulated kinematic capsules.

Method
------
* The 17x17 grid is wrapped into a closed tubular sleeve (row 0 welded to the
  last row) around the common capsule axis, so the initial state is a fitted,
  relaxed sleeve surrounding the aligned limbs.
* Thin-sheet mechanics: lumped-mass spring network (structural + shear +
  two-neighbor bending springs) with an asymmetric stiffness (cloth offers
  little in-plane compression resistance, so it buckles into folds),
  integrated with semi-implicit Euler over 60 substeps per sample interval.
* Contact: analytic signed distances to the ground plane and to the two
  kinematic capsules, penalty normal force with impulse clamping for
  unconditional substep stability, plus Coulomb friction against the limb
  surface velocity so the sleeve can slide.
* Articulation: the lower limb rotates about the shared joint at the origin
  with a smoothstep schedule (bends downward ~63 deg between t=1.0 and 4.5s),
  bringing sleeve regions near the joint into compression; the sleeve slides
  and develops localized folds, then settles.

All required StateFrame fields are exported per sample.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

GRAVITY = (0.0, 0.0, -9.81)

STRETCH_K = 900.0        # tension stiffness (N/m)
COMP_RATIO = 0.05        # compression stiffness ratio (buckling-friendly)
SHEAR_K = 250.0
BEND_K = 12.0
DAMP_EDGE = 0.9
DAMP_BEND = 0.02
AIR_DRAG = 0.5           # velocity decay rate (1/s)

FRICTION_MU = 0.35
CONTACT_STIFF = 8.0e3
CONTACT_DAMP = 40.0
EPS_CONTACT = 0.02
SD_SENTINEL = 0.1

SUBSTEPS = 60
MAX_TENSION = 0.30
MAX_COMPRESSION = 0.03

JOINT = np.array([0.0, 0.0, 1.0])
BEND_START, BEND_END = 1.0, 4.5
BEND_TOTAL = 1.1         # rad about +y (tip of lower limb swings downward)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def smoothstep(t0, t1, t):
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    ds = 2.0 * s * (1.0 - s) / (t1 - t0) if t0 < t < t1 else 0.0
    return s * s * (3.0 - 2.0 * s), ds


class Capsule:
    """Kinematic capsule; total tip-to-tip height h, rotating about JOINT."""

    def __init__(self, radius, height, center, quat):
        self.r = float(radius)
        self.half = max(0.5 * float(height) - self.r, 1e-6)
        self.ctr0 = np.asarray(center, dtype=float)
        self.R0 = quat_to_mat(np.asarray(quat, dtype=float))
        self.ctr = self.ctr0.copy()
        self.R = self.R0.copy()
        self.omega = np.zeros(3)

    def pose_at(self, angle, omega_y=0.0):
        c, s = np.cos(angle), np.sin(angle)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        self.R = Ry @ self.R0
        self.ctr = JOINT + Ry @ (self.ctr0 - JOINT)
        self.omega = np.array([0.0, omega_y, 0.0])


class BenchmarkSolver:
    # ------------------------------------------------------------ initialize
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.dt_sample = float(cfg["sample_dt"])
        self.g = np.asarray(cfg.get("gravity", GRAVITY), dtype=float)

        geom = cfg.get("scene_geometry", {})
        objs = {o["id"]: o for o in geom.get("objects", [])}

        def capsule_of(oid, r, h, c):
            o = objs.get(oid, {})
            gg = o.get("geometry", {})
            tt = o.get("transform", {})
            return Capsule(gg.get("radius", r), gg.get("height", h),
                           tt.get("translation", c),
                           tt.get("rotation_xyzw", [0.0, 0.7071068, 0.0, 0.7071068]))

        self.limb_upper = capsule_of("limb_upper", 0.34, 1.3, [-0.55, 0.0, 1.0])
        self.limb_lower = capsule_of("limb_lower", 0.31, 1.3, [0.55, 0.0, 1.0])
        self.limb_upper.pose_at(0.0, 0.0)
        self.limb_lower.pose_at(0.0, 0.0)

        go = objs.get("ground", {})
        gtr = go.get("transform", {}).get("translation", [0.0, 0.0, -0.05])
        ghe = go.get("geometry", {}).get("half_extents", [2.4, 2.0, 0.05])
        self.ground_z = float(gtr[2]) + float(ghe[2])

        frame0 = scene.initial_state()
        self.N = len(frame0.position)
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        nx, ny = (int(v) for v in cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        self.thickness = float(cloth.get("thickness", 0.015))

        # ---- wrap the flat grid into a closed tubular sleeve on the limb axis
        sx, sy = (float(v) for v in cloth["size"])
        cx = float(cloth["center"][0])
        axis_z = float(self.limb_upper.ctr0[2])
        R = sy / (2.0 * np.pi)
        axis_x = np.linspace(-sx * .5, sx * .5, nx) + cx
        theta = np.linspace(0.0, 2.0 * np.pi, ny)
        rows = [np.column_stack((axis_x,
                                 R * np.cos(t) * np.ones(nx),
                                 axis_z + R * np.sin(t) * np.ones(nx)))
                for t in theta]
        pos = np.vstack(rows)
        pos[self.N - nx:] = pos[:nx]           # weld seam onto row 0
        self.seam_idx = np.arange(self.N - nx, self.N)
        self.rest = pos.copy()

        # dynamic mass with seam rows merged into row 0
        self.m_dyn = self.mass.copy()
        self.m_dyn[:nx] += self.mass[self.seam_idx]

        # ---- topology (index = row*nx + col; row=y-wrap index, col=x)
        v = lambda a, b: a * nx + b
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
        i, j = i.ravel(), j.ravel()
        f1 = np.column_stack((v(i, j), v(i + 1, j), v(i + 1, j + 1)))
        f2 = np.column_stack((v(i, j), v(i + 1, j + 1), v(i, j + 1)))
        self.faces = np.vstack((f1, f2))

        def edges(i0, j0, i1, j1):
            return np.column_stack((v(i0, j0), v(i1, j1)))

        self.e_struct = np.unique(np.vstack((
            edges(i, j, i + 1, j), edges(i, j, i, j + 1),
            edges(i, j, i + 1, j + 1), edges(i + 1, j, i, j + 1),
        )), axis=0)
        ib, jb = np.meshgrid(np.arange(nx - 2), np.arange(ny - 2), indexing="ij")
        ib, jb = ib.ravel(), jb.ravel()
        self.e_bend = np.unique(np.vstack((
            edges(ib, jb, ib + 2, jb), edges(ib, jb, ib, jb + 2),
            edges(ib, jb, ib + 1, jb + 2), edges(ib, jb, ib + 2, jb + 1),
            edges(ib + 1, jb, ib + 1, jb + 2), edges(ib, jb + 1, ib + 2, jb + 1),
        )), axis=0)

        d0 = self.rest[self.e_struct[:, 0]] - self.rest[self.e_struct[:, 1]]
        self.l0_struct = np.linalg.norm(d0, axis=1)
        d0 = self.rest[self.e_bend[:, 0]] - self.rest[self.e_bend[:, 1]]
        self.l0_bend = np.linalg.norm(d0, axis=1)

        # lumped rest areas
        self.varea = np.zeros(self.N)
        p0, p1, p2 = (self.rest[self.faces[:, k]] for k in range(3))
        fa = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        for k in range(3):
            np.add.at(self.varea, self.faces[:, k], fa / 3.0)
        self.area_norm = max(self.varea.mean(), 1e-9)

        # rest material frames for deformation gradients
        d1 = self.rest[self.faces[:, 1]] - self.rest[self.faces[:, 0]]
        d2 = self.rest[self.faces[:, 2]] - self.rest[self.faces[:, 0]]
        e1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
        e2 = d2 - np.einsum("fi,fi->f", d2, e1)[:, None] * e1
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
        self.e1, self.e2 = e1, e2
        L11 = np.einsum("fi,fi->f", d1, e1)
        L21 = np.einsum("fi,fi->f", d2, e1)
        L22 = np.einsum("fi,fi->f", d2, e2)
        z = 0.0 * L11
        Dm = np.stack([np.stack([L11, L21], axis=1),
                       np.stack([z, L22], axis=1)], axis=1)   # [F,2,2]
        self.Dm_inv = np.linalg.inv(Dm)

        self.x = pos.copy()
        self.vel = np.zeros_like(pos)
        self.t = 0.0
        self._update_limbs(0.0)

    # ------------------------------------------------------- limb kinematics
    def _update_limbs(self, t):
        s, ds = smoothstep(BEND_START, BEND_END, t)
        angle = BEND_TOTAL * s
        self.limb_lower.pose_at(angle, BEND_TOTAL * ds)
        self.limb_upper.pose_at(0.0, 0.0)

    # --------------------------------------------------------------- forces
    def _edge_forces(self, f, edges, l0, k, kdamp, asymmetric):
        p0, p1 = self.x[edges[:, 0]], self.x[edges[:, 1]]
        v0, v1 = self.vel[edges[:, 0]], self.vel[edges[:, 1]]
        d = p1 - p0
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n = d / L[:, None]
        strain = (L - l0) / l0
        if asymmetric:
            strain = np.clip(strain, -MAX_COMPRESSION, MAX_TENSION)
            kk = np.where(strain >= 0.0, k, k * COMP_RATIO)
        else:
            strain = np.clip(strain, -MAX_TENSION, MAX_TENSION)
            kk = k
        dL = np.einsum("ei,ei->e", v1 - v0, n)
        fm = (kk * strain * l0 + kdamp * dL)[:, None] * n
        np.add.at(f, edges[:, 0], fm)
        np.add.at(f, edges[:, 1], -fm)

    def _compute_forces(self):
        f = self.mass[:, None] * self.g[None, :]
        f = np.ascontiguousarray(f)
        self._edge_forces(f, self.e_struct, self.l0_struct, STRETCH_K, DAMP_EDGE, True)
        self._edge_forces(f, self.e_bend, self.l0_bend, BEND_K, DAMP_BEND, False)
        return f

    # ------------------------------------------------------------- collision
    def _apply_contact(self, f, dt):
        pad = 0.5 * self.thickness
        # ground plane (normal +z)
        pen = (self.ground_z + pad) - self.x[:, 2]
        idx = np.where(pen > -EPS_CONTACT)[0]
        if idx.size:
            d = np.clip(pen[idx], -EPS_CONTACT, EPS_CONTACT)
            vn = self.vel[idx, 2]
            fn = np.maximum(CONTACT_STIFF * d - CONTACT_DAMP * vn, 0.0)
            fn *= self.varea[idx] / self.area_norm
            fn = np.minimum(fn, self.m_dyn[idx] * (np.abs(vn) + 0.5) / dt)
            f[idx, 2] += fn
            vt = self.vel[idx, :2]
            vtn = np.linalg.norm(vt, axis=1) + 1e-9
            ft = -FRICTION_MU * fn[:, None] * vt / vtn[:, None]
            cap = (self.m_dyn[idx] * vtn / dt)[:, None]
            ft = np.clip(ft, -cap, cap)
            f[idx, :2] += ft

        for cap in (self.limb_upper, self.limb_lower):
            r_eff = cap.r + pad
            local = (self.x - cap.ctr) @ cap.R
            a = np.clip(local[:, 0], -cap.half, cap.half)
            axis_pt = cap.ctr + (np.column_stack((a, np.zeros(self.N), np.zeros(self.N))) @ cap.R.T)
            diff = self.x - axis_pt
            rho = np.linalg.norm(diff, axis=1)
            sd = rho - r_eff
            idx = np.where(sd < EPS_CONTACT)[0]
            if not idx.size:
                continue
            n = diff[idx] / np.maximum(rho[idx, None], 1e-12)
            d = np.clip(-sd[idx], -EPS_CONTACT, EPS_CONTACT)
            v_surf = np.cross(cap.omega[None, :], axis_pt[idx] - JOINT[None, :])
            vrel = self.vel[idx] - v_surf
            vn = np.einsum("ci,ci->c", vrel, n)
            fn = np.maximum(CONTACT_STIFF * d - CONTACT_DAMP * vn, 0.0)
            fn *= self.varea[idx] / self.area_norm
            fn = np.minimum(fn, self.m_dyn[idx] * (np.abs(vn) + 0.5) / dt)
            vt = vrel - vn[:, None] * n
            vtn = np.linalg.norm(vt, axis=1) + 1e-9
            ft = -FRICTION_MU * fn[:, None] * vt / vtn[:, None]
            fcap = (self.m_dyn[idx] * vtn / dt)[:, None]
            ft = np.clip(ft, -fcap, fcap)
            f[idx] += fn[:, None] * n + ft

    # -------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        time = float(time)
        if time <= self.t:
            return self._make_frame(self.t)
        dt = (time - self.t) / SUBSTEPS
        for s in range(SUBSTEPS):
            self._update_limbs(self.t + s * dt)
            f = self._compute_forces()
            self._apply_contact(f, dt)
            # seam weld: merge seam forces into row 0, mirror afterwards
            nx = self.nx
            f[:nx] += f[self.seam_idx]
            f[self.seam_idx] = 0.0
            self.vel += dt * f / self.m_dyn[:, None]
            self.vel *= (1.0 - AIR_DRAG * dt)
            self.x += dt * self.vel
            self.x[self.seam_idx] = self.x[:nx]
            self.vel[self.seam_idx] = self.vel[:nx]
        self.t = time
        self._update_limbs(self.t)
        return self._make_frame(self.t)

    # ---------------------------------------------------------------- outputs
    def _face_gradients(self):
        q0 = self.x[self.faces[:, 0]]
        d1 = self.x[self.faces[:, 1]] - q0
        d2 = self.x[self.faces[:, 2]] - q0
        c11 = np.einsum("fi,fi->f", d1, self.e1)
        c12 = np.einsum("fi,fi->f", d1, self.e2)
        c21 = np.einsum("fi,fi->f", d2, self.e1)
        c22 = np.einsum("fi,fi->f", d2, self.e2)
        Ds = np.stack([np.stack([c11, c21], axis=1),
                       np.stack([c12, c22], axis=1)], axis=1)
        return Ds @ self.Dm_inv

    def _vertex_gradients(self):
        F = self._face_gradients()
        Fv = np.zeros((self.N, 2, 2))
        cnt = np.zeros(self.N)
        for k in range(3):
            np.add.at(Fv, self.faces[:, k], F)
            np.add.at(cnt, self.faces[:, k], 1.0)
        return Fv / np.maximum(cnt[:, None, None], 1.0)

    def _contact_signed_distance(self):
        pad = 0.5 * self.thickness
        best = self.x[:, 2] - (self.ground_z + pad)
        for cap in (self.limb_upper, self.limb_lower):
            local = (self.x - cap.ctr) @ cap.R
            a = np.clip(local[:, 0], -cap.half, cap.half)
            axis_pt = cap.ctr + (np.column_stack((a, np.zeros(self.N), np.zeros(self.N))) @ cap.R.T)
            rho = np.linalg.norm(self.x - axis_pt, axis=1) - (cap.r + pad)
            best = np.minimum(best, rho)
        return np.minimum(best, SD_SENTINEL)

    def _make_frame(self, t):
        ext = self.mass[:, None] * self.g[None, :]
        return StateFrame(
            float(t),
            self.x.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": np.zeros(self.N, dtype=np.int32),
                "deformation_gradient": self._vertex_gradients().astype(np.float64),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": np.ascontiguousarray(ext, dtype=np.float64),
                "contact_signed_distance": self._contact_signed_distance().astype(np.float64),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )

    def diagnostics(self):
        speeds = np.linalg.norm(self.vel, axis=1)
        return {
            "solver": "mass_spring_cloth_sleeve",
            "n_vertices": int(self.N),
            "n_faces": int(len(self.faces)),
            "substeps_per_sample": SUBSTEPS,
            "max_speed": float(speeds.max()) if self.N else 0.0,
            "min_signed_distance": float(self._contact_signed_distance().min()),
        }
