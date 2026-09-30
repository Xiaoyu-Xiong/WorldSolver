"""Rigid-fluid coupling solver for the "cube drops into water" scene.

Physical model (implemented directly with NumPy/SciPy; no external engines):

* The liquid is a three-dimensional *weakly compressible* SPH fluid:
  cubic-spline kernel, Tait equation of state for pressure and Monaghan
  artificial viscosity.  Mass is carried by Lagrangian particles, so the
  free surface, the splash and the wake emerge from the dynamics.
* The rigid cube is a genuine 6-DOF box (quaternion orientation, isotropic
  box inertia).  Its surface is discretised with Akinci-style boundary
  particles; fluid/boundary pair forces transfer momentum in both
  directions (force + torque on the box centre of mass), which reproduces
  the impact, the displaced volume and the buoyancy-driven motion.
* Non-penetration impulses between fluid particles and the moving box, plus
  spring-damper contacts between the box corners and the tank walls, keep
  the bodies inside the frozen container.
* Time integration is a symplectic (semi-implicit) Euler scheme with a
  fixed, CFL-safe time step; the sample interval is an exact multiple of the
  step so every requested sample time is hit exactly.

Everything is deterministic and produces finite states at every sample.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------
# small quaternion helpers (xyzw convention, world-frame angular velocity)
# --------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.asarray(q, dtype=float) / n


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    wq = np.array([omega[0], omega[1], omega[2], 0.0])
    dq = 0.5 * _quat_mul(wq, np.asarray(q, dtype=float)) * dt
    return _quat_normalize(np.asarray(q, dtype=float) + dq)


# --------------------------------------------------------------------------
# cubic-spline SPH kernel (support 2h) and its radial derivative
# --------------------------------------------------------------------------
def _kernel(r: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    q = r / h
    sigma = 1.0 / (4.0 * np.pi * h * h * h)
    W = np.zeros_like(r)
    dW = np.zeros_like(r)
    m1 = q < 1.0
    if np.any(m1):
        qq = q[m1]
        W[m1] = sigma * ((2.0 - qq) ** 3 - 4.0 * (1.0 - qq) ** 3)
        dW[m1] = sigma * (-3.0 * (2.0 - qq) ** 2 + 12.0 * (1.0 - qq) ** 2) / h
    m2 = (q >= 1.0) & (q < 2.0)
    if np.any(m2):
        qq = q[m2]
        W[m2] = sigma * (2.0 - qq) ** 3
        dW[m2] = sigma * (-3.0 * (2.0 - qq) ** 2) / h
    return W, dW


class BenchmarkSolver:
    """SPH fluid + 6-DOF rigid box with two-way momentum coupling."""

    # -- construction ------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        init = scene.initial_state()
        pos = np.asarray(init.position, dtype=float)
        vel = np.asarray(init.velocity, dtype=float)
        mass = np.asarray(init.mass, dtype=float)
        body = np.asarray(init.body_id)
        orient = np.asarray(init.orientation, dtype=float)

        water = body == 1
        cube = body == 2
        self.xw = pos[water].copy()
        self.vw = vel[water].copy()
        self.mw = mass[water].copy()
        self.nw = int(len(self.xw))

        self.mc = float(mass[cube][0])
        self.xc = pos[cube][0].copy()
        self.vc = vel[cube][0].copy()
        self.qc = _quat_normalize(orient[cube][0])
        self.wc = np.zeros(3)

        # frozen material / discretisation parameters
        self.rho0 = float(cfg["initial_conditions"]["water"]["rest_density"])
        self.dp = float((float(self.mw.mean()) / self.rho0) ** (1.0 / 3.0))
        self.h = self.dp
        self.r_int = 1.98 * self.h
        self.cs = 25.0
        self.B = self.rho0 * self.cs * self.cs / 7.0
        self.alpha = 0.10
        self.v_max = 30.0

        self.side = float(cfg["initial_conditions"]["cube"]["side_length"])
        self.Ic = self.mc * self.side * self.side / 6.0

        # rigid box surface samples (body frame)
        n = max(2, int(round(self.side / self.dp)))
        s = self.side / n
        offs = -0.5 * self.side + (np.arange(n) + 0.5) * s
        pts = []
        for axis in range(3):
            u, v = [k for k in range(3) if k != axis]
            g0, g1 = np.meshgrid(offs, offs, indexing="ij")
            for sign in (-1.0, 1.0):
                p = np.zeros((n * n, 3))
                p[:, axis] = sign * 0.5 * self.side
                p[:, u] = g0.ravel()
                p[:, v] = g1.ravel()
                pts.append(p)
        self.xb0 = np.vstack(pts)
        self.nb = int(len(self.xb0))
        self.mb = self.rho0 * s * s * self.dp
        self.m_all = np.concatenate((self.mw, np.full(self.nb, self.mb)))

        # container: interior planes (inward normal = nsign * e_axis)
        self.walls = self._container_planes(cfg)
        self.d0w = 0.5 * self.dp
        self.omega_w = 200.0
        self.zeta_w = 0.7
        self.c_w = 2.0 * self.zeta_w * self.omega_w

        # rigid box / wall contact parameters
        a = 0.5 * self.side
        self.corners = np.array(
            [[sx * a, sy * a, sz * a] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        self.d0c = 0.01
        self.omega_c = 60.0
        self.k_c = self.mc * self.omega_c * self.omega_c
        self.c_c = 2.0 * 0.7 * self.mc * self.omega_c
        self.mu_c = 0.35

        # time stepping
        self.dt = 5.0e-4
        self.step = 0

        # accumulators for per-sample averaged source terms
        self.acc_ext_w = np.zeros((self.nw, 3))
        self.acc_ext_c = np.zeros(3)
        self.acc_int_c = np.zeros(3)
        self.mean_ext_w = self.mw[:, None] * self.g[None, :]
        self.mean_ext_c = self.mc * self.g
        self.mean_int_c = np.zeros(3)

        self.xb = self.xb0 @ _quat_to_matrix(self.qc).T + self.xc
        self._diagnostics = {
            "method": "WCSPH + 6-DOF rigid box, boundary-particle coupling",
            "particles": self.nw,
            "boundary_samples": self.nb,
            "dt": self.dt,
            "h": self.h,
            "sound_speed": self.cs,
        }

    # -- frozen container geometry ----------------------------------------
    def _container_planes(self, cfg: dict):
        planes = []
        geo = cfg.get("scene_geometry") if isinstance(cfg, dict) else None
        objects = None
        if isinstance(geo, dict):
            objects = geo.get("objects")
        if isinstance(objects, list):
            for obj in objects:
                try:
                    if obj.get("semantic") != "boundary":
                        continue
                    g = obj.get("geometry", {})
                    if g.get("type") != "box":
                        continue
                    he = np.asarray(g["half_extents"], dtype=float)
                    tr = np.asarray(obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0]), dtype=float)
                    axis = int(np.argmin(he))
                    thickness = float(he[axis])
                    center = float(tr[axis])
                    nsign = -1.0 if center > 0.0 else 1.0
                    plane = center + nsign * thickness
                    planes.append((axis, plane, nsign))
                except Exception:
                    continue
        if not planes:
            lo = np.asarray(self.cfg.get("domain", {}).get("bounds_min", [-1.5, -1.2, 0.0]), dtype=float)
            hi = np.asarray(self.cfg.get("domain", {}).get("bounds_max", [1.5, 1.2, 2.5]), dtype=float)
            planes = [
                (2, float(lo[2]), 1.0),
                (0, float(lo[0]), 1.0),
                (0, float(hi[0]), -1.0),
                (1, float(lo[1]), 1.0),
                (1, float(hi[1]), -1.0),
            ]
        return planes

    # -- oriented box signed distance -------------------------------------
    def _box_sdf(self, pts: np.ndarray):
        R = _quat_to_matrix(self.qc)
        loc = (pts - self.xc) @ R
        a = 0.5 * self.side
        q = np.abs(loc) - a
        pos = np.maximum(q, 0.0)
        out = np.sqrt(np.einsum("ij,ij->i", pos, pos))
        inside = np.minimum(np.max(q, axis=1), 0.0)
        d = out + inside
        sgn = np.where(loc >= 0.0, 1.0, -1.0)
        nb = np.zeros_like(loc)
        outside = out > 1e-12
        if np.any(outside):
            nb[outside] = (pos[outside] * sgn[outside]) / out[outside, None]
        ins = ~outside
        if np.any(ins):
            idx = np.argmax(q[ins], axis=1)
            nrm = np.zeros((int(np.sum(ins)), 3))
            nrm[np.arange(len(idx)), idx] = 1.0
            nb[ins] = nrm * sgn[ins]
        n = nb @ R.T
        return d, n

    # -- one physical time step -------------------------------------------
    def _substep(self, dt: float) -> None:
        nw, nb = self.nw, self.nb
        g = self.g
        rho0 = self.rho0
        B = self.B
        alpha = self.alpha
        h = self.h

        R = _quat_to_matrix(self.qc)
        rb = self.xb0 @ R.T
        xb = rb + self.xc
        vb = self.vc + np.cross(self.wc, rb)
        self.xb = xb

        pts = np.vstack((self.xw, xb))
        vel = np.vstack((self.vw, vb))
        m_all = self.m_all

        tree = cKDTree(pts)
        pairs = tree.query_pairs(self.r_int, output_type="ndarray")

        acc = np.zeros((nw, 3))
        F_cube = np.zeros(3)
        tau_cube = np.zeros(3)

        if len(pairs):
            ii = pairs[:, 0]
            jj = pairs[:, 1]
            keep = ii < nw
            ii = ii[keep]
            jj = jj[keep]

            dr = pts[ii] - pts[jj]
            r2 = np.maximum(np.einsum("ij,ij->i", dr, dr), 1e-20)
            r = np.sqrt(r2)
            W, dWdr = _kernel(r, h)
            mi = m_all[ii]
            mj = m_all[jj]

            # density (all neighbours for fluid, fluid neighbours for boundary)
            rho = np.bincount(ii, weights=mj * W, minlength=nw + nb)
            mf = jj < nw
            if np.any(mf):
                rho[:nw] += np.bincount(jj[mf], weights=(mi * W)[mf], minlength=nw)
            # bounded density keeps the Tait EOS stiff but numerically tame
            # during the violent impact (standard WCSPH regularisation)
            rho_w = np.clip(rho[:nw], 0.5 * rho0, 1.5 * rho0)
            rho_b = np.clip(rho[nw:], rho0, 1.5 * rho0)

            p_w = np.maximum(B * ((rho_w / rho0) ** 7 - 1.0), 0.0)
            p_b = np.maximum(B * ((rho_b / rho0) ** 7 - 1.0), 0.0)
            p_all = np.concatenate((p_w, p_b))
            r_all = np.concatenate((rho_w, rho_b))

            # artificial (Monaghan) viscosity
            dv = vel[ii] - vel[jj]
            vdr = np.einsum("ij,ij->i", dv, dr)
            mu = h * vdr / (r2 + 0.01 * h * h)
            Pi = np.where(vdr < 0.0, -alpha * self.cs * mu / (0.5 * (r_all[ii] + r_all[jj])), 0.0)

            coef = p_all[ii] / (r_all[ii] ** 2) + p_all[jj] / (r_all[jj] ** 2) + Pi
            gcoef = dWdr / r

            # force on the fluid particle
            apair = -(mj * coef * gcoef)[:, None] * dr
            for k in range(3):
                acc[:, k] = np.bincount(ii, weights=apair[:, k], minlength=nw)

            # reaction on the rigid box (equal and opposite pair forces)
            mb = jj >= nw
            if np.any(mb):
                jb = jj[mb] - nw
                fb = (mi[mb] * mj[mb] * coef[mb] * gcoef[mb])[:, None] * dr[mb]
                rb_c = rb[jb]
                tb = np.cross(rb_c, fb)
                for k in range(3):
                    F_cube[k] = float(np.bincount(jb, weights=fb[:, k], minlength=nb).sum())
                    tau_cube[k] = float(np.bincount(jb, weights=tb[:, k], minlength=nb).sum())

        acc += g[None, :]

        # ---- container wall forces on the fluid -------------------------
        ext_w = self.mw[:, None] * g[None, :]
        for axis, plane, nsign in self.walls:
            d = nsign * (self.xw[:, axis] - plane)
            delta = self.d0w - d
            m = delta > 0.0
            if not np.any(m):
                continue
            vn = nsign * self.vw[m, axis]
            f = self.mw[m] * self.omega_w * self.omega_w * delta[m]
            f = f - self.mw[m] * self.c_w * np.minimum(vn, 0.0)
            acc[m, axis] += nsign * f / self.mw[m]
            ext_w[m, axis] += nsign * f

        # ---- fluid integration ------------------------------------------
        self.vw += acc * dt
        sp = np.linalg.norm(self.vw, axis=1)
        fast = sp > self.v_max
        if np.any(fast):
            self.vw[fast] *= (self.v_max / sp[fast])[:, None]
        self.xw += self.vw * dt

        # ---- hard containment (impulse accounted as external force) ------
        for axis, plane, nsign in self.walls:
            d = nsign * (self.xw[:, axis] - plane)
            bad = d < 0.0
            if not np.any(bad):
                continue
            self.xw[bad, axis] = plane
            vn = nsign * self.vw[bad, axis]
            mv = vn < 0.0
            if np.any(mv):
                idx = np.flatnonzero(bad)[mv]
                dvn = -vn[mv]
                self.vw[idx, axis] += nsign * dvn
                ext_w[idx, axis] += self.mw[idx] * dvn / dt

        # ---- fluid / box non-penetration impulses -----------------------
        d_sdf, n_sdf = self._box_sdf(self.xw)
        pen = d_sdf < 0.0
        if np.any(pen):
            idx = np.flatnonzero(pen)
            nn = n_sdf[idx]
            self.xw[idx] += nn * ((-d_sdf[idx])[:, None] + 0.02 * self.dp)
            r_contact = self.xw[idx] - self.xc
            v_surf = self.vc[None, :] + np.cross(self.wc[None, :], r_contact)
            v_rel = self.vw[idx] - v_surf
            vn = np.einsum("ij,ij->i", v_rel, nn)
            appr = vn < 0.0
            if np.any(appr):
                ia = idx[appr]
                na = nn[appr]
                dv = (-vn[appr])[:, None] * na
                self.vw[ia] += dv
                # equal and opposite impulse, applied to the box through the
                # rigid-body force/torque accumulators (integrated below)
                jpart = -self.mw[ia, None] * dv          # impulse on the box
                rc = self.xw[ia] - self.xc
                F_cube += jpart.sum(axis=0) / dt
                tau_cube += np.cross(rc, jpart).sum(axis=0) / dt

        # ---- rigid box contacts with the container walls -----------------
        corner_world = self.corners @ R.T
        corner_pos = corner_world + self.xc
        corner_vel = self.vc[None, :] + np.cross(np.broadcast_to(self.wc, (8, 3)), corner_world)
        F_wall_cube = np.zeros(3)
        tau_wall_cube = np.zeros(3)
        for axis, plane, nsign in self.walls:
            d = nsign * (corner_pos[:, axis] - plane)
            delta = self.d0c - d
            act = delta > 0.0
            if not np.any(act):
                continue
            vn = nsign * corner_vel[:, axis]
            fn = self.k_c * delta - self.c_c * np.minimum(vn, 0.0)
            fn = np.maximum(fn, 0.0) * act
            fvec = np.zeros((8, 3))
            fvec[:, axis] = nsign * fn
            nvec = np.zeros(3)
            nvec[axis] = nsign
            vtan = corner_vel - np.outer(vn, nvec)
            vt = np.linalg.norm(vtan, axis=1)
            ft = -self.mu_c * np.where(act, fn, 0.0)[:, None] * vtan / (vt[:, None] + 1e-9)
            ft[~act] = 0.0
            ftot = fvec + ft
            F_wall_cube += ftot.sum(axis=0)
            tau_wall_cube += np.cross(corner_world, ftot).sum(axis=0)

        # ---- rigid box integration --------------------------------------
        Ftot = self.mc * g + F_cube + F_wall_cube
        tautot = tau_cube + tau_wall_cube
        self.vc += (Ftot / self.mc) * dt
        self.wc += (tautot / self.Ic) * dt
        self.xc += self.vc * dt
        self.qc = _quat_integrate(self.qc, self.wc, dt)

        # ---- per-sample source-term accumulation -------------------------
        self.acc_ext_w += ext_w * dt
        self.acc_ext_c += (self.mc * g + F_wall_cube) * dt
        self.acc_int_c += F_cube * dt

    # -- frame assembly ----------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        n_target = int(round(t / self.dt))
        n_sub = 0
        if n_target > self.step:
            self.acc_ext_w[:] = 0.0
            self.acc_ext_c[:] = 0.0
            self.acc_int_c[:] = 0.0
            while self.step < n_target:
                self._substep(self.dt)
                self.step += 1
                n_sub += 1
        if n_sub > 0:
            inv = 1.0 / (n_sub * self.dt)
            self.mean_ext_w = self.acc_ext_w * inv
            self.mean_ext_c = self.acc_ext_c * inv
            self.mean_int_c = self.acc_int_c * inv
        else:
            self.mean_ext_w = self.mw[:, None] * self.g[None, :]
            self.mean_ext_c = self.mc * self.g
            self.mean_int_c = np.zeros(3)
        return self._make_frame(t)

    def _make_frame(self, t: float) -> StateFrame:
        nw = self.nw
        position = np.vstack((self.xw, self.xc[None, :]))
        velocity = np.vstack((self.vw, self.vc[None, :]))
        mass = np.concatenate((self.mw, [self.mc]))
        body_id = np.concatenate((np.ones(nw, dtype=np.int32), np.array([2], dtype=np.int32)))

        orientation = np.zeros((nw + 1, 4))
        orientation[:, 3] = 1.0
        orientation[-1] = self.qc
        angular_velocity = np.zeros((nw + 1, 3))
        angular_velocity[-1] = self.wc

        radius = np.full(nw + 1, 0.5 * self.dp)

        volume = mass / self.rho0
        volume[-1] = self.side ** 3

        inertia = np.zeros((nw + 1, 3, 3))
        inertia[-1, 0, 0] = inertia[-1, 1, 1] = inertia[-1, 2, 2] = self.Ic

        external_force = np.vstack((self.mean_ext_w, self.mean_ext_c[None, :]))
        interaction_force = np.array([self.mean_int_c, -self.mean_int_c])

        # contact geometry samples: box corners, nearest fluid/box gaps, wall gaps
        R = _quat_to_matrix(self.qc)
        corner_pos = self.corners @ R.T + self.xc
        corner_gap = np.full(8, np.inf)
        for axis, plane, nsign in self.walls:
            corner_gap = np.minimum(corner_gap, nsign * (corner_pos[:, axis] - plane))
        d_sdf, _ = self._box_sdf(self.xw)
        k = min(32, len(d_sdf))
        nearest_box = np.sort(d_sdf)[:k]
        wall_gap = np.full(nw, np.inf)
        for axis, plane, nsign in self.walls:
            wall_gap = np.minimum(wall_gap, nsign * (self.xw[:, axis] - plane))
        k2 = min(16, nw)
        nearest_wall = np.sort(wall_gap)[:k2]
        contact = np.concatenate((corner_gap, nearest_box, nearest_wall))

        fields = {
            "external_force": external_force,
            "interaction_force": interaction_force,
            "contact_signed_distance": contact,
            "cumulative_mass_source": np.zeros(2),
            "gravity_acceleration": self.g[None, :].copy(),
        }
        attributes = {
            "phase": np.concatenate((np.zeros(nw), np.ones(1))),
            "mass_quantity": mass,
            "volume_quantity": volume,
            "inertia_tensor": inertia,
            "particle_radius": radius,
        }
        scalars = {
            "contact_length_scale": float(self.dp),
            "grid_spacing": float(self.dp),
            "characteristic_length": float(self.dp),
            "reference_velocity": float(np.sqrt(2.0 * abs(float(self.g[2])) * 0.5)),
            "dynamic_viscosity": float(self.alpha * self.rho0 * self.cs * self.dp / 8.0),
        }
        frame = StateFrame(
            time=t,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -- diagnostics -------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diagnostics)
        out.update(
            {
                "steps": int(self.step),
                "water_particles": int(self.nw),
                "final_cube_position": [float(v) for v in self.xc],
                "final_cube_speed": float(np.linalg.norm(self.vc)),
                "max_water_speed": float(np.linalg.norm(self.vw, axis=1).max()) if self.nw else 0.0,
            }
        )
        return out
