"""Frictional mechanism jamming solver.

Self-contained planar (x-z) rigid-body dynamics for the frozen scene:

  * 8 identical L-shaped links (base slab + upstanding tab, exactly the union of
    the two boxes in ``assets/jam_link.obj``) interlaced inside a rigid channel
    formed by ``guide_lower`` / ``guide_upper`` (plus the static floor).
  * Contacts are detected as boundary-sample-point versus oriented-box
    penetrations, giving signed gaps and outward normals.
  * Velocity-level contact response: projected Jacobi (relaxed PGS) solve with
    non-negative normal impulses, Coulomb friction cones (mu from config),
    Newtonian restitution on impacting contacts and Baumgarte penetration
    stabilisation with a bounded push-out speed.
  * ``link_0`` is the driver: a force-limited prescribed-velocity actuator pushes
    on its tab, so once the assembly wedges between the guides the driver stalls
    and the mechanism locks (frictional jamming).

Integration is semi-implicit Euler at a fixed sub-step; every quantity is
deterministic (no random numbers) and finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# link geometry (local frame, body origin at the top-rear corner of the slab)
# rect = (cx, cz, hx, hz)
LINK_RECTS = ((0.0, -0.08, 0.24, 0.08), (0.16, 0.15, 0.08, 0.15))
_RECT_SAMPLES = ((7, 3), (3, 7))

# static geometry from scene_geometry.yaml, projected on the x-z plane
STATIC_RECTS = (
    (0.0, -0.05, 3.2, 0.05),   # 0 floor
    (0.0, 0.12, 2.25, 0.08),   # 1 guide_lower
    (0.0, 1.02, 2.25, 0.08),   # 2 guide_upper
)

DT = 1.0e-3
SOLVER_ITERS = 48
RELAX = 0.35
BETA = 0.25
SLOP = 1.5e-3
MAX_PUSHOUT = 0.50
REST_THRESHOLD = 0.30
CONTACT_MARGIN = 4.0e-3
MAX_CONTACTS = 512
LIN_DAMPING = 0.15
ANG_DAMPING = 0.20
V_CLAMP = 10.0
W_CLAMP = 20.0

PRERELAX_ITERS = 600
PRERELAX_ALPHA = 0.35

# driver actuator
DRIVE_START = 0.80
DRIVE_RAMP = 0.50
DRIVE_SPEED = 0.25
DRIVE_GAIN = 320.0
DRIVE_FMAX = 50.0
DRIVE_POINT = (0.16, 0.25)   # local application point on the driver tab


def _boundary_points(cx, cz, hx, hz, nx, nz):
    xs = np.linspace(cx - hx, cx + hx, nx)
    zs = np.linspace(cz - hz, cz + hz, nz)
    pts = []
    for x in xs:
        for z in zs:
            if abs(x - cx) > hx - 1e-9 or abs(z - cz) > hz - 1e-9:
                pts.append((x, z))
    return pts


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        frame0 = scene.initial_state()

        self.n = int(len(frame0.position))
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.length_scale = float(cfg["contact"]["length_scale"])
        self.mu = float(cfg["contact"]["friction"])
        self.restitution = float(cfg["contact"]["restitution"])
        g = np.asarray(cfg["gravity"], dtype=float)
        self.gravity = np.array([g[0], g[2]], dtype=float)   # planar (x, z)

        # ---- link mass properties (uniform density over the L cross-section)
        rects = np.asarray(LINK_RECTS, dtype=float)
        areas = 4.0 * rects[:, 2] * rects[:, 3]
        total_area = float(areas.sum())
        com = (rects[:, :2] * areas[:, None]).sum(axis=0) / total_area
        self.com_local = com
        second = 0.0
        for (cx, cz, hx, hz), a in zip(LINK_RECTS, areas):
            second += a * ((2.0 * hx) ** 2 + (2.0 * hz) ** 2) / 12.0
            second += a * ((cx - com[0]) ** 2 + (cz - com[1]) ** 2)
        density = self.mass[0] / total_area
        inertia = density * second
        self.inertia = np.full(self.n, inertia, dtype=float)

        # rect centres relative to the COM (body frame)
        self.rect_rel = rects[:, :2] - com[None, :]
        self.rect_half = rects[:, 2:].copy()

        # boundary sample points relative to the COM
        pts = []
        for r, (nx, nz) in zip(LINK_RECTS, _RECT_SAMPLES):
            pts.extend(_boundary_points(r[0], r[1], r[2], r[3], nx, nz))
        self.pts_local = np.asarray(pts, dtype=float) - com[None, :]
        self.n_pts = int(len(self.pts_local))

        # ---- state: COM position / angle about +y, planar velocities
        pos0 = np.asarray(frame0.position, dtype=float)
        self.com_pos = np.zeros((self.n + 1, 2), dtype=float)
        self.com_pos[: self.n, 0] = pos0[:, 0] + com[0]
        self.com_pos[: self.n, 1] = pos0[:, 2] + com[1]
        self.theta = np.zeros(self.n + 1, dtype=float)
        self.vel = np.zeros((self.n + 1, 2), dtype=float)
        self.omega = np.zeros(self.n + 1, dtype=float)

        self.inv_m = np.zeros(self.n + 1, dtype=float)
        self.inv_i = np.zeros(self.n + 1, dtype=float)
        self.inv_m[: self.n] = 1.0 / self.mass
        self.inv_i[: self.n] = 1.0 / self.inertia
        self.static_idx = self.n

        # ---- static + dynamic collision rects
        n_static = len(STATIC_RECTS)
        n_rects = n_static + 2 * self.n
        self.n_static_rects = n_static
        self.rc = np.zeros((n_rects, 2), dtype=float)
        self.rh = np.zeros((n_rects, 2), dtype=float)
        self.rcos = np.ones(n_rects, dtype=float)
        self.rsin = np.zeros(n_rects, dtype=float)
        self.rowner = np.full(n_rects, self.static_idx, dtype=np.int64)
        for k, (cx, cz, hx, hz) in enumerate(STATIC_RECTS):
            self.rc[k] = (cx, cz)
            self.rh[k] = (hx, hz)
        for i in range(self.n):
            for r in range(2):
                k = n_static + 2 * i + r
                self.rh[k] = self.rect_half[r]
                self.rowner[k] = i

        # point ownership
        self.pt_owner = np.repeat(np.arange(self.n, dtype=np.int64), self.n_pts)

        # ---- fixed export contact pairs
        #   0..7   : link i <-> guide_lower
        #   8..15  : link i <-> guide_upper
        #   16..22 : link i <-> link i+1
        self.n_pairs = 2 * self.n + (self.n - 1)
        self._imp_accum = np.zeros((self.n_pairs, 2), dtype=float)
        self._accum_time = 0.0
        self._gap = np.full(self.n_pairs, float(self.length_scale), dtype=float)
        self._sd_matrix = None

        self.t = 0.0
        self.dt = DT
        self._drive_force = np.zeros(2, dtype=float)
        self._steps = 0
        self._max_pen = 0.0
        self._relax_initial_overlap()
        self._detect()
        self._update_gaps()

    def _relax_initial_overlap(self) -> None:
        """Position-only projection so t=0 is a feasible, overlap-free state.

        The frozen initial layout interlaces neighbouring links and nominally
        overlaps ``guide_upper``; a Jacobi non-penetration projection (no
        velocity change, translation along contact normals weighted by inverse
        mass) moves the assembly to the nearest feasible configuration before
        any dynamics are stepped.
        """
        nb = self.n + 1
        for _ in range(PRERELAX_ITERS):
            contacts = self._detect()
            if contacts is None:
                break
            gap = contacts["gap"]
            pen = np.maximum(-gap, 0.0)
            act = pen > 1e-6
            if not np.any(act):
                break
            ia = contacts["ia"][act]
            ib = contacts["ib"][act]
            nvec = contacts["n"][act]
            pen = pen[act]
            ima, imb = self.inv_m[ia], self.inv_m[ib]
            w = PRERELAX_ALPHA * pen / np.maximum(ima + imb, 1e-12)
            px = w * nvec[:, 0]
            pz = w * nvec[:, 1]
            dx = (np.bincount(ia, ima * px, minlength=nb)
                  - np.bincount(ib, imb * px, minlength=nb))
            dz = (np.bincount(ia, ima * pz, minlength=nb)
                  - np.bincount(ib, imb * pz, minlength=nb))
            cnt = np.maximum(
                np.bincount(ia, minlength=nb) + np.bincount(ib, minlength=nb), 1
            ).astype(float)
            self.com_pos[: self.n, 0] += (dx[: self.n] / cnt[: self.n])
            self.com_pos[: self.n, 1] += (dz[: self.n] / cnt[: self.n])
        self.vel[:] = 0.0
        self.omega[:] = 0.0

    # -------------------------------------------------------------- geometry
    def _refresh_rects(self):
        n_s = self.n_static_rects
        c = np.cos(self.theta[: self.n])
        s = np.sin(self.theta[: self.n])
        for r in range(2):
            rel = self.rect_rel[r]
            k = n_s + np.arange(self.n) * 2 + r
            self.rc[k, 0] = self.com_pos[: self.n, 0] + c * rel[0] + s * rel[1]
            self.rc[k, 1] = self.com_pos[: self.n, 1] - s * rel[0] + c * rel[1]
            self.rcos[k] = c
            self.rsin[k] = s

    def _world_points(self):
        c = np.cos(self.theta[: self.n])[:, None]
        s = np.sin(self.theta[: self.n])[:, None]
        lx = self.pts_local[None, :, 0]
        lz = self.pts_local[None, :, 1]
        px = self.com_pos[: self.n, 0:1] + c * lx + s * lz
        pz = self.com_pos[: self.n, 1:2] - s * lx + c * lz
        return np.stack([px.ravel(), pz.ravel()], axis=1)

    def _detect(self):
        """Point-vs-box narrow phase; returns contact arrays and stores gaps."""
        self._refresh_rects()
        pts = self._world_points()
        rel_x = pts[None, :, 0] - self.rc[:, 0:1]
        rel_z = pts[None, :, 1] - self.rc[:, 1:2]
        c = self.rcos[:, None]
        s = self.rsin[:, None]
        lx = c * rel_x - s * rel_z
        lz = s * rel_x + c * rel_z
        dx = np.abs(lx) - self.rh[:, 0:1]
        dz = np.abs(lz) - self.rh[:, 1:2]
        dmax = np.maximum(dx, dz)
        sd = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2) + np.minimum(dmax, 0.0)
        self._sd_matrix = sd
        self._pts = pts

        owner_pt = self.pt_owner[None, :]
        owner_rc = self.rowner[:, None]
        is_static = owner_rc == self.static_idx
        diff = np.abs(owner_pt - owner_rc)
        valid = (owner_pt != owner_rc) & (is_static | (diff <= 2))
        hit = valid & (sd < CONTACT_MARGIN)
        rows, cols = np.nonzero(hit)
        if rows.size > MAX_CONTACTS:
            order = np.argsort(sd[rows, cols])[:MAX_CONTACTS]
            rows, cols = rows[order], cols[order]
        if rows.size == 0:
            return None

        inside = (dx[rows, cols] < 0.0) & (dz[rows, cols] < 0.0)
        lxc = lx[rows, cols]
        lzc = lz[rows, cols]
        hxc = self.rh[rows, 0]
        hzc = self.rh[rows, 1]
        dxc = dx[rows, cols]
        dzc = dz[rows, cols]

        # inside: separating axis of least penetration
        use_x = dxc >= dzc
        nl_in_x = np.where(use_x, np.sign(lxc) + (lxc == 0.0), 0.0)
        nl_in_z = np.where(use_x, 0.0, np.sign(lzc) + (lzc == 0.0))
        # outside: direction from nearest surface point to the sample point
        qx = np.clip(lxc, -hxc, hxc)
        qz = np.clip(lzc, -hzc, hzc)
        ox = lxc - qx
        oz = lzc - qz
        norm = np.sqrt(ox * ox + oz * oz)
        safe = norm > 1e-12
        nl_out_x = np.where(safe, ox / np.where(safe, norm, 1.0), 0.0)
        nl_out_z = np.where(safe, oz / np.where(safe, norm, 1.0), 1.0)

        nlx = np.where(inside, nl_in_x, nl_out_x)
        nlz = np.where(inside, nl_in_z, nl_out_z)
        cr = self.rcos[rows]
        sr = self.rsin[rows]
        nx = cr * nlx + sr * nlz
        nz = -sr * nlx + cr * nlz
        nn = np.sqrt(nx * nx + nz * nz)
        nn = np.where(nn > 1e-12, nn, 1.0)
        normal = np.stack([nx / nn, nz / nn], axis=1)

        ia = self.pt_owner[cols]
        ib = self.rowner[rows]
        cp = pts[cols]
        ra = cp - self.com_pos[ia]
        rb = cp - self.com_pos[ib]
        gap = sd[rows, cols]

        # export pair id and force sign (force reported on the first body)
        pid = np.full(rows.size, -1, dtype=np.int64)
        sign = np.ones(rows.size, dtype=float)
        m_low = rows == 1
        pid[m_low] = ia[m_low]
        m_up = rows == 2
        pid[m_up] = self.n + ia[m_up]
        m_link = (ib != self.static_idx) & (np.abs(ia - ib) == 1)
        if np.any(m_link):
            lo = np.minimum(ia[m_link], ib[m_link])
            pid[m_link] = 2 * self.n + lo
            sign[m_link] = np.where(ia[m_link] < ib[m_link], 1.0, -1.0)

        return {
            "ia": ia, "ib": ib, "n": normal, "ra": ra, "rb": rb,
            "gap": gap, "pid": pid, "sign": sign,
        }

    def _update_gaps(self):
        """Minimum signed gap for every exported contact pair."""
        sd = self._sd_matrix
        p = self.n_pts
        n_s = self.n_static_rects
        for i in range(self.n):
            sl = slice(i * p, (i + 1) * p)
            self._gap[i] = float(np.min(sd[1, sl]))
            self._gap[self.n + i] = float(np.min(sd[2, sl]))
        for i in range(self.n - 1):
            j = i + 1
            si = slice(i * p, (i + 1) * p)
            sj = slice(j * p, (j + 1) * p)
            vals = (
                np.min(sd[n_s + 2 * j: n_s + 2 * j + 2, si]),
                np.min(sd[n_s + 2 * i: n_s + 2 * i + 2, sj]),
            )
            self._gap[2 * self.n + i] = float(min(vals))

    # --------------------------------------------------------------- physics
    def _drive(self):
        t = self.t
        if t <= DRIVE_START:
            target = 0.0
        else:
            u = min(1.0, (t - DRIVE_START) / DRIVE_RAMP)
            target = DRIVE_SPEED * (u * u * (3.0 - 2.0 * u))
        fx = DRIVE_GAIN * (target - self.vel[0, 0])
        fx = float(np.clip(fx, -DRIVE_FMAX, DRIVE_FMAX))
        self._drive_force = np.array([fx, 0.0])
        c, s = np.cos(self.theta[0]), np.sin(self.theta[0])
        lx = DRIVE_POINT[0] - self.com_local[0]
        lz = DRIVE_POINT[1] - self.com_local[1]
        rx = c * lx + s * lz
        rz = -s * lx + c * lz
        self.vel[0, 0] += self.inv_m[0] * fx * self.dt
        self.omega[0] += self.inv_i[0] * (rz * fx) * self.dt

    def _step(self):
        dt = self.dt
        # ---- external forces (gravity + damping) then the driver actuator
        self.vel[: self.n] += self.gravity[None, :] * dt
        self.vel[: self.n] *= max(0.0, 1.0 - LIN_DAMPING * dt)
        self.omega[: self.n] *= max(0.0, 1.0 - ANG_DAMPING * dt)
        self._drive()

        contacts = self._detect()
        if contacts is not None:
            ia = contacts["ia"]
            ib = contacts["ib"]
            nvec = contacts["n"]
            ra = contacts["ra"]
            rb = contacts["rb"]
            gap = contacts["gap"]
            tvec = np.stack([-nvec[:, 1], nvec[:, 0]], axis=1)

            ca = ra[:, 1] * nvec[:, 0] - ra[:, 0] * nvec[:, 1]
            cb = rb[:, 1] * nvec[:, 0] - rb[:, 0] * nvec[:, 1]
            ta = ra[:, 1] * tvec[:, 0] - ra[:, 0] * tvec[:, 1]
            tb = rb[:, 1] * tvec[:, 0] - rb[:, 0] * tvec[:, 1]

            ima, imb = self.inv_m[ia], self.inv_m[ib]
            iia, iib = self.inv_i[ia], self.inv_i[ib]
            kn = np.maximum(ima + imb + iia * ca * ca + iib * cb * cb, 1e-12)
            kt = np.maximum(ima + imb + iia * ta * ta + iib * tb * tb, 1e-12)

            pen = np.maximum(-gap - SLOP, 0.0)
            bias = np.minimum(BETA * pen / dt, MAX_PUSHOUT)

            vn0 = ((self.vel[ia] - self.vel[ib]) * nvec).sum(axis=1) \
                + self.omega[ia] * ca - self.omega[ib] * cb
            rest = np.where(
                (vn0 < -REST_THRESHOLD) & (gap > -5.0 * SLOP),
                -self.restitution * vn0, 0.0,
            )

            ln = np.zeros(ia.size)
            lt = np.zeros(ia.size)
            nb = self.n + 1
            for _ in range(SOLVER_ITERS):
                vn = ((self.vel[ia] - self.vel[ib]) * nvec).sum(axis=1) \
                    + self.omega[ia] * ca - self.omega[ib] * cb
                dn = RELAX * (bias + rest - vn) / kn
                ln_new = np.maximum(ln + dn, 0.0)
                dn = ln_new - ln
                ln = ln_new

                vt = ((self.vel[ia] - self.vel[ib]) * tvec).sum(axis=1) \
                    + self.omega[ia] * ta - self.omega[ib] * tb
                dtl = RELAX * (-vt) / kt
                lim = self.mu * ln
                lt_new = np.clip(lt + dtl, -lim, lim)
                dtl = lt_new - lt
                lt = lt_new

                px = dn * nvec[:, 0] + dtl * tvec[:, 0]
                pz = dn * nvec[:, 1] + dtl * tvec[:, 1]
                self.vel[:, 0] += (
                    np.bincount(ia, ima * px, minlength=nb)
                    - np.bincount(ib, imb * px, minlength=nb)
                )
                self.vel[:, 1] += (
                    np.bincount(ia, ima * pz, minlength=nb)
                    - np.bincount(ib, imb * pz, minlength=nb)
                )
                self.omega += (
                    np.bincount(ia, iia * (dn * ca + dtl * ta), minlength=nb)
                    - np.bincount(ib, iib * (dn * cb + dtl * tb), minlength=nb)
                )
                self.vel[self.static_idx] = 0.0
                self.omega[self.static_idx] = 0.0

            # accumulate exported pair forces
            pid = contacts["pid"]
            sel = pid >= 0
            if np.any(sel):
                w = contacts["sign"][sel]
                fx = w * (ln[sel] * nvec[sel, 0] + lt[sel] * tvec[sel, 0])
                fz = w * (ln[sel] * nvec[sel, 1] + lt[sel] * tvec[sel, 1])
                self._imp_accum[:, 0] += np.bincount(pid[sel], fx, minlength=self.n_pairs)
                self._imp_accum[:, 1] += np.bincount(pid[sel], fz, minlength=self.n_pairs)
            self._max_pen = max(self._max_pen, float(-min(gap.min(), 0.0)))
        self._accum_time += dt

        # ---- guard against numerical blow-up, then integrate
        np.nan_to_num(self.vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.omega, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.vel, -V_CLAMP, V_CLAMP, out=self.vel)
        np.clip(self.omega, -W_CLAMP, W_CLAMP, out=self.omega)
        self.vel[self.static_idx] = 0.0
        self.omega[self.static_idx] = 0.0

        self.com_pos[: self.n] += self.vel[: self.n] * dt
        self.theta[: self.n] += self.omega[: self.n] * dt
        self.t += dt
        self._steps += 1

    # ---------------------------------------------------------------- output
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        c = np.cos(self.theta[:n])
        s = np.sin(self.theta[:n])
        clx, clz = self.com_local
        # body-origin offset from the COM, in world axes
        rox = -(c * clx + s * clz)
        roz = -(-s * clx + c * clz)
        ox = self.com_pos[:n, 0] + rox
        oz = self.com_pos[:n, 1] + roz
        vx = self.vel[:n, 0] + self.omega[:n] * roz
        vz = self.vel[:n, 1] - self.omega[:n] * rox

        zeros = np.zeros(n)
        position = np.stack([ox, zeros, oz], axis=1)
        velocity = np.stack([vx, zeros, vz], axis=1)
        half = 0.5 * self.theta[:n]
        orientation = np.stack([zeros, np.sin(half), zeros, np.cos(half)], axis=1)
        angular_velocity = np.stack([zeros, self.omega[:n], zeros], axis=1)

        ext = np.zeros((n, 3))
        ext[:, 2] = self.mass * self.gravity[1]
        ext[:, 0] = self.mass * self.gravity[0]
        ext[0, 0] += self._drive_force[0]
        ext[0, 2] += self._drive_force[1]

        dtacc = self._accum_time if self._accum_time > 1e-12 else self.dt
        forces = self._imp_accum / dtacc
        interaction = np.stack(
            [forces[:, 0], np.zeros(self.n_pairs), forces[:, 1]], axis=1
        )
        self._imp_accum[:] = 0.0
        self._accum_time = 0.0

        self._detect()
        self._update_gaps()
        frame = StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": np.zeros(n)},
            fields={
                "contact_signed_distance": self._gap.copy(),
                "external_force": ext,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        steps = int(round((target - self.t) / self.dt))
        for _ in range(max(0, steps)):
            self._step()
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "planar rigid bodies, relaxed PGS contacts with Coulomb friction",
            "sub_dt": self.dt,
            "steps": int(self._steps),
            "solver_iterations": SOLVER_ITERS,
            "friction": self.mu,
            "restitution": self.restitution,
            "max_penetration": float(self._max_pen),
            "driver_force_x": float(self._drive_force[0]),
            "driver_velocity_x": float(self.vel[0, 0]),
        }
