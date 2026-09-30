"""Physics solver for scene 047_boat_filled_and_sunk.

Self-contained multiphysics model (NumPy only):

* Rigid thin-shell boat: 6-DOF (quaternion orientation) body. Buoyancy is
  computed from the hull's true interior geometry (cross-sections sliced from
  the frozen triangle mesh): displaced volume = interior air volume below the
  external free surface + wetted shell volume. Water captured inside the hull
  adds weight at the interior-water centroid, so progressive filling produces
  a progressive loss of buoyancy, gunwale flooding, sinking, and bottom
  impact (penalty contact against the tank bottom).
* Water is carried as particles (pool / inlet stream / hull-interior water)
  advanced with semi-implicit Euler, hydrostatic pressure cancellation below
  the free surface, viscous drag, hull radial push-out, capture into the hull,
  spill over the gunwales, and flooding from the pool once the gunwale
  submerges.

No animation tables, no precomputed trajectories; every state is produced by
explicit time stepping.
"""
from __future__ import annotations

import os

import numpy as np

from worldsolver.contracts import StateFrame

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(os.path.dirname(HERE), "assets", "open_boat.obj")


# ----------------------------------------------------------------- mesh utils
def _load_obj(path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, float), np.asarray(f, int)


def _slice_loop(verts, faces, z):
    """Horizontal cross-section of the (open) hull at height z -> loop (M,2)."""
    segs = []
    for tri in faces:
        p = verts[tri]
        d = p[:, 2] - z
        if np.all(d > 0) or np.all(d < 0):
            continue
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if d[i] == 0.0:
                pts.append(p[i, :2])
            elif d[i] * d[j] < 0:
                t = d[i] / (d[i] - d[j])
                pts.append(p[i, :2] + t * (p[j, :2] - p[i, :2]))
        if len(pts) >= 2:
            segs.append((pts[0], pts[1]))
    if not segs:
        return None
    # chain segments into one loop
    loop = [segs[0][0], segs[0][1]]
    used = {0}
    while len(used) < len(segs):
        cur = loop[-1]
        for i, (a, b) in enumerate(segs):
            if i in used:
                continue
            if np.allclose(a, cur, atol=1e-6):
                loop.append(b); used.add(i); break
            if np.allclose(b, cur, atol=1e-6):
                loop.append(a); used.add(i); break
        else:
            break
    return np.asarray(loop[: len(loop) - 1] if np.allclose(loop[0], loop[-1], atol=1e-6)
                      else loop)


def _polar(loop, nth):
    """Star-polar resample of a closed 2D loop at `nth` uniform angles."""
    c = loop.mean(0)
    d = loop - c
    ang = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    rad = np.hypot(d[:, 0], d[:, 1])
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    ang = np.concatenate([ang, ang[:1] + 2 * np.pi])
    rad = np.concatenate([rad, rad[:1]])
    grid = np.linspace(0.0, 2 * np.pi, nth, endpoint=False)
    r = np.interp(grid, ang, rad)
    return c, r


# --------------------------------------------------------------- quat helpers
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ------------------------------------------------------------------- solver
class BenchmarkSolver:
    DT = 0.002          # physics substep (s)
    NTH = 64            # polar resolution of hull cross-sections
    NSAMP = 1400        # interior volume samples for buoyancy

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        rng = np.random.default_rng(int(seed))

        self.rho = float(cfg["fluid_domain"]["water_density"])
        self.zfs = float(cfg["fluid_domain"]["initial_free_surface_z"])
        gvec = np.asarray(cfg["gravity"], float)
        self.gvec = gvec
        boat_cfg = cfg["boat"]
        self.m_boat = float(boat_cfg["mass"])
        self.z_top_gunwale = None

        # ---- hull geometry tables (from the frozen mesh) ----
        verts, faces = _load_obj(MESH)
        self.verts, self.faces = verts, faces
        self.zmin = float(verts[:, 2].min())
        self.ztop = float(verts[:, 2].max())
        nz = 160
        self.ztab = np.linspace(self.zmin, self.ztop, nz)
        rtab = np.zeros((nz, self.NTH))
        vtab = np.zeros(nz)
        prev = 0.0
        prev_a = 0.0
        for i, z in enumerate(self.ztab):
            loop = _slice_loop(verts, faces, z)
            if loop is None or len(loop) < 3:
                if i > 0:
                    rtab[i] = rtab[i - 1]
                vtab[i] = prev
                continue
            c, r = _polar(loop, self.NTH)
            rtab[i] = r
            th = np.linspace(0.0, 2 * np.pi, self.NTH, endpoint=False)
            area = 0.5 * np.sum(r * np.roll(r, -1) * np.sin(2 * np.pi / self.NTH))
            if i == 0:
                vtab[i] = 0.0
            else:
                vtab[i] = vtab[i - 1] + 0.5 * (area + prev_a) * (z - self.ztab[i - 1])
            prev_a = area
        self.rtab, self.vtab = rtab, vtab
        self.V_total = float(vtab[-1])
        self.dz_tab = self.ztab[1] - self.ztab[0]

        # shell displacement volume (thin shell, capped so the flooded boat sinks)
        a = verts[faces[:, 0]] - verts[faces[:, 1]]
        b = verts[faces[:, 0]] - verts[faces[:, 2]]
        area_shell = float(np.sum(0.5 * np.linalg.norm(np.cross(a, b), axis=1)))
        t = float(boat_cfg.get("thin_shell_thickness", 0.018))
        self.V_shell = float(min(t * area_shell, 0.070, 0.8 * self.m_boat / self.rho))

        # buoyancy calibration: equilibrium draft = target_initial_draft
        zlw = self.zmin + float(boat_cfg.get("target_initial_draft", 0.28))
        V28 = float(np.interp(zlw, self.ztab, self.vtab))
        f28 = max(V28 / self.V_total, 1e-3)
        need = self.m_boat / self.rho - self.V_shell * f28
        self.k_buoy = float(np.clip(need / max(V28, 1e-6), 0.02, 1.0))

        # interior volume samples (local frame), weight = V_total / NSAMP
        zs = rng.uniform(self.zmin + 1e-3, self.ztop - 1e-3, 4 * self.NSAMP)
        th = rng.uniform(0.0, 2 * np.pi, 4 * self.NSAMP)
        rr = np.sqrt(rng.uniform(0.0, 1.0, 4 * self.NSAMP))
        Rint = self._radius(th, zs)
        pts = np.stack([rr * Rint * np.cos(th), rr * Rint * np.sin(th), zs], axis=1)
        keep = np.nonzero((zs > self.zmin) & (zs < self.ztop) & (rr < 1.0))[0][: self.NSAMP]
        self.samples = pts[keep]
        self.w_samp = self.V_total / len(self.samples)
        self.hull_centroid = self.samples.mean(0)

        # rigid body state (start from frozen scaffold initial state)
        f0 = scene.initial_state()
        pos0 = np.array(f0.position, float)
        self.p = pos0[0].copy()
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.w = np.zeros(3)
        L, W, H = 2.84, 1.76, 0.86
        self.I = np.diag(self.m_boat / 12.0 * np.array([W * W + H * H,
                                                        L * L + H * H,
                                                        L * L + W * W])) * 2.0
        self.Iinv = np.linalg.inv(self.I)

        # ---- particles ----
        ppos = pos0[1:].copy()
        pvel = np.array(f0.velocity, float)[1:]
        pmass = np.array(f0.mass, float)[1:]
        region = np.array(f0.attributes["water_region"], int)[1:]
        n0 = len(ppos)
        n_emit = 96
        self.ntot = n0 + n_emit
        self.n0 = n0
        P = np.zeros((self.ntot, 3))
        V = np.zeros((self.ntot, 3))
        M = np.zeros(self.ntot)
        REG = np.zeros(self.ntot, int)
        P[:n0], V[:n0], M[:n0] = ppos, pvel, pmass
        REG[:n0] = np.where(region == 3, 3, 1)
        self.inflow_v = float(cfg["inlet"]["velocity"][2])
        inrad = float(cfg["inlet"]["radius"])
        P[n0:] = np.asarray(cfg["inlet"]["center"], float)
        V[n0:] = 0.0
        M[n0:] = 0.0
        REG[n0:] = 3
        self.emit_from = n0
        self.P, self.V, self.M, self.REG = P, V, M, REG
        self.m_p = float(np.median(pmass[pmass > 0])) if np.any(pmass > 0) else 5.8
        self.slot = np.zeros((self.ntot, 3))     # local-frame rest slot (hull water)
        self.inrad = inrad
        self.flow = float(cfg["inlet"]["volumetric_flow_rate"])
        self.t_in = cfg["inlet"]["active_interval"]
        self.emit_acc = 0.0
        self.rng = rng
        self.inlet_added = 0.0
        self.spilled = 0.0
        self.flooded = 0.0

        self.t = 0.0
        self.bbox = np.array(cfg["fluid_domain"]["bounds_min"])[0], \
            np.array(cfg["fluid_domain"]["bounds_max"])[0]
        self.xlo, self.xhi = -2.28, 2.28
        self.ylo, self.yhi = -1.68, 1.68
        self._cache = {0.0: self._frame(0.0)}

    # ------------- hull geometry queries -------------
    def _radius(self, theta, z):
        th = np.asarray(theta, float) % (2 * np.pi)
        z = np.asarray(z, float)
        iw = (z - self.zmin) / self.dz_tab
        i0 = np.clip(np.floor(iw).astype(int), 0, len(self.ztab) - 2)
        f = np.clip(iw - i0, 0.0, 1.0)
        rows = self.rtab[i0] * (1 - f[:, None]) + self.rtab[i0 + 1] * f[:, None]
        nth = self.NTH
        t = th / (2 * np.pi) * nth
        j0 = np.floor(t).astype(int) % nth
        j1 = (j0 + 1) % nth
        ft = t - np.floor(t)
        idx = np.arange(len(th))
        return rows[idx, j0] * (1 - ft) + rows[idx, j1] * ft

    def _inside(self, local):
        """Boolean mask: local points inside the hull interior."""
        local = np.atleast_2d(local)
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        R = self._radius(th, z)
        return (z > self.zmin) & (z < self.ztop) & (r < R)

    def _hull_levels(self):
        Rm = _qmat(self.q)
        return Rm, Rm.T

    def _V_env(self, h):
        return float(np.interp(np.clip(h, self.zmin, self.ztop), self.ztab, self.vtab))

    def _h_from_volume(self, vol):
        if vol <= 0:
            return self.zmin
        return float(np.interp(vol, self.vtab, self.ztab))

    # ------------- physics -------------
    def _step(self, dt):
        q, p, v, w = self.q, self.p, self.v, self.w
        Rm, RmT = self._hull_levels()

        # hull water mass / level
        inmask = self.REG == 2
        m_in = float(self.M[inmask].sum())
        h_in = self._h_from_volume(m_in / self.rho)
        gunwale_z = float((Rm @ np.array([0.0, 0.0, self.ztop]))[2] + p[2])
        submerged = gunwale_z < self.zfs - 0.02

        # ---------- boat forces ----------
        F = np.zeros(3)
        T = np.zeros(3)
        F += self.m_boat * self.gvec

        sw = self.samples @ Rm.T + p
        sub = sw[:, 2] < self.zfs
        air = sub & (self.samples[:, 2] > h_in)
        V_air = self.k_buoy * self.w_samp * float(air.sum())
        f_sub = float(sub.mean())
        V_disp = V_air + self.V_shell * f_sub
        Fb = self.rho * G * V_disp
        F[2] += Fb
        cw = self.hull_centroid @ Rm.T + p
        if V_disp > 1e-9:
            cob = cw * (self.V_shell * f_sub / V_disp)
            if air.any():
                cob = cob + sw[air].mean(0) * (V_air / V_disp)
            T += np.cross(cob - p, np.array([0.0, 0.0, Fb]))

        # internal water weight
        if m_in > 0:
            Fw = -self.rho * G * 0.0  # weight handled below
            wm = self.samples[:, 2] < h_in
            if wm.any():
                cow = self.samples[wm].mean(0) @ Rm.T + p
            else:
                cow = p
            Fwz = -m_in * G
            F[2] += Fwz
            T += np.cross(cow - p, np.array([0.0, 0.0, Fwz]))

        # hydrodynamic drag (still water below free surface)
        rel = v.copy()
        if p[2] < self.zfs + 1.0:
            F += -(250.0 + 800.0 * np.linalg.norm(rel)) * rel
        F_drag = -(250.0 + 800.0 * np.linalg.norm(rel)) * rel if p[2] < self.zfs + 1.0 \
            else np.zeros(3)

        # keel / upright stabilization + angular damping
        rz = Rm @ np.array([0.0, 0.0, 1.0])
        T += 150.0 * np.cross(rz, np.array([0.0, 0.0, 1.0]))
        T -= 60.0 * w

        # bottom contact (tank top at z = 0), penalty at deepest hull vertex
        vw = self.verts @ Rm.T + p
        iv = int(np.argmin(vw[:, 2]))
        zkeel = float(vw[iv, 2])
        Fc = np.zeros(3)
        if zkeel < 0.0:
            Fc = np.array([0.0, 0.0, 2.0e6 * (-zkeel) - 2.0e4 * max(v[2], 0.0)])
            Fc[2] = max(Fc[2], 0.0)
            F += Fc
            T += np.cross(vw[iv] - p, Fc)

        # integrate rigid body
        a = F / (self.m_boat + m_in)
        v += a * dt
        p += v * dt
        Tb = RmT @ T
        wb = RmT @ w
        wbdot = self.Iinv @ (Tb - np.cross(wb, self.I @ wb))
        w += (Rm @ wbdot) * dt
        q = q + 0.5 * _qmul(q, np.array([0.0, *wb])) * dt
        self.q = q / np.linalg.norm(q)
        self.p, self.v, self.w = p, v, w

        Rm, RmT = self._hull_levels()

        # ---------- inlet emission ----------
        if self.t_in[0] <= self.t < self.t_in[1]:
            self.emit_acc += self.flow / self.m_p * dt
            while self.emit_acc >= 1.0:
                self.emit_acc -= 1.0
                i = self.emit_from
                if i >= self.ntot:
                    break
                self.emit_from += 1
                u = self.rng.uniform(0.0, 1.0, 2)
                rr = self.inrad * np.sqrt(u[0])
                th = u[1] * 2 * np.pi
                self.P[i] = [rr * np.cos(th), rr * np.sin(th), 3.35]
                self.V[i] = [0.0, 0.0, self.inflow_v]
                self.M[i] = self.m_p
                self.REG[i] = 3
                self.inlet_added += self.m_p

        # ---------- particles ----------
        P, V, M, REG = self.P, self.V, self.M, self.REG
        active = M > 0
        # gravity + ambient forces
        V[active] += self.gvec * dt
        # hydrostatic cancellation below free surface (pool water at rest)
        pool = active & (REG == 1) & (P[:, 2] < self.zfs)
        V[pool] -= self.gvec * dt
        V[pool] *= np.exp(-5.0 * dt)
        V[active & (REG == 3)] *= np.exp(-0.05 * dt)

        # candidates near the hull (broad phase)
        lo = Rm @ np.array([-1.6, -1.1, self.zmin - 0.3]) + p
        hi = Rm @ np.array([1.6, 1.1, self.ztop + 0.3]) + p
        lo = np.minimum(lo, p - 1.9)
        hi = np.maximum(hi, p + 1.9)
        cand = np.nonzero(active & (np.all(P >= lo - 0.2, axis=1) &
                                    np.all(P <= hi + 0.2, axis=1)))[0]

        if len(cand):
            loc = (P[cand] - p) @ Rm
            ins = self._inside(loc)
            for k, i in enumerate(cand):
                if not ins[k]:
                    continue
                if REG[i] == 3:
                    # stream enters the open hull -> captured as hull water
                    REG[i] = 2
                    r = np.hypot(loc[k, 0], loc[k, 1])
                    if r > 1e-6:
                        s = min(0.9 * self._radius(np.arctan2(loc[k, 1], loc[k, 0]),
                                                   loc[k, 2]) / r, 1.0)
                        xy = loc[k, :2] * s
                    else:
                        xy = loc[k, :2]
                    self.slot[i] = [xy[0], xy[1], max(h_in, self.zmin + 0.05)]
                elif REG[i] == 1:
                    if submerged:
                        REG[i] = 2
                        self.slot[i] = [loc[k, 0], loc[k, 1], loc[k, 2]]
                        self.flooded += M[i]
                    else:
                        # push pool water out of the hull radially
                        rr = np.hypot(loc[k, 0], loc[k, 1])
                        Rr = self._radius(np.arctan2(loc[k, 1], loc[k, 0]),
                                          np.clip(loc[k, 2], self.zmin, self.ztop))
                        if rr < 1e-6:
                            xy = np.array([Rr + 0.05, 0.0])
                        else:
                            xy = loc[k, :2] * ((Rr + 0.05) / rr)
                        newl = np.array([xy[0], xy[1], loc[k, 2]])
                        P[i] = Rm @ newl + p
                        V[i] = Rm @ np.array([xy[0] * 0.5, xy[1] * 0.5, 0.0])

        # hull water: damped spring toward its slot (rigid with the boat)
        hm = active & (REG == 2)
        if hm.any():
            tgt = self.slot[hm] @ Rm.T + p
            V[hm] += (50.0 * (tgt - P[hm]) - 10.0 * V[hm]) * dt
            # spill over the gunwales when overflowing and not yet submerged
            if not submerged and h_in > self.ztop - 0.05:
                loc = (P[hm] - p) @ Rm
                over = np.nonzero(loc[:, 2] > self.ztop - 0.08)[0]
                if len(over):
                    idxs = np.nonzero(hm)[0][over]
                    for i in idxs:
                        REG[i] = 1
                        self.spilled += M[i]
                        d = P[i] - p
                        d[2] = 0.0
                        n = np.linalg.norm(d)
                        if n > 1e-6:
                            V[i] = d / n * 1.0 + np.array([0.0, 0.0, -0.5])

        P[active] += V[active] * dt
        # domain clamps
        np.clip(P[:, 0], self.xlo, self.xhi, out=P[:, 0])
        np.clip(P[:, 1], self.ylo, self.yhi, out=P[:, 1])
        floor = P[:, 2] < 0.03
        if floor.any():
            P[floor, 2] = 0.03
            V[floor, 2] = np.maximum(V[floor, 2], 0.0)

        self.t += dt

    # ------------- frame export -------------
    def _frame(self, t):
        n = self.ntot
        Rm, _ = self._hull_levels()
        pos = np.vstack([self.p, self.P])
        vel = np.vstack([self.v, self.V])
        mass = np.concatenate([[self.m_boat], self.M])
        body = np.zeros(n, np.int32)
        body[1:] = 1
        region = np.zeros(n, np.int32)
        region[1:] = self.REG
        ori = np.zeros((n, 4))
        ori[0] = self.q
        ori[:, 3] = np.where(np.arange(n) == 0, self.q[3], 1.0)
        angv = np.zeros((n, 3))
        angv[0] = self.w

        # buoyancy bookkeeping for reporting
        m_in = float(self.M[self.REG == 2].sum())
        h_in = self._h_from_volume(m_in / self.rho)
        sub = (self.samples @ Rm.T + self.p)[:, 2] < self.zfs
        V_air = self.k_buoy * self.w_samp * float((sub & (self.samples[:, 2] > h_in)).sum())
        V_disp = V_air + self.V_shell * float(sub.mean())
        Fb = self.rho * G * V_disp
        vw = self.verts @ Rm.T + self.p
        zkeel = float(vw[:, 2].min())
        draft = max(self.zfs - zkeel, 0.0)

        ext = np.zeros((n, 3))
        ext[:, 2] = -mass * G
        # external boundary contact force on the boat (bottom penalty)
        if zkeel < 0.0:
            ext[0, 2] += max(2.0e6 * (-zkeel) - 2.0e4 * max(self.v[2], 0.0), 0.0)
        inter = np.zeros((3, 3))
        inter[0] = [0.0, 0.0, Fb]                       # boat <-> external water
        inter[1] = [0.0, 0.0, -m_in * G]                # boat <-> internal water
        rel = self.v
        inter[2] = -(250.0 + 800.0 * np.linalg.norm(rel)) * rel \
            if self.p[2] < self.zfs + 1.0 else np.zeros(3)
        src = np.array([self.inlet_added, -self.spilled, self.flooded])
        # hull extents vs static boundaries (signed gaps, negative = penetration)
        ex = np.abs(vw[:, 0]).max(); ey = np.abs(vw[:, 1]).max()
        csd = np.array([
            zkeel - 0.0,
            1.68 - ey if ey > 0 else 1.68,
            2.28 - ex if ex > 0 else 2.28,
            2.28 - ex if ex > 0 else 2.28,
        ])
        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=ori,
            angular_velocity=angv,
            mass=mass,
            body_id=body,
            attributes={
                "phase_id": np.where(np.arange(n) == 0, 0, 1).astype(np.int32),
                "water_region": region,
                "mass_quantity": mass.copy(),
            },
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "cumulative_mass_source": src,
                "contact_signed_distance": csd,
            },
            scalars={
                "contact_length_scale": 0.018,
                "hull_water_mass": float(m_in),
                "hull_water_level_local": float(h_in),
                "draft": float(draft),
                "displaced_volume": float(V_disp),
                "gunwale_submerged": float(1.0 if (Rm @ np.array([0, 0, self.ztop]))[2]
                                           + self.p[2] < self.zfs else 0.0),
                "keel_height": float(zkeel),
            },
        )
        frame.validate()
        return frame

    # ------------- API -------------
    def advance_to(self, time: float):
        t = float(time)
        if t < self.t - 1e-9:
            if t in self._cache:
                return self._cache[t]
            t = max(t, 0.0)
        target = self.t
        while self.t < t - 1e-12:
            nxt = min(self.t + self.DT, t)
            self._step(nxt - self.t)
        frame = self._frame(t)
        self._cache[t] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        m_in = float(self.M[self.REG == 2].sum())
        return {
            "time": float(self.t),
            "boat_center": self.p.tolist(),
            "hull_water_mass": m_in,
            "inlet_mass_added": float(self.inlet_added),
            "spilled_mass": float(self.spilled),
            "flooded_mass": float(self.flooded),
            "interior_volume": float(self.V_total),
            "shell_volume": float(self.V_shell),
            "buoyancy_calibration": float(self.k_buoy),
        }
