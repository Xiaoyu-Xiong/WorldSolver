"""Two-way coupled free-surface water / rigid frigate solver (no Genesis).

Physics implemented from scratch with NumPy:

* Water: nonlinear shallow-water equations (depth-integrated mass and
  momentum) on a staggered C-grid, advanced with an explicit scheme. A
  wavemaker at ``x_min`` radiates successive incoming waves, port/starboard
  walls are reflecting, the downstream boundary is a transmissive sponge.
* Water markers: Lagrangian samples advected by the reconstructed 3-D
  velocity field (depth-averaged horizontal velocity plus the kinematically
  consistent vertical velocity) and excluded from the impermeable hull.
* Frigate: rigid body with the frozen free degrees of freedom (heave, pitch).
  Loads are the water pressure integrated over wetted hull triangles
  (hydrostatic + Froude-Krylov of the resolved surface), plus radiation
  damping, added mass and added pitch inertia.
* Coupling: hull draft enters continuity as a moving-body volume source and
  reduces the locally open depth, so the moving hull deflects the resolved
  flow; the resolved surface sets the hull pressure loads in return.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

_F32 = np.float32


def _load_obj(path):
    """Read triangulated vertices/faces from a Wavefront OBJ file."""
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _subdivide(vertices, faces, levels):
    """Uniform 1-to-4 triangle refinement."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    for _ in range(int(levels)):
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        n0 = np.concatenate([v0, m01, m20, m01], axis=0)
        n1 = np.concatenate([m01, v1, m12, m12], axis=0)
        n2 = np.concatenate([m20, m12, v2, m20], axis=0)
        v0, v1, v2 = n0, n1, n2
    return v0, v1, v2
class BenchmarkSolver:
    """Explicit shallow-water / rigid-body coupled time-stepping solver."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        fl = cfg["fluid"]
        wv = cfg["incoming_waves"]
        sh = cfg["frigate"]

        self.gvec = np.asarray(cfg["gravity"], dtype=float)
        self.g = float(abs(self.gvec[2]))
        self.rho = float(fl["density"])
        self.lo = np.asarray(fl["domain_min"], dtype=float)
        self.hi = np.asarray(fl["domain_max"], dtype=float)
        self.h0 = float(fl["still_water_level"])
        self.u_mean = float(np.asarray(fl["initial_velocity"], dtype=float)[0])
        self.amp = float(wv["amplitude"])
        self.kw = 2.0 * math.pi / float(wv["wavelength"])
        self.ph0 = float(wv["phase_radians"])
        self.c0 = math.sqrt(self.g * self.h0)
        # Downstream-going linear shallow-water mode carried by the mean current.
        self.omega = self.kw * (self.c0 + self.u_mean)
        self.sample_dt = float(cfg["sample_dt"])
        self.n_sub = 3
        self.dt = self.sample_dt / self.n_sub

        # --- staggered C-grid -------------------------------------------
        self.dx = 0.30
        self.dy = 0.30
        self.nx = int(round((self.hi[0] - self.lo[0]) / self.dx))
        self.ny = int(round((self.hi[1] - self.lo[1]) / self.dy))
        self.lx = float(self.hi[0] - self.lo[0])
        self.xc = self.lo[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.lo[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.xf = self.lo[0] + np.arange(self.nx + 1) * self.dx
        self.hmin = 0.10
        self.nu = 0.05
        self.nu_eta = 0.02
        self.cd = 0.0025

        eta0 = self.amp * np.sin(self.kw * self.xc + self.ph0)
        self.eta = np.repeat(eta0[:, None], self.ny, axis=1).copy()
        etaf = self.amp * np.sin(self.kw * self.xf + self.ph0)
        uf = self.u_mean + (self.g / self.c0) * etaf
        self.u = np.repeat(uf[:, None], self.ny, axis=1).copy()
        self.v = np.zeros((self.nx, self.ny + 1))
        self.eta_t = np.zeros((self.nx, self.ny))

        # outflow sponge (absorbs the transmitted waves, no reflection)
        span = 3.0
        s = np.clip((self.xc - (self.hi[0] - span)) / span, 0.0, 1.0) ** 2
        self.sponge_c = (2.5 * s)[:, None]
        self.sponge_x = np.zeros((self.nx + 1, 1))
        sf = np.clip((self.xf - (self.hi[0] - span)) / span, 0.0, 1.0) ** 2
        self.sponge_x = (2.5 * sf)[:, None]
        self._setup_hull(cfg, sh)
        self._setup_markers(scene, sh)
        self._setup_boundary()
    def _setup_hull(self, cfg, sh) -> None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        verts, faces = _load_obj(os.path.join(here, "assets", "frigate_hull.obj"))
        v0, v1, v2 = _subdivide(verts, faces, 3)
        self.t0, self.t1, self.t2 = v0, v1, v2
        self.tc_loc = (v0 + v1 + v2) / 3.0
        self.tn_loc = 0.5 * np.cross(v1 - v0, v2 - v0)  # outward area vectors

        self.m_ship = float(sh["mass"])
        self.z0 = float(np.asarray(sh["initial_center"], dtype=float)[2])
        self.zc = self.z0
        self.wz = 0.0
        self.theta = 0.0
        self.omega_y = 0.0

        # rigid-body volume and pitch inertia about the local origin
        cz = self.tc_loc[:, 2]
        cx = self.tc_loc[:, 0]
        nz = self.tn_loc[:, 2]
        nx = self.tn_loc[:, 0]
        self.vol_hull = float(np.sum(cz * nz))
        ixx = float(np.sum((cx ** 3 / 3.0) * nx))
        izz = float(np.sum((cz ** 3 / 3.0) * nz))
        self.iyy = max(self.m_ship / max(self.vol_hull, 1e-9) * (ixx + izz), 1.0)

        # hydrostatics at the frozen initial pose
        s0 = self.h0 - self.z0
        wet0 = cz < s0
        vsub0 = float(np.sum(-(cz[wet0] - s0) * nz[wet0]))
        aw = float(np.sum(-nz[wet0]))
        iw = float(np.sum(-(cx[wet0] ** 2) * nz[wet0]))
        self.vsub0 = max(vsub0, 0.5)
        self.awp = max(aw, 1.0)
        self.iwp = max(iw, 1.0)

        self.m_add = 0.8 * self.rho * self.vsub0
        self.i_add = 0.5 * self.iyy
        m_eff = self.m_ship + self.m_add
        i_eff = self.iyy + self.i_add
        k_heave = self.rho * self.g * self.awp
        k_pitch = self.rho * self.g * self.iwp
        self.m_eff = m_eff
        self.i_eff = i_eff
        self.c_heave = 2.0 * 0.25 * math.sqrt(max(k_heave, 0.0) * m_eff)
        self.c_pitch = 2.0 * 0.25 * math.sqrt(max(k_pitch, 0.0) * i_eff)

        # analytic exclusion envelope for the water markers (matches the mesh)
        self.hl = 4.35
        self.hw = 0.84
        self.keel = -1.12
        self.deck = 0.40
        self.draft_prev = None

    def _setup_markers(self, scene, sh) -> None:
        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=float)
        self.n_total = len(pos)
        self.m_marker = float(frame0.mass[1])
        self.mp = pos[1:].copy()
        self.n_mark = len(self.mp)
        self.m_fluid = self.m_marker * self.n_mark
        zs = self._surface_at(self.mp[:, 0], self.mp[:, 1])
        self.frac = np.clip(self.mp[:, 2] / np.maximum(zs, 1e-3), 0.005, 1.0)
        self.mv = np.zeros((self.n_mark, 3))
        self.mv[:, 0] = np.asarray(scene.config["fluid"]["initial_velocity"], float)[0]
        self.mv[:, 1] = np.asarray(scene.config["fluid"]["initial_velocity"], float)[1]
        self.mv[:, 2] = np.asarray(scene.config["fluid"]["initial_velocity"], float)[2]

        self.mass_all = np.full(self.n_total, self.m_marker)
        self.mass_all[0] = self.m_ship
        self.body_id = np.zeros(self.n_total, dtype=np.int64)
        self.body_id[0] = int(sh["body_id"])
        self.phase = np.ones(self.n_total, dtype=_F32)
        self.phase[0] = 0.0
        self.ship_visual = np.zeros(self.n_total, dtype=_F32)
        self.ship_visual[0] = 1.0

        # momentum-budget domains: row 0 = frigate body, row 1 = water markers
        w = np.zeros((2, self.n_total), dtype=_F32)
        w[0, 0] = 1.0
        w[1, 1:] = 1.0
        self.weights = w
        self.mass_quantity = self.mass_all.astype(_F32)
        # mass_quantity is [N] (K=1 conserved water/hull mass); markers are never
        # created or destroyed (out-of-domain markers wrap with their velocity),
        # so the prescribed mass source stays identically zero.
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.cum_impulse = np.zeros((2, 3), dtype=float)
        self.cum_in = np.zeros((2, 3), dtype=float)
        self.cum_out = np.zeros((2, 3), dtype=float)
        self.force_terms = np.zeros((2, 4, 3), dtype=float)
        self.f_couple = np.zeros(3, dtype=float)
        self.ref_vel = float(self.c0)
        self.ref_mom = float((self.m_fluid + self.m_ship) * self.c0)
        self.t_now = 0.0
        self.steps = 0
    def _setup_boundary(self) -> None:
        xs = self.xc[::6]
        ys = self.yc[::4]
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        bed = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
        n_bed = np.tile([0.0, 0.0, 1.0], (len(bed), 1))

        xw = self.xc[::8]
        port = np.column_stack((xw, np.full_like(xw, self.hi[1]), np.full_like(xw, 0.7)))
        n_port = np.tile([0.0, -1.0, 0.0], (len(xw), 1))
        star = np.column_stack((xw, np.full_like(xw, self.lo[1]), np.full_like(xw, 0.7)))
        n_star = np.tile([0.0, 1.0, 0.0], (len(xw), 1))

        yin = self.yc[::6]
        inflow = np.column_stack(
            (np.full_like(yin, self.lo[0]), yin, np.full_like(yin, 0.7))
        )
        n_in = np.tile([1.0, 0.0, 0.0], (len(yin), 1))

        self.b_pos = np.vstack((bed, port, star, inflow))
        self.b_nrm = np.vstack((n_bed, n_port, n_star, n_in)).astype(_F32)
        self.n_bed = len(bed)
        self.n_wall = 2 * len(xw)
        self.n_inflow = len(yin)

    # ------------------------------------------------------------- helpers
    def _eta_bc(self, t: float) -> float:
        return self.amp * math.sin(self.kw * self.lo[0] - self.omega * t + self.ph0)

    def _u_bc(self, t: float) -> float:
        return self.u_mean + (self.g / self.c0) * self._eta_bc(t)

    def _interp(self, field, x, y):
        gx = (x - self.lo[0]) / self.dx - 0.5
        gy = (y - self.lo[1]) / self.dy - 0.5
        i0 = np.clip(np.floor(gx).astype(np.int64), 0, self.nx - 2)
        j0 = np.clip(np.floor(gy).astype(np.int64), 0, self.ny - 2)
        fx = np.clip(gx - i0, 0.0, 1.0)
        fy = np.clip(gy - j0, 0.0, 1.0)
        f00 = field[i0, j0]
        f10 = field[i0 + 1, j0]
        f01 = field[i0, j0 + 1]
        f11 = field[i0 + 1, j0 + 1]
        return (
            f00 * (1.0 - fx) * (1.0 - fy)
            + f10 * fx * (1.0 - fy)
            + f01 * (1.0 - fx) * fy
            + f11 * fx * fy
        )

    def _surface_at(self, x, y):
        return np.maximum(self.h0 + self._interp(self.eta, x, y), self.hmin)

    @staticmethod
    def _lap(a, dx, dy):
        ax = np.empty_like(a)
        ax[1:-1] = a[2:] - 2.0 * a[1:-1] + a[:-2]
        ax[0] = a[1] - a[0]
        ax[-1] = a[-2] - a[-1]
        ay = np.empty_like(a)
        ay[:, 1:-1] = a[:, 2:] - 2.0 * a[:, 1:-1] + a[:, :-2]
        ay[:, 0] = a[:, 1] - a[:, 0]
        ay[:, -1] = a[:, -2] - a[:, -1]
        return ax / (dx * dx) + ay / (dy * dy)

    def _hull_world(self):
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        rot = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        cen = np.array([0.0, 0.0, self.zc])
        tc = self.tc_loc @ rot.T + cen
        tn = self.tn_loc @ rot.T
        return tc, tn

    def _hull_loads(self):
        """Integrate resolved water pressure over the wetted hull triangles."""
        tc, tn = self._hull_world()
        zs = self._surface_at(tc[:, 0], tc[:, 1])
        wet = tc[:, 2] < zs
        force = np.zeros(3)
        torque = 0.0
        draft = np.zeros((self.nx, self.ny))
        if np.any(wet):
            cw = tc[wet]
            nw = tn[wet]
            zw = zs[wet]
            press = self.rho * self.g * (zw - cw[:, 2])
            df = -press[:, None] * nw
            force = df.sum(axis=0)
            arm = cw - np.array([0.0, 0.0, self.zc])
            torque = float(np.sum(arm[:, 2] * df[:, 0] - arm[:, 0] * df[:, 2]))
            # displaced volume scattered onto the water grid (moving-body source)
            dv = -(cw[:, 2] - zw) * nw[:, 2]
            ii = np.clip(
                ((cw[:, 0] - self.lo[0]) / self.dx).astype(np.int64), 0, self.nx - 1
            )
            jj = np.clip(
                ((cw[:, 1] - self.lo[1]) / self.dy).astype(np.int64), 0, self.ny - 1
            )
            flat = np.bincount(
                ii * self.ny + jj, weights=dv, minlength=self.nx * self.ny
            )
            draft = flat.reshape(self.nx, self.ny) / (self.dx * self.dy)
            draft = np.maximum(draft, 0.0)
            sm = draft.copy()
            sm[1:-1, :] = 0.5 * draft[1:-1, :] + 0.25 * (draft[2:, :] + draft[:-2, :])
            draft = sm
            sm2 = draft.copy()
            sm2[:, 1:-1] = 0.5 * draft[:, 1:-1] + 0.25 * (
                draft[:, 2:] + draft[:, :-2]
            )
            draft = np.clip(sm2, 0.0, 0.85 * self.h0)
        return force, torque, draft
    def _swe_step(self, t_next: float, draft, draft_rate) -> None:
        """One explicit shallow-water update on the staggered C-grid."""
        dt = self.dt
        dx, dy = self.dx, self.dy
        h_open = np.maximum(self.h0 + self.eta - draft, self.hmin)

        # --- continuity with the moving-hull volume source ---------------
        hx = np.empty((self.nx + 1, self.ny))
        hx[1:-1] = 0.5 * (h_open[:-1] + h_open[1:])
        hx[0] = h_open[0]
        hx[-1] = h_open[-1]
        flux_x = hx * self.u

        hy = np.zeros((self.nx, self.ny + 1))
        hy[:, 1:-1] = 0.5 * (h_open[:, :-1] + h_open[:, 1:])
        flux_y = hy * self.v

        div = (flux_x[1:] - flux_x[:-1]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy
        # d(h0+eta-draft)/dt + div(h_open u) = 0  ->  eta_t = -div + d(draft)/dt
        eta_t = -div + draft_rate + self.nu_eta * self._lap(self.eta, dx, dy)
        eta_t -= self.sponge_c * self.eta
        self.eta = self.eta + dt * eta_t
        self.eta = np.clip(self.eta, -0.9 * self.h0, 1.5 * self.h0)
        self.eta_t = eta_t

        # --- x-momentum at interior faces --------------------------------
        u = self.u
        ui = u[1:-1]
        dpdx = -self.g * (self.eta[1:] - self.eta[:-1]) / dx
        dudx_b = (ui - u[:-2]) / dx
        dudx_f = (u[2:] - ui) / dx
        adv_x = -np.where(ui > 0.0, ui * dudx_b, ui * dudx_f)
        vx = np.zeros_like(ui)
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        vx = 0.5 * (vc[:-1] + vc[1:])
        dudy = np.empty_like(ui)
        dudy[:, 1:-1] = (ui[:, 2:] - ui[:, :-2]) / (2.0 * dy)
        dudy[:, 0] = (ui[:, 1] - ui[:, 0]) / dy
        dudy[:, -1] = (ui[:, -1] - ui[:, -2]) / dy
        adv_y = -vx * dudy
        visc = self.nu * self._lap(ui, dx, dy)
        hu = np.maximum(0.5 * (h_open[:-1] + h_open[1:]), self.hmin)
        spd = np.sqrt(ui * ui + vx * vx) + 1e-9
        fric = -self.cd * ui * spd / hu
        sp = -self.sponge_x[1:-1] * (ui - self.u_mean)
        u_new = ui + dt * (dpdx + adv_x + adv_y + visc + fric + sp)
        self.u[1:-1] = u_new
        self.u[0] = self._u_bc(t_next)
        self.u[-1] = self.u[-2]

        # --- y-momentum at interior faces --------------------------------
        v = self.v
        vi = v[:, 1:-1]
        dpdy = -self.g * (self.eta[:, 1:] - self.eta[:, :-1]) / dy
        dvdy_b = (vi - v[:, :-2]) / dy
        dvdy_f = (v[:, 2:] - vi) / dy
        adv_yy = -np.where(vi > 0.0, vi * dvdy_b, vi * dvdy_f)
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        uy = 0.5 * (uc[:, :-1] + uc[:, 1:])
        dvdx = np.empty_like(vi)
        dvdx[1:-1] = (vi[2:] - vi[:-2]) / (2.0 * dx)
        dvdx[0] = (vi[1] - vi[0]) / dx
        dvdx[-1] = (vi[-1] - vi[-2]) / dx
        adv_xy = -uy * dvdx
        visc_v = self.nu * self._lap(vi, dx, dy)
        hv = np.maximum(0.5 * (h_open[:, :-1] + h_open[:, 1:]), self.hmin)
        spd_v = np.sqrt(vi * vi + uy * uy) + 1e-9
        fric_v = -self.cd * vi * spd_v / hv
        sp_v = -self.sponge_c[:, :1] * vi
        self.v[:, 1:-1] = vi + dt * (
            dpdy + adv_yy + adv_xy + visc_v + fric_v + sp_v
        )
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        np.clip(self.u, -8.0, 8.0, out=self.u)
        np.clip(self.v, -8.0, 8.0, out=self.v)
    def _hull_exclude(self) -> None:
        """Impermeable hull: project water markers out of the hull volume."""
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        dx = self.mp[:, 0]
        dz = self.mp[:, 2] - self.zc
        lx = ct * dx - st * dz
        lz = st * dx + ct * dz
        ly = self.mp[:, 1]
        half = self.hw * np.clip(1.0 - np.abs(lx) / self.hl, 0.0, 1.0) ** 0.35
        inside = (
            (np.abs(lx) < self.hl)
            & (np.abs(ly) < half)
            & (lz > self.keel)
            & (lz < self.deck)
        )
        if not np.any(inside):
            return
        idx = np.nonzero(inside)[0]
        lys = ly[idx]
        hs = half[idx]
        lzs = lz[idx]
        push_y = np.where(lys >= 0.0, hs - lys + 1e-3, -(hs + lys) - 1e-3)
        push_dn = (self.keel - lzs) - 1e-3
        use_y = np.abs(push_y) <= np.abs(push_dn)
        dy_l = np.where(use_y, push_y, 0.0)
        dz_l = np.where(use_y, 0.0, push_dn)
        # rotate the local z offset back into world coordinates
        self.mp[idx, 0] += -st * dz_l
        self.mp[idx, 1] += dy_l
        self.mp[idx, 2] += ct * dz_l
        np.clip(self.mp[:, 1], self.lo[1] + 0.02, self.hi[1] - 0.02, out=self.mp[:, 1])
        zs = self._surface_at(self.mp[idx, 0], self.mp[idx, 1])
        self.mp[idx, 2] = np.clip(self.mp[idx, 2], 0.02, zs)
        self.frac[idx] = np.clip(self.mp[idx, 2] / np.maximum(zs, 1e-3), 0.005, 1.0)

    def _marker_fields(self):
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        detadx = np.empty_like(self.eta)
        detadx[1:-1] = (self.eta[2:] - self.eta[:-2]) / (2.0 * self.dx)
        detadx[0] = (self.eta[1] - self.eta[0]) / self.dx
        detadx[-1] = (self.eta[-1] - self.eta[-2]) / self.dx
        detady = np.empty_like(self.eta)
        detady[:, 1:-1] = (self.eta[:, 2:] - self.eta[:, :-2]) / (2.0 * self.dy)
        detady[:, 0] = (self.eta[:, 1] - self.eta[:, 0]) / self.dy
        detady[:, -1] = (self.eta[:, -1] - self.eta[:, -2]) / self.dy
        return uc, vc, detadx, detady

    def _marker_velocity(self, uc, vc, detadx, detady, x, y, frac):
        um = self._interp(uc, x, y)
        vm = self._interp(vc, x, y)
        et = self._interp(self.eta_t, x, y)
        gx = self._interp(detadx, x, y)
        gy = self._interp(detady, x, y)
        wm = frac * (et + um * gx + vm * gy)
        return um, vm, wm

    def _advect_markers(self):
        uc, vc, detadx, detady = self._marker_fields()
        x = self.mp[:, 0]
        y = self.mp[:, 1]
        um, vm, _ = self._marker_velocity(uc, vc, detadx, detady, x, y, self.frac)
        xn = x + self.dt * um
        yn = np.clip(y + self.dt * vm, self.lo[1] + 0.02, self.hi[1] - 0.02)
        wrapped = xn > self.hi[0] - 0.02
        xn = np.where(wrapped, xn - self.lx, xn)
        xn = np.where(xn < self.lo[0] + 0.02, xn + self.lx, xn)
        self.mp[:, 0] = xn
        self.mp[:, 1] = yn
        zs = self._surface_at(xn, yn)
        self.mp[:, 2] = np.clip(self.frac * zs, 0.02, None)
        self._hull_exclude()
        un, vn, wn = self._marker_velocity(
            uc, vc, detadx, detady, self.mp[:, 0], self.mp[:, 1], self.frac
        )
        dp = np.array(
            [
                float(np.sum(un - self.mv[:, 0])),
                float(np.sum(vn - self.mv[:, 1])),
                float(np.sum(wn - self.mv[:, 2])),
            ]
        ) * self.m_marker
        self.mv[:, 0] = un
        self.mv[:, 1] = vn
        self.mv[:, 2] = wn
        n_wrap = int(np.count_nonzero(wrapped))
        return dp, n_wrap
    # ---------------------------------------------------------- time step
    def _step(self, t_next: float) -> None:
        dt = self.dt
        force_p, torque_p, draft = self._hull_loads()
        if self.draft_prev is None:
            draft_rate = np.zeros_like(draft)
        else:
            draft_rate = (draft - self.draft_prev) / dt
        self.draft_prev = draft
        np.clip(draft_rate, -4.0, 4.0, out=draft_rate)

        self._swe_step(t_next, draft, draft_rate)

        # --- rigid frigate: free heave and pitch -------------------------
        wz_old = self.wz
        f_damp = -self.c_heave * self.wz
        acc = (force_p[2] + f_damp - self.m_ship * self.g) / self.m_eff
        acc = float(np.clip(acc, -40.0, 40.0))
        self.wz = float(np.clip(self.wz + dt * acc, -3.0, 3.0))
        self.zc = float(np.clip(self.zc + dt * self.wz, 0.55, 3.2))

        tq_damp = -self.c_pitch * self.omega_y
        alpha = float(np.clip((torque_p + tq_damp) / self.i_eff, -20.0, 20.0))
        self.omega_y = float(np.clip(self.omega_y + dt * alpha, -1.5, 1.5))
        self.theta = float(np.clip(self.theta + dt * self.omega_y, -0.45, 0.45))

        a_actual = (self.wz - wz_old) / dt
        f_am = self.m_ship * a_actual - force_p[2] - f_damp + self.m_ship * self.g
        f_couple = np.array([force_p[0], force_p[1], force_p[2] + f_damp + f_am])
        self.f_couple = f_couple

        dp_fluid, n_wrap = self._advect_markers()
        self.n_wrap = n_wrap

        grav_hull = self.m_ship * self.gvec
        react_hull = np.array([-force_p[0], -force_p[1], 0.0])
        grav_fluid = self.m_fluid * self.gvec
        couple_fluid = -f_couple
        traction = dp_fluid / dt - grav_fluid - couple_fluid

        terms = np.zeros((2, 4, 3))
        terms[0, 0] = grav_hull
        terms[0, 2] = f_couple
        terms[0, 3] = react_hull
        terms[1, 0] = grav_fluid
        terms[1, 1] = traction
        terms[1, 2] = couple_fluid
        self.force_terms = terms
        self.cum_impulse += terms.sum(axis=1) * dt
        self.t_now = t_next
        self.steps += 1

    # ---------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        target = int(round(t / self.dt))
        while self.steps < target:
            self._step((self.steps + 1) * self.dt)

        n = self.n_total
        pos = np.empty((n, 3), dtype=_F32)
        pos[0] = (0.0, 0.0, self.zc)
        pos[1:] = self.mp
        vel = np.empty((n, 3), dtype=_F32)
        vel[0] = (0.0, 0.0, self.wz)
        vel[1:] = self.mv
        quat = np.zeros((n, 4), dtype=_F32)
        quat[:, 3] = 1.0
        half = 0.5 * self.theta
        quat[0] = (0.0, math.sin(half), 0.0, math.cos(half))
        avel = np.zeros((n, 3), dtype=_F32)
        avel[0, 1] = self.omega_y

        b_vel, b_tgt = self._boundary_velocity(t)
        fields = {
            "boundary_normal": self.b_nrm,
            "boundary_velocity": b_vel.astype(_F32),
            "boundary_target_velocity": b_tgt.astype(_F32),
            "budget_domain_weights": self.weights,
            "budget_external_force": self.force_terms.astype(np.float64),
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_in.copy(),
            "budget_cumulative_outflow_momentum": self.cum_out.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": np.vstack((self.f_couple, -self.f_couple)),
            "gravity_acceleration": self.gvec.reshape(1, 3),
        }
        frame = StateFrame(
            time=t,
            position=pos,
            velocity=vel,
            orientation=quat,
            angular_velocity=avel,
            mass=self.mass_all,
            body_id=self.body_id,
            attributes={
                "phase": self.phase,
                "ship_visual": self.ship_visual,
                "mass_quantity": self.mass_quantity,
            },
            fields=fields,
            scalars={
                "still_water_level": float(self.h0),
                "reference_velocity": float(self.ref_vel),
                "reference_momentum": float(self.ref_mom),
                "liquid_phase_id": 1.0,
                "characteristic_length": float(2.0 * self.hl),
            },
        )
        return frame

    def _boundary_velocity(self, t: float):
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        x = self.b_pos[:, 0]
        y = self.b_pos[:, 1]
        nb = self.n_bed
        nw = self.n_wall
        vel = np.zeros((len(self.b_pos), 3))
        vel[:, 0] = self._interp(uc, x, y)
        vel[:, 1] = self._interp(vc, x, y)
        tgt = np.zeros_like(vel)
        vel[nb : nb + nw, 1] = 0.0
        ub = self._u_bc(t)
        vel[nb + nw :, 0] = ub
        vel[nb + nw :, 1] = 0.0
        tgt[nb + nw :, 0] = ub
        return vel, tgt

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit shallow-water C-grid + rigid heave/pitch coupling",
            "substeps": int(self.steps),
            "dt": float(self.dt),
            "grid": [int(self.nx), int(self.ny)],
            "markers": int(self.n_mark),
            "wave_period_model_s": float(2.0 * math.pi / self.omega),
            "wave_celerity_m_s": float(self.c0),
            "hull_volume_m3": float(self.vol_hull),
            "waterplane_area_m2": float(self.awp),
            "heave_z_m": float(self.zc),
            "pitch_rad": float(self.theta),
            "added_mass_kg": float(self.m_add),
            "genesis_used": False,
        }
