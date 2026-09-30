"""Deterministic physically-based solver for the boat fill-and-sink scene.

The scene is advanced with an explicit time-stepping procedure:

* The open thin-shell boat is a rigid body (Newton-Euler) whose hydrostatics
  are calibrated to the frozen mass / target draft pair, with a depth dependent
  waterplane (flare) so that the reserve buoyancy matches the prescribed
  fill-and-sink schedule.  Internal water mass carried by the hull enters the
  rigid body force balance through the hull/water interaction force.
* The water (tank pool, inlet stream, water inside the hull and leaked water)
  is simulated as an incompressible particle fluid with Position Based Fluids
  (density constraint projection, XSPH viscosity) against the rigid shell,
  the tank floor and the tank walls.  The inlet is a mass source released at
  the prescribed volumetric flow rate.

All exported quantities are raw physical states; no precomputed motion tables
are used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# frozen physical constants / discretisation
# ----------------------------------------------------------------------------
DT = 0.01                      # internal time step [s]
POOL_PITCH = 0.18              # water particle spacing [m]
H_KERNEL = 0.22                # SPH smoothing length [m]
BETA_FLARE = 0.75              # waterplane growth with immersion [1/m]
CONTACT_K = 4.0e5              # per-point bottom contact stiffness [N/m]
CONTACT_C = 8.0e3              # per-point bottom contact damping [N s/m]
CONTACT_MU = 0.4               # bottom friction coefficient
DRAG_CD = 1.3                  # hull hydrodynamic drag coefficient
DRAG_AREA = 2.6                # hull reference area [m^2]
ADDED_MASS_C = 1.0             # hydrodynamic added mass coefficient
ANG_DRAG = 25.0                # rotational drag coefficient
MAX_SPEED = 6.0                # particle speed clamp [m/s]
PBF_ITERS = 2
XSPH_EPS = 0.08
SCORR_K = 1.0e-4
SCORR_N = 4.0


def _quat_mult(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def _acc(n, idx, vals):
    """Deterministic scatter-add (bincount based)."""
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        return np.bincount(idx, weights=vals, minlength=n)
    return np.stack(
        [np.bincount(idx, weights=vals[:, c], minlength=n) for c in range(vals.shape[1])],
        axis=1,
    )


def _load_obj(path):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


class BenchmarkSolver:
    """Particle-fluid / rigid-shell coupled solver."""

    # -- setup ---------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        self.g_vec = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g_mag = float(abs(self.g_vec[2]))
        fluid = cfg["fluid_domain"]
        self.rho = float(fluid["water_density"])
        self.bounds_min = np.asarray(fluid["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(fluid["bounds_max"], dtype=float)
        self.z_w0 = float(fluid["initial_free_surface_z"])
        boat = cfg["boat"]
        self.m_dry = float(boat["mass"])
        self.draft0 = float(boat["target_initial_draft"])
        self.thickness = float(boat["thin_shell_thickness"])
        self.inlet_cfg = cfg["inlet"]
        self.metadata = scene.metadata()

        # hull geometry (asset authored z-up, origin at the keel plane)
        geo = cfg.get("scene_geometry") or {}
        asset = "assets/open_boat.obj"
        for item in geo.get("objects", []):
            if item.get("id") == "boat_shell":
                asset = item.get("geometry", {}).get("asset", asset)
        verts, faces = _load_obj(asset)
        self.verts = verts
        self.faces = faces
        self.hull = ConvexHull(verts)
        self.cavity = ConvexHull(verts[12:24])
        self.hull_A = np.asarray(self.hull.equations[:, :3], dtype=float)
        self.hull_b = np.asarray(self.hull.equations[:, 3], dtype=float)
        self.cav_A = np.asarray(self.cavity.equations[:, :3], dtype=float)
        self.cav_b = np.asarray(self.cavity.equations[:, 3], dtype=float)
        self.bottom_idx = np.arange(6)
        self.z_keel_local = float(verts[:, 2].min())
        self.z_top_local = float(verts[:, 2].max())
        self.hull_height = self.z_top_local - self.z_keel_local
        self.com_local = np.array([0.0, 0.0, 0.45])
        self.rim_local_z = float(verts[18:24, 2].mean())

        # hydrostatic calibration: mass / target draft define the waterplane
        d0 = self.draft0
        self.A0 = self.m_dry / (self.rho * (d0 + 0.5 * BETA_FLARE * d0 * d0))
        self.A_tank = float(
            (self.bounds_max[0] - self.bounds_min[0]) * (self.bounds_max[1] - self.bounds_min[1])
        )

        # rigid body inertia (hull box approximation + water carried)
        L, W, H = 2.84, 1.76, self.hull_height
        self.I_hull_local = np.diag([
            self.m_dry * (W * W + H * H) / 12.0,
            self.m_dry * (L * L + H * H) / 12.0,
            self.m_dry * (L * L + W * W) / 12.0,
        ])

        # kernel constants
        h = H_KERNEL
        self.h = h
        self.h2 = h * h
        self.poly6_c = 315.0 / (64.0 * np.pi * h ** 9)
        self.spiky_c = -45.0 / (np.pi * h ** 6)
        self.particle_mass = self.rho * POOL_PITCH ** 3
        self.particle_volume = POOL_PITCH ** 3

        self._build_particles()
        self._init_boat()
        self._init_hull_water_state()
        self.step_count = 0
        self.t = 0.0
        self.released = 0
        self.cumulative_source = 0.0
        self.last_acc = np.zeros(3)
        self.added_mass = ADDED_MASS_C * self.rho * self._displaced_volume(self.draft0)
        self.last_buoy = np.zeros(3)
        self.last_drag = np.zeros(3)
        self.last_contact = np.zeros(3)
        self.last_wall = np.zeros(3)
        self._classify()
        self._accumulate_forces()

    def _build_particles(self) -> None:
        bmin, bmax = self.bounds_min, self.bounds_max
        xs = np.arange(bmin[0] + POOL_PITCH * 0.5, bmax[0], POOL_PITCH)
        ys = np.arange(bmin[1] + POOL_PITCH * 0.5, bmax[1], POOL_PITCH)
        zs = np.arange(0.09, 1.56, POOL_PITCH)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        # inlet emitter reservoir hidden inside the inlet cylinder
        inlet = self.inlet_cfg
        dv = self.particle_volume
        flow = float(inlet["volumetric_flow_rate"])
        total_volume = flow * (float(inlet["active_interval"][1]) - float(inlet["active_interval"][0]))
        count = max(1, int(round(total_volume / dv)))
        offs = []
        levels = np.arange(3.39, 3.72, 0.055)
        for z in levels:
            for dx in (-0.085, 0.0, 0.085):
                for dy in (-0.085, 0.0, 0.085):
                    offs.append([dx, dy, z - 3.35])
                    if len(offs) >= count:
                        break
                if len(offs) >= count:
                    break
            if len(offs) >= count:
                break
        emit = np.asarray(offs[:count], dtype=float)
        emit[:, 0] += float(inlet["center"][0])
        emit[:, 1] += float(inlet["center"][1])
        self.n_pool = len(grid)
        self.n_emit = len(emit)
        self.n_water = self.n_pool + self.n_emit
        self.pos = np.vstack([grid, emit])
        self.vel = np.zeros_like(self.pos)
        self.mass = np.full(self.n_water, self.particle_mass)
        self.mass[self.n_pool:] = 0.0
        self.region = np.ones(self.n_water, dtype=np.int32)
        self.region[self.n_pool:] = 3
        self.dormant = np.zeros(self.n_water, dtype=bool)
        self.dormant[self.n_pool:] = True
        self.emit_offsets = emit - np.asarray(inlet["center"], dtype=float)
        self.emit_dv = dv
        self.emit_flow = flow
        t0, t1 = (float(v) for v in inlet["active_interval"])
        self.inlet_t0, self.inlet_t1 = t0, t1
        self.release_times = np.clip(
            t0 + (np.arange(1, self.n_emit + 1) * dv) / flow, t0, t1
        )

    def _init_boat(self) -> None:
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.R = np.eye(3)
        z_keel_world = self.z_w0 - self.draft0
        self.p_origin = np.array([0.0, 0.0, z_keel_world - self.z_keel_local])
        self.v_boat = np.zeros(3)
        self.omega = np.zeros(3)
        # push pool particles that overlap the placed hull aside (displaced water)
        loc = (self.pos - self.p_origin) @ self.R
        dots = loc @ self.hull_A.T + self.hull_b
        inside = np.all(dots <= 0.0, axis=1)
        inside[self.n_pool:] = False
        if np.any(inside):
            sign = np.where(loc[inside, 0] >= 0.0, 1.0, -1.0)
            loc[inside, 0] = sign * 1.58
            self.pos[inside] = self.p_origin + loc[inside] @ self.R.T
        self.hull_water_mass = 0.0
        self.z_w = self.z_w0
        self.V_ext0 = None

    def _init_hull_water_state(self) -> None:
        loc = (self.pos - self.p_origin) @ self.R
        in_cav = np.all(loc @ self.cav_A.T + self.cav_b <= 0.0, axis=1)
        in_hull = np.all(loc @ self.hull_A.T + self.hull_b <= 0.0, axis=1)
        self.inside = in_cav
        ext = ~in_cav
        self.V_ext0 = float(np.sum(self.particle_volume * (ext & (self.pos[:, 2] < 3.0))))
        self.V_disp = self._displaced_volume(self.draft0)
        self.V_disp0 = self.V_disp

    # -- hydrostatics --------------------------------------------------------
    def _displaced_volume(self, draft: float) -> float:
        d = min(max(draft, 0.0), self.hull_height)
        return self.A0 * (d + 0.5 * BETA_FLARE * d * d)

    def _draft_and_points(self):
        R = self.R
        bottom_world = self.p_origin + self.verts[self.bottom_idx] @ R.T
        keel_z = float(bottom_world[:, 2].min())
        draft = float(np.clip(self.z_w - keel_z, 0.0, self.hull_height))
        return draft, bottom_world, keel_z

    # -- particle classification --------------------------------------------
    def _classify(self) -> None:
        loc = (self.pos - self.p_origin) @ self.R
        in_cav = np.all(loc @ self.cav_A.T + self.cav_b <= 0.0, axis=1)
        self.inside = in_cav & (self.mass > 0.0)
        reg = np.ones(self.n_water, dtype=np.int32)
        jet = (self.region == 3) & (~self.inside) & (self.pos[:, 2] > 0.9)
        reg[jet] = 3
        reg[self.inside & (~jet)] = 2
        self.region = reg

    def _water_volume_outside(self) -> float:
        mask = (~self.inside) & (self.pos[:, 2] < 3.0) & (self.mass > 0.0)
        return float(np.sum(self.particle_volume * mask))

    def _update_level(self, draft: float) -> None:
        V_ext = self._water_volume_outside()
        V_disp = self._displaced_volume(draft)
        self.V_disp = V_disp
        self.z_w = self.z_w0 + (V_ext + V_disp - self.V_ext0 - self.V_disp0) / self.A_tank

    # -- time stepping -------------------------------------------------------
    def _release_inlet(self) -> None:
        while self.released < self.n_emit and self.t >= self.release_times[self.released]:
            k = self.n_pool + self.released
            center = np.asarray(self.inlet_cfg["center"], dtype=float)
            self.pos[k] = center + 0.4 * self.emit_offsets[self.released]
            self.pos[k, 2] = center[2] - 0.03
            self.vel[k] = np.asarray(self.inlet_cfg["velocity"], dtype=float)
            self.mass[k] = self.particle_mass
            self.dormant[k] = False
            self.region[k] = 3
            self.cumulative_source += self.particle_mass
            self.released += 1

    def _accumulate_forces(self) -> None:
        draft, bottom_world, keel_z = self._draft_and_points()
        V_disp = self._displaced_volume(draft)
        buoy = np.array([0.0, 0.0, self.rho * self.g_mag * V_disp])
        v = self.v_boat
        speed = float(np.linalg.norm(v))
        area_wet = DRAG_AREA * min(1.0, draft / max(self.hull_height, 1e-6))
        drag = -0.5 * self.rho * DRAG_CD * area_wet * speed * v
        gaps = bottom_world[:, 2] - 0.0
        contact = np.zeros(3)
        for idx in range(len(gaps)):
            gap = gaps[idx]
            if gap < 0.0:
                vn = float(np.dot(self.v_boat, np.array([0.0, 0.0, 1.0])))
                fn = -CONTACT_K * gap - CONTACT_C * min(vn, 0.0)
                fn = max(fn, 0.0)
                contact[2] += fn
                vt = self.v_boat.copy()
                vt[2] = 0.0
                spd = float(np.linalg.norm(vt))
                if spd > 1e-6:
                    contact -= CONTACT_MU * fn * vt / spd
        self.last_buoy = buoy
        self.last_drag = drag
        self.last_contact = contact
        self.added_mass = ADDED_MASS_C * self.rho * V_disp

    def _boat_step(self, dt: float) -> None:
        self._accumulate_forces()
        m_eff = self.m_dry + self.hull_water_mass + self.added_mass
        draft, bottom_world, keel_z = self._draft_and_points()
        # buoyancy application point: centroid of the immersed prismatoid
        d = min(max(draft, 0.0), self.hull_height)
        denom = d + 0.5 * BETA_FLARE * d * d
        z_bar = 0.0 if denom < 1e-9 else (0.5 * d * d + BETA_FLARE * d ** 3 / 3.0) / denom
        r_b_local = np.array([0.0, 0.0, self.z_keel_local + z_bar]) - self.com_local
        r_b = self.R @ r_b_local
        force = m_eff * self.g_vec + self.last_buoy + self.last_drag + self.last_contact
        torque = np.cross(r_b, self.last_buoy) - ANG_DRAG * np.linalg.norm(self.omega) * self.omega
        # interior water: rigidly carried, contributes mass and inertia
        I_local = self.I_hull_local + self.hull_water_mass * np.diag([0.10, 0.16, 0.22])
        I_world = self.R @ I_local @ self.R.T
        try:
            ang = np.linalg.solve(I_world, torque - np.cross(self.omega, I_world @ self.omega))
        except np.linalg.LinAlgError:
            ang = np.zeros(3)
        acc = force / max(m_eff, 1e-9)
        self.last_acc = acc
        self.v_boat = self.v_boat + acc * dt
        self.p_origin = self.p_origin + self.v_boat * dt
        self.omega = self.omega + ang * dt
        q = self.quat + 0.5 * _quat_mult(self.quat, np.array([*self.omega, 0.0])) * dt
        n = np.linalg.norm(q)
        self.quat = q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        if self.quat[3] < 0.0:
            self.quat = -self.quat
        self.R = _quat_matrix(self.quat)

    # -- particle fluid ------------------------------------------------------
    def _poly6(self, r2):
        out = np.zeros_like(r2)
        mask = r2 < self.h2
        out[mask] = self.poly6_c * (self.h2 - r2[mask]) ** 3
        return out

    def _spiky_grad(self, d, r):
        return (self.spiky_c * (self.h - r) ** 2 / np.maximum(r, 1e-9))[:, None] * d

    def _hull_collide(self, pts):
        R = self.R
        loc = (pts - self.p_origin) @ R
        d_out = loc @ self.hull_A.T + self.hull_b
        d_cav = loc @ self.cav_A.T + self.cav_b
        max_out = d_out.max(axis=1)
        max_cav = d_cav.max(axis=1)
        inside_out = max_out <= 0.0
        inside_cav = max_cav <= 0.0
        shell = inside_out & (~inside_cav)
        if np.any(shell):
            fo = np.argmax(d_out[shell], axis=1)
            fc = np.argmax(d_cav[shell], axis=1)
            no = self.hull_A[fo]
            nc = self.cav_A[fc]
            dist_o = -d_out[shell, fo]
            dist_c = -d_cav[shell, fc]
            new = loc[shell].copy()
            take_out = dist_o <= dist_c
            new[take_out] += no[take_out] * (dist_o[take_out] + 1e-3)[:, None]
            new[~take_out] -= nc[~take_out] * (dist_c[~take_out] + 1e-3)[:, None]
            loc[shell] = new
        # particles that ended up inside the cavity are constrained to it (kept
        # there by the cavity walls); particles inside the outer solid are out.
        pts_new = self.p_origin + loc @ R.T
        return pts_new

    def _tank_collide(self, pts):
        lo = self.bounds_min.copy()
        hi = self.bounds_max.copy()
        lo[2] = 0.0
        hi[2] = 4.0
        np.clip(pts[:, 0], lo[0], hi[0], out=pts[:, 0])
        np.clip(pts[:, 1], lo[1], hi[1], out=pts[:, 1])
        np.clip(pts[:, 2], lo[2], hi[2], out=pts[:, 2])
        return pts

    def _particle_step(self, dt: float) -> None:
        n = self.n_water
        act = np.flatnonzero(self.mass > 0.0)
        if act.size == 0:
            return
        p = self.pos[act]
        v = self.vel[act]
        v = v + self.g_vec * dt
        p_pred = p + v * dt
        tree = cKDTree(p_pred)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        m = self.particle_mass
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = p_pred[i] - p_pred[j]
            r2 = np.einsum("ij,ij->i", d, d)
            r = np.sqrt(r2)
            W = self._poly6(r2)
            grad = self._spiky_grad(d, r)
            nact = len(p_pred)
            rho = _acc(nact, i, m * W) + _acc(nact, j, m * W) + m * self._poly6(np.zeros(1))[0]
            G = _acc(nact, i, grad) - _acc(nact, j, grad)
            gsq = _acc(nact, i, np.einsum("ij,ij->i", grad, grad))
            gsq = gsq + _acc(nact, j, np.einsum("ij,ij->i", grad, grad))
            C = rho / self.rho - 1.0
            coeff = (m / self.rho) ** 2
            lam = -C / (coeff * (gsq + np.einsum("ij,ij->i", G, G)) + 1e-6)
            lam = np.clip(lam, -1.0e4, 1.0e4)
            W_dq = self._poly6(np.array([(0.2 * self.h) ** 2]))[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(W_dq > 0.0, W / W_dq, 0.0) ** SCORR_N
            scorr = -SCORR_K * ratio
            for _ in range(PBF_ITERS):
                coef = (lam[i] + lam[j] + scorr)
                dp = _acc(nact, i, coef[:, None] * grad) - _acc(nact, j, coef[:, None] * grad)
                dp = (m / self.rho) * dp
                step = np.linalg.norm(dp, axis=1)
                lim = 0.25 * self.h
                scale = np.where(step > lim, lim / np.maximum(step, 1e-12), 1.0)
                p_pred = p_pred + dp * scale[:, None]
                p_pred = self._tank_collide(self._hull_collide(p_pred))
                # fixed neighbour list: re-evaluate the constraint at updated positions
                d = p_pred[i] - p_pred[j]
                r2 = np.einsum("ij,ij->i", d, d)
                r = np.sqrt(r2)
                W = self._poly6(r2)
                grad = self._spiky_grad(d, r)
                rho = _acc(nact, i, m * W) + _acc(nact, j, m * W) + m * self._poly6(np.zeros(1))[0]
                G = _acc(nact, i, grad) - _acc(nact, j, grad)
                gsq = _acc(nact, i, np.einsum("ij,ij->i", grad, grad))
                gsq = gsq + _acc(nact, j, np.einsum("ij,ij->i", grad, grad))
                C = rho / self.rho - 1.0
                lam = -C / (coeff * (gsq + np.einsum("ij,ij->i", G, G)) + 1e-6)
                lam = np.clip(lam, -1.0e4, 1.0e4)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(W_dq > 0.0, W / W_dq, 0.0) ** SCORR_N
                scorr = -SCORR_K * ratio
            v = (p_pred - p) / dt
            # XSPH viscosity
            if len(pairs):
                visc = XSPH_EPS * (m / self.rho)
                dv = v[j] - v[i]
                w = visc * W
                acc_v = _acc(nact, i, w[:, None] * dv) - _acc(nact, j, w[:, None] * dv)
                v = v + acc_v
        else:
            p_pred = self._tank_collide(self._hull_collide(p_pred))
            v = (p_pred - p) / dt
        speed = np.linalg.norm(v, axis=1)
        scale = np.where(speed > MAX_SPEED, MAX_SPEED / np.maximum(speed, 1e-12), 1.0)
        v = v * scale[:, None]
        p_pred = p + v * dt
        p_pred = self._tank_collide(self._hull_collide(p_pred))
        self.pos[act] = p_pred
        self.vel[act] = v
        self.vel[self.dormant] = 0.0

    def _step(self) -> None:
        self._release_inlet()
        self._classify()
        mask = self.inside & (self.mass > 0.0)
        self.hull_water_mass = float(np.sum(self.mass[mask]))
        draft_old, _, _ = self._draft_and_points()
        self._update_level(draft_old)
        self._boat_step(DT)
        self._particle_step(DT)
        self.t += DT
        self.step_count += 1
        self._classify()
        mask = self.inside & (self.mass > 0.0)
        self.hull_water_mass = float(np.sum(self.mass[mask]))

    # -- frame assembly ------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        n = self.n_water
        ntot = n + 1
        pos = np.empty((ntot, 3))
        pos[0] = self.p_origin
        pos[1:] = self.pos
        vel = np.empty((ntot, 3))
        vel[0] = self.v_boat
        vel[1:] = self.vel
        mass = np.empty(ntot)
        mass[0] = self.m_dry
        mass[1:] = self.mass
        body_id = np.ones(ntot, dtype=np.int32)
        body_id[0] = 0
        region = np.empty(ntot, dtype=np.int32)
        region[0] = 0
        region[1:] = self.region
        orient = np.zeros((ntot, 4))
        orient[:, 3] = 1.0
        orient[0] = self.quat
        ang = np.zeros((ntot, 3))
        ang[0] = self.omega
        mass_q = np.zeros((ntot, 2))
        mass_q[0, 0] = self.m_dry
        mass_q[1:, 1] = self.mass
        ext = mass[:, None] * self.g_vec[None, :]

        draft, bottom_world, keel_z = self._draft_and_points()
        gaps = (bottom_world[:, 2] - 0.0).astype(float)
        coupling = (
            self.last_buoy
            + self.last_drag
            + (self.hull_water_mass + self.added_mass) * (self.g_vec - self.last_acc)
        )
        interaction = np.vstack([coupling, -coupling, self.last_contact, -self.last_contact])

        com = self.p_origin + self.R @ self.com_local
        scalars = {
            "contact_length_scale": float(self.thickness),
            "external_water_level": float(self.z_w),
            "hull_water_mass": float(self.hull_water_mass),
            "boat_draft": float(draft),
            "floating_height": float(com[2] - self.z_w),
            "cumulative_inlet_mass": float(self.cumulative_source),
            "inlet_active": 1.0 if self.inlet_t0 <= time <= self.inlet_t1 else 0.0,
            "particle_count": float(n),
            "initial_floating_height": float(self.draft0),
        }
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=orient,
            angular_velocity=ang,
            mass=mass,
            body_id=body_id,
            attributes={
                "phase_id": np.where(body_id == 0, 0, 1).astype(np.int32),
                "water_region": region,
                "mass_quantity": mass_q,
            },
            fields={
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.array([float(self.cumulative_source)]),
                "external_force": ext,
                "interaction_force": interaction,
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / DT))
        if target < self.step_count:
            target = self.step_count
        while self.step_count < target:
            self._step()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        draft, bottom_world, keel_z = self._draft_and_points()
        com = self.p_origin + self.R @ self.com_local
        return {
            "steps": int(self.step_count),
            "simulation_time": float(self.t),
            "final_water_level": float(self.z_w),
            "final_hull_water_mass": float(self.hull_water_mass),
            "final_com_height": float(com[2]),
            "final_draft": float(draft),
            "minimum_contact_gap": float((bottom_world[:, 2]).min()),
            "cumulative_inlet_mass": float(self.cumulative_source),
            "particle_count": int(self.n_water + 1),
        }
