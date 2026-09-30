"""Physics solver for scene 049_cube_drop_water.

Rigid cube (rigid body, quaternion rotation) dropped into a tank of water
modeled with Particle-based Fluid (PBF / position based dynamics) SPH-style
particles.  Coupling: boundary-density half-space compensation around the
cube, penetration projection with impulse exchange, buoyancy from sampled
local water density, and quadratic drag.  Deterministic sub-stepping with a
fixed micro timestep.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------
# kernels (poly6 value, spiky gradient)
class Kernels:
    def __init__(self, h: float):
        self.h = float(h)
        self.h2 = h * h
        self.k_poly6 = 315.0 / (64.0 * np.pi * h ** 9)
        self.k_spiky = -45.0 / (np.pi * h ** 6)

    def w(self, r2: np.ndarray) -> np.ndarray:
        r2 = np.maximum(r2, 0.0)
        out = np.zeros_like(r2)
        m = r2 < self.h2
        out[m] = self.k_poly6 * (self.h2 - r2[m]) ** 3
        return out

    def grad_w(self, d: np.ndarray) -> np.ndarray:
        """gradient of spiky kernel wrt the argument vector d (rij)."""
        r = np.linalg.norm(d, axis=-1)
        out = np.zeros_like(d)
        m = (r > 1e-12) & (r < self.h)
        rm = d[m] / r[m][:, None]
        out[m] = (self.k_spiky * (self.h - r[m]) ** 2)[:, None] * rm
        return out


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


class BenchmarkSolver:
    MICRO_DT = 0.002
    K_NEIGHBORS = 48
    PBF_ITERS = 3

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        cfg = scene.config
        self.config = cfg
        self.t = 0.0
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        # --- water particles / cube rigid body -------------------------
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.array(frame.body_id, dtype=np.int32)
        self.phase = np.array(frame.attributes["phase"], dtype=float)
        self.n_water = int(np.sum(self.body_id == 1))

        cube_cfg = cfg["initial_conditions"]["cube"]
        self.cube_m = float(cube_cfg["density"]) * float(cube_cfg["side_length"]) ** 3
        s = float(cube_cfg["side_length"])
        self.cube_e = 0.5 * s
        self.cube_I_body = np.full(3, self.cube_m * s * s / 6.0)
        self.cube_x = np.array(cube_cfg["initial_position"], dtype=float)
        self.cube_v = np.array(cube_cfg["initial_velocity"], dtype=float)
        self.cube_q = np.array(cube_cfg["initial_orientation_xyzw"], dtype=float)
        self.cube_q = self.cube_q / np.linalg.norm(self.cube_q)
        self.cube_w = np.zeros(3)

        self.pos = self.pos.copy()
        self.vel = self.vel.copy()
        # remove the cube row from particle arrays (cube handled as rigid body)
        self.water_x = self.pos[:self.n_water].copy()
        self.water_v = self.vel[:self.n_water].copy()
        self.water_m = self.mass[:self.n_water].copy()

        self.rho0 = float(cfg["initial_conditions"]["water"]["rest_density"])
        self.spacing = 0.12
        self.h = 2.0 * self.spacing
        self.kern = Kernels(self.h)
        self.particle_r = 0.5 * self.spacing

        # tank inner half extents (from scene_geometry)
        self.wall_x = 1.38
        self.wall_y = 1.08
        self.floor_z = 0.0

        # --- rest density measured on the initial lattice ----------------
        self.rho0_num = self._measure_rest_density()

        # --- half-space kernel deficit table -----------------------------
        # f(d): fraction of the (normalized) kernel mass lying in the
        # half-space beyond a planar boundary at distance d from particle.
        rng = np.random.default_rng(0)
        m = 400000
        smp = rng.uniform(-self.h, self.h, size=(m, 3))
        r2 = np.sum(smp * smp, axis=1)
        wts = self.kern.w(r2)
        inside = r2 < self.h2 if False else (r2 < self.h * self.h)
        wts = wts * inside
        tot = wts.sum()
        ds = np.linspace(0.0, self.h, 65)
        self.deficit_d = ds
        fs = []
        for d in ds:
            fs.append(wts[smp[:, 2] < -d].sum() / tot)
        self.deficit_f = np.array(fs)

        # cube sampling points (body frame) for buoyancy
        g = np.linspace(-self.cube_e, self.cube_e, 5)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        self.cube_samples = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)

        # interaction force bookkeeping (per material, K = 2)
        self.inter_force_water = np.zeros(3)   # on water from cube
        self.inter_force_cube = np.zeros(3)    # on cube from water
        self.inter_window = 0.0
        self.contact_scale = self.spacing
        self.n_pts = self.n_water + 1

    # ------------------------------------------------------------------
    def _measure_rest_density(self) -> float:
        tree = cKDTree(self.water_x)
        d, idx = tree.query(self.water_x, k=min(self.K_NEIGHBORS, len(self.water_x)))
        r2 = d * d
        rho = (self.kern.w(r2) * self.water_m[idx]).sum(axis=1)
        return float(np.median(rho))

    # ------------------------------------------------------------------
    def _cube_R(self) -> np.ndarray:
        return quat_to_mat(self.cube_q)

    def _cube_sd(self, pts: np.ndarray) -> np.ndarray:
        """signed distance of world points to cube surface (<0 inside)."""
        R = self._cube_R()
        loc = (pts - self.cube_x) @ R
        q = np.abs(loc) - self.cube_e
        qo = np.maximum(q, 0.0)
        outside = np.linalg.norm(qo, axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _cube_local_normal(self, pts: np.ndarray) -> np.ndarray:
        R = self._cube_R()
        loc = (pts - self.cube_x) @ R
        q = np.abs(loc) - self.cube_e
        # face with largest penetration (or closest face outside)
        face = np.argmax(q, axis=1)
        n_loc = np.zeros_like(loc)
        for k in range(3):
            msk = face == k
            n_loc[msk, k] = np.sign(loc[msk, k])
        return n_loc @ R.T

    # ------------------------------------------------------------------
    def _boundary_deficit(self, x: np.ndarray, cube_R: np.ndarray,
                          cube_only: bool = False) -> np.ndarray:
        """estimated missing kernel mass fraction near planar walls + cube."""
        tot = np.zeros(len(x))
        def add(d):
            f = np.interp(np.clip(d, 0.0, self.h), self.deficit_d, self.deficit_f)
            return f
        if not cube_only:
            tot += add(x[:, 2] - self.floor_z)          # floor
            tot += add(self.wall_x - np.abs(x[:, 0]))   # side walls x
            tot += add(self.wall_y - np.abs(x[:, 1]))   # side walls y
        # cube (locally flat approximation)
        loc = (x - self.cube_x) @ cube_R
        q = np.abs(loc) - self.cube_e
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        sd = outside + inside
        tot += add(np.maximum(sd, 0.0))
        return np.minimum(tot, 0.95)

    # ------------------------------------------------------------------
    def _neighbors(self, x: np.ndarray):
        tree = cKDTree(x)
        k = min(self.K_NEIGHBORS, len(x))
        d, idx = tree.query(x, k=k)
        return d, idx

    # ------------------------------------------------------------------
    def _water_pbf_step(self, dt: float, cube_R: np.ndarray) -> None:
        h = self.h
        x_old = self.water_x.copy()
        v = self.water_v + self.gravity * dt
        x = self.water_x + v * dt

        d, idx = self._neighbors(x)
        r2 = d * d
        w = self.kern.w(r2)
        m_j = self.water_m[idx]
        rho = (w * m_j).sum(axis=1)
        rho += self.rho0_num * self._boundary_deficit(x, cube_R)

        # spiky gradient for each pair (i, idx[i,k])
        diff = x[:, None, :] - x[idx]                     # (N,k,3)
        gw = self.kern.grad_w(diff)                       # (N,k,3)
        gw_norm2 = np.sum(gw * gw, axis=2)                # (N,k)

        C = rho / self.rho0_num - 1.0
        C = np.maximum(C, -0.02)
        sum_grad2 = (np.sum(gw, axis=1) ** 2).sum(axis=1) + gw_norm2.sum(axis=1)
        lam = -C / (sum_grad2 / (self.rho0_num ** 2) + 1e-6)

        for _ in range(self.PBF_ITERS):
            # refresh density / gradients / lambda at current positions
            diff = x[:, None, :] - x[idx]
            dcur = np.linalg.norm(diff, axis=2)
            w = self.kern.w(dcur * dcur)
            rho = (w * m_j).sum(axis=1)
            rho += self.rho0_num * self._boundary_deficit(x, cube_R)
            gw = self.kern.grad_w(diff)
            C = np.maximum(rho / self.rho0_num - 1.0, -0.02)
            sum_grad2 = (np.sum(gw, axis=1) ** 2).sum(axis=1) + \
                np.sum(gw * gw, axis=2).sum(axis=1)
            lam = -C / (sum_grad2 / (self.rho0_num ** 2) + 1e-6)

            lw = (lam[:, None] + lam[idx])                # (N,k)
            dp = (1.0 / self.rho0_num) * np.sum(lw[:, :, None] * gw, axis=1)
            nrm = np.linalg.norm(dp, axis=1)
            fac = np.ones_like(nrm)
            big = nrm > 0.3 * h
            fac[big] = (0.3 * h) / nrm[big]
            x = x + dp * fac[:, None]

            # project out of cube and back inside tank
            x = self._project_solids(x, cube_R)

        v = (x - x_old) / dt
        # damped solid contacts for velocity
        v = self._contact_velocity(x, v, x_old, cube_R, dt)
        # mild viscosity (XSPH-like) using precomputed neighbor idx
        if len(x) > 1:
            nv = np.sum(self.water_v[idx], axis=1) / idx.shape[1]
            v = 0.92 * v + 0.08 * nv
        speed = np.linalg.norm(v, axis=1)
        sp = np.minimum(speed, 15.0)
        v = v * (sp / np.maximum(speed, 1e-12))[:, None]

        self.water_x = x
        self.water_v = v

    def _project_solids(self, x: np.ndarray, cube_R: np.ndarray) -> np.ndarray:
        # tank
        x[:, 2] = np.maximum(x[:, 2], self.floor_z + self.particle_r)
        x[:, 0] = np.clip(x[:, 0], -self.wall_x + self.particle_r,
                          self.wall_x - self.particle_r)
        x[:, 1] = np.clip(x[:, 1], -self.wall_y + self.particle_r,
                          self.wall_y - self.particle_r)
        x[:, 2] = np.minimum(x[:, 2], 2.4)
        # cube
        sd = self._cube_sd(x)
        bad = sd < self.particle_r
        if np.any(bad):
            pb = x[bad]
            nrm = self._cube_local_normal(pb)
            target = self.cube_e + self.particle_r
            loc = (pb - self.cube_x) @ cube_R
            q = np.abs(loc) - target
            qo = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            qin = np.minimum(np.max(q, axis=1), 0.0)
            sd_t = qo + qin
            # push along face normal to the target surface
            x[bad] = pb + nrm * (self.particle_r - sd)[:, None]
        return x

    def _contact_velocity(self, x, v, x_old, cube_R, dt):
        # walls / floor: kill inward normal velocity where clamped
        pad = self.particle_r + 1e-9
        m = x[:, 2] <= self.floor_z + pad
        v[m, 2] = np.maximum(v[m, 2], 0.0) * 0.1
        m = x[:, 0] <= -self.wall_x + pad
        v[m, 0] = np.maximum(v[m, 0], 0.0)
        m = x[:, 0] >= self.wall_x - pad
        v[m, 0] = np.minimum(v[m, 0], 0.0)
        m = x[:, 1] <= -self.wall_y + pad
        v[m, 1] = np.maximum(v[m, 1], 0.0)
        m = x[:, 1] >= self.wall_y - pad
        v[m, 1] = np.minimum(v[m, 1], 0.0)
        return v

    # ------------------------------------------------------------------
    def _cube_fluid_forces(self) -> tuple:
        """buoyancy + drag on the cube from the water."""
        R = self._cube_R()
        pts = self.cube_samples @ R.T + self.cube_x
        # local water density at the sample points
        tree = cKDTree(self.water_x)
        d, idx = tree.query(pts, k=min(8, self.n_water))
        rho_loc = (self.kern.w(d * d) * self.water_m[idx]).sum(axis=1)
        sub = np.clip(rho_loc / self.rho0_num, 0.0, 1.0)
        frac = float(np.mean(sub))
        if frac <= 1e-6:
            return np.zeros(3), np.zeros(3), 0.0, self.cube_x.copy()
        wsub = sub / max(sub.sum(), 1e-12)
        centroid = (pts * wsub[:, None]).sum(axis=0)

        V = (2 * self.cube_e) ** 3
        F_b = np.array([0.0, 0.0, self.rho0 * 9.81 * V * frac])

        # drag against surrounding water velocity
        near = self._cube_sd(self.water_x) < 0.45
        if np.any(near) and near.sum() > 3:
            v_rel = self.cube_v - self.water_v[near].mean(axis=0)
        else:
            v_rel = self.cube_v.copy()
        A = (2 * self.cube_e) ** 2
        F_d = -0.5 * self.rho0 * 1.1 * A * frac * np.linalg.norm(v_rel) * v_rel
        cap = 4.0 * self.cube_m * 9.81
        if np.linalg.norm(F_d) > cap:
            F_d = F_d * (cap / np.linalg.norm(F_d))
        F_d[2] += -0.8 * self.cube_m * 9.81 * frac * 0.0  # no extra term
        return F_b, F_d, frac, centroid

    def _apply_impulse_cube(self, J: np.ndarray, p: np.ndarray):
        self.cube_v += J / self.cube_m
        r = p - self.cube_x
        dL = np.cross(r, J)
        R = self._cube_R()
        Iw = R @ np.diag(self.cube_I_body) @ R.T
        self.cube_w += np.linalg.solve(Iw, dL)

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        cube_R = self._cube_R()

        # ---- cube forces: gravity + buoyancy + drag -------------------
        F_b, F_d, frac, centroid = self._cube_fluid_forces()
        F_coupling = F_b + F_d
        F = self.cube_m * self.gravity + F_coupling
        # rotational damping in water
        self.cube_w *= (1.0 - min(0.9, 3.0 * dt) * frac)

        # ---- water prediction / PBF (cube treated as moving solid) ----
        x_cube_old = self.cube_x.copy()
        # integrate cube first (semi-implicit)
        self.cube_v += F / self.cube_m * dt
        self.cube_x += self.cube_v * dt
        # quaternion update
        wlen = np.linalg.norm(self.cube_w)
        if wlen > 1e-12:
            axis = self.cube_w / wlen
            half = 0.5 * wlen * dt
            dq = np.array([axis[0] * np.sin(half), axis[1] * np.sin(half),
                           axis[2] * np.sin(half), np.cos(half)])
            self.cube_q = quat_mul(dq, self.cube_q)
            self.cube_q /= np.linalg.norm(self.cube_q)
        cube_R = self._cube_R()

        self._water_pbf_step(dt, cube_R)

        # ---- water <-> cube penetration impulses ----------------------
        sd = self._cube_sd(self.water_x)
        bad = sd < self.particle_r - 1e-9
        if np.any(bad):
            pb = self.water_x[bad]
            vb = self.water_v[bad]
            nrm = self._cube_local_normal(pb)
            push = (self.particle_r - sd[bad])[:, None] * nrm
            self.water_x[bad] = pb + push
            # surface velocity of the cube at those points
            r = pb - self.cube_x
            vs = self.cube_v + np.cross(self.cube_w, r)
            vrel = vb - vs
            vn = np.sum(vrel * nrm, axis=1)
            dv = np.minimum(vn, 0.0)[:, None] * nrm      # remove approaching
            vb_new = vb - dv + 0.35 * np.minimum(vn, 0.0)[:, None] * nrm
            self.water_v[bad] = vb_new
            Jtot = -((vb_new - vb) * self.water_m[bad, None]).sum(axis=0)
            # distribute to the cube as impulses at contact points
            for J_i, p_i, m_i in zip(-(vb_new - vb) * self.water_m[bad, None],
                                     self.water_x[bad], self.water_m[bad]):
                self._apply_impulse_cube(J_i, p_i)
            self.inter_force_water -= Jtot / dt
            self.inter_force_cube += Jtot / dt

        # accumulate coupling force on cube
        self.inter_force_cube += F_coupling
        self.inter_force_water -= F_coupling
        self.inter_window += dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9 and guard < 20000:
            dt = min(self.MICRO_DT, target - self.t)
            self._step(dt)
            self.t += dt
            guard += 1
        self.t = target
        return self._make_frame(target)

    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> "StateFrame":
        from worldsolver.contracts import StateFrame

        n_w = self.n_water
        pos = np.vstack((self.water_x, self.cube_x[None, :]))
        vel = np.vstack((self.water_v, self.cube_v[None, :]))
        mass = np.concatenate((self.water_m, [self.cube_m]))
        body_id = np.concatenate((np.ones(n_w, dtype=np.int32), [2]))
        phase = np.concatenate((np.zeros(n_w), [1.0]))
        orient = np.zeros((n_w + 1, 4))
        orient[:, 3] = 1.0
        orient[-1] = self.cube_q
        angv = np.zeros((n_w + 1, 3))
        angv[-1] = self.cube_w

        # external force: gravity only
        ext = np.repeat(self.gravity[None, :], n_w + 1, axis=0)
        ext[-1] = ext[-1] * self.cube_m
        ext[:n_w] = ext[:n_w] * self.water_m[:, None]

        # interaction forces (per material, averaged over window)
        if self.inter_window > 0:
            fw = self.inter_force_water / self.inter_window
            fc = self.inter_force_cube / self.inter_window
        else:
            fw = np.zeros(3)
            fc = np.zeros(3)
        self.inter_force_water[:] = 0.0
        self.inter_force_cube[:] = 0.0
        self.inter_window = 0.0
        interaction = np.vstack((fw, fc))

        # contact signed distances: water near walls / floor / cube,
        # plus cube near walls / floor.
        gap_wall = np.minimum(
            np.minimum(self.water_x[:, 2] - self.floor_z,
                       self.wall_x - np.abs(self.water_x[:, 0])),
            self.wall_y - np.abs(self.water_x[:, 1]))
        gap_cube = self._cube_sd(self.water_x)
        contacts = []
        thr = 2.0 * self.spacing
        contacts.append(gap_wall[gap_wall < thr])
        contacts.append(gap_cube[gap_cube < thr])
        cgap = min(self.wall_x - abs(self.cube_x[0]) - self.cube_e,
                   self.wall_y - abs(self.cube_x[1]) - self.cube_e,
                   self.cube_x[2] - self.floor_z - self.cube_e)
        contacts.append(np.array([cgap]))
        csd = np.concatenate([c for c in contacts if len(c)]) if any(
            len(c) for c in contacts) else np.array([10.0])
        csd = np.atleast_1d(np.asarray(csd, dtype=float))
        if not np.all(np.isfinite(csd)):
            csd = np.array([10.0])

        frame = StateFrame(
            time=t,
            position=pos,
            velocity=vel,
            orientation=orient,
            angular_velocity=angv,
            mass=mass,
            body_id=body_id,
            attributes={"phase": phase, "mass_quantity": mass},
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(2),
                "external_force": ext,
                "interaction_force": interaction,
            },
            scalars={
                "contact_length_scale": float(self.contact_scale),
                "water_rest_density": float(self.rho0),
                "cube_density": 400.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "PBF SPH water + rigid cube with impulse coupling",
            "particles": int(self.n_water),
            "micro_dt": self.MICRO_DT,
        }
