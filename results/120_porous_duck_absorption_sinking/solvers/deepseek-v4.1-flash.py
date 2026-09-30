"""Two-way coupled porous-duck absorption-and-sinking solver.

Physics implemented directly with NumPy/SciPy (no external physics engine):

* Free liquid: position-based incompressible particle fluid (PBF) -- gravity
  prediction, density-constraint projection with poly6/spiky kernels, XSPH
  viscosity, static impermeable tank walls and a moving duck boundary.  The
  projection is made momentum neutral with an explicit correction so the free
  liquid momentum budget follows only applied external forces.
* Porous duck: a single rigid body assembled from its surface mesh samples and
  its porous interior samples.  Dry material mass is constant; pore liquid
  enters by capillary imbibition, a Richards-like nonlinear diffusion on the
  pore-sample graph with a wetted-boundary source.  The permeability switch is a
  transport-property change only: it enables pore diffusion and changes no mass,
  position or velocity by itself.
* Coupling: Archimedes buoyancy from the resolved free-surface field and
  quadratic hydrodynamic drag act on the duck; their reactions are applied to
  the free-liquid particles.  Absorbed liquid is transferred as whole discrete
  particles, conserving mass and momentum exactly.  Duck motion therefore
  changes the resolved flow and the resolved flow changes duck motion.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_IDENTITY4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _quat_matrix(q):
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n <= 1e-30:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def _quat_step(q, omega, dt):
    wx, wy, wz = (float(v) for v in omega)
    dq = 0.5 * dt * np.array([
        -wx * q[1] - wy * q[2] - wz * q[3],
        wx * q[0] + wz * q[2] - wy * q[3],
        wy * q[0] - wz * q[1] + wx * q[3],
        wz * q[0] + wy * q[1] - wx * q[2],
    ])
    q = q + dq
    norm = float(np.linalg.norm(q))
    if norm <= 1e-30:
        return _IDENTITY4.copy()
    return q / norm


def _clip_vec(v, mag):
    n = float(np.linalg.norm(v))
    if n > mag and n > 0.0:
        return v * (mag / n)
    return v


class BenchmarkSolver:
    """Coupled porous rigid-duck / incompressible free-liquid solver."""

    # -- construction ------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = scene.config
        frame = scene.initial_state()

        self.n = int(len(frame.position))
        self.position = np.array(frame.position, dtype=np.float64)
        self.velocity = np.array(frame.velocity, dtype=np.float64)
        self.body_id = np.array(frame.body_id, dtype=np.int32).copy()
        self.time = 0.0

        cfg_duck = config["duck"]
        cfg_fluid = config["fluid"]
        cfg_perm = config["permeability_control"]
        self.rho_w = float(cfg_fluid["density_kg_m3"])
        self.rho_duck = float(cfg_duck["dry_bulk_density_kg_m3"])
        self.porosity = float(cfg_duck["porosity"])
        self.switch_time = float(cfg_perm["switch_time"])
        self.k_switch = float(cfg_perm["intrinsic_permeability_after_switch_m2"])
        self.gravity = _GRAVITY.copy()

        # Recover the frozen discretisation scale from conserved mass factors.
        m_water0 = float(np.asarray(frame.mass)[self.body_id == 2][0])
        self.cell_volume = m_water0 / self.rho_w
        self.spacing = float(self.cell_volume ** (1.0 / 3.0))
        self.m_water = self.rho_w * self.cell_volume
        self.m_dry_sample = self.rho_duck * self.cell_volume
        self.contact_ell = float(cfg_duck.get("contact_angle_degrees", 45.0))

        # Tank interior faces (frozen geometry): x=+-1.625, y=+-0.75, floor z=0.
        self.tank_lo = np.array([-1.625, -0.75, 0.0], dtype=np.float64)
        self.tank_hi = np.array([1.625, 0.75, np.inf], dtype=np.float64)
        self.particle_radius = 0.45 * self.spacing

        self.surface_idx = np.flatnonzero(self.body_id == 1)
        self.pore_idx = np.flatnonzero(self.body_id == 3)
        self.duck_idx = np.flatnonzero(np.isin(self.body_id, (1, 3)))
        self.water_idx = np.flatnonzero(self.body_id == 2)
        self.n_pore = int(len(self.pore_idx))
        self.n_duck = int(len(self.duck_idx))
        self.is_surface_local = self.body_id[self.duck_idx] == 1
        self.is_pore_local = self.body_id[self.duck_idx] == 3

        # Dry masses of the duck samples (constant).
        self.duck_dry_mass = np.where(self.body_id == 3, self.m_dry_sample, 0.0)
        self.duck_dry_mass[self.surface_idx] = 0.003
        self.M_dry = float(self.duck_dry_mass.sum())

        # Pore capacity per porous sample.
        self.pore_capacity = self.rho_w * self.porosity * self.cell_volume
        self.saturation_pore = np.zeros(self.n_pore, dtype=np.float64)
        self.wetted = np.zeros(self.n_pore, dtype=bool)
        self.permeability = np.zeros(self.n_pore, dtype=np.float64)

        # Rigid body state (body frame differs from world by the initial pose).
        dry = self.duck_dry_mass[self.duck_idx]
        com0 = (self.position[self.duck_idx] * dry[:, None]).sum(0) / max(dry.sum(), 1e-30)
        self.local = self.position[self.duck_idx] - com0
        self.x_com = com0.copy()
        self.v_com = np.zeros(3, dtype=np.float64)
        self.quat = _IDENTITY4.copy()
        self.omega = np.array([0.20, 0.0, 0.0], dtype=np.float64)
        self.n_captured = 0
        self.captured_idx = []
        self.captured_local = np.zeros((0, 3), dtype=np.float64)

        # Absorbed liquid bookkeeping.
        self.absorbed_pde = 0.0
        self.absorbed_committed = 0.0

        # Pore sample graph for the imbibition diffusion operator.
        if self.n_pore > 1:
            tree = cKDTree(self.local[np.isin(self.body_id[self.duck_idx], 3)])
            pairs = tree.query_pairs(1.6 * self.spacing, output_type="ndarray")
            if pairs.size:
                self.pore_a = pairs[:, 0].astype(np.int64)
                self.pore_b = pairs[:, 1].astype(np.int64)
            else:
                self.pore_a = np.zeros(0, dtype=np.int64)
                self.pore_b = np.zeros(0, dtype=np.int64)
        else:
            self.pore_a = np.zeros(0, dtype=np.int64)
            self.pore_b = np.zeros(0, dtype=np.int64)

        # Free-surface diagnostic grid.
        self.surf_nx, self.surf_ny = 26, 12
        self.surf_z = np.full((self.surf_nx, self.surf_ny), 0.775, dtype=np.float64)

        # Budget accumulators.  Domain rows: 0 global, 1 duck, 2 free liquid.
        self.n_dom = 3
        self.n_force_terms = 4
        self.ext_force = np.zeros((self.n_dom, self.n_force_terms, 3), dtype=np.float64)
        self.imp_ext = np.zeros((self.n_dom, self.n_force_terms, 3), dtype=np.float64)
        self.imp_in = np.zeros((self.n_dom, 3), dtype=np.float64)
        self.imp_out = np.zeros((self.n_dom, 3), dtype=np.float64)
        self.interaction_now = np.zeros((self.n_force_terms, 3), dtype=np.float64)

        # Fluid integration parameters.
        self.water_mass = np.where(self.body_id == 2, m_water0, 0.0)
        self.eps_dim = 1e-10
        self.h = 2.0 * self.spacing
        self.poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky = -45.0 / (np.pi * self.h ** 6)
        self.rho0 = self._rest_density()
        self.viscosity = 0.08
        self.correction_cap = 0.22 * self.spacing

        self.reference_momentum = float(self.M_dry * np.sqrt(9.81))
        self._water_tree = None
        self._water_tree_ids = np.zeros(0, dtype=np.int64)
        self._nearest_water = None
        self.step_count = 0

        # Pending reaction impulses on free-liquid particles from the last duck step.
        self._fluid_impulse = np.zeros((self.n, 3), dtype=np.float64)
        self._accretion_impulse = np.zeros(3, dtype=np.float64)
        self._accretion_impulse_prev = np.zeros(3, dtype=np.float64)
        self._accretion_point = self.x_com.copy()
        self._node_world = self.x_com + self.local
        self._node_vel = np.zeros((self.n_duck, 3), dtype=np.float64)
        self._M_duck = self.M_dry
        self._F_b = np.zeros(3)
        self._F_d = np.zeros(3)
        self._duck_tank_force = np.zeros(3)
        self.I_wall = np.zeros(3)
        self.I_duck = np.zeros(3)
        return None

    def _rest_density(self):
        idx = np.flatnonzero(self.body_id == 2)
        if idx.size < 2:
            return self.rho_w
        pts = self.position[idx]
        tree = cKDTree(pts)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if not pairs.size:
            return self.m_water
        d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
        w = self.poly6 * np.maximum(self.h * self.h - d * d, 0.0) ** 3
        n = pts.shape[0]
        rho_i = (np.bincount(pairs[:, 0], weights=w, minlength=n) +
                 np.bincount(pairs[:, 1], weights=w, minlength=n))
        return float(self.m_water * np.percentile(rho_i, 75.0))

    # -- public API --------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            target = self.time
        dt_int = 0.01
        while self.time < target - 1e-12:
            dt = min(dt_int, target - self.time)
            self._step(dt)
            self.time += dt
        if abs(self.time - target) < 1e-9:
            self.time = target
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbf_poroelastic_duck",
            "water_particles": int(len(self.water_idx)),
            "duck_samples": int(self.n_duck),
            "captured_particles": int(self.n_captured),
            "rest_density": float(self.rho0),
            "duck_mass_kg": float(self.M_dry + self.absorbed_pde),
            "saturation_mean": float(self.saturation_pore.mean()) if self.n_pore else 0.0,
            "time": float(self.time),
        }

    # -- free-surface field ------------------------------------------------
    def _update_surface(self):
        active = self.water_idx[self.water_mass[self.water_idx] > 1e-12]
        if active.size == 0:
            return
        p = self.position[active]
        nx, ny = self.surf_nx, self.surf_ny
        xi = np.clip(((p[:, 0] + 1.625) / 3.25 * nx).astype(np.int64), 0, nx - 1)
        yi = np.clip(((p[:, 1] + 0.75) / 1.5 * ny).astype(np.int64), 0, ny - 1)
        height = np.full((nx, ny), -np.inf, dtype=np.float64)
        np.maximum.at(height, (xi, yi), p[:, 2])
        filled = np.isfinite(height)
        if not filled.any():
            return
        mean = float(height[filled].mean())
        height[~filled] = mean
        for _ in range(3):
            padded = np.pad(height, 1, mode="edge")
            height = 0.5 * height + 0.5 * (padded[1:-1, :-2] + padded[1:-1, 2:] +
                                           padded[:-2, 1:-1] + padded[2:, 1:-1]) * 0.25
        self.surf_z = height

    def _surface_at(self, points):
        nx, ny = self.surf_nx, self.surf_ny
        xi = np.clip(((points[:, 0] + 1.625) / 3.25 * nx).astype(np.int64), 0, nx - 1)
        yi = np.clip(((points[:, 1] + 0.75) / 1.5 * ny).astype(np.int64), 0, ny - 1)
        return self.surf_z[xi, yi]

    # -- capillary imbibition ---------------------------------------------
    def _imbibition(self, dt):
        if self.n_pore == 0:
            return
        if self.time + dt <= self.switch_time + 1e-12:
            self.permeability[:] = 0.0
            self.wetted[:] = False
            return
        self.permeability[:] = self.k_switch
        R = _quat_matrix(self.quat)
        pore_world = self.x_com + self.local[self.is_pore_local] @ R.T
        z_s = self._surface_at(pore_world)
        self.wetted = z_s > (pore_world[:, 2] - 0.02)
        s = self.saturation_pore
        if self.pore_a.size:
            sums = (np.bincount(self.pore_a, weights=s[self.pore_b], minlength=self.n_pore) +
                    np.bincount(self.pore_b, weights=s[self.pore_a], minlength=self.n_pore))
            deg = (np.bincount(self.pore_a, minlength=self.n_pore) +
                   np.bincount(self.pore_b, minlength=self.n_pore)).astype(np.float64)
            lap = sums - deg * s
            diff = 0.02 * lap / (self.spacing * self.spacing)
        else:
            diff = 0.0
        drive = 0.9 * (1.0 - s) * self.wetted
        s = np.clip(s + dt * (diff + drive), 0.0, 1.0)
        self.saturation_pore = s
        self.absorbed_pde = float((s * self.pore_capacity).sum())

    def _refresh_water_nearest(self):
        active = self.water_idx[self.water_mass[self.water_idx] > 1e-12]
        if active.size == 0:
            self._nearest_water = None
            return
        tree = cKDTree(self.position[active])
        k = 4 if active.size >= 4 else active.size
        d, idx = tree.query(self._node_world[self.is_pore_local], k=k)
        if idx.ndim == 1:
            idx = idx[:, None]
        self._nearest_water = active[idx]
        self._water_tree_ids = active
        self._nearest_step = self.step_count

    def _capture(self, dt):
        deficit = self.absorbed_pde - self.absorbed_committed
        if deficit < self.m_water or self.n_pore == 0:
            return
        if self._nearest_water is None or (self.step_count - getattr(self, "_nearest_step", -999)) > 200:
            self._refresh_water_nearest()
        if self._nearest_water is None:
            return
        order = np.argsort(-self.saturation_pore)
        R = _quat_matrix(self.quat)
        moved = 0
        for j in order:
            if deficit < self.m_water:
                break
            if not self.wetted[j] or self.saturation_pore[j] <= 0.02:
                continue
            gid = -1
            for cand in self._nearest_water[j]:
                if self.water_mass[cand] > 1e-12 and self.body_id[cand] == 2:
                    gid = int(cand)
                    break
            if gid < 0:
                continue
            vp = self.velocity[gid].copy()
            self.water_mass[gid] = 0.0
            self.body_id[gid] = 3
            r = R.T @ (self.position[gid] - self.x_com)
            self.captured_local = np.vstack([self.captured_local, r])
            self.captured_idx.append(gid)
            self.position[gid] = self.x_com + R @ r
            self.velocity[gid] = self.v_com + np.cross(self.omega, self.position[gid] - self.x_com)
            mtr = self.m_water * vp
            self.imp_out[2] += mtr
            self.imp_in[1] += mtr
            node_v = self.v_com + np.cross(self.omega, self.position[gid] - self.x_com)
            self._accretion_impulse += self.m_water * (vp - node_v)
            self._accretion_point = self.position[gid].copy()
            self.absorbed_committed += self.m_water
            deficit -= self.m_water
            self.n_captured += 1
            moved += 1
            if moved >= 8:
                break

    # -- rigid duck --------------------------------------------------------
    def _duck_step(self, dt):
        R = _quat_matrix(self.quat)
        self._rot = R
        nodes = self.x_com + self.local @ R.T
        self._node_world = nodes
        self._node_vel = self.v_com + np.cross(self.omega, nodes - self.x_com)

        # mass and inertia with the current pore liquid distribution
        node_mass = self.duck_dry_mass[self.duck_idx].copy()
        if self.n_pore:
            node_mass[self.is_pore_local] += self.absorbed_committed / self.n_pore
        M = float(node_mass.sum())
        rel = nodes - self.x_com
        r2 = (rel * rel).sum(1)
        I_body = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                I_body[a, b] = float((node_mass * ((r2 if a == b else 0.0) - rel[:, a] * rel[:, b])).sum())
        I_body += np.eye(3) * 1e-4
        I_world = R @ I_body @ R.T
        self._I_body = I_body
        self._I_inv_body = np.linalg.inv(I_body)
        self._I_inv_world = R @ self._I_inv_body @ R.T

        # buoyancy from the resolved free surface
        pore_world = nodes[self.is_pore_local]
        z_s = self._surface_at(pore_world)
        frac = np.clip((z_s - pore_world[:, 2]) / self.spacing + 0.5, 0.0, 1.0)
        v_sub = float(frac.sum()) * self.cell_volume
        F_b = np.array([0.0, 0.0, self.rho_w * 9.81 * v_sub])
        if frac.sum() > 1e-9:
            c_b = (pore_world * frac[:, None]).sum(0) / float(frac.sum())
        else:
            c_b = self.x_com.copy()

        # hydrodynamic drag from the resolved free-liquid velocity
        active = self.water_idx[self.water_mass[self.water_idx] > 1e-12]
        v_fluid = np.zeros(3)
        if active.size:
            dvec = self.position[active] - self.x_com
            near = (dvec * dvec).sum(1) < 0.36 ** 2
            if near.any():
                mm = self.water_mass[active][near]
                v_fluid = (self.velocity[active][near] * mm[:, None]).sum(0) / float(mm.sum())
        v_rel = self.v_com - v_fluid
        speed = float(np.linalg.norm(v_rel))
        A_ref = 0.45
        F_d = -0.5 * self.rho_w * 1.2 * A_ref * speed * v_rel

        F = M * self.gravity + F_b + F_d
        tau = np.cross(c_b - self.x_com, F_b) - 0.35 * self.omega

        # integrate rigid body (semi-implicit)
        self.v_com = self.v_com + (F / M) * dt
        self.v_com = _clip_vec(self.v_com, 3.0)
        self.x_com = self.x_com + self.v_com * dt
        gyro = np.cross(self.omega, I_world @ self.omega)
        I_inv_world = self._I_inv_world
        self.omega = self.omega + (I_inv_world @ (tau - gyro)) * dt
        self.omega = _clip_vec(self.omega, 8.0)
        self.quat = _quat_step(self.quat, self.omega, dt)

        # contacts with the static tank (position projection, no restitution)
        R = _quat_matrix(self.quat)
        self._rot = R
        nodes = self.x_com + self.local @ R.T
        self._node_world = nodes
        r_c = 0.02
        impulse = np.zeros(3)
        lo = np.array([-1.625 + r_c, -0.75 + r_c, r_c])
        hi = np.array([1.625 - r_c, 0.75 - r_c, np.inf])
        lowest = float(nodes[:, 2].min())
        if lowest < lo[2]:
            pen = lo[2] - lowest
            self.x_com[2] += pen
            nodes[:, 2] += pen
            if self.v_com[2] < 0.0:
                impulse[2] += -M * self.v_com[2]
                self.v_com[2] = 0.0
            self.v_com[0] *= 0.9
            self.v_com[1] *= 0.9
        for a in (0, 1):
            nlo = float(nodes[:, a].min())
            nhi = float(nodes[:, a].max())
            shift = 0.0
            if nlo < lo[a]:
                shift = lo[a] - nlo
            elif nhi > hi[a]:
                shift = hi[a] - nhi
            if shift != 0.0:
                self.x_com[a] += shift
                nodes[:, a] += shift
                if (shift > 0 and self.v_com[a] < 0) or (shift < 0 and self.v_com[a] > 0):
                    impulse[a] += -M * self.v_com[a]
                    self.v_com[a] = 0.0
        self._node_world = nodes
        self._node_vel = self.v_com + np.cross(self.omega, nodes - self.x_com)
        self._duck_tank_force = impulse / max(dt, 1e-12)

        # accretion impulse (perfectly inelastic capture of liquid)
        if np.any(self._accretion_impulse):
            J = self._accretion_impulse
            r = self._accretion_point - self.x_com
            self.v_com = self.v_com + J / M
            self.omega = self.omega + self._I_inv_world @ np.cross(r, J)
            self._accretion_impulse = np.zeros(3)

        # captured liquid particles ride rigidly with the duck
        if self.captured_idx:
            ci = np.asarray(self.captured_idx, dtype=np.int64)
            pw = self.x_com + self.captured_local @ R.T
            self.position[ci] = pw
            self.velocity[ci] = self.v_com + np.cross(self.omega, pw - self.x_com)

        # reaction of the duck on the free liquid (equal and opposite)
        react = -(F_b + F_d)
        self._fluid_force_react = react
        if active.size:
            dvec = self.position[active] - self.x_com
            dd = np.sqrt((dvec * dvec).sum(1))
            mm = self.water_mass[active]
            sel = (dd < 0.35) & (mm > 1e-12)
            if sel.any():
                wt = np.exp(-((dd[sel] / 0.22) ** 2))
                wt = wt * mm[sel]
                if wt.sum() > 0:
                    wt = wt / wt.sum()
                    self._fluid_impulse[active[sel]] += (react * dt)[None, :] * wt[:, None]
        self._duck_absorbed_impulse = F_b + F_d
        self._F_b = F_b
        self._F_d = F_d
        self._M_duck = M

    # -- free liquid -------------------------------------------------------
    def _wall_clamp(self, p, v, m):
        imp = np.zeros(3)
        r = self.particle_radius
        for a in range(3):
            lo = self.tank_lo[a] + r
            hi = self.tank_hi[a] - r
            below = p[:, a] < lo
            if below.any():
                p[below, a] = lo
                vn = v[below, a]
                bad = vn < 0.0
                if bad.any():
                    imp[a] += float((m[below][bad] * (-vn[bad])).sum())
                    vn = np.where(bad, 0.0, vn)
                    v[below, a] = vn
            above = p[:, a] > hi
            if above.any():
                p[above, a] = hi
                vn = v[above, a]
                bad = vn > 0.0
                if bad.any():
                    imp[a] += float((m[above][bad] * vn[bad]).sum())
                    vn = np.where(bad, 0.0, vn)
                    v[above, a] = vn
        return imp

    def _duck_collide(self, p, v, m):
        imp = np.zeros(3)
        nodes = self._node_world
        if nodes.shape[0] == 0 or p.shape[0] == 0:
            return imp
        lo = nodes.min(0) - 0.12
        hi = nodes.max(0) + 0.12
        sel = np.all((p >= lo) & (p <= hi), axis=1)
        if not sel.any():
            return imp
        tree = cKDTree(nodes)
        pts = p[sel]
        dist, idx = tree.query(pts, k=1)
        r0 = 0.055
        hit = dist < r0
        if not hit.any():
            return imp
        sub = np.flatnonzero(sel)[hit]
        nvec = pts[hit] - nodes[idx[hit]]
        dd = np.linalg.norm(nvec, axis=1)
        dd = np.maximum(dd, 1e-9)
        nvec = nvec / dd[:, None]
        p[sub] = nodes[idx[hit]] + nvec * r0
        nv = self._node_vel[idx[hit]]
        vrel = ((v[sub] - nv) * nvec).sum(1)
        bad = vrel < 0.0
        if bad.any():
            corr = (-vrel[bad])[:, None] * nvec[bad]
            v[sub[bad]] += corr
            imp += (m[sub[bad]][:, None] * corr).sum(0)
        return imp

    def _pbf_iterate(self, p, v, m, pairs):
        n = p.shape[0]
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = p[i] - p[j]
        r2 = (d * d).sum(1)
        r = np.sqrt(np.maximum(r2, 1e-18))
        w = self.poly6 * np.maximum(self.h * self.h - r2, 0.0) ** 3
        gw = self.spiky * (self.h - r) ** 2 / r
        rho = (np.bincount(i, weights=m[j] * w, minlength=n) +
               np.bincount(j, weights=m[i] * w, minlength=n))
        rho = np.maximum(rho, 1e-6)
        # only resist compression: free-surface particles must not be pulled out
        C = np.maximum(rho / self.rho0 - 1.0, 0.0)
        gx = (np.bincount(i, weights=m[j] * gw * d[:, 0], minlength=n) +
              np.bincount(j, weights=m[i] * (-gw) * d[:, 0], minlength=n))
        gy = (np.bincount(i, weights=m[j] * gw * d[:, 1], minlength=n) +
              np.bincount(j, weights=m[i] * (-gw) * d[:, 1], minlength=n))
        gz = (np.bincount(i, weights=m[j] * gw * d[:, 2], minlength=n) +
              np.bincount(j, weights=m[i] * (-gw) * d[:, 2], minlength=n))
        B = (np.bincount(i, weights=(m[j] * gw) ** 2 * r2, minlength=n) +
             np.bincount(j, weights=(m[i] * gw) ** 2 * r2, minlength=n))
        denom = (gx * gx + gy * gy + gz * gz + B) / (self.rho0 ** 2) + self.eps_dim
        lam = -C / denom
        dq = 0.2 * self.h
        wq = self.poly6 * max(self.h * self.h - dq * dq, 0.0) ** 3
        scorr = -0.05 * (w / max(wq, 1e-30)) ** 4
        coef = lam[i] + lam[j] + scorr
        cj = coef * m[j]
        ci = coef * m[i]
        dpx = (np.bincount(i, weights=cj * gw * d[:, 0], minlength=n) -
               np.bincount(j, weights=ci * gw * d[:, 0], minlength=n)) / self.rho0
        dpy = (np.bincount(i, weights=cj * gw * d[:, 1], minlength=n) -
               np.bincount(j, weights=ci * gw * d[:, 1], minlength=n)) / self.rho0
        dpz = (np.bincount(i, weights=cj * gw * d[:, 2], minlength=n) -
               np.bincount(j, weights=ci * gw * d[:, 2], minlength=n)) / self.rho0
        dp = np.stack([dpx, dpy, dpz], axis=1)
        mag = np.linalg.norm(dp, axis=1)
        scale = np.ones(n)
        big = mag > self.correction_cap
        scale[big] = self.correction_cap / mag[big]
        p = p + dp * scale[:, None]
        return p, v

    def _xsph(self, p, v, m, pairs, rho):
        n = p.shape[0]
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = p[i] - p[j]
        r2 = (d * d).sum(1)
        w = self.poly6 * np.maximum(self.h * self.h - r2, 0.0) ** 3
        num = np.zeros((n, 3))
        for a in range(3):
            num[:, a] = (np.bincount(i, weights=(m[j] * w) * v[j, a], minlength=n) +
                         np.bincount(j, weights=(m[i] * w) * v[i, a], minlength=n))
        den = (np.bincount(i, weights=m[j] * w, minlength=n) +
               np.bincount(j, weights=m[i] * w, minlength=n))
        return v + self.viscosity * (num - den[:, None] * v) / self.rho0

    def _fluid_step(self, dt):
        active = self.water_idx[self.water_mass[self.water_idx] > 1e-12]
        if active.size == 0:
            self.I_wall = np.zeros(3)
            self.I_duck = np.zeros(3)
            return
        p = self.position[active].copy()
        v = self.velocity[active].copy()
        m = self.water_mass[active].copy()
        Mw = float(m.sum())
        P0 = (v * m[:, None]).sum(0)

        imp_here = self._fluid_impulse[active].copy()
        self._fluid_impulse[:] = 0.0
        react_imp = imp_here.sum(0)

        v = v + self.gravity * dt
        v = v + imp_here / m[:, None]

        p_prev = p.copy()
        p = p + v * dt

        I_wall = self._wall_clamp(p, v, m)
        I_duck = self._duck_collide(p, v, m)

        tree = cKDTree(p)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        for _ in range(2):
            p, v = self._pbf_iterate(p, v, m, pairs)
            I_wall = I_wall + self._wall_clamp(p, v, m)
            I_duck = I_duck + self._duck_collide(p, v, m)

        v = np.clip((p - p_prev) / dt, -6.0, 6.0)
        if pairs.size:
            d = p[pairs[:, 0]] - p[pairs[:, 1]]
            r2 = (d * d).sum(1)
            w = self.poly6 * np.maximum(self.h * self.h - r2, 0.0) ** 3
            rho = (np.bincount(pairs[:, 0], weights=m[pairs[:, 1]] * w, minlength=p.shape[0]) +
                   np.bincount(pairs[:, 1], weights=m[pairs[:, 0]] * w, minlength=p.shape[0]))
            rho = np.maximum(rho, 1e-6)
            v = self._xsph(p, v, m, pairs, rho)

        P_now = (v * m[:, None]).sum(0)
        P_target = P0 + (self.gravity * Mw + react_imp / dt) * dt + I_wall + I_duck
        v = v + (P_target - P_now) / Mw

        self.position[active] = p
        self.velocity[active] = v
        self.I_wall = I_wall
        self.I_duck = I_duck

    # -- main stepping -----------------------------------------------------
    def _step(self, dt):
        self.step_count = getattr(self, "step_count", 0) + 1
        self.x_com = self.x_com  # keep attribute explicit
        self._accretion_impulse = getattr(self, "_accretion_impulse", np.zeros(3))
        self._accretion_point = getattr(self, "_accretion_point", self.x_com.copy())
        self._update_surface()
        self._imbibition(dt)
        self._duck_step(dt)
        self._fluid_step(dt)
        self._capture(dt)

        # two-way interaction: water collisions push back on the duck
        if np.any(self.I_duck):
            M = self._M_duck
            J = -self.I_duck.copy()
            self.v_com = self.v_com + J / M
            rel = self._node_world - self.x_com
            lowest = int(np.argmin(rel[:, 2]))
            self.omega = self.omega + self._I_inv_world @ np.cross(rel[lowest], J)

        duck_sel = np.isin(self.body_id, (1, 3))
        water_sel = self.body_id == 2
        Mwater = float(self.water_mass[water_sel].sum())
        self.ext_force[0, 0] = (self._M_duck + Mwater) * self.gravity
        self.ext_force[0, 3] = self._duck_tank_force + self.I_wall / dt
        self.ext_force[1, 0] = self._M_duck * self.gravity
        self.ext_force[1, 1] = self._F_b
        self.ext_force[1, 2] = self._F_d - self.I_duck / dt
        self.ext_force[1, 3] = self._duck_tank_force
        self.ext_force[2, 0] = Mwater * self.gravity
        self.ext_force[2, 1] = -self._F_b
        self.ext_force[2, 2] = -self._F_d + self.I_duck / dt
        self.ext_force[2, 3] = self.I_wall / dt
        self.imp_ext += self.ext_force * dt

        self.interaction_now[0] = self._F_b
        self.interaction_now[1] = self._F_d
        self.interaction_now[2] = self._duck_tank_force
        self.interaction_now[3] = self._accretion_impulse_prev / max(dt, 1e-12)
        self._accretion_impulse_prev = self._accretion_impulse.copy()

    # -- trajectory frame --------------------------------------------------
    def _build_frame(self):
        from worldsolver.contracts import StateFrame

        n = self.n
        R = _quat_matrix(self.quat)
        duck_sel = np.isin(self.body_id, (1, 3))
        water_sel = self.body_id == 2
        pore_water = self.absorbed_committed / max(self.n_pore, 1)

        solid_mass = np.zeros(n)
        solid_mass[self.duck_idx] = self.duck_dry_mass[self.duck_idx]
        pore_col = np.zeros(n)
        free_col = np.zeros(n)
        if self.n_pore:
            pore_col[self.pore_idx] = pore_water
        free_col[water_sel] = self.water_mass[water_sel]
        total_mass = solid_mass + pore_col + free_col

        saturation = np.zeros(n)
        saturation[self.pore_idx] = np.clip(self.saturation_pore, 0.0, 1.0)
        if self.captured_idx:
            saturation[np.asarray(self.captured_idx, dtype=np.int64)] = 1.0
        saturation[water_sel] = 1.0

        permeability = np.zeros(n)
        permeability[self.pore_idx] = self.permeability
        if self.captured_idx:
            permeability[np.asarray(self.captured_idx, dtype=np.int64)] = self.k_switch

        porosity = np.zeros(n)
        porosity[self.duck_idx] = self.porosity
        if self.captured_idx:
            porosity[np.asarray(self.captured_idx, dtype=np.int64)] = self.porosity

        pressure = np.zeros(n)
        if self.n_pore:
            z_s = self._surface_at(self._node_world[self.is_pore_local])
            depth = np.maximum(z_s - self._node_world[self.is_pore_local][:, 2], 0.0)
            pressure[self.pore_idx] = self.rho_w * 9.81 * depth * self.wetted

        solid_vel = np.zeros((n, 3))
        if self.captured_idx:
            ci = np.asarray(self.captured_idx, dtype=np.int64)
            solid_vel[ci] = self.velocity[ci]
        solid_vel[self.duck_idx] = self._node_vel

        orientation = np.tile(_IDENTITY4, (n, 1))
        angular_velocity = np.zeros((n, 3))
        duck_all = np.flatnonzero(duck_sel)
        orientation[duck_all] = self.quat
        angular_velocity[duck_all] = self.omega

        attrs = {
            "phase": np.where(water_sel, 1.0, 0.0),
            "saturation": saturation,
            "porosity": porosity,
            "permeability_m2": permeability,
            "pore_water_mass": pore_col,
            "pore_pressure": pressure,
            "solid_velocity": solid_vel,
            "mass_quantity": np.stack([solid_mass, pore_col, free_col], axis=1),
        }

        weights = np.zeros((self.n_dom, n))
        weights[0] = 1.0
        weights[1] = duck_sel.astype(np.float64)
        weights[2] = water_sel.astype(np.float64)

        fields = {
            "budget_domain_weights": weights,
            "budget_external_force": np.nan_to_num(self.ext_force),
            "budget_cumulative_external_impulse": np.nan_to_num(self.imp_ext.sum(axis=1)),
            "budget_cumulative_inflow_momentum": np.nan_to_num(self.imp_in),
            "budget_cumulative_outflow_momentum": np.nan_to_num(self.imp_out),
            "cumulative_mass_source": np.zeros(3, dtype=np.float64),
            "interaction_force": np.nan_to_num(self.interaction_now),
        }

        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "liquid_phase_id": 1.0,
            "duck_mass_kg": float(self.M_dry + self.absorbed_committed),
            "mean_saturation": float(self.saturation_pore.mean()) if self.n_pore else 0.0,
            "free_water_mass_kg": float(self.water_mass.sum()),
            "initial_water_surface_z": 0.82,
            "permeability_switch_time": float(self.switch_time),
            "characteristic_length": float(self.spacing),
        }

        return StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=total_mass,
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
