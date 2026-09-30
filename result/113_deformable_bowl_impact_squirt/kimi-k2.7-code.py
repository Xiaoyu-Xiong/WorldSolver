"""WorldSolver benchmark solver for 113_deformable_bowl_impact_squirt.

A simple explicit time-stepping particle solver for a deformable bowl
filled with liquid that falls into a shallow water layer.  The bowl is
treated as a deformable shell, the contained liquid and the shallow water
as weakly-compressible SPH fluids.  Coupling is via penalty contact between
the bowl shell particles and the fluid particles.  Gravity, tank walls,
internal elastic forces and fluid pressure are integrated together with
symplectic Euler substeps.  The implementation is deterministic and produces
a StateFrame at every requested sample time.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from worldsolver.contracts import StateFrame


@dataclass
class _BoundaryPlane:
    center: np.ndarray
    normal: np.ndarray
    width: float
    height: float
    axis_u: np.ndarray
    axis_v: np.ndarray


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        np.random.seed(seed)

        cfg = scene.config
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        init = cfg["initial_conditions"]
        self.bowl_total_mass = float(init["bowl"]["total_mass"])
        self.bowl_thickness = float(init["bowl"]["thickness"])
        self.bowl_impermeable = bool(init["bowl"].get("impermeable", True))

        liq = init["liquid"]
        self.liq_rho = float(liq["rest_density"])
        self.liq_spacing = float(liq["particle_spacing"])

        wat = init["shallow_water"]
        self.wat_rho = float(wat["rest_density"])
        self.wat_spacing = float(wat["particle_spacing"])

        bc = cfg["boundary_conditions"]
        tank = bc["tank"]
        self.tank_min = np.asarray(tank["interior_min"], float)
        self.tank_max = np.asarray(tank["interior_max"], float)
        self.tank_open_top = bool(tank.get("open_top", True))

        # Bowl mesh topology
        asset = pathlib.Path(__file__).parent.parent / "assets" / "bowl.npz"
        data = np.load(str(asset))
        self.faces = np.asarray(data["faces"], np.int32)
        self.rest_vertices = np.asarray(data["vertices"], float)

        # Initial state from the scene
        f0 = scene.initial_state()
        self.position = np.asarray(f0.position, float).copy()
        self.velocity = np.asarray(f0.velocity, float).copy()
        self.mass = np.asarray(f0.mass, float).copy()
        self.body_id = np.asarray(f0.body_id, np.int32).copy()

        # Partition by body_id: 1 = bowl shell, 2 = contained liquid, 3 = shallow water
        self.bowl_idx = np.nonzero(self.body_id == 1)[0]
        self.liq_idx = np.nonzero(self.body_id == 2)[0]
        self.wat_idx = np.nonzero(self.body_id == 3)[0]

        # Per-bowl-vertex rest positions and mass
        self.bowl_rest = self.position[self.bowl_idx].copy()
        self.bowl_rest_center = self.bowl_rest.mean(axis=0)
        self.bowl_local = self.bowl_rest - self.bowl_rest_center
        self.bowl_mass = self.mass[self.bowl_idx].copy()

        self.liq_m0 = float(self.mass[self.liq_idx[0]]) if len(self.liq_idx) else 1e-3
        self.wat_m0 = float(self.mass[self.wat_idx[0]]) if len(self.wat_idx) else 1e-3

        # Build edge springs from the rest mesh
        self._build_edge_springs()

        # Material / numerical parameters
        self.young = 1.5e5
        self.damping = 8.0
        self.wall_friction = 0.15
        self.contact_k = 3.0e4
        self.contact_damp = 0.5
        self.stiffness = 1.0e4  # fluid pressure scale (Pa)
        self.viscosity = 0.05
        self.fluid_cutoff = 1.8 * max(self.liq_spacing, self.wat_spacing)
        self.cross_cutoff = 0.5 * (self.liq_spacing + self.wat_spacing)

        # Time step: choose a small explicit sub-step that evenly divides sample_dt
        self.dt = self.sample_dt / 40.0

        self.current_time = 0.0
        self._sample_time = 0.0
        self._step = 0

        # Boundary wall samples for trajectory export
        self._init_boundary_samples()

        # Budget / cumulative bookkeeping
        self._cumulative_ext_impulse = np.zeros((3, 3), float)
        self._cumulative_in_mom = np.zeros((3, 3), float)
        self._cumulative_out_mom = np.zeros((3, 3), float)

        # Reference scales
        if len(self.bowl_idx):
            self._ref_momentum = float(
                np.sum(self.bowl_mass * np.linalg.norm(self.velocity[self.bowl_idx], axis=1))
            )
        else:
            self._ref_momentum = 1.0
        v0 = self.velocity[self.bowl_idx[0]] if len(self.bowl_idx) else np.zeros(3)
        self._ref_velocity = max(float(np.linalg.norm(v0)), 1.2)

        self._diag = {
            "dt": self.dt,
            "n_particles": len(self.position),
            "n_bowl": len(self.bowl_idx),
            "n_liquid": len(self.liq_idx),
            "n_water": len(self.wat_idx),
        }

    # -----------------------------------------------------------------------
    #  helpers
    # -----------------------------------------------------------------------
    def _build_edge_springs(self):
        edges = set()
        for f in self.faces:
            edges.add((int(min(f[0], f[1])), int(max(f[0], f[1]))))
            edges.add((int(min(f[1], f[2])), int(max(f[1], f[2]))))
            edges.add((int(min(f[2], f[0])), int(max(f[2], f[0]))))
        self.edges = np.asarray(list(edges), np.int32)
        self.rest_len = np.linalg.norm(
            self.bowl_local[self.edges[:, 0]] - self.bowl_local[self.edges[:, 1]], axis=1
        )

    def _init_boundary_samples(self):
        planes = [
            _BoundaryPlane(
                np.array([0.0, 0.0, self.tank_min[2]]),
                np.array([0.0, 0.0, 1.0]),
                self.tank_max[0] - self.tank_min[0],
                self.tank_max[1] - self.tank_min[1],
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
            _BoundaryPlane(
                np.array([self.tank_min[0], 0.0, (self.tank_min[2] + self.tank_max[2]) * 0.5]),
                np.array([1.0, 0.0, 0.0]),
                self.tank_max[1] - self.tank_min[1],
                self.tank_max[2] - self.tank_min[2],
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
            _BoundaryPlane(
                np.array([self.tank_max[0], 0.0, (self.tank_min[2] + self.tank_max[2]) * 0.5]),
                np.array([-1.0, 0.0, 0.0]),
                self.tank_max[1] - self.tank_min[1],
                self.tank_max[2] - self.tank_min[2],
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
            _BoundaryPlane(
                np.array([0.0, self.tank_min[1], (self.tank_min[2] + self.tank_max[2]) * 0.5]),
                np.array([0.0, 1.0, 0.0]),
                self.tank_max[0] - self.tank_min[0],
                self.tank_max[2] - self.tank_min[2],
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
            _BoundaryPlane(
                np.array([0.0, self.tank_max[1], (self.tank_min[2] + self.tank_max[2]) * 0.5]),
                np.array([0.0, -1.0, 0.0]),
                self.tank_max[0] - self.tank_min[0],
                self.tank_max[2] - self.tank_min[2],
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
        ]
        if not self.tank_open_top:
            planes.append(
                _BoundaryPlane(
                    np.array([0.0, 0.0, self.tank_max[2]]),
                    np.array([0.0, 0.0, -1.0]),
                    self.tank_max[0] - self.tank_min[0],
                    self.tank_max[1] - self.tank_min[1],
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                )
            )
        self._walls = planes
        pts, normals = [], []
        for w in planes:
            nu = max(1, int(np.ceil(w.width / 0.25)))
            nv = max(1, int(np.ceil(w.height / 0.25)))
            for i in range(nu):
                for j in range(nv):
                    u = (i + 0.5) / nu - 0.5
                    v = (j + 0.5) / nv - 0.5
                    pts.append(w.center + u * w.width * w.axis_u + v * w.height * w.axis_v)
                    normals.append(w.normal.copy())
        self._boundary_points = np.asarray(pts, float)
        self._boundary_normals = np.asarray(normals, float)

    # -----------------------------------------------------------------------
    #  substep
    # -----------------------------------------------------------------------
    def _substep(self):
        pos = self.position
        vel = self.velocity
        acc = np.zeros_like(vel)
        acc += self.gravity

        # Bowl shell: edge springs + bending + damping
        self._bowl_elastic(acc)

        # Fluid pressure (SPH) for contained liquid and shallow water
        if len(self.liq_idx):
            self._sph_forces(self.liq_idx, self.liq_m0, pos, vel, acc)
        if len(self.wat_idx):
            self._sph_forces(self.wat_idx, self.wat_m0, pos, vel, acc)

        # Bowl - contained liquid coupling
        if len(self.liq_idx):
            self._couple_bowl_fluid(pos, vel, acc, self.liq_idx)

        # Bowl - shallow water coupling
        if len(self.wat_idx):
            self._couple_bowl_fluid(pos, vel, acc, self.wat_idx)

        # Liquid - shallow water coupling when liquid escapes through the opening
        if len(self.liq_idx) and len(self.wat_idx):
            self._couple_fluids(pos, vel, acc)

        # Wall / boundary collisions
        self._wall_collisions(pos, vel, acc)

        # Explicit update
        vel_new = vel + acc * self.dt
        pos_new = pos + vel_new * self.dt
        pos_new = self._clip_to_tank(pos_new)

        self.position = pos_new
        self.velocity = vel_new
        self.current_time += self.dt
        self._step += 1

    def _bowl_elastic(self, acc):
        bpos = self.position[self.bowl_idx]
        bvel = self.velocity[self.bowl_idx]
        bacc = acc[self.bowl_idx]

        # Edge spring forces (linear-elastic)
        p0 = bpos[self.edges[:, 0]]
        p1 = bpos[self.edges[:, 1]]
        dx = p1 - p0
        l = np.linalg.norm(dx, axis=1)
        l = np.where(l < 1e-12, 1e-12, l)
        strain = (l - self.rest_len) / self.rest_len
        fmag = self.young * self.bowl_thickness * strain / l
        f = -fmag[:, None] * dx
        m0 = self.bowl_mass[self.edges[:, 0]]
        m1 = self.bowl_mass[self.edges[:, 1]]
        np.add.at(bacc, self.edges[:, 0], f / m0[:, None])
        np.add.at(bacc, self.edges[:, 1], -f / m1[:, None])

        # Bending / area-preservation via Laplacian smoothing
        n = len(bpos)
        adj = np.zeros((n, n), bool)
        adj[self.edges[:, 0], self.edges[:, 1]] = True
        adj[self.edges[:, 1], self.edges[:, 0]] = True
        neighbor_sum = adj.astype(float).dot(bpos)
        neighbor_count = adj.sum(axis=1)[:, None]
        neighbor_count = np.where(neighbor_count < 1, 1, neighbor_count)
        target = neighbor_sum / neighbor_count
        disp = target - bpos
        bacc += 1.5e3 * disp

        # Velocity damping
        bacc -= self.damping * bvel

    def _sph_forces(self, idx, m0, pos, vel, acc):
        p = pos[idx]
        if len(p) == 0:
            return
        h = self.fluid_cutoff
        tree = KDTree(p)
        pairs = tree.query_pairs(h, output_type="ndarray")

        # Self-contribution to density
        w0 = (21.0 / (2.0 * np.pi * h ** 3))
        rho = np.full(len(p), m0 * w0)

        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            dvec = p[j] - p[i]
            r = np.linalg.norm(dvec, axis=1)
            r = np.where(r < 1e-12, 1e-12, r)
            W = self._wendland_kernel(r, h)
            np.add.at(rho, i, m0 * W)
            np.add.at(rho, j, m0 * W)

            # Pressure
            rho0 = 1000.0
            P = np.maximum(0.0, (rho / rho0 - 1.0)) * self.stiffness
            Pi = P[i]
            Pj = P[j]
            grad_scalar = self._wendland_kernel_grad(r, h) / r
            # grad W_ij = (dW/dr / r) * (p_j - p_i)
            a_pres = -m0 * (Pi / (rho[i] ** 2) + Pj / (rho[j] ** 2))[:, None] * grad_scalar[:, None] * dvec
            np.add.at(acc, idx[i], a_pres)
            np.add.at(acc, idx[j], -a_pres)

            # Artificial viscosity (vectorized over pairs)
            dv = vel[idx[i]] - vel[idx[j]]
            vdot = np.sum(dv * dvec, axis=1)
            mu = 2.0 * self.viscosity * 10.0 * (1.0 - r / h) / (rho[i] + rho[j])
            a_vis = (mu * vdot / (r ** 2 + 1e-8))[:, None] * dvec
            np.add.at(acc, idx[i], a_vis)
            np.add.at(acc, idx[j], -a_vis)

    def _couple_bowl_fluid(self, pos, vel, acc, fluid_idx):
        bpos = pos[self.bowl_idx]
        fpos = pos[fluid_idx]
        if len(bpos) == 0 or len(fpos) == 0:
            return
        tree = KDTree(bpos)
        d, close = tree.query(fpos, k=1)
        d = np.asarray(d, float)
        close = np.asarray(close, int)
        rc = 0.75 * self.liq_spacing
        penetr = rc - d
        active = penetr > 0.0
        if not np.any(active):
            return
        dist_vec = fpos - bpos[close]
        dist = np.linalg.norm(dist_vec, axis=1)
        normal = dist_vec / (dist[:, None] + 1e-12)
        rel = vel[fluid_idx] - vel[self.bowl_idx[close]]
        v_close = np.sum(rel * normal, axis=1)
        fn = self.contact_k * penetr - self.contact_damp * v_close * np.sqrt(self.contact_k * self.bowl_total_mass)
        fn = np.maximum(fn, 0.0)
        fvec = np.zeros_like(fpos)
        fvec[active] = fn[active, None] * normal[active]
        # Fluid side
        acc[fluid_idx] += fvec / np.mean(self.mass[fluid_idx])
        # Bowl side (equal and opposite)
        np.add.at(acc, self.bowl_idx, -fvec / self.bowl_mass[:, None])

    def _couple_fluids(self, pos, vel, acc):
        # Simple stiff repulsion between escaped contained liquid and shallow water
        p_liq = pos[self.liq_idx]
        p_wat = pos[self.wat_idx]
        if len(p_lq := p_liq) == 0 or len(p_wat) == 0:
            return
        tree = KDTree(p_wat)
        d, close = tree.query(p_liq, k=1)
        d = np.asarray(d, float)
        close = np.asarray(close, int)
        rc = self.cross_cutoff
        active = d < rc
        if not np.any(active):
            return
        idx_i = self.liq_idx[active]
        idx_j = self.wat_idx[close[active]]
        dist_vec = p_liq[active] - p_wat[close[active]]
        dist = np.linalg.norm(dist_vec, axis=1)
        normal = dist_vec / (dist[:, None] + 1e-12)
        rel = vel[idx_i] - vel[idx_j]
        v_close = np.sum(rel * normal, axis=1)
        fn = self.contact_k * (rc - d[active]) - self.contact_damp * v_close * np.sqrt(self.contact_k * self.bowl_total_mass)
        fn = np.maximum(fn, 0.0)
        fvec = fn[:, None] * normal
        m_l = self.liq_m0
        m_w = self.wat_m0
        acc[idx_i] += fvec / m_l
        acc[idx_j] -= fvec / m_w

    def _wall_collisions(self, pos, vel, acc):
        minx, miny, minz = self.tank_min
        maxx, maxy, maxz = self.tank_max
        for idx_group in (self.bowl_idx, self.liq_idx, self.wat_idx):
            p = pos[idx_group]
            v = vel[idx_group]
            if len(p) == 0:
                continue
            m_group = self.mass[idx_group]
            # floor
            pen = minz - p[:, 2]
            active = pen > 0.0
            if np.any(active):
                vn = v[active, 2]
                f_n = self.contact_k * pen[active] - self.contact_damp * vn
                f_n = np.maximum(f_n, 0.0)
                acc[idx_group[active], 2] += f_n / m_group[active]
                # friction
                vt = v[active].copy()
                vt[:, 2] = 0.0
                vt_mag = np.linalg.norm(vt, axis=1)
                f_t = np.minimum(self.wall_friction * f_n, vt_mag + 1e-6)
                acc[idx_group[active], :2] -= (f_t[:, None] * vt / (vt_mag[:, None] + 1e-8))[:, :2] / m_group[active, None]

            # x walls
            for dim, lo, hi in ((0, minx, maxx), (1, miny, maxy)):
                pen_lo = lo - p[:, dim]
                pen_hi = p[:, dim] - hi
                active_lo = pen_lo > 0.0
                active_hi = pen_hi > 0.0
                if np.any(active_lo):
                    f = self.contact_k * pen_lo[active_lo] + self.contact_damp * v[active_lo, dim]
                    f = np.maximum(f, 0.0)
                    acc[idx_group[active_lo], dim] -= f / m_group[active_lo]
                if np.any(active_hi):
                    f = self.contact_k * pen_hi[active_hi] - self.contact_damp * v[active_hi, dim]
                    f = np.maximum(f, 0.0)
                    acc[idx_group[active_hi], dim] += f / m_group[active_hi]

            # ceiling if closed
            if not self.tank_open_top:
                pen = p[:, 2] - maxz
                active = pen > 0.0
                if np.any(active):
                    f = self.contact_k * pen[active] + self.contact_damp * v[active, 2]
                    f = np.maximum(f, 0.0)
                    acc[idx_group[active], 2] -= f / m_group[active]

    def _clip_to_tank(self, pos_new):
        pos_new[:, 2] = np.maximum(pos_new[:, 2], self.tank_min[2])
        eps = 1e-3
        pos_new[:, 0] = np.clip(pos_new[:, 0], self.tank_min[0] + eps, self.tank_max[0] - eps)
        pos_new[:, 1] = np.clip(pos_new[:, 1], self.tank_min[1] + eps, self.tank_max[1] - eps)
        return pos_new

    def _wendland_kernel(self, r, h):
        q = r / h
        out = np.zeros_like(r)
        mask = q < 2.0
        if not np.any(mask):
            return out
        x = 1.0 - 0.5 * q[mask]
        out[mask] = (21.0 / (2.0 * np.pi * h ** 3)) * x ** 4 * (1.0 + 2.0 * q[mask])
        return out

    def _wendland_kernel_grad(self, r, h):
        # returns dW/dr
        q = r / h
        out = np.zeros_like(r)
        mask = q < 2.0
        if not np.any(mask):
            return out
        x = 1.0 - 0.5 * q[mask]
        out[mask] = -(21.0 * 5.0 / (np.pi * h ** 4)) * x ** 3
        return out

    # -----------------------------------------------------------------------
    #  public API
    # -----------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.current_time < target - 0.5 * self.dt:
            self._substep()
        self._sample_time = target
        return self._make_frame()

    def _make_frame(self):
        n = len(self.position)
        mass_quantity = self.mass.copy()

        B = len(self._boundary_points)
        boundary_normal = self._boundary_normals.copy()
        boundary_target_velocity = np.zeros((B, 3), float)
        boundary_velocity = boundary_target_velocity.copy()

        G = 3
        budget_domain_weights = np.zeros((G, n), float)
        for g, idx in enumerate((self.bowl_idx, self.liq_idx, self.wat_idx)):
            budget_domain_weights[g, idx] = 1.0

        budget_external_force = np.zeros((G, 1, 3), float)
        for g, idx in enumerate((self.bowl_idx, self.liq_idx, self.wat_idx)):
            if len(idx):
                budget_external_force[g, 0] = np.sum(self.mass[idx][:, None] * self.gravity, axis=0)

        interaction_force = np.zeros((3, 3), float)
        cumulative_mass_source = np.zeros(3, float)

        frame = StateFrame(
            time=float(self._sample_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": mass_quantity},
            fields={
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_target_velocity,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_impulse": self._cumulative_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self._cumulative_in_mom.copy(),
                "budget_cumulative_outflow_momentum": self._cumulative_out_mom.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "reference_momentum": float(self._ref_momentum),
                "reference_velocity": float(self._ref_velocity),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return self._diag
