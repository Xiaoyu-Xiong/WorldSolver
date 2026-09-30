"""Soft-sphere DEM solver for a long horizontal rotating drum.

Bidisperse dry granular assembly (two species with distinct radius, mass and
friction) inside a rigid cylinder whose axis is horizontal (along +x) and which
rotates kinematically at the frozen angular velocity.  Each grain is a point
mass with a finite contact radius.  Contacts use a regularized soft-sphere law:

    normal      F_n = max(0, k_n * delta - gamma_n * (v_rel . n)) * n
    tangential  F_t = -min(mu * |F_n|, c_t * |v_t|) * v_t / |v_t|

with ``delta`` the geometric overlap and ``n`` the unit contact normal.  Wall
contacts use the same law against the analytic barrel and the two end caps,
evaluated with the local wall velocity ``omega x r`` so that wall friction
actually drives the bed into solid-body-plus-surface-flow motion.

Time stepping is semi-implicit (symplectic) Euler with a fixed step well below
the contact period, and a Verlet neighbour list rebuilt from a kd-tree whenever
a grain has drifted more than half the skin.  Size and friction contrast between
the species drives percolation of the small grains toward the rotating core
(radial segregation) and the slow growth of axial composition fluctuations.
No motion is scripted: all displacements come from integrated forces.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- numerical / material constants ------------------------------------
    DT = 8.0e-4  # target integration step [s]
    SKIN = 0.055  # Verlet neighbour-list skin [m]
    KN_PP = 3.0e4  # grain-grain normal stiffness [N/m]
    KN_W = 6.0e4  # grain-wall normal stiffness [N/m]
    ZETA = 0.35  # normal contact damping ratio
    CT_PP = 200.0  # tangential viscous regularization, grains [N s/m]
    CT_W = 350.0  # tangential viscous regularization, wall [N s/m]
    MU_SPECIES = (0.55, 0.32)  # sliding friction of species 0 / species 1
    MU_WALL = 0.85  # grain-wall sliding friction
    V_MAX = 12.0  # numerical velocity ceiling [m/s]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        centrifuge = config["centrifuge"]

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=np.float64)
        self.velocity = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.species = np.array(frame0.attributes["species_id"], dtype=np.int32)
        self.radii = np.array(frame0.attributes["particle_radius"], dtype=np.float64)
        self.mass_quantity = np.array(frame0.attributes["mass_quantity"], dtype=np.float64)
        self.n = int(len(self.position))
        self.body_id = np.zeros(self.n, dtype=np.int32)
        self.species_count = int(self.mass_quantity.shape[1]) if self.mass_quantity.ndim > 1 else 1

        # frozen geometry / drive
        self.radius = float(centrifuge["radius"])
        self.half_length = 0.5 * float(centrifuge["length"])
        self.center_z = float(centrifuge["center_height"])
        self.omega = float(centrifuge["angular_velocity_rad_s"])
        self.grid_shape = tuple(int(v) for v in centrifuge["field_grid_shape"])
        self.gravity = np.asarray(config["gravity"], dtype=np.float64)

        self.contact_length_scale = float(frame0.scalars["contact_length_scale"])
        self.characteristic_length = float(frame0.scalars["characteristic_length"])
        self.grid_spacing = float(frame0.scalars["grid_spacing"])

        # per-grain material data
        self.mu = np.where(self.species == 0, self.MU_SPECIES[0], self.MU_SPECIES[1])
        self.inv_mass = 1.0 / self.mass
        self.gravity_force = self.mass[:, None] * self.gravity[None, :]

        # neighbour-list bookkeeping
        self.cutoff = 2.0 * float(self.radii.max()) + self.SKIN
        self.pair_i = np.zeros(0, dtype=np.int64)
        self.pair_j = np.zeros(0, dtype=np.int64)
        self.pair_radsum = np.zeros(0)
        self.pair_gamma = np.zeros(0)
        self.pair_mu = np.zeros(0)
        self.reference_position = self.position.copy()
        self._build_neighbours()

        sample_dt = float(config["sample_dt"])
        self.substeps = max(1, int(np.ceil(sample_dt / self.DT)))
        self.dt = sample_dt / self.substeps
        self.time = 0.0
        self.step_index = 0
        self.rebuilds = 0
        self.max_speed = 0.0
        self.max_penetration = 0.0
        self.contact_gaps = np.zeros(1, dtype=np.float64)
        self.total_force = self.gravity_force.copy()
        self._compute_forces(record_contacts=True)

    # ------------------------------------------------------- neighbour search
    def _build_neighbours(self) -> None:
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if pairs is None or len(pairs) == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if len(pairs):
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs = pairs[order]
        self.pair_i = np.ascontiguousarray(pairs[:, 0])
        self.pair_j = np.ascontiguousarray(pairs[:, 1])
        mi = self.mass[self.pair_i]
        mj = self.mass[self.pair_j]
        m_eff = mi * mj / (mi + mj)
        self.pair_radsum = self.radii[self.pair_i] + self.radii[self.pair_j]
        self.pair_gamma = 2.0 * self.ZETA * np.sqrt(np.maximum(m_eff * self.KN_PP, 1e-12))
        self.pair_mu = np.minimum(self.mu[self.pair_i], self.mu[self.pair_j])
        self.reference_position = self.position.copy()
        self.rebuilds = getattr(self, "rebuilds", 0) + 1

    def _maybe_rebuild(self) -> None:
        drift = self.position - self.reference_position
        if float(np.max(np.abs(drift))) > 0.5 * self.SKIN:
            self._build_neighbours()

    # -------------------------------------------------------------- mechanics
    def _wall_velocity(self, position: np.ndarray) -> np.ndarray:
        """Rigid-body velocity (omega * x_hat) x r of the rotating drum surface."""
        out = np.zeros_like(position)
        out[:, 1] = -self.omega * (position[:, 2] - self.center_z)
        out[:, 2] = self.omega * position[:, 1]
        return out

    def _wall_contact(self, indices, normals, delta, force):
        """Accumulate a normal + Coulomb-capped frictional wall contact."""
        pos = self.position[indices]
        v_rel = self.velocity[indices] - self._wall_velocity(pos)
        vn = np.einsum("ij,ij->i", v_rel, normals)
        gamma = 2.0 * self.ZETA * np.sqrt(self.mass[indices] * self.KN_W)
        fn = np.maximum(self.KN_W * delta - gamma * vn, 0.0)
        fvec = fn[:, None] * normals
        v_t = v_rel - vn[:, None] * normals
        vt_mag = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        ft = np.minimum(self.MU_WALL * fn, self.CT_W * vt_mag)
        fvec -= (ft / np.maximum(vt_mag, 1e-12))[:, None] * v_t
        np.add.at(force, indices, fvec)

    def _compute_forces(self, record_contacts: bool = False) -> np.ndarray:
        n = self.n
        pos = self.position
        vel = self.velocity
        force = self.gravity_force.copy()
        gaps: list[np.ndarray] = []
        near = 0.30 * float(self.radii.mean())

        # ---------------- grain-grain contacts -----------------------------
        if len(self.pair_i):
            i = self.pair_i
            j = self.pair_j
            d = pos[i] - pos[j]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            dist_safe = np.maximum(dist, 1e-12)
            gap = dist - self.pair_radsum
            if record_contacts:
                sel = gap < near
                if np.any(sel):
                    gaps.append(gap[sel])
            active = gap < 0.0
            if np.any(active):
                a = np.flatnonzero(active)
                nrm = d[a] / dist_safe[a][:, None]
                delta = -gap[a]
                v_rel = vel[i[a]] - vel[j[a]]
                vn = np.einsum("ij,ij->i", v_rel, nrm)
                fn = np.maximum(self.KN_PP * delta - self.pair_gamma[a] * vn, 0.0)
                fvec = fn[:, None] * nrm
                v_t = v_rel - vn[:, None] * nrm
                vt_mag = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
                ft = np.minimum(self.pair_mu[a] * fn, self.CT_PP * vt_mag)
                fvec -= (ft / np.maximum(vt_mag, 1e-12))[:, None] * v_t
                idx = np.concatenate((i[a], j[a]))
                for k in range(3):
                    w = np.concatenate((fvec[:, k], -fvec[:, k]))
                    force[:, k] += np.bincount(idx, weights=w, minlength=n)
                pen = float(delta.max())
                if pen > self.max_penetration:
                    self.max_penetration = pen

        # ---------------- cylindrical barrel -------------------------------
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center_z
        r = np.sqrt(dy * dy + dz * dz)
        r_safe = np.maximum(r, 1e-12)
        gap_w = (self.radius - r) - self.radii
        if record_contacts:
            sel = gap_w < near
            if np.any(sel):
                gaps.append(gap_w[sel])
        active = gap_w < 0.0
        if np.any(active):
            a = np.flatnonzero(active)
            nrm = np.zeros((len(a), 3))
            nrm[:, 1] = -dy[a] / r_safe[a]
            nrm[:, 2] = -dz[a] / r_safe[a]
            self._wall_contact(a, nrm, -gap_w[a], force)

        # ---------------- flat end caps ------------------------------------
        for sign in (-1.0, 1.0):
            gap_c = (self.half_length - sign * pos[:, 0]) - self.radii
            if record_contacts:
                sel = gap_c < near
                if np.any(sel):
                    gaps.append(gap_c[sel])
            active = gap_c < 0.0
            if not np.any(active):
                continue
            a = np.flatnonzero(active)
            nrm = np.zeros((len(a), 3))
            nrm[:, 0] = -sign
            self._wall_contact(a, nrm, -gap_c[a], force)

        if record_contacts:
            self.contact_gaps = (
                np.concatenate(gaps).astype(np.float64) if gaps else np.zeros(1, dtype=np.float64)
            )
        self.total_force = force
        return force

    def _confine(self) -> None:
        """Non-penetration backstop: keep every grain strictly inside the drum."""
        pos = self.position
        vel = self.velocity
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center_z
        r = np.sqrt(dy * dy + dz * dz)
        limit = np.maximum(self.radius - self.radii, 1e-6)
        outside = r > limit
        if np.any(outside):
            a = np.flatnonzero(outside)
            r_safe = np.maximum(r[a], 1e-12)
            uy = dy[a] / r_safe
            uz = dz[a] / r_safe
            pos[a, 1] = uy * limit[a]
            pos[a, 2] = self.center_z + uz * limit[a]
            vr = vel[a, 1] * uy + vel[a, 2] * uz
            approaching = vr > 0.0
            if np.any(approaching):
                b = a[approaching]
                vel[b, 1] -= vr[approaching] * uy[approaching]
                vel[b, 2] -= vr[approaching] * uz[approaching]
        cap = np.maximum(self.half_length - self.radii, 1e-6)
        hi = pos[:, 0] > cap
        if np.any(hi):
            a = np.flatnonzero(hi)
            pos[a, 0] = cap[a]
            vel[a, 0] = np.minimum(vel[a, 0], 0.0)
        lo = pos[:, 0] < -cap
        if np.any(lo):
            a = np.flatnonzero(lo)
            pos[a, 0] = -cap[a]
            vel[a, 0] = np.maximum(vel[a, 0], 0.0)

    def _step(self, record_contacts: bool) -> None:
        force = self._compute_forces(record_contacts=record_contacts)
        self.velocity += (self.dt * force) * self.inv_mass[:, None]
        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        fast = speed > self.V_MAX
        if np.any(fast):
            self.velocity[fast] *= (self.V_MAX / np.maximum(speed[fast], 1e-12))[:, None]
            speed[fast] = self.V_MAX
        if self.n:
            self.max_speed = max(self.max_speed, float(speed.max()))
        self.position += self.dt * self.velocity
        self._confine()
        self._maybe_rebuild()
        self.step_index += 1

    # ----------------------------------------------------------- diagnostics
    def _velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.grid_shape
        lo = np.array([-self.half_length, -self.radius, self.center_z - self.radius])
        hi = np.array([self.half_length, self.radius, self.center_z + self.radius])
        frac = (self.position - lo) / np.maximum(hi - lo, 1e-9)
        ix = np.clip((frac[:, 0] * nx).astype(np.int64), 0, nx - 1)
        iy = np.clip((frac[:, 1] * ny).astype(np.int64), 0, ny - 1)
        iz = np.clip((frac[:, 2] * nz).astype(np.int64), 0, nz - 1)
        flat = (ix * ny + iy) * nz + iz
        cells = nx * ny * nz
        count = np.bincount(flat, minlength=cells).astype(np.float64)
        field = np.zeros((cells, 3))
        for k in range(3):
            field[:, k] = np.bincount(flat, weights=self.velocity[:, k], minlength=cells)
        occupied = count > 0
        field[occupied] /= count[occupied][:, None]
        return field.reshape(nx, ny, nz, 3), occupied.reshape(nx, ny, nz)

    def _frame(self, time: float) -> StateFrame:
        gaps = self.contact_gaps
        if gaps.size == 0:
            gaps = np.zeros(1, dtype=np.float64)
        velocity_field, active_mask = self._velocity_field()
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.position, dtype=np.float64),
            velocity=np.ascontiguousarray(self.velocity, dtype=np.float64),
            mass=np.ascontiguousarray(self.mass, dtype=np.float64),
            body_id=self.body_id,
            attributes={
                "species_id": self.species,
                "particle_radius": self.radii,
                "mass_quantity": self.mass_quantity,
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(gaps, dtype=np.float64),
                "cumulative_mass_source": np.zeros(self.species_count, dtype=np.float64),
                "external_force": np.ascontiguousarray(self.gravity_force, dtype=np.float64),
                "velocity_field": velocity_field,
                "active_mask": active_mask,
            },
            scalars={
                "grid_spacing": self.grid_spacing,
                "characteristic_length": self.characteristic_length,
                "contact_length_scale": self.contact_length_scale,
                "time_step": self.dt,
                "angular_velocity_rad_s": self.omega,
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- advancing
    def advance_to(self, time: float):
        target = float(time)
        remaining = int(round((target - self.time) / self.dt))
        for k in range(max(remaining, 0)):
            self._step(record_contacts=(k == remaining - 1))
        if remaining > 0:
            self.time += remaining * self.dt
        if not (np.isfinite(self.position).all() and np.isfinite(self.velocity).all()):
            raise FloatingPointError("DEM state diverged: non-finite grain state")
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "soft_sphere_DEM_semi_implicit_euler",
            "particles": int(self.n),
            "time_step": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps_taken": int(self.step_index),
            "neighbour_rebuilds": int(self.rebuilds),
            "neighbour_pairs": int(len(self.pair_i)),
            "max_speed": float(self.max_speed),
            "max_penetration": float(self.max_penetration),
            "normal_stiffness_grain": float(self.KN_PP),
            "normal_stiffness_wall": float(self.KN_W),
            "friction_species": list(self.MU_SPECIES),
            "wall_friction": float(self.MU_WALL),
        }
