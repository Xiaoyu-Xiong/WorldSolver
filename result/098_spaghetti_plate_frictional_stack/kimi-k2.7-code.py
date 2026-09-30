"""Mass-spring chain solver with bending, plate collision, and dry friction.

The solver advances 31 spaghetti strands (each a 16-node elastic rod) under
gravity inside an axisymmetric plate.  Stretch and bending springs model the
fibres, penalty springs plus Coulomb friction handle strand-plate and
strand-strand contact, and an explicit second-order stable time stepper is
used with sub-stepping.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        frame0 = scene.initial_state()
        self.t = float(frame0.time)
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=int).copy()
        self.attr = {
            k: np.asarray(v).copy() for k, v in frame0.attributes.items()
        }
        self.N = self.pos.shape[0]

        meta = scene.metadata()
        self.strand_radius = float(meta.get("strand_radius", 0.006))
        self.n_strands = int(meta.get("strand_count", 31))
        self.n_nodes = int(meta.get("nodes_per_strand", 16))
        self.strand_length = float(meta.get("strand_length", 0.72))

        # Gravity
        self.g = np.array([0.0, 0.0, -9.81])
        self.grav_force = self.mass[:, None] * self.g[None, :]

        # Axisymmetric plate inner profile (r, z).  The interior of the bowl is
        # to the left of each directed segment.  Add a tall vertical rim to keep
        # strands from escaping over the lip.
        self.plate_r = np.array(
            [0.0, 0.1, 0.4, 0.52, 0.57, 0.57], dtype=float
        )
        self.plate_z = np.array(
            [0.0, 0.0, 0.018, 0.075, 0.125, 10.0], dtype=float
        )

        # Topology: straight chain edges and bending triples
        self.L0 = self.strand_length / (self.n_nodes - 1)
        edge_i0, edge_i1 = [], []
        bend_im1, bend_i, bend_ip1 = [], [], []
        for s in range(self.n_strands):
            base = s * self.n_nodes
            for e in range(self.n_nodes - 1):
                edge_i0.append(base + e)
                edge_i1.append(base + e + 1)
            for n in range(1, self.n_nodes - 1):
                bend_im1.append(base + n - 1)
                bend_i.append(base + n)
                bend_ip1.append(base + n + 1)
        self.edge_i0 = np.array(edge_i0)
        self.edge_i1 = np.array(edge_i1)
        self.bend_im1 = np.array(bend_im1)
        self.bend_i = np.array(bend_i)
        self.bend_ip1 = np.array(bend_ip1)

        # Material / numerical parameters
        self.k_stretch = 1.0e3
        self.k_bend = 10.0
        self.k_contact = 2.0e3
        self.mu = 0.3
        self.vel_damping = 1.0  # mass-proportional Rayleigh damping [1/s]
        self.contact_damping_zeta = 1.0  # critical-ish contact damping
        self.stick_tol = 1e-3  # m/s
        self.contact_cutoff = 2.0 * self.strand_radius

        # Budget bookkeeping
        self.G = self.n_strands
        self.K = 1
        self.domain_weights = np.zeros((self.G, self.N), dtype=float)
        for g in range(self.G):
            self.domain_weights[g, self.body_id == g] = 1.0
        self.body_gravity = np.zeros((self.G, 3), dtype=float)
        for g in range(self.G):
            self.body_gravity[g] = self.grav_force[self.body_id == g].sum(axis=0)
        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=float)
        self.cum_in = np.zeros((self.G, 3), dtype=float)
        self.cum_out = np.zeros((self.G, 3), dtype=float)

        # Time stepper control
        times = scene.sample_times()
        if len(times) >= 2:
            self.dt_sample = float(times[1] - times[0])
        else:
            self.dt_sample = 0.02
        self.substep_dt = 2.0e-4

    def advance_to(self, time: float) -> StateFrame:
        h = time - self.t
        if h < 0.0:
            raise ValueError("advance_to time must be non-decreasing")
        if h > 0.0:
            n_sub = max(1, int(np.ceil(h / self.substep_dt)))
            dt = h / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.t = time
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_dt": self.substep_dt,
            "k_stretch": self.k_stretch,
            "k_bend": self.k_bend,
            "k_contact": self.k_contact,
            "mu": self.mu,
        }

    # --------------------------------------------------------------------- #
    # Physics core
    # --------------------------------------------------------------------- #
    def _step(self, dt: float) -> None:
        F = self._base_forces(self.pos, self.vel)

        # Plate contacts
        plate = self._plate_contact_tensors(self.pos, self.vel, F, dt)
        if plate is not None:
            idx = plate["index"]
            f = plate["normal_force"][:, None] * plate["normal"] + plate["tangential_force"]
            F[idx] += f

        # Strand contacts
        strand = self._strand_contact_tensors(self.pos, self.vel, F, dt)
        if strand is not None:
            i = strand["i"]
            j = strand["j"]
            f = strand["normal_force"][:, None] * strand["normal"] + strand["tangential_force"]
            np.add.at(F, i, f)
            np.subtract.at(F, j, f)

        # Update budgets (gravity is the only external force tracked)
        self.cum_ext_impulse += self.body_gravity * dt

        # Symplectic Euler integration
        self.vel += (F / self.mass[:, None]) * dt
        self.pos += self.vel * dt

    def _base_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        F = np.zeros_like(pos)

        # Gravity
        F += self.grav_force

        # Stretch springs
        dx = pos[self.edge_i1] - pos[self.edge_i0]
        L = np.linalg.norm(dx, axis=1, keepdims=True)
        L = np.where(L < 1e-12, 1.0, L)
        f = self.k_stretch * (L - self.L0) * (dx / L)
        np.add.at(F, self.edge_i0, -f)
        np.add.at(F, self.edge_i1, f)

        # Bending resistance (midpoint spring, momentum-conserving)
        target = 0.5 * (pos[self.bend_im1] + pos[self.bend_ip1])
        f_mid = self.k_bend * (target - pos[self.bend_i])
        F[self.bend_i] += f_mid
        F[self.bend_im1] -= 0.5 * f_mid
        F[self.bend_ip1] -= 0.5 * f_mid

        # Velocity damping
        F -= self.mass[:, None] * self.vel_damping * vel

        return F

    # --------------------------------------------------------------------- #
    # Plate signed distance and contact
    # --------------------------------------------------------------------- #
    def _plate_sdf(self, pos: np.ndarray):
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        r = np.sqrt(x * x + y * y)

        rA = self.plate_r[:-1]
        zA = self.plate_z[:-1]
        rB = self.plate_r[1:]
        zB = self.plate_z[1:]
        dr = rB - rA
        dz = zB - zA
        seg_len2 = dr * dr + dz * dz

        # Projection onto each segment in the (r,z) plane
        t = ((r[:, None] - rA) * dr + (z[:, None] - zA) * dz) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        cr = rA + t * dr
        cz = zA + t * dz
        d2 = (r[:, None] - cr) ** 2 + (z[:, None] - cz) ** 2
        seg_idx = np.argmin(d2, axis=1)

        N = len(r)
        closest_r = cr[np.arange(N), seg_idx]
        closest_z = cz[np.arange(N), seg_idx]

        drs = dr[seg_idx]
        dzs = dz[seg_idx]
        seg_len = np.sqrt(seg_len2[seg_idx])

        # Interior is the left side of each directed segment for this convex chain
        cross = drs * (z - closest_z) - dzs * (r - closest_r)
        sign = np.where(cross >= 0.0, 1.0, -1.0)
        dist = np.sqrt(np.min(d2, axis=1))
        sd = sign * dist

        # Inward unit normal in 3D (radial part times r direction)
        n_r = -dzs / seg_len
        n_z = drs / seg_len
        nx = np.empty(N, dtype=float)
        ny = np.empty(N, dtype=float)
        mask = r > 1e-12
        nx[mask] = x[mask] / r[mask] * n_r[mask]
        ny[mask] = y[mask] / r[mask] * n_r[mask]
        nx[~mask] = 0.0
        ny[~mask] = 0.0
        n = np.stack([nx, ny, n_z], axis=1)
        return sd, n

    def _plate_contact_tensors(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        F_base: np.ndarray,
        dt: float,
    ):
        sd, n = self._plate_sdf(pos)
        pen = self.strand_radius - sd
        active = pen > 0.0
        if not np.any(active):
            return None

        idx = np.where(active)[0]
        v = vel[idx]
        n_active = n[idx]
        m = self.mass[idx]

        vn = np.sum(v * n_active, axis=1)
        c = self.contact_damping_zeta * 2.0 * np.sqrt(m * self.k_contact)
        Nmag = self.k_contact * pen[idx] - c * vn
        Nmag = np.maximum(Nmag, 0.0)

        v_t = v - vn[:, None] * n_active
        vt_norm = np.linalg.norm(v_t, axis=1)

        # Tangential component of non-contact acceleration for static friction
        a_base = F_base[idx] / m[:, None]
        a_n = np.sum(a_base * n_active, axis=1)
        a_t = a_base - a_n[:, None] * n_active

        f_t = self._friction_vector(v_t, vt_norm, Nmag, m, dt, a_t)

        return {
            "index": idx,
            "normal": n_active,
            "rel_vel": v,
            "normal_force": Nmag,
            "tangential_force": f_t,
            "active_mask": np.ones_like(Nmag, dtype=float),
            "static_coeff": np.full_like(Nmag, self.mu),
            "dynamic_coeff": np.full_like(Nmag, self.mu),
        }

    # --------------------------------------------------------------------- #
    # Strand-strand contact
    # --------------------------------------------------------------------- #
    def _strand_pairs(self, pos: np.ndarray) -> np.ndarray:
        tree = cKDTree(pos, leafsize=16)
        pairs = tree.query_pairs(self.contact_cutoff)
        if not pairs:
            return np.empty((0, 2), dtype=int)
        pairs = np.array(list(pairs), dtype=int)
        # Different-body contacts only
        mask = self.body_id[pairs[:, 0]] != self.body_id[pairs[:, 1]]
        return pairs[mask]

    def _strand_contact_tensors(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        F_base: np.ndarray,
        dt: float,
    ):
        pairs = self._strand_pairs(pos)
        if pairs.size == 0:
            return None

        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = pos[i] - pos[j]
        d = np.linalg.norm(dx, axis=1)
        active = d < self.contact_cutoff
        if not np.any(active):
            return None

        i = i[active]
        j = j[active]
        d = d[active]
        dx = dx[active]

        n = dx / d[:, None]
        pen = self.contact_cutoff - d

        vrel = vel[i] - vel[j]
        vn = np.sum(vrel * n, axis=1)
        m_i = self.mass[i]
        m_j = self.mass[j]
        m_eff = (m_i * m_j) / (m_i + m_j)

        c = self.contact_damping_zeta * 2.0 * np.sqrt(m_eff * self.k_contact)
        Nmag = self.k_contact * pen - c * vn
        Nmag = np.maximum(Nmag, 0.0)

        v_t = vrel - vn[:, None] * n
        vt_norm = np.linalg.norm(v_t, axis=1)

        a_i0 = F_base[i] / m_i[:, None]
        a_j0 = F_base[j] / m_j[:, None]
        a_rel = a_i0 - a_j0
        a_n = np.sum(a_rel * n, axis=1)
        a_t = a_rel - a_n[:, None] * n

        f_t = self._friction_vector(v_t, vt_norm, Nmag, m_eff, dt, a_t)

        return {
            "i": i,
            "j": j,
            "normal": n,
            "rel_vel": vrel,
            "normal_force": Nmag,
            "tangential_force": f_t,
            "active_mask": np.ones_like(Nmag, dtype=float),
            "static_coeff": np.full_like(Nmag, self.mu),
            "dynamic_coeff": np.full_like(Nmag, self.mu),
        }

    # --------------------------------------------------------------------- #
    # Friction helper
    # --------------------------------------------------------------------- #
    def _friction_vector(
        self,
        v_t: np.ndarray,
        vt_norm: np.ndarray,
        Nmag: np.ndarray,
        m_eff: np.ndarray,
        dt: float,
        a_t: np.ndarray,
    ) -> np.ndarray:
        # Static friction when slipping slowly
        static = vt_norm < self.stick_tol
        f = np.zeros_like(v_t)

        # Static branch
        if np.any(static):
            a = a_t[static]
            a_norm = np.linalg.norm(a, axis=1)
            mask = a_norm > 1e-12
            dir_a = np.zeros_like(a)
            dir_a[mask] = a[mask] / a_norm[mask, None]
            f_mag = np.minimum(
                self.mu * Nmag[static], m_eff[static] * a_norm
            )
            f[static] = -f_mag[:, None] * dir_a

        # Kinetic branch
        kinetic = ~static
        if np.any(kinetic):
            v = v_t[kinetic]
            vn = vt_norm[kinetic]
            dir_v = -v / np.maximum(vn, 1e-12)[:, None]
            f_mag = np.minimum(self.mu * Nmag[kinetic], m_eff[kinetic] * vn / dt)
            f[kinetic] = f_mag[:, None] * dir_v

        return f

    # --------------------------------------------------------------------- #
    # Frame export
    # --------------------------------------------------------------------- #
    def _build_frame(self, time: float) -> StateFrame:
        # Recompute contact state at the sample instant for trajectory fields
        F_base = self._base_forces(self.pos, self.vel)
        plate = self._plate_contact_tensors(self.pos, self.vel, F_base, self.substep_dt)
        strand = self._strand_contact_tensors(self.pos, self.vel, F_base, self.substep_dt)

        normals = []
        rel_vels = []
        normal_forces = []
        tang_forces = []
        active_masks = []
        static_coeffs = []
        dynamic_coeffs = []

        if plate is not None:
            normals.append(plate["normal"])
            rel_vels.append(plate["rel_vel"])
            normal_forces.append(plate["normal_force"])
            tang_forces.append(plate["tangential_force"])
            active_masks.append(plate["active_mask"])
            static_coeffs.append(plate["static_coeff"])
            dynamic_coeffs.append(plate["dynamic_coeff"])

        if strand is not None:
            normals.append(strand["normal"])
            rel_vels.append(strand["rel_vel"])
            normal_forces.append(strand["normal_force"])
            tang_forces.append(strand["tangential_force"])
            active_masks.append(strand["active_mask"])
            static_coeffs.append(strand["static_coeff"])
            dynamic_coeffs.append(strand["dynamic_coeff"])

        if normals:
            friction_normal = np.concatenate(normals, axis=0)
            friction_rel_vel = np.concatenate(rel_vels, axis=0)
            friction_normal_force = np.concatenate(normal_forces, axis=0)
            friction_tangential_force = np.concatenate(tang_forces, axis=0)
            friction_active_mask = np.concatenate(active_mask, axis=0)
            friction_static_coeff = np.concatenate(static_coeffs, axis=0)
            friction_dynamic_coeff = np.concatenate(dynamic_coeffs, axis=0)
        else:
            friction_normal = np.zeros((0, 3), dtype=float)
            friction_rel_vel = np.zeros((0, 3), dtype=float)
            friction_normal_force = np.zeros(0, dtype=float)
            friction_tangential_force = np.zeros((0, 3), dtype=float)
            friction_active_mask = np.zeros(0, dtype=float)
            friction_static_coeff = np.zeros(0, dtype=float)
            friction_dynamic_coeff = np.zeros(0, dtype=float)

        total_mass = float(self.mass.sum())
        total_weight = float((self.mass * (-self.g[2])).sum())

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_in.copy(),
            "budget_cumulative_outflow_momentum": self.cum_out.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.body_gravity[:, None, :].copy(),
            "cumulative_mass_source": np.zeros(self.K, dtype=float),
            "friction_active_mask": friction_active_mask,
            "friction_dynamic_coefficient": friction_dynamic_coeff,
            "friction_normal": friction_normal,
            "friction_normal_force": friction_normal_force,
            "friction_relative_velocity": friction_rel_vel,
            "friction_static_coefficient": friction_static_coeff,
            "friction_tangential_force": friction_tangential_force,
        }

        attributes = dict(self.attr)
        attributes["mass_quantity"] = self.mass.copy()

        scalars = {
            "friction_stick_speed_tolerance": self.stick_tol,
            "reference_force": max(total_weight, 1e-6),
            "reference_momentum": max(total_mass, 1e-6),
            "reference_velocity": 1.0,
        }

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
