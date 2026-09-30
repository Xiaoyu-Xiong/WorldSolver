"""Solver for the frictionless spaghetti plate benchmark.

The scene contains 31 flexible mass-spring strands falling into an axisymmetric
plate.  The implementation uses symplectic Euler time stepping with explicit
elastic forces, a simple discrete bending model, and penalty-based frictionless
contact (strand-strand and strand-plate).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.config = scene.metadata()
        self.current_time = 0.0

        init = scene.initial_state()
        self.position = np.asarray(init.position, dtype=float).copy()
        self.velocity = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.body_id = np.asarray(init.body_id, dtype=int).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in init.attributes.items()}

        self.n = self.position.shape[0]

        cfg = self.config
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.strand_count = int(cfg["strand_count"])
        strand_cfg = cfg["strand"]
        self.radius = float(strand_cfg["radius"])
        self.contact_d = 2.0 * self.radius
        self.strand_length = float(strand_cfg["length"])
        self.vps = int(strand_cfg["vertex_count"])

        # Material parameters tuned for stable explicit integration while
        # keeping the strands nearly inextensible and bending-resistant.
        E = 2.0e5  # Pa (effective Young's modulus)
        r = self.radius
        A = np.pi * r * r
        self.l0 = self.strand_length / (self.vps - 1)
        self.k_stretch = E * A / self.l0
        self.k_bend = 20.0
        self.k_contact = 2000.0
        self.c_contact = 0.05
        self.c_plate = 0.05
        self.c_damp = 0.001

        # Edge connectivity: edges for stretch, triplets for bending.
        edges = []
        bends = []
        for s in range(self.strand_count):
            base = s * self.vps
            for i in range(self.vps - 1):
                edges.append((base + i, base + i + 1))
            for i in range(1, self.vps - 1):
                bends.append((base + i - 1, base + i, base + i + 1))
        self.edges = np.asarray(edges, dtype=int)
        self.bends = np.asarray(bends, dtype=int)

        # Plate inner profile in the (radius, height) plane.  The interior of
        # the plate is to the left of each directed segment.
        self.plate_profile = np.asarray(
            [
                [0.10, 0.075],
                [0.88, 0.080],
                [1.05, 0.135],
                [1.14, 0.220],
            ],
            dtype=float,
        )

        # Cumulative budget fields (single global domain, one external force term).
        self.cum_external_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow = np.zeros((1, 3), dtype=float)
        self.cum_outflow = np.zeros((1, 3), dtype=float)
        self.cum_mass_source = np.zeros((1,), dtype=float)
        self._external_force = np.zeros((1, 1, 3), dtype=float)

        # Reference scalars used for normalization.
        self._scalars = {
            "friction_stick_speed_tolerance": 1e-3,
            "reference_force": 1.0,
            "reference_momentum": 0.1,
            "reference_velocity": 1.0,
        }

        # Fixed substep for explicit integration.
        self.dt_sub = 1.0e-4

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _plate_sdf_and_normal(self, pos: np.ndarray):
        """Return signed distance to the inner plate surface and the left normal.

        The signed distance is positive inside the hollow and negative inside
        the plate material.  ``nr`` and ``nz`` are the radial and vertical
        components of the 2-D left (hollow-side) normal.
        """
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        z = pos[:, 2]
        n = pos.shape[0]
        min_s = np.full(n, 1.0e6, dtype=float)
        nr_arr = np.zeros(n, dtype=float)
        nz_arr = np.zeros(n, dtype=float)

        for i in range(len(self.plate_profile) - 1):
            p1 = self.plate_profile[i]
            p2 = self.plate_profile[i + 1]
            dr = p2[0] - p1[0]
            dz = p2[1] - p1[1]
            seg_len = np.hypot(dr, dz)
            if seg_len < 1e-12:
                continue
            tr = dr / seg_len
            tz = dz / seg_len
            # Left (hollow-side) normal in the r-z plane.
            nr = -tz
            nz = tr
            s = (r - p1[0]) * nr + (z - p1[1]) * nz
            update = s < min_s
            min_s = np.minimum(min_s, s)
            nr_arr[update] = nr
            nz_arr[update] = nz

        return min_s, nr_arr, nz_arr

    def _collect_contacts(self):
        """Gather contact data required by the friction fields."""
        # Strand-strand contacts.
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(r=self.contact_d, output_type="ndarray")
        if pairs.shape[0] > 0:
            mask = self.body_id[pairs[:, 0]] != self.body_id[pairs[:, 1]]
            pairs = pairs[mask]

        # Strand-plate contacts.
        s, nr, nz = self._plate_sdf_and_normal(self.position)
        inside = s < 0.0
        n_plate = int(np.count_nonzero(inside))

        c = pairs.shape[0] + n_plate
        if c == 0:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        normals = np.zeros((c, 3), dtype=float)
        relvels = np.zeros((c, 3), dtype=float)
        normal_force = np.zeros(c, dtype=float)
        dyn = np.zeros(c, dtype=float)
        stat = np.zeros(c, dtype=float)

        idx = 0
        # Strand-strand contacts: normal points from the first vertex to the
        # second vertex; the acting side is the first vertex.
        if pairs.shape[0] > 0:
            pi = self.position[pairs[:, 0]]
            pj = self.position[pairs[:, 1]]
            vi = self.velocity[pairs[:, 0]]
            vj = self.velocity[pairs[:, 1]]
            diff = pj - pi
            dist = np.linalg.norm(diff, axis=1)
            dist[dist < 1e-12] = 1e-12
            nvec = diff / dist[:, None]
            overlap = self.contact_d - dist
            nf = self.k_contact * overlap
            n_contacts = pairs.shape[0]
            normals[:n_contacts] = nvec
            relvels[:n_contacts] = vi - vj
            normal_force[:n_contacts] = nf
            dyn[:n_contacts] = 0.0
            stat[:n_contacts] = 0.0
            idx = n_contacts

        # Strand-plate contacts: normal points from the strand into the plate
        # material (outward from the hollow-side normal).
        if n_plate > 0:
            plate_indices = np.nonzero(inside)[0]
            pos_in = self.position[plate_indices]
            vel_in = self.velocity[plate_indices]
            r_in = np.sqrt(pos_in[:, 0] ** 2 + pos_in[:, 1] ** 2)
            r_in[r_in < 1e-12] = 1e-12
            nx = -nr[inside] * pos_in[:, 0] / r_in
            ny = -nr[inside] * pos_in[:, 1] / r_in
            nz_out = -nz[inside]
            nvec = np.column_stack((nx, ny, nz_out))
            norms = np.linalg.norm(nvec, axis=1)
            norms[norms < 1e-12] = 1e-12
            nvec = nvec / norms[:, None]
            penetration = -s[inside]
            nf = self.k_contact * penetration
            normals[idx:] = nvec
            relvels[idx:] = vel_in
            normal_force[idx:] = nf
            dyn[idx:] = 0.0
            stat[idx:] = 0.0

        active = np.ones(c, dtype=float)
        tangential = np.zeros((c, 3), dtype=float)
        return normals, normal_force, relvels, tangential, dyn, stat, active

    def _step(self, dt: float) -> None:
        force = np.zeros_like(self.position)

        # Gravity.
        force += self.mass[:, None] * self.gravity[None, :]

        # Stretch forces along each edge.
        p1 = self.position[self.edges[:, 0]]
        p2 = self.position[self.edges[:, 1]]
        e_vec = p2 - p1
        e_len = np.linalg.norm(e_vec, axis=1)
        e_len[e_len < 1e-12] = 1e-12
        n = e_vec / e_len[:, None]
        delta = e_len - self.l0
        f = self.k_stretch * delta[:, None] * n
        np.add.at(force, self.edges[:, 0], f)
        np.add.at(force, self.edges[:, 1], -f)

        # Bending forces: each interior vertex is pulled toward the midpoint of
        # its two neighbours.
        i0 = self.bends[:, 0]
        i1 = self.bends[:, 1]
        i2 = self.bends[:, 2]
        p_i = self.position[i0]
        p_j = self.position[i1]
        p_k = self.position[i2]
        mid = 0.5 * (p_i + p_k)
        disp = mid - p_j
        f_b = self.k_bend * disp
        np.add.at(force, i1, f_b)
        np.add.at(force, i0, -0.5 * f_b)
        np.add.at(force, i2, -0.5 * f_b)

        # Strand-strand contact (frictionless penalty).
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(r=self.contact_d, output_type="ndarray")
        if pairs.shape[0] > 0:
            mask = self.body_id[pairs[:, 0]] != self.body_id[pairs[:, 1]]
            pairs = pairs[mask]
        if pairs.shape[0] > 0:
            pi = self.position[pairs[:, 0]]
            pj = self.position[pairs[:, 1]]
            vi = self.velocity[pairs[:, 0]]
            vj = self.velocity[pairs[:, 1]]
            diff = pj - pi
            dist = np.linalg.norm(diff, axis=1)
            dist[dist < 1e-12] = 1e-12
            nvec = diff / dist[:, None]
            overlap = self.contact_d - dist
            rel_v = np.sum((vi - vj) * nvec, axis=1)
            f_mag = self.k_contact * overlap + self.c_contact * np.maximum(0.0, rel_v)
            f = f_mag[:, None] * nvec
            np.add.at(force, pairs[:, 0], -f)
            np.add.at(force, pairs[:, 1], f)

        # Strand-plate contact (frictionless penalty).
        s, nr, nz = self._plate_sdf_and_normal(self.position)
        inside = s < 0.0
        total_plate_force = np.zeros(3, dtype=float)
        if np.any(inside):
            idx = np.nonzero(inside)[0]
            pos_in = self.position[idx]
            vel_in = self.velocity[idx]
            r_in = np.sqrt(pos_in[:, 0] ** 2 + pos_in[:, 1] ** 2)
            r_in[r_in < 1e-12] = 1e-12
            # Hollow-side normal in 3-D (pushes strands back into the interior).
            nx = nr[inside] * pos_in[:, 0] / r_in
            ny = nr[inside] * pos_in[:, 1] / r_in
            nvec = np.column_stack((nx, ny, nz[inside]))
            norms = np.linalg.norm(nvec, axis=1)
            norms[norms < 1e-12] = 1e-12
            nvec = nvec / norms[:, None]
            penetration = -s[inside]
            v_dot_n = np.sum(vel_in * nvec, axis=1)
            f_mag = self.k_contact * penetration + self.c_plate * np.maximum(0.0, -v_dot_n)
            f = f_mag[:, None] * nvec
            np.add.at(force, idx, f)
            total_plate_force = f.sum(axis=0)

        # Light velocity damping.
        force -= self.c_damp * self.velocity

        # Update external-force budget (gravity + plate contact) and its
        # cumulative impulse for the global strands domain.
        total_gravity = self.mass.sum() * self.gravity
        self._external_force[0, 0] = total_gravity + total_plate_force
        self.cum_external_impulse[0] += self._external_force[0, 0] * dt

        # Symplectic Euler integration.
        self.velocity += dt * force / self.mass[:, None]
        self.position += dt * self.velocity

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            time = self.current_time
        while self.current_time < time - 1e-12:
            dt = min(self.dt_sub, time - self.current_time)
            self._step(dt)
            self.current_time += dt
        return self._build_frame()

    def _build_frame(self) -> StateFrame:
        normals, normal_force, relvel, tangential, dyn, stat, active = self._collect_contacts()

        attributes = dict(self.attributes)
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": self._external_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "friction_active_mask": active,
            "friction_dynamic_coefficient": dyn,
            "friction_normal": normals,
            "friction_normal_force": normal_force,
            "friction_relative_velocity": relvel,
            "friction_static_coefficient": stat,
            "friction_tangential_force": tangential,
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=self._scalars.copy(),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt_sub": self.dt_sub,
            "n_particles": self.n,
            "seed": self.seed,
        }
