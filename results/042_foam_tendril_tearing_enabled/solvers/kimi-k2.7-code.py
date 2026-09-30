"""Particle-spring foam tendril solver with explicit tearing."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initial state from the scene scaffold.
        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=np.float64).copy()
        self.vel = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        if frame0.body_id is None:
            self.body_id = np.zeros(len(self.pos), dtype=np.int32)
        else:
            self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.radius = np.asarray(
            frame0.attributes.get("particle_radius", np.full(len(self.pos), 0.026)),
            dtype=np.float64,
        ).copy()
        self.foam_material = np.asarray(
            frame0.attributes.get("foam_material", np.ones(len(self.pos))),
            dtype=np.float64,
        ).copy()
        self.tear_candidate = np.asarray(
            frame0.attributes.get("tear_candidate", np.zeros(len(self.pos))),
            dtype=np.float64,
        ).copy()
        self.pos0 = self.pos.copy()
        self.time = float(frame0.time)

        # Read configuration safely. The runner passes a Scene whose .config
        # may not be available, so fall back to stored defaults.
        try:
            ic = scene.config["initial_conditions"]
        except AttributeError:
            # Build a default from the public config fields known at runtime.
            ic = {
                "particle_spacing": 0.045,
                "nominal_particle_radius": 0.026,
                "bulk_density": 180.0,
                "upper_pull_velocity": [0.0, 0.0, 0.2],
                "lower_settle_velocity": [0.0, 0.0, -0.03],
                "explicit_tearing": "enabled",
                "intended_tear_zone_z": [0.78, 1.12],
            }
        self.spacing = float(ic["particle_spacing"])
        self.nominal_radius = float(ic["nominal_particle_radius"])
        self.bulk_density = float(ic["bulk_density"])
        self.gravity = np.asarray(scene.config["gravity"], dtype=np.float64)
        self.upper_vel = np.asarray(ic["upper_pull_velocity"], dtype=np.float64)
        self.lower_vel = np.asarray(ic["lower_settle_velocity"], dtype=np.float64)
        self.tear_zone = np.asarray(ic["intended_tear_zone_z"], dtype=np.float64)
        self.z_tear = 0.5 * (self.tear_zone[0] + self.tear_zone[1])

        # Solver / material parameters.
        self.k = 1500.0
        self.c = 15.0
        self.bond_cutoff = 2.3 * self.spacing
        self.yield_strain = 1.02
        self.plastic_rate = 0.5
        self.tear_diameter = 0.07
        self.tear_time_fallback = 1.5
        self.dt_sub = 0.0005

        # Scene geometry for contact.
        # Floor top surface is at z=0 (cylinder centered at -0.04, height 0.08).
        self.floor_z = 0.0
        self.plate_z0 = 1.84
        self.plate_half_extents = np.array([0.78, 0.62], dtype=np.float64)
        self.plate_velocity = float(self.upper_vel[2])

        # Upper cap is kinematically pulled with the moving plate.
        self.adhered = self.pos[:, 2] > 1.45

        # Build initial spring network from nearest-neighbor pairs.
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.bond_cutoff, output_type="ndarray")
        self.bonds_i = pairs[:, 0].astype(np.int64)
        self.bonds_j = pairs[:, 1].astype(np.int64)
        dx = self.pos[self.bonds_j] - self.pos[self.bonds_i]
        self.bonds_rest = np.linalg.norm(dx, axis=1)
        self.bonds_active = np.ones(len(self.bonds_i), dtype=bool)

        # Deformation gradient precompute.
        self._build_deformation_precompute()

        self.torn = False
        self.torn_time: float | None = None

    def _build_deformation_precompute(self) -> None:
        N = len(self.pos)
        adj_to_i: list[list[int]] = [[] for _ in range(N)]
        adj_to_j: list[list[int]] = [[] for _ in range(N)]
        for b, (i, j) in enumerate(zip(self.bonds_i, self.bonds_j)):
            adj_to_i[i].append(b)
            adj_to_j[j].append(b)

        deform_nbrs: list[np.ndarray] = []
        deform_M_inv: list[np.ndarray] = []
        for i in range(N):
            nbrs: list[int] = []
            for b in adj_to_i[i]:
                nbrs.append(int(self.bonds_j[b]))
            for b in adj_to_j[i]:
                nbrs.append(int(self.bonds_i[b]))
            if len(nbrs) == 0:
                deform_nbrs.append(np.array([], dtype=np.int64))
                deform_M_inv.append(np.zeros((3, 3), dtype=np.float64))
                continue
            js = np.array(nbrs, dtype=np.int64)
            r_vecs = self.pos0[js] - self.pos0[i]
            M = r_vecs.T @ r_vecs
            trace = np.trace(M)
            reg = 1e-10 * max(trace, 1e-12)
            M += np.eye(3) * reg
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            deform_nbrs.append(js)
            deform_M_inv.append(M_inv)

        self.deform_nbrs = deform_nbrs
        self.deform_M_inv = np.array(deform_M_inv, dtype=np.float64)

    def _compute_F(self) -> np.ndarray:
        N = len(self.pos)
        S = np.zeros((N, 3, 3), dtype=np.float64)
        active = self.bonds_active
        if active.any():
            i = self.bonds_i[active]
            j = self.bonds_j[active]
            dx = self.pos[j] - self.pos[i]
            r = self.pos0[j] - self.pos0[i]
            contrib = dx[:, :, None] * r[:, None, :]
            np.add.at(S, i, contrib)
            np.add.at(S, j, contrib)
        F = np.einsum("nij,njk->nik", S, self.deform_M_inv)
        has_nbrs = np.array([len(js) > 0 for js in self.deform_nbrs])
        if not has_nbrs.all():
            F[~has_nbrs] = np.eye(3, dtype=np.float64)
        return F

    def _plate_bottom_z(self, t: float) -> float:
        return self.plate_z0 + self.plate_velocity * t

    def _step(self, dt: float) -> None:
        N = len(self.pos)

        # Gravity (external force).
        f_ext = self.mass[:, None] * self.gravity[None, :]

        # Internal cohesive spring forces.
        f_int = np.zeros((N, 3), dtype=np.float64)
        active = self.bonds_active
        if active.any():
            i = self.bonds_i[active]
            j = self.bonds_j[active]
            dx = self.pos[j] - self.pos[i]
            l = np.linalg.norm(dx, axis=1)
            l_safe = np.maximum(l, 1e-12)
            n = dx / l_safe[:, None]
            dv = self.vel[j] - self.vel[i]
            f_mag = self.k * (l - self.bonds_rest[active]) + self.c * np.sum(dv * n, axis=1)
            f_bond = f_mag[:, None] * n
            np.add.at(f_int, i, f_bond)
            np.add.at(f_int, j, -f_bond)

            # Plastic (irreversible) flow: foam yields under large tensile stretch.
            strain = l / self.bonds_rest[active]
            plastic_mask = strain > self.yield_strain
            if plastic_mask.any():
                active_idx = np.where(active)[0]
                idx = active_idx[plastic_mask]
                self.bonds_rest[idx] += self.plastic_rate * (l[plastic_mask] - self.bonds_rest[idx])

        # Symplectic Euler update.
        f_total = f_ext + f_int
        self.vel += dt * (f_total / self.mass[:, None])
        self.pos += dt * self.vel

        # Kinematic upper cap follows the moving plate.
        if self.adhered.any():
            self.vel[self.adhered] = self.upper_vel[None, :]

        # Floor collision.
        pen_floor = self.floor_z + self.radius - self.pos[:, 2]
        mask_floor = pen_floor > 0.0
        if mask_floor.any():
            self.pos[mask_floor, 2] = self.floor_z + self.radius[mask_floor]
            self.vel[mask_floor, 2] = -self.vel[mask_floor, 2] * 0.2

        # Moving upper plate collision for non-adhered particles.
        t_new = self.time + dt
        z_plate = self._plate_bottom_z(t_new)
        non_adhered = ~self.adhered
        if non_adhered.any():
            inside_xy = (
                np.abs(self.pos[non_adhered, 0]) < self.plate_half_extents[0]
            ) & (np.abs(self.pos[non_adhered, 1]) < self.plate_half_extents[1])
            pen_plate = self.pos[non_adhered, 2] + self.radius[non_adhered] - z_plate
            mask_plate = inside_xy & (pen_plate > 0.0)
            if mask_plate.any():
                idx = np.where(non_adhered)[0][mask_plate]
                self.pos[idx, 2] = z_plate - self.radius[idx]
                self.vel[idx, 2] = -self.vel[idx, 2] * 0.2

        self.time = t_new

        # Explicit tearing check.
        if not self.torn:
            self._check_and_tear()

    def _neck_diameter(self) -> float:
        mask = (self.pos[:, 2] >= self.tear_zone[0]) & (self.pos[:, 2] <= self.tear_zone[1])
        if mask.sum() == 0:
            return 1.0
        r = np.sqrt(self.pos[mask, 0] ** 2 + self.pos[mask, 1] ** 2)
        r95 = np.percentile(r, 95)
        return float(2.0 * r95)

    def _tear(self) -> None:
        # Remove all springs that straddle the tearing plane.
        cross = (self.pos[self.bonds_i, 2] <= self.z_tear) != (
            self.pos[self.bonds_j, 2] <= self.z_tear
        )
        active_cross = self.bonds_active & cross
        self.bonds_active[active_cross] = False
        self.torn = True
        self.torn_time = float(self.time)
        # Split into two disconnected fragments.
        self.body_id[:] = (self.pos[:, 2] > self.z_tear).astype(np.int32)

    def _check_and_tear(self) -> None:
        diameter = self._neck_diameter()
        if diameter < self.tear_diameter or self.time >= self.tear_time_fallback:
            self._tear()

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("Cannot advance backwards in time")
        dt_total = time - self.time
        if dt_total > 0.0:
            n_steps = max(1, int(np.ceil(dt_total / self.dt_sub)))
            dt = dt_total / n_steps
            for _ in range(n_steps):
                self._step(dt)
        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        F = self._compute_F()
        z_plate = self._plate_bottom_z(self.time)
        sd_floor = self.pos[:, 2] - self.floor_z
        sd_plate = z_plate - self.pos[:, 2]
        contact_sd = np.minimum(sd_floor, sd_plate)
        ext_force = self.mass[:, None] * self.gravity[None, :]
        mass_quantity = self.mass[:, None]

        attributes = {
            "particle_radius": self.radius,
            "foam_material": self.foam_material,
            "tear_candidate": self.tear_candidate,
            "deformation_gradient": F,
            "mass_quantity": mass_quantity,
        }
        fields = {
            "contact_signed_distance": contact_sd,
            "cumulative_mass_source": np.array([0.0], dtype=np.float64),
            "external_force": ext_force,
        }
        scalars = {
            "contact_length_scale": self.spacing,
            "explicit_tearing_enabled": 1.0,
            "initial_particle_count": float(len(self.pos)),
        }

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "particle_spring_foam",
            "num_particles": len(self.pos),
            "num_bonds_active": int(self.bonds_active.sum()),
            "torn": self.torn,
            "torn_time": self.torn_time,
        }
