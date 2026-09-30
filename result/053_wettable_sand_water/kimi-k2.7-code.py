"""Particle-based physics solver for wettable sand and water.

The solver advances a mixed granular/liquid particle system with explicit
semi-implicit Euler time stepping, penalty contact, Coulomb friction, and
per-sand-particle wetting transport.  It exports the required StateFrame
fields on the frozen diagnostic grid.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit particle solver for the wettable-sand-water scene."""

    # --- frozen simulation parameters (SI) ---------------------------------
    DT = 0.001                      # sub-step size (s)
    R_CUT = 0.09                    # particle interaction cutoff (m)
    K_CONTACT = 500.0               # contact stiffness (N/m)
    C_DAMP = 10.0                  # normal damping (kg/s)
    MU_SAND = 0.55                  # sand-sand friction coefficient
    MU_WATER = 0.02               # water-water friction
    MU_SAND_WATER = 0.15            # sand-water friction
    K_FRICTION = 200.0             # tangential friction rate scale (kg/s)
    K_FLOOR = 1000.0               # floor normal stiffness (N/m)
    C_FLOOR = 20.0                 # floor damping (kg/s)
    WET_RATE = 2.0                 # wetness transfer rate (1/s at contact)
    WET_DIFF = 0.0005              # wetness diffusion among sand (1/s)
    WET_SAT = 0.05                 # saturation wetness for dry->wet color map
    FLOOR_Z = 0.0

    # diagnostic grid
    GRID_ORIGIN = np.array([-1.30, -0.80, 0.00])
    GRID_H = 0.10
    GRID_SHAPE = (26, 16, 24)      # covers [-1.3,1.3] x [-0.8,0.8] x [0,2.4]


    def initialize(self, scene, seed: int) -> None:
        """Capture the frozen initial state and prepare buffers."""
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.scene = scene

        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()

        # attributes are required to be mutable copies
        self.attributes = {
            k: np.asarray(v).copy() for k, v in frame0.attributes.items()
        }
        # ensure exact expected names exist
        self.wetness = np.asarray(self.attributes["wetness"], dtype=np.float64)
        self.water_conc = np.asarray(
            self.attributes["water_concentration"], dtype=np.float64
        )
        self.solid_frac = np.asarray(
            self.attributes["solid_phase_fraction"], dtype=np.float64
        )
        self.material_id = np.asarray(
            self.attributes["material_id"], dtype=np.int32
        )

        self.sand_mask = self.body_id == 1
        self.water_mask = self.body_id == 2
        self.n = len(self.position)

        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.current_time = 0.0

        # cumulative sources / interaction bookkeeping
        self.cumulative_mass_source = np.zeros(2, dtype=np.float64)
        self.contact_length_scale = self.GRID_H

        # small random velocity perturbation to break perfect lattice lock
        noise = self.rng.normal(scale=0.005, size=(self.n, 3))
        self.velocity += noise

    def _build_frame(self, time: float) -> StateFrame:
        """Build a StateFrame from the current particle state."""
        # diagnostic grid quantities
        grid = self._diagnostic_grid()

        # contact signed distance to floor for all near-contact samples
        contact_dist = self.position[:, 2] - self.FLOOR_Z
        near = contact_dist < 0.02
        if np.any(near):
            signed = contact_dist[near]
        else:
            signed = np.zeros(0, dtype=np.float64)

        # interaction force on each phase (floor contact only; internal pairs cancel)
        interaction_force = self._floor_contact_forces()

        attributes = {
            "material_id": self.material_id.copy(),
            "solid_phase_fraction": self.solid_frac.copy(),
            "water_concentration": self.water_conc.copy(),
            "wetness": self.wetness.copy(),
            "mass_quantity": self.mass.copy(),
        }

        sand_mass = float(self.mass[self.sand_mask].sum())
        water_mass = float(self.mass[self.water_mask].sum())

        frame = StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields={
                "active_mask": grid["active_mask"],
                "contact_signed_distance": signed,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "interaction_force": interaction_force,
                "scalar_source": grid["scalar_source"],
                "transported_scalar": grid["transported_scalar"],
                "velocity_field": grid["velocity_field"],
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "grid_spacing": float(self.GRID_H),
                "sand_mass": sand_mass,
                "water_mass": water_mass,
            },
        )
        return frame

    def _diagnostic_grid(self) -> dict:
        """Return diagnostic 3-D grid fields from current particles."""
        nx, ny, nz = self.GRID_SHAPE
        h = self.GRID_H
        origin = self.GRID_ORIGIN

        # per-particle grid integer indices
        idx_f = (self.position - origin) / h
        idx = np.floor(idx_f).astype(np.int64)
        # clamp to grid bounds
        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
        flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]

        active_mask = np.zeros(self.GRID_SHAPE, dtype=np.float64)
        scalar = np.zeros(self.GRID_SHAPE, dtype=np.float64)
        velocity = np.zeros((*self.GRID_SHAPE, 3), dtype=np.float64)
        scalar_src = np.zeros(self.GRID_SHAPE, dtype=np.float64)

        if self.n > 0:
            counts = np.bincount(flat, minlength=nx * ny * nz).astype(np.float64)
            counts = counts.reshape(self.GRID_SHAPE)
            active_mask = (counts > 0).astype(np.float64)

            # scalar: wetness for sand, water concentration for water
            particle_scalar = np.where(self.sand_mask, self.wetness, self.water_conc)
            scalar_sum = np.bincount(flat, weights=particle_scalar, minlength=nx * ny * nz)
            scalar = (scalar_sum / np.maximum(counts.ravel(), 1.0)).reshape(self.GRID_SHAPE)

            for dim in range(3):
                v_sum = np.bincount(
                    flat, weights=self.velocity[:, dim], minlength=nx * ny * nz
                )
                velocity[:, :, :, dim] = (v_sum / np.maximum(counts.ravel(), 1.0)).reshape(
                    self.GRID_SHAPE
                )

        return {
            "active_mask": active_mask,
            "transported_scalar": scalar,
            "velocity_field": velocity,
            "scalar_source": scalar_src,
        }

    def _floor_contact_forces(self) -> np.ndarray:
        """Total floor contact force per phase (sand/water)."""
        forces = np.zeros((2, 3), dtype=np.float64)
        # include wall contacts
        z = self.position[:, 2]
        v_z = self.velocity[:, 2]
        # simple penalty against the floor plane
        overlap = np.maximum(-z, 0.0)
        mask = overlap > 0.0
        if not np.any(mask):
            return forces
        # upward normal force (positive z)
        fn = self.K_FLOOR * overlap[mask] - self.C_FLOOR * v_z[mask]
        fn = np.maximum(fn, 0.0)
        # floor tangential friction
        v_tan = self.velocity[mask, :2].copy()
        v_tan_mag = np.linalg.norm(v_tan, axis=1)
        mu = np.where(self.sand_mask[mask], self.MU_SAND, self.MU_WATER)
        ftan_mag = np.minimum(mu * fn, self.K_FRICTION * v_tan_mag)
        # friction opposes motion; leave as zero when stationary
        dir_tan = np.zeros_like(v_tan)
        non_zero = v_tan_mag > 1e-12
        dir_tan[non_zero] = -v_tan[non_zero] / v_tan_mag[non_zero, None]
        ftan = ftan_mag[:, None] * dir_tan

        fvec = np.column_stack((ftan[:, 0], ftan[:, 1], fn))
        for phase in (0, 1):
            pmask = self.body_id[mask] == (phase + 1)
            if np.any(pmask):
                forces[phase] = fvec[pmask].sum(axis=0)
        return forces

    def _pairwise_forces(self) -> np.ndarray:
        """Compute internal pairwise contact forces on every particle."""
        force = self.mass[:, None] * self.gravity[None, :]
        if self.n < 2:
            return force

        tree = cKDTree(self.position)
        pairs = tree.sparse_distance_matrix(
            tree, max_distance=self.R_CUT, output_type="coo_matrix"
        )
        i = pairs.row
        j = pairs.col
        d = pairs.data
        # remove self-pairs and keep each unordered pair once
        mask = (i != j) & (i < j)
        i = i[mask]
        j = j[mask]
        d = d[mask]
        if i.size == 0:
            return force

        dx = self.position[j] - self.position[i]
        d = np.maximum(d, 1e-12)
        n = dx / d[:, None]
        overlap = self.R_CUT - d
        overlap = np.maximum(overlap, 0.0)

        vrel = self.velocity[j] - self.velocity[i]
        vrel_n = np.einsum("ij,ij->i", vrel, n)
        vrel_t = vrel - vrel_n[:, None] * n
        vrel_t_mag = np.linalg.norm(vrel_t, axis=1)

        # normal contact (repulsive only)
        fn_mag = self.K_CONTACT * overlap - self.C_DAMP * vrel_n
        fn_mag = np.maximum(fn_mag, 0.0)

        # friction model
        ti = self.body_id[i]
        tj = self.body_id[j]
        # sand-sand = 1, water-water = 2, cross = 3
        pair_type = np.where((ti == 1) & (tj == 1), 0,
                     np.where((ti == 2) & (tj == 2), 1, 2))
        mu = np.empty_like(pair_type, dtype=np.float64)
        mu[pair_type == 0] = self.MU_SAND
        mu[pair_type == 1] = self.MU_WATER
        mu[pair_type == 2] = self.MU_SAND_WATER

        ftan_mag = np.minimum(mu * fn_mag, self.K_FRICTION * vrel_t_mag)
        ftan = np.zeros_like(vrel_t)
        moving = vrel_t_mag > 1e-12
        ftan[moving] = -(ftan_mag[moving] / vrel_t_mag[moving])[:, None] * (
            vrel_t[moving] / vrel_t_mag[moving, None]
        )

        f_on_i = fn_mag[:, None] * n + ftan
        # scatter to particles
        np.add.at(force, i, f_on_i)
        np.add.at(force, j, -f_on_i)
        return force

    def _transfer_wetness(self, pairs_i, pairs_j, pairs_d, dt: float) -> None:
        """Update per-sand-particle wetness from water contact and diffusion."""
        if pairs_i.size == 0:
            return

        ti = self.body_id[pairs_i]
        tj = self.body_id[pairs_j]

        # Sand-water contact: water wets sand
        sw = ((ti == 1) & (tj == 2)) | ((ti == 2) & (tj == 1))
        if np.any(sw):
            # For each pair figure out sand index and water amount
            sand_idx = np.where(sw, np.where(ti == 1, pairs_i, pairs_j), -1)
            water_conc_pair = np.where(ti == 2, self.water_conc[pairs_i], self.water_conc[pairs_j])
            valid = sw & (sand_idx >= 0)
            if np.any(valid):
                inc = self.WET_RATE * dt * water_conc_pair[valid]
                # accumulate per sand particle
                accum = np.bincount(sand_idx[valid], weights=inc, minlength=self.n)
                self.wetness += accum
                np.clip(self.wetness, 0.0, self.WET_SAT, out=self.wetness)

        # Sand-sand diffusion of wetness
        ss = (ti == 1) & (tj == 1)
        if np.any(ss):
            si = pairs_i[ss]
            sj = pairs_j[ss]
            wi = self.wetness[si]
            wj = self.wetness[sj]
            # only diffuse from wetter to dryer; sign handled below
            delta = self.WET_DIFF * dt * (wj - wi)
            self.wetness[si] += delta
            self.wetness[sj] -= delta
            np.clip(self.wetness, 0.0, self.WET_SAT, out=self.wetness)

    def _enforce_boundaries(self) -> None:
        """Keep particles inside the tray and above the floor."""
        # floor: reflect and damp vertical velocity
        below = self.position[:, 2] < self.FLOOR_Z
        if np.any(below):
            self.position[below, 2] = self.FLOOR_Z
            self.velocity[below, 2] = -0.3 * self.velocity[below, 2]
            # apply floor friction to tangential components
            self.velocity[below, :2] *= 0.8

    def _substep(self, dt: float) -> None:
        """Advance the particle state by one sub-step."""
        # Pairwise contact and gravity
        force = self._pairwise_forces()

        # Semi-implicit Euler integration
        acc = force / self.mass[:, None]
        self.velocity += acc * dt
        self.position += self.velocity * dt

        # Boundary enforcement
        self._enforce_boundaries()

        # Wetting: reuse the same pair list (rebuild tree, cheap)
        tree = cKDTree(self.position)
        pairs = tree.sparse_distance_matrix(
            tree, max_distance=self.R_CUT, output_type="coo_matrix"
        )
        i = pairs.row
        j = pairs.col
        d = pairs.data
        mask = (i != j) & (i < j)
        i = i[mask]
        j = j[mask]
        self._transfer_wetness(i, j, d[mask], dt)

    def advance_to(self, time: float) -> StateFrame:
        """Step the simulation to *time* and return the sampled StateFrame."""
        if time < self.current_time:
            raise ValueError("advance_to time must be monotonically non-decreasing")

        while self.current_time < time - 1e-12:
            dt = min(self.DT, time - self.current_time)
            self._substep(dt)
            self.current_time += dt

        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_particle_dem_wetting",
            "dt": self.DT,
            "n_particles": self.n,
            "cutoff": self.R_CUT,
        }
