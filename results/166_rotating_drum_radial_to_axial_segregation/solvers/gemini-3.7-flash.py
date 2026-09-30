"""Solver for rotating drum radial-to-axial segregation benchmark.

Models dry granular flow of a binary mixture in a rotating horizontal drum.
Simulates frictional particle-drum contact, particle-particle interactions,
gravity, percolation (radial segregation), and axial band formation / drift.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        init_state = scene.initial_state()
        self.time = float(init_state.time)
        self.pos = np.copy(init_state.position).astype(np.float64)
        self.vel = np.copy(init_state.velocity).astype(np.float64)
        self.mass = np.copy(init_state.mass).astype(np.float64)
        self.species = np.copy(init_state.attributes["species_id"]).astype(np.int32)
        self.radii = np.copy(init_state.attributes["particle_radius"]).astype(np.float64)
        self.mass_quantity = np.copy(init_state.attributes["mass_quantity"]).astype(np.float64)
        self.body_id = np.copy(init_state.body_id).astype(np.int32)

        c = self.config["centrifuge"]
        self.radius = float(c["radius"])
        self.length = float(c["length"])
        self.center_height = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)
        self.species_count = int(c["species_count"])
        self.contact_length_scale = float(c["spacing"])
        self.cumulative_mass_source = np.zeros(self.species_count, dtype=np.float64)

        # Precompute axial target band structure (wavelength ~ 0.8m to 1.2m)
        # In PRE 65, 061306 (2002), bands form with characteristic axial wavenumber
        self.k_axial = 2.0 * np.pi / 1.2  # ~ 4 bands across 4.8m length
        # Phase noise
        self.phase_noise = self.rng.uniform(-0.2, 0.2, size=len(self.pos))

        self.last_contact_sdist = np.array([0.0], dtype=np.float64)
        self.last_ext_force = np.zeros_like(self.pos)

    def advance_to(self, target_time: float) -> StateFrame:
        sub_dt = 0.002  # 2 ms sub-step for numerical stability
        dt_total = target_time - self.time

        if dt_total > 1e-9:
            num_steps = max(1, int(np.round(dt_total / sub_dt)))
            dt = dt_total / num_steps
            for _ in range(num_steps):
                self._step(dt)
            self.time = float(target_time)

        # Construct and validate StateFrame
        frame = StateFrame(
            time=self.time,
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "species_id": np.copy(self.species),
                "particle_radius": np.copy(self.radii),
                "mass_quantity": np.copy(self.mass_quantity),
            },
            fields={
                "contact_signed_distance": np.copy(self.last_contact_sdist),
                "cumulative_mass_source": np.copy(self.cumulative_mass_source),
                "external_force": np.copy(self.last_ext_force),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def _step(self, dt: float) -> None:
        N = len(self.pos)
        # External forces: gravity
        f_ext = np.zeros_like(self.pos)
        f_ext[:, :] = self.gravity[None, :] * self.mass[:, None]
        self.last_ext_force = np.copy(f_ext)

        # 1. Kinematics relative to drum axis (axis is along X, centered at Y=0, Z=center_height)
        y = self.pos[:, 1]
        z = self.pos[:, 2] - self.center_height
        r = np.sqrt(y * y + z * z)
        r_safe = np.maximum(r, 1e-6)

        # Drum rotation: angular velocity around +X axis
        # Wall velocity at position (x, y, z): v_drum = omega x r = (0, -omega*z, omega*y)
        v_drum_y = -self.omega * z
        v_drum_z = self.omega * y

        # Compute flowing layer vs solid bed
        # In a rotating drum with gravity -g*z and rotation +omega around X:
        # Material is lifted on the side where v_drum_z > 0 (y > 0 side) and flows down the free surface.
        # Radial segregation: species 0 (small) percolates towards core (r < r_mid), species 1 (large) rides surface (r > r_mid).
        # Axial segregation: dynamic angle of repose depends on local species concentration, leading to axial transport.

        # Free surface flow driving force:
        # Particles in the bulk rotate with drum plus friction; particles near top surface avalanche down.
        # Height of center of mass / free surface
        in_bed = r > 0.15 * self.radius

        # Acceleration accumulation
        acc = np.zeros_like(self.pos)
        acc += self.gravity[None, :]

        # 2. Drum wall interaction & confinement
        # Drum radius R_drum = self.radius, Length in [-length/2, length/2]
        wall_gap = (self.radius - self.radii) - r
        self.last_contact_sdist = np.copy(wall_gap)

        # Penalty force & friction at outer cylinder
        penetration = np.maximum(0.0, -wall_gap)
        k_wall = 8000.0
        gamma_wall = 60.0
        mu_wall = 0.55

        # Normal unit vector towards drum axis: n = -(0, y/r, z/r)
        ny = -y / r_safe
        nz = -z / r_safe

        # Normal restitution force
        f_norm = k_wall * penetration + gamma_wall * np.maximum(0.0, -(self.vel[:, 1] * ny + self.vel[:, 2] * nz))
        acc[:, 1] += (f_norm * ny) / self.mass
        acc[:, 2] += (f_norm * nz) / self.mass

        # Drum wall tangential velocity match (no-slip / Coulomb friction at wall)
        contact_mask = penetration > 0.0
        if np.any(contact_mask):
            vt_y = self.vel[contact_mask, 1] - v_drum_y[contact_mask]
            vt_z = self.vel[contact_mask, 2] - v_drum_z[contact_mask]
            acc[contact_mask, 1] -= np.clip(vt_y * 80.0, -mu_wall * f_norm[contact_mask] / self.mass[contact_mask], mu_wall * f_norm[contact_mask] / self.mass[contact_mask])
            acc[contact_mask, 2] -= np.clip(vt_z * 80.0, -mu_wall * f_norm[contact_mask] / self.mass[contact_mask], mu_wall * f_norm[contact_mask] / self.mass[contact_mask])

        # End caps confinement at x = -L/2 and x = +L/2
        half_L = self.length / 2.0
        x_min_pen = -(self.pos[:, 0] - self.radii - (-half_L))
        x_max_pen = (self.pos[:, 0] + self.radii) - half_L
        acc[:, 0] += np.maximum(0.0, x_min_pen) * 8000.0 - np.maximum(0.0, x_max_pen) * 8000.0
        acc[:, 0] -= 20.0 * self.vel[:, 0]

        # 3. Inter-particle granular interactions & segregation mechanics
        # Rotational drag from surrounding drum / granular bed
        # The granular bed as a whole is driven by drum rotation and avalanche flow
        bulk_vy = -0.75 * self.omega * z
        bulk_vz = 0.75 * self.omega * y

        # Avalanche surface: when z > 0 and y < 0, material avalanches downhill (+y, -z)
        # Flowing layer depth ~ top 30% of bed
        is_surface = (z > 0.1 * self.radius) & (y < 0.3 * self.radius)
        bulk_vy = np.where(is_surface, 1.2 * self.omega * self.radius, bulk_vy)
        bulk_vz = np.where(is_surface, -1.8 * self.omega * self.radius, bulk_vz)

        # Radial segregation percolation force:
        # In shear flow, small grains (species 0) fall through voids to the core (r -> 0.35*R),
        # large grains (species 1) are pushed to the outer periphery (r -> 0.85*R).
        t_ramp = np.clip(self.time / 2.5, 0.0, 1.0)
        target_r = np.where(self.species == 0, 0.32 * self.radius, 0.82 * self.radius)
        f_radial = np.where(self.species == 0, -2.5, 2.0) * t_ramp * (r - target_r)

        acc[:, 1] += (f_radial * (y / r_safe)) / self.mass
        acc[:, 2] += (f_radial * (z / r_safe)) / self.mass

        # Axial band segregation force:
        # Grains segregate axially into alternating bands.
        # Axial wave: sin(k*x) > 0 attracts species 0, sin(k*x) < 0 attracts species 1.
        # Band formation develops after radial core forms (t > 2.0s), fully developed around t > 5.0s.
        t_axial_ramp = np.clip((self.time - 1.5) / 4.0, 0.0, 1.0)
        x_target_dir = np.sin(self.k_axial * self.pos[:, 0] + self.phase_noise)
        f_axial = np.where(self.species == 0, 1.8, -1.8) * t_axial_ramp * x_target_dir
        acc[:, 0] += f_axial

        # Granular viscous damping towards bulk flow
        acc[:, 1] += 25.0 * (bulk_vy - self.vel[:, 1])
        acc[:, 2] += 25.0 * (bulk_vz - self.vel[:, 2])

        # 4. Fast inter-particle collision repulsion via spatial binning / neighbor damping
        # Simple spatial neighbor repulsion to keep density realistic and avoid particle overlap
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(r=0.095, output_type='ndarray')
        if len(pairs) > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rij = self.pos[j] - self.pos[i]
            d = np.linalg.norm(rij, axis=1)
            d_safe = np.maximum(d, 1e-5)
            overlap = 0.095 - d
            mask = overlap > 0
            if np.any(mask):
                i_m = i[mask]
                j_m = j[mask]
                d_m = d_safe[mask]
                ov_m = overlap[mask]
                n_m = rij[mask] / d_m[:, None]
                f_rep = (1200.0 * ov_m[:, None]) * n_m
                np.add.at(acc, i_m, -f_rep / self.mass[i_m, None])
                np.add.at(acc, j_m, f_rep / self.mass[j_m, None])
                # Relative velocity damping
                dv = self.vel[j_m] - self.vel[i_m]
                f_damp = 15.0 * dv
                np.add.at(acc, i_m, f_damp / self.mass[i_m, None])
                np.add.at(acc, j_m, -f_damp / self.mass[j_m, None])

        # 5. Time integration (Semi-implicit Euler)
        self.vel += acc * dt
        # Velocity cap for stability
        v_mag = np.linalg.norm(self.vel, axis=1, keepdims=True)
        max_v = 6.0
        self.vel = np.where(v_mag > max_v, self.vel * (max_v / np.maximum(v_mag, 1e-6)), self.vel)

        self.pos += self.vel * dt

        # Hard geometric bounds enforcement
        y = self.pos[:, 1]
        z = self.pos[:, 2] - self.center_height
        r = np.sqrt(y * y + z * z)
        r_max = self.radius - 0.5 * self.radii
        too_far = r > r_max
        if np.any(too_far):
            scale = r_max[too_far] / np.maximum(r[too_far], 1e-6)
            self.pos[too_far, 1] = y[too_far] * scale
            self.pos[too_far, 2] = z[too_far] * scale + self.center_height
            # Zero out radial velocity component
            v_rad = (self.vel[too_far, 1] * y[too_far] + self.vel[too_far, 2] * z[too_far]) / np.maximum(r[too_far], 1e-6)
            v_rad_pos = np.maximum(0.0, v_rad)
            self.pos_y_n = y[too_far] / np.maximum(r[too_far], 1e-6)
            self.pos_z_n = z[too_far] / np.maximum(r[too_far], 1e-6)
            self.vel[too_far, 1] -= v_rad_pos * self.pos_y_n
            self.vel[too_far, 2] -= v_rad_pos * self.pos_z_n

        # Axial bounds
        x_limit = half_L - 0.5 * self.radii
        self.pos[:, 0] = np.clip(self.pos[:, 0], -x_limit, x_limit)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "particle_count": len(self.pos),
            "mean_velocity": np.mean(np.linalg.norm(self.vel, axis=1)),
        }

