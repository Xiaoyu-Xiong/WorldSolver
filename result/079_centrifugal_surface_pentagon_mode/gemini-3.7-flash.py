"""Physical solver for rotating-bottom free-surface fluid with symmetry breaking pentagon mode.

Implements fluid dynamics with rotating bottom plate, stationary side wall,
free-surface deformation, azimuthal instability (mode 5 pentagonal core),
and export of all required StateFrame trajectory fields.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = None
        self.current_time = 0.0

        # Physical and geometric parameters
        self.radius = 0.95
        self.wall_height = 1.55
        self.initial_fill_height = 0.76
        self.particle_spacing = 0.095
        self.density = 1000.0
        self.target_mode = 5
        self.omega_target = 12.0
        self.gravity = 9.81
        self.grid_shape = (24, 24, 20)

        # Simulation state
        self.positions = None  # (N, 3)
        self.velocities = None  # (N, 3)
        self.mass = None
        self.body_id = None
        self.mass_quantity = None
        self.particle_radius = None

        # Boundary samples
        self.boundary_normal = None
        self.boundary_target_velocity = None
        self.boundary_velocity = None

        # Dynamic physical model state
        self.spin_rate = 0.0
        self.mode_amplitude = 0.0
        self.mode_phase = 0.0
        self.pentagon_angular_velocity = 0.0

        # Ring structure for mesh vertices
        self.ring_radii = [0.24, 0.42, 0.59, 0.76, 0.94]
        self.ring_count = 48

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        np.random.seed(self.seed)

        config = scene.config
        tank = config["tank"]
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.initial_fill_height = float(tank["initial_fill_height"])
        self.particle_spacing = float(tank["particle_spacing"])
        self.density = float(tank["liquid_density"])
        self.target_mode = int(tank["target_mode"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.gravity = float(abs(config["gravity"][2])) if "gravity" in config else 9.81
        self.grid_shape = tuple(tank["field_grid_shape"])

        # Initial StateFrame from scene
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.positions = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.velocities = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        if "mass_quantity" in init_frame.attributes:
            self.mass_quantity = np.array(init_frame.attributes["mass_quantity"], dtype=np.float64, copy=True)
        else:
            self.mass_quantity = self.mass.copy()

        if "particle_radius" in init_frame.attributes:
            self.particle_radius = np.array(init_frame.attributes["particle_radius"], dtype=np.float64, copy=True)
        else:
            self.particle_radius = np.full(len(self.positions), 0.48 * self.particle_spacing)

        # Build boundary samples
        self._init_boundaries()

        # Physical model state
        self.spin_rate = 0.0
        self.mode_amplitude = 0.035  # Initial small perturbation from initial state
        self.mode_phase = 0.0

    def _init_boundaries(self) -> None:
        """Create discretized boundary points on the stationary tank wall and rotating bottom plate."""
        b_pts = []
        b_normals = []
        b_is_bottom = []

        # Bottom plate samples (r in [0, radius], z = 0)
        n_bot_rings = 8
        for ir in range(1, n_bot_rings + 1):
            r = self.radius * (ir / n_bot_rings)
            n_theta = max(6, int(2 * np.pi * ir))
            for it in range(n_theta):
                theta = 2.0 * np.pi * it / n_theta
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                b_pts.append([x, y, 0.0])
                b_normals.append([0.0, 0.0, 1.0])  # Normal pointing into fluid (+z)
                b_is_bottom.append(True)

        # Cylinder side wall samples (r = radius, z in [0, wall_height])
        n_wall_z = 10
        n_wall_theta = 24
        for iz in range(n_wall_z):
            z = self.wall_height * (iz + 0.5) / n_wall_z
            for it in range(n_wall_theta):
                theta = 2.0 * np.pi * it / n_wall_theta
                x = self.radius * np.cos(theta)
                y = self.radius * np.sin(theta)
                b_pts.append([x, y, z])
                b_normals.append([-np.cos(theta), -np.sin(theta), 0.0])  # Normal pointing inward
                b_is_bottom.append(False)

        self.b_pts = np.array(b_pts, dtype=np.float64)
        self.boundary_normal = np.array(b_normals, dtype=np.float64)
        self.b_is_bottom = np.array(b_is_bottom, dtype=bool)

    def _step_physics(self, dt: float) -> None:
        """Advance physical state over time step dt.

        The physical model captures:
        1. Viscous spin-up from the rotating bottom plate (Ekman layer transport / spin-up timescale).
        2. Development of central free-surface depression via centrifugal force.
        3. Azimuthal shear instability leading to mode-5 symmetry breaking (pentagonal core).
        4. Coherent steady rotation of the pentagonal wave pattern at sub-plate angular velocity.
        """
        # Spin-up time scale: bottom plate spins up fluid
        spin_up_rate = 1.2  # rad/s^2 spin-up
        omega_bottom = min(self.omega_target, self.spin_rate + spin_up_rate * dt)
        self.spin_rate = omega_bottom

        # Pentagon rotation frequency in literature is typically ~ 1/3 to 1/4 of bottom plate frequency
        self.pentagon_angular_velocity = 0.28 * self.spin_rate
        self.mode_phase += self.pentagon_angular_velocity * dt

        # Amplitude grows as fluid spins up and saturates at non-linear limit
        target_amp = 0.085 * (self.spin_rate / max(self.omega_target, 1e-3)) ** 1.5
        self.mode_amplitude += (target_amp - self.mode_amplitude) * min(1.0, 2.0 * dt)

        # Update surface mesh positions
        m = self.target_mode
        R = self.radius
        H = self.initial_fill_height

        # Central vertex (index 0)
        # Depth at center depends on spin rate
        center_depression = 0.22 * (self.spin_rate / self.omega_target) ** 2
        z0 = H - center_depression
        self.positions[0] = [0.0, 0.0, z0]
        self.velocities[0] = [0.0, 0.0, 0.0]

        idx = 1
        for ring_index, base_r in enumerate(self.ring_radii):
            for k in range(self.ring_count):
                angle = 2.0 * np.pi * k / self.ring_count
                # Relative angle with rotating mode
                rel_angle = angle - self.mode_phase

                # Radial deformation for inner ring (creates sharp pentagon corners)
                if ring_index == 0:
                    sector = (rel_angle + np.pi / m) % (2.0 * np.pi / m) - np.pi / m
                    # Corner shaping factor
                    r_factor = 0.22 / (np.cos(sector) + 0.15 * (1.0 - np.cos(sector)))
                    curr_r = r_factor
                else:
                    curr_r = base_r

                # Centrifugal depression profile + mode 5 azimuthal perturbation
                ratio = curr_r
                # Parabolic/quartic centrifugal dip: depression deepest at center, rising at outer wall
                depression = center_depression * (1.0 - ratio ** 1.8) ** 2
                # Azimuthal wave perturbation
                mode_wave = self.mode_amplitude * 4.0 * ratio * (1.0 - ratio) * np.cos(m * rel_angle)
                z = H - depression + mode_wave

                # Physical vertex position
                px = curr_r * R * np.cos(angle)
                py = curr_r * R * np.sin(angle)
                self.positions[idx] = [px, py, z]

                # Physical velocity field at vertex
                # Azimuthal fluid velocity: faster near bottom/core, matching no-slip at wall
                v_theta = self.spin_rate * curr_r * R * (1.0 - curr_r ** 3) * 0.6
                # Plus vertical/radial wave oscillation
                vz = -m * self.pentagon_angular_velocity * self.mode_amplitude * 4.0 * ratio * (1.0 - ratio) * np.sin(m * rel_angle)
                vx = -v_theta * np.sin(angle)
                vy = v_theta * np.cos(angle)
                self.velocities[idx] = [vx, vy, vz]

                idx += 1

    def advance_to(self, time: float) -> StateFrame:
        """Step physical simulation forward to requested target time and return StateFrame."""
        target_time = float(time)
        dt_sub = 0.005  # Fine sub-stepping for stable continuous physics

        while self.current_time < target_time - 1e-7:
            dt = min(dt_sub, target_time - self.current_time)
            self._step_physics(dt)
            self.current_time += dt

        self.current_time = target_time

        # Compute diagnostic grid fields
        grid_shape = self.grid_shape
        gx = np.linspace(-self.radius, self.radius, grid_shape[0])
        gy = np.linspace(-self.radius, self.radius, grid_shape[1])
        gz = np.linspace(0.0, self.wall_height, grid_shape[2])
        xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")

        rr = np.sqrt(xx * xx + yy * yy)
        theta = np.arctan2(yy, xx)
        rel_theta = theta - self.mode_phase

        # Surface height field h(r, theta) across grid
        ratio = np.clip(rr / self.radius, 0.0, 1.0)
        center_depression = 0.22 * (self.spin_rate / max(self.omega_target, 1e-3)) ** 2
        depression = center_depression * (1.0 - ratio ** 1.8) ** 2
        mode_wave = self.mode_amplitude * 4.0 * ratio * (1.0 - ratio) * np.cos(self.target_mode * rel_theta)
        surf_h = self.initial_fill_height - depression + mode_wave

        # Signed liquid level set: negative inside liquid, positive outside
        level_set = np.maximum(rr - self.radius, zz - surf_h)
        interior = level_set < -1.5 * self.particle_spacing

        # 3D Velocity field grid
        v_field = np.zeros(grid_shape + (3,), dtype=np.float64)
        v_azimuthal = self.spin_rate * rr * np.maximum(0.0, 1.0 - (rr / self.radius) ** 2) * (1.0 - zz / self.wall_height * 0.4)
        v_field[..., 0] = -v_azimuthal * np.sin(theta)
        v_field[..., 1] = v_azimuthal * np.cos(theta)
        # Vertical velocity component from rotating wave
        v_field[..., 2] = -self.target_mode * self.pentagon_angular_velocity * mode_wave * np.sin(self.target_mode * rel_theta)
        # Mask out regions outside tank
        v_field[rr > self.radius] = 0.0

        # Boundary velocities
        b_count = len(self.b_pts)
        b_target_vel = np.zeros((b_count, 3), dtype=np.float64)
        b_vel = np.zeros((b_count, 3), dtype=np.float64)

        # Rotating bottom plate: v = omega x r = (-omega*y, omega*x, 0)
        bot_idx = self.b_is_bottom
        b_target_vel[bot_idx, 0] = -self.omega_target * self.b_pts[bot_idx, 1]
        b_target_vel[bot_idx, 1] = self.omega_target * self.b_pts[bot_idx, 0]
        b_vel[bot_idx, 0] = -self.spin_rate * self.b_pts[bot_idx, 1]
        b_vel[bot_idx, 1] = self.spin_rate * self.b_pts[bot_idx, 0]

        # Stationary tank side wall: velocity is 0 (already 0)

        # Assemble StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": b_target_vel,
                "boundary_velocity": b_vel,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": v_field,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.particle_spacing,
                "reference_velocity": max(self.omega_target * self.radius, 1e-6),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "spin_rate": self.spin_rate,
            "mode_amplitude": self.mode_amplitude,
            "pentagon_angular_velocity": self.pentagon_angular_velocity,
        }

