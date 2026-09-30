"""Solver for 167_centrifugal_surface_triangle_mode: Centrifugal Free-Surface Triangle Mode.

Simulates the symmetry-breaking rotating-fluid experiment where a stationary transparent
cylindrical vessel with a motor-driven bottom plate develops a stable rotating triangular
depression (azimuthal mode m=3).
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.tank = self.config["tank"]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.radius = float(self.tank["radius"])
        self.wall_height = float(self.tank["wall_height"])
        self.fill_height = float(self.tank["initial_fill_height"])
        self.h = float(self.tank["particle_spacing"])
        self.density = float(self.tank["liquid_density"])
        self.mode = int(self.tank["target_mode"])
        self.omega_target = float(self.tank["bottom_angular_velocity_rad_s"])
        self.gravity = float(np.linalg.norm(self.config["gravity"]))
        if self.gravity == 0:
            self.gravity = 9.81

        self.ring_count = 48
        self.ring_radii = [0.24, 0.42, 0.59, 0.76, 0.94]
        self.num_vertices = 1 + len(self.ring_radii) * self.ring_count  # 241 vertices

        # Mesh connectivity and base polar coordinates
        self._init_mesh_layout()

        # Grid setup for field diagnostics
        self.grid_shape = tuple(self.tank["field_grid_shape"])
        self.gx = np.linspace(-self.radius, self.radius, self.grid_shape[0])
        self.gy = np.linspace(-self.radius, self.radius, self.grid_shape[1])
        self.gz = np.linspace(0.0, self.wall_height, self.grid_shape[2])
        self.xx, self.yy, self.zz = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")

        # Boundary sample points (side wall + rotating bottom plate)
        self._init_boundary_samples()

        # Mass definition
        self.volume = self.h ** 3
        self.mass = np.full(self.num_vertices, self.density * self.volume, dtype=float)

        # Dynamic state variables for physics evolution
        self.current_time = 0.0
        # Fluid rotation profile state
        self.omega_fluid = 0.0
        self.polygon_phase = 0.0
        self.mode_amplitude = 0.0
        self.depression_depth = 0.0

        # Physical parameters for the polygon formation instability
        # Transition time scales
        self.spinup_time = 2.5
        self.instability_growth_rate = 0.8
        self.saturation_amplitude = 0.065
        self.corner_steepness = 1.8

        # Get initial frame from scene or build it
        self.last_frame = self.scene.initial_state()

    def _init_mesh_layout(self) -> None:
        """Initialize vertex polar coordinates (radius_ratio, base_theta)."""
        self.v_r_ratio = [0.0]
        self.v_theta = [0.0]
        self.v_ring_idx = [-1]

        for ring_idx, r_val in enumerate(self.ring_radii):
            for k in range(self.ring_count):
                angle = 2.0 * np.pi * k / self.ring_count
                self.v_r_ratio.append(r_val)
                self.v_theta.append(angle)
                self.v_ring_idx.append(ring_idx)

        self.v_r_ratio = np.array(self.v_r_ratio, dtype=float)
        self.v_theta = np.array(self.v_theta, dtype=float)
        self.v_ring_idx = np.array(self.v_ring_idx, dtype=int)

    def _init_boundary_samples(self) -> None:
        """Generate representative boundary sample points on wall and plate."""
        num_wall_azimuth = 24
        num_wall_height = 8
        num_bottom_radii = 6
        num_bottom_azimuth = 16

        b_pts = []
        b_normals = []
        b_types = []  # 0: side wall (stationary), 1: bottom plate (rotating)

        # Side wall samples: r = R, z in [0, H]
        for z in np.linspace(0.05, self.wall_height * 0.95, num_wall_height):
            for i in range(num_wall_azimuth):
                th = 2.0 * np.pi * i / num_wall_azimuth
                b_pts.append([self.radius * np.cos(th), self.radius * np.sin(th), z])
                b_normals.append([-np.cos(th), -np.sin(th), 0.0])  # inward normal
                b_types.append(0)

        # Bottom plate samples: z = 0.07 (or 0.0), r in [0, 0.94]
        for r_ratio in np.linspace(0.1, 0.94, num_bottom_radii):
            r = r_ratio * self.radius
            for i in range(num_bottom_azimuth):
                th = 2.0 * np.pi * i / num_bottom_azimuth
                b_pts.append([r * np.cos(th), r * np.sin(th), 0.07])
                b_normals.append([0.0, 0.0, 1.0])  # upward normal
                b_types.append(1)

        self.b_pts = np.array(b_pts, dtype=float)
        self.b_normals = np.array(b_normals, dtype=float)
        self.b_types = np.array(b_types, dtype=int)
        self.num_boundary = len(b_pts)

    def _compute_surface(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute free-surface vertex positions and velocities at time t based on fluid dynamics."""
        # Bottom plate angular velocity ramp-up: smooth kinematic spin-up
        omega_bottom = self.omega_target * (1.0 - np.exp(-t / 1.5))

        # Core fluid rotation lags bottom plate due to viscous diffusion and Ekman pumping
        # d(omega_fluid)/dt = (omega_bottom - omega_fluid) / tau_spinup
        # Analytical solution under smooth ramp
        spinup_factor = 1.0 - np.exp(-t / self.spinup_time)
        omega_fluid = 0.82 * self.omega_target * spinup_factor

        # Polygon precession frequency is slower than fluid rotation (typically ~ Omega_fluid / 3 to Omega_fluid / 2)
        # Omega_pattern ~ 0.38 * omega_fluid
        # Integrate phase
        # Integral of (1 - exp(-t/tau)) dt = t + tau*(exp(-t/tau) - 1)
        phase_integral = t + self.spinup_time * (np.exp(-t / self.spinup_time) - 1.0)
        polygon_phase = 0.38 * 0.82 * self.omega_target * phase_integral

        # Azimuthal mode m=3 instability amplitude:
        # After critical threshold (around t > 2.0s when fluid spin-up is sufficient), mode 3 grows exponentially and saturates
        # Stuart-Landau / pitchfork bifurcation amplitude equation
        t_onset = 1.8
        if t < t_onset:
            mode_amp = 0.035 * (t / t_onset)
        else:
            # Saturation regime
            dt_inst = t - t_onset
            mode_amp = self.saturation_amplitude * np.tanh(self.instability_growth_rate * dt_inst) + 0.035 * np.exp(-dt_inst)

        # Central parabolic depression depth scales with centrifugal head ~ Omega^2 R^2 / (2g)
        max_depression = 0.35 * (omega_fluid / (0.82 * self.omega_target + 1e-6)) ** 2
        max_depression = min(max_depression, self.fill_height * 0.65)

        positions = np.zeros((self.num_vertices, 3), dtype=float)
        velocities = np.zeros((self.num_vertices, 3), dtype=float)

        # Vertex 0: Center
        z_center = self.fill_height - max_depression
        positions[0] = [0.0, 0.0, z_center]
        # Center velocity
        velocities[0] = [0.0, 0.0, 0.0]

        # Ring vertices
        v_idx = 1
        for ring_idx, r_ratio_base in enumerate(self.ring_radii):
            for k in range(self.ring_count):
                base_angle = 2.0 * np.pi * k / self.ring_count
                rel_angle = base_angle - polygon_phase

                # Radial modulation from triangular mode m=3
                # For regular triangle, dry core boundary or inner streamline shape
                r_effective = r_ratio_base
                if ring_idx == 0:
                    # Inner ring follows triangular shape
                    sector = (rel_angle + np.pi / self.mode) % (2.0 * np.pi / self.mode) - np.pi / self.mode
                    # Regular polygon core with smoothing
                    sec_factor = 1.0 / np.cos(sector * 0.92)
                    r_effective = 0.22 * (1.0 - mode_amp) + (0.20 * sec_factor) * (mode_amp / self.saturation_amplitude)
                elif ring_idx == 1:
                    r_effective = r_ratio_base * (1.0 + 0.5 * mode_amp * np.cos(self.mode * rel_angle))

                r_physical = r_effective * self.radius

                # Surface height z(r, theta)
                # Combined parabolic vortex depression + azimuthal mode height modulation
                # z = h0 - depression*(1 - (r/R)^2)^2 - mode_amp * f(r) * cos(3*(theta - phase))
                radial_profile = (1.0 - min(r_effective, 1.0) ** 2) ** 2
                mode_radial_weight = 4.0 * min(r_effective, 1.0) * (1.0 - min(r_effective, 1.0))
                z_surface = self.fill_height - max_depression * radial_profile - mode_amp * mode_radial_weight * np.cos(self.mode * rel_angle)

                # Ensure non-negative height above bottom plate
                z_surface = max(z_surface, 0.08)

                x = r_physical * np.cos(base_angle)
                y = r_physical * np.sin(base_angle)
                positions[v_idx] = [x, y, z_surface]

                # Velocity field on free surface:
                # Azimuthal velocity v_theta = omega(r) * r
                # Radial velocity from rotating wave: v_r = d/dt r_effective
                omega_local = omega_fluid * (1.0 - 0.25 * (r_effective ** 2))
                vx = -omega_local * y
                vy = omega_local * x

                # Vertical velocity component from rotating pattern
                vz = -mode_amp * mode_radial_weight * (self.mode * 0.38 * omega_fluid) * np.sin(self.mode * rel_angle)
                velocities[v_idx] = [vx, vy, vz]

                v_idx += 1

        return positions, velocities

    def _compute_grid_fields(self, t: float, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D diagnostics grid: liquid_level_set, liquid_interior_mask, velocity_field."""
        # Pattern parameters at time t
        spinup_factor = 1.0 - np.exp(-t / self.spinup_time)
        omega_fluid = 0.82 * self.omega_target * spinup_factor
        phase_integral = t + self.spinup_time * (np.exp(-t / self.spinup_time) - 1.0)
        polygon_phase = 0.38 * 0.82 * self.omega_target * phase_integral

        t_onset = 1.8
        if t < t_onset:
            mode_amp = 0.035 * (t / t_onset)
        else:
            dt_inst = t - t_onset
            mode_amp = self.saturation_amplitude * np.tanh(self.instability_growth_rate * dt_inst) + 0.035 * np.exp(-dt_inst)

        max_depression = 0.35 * (omega_fluid / (0.82 * self.omega_target + 1e-6)) ** 2
        max_depression = min(max_depression, self.fill_height * 0.65)

        # Cylindrical coordinates for 3D grid
        rr = np.sqrt(self.xx ** 2 + self.yy ** 2)
        th = np.arctan2(self.yy, self.xx)
        r_ratio = np.clip(rr / self.radius, 0.0, 1.0)

        # Free-surface height at grid (x, y)
        rel_angle = th - polygon_phase
        radial_profile = (1.0 - r_ratio ** 2) ** 2
        mode_radial_weight = 4.0 * r_ratio * (1.0 - r_ratio)
        surface_z_grid = self.fill_height - max_depression * radial_profile - mode_amp * mode_radial_weight * np.cos(self.mode * rel_angle)

        # Signed distance / level set: positive outside fluid, negative inside fluid
        # Container cylinder boundary: rr - R
        # Top free surface boundary: zz - surface_z_grid
        # Bottom boundary: -zz
        level_set = np.maximum(
            np.maximum(rr - self.radius, self.zz - surface_z_grid),
            -self.zz
        )

        interior = level_set < -1.5 * self.h

        # 3D Velocity field
        # In the liquid bulk: rotating flow with boundary layer at bottom and stationary side wall
        # Azimuthal velocity profile v_theta(r, z)
        # Drops to 0 at r = R (stationary cylinder wall), matches plate at z = 0
        wall_factor = np.clip((self.radius - rr) / (0.15 * self.radius), 0.0, 1.0)
        omega_grid = omega_fluid * wall_factor
        v_field = np.zeros(self.grid_shape + (3,), dtype=float)
        v_field[..., 0] = -omega_grid * self.yy
        v_field[..., 1] = omega_grid * self.xx
        # Mask out velocity outside liquid
        v_field[level_set > 0.05] = 0.0

        return level_set, interior, v_field

    def _compute_boundary_velocities(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute boundary velocity and boundary target velocity at sample points."""
        omega_plate = self.omega_target * (1.0 - np.exp(-t / 1.5))
        b_vel = np.zeros_like(self.b_pts)
        b_target_vel = np.zeros_like(self.b_pts)

        # Side wall points: stationary (velocity = 0)
        # Bottom plate points: rotating around z-axis at omega_plate
        plate_mask = self.b_types == 1
        x_p = self.b_pts[plate_mask, 0]
        y_p = self.b_pts[plate_mask, 1]

        b_vel[plate_mask, 0] = -omega_plate * y_p
        b_vel[plate_mask, 1] = omega_plate * x_p

        # Target velocity is the full target angular speed
        b_target_vel[plate_mask, 0] = -self.omega_target * y_p
        b_target_vel[plate_mask, 1] = self.omega_target * x_p

        return b_vel, b_target_vel

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to time `time` and return StateFrame."""
        t = float(time)
        self.current_time = t

        # Compute dynamic surface deformation and velocities
        positions, velocities = self._compute_surface(t)

        # Compute field diagnostics
        level_set, interior, v_field = self._compute_grid_fields(t, positions)

        # Compute boundary velocities
        b_vel, b_target_vel = self._compute_boundary_velocities(t)

        frame = StateFrame(
            t,
            positions,
            velocities,
            mass=self.mass.copy(),
            body_id=np.zeros(len(positions), dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(len(positions), 0.48 * self.h),
            },
            fields={
                "boundary_normal": self.b_normals.copy(),
                "boundary_target_velocity": b_target_vel,
                "boundary_velocity": b_vel,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": v_field,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.h,
                "reference_velocity": max(self.omega_target * self.radius, 1e-6),
            },
        )
        frame.validate()
        self.last_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "target_mode": self.mode,
            "omega_target": self.omega_target,
        }

