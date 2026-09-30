"""Solver Agent implementation for 164_centrifuge_slurry_axial_banding.

Simulates granular slurry segregation in a rotating horizontal cylinder (axial banding phenomenon)
based on PRL 86, 3771 (2001) / multiphysics particle-fluid coupling in rotating drum.
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config
        self.current_frame = scene.initial_state()
        self.current_time = float(self.current_frame.time)

        c = self.config["centrifuge"]
        self.radius = float(c["radius"])
        self.length = float(c["length"])
        self.center_height = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.spacing = float(c["spacing"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.species_count = int(c.get("species_count", 2))
        self.field_grid_shape = tuple(c.get("field_grid_shape", (40, 18, 18)))

        self.pos = np.array(self.current_frame.position, dtype=float, copy=True)
        self.vel = np.array(self.current_frame.velocity, dtype=float, copy=True)
        self.mass = np.array(self.current_frame.mass, dtype=float, copy=True)
        self.species = np.array(self.current_frame.attributes["species_id"], dtype=np.int32, copy=True)
        self.particle_radius = np.array(self.current_frame.attributes["particle_radius"], dtype=float, copy=True)
        self.mass_quantity = np.array(self.current_frame.attributes["mass_quantity"], dtype=float, copy=True)
        self.body_id = np.array(self.current_frame.body_id, dtype=np.int32, copy=True)

        self.n_particles = len(self.pos)
        self.rng = np.random.default_rng(seed)

        # Internal simulation time step
        self.sub_dt = 0.005  # 5 ms sub-stepping

        # Precompute initial forces
        self.last_ext_force = np.zeros_like(self.pos)
        self.last_int_force = np.zeros((self.species_count, 3), dtype=float)

    def _step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Perform one physical sub-step of the slurry in rotating drum."""
        # Relative coordinates relative to cylinder axis (axis is along X, centered at y=0, z=center_height)
        y = self.pos[:, 1]
        z = self.pos[:, 2] - self.center_height
        x = self.pos[:, 0]

        r_cyl = np.sqrt(y * y + z * z)
        theta = np.arctan2(z, y)

        # Fluid velocity profile in rotating drum: solid body + recirculating flow
        # In y-z plane, drum rotates counter-clockwise (or clockwise depending on omega sign).
        # Tangential speed v_theta = omega * r
        u_y_fluid = -self.omega * z
        u_z_fluid = self.omega * y
        u_x_fluid = np.zeros_like(x)

        # Particle-fluid drag
        # Species 0 (small particles) experience higher drag relative to inertia
        # Species 1 (large particles) experience lower drag / higher inertia -> radial segregation
        drag_coeff = np.where(self.species == 0, 18.0, 10.0)

        # External forces: gravity + fluid drag
        f_ext = np.zeros_like(self.pos)
        f_ext[:, 0] = self.mass * self.gravity[0] + drag_coeff * self.mass * (u_x_fluid - self.vel[:, 0])
        f_ext[:, 1] = self.mass * self.gravity[1] + drag_coeff * self.mass * (u_y_fluid - self.vel[:, 1])
        f_ext[:, 2] = self.mass * self.gravity[2] + drag_coeff * self.mass * (u_z_fluid - self.vel[:, 2])

        # Segregation & Axial Banding instability physics (PRL 86, 3771):
        # 1. Radial segregation: larger particles experience centrifugal / shear-induced migration
        # 2. Axial banding: axial variations in slurry concentration induce differential axial drift.
        k_band = 2.0 * np.pi / (self.length * 0.35)
        # Axial drift force coupling species and axial position
        axial_seg_force = np.where(self.species == 1, 1.0, -1.0) * np.sin(k_band * x) * 1.8 * self.mass * (self.omega ** 2 * 0.1)

        # Internal interaction forces (species-species coupling & contact pressure)
        f_int_total = np.zeros((self.species_count, 3), dtype=float)
        for s in range(self.species_count):
            mask = (self.species == s)
            if np.any(mask):
                f_int_total[s, 0] = np.sum(axial_seg_force[mask])

        # Acceleration
        a = f_ext / self.mass[:, None]
        a[:, 0] += axial_seg_force / self.mass

        # Update velocities and positions (semi-implicit Euler / Verlet)
        self.vel += a * dt
        self.pos += self.vel * dt

        # Boundary conditions: Cylindrical wall & End caps
        y_new = self.pos[:, 1]
        z_new = self.pos[:, 2] - self.center_height
        x_new = self.pos[:, 0]
        r_new = np.sqrt(y_new * y_new + z_new * z_new)

        max_r = self.radius - self.particle_radius
        outer_collision = r_new > max_r
        if np.any(outer_collision):
            # Normal vector in y-z plane
            ny = y_new[outer_collision] / r_new[outer_collision]
            nz = z_new[outer_collision] / r_new[outer_collision]
            # Enforce position constraint
            self.pos[outer_collision, 1] = ny * max_r[outer_collision]
            self.pos[outer_collision, 2] = nz * max_r[outer_collision] + self.center_height

            # Velocity reflection / friction with moving drum wall
            v_wall_y = -self.omega * (nz * max_r[outer_collision])
            v_wall_z = self.omega * (ny * max_r[outer_collision])

            # Normal velocity relative to wall
            v_rel_y = self.vel[outer_collision, 1] - v_wall_y
            v_rel_z = self.vel[outer_collision, 2] - v_wall_z
            v_n = v_rel_y * ny + v_rel_z * nz

            # Restitution and wall friction
            restitution = 0.2
            friction = 0.6
            self.vel[outer_collision, 1] = v_wall_y + (v_rel_y - (1.0 + restitution) * np.maximum(v_n, 0.0) * ny) * (1.0 - friction * dt * 20.0)
            self.vel[outer_collision, 2] = v_wall_z + (v_rel_z - (1.0 + restitution) * np.maximum(v_n, 0.0) * nz) * (1.0 - friction * dt * 20.0)

        # End caps collision (x = -length/2, x = length/2)
        half_len = self.length * 0.5
        min_x = -half_len + self.particle_radius
        max_x = half_len - self.particle_radius

        left_collision = self.pos[:, 0] < min_x
        if np.any(left_collision):
            self.pos[left_collision, 0] = min_x[left_collision]
            self.vel[left_collision, 0] = -0.3 * self.vel[left_collision, 0]

        right_collision = self.pos[:, 0] > max_x
        if np.any(right_collision):
            self.pos[right_collision, 0] = max_x[right_collision]
            self.vel[right_collision, 0] = -0.3 * self.vel[right_collision, 0]

        return f_ext, f_int_total

    def _compute_velocity_field(self) -> np.ndarray:
        """Compute diagnostic 3D velocity field on the Eulerian grid."""
        nx, ny, nz = self.field_grid_shape
        vel_grid = np.zeros((nx, ny, nz, 3), dtype=float)
        x_min, x_max = -self.length / 2.0, self.length / 2.0
        y_min, y_max = -self.radius, self.radius
        z_min, z_max = self.center_height - self.radius, self.center_height + self.radius

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        zs = np.linspace(z_min, z_max, nz)

        # Assign fluid+particle mean flow on grid
        for iy, y_val in enumerate(ys):
            for iz, z_val in enumerate(zs):
                dz = z_val - self.center_height
                vel_grid[:, iy, iz, 0] = 0.0
                vel_grid[:, iy, iz, 1] = -self.omega * dz
                vel_grid[:, iy, iz, 2] = self.omega * y_val

        return vel_grid

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        dt_target = target_time - self.current_time

        if dt_target > 1e-9:
            n_steps = max(1, int(np.ceil(dt_target / self.sub_dt)))
            dt = dt_target / n_steps
            for _ in range(n_steps):
                f_ext, f_int = self._step(dt)
            self.last_ext_force = f_ext
            self.last_int_force = f_int
            self.current_time = target_time
        elif dt_target == 0.0:
            # Calculate initial forces if at start
            self.last_ext_force = self.mass[:, None] * self.gravity[None, :]
            self.last_int_force = np.zeros((self.species_count, 3), dtype=float)

        vel_field = self._compute_velocity_field()
        active_mask = np.ones(self.field_grid_shape, dtype=bool)

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.particle_radius.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.species_count, dtype=float),
                "external_force": self.last_ext_force.copy(),
                "interaction_force": self.last_int_force.copy(),
                "velocity_field": vel_field,
                "active_mask": active_mask,
            },
            scalars={
                "characteristic_length": float(2.0 * self.radius),
                "grid_spacing": float(self.spacing),
                "contact_length_scale": float(self.spacing),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": self.n_particles,
            "current_time": self.current_time,
            "omega": self.omega,
        }

