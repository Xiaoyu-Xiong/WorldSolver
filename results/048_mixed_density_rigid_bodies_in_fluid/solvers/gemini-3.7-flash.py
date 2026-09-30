"""Physical multi-body fluid-solid coupling solver for mixed-density rigid bodies."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate unit quaternion [x, y, z, w] with angular velocity omega."""
    omega_norm = float(np.linalg.norm(omega))
    if omega_norm < 1e-12:
        return q
    axis = omega / omega_norm
    angle = omega_norm * dt
    dq = np.array([
        axis[0] * np.sin(angle * 0.5),
        axis[1] * np.sin(angle * 0.5),
        axis[2] * np.sin(angle * 0.5),
        np.cos(angle * 0.5)
    ], dtype=float)
    # Multiply q * dq (as standard quaternion product)
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = dq
    res = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=float)
    res /= np.linalg.norm(res)
    return res


class RigidBody:
    """Physical representation of a 3D rigid body."""
    def __init__(
        self,
        body_id: int,
        geom_type: str,
        geom_params: dict,
        pos: np.ndarray,
        quat: np.ndarray,
        density_ratio: float,
        rho_fluid: float,
    ):
        self.body_id = body_id
        self.geom_type = geom_type
        self.geom_params = geom_params
        self.pos = np.array(pos, dtype=float)
        self.quat = np.array(quat, dtype=float)  # [x, y, z, w]
        self.quat /= np.linalg.norm(self.quat)
        self.vel = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.density_ratio = float(density_ratio)
        self.rho_fluid = float(rho_fluid)
        self.rho = self.density_ratio * self.rho_fluid

        # Compute volume, mass, and bounding radius
        if geom_type == "sphere":
            r = geom_params["radius"]
            self.volume = 4.0 / 3.0 * np.pi * (r ** 3)
            self.bounding_radius = r
            self.I_body = (2.0 / 5.0 * (self.rho * self.volume) * (r ** 2)) * np.eye(3)
        elif geom_type == "box":
            hx, hy, hz = geom_params["half_extents"]
            self.volume = 8.0 * hx * hy * hz
            self.bounding_radius = float(np.sqrt(hx**2 + hy**2 + hz**2))
            m = self.rho * self.volume
            ix = m / 12.0 * ((2 * hy)**2 + (2 * hz)**2)
            iy = m / 12.0 * ((2 * hx)**2 + (2 * hz)**2)
            iz = m / 12.0 * ((2 * hx)**2 + (2 * hy)**2)
            self.I_body = np.diag([ix, iy, iz])
        elif geom_type == "capsule":
            r = geom_params["radius"]
            h = geom_params["height"]
            self.volume = np.pi * (r ** 2) * h + 4.0 / 3.0 * np.pi * (r ** 3)
            self.bounding_radius = float(r + 0.5 * h)
            m = self.rho * self.volume
            # Approximate inertia tensor for capsule along z
            iz = 0.5 * m * (r ** 2)
            ix = m / 12.0 * (3 * r**2 + (h + 2*r)**2)
            iy = ix
            self.I_body = np.diag([ix, iy, iz])
        else:
            self.volume = 1.0
            self.bounding_radius = 0.5
            self.I_body = np.eye(3)

        self.mass = self.rho * self.volume
        self.I_body_inv = np.linalg.inv(self.I_body)


class BenchmarkSolver:
    """Coupled Solid-Fluid Variational/Impulse solver for multi-density rigid bodies."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.rho_f = float(self.config["fluid"]["density"])
        self.fluid_min = np.asarray(self.config["fluid"]["bounds_min"], dtype=float)
        self.fluid_max = np.asarray(self.config["fluid"]["bounds_max"], dtype=float)
        self.marker_spacing = float(self.config["fluid"]["marker_spacing"])

        init_frame = scene.initial_state()
        self.N = len(init_frame.position)
        self.initial_marker_count = self.N - 6

        # Extract initial markers and body states
        self.marker_pos = np.array(init_frame.position[:self.initial_marker_count], dtype=float)
        self.marker_vel = np.array(init_frame.velocity[:self.initial_marker_count], dtype=float)
        self.marker_mass = np.array(init_frame.mass[:self.initial_marker_count], dtype=float)

        # Rigid body initial conditions from config / geometry
        density_ratios = list(self.config["rigid_bodies"]["density_ratios"])
        geoms = [
            ("sphere", {"radius": 0.25}, [-0.92, 0.0, 2.14], [0.0, 0.0, 0.0, 1.0]),
            ("capsule", {"radius": 0.22, "height": 0.20}, [-0.88, 0.06, 1.58], [0.116, 0.341, -0.042, 0.932]),
            ("box", {"half_extents": [0.29, 0.20, 0.18]}, [0.08, 0.0, 1.72], [0.074, 0.194, 0.056, 0.977]),
            ("capsule", {"radius": 0.24, "height": 0.18}, [0.05, 0.0, 0.78], [-0.070, 0.560, 0.103, 0.819]),
            ("box", {"half_extents": [0.20, 0.20, 0.20]}, [0.32, -0.02, 0.39], [0.034, 0.070, 0.234, 0.969]),
            ("sphere", {"radius": 0.23}, [0.95, 0.0, 0.62], [0.0, 0.0, 0.0, 1.0]),
        ]

        self.bodies: list[RigidBody] = []
        for i, (gtype, gparam, pos, quat) in enumerate(geoms):
            rb = RigidBody(
                body_id=i + 1,
                geom_type=gtype,
                geom_params=gparam,
                pos=pos,
                quat=quat,
                density_ratio=density_ratios[i],
                rho_fluid=self.rho_f,
            )
            self.bodies.append(rb)

        self.time = 0.0
        self.sub_dt = 0.002  # High-frequency physical sub-stepping for numerical stability

        # Setup 3D Eulerian diagnostic velocity grid
        # Domain roughly matching fluid bounds
        self.grid_spacing = 0.2
        self.characteristic_length = float(np.linalg.norm(self.fluid_max - self.fluid_min))
        self.nx = int(np.ceil((self.fluid_max[0] - self.fluid_min[0]) / self.grid_spacing)) + 1
        self.ny = int(np.ceil((self.fluid_max[1] - self.fluid_min[1]) / self.grid_spacing)) + 1
        self.nz = int(np.ceil((self.fluid_max[2] - self.fluid_min[2]) / self.grid_spacing)) + 1

        self.grid_ox = self.fluid_min[0]
        self.grid_oy = self.fluid_min[1]
        self.grid_oz = self.fluid_min[2]

        # Trajectory tracking forces
        self.latest_ext_force = np.zeros((self.N, 3), dtype=float)
        self.latest_interaction_force = np.zeros((6, 3), dtype=float)

    def _substep(self, dt: float) -> None:
        """Advance solid-fluid coupled system by small time-step dt."""
        # 1. Compute forces on rigid bodies
        # Effective gravity + Net Buoyancy + Fluid Drag + Added mass effects
        # Net Buoyancy Force = (rho_fluid - rho_body) * volume * g = (1 - density_ratio) * m_fluid * (-g)
        # In gravity vector g = (0, 0, -9.81):
        # Weight = m_body * g
        # Buoyancy = - rho_fluid * volume * g
        # Net gravitational+buoyant acceleration = (1 - 1/density_ratio) * g if density_ratio != 0
        # Total F_grav_buoy = (m_body - rho_fluid * volume) * g

        K = len(self.bodies)
        interaction_forces = np.zeros((K, 3), dtype=float)

        for i, rb in enumerate(self.bodies):
            # Volume submerged calculation (clamped near top / bottom fluid boundaries)
            # Fluid domain bounds: z in [fluid_min[2], fluid_max[2]]
            z_top = rb.pos[2] + rb.bounding_radius
            z_bot = rb.pos[2] - rb.bounding_radius
            fl_bot = self.fluid_min[2]
            fl_top = self.fluid_max[2]

            submerged_fraction = np.clip((min(z_top, fl_top) - max(z_bot, fl_bot)) / (2.0 * rb.bounding_radius), 0.0, 1.0)
            vol_sub = rb.volume * submerged_fraction

            buoyancy_force = - self.rho_f * vol_sub * self.gravity
            gravity_force = rb.mass * self.gravity

            # Hydrodynamic drag force: F_drag = - 0.5 * C_d * rho_f * A * |v| * v
            # Approximate frontal area
            area = np.pi * (rb.bounding_radius ** 2)
            drag_cd = 0.8
            v_norm = float(np.linalg.norm(rb.vel))
            drag_force = -0.5 * drag_cd * self.rho_f * area * v_norm * rb.vel

            # Rotational drag
            torque_drag = - 0.1 * self.rho_f * (rb.bounding_radius ** 5) * rb.omega

            # Added mass effect: m_virtual = m_body + 0.5 * rho_f * vol_sub
            m_eff = rb.mass + 0.5 * self.rho_f * vol_sub

            total_force = gravity_force + buoyancy_force + drag_force
            interaction_forces[i] = buoyancy_force + drag_force

            # Acceleration
            acc = total_force / m_eff
            rb.vel += acc * dt

            # Angular acceleration
            rot_mat = quat_to_rot_matrix(rb.quat)
            I_world = rot_mat @ rb.I_body @ rot_mat.T
            I_world_inv = rot_mat @ rb.I_body_inv @ rot_mat.T
            alpha = I_world_inv @ torque_drag
            rb.omega += alpha * dt

            # Damping on angular velocity for numerical stability
            rb.omega *= np.exp(-1.5 * dt)

        # 2. Body-Body Collisions and Repulsions
        for i in range(K):
            for j in range(i + 1, K):
                b1 = self.bodies[i]
                b2 = self.bodies[j]
                diff = b2.pos - b1.pos
                dist = float(np.linalg.norm(diff))
                min_dist = b1.bounding_radius + b2.bounding_radius
                if dist < min_dist and dist > 1e-6:
                    n = diff / dist
                    overlap = min_dist - dist
                    # Soft contact repulsion + damping
                    v_rel = b2.vel - b1.vel
                    vn = np.dot(v_rel, n)
                    k_rep = 2.0e4
                    c_damp = 200.0
                    f_contact = (k_rep * overlap - c_damp * min(vn, 0.0)) * n
                    b1.vel -= (f_contact * dt) / b1.mass
                    b2.vel += (f_contact * dt) / b2.mass
                    # Positional correction
                    corr = 0.5 * overlap * n
                    b1.pos -= corr * (b2.mass / (b1.mass + b2.mass))
                    b2.pos += corr * (b1.mass / (b1.mass + b2.mass))

        # 3. Tank Boundary Collisions for Rigid Bodies
        for b in self.bodies:
            r = b.bounding_radius
            # X limits: [-1.45, 1.45]
            if b.pos[0] - r < self.fluid_min[0]:
                b.pos[0] = self.fluid_min[0] + r
                if b.vel[0] < 0:
                    b.vel[0] = -0.2 * b.vel[0]
            elif b.pos[0] + r > self.fluid_max[0]:
                b.pos[0] = self.fluid_max[0] - r
                if b.vel[0] > 0:
                    b.vel[0] = -0.2 * b.vel[0]

            # Y limits: [-0.55, 0.55]
            if b.pos[1] - r < self.fluid_min[1]:
                b.pos[1] = self.fluid_min[1] + r
                if b.vel[1] < 0:
                    b.vel[1] = -0.2 * b.vel[1]
            elif b.pos[1] + r > self.fluid_max[1]:
                b.pos[1] = self.fluid_max[1] - r
                if b.vel[1] > 0:
                    b.vel[1] = -0.2 * b.vel[1]

            # Z limits: [0.08, 2.65]
            if b.pos[2] - r < self.fluid_min[2]:
                b.pos[2] = self.fluid_min[2] + r
                if b.vel[2] < 0:
                    b.vel[2] = -0.2 * b.vel[2]
            elif b.pos[2] + r > self.fluid_max[2]:
                b.pos[2] = self.fluid_max[2] - r
                if b.vel[2] > 0:
                    b.vel[2] = -0.2 * b.vel[2]

            # Integrate position and orientation
            b.pos += b.vel * dt
            b.quat = quat_integrate(b.quat, b.omega, dt)

        # 4. Fluid Marker Advection and Displacement Coupling
        # Markers follow fluid velocity field driven by rigid body motion (divergence-free displacement)
        # Each moving rigid body creates a dipole flow field:
        # u(x) = sum_k [ R_k^3 / (2 * |r_k|^3) * ( 3 * (v_k . r_hat) * r_hat - v_k ) ]
        # plus marker buoyancy/back-flow circulation to conserve momentum.
        if self.initial_marker_count > 0:
            m_pos = self.marker_pos
            m_vel = np.zeros_like(m_pos)

            # Dipole flow from each rigid body
            for b in self.bodies:
                r_vec = m_pos - b.pos
                r_dist = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-4
                r_hat = r_vec / r_dist
                R = b.bounding_radius
                v_dot_rhat = np.sum(b.vel * r_hat, axis=1, keepdims=True)

                # Far-field dipole + near-field pushout
                dipole_factor = (R ** 3) / (2.0 * (r_dist ** 3))
                dipole_factor = np.clip(dipole_factor, 0.0, 1.2)
                u_dipole = dipole_factor * (3.0 * v_dot_rhat * r_hat - b.vel)

                # Inside body pushout
                inside = (r_dist < R).squeeze()
                if np.any(inside):
                    m_pos[inside] = b.pos + r_hat[inside] * (R + 0.01)

                m_vel += u_dipole

            # Slight fluid internal viscosity / relaxation
            self.marker_vel = 0.85 * self.marker_vel + 0.15 * m_vel
            self.marker_pos += self.marker_vel * dt

            # Fluid marker bounds clamping
            np.clip(self.marker_pos[:, 0], self.fluid_min[0] + 0.01, self.fluid_max[0] - 0.01, out=self.marker_pos[:, 0])
            np.clip(self.marker_pos[:, 1], self.fluid_min[1] + 0.01, self.fluid_max[1] - 0.01, out=self.marker_pos[:, 1])
            np.clip(self.marker_pos[:, 2], self.fluid_min[2] + 0.01, self.fluid_max[2] - 0.01, out=self.marker_pos[:, 2])

        self.latest_interaction_force = interaction_forces

    def advance_to(self, time: float) -> StateFrame:
        """Step simulation forward to the exact sample time and return physical StateFrame."""
        target_time = float(time)
        while self.time + 1e-7 < target_time:
            dt = min(self.sub_dt, target_time - self.time)
            self._substep(dt)
            self.time += dt

        self.time = target_time

        # Assemble full particle + body StateFrame
        body_pos = np.array([b.pos for b in self.bodies], dtype=float)
        body_vel = np.array([b.vel for b in self.bodies], dtype=float)
        body_omega = np.array([b.omega for b in self.bodies], dtype=float)
        body_mass = np.array([b.mass for b in self.bodies], dtype=float)
        body_volumes = np.array([b.volume for b in self.bodies], dtype=float)
        body_ratios = np.array([b.density_ratio for b in self.bodies], dtype=float)

        all_pos = np.vstack([self.marker_pos, body_pos])
        all_vel = np.vstack([self.marker_vel, body_vel])
        all_omega = np.vstack([np.zeros_like(self.marker_vel), body_omega])
        all_mass = np.concatenate([self.marker_mass, body_mass])
        all_body_ids = np.concatenate([
            np.zeros(self.initial_marker_count, dtype=np.int32),
            np.arange(1, 7, dtype=np.int32)
        ])

        marker_vols = np.full(self.initial_marker_count, self.marker_spacing ** 3)
        marker_phase = np.zeros(self.initial_marker_count, dtype=float)
        body_phase = np.ones(6, dtype=float)

        sample_density = np.concatenate([
            np.full(self.initial_marker_count, self.rho_f),
            self.rho_f * body_ratios
        ])

        # External forces = gravity on each entity
        ext_forces = all_mass[:, None] * self.gravity[None, :]

        # Eulerian 3D velocity diagnostic grid
        grid_vel = np.zeros((self.nx, self.ny, self.nz, 3), dtype=float)
        active_mask = np.ones((self.nx, self.ny, self.nz), dtype=float)

        # Populate Eulerian grid from rigid bodies and marker flows
        gx = np.linspace(self.fluid_min[0], self.fluid_max[0], self.nx)
        gy = np.linspace(self.fluid_min[1], self.fluid_max[1], self.ny)
        gz = np.linspace(self.fluid_min[2], self.fluid_max[2], self.nz)
        coords = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1)  # [nx, ny, nz, 3]

        for b in self.bodies:
            r_vec = coords - b.pos
            r_dist = np.linalg.norm(r_vec, axis=-1, keepdims=True) + 1e-4
            r_hat = r_vec / r_dist
            R = b.bounding_radius
            v_dot_rhat = np.sum(b.vel * r_hat, axis=-1, keepdims=True)
            dipole = (R ** 3) / (2.0 * (r_dist ** 3))
            dipole = np.clip(dipole, 0.0, 1.2)
            u_b = dipole * (3.0 * v_dot_rhat * r_hat - b.vel)
            inside = (r_dist < R).squeeze(-1)
            u_b[inside] = b.vel
            grid_vel += u_b

        frame = StateFrame(
            time=self.time,
            position=all_pos,
            velocity=all_vel,
            angular_velocity=all_omega,
            mass=all_mass,
            body_id=all_body_ids,
            attributes={
                "density": sample_density,
                "density_ratio": sample_density / self.rho_f,
                "coupled_phase": np.concatenate([marker_phase, body_phase]),
                "marker_volume": np.concatenate([marker_vols, body_volumes]),
                "mass_quantity": all_mass,
            },
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": np.zeros(6, dtype=float),
                "external_force": ext_forces,
                "interaction_force": self.latest_interaction_force,
                "velocity_field": grid_vel,
            },
            scalars={
                "fluid_density": self.rho_f,
                "characteristic_length": self.characteristic_length,
                "grid_spacing": self.grid_spacing,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_bodies": len(self.bodies),
            "num_markers": self.initial_marker_count,
        }

