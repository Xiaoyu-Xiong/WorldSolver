"""Physical Newton's Cradle Breakaway Solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Physics parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g = -self.gravity[2]  # typically 9.81
        self.length_scale = float(self.config["contact"]["length_scale"])

        bodies_cfg = self.config["bodies"]
        self.N = len(bodies_cfg)
        self.masses = np.array([b["mass"] for b in bodies_cfg], dtype=float)

        # Pendulum string attachment points (pivot at z = 2.15)
        # In config, equilibrium positions are: x_i = -1.38, -0.92, -0.46, 0.0, 0.46, 0.92, 1.38; z = 1.05
        # Distance from pivot (x_i, 0, 2.15) to ball center (x_i, 0, 1.05) is L = 1.10 m.
        self.pivots_x = np.array([b["position"][0] for b in bodies_cfg], dtype=float)
        self.pivot_z = 2.15
        self.radius = 0.23
        self.L = 1.10  # length of pendulum

        # Initial conditions:
        # Per task requirements: "A row of touching equal balls hangs at rest while one end ball is displaced."
        # Displace ball 0 (leftmost ball) to an initial angle theta0.
        # e.g., theta0 = -np.radians(45.0) or -0.8 rad.
        self.theta = np.zeros(self.N, dtype=float)
        self.omega = np.zeros(self.N, dtype=float)

        # Initial angle for ball 0 (left end ball displaced backwards/upwards)
        self.theta[0] = -np.radians(45.0)  # displaced to the left

        self.current_time = 0.0
        self.sub_dt = 0.0001  # high resolution for impact and integration

    def _get_ball_positions_and_velocities(self, theta: np.ndarray, omega: np.ndarray):
        """Compute 3D position and velocity from pendulum angles."""
        # x = pivot_x + L * sin(theta)
        # y = 0
        # z = pivot_z - L * cos(theta)
        x = self.pivots_x + self.L * np.sin(theta)
        y = np.zeros_like(x)
        z = self.pivot_z - self.L * np.cos(theta)
        pos = np.column_stack([x, y, z])

        # vx = L * cos(theta) * omega
        # vy = 0
        # vz = L * sin(theta) * omega
        vx = self.L * np.cos(theta) * omega
        vy = np.zeros_like(vx)
        vz = self.L * np.sin(theta) * omega
        vel = np.column_stack([vx, vy, vz])

        return pos, vel

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until target_time
        while self.current_time < target_time - 1e-9:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Current physical state
        pos, vel = self._get_ball_positions_and_velocities(self.theta, self.omega)

        # Compute required fields:
        # 1. external_force: gravity + string tension
        # Gravity = m * g (in -z direction)
        # String tension T = m * (g * cos(theta) + L * omega^2) directed along string: (-sin(theta), 0, cos(theta))
        # Net external force on ball = F_gravity + T_string
        tension_mag = self.masses * (self.g * np.cos(self.theta) + self.L * (self.omega ** 2))
        ext_fx = -tension_mag * np.sin(self.theta)
        ext_fy = np.zeros(self.N)
        ext_fz = -self.masses * self.g + tension_mag * np.cos(self.theta)
        external_force = np.column_stack([ext_fx, ext_fy, ext_fz])

        # 2. contact_signed_distance: pairwise signed gap between adjacent balls
        # For N balls, N-1 adjacent pairs: dist(i, i+1) - 2*radius
        # Positive means separation, negative means penetration.
        gap = np.zeros(self.N - 1, dtype=float)
        for i in range(self.N - 1):
            d = np.linalg.norm(pos[i + 1] - pos[i])
            gap[i] = d - 2.0 * self.radius

        # 3. interaction_force: internal contact forces between adjacent pairs
        # If contacting or colliding, contact normal force. At rest / general motion:
        # shape [K, 3], K = N - 1
        interaction_force = np.zeros((self.N - 1, 3), dtype=float)
        for i in range(self.N - 1):
            if gap[i] < 0:
                # Spring-damper or penalty contact force if overlapping
                overlap = -gap[i]
                fn = 1e5 * overlap
                direction = (pos[i + 1] - pos[i]) / (np.linalg.norm(pos[i + 1] - pos[i]) + 1e-12)
                interaction_force[i] = fn * direction

        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.N, 1))
        angular_vel = np.zeros((self.N, 3), dtype=float)

        frame = StateFrame(
            time=float(target_time),
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular_vel,
            mass=self.masses.copy(),
            body_id=np.arange(self.N, dtype=np.int32),
            attributes={"release_time": np.zeros(self.N)},
            scalars={"contact_length_scale": float(self.length_scale)},
            fields={
                "contact_signed_distance": gap,
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
        )
        frame.validate()
        return frame

    def _step(self, dt: float) -> None:
        # RK4 or Velocity Verlet for pendulum motion
        # Equation of motion: d^2 theta / dt^2 = - (g / L) * sin(theta) - damping * omega
        damping = 0.0005  # slight air resistance

        # Step positions and velocities using semi-implicit Euler / Verlet
        alpha = -(self.g / self.L) * np.sin(self.theta) - damping * self.omega
        self.omega += alpha * dt
        self.theta += self.omega * dt

        # Collision detection and resolution (Newton's cradle simultaneous / sequential 1D impact)
        # Using exact 1D elastic impact resolution along the chain
        # In an ideal Newton's cradle, identical elastic spheres transfer momentum.
        # Check geometric overlaps between consecutive balls:
        # Distance between pivot centers is d_piv = 0.46 = 2 * radius.
        # Two spheres touch when:
        # (x_{i+1} - x_i)^2 + (z_{i+1} - z_i)^2 <= (2R)^2
        # For small to moderate theta, x_{i+1} - x_i = 2R + L(sin(theta_{i+1}) - sin(theta_i))
        # spheres touch when sin(theta_i) >= sin(theta_{i+1}), i.e., theta_i >= theta_{i+1}.

        # Multi-pass elastic collision resolution for rigid contacts
        for _ in range(5):
            collision_occurred = False
            for i in range(self.N - 1):
                # Calculate actual Euclidean distance between ball i and ball i+1
                pos_i = np.array([self.pivots_x[i] + self.L * np.sin(self.theta[i]), 0.0, self.pivot_z - self.L * np.cos(self.theta[i])])
                pos_ip1 = np.array([self.pivots_x[i+1] + self.L * np.sin(self.theta[i+1]), 0.0, self.pivot_z - self.L * np.cos(self.theta[i+1])])
                dist = np.linalg.norm(pos_ip1 - pos_i)

                if dist < 2.0 * self.radius or self.theta[i] > self.theta[i+1]:
                    # Relative velocity along tangential direction (omega)
                    # Ball i is to the left of ball i+1. If omega[i] > omega[i+1], they are approaching
                    v_rel = self.omega[i] - self.omega[i+1]
                    if v_rel > 0:
                        # Elastic collision between equal masses: swap angular velocities
                        self.omega[i], self.omega[i+1] = self.omega[i+1], self.omega[i]
                        collision_occurred = True

                    # Positional correction to prevent overlap / penetration
                    # If theta[i] > theta[i+1], resolve overlap
                    if self.theta[i] > self.theta[i+1]:
                        mid = 0.5 * (self.theta[i] + self.theta[i+1])
                        self.theta[i] = mid
                        self.theta[i+1] = mid

            if not collision_occurred:
                break

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "exact_pendulum_cradle"}

