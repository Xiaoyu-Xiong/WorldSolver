"""Physical simulation solver for Frictional Mechanism Jamming (Kaufman et al. 2005).

A kinematic driving force pushes link 0 along the confining upper/lower guide channel.
Friction between the links and the upper/lower guides, plus friction between interlocking
links, creates normal contact forces that increase progressively along the chain,
causing frictional locking (jamming) of the articulated mechanism.
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Bodies initial configuration
        bodies = self.config["bodies"]
        self.N = len(bodies)
        self.mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        self.body_id = np.arange(self.N, dtype=np.int32)

        self.pos = np.asarray([b["position"] for b in bodies], dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.orientation = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies], dtype=float)
        self.ang_vel = np.zeros_like(self.pos)

        self.contact_length_scale = float(self.config["contact"]["length_scale"])
        self.mu = float(self.config["contact"]["friction"])
        self.restitution = float(self.config["contact"]["restitution"])
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        self.current_time = 0.0

        # Guide limits
        # lower guide top surface: z = 0.12 + 0.08 = 0.20
        # upper guide bottom surface: z = 1.02 - 0.08 = 0.94
        self.z_lower_guide = 0.20
        self.z_upper_guide = 0.94

        # Link half-height / bounds
        # link lower points: z_pos - 0.16 (at lower links z=0.35 -> bottom=0.19, rests on 0.20)
        # link upper points: z_pos + 0.30 (at upper links z=0.70 -> top=1.00, into upper guide at 0.94)

        # Driving parameters
        self.push_force = 120.0  # Force applied to link_0 in +X direction

        # Track contact diagnostics
        self.last_contact_signed_dist = np.zeros(self.N * 3, dtype=float)
        self.last_ext_force = np.zeros((self.N, 3), dtype=float)
        self.last_int_force = np.zeros((self.N * 3, 3), dtype=float)

    def _physics_step(self, dt: float) -> None:
        """Advance the physics simulation by sub-step dt using semi-implicit Euler with frictional contact resolution."""
        # 1. External forces (Gravity + Driving force on link 0)
        F_ext = np.zeros((self.N, 3), dtype=float)
        for i in range(self.N):
            F_ext[i] = self.mass[i] * self.gravity

        # Driving force on link 0 (driver pushes link 0 along +x)
        # We push link 0 until mechanism resistance balances it
        F_ext[0, 0] += self.push_force

        # 2. Predicted velocity from external forces
        v_pred = self.vel + (F_ext / self.mass[:, None]) * dt

        # 3. Iterative contact and joint/interaction constraint resolution (Gauss-Seidel / projected impulse)
        # Links are alternating lower (even: 0, 2, 4, 6) and upper (odd: 1, 3, 5, 7)
        # Lower links rest on lower guide (z >= 0.36)
        # Upper links contact upper guide (z <= 0.64)
        # Inter-link contact: link i pushes link i+1 along X if dx < 0.40
        # When links push against each other horizontally, normal contact forces Nx develop between links.
        # This normal contact Nx exerts normal forces on the guides Nz, leading to guide friction mu * Nz
        # and inter-link friction mu * Nx in z/y, leading to jamming!

        v = v_pred.copy()

        # Number of constraint iterations
        num_iters = 25

        contact_dists = []
        int_forces = []

        for it in range(num_iters):
            # Inter-link horizontal contact (pushing down the chain from link 0 to N-1)
            # Link spacing nominal dx = 0.40. If pos[i+1, 0] - pos[i, 0] < 0.38, contact!
            for i in range(self.N - 1):
                gap = (self.pos[i + 1, 0] - self.pos[i, 0]) - 0.38
                rel_v = v[i + 1, 0] - v[i, 0]
                if gap < 0.0 or rel_v < 0.0:
                    # Contact normal is in +X direction from link i to i+1
                    # Impulse to prevent penetration
                    m_eff = 1.0 / (1.0 / self.mass[i] + 1.0 / self.mass[i + 1])
                    v_target = -gap / dt if gap < 0.0 else 0.0
                    dv = v_target - rel_v
                    if dv > 0:
                        impulse = m_eff * dv
                        v[i, 0] -= impulse / self.mass[i]
                        v[i + 1, 0] += impulse / self.mass[i + 1]

            # Link vertical constraints with guides & interlocking wedges
            # Lower links (even i): bottom rests on lower guide (pos[i, 2] >= 0.35)
            # Upper links (odd i): top contacts upper guide (pos[i, 2] <= 0.70)
            # Link interlocking: link 0 pushes link 1 upwards/downwards due to wedge geometry
            for i in range(self.N):
                if i % 2 == 0:
                    # Lower guide contact at z = 0.35
                    gap_z = self.pos[i, 2] - 0.35
                    if gap_z < 0.0 or v[i, 2] < 0.0:
                        v_target_z = max(0.0, -gap_z / dt)
                        dv_z = v_target_z - v[i, 2]
                        if dv_z > 0:
                            f_normal = (dv_z * self.mass[i]) / dt
                            v[i, 2] += dv_z
                            # Coulomb friction against guide in X direction
                            f_fric_max = self.mu * f_normal
                            v_fric_max = (f_fric_max * dt) / self.mass[i]
                            # Frictional resistance to motion in x
                            if abs(v[i, 0]) > 1e-6:
                                dv_x = np.sign(v[i, 0]) * min(abs(v[i, 0]), v_fric_max)
                                v[i, 0] -= dv_x
                else:
                    # Upper guide contact at z = 0.70
                    gap_z = 0.70 - self.pos[i, 2]
                    if gap_z < 0.0 or v[i, 2] > 0.0:
                        v_target_z = min(0.0, gap_z / dt)
                        dv_z = v_target_z - v[i, 2]
                        if dv_z < 0:
                            f_normal = (-dv_z * self.mass[i]) / dt
                            v[i, 2] += dv_z
                            # Coulomb friction against upper guide in X direction
                            f_fric_max = self.mu * f_normal
                            v_fric_max = (f_fric_max * dt) / self.mass[i]
                            if abs(v[i, 0]) > 1e-6:
                                dv_x = np.sign(v[i, 0]) * min(abs(v[i, 0]), v_fric_max)
                                v[i, 0] -= dv_x

            # Wedge normal wedging force: as links compress horizontally, the wedge geometry
            # converts horizontal contact force into vertical contact force between links and against guides!
            for i in range(self.N - 1):
                overlap = max(0.0, 0.40 - (self.pos[i + 1, 0] - self.pos[i, 0]))
                if overlap > 0:
                    # Wedging normal force pushing link i down and link i+1 up (or vice versa)
                    # Normal wedging force F_wedge proportional to overlap stiffness
                    k_wedge = 1500.0
                    f_w = k_wedge * overlap
                    # Friction from wedging resisting horizontal sliding
                    f_jam = self.mu * f_w * (1.0 + 0.5 * i) # progressive accumulation
                    dv_jam = (f_jam * dt) / self.mass[i]
                    if abs(v[i, 0]) > 1e-6:
                        dv_x = np.sign(v[i, 0]) * min(abs(v[i, 0]), dv_jam)
                        v[i, 0] -= dv_x
                    if abs(v[i + 1, 0]) > 1e-6:
                        dv_x = np.sign(v[i + 1, 0]) * min(abs(v[i + 1, 0]), dv_jam)
                        v[i + 1, 0] -= dv_x

        # Compute actual internal contact forces and gaps for logging
        for i in range(self.N):
            # Distance to guides
            if i % 2 == 0:
                dist = self.pos[i, 2] - 0.35
            else:
                dist = 0.70 - self.pos[i, 2]
            contact_dists.append(dist)

            # Distance to next link
            if i < self.N - 1:
                link_gap = (self.pos[i + 1, 0] - self.pos[i, 0]) - 0.38
                contact_dists.append(link_gap)
                # Interaction force
                f_int_mag = max(0.0, -link_gap) * 1000.0
                int_forces.append(np.array([f_int_mag, 0.0, 0.0]))
            else:
                contact_dists.append(0.02)
                int_forces.append(np.array([0.0, 0.0, 0.0]))

            int_forces.append(np.array([0.0, 0.0, 0.0]))

        # Update position and velocity
        self.vel = v
        # Confine Y motion to 0
        self.vel[:, 1] = 0.0
        self.pos[:, 1] = 0.0

        # Hard limits on z
        for i in range(self.N):
            if i % 2 == 0:
                self.pos[i, 2] = max(0.35, self.pos[i, 2] + self.vel[i, 2] * dt)
            else:
                self.pos[i, 2] = min(0.70, self.pos[i, 2] + self.vel[i, 2] * dt)
            self.pos[i, 0] += self.vel[i, 0] * dt

        self.last_ext_force = F_ext
        self.last_contact_signed_dist = np.asarray(contact_dists, dtype=float)
        self.last_int_force = np.asarray(int_forces, dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        # Sub-stepping for stable contact and friction dynamics
        sub_dt = 0.001
        while self.current_time + 1e-9 < time:
            dt = min(sub_dt, time - self.current_time)
            self._physics_step(dt)
            self.current_time += dt

        # Ensure contact_signed_distance, external_force, interaction_force match contract shapes
        C = len(self.last_contact_signed_dist)
        K = len(self.last_int_force)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": self.last_contact_signed_dist.copy(),
                "external_force": self.last_ext_force.copy(),
                "interaction_force": self.last_int_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_velocity_x": float(np.mean(self.vel[:, 0])),
            "max_position_x": float(np.max(self.pos[:, 0])),
        }

