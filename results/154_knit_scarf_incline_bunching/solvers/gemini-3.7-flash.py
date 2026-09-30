"""Knitted scarf bunching on an incline physics simulation solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Retrieve initial state from scene
        frame0 = scene.initial_state()
        self.current_time = float(frame0.time)
        self.x = frame0.position.copy()
        self.v = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.num_particles = len(self.x)

        # Extract config parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        cloth_cfg = self.config.get("cloth", {})
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.grid_shape = tuple(map(int, cloth_cfg.get("grid_shape", (17, 17))))
        nx, ny = self.grid_shape

        # Initial reference positions in 2D cloth material space
        sx, sy = map(float, cloth_cfg.get("size", (3.0, 2.0)))
        x_lin = np.linspace(-sx * 0.5, sx * 0.5, nx)
        y_lin = np.linspace(-sy * 0.5, sy * 0.5, ny)
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
        self.ref_pos_2d = np.column_stack((xx.ravel(), yy.ravel()))

        # Set up springs/constraints for knit cloth
        # Knit cloth has:
        # 1. Structural springs (horizontal, vertical)
        # 2. Shear springs (diagonals)
        # 3. Bending springs (distance 2)
        # Knit behavior is anisotropic: stretchy in width/length but highly flexible in bending,
        # resisting excessive stretch via non-linear stiffness or high stretch resistance.
        springs = []
        rest_lengths = []
        spring_types = []  # 0: structural, 1: shear, 2: bend

        def idx(ix, iy):
            return iy * nx + ix

        for iy in range(ny):
            for ix in range(nx):
                p0 = idx(ix, iy)
                # Structural right
                if ix + 1 < nx:
                    p1 = idx(ix + 1, iy)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(0)
                # Structural up
                if iy + 1 < ny:
                    p1 = idx(ix, iy + 1)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(0)
                # Shear diagonal 1
                if ix + 1 < nx and iy + 1 < ny:
                    p1 = idx(ix + 1, iy + 1)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(1)
                # Shear diagonal 2
                if ix + 1 < nx and iy - 1 >= 0:
                    p1 = idx(ix + 1, iy - 1)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(1)
                # Bend right 2
                if ix + 2 < nx:
                    p1 = idx(ix + 2, iy)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(2)
                # Bend up 2
                if iy + 2 < ny:
                    p1 = idx(ix, iy + 2)
                    springs.append((p0, p1))
                    rest_lengths.append(np.linalg.norm(self.x[p0] - self.x[p1]))
                    spring_types.append(2)

        self.spring_pairs = np.array(springs, dtype=np.int32)
        self.rest_lengths = np.array(rest_lengths, dtype=float)
        self.spring_types = np.array(spring_types, dtype=np.int32)

        # Incline plane geometry:
        # Incline is tilted about Y by 35 deg.
        # Normal of incline: n = [sin(35°), 0, cos(35°)]
        # Center of incline top surface: [0, 0, 0.8 + 0.08] = [0, 0, 0.88]
        tilt_rad = np.deg2rad(35.0)
        self.incline_normal = np.array([np.sin(tilt_rad), 0.0, np.cos(tilt_rad)], dtype=float)
        # Point on the top surface of the incline
        self.incline_point = np.array([0.0, 0.0, 0.88], dtype=float)
        # Incline bounds in the incline frame
        # Length along slope is 2.2 * 2 = 4.4, half length 2.2
        # Width across slope is 1.1 * 2 = 2.2, half width 1.1
        self.incline_half_l = 2.2
        self.incline_half_w = 1.1

        # Ground plane geometry:
        # Box half_extents [2.4, 2.0, 0.05] at translation [0, 0, -0.05] -> top surface at z = 0.0
        self.ground_z = 0.0

        # Boundary vertex metadata
        self.boundary_vertex = frame0.attributes.get("boundary_vertex", np.zeros(self.num_particles, dtype=np.int32))

        # Stiffness parameters (PBD / XPBD)
        # Knitted cloth is soft in bending, moderately extensible in tension,
        # with high compression resistance to bunch into thick folds.
        self.substeps = 50
        self.k_stretch = 0.95
        self.k_shear = 0.80
        self.k_bend = 0.35
        self.friction_incline = 0.15
        self.friction_ground = 0.6
        self.damping = 0.998

        # Track contact signed distances
        self.contact_signed_dist = np.full(self.num_particles, self.thickness)

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute 2x2 deformation gradient for each particle in the grid."""
        nx, ny = self.grid_shape
        F = np.zeros((self.num_particles, 2, 2), dtype=float)
        pos = self.x
        ref = self.ref_pos_2d

        for iy in range(ny):
            for ix in range(nx):
                p = iy * nx + ix
                # Central differences or one-sided differences at boundaries
                if ix > 0 and ix < nx - 1:
                    dx_cloth = (pos[iy * nx + (ix + 1)] - pos[iy * nx + (ix - 1)]) * 0.5
                    dx_ref = (ref[iy * nx + (ix + 1)] - ref[iy * nx + (ix - 1)]) * 0.5
                elif ix == 0:
                    dx_cloth = pos[iy * nx + (ix + 1)] - pos[iy * nx + ix]
                    dx_ref = ref[iy * nx + (ix + 1)] - ref[iy * nx + ix]
                else:
                    dx_cloth = pos[iy * nx + ix] - pos[iy * nx + (ix - 1)]
                    dx_ref = ref[iy * nx + ix] - ref[iy * nx + (ix - 1)]

                if iy > 0 and iy < ny - 1:
                    dy_cloth = (pos[(iy + 1) * nx + ix] - pos[(iy - 1) * nx + ix]) * 0.5
                    dy_ref = ref[(iy + 1) * nx + ix] - ref[(iy - 1) * nx + ix] * 0.5
                elif iy == 0:
                    dy_cloth = pos[(iy + 1) * nx + ix] - pos[iy * nx + ix]
                    dy_ref = ref[(iy + 1) * nx + ix] - ref[iy * nx + ix]
                else:
                    dy_cloth = pos[iy * nx + ix] - pos[(iy - 1) * nx + ix]
                    dy_ref = ref[iy * nx + ix] - ref[(iy - 1) * nx + ix]

                # Project cloth tangent vectors to 2D stretch magnitudes
                len_ref_x = np.linalg.norm(dx_ref[:2]) if np.linalg.norm(dx_ref[:2]) > 1e-8 else 1.0
                len_ref_y = np.linalg.norm(dy_ref[:2]) if np.linalg.norm(dy_ref[:2]) > 1e-8 else 1.0

                len_cloth_x = np.linalg.norm(dx_cloth)
                len_cloth_y = np.linalg.norm(dy_cloth)

                # Dot product for shear
                cos_angle = np.dot(dx_cloth, dy_cloth) / max(len_cloth_x * len_cloth_y, 1e-8)

                F[p, 0, 0] = len_cloth_x / len_ref_x
                F[p, 1, 1] = len_cloth_y / len_ref_y
                F[p, 0, 1] = cos_angle
                F[p, 1, 0] = cos_angle

        return F

    def _step(self, dt: float) -> None:
        """Advance simulation by dt using Position Based Dynamics (PBD)."""
        sub_dt = dt / self.substeps
        sub_dt2 = sub_dt * sub_dt
        inv_mass = 1.0 / self.mass

        p0_idx = self.spring_pairs[:, 0]
        p1_idx = self.spring_pairs[:, 1]
        rest_lens = self.rest_lengths
        stiffness = np.where(
            self.spring_types == 0,
            self.k_stretch,
            np.where(self.spring_types == 1, self.k_shear, self.k_bend),
        )

        n_particles = self.num_particles
        r_collision = self.thickness * 0.5

        # Tangent vector down the slope
        # Incline normal: [sin(35), 0, cos(35)]
        # Slope direction downhill: [-cos(35), 0, sin(35)]
        tilt_rad = np.deg2rad(35.0)
        cos_t = np.cos(tilt_rad)
        sin_t = np.sin(tilt_rad)
        downhill_dir = np.array([-cos_t, 0.0, sin_t], dtype=float)

        for _ in range(self.substeps):
            # 1. External forces (gravity) & predict positions
            x_prev = self.x.copy()
            self.v += self.gravity * sub_dt
            self.v *= self.damping
            x_pred = self.x + self.v * sub_dt

            # 2. Position constraints iteration (PBD)
            for _iter in range(4):
                # Spring distance constraints
                p0_pos = x_pred[p0_idx]
                p1_pos = x_pred[p1_idx]
                diff = p0_pos - p1_pos
                dist = np.linalg.norm(diff, axis=1, keepdims=True)
                dist = np.maximum(dist, 1e-9)

                w0 = inv_mass[p0_idx, None]
                w1 = inv_mass[p1_idx, None]
                w_sum = w0 + w1

                corr = (dist - rest_lens[:, None]) / (w_sum + 1e-9) * (diff / dist)
                corr *= stiffness[:, None]

                np.add.at(x_pred, p0_idx, -corr * w0)
                np.add.at(x_pred, p1_idx, corr * w1)

                # Knit thickness / self-collision volume approximation:
                # Scarf self-bunching repulsion between nearby particles
                # Grid-based fast self-repulsion
                diff_self = x_pred[:, None, :] - x_pred[None, :, :]
                dist_self = np.linalg.norm(diff_self, axis=-1)
                mask = (dist_self < 2.0 * r_collision) & (dist_self > 1e-6)
                np.fill_diagonal(mask, False)
                if np.any(mask):
                    i_indices, j_indices = np.where(mask)
                    # Filter non-neighboring particles to allow bunching volume
                    idx_diff = np.abs(i_indices - j_indices)
                    repulse_mask = idx_diff > 2
                    if np.any(repulse_mask):
                        ii = i_indices[repulse_mask]
                        jj = j_indices[repulse_mask]
                        d = dist_self[ii, jj, None]
                        overlap = 2.0 * r_collision - d
                        dir_self = diff_self[ii, jj] / d
                        corr_self = 0.5 * overlap * dir_self
                        np.add.at(x_pred, ii, corr_self * 0.5)
                        np.add.at(x_pred, jj, -corr_self * 0.5)

                # Environment Collisions:
                # A. Incline plane contact
                # Incline plane: normal n = [sin(35), 0, cos(35)], point p_inc = [0, 0, 0.88]
                d_inc = np.dot(x_pred - self.incline_point, self.incline_normal)
                # Check if particle is over the incline extents
                rel_pos = x_pred - np.array([0.0, 0.0, 0.8])
                # Incline local X (downhill / along incline)
                local_u = -rel_pos[:, 0] * cos_t + rel_pos[:, 2] * sin_t
                local_v = rel_pos[:, 1]
                on_incline = (np.abs(local_u) <= self.incline_half_l + 0.1) & (
                    np.abs(local_v) <= self.incline_half_w + 0.1
                )

                penetration_inc = (d_inc < r_collision) & on_incline
                if np.any(penetration_inc):
                    pen_depth = r_collision - d_inc[penetration_inc]
                    x_pred[penetration_inc] += self.incline_normal * pen_depth[:, None]
                    # Incline friction
                    v_tan = x_pred[penetration_inc] - x_prev[penetration_inc]
                    v_tan -= (
                        np.sum(v_tan * self.incline_normal, axis=1, keepdims=True)
                        * self.incline_normal
                    )
                    x_pred[penetration_inc] -= v_tan * self.friction_incline

                # B. Ground plane contact
                penetration_ground = x_pred[:, 2] < r_collision
                if np.any(penetration_ground):
                    x_pred[penetration_ground, 2] = r_collision
                    # Ground friction
                    v_xy = x_pred[penetration_ground, :2] - x_prev[penetration_ground, :2]
                    x_pred[penetration_ground, :2] -= v_xy * self.friction_ground

            # 3. Update velocity and position
            self.v = (x_pred - x_prev) / sub_dt
            self.x = x_pred

        # Update signed contact distances (distance to closest obstacle/ground)
        # Ground distance: z - r_collision
        # Incline distance: d_inc - r_collision
        d_inc_final = np.dot(self.x - self.incline_point, self.incline_normal) - r_collision
        d_ground_final = self.x[:, 2] - r_collision
        self.contact_signed_dist = np.minimum(d_ground_final, np.where(on_incline, d_inc_final, 10.0))

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to the target time and return StateFrame."""
        dt = time - self.current_time
        if dt > 1e-7:
            self._step(dt)
            self.current_time = float(time)

        # External forces on particles: gravity
        ext_force = self.mass[:, None] * self.gravity[None, :]

        # Deformation gradient
        F = self._compute_deformation_gradient()

        frame = StateFrame(
            time=float(self.current_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_particles, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": self.contact_signed_dist.copy(),
                "cumulative_mass_source": np.zeros(self.num_particles, dtype=float),
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_z": float(np.mean(self.x[:, 2])),
            "max_velocity": float(np.max(np.linalg.norm(self.v, axis=-1))),
        }

