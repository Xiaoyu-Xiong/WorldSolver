"""Cloth Sleeve on Articulated Cylinders Solver.

Simulates a tubular cloth sleeve undergoing bending and contact around two articulated kinematic capsules.
"""
from __future__ import annotations

import json
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.current_frame = None

        # Mesh data
        self.N = 0
        self.grid_shape = (17, 17)
        self.X_rest = None # [N, 3]
        self.pos = None    # [N, 3]
        self.vel = None    # [N, 3]
        self.mass = None   # [N]
        self.inv_mass = None # [N]
        self.faces = None  # [F, 3]
        self.springs = None # list of (i, j, rest_len, stiffness)
        self.bending_springs = None

        # Simulation params
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.sub_dt = 0.001
        self.thickness = 0.015

        # Limb kinematics parameters
        # Upper limb: pivot at (0, 0, 1.0), points along -x initially
        # Lower limb: pivot at (0, 0, 1.0), points along +x initially
        self.pivot = np.array([0.0, 0.0, 1.0])
        self.limb_upper_r = 0.34
        self.limb_upper_half_h = 0.65
        self.limb_lower_r = 0.31
        self.limb_lower_half_h = 0.65

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        cloth_cfg = scene.config.get("cloth", {})
        self.grid_shape = tuple(cloth_cfg.get("grid_shape", [17, 17]))
        nx, ny = self.grid_shape
        self.N = nx * ny

        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Load topology
        try:
            with open("assets/cloth_topology.json", "r") as f:
                topo = json.load(f)
            self.faces = np.array(topo["faces"], dtype=np.int32)
        except Exception:
            faces = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    v0 = j * nx + i
                    v1 = j * nx + (i + 1)
                    v2 = (j + 1) * nx + (i + 1)
                    v3 = (j + 1) * nx + i
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
            self.faces = np.array(faces, dtype=np.int32)

        # Form tubular sleeve around the cylinders
        # Cylinder axis is along x from -1.2 to +1.2 at z=1.0, y=0.0
        # Mesh uv: nx along x (-1.5 to 1.5), ny around perimeter (theta: 0 to 2*pi)
        sx, sy = float(cloth_cfg.get("size", [3.0, 2.5])[0]), float(cloth_cfg.get("size", [3.0, 2.5])[1])
        x_coords = np.linspace(-sx * 0.5, sx * 0.5, nx)
        theta_coords = np.linspace(0, 2 * np.pi, ny, endpoint=False)

        # Sleeve radius slightly larger than limb radius to fit smoothly
        sleeve_radius = 0.36
        pos = np.zeros((self.N, 3), dtype=float)
        for j in range(ny):
            th = theta_coords[j]
            y_val = sleeve_radius * np.sin(th)
            z_val = 1.0 + sleeve_radius * np.cos(th)
            for i in range(nx):
                idx = j * nx + i
                pos[idx] = [x_coords[i], y_val, z_val]

        self.pos = pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.X_rest = self.pos.copy()

        total_mass = float(cloth_cfg.get("total_mass", 0.32))
        self.mass = np.full(self.N, total_mass / self.N)
        self.inv_mass = 1.0 / self.mass

        # Build springs (structural, shear, bending)
        springs_dict = {}
        def add_spring(i1, i2, k_type):
            if i1 == i2:
                return
            pair = (min(i1, i2), max(i1, i2))
            if pair not in springs_dict:
                d = np.linalg.norm(self.X_rest[pair[0]] - self.X_rest[pair[1]])
                springs_dict[pair] = (d, k_type)

        for j in range(ny):
            j_next = (j + 1) % ny
            j_next2 = (j + 2) % ny
            for i in range(nx):
                idx = j * nx + i
                # Longitudinal (along x)
                if i + 1 < nx:
                    idx_right = j * nx + (i + 1)
                    add_spring(idx, idx_right, 'struct')
                if i + 2 < nx:
                    idx_right2 = j * nx + (i + 2)
                    add_spring(idx, idx_right2, 'bend')

                # Circumferential (around tube)
                idx_up = j_next * nx + i
                add_spring(idx, idx_up, 'struct')
                idx_up2 = j_next2 * nx + i
                add_spring(idx, idx_up2, 'bend')

                # Shear
                if i + 1 < nx:
                    idx_diag1 = j_next * nx + (i + 1)
                    add_spring(idx, idx_diag1, 'shear')
                    idx_diag2 = ((j - 1) % ny) * nx + (i + 1)
                    add_spring(idx, idx_diag2, 'shear')

        # Convert to arrays for fast PBD / Mass-Spring
        spring_list = []
        for (i1, i2), (rest_d, k_type) in springs_dict.items():
            if k_type == 'struct':
                stiff = 0.95
            elif k_type == 'shear':
                stiff = 0.85
            else: # bend
                stiff = 0.5
            spring_list.append((i1, i2, rest_d, stiff))

        self.spring_edges = np.array([[s[0], s[1]] for s in spring_list], dtype=np.int32)
        self.spring_rest = np.array([s[2] for s in spring_list], dtype=float)
        self.spring_stiff = np.array([s[3] for s in spring_list], dtype=float)

        self.current_time = 0.0
        self.current_frame = self._build_state_frame(0.0)

    def _get_limb_transforms(self, t: float):
        """Compute positions and orientations of the two articulated limbs at time t."""
        # Joint bends gradually from t=1.0 to t=5.0 up to ~70 degrees, then holds
        max_angle = np.radians(70.0)
        if t < 1.0:
            angle = 0.0
        elif t < 5.0:
            s = (t - 1.0) / 4.0
            # Smooth step
            smooth = s * s * (3.0 - 2.0 * s)
            angle = max_angle * smooth
        else:
            angle = max_angle

        # Upper limb rotates slightly or stays fixed, lower limb rotates upwards in xz plane
        # Pivot is at (0, 0, 1.0)
        # Upper limb segment: from (0, 0, 1.0) to (-1.1, 0, 1.0)
        # Upper limb center: (-0.55, 0, 1.0)
        # Lower limb segment: rotates by angle around y-axis (bending up in +z direction)
        # Lower limb direction: dir_vec = [cos(angle), 0, sin(angle)]
        # Lower limb start: (0, 0, 1.0), end: dir_vec * 1.1 + (0, 0, 1.0)
        # Upper limb start: (0, 0, 1.0), end: (-1.1, 0, 1.0)

        # Upper limb capsule line segment:
        p0_upper = self.pivot + np.array([-0.05, 0.0, 0.0])
        p1_upper = self.pivot + np.array([-1.1, 0.0, 0.0])

        # Lower limb capsule line segment:
        rot_mat = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)]
        ])
        p0_lower = self.pivot + rot_mat @ np.array([0.05, 0.0, 0.0])
        p1_lower = self.pivot + rot_mat @ np.array([1.1, 0.0, 0.0])

        return (p0_upper, p1_upper, self.limb_upper_r), (p0_lower, p1_lower, self.limb_lower_r)

    def _point_segment_distance(self, p, a, b):
        """Vectorized distance from points p [N, 3] to segment [a, b]."""
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-12:
            ap = p - a
            dist = np.linalg.norm(ap, axis=1)
            closest = np.repeat(a[None, :], len(p), axis=0)
            return dist, closest

        ap = p - a
        t = np.sum(ap * ab, axis=1) / ab_len2
        t = np.clip(t, 0.0, 1.0)
        closest = a + t[:, None] * ab
        dist = np.linalg.norm(p - closest, axis=1)
        return dist, closest

    def _step(self, dt: float, target_time: float):
        # Semi-implicit Euler / Position Based Dynamics step
        # 1. External forces
        ext_acc = self.gravity.copy()
        self.vel += ext_acc * dt

        # Damping
        self.vel *= (1.0 - 0.05 * dt)

        # 2. Predicted positions
        p_pred = self.pos + self.vel * dt

        # 3. Solver iterations for distance constraints and collision
        (p0_u, p1_u, r_u), (p0_l, p1_l, r_l) = self._get_limb_transforms(target_time)

        e0 = self.spring_edges[:, 0]
        e1 = self.spring_edges[:, 1]
        rest = self.spring_rest
        stiff = self.spring_stiff

        iters = 12
        for _ in range(iters):
            # Distance constraints
            dp = p_pred[e0] - p_pred[e1]
            dist = np.linalg.norm(dp, axis=1)
            mask = dist > 1e-8
            dist_safe = np.where(mask, dist, 1.0)
            corr_mag = (dist_safe - rest) / (self.inv_mass[e0] + self.inv_mass[e1]) * stiff
            corr_vec = (dp / dist_safe[:, None]) * corr_mag[:, None]

            # Accumulate corrections
            np.add.at(p_pred, e0, -corr_vec * self.inv_mass[e0, None] * 0.5)
            np.add.at(p_pred, e1, corr_vec * self.inv_mass[e1, None] * 0.5)

            # Capsule collisions (Upper limb)
            dist_u, closest_u = self._point_segment_distance(p_pred, p0_u, p1_u)
            pen_u = (r_u + self.thickness * 0.5) - dist_u
            coll_mask_u = pen_u > 0.0
            if np.any(coll_mask_u):
                n_u = p_pred[coll_mask_u] - closest_u[coll_mask_u]
                n_norm_u = np.linalg.norm(n_u, axis=1, keepdims=True)
                n_norm_u = np.where(n_norm_u > 1e-8, n_norm_u, 1.0)
                n_dir_u = n_u / n_norm_u
                p_pred[coll_mask_u] = closest_u[coll_mask_u] + n_dir_u * (r_u + self.thickness * 0.5)

            # Capsule collisions (Lower limb)
            dist_l, closest_l = self._point_segment_distance(p_pred, p0_l, p1_l)
            pen_l = (r_l + self.thickness * 0.5) - dist_l
            coll_mask_l = pen_l > 0.0
            if np.any(coll_mask_l):
                n_l = p_pred[coll_mask_l] - closest_l[coll_mask_l]
                n_norm_l = np.linalg.norm(n_l, axis=1, keepdims=True)
                n_norm_l = np.where(n_norm_l > 1e-8, n_norm_l, 1.0)
                n_dir_l = n_l / n_norm_l
                p_pred[coll_mask_l] = closest_l[coll_mask_l] + n_dir_l * (r_l + self.thickness * 0.5)

            # Ground collision (z >= 0.0)
            ground_pen = self.thickness * 0.5 - p_pred[:, 2]
            ground_coll = ground_pen > 0.0
            if np.any(ground_coll):
                p_pred[ground_coll, 2] = self.thickness * 0.5

        # 4. Update velocity and position
        self.vel = (p_pred - self.pos) / dt
        self.pos = p_pred.copy()

    def advance_to(self, time: float) -> StateFrame:
        if np.isclose(time, self.current_time):
            return self.current_frame

        # Substep to target time
        while self.current_time < time - 1e-8:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt, self.current_time + dt)
            self.current_time += dt

        self.current_time = time
        self.current_frame = self._build_state_frame(time)
        return self.current_frame

    def _build_state_frame(self, t: float) -> StateFrame:
        nx, ny = self.grid_shape

        # Compute 2D deformation gradients on the surface grid
        # Rest grid spacing
        # u along x, v along theta
        pos_grid = self.pos.reshape(ny, nx, 3)
        rest_grid = self.X_rest.reshape(ny, nx, 3)

        # Gradients in x (along tube) and y (around tube)
        # Using central differences
        dx = np.gradient(pos_grid, axis=1) # [ny, nx, 3]
        dy = np.gradient(pos_grid, axis=0) # [ny, nx, 3]
        dx_rest = np.gradient(rest_grid, axis=1) # [ny, nx, 3]
        dy_rest = np.gradient(rest_grid, axis=0) # [ny, nx, 3]

        # Metric stretch
        dx_len = np.linalg.norm(dx, axis=-1)
        dy_len = np.linalg.norm(dy, axis=-1)
        dx_rest_len = np.linalg.norm(dx_rest, axis=-1)
        dy_rest_len = np.linalg.norm(dy_rest, axis=-1)
        dx_rest_len = np.where(dx_rest_len > 1e-6, dx_rest_len, 1.0)
        dy_rest_len = np.where(dy_rest_len > 1e-6, dy_rest_len, 1.0)

        lambda_u = (dx_len / dx_rest_len).ravel()
        lambda_v = (dy_len / dy_rest_len).ravel()

        def_grad = np.zeros((self.N, 2, 2), dtype=float)
        def_grad[:, 0, 0] = lambda_u
        def_grad[:, 1, 1] = lambda_v

        # External forces (gravity)
        ext_force = self.mass[:, None] * self.gravity[None, :]

        # Contact signed distance
        (p0_u, p1_u, r_u), (p0_l, p1_l, r_l) = self._get_limb_transforms(t)
        dist_u, _ = self._point_segment_distance(self.pos, p0_u, p1_u)
        gap_u = dist_u - r_u
        dist_l, _ = self._point_segment_distance(self.pos, p0_l, p1_l)
        gap_l = dist_l - r_l
        signed_dist = np.minimum(gap_u, gap_l)

        # Boundary vertex indicator
        boundary = np.zeros(self.N, dtype=np.int32)
        # Tube ends (first and last longitudinal rings)
        for j in range(ny):
            boundary[j * nx] = 1
            boundary[j * nx + (nx - 1)] = 1

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": def_grad,
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N, dtype=float),
                "external_force": ext_force,
                "contact_signed_distance": signed_dist,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.N,
        }

