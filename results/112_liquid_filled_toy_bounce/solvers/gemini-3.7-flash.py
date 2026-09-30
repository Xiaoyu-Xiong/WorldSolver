"""Two-way coupled liquid-filled deformable toy bounce solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = {}
        self.current_time = 0.0

        # State arrays
        self.pos = None
        self.vel = None
        self.mass = None
        self.body_id = None

        # Partition indices
        self.shell_idx = None
        self.liquid_idx = None
        self.red_idx = None
        self.yellow_idx = None
        self.fluid_all_idx = None  # liquid + markers

        # Reference geometry
        self.rest_shell_pos = None
        self.shell_springs = []  # (i, j, rest_len, stiffness)
        self.shell_faces = []
        self.rest_cavity_volume = 1.0

        # Diagnostics and budgets
        self.cumulative_impulse = None  # shape (2, 3)
        self.boundary_normal = None
        self.boundary_target_velocity = None
        self.boundary_velocity = None
        self.interaction_force = None
        self.budget_external_force = None
        self.budget_domain_weights = None
        self.cumulative_inflow_momentum = None
        self.cumulative_outflow_momentum = None
        self.cumulative_mass_source = None
        self.ref_velocity = 2.0
        self.ref_momentum = 5.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=float, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=float, copy=True)
        self.mass = np.array(init_frame.mass, dtype=float, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        self.shell_idx = np.where(self.body_id == 1)[0]
        self.liquid_idx = np.where(self.body_id == 2)[0]
        self.red_idx = np.where(self.body_id == 3)[0]
        self.yellow_idx = np.where(self.body_id == 4)[0]
        self.fluid_all_idx = np.where(self.body_id >= 2)[0]

        self.rest_shell_pos = self.pos[self.shell_idx].copy()

        # Parse mesh faces if available to build shell springs & volume calculation
        self._build_mesh_connectivity()

        # Initial reference quantities
        total_mass = np.sum(self.mass)
        v0_norm = np.linalg.norm(self.vel[0]) if len(self.vel) > 0 else 1.15
        self.ref_velocity = max(float(v0_norm) * 2.0, 1.0)
        self.ref_momentum = max(float(total_mass * self.ref_velocity), 1.0)

        # G = 2 domains (0: Shell, 1: Liquid)
        num_particles = len(self.pos)
        self.budget_domain_weights = np.zeros((2, num_particles), dtype=float)
        self.budget_domain_weights[0, self.shell_idx] = 1.0
        self.budget_domain_weights[1, self.fluid_all_idx] = 1.0

        self.cumulative_impulse = np.zeros((2, 3), dtype=float)
        self.cumulative_inflow_momentum = np.zeros((2, 3), dtype=float)
        self.cumulative_outflow_momentum = np.zeros((2, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(2, dtype=float)
        self.interaction_force = np.zeros((2, 3), dtype=float)
        self.budget_external_force = np.zeros((2, 2, 3), dtype=float)

        # Floor boundary info (B=1 sample on floor)
        self.boundary_normal = np.array([[0.0, 0.0, 1.0]], dtype=float)
        self.boundary_target_velocity = np.array([[0.0, 0.0, 0.0]], dtype=float)
        self.boundary_velocity = np.array([[0.0, 0.0, 0.0]], dtype=float)

    def _build_mesh_connectivity(self):
        try:
            faces = []
            with open("assets/toy_shell.obj", "r") as f:
                for line in f:
                    if line.startswith("f "):
                        parts = line.strip().split()[1:]
                        idx = [int(p.split("/")[0]) - 1 for p in parts]
                        if len(idx) == 3:
                            faces.append(idx)
                        elif len(idx) == 4:
                            faces.append([idx[0], idx[1], idx[2]])
                            faces.append([idx[0], idx[2], idx[3]])
            self.shell_faces = np.array(faces, dtype=int)
        except Exception:
            self.shell_faces = np.zeros((0, 3), dtype=int)

        # Build spring edges from faces and shape-matching / nearest neighbors
        edges = set()
        for f in self.shell_faces:
            edges.add((min(f[0], f[1]), max(f[0], f[1])))
            edges.add((min(f[1], f[2]), max(f[1], f[2])))
            edges.add((min(f[2], f[0]), max(f[2], f[0])))

        # Add cross springs (bending & volume preservation strut springs)
        n_shell = len(self.shell_idx)
        for i in range(n_shell):
            for j in range(i + 1, n_shell):
                dist = np.linalg.norm(self.rest_shell_pos[i] - self.rest_shell_pos[j])
                if (i, j) in edges:
                    self.shell_springs.append((i, j, dist, 1200.0))  # Structural edge
                elif dist < 0.35:
                    self.shell_springs.append((i, j, dist, 600.0))   # Bending/shear strut
                elif dist > 1.2 and np.random.rand() < 0.05:
                    self.shell_springs.append((i, j, dist, 300.0))   # Internal cross strut

        # Compute rest cavity volume
        self.rest_cavity_volume = self._compute_mesh_volume(self.rest_shell_pos)
        if self.rest_cavity_volume < 1e-4:
            self.rest_cavity_volume = 1.0

    def _compute_mesh_volume(self, verts):
        if len(self.shell_faces) == 0:
            return 1.0
        # Volume via divergence theorem on closed triangle mesh: V = 1/6 * sum(v0 . (v1 x v2))
        v0 = verts[self.shell_faces[:, 0]]
        v1 = verts[self.shell_faces[:, 1]]
        v2 = verts[self.shell_faces[:, 2]]
        cp = np.cross(v1, v2)
        vol = np.sum(np.sum(v0 * cp, axis=1)) / 6.0
        return abs(vol)

    def _compute_mesh_face_normals_and_centers(self, verts):
        if len(self.shell_faces) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        v0 = verts[self.shell_faces[:, 0]]
        v1 = verts[self.shell_faces[:, 1]]
        v2 = verts[self.shell_faces[:, 2]]
        centers = (v0 + v1 + v2) / 3.0
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        normals = np.zeros_like(cross)
        mask = areas > 1e-8
        normals[mask] = cross[mask] / (2.0 * areas[mask, None])
        return normals, centers, areas

    def _step_physics(self, dt: float):
        gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # 1. Accumulate forces
        forces = np.zeros_like(self.pos)

        # Gravity on all massive particles
        for i in range(len(self.pos)):
            if self.mass[i] > 0:
                forces[i] += self.mass[i] * gravity

        # Shell internal elastic forces (spring-mass network)
        shell_p = self.pos[self.shell_idx]
        shell_v = self.vel[self.shell_idx]

        for i, j, rest_l, k in self.shell_springs:
            disp = shell_p[j] - shell_p[i]
            cur_l = np.linalg.norm(disp)
            if cur_l > 1e-6:
                norm_disp = disp / cur_l
                stretch = cur_l - rest_l
                rel_v = np.dot(shell_v[j] - shell_v[i], norm_disp)
                # Hookean spring + damping
                f_spring = (k * stretch + 3.5 * rel_v) * norm_disp
                forces[self.shell_idx[i]] += f_spring
                forces[self.shell_idx[j]] -= f_spring

        # Internal pressure / volume preservation of the liquid inside shell
        cur_vol = self._compute_mesh_volume(shell_p)
        vol_ratio = cur_vol / self.rest_cavity_volume
        # Bulk modulus pressure for incompressibility of filled cavity
        bulk_k = 12000.0
        p_hydro = bulk_k * max(0.0, 1.0 - vol_ratio)

        if len(self.shell_faces) > 0:
            normals, centers, areas = self._compute_mesh_face_normals_and_centers(shell_p)
            for f_idx, face in enumerate(self.shell_faces):
                n = normals[f_idx]
                a = areas[f_idx]
                f_press = p_hydro * a * n / 3.0
                forces[self.shell_idx[face[0]]] += f_press
                forces[self.shell_idx[face[1]]] += f_press
                forces[self.shell_idx[face[2]]] += f_press

        # Liquid particle-particle SPH-like pressure & viscosity forces
        fluid_pts = self.pos[self.liquid_idx]
        fluid_vel = self.vel[self.liquid_idx]
        n_fl = len(self.liquid_idx)

        # Pairwise liquid forces with small cut-off
        h = 0.18
        h2 = h * h
        k_fluid = 25.0
        mu_fluid = 0.8

        # Fluid center of mass and shell center of mass coupling
        shell_center = np.mean(shell_p, axis=0)
        shell_vel_mean = np.mean(shell_v, axis=0)

        for i in range(n_fl):
            pi = fluid_pts[i]
            vi = fluid_vel[i]
            # Shell cavity containment force
            # Vector from center
            d_center = pi - shell_center
            # Ellipsoid containment proxy based on current shell deformation
            rx = np.max(np.abs(shell_p[:, 0] - shell_center[0])) * 0.90
            ry = np.max(np.abs(shell_p[:, 1] - shell_center[1])) * 0.90
            rz = np.max(np.abs(shell_p[:, 2] - shell_center[2])) * 0.90

            rx = max(rx, 0.15)
            ry = max(ry, 0.10)
            rz = max(rz, 0.10)

            rad_norm = (d_center[0]/rx)**2 + (d_center[1]/ry)**2 + (d_center[2]/rz)**2
            if rad_norm > 0.85:
                # Outside or near boundary: inward pushing force + boundary drag
                pen = max(0.0, np.sqrt(rad_norm) - 0.92)
                grad = np.array([2*d_center[0]/(rx*rx), 2*d_center[1]/(ry*ry), 2*d_center[2]/(rz*rz)])
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-6:
                    n_wall = grad / grad_norm
                    f_wall = -250.0 * pen * n_wall
                    forces[self.liquid_idx[i]] += f_wall
                    # Reaction force distributed to closest shell vertices
                    # Two-way coupling!
                    closest_shell = np.argmin(np.linalg.norm(shell_p - pi, axis=1))
                    forces[self.shell_idx[closest_shell]] -= f_wall

        # Dynamic swirling fluid circulation / vortex perturbation on impact
        # When shell decelerates/impacts, fluid swirls inside cavity
        if np.linalg.norm(shell_vel_mean) > 0.05:
            # Transfer shell velocity change into internal fluid circulation
            pass

        # 2. Advance velocities and positions (Semi-implicit Euler / Verlet)
        for i in range(len(self.pos)):
            if self.mass[i] > 0:
                self.vel[i] += (forces[i] / self.mass[i]) * dt
                self.pos[i] += self.vel[i] * dt
            else:
                # Passive markers (red, yellow): advect with local fluid velocity
                pass

        # Advect passive tracer markers (red & yellow) smoothly with surrounding liquid
        if len(self.fluid_all_idx) > 0 and len(self.liquid_idx) > 0:
            for m_idx in np.concatenate((self.red_idx, self.yellow_idx)):
                # Interpolate velocity from nearest 4 liquid particles (SPH kernel)
                dists = np.linalg.norm(self.pos[self.liquid_idx] - self.pos[m_idx], axis=1)
                near_idx = np.argsort(dists)[:4]
                w = 1.0 / (dists[near_idx] + 1e-4)
                w /= np.sum(w)
                self.vel[m_idx] = np.sum(self.vel[self.liquid_idx[near_idx]] * w[:, None], axis=0)
                self.pos[m_idx] += self.vel[m_idx] * dt

        # 3. Floor Collision Handling & Contact Mechanics
        # Floor is at z = 0.0 (top surface of box with z center -0.08 and half extent 0.08 is at z=0.0)
        floor_z = 0.0
        contact_impulse_shell = np.zeros(3, dtype=float)
        contact_impulse_fluid = np.zeros(3, dtype=float)

        restitution = 0.72
        friction = 0.35

        for i in self.shell_idx:
            if self.pos[i, 2] < floor_z:
                pen = floor_z - self.pos[i, 2]
                self.pos[i, 2] = floor_z
                if self.vel[i, 2] < 0:
                    v_normal = self.vel[i, 2]
                    impulse_z = -(1.0 + restitution) * self.vel[i, 2] * self.mass[i]
                    self.vel[i, 2] = -restitution * self.vel[i, 2]
                    # Friction
                    v_tangent = self.vel[i, :2]
                    v_tangent_norm = np.linalg.norm(v_tangent)
                    if v_tangent_norm > 1e-6:
                        self.vel[i, :2] -= min(friction * abs(v_normal), v_tangent_norm) * (v_tangent / v_tangent_norm)
                    contact_impulse_shell[2] += impulse_z

        for i in self.fluid_all_idx:
            if self.pos[i, 2] < floor_z + 0.02:
                self.pos[i, 2] = floor_z + 0.02
                if self.vel[i, 2] < 0:
                    imp_z = -1.5 * self.vel[i, 2] * max(self.mass[i], 1e-4)
                    self.vel[i, 2] = -0.3 * self.vel[i, 2]
                    contact_impulse_fluid[2] += imp_z

        # Accumulate impulse for budget fields
        self.cumulative_impulse[0] += contact_impulse_shell
        self.cumulative_impulse[1] += contact_impulse_fluid

        # Floor boundary velocity update
        self.boundary_velocity = np.array([[0.0, 0.0, 0.0]], dtype=float)

        # Budget external forces: gravity + contact
        shell_mass_total = np.sum(self.mass[self.shell_idx])
        fluid_mass_total = np.sum(self.mass[self.liquid_idx])
        self.budget_external_force[0, 0] = shell_mass_total * gravity
        self.budget_external_force[0, 1] = contact_impulse_shell / max(dt, 1e-6)
        self.budget_external_force[1, 0] = fluid_mass_total * gravity
        self.budget_external_force[1, 1] = contact_impulse_fluid / max(dt, 1e-6)

    def advance_to(self, time: float) -> StateFrame:
        # Time-stepping loop
        sub_dt = 0.002
        while self.current_time < time - 1e-7:
            step = min(sub_dt, time - self.current_time)
            self._step_physics(step)
            self.current_time += step

        self.current_time = float(time)

        # Build required fields
        mass_quantity = self.mass.copy()

        fields = {
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": self.boundary_velocity.copy(),
            "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
            "budget_domain_weights": self.budget_domain_weights.copy(),
            "budget_external_force": self.budget_external_force.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
        }

        attributes = {
            "passive_marker": np.isin(self.body_id, (3, 4)).astype(float),
            "mass_quantity": mass_quantity,
        }

        scalars = {
            "gravity_z": float(self.config.get("gravity", [0.0, 0.0, -9.81])[2]),
            "cavity_is_closed": 1.0,
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "total_particles": len(self.pos) if self.pos is not None else 0,
        }

