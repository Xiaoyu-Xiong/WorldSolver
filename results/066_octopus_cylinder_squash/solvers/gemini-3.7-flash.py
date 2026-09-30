"""Physical solver for Figure 11 octopus cylinder squash experiment."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from worldsolver.contracts import StateFrame


def _load_obj_faces(path: Path) -> list[tuple[int, int, int]]:
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("f "):
            parts = line.split()[1:4]
            # 1-indexed to 0-indexed
            indices = [int(p.split("/")[0]) - 1 for p in parts]
            faces.append(tuple(indices))
    return faces


class BenchmarkSolver:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.rest_pos = np.array(init_frame.position, dtype=np.float64)  # (N, 3)
        self.pos = self.rest_pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.num_nodes = len(self.pos)
        self.mass = np.array(init_frame.mass, dtype=np.float64)  # (N,)
        self.total_mass = float(np.sum(self.mass))
        self.body_id = np.array(init_frame.body_id, dtype=np.int32)

        # Load topology
        obj_path = Path(__file__).parent.parent / "assets" / "octopus.obj"
        self.faces = _load_obj_faces(obj_path)
        self.num_elements = len(self.faces)

        # Extract all edges from faces and shape matching / spring network
        edges_set = set()
        for f in self.faces:
            edges_set.add((min(f[0], f[1]), max(f[0], f[1])))
            edges_set.add((min(f[1], f[2]), max(f[1], f[2])))
            edges_set.add((min(f[2], f[0]), max(f[2], f[0])))

        # Also add internal cross-springs across mantle and arm bases to prevent collapse
        # Mantle core is vertex 5 (0,0,2.35) and top is vertex 0 (0,0,3.25)
        # Add springs from vertex 0 and 5 to base nodes 1..10
        for i in range(1, 10):
            edges_set.add((min(0, i), max(0, i)))
            edges_set.add((min(5, i), max(5, i)))

        # Add diagonal / opposite springs across mantle
        edges_set.add((1, 3))
        edges_set.add((2, 4))
        edges_set.add((6, 8))
        edges_set.add((7, 9))

        self.edges = list(edges_set)
        self.edge_rest_lengths = np.array([
            np.linalg.norm(self.rest_pos[i] - self.rest_pos[j])
            for i, j in self.edges
        ], dtype=np.float64)

        # Precompute material coordinates for 2D/3D deformation gradient per triangle element
        # For each triangle face (i, j, k), define 2 rest edge vectors e1 = X_j - X_i, e2 = X_k - X_i
        # normal e3 = (e1 x e2) / ||e1 x e2||
        # D_m = [e1, e2, e3] -> D_m_inv
        self.Dm_inv = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for e_idx, f in enumerate(self.faces):
            i, j, k = f
            e1 = self.rest_pos[j] - self.rest_pos[i]
            e2 = self.rest_pos[k] - self.rest_pos[i]
            e3 = np.cross(e1, e2)
            norm_e3 = np.linalg.norm(e3)
            if norm_e3 > 1e-12:
                e3 = e3 / norm_e3
            else:
                e3 = np.array([0.0, 0.0, 1.0])
            Dm = np.column_stack([e1, e2, e3])
            try:
                self.Dm_inv[e_idx] = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                self.Dm_inv[e_idx] = np.eye(3)

        # Simulation parameters
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.sim_time = 0.0
        self.sub_dt = 0.0005  # 0.5 ms for stable integration

        # Elastic parameters (Stable Neo-Hookean style elastic body)
        self.k_spring = 2800.0  # N/m spring stiffness
        self.damping = 2.5      # Damping coefficient
        self.k_shape = 250.0    # Soft global/local shape recovery stiffness

        # Cylinders:
        # 3 static cylinders: [x, z, r]
        self.static_cylinders = [
            {"center_x": -0.72, "center_z": 0.72, "radius": 0.58},
            {"center_x": 0.72, "center_z": 0.72, "radius": 0.58},
            {"center_x": 0.0, "center_z": 1.72, "radius": 0.58},
        ]
        self.upper_cyl_radius = 0.62
        self.upper_keyframes = scene.config.get("upper_cylinder_motion", {}).get("keyframes", [])

        # Budget tracking
        self.cum_external_impulse = np.zeros(3, dtype=np.float64)
        self.cum_inflow_momentum = np.zeros(3, dtype=np.float64)
        self.cum_outflow_momentum = np.zeros(3, dtype=np.float64)
        self.last_ext_force = np.zeros(3, dtype=np.float64)
        self.reference_momentum = self.total_mass * 9.81 * 1.0  # Positive momentum scale

        self.initialized = True

    def _get_upper_cylinder_z(self, t: float) -> float:
        if not self.upper_keyframes:
            return 4.20
        if t <= self.upper_keyframes[0]["time"]:
            return float(self.upper_keyframes[0]["translation"][2])
        if t >= self.upper_keyframes[-1]["time"]:
            return float(self.upper_keyframes[-1]["translation"][2])
        for i in range(len(self.upper_keyframes) - 1):
            t0 = self.upper_keyframes[i]["time"]
            t1 = self.upper_keyframes[i + 1]["time"]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                z0 = self.upper_keyframes[i]["translation"][2]
                z1 = self.upper_keyframes[i + 1]["translation"][2]
                return float((1.0 - alpha) * z0 + alpha * z1)
        return 4.20

    def _step(self, dt: float) -> None:
        # Forces accumulation
        forces = np.zeros_like(self.pos)

        # 1. Gravity
        for i in range(self.num_nodes):
            forces[i] += self.mass[i] * self.gravity

        # 2. Elastic spring forces
        for (i, j), L0 in zip(self.edges, self.edge_rest_lengths):
            diff = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                dir_vec = diff / dist
                # Spring tension/compression
                stretch = dist - L0
                f_mag = self.k_spring * stretch
                # Relative velocity damping along spring
                v_rel = self.vel[j] - self.vel[i]
                damp_force = 15.0 * np.dot(v_rel, dir_vec)
                f_total = (f_mag + damp_force) * dir_vec
                forces[i] += f_total
                forces[j] -= f_total

        # 3. Shape matching / Neo-Hookean restoration force toward optimal rigid/soft rest pose
        # Optimal translation matching
        com_cur = np.mean(self.pos, axis=0)
        com_rest = np.mean(self.rest_pos, axis=0)
        p_cur = self.pos - com_cur
        p_rest = self.rest_pos - com_rest

        # Optimal rotation R via SVD
        H = p_rest.T @ p_cur
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Goal positions in current orientation
        goal_pos = (self.rest_pos - com_rest) @ R.T + com_cur
        forces += self.k_shape * (goal_pos - self.pos)

        # 4. Overall velocity damping
        forces -= self.damping * self.vel

        # Semi-implicit Euler integration
        acc = forces / self.mass[:, None]
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Collision Handling
        upper_z = self._get_upper_cylinder_z(self.sim_time + dt)
        contact_force_total = np.zeros(3, dtype=np.float64)

        # (a) Static cylinders along Y axis
        for cyl in self.static_cylinders:
            cx = cyl["center_x"]
            cz = cyl["center_z"]
            r = cyl["radius"]

            for i in range(self.num_nodes):
                px = self.pos[i, 0]
                pz = self.pos[i, 2]
                dx = px - cx
                dz = pz - cz
                d_xz = np.sqrt(dx * dx + dz * dz)

                if d_xz < r:
                    # Inside cylinder: project out
                    if d_xz > 1e-12:
                        nx = dx / d_xz
                        nz = dz / d_xz
                    else:
                        nx = 0.0
                        nz = 1.0

                    overlap = r - d_xz
                    self.pos[i, 0] = cx + nx * r
                    self.pos[i, 2] = cz + nz * r

                    # Normal impulse to cancel inward velocity
                    normal = np.array([nx, 0.0, nz])
                    vn = np.dot(self.vel[i], normal)
                    if vn < 0:
                        normal_impulse = -vn * normal
                        self.vel[i] += normal_impulse
                        contact_force_total += self.mass[i] * normal_impulse / dt
                        # Tangential friction
                        vt = self.vel[i] - np.dot(self.vel[i], normal) * normal
                        self.vel[i] -= 0.3 * vt

        # (b) Kinematic upper cylinder along Y axis: (cx=0.0, cz=upper_z, r=0.62)
        u_cx = 0.0
        u_cz = upper_z
        u_r = self.upper_cyl_radius

        for i in range(self.num_nodes):
            px = self.pos[i, 0]
            pz = self.pos[i, 2]
            dx = px - u_cx
            dz = pz - u_cz
            d_xz = np.sqrt(dx * dx + dz * dz)

            if d_xz < u_r:
                if d_xz > 1e-12:
                    nx = dx / d_xz
                    nz = dz / d_xz
                else:
                    nx = 0.0
                    nz = -1.0

                self.pos[i, 0] = u_cx + nx * u_r
                self.pos[i, 2] = u_cz + nz * u_r

                normal = np.array([nx, 0.0, nz])
                vn = np.dot(self.vel[i], normal)
                if vn < 0:
                    normal_impulse = -vn * normal
                    self.vel[i] += normal_impulse
                    contact_force_total += self.mass[i] * normal_impulse / dt
                    vt = self.vel[i] - np.dot(self.vel[i], normal) * normal
                    self.vel[i] -= 0.3 * vt

        # Record external forces and impulses
        ext_force = self.total_mass * self.gravity + contact_force_total
        self.last_ext_force = ext_force
        self.cum_external_impulse += ext_force * dt
        self.sim_time += dt

    def advance_to(self, time: float) -> StateFrame:
        if not self.initialized:
            raise RuntimeError("Solver not initialized")

        # Step simulation until requested time
        while self.sim_time < time - 1e-9:
            dt = min(self.sub_dt, time - self.sim_time)
            self._step(dt)

        # Compute deformation gradient F per element
        F_elements = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for e_idx, f in enumerate(self.faces):
            i, j, k = f
            e1 = self.pos[j] - self.pos[i]
            e2 = self.pos[k] - self.pos[i]
            e3 = np.cross(e1, e2)
            norm_e3 = np.linalg.norm(e3)
            if norm_e3 > 1e-12:
                e3 = e3 / norm_e3
            else:
                e3 = np.array([0.0, 0.0, 1.0])
            Ds = np.column_stack([e1, e2, e3])
            F_elements[e_idx] = Ds @ self.Dm_inv[e_idx]

        # Domain weights: G=1 global domain, N particles -> [1, N]
        domain_weights = np.ones((1, self.num_nodes), dtype=np.float64)

        # Budget external forces: [G, K, 3] -> [1, 1, 3]
        budget_ext_force = self.last_ext_force.reshape(1, 1, 3)

        # Cumulative external impulse: [G, 3] -> [1, 3]
        budget_ext_impulse = self.cum_external_impulse.reshape(1, 3)

        # Inflow / Outflow momentum: [G, 3] -> [1, 3]
        inflow_mom = self.cum_inflow_momentum.reshape(1, 3)
        outflow_mom = self.cum_outflow_momentum.reshape(1, 3)

        # Mass source: [K] -> [1]
        mass_source = np.zeros(1, dtype=np.float64)

        # Attributes: mass_quantity [N]
        mass_quantity = self.mass.copy()

        # Build StateFrame
        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "rest_position": self.rest_pos.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": budget_ext_impulse,
                "budget_cumulative_inflow_momentum": inflow_mom,
                "budget_cumulative_outflow_momentum": outflow_mom,
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": mass_source,
                "element_deformation_gradient": F_elements,
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": float(self.gravity[2]),
                "upper_cylinder_z": float(self._get_upper_cylinder_z(time)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "sim_time": self.sim_time,
        }

