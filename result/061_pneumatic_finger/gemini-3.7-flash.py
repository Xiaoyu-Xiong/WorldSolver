"""Physics solver for pressure-driven bending of a soft finger."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config
        self.pressure_cfg = self.config["pressure"]
        self.p_target = float(self.pressure_cfg["chamber_pressure_pa"])
        self.ramp_start = float(self.pressure_cfg["ramp_start_s"])
        self.ramp_end = float(self.pressure_cfg["ramp_end_s"])
        self.hold_until = float(self.pressure_cfg["hold_until_s"])

        # Load mesh vertices
        asset = Path(__file__).parent.parent / "assets" / "pneumatic_finger.obj"
        vertices = []
        for line in asset.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
        self.rest_pos = np.asarray(vertices, dtype=float)
        self.num_nodes = len(self.rest_pos)
        self.pos = self.rest_pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.sim_time = 0.0

        # Fixed base indicator
        self.fixed_mask = (self.rest_pos[:, 0] <= 0.030)

        # Mass per node
        density = 1100.0  # kg/m^3
        total_vol = 0.21 * 0.04 * 0.014 + 6 * (0.027 * 0.048 * 0.036) + 0.02 * 0.04 * 0.030
        total_mass = density * total_vol
        node_mass = total_mass / self.num_nodes
        self.mass = np.full(self.num_nodes, node_mass, dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.inv_mass[self.fixed_mask] = 0.0

        # Define 8 box elements (sub-blocks)
        # Box 0: bottom base substrate [0..7]
        # Box 1..6: chambers 1 to 6 [8..55] (8 nodes per chamber)
        # Box 7: fingertip cap [56..63]
        self.elements = []
        for b in range(8):
            self.elements.append(np.arange(b * 8, (b + 1) * 8))

        # Build structural springs (edges of hexahedra + face diagonals + body diagonals)
        spring_set = set()
        for b in range(8):
            nodes = self.elements[b]
            for i in range(8):
                for j in range(i + 1, 8):
                    ni, nj = nodes[i], nodes[j]
                    spring_set.add((min(ni, nj), max(ni, nj)))

        # Connect chambers to the bottom substrate and to adjacent chambers
        # Box 0 (substrate) top surface nodes: 4, 5, 6, 7 (z=0.014)
        # Chamber bases are also at z=0.014
        # Connect chamber base nodes to nearest substrate nodes
        for b in range(1, 7):
            ch_base = self.elements[b][:4]  # bottom 4 nodes of chamber
            for c_node in ch_base:
                c_pos = self.rest_pos[c_node]
                # find closest nodes in box 0
                for s_node in self.elements[0]:
                    s_pos = self.rest_pos[s_node]
                    dist = np.linalg.norm(c_pos - s_pos)
                    if dist < 0.06:
                        spring_set.add((min(c_node, s_node), max(c_node, s_node)))

        # Connect adjacent chambers (chambers expand and push on each other)
        for b in range(1, 6):
            ch1_top_x = self.elements[b][[1, 2, 5, 6]]  # +x face of chamber b
            ch2_bot_x = self.elements[b + 1][[0, 3, 4, 7]]  # -x face of chamber b+1
            for n1 in ch1_top_x:
                for n2 in ch2_bot_x:
                    spring_set.add((min(n1, n2), max(n1, n2)))

        # Connect chamber 6 to tip (box 7)
        ch6_top_x = self.elements[6][[1, 2, 5, 6]]
        tip_bot_x = self.elements[7][[0, 3, 4, 7]]
        for n1 in ch6_top_x:
            for n2 in tip_bot_x:
                spring_set.add((min(n1, n2), max(n1, n2)))

        # Connect tip to substrate
        for n1 in self.elements[7]:
            for n2 in self.elements[0]:
                if np.linalg.norm(self.rest_pos[n1] - self.rest_pos[n2]) < 0.05:
                    spring_set.add((min(n1, n2), max(n1, n2)))

        self.spring_indices = np.array(list(spring_set), dtype=int)
        p1 = self.rest_pos[self.spring_indices[:, 0]]
        p2 = self.rest_pos[self.spring_indices[:, 1]]
        self.spring_rest_lens = np.linalg.norm(p1 - p2, axis=1)

        # Chamber internal volume / pressure surface normals
        # Each chamber b in 1..6 has 6 quad faces:
        # bottom (z=0.014), top (z=0.050), -x, +x, -y, +y
        self.chamber_faces = []
        for b in range(1, 7):
            idx = b * 8
            # 6 faces of hex: (quad node indices, outward normal direction in rest pose)
            # nodes 0..3: bottom z=0.014 [idx+0, idx+3, idx+2, idx+1]
            # nodes 4..7: top z=0.050    [idx+4, idx+5, idx+6, idx+7]
            # -x face: [idx+0, idx+4, idx+7, idx+3]
            # +x face: [idx+1, idx+2, idx+6, idx+5]
            # -y face: [idx+0, idx+1, idx+5, idx+4]
            # +y face: [idx+3, idx+7, idx+6, idx+2]
            faces = [
                ([idx + 0, idx + 3, idx + 2, idx + 1], np.array([0.0, 0.0, -1.0])),
                ([idx + 4, idx + 5, idx + 6, idx + 7], np.array([0.0, 0.0, 1.0])),
                ([idx + 0, idx + 4, idx + 7, idx + 3], np.array([-1.0, 0.0, 0.0])),
                ([idx + 1, idx + 2, idx + 6, idx + 5], np.array([1.0, 0.0, 0.0])),
                ([idx + 0, idx + 1, idx + 5, idx + 4], np.array([0.0, -1.0, 0.0])),
                ([idx + 3, idx + 7, idx + 6, idx + 2], np.array([0.0, 1.0, 0.0])),
            ]
            self.chamber_faces.append(faces)

        # Precompute initial element rest inverse matrices for deformation gradient
        # For each box element (8 nodes), compute least-squares deformation gradient F
        # X_i - X_c, D_m = sum (X_i - X_c)(X_i - X_c)^T
        self.elem_Dm_inv = []
        self.elem_X_rel = []
        for b in range(8):
            nodes = self.elements[b]
            X = self.rest_pos[nodes]
            Xc = np.mean(X, axis=0)
            Xrel = X - Xc
            Dm = Xrel.T @ Xrel
            self.elem_Dm_inv.append(np.linalg.inv(Dm))
            self.elem_X_rel.append(Xrel)

        self.last_ext_force = np.zeros((1, 1, 3))
        self.ref_force = 10.0  # characteristic force scale (N)

    def _get_pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return 0.0
        elif t < self.ramp_end:
            alpha = (t - self.ramp_start) / (self.ramp_end - self.ramp_start)
            # smooth hermite ramp
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            return self.p_target * smooth_alpha
        elif t <= self.hold_until:
            return self.p_target
        else:
            return self.p_target

    def _compute_forces(self, pos: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros_like(pos)

        # 1. Elastic spring forces
        i1 = self.spring_indices[:, 0]
        i2 = self.spring_indices[:, 1]
        delta = pos[i2] - pos[i1]
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        dir_vec = delta / dist
        strain = (dist[:, 0] - self.spring_rest_lens) / np.maximum(self.spring_rest_lens, 1e-6)

        # Non-linear hyperelastic-like stiffness
        k_base = 2500.0  # N/m
        spring_force_mag = k_base * (strain + 2.0 * np.sign(strain) * (strain ** 2))
        spring_force_vec = dir_vec * spring_force_mag[:, np.newaxis]

        np.add.at(forces, i1, spring_force_vec)
        np.add.at(forces, i2, -spring_force_vec)

        # 2. Chamber Pressure Forces
        # Internal chamber pressure exerts outward forces on chamber walls.
        # As pressure expands the chambers, the asymmetric wall structure
        # (thick / non-expanding bottom substrate vs thin expanding top chambers)
        # creates a bending moment, curling the finger toward +z (or into an arc).
        p_current = self._get_pressure(t)
        total_p_force = np.zeros(3)

        if p_current > 0.0:
            for chamber in self.chamber_faces:
                for face_nodes, rest_normal in chamber:
                    # Current face vertices
                    p0 = pos[face_nodes[0]]
                    p1 = pos[face_nodes[1]]
                    p2 = pos[face_nodes[2]]
                    p3 = pos[face_nodes[3]]

                    # Compute normal and area using cross product of diagonals
                    v02 = p2 - p0
                    v13 = p3 - p1
                    n_cross = np.cross(v02, v13)
                    area = 0.5 * np.linalg.norm(n_cross)
                    if area > 1e-8:
                        normal = n_cross / np.linalg.norm(n_cross)
                        # Ensure normal points outward matching rest normal
                        if np.dot(normal, rest_normal) < 0:
                            normal = -normal
                    else:
                        normal = rest_normal.copy()
                        area = 1e-6

                    # Pressure force = P * Area * normal
                    # Applied equally to the 4 nodes of the face
                    face_force = p_current * area * normal
                    node_force = face_force / 4.0

                    for fn in face_nodes:
                        forces[fn] += node_force
                    total_p_force += face_force

        # 3. Inter-chamber contact / expansion bending moment
        # When chambers expand in +x / -x, adjacent chamber walls press against each other
        # The contact between chamber tops pushes adjacent chamber sections away from each other at the top (z > 0.02),
        # while the bottom is constrained by the substrate, generating strong bending.
        for b in range(1, 6):
            # top nodes of +x face of chamber b:
            top_x_b = self.elements[b][[5, 6]]  # z=0.05 nodes
            # top nodes of -x face of chamber b+1:
            top_x_bnext = self.elements[b + 1][[4, 7]]  # z=0.05 nodes

            mid_b = np.mean(pos[top_x_b], axis=0)
            mid_next = np.mean(pos[top_x_bnext], axis=0)
            dx = mid_next[0] - mid_b[0]
            rest_gap = 0.005  # 5mm rest gap between chambers
            # Chamber expansion reduces gap; when pressurized, repulsion forces push chambers apart
            if p_current > 0:
                push = 0.08 * p_current * (1.0 + np.maximum(0.0, rest_gap - dx) * 100.0)
                forces[top_x_b, 0] -= push * 0.5
                forces[top_x_bnext, 0] += push * 0.5
                # also upward tilt tendency
                forces[top_x_bnext, 2] += push * 0.25

        # 4. Enforce fixed base boundary condition
        forces[self.fixed_mask] = 0.0

        return forces, total_p_force

    def _compute_deformation_gradients(self) -> np.ndarray:
        F_all = np.zeros((8, 3, 3))
        for b in range(8):
            nodes = self.elements[b]
            x = self.pos[nodes]
            xc = np.mean(x, axis=0)
            xrel = x - xc
            # F = (xrel^T @ Xrel) @ Dm_inv
            Dx = xrel.T @ self.elem_X_rel[b]
            F = Dx @ self.elem_Dm_inv[b]
            F_all[b] = F
        return F_all

    def advance_to(self, time: float) -> StateFrame:
        # Sub-step simulation from self.sim_time to time
        sub_dt = 0.0005  # 0.5 ms sub-steps for stability
        while self.sim_time < time - 1e-8:
            dt_step = min(sub_dt, time - self.sim_time)

            # Semi-implicit Euler / Verlet with damping
            forces, total_p_force = self._compute_forces(self.pos, self.sim_time)
            self.last_ext_force[0, 0] = total_p_force

            # Velocity update
            acc = forces * self.inv_mass[:, np.newaxis]
            self.vel += acc * dt_step

            # Damping (Rayleigh-like mass and structural damping)
            damping = np.exp(-12.0 * dt_step)
            self.vel *= damping

            # Position update
            self.pos += self.vel * dt_step

            # Hard constraint for fixed base
            self.pos[self.fixed_mask] = self.rest_pos[self.fixed_mask]
            self.vel[self.fixed_mask] = 0.0

            self.sim_time += dt_step

        self.sim_time = time
        p_now = self._get_pressure(time)
        F_elements = self._compute_deformation_gradients()

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "fixed_base": self.fixed_mask.astype(float),
                "rest_position": self.rest_pos.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": self.last_ext_force.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": F_elements,
            },
            scalars={
                "chamber_pressure_pa": float(p_now),
                "reference_force": float(self.ref_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "sim_time": self.sim_time,
            "p_target": self.p_target,
        }

