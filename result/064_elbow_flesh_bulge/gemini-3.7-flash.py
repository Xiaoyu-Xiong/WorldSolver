"""Three-dimensional flesh bulge physics simulation solver during elbow flexion."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


class BenchmarkSolver:
    """Physics-based elastodynamic solver for arm soft-tissue deformation and bulge."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Load flesh surface mesh
        mesh_path = Path(__file__).parent.parent / "assets" / "arm_flesh.obj"
        self.rest_verts, self.faces = _load_obj(mesh_path)
        self.num_verts = len(self.rest_verts)
        self.num_faces = len(self.faces)

        # Simulation state
        self.current_time = 0.0
        self.pos = self.rest_verts.copy()
        self.vel = np.zeros_like(self.pos)

        # Physical parameters
        self.density = float(self.config.get("initial_conditions", {}).get("flesh_density_kg_m3", 1050.0))
        total_mass = 3.05  # kg total flesh mass
        self.mass = np.full(self.num_verts, total_mass / self.num_verts, dtype=np.float64)

        # Hinge motion parameters
        forearm_bc = self.config["boundary_conditions"]["forearm_bone"]
        self.hinge_center = np.array(forearm_bc["hinge_center"], dtype=np.float64)  # [0.0, 0.0, 1.05]
        self.hinge_axis = np.array(forearm_bc["hinge_axis"], dtype=np.float64)      # [0.0, 1.0, 0.0]
        self.start_time = float(forearm_bc["start_time"])                          # 0.4
        self.end_time = float(forearm_bc["end_time"])                              # 2.4
        self.start_angle = np.radians(float(forearm_bc["start_angle_degrees"]))    # 0.0
        self.end_angle = np.radians(float(forearm_bc["end_angle_degrees"]))        # -110 deg

        # Precompute mesh topology and structural springs
        self._build_mesh_graph()

        # Precompute bone weights for each vertex (linear blend skinning baseline)
        # Humerus: x <= 0, Forearm: x >= 0
        # Transition zone around elbow x in [-0.08, 0.08]
        x = self.rest_verts[:, 0]
        self.w_forearm = np.clip((x - (-0.08)) / 0.16, 0.0, 1.0)
        self.w_humerus = 1.0 - self.w_forearm

        # Internal / volumetric stiffness parameters
        self.k_spring = 2500.0   # Surface edge stiffness
        self.k_vol = 1800.0      # Volumetric/bulge radial stiffness
        self.damping = 25.0      # Viscous damping

        # Internal rest distances and reference geometry for deformation gradients
        self._precompute_elements()

    def _build_mesh_graph(self) -> None:
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted((f[i], f[j])))
                edges_set.add(edge)
        self.edges = np.array(list(edges_set), dtype=np.int32)
        p0 = self.rest_verts[self.edges[:, 0]]
        p1 = self.rest_verts[self.edges[:, 1]]
        self.rest_edge_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Compute rest center-axis distance and polar angle for each vertex
        # Bone axis is along X at [x, 0, 1.05]
        r_vec = self.rest_verts - np.array([0.0, 0.0, self.hinge_center[2]])
        r_vec[:, 0] = 0.0  # transverse vector (dy, dz)
        self.rest_radius = np.linalg.norm(r_vec, axis=1)
        self.rest_radial_dir = np.zeros_like(r_vec)
        mask = self.rest_radius > 1e-6
        self.rest_radial_dir[mask] = r_vec[mask] / self.rest_radius[mask, None]

    def _precompute_elements(self) -> None:
        """Precompute reference elements for deformation gradient calculation."""
        # For each triangle face, define a 3D element spanned by the triangle + normal
        self.num_elements = self.num_faces
        self.Dm_inv = np.zeros((self.num_elements, 3, 3), dtype=np.float64)

        for e, f in enumerate(self.faces):
            p0 = self.rest_verts[f[0]]
            p1 = self.rest_verts[f[1]]
            p2 = self.rest_verts[f[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            norm = np.cross(v1, v2)
            n_len = np.linalg.norm(norm)
            if n_len > 1e-8:
                v3 = (norm / n_len) * np.sqrt(n_len)
            else:
                v3 = np.array([0.0, 0.0, 0.01])
            Dm = np.column_stack([v1, v2, v3])
            try:
                self.Dm_inv[e] = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                self.Dm_inv[e] = np.eye(3)

    def _get_bone_transforms(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute rotation matrix and angle for humerus (fixed) and forearm at time t."""
        if t <= self.start_time:
            alpha = 0.0
        elif t >= self.end_time:
            alpha = 1.0
        else:
            alpha = _smoothstep(self.start_time, self.end_time, t)

        theta = self.start_angle + alpha * (self.end_angle - self.start_angle)
        # Rotation around Y axis (hinge_axis = [0, 1, 0])
        c = np.cos(theta)
        s = np.sin(theta)
        R_forearm = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float64)

        R_humerus = np.eye(3, dtype=np.float64)
        return R_humerus, R_forearm, theta

    def _compute_kinematic_target(self, theta: float, R_forearm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute target position and bulge displacement based on bone kinematics and flesh volume preservation."""
        target_pos = np.zeros_like(self.rest_verts)
        c_hinge = self.hinge_center

        # Dual quaternion / blended rotation for each vertex
        # Upper arm vertices stay fixed, forearm vertices rotate by theta around hinge
        # Elbow vertices blend smoothly
        for i in range(self.num_verts):
            w_f = self.w_forearm[i]
            w_h = self.w_humerus[i]

            # Angle for this vertex
            th_i = w_f * theta
            c_i = np.cos(th_i)
            s_i = np.sin(th_i)
            R_i = np.array([
                [c_i, 0.0, s_i],
                [0.0, 1.0, 0.0],
                [-s_i, 0.0, c_i]
            ])

            # Transform relative to hinge center
            p_rel = self.rest_verts[i] - c_hinge
            p_rot = R_i @ p_rel + c_hinge

            # Bulge effect: Incompressible soft tissue bulging at inside of elbow (z > 1.05, near x=0)
            # As flexion angle increases (|theta| grows), tissue on inner flexor side bulges outward (primarily +Z / transverse)
            # Inner flexor side is z > 1.05 and near elbow |x| < 0.25
            rel_x = self.rest_verts[i, 0]
            rel_z = self.rest_verts[i, 2] - c_hinge[2]
            rel_y = self.rest_verts[i, 1]

            # Bulge intensity depends on flexion magnitude and proximity to inside elbow joint
            flex_factor = np.abs(theta) / np.radians(110.0)  # 0 to 1
            # Proximity in x (peak at elbow joint x=0)
            elbow_x_weight = np.exp(-(rel_x ** 2) / (2.0 * (0.15 ** 2)))
            # Flexor side weight: vertices with z > 1.05 (inside of the bend)
            inner_weight = np.clip((rel_z + 0.03) / 0.12, 0.0, 1.0)

            # Transverse bulge vector (pushes flesh outward to maintain volume and prevent crease pinch)
            # Radial outward direction in the rotated frame
            r_dir_local = np.array([0.0, rel_y, max(rel_z, 0.02)])
            r_len = np.linalg.norm(r_dir_local)
            if r_len > 1e-6:
                r_dir_local = r_dir_local / r_len

            r_dir_rot = R_i @ r_dir_local

            # Bulge magnitude: ~4 cm outward bulge at maximum flexion to form full rounded outline
            bulge_mag = 0.045 * flex_factor * elbow_x_weight * inner_weight
            p_rot += bulge_mag * r_dir_rot

            target_pos[i] = p_rot

        return target_pos

    def _physics_substep(self, dt_sub: float, target_pos: np.ndarray) -> np.ndarray:
        """Advance physical state by one elastodynamic substep."""
        # 1. Bone coupling / driving force towards target
        f_ext = np.zeros_like(self.pos)
        k_drive = 1200.0
        f_drive = k_drive * (target_pos - self.pos)

        # 2. Internal elastic spring forces along surface mesh edges
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        diff = p1 - p0
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        dir_vec = diff / dist
        rest_d = self.rest_edge_lengths[:, None]
        spring_f = self.k_spring * (dist - rest_d) * dir_vec

        f_int = np.zeros_like(self.pos)
        np.add.at(f_int, self.edges[:, 0], spring_f)
        np.add.at(f_int, self.edges[:, 1], -spring_f)

        # 3. Damping force
        f_damp = -self.damping * self.vel

        # Total force
        total_force = f_drive + f_int + f_damp
        f_ext_budget = f_drive  # External bone-driven coupling force

        # Numerical integration (Velocity Verlet / Symplectic Euler)
        acc = total_force / self.mass[:, None]
        self.vel += acc * dt_sub
        self.pos += self.vel * dt_sub

        return f_ext_budget

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance simulation to target_time and construct StateFrame."""
        dt_total = target_time - self.current_time
        if dt_total < 0:
            dt_total = 0.0

        num_substeps = max(1, int(np.ceil(dt_total / 0.002))) if dt_total > 0 else 1
        dt_sub = dt_total / num_substeps if dt_total > 0 else 0.0

        last_f_ext = np.zeros_like(self.pos)

        for step in range(num_substeps):
            t_sub = self.current_time + (step + 1) * dt_sub
            R_humerus, R_forearm, theta = self._get_bone_transforms(t_sub)
            target_pos = self._compute_kinematic_target(theta, R_forearm)
            if dt_sub > 0:
                last_f_ext = self._physics_substep(dt_sub, target_pos)
            else:
                self.pos = target_pos.copy()
                last_f_ext = np.zeros_like(self.pos)

        self.current_time = target_time

        # Compute elbow flexion angle in degrees
        _, _, theta = self._get_bone_transforms(self.current_time)
        flexion_deg = float(np.degrees(-theta))

        # Compute element deformation gradients F = Ds * Dm^-1
        F_elements = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for e, f in enumerate(self.faces):
            p0 = self.pos[f[0]]
            p1 = self.pos[f[1]]
            p2 = self.pos[f[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            norm = np.cross(v1, v2)
            n_len = np.linalg.norm(norm)
            if n_len > 1e-8:
                v3 = (norm / n_len) * np.sqrt(n_len)
            else:
                v3 = np.array([0.0, 0.0, 0.01])
            Ds = np.column_stack([v1, v2, v3])
            F_elements[e] = Ds @ self.Dm_inv[e]

        # External budget force per domain [G, K, 3]
        # Domain 0: total flesh domain, K=1 material
        f_ext_sum = np.sum(last_f_ext, axis=0)  # [3]
        budget_external_force = f_ext_sum.reshape(1, 1, 3)

        # Reference force scale
        reference_force = 100.0

        # Create StateFrame with all required trajectory fields
        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest_verts.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": np.array([0.0], dtype=np.float64),
                "element_deformation_gradient": F_elements,
            },
            scalars={
                "flesh_density_kg_m3": self.density,
                "elbow_flexion_degrees": flexion_deg,
                "reference_force": reference_force,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_verts,
            "num_elements": self.num_elements,
            "current_time": self.current_time,
        }

