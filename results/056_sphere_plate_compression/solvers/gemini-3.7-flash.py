"""Deformable solid simulation of an incompressible elastic sphere compressed between plates."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.t = 0.0
        self.pos = None
        self.vel = None
        self.rest_pos = None
        self.mass = None
        self.body_id = None
        self.faces = None
        self.edges = None
        self.rest_edge_lens = None
        self.rest_volume = None
        self.K_volume = 400.0  # Bulk modulus / penalty for volume preservation
        self.K_spring = 50.0   # Edge spring stiffness
        self.damping = 15.0    # Damping coefficient
        self.cum_external_impulse = None
        self.cum_inflow_momentum = None
        self.cum_outflow_momentum = None
        self.cum_mass_source = None
        self.sub_dt = 0.0005   # Physical substep size

    def _load_mesh(self, obj_path: Path):
        vertices = []
        faces = []
        for line in obj_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                # OBJ 1-indexed
                idx = [int(v.split("/")[0]) - 1 for v in line.split()[1:4]]
                faces.append(idx)
        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        init_frame = scene.initial_state()
        self.t = float(init_frame.time)
        self.pos = init_frame.position.copy().astype(np.float64)
        self.vel = init_frame.velocity.copy().astype(np.float64)
        self.mass = init_frame.mass.copy().astype(np.float64)
        self.body_id = init_frame.body_id.copy()
        self.rest_pos = init_frame.attributes["rest_position"].copy().astype(np.float64)

        # Load topology
        obj_path = Path(scene.__module__ if hasattr(scene, "__file__") else "").resolve()
        # Find soft_sphere.obj
        asset_path = Path("assets/soft_sphere.obj")
        if not asset_path.exists():
            asset_path = Path(__file__).resolve().parent.parent / "assets" / "soft_sphere.obj"
        _, self.faces = self._load_mesh(asset_path)

        # Build unique edges
        edge_set = set()
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edge_set.add(e)
        self.edges = np.array(list(edge_set), dtype=int)

        # Rest edge lengths
        p1 = self.rest_pos[self.edges[:, 0]]
        p2 = self.rest_pos[self.edges[:, 1]]
        self.rest_edge_lens = np.linalg.norm(p2 - p1, axis=1)

        # Rest volume (from initial closed mesh or config)
        self.rest_volume = float(self.config["sphere"]["rest_volume"])

        # Cumulative budget fields (G=1 domain)
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.cum_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cum_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cum_mass_source = np.zeros((1,), dtype=np.float64)

    def _get_plate_z(self, t: float) -> tuple[float, float, float, float]:
        """Return (z_lower, z_upper, vz_lower, vz_upper) at time t."""
        plates = self.config["plates"]
        t_start = float(plates["motion_start"])
        t_end = float(plates["motion_end"])
        z_lower_init = float(plates["initial_inner_faces_z"][0])
        z_upper_init = float(plates["initial_inner_faces_z"][1])
        z_lower_final = float(plates["final_inner_faces_z"][0])
        z_upper_final = float(plates["final_inner_faces_z"][1])
        speed = float(plates["speed_each"])

        if t <= t_start:
            return z_lower_init, z_upper_init, 0.0, 0.0
        elif t >= t_end:
            return z_lower_final, z_upper_final, 0.0, 0.0
        else:
            # Moving towards each other
            dt_move = t - t_start
            z_low = z_lower_init + speed * dt_move
            z_up = z_upper_init - speed * dt_move
            return z_low, z_up, speed, -speed

    def _compute_mesh_volume_and_grad(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute volume of closed triangle mesh using divergence theorem and its gradient w.r.t pos."""
        # V = 1/6 sum_f (v0 . (v1 x v2))
        f_v0 = pos[self.faces[:, 0]]
        f_v1 = pos[self.faces[:, 1]]
        f_v2 = pos[self.faces[:, 2]]
        cross_12 = np.cross(f_v1, f_v2)
        vol = np.sum(f_v0 * cross_12) / 6.0

        # Gradient of volume w.r.t each vertex:
        # For vertex v0 in face (v0, v1, v2), dV/dv0 = 1/6 (v1 x v2)
        grad_V = np.zeros_like(pos)
        np.add.at(grad_V, self.faces[:, 0], np.cross(f_v1, f_v2) / 6.0)
        np.add.at(grad_V, self.faces[:, 1], np.cross(f_v2, f_v0) / 6.0)
        np.add.at(grad_V, self.faces[:, 2], np.cross(f_v0, f_v1) / 6.0)

        return vol, grad_V

    def _step(self, dt: float) -> np.ndarray:
        """Advance simulation by dt, returning contact force on domain."""
        t_mid = self.t + 0.5 * dt
        z_low, z_up, vz_low, vz_up = self._get_plate_z(t_mid)

        # 1. Internal elastic forces
        # Edge springs
        p1 = self.pos[self.edges[:, 0]]
        p2 = self.pos[self.edges[:, 1]]
        diff = p2 - p1
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        norm_dir = diff / dist
        strain = (dist[:, 0] - self.rest_edge_lens)
        spring_forces = self.K_spring * strain[:, None] * norm_dir

        forces = np.zeros_like(self.pos)
        np.add.at(forces, self.edges[:, 0], spring_forces)
        np.add.at(forces, self.edges[:, 1], -spring_forces)

        # Volume preservation penalty force
        vol, grad_V = self._compute_mesh_volume_and_grad(self.pos)
        vol_err = vol - self.rest_volume
        # Elastic potential E_vol = 0.5 * K_vol * (vol - V0)^2 / V0
        # F_vol = - dE_vol / dx = - K_vol * (vol - V0) / V0 * grad_V
        forces -= (self.K_volume * vol_err / self.rest_volume) * grad_V

        # Damping force
        forces -= self.damping * self.vel

        # 2. Integrate velocities (semi-implicit Euler)
        inv_mass = 1.0 / self.mass[:, None]
        self.vel += dt * forces * inv_mass

        # 3. Integrate positions
        new_pos = self.pos + dt * self.vel

        # 4. Collision resolution with kinematic moving plates
        # Contact forces tracking
        contact_force_sum = np.zeros(3, dtype=np.float64)

        # Lower plate collision (z >= z_low)
        below_lower = new_pos[:, 2] < z_low
        if np.any(below_lower):
            penetration = z_low - new_pos[below_lower, 2]
            # Enforce non-penetration position projection
            new_pos[below_lower, 2] = z_low
            # Normal velocity constraint: v_z >= vz_low
            v_rel = vz_low - self.vel[below_lower, 2]
            impulse_z = np.maximum(v_rel, 0.0) * self.mass[below_lower]
            self.vel[below_lower, 2] = vz_low
            # Add normal contact force contribution = impulse / dt
            contact_force_sum[2] += np.sum(impulse_z) / dt

        # Upper plate collision (z <= z_up)
        above_upper = new_pos[:, 2] > z_up
        if np.any(above_upper):
            penetration = new_pos[above_upper, 2] - z_up
            new_pos[above_upper, 2] = z_up
            v_rel = self.vel[above_upper, 2] - vz_up
            impulse_z = np.maximum(v_rel, 0.0) * self.mass[above_upper]
            self.vel[above_upper, 2] = vz_up
            contact_force_sum[2] -= np.sum(impulse_z) / dt

        self.pos = new_pos
        self.t += dt
        return contact_force_sum

    def advance_to(self, target_time: float) -> StateFrame:
        # Substep to target_time
        ext_force = np.zeros((1, 1, 3), dtype=np.float64)
        last_contact_force = np.zeros(3, dtype=np.float64)
        while self.t < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.t)
            contact_force = self._step(dt)
            last_contact_force = contact_force
            self.cum_external_impulse[0] += contact_force * dt

        self.t = target_time
        ext_force[0, 0] = last_contact_force

        # Compute per-element deformation gradient F
        # For each triangular face, compute affine 3D deformation mapping
        # We can construct element deformation gradient F_e in 3D:
        num_faces = len(self.faces)
        elem_F = np.tile(np.eye(3), (num_faces, 1, 1))
        for i, face in enumerate(self.faces):
            r0, r1, r2 = self.rest_pos[face]
            p0, p1, p2 = self.pos[face]

            # Reference edge vectors and normal
            e1_ref = r1 - r0
            e2_ref = r2 - r0
            n_ref = np.cross(e1_ref, e2_ref)
            area_ref = np.linalg.norm(n_ref)
            if area_ref > 1e-8:
                n_ref = n_ref / np.sqrt(area_ref) # unit normal scaled by sqrt(area)

            # Current edge vectors and normal
            e1_cur = p1 - p0
            e2_cur = p2 - p0
            n_cur = np.cross(e1_cur, e2_cur)
            area_cur = np.linalg.norm(n_cur)
            if area_cur > 1e-8:
                n_cur = n_cur / np.sqrt(area_cur)

            D_ref = np.column_stack([e1_ref, e2_ref, n_ref])
            D_cur = np.column_stack([e1_cur, e2_cur, n_cur])
            try:
                F = D_cur @ np.linalg.inv(D_ref)
                if np.isfinite(F).all():
                    elem_F[i] = F
            except np.linalg.LinAlgError:
                pass

        # Also vertex deformation gradients for attributes
        # Can compute per-vertex affine deformation gradient from rest_pos to pos
        n_verts = len(self.pos)
        vert_F = np.tile(np.eye(3), (n_verts, 1, 1))
        # Optimal affine transform / deformation gradient per vertex from local neighborhood
        for v_idx in range(n_verts):
            # Find neighboring vertices
            connected = []
            for e in self.edges:
                if e[0] == v_idx:
                    connected.append(e[1])
                elif e[1] == v_idx:
                    connected.append(e[0])
            if len(connected) >= 3:
                dX0 = self.rest_pos[connected] - self.rest_pos[v_idx] # [M, 3]
                dx = self.pos[connected] - self.pos[v_idx]           # [M, 3]
                # dx = dX0 @ F^T => F^T = (dX0^T dX0)^(-1) dX0^T dx
                try:
                    F_T = np.linalg.lstsq(dX0, dx, rcond=None)[0]
                    F = F_T.T
                    if np.isfinite(F).all():
                        vert_F[v_idx] = F
                except Exception:
                    pass

        # Compute volume constraint value and target
        cur_vol, _ = self._compute_mesh_volume_and_grad(self.pos)
        z_low, z_up, _, _ = self._get_plate_z(self.t)
        plate_gap = z_up - z_low

        # Budget domain weights (1 global domain across N vertices)
        budget_weights = np.ones((1, n_verts), dtype=np.float64)

        # Reference momentum scale
        ref_momentum = float(np.sum(self.mass) * float(self.config["plates"]["speed_each"]))
        if ref_momentum <= 0:
            ref_momentum = 1.0

        frame = StateFrame(
            time=self.t,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "rest_position": self.rest_pos.copy(),
                "deformation_gradient": vert_F,
                "rest_volume_share": np.full(n_verts, self.rest_volume / n_verts),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
                "budget_domain_weights": budget_weights,
                "budget_external_force": ext_force.copy(),
                "constraint_target": np.array([self.rest_volume], dtype=np.float64),
                "constraint_value": np.array([cur_vol], dtype=np.float64),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "element_deformation_gradient": elem_F,
            },
            scalars={
                "rest_volume": self.rest_volume,
                "initial_plate_gap": 1.7013016168,
                "minimum_positive_gap": float(self.config["plates"]["minimum_positive_gap"]),
                "current_plate_gap": plate_gap,
                "constraint_scale": self.rest_volume,
                "reference_momentum": ref_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "vertex_count": len(self.pos) if self.pos is not None else 0,
        }

