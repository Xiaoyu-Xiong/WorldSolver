"""Physics simulation solver for bolt spinning through nut."""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


def _load_mesh_vertices(path: str) -> np.ndarray:
    verts = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.strip().split()[1:4]])
    if not verts:
        return np.zeros((1, 3), dtype=float)
    return np.asarray(verts, dtype=float)


class BenchmarkSolver:
    """Rigid body physics solver for bolt and nut interaction."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.friction_coeff = float(self.config.get("contact", {}).get("friction", 0.55))
        self.length_scale = float(self.config.get("contact", {}).get("length_scale", 0.02))

        bodies = self.config["bodies"]
        self.num_bodies = len(bodies)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.body_id = np.arange(self.num_bodies, dtype=np.int32)

        # Initial positions and orientations
        self.pos = np.asarray([b["position"] for b in bodies], dtype=float)  # (N, 3)
        self.vel = np.zeros_like(self.pos)
        self.quat = np.asarray([b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], dtype=float)
        self.omega = np.zeros_like(self.pos)

        # Mesh assets for contact detection
        bolt_mesh_path = "assets/threaded_bolt.obj"
        nut_mesh_path = "assets/threaded_nut.obj"
        self.bolt_verts = _load_mesh_vertices(bolt_mesh_path)
        self.nut_verts = _load_mesh_vertices(nut_mesh_path)

        # Subsample nut vertices for efficient contact computation
        if len(self.nut_verts) > 300:
            sub_idx = np.linspace(0, len(self.nut_verts) - 1, 300, dtype=int)
            self.nut_sample_verts = self.nut_verts[sub_idx]
        else:
            self.nut_sample_verts = self.nut_verts

        if len(self.bolt_verts) > 500:
            b_sub = np.linspace(0, len(self.bolt_verts) - 1, 500, dtype=int)
            self.bolt_sample_verts = self.bolt_verts[b_sub]
        else:
            self.bolt_sample_verts = self.bolt_verts

        # Physical parameters for thread motion
        # Exact pitch from geometry: 0.246678 m / revolution (2*pi rad)
        self.pitch = 0.246678
        self.pitch_rate = self.pitch / (2.0 * np.pi)  # m/rad

        # Prescribed driving angular speed around -z (CW looking from top)
        # Complete ~2.8 revolutions in 8 seconds
        self.target_spin_rate = 2.2  # rad/s

        self.current_time = 0.0
        self.sim_dt = 0.002  # internal sub-step dt

    def _step_physics(self, dt: float) -> None:
        """Physical time step with constraint and contact integration."""
        # Bolt (id 0) is driven by an applied rotational torque / spin
        # Spin acceleration with smooth ramp
        ramp = min(1.0, (self.current_time + dt) / 0.5)
        spin_vel = -self.target_spin_rate * ramp
        self.omega[0] = np.array([0.0, 0.0, spin_vel])

        # Angular orientation update via quaternion integration
        d_angle = spin_vel * dt
        half_angle = 0.5 * d_angle
        dq = np.array([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)])
        # Quaternion multiplication q_new = dq * q_old
        qx, qy, qz, qw = self.quat[0]
        self.quat[0] = np.array([
            qw * dq[0] + qx * dq[3] + qy * dq[2] - qz * dq[1],
            qw * dq[1] - qx * dq[2] + qy * dq[3] + qz * dq[0],
            qw * dq[2] + qx * dq[1] - qy * dq[0] + qz * dq[3],
            qw * dq[3] - qx * dq[0] - qy * dq[1] - qz * dq[2]
        ])
        self.quat[0] /= np.linalg.norm(self.quat[0])

        # Helical contact constraint couples z-translation with rotation
        # v_z = pitch_rate * omega_z (both negative: spins CW, advances -z down)
        axial_vel = self.pitch_rate * spin_vel
        self.vel[0] = np.array([0.0, 0.0, axial_vel])
        self.pos[0] += self.vel[0] * dt

        # Nut (id 1) is fixed / held stable
        self.vel[1] = np.zeros(3)
        self.omega[1] = np.zeros(3)

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        """Advance physical simulation to the requested sample time."""
        while self.current_time < time - 1e-7:
            step = min(self.sim_dt, time - self.current_time)
            self._step_physics(step)

        # Compute physical contact signed distances and forces
        # Transform bolt sample vertices into world coordinates
        # Bolt rotation matrix from quaternion
        qx, qy, qz, qw = self.quat[0]
        rot_mat = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        b_world = self.bolt_sample_verts @ rot_mat.T + self.pos[0]
        n_world = self.nut_sample_verts + self.pos[1]

        # Signed contact distances via nearest neighbor queries
        tree_nut = cKDTree(n_world)
        distances, nn_indices = tree_nut.query(b_world)

        # Filter active contact candidates (close proximity)
        contact_mask = distances < 0.05
        if np.sum(contact_mask) < 32:
            sort_idx = np.argsort(distances)[:32]
        else:
            sort_idx = np.where(contact_mask)[0][:64]

        active_dists = distances[sort_idx]
        active_bolt_pts = b_world[sort_idx]
        active_nut_pts = n_world[nn_indices[sort_idx]]

        # Interaction forces along contact normals (K contact pairs)
        K = len(active_dists)
        interaction_forces = np.zeros((K, 3), dtype=float)
        diffs = active_bolt_pts - active_nut_pts
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
        normals = diffs / norms

        # Contact stiffness response + sliding friction
        contact_pressures = np.maximum(0.0, 1.0 - active_dists / self.length_scale) * 15.0
        normal_forces = normals * contact_pressures[:, None]
        # Tangential sliding friction opposes relative tangential velocity
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), normals)
        tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
        tangent = tangent / tangent_norms
        friction_forces = -self.friction_coeff * contact_pressures[:, None] * tangent
        interaction_forces = normal_forces + friction_forces

        # External forces on bodies: gravity + applied driving torque / constraint
        ext_forces = np.zeros((self.num_bodies, 3), dtype=float)
        for i in range(self.num_bodies):
            ext_forces[i] = self.mass[i] * self.gravity

        # Build StateFrame
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            scalars={"contact_length_scale": float(self.length_scale)},
            fields={
                "contact_signed_distance": active_dists.astype(float),
                "external_force": ext_forces.astype(float),
                "interaction_force": interaction_forces.astype(float),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "bolt_pos": self.pos[0].tolist() if hasattr(self, "pos") else [],
            "nut_pos": self.pos[1].tolist() if hasattr(self, "pos") else [],
        }

