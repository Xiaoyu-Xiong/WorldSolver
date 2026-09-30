"""Physical simulation solver for 073_octocat_six_wall_compactor.

Simulates 3D elastic solid deformation under 6-wall compression and release.
Uses a volumetric/surface elastic mass-spring-volume system with kinematic wall contact,
penalty-based internal pressure / volume preservation, and restitution.
Accelerated with Numba for real-time physical stepping.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import numba
from worldsolver.contracts import StateFrame


@numba.njit(fastmath=True)
def compute_face_deformation_gradients(pos, x0, faces, Dm_inv):
    """Compute per-triangle deformation gradient F = Ds * Dm_inv."""
    num_faces = faces.shape[0]
    F = np.zeros((num_faces, 3, 3), dtype=np.float64)
    for i in range(num_faces):
        i0 = faces[i, 0]
        i1 = faces[i, 1]
        i2 = faces[i, 2]

        # Current edge vectors in 3D
        e1 = pos[i1] - pos[i0]
        e2 = pos[i2] - pos[i0]

        # Ds is 3x2 matrix [e1, e2]
        # Dm_inv is 2x2
        # F = Ds @ Dm_inv -> 3x2, embed in 3x3 with normal
        f00 = e1[0] * Dm_inv[i, 0, 0] + e2[0] * Dm_inv[i, 1, 0]
        f01 = e1[0] * Dm_inv[i, 0, 1] + e2[0] * Dm_inv[i, 1, 1]
        f10 = e1[1] * Dm_inv[i, 0, 0] + e2[1] * Dm_inv[i, 1, 0]
        f11 = e1[1] * Dm_inv[i, 0, 1] + e2[1] * Dm_inv[i, 1, 1]
        f20 = e1[2] * Dm_inv[i, 0, 0] + e2[2] * Dm_inv[i, 1, 0]
        f21 = e1[2] * Dm_inv[i, 0, 1] + e2[2] * Dm_inv[i, 1, 1]

        # Normal for 3rd column
        n_x = f10 * f21 - f20 * f11
        n_y = f20 * f01 - f00 * f21
        n_z = f00 * f11 - f10 * f01
        n_len = np.sqrt(n_x * n_x + n_y * n_y + n_z * n_z) + 1e-12

        F[i, 0, 0] = f00
        F[i, 0, 1] = f01
        F[i, 0, 2] = n_x / n_len
        F[i, 1, 0] = f10
        F[i, 1, 1] = f11
        F[i, 1, 2] = n_y / n_len
        F[i, 2, 0] = f20
        F[i, 2, 1] = f21
        F[i, 2, 2] = n_z / n_len
    return F


@numba.njit(fastmath=True)
def pbd_substep(
    pos,
    prev_pos,
    vel,
    x0,
    edges,
    rest_lengths,
    spring_k,
    vol_pairs,
    vol_rest_lens,
    vol_k,
    inv_mass,
    dt,
    damping,
    wall_h,
    shape_k,
):
    """PBD / XPBD substep for elastic solid with kinematic box boundaries."""
    num_verts = pos.shape[0]
    num_edges = edges.shape[0]
    num_vol = vol_pairs.shape[0]

    # Semi-implicit Euler integration: predict positions
    for i in range(num_verts):
        vel[i] *= (1.0 - damping * dt)
        prev_pos[i, 0] = pos[i, 0]
        prev_pos[i, 1] = pos[i, 1]
        prev_pos[i, 2] = pos[i, 2]
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt

    # Distance constraints (springs)
    for e in range(num_edges):
        i0 = edges[e, 0]
        i1 = edges[e, 1]
        dx = pos[i0, 0] - pos[i1, 0]
        dy = pos[i0, 1] - pos[i1, 1]
        dz = pos[i0, 2] - pos[i1, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        r_len = rest_lengths[e]
        C = dist - r_len
        corr = (C / dist) * spring_k * 0.5
        pos[i0, 0] -= dx * corr
        pos[i0, 1] -= dy * corr
        pos[i0, 2] -= dz * corr
        pos[i1, 0] += dx * corr
        pos[i1, 1] += dy * corr
        pos[i1, 2] += dz * corr

    # Internal volumetric strut constraints
    for v in range(num_vol):
        i0 = vol_pairs[v, 0]
        i1 = vol_pairs[v, 1]
        dx = pos[i0, 0] - pos[i1, 0]
        dy = pos[i0, 1] - pos[i1, 1]
        dz = pos[i0, 2] - pos[i1, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        r_len = vol_rest_lens[v]
        C = dist - r_len
        corr = (C / dist) * vol_k * 0.5
        pos[i0, 0] -= dx * corr
        pos[i0, 1] -= dy * corr
        pos[i0, 2] -= dz * corr
        pos[i1, 0] += dx * corr
        pos[i1, 1] += dy * corr
        pos[i1, 2] += dz * corr

    # Shape matching / rest pose restoring pull
    if shape_k > 0.0:
        for i in range(num_verts):
            pos[i, 0] += shape_k * (x0[i, 0] - pos[i, 0])
            pos[i, 1] += shape_k * (x0[i, 1] - pos[i, 1])
            pos[i, 2] += shape_k * (x0[i, 2] - pos[i, 2])

    # Wall collisions (6 moving compactor walls)
    total_contact_force = np.zeros(3, dtype=np.float64)
    for i in range(num_verts):
        for dim in range(3):
            if pos[i, dim] > wall_h:
                diff = pos[i, dim] - wall_h
                total_contact_force[dim] -= diff / (dt * dt + 1e-12) / num_verts
                pos[i, dim] = wall_h
            elif pos[i, dim] < -wall_h:
                diff = -wall_h - pos[i, dim]
                total_contact_force[dim] += diff / (dt * dt + 1e-12) / num_verts
                pos[i, dim] = -wall_h

    # Update velocities from actual position delta
    inv_dt = 1.0 / dt
    for i in range(num_verts):
        vel[i, 0] = (pos[i, 0] - prev_pos[i, 0]) * inv_dt
        vel[i, 1] = (pos[i, 1] - prev_pos[i, 1]) * inv_dt
        vel[i, 2] = (pos[i, 2] - prev_pos[i, 2]) * inv_dt

    return total_contact_force


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load mesh
        mesh_path = Path(__file__).resolve().parent.parent / "assets" / "octocat.obj"
        verts = []
        faces = []
        with open(mesh_path) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.strip().split()[1:4]])
                elif line.startswith("f "):
                    faces.append([int(x.split("/")[0]) - 1 for x in line.strip().split()[1:4]])

        self.x0 = np.ascontiguousarray(verts, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.num_verts = self.x0.shape[0]
        self.num_faces = self.faces.shape[0]

        self.pos = self.x0.copy()
        self.prev_pos = self.x0.copy()
        self.vel = np.zeros_like(self.pos)
        self.mass = np.full(self.num_verts, 1.0 / self.num_verts, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass

        # Extract unique edges
        edges_set = set()
        for f in self.faces:
            edges_set.add(tuple(sorted([f[0], f[1]])))
            edges_set.add(tuple(sorted([f[1], f[2]])))
            edges_set.add(tuple(sorted([f[2], f[0]])))
        self.edges = np.ascontiguousarray(list(edges_set), dtype=np.int64)
        e_diff = self.x0[self.edges[:, 0]] - self.x0[self.edges[:, 1]]
        self.rest_lengths = np.ascontiguousarray(np.sqrt(np.sum(e_diff ** 2, axis=1)), dtype=np.float64)

        # Precompute reference 2D coordinate system Dm_inv for each face
        self.Dm_inv = np.zeros((self.num_faces, 2, 2), dtype=np.float64)
        for i in range(self.num_faces):
            i0, i1, i2 = self.faces[i]
            e1 = self.x0[i1] - self.x0[i0]
            e2 = self.x0[i2] - self.x0[i0]
            l1 = np.linalg.norm(e1)
            u = e1 / (l1 + 1e-12)
            v = e2 - np.dot(e2, u) * u
            l2 = np.linalg.norm(v)
            v = v / (l2 + 1e-12)
            # Local 2D coordinates
            x1, y1 = l1, 0.0
            x2, y2 = float(np.dot(e2, u)), float(np.dot(e2, v))
            Dm = np.array([[x1, x2], [y1, y2]], dtype=np.float64)
            det = Dm[0, 0] * Dm[1, 1] - Dm[0, 1] * Dm[1, 0]
            if abs(det) > 1e-12:
                self.Dm_inv[i] = np.linalg.inv(Dm)
            else:
                self.Dm_inv[i] = np.eye(2)

        # Construct internal volumetric struts (cross-body springs)
        # Sample vertices inside body / tentacles to form volumetric structural network
        rng = np.random.RandomState(seed)
        sample_stride = 10
        sampled_indices = np.arange(0, self.num_verts, sample_stride)
        sampled_pts = self.x0[sampled_indices]

        # Connect nearby vertices in rest pose (volumetric tetrahedral-like network)
        vol_list = []
        for i_idx, i in enumerate(sampled_indices):
            dists = np.linalg.norm(sampled_pts - self.x0[i], axis=1)
            # Neighbors within radius [0.05, 0.4]
            neighbors = sampled_indices[(dists > 0.05) & (dists < 0.35)]
            for nb in neighbors:
                if i < nb:
                    vol_list.append((i, nb))

        self.vol_pairs = np.ascontiguousarray(vol_list, dtype=np.int64)
        v_diff = self.x0[self.vol_pairs[:, 0]] - self.x0[self.vol_pairs[:, 1]]
        self.vol_rest_lens = np.ascontiguousarray(np.sqrt(np.sum(v_diff ** 2, axis=1)), dtype=np.float64)

        # Physics simulation parameters
        self.substeps = 10
        self.spring_k = 0.85
        self.vol_k = 0.65
        self.shape_k = 0.04
        self.damping = 0.5

        # Budget accounting
        self.current_time = 0.0
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.current_external_force = np.zeros((1, 1, 3), dtype=np.float64)

        # Warm up JIT
        _ = compute_face_deformation_gradients(self.pos, self.x0, self.faces, self.Dm_inv)
        _ = pbd_substep(
            self.pos.copy(),
            self.prev_pos.copy(),
            self.vel.copy(),
            self.x0,
            self.edges,
            self.rest_lengths,
            self.spring_k,
            self.vol_pairs,
            self.vol_rest_lens,
            self.vol_k,
            self.inv_mass,
            0.002,
            self.damping,
            1.45,
            self.shape_k,
        )

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        dt_total = target_time - self.current_time

        if dt_total > 1e-9:
            num_steps = max(1, int(np.round(dt_total / (self.config["sample_dt"] / self.substeps))))
            sub_dt = dt_total / num_steps

            step_force = np.zeros(3, dtype=np.float64)
            for s in range(num_steps):
                t_sub = self.current_time + (s + 0.5) * sub_dt
                wall_h = self.scene.inner_half_extent(t_sub)
                f_contact = pbd_substep(
                    self.pos,
                    self.prev_pos,
                    self.vel,
                    self.x0,
                    self.edges,
                    self.rest_lengths,
                    self.spring_k,
                    self.vol_pairs,
                    self.vol_rest_lens,
                    self.vol_k,
                    self.inv_mass,
                    sub_dt,
                    self.damping,
                    wall_h,
                    self.shape_k,
                )
                step_force += f_contact
                self.cum_external_impulse[0] += f_contact * sub_dt

            self.current_external_force[0, 0] = step_force / num_steps
            self.current_time = target_time

        # Compute element deformation gradients
        F = compute_face_deformation_gradients(self.pos, self.x0, self.faces, self.Dm_inv)

        # Build StateFrame
        attributes = {
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.ones((1, self.num_verts), dtype=np.float64),
            "budget_external_force": self.current_external_force.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": F,
        }
        scalars = {
            "reference_momentum": 1.0,
        }

        frame = StateFrame(
            time=target_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_verts": self.num_verts,
            "num_faces": self.num_faces,
            "num_edges": len(self.edges),
            "num_vol_struts": len(self.vol_pairs),
        }

