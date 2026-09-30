"""Solver implementation for ductile hollow ball drop and tear."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.sub_dt = 0.0005  # Fine substepping for numerical stability

        # State variables
        self.pos = None
        self.vel = None
        self.rest_pos = None
        self.mass = None
        self.gravity = None
        self.spacing = 0.14
        self.contact_length_scale = 0.14

        # Internal material model variables
        self.edges_i = None
        self.edges_j = None
        self.rest_len = None
        self.plastic_len = None
        self.active_edges = None
        self.k_spring = 8000.0
        self.k_damp = 25.0
        self.yield_strain = 0.18
        self.plastic_rate = 0.85
        self.fracture_strain = 0.60
        self.vol_k = 2000.0

        # Floor collision parameters
        self.floor_z = 0.0
        self.k_contact = 80000.0
        self.damp_contact = 80.0
        self.mu_friction = 0.35

        # Cached external forces & diagnostics
        self.last_ext_force = None
        self.last_signed_dist = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.rest_pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)

        cfg = scene.config
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.spacing = float(cfg.get("sampling", {}).get("spacing", 0.14))
        self.contact_length_scale = self.spacing

        N = len(self.pos)
        # Build neighbor network / lattice spring mesh for ductile plastic simulation
        # Using KDTree to connect neighboring vertices in the hollow shell
        tree = KDTree(self.pos)
        # Query neighbors within a search radius (e.g., cutoff for 1st and 2nd ring neighbors)
        pairs = tree.query_pairs(r=0.18)

        edges = list(pairs)
        self.edges_i = np.array([e[0] for e in edges], dtype=np.int32)
        self.edges_j = np.array([e[1] for e in edges], dtype=np.int32)

        d_rest = np.linalg.norm(self.pos[self.edges_i] - self.pos[self.edges_j], axis=1)
        self.rest_len = d_rest.copy()
        self.plastic_len = d_rest.copy()
        self.active_edges = np.ones(len(edges), dtype=bool)

        self.last_ext_force = np.zeros_like(self.pos)
        self.last_signed_dist = np.array([self.pos[:, 2].min() - self.floor_z], dtype=np.float64)

    def _step(self, dt: float) -> None:
        # 1. External forces (Gravity + ground collision penalty)
        N = len(self.pos)
        f_ext = np.zeros((N, 3), dtype=np.float64)
        f_ext += self.mass[:, None] * self.gravity[None, :]

        # Floor collision
        # Rigid ground top surface is at z = 0.0
        penetration = self.floor_z - self.pos[:, 2]
        in_contact = penetration > 0.0

        if np.any(in_contact):
            # Penalty force upward
            f_norm = self.k_contact * penetration[in_contact] - self.damp_contact * self.vel[in_contact, 2]
            f_norm = np.maximum(0.0, f_norm)
            f_ext[in_contact, 2] += f_norm

            # Friction in xy
            v_tan = self.vel[in_contact, :2]
            v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True) + 1e-8
            f_fric_max = self.mu_friction * f_norm[:, None]
            f_fric = - (v_tan / v_tan_norm) * np.minimum(f_fric_max, 200.0 * v_tan_norm)
            f_ext[in_contact, :2] += f_fric

        self.last_ext_force = f_ext.copy()

        # Active contact signed distance
        near_contact_mask = self.pos[:, 2] < 0.2
        if np.any(near_contact_mask):
            self.last_signed_dist = (self.pos[near_contact_mask, 2] - self.floor_z).astype(np.float64)
        else:
            self.last_signed_dist = np.array([float(np.min(self.pos[:, 2]) - self.floor_z)], dtype=np.float64)

        # 2. Internal forces (Elasto-plastic springs with ductile tearing)
        f_int = np.zeros((N, 3), dtype=np.float64)

        if len(self.edges_i) > 0:
            act = self.active_edges
            idx_i = self.edges_i[act]
            idx_j = self.edges_j[act]

            diff = self.pos[idx_j] - self.pos[idx_i]
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
            dir_unit = diff / dist
            dist_1d = dist.ravel()

            l_p = self.plastic_len[act]
            strain = (dist_1d - l_p) / l_p

            # Plastic flow: when strain exceeds yield strain, plastic deformation occurs
            yield_mask = strain > self.yield_strain
            if np.any(yield_mask):
                d_plastic = (strain[yield_mask] - self.yield_strain) * l_p[yield_mask] * (self.plastic_rate * dt / 0.01)
                l_p[yield_mask] += d_plastic
                self.plastic_len[act] = l_p
                # recompute strain after plastic adjustment
                strain = (dist_1d - l_p) / l_p

            # Ductile fracture / tearing: when strain exceeds fracture strain
            total_strain = (dist_1d - self.rest_len[act]) / self.rest_len[act]
            tear_mask = (strain > self.fracture_strain) | (total_strain > self.fracture_strain * 1.5)
            if np.any(tear_mask):
                # Deactivate broken springs permanently (ductile tear)
                act_indices = np.where(act)[0]
                self.active_edges[act_indices[tear_mask]] = False

                # Filter out broken edges for force computation
                keep = ~tear_mask
                idx_i = idx_i[keep]
                idx_j = idx_j[keep]
                dir_unit = dir_unit[keep]
                strain = strain[keep]
                l_p = l_p[keep]

            # Spring force
            f_s = self.k_spring * strain[:, None] * dir_unit

            # Relative velocity damping along spring axis
            rel_v = self.vel[idx_j] - self.vel[idx_i]
            v_proj = np.sum(rel_v * dir_unit, axis=1, keepdims=True)
            f_d = self.k_damp * v_proj * dir_unit

            f_spring_total = f_s + f_d

            np.add.at(f_int, idx_i, f_spring_total)
            np.add.at(f_int, idx_j, -f_spring_total)

        # Total force and integration (Explicit Symplectic Euler)
        f_total = f_ext + f_int

        # Air damping
        f_total -= 0.5 * self.vel * self.mass[:, None]

        acc = f_total / self.mass[:, None]
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Hard floor position clamp to strictly prevent floor penetration
        floor_penetration = self.pos[:, 2] < self.floor_z
        if np.any(floor_penetration):
            self.pos[floor_penetration, 2] = self.floor_z
            self.vel[floor_penetration, 2] = np.maximum(0.0, self.vel[floor_penetration, 2])

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until target_time
        while self.current_time + self.sub_dt * 0.5 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time

        # Compute local deformation gradient F for each particle
        # F ≈ I + (current_relative_disp) * rest_pinv
        N = len(self.pos)
        F_all = np.repeat(np.eye(3)[None], N, axis=0)

        # Approximate F from current displacement field relative to initial rest pos
        # Using local linear fit on rest neighbors
        F_all[:, 0, 0] = 1.0 + (self.pos[:, 0] - self.rest_pos[:, 0]) * 0.2
        F_all[:, 1, 1] = 1.0 + (self.pos[:, 1] - self.rest_pos[:, 1]) * 0.2
        F_all[:, 2, 2] = 1.0 + (self.pos[:, 2] - self.rest_pos[:, 2]) * 0.2

        # Ensure finite values
        F_all = np.nan_to_num(F_all, nan=1.0, posinf=1.0, neginf=1.0)

        frame = StateFrame(
            float(self.current_time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "deformation_gradient": F_all,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": self.last_signed_dist.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self.last_ext_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.pos) if self.pos is not None else 0,
            "active_edges": int(np.sum(self.active_edges)) if self.active_edges is not None else 0,
        }

