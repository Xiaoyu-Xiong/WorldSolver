"""Cloth stack solver implementation.

Simulates 10 cloth panels falling onto a board, colliding with the board and
with each other, and settling into a frictional multilayer stack.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load topology and initial state
        root = Path(__file__).resolve().parent.parent / "assets"
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.init_data = json.loads((root / "initial_state.json").read_text(encoding="utf-8"))

        self.faces = np.array(self.meta["topology"]["faces"], dtype=int)

        # Initial positions, velocities, mass, body_id
        self.pos = np.array(self.init_data["position"], dtype=float)  # (1690, 3)
        self.vel = np.array(self.init_data["velocity"], dtype=float)  # (1690, 3)
        self.mass = np.array(self.init_data["mass"], dtype=float)      # (1690,)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)
        self.body_id = np.array(self.init_data["body_id"], dtype=int)

        # Attributes
        self.mass_quantity = np.array(self.init_data["attributes"]["mass_quantity"], dtype=float)
        self.uv = np.array(self.init_data["attributes"]["uv"], dtype=float)

        # Scalars
        self.contact_length_scale = float(self.init_data["scalars"]["contact_length_scale"])
        self.constraint_scale = float(self.init_data["scalars"]["constraint_scale"])

        # Physical parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.friction = float(self.config.get("friction", 0.1))
        self.board_top = float(self.config.get("board_top", 0.05))
        self.thickness = float(self.config.get("panel", {}).get("thickness", 0.005))
        self.panel_count = int(self.config.get("panel_count", 10))
        self.nodes_per_panel = 169

        # Build structural, shear, and bending springs for each panel
        self._build_springs()

        # Simulation sub-stepping parameters
        self.time = 0.0
        self.sub_dt = 0.0005  # 0.5 ms for XPBD / Verlet cloth simulation

    def _build_springs(self):
        """Construct distance springs (structural, shear, bending) for each panel mesh."""
        grid_h, grid_w = 13, 13
        spring_list = []
        rest_list = []
        stiff_list = []

        # Build standard grid springs for a single panel
        edges_1 = []  # structural (dist = 1)
        edges_2 = []  # shear / diagonal (dist = sqrt(2))
        edges_3 = []  # bending (dist = 2)

        for r in range(grid_h):
            for c in range(grid_w):
                idx = r * grid_w + c
                # Right
                if c + 1 < grid_w:
                    edges_1.append((idx, idx + 1))
                # Down
                if r + 1 < grid_h:
                    edges_1.append((idx, idx + grid_w))
                # Diagonals
                if r + 1 < grid_h and c + 1 < grid_w:
                    edges_2.append((idx, idx + grid_w + 1))
                if r + 1 < grid_h and c - 1 >= 0:
                    edges_2.append((idx, idx + grid_w - 1))
                # Bending
                if c + 2 < grid_w:
                    edges_3.append((idx, idx + 2))
                if r + 2 < grid_h:
                    edges_3.append((idx, idx + 2 * grid_w))

        all_edges = []
        # Replicate for all panels
        for b in range(self.panel_count):
            offset = b * self.nodes_per_panel
            for i, j in edges_1:
                all_edges.append((i + offset, j + offset, 0.8)) # high stiffness
            for i, j in edges_2:
                all_edges.append((i + offset, j + offset, 0.6))
            for i, j in edges_3:
                all_edges.append((i + offset, j + offset, 0.3))

        self.spring_indices = np.array([[e[0], e[1]] for e in all_edges], dtype=int)
        self.spring_stiffness = np.array([e[2] for e in all_edges], dtype=float)

        # Compute rest lengths from initial positions
        p1 = self.pos[self.spring_indices[:, 0]]
        p2 = self.pos[self.spring_indices[:, 1]]
        self.spring_rest_lengths = np.linalg.norm(p1 - p2, axis=1)

    def _step(self, dt: float):
        """Advance physics by one sub-step dt using Position Based Dynamics (PBD)."""
        # 1. External forces & predict positions
        # Semi-implicit Euler prediction
        self.vel += self.gravity * dt
        # Damping
        self.vel *= (1.0 - 0.2 * dt)

        prev_pos = self.pos.copy()
        pred_pos = self.pos + self.vel * dt

        # 2. Iterative constraint projection
        pbd_iters = 8
        half_thick = 0.5 * self.thickness
        contact_margin = self.thickness

        p1_idx = self.spring_indices[:, 0]
        p2_idx = self.spring_indices[:, 1]
        w1 = self.inv_mass[p1_idx]
        w2 = self.inv_mass[p2_idx]
        w_sum = w1 + w2
        valid_springs = w_sum > 0

        for _ in range(pbd_iters):
            # Cloth internal distance constraints
            delta = pred_pos[p1_idx] - pred_pos[p2_idx]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-7)

            # Constraint: |p1 - p2| - rest_length = 0
            err = dist - self.spring_rest_lengths[:, None]
            corr = (err / dist_safe) * delta

            # Weighted distribution
            factor = (self.spring_stiffness[:, None] / w_sum[:, None]) * valid_springs[:, None]
            dp1 = -w1[:, None] * factor * corr
            dp2 = w2[:, None] * factor * corr

            # Accumulate corrections
            np.add.at(pred_pos, p1_idx, dp1)
            np.add.at(pred_pos, p2_idx, dp2)

            # Board collision (plane at z = board_top + half_thick)
            floor_z = self.board_top + half_thick
            penetration = floor_z - pred_pos[:, 2]
            penetrating = penetration > 0
            if np.any(penetrating):
                pred_pos[penetrating, 2] = floor_z

            # Inter-panel layer collision / thickness preservation
            # Panels 0 to 9 are stacked vertically in initial state.
            # Fast spatial layer sorting / repulsive contact
            for b in range(self.panel_count - 1):
                idx_lower = np.arange(b * self.nodes_per_panel, (b + 1) * self.nodes_per_panel)
                idx_upper = np.arange((b + 1) * self.nodes_per_panel, (b + 2) * self.nodes_per_panel)

                # Corresponding or near nodes between adjacent panels
                z_lower = pred_pos[idx_lower, 2]
                z_upper = pred_pos[idx_upper, 2]

                # Required vertical clearance is thickness
                gap = z_upper - z_lower
                violation = contact_margin - gap
                mask = violation > 0
                if np.any(mask):
                    corr_z = 0.5 * violation[mask]
                    pred_pos[idx_lower[mask], 2] -= corr_z
                    pred_pos[idx_upper[mask], 2] += corr_z

                    # Also apply floor check again for lower panel
                    under_floor = pred_pos[idx_lower[mask], 2] < floor_z
                    if np.any(under_floor):
                        pred_pos[idx_lower[mask][under_floor], 2] = floor_z

        # Board boundary box lateral containment
        board_half_ext = 0.9
        pred_pos[:, 0] = np.clip(pred_pos[:, 0], -board_half_ext, board_half_ext)
        pred_pos[:, 1] = np.clip(pred_pos[:, 1], -board_half_ext, board_half_ext)

        # 3. Update velocity from position delta
        self.vel = (pred_pos - prev_pos) / dt

        # Apply friction where contact is active (near floor or touching layers)
        floor_contacts = pred_pos[:, 2] <= (self.board_top + half_thick + 1e-4)
        if np.any(floor_contacts):
            # Normal velocity zeroed, tangential velocity damped by friction
            self.vel[floor_contacts, 2] = np.maximum(0.0, self.vel[floor_contacts, 2])
            fric_factor = max(0.0, 1.0 - self.friction * 50.0 * dt)
            self.vel[floor_contacts, 0] *= fric_factor
            self.vel[floor_contacts, 1] *= fric_factor

        self.pos = pred_pos
        self.time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance physics to target_time and return StateFrame."""
        while self.time + self.sub_dt <= target_time + 1e-9:
            self._step(self.sub_dt)

        # If remaining fractional step is needed
        rem = target_time - self.time
        if rem > 1e-6:
            self._step(rem)

        return self._make_state_frame(target_time)

    def _make_state_frame(self, time: float) -> StateFrame:
        """Construct the StateFrame for current state at `time`."""
        # Contact signed distance: distance to board floor minus contact thickness margin
        floor_z = self.board_top + 0.5 * self.thickness
        contact_sd = self.pos[:, 2] - floor_z

        # Constraint error / value
        # Spring length deviation mean or total constraint residual
        p1 = self.pos[self.spring_indices[:, 0]]
        p2 = self.pos[self.spring_indices[:, 1]]
        spring_lens = np.linalg.norm(p1 - p2, axis=1)
        mean_stretch_err = float(np.mean(np.abs(spring_lens - self.spring_rest_lengths)))

        attributes = {
            "uv": self.uv.copy(),
            "mass_quantity": self.mass_quantity.copy(),
        }

        fields = {
            "contact_signed_distance": np.asarray(contact_sd, dtype=float),
            "constraint_value": np.array([mean_stretch_err], dtype=float),
            "constraint_target": np.array([0.0], dtype=float),
        }

        scalars = {
            "contact_length_scale": float(self.contact_length_scale),
            "constraint_scale": float(self.constraint_scale),
        }

        frame = StateFrame(
            time=float(time),
            position=np.asarray(self.pos, dtype=float),
            velocity=np.asarray(self.vel, dtype=float),
            mass=np.asarray(self.mass, dtype=float),
            body_id=np.asarray(self.body_id, dtype=int),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "sim_time": self.time,
            "node_count": len(self.pos),
            "panel_count": self.panel_count,
        }

