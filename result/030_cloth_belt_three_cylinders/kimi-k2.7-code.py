"""BenchmarkSolver for 030_cloth_belt_three_cylinders.

A lightweight mass-spring / PBD cloth solver that runs without Genesis.
The belt is represented by the frozen closed-loop mesh and interacts with
three prescribed kinematic cylinders.  The solver uses a stable sub-stepped
position-based dynamics loop with edge-length constraints, hard radial
contact projection, and velocity-level Coulomb friction to reproduce the
tensioning, forward drive, pause, and reverse drive of the belt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        # Populated in initialize().
        self.scene = None
        self.config: dict = {}
        self.sample_dt: float = 1.0 / 60.0
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)

        # Cylinder parameters.
        self.cyl_radius: float = 0.28
        self.cyl_length: float = 1.5
        self.centerline_clearance: float = 0.008
        self.belt_thickness: float = 0.006
        self.friction_coeff: float = 0.4
        self.outer_omega: float = np.pi
        self.outer_signs = np.array([1.0, -1.0], dtype=float)

        # Belt / cloth.
        self.N_cloth: int = 0
        self.N_total: int = 0
        self.faces: np.ndarray | None = None
        self.ei: np.ndarray | None = None
        self.ej: np.ndarray | None = None
        self.edge_L0: np.ndarray | None = None
        self.uv: np.ndarray | None = None
        self.Dm_inv: np.ndarray | None = None
        self.face_area: np.ndarray | None = None
        self.vertex_face_indices: list[np.ndarray] | None = None

        # State.
        self.X: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.body_id: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.mass_quantity: np.ndarray | None = None
        self.orientation: np.ndarray | None = None
        self.angular_velocity: np.ndarray | None = None

        # Cylinder state caches.
        self.cyl_initial_q = np.array([0.707106781, 0.0, 0.0, 0.707106781], dtype=float)
        self.cyl_body_ids = np.array([1, 2, 3], dtype=int)
        self.cyl_centers_t = np.zeros((3, 3), dtype=float)
        self.cyl_omegas_t = np.zeros((3, 3), dtype=float)

        self.current_time: float = 0.0
        self.substeps: int = 10
        self.edge_iterations: int = 8
        self.damping: float = 1.5

    # ------------------------------------------------------------------
    # Kinematic helpers
    # ------------------------------------------------------------------
    def _top_z(self, t: float) -> float:
        z0, z1 = self.config["cylinders"]["initial_centers_xz"][2][1], \
                 self.config["cylinders"]["final_centers_xz"][2][1]
        t0, t1 = self.config["timeline"]["lift"]
        if t <= t0:
            return z0
        if t >= t1:
            return z1
        s = (t - t0) / (t1 - t0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        return z0 + (z1 - z0) * s

    def _outer_phase(self, t: float) -> float:
        f0, f1 = self.config["timeline"]["forward"]
        p0, p1 = self.config["timeline"]["pause"]
        r0, r1 = self.config["timeline"]["reverse"]
        omega = self.outer_omega

        if t <= f0:
            return 0.0
        if t < f1:
            return omega * (t - f0)
        if t <= p1:
            return omega * (f1 - f0)
        if t < r1:
            return omega * (f1 - f0) - omega * (t - r0)
        return 0.0

    def _outer_omega_world(self, t: float) -> float:
        f0, f1 = self.config["timeline"]["forward"]
        r0, r1 = self.config["timeline"]["reverse"]
        if f0 < t < f1:
            return self.outer_omega
        if r0 < t < r1:
            return -self.outer_omega
        return 0.0

    def _update_cylinder_kinematics(self, t: float) -> None:
        # Left / right centers are fixed.
        left_xz = self.config["cylinders"]["initial_centers_xz"][0]
        right_xz = self.config["cylinders"]["initial_centers_xz"][1]
        top_z = self._top_z(t)

        self.cyl_centers_t[0] = [left_xz[0], 0.0, left_xz[1]]
        self.cyl_centers_t[1] = [right_xz[0], 0.0, right_xz[1]]
        self.cyl_centers_t[2] = [0.0, 0.0, top_z]

        # Angular velocities in world frame (axis y).
        omega_left = self.outer_signs[0] * self._outer_omega_world(t)
        omega_right = self.outer_signs[1] * self._outer_omega_world(t)
        self.cyl_omegas_t[0] = [0.0, omega_left, 0.0]
        self.cyl_omegas_t[1] = [0.0, omega_right, 0.0]
        self.cyl_omegas_t[2] = [0.0, 0.0, 0.0]

    def _cylinder_orientation(self, body_idx: int, t: float) -> np.ndarray:
        # body_idx 0 -> left, 1 -> right, 2 -> top.
        if body_idx == 2:
            return self.cyl_initial_q.copy()
        sign = self.outer_signs[body_idx]
        phase = sign * self._outer_phase(t)
        # q_y * q_initial, where q_y rotates about y.
        half = 0.5 * phase
        qy = np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=float)
        # Hamilton product qy * q_initial.
        x1, y1, z1, w1 = qy
        x2, y2, z2, w2 = self.cyl_initial_q
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config
        self.sample_dt = float(self.config["sample_dt"])

        cyl_cfg = self.config["cylinders"]
        self.cyl_radius = float(cyl_cfg["radius"])
        self.cyl_length = float(cyl_cfg["length"])
        self.outer_omega = float(cyl_cfg["outer_angular_speed"])
        self.outer_signs = np.asarray(cyl_cfg["outer_rotation_signs"], dtype=float)

        belt_cfg = self.config["belt"]
        self.belt_thickness = float(belt_cfg["thickness"])
        self.centerline_clearance = float(belt_cfg["centerline_clearance"])
        self.friction_coeff = float(self.config["friction"]["coefficient"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Initial state from the scene.
        init_frame = scene.initial_state()
        self.X = np.asarray(init_frame.position, dtype=float).copy()
        self.V = np.asarray(init_frame.velocity, dtype=float).copy()
        self.body_id = np.asarray(init_frame.body_id, dtype=int).copy()
        self.mass = np.asarray(init_frame.mass, dtype=float).copy()
        self.orientation = np.asarray(init_frame.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(init_frame.angular_velocity, dtype=float).copy()
        self.uv = np.asarray(init_frame.attributes["uv"], dtype=float).copy()
        self.mass_quantity = np.asarray(init_frame.attributes["mass_quantity"], dtype=float).copy()

        self.N_total = len(self.X)
        self.N_cloth = int(self.mass_quantity.nonzero()[0].size)
        # Recompute cloth count from topology if possible; otherwise keep nonzeros.
        faces = np.asarray(scene._faces, dtype=int)
        if faces.max() + 1 < self.N_total:
            self.N_cloth = int(faces.max() + 1)

        self.faces = faces

        # Build unique undirected edge list from faces.
        e0 = np.column_stack([faces[:, 0], faces[:, 1]])
        e1 = np.column_stack([faces[:, 1], faces[:, 2]])
        e2 = np.column_stack([faces[:, 2], faces[:, 0]])
        edges = np.vstack([e0, e1, e2])
        edges = np.sort(edges, axis=1)
        self.ei, self.ej = np.unique(edges, axis=0, return_index=False)[0], np.unique(edges, axis=0, return_index=False)[1]
        # Use a single unique call to avoid mismatched ordering.
        edges_unique = np.unique(edges, axis=0)
        self.ei = edges_unique[:, 0]
        self.ej = edges_unique[:, 1]

        # Rest lengths with prestrain -> taut belt.
        prestrain = float(belt_cfg.get("prestrain", 0.0))
        X0 = self.X[: self.N_cloth]
        diff = X0[self.ei] - X0[self.ej]
        self.edge_L0 = np.linalg.norm(diff, axis=1) * (1.0 - prestrain)

        # Per-face material-to-world maps for deformation gradient.
        self._build_deformation_helpers(X0)

        # Make sure cylinder samples have identity-ish orientation and zero angular velocity.
        for i, bid in enumerate(self.cyl_body_ids):
            idx = self.N_cloth + i
            self.orientation[idx] = self._cylinder_orientation(i, 0.0)
        self.angular_velocity[self.N_cloth:] = 0.0

        self.current_time = 0.0

    def _build_deformation_helpers(self, X0: np.ndarray) -> None:
        faces = self.faces
        uv0 = self.uv[: self.N_cloth]
        u0 = uv0[faces[:, 0]]
        u1 = uv0[faces[:, 1]]
        u2 = uv0[faces[:, 2]]
        Du = np.stack([u1 - u0, u2 - u0], axis=2)  # (F,2,2)
        det = Du[:, 0, 0] * Du[:, 1, 1] - Du[:, 0, 1] * Du[:, 1, 0]
        # Guard against degenerate faces.
        det = np.where(np.abs(det) < 1e-12, 1.0, det)
        inv = np.empty((len(Du), 2, 2), dtype=float)
        inv[:, 0, 0] = Du[:, 1, 1] / det
        inv[:, 0, 1] = -Du[:, 0, 1] / det
        inv[:, 1, 0] = -Du[:, 1, 0] / det
        inv[:, 1, 1] = Du[:, 0, 0] / det
        self.Dm_inv = inv

        # Vertex -> face list for averaging per-vertex F.
        nfaces = len(faces)
        vfi = [[] for _ in range(self.N_cloth)]
        for fi in range(nfaces):
            for v in faces[fi]:
                vfi[v].append(fi)
        self.vertex_face_indices = [np.asarray(lst, dtype=int) for lst in vfi]

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _project_edges(self, X: np.ndarray) -> None:
        """PBD edge-length projection (inextensible)."""
        for _ in range(self.edge_iterations):
            p_i = X[self.ei]
            p_j = X[self.ej]
            diff = p_i - p_j
            len_d = np.linalg.norm(diff, axis=1)
            # Avoid division by very small numbers.
            inv_len = np.where(len_d > 1e-10, 1.0 / len_d, 0.0)
            scale = (len_d - self.edge_L0) * inv_len
            delta = 0.5 * scale[:, None] * diff

            # Accumulate corrections with bincount to avoid races.
            disp = np.zeros_like(X)
            np.add.at(disp, self.ei, delta)
            np.add.at(disp, self.ej, -delta)
            X[: self.N_cloth] += disp[: self.N_cloth]

    def _apply_friction(self, dt: float, t: float) -> None:
        """Velocity-level frictional coupling to the kinematic cylinders."""
        cloth = np.s_[: self.N_cloth]
        Xc = self.X[cloth]
        Vc = self.V[cloth]
        for ci in range(3):
            c = self.cyl_centers_t[ci]
            omega = self.cyl_omegas_t[ci, 1]
            dx = Xc[:, 0] - c[0]
            dz = Xc[:, 2] - c[2]
            dist = np.sqrt(dx * dx + dz * dz)
            # Avoid division by zero if a vertex coincides with a cylinder center.
            dist = np.where(dist < 1e-12, 1.0, dist)
            # Radial normal in xz.
            n_x = dx / dist
            n_z = dz / dist
            # Tangential surface velocity of the cylinder.
            # (0, omega, 0) x (dx, 0, dz) = (-omega*dz, 0, omega*dx)
            target_x = -omega * dz
            target_z = omega * dx

            # Normal/tangential split of cloth velocity.
            vn = Vc[:, 0] * n_x + Vc[:, 2] * n_z
            v_t_x = Vc[:, 0] - vn * n_x
            v_t_z = Vc[:, 2] - vn * n_z

            rel_x = v_t_x - target_x
            rel_z = v_t_z - target_z

            # Exponential decay of relative tangential velocity (Coulomb-like).
            decay = np.exp(-self.friction_coeff * 8.0 * dt)
            rel_x *= decay
            rel_z *= decay

            # Apply only to samples near the contact surface.
            margin = self.cyl_radius * 0.25
            active = dist < (self.cyl_radius + self.centerline_clearance + margin)
            mask = active
            Vc[mask, 0] = (vn[mask] * n_x[mask]) + target_x[mask] + rel_x[mask]
            Vc[mask, 2] = (vn[mask] * n_z[mask]) + target_z[mask] + rel_z[mask]

    def _project_contact(self, X: np.ndarray) -> None:
        """Hard radial contact with all three cylinders."""
        cloth = np.s_[: self.N_cloth]
        Xc = X[cloth]
        for ci in range(3):
            c = self.cyl_centers_t[ci]
            dx = Xc[:, 0] - c[0]
            dz = Xc[:, 2] - c[2]
            dist = np.sqrt(dx * dx + dz * dz)
            target_r = self.cyl_radius + self.centerline_clearance
            inside = dist < target_r
            if not np.any(inside):
                continue
            ratio = target_r / dist[inside]
            # Clamp to avoid NaN if dist==0.
            ratio = np.clip(ratio, 1.0, 1e6)
            Xc[inside, 0] = c[0] + (Xc[inside, 0] - c[0]) * ratio
            Xc[inside, 2] = c[2] + (Xc[inside, 2] - c[2]) * ratio

    def _step(self, dt: float, t: float) -> None:
        # Store previous positions for velocity update.
        X_prev = self.X.copy()

        # Update prescribed cylinder positions/orientations at the end of the step.
        self._update_cylinder_kinematics(t + dt)

        # Cylinder positions (linear).
        for i, bid in enumerate(self.cyl_body_ids):
            idx = self.N_cloth + i
            self.X[idx] = self.cyl_centers_t[i]
            self.orientation[idx] = self._cylinder_orientation(i, t + dt)
            self.angular_velocity[idx] = self.cyl_omegas_t[i]

        # Cloth dynamics.
        cloth = np.s_[: self.N_cloth]
        self.V[cloth] += self.gravity * dt
        # Velocity damping.
        self.V[cloth] *= np.exp(-self.damping * dt)

        # Apply friction/coupling before the position prediction.
        self._apply_friction(dt, t)

        # Predict positions.
        self.X[cloth] = self.X[cloth] + dt * self.V[cloth]

        # Enforce hard contact.
        self._project_contact(self.X)

        # Enforce inextensible edge constraints.
        self._project_edges(self.X)

        # Update cloth velocities.
        self.V[cloth] = (self.X[cloth] - X_prev[cloth]) / dt

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        """Per-sample deformation gradient (per-vertex averaged)."""
        F = np.tile(np.eye(3, dtype=float)[None], (self.N_total, 1, 1))
        if self.Dm_inv is None:
            return F

        faces = self.faces
        p0 = self.X[faces[:, 0]]
        p1 = self.X[faces[:, 1]]
        p2 = self.X[faces[:, 2]]
        Ds = np.stack([p1 - p0, p2 - p0], axis=2)  # (F,3,2)
        F2 = Ds @ self.Dm_inv  # (F,3,2)
        c0 = F2[:, :, 0]
        c1 = F2[:, :, 1]
        c2 = np.cross(c0, c1)
        # Normalize the third column to avoid scaling artifacts.
        norm_c2 = np.linalg.norm(c2, axis=1, keepdims=True)
        norm_c2 = np.where(norm_c2 < 1e-12, 1.0, norm_c2)
        c2 = c2 / norm_c2

        accum = np.zeros((self.N_cloth, 3, 3), dtype=float)
        counts = np.zeros(self.N_cloth, dtype=float)
        for fi in range(len(faces)):
            fmat = np.stack([c0[fi], c1[fi], c2[fi]], axis=1)
            for v in faces[fi]:
                accum[v] += fmat
                counts[v] += 1.0
        counts = np.where(counts < 1.0, 1.0, counts)
        F_cloth = accum / counts[:, None, None]
        F[: self.N_cloth] = F_cloth
        return F

    def _contact_signed_distance(self) -> np.ndarray:
        """Signed gap for every cloth vertex to the nearest cylinder surface."""
        Xc = self.X[: self.N_cloth]
        sd = np.full(self.N_cloth, 1e6, dtype=float)
        for ci in range(3):
            c = self.cyl_centers_t[ci]
            dx = Xc[:, 0] - c[0]
            dz = Xc[:, 2] - c[2]
            dist = np.sqrt(dx * dx + dz * dz)
            sd = np.minimum(sd, dist - (self.cyl_radius + self.centerline_clearance))
        return sd

    def _make_frame(self, t: float) -> StateFrame:
        F = self._deformation_gradient()
        contact_sd = self._contact_signed_distance()
        return StateFrame(
            time=float(t),
            position=self.X.copy(),
            velocity=self.V.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass_quantity.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "contact_signed_distance": contact_sd,
            },
            scalars={
                "contact_length_scale": float(self.belt_thickness),
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time - 1e-12:
            raise ValueError("advance_to time must be non-decreasing")

        # Ensure at least the initial frame is returned for t == 0.
        if time == 0.0 and self.current_time == 0.0:
            return self._make_frame(0.0)

        target = float(time)
        total_dt = target - self.current_time
        if total_dt <= 0.0:
            return self._make_frame(self.current_time)

        h = total_dt / self.substeps
        for _ in range(self.substeps):
            self._step(h, self.current_time)
            self.current_time += h

        return self._make_frame(self.current_time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": self.substeps,
            "edge_iterations": self.edge_iterations,
            "cloth_vertices": self.N_cloth,
            "total_samples": self.N_total,
        }
