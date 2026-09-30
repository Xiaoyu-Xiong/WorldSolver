"""Physics solver for the torus-through-hole scene.

Implements an explicit mass-spring deformable body with gravity, damping, and
constraint collisions against the ground, the rigid hole plate, and the
kinematic pusher.  Genesis / gs is not used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic explicit mass-spring + constraint collision solver."""

    def initialize(self, scene, seed: int) -> None:
        self.cfg = scene.config
        self.rng = np.random.default_rng(seed)

        # Particle state.
        self.x = scene.vertices.copy().astype(float)
        self.n = self.x.shape[0]
        self.v = np.zeros((self.n, 3), dtype=float)
        init_vel = np.asarray(self.cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        if init_vel.shape == (3,):
            self.v[:] = init_vel

        total_mass = float(self.cfg["subject_mass"])
        self.m = np.full(self.n, total_mass / self.n, dtype=float)
        self.inv_m = np.full(self.n, self.n / total_mass, dtype=float)

        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.hole_z = float(self.cfg["hole_z"])
        self.hole_radius = float(self.cfg["hole_radius"])
        self.push_speed = float(self.cfg["push_speed"])
        self.pusher_init_z = 1.88

        # Material parameters.
        self.stiffness = float(self.cfg["stiffness"])
        self.damping = float(self.cfg["damping"])
        self.restitution = float(self.cfg["restitution"])
        self.friction = float(self.cfg["friction"])
        self.contact_length_scale = float(self.cfg["contact_length_scale"])

        # Load mesh topology.
        asset_path = Path(self.cfg["subject_asset"])
        if not asset_path.is_absolute():
            asset_path = Path(__file__).resolve().parent.parent / asset_path
        with np.load(asset_path) as data:
            if "faces" in data:
                faces = np.asarray(data["faces"], dtype=int)
            elif "elements" in data and np.asarray(data["elements"]).shape[1] in (3, 4):
                faces = np.asarray(data["elements"], dtype=int)
            else:
                faces = None
            self.rest_x = np.asarray(data["vertices"], dtype=float)

        # Build unique edges.
        if faces is not None and faces.ndim == 2 and faces.shape[1] == 3:
            edges = set()
            for f in faces:
                e = tuple(sorted((int(f[0]), int(f[1]))))
                edges.add(e)
                e = tuple(sorted((int(f[1]), int(f[2]))))
                edges.add(e)
                e = tuple(sorted((int(f[2]), int(f[0]))))
                edges.add(e)
            self.edges = np.asarray(list(edges), dtype=int)
        else:
            # Fallback: connect each vertex to nearby vertices using the rest shape.
            tree = KDTree(self.rest_x)
            # Pick a connectivity radius from the mean nearest-neighbor distance.
            dd, _ = tree.query(self.rest_x, k=2)
            mean_nn = float(np.mean(dd[:, 1]))
            pairs = tree.query_pairs(r=mean_nn * 2.5, output_type="ndarray")
            self.edges = np.asarray(pairs, dtype=int)

        # Edge rest lengths and directions.
        self.rest_len = np.linalg.norm(
            self.rest_x[self.edges[:, 1]] - self.rest_x[self.edges[:, 0]], axis=1
        )
        self.rest_dir = (
            self.rest_x[self.edges[:, 1]] - self.rest_x[self.edges[:, 0]]
        ) / self.rest_len[:, None]

        # Per-vertex neighbor lists and rest-shape matrices for deformation gradient.
        self.neighbors = [[] for _ in range(self.n)]
        for (i, j) in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
        # Precompute B^{-1} for the deformation-gradient least-squares fit.
        self.B_inv = []
        for i in range(self.n):
            nb = self.neighbors[i]
            if len(nb) < 3:
                self.B_inv.append(None)
                continue
            dX = self.rest_x[nb] - self.rest_x[i]
            # Weight by inverse squared rest length to avoid bias toward far neighbors.
            w = 1.0 / (np.linalg.norm(dX, axis=1) ** 2 + 1e-12)
            B = (dX.T * w) @ dX
            try:
                Bi = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                Bi = np.linalg.pinv(B)
            self.B_inv.append((Bi, np.asarray(nb, dtype=int), w))

        self.F = np.tile(np.eye(3, dtype=float), (self.n, 1, 1))

        self.time = 0.0
        self.sample_dt = float(self.cfg["sample_dt"])
        # Use a fixed small sub-step for stability.
        self.sub_dt = self.sample_dt / 8.0

    def _pusher_z(self, t: float) -> tuple[float, float, float]:
        """Return (center_z, bottom_z, top_z) of the kinematic cylinder."""
        cz = self.pusher_init_z - self.push_speed * t
        half_h = 0.18 / 2.0
        return cz, cz - half_h, cz + half_h

    def _compute_deformation_gradient(self) -> np.ndarray:
        F = np.empty((self.n, 3, 3), dtype=float)
        for i in range(self.n):
            item = self.B_inv[i]
            if item is None:
                F[i] = np.eye(3)
                continue
            Bi, nb, w = item
            dx = self.x[nb] - self.x[i]
            dX = self.rest_x[nb] - self.rest_x[i]
            A = (dx.T * w) @ dX
            F[i] = A @ Bi
        # Clamp singular values to avoid blow-up in diagnostics.
        u, s, vt = np.linalg.svd(F, full_matrices=False)
        s = np.clip(s, 0.1, 10.0)
        F = u * s[:, :, None] @ vt
        return F

    def _spring_forces(self) -> np.ndarray:
        f = np.zeros((self.n, 3), dtype=float)
        if self.edges.size == 0:
            return f
        dx = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        lengths = np.linalg.norm(dx, axis=1)
        nonzero = lengths > 1e-12
        if not nonzero.all():
            lengths = np.where(nonzero, lengths, 1.0)
        # Stiffness is scaled by edge count so the total object stiffness matches cfg.
        k = self.stiffness / max(1, len(self.edges))
        magnitudes = k * (lengths - self.rest_len)
        dirs = dx / lengths[:, None]
        edge_forces = magnitudes[:, None] * dirs
        np.add.at(f, self.edges[:, 0], edge_forces)
        np.add.at(f, self.edges[:, 1], -edge_forces)
        return f

    def _contact_response(self, t: float) -> None:
        # Ground.
        pen = self.ground_z - self.x[:, 2]
        mask = pen > 0.0
        if mask.any():
            self.x[mask, 2] = self.ground_z + pen[mask] * 0.0  # place on ground
            self.x[mask, 2] = self.ground_z
            vmask = self.v[mask, 2] < 0.0
            self.v[mask, 2] = np.where(vmask, -self.restitution * self.v[mask, 2], self.v[mask, 2])
            # Friction.
            self.v[mask, :2] *= max(0.0, 1.0 - self.friction * 0.5)

        # Rigid hole plate: one-sided barrier at hole_z with circular hole.
        r_xy = np.linalg.norm(self.x[:, :2], axis=1)
        plate_mask = (self.x[:, 2] < self.hole_z) & (r_xy > self.hole_radius)
        if plate_mask.any():
            self.x[plate_mask, 2] = self.hole_z
            vmask = self.v[plate_mask, 2] < 0.0
            idx = np.where(plate_mask)[0]
            self.v[idx, 2] = np.where(vmask, -self.restitution * self.v[idx, 2], self.v[idx, 2])
            self.v[idx, :2] *= max(0.0, 1.0 - self.friction * 0.5)

        # Kinematic pusher cylinder.
        cz, bottom, top = self._pusher_z(t)
        pusher_radius = 0.48
        inside = (r_xy < pusher_radius) & (self.x[:, 2] > bottom) & (self.x[:, 2] < top)
        if inside.any():
            idx = np.where(inside)[0]
            d_bottom = self.x[idx, 2] - bottom
            d_top = top - self.x[idx, 2]
            d_side = pusher_radius - r_xy[idx]
            choice = np.argmin(np.column_stack([d_bottom, d_top, d_side]), axis=1)
            for k, c in enumerate(choice):
                ii = idx[k]
                if c == 0:
                    # Nearest the bottom face; push down and reflect upward velocity.
                    self.x[ii, 2] = bottom
                    if self.v[ii, 2] > 0.0:
                        self.v[ii, 2] = -self.restitution * self.v[ii, 2]
                elif c == 1:
                    # Nearest the top face (unlikely while descending).
                    self.x[ii, 2] = top
                    if self.v[ii, 2] < 0.0:
                        self.v[ii, 2] = -self.restitution * self.v[ii, 2]
                else:
                    # Radial side contact.
                    scale = pusher_radius / (r_xy[ii] + 1e-12)
                    self.x[ii, 0] *= scale
                    self.x[ii, 1] *= scale
                    # Reflect radial velocity outward.
                    radial_v = self.v[ii, :2]
                    radial_pos = self.x[ii, :2]
                    rv = np.dot(radial_v, radial_pos)
                    if rv < 0.0:
                        self.v[ii, :2] -= (1.0 + self.restitution) * rv * radial_pos / (
                            np.dot(radial_pos, radial_pos) + 1e-12
                        )
                self.v[ii, :2] *= max(0.0, 1.0 - self.friction * 0.5)

    def _step(self, dt: float) -> None:
        t = self.time + dt
        # Internal elastic force.
        f_spring = self._spring_forces()
        # Damping (mass-proportional).
        f_damp = -self.damping * self.m[:, None] * self.v
        # Gravity.
        f_grav = self.m[:, None] * self.gravity

        f_total = f_spring + f_damp + f_grav

        # Semi-implicit Euler.
        self.v += dt * f_total * self.inv_m[:, None]
        self.x += dt * self.v

        # Enforce collision constraints.
        self._contact_response(t)

        self.time = t

    def _signed_distances(self) -> np.ndarray:
        # Ground signed distance (positive above).
        sd_ground = self.x[:, 2] - self.ground_z

        # Hole plate: only solid annular region blocks.
        r_xy = np.linalg.norm(self.x[:, :2], axis=1)
        sd_plate = np.full(self.n, np.inf)
        mask = r_xy > self.hole_radius
        sd_plate[mask] = self.x[mask, 2] - self.hole_z

        # Pusher: signed distance to exterior of cylinder.
        _, bottom, top = self._pusher_z(self.time)
        pusher_radius = 0.48
        sd_pusher = np.maximum.reduce(
            [bottom - self.x[:, 2], self.x[:, 2] - top, r_xy - pusher_radius]
        )

        return np.minimum.reduce([sd_ground, sd_plate, sd_pusher])

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.time:
            # State cannot go backwards; return current state at requested time.
            return self._build_frame(time)

        # Sub-step until we reach the target.
        remaining = time - self.time
        n_sub = int(np.floor(remaining / self.sub_dt))
        for _ in range(n_sub):
            self._step(self.sub_dt)
        # Final fractional step.
        rem = time - self.time
        if rem > 1e-12:
            self._step(rem)

        self.F = self._compute_deformation_gradient()
        return self._build_frame(self.time)

    def _build_frame(self, time: float) -> StateFrame:
        mass_quantity = self.m.copy()
        external_force = self.m[:, None] * self.gravity
        contact_sd = self._signed_distances()
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": mass_quantity,
                "deformation_gradient": self.F.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
                "contact_signed_distance": contact_sd,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_explicit",
            "particles": int(self.n),
            "edges": int(len(self.edges)),
            "sub_dt": float(self.sub_dt),
            "time": float(self.time),
        }
