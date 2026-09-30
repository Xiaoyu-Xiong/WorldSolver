"""Solver Agent implementation for the self-supporting parabolic arch.

A lightweight 3-D rigid-particle penalty contact solver.  Each voussoir is
treated as a point mass located at the world position supplied by the scene;
gravity, ground contact, and inter-block penalty contacts advance the state.
The orientation and angular velocity are held fixed because the integrator
works with the translational degrees of freedom only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Penalty-contact time-stepper for the self-supporting arch."""

    def initialize(self, scene, seed: int) -> None:
        state = scene.initial_state()

        self.n = int(len(state.position))
        self.mass = np.asarray(state.mass, dtype=float).reshape(self.n)
        self.g = np.asarray([0.0, 0.0, -9.81], dtype=float)

        # Copy the initial kinematic state.
        self.pos = np.asarray(state.position, dtype=float).copy()
        self.vel = np.asarray(state.velocity, dtype=float).copy()
        self.ori = np.asarray(state.orientation, dtype=float).copy()
        self.ang = np.asarray(state.angular_velocity, dtype=float).copy()
        self.body_id = np.asarray(state.body_id, dtype=int).copy()

        # Reference geometry: rest distances for every pair, and per-body
        # ground-contact half-height.
        self.radius = self._load_ground_radii(scene)
        self.rest_dist = self._pairwise_distance(self.pos)

        # Material / contact parameters.
        self.mu = 0.6
        self.k = 1.0e6
        self.c_damp = 2.0 * np.sqrt(self.k / np.mean(self.mass))
        self.substep = 1e-4
        self.time = 0.0

        # Constant external force: gravity only (N).
        self.external_force = (self.mass[:, None] * self.g).copy()

        # For stable equilibrium, seed with a tiny non-zero tangential velocity so
        # the penalty solver can resolve the initial contact stack rather than
        # remaining exactly at rest.
        self.vel[:, 0] += 1e-5

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "num_bodies": self.n,
            "contact_stiffness": self.k,
            "damping": self.c_damp,
            "friction_coefficient": self.mu,
            "substep": self.substep,
        }

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._frame(time)

        while self.time < time:
            dt = min(self.substep, time - self.time)
            self._step(dt)
            self.time += dt

        return self._frame(time)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _step(self, dt: float) -> None:
        """One symplectic Euler step of the penalty contact system."""
        n = self.n
        f = np.zeros((n, 3), dtype=float)

        # Gravity.
        f += self.external_force

        # Ground contact (plane z = 0) plus Coulomb friction.
        for i in range(n):
            gap = self.pos[i, 2] - self.radius[i]
            if gap < 0.0:
                normal_force = -self.k * gap
                # Normal repulsion and damping.
                f[i, 2] += normal_force - self.c_damp * self.vel[i, 2]
                # Tangential friction opposes horizontal motion.
                vh = self.vel[i, :2].copy()
                vh_norm = np.linalg.norm(vh)
                if vh_norm > 0.0:
                    f_limit = self.mu * max(normal_force, 0.0)
                    f[i, :2] -= f_limit * np.tanh(vh_norm * 10.0) * (vh / vh_norm)

        # Pairwise penalty contacts for all body pairs.
        diff = self.pos[:, None, :] - self.pos[None, :, :]  # n,n,3
        dist = np.linalg.norm(diff, axis=2)
        # Avoid division by zero on the diagonal.
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        gap = dist - self.rest_dist  # negative => compressed/overlapping
        mask = (gap < 0.0) & (~np.eye(n, dtype=bool))
        if np.any(mask):
            mag = -self.k * gap  # non-negative where mask
            mag[~mask] = 0.0
            unit = diff / dist_safe[..., None]
            force_tensor = mag[..., None] * unit
            f += force_tensor.sum(axis=1)

        # Global mass-proportional Rayleigh damping.
        f -= self.c_damp * self.mass[:, None] * self.vel

        # Symplectic Euler update.
        acc = f / self.mass[:, None]
        self.vel += dt * acc
        self.pos += dt * self.vel

    def _frame(self, time: float) -> StateFrame:
        """Build a StateFrame from the current internal state."""
        n = self.n
        # Build contact data for the canonical adjacent + ground contact set.
        contact_gaps = []

        # Ground contacts for every body (1 per body).
        for i in range(n):
            contact_gaps.append(self.pos[i, 2] - self.radius[i])

        # Adjacent block-block contacts (i, i+1).
        for i in range(n - 1):
            j = i + 1
            d = np.linalg.norm(self.pos[i] - self.pos[j])
            contact_gaps.append(d - self.rest_dist[i, j])

        contact_gaps = np.asarray(contact_gaps, dtype=float)
        constraint_target = np.zeros_like(contact_gaps)

        # A representative contact length scale: mean ground half-height.
        contact_length_scale = float(np.mean(self.radius))
        if not np.isfinite(contact_length_scale) or contact_length_scale <= 0.0:
            contact_length_scale = 0.25

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.ori.copy(),
            angular_velocity=self.ang.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "constraint_value": contact_gaps.copy(),
                "constraint_target": constraint_target.copy(),
                "contact_signed_distance": contact_gaps.copy(),
                "external_force": self.external_force.copy(),
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": contact_length_scale,
            },
        )

    def _pairwise_distance(self, pos: np.ndarray) -> np.ndarray:
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _load_ground_radii(self, scene: Any) -> np.ndarray:
        """Return the per-body distance from the local origin to the bottom face.

        This is the negative of the minimum local z-coordinate of each
        voussoir mesh.  If the geometry files cannot be read, fall back to a
        conservative constant.
        """
        n = self.n
        radii = np.full(n, 0.25, dtype=float)

        geom_path = Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
        if not geom_path.is_file():
            return radii

        try:
            import yaml
            data = yaml.safe_load(geom_path.read_text(encoding="utf-8"))
        except Exception:
            return radii

        assets: dict[int, Path] = {}
        base = geom_path.parent
        for obj in data.get("objects", []):
            if not isinstance(obj, dict):
                continue
            binding = obj.get("state_binding") or {}
            if binding.get("selector") != "body_id":
                continue
            body_id = binding.get("value")
            if body_id is None:
                continue
            geom = obj.get("geometry", {})
            asset = geom.get("asset")
            if asset:
                assets[body_id] = base / asset

        for body_id, path in assets.items():
            if not (0 <= body_id < n):
                continue
            try:
                min_z = self._obj_min_z(path)
                # Distance from local origin to the lowest point of the mesh.
                radii[body_id] = max(-min_z, 0.01)
            except Exception:
                pass

        return radii

    @staticmethod
    def _obj_min_z(path: Path) -> float:
        min_z = float("inf")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    z = float(parts[3])
                    if z < min_z:
                        min_z = z
        if not np.isfinite(min_z):
            raise ValueError("No vertices found")
        return min_z
