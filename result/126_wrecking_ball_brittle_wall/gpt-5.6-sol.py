"""Particle based brittle-wall solver for the wrecking-ball scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self._scene = scene
        self._rng = np.random.default_rng(int(seed))
        self._time = 0.0
        self._dt = 0.004
        self._spacing = float(scene.config["sampling"]["spacing"])
        self._gravity = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._x0 = np.asarray(initial.position, dtype=float).copy()
        self._x = self._x0.copy()
        self._v = np.asarray(initial.velocity, dtype=float).copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._n = len(self._x)
        self._damage = np.zeros(self._n, dtype=float)
        self._base = self._x0[:, 2] <= float(np.min(self._x0[:, 2])) + 1e-7

        # Six-neighbour bonds from the regular sampling lattice.
        index = {tuple(np.round(p, 8)): i for i, p in enumerate(self._x0)}
        pairs = []
        for i, p in enumerate(self._x0):
            for k in range(3):
                q = p.copy()
                q[k] += self._spacing
                j = index.get(tuple(np.round(q, 8)))
                if j is not None:
                    pairs.append((i, j))
        self._pairs = np.asarray(pairs, dtype=np.int32)
        self._rest = self._x0[self._pairs[:, 1]] - self._x0[self._pairs[:, 0]]
        self._rest_len = np.linalg.norm(self._rest, axis=1)

        # Heavy swinging pendulum, integrated independently and coupled by contact.
        self._pivot = np.array([-1.75, 0.0, 3.35], dtype=float)
        self._cable_length = 2.0
        self._ball_radius = 0.48
        self._theta = 0.0
        self._omega = 4.25
        self._ball_center = self._pendulum_center()
        self._ball_velocity = np.zeros(3)
        self._initialized = True

    def _pendulum_center(self) -> np.ndarray:
        s, c = np.sin(self._theta), np.cos(self._theta)
        return self._pivot + np.array([self._cable_length * s, 0.0, -self._cable_length * c])

    def _step(self, h: float) -> None:
        self._omega += h * (-np.linalg.norm(self._gravity) / self._cable_length) * np.sin(self._theta)
        self._omega *= np.exp(-0.025 * h)
        self._theta += h * self._omega
        old_ball = self._ball_center
        self._ball_center = self._pendulum_center()
        self._ball_velocity = (self._ball_center - old_ball) / h

        force = self._mass[:, None] * self._gravity
        # The lowest masonry course is seated on the rigid foundation.  As its
        # damage rises this support releases, allowing a fractured base chunk to
        # move and settle on the floor like the rest of the debris.
        if np.any(self._base):
            base_intact = np.clip(1.0 - self._damage[self._base], 0.0, 1.0)
            base_disp = self._x[self._base] - self._x0[self._base]
            base_relv = self._v[self._base]
            force[self._base] += (-12000.0 * base_disp - 90.0 * base_relv) * base_intact[:, None]
        if len(self._pairs):
            a, b = self._pairs[:, 0], self._pairs[:, 1]
            d = self._x[b] - self._x[a]
            ln = np.linalg.norm(d, axis=1)
            unit = d / np.maximum(ln[:, None], 1e-9)
            extension = ln - self._rest_len
            intact = np.clip(1.0 - np.maximum(self._damage[a], self._damage[b]), 0.0, 1.0)
            relv = self._v[b] - self._v[a]
            damp = np.sum(relv * unit, axis=1)
            fs = (8500.0 * intact * extension + 34.0 * damp)[:, None] * unit
            np.add.at(force, a, fs)
            np.add.at(force, b, -fs)

        # Sphere contact with the moving wrecking ball.
        delta = self._x - self._ball_center[None, :]
        dist = np.linalg.norm(delta, axis=1)
        # The particle spacing undersamples the brick surface (the frozen bricks
        # extend above the particle cloud), so use a compact kernel radius that
        # transfers a masonry-surface impact to nearby material samples.
        penetration = self._ball_radius + 0.30 - dist
        hit = penetration > 0.0
        if np.any(hit):
            normal = delta[hit] / np.maximum(dist[hit, None], 1e-8)
            rel = self._v[hit] - self._ball_velocity[None, :]
            vn = np.sum(rel * normal, axis=1)
            rep = np.maximum(0.0, 18000.0 * penetration[hit] - 240.0 * vn)
            force[hit] += rep[:, None] * normal
            self._damage[hit] += h * (2.0 * penetration[hit] / max(self._spacing, 1e-6) +
                                      np.maximum(0.0, -vn) / 2.5)

        floor = 0.055
        below = self._x[:, 2] < floor
        if np.any(below):
            self._x[below, 2] = floor
            self._v[below, 2] = np.maximum(self._v[below, 2], 0.0)
            self._v[below, :2] *= 0.88

        force -= 1.8 * self._v
        self._v += h * force / np.maximum(self._mass[:, None], 1e-9)
        self._v = np.clip(self._v, -35.0, 35.0)
        self._x += h * self._v

        # Irreversible tensile failure weakens bonds and lets fragments separate.
        if len(self._pairs):
            a, b = self._pairs[:, 0], self._pairs[:, 1]
            d = self._x[b] - self._x[a]
            strain = np.maximum(0.0, np.linalg.norm(d, axis=1) / self._rest_len - 1.0)
            bond_damage = np.clip((strain - 0.055) / 0.10, 0.0, 1.0)
            np.maximum.at(self._damage, a, bond_damage)
            np.maximum.at(self._damage, b, bond_damage)
        self._damage = np.clip(self._damage, 0.0, 1.0)
        below = self._x[:, 2] < floor
        if np.any(below):
            self._x[below, 2] = floor
            self._v[below, 2] = np.maximum(self._v[below, 2], 0.0)

    def _deformation_gradients(self) -> np.ndarray:
        F = np.repeat(np.eye(3)[None], self._n, axis=0)
        neighbours = [[] for _ in range(self._n)]
        for i, j in self._pairs:
            neighbours[int(i)].append(int(j))
            neighbours[int(j)].append(int(i))
        for i, nb in enumerate(neighbours):
            if len(nb) < 3:
                continue
            X = self._x0[np.asarray(nb)] - self._x0[i]
            Y = self._x[np.asarray(nb)] - self._x[i]
            try:
                Fi = np.linalg.lstsq(X, Y, rcond=None)[0].T
                if np.isfinite(Fi).all():
                    F[i] = np.clip(Fi, -4.0, 4.0)
            except np.linalg.LinAlgError:
                pass
        return F

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if not np.isfinite(target) or target < self._time:
            raise ValueError("advance_to time must be finite and monotonic")
        while self._time + 1e-12 < target:
            h = min(self._dt, target - self._time)
            self._step(h)
            self._time += h

        F = self._deformation_gradients()
        floor_dist = self._x[:, 2] - 0.055
        ball_dist = np.linalg.norm(self._x - self._ball_center[None, :], axis=1) - self._ball_radius
        frame = StateFrame(
            self._time, self._x.copy(), self._v.copy(), mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "particle_radius": np.full(self._n, 0.45 * self._spacing),
                "material_label": self._body_id.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": np.minimum(floor_dist, ball_dist),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self._mass[:, None] * self._gravity,
            },
            scalars={
                "contact_length_scale": self._spacing,
                "grid_spacing": self._spacing,
                "characteristic_length": max(float(np.ptp(self._x0, axis=0).max()), self._spacing),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_particle_springs",
            "particle_count": int(self._n),
            "bond_count": int(len(self._pairs)),
            "max_damage": float(np.max(self._damage)) if self._n else 0.0,
            "pendulum_angle": float(self._theta),
        }
