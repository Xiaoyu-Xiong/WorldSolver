"""Solver Agent-owned implementation for 075_soft_ball_pin_compression.

A simple explicit corotational-elastic tetrahedral solver with penalty contact
against the lower and upper pin arrays.  The ball falls under gravity, lands on
the lower pins, and is compressed by the kinematic upper array.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


Array = NDArray[np.floating]


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.scene = scene
        self.config = scene.config
        self._root = Path(__file__).resolve().parent.parent

        # Time settings
        self.duration = float(self.config["duration"])
        self.sample_dt = float(self.config["sample_dt"])
        self.gravity = np.array(self.config["gravity"], dtype=float)

        # Ball geometry and state
        init = self.config["initial_conditions"]
        self.ball_radius = float(init["ball_radius"])
        self.ball_density = float(init["ball_density"])

        # Load tetrahedral mesh
        ball_asset = self._root / "assets" / "soft_ball_volume.json"
        ball_data = json.loads(ball_asset.read_text())
        self.rest_position = np.array(ball_data["vertices"], dtype=float)
        self.tetrahedra = np.array(ball_data["tetrahedra"], dtype=int)
        self.n_vertices = self.rest_position.shape[0]

        # Initial state from scene
        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float, copy=True)
        self.velocity = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)

        # Compute per-element rest volume and lumped nodal mass
        self.rest_volume = np.zeros(len(self.tetrahedra), dtype=float)
        for i, tet in enumerate(self.tetrahedra):
            p = self.rest_position[tet]
            vol = abs(np.linalg.det(np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1))) / 6.0
            self.rest_volume[i] = max(vol, 1e-18)

        # Pin obstacles
        lower = json.loads((self._root / "assets" / "lower_pins.json").read_text())
        upper = json.loads((self._root / "assets" / "upper_pins.json").read_text())
        self.lower_segments = self._segments(lower)
        self.upper_segments = self._segments(upper)
        self.pin_radius = 0.009

        # Upper array motion
        self.upper_axis = np.array(self.config["boundary_conditions"]["upper_pin_array"]["axis"], dtype=float)
        self.upper_keyframes = self.config["boundary_conditions"]["upper_pin_array"]["keyframes"]

        # Material parameters (soft rubber-like)
        self.young = 8.0e4  # Pa
        self.poisson = 0.45
        self.mu = self.young / (2.0 * (1.0 + self.poisson))
        self.lam = self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

        # Damping and contact
        self.damping = 5.0
        self.contact_stiffness = 1.0e6
        self.contact_damping = 1.0e3
        self.time_step = 1.0 / 1200.0  # simulation substep ~1.2 kHz

        # Current simulation time
        self.current_time = 0.0

        # Precompute per-element Dm_inv and rest orientation
        self.Dm_inv = np.zeros((len(self.tetrahedra), 3, 3), dtype=float)
        self.rest_orientation = np.zeros((len(self.tetrahedra), 3, 3), dtype=float)
        for i, tet in enumerate(self.tetrahedra):
            p = self.rest_position[tet]
            Dm = np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1)
            self.Dm_inv[i] = np.linalg.inv(Dm)

        # Cache initial upper segment positions for prescribed motion
        self.upper_segments_base = [seg.copy() for seg in self.upper_segments]

    def _segments(self, data: dict) -> list:
        verts = np.array(data["vertices"], dtype=float)
        segments = []
        for i in range(0, len(verts), 2):
            segments.append(verts[i : i + 2])
        return segments

    def _upper_displacement(self, t: float) -> float:
        kf = self.upper_keyframes
        for j in range(len(kf) - 1):
            if kf[j]["time"] <= t <= kf[j + 1]["time"]:
                a = kf[j]["time"]
                b = kf[j + 1]["time"]
                da = kf[j]["displacement"]
                db = kf[j + 1]["displacement"]
                if b == a:
                    return float(da)
                s = (t - a) / (b - a)
                return float(da + s * (db - da))
        return float(kf[-1]["displacement"])

    def _upper_segments_at(self, t: float) -> list:
        disp = self._upper_displacement(t)
        offset = disp * self.upper_axis
        return [seg + offset for seg in self.upper_segments_base]

    def _elastic_forces(self, position: Array) -> Array:
        f = np.zeros_like(position)
        for i, tet in enumerate(self.tetrahedra):
            p = position[tet]
            Dm_inv = self.Dm_inv[i]
            # Deformation gradient F = Ds * Dm_inv
            Ds = np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1)
            F = Ds @ Dm_inv
            detF = np.linalg.det(F)
            if detF <= 1e-12:
                # Avoid inversion; add stability with small clamp
                F = F + (abs(detF) + 1e-8) * np.eye(3)
                detF = np.linalg.det(F)
            # Neo-Hookean energy: psi = mu/2*(trace(F^T F)-3) - mu*log(J) + lambda/2*log(J)^2
            # First Piola Kirchhoff P = dpsi/dF
            J = detF
            F_inv_T = np.linalg.inv(F).T
            logJ = np.log(max(J, 1e-12))
            P = self.mu * (F - F_inv_T) + self.lam * logJ * F_inv_T
            # Force on nodes (negative gradient wrt p)
            H = -P @ Dm_inv.T
            # Distribute to nodes 1,2,3; node 0 gets negative sum
            f_tet = np.zeros((4, 3), dtype=float)
            f_tet[1] = H[:, 0]
            f_tet[2] = H[:, 1]
            f_tet[3] = H[:, 2]
            f_tet[0] = -f_tet[1] - f_tet[2] - f_tet[3]
            f[tet] += f_tet
        return f

    def _point_segment_distance(self, p: Array, a: Array, b: Array):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        d = p - closest
        dist = np.linalg.norm(d)
        return dist, d / (dist + 1e-12), closest

    def _contact_forces(self, position: Array, velocity: Array, time: float) -> Array:
        f = np.zeros_like(position)
        upper_segments = self._upper_segments_at(time)
        all_segments = [(seg, -1.0) for seg in self.lower_segments] + [(seg, 1.0) for seg in upper_segments]
        # Normal convention: obstacle normals point into the ball. Lower pins push up (-z), upper pins push down (+z)
        for seg in all_segments:
            a, b = seg
            for i, p in enumerate(position):
                d_vec = p - a
                ab = b - a
                t = np.dot(d_vec, ab) / np.dot(ab, ab)
                t = np.clip(t, 0.0, 1.0)
                closest = a + t * ab
                diff = p - closest
                dist = np.linalg.norm(diff)
                # contact radius + small margin
                if dist < self.ball_radius + self.pin_radius + 1e-4:
                    n = diff / (dist + 1e-12)
                    # push along normal to keep ball outside pins
                    penetration = self.ball_radius + self.pin_radius - dist
                    # Stiff penalty + damping
                    fn = self.contact_stiffness * penetration
                    vn = np.dot(velocity[i], n)
                    fd = -self.contact_damping * vn
                    force_mag = max(fn + fd, 0.0)
                    f[i] += force_mag * n
        return f

    def _step(self) -> None:
        dt = self.time_step
        # Explicit elastic forces and contact
        f_elastic = self._elastic_forces(self.position)
        f_contact = self._contact_forces(self.position, self.velocity, self.current_time)
        f = f_elastic + f_contact
        # Gravity
        f += self.mass[:, None] * self.gravity[None, :]
        # Damping
        f -= self.damping * self.velocity

        a = f / self.mass[:, None]
        self.velocity += a * dt
        self.position += self.velocity * dt
        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time - 1e-12:
            raise ValueError("advance_to time must be non-decreasing")
        while self.current_time < time - 1e-12:
            self._step()
        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        # Deformation gradient per element at current state
        F_array = np.zeros((len(self.tetrahedra), 3, 3), dtype=float)
        for i, tet in enumerate(self.tetrahedra):
            p = self.position[tet]
            Ds = np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1)
            F_array[i] = Ds @ self.Dm_inv[i]

        # Budget / external-force bookkeeping: single domain (the ball)
        G = 1
        # Gravity as external force on the ball
        ext_force = self.mass[:, None] * self.gravity[None, :]
        budget_external_force = ext_force.reshape(1, -1, 3)
        # Cumulative impulse: integral of gravity + contact (very rough approx)
        # We keep it simple: cumulative gravity impulse over time
        cumulative_impulse = (self.gravity * self.mass.sum() * time).reshape(1, 3)
        budget_domain_weights = np.ones((G, self.n_vertices), dtype=float)

        # Reference momentum: positive scale; use sqrt(mass * gravity * radius * mass) type scale
        ref_momentum = float(self.mass.sum() * np.linalg.norm(self.gravity) * self.ball_radius)
        if not np.isfinite(ref_momentum) or ref_momentum <= 0.0:
            ref_momentum = 1.0

        frame = StateFrame(
            time=float(time),
            position=np.asarray(self.position, dtype=float, copy=True),
            velocity=np.asarray(self.velocity, dtype=float, copy=True),
            mass=np.asarray(self.mass, dtype=float, copy=True),
            attributes={
                "mass_quantity": np.asarray(self.mass, dtype=float, copy=True),
                "rest_position": np.asarray(self.rest_position, dtype=float, copy=True),
            },
            fields={
                "element_deformation_gradient": F_array,
                "budget_cumulative_external_impulse": cumulative_impulse,
                "budget_cumulative_inflow_momentum": np.zeros((G, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((G, 3), dtype=float),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": ref_momentum,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_corotational_tetrahedral",
            "time_step": self.time_step,
            "substeps": int(round(self.current_time / self.time_step)),
            "seed": self.seed,
        }
