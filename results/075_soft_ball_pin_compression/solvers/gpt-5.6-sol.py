"""Physical solver for the soft ball / pin-array compression scene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic tetra-scaffold deformable-body integrator."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.rest = np.asarray(initial.position, dtype=float).copy()
        self.x = self.rest.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = len(self.x)
        self.t = 0.0
        self.cumulative_impulse = np.zeros(3, dtype=float)
        self.latest_external_force = np.sum(self.mass[:, None] * np.asarray(
            self.config["gravity"], dtype=float)[None, :], axis=0)
        asset = json.loads((Path(scene._root) / "assets" / "soft_ball_volume.json").read_text())
        self.tets = np.asarray(asset["tetrahedra"], dtype=int)
        edges = set()
        for tet in self.tets:
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.add(tuple(sorted((int(tet[a]), int(tet[b])))))
        self.edges = np.asarray(sorted(edges), dtype=int)
        d = self.rest[self.edges[:, 1]] - self.rest[self.edges[:, 0]]
        self.rest_lengths = np.linalg.norm(d, axis=1)
        self.dm_inv = []
        for tet in self.tets:
            p = self.rest[tet]
            dm = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
            self.dm_inv.append(np.linalg.inv(dm))
        self.dm_inv = np.asarray(self.dm_inv)
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.lower_tip = float(self.config["boundary_conditions"]["lower_pin_array"]["tip_height"])
        self.upper_axis = np.asarray(self.config["boundary_conditions"]["upper_pin_array"]["axis"], dtype=float)
        self.upper_keys = self.config["boundary_conditions"]["upper_pin_array"]["keyframes"]
        self.pin_xy = np.array([(a, b) for a in np.arange(-0.72, 0.7201, 0.18)
                                for b in np.arange(-0.72, 0.7201, 0.18)], dtype=float)
        self.spring_k, self.spring_d = 2600.0, 18.0
        self.drag = 1.8
        self.contact_k, self.contact_d, self.contact_radius = 18000.0, 45.0, 0.055
        self.sub_dt = 0.0015
        self.reference_momentum = max(float(np.sum(self.mass)) * np.sqrt(9.81 * 0.46), 1.0e-6)

    def _upper_displacement(self, time: float) -> float:
        keys = self.upper_keys
        if time <= float(keys[0]["time"]):
            return float(keys[0]["displacement"])
        for k0, k1 in zip(keys[:-1], keys[1:]):
            t0, t1 = float(k0["time"]), float(k1["time"])
            if time <= t1:
                u = (time - t0) / max(t1 - t0, 1e-12)
                return (1.0 - u) * float(k0["displacement"]) + u * float(k1["displacement"])
        return float(keys[-1]["displacement"])

    def _contact_forces(self, x: np.ndarray, v: np.ndarray, time: float) -> np.ndarray:
        forces = np.zeros_like(x)
        upper_shift = self.upper_axis * self._upper_displacement(time)
        for i, p in enumerate(x):
            vi = v[i]
            for px, py in self.pin_xy:
                if (p[0] - px) ** 2 + (p[1] - py) ** 2 > 0.22 ** 2:
                    continue
                for lo, hi in ((0.08, self.lower_tip),
                               (1.86 + upper_shift[2], 2.32 + upper_shift[2])):
                    zc = min(max(p[2], lo), hi)
                    dvec = p - np.array([px, py, zc])
                    dist = float(np.linalg.norm(dvec))
                    if dist < self.contact_radius:
                        nrm = dvec / max(dist, 1e-8)
                        rel = float(np.dot(vi, nrm))
                        forces[i] += (self.contact_k * (self.contact_radius - dist)
                                      - self.contact_d * min(rel, 0.0)) * nrm
        return forces

    def _step(self, dt: float) -> None:
        f = self.mass[:, None] * self.gravity[None, :]
        a, b = self.edges[:, 0], self.edges[:, 1]
        d = self.x[b] - self.x[a]
        length = np.linalg.norm(d, axis=1)
        nrm = d / np.maximum(length[:, None], 1e-9)
        extension = length - self.rest_lengths
        damp = np.sum((self.v[b] - self.v[a]) * nrm, axis=1)
        ef = (self.spring_k * extension + self.spring_d * damp)[:, None] * nrm
        np.add.at(f, a, ef)
        np.add.at(f, b, -ef)
        drag_force = -self.drag * self.v
        f += drag_force
        cf = self._contact_forces(self.x, self.v, self.t)
        f += cf
        self.v += dt * f / np.maximum(self.mass[:, None], 1e-9)
        speed = np.linalg.norm(self.v, axis=1)
        self.v *= np.minimum(1.0, 18.0 / np.maximum(speed, 18.0))[:, None]
        self.x += dt * self.v
        external = np.sum(self.mass[:, None] * self.gravity[None, :] + drag_force + cf, axis=0)
        self.latest_external_force = external
        self.cumulative_impulse += dt * external

    def _deformation_gradients(self) -> np.ndarray:
        out = np.empty((len(self.tets), 3, 3), dtype=float)
        for j, tet in enumerate(self.tets):
            p = self.x[tet]
            ds = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
            out[j] = ds @ self.dm_inv[j]
        return out

    def _frame(self) -> StateFrame:
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": self.latest_external_force.reshape(1, 1, 3).copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "element_deformation_gradient": self._deformation_gradients(),
        }
        frame = StateFrame(time=float(self.t), position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
                           attributes={"rest_position": self.rest.copy(), "mass_quantity": self.mass.copy()},
                           fields=fields, scalars={"reference_momentum": float(self.reference_momentum)})
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-10:
            raise ValueError("advance_to requires nondecreasing time")
        while self.t + 1e-12 < target:
            dt = min(self.sub_dt, target - self.t)
            self._step(dt)
            self.t += dt
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler", "sub_dt": self.sub_dt, "vertices": self.n}
