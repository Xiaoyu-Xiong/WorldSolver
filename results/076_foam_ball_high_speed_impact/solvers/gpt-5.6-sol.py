"""Deterministic deformable-ball impact solver using springs and penalty contact."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._time = 0.0
        self._position = np.empty((0, 3), dtype=np.float64)
        self._velocity = np.empty((0, 3), dtype=np.float64)
        self._mass = np.empty(0, dtype=np.float64)
        self._rest_position = np.empty((0, 3), dtype=np.float64)
        self._edges = np.empty((0, 2), dtype=np.int64)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._rest_lengths = np.empty(0, dtype=np.float64)
        self._impulse = np.zeros(3, dtype=np.float64)
        self._last_external = np.zeros((2, 3), dtype=np.float64)
        self._total_mass = 1.0
        self._reference_momentum = 1.0
        self._gravity = np.array([0.0, 0.0, -9.81])

    def initialize(self, scene, seed: int) -> None:
        del seed
        cfg = scene.config
        ball = cfg["foam_ball"]
        radius = float(ball["radius"])
        center = np.asarray(ball["initial_center"], dtype=np.float64)
        self._total_mass = float(ball["density"]) * (4.0 / 3.0) * np.pi * radius**3
        self._gravity = np.asarray(cfg["initial_conditions"]["gravity"], dtype=np.float64)
        verts = np.array([
            [-.525731,.850651,0],[.525731,.850651,0],[-.525731,-.850651,0],[.525731,-.850651,0],
            [0,-.525731,.850651],[0,.525731,.850651],[0,-.525731,-.850651],[0,.525731,-.850651],
            [.850651,0,-.525731],[.850651,0,.525731],[-.850651,0,-.525731],[-.850651,0,.525731]], dtype=np.float64)
        verts /= np.linalg.norm(verts, axis=1)[:, None]
        self._rest_position = center + radius * verts
        self._position = self._rest_position.copy()
        self._velocity = np.broadcast_to(np.asarray(ball["initial_velocity"], dtype=np.float64), (12,3)).copy()
        self._mass = np.full(12, self._total_mass / 12.0)
        self._faces = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], dtype=np.int64)
        self._edges = np.unique(np.sort(np.vstack((self._faces[:,[0,1]], self._faces[:,[1,2]], self._faces[:,[2,0]])), axis=1), axis=0)
        self._rest_lengths = np.linalg.norm(self._rest_position[self._edges[:,1]] - self._rest_position[self._edges[:,0]], axis=1)
        self._time = 0.0; self._impulse.fill(0.0); self._last_external.fill(0.0)
        self._last_external[0] = self._total_mass * self._gravity
        self._reference_momentum = max(self._total_mass * np.linalg.norm(self._velocity.mean(axis=0)), 1e-12)

    def _step(self, dt: float) -> None:
        n = len(self._position)
        force = self._mass[:,None] * self._gravity[None,:]
        elastic = np.zeros((n,3)); k = 1.0e7 * 0.02; c = 0.015 * 2*np.sqrt(k*self._total_mass/n)
        for q,(i,j) in enumerate(self._edges):
            d = self._position[j]-self._position[i]; length = np.linalg.norm(d)
            if length < 1e-12: continue
            direction = d/length; rel = np.dot(self._velocity[j]-self._velocity[i], direction)
            f = (k*(length-self._rest_lengths[q]) + c*rel)*direction
            elastic[i] += f; elastic[j] -= f
        force += elastic
        contact = np.zeros((n,3)); penetration = np.maximum(self._position[:,0], 0.0); active = penetration > 0
        kc = 3.0e6; cc = 0.08 * 2*np.sqrt(kc*self._total_mass/n)
        contact[active,0] = -kc*penetration[active] - cc*np.maximum(self._velocity[active,0], 0.0)
        force += contact
        self._velocity += dt * force / self._mass[:,None]
        self._position += dt*self._velocity
        gravity_force = np.sum(self._mass[:,None]*self._gravity[None,:], axis=0)
        self._impulse += (gravity_force + np.sum(contact, axis=0))*dt
        self._last_external[0] = gravity_force; self._last_external[1] = np.sum(contact, axis=0)

    def _deformation_gradient(self) -> np.ndarray:
        rest_center = self._rest_position.mean(axis=0)
        center = self._position.mean(axis=0)
        gradients = np.empty((len(self._faces), 3, 3), dtype=np.float64)
        for q, face in enumerate(self._faces):
            dm = (self._rest_position[face] - rest_center).T
            ds = (self._position[face] - center).T
            try: gradients[q] = ds @ np.linalg.inv(dm)
            except np.linalg.LinAlgError: gradients[q] = np.eye(3)
        gradients[~np.isfinite(gradients)] = 0.0
        return gradients

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1e-12: raise ValueError("advance_to requires finite nondecreasing time")
        while self._time < target - 1e-15:
            dt = min(2.0e-6, target-self._time); self._step(dt); self._time += dt
        n = len(self._position)
        frame = StateFrame(time=self._time, position=self._position.copy(), velocity=self._velocity.copy(), mass=self._mass.copy(), attributes={"rest_position":self._rest_position.copy(), "mass_quantity":self._mass.copy()}, fields={
            "budget_cumulative_external_impulse":self._impulse[None,:].copy(), "budget_cumulative_inflow_momentum":np.zeros((1,3)), "budget_cumulative_outflow_momentum":np.zeros((1,3)), "budget_domain_weights":np.ones((1,n)), "budget_external_force":self._last_external[None,:,:].copy(), "cumulative_mass_source":np.zeros(1), "element_deformation_gradient":self._deformation_gradient()}, scalars={"reference_momentum":self._reference_momentum})
        frame.validate(); return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator":"semi_implicit_euler_edge_springs_penalty_contact", "particle_count":int(len(self._position))}
