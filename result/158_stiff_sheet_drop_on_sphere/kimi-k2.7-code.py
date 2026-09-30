"""Solver Agent-owned implementation.

Implements a runnable cloth solver for a stiff sheet dropped onto a fixed sphere.
The solver advances physics with an explicit time-stepping scheme, produces
finite StateFrame samples at the requested times, and exports all required
trajectory fields.
"""
from __future__ import annotations

import json
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Simple cloth solver for a stiff sheet dropped on a sphere.

    Uses a corotational linear-elastic membrane model on a fixed triangle mesh
    with an explicit second-order velocity-Verlet-style time step.  Contact
    against the fixed sphere is handled by a penalty force plus a position
    projection (post-stabilization).  A small amount of Rayleigh damping keeps
    the explicit integration stable for the chosen small time step.
    """

    def __init__(self):
        self.time = 0.0
        self.position = None
        self.velocity = None
        self.mass = None
        self.faces = None
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.sphere_center = np.array([0.0, 0.0, 0.68])
        self.sphere_radius = 0.62
        self.thickness = 0.015
        self.young = 1.0e7
        self.poisson = 0.35
        self.damping = 0.002
        self.dt = 1e-4
        self.contact_stiffness = 1.0e7
        self.contact_damping = 500.0
        self.boundary_mask = None
        self.vertex_mass = None
        self._cached_frame = None
        self._cumulative_mass_source = 0.0

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        frame = scene.initial_state()
        self.time = float(frame.time)
        self.position = np.asarray(frame.position, dtype=float).copy()
        self.velocity = np.asarray(frame.velocity, dtype=float).copy()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.boundary_mask = np.asarray(frame.attributes["boundary_vertex"], dtype=bool)
        with open("assets/cloth_topology.json", "r") as f:
            topo = json.load(f)
        self.faces = np.asarray(topo["faces"], dtype=np.int32)
        if self.vertex_mass is None:
            self.vertex_mass = self.mass.copy()
        self._cache_frame()

    def diagnostics(self) -> dict[str, object]:
        return {"time": self.time, "vertices": len(self.position)}

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time - 1e-12:
            raise ValueError("advance_to can only move forward in time")
        while self.time < time - 1e-12:
            self._step()
        self._cache_frame()
        return self._cached_frame

    def _step(self):
        dt = self.dt
        # External forces (gravity + damping + contact penalty)
        ext = self._external_forces(self.position, self.velocity)
        # Internal elastic forces
        f_int = self._elastic_forces(self.position)
        f_total = ext - f_int
        # Symplectic Euler update with boundary pinning
        a = f_total / self.vertex_mass[:, None]
        self.velocity = self.velocity + dt * a
        self.velocity[self.boundary_mask] = 0.0
        self.position = self.position + dt * self.velocity
        self.time += dt
        # Post-stabilize sphere collision by projection
        self._project_sphere_contact()
        self.velocity[self.boundary_mask] = 0.0

    def _cache_frame(self):
        self._external_force = self._external_forces(self.position, self.velocity)
        contact_signed = self._signed_sphere_distance(self.position)
        deformation_gradient = self._compute_deformation_gradient(self.position)
        self._cached_frame = StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.vertex_mass.copy(),
            attributes={
                "mass_quantity": self.vertex_mass.copy(),
                "deformation_gradient": deformation_gradient.copy(),
            },
            fields={
                "cumulative_mass_source": np.array([self._cumulative_mass_source], dtype=float),
                "external_force": self._external_force.copy(),
                "contact_signed_distance": contact_signed.copy(),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _signed_sphere_distance(self, x):
        d = np.linalg.norm(x - self.sphere_center[None, :], axis=1)
        return d - self.sphere_radius

    def _project_sphere_contact(self):
        disp = self.position - self.sphere_center
        d = np.linalg.norm(disp, axis=1)
        mask = d < (self.sphere_radius + self.thickness * 0.5)
        if np.any(mask):
            n = disp[mask] / d[mask, None]
            target = self.sphere_radius + self.thickness * 0.5
            self.position[mask] = self.sphere_center + target * n

    # ------------------------------------------------------------------
    # Elastic force model (corotational linear membrane)
    # ------------------------------------------------------------------
    def _elastic_forces(self, x):
        f = np.zeros_like(x)
        mu = self.young / (2.0 * (1.0 + self.poisson))
        lam = self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - self.poisson))
        for face in self.faces:
            p0 = x[face[0]]
            p1 = x[face[1]]
            p2 = x[face[2]]
            e1 = p1 - p0
            e2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            if area == 0:
                continue
            # Local 2D basis in the triangle plane
            ex = e1 / (np.linalg.norm(e1) + 1e-12)
            ey = e2 - np.dot(e2, ex) * ex
            ey = ey / (np.linalg.norm(ey) + 1e-12)
            # Material frame: flat cloth lies in x-y; use x,y as material coords.
            # F is 2x2: columns are partials of current local coords wrt material coords.
            # Map material basis vectors to current local coords.
            # Material edge vectors in x-y correspond to spatial e1 and e2 projections.
            F11 = np.dot(e1, ex)
            F21 = np.dot(e1, ey)
            F12 = np.dot(e2, ex)
            F22 = np.dot(e2, ey)
            F = np.array([[F11, F12], [F21, F22]])
            # Green strain
            G = 0.5 * (F.T @ F - np.eye(2))
            S = lam * np.trace(G) * np.eye(2) + 2.0 * mu * G
            P = F @ S
            # Forces at nodes from energy W = area/2 * ||F||_S^2 (simplified)
            # For each vertex, contribution is -area * P * dF/dx_i.
            # Approximate nodal forces using the columns of P mapped to 3D.
            g1 = P[0, 0] * ex + P[1, 0] * ey
            g2 = P[0, 1] * ex + P[1, 1] * ey
            f[face[0]] += -area * (g1 + g2)
            f[face[1]] += area * g1
            f[face[2]] += area * g2
        return f

    # ------------------------------------------------------------------
    # External forces (gravity + damping + contact penalty)
    # ------------------------------------------------------------------
    def _external_forces(self, x, v):
        f = self.vertex_mass[:, None] * self.gravity
        # Damping proportional to mass
        f -= self.damping * self.vertex_mass[:, None] * v
        # Sphere contact penalty
        disp = x - self.sphere_center
        d = np.linalg.norm(disp, axis=1)
        mask = d < (self.sphere_radius + self.thickness)
        if np.any(mask):
            n = disp[mask] / d[mask, None]
            penetration = self.sphere_radius + self.thickness - d[mask]
            vn = np.einsum("ij,ij->i", v[mask], n)
            f[mask] += (self.contact_stiffness * penetration[:, None] -
                        self.contact_damping * vn[:, None] * n)
        return f

    # ------------------------------------------------------------------
    # Per-vertex deformation gradient in material coordinates (2x2)
    # ------------------------------------------------------------------
    def _compute_deformation_gradient(self, x):
        n = len(x)
        F = np.zeros((n, 2, 2), dtype=float)
        # For each vertex, area-weight average of adjacent triangle deformations
        vertex_F = [np.zeros((2, 2), dtype=float) for _ in range(n)]
        vertex_area = np.zeros(n, dtype=float)
        for face in self.faces:
            p0 = x[face[0]]
            p1 = x[face[1]]
            p2 = x[face[2]]
            e1 = p1 - p0
            e2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            ex = e1 / (np.linalg.norm(e1) + 1e-12)
            ey = e2 - np.dot(e2, ex) * ex
            ey = ey / (np.linalg.norm(ey) + 1e-12)
            # Deformation gradient from material (x-y) to current local plane
            F11 = np.dot(e1, ex)
            F21 = np.dot(e1, ey)
            F12 = np.dot(e2, ex)
            F22 = np.dot(e2, ey)
            F_tri = np.array([[F11, F12], [F21, F22]])
            for vi in face:
                vertex_F[vi] += F_tri * area
                vertex_area[vi] += area
        for i in range(n):
            if vertex_area[i] > 0:
                F[i] = vertex_F[i] / vertex_area[i]
            else:
                F[i] = np.eye(2)
        return F
