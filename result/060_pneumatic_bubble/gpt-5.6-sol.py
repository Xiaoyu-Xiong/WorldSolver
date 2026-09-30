"""Pressure-driven axisymmetric solid solver for the pneumatic bubble."""
from __future__ import annotations
import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.config = scene.config
        self.x0_vertices = np.asarray(initial.position, dtype=float).copy()
        self.faces = np.asarray(initial.fields["surface_faces"], dtype=np.int32).copy()
        self.vertex_mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.fixed_vertices = np.asarray(initial.attributes["fixed"], dtype=float).copy()
        self.nr = 7
        self.ntheta = len(self.x0_vertices) // (2 * self.nr)
        self.nn = 2 * self.nr
        self.vertex_node = np.repeat(np.arange(self.nn, dtype=np.int32), self.ntheta)
        radius = np.hypot(self.x0_vertices[:, 0], self.x0_vertices[:, 1])
        self.cos_theta = self.x0_vertices[:, 0] / radius
        self.sin_theta = self.x0_vertices[:, 1] / radius
        self.X = np.zeros((self.nn, 2), dtype=float)
        for node in range(self.nn):
            sl = slice(node * self.ntheta, (node + 1) * self.ntheta)
            self.X[node] = (np.mean(radius[sl]), np.mean(self.x0_vertices[sl, 2]))
        self.x = self.X.copy()
        self.v = np.zeros_like(self.x)
        self.node_mass = np.bincount(self.vertex_node, weights=self.vertex_mass, minlength=self.nn)
        self.fixed_nodes = np.array([np.all(self.fixed_vertices[self.vertex_node == i] > .5) for i in range(self.nn)])
        self.elements = np.asarray([[j, j + 1, self.nr + j + 1, self.nr + j] for j in range(self.nr - 1)], dtype=np.int32)
        self._prepare_quadrature()
        mat = self.config.get("material_reference", {})
        E, nu = float(mat.get("youngs_modulus_pa", 1e5)), float(mat.get("poisson_ratio", .499))
        self.mu = E / (2 * (1 + nu))
        self.lam = min(E * nu / ((1 + nu) * (1 - 2 * nu)), 8 * self.mu)
        loading = self.config["loading"]
        self.pressure_max = float(loading["pressure_pa"])
        self.ramp_start, self.ramp_end = float(loading["ramp_start_s"]), float(loading["ramp_end_s"])
        self.hold_until = float(loading.get("hold_until_s", self.config["duration"]))
        self.reference_force = self.pressure_max * math.pi * float(np.max(self.X[self.nr:, 0]) ** 2)
        self.dt_max, self.damping = 2e-5, 15.0
        self.time, self.step_count = 0.0, 0

    def _prepare_quadrature(self):
        a = 1 / math.sqrt(3)
        self.quadrature = []
        for elem in self.elements:
            Xe = self.X[elem]
            for xi, eta in ((-a, -a), (a, -a), (a, a), (-a, a)):
                N = .25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
                dN = .25 * np.array([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)], [(1+eta), (1+xi)], [-(1+eta), (1-xi)]])
                J0 = Xe.T @ dN
                grad = dN @ np.linalg.inv(J0)
                self.quadrature.append((elem, N, grad, 2 * math.pi * float(N @ Xe[:, 0]) * np.linalg.det(J0)))

    def _pressure(self, t):
        if t <= self.ramp_start: return 0.0
        if t < self.ramp_end: return self.pressure_max * (t-self.ramp_start)/(self.ramp_end-self.ramp_start)
        return self.pressure_max

    def _elastic_force(self):
        force = np.zeros_like(self.x)
        for elem, N, grad, weight in self.quadrature:
            F2 = self.x[elem].T @ grad
            R, r = float(N @ self.X[elem, 0]), float(N @ self.x[elem, 0])
            F = np.array([[F2[0,0], 0, F2[0,1]], [0, r/R, 0], [F2[1,0], 0, F2[1,1]]])
            J = max(float(np.linalg.det(F)), 1e-8)
            FinvT = np.linalg.inv(F).T
            P = self.mu * (F-FinvT) + self.lam * math.log(J) * FinvT
            for q, node in enumerate(elem):
                gR, gZ = grad[q]
                force[node, 0] -= weight * (P[0,0]*gR + P[0,2]*gZ + P[1,1]*N[q]/R)
                force[node, 1] -= weight * (P[2,0]*gR + P[2,2]*gZ)
        return force

    def _pressure_force(self, pressure):
        force = np.zeros_like(self.x)
        for a, b in zip(range(self.nr, 2*self.nr-1), range(self.nr+1, 2*self.nr)):
            ra, za = self.x[a]; rb, zb = self.x[b]
            resultant = pressure * math.pi * (ra+rb) * np.array([zb-za, -(rb-ra)])
            force[a] += .5 * resultant; force[b] += .5 * resultant
        return force

    def _total_force(self, t):
        f = self._elastic_force() + self._pressure_force(self._pressure(t)) - self.damping * self.v
        f[self.fixed_nodes] = 0
        return f

    def _step(self, h):
        f = self._total_force(self.time)
        free = ~self.fixed_nodes
        self.v[free] += h * f[free] / self.node_mass[free, None]
        self.x[free] += h * self.v[free]
        self.x[self.fixed_nodes] = self.X[self.fixed_nodes]; self.v[self.fixed_nodes] = 0
        self.time += h; self.step_count += 1

    def _element_gradients(self):
        out = np.empty((len(self.elements), 3, 3))
        dN = .25 * np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        for k, elem in enumerate(self.elements):
            grad = dN @ np.linalg.inv(self.X[elem].T @ dN)
            F2 = self.x[elem].T @ grad; R = np.mean(self.X[elem,0]); r = np.mean(self.x[elem,0])
            out[k] = [[F2[0,0],0,F2[0,1]], [0,r/R,0], [F2[1,0],0,F2[1,1]]]
        return out

    def _make_frame(self):
        node = self.vertex_node
        pos, vel = np.empty_like(self.x0_vertices), np.empty_like(self.x0_vertices)
        pos[:,0] = self.x[node,0] * self.cos_theta; pos[:,1] = self.x[node,0] * self.sin_theta; pos[:,2] = self.x[node,1]
        vel[:,0] = self.v[node,0] * self.cos_theta; vel[:,1] = self.v[node,0] * self.sin_theta; vel[:,2] = self.v[node,1]
        pressure_force = self._pressure_force(self._pressure(self.time))
        damping_force = -self.damping * self.v
        elastic_force = self._elastic_force()
        support_force = np.zeros_like(self.x)
        support_force[self.fixed_nodes] = -(elastic_force + pressure_force + damping_force)[self.fixed_nodes]
        # Axisymmetric radial components cancel around each complete ring.
        budget = np.zeros((1, 3, 3), dtype=float)
        budget[0, 0, 2] = np.sum(pressure_force[:, 1])
        budget[0, 1, 2] = np.sum(damping_force[:, 1])
        budget[0, 2, 2] = np.sum(support_force[:, 1])
        frame = StateFrame(time=float(self.time), position=pos, velocity=vel, mass=self.vertex_mass.copy(), body_id=self.body_id.copy(),
            attributes={"fixed": self.fixed_vertices.copy(), "mass_quantity": self.vertex_mass.copy()},
            fields={"surface_faces": self.faces.copy(), "budget_external_force": budget, "cumulative_mass_source": np.zeros(1), "element_deformation_gradient": self._element_gradients()},
            scalars={"internal_pressure_pa": float(self._pressure(self.time)), "reference_force": float(self.reference_force)})
        frame.validate(); return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12: raise ValueError("sample times must be nondecreasing")
        while self.time < target - 1e-14: self._step(min(self.dt_max, target-self.time))
        self.time = target
        return self._make_frame()

    def diagnostics(self):
        return {"model": "axisymmetric_neo_hookean", "steps": int(self.step_count), "maximum_radial_displacement_m": float(np.max(self.x[:,0]-self.X[:,0])), "finite": bool(np.all(np.isfinite(self.x)))}
