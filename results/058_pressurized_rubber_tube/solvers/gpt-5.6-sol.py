"""Axisymmetric finite-volume solver for a pressurized rubber tube."""
from __future__ import annotations
import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene, self.config = scene, scene.config
        initial = scene.initial_state()
        g = self.config["geometry"]
        self.n_theta = int(g["circumferential_segments"])
        self.n_z_cells = int(g["axial_segments"])
        self.n_ring = self.n_z_cells + 1
        self.inner0 = float(g["inner_radius"])
        self.thickness0 = float(g["wall_thickness"])
        self.outer0 = self.inner0 + self.thickness0
        self.mid0 = 0.5 * (self.inner0 + self.outer0)
        self.height = float(g["height"])
        self.dz = self.height / self.n_z_cells
        self.mu, self.damping_rate, self.max_step = 4.5e4, 11.0, 2.5e-4
        self.rest_position = np.asarray(initial.position, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.fixed = np.asarray(initial.attributes["fixed_rim"], dtype=np.int32).copy()
        self.pressure_surface = np.asarray(initial.attributes["pressure_surface"], dtype=np.int32).copy()
        self.surface_region = np.asarray(initial.attributes["surface_region"], dtype=np.int32).copy()
        self.q = np.full(self.n_ring, self.mid0, dtype=np.float64)
        self.v = np.zeros(self.n_ring, dtype=np.float64)
        wall_area = math.pi * (self.outer0**2 - self.inner0**2)
        self.ring_mass = np.full(self.n_ring, self.config["material"]["density"] * wall_area * self.dz)
        self.ring_mass[[0, -1]] *= 0.5
        self.dual_length = np.full(self.n_ring, self.dz)
        self.dual_length[[0, -1]] *= 0.5
        self.theta = 2.0 * math.pi * np.arange(self.n_theta) / self.n_theta
        self.cos_theta, self.sin_theta = np.cos(self.theta), np.sin(self.theta)
        self.z = np.linspace(-0.5 * self.height, 0.5 * self.height, self.n_ring)
        self.time, self.step_count, self.max_mid_radius = 0.0, 0, self.mid0
        pmax = float(self.config["boundary_conditions"]["cavity_pressure"]["pressure_pa"])
        self.reference_force = pmax * math.pi * self.inner0**2

    def _acceleration(self, time: float) -> np.ndarray:
        q, force = self.q, np.zeros_like(self.q)
        v0 = math.pi * (self.outer0**2 - self.inner0**2) * self.dz
        qa = 0.5 * (q[:-1] + q[1:])
        lam, slope = qa / self.mid0, (q[1:] - q[:-1]) / self.dz
        dmean, dslope = v0 * self.mu * (lam - lam ** -3) / self.mid0, v0 * self.mu * slope
        force[:-1] += -0.5 * dmean + dslope / self.dz
        force[1:] += -0.5 * dmean - dslope / self.dz
        thick = self.thickness0 * self.mid0 / q
        inner = q - 0.5 * thick
        dinner = 1.0 + 0.5 * self.thickness0 * self.mid0 / (q * q)
        force += self.scene.pressure_at(time) * 2.0 * math.pi * inner * dinner * self.dual_length
        force -= self.damping_rate * self.ring_mass * self.v
        acc = force / self.ring_mass
        acc[[0, -1]] = 0.0
        return acc

    def _step(self, dt: float) -> None:
        self.v += 0.5 * dt * self._acceleration(self.time)
        self.q += dt * self.v
        self.q[[0, -1]], self.v[[0, -1]] = self.mid0, 0.0
        self.v += 0.5 * dt * self._acceleration(self.time + dt)
        self.v[[0, -1]] = 0.0
        self.time += dt
        self.step_count += 1

    def _surface_state(self):
        thick = self.thickness0 * self.mid0 / self.q
        rate = -thick * self.v / self.q
        outer, inner = self.q + 0.5 * thick, self.q - 0.5 * thick
        outer_v, inner_v = self.v + 0.5 * rate, self.v - 0.5 * rate
        def layer(radius, speed):
            x = (radius[:, None] * self.cos_theta[None, :]).ravel()
            y = (radius[:, None] * self.sin_theta[None, :]).ravel()
            z = np.repeat(self.z, self.n_theta)
            vx = (speed[:, None] * self.cos_theta[None, :]).ravel()
            vy = (speed[:, None] * self.sin_theta[None, :]).ravel()
            return np.column_stack((x, y, z)), np.column_stack((vx, vy, np.zeros_like(vx)))
        ox, ov = layer(outer, outer_v)
        ix, iv = layer(inner, inner_v)
        return np.vstack((ox, ix)), np.vstack((ov, iv))

    def _element_deformation_gradients(self):
        qa = 0.5 * (self.q[:-1] + self.q[1:])
        lt, lr, shear = qa / self.mid0, self.mid0 / qa, (self.q[1:] - self.q[:-1]) / self.dz
        out, k = np.empty((self.n_z_cells * self.n_theta, 3, 3)), 0
        for j in range(self.n_z_cells):
            local = np.array([[lr[j], 0., shear[j]], [0., lt[j], 0.], [0., 0., 1.]])
            for c, s in zip(self.cos_theta, self.sin_theta):
                basis = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                out[k] = basis @ local @ basis.T
                k += 1
        return out

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time + 1.e-12 < target:
            self._step(min(self.max_step, target - self.time))
        self.time = target
        position, velocity = self._surface_state()
        self.max_mid_radius = max(self.max_mid_radius, float(self.q[self.n_ring // 2]))
        frame = StateFrame(
            time=target, position=position, velocity=velocity, mass=self.mass.copy(),
            attributes={"fixed_rim": self.fixed.copy(), "pressure_surface": self.pressure_surface.copy(),
                        "surface_region": self.surface_region.copy(), "rest_position": self.rest_position.copy(),
                        "mass_quantity": self.mass.copy()},
            fields={"budget_external_force": np.zeros((1, 2, 3)), "cumulative_mass_source": np.zeros(1),
                    "element_deformation_gradient": self._element_deformation_gradients()},
            scalars={"cavity_pressure_pa": float(self.scene.pressure_at(target)), "inner_radius": self.inner0,
                     "outer_radius": self.outer0, "wall_thickness": self.thickness0, "reference_force": self.reference_force})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "axisymmetric_incompressible_neo_hookean_finite_volume", "integration_steps": self.step_count,
                "shear_modulus_pa": self.mu, "maximum_mid_surface_radius_m": self.max_mid_radius}
