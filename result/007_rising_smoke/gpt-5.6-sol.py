"""Deterministic buoyant-smoke solver using a stable-fluids scheme."""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-Lagrangian density transport with buoyancy and pressure projection."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        dom, initial = cfg["domain"], cfg["initial_conditions"]
        source = initial["smoke_source"]
        self.out_shape = tuple(map(int, dom["grid_shape"]))
        # Coarsen only camera-depth y; preserve the entire visible x-z plane.
        self.y_scale = 2 if self.out_shape[1] % 2 == 0 else 1
        self.shape = (self.out_shape[0], self.out_shape[1] // self.y_scale,
                      self.out_shape[2])
        self.low = np.asarray(dom["bounds_min"], np.float32)
        self.high = np.asarray(dom["bounds_max"], np.float32)
        self.h = (self.high - self.low) / np.asarray(self.shape, np.float32)
        self.out_h = (self.high - self.low) / np.asarray(self.out_shape, np.float32)
        axes = [self.low[i] + (np.arange(self.shape[i], dtype=np.float32) + .5) * self.h[i]
                for i in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
        center = np.asarray(source["center"], np.float32)
        q = (((x-center[0])**2 + (y-center[1])**2) / float(source["radius"])**2
             + (z-center[2])**2 / float(source["half_height"])**2)
        self.density = np.asarray(float(source["peak_density"]) * np.exp(-2.5*q),
                                  np.float32)
        self.density[self.density < 1.e-4] = 0.
        envelope = np.clip(self.density / float(source["peak_density"]), 0., 1.)
        self.velocity = envelope[..., None] * np.asarray(source["initial_velocity"], np.float32)
        self.velocity = np.asarray(self.velocity, np.float32)
        amp = float(initial["perturbation_amplitude"])
        phase = (int(seed) % 360) * np.pi / 180.
        self.velocity[..., 0] += amp * envelope * np.sin(2*np.pi*y/(self.high[1]-self.low[1])+phase)
        self.velocity[..., 1] += amp * envelope * np.sin(2*np.pi*x/(self.high[0]-self.low[0])-phase)

        self.time, self.steps = 0., 0
        self.center, self.radius = center, float(source["radius"])
        self.temp = float(source["temperature_excess"])
        self.decay = np.log(2.) / float(initial["density_half_life"])
        self.buoyancy, self.vort_strength = 1.35*self.temp, 4.0
        self.cell_volume = float(np.prod(self.h))
        self.removed_mass = 0.
        self.coords = np.indices(self.shape, dtype=np.float32)
        self.departure = np.empty((3,) + self.shape, np.float32)

        # DCT diagonalization of the Neumann discrete Laplacian.
        eig = np.zeros(self.shape, np.float32)
        for axis, (n, spacing) in enumerate(zip(self.shape, self.h)):
            values = 2*(np.cos(np.pi*np.arange(n, dtype=np.float32)/n)-1)/(spacing*spacing)
            rs = [1, 1, 1]
            rs[axis] = n
            eig += values.reshape(rs)
        eig[0, 0, 0] = 1.
        self.poisson_eig = eig
        self.active_mask = np.ones(self.out_shape, np.uint8)
        self.last_divergence = 0.

    @staticmethod
    def _d(a, h, axis):
        return np.gradient(a, h, axis=axis, edge_order=1)

    def _set_departure(self, vel, dt):
        for a in range(3):
            np.subtract(self.coords[a], dt/float(self.h[a])*vel[..., a],
                        out=self.departure[a])

    def _advect(self, a):
        return map_coordinates(a, self.departure, order=1, mode="nearest",
                               prefilter=False).astype(np.float32, copy=False)

    def _boundaries(self, vel):
        # x-min, y walls, and floor are closed; x-max and z-max are outflow.
        vel[0, :, :, 0] = np.maximum(vel[0, :, :, 0], 0.)
        vel[-1, :, :, 0] = np.maximum(vel[-1, :, :, 0], 0.)
        vel[:, 0, :, 1] = vel[:, -1, :, 1] = 0.
        vel[:, :, 0, 2] = np.maximum(vel[:, :, 0, 2], 0.)
        vel[:, :, -1, 2] = np.maximum(vel[:, :, -1, 2], 0.)

    def _vorticity_force(self, vel):
        hx, hy, hz = map(float, self.h)
        u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
        wx = self._d(w, hy, 1) - self._d(v, hz, 2)
        wy = self._d(u, hz, 2) - self._d(w, hx, 0)
        wz = self._d(v, hx, 0) - self._d(u, hy, 1)
        mag = np.sqrt(wx*wx + wy*wy + wz*wz)
        nx, ny, nz = self._d(mag, hx, 0), self._d(mag, hy, 1), self._d(mag, hz, 2)
        inv = 1/np.maximum(np.sqrt(nx*nx + ny*ny + nz*nz), 1.e-6)
        nx *= inv; ny *= inv; nz *= inv
        s = self.vort_strength * min(hx, hy, hz)
        return (s*(ny*wz-nz*wy), s*(nz*wx-nx*wz), s*(nx*wy-ny*wx))

    def _divergence(self, vel):
        div = np.zeros(self.shape, np.float32)
        for a in range(3):
            c = vel[..., a]
            first, upper, lower = [slice(None)]*3, [slice(None)]*3, [slice(None)]*3
            first[a], upper[a], lower[a] = 0, slice(1, None), slice(None, -1)
            div[tuple(first)] += c[tuple(first)] / self.h[a]
            div[tuple(upper)] += (c[tuple(upper)]-c[tuple(lower)]) / self.h[a]
        return div

    def _project(self, vel):
        div = self._divergence(vel)
        rhs = div - np.mean(div, dtype=np.float64)
        phat = dctn(rhs, type=2, norm="ortho", workers=1)
        phat /= self.poisson_eig
        phat[0, 0, 0] = 0.
        p = idctn(phat, type=2, norm="ortho", workers=1)
        for a in range(3):
            lo, hi = [slice(None)]*3, [slice(None)]*3
            lo[a], hi[a] = slice(None, -1), slice(1, None)
            target = tuple(lo) + (a,)
            vel[target] -= (p[tuple(hi)]-p[tuple(lo)]) / self.h[a]
        self._boundaries(vel)
        div = self._divergence(vel)
        self.last_divergence = float(np.sqrt(np.mean(div*div)))

    def _step(self, dt):
        self._set_departure(self.velocity, dt)
        vel = np.empty_like(self.velocity)
        for a in range(3):
            vel[..., a] = self._advect(self.velocity[..., a])
        density = self._advect(self.density)
        np.maximum(density, 0., out=density)
        before = float(np.sum(density, dtype=np.float64)*self.cell_volume)
        factor = float(np.exp(-self.decay*dt))
        density *= factor
        self.removed_mass += before*(1-factor)
        vel[..., 2] += dt*self.buoyancy*density
        force = self._vorticity_force(vel)
        for a in range(3):
            vel[..., a] += dt*force[a]
        vel *= float(np.exp(-.025*dt))
        np.clip(vel, -5., 5., out=vel)
        self._boundaries(vel)
        self._project(vel)
        self.density, self.velocity = density, vel
        self.steps += 1

    def _expand(self, a):
        return a.copy() if self.y_scale == 1 else np.repeat(a, self.y_scale, axis=1)

    def _frame(self, time):
        density, velocity = self._expand(self.density), self._expand(self.velocity)
        mass = float(np.sum(self.density, dtype=np.float64)*self.cell_volume)
        total = float(np.sum(self.density, dtype=np.float64))
        if total > 1.e-12:
            axes = [self.low[i]+(np.arange(self.shape[i], dtype=np.float32)+.5)*self.h[i]
                    for i in range(3)]
            pos = np.asarray([np.sum(self.density*axes[0][:, None, None])/total,
                              np.sum(self.density*axes[1][None, :, None])/total,
                              np.sum(self.density*axes[2][None, None, :])/total], np.float64)
            mean_vel = np.asarray([np.sum(self.density*self.velocity[..., i])/total
                                   for i in range(3)], np.float64)
        else:
            pos, mean_vel = self.center.astype(np.float64), np.zeros(3)
        frame = StateFrame(
            float(time), pos.reshape(1, 3), mean_vel.reshape(1, 3),
            mass=np.asarray([mass], np.float64),
            attributes={"density": np.asarray([float(np.max(density))], np.float32),
                        "temperature_excess": np.asarray([self.temp], np.float32),
                        "mass_quantity": np.asarray([mass], np.float64)},
            fields={"density": density, "velocity": velocity,
                    "active_mask": self.active_mask,
                    "cumulative_mass_source": np.asarray([-self.removed_mass], np.float64),
                    "scalar_source": np.asarray(-self.decay*density, np.float32),
                    "transported_scalar": density, "velocity_field": velocity},
            scalars={"ambient_density": 0., "characteristic_length": 2*self.radius,
                     "grid_spacing": float(np.min(self.out_h))})
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time-1.e-10:
            raise ValueError("advance_to cannot move backward")
        while self.time < target-1.e-10:
            dt = min(.02, target-self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"method": "semi_lagrangian_buoyancy_projection",
                "internal_grid_shape": list(self.shape), "steps": self.steps,
                "time": self.time,
                "density_mass": float(np.sum(self.density)*self.cell_volume),
                "rms_divergence": self.last_divergence}
