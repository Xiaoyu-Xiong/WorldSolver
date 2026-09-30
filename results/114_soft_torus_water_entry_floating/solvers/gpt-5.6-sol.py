"""Deterministic deformable-torus / shallow-water solver."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.scene = scene
        self.rest = np.asarray(initial.position, dtype=float).copy()
        self.pos = self.rest.copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.nt = int(np.count_nonzero(self.body == 1)); self.n = len(self.rest); self.nw = self.n - self.nt
        self.torus_local = self.rest[:self.nt] - np.array([0., 0., .66])
        self.nx = self.ny = 24; self.nz = self.nw // (self.nx * self.ny); self.dx = .04
        self.eta = np.zeros((self.nx, self.ny)); self.eta_v = np.zeros_like(self.eta)
        self.ux = np.zeros_like(self.eta); self.uy = np.zeros_like(self.eta)
        self.center = np.array([0., 0., .66]); self.center_v = np.array([0., 0., -.15])
        self.qz = 0.; self.qz_v = 0.; self.qr = 0.; self.qr_v = 0.; self.time = 0.; self.last_dt = .02
        self.rho_water = 1000.; self.rho_torus = 650.; self.gravity = -9.81
        self.torus_volume = 2. * math.pi**2 * .255 * .105**2; self.torus_mass = self.rho_torus * self.torus_volume
        self.water_mass = float(np.sum(self.mass[self.nt:]))
        self.impulse = np.zeros((2, 3)); self.interaction = np.zeros((2, 3))
        self.external_terms = np.zeros((2, 2, 3))
        self.external_terms[0, 0, 2] = -self.torus_mass * 9.81
        self.external_terms[1, 0, 2] = -self.water_mass * 9.81
        self.external_terms[1, 1, 2] = self.water_mass * 9.81
        self._update_particles()

    def _grid_index(self, x, y):
        return (int(np.clip(round((x + .46) / self.dx), 0, self.nx - 1)), int(np.clip(round((y + .46) / self.dx), 0, self.ny - 1)))

    def _lap(self, a):
        p = np.pad(a, 1, mode="edge")
        return (p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1] - 4. * a) / (self.dx * self.dx)

    def _update_particles(self):
        sx = 1. + self.qr; sz = max(.72, 1. + self.qz)
        local = self.torus_local.copy(); local[:, :2] *= sx; local[:, 2] *= sz
        self.pos[:self.nt] = self.center + local
        lv = np.empty_like(local); lv[:, :2] = self.torus_local[:, :2] * self.qr_v; lv[:, 2] = self.torus_local[:, 2] * self.qz_v
        self.vel[:self.nt] = self.center_v + lv
        w = self.rest[self.nt:].copy(); vv = np.zeros_like(w)
        for ix in range(self.nx):
            for iy in range(self.ny):
                sl = (ix * self.ny + iy) * self.nz; ids = slice(sl, sl + self.nz)
                frac = np.arange(self.nz, dtype=float) / max(1, self.nz - 1)
                w[ids, 2] += self.eta[ix, iy] * frac
                w[ids, 0] += self.ux[ix, iy] * self.last_dt * frac; w[ids, 1] += self.uy[ix, iy] * self.last_dt * frac
                vv[ids, 0] = self.ux[ix, iy] * frac; vv[ids, 1] = self.uy[ix, iy] * frac; vv[ids, 2] = self.eta_v[ix, iy] * frac
        w[:, :2] = np.clip(w[:, :2], -.49, .49); w[:, 2] = np.clip(w[:, 2], .015, .405)
        self.pos[self.nt:] = w; self.vel[self.nt:] = vv

    def _step(self, dt):
        old_water_momentum = np.sum(self.mass[self.nt:, None] * self.vel[self.nt:], axis=0)
        ix, iy = self._grid_index(self.center[0], self.center[1]); surface = .4 + float(self.eta[ix, iy])
        sz = max(.72, 1. + self.qz); torus_z = self.center[2] + self.torus_local[:, 2] * sz
        submerged = float(np.clip(np.mean(torus_z < surface), 0., 1.))
        buoy = self.rho_water * 9.81 * self.torus_volume * submerged; weight = self.torus_mass * 9.81
        relz = self.center_v[2] - float(self.eta_v[ix, iy]); drag = -85. * submerged * relz; pressure = 130. * float(self.eta[ix, iy])
        fz = buoy - weight + drag - pressure; self.center_v[2] += (fz / self.torus_mass) * dt; self.center[2] += self.center_v[2] * dt
        min_z = .025 + .105 * sz
        if self.center[2] < min_z:
            self.center[2] = min_z
            if self.center_v[2] < 0.: self.center_v[2] *= -.25
        self.center[:2] = np.clip(self.center[:2], -.28, .28)
        impact = submerged * max(0., -self.center_v[2])
        self.qz_v += (-42. * self.qz - 7. * self.qz_v - 1.8 * impact) * dt; self.qr_v += (-30. * self.qr - 5. * self.qr_v + .8 * impact) * dt
        self.qz += self.qz_v * dt; self.qr += self.qr_v * dt; self.qz = float(np.clip(self.qz, -.27, .2)); self.qr = float(np.clip(self.qr, -.18, .18))
        X = -.46 + self.dx * np.arange(self.nx); xx, yy = np.meshgrid(X, X, indexing="ij"); sig = .18 + .04 * submerged
        footprint = np.exp(-((xx - self.center[0])**2 + (yy - self.center[1])**2) / (2. * sig * sig)); source = -.32 * submerged * self.center_v[2] * footprint
        div = np.gradient(self.ux, self.dx, axis=0) + np.gradient(self.uy, self.dx, axis=1)
        # Finite-volume shallow-water continuity: eta_t = -H div(u) plus
        # the moving-body displacement source.  A small Laplacian supplies
        # physical/numerical viscosity at the particle sampling scale.
        self.eta_v = -.4 * div + source + .002 * self._lap(self.eta)
        self.eta += self.eta_v * dt
        self.ux += (-9.81 * np.gradient(self.eta, self.dx, axis=0) - 1.8 * self.ux - .15 * footprint * self.center_v[0]) * dt
        self.uy += (-9.81 * np.gradient(self.eta, self.dx, axis=1) - 1.8 * self.uy - .15 * footprint * self.center_v[1]) * dt
        self.eta *= .999; self.eta = np.clip(self.eta, -.2, .2); self.ux = np.clip(self.ux, -2., 2.); self.uy = np.clip(self.uy, -2., 2.)
        self.ux[[0, -1], :] = 0.; self.ux[:, [0, -1]] = 0.; self.uy[[0, -1], :] = 0.; self.uy[:, [0, -1]] = 0.
        fluid_on_torus = np.array([0., 0., buoy + drag - pressure])
        self.interaction[0] = fluid_on_torus; self.interaction[1] = -fluid_on_torus
        self.time += dt; self.last_dt = dt; self._update_particles()
        new_water_momentum = np.sum(self.mass[self.nt:, None] * self.vel[self.nt:], axis=0)
        fluid_gravity = np.array([0., 0., -self.water_mass * 9.81])
        wall_force = (new_water_momentum - old_water_momentum) / dt - self.interaction[1] - fluid_gravity
        self.external_terms[:] = 0.
        self.external_terms[0, 0] = [0., 0., -weight]
        self.external_terms[1, 0] = fluid_gravity
        self.external_terms[1, 1] = wall_force
        self.impulse += np.sum(self.external_terms, axis=1) * dt

    def _frame(self, time):
        B = np.array([[0.,0.,1.],[0.,0.,-1.],[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
        weights = np.zeros((2, self.n)); weights[0,:self.nt] = 1.; weights[1,self.nt:] = 1.
        fields = {"boundary_normal":B, "boundary_target_velocity":np.zeros_like(B), "boundary_velocity":np.zeros_like(B), "budget_cumulative_external_impulse":self.impulse.copy(), "budget_cumulative_inflow_momentum":np.zeros((2,3)), "budget_cumulative_outflow_momentum":np.zeros((2,3)), "budget_domain_weights":weights, "budget_external_force":self.external_terms.copy(), "interaction_force":self.interaction.copy(), "cumulative_mass_source":np.zeros(2)}
        f = StateFrame(time=float(time), position=self.pos.copy(), velocity=self.vel.copy(), mass=self.mass.copy(), body_id=self.body.copy(), attributes={"rest_position":self.rest.copy(), "mass_quantity":self.mass.copy()}, fields=fields, scalars={"reference_momentum":max(1.,self.torus_mass), "reference_velocity":1., "water_free_surface_z":.4+float(np.mean(self.eta))})
        f.validate(); return f

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10: raise ValueError("advance_to requires nondecreasing time")
        while self.time < target - 1e-10: self._step(min(.003, target - self.time))
        return self._frame(target)

    def diagnostics(self):
        return {"time":float(self.time), "torus_center":self.center.copy(), "free_surface_mean":float(np.mean(self.eta))}
