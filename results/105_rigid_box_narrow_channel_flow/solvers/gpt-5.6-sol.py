"""Two-way coupled grid-fluid and rigid-body solver for the channel scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene, self.cfg = scene, scene.config
        initial = scene.initial_state()
        fluid, box = self.cfg["fluid"], self.cfg["rigid_box"]
        self.t = 0.0
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int64).copy()
        self.phase = np.asarray(initial.attributes["phase"], dtype=float).copy()
        self.box_mass = float(box["mass"])
        self.box_size = np.asarray(box["size"], dtype=float)
        self.angle = math.radians(float(box["initial_angle_degrees"]))
        self.omega = float(np.asarray(box["initial_angular_velocity"])[1])
        self.box_v = np.asarray(box["initial_velocity"], dtype=float)[[0, 2]].copy()
        lo, hi = np.asarray(fluid["domain_min"], float), np.asarray(fluid["domain_max"], float)
        self.nx, self.nz = int(fluid["nominal_resolution"][0]), int(fluid["nominal_resolution"][2])
        self.xmin, self.xmax, self.zmin, self.zmax = float(lo[0]), float(hi[0]), float(lo[2]), float(hi[2])
        self.dx, self.dz = (self.xmax - self.xmin) / self.nx, (self.zmax - self.zmin) / self.nz
        self.xc = self.xmin + (np.arange(self.nx) + .5) * self.dx
        self.zc = self.zmin + (np.arange(self.nz) + .5) * self.dz
        self.X, self.Z = np.meshgrid(self.xc, self.zc)
        self.u = np.zeros((self.nz, self.nx)); self.w = np.zeros_like(self.u); self.p = np.zeros_like(self.u)
        self.rho, self.depth, self.nu = float(fluid["density"]), float(hi[1] - lo[1]), 1e-4
        self.cell_mass = self.rho * self.dx * self.dz * self.depth
        self.gravity = float(self.cfg["gravity"][2]); self.controls = self.cfg.get("controls", [])
        self.last_interaction = np.zeros((2, 3)); self.last_external = np.zeros((3, 3, 3))
        self.cumulative_external = np.zeros((3, 3)); self.cumulative_inflow = np.zeros((3, 3)); self.cumulative_outflow = np.zeros((3, 3))
        fluid_mass = float(np.sum(self.mass[self.phase > .5]))
        self.reference_velocity = max(1., abs(self.gravity) * .25)
        self.reference_momentum = (fluid_mass + self.box_mass) * self.reference_velocity
        self.steps = 0

    def _drive(self, t):
        a = 0.
        for c in self.controls:
            if c.get("target") == "fluid" and float(c["start_time"]) <= t <= float(c["end_time"]):
                a += float(c["magnitude"]) * min(1., (t - float(c["start_time"])) / max(float(c.get("ramp_duration", 0.)), 1e-12))
        return a

    def _sample(self, f, x, z):
        gx = np.clip((x - self.xmin) / self.dx - .5, 0, self.nx - 1.000001)
        gz = np.clip((z - self.zmin) / self.dz - .5, 0, self.nz - 1.000001)
        i, k = np.floor(gx).astype(int), np.floor(gz).astype(int); j, l = np.minimum(i + 1, self.nx - 1), np.minimum(k + 1, self.nz - 1)
        ax, az = gx - i, gz - k
        return (1 - az) * ((1 - ax) * f[k, i] + ax * f[k, j]) + az * ((1 - ax) * f[l, i] + ax * f[l, j])

    def _project(self, dt):
        div = np.zeros_like(self.u)
        div[:, 1:-1] += (self.u[:, 2:] - self.u[:, :-2]) / (2 * self.dx)
        div[1:-1] += (self.w[2:] - self.w[:-2]) / (2 * self.dz)
        div[0] += self.w[1] / self.dz; div[-1] += (self.w[-1] - self.w[-2]) / self.dz
        rhs = self.rho * div / dt; p = self.p; ix, iz = 1 / self.dx**2, 1 / self.dz**2; den = 2 * (ix + iz)
        for _ in range(28):
            left, right = np.empty_like(p), np.empty_like(p); down, up = np.empty_like(p), np.empty_like(p)
            left[:, 0], left[:, 1:] = p[:, 0], p[:, :-1]; right[:, -1], right[:, :-1] = p[:, -1], p[:, 1:]
            down[0], down[1:] = p[0], p[:-1]; up[:-1], up[-1] = p[1:], 0.
            p = ((left + right) * ix + (down + up) * iz - rhs) / den; p[-1] = 0.
        self.p = p; self.u[:, 1:-1] -= dt * (p[:, 2:] - p[:, :-2]) / (2 * self.rho * self.dx)
        self.w[1:-1] -= dt * (p[2:] - p[:-2]) / (2 * self.rho * self.dz)
        self.u[:, 0] = self.u[:, -1] = 0.; self.w[0] = 0.; self.w[-1] -= dt * (0 - p[-2]) / (self.rho * self.dz)

    def _mask(self):
        ca, sa = math.cos(self.angle), math.sin(self.angle); rx, rz = self.X - self.pos[0, 0], self.Z - self.pos[0, 2]
        lx, lz = ca * rx - sa * rz, sa * rx + ca * rz
        mask = (np.abs(lx) <= .5 * self.box_size[0] + .35 * self.dx) & (np.abs(lz) <= .5 * self.box_size[2] + .35 * self.dz)
        return mask, lx, lz, rx, rz

    def _hydro(self, dt):
        mask, lx, lz, rx, rz = self._mask(); ca, sa = math.cos(self.angle), math.sin(self.angle)
        tu, tw = self.box_v[0] + self.omega * rz, self.box_v[1] - self.omega * rx; alpha = 1 - math.exp(-dt / .008)
        ou, ow = self.u[mask].copy(), self.w[mask].copy(); self.u[mask] += alpha * (tu[mask] - self.u[mask]); self.w[mask] += alpha * (tw[mask] - self.w[mask])
        fp = -self.cell_mass * np.array([np.sum(self.u[mask] - ou), np.sum(self.w[mask] - ow)]) / dt
        tp = -self.cell_mass * np.sum(rx[mask] * (self.w[mask] - ow) - rz[mask] * (self.u[mask] - ou)) / dt
        near = (np.abs(lx) <= .5 * self.box_size[0] + 2.5 * self.dx) & (np.abs(lz) <= .5 * self.box_size[2] + 2.5 * self.dz) & ~mask
        flow = np.array([np.mean(self.u[near]), np.mean(self.w[near])]) if np.any(near) else np.zeros(2)
        rel = flow - self.box_v; blockage = min(.94, self.box_size[0] / (self.xmax - self.xmin)); cdz = min(65., 1.2 / max((1 - blockage)**2, .02))
        areas = np.array([self.box_size[2] * self.depth, self.box_size[0] * self.depth]); drag = .5 * self.rho * areas * np.array([4., cdz]) * rel * np.abs(rel)
        buoy = np.array([0., -self.rho * np.prod(self.box_size) * self.gravity]); force = fp + drag + buoy
        torque = tp - 2.5 * self.omega
        if np.any(near): self.u[near] -= dt * (drag[0] + buoy[0]) / (np.count_nonzero(near) * self.cell_mass); self.w[near] -= dt * (drag[1] + buoy[1]) / (np.count_nonzero(near) * self.cell_mass)
        return force, torque

    def _contact(self):
        hx, hz, ca, sa = .5 * self.box_size[0], .5 * self.box_size[2], math.cos(self.angle), math.sin(self.angle)
        corners = np.array([[sx * hx, sz * hz] for sx in (-1., 1.) for sz in (-1., 1.)]); world = np.empty_like(corners)
        world[:, 0] = self.pos[0, 0] + ca * corners[:, 0] + sa * corners[:, 1]; world[:, 1] = self.pos[0, 2] - sa * corners[:, 0] + ca * corners[:, 1]
        force = np.zeros(2); torque = 0.; stiff, damp = 7e4, 900.
        for x, z in world:
            rx, rz = x - self.pos[0, 0], z - self.pos[0, 2]; pv = np.array([self.box_v[0] + self.omega * rz, self.box_v[1] - self.omega * rx])
            for pen, n in ((self.xmin - x, np.array([1., 0.])) if x < self.xmin else (0., np.zeros(2)), (x - self.xmax, np.array([-1., 0.])) if x > self.xmax else (0., np.zeros(2)), (self.zmin - z, np.array([0., 1.])) if z < self.zmin else (0., np.zeros(2))):
                if pen > 0: f = max(0., stiff * pen - damp * np.dot(pv, n)) * n; force += f; torque += rx * f[1] - rz * f[0]
        return force, torque

    def _advect(self, dt):
        x, z = self.pos[1:, 0], self.pos[1:, 2]; pu, pw = self._sample(self.u, x, z), self._sample(self.w, x, z)
        mx, mz = x + .5 * dt * pu, z + .5 * dt * pw; pu, pw = self._sample(self.u, mx, mz), self._sample(self.w, mx, mz); x += dt * pu; z += dt * pw
        x[x < self.xmin] = 2 * self.xmin - x[x < self.xmin]; x[x > self.xmax] = 2 * self.xmax - x[x > self.xmax]; z[z < self.zmin] = 2 * self.zmin - z[z < self.zmin]; z[z > self.zmax] = self.zmin + np.mod(z[z > self.zmax] - self.zmax, self.zmax - self.zmin)
        mask, lx, lz, _, _ = self._mask(); inside = mask[0:0]  # keep shape-independent marker correction below
        ca, sa = math.cos(self.angle), math.sin(self.angle); rx, rz = x - self.pos[0, 0], z - self.pos[0, 2]; lx, lz = ca * rx - sa * rz, sa * rx + ca * rz
        inside = (np.abs(lx) < .5 * self.box_size[0]) & (np.abs(lz) < .5 * self.box_size[2])
        if np.any(inside):
            ix = np.where(inside)[0]; gx = .5 * self.box_size[0] - np.abs(lx[ix]); gz = .5 * self.box_size[2] - np.abs(lz[ix]); ux = gx < gz
            lx[ix[ux]] = np.sign(lx[ix[ux]] + 1e-12) * (.5 * self.box_size[0] + .25 * self.dx); lz[ix[~ux]] = np.sign(lz[ix[~ux]] + 1e-12) * (.5 * self.box_size[2] + .25 * self.dz)
            x[ix] = self.pos[0, 0] + ca * lx[ix] + sa * lz[ix]; z[ix] = self.pos[0, 2] - sa * lx[ix] + ca * lz[ix]
        self.vel[1:, 0], self.vel[1:, 1], self.vel[1:, 2] = pu, 0., pw

    def _step(self, dt):
        drive = self._drive(self.t + .5 * dt)
        for q in (self.u, self.w):
            lap = np.zeros_like(q); lap[1:-1, 1:-1] = (q[1:-1, 2:] - 2 * q[1:-1, 1:-1] + q[1:-1, :-2]) / self.dx**2 + (q[2:, 1:-1] - 2 * q[1:-1, 1:-1] + q[:-2, 1:-1]) / self.dz**2; q += dt * self.nu * lap
        self.w += dt * (self.gravity + drive); self._project(dt); hf, ht = self._hydro(dt); cf, ct = self._contact(); gf = np.array([0., self.box_mass * self.gravity]); total = hf + gf + cf
        inertia = self.box_mass * (self.box_size[0]**2 + self.box_size[2]**2) / 12.; self.box_v += dt * total / self.box_mass; self.omega += dt * (ht + ct) / inertia; self.pos[0, 0] += dt * self.box_v[0]; self.pos[0, 2] += dt * self.box_v[1]; self.angle += dt * self.omega; self._advect(dt)
        fm = float(np.sum(self.mass[self.phase > .5])); fg = np.array([0., 0., fm * self.gravity]); fd = np.array([0., 0., fm * drive]); fb = np.array([0., 0., self.box_mass * self.gravity]); fc = np.array([cf[0], 0., cf[1]])
        self.last_external.fill(0.); self.last_external[0, 0] = fg + fb; self.last_external[0, 1] = fd; self.last_external[0, 2] = fc; self.last_external[1, 0] = fg; self.last_external[1, 1] = fd; self.last_external[2, 0] = fb; self.last_external[2, 2] = fc; self.cumulative_external += dt * np.sum(self.last_external, axis=1)
        h3 = np.array([hf[0], 0., hf[1]]); self.last_interaction[0], self.last_interaction[1] = h3, -h3; self.t += dt; self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-10: raise ValueError("advance_to times must be nondecreasing")
        while self.t < target - 1e-12: self._step(min(.004, target - self.t))
        self.vel[0] = [self.box_v[0], 0., self.box_v[1]]; orientation = np.tile([0., 0., 0., 1.], (len(self.pos), 1)); orientation[0] = [0., math.sin(.5 * self.angle), 0., math.cos(.5 * self.angle)]; angular = np.zeros_like(self.pos); angular[0, 1] = self.omega
        weights = np.zeros((3, len(self.pos))); weights[0] = 1.; weights[1, self.phase > .5] = 1.; weights[2, 0] = 1.; normals = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.]])
        bv = np.array([[0., 0., np.mean(self.w[:, 0])], [0., 0., np.mean(self.w[:, -1])], [np.mean(self.u[0]), 0., 0.]])
        fv = np.zeros((self.nz, self.nx, 3)); fv[:, :, 0], fv[:, :, 2] = self.u, self.w
        fields = {"fluid_velocity": fv, "boundary_normal": normals, "boundary_target_velocity": np.zeros((3, 3)), "boundary_velocity": bv, "budget_cumulative_external_impulse": self.cumulative_external.copy(), "budget_cumulative_inflow_momentum": self.cumulative_inflow.copy(), "budget_cumulative_outflow_momentum": self.cumulative_outflow.copy(), "budget_domain_weights": weights, "budget_external_force": self.last_external.copy(), "cumulative_mass_source": np.zeros(1), "interaction_force": self.last_interaction.copy()}
        return StateFrame(time=self.t, position=self.pos.copy(), velocity=self.vel.copy(), orientation=orientation, angular_velocity=angular, mass=self.mass.copy(), body_id=self.body_id.copy(), attributes={"phase": self.phase.copy(), "mass_quantity": self.mass.copy()}, fields=fields, scalars={"fluid_drive": self._drive(self.t), "reference_momentum": self.reference_momentum, "reference_velocity": self.reference_velocity})

    def diagnostics(self):
        return {"method": "finite_difference_projection_immersed_boundary", "steps": self.steps, "grid": [self.nx, self.nz], "simulated_time": self.t}
