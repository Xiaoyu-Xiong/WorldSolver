"""Deterministic semi-Lagrangian vortex smoke solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene, self.config = scene, scene.config
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.particles = np.asarray(initial.position, float).copy()
        self.pmass = np.asarray(initial.mass, float).copy()
        self.body_id = np.asarray(initial.body_id, np.int32).copy()
        self.radius = np.asarray(initial.attributes["particle_radius"], float).copy()
        self.shape = tuple(int(v) for v in self.config["sampling"]["field_grid_shape"])
        self.lo = np.array([-3.0, -1.2, 0.0]); self.hi = np.array([3.0, 1.2, 3.8])
        self.coords = [np.linspace(self.lo[d], self.hi[d], self.shape[d]) for d in range(3)]
        self.X, self.Y, self.Z = np.meshgrid(*self.coords, indexing="ij")
        self.dx = np.array([(self.hi[d] - self.lo[d]) / (self.shape[d] - 1) for d in range(3)])
        self.scalar = np.asarray(initial.fields["transported_scalar"], float).copy()
        self.velocity = np.zeros(self.shape + (3,), float)
        self.particle_velocity = np.zeros_like(self.particles)
        self.t = 0.0
        self.substep_dt = 0.018
        self._last_source = self._source(0.0)

    @staticmethod
    def _interp(arr, points, lo, dx):
        q = np.clip((np.asarray(points) - lo) / dx, 0.0, np.array(arr.shape[:3]) - 1.000001)
        i0 = np.floor(q).astype(np.intp); f = q - i0
        i1 = np.minimum(i0 + 1, np.array(arr.shape[:3]) - 1)
        a, b, c = i0[:, 0], i0[:, 1], i0[:, 2]; d, e, g = i1[:, 0], i1[:, 1], i1[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        v = [arr[a, b, c], arr[d, b, c], arr[a, e, c], arr[d, e, c],
             arr[a, b, g], arr[d, b, g], arr[a, e, g], arr[d, e, g]]
        w = [(1-fx)*(1-fy)*(1-fz), fx*(1-fy)*(1-fz), (1-fx)*fy*(1-fz), fx*fy*(1-fz),
             (1-fx)*(1-fy)*fz, fx*(1-fy)*fz, (1-fx)*fy*fz, fx*fy*fz]
        if arr.ndim == 4:
            out = v[0] * w[0][..., None]
            for vi, wi in zip(v[1:], w[1:]):
                out = out + vi * wi[..., None]
        else:
            out = v[0] * w[0]
            for vi, wi in zip(v[1:], w[1:]):
                out = out + vi * wi
        return out

    def _source(self, t):
        r2 = self.X*self.X + self.Y*self.Y + (self.Z - .85)**2
        return 7.0 * np.exp(-t/.32) * np.exp(-r2/.16**2)

    def _project(self, u):
        div = (np.gradient(u[..., 0], self.dx[0], axis=0) + np.gradient(u[..., 1], self.dx[1], axis=1) +
               np.gradient(u[..., 2], self.dx[2], axis=2))
        p = np.zeros(self.shape); denom = 2.0 * np.sum(self.dx**-2)
        for _ in range(4):
            pn = p.copy()
            pn[1:-1,1:-1,1:-1] = ((p[2:,1:-1,1:-1]+p[:-2,1:-1,1:-1])/self.dx[0]**2 +
                (p[1:-1,2:,1:-1]+p[1:-1,:-2,1:-1])/self.dx[1]**2 +
                (p[1:-1,1:-1,2:]+p[1:-1,1:-1,:-2])/self.dx[2]**2 - div[1:-1,1:-1,1:-1]) / denom
            p = pn
        u[...,0] -= np.gradient(p, self.dx[0], axis=0); u[...,1] -= np.gradient(p, self.dx[1], axis=1); u[...,2] -= np.gradient(p, self.dx[2], axis=2)
        return u

    def _step(self, dt):
        pts = np.column_stack((self.X.ravel(), self.Y.ravel(), self.Z.ravel()))
        back = pts - self.velocity.reshape(-1,3) * dt
        adv_u = self._interp(self.velocity, back, self.lo, self.dx).reshape(self.shape+(3,))
        adv_s = self._interp(self.scalar, back, self.lo, self.dx).reshape(self.shape)
        source = self._source(self.t)
        lap = sum(np.gradient(np.gradient(adv_s, self.dx[d], axis=d), self.dx[d], axis=d) for d in range(3))
        self.scalar = np.maximum(0.0, adv_s + dt*(.08*lap + source)); self._last_source = source
        r = np.sqrt(self.X*self.X+self.Y*self.Y+(self.Z-.85)**2) + 1e-6
        radial = np.stack((self.X/r, self.Y/r, (self.Z-.85)/r), axis=-1)
        impulse = 11*np.exp(-self.t/.42)*np.exp(-r/.9)[...,None]*radial
        buoy = np.zeros_like(self.velocity); buoy[...,2] = 3.8*self.scalar
        swirl = np.stack((-self.Y, self.X, .15*np.sin(2.5*self.Z)), axis=-1) * (1.8*np.exp(-r/1.5)*np.exp(-self.t/5))[...,None]
        u = adv_u + dt*(buoy + impulse + swirl)
        wx = np.gradient(u[...,2], self.dx[1], axis=1)-np.gradient(u[...,1], self.dx[2], axis=2)
        wy = np.gradient(u[...,0], self.dx[2], axis=2)-np.gradient(u[...,2], self.dx[0], axis=0)
        wz = np.gradient(u[...,1], self.dx[0], axis=0)-np.gradient(u[...,0], self.dx[1], axis=1)
        om = np.stack((wx,wy,wz), axis=-1); mag = np.sqrt(np.sum(om*om, axis=-1))
        grad = np.stack(np.gradient(mag, self.dx[0], self.dx[1], self.dx[2]), axis=-1)
        nrm = grad/(np.linalg.norm(grad, axis=-1, keepdims=True)+1e-5)
        u += dt*.32*np.cross(nrm, om)*np.clip(self.scalar[...,None], 0, 1.5)
        u *= np.exp(-.11*dt); u = self._project(u)
        u[:,:,0,:] = 0.0
        self.velocity = np.nan_to_num(u)
        pv = self._interp(self.velocity, self.particles, self.lo, self.dx)
        self.particles += dt*pv; self.particles[:,0]=np.clip(self.particles[:,0],self.lo[0],self.hi[0]); self.particles[:,1]=np.clip(self.particles[:,1],self.lo[1],self.hi[1])
        hit = self.particles[:,2] < .02; self.particles[:,2] = np.maximum(self.particles[:,2], .02); pv[hit,2] = np.maximum(pv[hit,2], 0)
        self.particle_velocity = np.nan_to_num(pv); self.t += dt

    def _frame(self, time):
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (9, 1))
        bv = np.zeros((len(normals), 3))
        fields = {"transported_scalar": self.scalar.copy(), "scalar_source": self._last_source.copy(), "active_mask": np.ones(self.shape, bool), "velocity_field": self.velocity.copy(), "boundary_normal": normals, "boundary_target_velocity": bv.copy(), "boundary_velocity": bv, "cumulative_mass_source": np.array([0.0])}
        attrs = {"mass_quantity": self.pmass.copy(), "particle_radius": self.radius.copy(), "material_label": self.body_id.copy(), "deformation_gradient": np.repeat(np.eye(3)[None], len(self.particles), axis=0)}
        scalars = {"contact_length_scale": float(self.config["sampling"]["spacing"]), "grid_spacing": float(self.config["sampling"]["spacing"]), "characteristic_length": 3.8, "reference_velocity": 6.0}
        frame = StateFrame(float(time), self.particles.copy(), self.particle_velocity.copy(), mass=self.pmass.copy(), body_id=self.body_id.copy(), attributes=attrs, fields=fields, scalars=scalars)
        frame.validate(); return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.t - 1e-10: raise ValueError("advance_to requires finite nondecreasing time")
        while self.t + 1e-10 < target: self._step(min(self.substep_dt, target-self.t))
        self.t = target; return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "eulerian_semi_lagrangian_vortex", "substeps_dt": self.substep_dt}
