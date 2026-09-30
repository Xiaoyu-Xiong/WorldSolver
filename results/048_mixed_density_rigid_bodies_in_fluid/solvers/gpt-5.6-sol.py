"""Deterministic buoyant rigid bodies coupled to a marker fluid."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        f = scene.initial_state()
        self.pos = np.asarray(f.position, float).copy()
        self.vel = np.asarray(f.velocity, float).copy()
        self.mass = np.asarray(f.mass, float).copy()
        self.body_id = np.asarray(f.body_id, np.int32).copy()
        self.n = len(self.pos)
        self.nm = int(np.count_nonzero(self.body_id == 0))
        self.nb = 6
        cfg = scene.config
        fluid = cfg["fluid"]
        self.rho = float(fluid["density"])
        self.g = np.asarray(cfg.get("gravity", [0, 0, -9.81]), float)
        self.lo = np.asarray(fluid["bounds_min"], float)
        self.hi = np.asarray(fluid["bounds_max"], float)
        self.dx = float(fluid["marker_spacing"])
        self.ratio = np.asarray(cfg["rigid_bodies"]["density_ratios"], float)
        self.volume = np.asarray(f.attributes["marker_volume"][self.nm:], float)
        self.bmass = self.mass[self.nm:].copy()
        self.br = np.array([.25, .30, .40, .32, .30, .24])
        self.time = 0.0
        self.external = np.zeros((self.n, 3))
        self.external[self.nm:] = ((self.rho-self.rho*self.ratio)*self.volume)[:, None]*(-self.g)
        self.interaction = np.zeros((self.nb+1, 3))
        self.orientation = np.zeros((self.n, 4)); self.orientation[:, 3] = 1.0
        axes = [np.arange(self.lo[i], self.hi[i] + .5*self.dx, self.dx) for i in range(3)]
        self.grid_shape = tuple(len(a) for a in axes)
        self.grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def _step(self, dt: float) -> None:
        bp = self.pos[self.nm:]
        bv = self.vel[self.nm:]
        ext = np.zeros((self.n, 3))
        inter = np.zeros((self.nb+1, 3))
        # Archimedes force plus viscous momentum exchange with nearby fluid.
        for i in range(self.nb):
            net = (self.rho - self.rho*self.ratio[i])*self.volume[i]*(-self.g)
            d = self.pos[:self.nm] - bp[i]
            near = np.linalg.norm(d, axis=1) < self.br[i] + .28
            uf = np.mean(self.vel[:self.nm][near], axis=0) if np.any(near) else np.zeros(3)
            relative = uf-bv[i]
            area = np.pi*self.br[i]**2
            drag = (18.0*self.volume[i] + .35*self.rho*area*np.linalg.norm(relative))*relative
            net += drag; inter[i] += drag
            ext[self.nm+i] = self.rho*self.ratio[i]*self.volume[i]*self.g + self.rho*self.volume[i]*(-self.g)
            if np.any(near):
                ext[np.flatnonzero(near)] += -drag/np.count_nonzero(near)
            # Potential-flow added mass prevents singular acceleration of very light bodies.
            effective_mass = self.bmass[i] + .5*self.rho*self.volume[i]
            bv[i] += dt*net/max(effective_mass, 1e-9)
            bp[i] += dt*bv[i]
        inter[-1] = -np.sum(inter[:-1], axis=0)
        # Equal-and-opposite compliant rigid contact.
        for i in range(self.nb):
            for j in range(i+1, self.nb):
                d = bp[j]-bp[i]; dist = float(np.linalg.norm(d))
                gap = dist-self.br[i]-self.br[j]
                if gap < 0:
                    n = d/max(dist, 1e-9)
                    force = (1800*(-gap)-90*np.dot(bv[j]-bv[i], n))*n
                    inter[i] -= force; inter[j] += force
                    bv[i] -= dt*force/max(self.bmass[i], 1e-9)
                    bv[j] += dt*force/max(self.bmass[j], 1e-9)
                    correction = .5*(-gap+1e-4)*n
                    bp[i] -= correction; bp[j] += correction
        # Contact is internal and cancels pairwise; retain the fluid reaction to coupling.
        inter[-1] = -np.sum(inter[:-1], axis=0)
        # Closed no-penetration tank.
        for i in range(self.nb):
            for a in range(3):
                low, high = self.lo[a]+self.br[i], self.hi[a]-self.br[i]
                if bp[i, a] < low: bp[i, a] = low; bv[i, a] = abs(bv[i, a])*.35
                if bp[i, a] > high: bp[i, a] = high; bv[i, a] = -abs(bv[i, a])*.35
        # Marker-fluid time integration with damping and reflecting walls.
        self.vel[:self.nm] += dt*ext[:self.nm]/self.mass[:self.nm, None]
        self.vel[:self.nm] *= np.exp(-.8*dt)
        self.pos[:self.nm] += dt*self.vel[:self.nm]
        for a in range(3):
            bad = (self.pos[:self.nm, a] < self.lo[a]) | (self.pos[:self.nm, a] > self.hi[a])
            self.pos[:self.nm, a] = np.clip(self.pos[:self.nm, a], self.lo[a], self.hi[a])
            self.vel[:self.nm, a] = np.where(bad, -.25*self.vel[:self.nm, a], self.vel[:self.nm, a])
        self.pos[self.nm:] = bp; self.vel[self.nm:] = bv
        self.external, self.interaction = ext, inter

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time-1e-10: raise ValueError("time must be nondecreasing")
        remaining = target-self.time
        while remaining > 1e-12:
            dt = min(.005, remaining); self._step(dt); remaining -= dt
        self.time = target
        vf = np.zeros(self.grid_shape+(3,))
        for i in range(self.nb):
            d = self.grid-self.pos[self.nm+i]
            w = np.exp(-np.sum(d*d, axis=-1)/(2*(self.br[i]+.18)**2))
            vf += self.vel[self.nm+i]*w[..., None]
        attrs = {
            "density": np.r_[np.full(self.nm, self.rho), self.rho*self.ratio],
            "density_ratio": np.r_[np.ones(self.nm), self.ratio],
            "coupled_phase": np.r_[np.zeros(self.nm), np.ones(self.nb)],
            "marker_volume": np.r_[np.full(self.nm, self.dx**3), self.volume],
            "mass_quantity": self.mass.copy(),
        }
        fields = {"velocity_field": vf, "active_mask": np.ones(self.grid_shape),
                  "cumulative_mass_source": np.zeros(1), "external_force": self.external.copy(),
                  "interaction_force": self.interaction.copy()}
        scalars = {"fluid_density": self.rho, "grid_spacing": self.dx,
                   "characteristic_length": .5, "initial_total_momentum": 0.0}
        out = StateFrame(target, self.pos.copy(), self.vel.copy(), self.orientation.copy(),
                         np.zeros((self.n, 3)), self.mass.copy(), self.body_id.copy(), attrs, fields, scalars)
        out.validate()
        return out

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "semi-implicit buoyancy/contact/marker coupling", "substep_dt": .005}
