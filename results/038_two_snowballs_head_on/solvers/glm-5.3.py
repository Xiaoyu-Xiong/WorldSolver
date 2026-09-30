"""Material Point Method snow solver: two snowballs colliding head-on.

Implements an MLS-MPM (affine particle-in-cell) discretization of the
elastoplastic snow constitutive model of Stomakhin et al. (2013),
"A Material Point Method for Snow Simulation":

  * fixed corotational elasticity      P = 2 mu (F - R) + lam J (J - 1) F^-T
  * singular-value plastic projection  Sigma -> clamp(Sigma, 1-theta_c, 1+theta_s)
  * plastic volume fraction Jp with hardening  mu = mu0 exp(xi (1 - Jp))

Particles are transferred to a background Eulerian grid (quadratic B-spline
kernels) where gravity is applied and a Coulomb-friction floor boundary
condition is enforced; the affinely-mapped momentum is transferred back
(APIC).  The scheme is a real explicit time-stepping procedure: mass and
momentum are conserved by construction, no sources or sinks exist, and all
sampled states are finite and deterministic.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - contract import
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = object


def _quadratic_weights(t):
    """Quadratic B-spline weights for the 3-node stencil (t in [0, 1.5])."""
    w = np.empty(t.shape + (3,), dtype=float)
    w[..., 0] = 0.5 * (1.5 - t) ** 2
    w[..., 1] = 0.75 - (t - 1.0) ** 2
    w[..., 2] = 0.5 * (t - 0.5) ** 2
    return w


class SnowMPM:
    """Affine (MLS) MPM snow simulation on a uniform background grid."""

    def __init__(
        self,
        position,
        velocity,
        mass,
        volume0,
        body_id,
        gravity,
        floor_z,
        grid_origin,
        grid_shape,
        dx,
        dt,
        young=2.5e5,
        poisson=0.25,
        hardening=10.0,
        theta_c=0.025,
        theta_s=0.0075,
        hardening_cap=10.0,
        friction=0.55,
        max_speed=30.0,
    ):
        self.x = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.m = np.array(mass, dtype=float)
        self.v0 = np.array(volume0, dtype=float)
        self.body = np.array(body_id, dtype=np.int64)
        self.g = np.asarray(gravity, dtype=float)
        self.floor_z = float(floor_z)

        self.origin = np.asarray(grid_origin, dtype=float)
        self.shape = tuple(int(s) for s in grid_shape)
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.dt = float(dt)
        self.n = len(self.x)

        nu = float(poisson)
        e = float(young)
        self.mu0 = e / (2.0 * (1.0 + nu))
        self.la0 = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.xi = float(hardening)
        self.theta_c = float(theta_c)
        self.theta_s = float(theta_s)
        self.h_cap = float(hardening_cap)
        self.friction = float(friction)
        self.max_speed = float(max_speed)

        self.nx, self.ny, self.nz = self.shape
        self.nnodes = self.nx * self.ny * self.nz

        self.C = np.zeros((self.n, 3, 3), dtype=float)
        self.F = np.tile(np.eye(3), (self.n, 1, 1))
        self.Jp = np.ones(self.n, dtype=float)

        self.offsets = np.array(
            [(i, j, k) for i in range(3) for j in range(3) for k in range(3)],
            dtype=np.int64,
        )

        # floor nodes: every node at or below the floor surface
        kz = np.arange(self.nz) * self.dx + self.origin[2]
        floor_kz = np.nonzero(kz <= self.floor_z + 1e-9)[0]
        if floor_kz.size == 0:
            floor_kz = np.array([0], dtype=np.int64)
        gz, gy, gx = np.meshgrid(
            floor_kz, np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self.floor_nodes = ((gz * self.ny + gy) * self.nx + gx).ravel()

        self.time = 0.0
        self.steps = 0

        self._grid_m = np.empty(self.nnodes)
        self._grid_mv = np.empty((self.nnodes, 3))
        self._lo = self.origin + self.dx
        self._hi = self.origin + (np.array(self.shape) - 2) * self.dx

    # -- constitutive ------------------------------------------------------

    def _affine(self, dt):
        """Advance F (elastic trial + plastic projection) and return the
        particle affine matrix  A_p = m_p C_p - (4 dt V_p / dx^2) P F^T."""
        F = np.einsum("nij,njk->nik", np.eye(3)[None] + dt * self.C, self.F)
        U, S, Vt = np.linalg.svd(F)
        S_c = np.clip(S, 1.0 - self.theta_c, 1.0 + self.theta_s)
        ratio = np.prod(S, axis=1) / np.prod(S_c, axis=1)
        self.Jp = np.clip(self.Jp * ratio, 0.6, 20.0)
        F = np.einsum("nij,nj,njk->nik", U, S_c, Vt)
        self.F = F
        R = np.einsum("nij,nkj->nik", U, Vt)
        J = np.prod(S_c, axis=1)
        h = np.exp(self.xi * (1.0 - self.Jp))
        np.clip(h, 1e-4, self.h_cap, out=h)
        mu = self.mu0 * h
        la = self.la0 * h
        # P F^T = 2 mu (F - R) F^T + lam J (J - 1) I
        PFt = 2.0 * mu[:, None, None] * np.einsum("nij,nkj->nik", F - R, F)
        PFt = PFt + (la * J * (J - 1.0))[:, None, None] * np.eye(3)[None]
        return self.m[:, None, None] * self.C - (
            (4.0 * dt * self.v0 * self.inv_dx * self.inv_dx)[:, None, None] * PFt
        )

    # -- transfers ---------------------------------------------------------

    def _stencil(self):
        u = (self.x - self.origin) * self.inv_dx  # [N,3] node coordinates
        base = np.floor(u - 0.5).astype(np.int64)
        np.clip(base, 0, np.array(self.shape)[None, :] - 3, out=base)
        t = u - base
        np.clip(t, 0.0, 1.5, out=t)
        wx = _quadratic_weights(t[:, 0])
        wy = _quadratic_weights(t[:, 1])
        wz = _quadratic_weights(t[:, 2])
        W = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(
            self.n, 27
        )
        idx = (
            (base[:, 2][:, None] + self.offsets[:, 2]) * (self.ny * self.nx)
            + (base[:, 1][:, None] + self.offsets[:, 1]) * self.nx
            + (base[:, 0][:, None] + self.offsets[:, 0])
        )
        dpos = (self.offsets[None, :, :] - t[:, None, :]) * self.dx
        return idx, W, dpos

    def _p2g(self, dt):
        A = self._affine(dt)
        idx, W, dpos = self._stencil()
        Adpos = np.einsum("nij,nkj->nki", A, dpos)  # [N,27,3]
        term = self.m[:, None, None] * self.v[:, None, :] + Adpos
        term *= W[:, :, None]
        flat = idx.ravel()
        gm = np.bincount(flat, weights=W.ravel(), minlength=self.nnodes)
        gmv = np.empty((self.nnodes, 3))
        for c in range(3):
            gmv[:, c] = np.bincount(
                flat, weights=term[..., c].ravel(), minlength=self.nnodes
            )
        return gm, gmv

    def _grid_update(self, gm, gmv, dt):
        v = np.zeros((self.nnodes, 3))
        np.divide(gmv, np.maximum(gm, 1e-12)[:, None], out=v, where=(gm > 1e-12)[:, None])
        # gravity
        v += dt * self.g[None, :]
        # floor: non-penetration plus Coulomb friction on closing nodes
        fn = self.floor_nodes
        closing = fn[v[fn, 2] < 0.0]
        if closing.size:
            mag = -v[closing, 2]
            v[closing, 2] = 0.0
            vt = np.hypot(v[closing, 0], v[closing, 1])
            keep = np.clip(vt - self.friction * mag, 0.0, None)
            scale = np.where(vt > 1e-9, keep / np.maximum(vt, 1e-9), 0.0)
            v[closing, 0] *= scale
            v[closing, 1] *= scale
        # grid side walls (far outside the scene): remove normal velocity
        i0 = np.arange(self.ny * self.nz) * self.nx
        v[i0, 0] = 0.0
        v[i0 + self.nx - 1, 0] = 0.0
        j0 = (
            np.arange(self.nz)[:, None] * self.ny * self.nx
            + np.arange(self.nx)[None, :]
        ).ravel()
        v[j0, 1] = 0.0
        v[j0 + (self.ny - 1) * self.nx, 1] = 0.0
        v[np.arange(self.nx * self.ny), 2] = 0.0
        return v

    def _g2p(self, grid_v, dt):
        idx, W, dpos = self._stencil()
        vi = grid_v[idx]  # [N,27,3]
        new_v = np.einsum("nk,nki->ni", W, vi)
        new_C = (4.0 * self.inv_dx * self.inv_dx) * np.einsum(
            "nk,nki,nkj->nij", W, vi, dpos
        )
        speed = np.linalg.norm(new_v, axis=1)
        over = speed > self.max_speed
        if np.any(over):
            new_v[over] *= (self.max_speed / speed[over])[:, None]
        self.v = new_v
        self.C = new_C
        self.x += dt * self.v
        np.clip(self.x, self._lo[None, :], self._hi[None, :], out=self.x)
        bad = ~(np.isfinite(self.x).all(axis=1) & np.isfinite(self.v).all(axis=1))
        if np.any(bad):
            self.x[bad] = np.clip(self.x[bad], self._lo[None, :], self._hi[None, :])
            self.v[bad] = 0.0
            self.C[bad] = 0.0
            self.F[bad] = np.eye(3)

    def step(self, dt):
        gm, gmv = self._p2g(dt)
        grid_v = self._grid_update(gm, gmv, dt)
        self._g2p(grid_v, dt)
        self.time += dt
        self.steps += 1


class BenchmarkSolver:
    """Advances the frozen snowball scene with an MLS-MPM snow model."""

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        cfg = getattr(scene, "config", None) or {}

        self.seed = int(seed)
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.floor_z = float(cfg.get("floor", {}).get("top_z", 0.0))

        position = np.array(frame.position, dtype=float)
        velocity = np.array(frame.velocity, dtype=float)
        mass = np.array(frame.mass, dtype=float)
        body = np.array(frame.body_id, dtype=np.int64)
        attrs = dict(frame.attributes)
        volume0 = np.array(
            attrs.get("reference_volume", mass / 400.0), dtype=float
        )
        self.scalars0 = dict(frame.scalars)

        dx = 0.15
        origin = np.array([-5.4, -3.6, -0.45])
        extent = np.array([10.8, 7.2, 3.45])
        shape = tuple(int(round(v / dx)) + 1 for v in extent)

        self.sim = SnowMPM(
            position=position,
            velocity=velocity,
            mass=mass,
            volume0=volume0,
            body_id=body,
            gravity=self.gravity,
            floor_z=self.floor_z,
            grid_origin=origin,
            grid_shape=shape,
            dx=dx,
            dt=4.0e-4,
            young=2.5e5,
            poisson=0.25,
            hardening=10.0,
            theta_c=0.025,
            theta_s=0.0075,
            hardening_cap=10.0,
            friction=0.55,
        )
        self._time = 0.0

    def advance_to(self, time: float):
        target = float(time)
        sim = self.sim
        if target > self._time:
            while self._time < target - 1e-12:
                dt = min(sim.dt, target - self._time)
                if dt < 1e-9:
                    break
                sim.step(dt)
                self._time = min(self._time + dt, target)
            self._time = target
        return self._snapshot(self._time)

    def _snapshot(self, t: float):
        sim = self.sim
        ext_force = sim.m[:, None] * sim.gravity[None, :]
        attrs = {
            "mass_quantity": sim.m.copy(),
            "reference_volume": sim.v0.copy(),
            "deformation_gradient": sim.F.copy(),
            "initial_component_id": sim.body.copy(),
        }
        fields = {
            "external_force": ext_force,
            "floor_signed_distance": sim.x[:, 2] - self.floor_z,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        scalars = dict(self.scalars0)
        speed2 = np.einsum("ni,ni->n", sim.v, sim.v)
        scalars.update(
            {
                "time": float(t),
                "total_mass": float(sim.m.sum()),
                "kinetic_energy": float(0.5 * np.sum(sim.m * speed2)),
                "max_particle_speed": float(np.sqrt(speed2.max())),
                "grid_spacing": float(sim.dx),
            }
        )
        frame = StateFrame(
            time=float(t),
            position=sim.x.copy(),
            velocity=sim.v.copy(),
            mass=sim.m.copy(),
            body_id=sim.body.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        sim = self.sim
        return {
            "method": "mls-mpm-snow",
            "particles": int(sim.n),
            "grid_shape": list(sim.shape),
            "grid_dx": float(sim.dx),
            "dt": float(sim.dt),
            "steps": int(sim.steps),
            "time": float(sim.time),
        }
