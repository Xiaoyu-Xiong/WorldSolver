"""Shallow-water solver for scene 005_shallow_water_obstacle.

Finite-volume shallow-water equations (conservative variables h, hu, hv)
solved with a Rusanov (local Lax-Friedrichs) flux on the 64x64
cell-centered grid.  Domain walls and the octagonal obstacle are handled
as reflecting boundaries via mirrored ghost states (no mass flux,
hydrostatic pressure reflection), guaranteeing no penetration and global
height (mass) conservation.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        domain = cfg["domain"]
        self.nx, self.ny = domain["grid_shape"]
        lo = np.asarray(domain["bounds_min"], dtype=float)
        hi = np.asarray(domain["bounds_max"], dtype=float)
        self.dx = (hi[0] - lo[0]) / self.nx
        self.dy = (hi[1] - lo[1]) / self.ny
        self.g = abs(float(cfg["gravity"][2]))
        self.rho = float(cfg["fluid"]["density"])

        # Initial state from the scene (height field, active mask).
        frame0 = scene.initial_state()
        self.h = np.array(frame0.fields["density_field"], dtype=float)
        vel0 = np.array(frame0.fields["velocity_field"], dtype=float)
        self.hu = self.h * vel0[:, :, 0]
        self.hv = self.h * vel0[:, :, 1]
        self.solid = ~(np.asarray(frame0.fields["active_mask"]) > 0.5)
        self.h[self.solid] = 0.0
        self.hu[self.solid] = 0.0
        self.hv[self.solid] = 0.0

        self.x = lo[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.y = lo[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

        # Padded solid mask: outside domain counts as solid (reflecting).
        self.solid_p = np.pad(self.solid, 1, constant_values=True)

        # Boundary normals (static walls + obstacle faces), matching scene.
        self.boundary_normals = self._boundary_normals(cfg)

        self.t = 0.0
        self.h_max0 = float(self.h.max())
        self.ref_velocity = float(np.sqrt(self.g * max(self.h_max0, 1e-6)))
        self.initial_mass = float(self.h.sum()) * self.dx * self.dy * self.rho

        # Max stable step (CFL ~ 0.35 with local wave speed).
        self.dt_max = 0.35 * min(self.dx, self.dy) / (self.ref_velocity + 1e-6)

    # ------------------------------------------------------------------
    def _boundary_normals(self, cfg) -> np.ndarray:
        obstacle = cfg["obstacle"]
        theta = np.arange(8) * np.pi / 4 + obstacle["rotation_radians"]
        verts = np.asarray(obstacle["center"], dtype=float) + obstacle[
            "circumradius"
        ] * np.column_stack((np.cos(theta), np.sin(theta)))
        obstacle_normals = []
        for a, b in zip(verts, np.roll(verts, -1, axis=0)):
            edge = b - a
            n = np.array([-edge[1], edge[0]])
            n /= np.linalg.norm(n)
            obstacle_normals.extend([n] * 8)
        ny, nx = self.ny, self.nx
        return np.vstack(
            (
                np.tile([1.0, 0.0], (ny, 1)),
                np.tile([-1.0, 0.0], (ny, 1)),
                np.tile([0.0, 1.0], (nx, 1)),
                np.tile([0.0, -1.0], (nx, 1)),
                np.array(obstacle_normals),
            )
        )

    # ------------------------------------------------------------------
    def _padded(self):
        """Padded state arrays with edge-replicated ghosts (walls)."""
        hp = np.pad(self.h, 1, mode="edge")
        hup = np.pad(self.hu, 1, mode="edge")
        hvp = np.pad(self.hv, 1, mode="edge")
        # Reflect normal velocity on domain walls.
        hup[0, :] = -hup[1, :]
        hup[-1, :] = -hup[-2, :]
        hvp[:, 0] = -hvp[:, 1]
        hvp[:, -1] = -hvp[:, -2]
        return hp, hup, hvp

    @staticmethod
    def _rusanov(Lh, Lhu, Lhv, Rh, Rhu, Rhv, g, nx_dir):
        eps = 1e-9
        Lu = Lhu / np.maximum(Lh, eps)
        Lv = Lhv / np.maximum(Lh, eps)
        Ru = Rhu / np.maximum(Rh, eps)
        Rv = Rhv / np.maximum(Rh, eps)
        cL = np.sqrt(g * np.maximum(Lh, 0.0))
        cR = np.sqrt(g * np.maximum(Rh, 0.0))
        if nx_dir:
            alpha = np.maximum(np.abs(Lu) + cL, np.abs(Ru) + cR)
        else:
            alpha = np.maximum(np.abs(Lv) + cL, np.abs(Rv) + cR)
        Fh = 0.5 * ((Lhu + Rhu) if nx_dir else (Lhv + Rhv)) - 0.5 * alpha * (Rh - Lh)
        if nx_dir:
            Fhu = (
                0.5 * (Lhu * Lu + Rhu * Ru + 0.5 * g * (Lh * Lh + Rh * Rh))
                - 0.5 * alpha * (Rhu - Lhu)
            )
            Fhv = 0.5 * (Lhu * Lv + Rhu * Rv) - 0.5 * alpha * (Rhv - Lhv)
        else:
            Fhu = 0.5 * (Lhv * Lu + Rhv * Ru) - 0.5 * alpha * (Rhu - Lhu)
            Fhv = (
                0.5 * (Lhv * Lv + Rhv * Rv + 0.5 * g * (Lh * Lh + Rh * Rh))
                - 0.5 * alpha * (Rhv - Lhv)
            )
        return Fh, Fhu, Fhv

    def _face_states(self, hp, hup, hvp, axis):
        """Mirror fluid states into solid neighbours (reflecting walls)."""
        if axis == 0:
            sl = (slice(0, self.nx + 1), slice(None))
            sr = (slice(1, self.nx + 2), slice(None))
        else:
            sl = (slice(None), slice(0, self.ny + 1))
            sr = (slice(None), slice(1, self.ny + 2))
        Lh, Lhu, Lhv = hp[sl].copy(), hup[sl].copy(), hvp[sl].copy()
        Rh, Rhu, Rhv = hp[sr].copy(), hup[sr].copy(), hvp[sr].copy()
        Lm, Rm = self.solid_p[sl], self.solid_p[sr]
        # Right solid, left fluid -> mirror left state into right.
        w = Rm & ~Lm
        Rh = np.where(w, Lh, Rh)
        Rhu = np.where(w, -Lhu, Rhu)
        Rhv = np.where(w, Lhv, Rhv)
        # Left solid, right fluid -> mirror right state into left.
        w2 = Lm & ~Rm
        Lh = np.where(w2, Rh, Lh)
        Lhu = np.where(w2, -Rhu, Lhu)
        Lhv = np.where(w2, Rhv, Lhv)
        # Both solid: states remain zero -> zero flux.
        both = Lm & Rm
        z = np.zeros_like(Lh)
        Lh = np.where(both, z, Lh); Lhu = np.where(both, z, Lhu); Lhv = np.where(both, z, Lhv)
        Rh = np.where(both, z, Rh); Rhu = np.where(both, z, Rhu); Rhv = np.where(both, z, Rhv)
        return Lh, Lhu, Lhv, Rh, Rhu, Rhv

    def _step(self, dt):
        hp, hup, hvp = self._padded()
        g = self.g
        # x-direction faces (nx+1, ny+2)
        Lh, Lhu, Lhv, Rh, Rhu, Rhv = self._face_states(hp, hup, hvp, 0)
        Fxh, Fxhu, Fxhv = self._rusanov(Lh, Lhu, Lhv, Rh, Rhu, Rhv, g, True)
        # y-direction faces (nx+2, ny+1)
        Lh, Lhu, Lhv, Rh, Rhu, Rhv = self._face_states(hp, hup, hvp, 1)
        Fyh, Fyhu, Fyhv = self._rusanov(Lh, Lhu, Lhv, Rh, Rhu, Rhv, g, False)

        fluid = ~self.solid
        ix = (slice(1, self.nx + 1), slice(1, self.ny + 1))
        divh = (Fxh[1:, 1:-1] - Fxh[:-1, 1:-1]) / self.dx + (Fyh[1:-1, 1:] - Fyh[1:-1, :-1]) / self.dy
        divhu = (Fxhu[1:, 1:-1] - Fxhu[:-1, 1:-1]) / self.dx + (Fyhu[1:-1, 1:] - Fyhu[1:-1, :-1]) / self.dy
        divhv = (Fxhv[1:, 1:-1] - Fxhv[:-1, 1:-1]) / self.dx + (Fyhv[1:-1, 1:] - Fyhv[1:-1, :-1]) / self.dy
        self.h[ix] = np.where(fluid, self.h - dt * divh, 0.0)
        self.hu[ix] = np.where(fluid, self.hu - dt * divhu, 0.0)
        self.hv[ix] = np.where(fluid, self.hv - dt * divhv, 0.0)
        np.maximum(self.h, 0.0, out=self.h)
        self.t += dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        remaining = time - self.t
        if remaining > 1e-12:
            n = int(np.ceil(remaining / self.dt_max))
            dt = remaining / n
            for _ in range(n):
                self._step(dt)
        return self._frame(time)

    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        h = self.h
        fluid = ~self.solid
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        np.divide(self.hu, np.maximum(h, 1e-6), out=u, where=fluid)
        np.divide(self.hv, np.maximum(h, 1e-6), out=v, where=fluid)
        velocity_field = np.stack((u, v), axis=-1)

        n = int(fluid.sum())
        xx = self.xx[fluid]
        yy = self.yy[fluid]
        hf = h[fluid]
        position = np.column_stack((xx, yy, 0.5 * hf))
        velocity = np.column_stack((u[fluid], v[fluid], np.zeros(n)))
        mass = self.rho * hf * self.dx * self.dy

        zeros_nb = np.zeros_like(self.boundary_normals)
        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"mass_quantity": mass.copy(), "water_height": hf.copy()},
            fields={
                "density_field": h.copy(),
                "velocity_field": velocity_field,
                "active_mask": fluid.astype(np.uint8),
                "continuity_source": np.zeros_like(h),
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_target_velocity": zeros_nb,
                "boundary_velocity": zeros_nb,
            },
            scalars={
                "grid_spacing": float(self.dx),
                "reference_velocity": self.ref_velocity,
                "water_volume": float(h.sum()) * self.dx * self.dy,
                "max_height": float(h.max()),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        total = float(self.h.sum()) * self.dx * self.dy * self.rho
        return {
            "time": self.t,
            "mass": total,
            "mass_drift": (total - self.initial_mass) / max(self.initial_mass, 1e-12),
            "max_height": float(self.h.max()),
        }
