"""Snowball-into-sticky-wall solver.

Physical model
--------------
A continuum snow model is discretised with the Material Point Method (MPM):
material point particles carry mass, velocity, a deformation gradient ``F`` and a
small set of internal plastic variables.  Every substep

1. particles scatter mass, momentum (APIC affine) and internal stress forces onto
   a uniform background grid (quadratic B-spline shape functions),
2. the grid momentum equation is solved explicitly with gravity and the frozen
   boundary conditions (sticky, i.e. no-slip velocity, nodes inside the wall and
   the floor; the wall's inner face is ``contact.wall_inner_x``),
3. grid velocity/affine fields are gathered back to the particles,
4. the deformation gradient is advanced with ``F <- (I + dt * grad v) F`` and a
   dissipative Drucker-Prager / von Mises return mapping (Hencky log-strain
   space) with volumetric compaction and tensile limits is applied, which yields
   the first Piola stress used by the next substep,
5. positions are advected and a sticky surface projection removes residual
   penetration so attached material rests on the wall/floor faces.

No animation tables, keyframes or scene specific motion are used: all motion
comes from the discretised balance laws and the constitutive update.  The
scheme is fully deterministic.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_I3 = np.eye(3)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg

        spec = cfg["snowball"]
        contact = cfg["contact"]
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.wall_x = float(contact["wall_inner_x"])
        self.floor_z = float(contact["floor_z"])
        self.ls = float(contact["contact_length_scale"])
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.rho0 = float(spec["nominal_density"])

        # Elastic and plastic snow parameters (fixed material constants).
        self.E = 1.4e5
        self.nu = 0.2
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.q0 = 0.05          # initial cohesion / yield shear strain
        self.harden = 0.35      # linear hardening with accumulated plastic strain
        self.J_min = 0.60       # plastic compaction limit
        self.J_max = 1.05       # plastic tensile (decohesion) limit
        self.viscosity = 20.0   # small physical shear viscosity (Pa s)

        # Initial physical state comes from the frozen scene contract.
        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.n = int(self.mass.size)
        self.volume0 = self.mass / self.rho0
        self.initial_component_id = np.zeros(self.n, dtype=np.int64)
        self.body_id = np.zeros(self.n, dtype=np.int64)

        # Internal state.
        self.F = np.tile(_I3, (self.n, 1, 1))
        self.C = np.zeros((self.n, 3, 3))
        self.P = np.zeros((self.n, 3, 3))
        self.qp = np.zeros(self.n)           # accumulated deviatoric plastic strain

        # Background grid (fixed, covers the whole frozen workspace).
        self.dx = 0.05
        self.grid_min = np.array([-3.00, -1.15, -0.10], dtype=float)
        self.grid_max = np.array([0.20, 1.15, 2.70], dtype=float)
        self.dims = (
            np.rint((self.grid_max - self.grid_min) / self.dx).astype(np.int64) + 1
        )
        self.n_nodes = int(self.dims[0] * self.dims[1] * self.dims[2])
        self.ny = int(self.dims[1])
        self.nz = int(self.dims[2])

        # Integrator: exact integer subdivision of the sampling interval.
        self.nsub = 25
        self.dt = self.sample_dt / float(self.nsub)
        self.step_index = 0
        self.sim_time = 0.0

        self._constitutive()  # P = 0 for F = I
        self._initial_mass = float(self.mass.sum())
        self._initial_volume = float(self.volume0.sum())

    # -------------------------------------------------------------- machinery
    def _weights(self):
        xi = (self.position - self.grid_min[None, :]) / self.dx
        base = np.floor(xi).astype(np.int64)
        base = np.clip(base, 1, self.dims - 2)
        f = xi - base
        w = (
            0.5 * (1.0 - f) ** 2,
            0.75 - (f - 0.5) ** 2,
            0.5 * f ** 2,
        )
        g = (-(1.0 - f), 1.0 - 2.0 * f, f)
        return base, f, w, g

    def _constitutive(self) -> None:
        F = self.F
        bad = ~np.isfinite(F).all(axis=(1, 2))
        if bad.any():
            F[bad] = _I3

        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, 1.0e-5, 10.0)
        R = np.einsum("nij,njk->nik", U, Vt)

        # Drucker-Prager / von Mises plasticity in Hencky log-strain space.
        e = np.log(s)
        em = e.mean(axis=1, keepdims=True)
        ed = e - em
        q = np.linalg.norm(ed, axis=1)
        qy = self.q0 + self.harden * self.qp
        over = q > qy
        ratio = np.ones_like(q)
        ratio[over] = qy[over] / np.maximum(q[over], 1.0e-12)
        self.qp = self.qp + np.where(over, (1.0 - ratio) * q, 0.0)
        ed = ed * ratio[:, None]
        e_new = ed + np.clip(em, np.log(self.J_min), np.log(self.J_max))
        s_new = np.exp(e_new)

        Fn = np.einsum("nij,nj,nkj->nik", U, s_new, Vt)
        self.F = Fn
        J = np.prod(s_new, axis=1)
        FinvT = np.einsum("nij,nj,nkj->nik", U, 1.0 / s_new, Vt)

        P = 2.0 * self.mu * (Fn - R) + (self.lam * J * (J - 1.0))[:, None, None] * FinvT
        if self.viscosity > 0.0:
            L = self.C
            sym = L + np.transpose(L, (0, 2, 1))
            sig = self.viscosity * sym
            P = P + (J[:, None, None]) * np.einsum("nij,njk->nik", sig, FinvT)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        self.P = P

    def _step(self) -> None:
        dt = self.dt
        dx = self.dx
        n = self.n

        base, f, W, G = self._weights()

        mass_g = np.zeros(self.n_nodes)
        mom_g = np.zeros((self.n_nodes, 3))
        force_g = np.zeros((self.n_nodes, 3))

        m = self.mass
        vel = self.velocity
        Cm = self.C
        Pp = self.P
        V0 = self.volume0

        idx_all = []
        w_all = []
        d_all = []
        for a in range(3):
            wxa, gxa = W[a][:, a], G[a][:, a]
            for b in range(3):
                wyb, gyb = W[b][:, b], G[b][:, b]
                for c in range(3):
                    wzc, gzc = W[c][:, c], G[c][:, c]
                    w = wxa * wyb * wzc
                    gvec = np.stack(
                        (gxa * wyb * wzc, wxa * gyb * wzc, wxa * wyb * gzc), axis=1
                    ) / dx
                    dvec = np.stack(
                        ((a - 1) - f[:, 0], (b - 1) - f[:, 1], (c - 1) - f[:, 2]),
                        axis=1,
                    ) * dx
                    idx = (
                        (base[:, 0] + (a - 1)) * self.ny + (base[:, 1] + (b - 1))
                    ) * self.nz + (base[:, 2] + (c - 1))
                    idx_all.append(idx)
                    w_all.append(w)
                    d_all.append(dvec)

                    wm = w * m
                    mass_g += np.bincount(idx, weights=wm, minlength=self.n_nodes)
                    aff = np.einsum("nij,nj->ni", Cm, dvec)
                    contrib = wm[:, None] * (vel + aff)
                    for k in range(3):
                        mom_g[:, k] += np.bincount(
                            idx, weights=contrib[:, k], minlength=self.n_nodes
                        )
                    Pg = np.einsum("nij,nj->ni", Pp, gvec)
                    fv = -V0[:, None] * Pg
                    for k in range(3):
                        force_g[:, k] += np.bincount(
                            idx, weights=fv[:, k], minlength=self.n_nodes
                        )

        # Grid momentum update.
        vg = np.zeros_like(mom_g)
        good = mass_g > 1.0e-12
        vg[good] = mom_g[good] / mass_g[good, None]
        vg[good] += dt * self.gravity[None, :]
        vg = vg + dt * force_g / np.where(good, mass_g, 1.0)[:, None]

        vg = vg.reshape(self.dims[0], self.ny, self.nz, 3)
        x_nodes = self.grid_min[0] + dx * np.arange(self.dims[0])
        z_nodes = self.grid_min[2] + dx * np.arange(self.nz)
        wall_slice = x_nodes >= self.wall_x
        floor_slice = z_nodes <= self.floor_z
        if wall_slice.any():
            vg[wall_slice, :, :, :] = 0.0
        if floor_slice.any():
            vg[:, :, floor_slice, :] = 0.0
        np.clip(vg, -80.0, 80.0, out=vg)
        vg = vg.reshape(-1, 3)

        # Gather (APIC) back to particles.
        v_new = np.zeros((n, 3))
        Maff = np.zeros((n, 3, 3))
        Dmat = np.zeros((n, 3, 3))
        for idx, w, dvec in zip(idx_all, w_all, d_all):
            vn = vg[idx]
            v_new += w[:, None] * vn
            Maff += w[:, None, None] * vn[:, :, None] * dvec[:, None, :]
            Dmat += w[:, None, None] * dvec[:, :, None] * dvec[:, None, :]

        self.velocity = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
        reg = 1.0e-9 * _I3[None, :, :]
        self.C = np.nan_to_num(
            np.einsum("nij,njk->nik", Maff, np.linalg.inv(Dmat + reg)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Advance deformation gradient and evaluate the constitutive response.
        self.F = np.einsum(
            "nij,njk->nik", _I3[None, :, :] + dt * self.C, self.F
        )
        self._constitutive()

        # Advect and enforce the sticky surface constraint.
        self.position = self.position + dt * self.velocity
        self.position = np.nan_to_num(self.position, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.velocity, -100.0, 100.0, out=self.velocity)

        pen_wall = self.position[:, 0] > self.wall_x
        if pen_wall.any():
            self.position[pen_wall, 0] = self.wall_x
            self.velocity[pen_wall, 0] = np.minimum(self.velocity[pen_wall, 0], 0.0)
        pen_floor = self.position[:, 2] < self.floor_z
        if pen_floor.any():
            self.position[pen_floor, 2] = self.floor_z
            self.velocity[pen_floor, 2] = np.maximum(self.velocity[pen_floor, 2], 0.0)

        self.sim_time += dt

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        max_steps = int(round(self.duration / self.dt))
        target = max(0, min(target, max_steps))
        while self.step_index < target:
            self._step()
            self.step_index += 1
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        pos = self.position
        d_wall = self.wall_x - pos[:, 0]
        d_floor = pos[:, 2] - self.floor_z
        gap = np.minimum(d_wall, d_floor)
        sel = np.flatnonzero(gap < 2.0 * self.ls)
        if sel.size == 0:
            sel = np.array([int(np.argmin(gap))], dtype=np.int64)
        frame = StateFrame(
            time=float(time),
            position=pos.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "reference_volume": self.volume0.copy(),
                "initial_component_id": self.initial_component_id.copy(),
            },
            fields={
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": gap[sel],
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "contact_length_scale": self.ls,
                "initial_total_mass": self._initial_mass,
                "initial_total_volume": self._initial_volume,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        gap = np.minimum(self.wall_x - self.position[:, 0], self.position[:, 2] - self.floor_z)
        attached = int(np.count_nonzero(np.abs(gap) < self.ls))
        return {
            "steps": int(self.step_index),
            "sim_time": float(self.sim_time),
            "particle_count": int(self.n),
            "attached_samples": attached,
            "total_mass": float(self.mass.sum()),
            "max_speed": float(np.linalg.norm(self.velocity, axis=1).max()),
            "min_gap": float(gap.min()),
        }
