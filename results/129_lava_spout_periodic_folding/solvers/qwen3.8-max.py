"""Physically-based MPM solver for the lava spout periodic folding scene.

Method: explicit affine-PIC (APIC) material point method on a fixed Eulerian
grid with a fixed-corotated elastoplastic constitutive model plus a Newtonian
viscous Cauchy stress (sigma_v = mu_v (L + L^T)) built from the APIC affine
velocity term.  Plastic return mapping (clamped singular values) lets the
highly viscous lava stream buckle, fold and accumulate permanently, while
elasticity and viscosity transmit the compressive load that drives the
periodic coiling instability.  Continuous mass inflow is prescribed at the
nozzle (emitter) and tracked in fields.cumulative_mass_source.

Collisions: analytic rigid boundaries (ground top plane z=0, catch-basin
cylinder r<=1.35, 0<=z<=0.12).  Grid nodes inside the solids have their
velocity projected (no penetration, tangential friction); particles receive a
final position/velocity projection so nothing tunnels through the thin basin.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- contract ----------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=np.float64)
        m0 = np.asarray(frame0.mass, dtype=np.float64)

        spacing = float(cfg["sampling"]["spacing"])          # 0.14
        self.spacing = spacing
        rho = float(cfg["sampling"]["density"])              # 1350
        self.rho = rho
        self.g_vec = np.asarray(cfg["gravity"], dtype=np.float64)

        # material parameters (lava: stiff-ish elastic, strongly viscous,
        # elastoplastic with a modest yield window so folds accumulate)
        E, nu = 1.0e5, 0.25
        self.mu_e = E / (2.0 * (1.0 + nu))
        self.lam_e = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu_visc = 1200.0        # Pa*s, highly viscous
        self.theta = 0.07            # plastic yield window on singular values
        self.dt = 0.004

        # background grid (covers the particle domain with margin)
        dx = spacing
        self.dx = dx
        self.origin = np.array([-3.22, -1.54, -0.28])
        self.dims = np.array([46, 22, 32])
        self.nn = int(np.prod(self.dims))

        # static solid nodes: ground top (z<=0) and basin cylinder
        gx = self.origin[0] + np.arange(self.dims[0]) * dx
        gy = self.origin[1] + np.arange(self.dims[1]) * dx
        gz = self.origin[2] + np.arange(self.dims[2]) * dx
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        rr = np.hypot(GX, GY)
        solid = (GZ <= 1e-9) | ((rr <= 1.35) & (GZ < 0.12))
        self.solid_idx = np.flatnonzero(solid.ravel())

        # particle storage
        cap = 6000
        self.cap = cap
        self.x = np.zeros((cap, 3))
        self.v = np.zeros((cap, 3))
        self.Cp = np.zeros((cap, 3, 3))
        self.F = np.zeros((cap, 3, 3))
        self.m = np.zeros(cap)
        self.V0 = np.zeros(cap)
        self.rad = np.zeros(cap)
        n0 = len(pos)
        self.x[:n0] = pos
        self.v[:n0] = np.asarray(frame0.velocity, float)
        self.F[:n0] = np.eye(3)
        self.m[:n0] = m0
        self.V0[:n0] = m0 / rho
        self.rad[:n0] = 0.45 * spacing
        self.n = n0

        # emitter (nozzle) inflow
        self.emit_rate = 520.0                     # particles / s
        self.emit_pos_z = 2.92
        self.emit_r = 0.20
        self.emit_vz = -1.8
        self.emit_mass = rho * 0.07 ** 3           # per spawned particle
        self.emit_acc = 0.0
        self.emitted_mass = 0.0

        self.t = 0.0
        self.steps = 0
        self.frozen = dict(frame0.scalars)

        # grid work arrays
        self.m_g = np.zeros(self.nn)
        self.p_g = np.zeros((self.nn, 3))
        self.v_g = np.zeros((self.nn, 3))

    # ---------------- helpers ----------------
    def _weights(self, x):
        dx = self.dx
        fx = (x - self.origin) / dx - 0.5
        base = np.floor(fx).astype(np.int64)
        d = fx - base
        lo = np.zeros(3, np.int64)
        hi = self.dims - 3
        base = np.clip(base, lo, hi)
        n = len(x)
        w = np.empty((n, 3, 3))
        w[:, 0, :] = 0.5 * (1.0 - d) ** 2
        w[:, 1, :] = 0.75 - (d - 0.5) ** 2
        w[:, 2, :] = 0.5 * d ** 2
        dw = np.empty((n, 3, 3))
        dw[:, 0, :] = -(1.0 - d) / dx
        dw[:, 1, :] = -2.0 * (d - 0.5) / dx
        dw[:, 2, :] = d / dx
        k = np.arange(3)[None, :, None]
        dpos = (k - 0.5 - d[:, None, :]) * dx      # [n, axis-slot, xyz] -> dpos[:,k,ax]
        return base, w, dw, dpos

    def _emit(self, h):
        self.emit_acc += self.emit_rate * h
        k = int(self.emit_acc)
        if k <= 0:
            return
        self.emit_acc -= k
        k = min(k, self.cap - self.n)
        if k <= 0:
            return
        s, e = self.n, self.n + k
        rr = self.emit_r * np.sqrt(self.rng.random(k))
        th = 2.0 * np.pi * self.rng.random(k)
        self.x[s:e, 0] = rr * np.cos(th)
        self.x[s:e, 1] = rr * np.sin(th)
        self.x[s:e, 2] = self.emit_pos_z + 0.02 * self.rng.random(k)
        self.v[s:e, 0] = 0.05 * self.rng.standard_normal(k)
        self.v[s:e, 1] = 0.05 * self.rng.standard_normal(k)
        self.v[s:e, 2] = self.emit_vz
        self.Cp[s:e] = 0.0
        self.F[s:e] = np.eye(3)
        self.m[s:e] = self.emit_mass
        self.V0[s:e] = self.emit_mass / self.rho
        self.rad[s:e] = 0.0315
        self.n = e
        self.emitted_mass += k * self.emit_mass

    def _collide_particles(self):
        x, v = self.x[: self.n], self.v[: self.n]
        # catch basin cylinder (top z=0.12, radius 1.35)
        r = np.hypot(x[:, 0], x[:, 1])
        inside = (r < 1.35) & (x[:, 2] < 0.12)
        if np.any(inside):
            dz_out = 0.12 - x[inside, 2]
            dr_out = 1.35 - r[inside]
            top = dz_out <= dr_out
            ii = np.flatnonzero(inside)
            it, ir = ii[top], ii[~top]
            if len(it):
                x[it, 2] = 0.12
                v[it, 2] = np.maximum(v[it, 2], 0.0)
                v[it, 0] *= 0.85
                v[it, 1] *= 0.85
            if len(ir):
                rr = np.maximum(r[ir], 1e-12)
                nx, ny = x[ir, 0] / rr, x[ir, 1] / rr
                x[ir, 0] = nx * 1.35
                x[ir, 1] = ny * 1.35
                vr = v[ir, 0] * nx + v[ir, 1] * ny
                bad = vr < 0.0
                v[ir[bad], 0] -= vr[bad] * nx[bad]
                v[ir[bad], 1] -= vr[bad] * ny[bad]
        # ground plane z = 0
        g = x[:, 2] < 0.0
        if np.any(g):
            x[g, 2] = 0.0
            v[g, 2] = np.maximum(v[g, 2], 0.0)
            v[g, 0] *= 0.9
            v[g, 1] *= 0.9
        # keep inside the grid interior (quadratic support)
        lo = self.origin + 0.6 * self.dx
        hi = self.origin + (self.dims - 2.6) * self.dx
        np.clip(x, lo, hi, out=x)

    def _substep(self, h):
        n = self.n
        if n == 0:
            return
        dx, nn = self.dx, self.nn
        x, v, F, Cp = self.x[:n], self.v[:n], self.F[:n], self.Cp[:n]
        m, V0 = self.m[:n], self.V0[:n]

        # ---- constitutive: SVD, fixed-corotated stress, plastic return, viscosity
        U, S, Vt = np.linalg.svd(F)
        Sc = np.clip(S, 1.0 - self.theta, 1.0 + self.theta)
        Fhat = np.einsum("nij,nj,njk->nik", U, Sc, Vt)
        R = U @ Vt
        J = np.prod(S, axis=1)
        FinvT = np.einsum("nij,nj,njk->nik", U, 1.0 / np.maximum(S, 1e-12), Vt)
        P = 2.0 * self.mu_e * (F - R) + (self.lam_e * (J - 1.0) * J)[:, None, None] * FinvT
        Jc = np.prod(Sc, axis=1)
        sigma_v = self.mu_visc * (Cp + np.swapaxes(Cp, 1, 2))
        M = P @ np.swapaxes(Fhat, 1, 2) + Jc[:, None, None] * sigma_v

        # ---- P2G
        base, w, dw, dpos = self._weights(x)
        self.m_g[:] = 0.0
        self.p_g[:] = 0.0
        mg, pg = self.m_g, self.p_g
        nx, ny, nz = self.dims
        for a in range(3):
            ix = base[:, 0] + a
            wa, dwa = w[:, a, 0], dw[:, a, 0]
            for b in range(3):
                iy = base[:, 1] + b
                wb, dwb = w[:, b, 1], dw[:, b, 1]
                base_xz = (ix * ny + iy) * nz
                for c in range(3):
                    iz = base[:, 2] + c
                    flat = base_xz + iz
                    wc, dwc = w[:, c, 2], dw[:, c, 2]
                    ww = wa * wb * wc
                    dp = np.empty((n, 3))
                    dp[:, 0] = dpos[:, a, 0]
                    dp[:, 1] = dpos[:, b, 1]
                    dp[:, 2] = dpos[:, c, 2]
                    gr = np.empty((n, 3))
                    gr[:, 0] = dwa * wb * wc
                    gr[:, 1] = wa * dwb * wc
                    gr[:, 2] = wa * wb * dwc
                    wwm = ww * m
                    mg += np.bincount(flat, weights=wwm, minlength=nn)
                    vel = v + np.einsum("nij,nj->ni", Cp, dp)
                    frc = -h * (V0 * ww)[:, None] * np.einsum("nij,nj->ni", M, gr)
                    mom = wwm[:, None] * vel + frc
                    for ax in range(3):
                        pg[:, ax] += np.bincount(flat, weights=mom[:, ax], minlength=nn)
        # ---- grid solve
        vg = self.v_g
        occ = mg > 1e-9
        vg[:] = 0.0
        vg[occ] = pg[occ] / mg[occ, None] + h * self.g_vec
        sv = vg[self.solid_idx]
        sv[:, 2] = np.maximum(sv[:, 2], 0.0)
        sv[:, 0] *= 0.5
        sv[:, 1] *= 0.5
        vg[self.solid_idx] = sv

        # ---- G2P
        v[:] = 0.0
        Cp[:] = 0.0
        c4 = 4.0 / (dx * dx)
        for a in range(3):
            ix = base[:, 0] + a
            wa = w[:, a, 0]
            for b in range(3):
                iy = base[:, 1] + b
                wb = w[:, b, 1]
                base_xz = (ix * ny + iy) * nz
                for c in range(3):
                    flat = base_xz + base[:, 2] + c
                    wc = w[:, c, 2]
                    ww = wa * wb * wc
                    dp = np.empty((n, 3))
                    dp[:, 0] = dpos[:, a, 0]
                    dp[:, 1] = dpos[:, b, 1]
                    dp[:, 2] = dpos[:, c, 2]
                    g = vg[flat]
                    v += ww[:, None] * g
                    Cp += (c4 * ww)[:, None, None] * np.einsum("na,nb->nab", g, dp)
        # ---- integrate F and positions
        A = np.eye(3)[None] + h * Cp
        F[:] = np.einsum("nij,njk->nik", A, Fhat)
        x += h * v
        self._collide_particles()
        self.steps += 1

    # ---------------- main API ----------------
    def advance_to(self, time: float):
        time = float(time)
        eps = 1e-12
        while self.t < time - 1e-9:
            rem = time - self.t
            ns = max(1, int(np.ceil(rem / self.dt - 1e-6)))
            h = rem / ns
            for _ in range(ns):
                self._emit(h)
                self._substep(h)
                self.t += h
        return self._frame(time)

    def _frame(self, time):
        n = self.n
        pos = self.x[:n].copy()
        vel = self.v[:n].copy()
        mass = self.m[:n].copy()
        F = self.F[:n].copy()
        F[~np.isfinite(F).all(axis=(1, 2))] = np.eye(3)

        # contact signed distance to the rigid boundaries (ground + basin)
        z = pos[:, 2]
        r = np.hypot(pos[:, 0], pos[:, 1])
        dz = z - 0.12
        dr = r - 1.35
        d_basin = np.where(r <= 1.35, dz,
                           np.where(dz >= 0.0, np.hypot(dr, np.maximum(dz, 0.0)), dr))
        dist = np.minimum(z, d_basin)
        near = dist < 0.25
        if np.count_nonzero(near) < 4:
            near = np.zeros_like(near)
            near[np.argsort(dist)[:4]] = True
        csd = dist[near].astype(np.float64)

        ext = mass[:, None] * self.g_vec[None, :]
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=np.zeros(n, np.int32),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": F,
                "particle_radius": self.rad[:n].copy(),
                "material_label": np.zeros(n, np.int32),
            },
            fields={
                "cumulative_mass_source": np.array([self.emitted_mass]),
                "external_force": ext,
                "contact_signed_distance": csd,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": float(self.frozen.get("characteristic_length", 1.0)),
                "reference_velocity": float(self.frozen.get("reference_velocity", 1.0)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"n_particles": int(self.n), "substeps": int(self.steps),
                "emitted_kg": float(self.emitted_mass), "t": float(self.t)}
