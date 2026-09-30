"""Drucker-Prager elastoplastic sand bed with a rigid spherical impactor.

The sand is discretised with the Material Point Method (MLS/APIC style
transfer, quadratic B-spline shape functions on a fixed background grid).
Each material point carries a deformation gradient ``F``; the constitutive
response is a Hencky (log-strain) elastic law combined with a Drucker-Prager
plastic return mapping on the principal Kirchhoff stresses, following the
model used for sand animation by Klár et al.  The impact ball is a rigid
body that exchanges momentum with the sand through the same background grid
plus a particle level non-penetration projection.  Everything is advanced
with an explicit, CFL limited time stepping scheme.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
_IDENTITY = np.eye(3, dtype=float)


class BenchmarkSolver:
    """Explicit MPM sand solver with a dynamic rigid sphere."""

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self._configure(cfg)

        frame = scene.initial_state()
        self.pos = np.asarray(frame.position, dtype=float).copy()
        self.vel = np.zeros_like(self.pos)
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.body = np.asarray(frame.body_id).astype(np.int64).copy()
        self.n = int(self.pos.shape[0])

        attr_radius = frame.attributes.get("particle_radius")
        if attr_radius is None:
            self.radius = np.full(self.n, 0.45 * self.spacing)
        else:
            self.radius = np.asarray(attr_radius, dtype=float).copy()

        self.sand = self.body == 0
        self.ball_mask = self.body == 1
        self.sand_index = np.flatnonzero(self.sand)
        self.ball_index = np.flatnonzero(self.ball_mask)

        self.F = np.tile(_IDENTITY, (self.n, 1, 1))
        self.B = np.zeros((self.n, 3, 3), dtype=float)
        self.vol0 = self.mass / self.density
        self.vol0[self.vol0 <= 0.0] = self.spacing ** 3

        self.has_ball = self.ball_index.size > 0
        if self.has_ball:
            bi = int(self.ball_index[0])
            self.ball_mass = float(self.mass[bi])
            self.ball_pos = self.pos[bi].copy()
        else:
            self.ball_mass = 0.0
            self.ball_pos = np.zeros(3)
        self.ball_vel = np.zeros(3)

        self.gnode_pos = self._grid_nodes()
        self.time = 0.0

    def _configure(self, cfg) -> None:
        sampling = cfg.get("sampling", {})
        self.density = float(sampling.get("density", 1350.0))
        self.spacing = float(sampling.get("spacing", 0.12))
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        # Substeps per sample: keeps the explicit CFL number comfortably below
        # one for the frozen sample spacing and elastic wave speed.
        self.n_sub = 16
        self.dt_sub = self.sample_dt / self.n_sub

        geometry = cfg.get("scene_geometry") or {}
        objects = {obj.get("id"): obj for obj in geometry.get("objects", [])}
        sg = (objects.get("sand", {}) or {}).get("geometry", {}) or {}
        bmin = np.asarray(sg.get("bounds_min", [-3.0, -1.2, 0.0]), dtype=float)
        bmax = np.asarray(sg.get("bounds_max", [3.0, 1.2, 3.8]), dtype=float)
        self.gmin = np.minimum(bmin, bmax)
        self.gmax = np.maximum(bmin, bmax)
        shape = np.asarray(sg.get("grid_shape", [24, 16, 28]), dtype=np.int64)
        self.cells = np.maximum(shape, 2)
        self.nnodes = self.cells + 1
        self.nnode = int(self.nnodes[0]) * int(self.nnodes[1]) * int(self.nnodes[2])
        self.dx = (self.gmax - self.gmin) / self.cells.astype(float)

        bg = (objects.get("ball", {}) or {}).get("geometry", {}) or {}
        self.ball_radius = float(bg.get("radius", 0.54))

        young = 3.0e5
        poisson = 0.3
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        # Drucker-Prager: ||dev tau|| <= cohesion + alpha * p (p in compression).
        self.cohesion = 250.0
        self.dp_alpha = 1.0
        self.friction = 0.5

    def _grid_nodes(self) -> np.ndarray:
        axes = [self.gmin[i] + np.arange(self.nnodes[i]) * self.dx[i] for i in range(3)]
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        return np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    # ------------------------------------------------------------- mechanics
    def _shape(self, x: np.ndarray):
        """Quadratic B-spline weights/gradients for the 27 neighbouring nodes."""
        n = x.shape[0]
        p = (x - self.gmin) / self.dx
        base = np.floor(p - 0.5).astype(np.int64)
        frac = p - base
        bc = np.clip(base, 0, self.cells - 1)

        w = np.empty((3, n, 3), dtype=float)
        dw = np.empty((3, n, 3), dtype=float)
        w[0] = 0.5 * (1.0 - frac) ** 2
        dw[0] = -(1.0 - frac)
        w[1] = 0.75 - (frac - 0.5) ** 2
        dw[1] = -2.0 * (frac - 0.5)
        w[2] = 0.5 * frac ** 2
        dw[2] = frac

        offsets = np.array([-1, 0, 1], dtype=np.int64)
        node = bc[:, :, None] + offsets[None, None, :]          # [n,3,3]
        np.clip(node, 0, (self.nnodes - 1)[None, None, :], out=node)

        ny, nz = int(self.nnodes[1]), int(self.nnodes[2])
        weight = np.empty((27, n), dtype=float)
        grad = np.empty((27, n, 3), dtype=float)
        index = np.empty((27, n), dtype=np.int64)
        xnode = np.empty((27, n, 3), dtype=float)
        k = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    wa, wb, wc = w[a, :, 0], w[b, :, 1], w[c, :, 2]
                    weight[k] = wa * wb * wc
                    grad[k, :, 0] = dw[a, :, 0] * wb * wc / self.dx[0]
                    grad[k, :, 1] = wa * dw[b, :, 1] * wc / self.dx[1]
                    grad[k, :, 2] = wa * wb * dw[c, :, 2] / self.dx[2]
                    ia, ib, ic = node[:, a, 0], node[:, b, 1], node[:, c, 2]
                    index[k] = ia * ny * nz + ib * nz + ic
                    xnode[k, :, 0] = self.gmin[0] + ia * self.dx[0]
                    xnode[k, :, 1] = self.gmin[1] + ib * self.dx[1]
                    xnode[k, :, 2] = self.gmin[2] + ic * self.dx[2]
                    k += 1
        return weight, grad, index, xnode

    def _stress(self, F: np.ndarray):
        """Hencky elasticity plus Drucker-Prager return mapping.

        Returns the Kirchhoff stress used for the momentum transfer together
        with the plastically admissibility-restored deformation gradient.
        """
        C = np.einsum("nki,nkj->nij", F, F)
        eig, vec = np.linalg.eigh(C)
        s = np.clip(np.sqrt(np.clip(eig, 1e-12, None)), 0.4, 2.5)
        U = np.einsum("nij,njk->nik", F, vec) / s[:, None, :]
        eps = np.log(s)
        tr = eps.sum(axis=1)

        # principal Kirchhoff stresses (Hencky elasticity)
        tau_p = 2.0 * self.mu * eps + self.lam * tr[:, None]
        mean_tau = tau_p.sum(axis=1)
        dev = tau_p - (mean_tau / 3.0)[:, None]

        pressure = -mean_tau / 3.0                       # positive in compaction
        limit = np.maximum(self.cohesion + self.dp_alpha * pressure, 0.0)
        q = np.linalg.norm(dev, axis=1)
        scale = np.minimum(1.0, limit / np.maximum(q, 1e-12))
        dev = dev * scale[:, None]

        tau_p = dev + (mean_tau / 3.0)[:, None]
        # rotate the principal stresses back into the spatial frame
        tau = np.einsum("nij,nj,nkj->nik", U, tau_p, U)
        eps_new = dev / (2.0 * self.mu) + (mean_tau / (3.0 * self.lam + 2.0 * self.mu))[:, None]
        s_new = np.exp(np.clip(eps_new, -0.8, 0.8))
        F_new = np.einsum("nij,nj,nkj->nik", U, s_new, vec)
        return tau, F_new

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        pos, vel, mass = self.pos, self.vel, self.mass
        idx = self.sand_index
        pos_s = pos[idx]
        vel_s = vel[idx]
        m_s = mass[idx]
        F_s = self.F[idx]
        B_s = self.B[idx]

        if self.has_ball:
            self.ball_vel = self.ball_vel + dt * _GRAVITY

        weight, grad, node_index, xnode = self._shape(pos_s)
        flat = node_index.ravel()
        wm = weight * m_s[None, :]
        M = np.bincount(flat, weights=wm.ravel(), minlength=self.nnode)

        dy = xnode - pos_s[None, :, :]
        vp = vel_s[None, :, :] + np.einsum("nij,knj->kni", B_s, dy)
        mom = wm[:, :, None] * vp
        MOM = np.empty((self.nnode, 3), dtype=float)
        for c in range(3):
            MOM[:, c] = np.bincount(flat, weights=mom[:, :, c].ravel(), minlength=self.nnode)

        gv = np.zeros((self.nnode, 3), dtype=float)
        mass_floor = 1e-6 * float(m_s.min())
        active = M > mass_floor
        gv[active] = MOM[active] / M[active, None]

        tau, _ = self._stress(F_s)
        fvol = tau * (-self.vol0[idx])[:, None, None]
        fint = np.einsum("nij,knj->kni", fvol, grad)
        FINT = np.empty((self.nnode, 3), dtype=float)
        for c in range(3):
            FINT[:, c] = np.bincount(flat, weights=fint[:, :, c].ravel(), minlength=self.nnode)

        acc = np.zeros((self.nnode, 3), dtype=float)
        acc[active] = FINT[active] / M[active, None] + _GRAVITY
        anorm = np.linalg.norm(acc, axis=1)
        fast = anorm > 1.0e5
        if np.any(fast):
            acc[fast] *= (1.0e5 / anorm[fast])[:, None]
        gv = gv + dt * acc
        gv = np.clip(gv, -80.0, 80.0)

        if self.has_ball:
            self._ball_grid_contact(gv, M)

        # rigid container floor at z = 0 (top face of the frozen ground box)
        floor = self.gnode_pos[:, 2] <= 1e-9
        if np.any(floor):
            gv[floor, 2] = np.maximum(gv[floor, 2], 0.0)
            gv[floor, 0] *= 0.9
            gv[floor, 1] *= 0.9

        vnode = gv[node_index]
        v_new = np.einsum("kn,kni->ni", weight, vnode)
        B_new = np.einsum("kn,kni,knj->nij", weight, vnode, dy)
        G = np.einsum("kn,kni,knj->nij", weight, vnode, grad)
        F_upd = F_s + dt * np.einsum("nij,njk->nik", G, F_s)
        _, F_pl = self._stress(F_upd)

        self.B[idx] = B_new
        self.F[idx] = F_pl
        pos_s = pos_s + dt * v_new
        vel_s = v_new

        if self.has_ball:
            bp = self.ball_pos
            bv = self.ball_vel
            d = pos_s - bp
            dist = np.linalg.norm(d, axis=1)
            rad = self.radius[idx]
            pen = (self.ball_radius + rad) - dist
            inside = pen > 0.0
            if np.any(inside):
                safe = np.maximum(dist[inside], 1e-9)
                nrm = d[inside] / safe[:, None]
                pos_s[inside] = bp + nrm * (self.ball_radius + rad[inside])[:, None]
                vrel = vel_s[inside] - bv[None, :]
                vn = np.einsum("ij,ij->i", vrel, nrm)
                vtan = vrel - vn[:, None] * nrm
                vn = np.maximum(vn, 0.0)
                vel_s[inside] = bv[None, :] + vn[:, None] * nrm + 0.6 * vtan

        below = pos_s[:, 2] < 0.0
        if np.any(below):
            pos_s[below, 2] = 0.0
            vel_s[below, 2] = np.maximum(vel_s[below, 2], 0.0)
            vel_s[below, 0] *= 0.9
            vel_s[below, 1] *= 0.9

        np.clip(pos_s[:, 0], self.gmin[0] + 1e-9, self.gmax[0] - 1e-9, out=pos_s[:, 0])
        np.clip(pos_s[:, 1], self.gmin[1] + 1e-9, self.gmax[1] - 1e-9, out=pos_s[:, 1])
        np.clip(pos_s[:, 2], 0.0, self.gmax[2] - 1e-9, out=pos_s[:, 2])

        speed = np.linalg.norm(vel_s, axis=1)
        too_fast = speed > 60.0
        if np.any(too_fast):
            vel_s[too_fast] *= (60.0 / speed[too_fast])[:, None]

        pos[idx] = pos_s
        vel[idx] = vel_s

        if self.has_ball:
            self.ball_pos = self.ball_pos + dt * self.ball_vel
            if self.ball_pos[2] < self.ball_radius:
                self.ball_pos[2] = self.ball_radius
                self.ball_vel[2] = max(self.ball_vel[2], 0.0)
                self.ball_vel[0] *= 0.9
                self.ball_vel[1] *= 0.9
            self.pos[self.ball_index[0]] = self.ball_pos
            self.vel[self.ball_index[0]] = self.ball_vel

        self.pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.F = np.nan_to_num(self.F, nan=0.0, posinf=0.0, neginf=0.0)

    def _ball_grid_contact(self, gv: np.ndarray, M: np.ndarray) -> None:
        """Momentum conserving inelastic frictional contact on grid nodes."""
        rel = self.gnode_pos - self.ball_pos[None, :]
        dist = np.linalg.norm(rel, axis=1)
        hit = dist < self.ball_radius
        if not np.any(hit):
            return
        keep = hit & (M > 1e-12)
        if not np.any(keep):
            return
        ii = np.flatnonzero(keep)
        nrm = rel[ii] / np.maximum(dist[ii], 1e-9)[:, None]
        bv = self.ball_vel
        vrel = gv[ii] - bv[None, :]
        vn = np.einsum("ij,ij->i", vrel, nrm)
        vtan = vrel - vn[:, None] * nrm
        approach = np.maximum(-vn, 0.0)
        vn_new = np.maximum(vn, 0.0)
        tmag = np.linalg.norm(vtan, axis=1)
        tmax = self.friction * approach
        shrink = np.where(tmag > 1e-12, np.minimum(1.0, tmax / np.maximum(tmag, 1e-12)), 0.0)
        vrel_new = vn_new[:, None] * nrm + shrink[:, None] * vtan
        v_new = bv[None, :] + vrel_new
        impulse = M[ii, None] * (v_new - gv[ii])
        gv[ii] = v_new
        if self.ball_mass > 0.0:
            self.ball_vel = bv - impulse.sum(axis=0) / self.ball_mass

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-12:
            dt = min(self.dt_sub, target - self.time)
            self._step(dt)
            self.time += dt
            guard += 1
            if guard > 200000:
                break
        if self.time > target:
            self.time = target
        return self._frame(target)

    def _contact_distance(self) -> np.ndarray:
        """Signed gap to the nearest obstacle; negative means penetration."""
        out = self.pos[:, 2] - 0.0                       # gap to the rigid floor
        if self.has_ball and self.sand_index.size:
            sand = self.sand_index
            gap = np.linalg.norm(self.pos[sand] - self.ball_pos[None, :], axis=1)
            gap = gap - (self.ball_radius + self.radius[sand])
            out[sand] = np.minimum(out[sand], gap)
            ball_gap = min(float(self.ball_pos[2] - self.ball_radius), float(gap.min()))
            out[self.ball_index] = ball_gap
        return np.asarray(out, dtype=float)

    def _frame(self, time: float) -> StateFrame:
        n = self.n
        mass = self.mass.astype(float)
        external = mass[:, None] * _GRAVITY[None, :]
        frame = StateFrame(
            float(time),
            self.pos.astype(float).copy(),
            self.vel.astype(float).copy(),
            mass=mass.copy(),
            body_id=self.body.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": self.F.astype(float).copy(),
                "particle_radius": self.radius.astype(float).copy(),
            },
            fields={
                "contact_signed_distance": self._contact_distance(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(np.min(self.dx)),
                "characteristic_length": float(self.spacing),
            },
        )
        if frame.position.shape[0] != n:
            raise RuntimeError("state size changed during simulation")
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "solver": "mpm_drucker_prager_rigid_ball",
            "substeps_per_sample": int(self.n_sub),
            "dt_sub": float(self.dt_sub),
            "particle_count": int(self.n),
            "sand_count": int(self.sand_index.size),
            "grid_nodes": int(self.nnode),
            "time": float(self.time),
            "ball_position": self.ball_pos.tolist(),
            "ball_velocity": self.ball_vel.tolist(),
        }
