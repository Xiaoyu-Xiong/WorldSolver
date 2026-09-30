"""Physically-based inclined-cloth solver.

Simulates a square cloth sheet (mass-spring membrane) that free-falls onto a
rigid inclined plane and then slides under gravity with Coulomb friction.

Method
------
* Semi-implicit (symplectic) Euler with fixed sub-stepping.
* Cloth elasticity: structural / shear / bend springs with internal damping.
* Contact: impulse-based non-penetration (restitution 0) + Coulomb friction
  tied to the normal impulse, plus positional projection out of the plane.
* Diagnostics exported as raw physical fields; the platform computes residuals.

The scheme reproduces the analytical sliding acceleration
``a = g (sin(theta) - mu cos(theta))`` for the downslope motion once the sheet
is in persistent contact, because the friction impulse per step is
``mu * j_n`` with ``j_n = m g cos(theta) dt``.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg

        init = scene.initial_state()
        self.ref_pos = np.asarray(init.position, dtype=float)      # [N,3]
        self.N = int(self.ref_pos.shape[0])
        self.mass = np.asarray(init.mass, dtype=float)             # [N]
        self.uv = np.asarray(init.attributes["uv"], dtype=float)   # [N,2]
        self.vel = np.asarray(init.velocity, dtype=float).copy()   # [N,3]

        meta = scene.metadata()
        self.faces = np.asarray(meta["topology"]["faces"], dtype=np.int64)  # [Ne,3]
        nx, ny = map(int, cfg["cloth"]["grid_shape"])
        self.nx, self.ny = nx, ny

        # Gravity.
        self.g = np.asarray(cfg["gravity"], dtype=float)           # [3]

        # Inclined plane (matches scene.py construction).
        theta = np.deg2rad(float(cfg["incline"]["angle_degrees"]))
        self.n = np.array([-np.sin(theta), 0.0, np.cos(theta)])    # outward unit normal
        plane_pt = np.asarray(cfg["incline"]["plane_translation"], dtype=float)
        half_thick = float(cfg["incline"]["plane_half_thickness"])
        self.surface_point = plane_pt + half_thick * self.n        # contact-surface point

        # Contact / friction parameters.
        self.mu = float(cfg["friction"]["coefficient"])
        self.half_cloth = float(cfg["contact"]["thickness"]) / 2.0

        # Frozen length scales for exported fields.
        side = float(cfg["cloth"]["side_length"])
        self.mean_elem = side / float(max(nx, ny) - 1)
        self.contact_length_scale = float(cfg["contact"]["thickness"])
        self.constraint_scale = float(self.mean_elem)

        # Spring network + reference deformation data.
        self._build_edges()
        self._build_reference_deformation()

        # Dynamic state.
        self.pos = self.ref_pos.copy()
        self.t = 0.0
        self.sub_dt = 2.5e-4
        self._steps = 0

    # ------------------------------------------------------------- topology
    def _build_edges(self) -> None:
        nx, ny = self.nx, self.ny
        ref = self.ref_pos

        def vid(i, j):
            return j * nx + i

        ea, eb, ks = [], [], []
        # Moderate stiffness: the sheet undergoes near-rigid motion (uniform
        # gravity + uniform contact), so internal stress is tiny. These values
        # keep the worst-case nodal spring mode well inside the symplectic-Euler
        # stability limit (omega*dt < 2) at sub_dt = 2.5e-4.
        k_struct, k_shear, k_bend = 200.0, 100.0, 20.0

        # Structural (grid neighbours).
        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx:
                    ea.append(vid(i, j)); eb.append(vid(i + 1, j)); ks.append(k_struct)
                if j + 1 < ny:
                    ea.append(vid(i, j)); eb.append(vid(i, j + 1)); ks.append(k_struct)
        # Shear (diagonals).
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b = vid(i, j), vid(i + 1, j + 1)
                c, d = vid(i + 1, j), vid(i, j + 1)
                ea.append(a); eb.append(b); ks.append(k_shear)
                ea.append(c); eb.append(d); ks.append(k_shear)
        # Bend (second neighbours).
        for j in range(ny):
            for i in range(nx):
                if i + 2 < nx:
                    ea.append(vid(i, j)); eb.append(vid(i + 2, j)); ks.append(k_bend)
                if j + 2 < ny:
                    ea.append(vid(i, j)); eb.append(vid(i, j + 2)); ks.append(k_bend)

        self.ea = np.asarray(ea, dtype=np.int64)
        self.eb = np.asarray(eb, dtype=np.int64)
        self.k = np.asarray(ks, dtype=float)
        d = ref[self.eb] - ref[self.ea]
        self.rest = np.linalg.norm(d, axis=1)
        # Internal (relative-velocity) damping; keeps centre-of-mass motion physical.
        zeta = 0.3
        m_red = 0.5 * self.mass[self.ea]
        self.c = zeta * 2.0 * np.sqrt(self.k * m_red)

    def _build_reference_deformation(self) -> None:
        f = self.faces
        X0 = self.ref_pos[f[:, 0]]
        X1 = self.ref_pos[f[:, 1]]
        X2 = self.ref_pos[f[:, 2]]
        Dm = np.stack([X1 - X0, X2 - X0], axis=2)          # [Ne,3,2]
        self.ref_pinv = np.linalg.pinv(Dm)                 # [Ne,2,3]

    # -------------------------------------------------------------- physics
    def _spring_forces(self) -> np.ndarray:
        forces = np.zeros((self.N, 3), dtype=float)
        d = self.pos[self.eb] - self.pos[self.ea]
        dist = np.linalg.norm(d, axis=1)
        dsafe = np.where(dist > 1e-12, dist, 1e-12)
        dirn = d / dsafe[:, None]
        dv = self.vel[self.eb] - self.vel[self.ea]
        rel = np.einsum("ij,ij->i", dv, dirn)
        fmag = self.k * (dist - self.rest) + self.c * rel
        fvec = fmag[:, None] * dirn
        np.add.at(forces, self.ea, fvec)
        np.add.at(forces, self.eb, -fvec)
        return forces

    def _contact(self) -> None:
        n = self.n
        d = (self.pos - self.surface_point) @ n
        gap = d - self.half_cloth
        idx = np.flatnonzero(gap < 0.0)
        if idx.size == 0:
            return
        vi = self.vel[idx]
        vn = vi @ n                                   # normal component
        approaching = vn < 0.0
        jn = self.mass[idx] * np.maximum(0.0, -vn)    # normal impulse magnitude
        vt = vi - vn[:, None] * n[None, :]            # tangential velocity
        speed = np.linalg.norm(vt, axis=1)
        fric = self.mu * jn / self.mass[idx]          # friction dv magnitude
        fric = np.minimum(fric, speed)                # do not reverse motion
        dir_t = np.zeros_like(vt)
        nz = speed > 1e-12
        dir_t[nz] = vt[nz] / speed[nz, None]
        new_vi = vt - fric[:, None] * dir_t           # normal velocity zeroed
        keep = ~approaching                           # separating contacts: unchanged
        new_vi[keep] = vi[keep]
        self.vel[idx] = new_vi
        # Positional projection out of penetration (non-penetration constraint).
        self.pos[idx] += (-gap[idx])[:, None] * n[None, :]

    def _step(self, dt: float) -> None:
        forces = self.mass[:, None] * self.g[None, :]
        forces = forces + self._spring_forces()
        self.vel += dt * forces / self.mass[:, None]
        self.pos += dt * self.vel
        self._contact()
        self._steps += 1

    # -------------------------------------------------------------- output
    def _deformation_gradients(self) -> np.ndarray:
        f = self.faces
        x0 = self.pos[f[:, 0]]
        x1 = self.pos[f[:, 1]]
        x2 = self.pos[f[:, 2]]
        Ds = np.stack([x1 - x0, x2 - x0], axis=2)     # [Ne,3,2]
        F_tri = Ds @ self.ref_pinv                    # [Ne,3,3]
        Fv = np.zeros((self.N, 3, 3), dtype=float)
        cnt = np.zeros(self.N, dtype=float)
        for c in range(3):
            np.add.at(Fv, f[:, c], F_tri)
            np.add.at(cnt, f[:, c], 1.0)
        cnt[cnt == 0.0] = 1.0
        return Fv / cnt[:, None, None]

    def _make_frame(self, time: float) -> StateFrame:
        d = (self.pos - self.surface_point) @ self.n
        gap = d - self.half_cloth                     # signed geometric gap [N]
        F = self._deformation_gradients()
        attributes = {
            "deformation_gradient": F,                # [N,3,3]
            "mass_quantity": self.mass.copy(),        # [N]
            "uv": self.uv.copy(),
            "cloth_vertex": np.ones(self.N, dtype=float),
        }
        fields = {
            "constraint_value": gap.copy(),                    # [C] signed gap
            "constraint_target": np.zeros(self.N, dtype=float),# [C] non-penetration target
            "contact_signed_distance": gap.copy(),             # [C]
            "cumulative_mass_source": np.zeros(1, dtype=float),# [K]
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "contact_length_scale": float(self.contact_length_scale),
            "friction_coefficient": float(self.mu),
        }
        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        total = time - self.t
        if total > 0.0:
            n_sub = max(1, int(np.ceil(total / self.sub_dt)))
            dt = total / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.t = time
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        # Downhill unit vector recovered from the plane normal
        # n = [-sin(theta), 0, cos(theta)].
        theta = np.arctan2(-self.n[0], self.n[2])
        downhill = np.array([-np.cos(theta), 0.0, -np.sin(theta)])
        sin_t = float(-self.n[0])
        cos_t = float(self.n[2])
        g_mag = float(np.linalg.norm(self.g))
        com = self.pos.mean(axis=0)
        disp = float((com - self.ref_pos.mean(axis=0)) @ downhill)
        vel = float(self.vel.mean(axis=0) @ downhill)
        return {
            "solver": "inclined_cloth_mass_spring",
            "substeps": int(self._steps),
            "downslope_displacement": disp,
            "downslope_velocity": vel,
            "friction_coefficient": float(self.mu),
            "analytical_sliding_acceleration": float(
                g_mag * max(0.0, sin_t - self.mu * cos_t)
            ),
        }
