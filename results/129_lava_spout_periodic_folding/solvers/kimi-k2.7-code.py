"""Particle-based viscous lava-spout solver.

A lightweight explicit particle/PIC-like scheme is used.  A stream of highly
viscous material is continuously injected from a nozzle, falls under gravity,
collides with a rigid receiving surface (catch basin + ground), and folds
periodically because the injection velocity carries a small deterministic
lateral oscillation.  Pairwise spring/damping forces between particles provide
cohesion and resistance to compression, while per-particle deformation gradients
are tracked with an SPH-style velocity gradient.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.rng = np.random.default_rng(int(seed))

        config = scene.config
        self.g = np.asarray(config["gravity"], dtype=float)

        # Material / particle parameters.
        self.spacing = float(config["sampling"]["spacing"])
        self.density = float(config["sampling"]["density"])
        self.vol = self.spacing ** 3
        self.mass_part = self.density * self.vol
        self.radius = 0.45 * self.spacing
        self.contact_length_scale = self.spacing

        # Numerical settings.
        self.dt = 1e-3
        self.k = 2.0e4            # contact stiffness
        self.zeta = 0.5           # damping ratio for pairwise contacts
        self.global_damp = 1.0    # viscous damping (1/s)
        self.friction = 0.4       # tangential velocity reduction on contact

        # Source / nozzle parameters.
        self.nozzle_radius = 0.27
        self.nozzle_z = 3.02
        self.emit_speed = 1.4
        self.emit_interval = self.spacing / self.emit_speed
        self.sway_amp = 0.12
        self.fold_period = 1.6
        self._disc = self._make_disc()

        # Initial state from the scene.
        f0 = scene.initial_state()
        self.pos = np.asarray(f0.position, dtype=float).copy()
        self.vel = np.asarray(f0.velocity, dtype=float).copy()
        self.mass = np.asarray(f0.mass, dtype=float).copy()
        self.body_id = np.asarray(f0.body_id, dtype=int).copy()
        self.F = np.asarray(f0.attributes["deformation_gradient"], dtype=float).copy()
        self.material_label = np.asarray(f0.attributes["material_label"], dtype=int).copy()
        self.scalars = dict(f0.scalars)
        self.scalars.setdefault("contact_length_scale", self.contact_length_scale)

        # Book-keeping.
        self.t = 0.0
        self.emit_timer = 0.0
        self.cum_mass_source = np.zeros(1, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        while self.t < time:
            dt = min(self.dt, time - self.t)
            self._substep(dt)
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "particle_viscous_lava",
            "particles": len(self.pos),
            "cumulative_injected_mass_kg": float(self.cum_mass_source[0]),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_disc(self) -> np.ndarray:
        """Return a 2-D point pattern covering the nozzle cross-section."""
        r = max(0.0, self.nozzle_radius - 0.5 * self.spacing)
        n = max(2, int(np.ceil(2.0 * r / self.spacing)) + 1)
        ax = np.linspace(-r, r, n)
        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        pts = pts[np.sum(pts ** 2, axis=1) <= self.nozzle_radius ** 2]
        return pts

    def _make_frame(self, time: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()
        body_id = self.body_id.copy()
        F = self.F.copy()
        material_label = self.material_label.copy()

        attributes = {
            "mass_quantity": mass.copy(),
            "deformation_gradient": F,
            "material_label": material_label,
            "particle_radius": np.full(len(mass), self.radius),
        }

        external_force = mass[:, None] * self.g

        fields = {
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "external_force": external_force,
            "contact_signed_distance": self._signed_distance(),
        }

        return StateFrame(
            float(time),
            pos,
            vel,
            mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=self.scalars.copy(),
        )

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        # Continuous injection from the nozzle.
        self.emit_timer += dt
        while self.emit_timer >= self.emit_interval:
            emit_t = self.t + self.emit_timer - 0.5 * self.emit_interval
            self._emit(emit_t)
            self.emit_timer -= self.emit_interval

        # Pairwise interactions + velocity gradient.
        acc, L = self._compute_acc_and_L()

        # Integrate velocity and position (symplectic Euler).
        self.vel += dt * (acc + self.g)
        self.vel *= 1.0 - min(self.global_damp * dt, 0.95)
        self.pos += dt * self.vel

        # Enforce rigid boundaries.
        self._apply_boundary()

        # Advect deformation gradient.
        if self.pos.shape[0] > 0:
            self.F += dt * np.einsum("nij,njk->nik", L, self.F)
            self._stabilize_F()

        self.t += dt

    def _compute_acc_and_L(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.pos.shape[0]
        acc = np.zeros_like(self.pos)
        L = np.zeros((N, 3, 3))

        if N < 2:
            return acc, L

        # Neighbour search with smoothing length as the cutoff.
        h = 2.0 * self.spacing
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(h, output_type="ndarray")

        if pairs.size == 0:
            return acc, L

        i = pairs["i"]
        j = pairs["j"]

        dx = self.pos[i] - self.pos[j]
        r = np.linalg.norm(dx, axis=1)
        r = np.where(r < 1e-12, 1e-12, r)
        n = dx / r[:, None]

        # Pairwise repulsive/damping force (DEM-like).
        vrel = self.vel[i] - self.vel[j]
        vdn = np.sum(vrel * n, axis=1)
        d0 = self.spacing
        overlap = d0 - r

        mi = self.mass[i]
        mj = self.mass[j]
        m_eff = mi * mj / (mi + mj)
        c_pair = 2.0 * self.zeta * np.sqrt(self.k * m_eff)

        fn = self.k * overlap - c_pair * vdn
        fn = np.where((overlap > 0.0) & (fn > 0.0), fn, 0.0)
        force = fn[:, None] * n

        np.add.at(acc, i, force / mi[:, None])
        np.add.at(acc, j, -force / mj[:, None])

        # SPH-style velocity gradient for the deformation gradient update.
        q = r / h
        q = np.clip(q, 0.0, 1.0)
        # 3-D cubic-spline-like kernel: W = alpha*(1-q^2)^2, dW/dr = alpha*(-4q(1-q^2)/h)
        alpha = 315.0 / (64.0 * np.pi * h ** 3)
        f = 1.0 - q ** 2
        dWdr = alpha * (-4.0 * q * f / h)

        dv = self.vel[j] - self.vel[i]
        gradW = n * dWdr[:, None]

        # L_i += V * (v_j - v_i) (x) grad_i W
        # Vectorised accumulation for each pair.
        weight = self.vol
        dL = weight * np.einsum("pi,pj->pij", dv, gradW)

        np.add.at(L, i, dL)
        np.add.at(L, j, dL)

        return acc, L

    def _apply_boundary(self) -> None:
        # Ground / catch-basin floor height.
        r_xy = np.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)
        floor_z = np.where(r_xy < 1.35, 0.12, 0.0)

        pen = floor_z + self.radius - self.pos[:, 2]
        contact = pen > 0.0
        if np.any(contact):
            self.pos[contact, 2] = floor_z[contact] + self.radius
            self.vel[contact, 2] = np.maximum(self.vel[contact, 2], 0.0)
            # Tangential friction to make the pile stick.
            self.vel[contact, :2] *= 1.0 - self.friction

        # Domain walls (soft bounce).
        bounds = [(-3.0, 3.0), (-1.2, 1.2), (0.0, 3.8)]
        for dim, (lo, hi) in enumerate(bounds):
            if dim == 2:
                # z already handled by the floor; just keep a ceiling.
                too_high = self.pos[:, 2] > hi
                if np.any(too_high):
                    self.pos[too_high, 2] = hi - 0.5 * (self.pos[too_high, 2] - hi)
                    self.vel[too_high, 2] *= -0.1
                continue
            too_low = self.pos[:, dim] < lo
            too_high = self.pos[:, dim] > hi
            if np.any(too_low):
                self.pos[too_low, dim] = lo + 0.5 * (lo - self.pos[too_low, dim])
                self.vel[too_low, dim] = 0.0
            if np.any(too_high):
                self.pos[too_high, dim] = hi - 0.5 * (self.pos[too_high, dim] - hi)
                self.vel[too_high, dim] = 0.0

    def _emit(self, t: float) -> None:
        if len(self._disc) == 0:
            return

        # Slightly rotate the pattern every layer to avoid persistent alignment.
        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        pts = self._disc @ rot.T
        pts += self.rng.normal(scale=0.003, size=pts.shape)

        n = pts.shape[0]
        new_pos = np.zeros((n, 3), dtype=float)
        new_pos[:, :2] = pts
        new_pos[:, 2] = self.nozzle_z

        phase = 2.0 * np.pi * t / self.fold_period
        new_vel = np.zeros((n, 3), dtype=float)
        new_vel[:, 0] = self.sway_amp * np.sin(phase)
        new_vel[:, 1] = self.sway_amp * 0.5 * np.cos(phase)
        new_vel[:, 2] = -self.emit_speed
        new_vel += self.rng.normal(scale=0.02, size=new_vel.shape)

        m = np.full(n, self.mass_part, dtype=float)

        self.pos = np.vstack([self.pos, new_pos])
        self.vel = np.vstack([self.vel, new_vel])
        self.mass = np.concatenate([self.mass, m])
        self.body_id = np.concatenate([self.body_id, np.zeros(n, dtype=int)])
        self.material_label = np.concatenate([self.material_label, np.zeros(n, dtype=int)])
        F0 = np.repeat(np.eye(3, dtype=float)[None], n, axis=0)
        self.F = np.vstack([self.F, F0])

        self.cum_mass_source[0] += float(n) * self.mass_part

    def _stabilize_F(self) -> None:
        det = np.linalg.det(self.F)
        bad = ~np.isfinite(det) | (det <= 0.0) | (det > 1e3)
        if np.any(bad):
            self.F[bad] = np.eye(3, dtype=float)

    def _signed_distance(self) -> np.ndarray:
        # Signed distance to the ground / catch-basin composite surface.
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        z = self.pos[:, 2]
        r = np.sqrt(x * x + y * y)

        # Floor: 0.12 m inside basin radius, 0 otherwise.
        floor_z = np.where(r < 1.35, 0.12, 0.0)
        sd_floor = z - floor_z

        # Catch-basin cylinder (solid when r <= 1.35 and 0 <= z <= 0.12).
        dr = r - 1.35
        dz_top = z - 0.12
        dz_bot = -z
        inside = (r <= 1.35) & (z >= 0.0) & (z <= 0.12)
        sd_basin = np.where(
            inside,
            np.maximum(np.maximum(dr, dz_top), dz_bot),
            np.sqrt(
                np.maximum(0.0, dr) ** 2
                + np.maximum(0.0, dz_top) ** 2
                + np.maximum(0.0, dz_bot) ** 2,
            ),
        )

        return np.minimum(sd_floor, sd_basin)
