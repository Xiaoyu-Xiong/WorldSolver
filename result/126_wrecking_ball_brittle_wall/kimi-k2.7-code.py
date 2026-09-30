"""Solver Agent-owned implementation for the wrecking ball brittle wall scene.

A simple explicit elasto-brittle particle model is used for the masonry wall.
A kinematic wrecking ball swings on a pendular arc and collides with the wall,
breaking cohesive bonds near the impact zone.  Gravity and floor contact then
drive the resulting debris.
"""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = scene.config
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.spacing = float(config["sampling"]["spacing"])
        self.density = float(config["sampling"]["density"])
        self.vol = self.spacing ** 3

        init = scene.initial_state()
        self.pos = init.position.copy()
        self.vel = init.velocity.copy()
        self.mass = init.mass.copy()
        self.body_id = (
            init.body_id.copy()
            if init.body_id is not None
            else np.zeros(len(self.pos), dtype=np.int32)
        )
        self.n = len(self.pos)

        self.radius = init.attributes.get(
            "particle_radius", np.full(self.n, 0.45 * self.spacing)
        ).copy()
        self.mass_quantity = init.attributes.get(
            "mass_quantity", self.mass.copy()
        ).copy()
        self.material_label = init.attributes.get(
            "material_label", self.body_id.copy()
        ).copy()
        self.F0 = init.attributes.get(
            "deformation_gradient", np.repeat(np.eye(3)[None], self.n, axis=0)
        ).copy()

        self.base_scalars = dict(init.scalars) if init.scalars else {}
        self.base_scalars["contact_length_scale"] = self.spacing
        self.base_fields = dict(init.fields) if init.fields else {}

        # Cohesive bonds between initially close particles.
        self.bond_i = []
        self.bond_j = []
        self.bond_rest = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = np.linalg.norm(self.pos[i] - self.pos[j])
                if d < 1.25 * self.spacing and d > 1e-9:
                    self.bond_i.append(i)
                    self.bond_j.append(j)
                    self.bond_rest.append(d)
        self.bond_i = np.asarray(self.bond_i, dtype=np.int64)
        self.bond_j = np.asarray(self.bond_j, dtype=np.int64)
        self.bond_rest = np.asarray(self.bond_rest, dtype=float)
        self.bond_alive = np.ones(len(self.bond_i), dtype=bool)

        rng = np.random.default_rng(seed)
        self.bond_break_threshold = rng.uniform(0.07, 0.13, size=len(self.bond_i))

        self.bonded_pair = np.zeros((self.n, self.n), dtype=bool)
        self.bonded_pair[self.bond_i, self.bond_j] = True
        self.bonded_pair[self.bond_j, self.bond_i] = True

        # Neighbor lists for computing per-particle deformation gradients.
        self.F_neighbor_idx = []
        self.F_init_offsets = []
        cutoff_F = 1.6 * self.spacing
        for i in range(self.n):
            idx = []
            for j in range(self.n):
                if i == j:
                    continue
                d = np.linalg.norm(self.pos[j] - self.pos[i])
                if d < cutoff_F:
                    idx.append(j)
            idx = np.asarray(idx, dtype=np.int64)
            self.F_neighbor_idx.append(idx)
            if len(idx):
                self.F_init_offsets.append((self.pos[idx] - self.pos[i]).T)
            else:
                self.F_init_offsets.append(np.zeros((3, 0)))

        self.contact_i, self.contact_j = np.triu_indices(self.n, k=1)

        # Kinematic wrecking ball on a pendulum arc.
        self.ball_pivot = np.array([-1.75, 0.0, 3.35])
        self.ball_L = 2.0
        self.ball_theta0 = math.radians(-55.0)
        self.ball_omega = math.sqrt(9.81 / self.ball_L)
        self.ball_R = 0.48

        # Explicit integration parameters.
        self.dt = 5.0e-4
        self.k_bond = 8.0e4
        self.k_contact_ball = 2.0e5
        self.k_contact_ground = 5.0e5
        self.k_particle = 1.0e5
        self.damping_global = 1.0
        m = float(self.mass[0])
        self.c_damp_ball = 2.0 * 0.2 * math.sqrt(self.k_contact_ball * m)
        self.c_damp_ground = 2.0 * 0.2 * math.sqrt(self.k_contact_ground * m)
        self.c_damp_bond = 2.0 * 0.05 * math.sqrt(self.k_bond * (m / 2.0))
        self.c_damp_particle = 2.0 * 0.2 * math.sqrt(self.k_particle * (m / 2.0))

        self.current_time = 0.0

    def _ball_state(self, t: float):
        theta = self.ball_theta0 * math.cos(self.ball_omega * t)
        dtheta = -self.ball_theta0 * self.ball_omega * math.sin(self.ball_omega * t)
        c = math.cos(theta)
        s = math.sin(theta)
        x = self.ball_pivot[0] + self.ball_L * s
        z = self.ball_pivot[2] - self.ball_L * c
        center = np.array([x, 0.0, z])
        v = np.array([self.ball_L * c * dtheta, 0.0, self.ball_L * s * dtheta])
        return center, v

    def _compute_F(self) -> np.ndarray:
        F = np.zeros((self.n, 3, 3))
        for i in range(self.n):
            idx = self.F_neighbor_idx[i]
            if len(idx) < 3:
                F[i] = np.eye(3)
                continue
            X = self.F_init_offsets[i]
            x = (self.pos[idx] - self.pos[i]).T
            M = X @ X.T
            try:
                Minv = np.linalg.inv(M)
                F[i] = x @ X.T @ Minv
            except np.linalg.LinAlgError:
                F[i] = np.eye(3)
        return F

    def _add_particle_contact(self, f: np.ndarray) -> None:
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        i, j = self.contact_i, self.contact_j
        mask = ~self.bonded_pair[i, j]
        i = i[mask]
        j = j[mask]
        d = dist[i, j]
        r_sum = self.radius[i] + self.radius[j]
        overlap = r_sum - d
        mask2 = overlap > 0.0
        if not np.any(mask2):
            return
        i = i[mask2]
        j = j[mask2]
        d = d[mask2]
        overlap = overlap[mask2]
        nvec = diff[i, j] / d[:, None]
        vrel = self.vel[j] - self.vel[i]
        v_n = np.sum(vrel * nvec, axis=1)
        f_mag = self.k_particle * overlap - self.c_damp_particle * v_n
        fvec = f_mag[:, None] * nvec
        np.add.at(f, i, -fvec)
        np.add.at(f, j, fvec)

    def _step(self, dt: float) -> None:
        t = self.current_time
        f = self.mass[:, None] * self.gravity[None, :]

        # Contact with the kinematic wrecking ball.
        c, v_b = self._ball_state(t)
        dvec = self.pos - c
        dist = np.linalg.norm(dvec, axis=1)
        r_eff = self.ball_R + self.radius
        overlap = r_eff - dist
        mask = overlap > 0.0
        if np.any(mask):
            nvec = dvec[mask] / dist[mask][:, None]
            rel_v = self.vel[mask] - v_b
            v_n = np.sum(rel_v * nvec, axis=1)
            f_mag = self.k_contact_ball * overlap[mask] - self.c_damp_ball * v_n
            f[mask] += f_mag[:, None] * nvec

        # Floor contact.
        overlap_g = self.radius - self.pos[:, 2]
        mask_g = overlap_g > 0.0
        if np.any(mask_g):
            v_z = self.vel[mask_g, 2]
            f[mask_g, 2] += (
                self.k_contact_ground * overlap_g[mask_g]
                - self.c_damp_ground * v_z
            )

        # Cohesive bonds: break when tensile strain exceeds a random threshold.
        if len(self.bond_i):
            ri = self.pos[self.bond_i]
            rj = self.pos[self.bond_j]
            dr = rj - ri
            dists = np.linalg.norm(dr, axis=1)
            nvec = dr / dists[:, None]
            strain = dists / self.bond_rest - 1.0
            broken = (strain > self.bond_break_threshold) & self.bond_alive
            if np.any(broken):
                self.bond_alive[broken] = False
                self.bonded_pair[self.bond_i[broken], self.bond_j[broken]] = False
                self.bonded_pair[self.bond_j[broken], self.bond_i[broken]] = False
            alive = self.bond_alive
            vrel = self.vel[self.bond_j] - self.vel[self.bond_i]
            vrel_n = np.sum(vrel * nvec, axis=1)
            f_mag = self.k_bond * (dists - self.bond_rest) - self.c_damp_bond * vrel_n
            f_mag = f_mag * alive
            fvec = f_mag[:, None] * nvec
            np.add.at(f, self.bond_i, fvec)
            np.add.at(f, self.bond_j, -fvec)

        # Repulsive contact between particles that are no longer bonded.
        self._add_particle_contact(f)

        # Global velocity-proportional damping.
        f -= self.damping_global * self.mass[:, None] * self.vel

        # Semi-implicit Euler integration.
        self.vel += dt * (f / self.mass[:, None])
        self.pos += dt * self.vel
        self.current_time += dt

    def _frame(self, time: float) -> StateFrame:
        F = self._compute_F()
        c, _ = self._ball_state(time)
        sd_ball = np.linalg.norm(self.pos - c, axis=1) - (self.ball_R + self.radius)
        sd_ground = self.pos[:, 2] - self.radius
        contact_sd = np.minimum(sd_ball, sd_ground)
        external_force = self.mass[:, None] * self.gravity
        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.radius.copy(),
                "material_label": self.material_label.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
            },
            scalars=dict(self.base_scalars),
        )

    def advance_to(self, time: float) -> StateFrame:
        while self.current_time < time - 1e-12:
            dt = min(self.dt, time - self.current_time)
            self._step(dt)
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": int(self.n),
            "num_bonds": int(len(self.bond_i)),
            "broken_bonds": int(np.sum(~self.bond_alive)),
            "dt": float(self.dt),
        }
