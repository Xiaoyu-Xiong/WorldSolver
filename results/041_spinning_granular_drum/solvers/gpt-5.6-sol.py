"""Deterministic soft-sphere DEM for a rotating granular drum."""
from __future__ import annotations

import numpy as np
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        src = scene.initial_state()
        self.pos = np.asarray(src.position, dtype=np.float64).copy()
        self.vel = np.asarray(src.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(src.mass, dtype=np.float64).copy()
        self.attrs = {k: np.asarray(v).copy() for k, v in src.attributes.items()}
        self.body_id = np.asarray(src.body_id, dtype=np.int32).copy()
        cfg = scene.config; drum = cfg["initial_conditions"]["drum"]; grains = cfg["initial_conditions"]["grains"]
        self.center = np.asarray(drum["center"], dtype=np.float64); self.radius = float(drum["radius"]); self.depth = float(drum["depth"])
        self.omega = float(drum["angular_velocity_rad_s"]); self.rp = float(grains["radius"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.dt = min(0.004, float(cfg.get("sample_dt", 0.02)) / 4.0); self.time = 0.0
        self.k_contact = 900.0; self.gamma_contact = 5.0; self.mu_wall = 0.65; self.wall_skin = 0.012
        self.last_external = self.mass[:, None] * self.gravity[None, :]
        self.escaped = 0; self._diag = {}

    def _pair_forces(self, force):
        if cKDTree is None:
            return
        pairs = np.asarray(list(cKDTree(self.pos).query_pairs(2.0 * self.rp * 1.08)), dtype=np.int64)
        if pairs.size == 0: return
        pairs = pairs.reshape((-1, 2)); i, j = pairs[:, 0], pairs[:, 1]
        d = self.pos[j] - self.pos[i]; dist = np.linalg.norm(d, axis=1); valid = dist > 1e-10
        i, j, d, dist = i[valid], j[valid], d[valid], dist[valid]; overlap = 2.0 * self.rp - dist; active = overlap > 0.0
        if not np.any(active): return
        i, j, d, dist, overlap = i[active], j[active], d[active], dist[active], overlap[active]
        normal = d / dist[:, None]; vn = np.sum((self.vel[j] - self.vel[i]) * normal, axis=1)
        pf = (self.k_contact * overlap + self.gamma_contact * np.maximum(0.0, -vn))[:, None] * normal
        np.add.at(force, i, -pf); np.add.at(force, j, pf)

    def _boundary_forces(self, force):
        rel = self.pos - self.center; xz = rel[:, (0, 2)]; radial = np.linalg.norm(xz, axis=1); safe = np.maximum(radial, 1e-12)
        nout = np.zeros_like(self.pos); nout[:, 0] = xz[:, 0] / safe; nout[:, 2] = xz[:, 1] / safe
        limit = self.radius - self.rp; penetration = radial - limit; outside = penetration > 0.0
        if np.any(outside): force[outside] -= (self.k_contact * penetration[outside])[:, None] * nout[outside]
        near = radial > (limit - self.wall_skin)
        if np.any(near):
            wall_v = np.zeros_like(self.pos); wall_v[:, 0] = self.omega * rel[:, 2]; wall_v[:, 2] = -self.omega * rel[:, 0]
            tang = (wall_v - self.vel) - np.sum((wall_v - self.vel) * nout, axis=1)[:, None] * nout
            load = self.mass * 9.81 * 0.8 + self.k_contact * np.maximum(penetration, 0.0); friction = np.minimum(self.mu_wall * load, 18.0 * self.mass)
            tn = np.linalg.norm(tang, axis=1); mask = near & (tn > 1e-10)
            force[mask] += friction[mask, None] * tang[mask] / tn[mask, None]
        half = 0.5 * self.depth - self.rp; q = self.pos[:, 1]; lo, hi = self.center[1] - half, self.center[1] + half
        below, above = q < lo, q > hi
        if np.any(below): force[below, 1] += self.k_contact * (lo - q[below])
        if np.any(above): force[above, 1] -= self.k_contact * (q[above] - hi)

    def _step(self, dt):
        force = self.mass[:, None] * self.gravity[None, :]; self._pair_forces(force); self._boundary_forces(force)
        self.last_external = self.mass[:, None] * self.gravity[None, :]; self.vel += dt * force / self.mass[:, None]; self.vel *= 0.9992; self.pos += dt * self.vel
        rel = self.pos - self.center; rr = np.linalg.norm(rel[:, (0, 2)], axis=1); bad = rr > (self.radius - self.rp)
        if np.any(bad):
            scale = (self.radius - self.rp) / np.maximum(rr[bad], 1e-12); self.pos[bad, 0] = self.center[0] + rel[bad, 0] * scale; self.pos[bad, 2] = self.center[2] + rel[bad, 2] * scale
            rn = np.zeros((bad.sum(), 3)); rn[:, 0] = rel[bad, 0] / np.maximum(rr[bad], 1e-12); rn[:, 2] = rel[bad, 2] / np.maximum(rr[bad], 1e-12)
            self.vel[bad] -= np.maximum(np.sum(self.vel[bad] * rn, axis=1), 0.0)[:, None] * rn
        half = 0.5 * self.depth - self.rp; self.pos[:, 1] = np.clip(self.pos[:, 1], self.center[1] - half, self.center[1] + half); self.time += dt

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-9: raise ValueError("advance_to requires finite nondecreasing time")
        while self.time < target - 1e-10: self._step(min(self.dt, target - self.time))
        rel = self.pos - self.center; radial = np.linalg.norm(rel[:, (0, 2)], axis=1)
        half = 0.5 * self.depth - self.rp
        contacts = np.concatenate((self.radius - self.rp - radial, self.pos[:, 1] - (self.center[1] - half), (self.center[1] + half) - self.pos[:, 1]))
        bins = np.linspace(-self.radius, self.radius, 17); tops = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (rel[:, 0] >= a) & (rel[:, 0] < b)
            if np.any(m): tops.append(float(np.max(rel[m, 2])))
        slope = float(np.polyfit(np.arange(len(tops)), tops, 1)[0] / (self.radius / 8.0)) if len(tops) > 2 else 0.0
        flow = (rel[:, 2] > 0.05) & (np.abs(rel[:, 0]) < 0.65); speed = float(np.mean(np.linalg.norm(self.vel[flow], axis=1))) if np.any(flow) else 0.0
        self._diag = {"drum_angle": self.omega * target, "lifted_grain_height": float(np.max(rel[:, 2])), "free_surface_slope": slope, "avalanche_layer_speed": speed, "escaped_particle_count": int(self.escaped)}
        frame = StateFrame(time=target, position=self.pos.copy(), velocity=self.vel.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(), attributes={**self.attrs, "mass_quantity": self.mass.copy()}, fields={"contact_signed_distance": contacts.astype(np.float64), "cumulative_mass_source": np.zeros(1, dtype=np.float64), "external_force": self.last_external.copy()}, scalars={"drum_radius": self.radius, "drum_depth": self.depth, "drum_angular_velocity_rad_s": self.omega, "initial_particle_count": float(len(self.pos)), "drum_angle": self.omega * target, "contact_length_scale": 2.0 * self.rp})
        frame.validate(); return frame

    def diagnostics(self) -> dict[str, object]: return dict(self._diag)
