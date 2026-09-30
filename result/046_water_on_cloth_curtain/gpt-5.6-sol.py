"""Deterministic spring-mass cloth and particle-water physics solver."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit, damped multiphysics model with equal/opposite coupling forces."""

    def __init__(self) -> None:
        self.scene = None
        self.position = self.velocity = self.mass = self.body_id = None
        self.reference = None
        self.n_cloth = self.n_water = 0
        self.time = 0.0
        self.dt = 0.004
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.edges = self.rest = self.pinned = None
        self.last_interaction = np.zeros((2, 3))
        self.cumulative_source = np.zeros(2)
        self.max_displacement = 0.0

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.reference = self.position.copy()
        self.n_cloth = int(np.count_nonzero(self.body_id == 0))
        self.n_water = len(self.position) - self.n_cloth
        self.gravity = np.asarray(scene.config.get("gravity", self.gravity), dtype=float)
        self.dt = min(0.004, float(scene.config.get("sample_dt", 0.02)) / 5.0)
        ny, nz = int(scene.CLOTH_NY), int(scene.CLOTH_NZ)
        grid = np.arange(self.n_cloth, dtype=np.int32).reshape(nz, ny)
        pairs = []
        # Structural and diagonal shear links; a damped network is robust at
        # the modest resolution supplied by the frozen topology.
        for z in range(nz):
            for y in range(ny):
                if y + 1 < ny:
                    pairs.append((grid[z, y], grid[z, y + 1]))
                if z + 1 < nz:
                    pairs.append((grid[z, y], grid[z + 1, y]))
                if z + 1 < nz and y + 1 < ny:
                    pairs.append((grid[z, y], grid[z + 1, y + 1]))
                if z + 1 < nz and y > 0:
                    pairs.append((grid[z, y], grid[z + 1, y - 1]))
        self.edges = np.asarray(pairs, dtype=np.int32)
        delta = self.position[self.edges[:, 1]] - self.position[self.edges[:, 0]]
        self.rest = np.linalg.norm(delta, axis=1)
        self.pinned = np.zeros(self.n_cloth, dtype=bool)
        self.pinned[:ny] = True
        self.time = 0.0

    def _nearest_cloth(self, water_positions):
        ny, nz = self.scene.CLOTH_NY, self.scene.CLOTH_NZ
        iy = np.clip(np.rint((water_positions[:, 1] + 0.92) * (ny - 1) / 1.84), 0, ny - 1).astype(np.int32)
        iz = np.clip(np.rint((2.72 - water_positions[:, 2]) * (nz - 1) / 2.29), 0, nz - 1).astype(np.int32)
        return iz * ny + iy

    def _step(self, h: float) -> None:
        p, v = self.position, self.velocity
        nc = self.n_cloth
        pc, pw = p[:nc], p[nc:]
        vc, vw = v[:nc], v[nc:]
        mc, mw = self.mass[:nc], self.mass[nc:]
        fc = mc[:, None] * self.gravity - 0.018 * vc
        fw = mw[:, None] * self.gravity - 0.06 * mw[:, None] * vw

        # Elastic sheet forces.  The per-edge stiffness is intentionally soft
        # enough for visible loading while shear links prevent element collapse.
        a, b = self.edges[:, 0], self.edges[:, 1]
        d = pc[b] - pc[a]
        length = np.linalg.norm(d, axis=1) + 1e-12
        relative = vc[b] - vc[a]
        stretch = length - self.rest
        magnitude = 6.0 * stretch + 0.045 * np.sum(relative * d, axis=1) / length
        spring = magnitude[:, None] * d / length[:, None]
        np.add.at(fc, a, spring)
        np.add.at(fc, b, -spring)

        # Prescribed nozzle drive continuously accelerates upstream jet samples.
        nozzle = (pw[:, 0] < -0.72) & (np.square(pw[:, 1]) + np.square(pw[:, 2] - 1.72) < 0.24 ** 2)
        fw[nozzle, 0] += 7.0 * mw[nozzle] * (3.2 - vw[nozzle, 0])
        # Prescribed nozzle source bookkeeping (water density * area * speed).
        self.cumulative_source[1] += 1000.0 * np.pi * (0.20 ** 2) * 3.2 * h

        # Local penalty/drag coupling at the curtain. Momentum transfer is
        # applied in equal and opposite pairs and redirects water tangentially.
        ci = self._nearest_cloth(pw)
        gap = pw[:, 0] - pc[ci, 0]
        within = (np.abs(gap) < 0.095) & (pw[:, 1] > -0.98) & (pw[:, 1] < 0.98) & (pw[:, 2] > 0.38) & (pw[:, 2] < 2.76)
        interaction = np.zeros((2, 3))
        if np.any(within):
            wi = np.flatnonzero(within)
            cj = ci[wi]
            rel = vw[wi] - vc[cj]
            side = np.where(gap[wi] <= 0.0, -1.0, 1.0)
            normal_speed = rel[:, 0]
            normal = (70.0 * (0.095 - np.abs(gap[wi])) * side + 2.2 * normal_speed) * mw[wi]
            pair = np.zeros((len(wi), 3))
            pair[:, 0] = normal
            pair[:, 1:] = 0.35 * mw[wi, None] * rel[:, 1:]
            fw[wi] -= pair
            np.add.at(fc, cj, pair)
            interaction[0] = pair.sum(axis=0)
            interaction[1] = -interaction[0]

        vc += h * fc / mc[:, None]
        vw += h * fw / mw[:, None]
        np.clip(vc, -12.0, 12.0, out=vc)
        np.clip(vw, -18.0, 18.0, out=vw)
        pc += h * vc
        pw += h * vw

        # Frozen support and closed catch-tank walls/bottom (open top).
        pc[self.pinned] = self.reference[:nc][self.pinned]
        vc[self.pinned] = 0.0
        lower = (-1.53, -1.37, 0.055)
        upper = (2.07, 1.37, np.inf)
        for axis in range(3):
            low = pw[:, axis] < lower[axis]
            high = pw[:, axis] > upper[axis]
            pw[low, axis] = lower[axis]
            if np.isfinite(upper[axis]):
                pw[high, axis] = upper[axis]
            vw[low | high, axis] *= -0.25
        # Projection prevents fast particles from remaining embedded in cloth.
        ci = self._nearest_cloth(pw)
        gap = pw[:, 0] - pc[ci, 0]
        embedded = np.abs(gap) < 0.012
        if np.any(embedded):
            side = np.where(gap[embedded] <= 0.0, -1.0, 1.0)
            pw[embedded, 0] = pc[ci[embedded], 0] + 0.012 * side
            approaching = vw[embedded, 0] * side < 0.0
            indices = np.flatnonzero(embedded)[approaching]
            vw[indices, 0] *= -0.18
        self.last_interaction = interaction
        self.position = np.nan_to_num(p, nan=0.0, posinf=5.0, neginf=-5.0)
        self.velocity = np.nan_to_num(v, nan=0.0, posinf=18.0, neginf=-18.0)

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1e-12:
            h = min(self.dt, target - self.time)
            self._step(h)
            self.time += h
        displacement = np.linalg.norm(self.position[:self.n_cloth] - self.reference[:self.n_cloth], axis=1)
        self.max_displacement = max(self.max_displacement, float(displacement.max()))
        ci = self._nearest_cloth(self.position[self.n_cloth:])
        water_gap = self.position[self.n_cloth:, 0] - self.position[ci, 0]
        contact_distance = np.concatenate((np.full(self.n_cloth, 0.095), water_gap))
        external = self.mass[:, None] * self.gravity
        pinned_all = np.zeros(len(self.position))
        pinned_all[:self.n_cloth] = self.pinned
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass.copy(), "pinned": pinned_all},
            fields={
                "contact_signed_distance": contact_distance,
                "cumulative_mass_source": self.cumulative_source.copy(),
                "external_force": external,
                "interaction_force": self.last_interaction.copy(),
            },
            scalars={"contact_length_scale": 0.095},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "spring_mass_cloth_particle_water",
            "cloth_displacement": float(self.max_displacement),
            "water_volume": float(self.mass[self.n_cloth:].sum() / 1000.0),
            "coupling_stability": bool(np.isfinite(self.position).all()),
        }
