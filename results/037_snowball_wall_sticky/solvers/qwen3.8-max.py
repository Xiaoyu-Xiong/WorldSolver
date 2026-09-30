"""Physically-based bonded-DEM solver for the sticky snowball/wall impact scene.

Model
-----
The snowball is discretised as the granular particle set supplied by the scene
contract.  Snow cohesion is represented by a network of damped linear springs
between neighbouring particles (built once at t=0).  Springs break permanently
when the local strain exceeds a fracture threshold, which reproduces snowball
deformation, breakup and fragment distribution on impact.

Boundaries:
* Sticky wall at ``x = wall_inner_x``: particles entering the adhesion band
  attach locally (velocity frozen, position pinned to the wall face).  Material
  that never reaches the wall stays free to rebound, break up and fall.
* Floor at ``z = floor_z``: unilateral normal spring contact with Coulomb-like
  tangential friction and dissipation.

Integration is symplectic Euler with a fixed substep chosen from the bond
stability limit.  Mass is strictly conserved (no mass sources).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = getattr(scene, "config", None) or {}
        sb = config.get("snowball", {})
        contact = config.get("contact", {})
        gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)

        frame0 = scene.initial_state()
        self._pos = np.array(frame0.position, dtype=float, copy=True)
        self._vel = np.array(frame0.velocity, dtype=float, copy=True)
        self._mass = np.array(frame0.mass, dtype=float, copy=True)
        self._n = len(self._pos)
        self._t = 0.0

        self._gravity = gravity
        self._wall_x = float(contact.get("wall_inner_x", 0.0))
        self._floor_z = float(contact.get("floor_z", 0.0))
        self._cls = float(contact.get("contact_length_scale", 0.065))
        spacing = float(sb.get("particle_spacing", 0.065))

        # ---------------------------------------------------------- bonds
        tree = cKDTree(self._pos)
        cutoff = 1.45 * spacing
        pairs = np.asarray(tree.query_pairs(r=cutoff, output_type="ndarray"), dtype=np.int64)
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        self._bi = pairs[:, 0]
        self._bj = pairs[:, 1]
        delta = self._pos[self._bj] - self._pos[self._bi]
        self._rest = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        self._active = np.ones(len(self._bi), dtype=bool)

        m_eff = float(np.mean(self._mass))
        # Elastic/cohesive parameters (snow-like, tuned for stability).
        self._k_bond = 2.0e5          # spring stiffness [N/m]
        self._c_bond = 6.0            # bond damping [N s/m]
        self._break_strain = 0.28     # tensile/compressive fracture strain
        # Stability-limited substep.
        dt_crit = np.sqrt(m_eff / self._k_bond)
        self._dt = float(min(1.0e-4, 0.25 * dt_crit))
        self._drag = 0.4              # weak global viscous drag [1/s]

        # ------------------------------------------------------- boundaries
        self._stuck = np.zeros(self._n, dtype=bool)
        self._stick_band = 0.75 * spacing
        self._wall_offset = 0.30 * spacing     # pinned standoff from wall face
        self._k_floor = 2.0e6
        self._c_floor = 40.0
        self._friction = 0.55
        self._p_radius = 0.5 * spacing

        self._ext_force = self._mass[:, None] * gravity[None, :]
        self._cum_mass_source = np.zeros(1, dtype=float)
        self._nbonds_broken = 0
        self._nstuck = 0

    # ------------------------------------------------------------- stepping
    def _compute_forces(self) -> np.ndarray:
        forces = np.tile(self._ext_force, (1, 1))
        pos, vel = self._pos, self._vel
        bi, bj, active = self._bi, self._bj, self._active
        if np.any(active):
            ia = bi[active]
            ja = bj[active]
            delta = pos[ja] - pos[ia]
            r = np.linalg.norm(delta, axis=1)
            r = np.maximum(r, 1e-9)
            nhat = delta / r[:, None]
            rest = self._rest[active]
            strain = (r - rest) / rest
            # break bonds beyond fracture strain (permanent)
            broken_local = np.abs(strain) > self._break_strain
            if np.any(broken_local):
                idx = np.flatnonzero(active)[broken_local]
                self._active[idx] = False
                self._nbonds_broken += int(broken_local.sum())
            keep = ~broken_local
            ia, ja, nhat = ia[keep], ja[keep], nhat[keep]
            dlen = (r - rest)[keep]
            dvel = vel[ja] - vel[ia]
            rel_n = np.einsum("ij,ij->i", dvel, nhat)
            fmag = self._k_bond * dlen + self._c_bond * rel_n
            fbond = fmag[:, None] * nhat
            np.add.at(forces, ia, fbond)
            np.add.at(forces, ja, -fbond)

        # floor contact (unilateral)
        pen = (self._floor_z + self._p_radius) - pos[:, 2]
        act = pen > 0.0
        if np.any(act):
            vn = vel[act, 2]
            fn = self._k_floor * pen[act] - self._c_floor * vn
            fn = np.maximum(fn, 0.0)
            forces[act, 2] += fn
            # tangential friction opposing slip
            vt = vel[act, :2]
            speed = np.linalg.norm(vt, axis=1)
            safe = np.maximum(speed, 1e-9)
            ft_mag = np.minimum(self._friction * fn, self._c_floor * speed + 1e-9)
            forces[act, 0] -= ft_mag * vt[:, 0] / safe
            forces[act, 1] -= ft_mag * vt[:, 1] / safe
        return forces

    def _step(self, dt: float) -> None:
        forces = self._compute_forces()
        forces -= self._drag * self._mass[:, None] * self._vel
        acc = forces / self._mass[:, None]
        self._vel += acc * dt
        self._pos += self._vel * dt

        # sticky wall attachment
        newly = (~self._stuck) & (self._pos[:, 0] >= self._wall_x - self._stick_band)
        if np.any(newly):
            self._stuck |= newly
            self._nstuck = int(self._stuck.sum())
        if np.any(self._stuck):
            s = self._stuck
            self._vel[s] = 0.0
            self._pos[s, 0] = min(self._pos[s, 0], self._wall_x - self._wall_offset)

        # hard floor clamp (no penetration)
        below = self._pos[:, 2] < self._floor_z + 1e-4
        if np.any(below):
            self._pos[below, 2] = self._floor_z + 1e-4
            neg = below & (self._vel[:, 2] < 0.0)
            self._vel[neg, 2] = 0.0

        # lateral containment (keep debris in finite domain)
        np.clip(self._pos[:, 1], -3.0, 3.0, out=self._pos[:, 1])
        self._t += dt

    def _advance(self, target: float) -> None:
        dt = self._dt
        while self._t < target - 1e-12:
            h = min(dt, target - self._t)
            self._step(h)

    # --------------------------------------------------------------- output
    def advance_to(self, time: float):
        target = float(time)
        if target > self._t:
            self._advance(target)
        return self._frame()

    def _frame(self) -> StateFrame:
        pos = self._pos
        wall_d = self._wall_x - pos[:, 0]      # negative => penetrated wall
        floor_d = pos[:, 2] - self._floor_z    # negative => penetrated floor
        contact_d = np.minimum(wall_d, floor_d)
        band = 3.0 * self._cls
        sel = contact_d < band
        contact_signed = contact_d[sel]

        frame = StateFrame(
            time=float(self._t),
            position=pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=np.zeros(self._n, dtype=np.int64),
            attributes={
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "external_force": self._ext_force.copy(),
                "contact_signed_distance": np.asarray(contact_signed, dtype=float),
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "wall_signed_distance": wall_d,
                "floor_signed_distance": floor_d,
            },
            scalars={
                "contact_length_scale": float(self._cls),
                "total_mass": float(self._mass.sum()),
                "broken_bonds": float(self._nbonds_broken),
                "attached_particles": float(self._nstuck),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": self._n,
            "n_bonds_total": int(len(self._bi)),
            "n_bonds_broken": self._nbonds_broken,
            "n_attached": self._nstuck,
            "dt": self._dt,
            "time": self._t,
        }
