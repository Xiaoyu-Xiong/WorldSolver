"""Physically-based solver for the ground-impact oobleck benchmark.

Method: explicit DEM-style bonded-particle dynamics with a rate-dependent
(shear-thickening) constitutive law for the oobleck body and a low-yield
viscoplastic law for the comparison body.

* Gravity is integrated explicitly (symplectic Euler).
* The floor is a penalty contact with rate-dependent restitution for the
  shear-thickening material (stiff/elastic at high impact rate, dissipative
  at low rate).
* Inter-particle bonds carry a spring + damper whose stiffness, damping and
  plastic yield threshold depend on the local relative (shear) rate:
  - oobleck (body 2): high rate -> nearly rigid elastic solid (coherent
    rebound at the first fast impact); low rate -> soft, low-yield, highly
    damped material that flows and spreads (after the second, slower impact).
  - viscoplastic comparison (body 1): soft, low yield, heavily damped at all
    rates, so it splats and flows at the first impact.

No keyframes, no scripted motion: every state is produced by time-stepping.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

VEC3 = np.float64


def _stiffness_fraction(v: np.ndarray, vc: float, p: int = 8) -> np.ndarray:
    """Smooth shear-thickening switch in [0, 1]: ~0 slow, ~1 fast."""
    a = np.power(np.maximum(v, 0.0) / vc, p)
    return a / (1.0 + a)


def _zeta_from_restitution(e: np.ndarray) -> np.ndarray:
    """Damping ratio giving restitution e for a penalty contact."""
    e = np.clip(e, 1e-3, 0.999)
    ln = -np.log(e)
    return ln / np.sqrt(math.pi**2 + ln**2)


class BenchmarkSolver:
    """Bonded-particle solver with shear-rate-dependent material laws."""

    SUBSTEP = 2.5e-4
    BOND_CUTOFF_SCALE = 1.55
    VC = 4.4           # rate (m/s) at which the oobleck is half-stiffened
    E_MAX = 0.90       # wall restitution of the fully stiffened oobleck
    K_WALL = 6.0e4     # floor penalty stiffness (N/m)
    K_HI = 9.0e4       # oobleck bond stiffness when fully shear-thiffened
    K_LO = 2.5e3       # oobleck bond stiffness at low rate
    K_VP = 3.0e3       # viscoplastic comparison bond stiffness
    REPULSION_CUTOFF = 0.90   # fraction of particle spacing
    VMAX = 30.0

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.spacing = float(cfg["initial_conditions"]["particle_spacing"])
        self.plane_z = float(cfg["boundary_conditions"]["ground"]["plane_z"])

        self.pos = np.array(frame.position, dtype=np.float64)
        self.vel = np.array(frame.velocity, dtype=np.float64)
        self.mass = np.array(frame.mass, dtype=np.float64)
        self.body_id = np.array(frame.body_id, dtype=np.int32)
        self.n = len(self.pos)
        self.radius = 0.5 * self.spacing
        self.time = 0.0

        self.rest = self.pos.copy()
        self.rest_attr = dict(frame.attributes)
        self.scalars0 = dict(frame.scalars)

        self._build_bonds()
        self._build_defo_precompute()
        self._last_frame_time = None
        self._steps = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _build_bonds(self) -> None:
        idx = np.arange(self.n)
        left = idx[self.body_id == 1]
        right = idx[self.body_id == 2]
        bi_list, bj_list = [], []
        cutoff = self.BOND_CUTOFF_SCALE * self.spacing
        for group in (left, right):
            if len(group) < 2:
                continue
            tree = cKDTree(self.pos[group])
            pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
            if len(pairs) == 0:
                continue
            bi_list.append(group[pairs[:, 0]])
            bj_list.append(group[pairs[:, 1]])
        self.bond_i = np.concatenate(bi_list) if bi_list else np.zeros(0, dtype=int)
        self.bond_j = np.concatenate(bj_list) if bj_list else np.zeros(0, dtype=int)
        self.bond_body = self.body_id[self.bond_i]
        self.bond_l0 = np.linalg.norm(
            self.pos[self.bond_j] - self.pos[self.bond_i], axis=1
        )
        # repulsion lookups exclude the (initially spaced) bonded pairs by the
        # cutoff choice; nothing else is needed.

    def _build_defo_precompute(self) -> None:
        """Precompute the rest-shape tensor per particle for F estimation."""
        n = self.n
        i = np.concatenate((self.bond_i, self.bond_j))
        j = np.concatenate((self.bond_j, self.bond_i))
        rest = self.rest
        r = rest[j] - rest[i]
        b = np.einsum("nk,nl->nkl", r, r)
        bsum = np.zeros((n, 3, 3))
        np.add.at(bsum, i, b)
        trace = np.trace(bsum, axis1=1, axis2=2)
        reg = 1e-9 * np.maximum(trace, 1e-12) + 1e-18
        bsum += reg[:, None, None] * np.eye(3)[None, :, :]
        self._defo_i = i
        self._defo_j = j
        self._defo_r = r
        self._defo_binv = np.linalg.inv(bsum)

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        pos, vel, mass = self.pos, self.vel, self.mass
        n = self.n

        # gravity
        vel += self.gravity[None, :] * h

        force = np.zeros((n, 3))
        # (gravity handled through velocity; forces below are contact forces)

        # ---------------- floor penalty contact ----------------
        gap = pos[:, 2] - self.plane_z - self.radius
        pen = gap < 0.0
        if np.any(pen):
            idx = np.flatnonzero(pen)
            m = mass[idx]
            kw = self.K_WALL
            vz = vel[idx, 2]
            approach = np.abs(vz)
            body = self.body_id[idx]
            # restitution of the shear-thickening material grows with rate
            e = self.E_MAX * _stiffness_fraction(approach, self.VC)
            e = np.where(body == 2, e, 0.02)
            zeta = np.clip(_zeta_from_restitution(e), 0.02, 1.5)
            c = 2.0 * zeta * np.sqrt(kw * m)
            fn = kw * (-gap[idx])
            fz = fn - np.where(vz < 0.0, c * vz, 0.6 * c * vz)
            force[idx, 2] += fz
            # tangential friction while in contact
            damp_t = np.exp(-8.0 * h)
            vel[idx, 0] *= damp_t
            vel[idx, 1] *= damp_t

        # ---------------- bonded interactions ----------------
        if len(self.bond_i):
            bi, bj = self.bond_i, self.bond_j
            d = pos[bj] - pos[bi]
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 1e-9)
            u = d / dist[:, None]
            rel = vel[bj] - vel[bi]
            vn = np.einsum("nk,nk->n", rel, u)   # signed normal rate
            rate = np.abs(vn)

            body = self.bond_body
            s = _stiffness_fraction(rate, self.VC)
            # stiffness: oobleck stiffens with rate, viscoplastic stays soft
            k = np.where(body == 2, self.K_LO + (self.K_HI - self.K_LO) * s, self.K_VP)
            # damping ratio: oobleck nearly elastic when stiffened
            zeta = np.where(body == 2, 0.05 + 0.45 * (1.0 - s), 0.55)
            mi = mass[bi]
            mj = mass[bj]
            c = 2.0 * zeta * np.sqrt(k * 0.5 * (mi + mj))

            strain = (dist - self.bond_l0) / self.bond_l0
            # plastic yield threshold: large when shear-stiffened
            yld = np.where(body == 2, 0.012 + 0.4 * s, 0.006)
            over = np.abs(strain) - yld
            plastic = over > 0.0
            if np.any(plastic):
                self.bond_l0[plastic] *= 1.0 + np.sign(strain[plastic]) * over[plastic]

            fmag = k * (dist - self.bond_l0) + c * vn
            fd = fmag[:, None] * u
            np.add.at(force, bi, fd)
            np.add.at(force, bj, -fd)

        # ---------------- non-bond repulsion (anti-interpenetration) -------
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=self.REPULSION_CUTOFF * self.spacing,
                                 output_type="ndarray")
        if len(pairs):
            pi, pj = pairs[:, 0], pairs[:, 1]
            d = pos[pj] - pos[pi]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            u = d / dist[:, None]
            overlap = self.spacing - dist
            act = overlap > 0.0
            if np.any(act):
                pi, pj, u, dist, ov = pi[act], pj[act], u[act], dist[act], overlap[act]
                rel = vel[pj] - vel[pi]
                vn = np.einsum("nk,nk->n", rel, u)
                k = 2.0e4
                mred = 0.5 * (mass[pi] + mass[pj])
                c = 2.0 * 0.3 * np.sqrt(k * mred)
                fmag = k * ov + c * vn
                fd = fmag[:, None] * u
                np.add.at(force, pi, -fd)
                np.add.at(force, pj, fd)

        # ---------------- integrate ----------------
        vel += (force / mass[:, None]) * h
        np.clip(vel, -self.VMAX, self.VMAX, out=vel)
        pos += vel * h

        # hard safety: never tunnel deeply through the floor
        zmin = self.plane_z + 0.2 * self.radius
        deep = pos[:, 2] < zmin
        if np.any(deep):
            pos[deep, 2] = zmin
            vel[deep, 2] = np.maximum(vel[deep, 2], 0.0)

        self.time += h
        self._steps += 1

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        i, j = self._defo_i, self._defo_j
        d = self.pos[j] - self.pos[i]
        a = np.einsum("nk,nl->nkl", d, self._defo_r)
        asum = np.zeros((self.n, 3, 3))
        np.add.at(asum, i, a)
        f = asum @ self._defo_binv
        return f

    def _snapshot(self) -> "StateFrame":
        from worldsolver.contracts import StateFrame

        n = self.n
        fdef = self._deformation_gradient()
        gap = self.pos[:, 2] - self.plane_z - self.radius
        near = np.flatnonzero(gap < 3.0 * self.spacing)
        if near.size == 0:
            near = np.array([int(np.argmin(gap))], dtype=int)
        contact_sd = gap[near]

        ext_force = self.mass[:, None] * self.gravity[None, :]

        attributes = {
            "particle_radius": np.full(n, self.radius, dtype=np.float64),
            "rest_position": self.rest.copy(),
            "material_label": self.body_id.astype(np.float64),
            "deformation_gradient": fdef,
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "contact_signed_distance": contact_sd,
            "cumulative_mass_source": np.zeros(2, dtype=np.float64),
            "external_force": ext_force,
        }
        scalars = {
            "contact_length_scale": float(self.spacing),
            "total_mass": float(self.mass.sum()),
            "kinetic_energy": float(
                0.5 * np.sum(self.mass * np.einsum("nk,nk->n", self.vel, self.vel))
            ),
        }
        frame = StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-9:
            # times are requested monotonically; guard anyway
            target = self.time
        eps = 1e-9
        while self.time < target - eps:
            h = min(self.SUBSTEP, target - self.time)
            self._step(h)
        self.time = target
        return self._snapshot()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "rate_dependent_bonded_particle_dem",
            "substeps": self._steps,
            "particles": int(self.n),
            "bonds": int(len(self.bond_i)),
            "time": float(self.time),
        }
