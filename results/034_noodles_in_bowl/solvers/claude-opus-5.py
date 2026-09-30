"""Noodles-in-bowl solver.

Physical model
--------------
625 noodles, each discretised as a 2-node inextensible rod (a capsule of
radius r = 6 mm, rest length 0.4 m).  Dynamics are advanced with a substepped
position-based (XPBD-style) scheme:

    1. explicit gravity prediction of node positions,
    2. projection of the rod inextensibility constraints,
    3. projection of unilateral contact constraints against the frozen
       axisymmetric bowl (inner profile + wall thickness) and the table plane,
    4. projection of capsule-capsule (segment-segment) non-penetration
       constraints between noodles at the exact closest-point pair,
    5. velocity recovery v = (x - x_prev)/dt with frictional damping on
       contacting nodes.

The projection scheme is stable for stiff contacts, so the rods fall, pile,
intertwine and settle into a stable equilibrium with finite pile thickness.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# bowl inner profile (radius, height) from scene_geometry.yaml
_PROFILE_R = np.array([0.12, 0.38, 0.66, 0.72])
_PROFILE_Z = np.array([0.00, 0.08, 0.34, 0.48])
_WALL_THICKNESS = 0.025


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        frame0 = scene.initial_state()
        meta = scene.metadata()

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.n = self.pos.shape[0]

        mass = frame0.mass
        if mass is None:
            mass = np.full(self.n, 0.00125)
        self.mass = np.asarray(mass, dtype=float).reshape(self.n)
        self.mass = np.where(self.mass > 0.0, self.mass, 1e-6)
        self.inv_mass = 1.0 / self.mass

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.radius = float(cfg.get("radius", meta.get("radius", 0.006)))

        # rod topology: edge list of the curve network
        edges = meta.get("edges")
        if edges is None:
            npn = int(cfg.get("nodes_per_noodle", 2))
            count = int(cfg.get("noodle_count", self.n // max(npn, 1)))
            edges = [[i * npn + k, i * npn + k + 1]
                     for i in range(count) for k in range(npn - 1)]
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.e0 = self.edges[:, 0]
        self.e1 = self.edges[:, 1]
        self.rest_len = np.maximum(
            np.linalg.norm(self.pos[self.e1] - self.pos[self.e0], axis=1), 1e-6)
        self.n_seg = self.edges.shape[0]

        # time stepping
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.n_sub = max(1, int(np.ceil(self.sample_dt / 1.5e-3)))
        self.dt = self.sample_dt / self.n_sub
        self.time = 0.0
        self.steps = 0

        # solver parameters
        self.contact_iters = 2
        self.relax = 0.5
        self.max_speed = 8.0
        self.linear_damping = 0.9995
        self.contact_damping = 0.9
        self.broad_every = 8
        self.pairs = np.zeros((0, 2), dtype=np.int64)
        self._bp_i, self._bp_j = np.triu_indices(self.n_seg, k=1)

        self.contact_length_scale = float(
            frame0.scalars.get("contact_length_scale", self.radius))
        self._min_gap = float("inf")

        # tiny seeded symmetry-breaking perturbation applied on the first step
        self._perturb = 1.0e-3 * self.rng.standard_normal((self.n, 3))
        self._perturbed = False

    # -------------------------------------------------------------- bowl SDF
    def _bowl_contact(self, pos):
        """Distance to the bowl inner surface and the outward free-side normal.

        Returns (signed, normal, inside): ``signed`` is the unsigned distance
        to the profile surface with the sign flipped when the node has left the
        bowl cavity, ``normal`` points from the surface towards the node's own
        free region, and ``inside`` marks nodes inside the cavity.
        """
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        rho = np.hypot(x, y)
        safe_rho = np.where(rho > 1e-9, rho, 1.0)
        er_x = np.where(rho > 1e-9, x / safe_rho, 1.0)
        er_y = np.where(rho > 1e-9, y / safe_rho, 0.0)

        # closest point on the inner profile polyline in the (rho, z) half plane
        ar = np.concatenate(([0.0], _PROFILE_R[:-1]))
        az = np.concatenate(([0.0], _PROFILE_Z[:-1]))
        br, bz = _PROFILE_R, _PROFILE_Z
        pr = rho[:, None]
        pz = z[:, None]
        dr = (br - ar)[None, :]
        dz = (bz - az)[None, :]
        seg2 = np.maximum(dr * dr + dz * dz, 1e-12)
        t = np.clip(((pr - ar[None, :]) * dr + (pz - az[None, :]) * dz) / seg2, 0.0, 1.0)
        cr = ar[None, :] + t * dr
        cz = az[None, :] + t * dz
        d2 = (pr - cr) ** 2 + (pz - cz) ** 2
        k = np.argmin(d2, axis=1)
        idx = np.arange(pos.shape[0])
        cr = cr[idx, k]
        cz = cz[idx, k]
        dist = np.sqrt(np.maximum(d2[idx, k], 0.0))

        # cavity test: above the bottom, inside the flaring wall, below the rim
        r_wall = np.interp(z, _PROFILE_Z, _PROFILE_R)
        below_rim = z < _PROFILE_Z[-1]
        inside = (z > 0.0) & (rho < r_wall) & below_rim

        safe_d = np.where(dist > 1e-9, dist, 1.0)
        ur = (rho - cr) / safe_d
        uz = (z - cz) / safe_d
        sgn = np.where(inside, 1.0, -1.0)
        nr = sgn * ur
        nz = sgn * uz
        signed = sgn * dist

        deg = dist <= 1e-9
        if np.any(deg):
            nr = np.where(deg, 0.0, nr)
            nz = np.where(deg, 1.0, nz)

        normal = np.stack([nr * er_x, nr * er_y, nz], axis=1)
        normal = normal / np.maximum(
            np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)

        # nodes above the rim plane are in free air: no wall constraint
        signed = np.where(below_rim, signed, 1.0e3)
        return signed, normal, inside

    def _wall_gap(self, pts):
        """Unilateral wall gap: clearance to the bowl shell, plus free normal."""
        signed, normal, inside = self._bowl_contact(pts)
        # required clearance: rod radius in the cavity, radius + wall outside
        need = np.where(inside, self.radius, self.radius + _WALL_THICKNESS)
        return np.abs(signed) - need, normal

    def _project_bowl(self, pos):
        active = np.zeros(self.n, dtype=bool)

        # rod endpoints
        gap, normal = self._wall_gap(pos)
        hit = gap < 0.0
        if np.any(hit):
            pos += np.where(hit, -gap, 0.0)[:, None] * normal
            active |= hit

        # interior samples of each capsule, so rods cannot straddle the wall
        p = pos[self.e0]
        d = pos[self.e1] - p
        for s in (0.25, 0.5, 0.75):
            pts = p + s * d
            gap, normal = self._wall_gap(pts)
            hit = gap < 0.0
            if not np.any(hit):
                continue
            w0 = 1.0 - s
            w1 = s
            im0 = self.inv_mass[self.e0]
            im1 = self.inv_mass[self.e1]
            denom = np.maximum(w0 * w0 * im0 + w1 * w1 * im1, 1e-12)
            lam = np.where(hit, -gap, 0.0) / denom
            corr = np.zeros_like(pos)
            np.add.at(corr, self.e0, (lam * w0 * im0)[:, None] * normal)
            np.add.at(corr, self.e1, (lam * w1 * im1)[:, None] * normal)
            pos += corr
            active[self.e0[hit]] = True
            active[self.e1[hit]] = True

        # table plane at z = 0 supporting the bowl exterior
        under = pos[:, 2] < self.radius
        if np.any(under):
            pos[under, 2] = self.radius
        return active | under

    # --------------------------------------------------------- collision pairs
    def _broadphase(self, pos):
        p = pos[self.e0]
        q = pos[self.e1]
        margin = 2.0 * self.radius + 6.0e-3
        lo = np.minimum(p, q) - margin
        hi = np.maximum(p, q) + margin
        i, j = self._bp_i, self._bp_j
        ok = np.ones(i.shape[0], dtype=bool)
        for ax in range(3):
            ok &= (lo[i, ax] <= hi[j, ax]) & (lo[j, ax] <= hi[i, ax])
            if not ok.any():
                break
        self.pairs = np.stack([i[ok], j[ok]], axis=1)

    @staticmethod
    def _closest_params(p1, d1, p2, d2):
        """Closest-point parameters (s, t) for two segment families."""
        r0 = p1 - p2
        a = np.maximum(np.einsum("ij,ij->i", d1, d1), 1e-12)
        e = np.maximum(np.einsum("ij,ij->i", d2, d2), 1e-12)
        b = np.einsum("ij,ij->i", d1, d2)
        c = np.einsum("ij,ij->i", d1, r0)
        f = np.einsum("ij,ij->i", d2, r0)
        denom = a * e - b * b
        ok = denom > 1e-12
        s = np.where(ok, (b * f - c * e) / np.where(ok, denom, 1.0), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((b * s + f) / e, 0.0, 1.0)
        s = np.clip((t * b - c) / a, 0.0, 1.0)
        return s, t

    def _self_gaps(self, pos):
        """Gap, normal and barycentric weights for candidate noodle pairs."""
        if self.pairs.shape[0] == 0:
            z = np.zeros(0)
            return self.pairs, z, np.zeros((0, 3)), z, z
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        p1 = pos[self.e0[i]]
        p2 = pos[self.e0[j]]
        d1 = pos[self.e1[i]] - p1
        d2 = pos[self.e1[j]] - p2
        s, t = self._closest_params(p1, d1, p2, d2)
        delta = (p1 + s[:, None] * d1) - (p2 + t[:, None] * d2)
        dist = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(dist, 1e-9)[:, None]
        deg = dist <= 1e-9
        if np.any(deg):
            normal[deg] = np.array([0.0, 0.0, 1.0])
        return self.pairs, dist - 2.0 * self.radius, normal, s, t

    def _project_self_contacts(self, pos, contact_flag):
        pairs, gap, normal, s, t = self._self_gaps(pos)
        if pairs.shape[0] == 0:
            return
        act = gap < 0.0
        if not np.any(act):
            return
        pen = -gap[act]
        normal = normal[act]
        s = s[act]
        t = t[act]
        i = pairs[act, 0]
        j = pairs[act, 1]
        a0, a1 = self.e0[i], self.e1[i]
        b0, b1 = self.e0[j], self.e1[j]
        w_a0, w_a1 = 1.0 - s, s
        w_b0, w_b1 = 1.0 - t, t
        k1 = w_a0 ** 2 * self.inv_mass[a0] + w_a1 ** 2 * self.inv_mass[a1]
        k2 = w_b0 ** 2 * self.inv_mass[b0] + w_b1 ** 2 * self.inv_mass[b1]
        lam = self.relax * pen / np.maximum(k1 + k2, 1e-12)

        corr = np.zeros_like(pos)
        np.add.at(corr, a0, (lam * w_a0 * self.inv_mass[a0])[:, None] * normal)
        np.add.at(corr, a1, (lam * w_a1 * self.inv_mass[a1])[:, None] * normal)
        np.add.at(corr, b0, -(lam * w_b0 * self.inv_mass[b0])[:, None] * normal)
        np.add.at(corr, b1, -(lam * w_b1 * self.inv_mass[b1])[:, None] * normal)

        # cap the Jacobi correction magnitude for robustness in dense piles
        mag = np.linalg.norm(corr, axis=1)
        cap = 4.0 * self.radius
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        pos += corr * scale[:, None]

        for idx in (a0, a1, b0, b1):
            contact_flag[idx] = True

    def _project_rods(self, pos):
        p = pos[self.e0]
        d = pos[self.e1] - p
        length = np.linalg.norm(d, axis=1)
        n = d / np.maximum(length, 1e-9)[:, None]
        w0 = self.inv_mass[self.e0]
        w1 = self.inv_mass[self.e1]
        lam = (length - self.rest_len) / np.maximum(w0 + w1, 1e-12)
        corr = np.zeros_like(pos)
        np.add.at(corr, self.e0, (lam * w0)[:, None] * n)
        np.add.at(corr, self.e1, -(lam * w1)[:, None] * n)
        pos += corr

    # ------------------------------------------------------------- integration
    def _substep(self, dt):
        pos_prev = self.pos
        vel = self.vel
        if not self._perturbed:
            vel = vel + self._perturb
            self._perturbed = True
        vel = vel + dt * self.gravity[None, :]
        vel = vel * self.linear_damping
        speed = np.linalg.norm(vel, axis=1)
        over = speed > self.max_speed
        if np.any(over):
            vel[over] *= (self.max_speed / speed[over])[:, None]
        pos = pos_prev + dt * vel

        if self.steps % self.broad_every == 0:
            self._broadphase(pos)

        contact = np.zeros(self.n, dtype=bool)
        for _ in range(self.contact_iters):
            self._project_rods(pos)
            contact |= self._project_bowl(pos)
            self._project_self_contacts(pos, contact)
        self._project_rods(pos)
        contact |= self._project_bowl(pos)

        new_vel = (pos - pos_prev) / dt
        if np.any(contact):
            new_vel[contact] *= self.contact_damping
        speed = np.linalg.norm(new_vel, axis=1)
        over = speed > self.max_speed
        if np.any(over):
            new_vel[over] *= (self.max_speed / speed[over])[:, None]

        self.pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.vel = np.nan_to_num(new_vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.steps += 1
        self.time += dt

    # ----------------------------------------------------------------- output
    def _contact_signed_distance(self):
        """Per-node signed gap, clipped to near contact; negative = penetration."""
        gap, _ = self._wall_gap(self.pos)
        gap = np.minimum(gap, self.pos[:, 2] - self.radius)

        self._broadphase(self.pos)
        pairs, pgap, _, _, _ = self._self_gaps(self.pos)
        if pairs.shape[0] > 0:
            near = pgap < 0.01
            if np.any(near):
                i = pairs[near, 0]
                j = pairs[near, 1]
                g = pgap[near]
                for idx in (self.e0[i], self.e1[i], self.e0[j], self.e1[j]):
                    np.minimum.at(gap, idx, g)
        gap = np.minimum(gap, 0.01)
        self._min_gap = min(self._min_gap, float(gap.min()))
        return np.nan_to_num(gap, nan=0.0, posinf=0.01, neginf=0.0)

    def _frame(self):
        ext = self.mass[:, None] * self.gravity[None, :]
        fields = {
            "contact_signed_distance": self._contact_signed_distance(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "external_force": np.ascontiguousarray(ext, dtype=float),
        }
        frame = StateFrame(
            time=float(self.time),
            position=np.array(self.pos, dtype=float),
            velocity=np.array(self.vel, dtype=float),
            mass=np.array(self.mass, dtype=float),
            attributes={"mass_quantity": np.array(self.mass, dtype=float)},
            fields=fields,
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-12:
                break
            self._substep(dt)
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        z = np.clip(self.pos[:, 2], 0.0, _PROFILE_Z[-1])
        rho = np.hypot(self.pos[:, 0], self.pos[:, 1])
        inside = rho < np.interp(z, _PROFILE_Z, _PROFILE_R)
        return {
            "method": "substepped position-based rod dynamics (XPBD-style)",
            "substep_dt": float(self.dt),
            "substeps": int(self.steps),
            "nodes": int(self.n),
            "rods": int(self.n_seg),
            "contact_pairs_last": int(self.pairs.shape[0]),
            "min_signed_distance": float(self._min_gap),
            "fill_height": float(np.max(self.pos[:, 2], initial=0.0)),
            "nodes_inside_bowl": int(np.count_nonzero(inside)),
            "max_speed": float(np.linalg.norm(self.vel, axis=1).max()),
        }
