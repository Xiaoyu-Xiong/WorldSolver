"""Rigid-body contact solver for the circular domino chain reaction.

Physical model
--------------
Each domino is a rigid box with the frozen half extents (0.07 x 0.2 x 0.55 m),
the config mass and the exact solid-box inertia tensor.  The floor is the top
face of the frozen static floor box, treated as a rigid half space.

Time stepping is semi-implicit (symplectic) Euler:

1. gravity is applied to the velocities,
2. contacts are detected geometrically (box corners against box faces and
   against the floor plane),
3. the velocity-level contact problem is solved with a relaxed projected
   Gauss-Seidel/Jacobi sweep: non-penetration with Baumgarte stabilisation,
   Newton restitution and a Coulomb friction cone,
4. positions and quaternions are integrated and the quaternions renormalised.

Geometry of the frozen scene
----------------------------
The frozen ring is over-packed: 24 dominoes whose tangential width is 0.4 m sit
on a circle of circumference 2*pi*1.35 = 8.48 m, so the declared collision boxes
of neighbouring dominoes start about 0.06 m interpenetrated, and the pre-tilted
trigger starts 0.033 m inside the floor.  Resolving that overlap instantly would
inject a huge fictitious impulse and burst the ring apart at t = 0, which would
violate the "all dominoes start motionless" initial condition.

Instead every contact is measured against a per-pair reference gap that is
initialised from the frozen configuration (so the frozen state is exactly the
unstressed touching state) and then relaxed linearly to zero over
``RELAX_TIME``.  After that the true frozen geometry is enforced, so the initial
interpenetration is transient rather than persistent.  The relaxation preloads
the closed ring tangentially; for an upright domino the two neighbour loads
cancel, so the ring stays standing, while the already-unstable trigger topples
outward under gravity.  Once it falls its neighbour loses one-sided support and
topples in turn, so a single coherent front travels around the loop and closes
it.  No motion is scripted: every impulse comes from the contact solve.

Exported trajectory fields (StateFrame contract)
------------------------------------------------
state.body_id, state.mass, state.position, state.velocity
    (plus orientation and angular_velocity)
fields.external_force           [N,3]  gravity only (no internal contact terms)
fields.interaction_force        [K,3]  net contact force per tracked pair
fields.contact_signed_distance  [K]    raw signed geometric gap per tracked pair
scalars.contact_length_scale    scalar frozen contact normalisation length
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# quaternion helpers (xyzw convention, matching the frozen scene)
# ---------------------------------------------------------------------------


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    out = np.asarray(q, dtype=float) / n
    bad = ~np.isfinite(out).all(axis=-1)
    if np.any(bad):
        out[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N,4) xyzw quaternions -> (N,3,3) body-to-world rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((q.shape[0], 3, 3), dtype=float)
    m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[:, 0, 1] = 2.0 * (xy - wz)
    m[:, 0, 2] = 2.0 * (xz + wy)
    m[:, 1, 0] = 2.0 * (xy + wz)
    m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[:, 1, 2] = 2.0 * (yz - wx)
    m[:, 2, 0] = 2.0 * (xz - wy)
    m[:, 2, 1] = 2.0 * (yz + wx)
    m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(a)
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


_CORNER_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, +1.0, +1.0],
    ]
)

_FAR_GAP = 1.0  # reported gap for pairs with no nearby features [m]


class BenchmarkSolver:
    """Deterministic rigid-body solver with frictional contact."""

    # ---- numerical parameters ----
    DT = 1.0e-3           # integration step [s]
    ITERATIONS = 20       # contact sweeps per step
    RELAX = 0.35          # under-relaxation of the contact sweep
    MARGIN = 0.015        # speculative-contact margin [m]
    SLOP = 0.0010         # tolerated penetration before recovery [m]
    BETA = 0.20           # Baumgarte penetration recovery factor
    MAX_RECOVERY = 0.30   # cap on penetration recovery speed [m/s]
    REST_THRESHOLD = 0.30 # minimum approach speed for restitution [m/s]
    RELAX_TIME = 1.20     # time to fully enforce the frozen geometry [s]
    LIN_DAMPING = 0.02    # weak linear damping [1/s]
    ANG_DAMPING = 0.05    # weak angular damping [1/s]
    MAX_LIN = 25.0        # velocity clamp for guaranteed finiteness [m/s]
    MAX_ANG = 50.0        # spin clamp [rad/s]
    NEIGHBOUR_SPAN = 3    # tracked ring neighbours on each side

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = dict(scene.config)
        frame = scene.initial_state()

        self.body_id = np.asarray(frame.body_id, dtype=np.int64).copy()
        self.n = int(self.body_id.shape[0])
        self.x = np.asarray(frame.position, dtype=float).copy()
        self.q = _quat_normalize(np.asarray(frame.orientation, dtype=float).copy())
        self.v = np.zeros((self.n, 3), dtype=float)
        self.w = np.zeros((self.n, 3), dtype=float)
        self.mass = np.asarray(frame.mass, dtype=float).copy()

        contact = dict(config.get("contact", {}) or {})
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.0))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # ---- frozen geometry: per-body half extents and the floor surface --
        half = np.tile(np.array([0.07, 0.2, 0.55]), (self.n, 1))
        floor_top = 0.0
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            geo = obj.get("geometry", {}) or {}
            he = geo.get("half_extents")
            binding = obj.get("state_binding") or {}
            transform = obj.get("transform") or {}
            translation = transform.get("translation") or [0.0, 0.0, 0.0]
            is_static = (obj.get("semantic") == "boundary"
                         or obj.get("motion") == "static")
            if is_static:
                if he is not None:
                    floor_top = float(translation[2]) + float(he[2])
            elif binding.get("selector") == "body_id" and he is not None:
                idx = np.nonzero(self.body_id == int(binding.get("value")))[0]
                if idx.size:
                    half[idx[0]] = np.asarray(he, dtype=float)
        self.half = half
        self.floor_z = float(floor_top)

        # solid box inertia in the body frame (diagonal)
        full = 2.0 * self.half
        a2, b2, c2 = full[:, 0] ** 2, full[:, 1] ** 2, full[:, 2] ** 2
        self.inertia = np.stack(
            [
                self.mass * (b2 + c2) / 12.0,
                self.mass * (a2 + c2) / 12.0,
                self.mass * (a2 + b2) / 12.0,
            ],
            axis=1,
        )
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia_body = 1.0 / self.inertia
        self.corners_local = _CORNER_SIGNS[None, :, :] * self.half[:, None, :]

        # ---- tracked interaction pairs, fixed deterministic ordering ------
        pair_a: list[int] = []
        pair_b: list[int] = []
        seen: set[tuple[int, int]] = set()
        for offset in range(1, min(self.NEIGHBOUR_SPAN, self.n - 1) + 1):
            for i in range(self.n):
                j = (i + offset) % self.n
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                pair_a.append(i)
                pair_b.append(j)
        self.pair_a = np.asarray(pair_a, dtype=np.int64)
        self.pair_b = np.asarray(pair_b, dtype=np.int64)
        self.n_pairs = int(self.pair_a.shape[0])
        # tracked block layout: [domino-domino pairs ..., domino-floor ...]
        self.n_tracked = self.n_pairs + self.n

        self.dt = float(self.DT)
        self.time = 0.0
        self.step_index = 0
        self._impulse_accum = np.zeros((self.n_tracked, 3), dtype=float)
        self._accum_time = 0.0
        self._max_penetration = 0.0
        self._contact_count = 0

        # ---- reference gaps calibrated from the frozen configuration ------
        # allow0 <= 0 is the initial interpenetration of each tracked pair; the
        # constraint gap is (raw gap - allow), so at t = 0 every pair sits
        # exactly at its unstressed touching state.  allow is then ramped to
        # zero over RELAX_TIME, after which the frozen geometry is enforced
        # exactly.
        self.pair_allow = np.zeros(self.n_tracked, dtype=float)
        rot0 = _quat_to_matrix(self.q)
        raw0 = self._raw_gaps(rot0, self._world_corners(rot0))
        self._allow0 = np.minimum(0.0, raw0)
        self.pair_allow = self._allow0.copy()
        self._signed_distance = raw0.copy()

        # gravity is the only external (non-contact) force
        self._external_force = self.mass[:, None] * self.gravity[None, :]

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _world_corners(self, rot: np.ndarray) -> np.ndarray:
        return self.x[:, None, :] + np.einsum(
            "nij,nkj->nki", rot, self.corners_local)

    def _point_box_query(self, points_local: np.ndarray, half: np.ndarray):
        """Signed distance, outward normal and closest point against a box.

        points_local : (P,K,3) points expressed in the box frame
        half         : (P,1,3) half extents
        returns      : sd (P,K), normal_local (P,K,3), closest_local (P,K,3)
        """
        absl = np.abs(points_local)
        gaps = half - absl                        # > 0 means inside on that axis
        inside = np.all(gaps > 0.0, axis=-1)      # (P,K)

        # interior: minimum translation distance to the nearest face
        axis = np.argmin(gaps, axis=-1)
        min_gap = np.take_along_axis(gaps, axis[..., None], axis=-1)[..., 0]
        sign = np.sign(
            np.take_along_axis(points_local, axis[..., None], axis=-1)[..., 0])
        sign = np.where(sign == 0.0, 1.0, sign)
        n_in = np.zeros_like(points_local)
        np.put_along_axis(n_in, axis[..., None], sign[..., None], axis=-1)
        sd_in = -min_gap

        # exterior: distance to the clamped closest point on the box
        closest = np.clip(points_local, -half, half)
        delta = points_local - closest
        dist = np.linalg.norm(delta, axis=-1)
        safe = np.where(dist < 1e-12, 1.0, dist)
        n_out = delta / safe[..., None]
        n_out = np.where((dist < 1e-12)[..., None], n_in, n_out)

        sd = np.where(inside, sd_in, dist)
        normal = np.where(inside[..., None], n_in, n_out)
        closest_pt = np.where(inside[..., None], points_local, closest)
        return sd, normal, closest_pt

    def _raw_gaps(self, rot: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Raw signed geometric gap per tracked pair (no reference offset)."""
        gaps = np.full(self.n_tracked, _FAR_GAP, dtype=float)
        gaps[self.n_pairs:] = (corners[:, :, 2] - self.floor_z).min(axis=1)
        if self.n_pairs:
            for first, second in ((self.pair_a, self.pair_b),
                                  (self.pair_b, self.pair_a)):
                pts = corners[first]
                rb = rot[second]
                d = pts - self.x[second][:, None, :]
                loc = np.einsum("pji,pkj->pki", rb, d)
                sd, _, _ = self._point_box_query(
                    loc, self.half[second][:, None, :])
                gaps[: self.n_pairs] = np.minimum(
                    gaps[: self.n_pairs], sd.min(axis=1))
        return np.minimum(gaps, _FAR_GAP)

    def _collide(self, rot: np.ndarray, corners: np.ndarray):
        """Build the contact set plus the raw per-pair signed distances."""
        allow = self.pair_allow
        raw = np.full(self.n_tracked, _FAR_GAP, dtype=float)

        body_a: list[np.ndarray] = []
        body_b: list[np.ndarray] = []
        c_point: list[np.ndarray] = []
        c_normal: list[np.ndarray] = []
        c_gap: list[np.ndarray] = []
        c_track: list[np.ndarray] = []

        # ---------------- floor: static half space z >= floor_z ------------
        cz = corners[:, :, 2] - self.floor_z                 # (N,8)
        raw[self.n_pairs:] = cz.min(axis=1)
        gap_floor = cz - allow[self.n_pairs:][:, None]
        sel = gap_floor < self.MARGIN
        if np.any(sel):
            bi, ci = np.nonzero(sel)
            normals = np.zeros((bi.shape[0], 3))
            normals[:, 2] = 1.0
            body_a.append(bi)
            body_b.append(np.full(bi.shape, -1, dtype=np.int64))
            c_point.append(corners[bi, ci])
            c_normal.append(normals)
            c_gap.append(gap_floor[bi, ci])
            c_track.append(self.n_pairs + bi)

        # ---------------- domino against domino ----------------------------
        if self.n_pairs:
            allow_pair = allow[: self.n_pairs]
            for first, second in ((self.pair_a, self.pair_b),
                                  (self.pair_b, self.pair_a)):
                pts = corners[first]                          # (P,8,3)
                rb = rot[second]                              # (P,3,3)
                d = pts - self.x[second][:, None, :]
                loc = np.einsum("pji,pkj->pki", rb, d)        # into B's frame
                sd, n_loc, closest = self._point_box_query(
                    loc, self.half[second][:, None, :])
                raw[: self.n_pairs] = np.minimum(
                    raw[: self.n_pairs], sd.min(axis=1))
                gap = sd - allow_pair[:, None]
                mask = gap < self.MARGIN
                if np.any(mask):
                    pi, ki = np.nonzero(mask)
                    n_world = np.einsum("pij,pj->pi", rb[pi], n_loc[pi, ki])
                    nn = np.linalg.norm(n_world, axis=1, keepdims=True)
                    n_world = n_world / np.where(nn < 1e-12, 1.0, nn)
                    cp_world = self.x[second][pi] + np.einsum(
                        "pij,pj->pi", rb[pi], closest[pi, ki])
                    body_a.append(first[pi])
                    body_b.append(second[pi])
                    c_point.append(0.5 * (pts[pi, ki] + cp_world))
                    c_normal.append(n_world)
                    c_gap.append(gap[pi, ki])
                    # tracked slot pi in both corner-vs-face directions
                    c_track.append(pi)

        if body_a:
            out = (
                np.concatenate(body_a),
                np.concatenate(body_b),
                np.concatenate(c_point, axis=0),
                np.concatenate(c_normal, axis=0),
                np.concatenate(c_gap),
                np.concatenate(c_track),
            )
        else:
            out = (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                np.zeros(0),
                np.zeros(0, dtype=np.int64),
            )
        raw = np.where(np.isfinite(raw), raw, _FAR_GAP)
        return out + (raw,)

    # ------------------------------------------------------------------
    # contact solve
    # ------------------------------------------------------------------
    @staticmethod
    def _tangents(normal: np.ndarray):
        ref = np.zeros_like(normal)
        pick = np.abs(normal[:, 2]) < 0.9
        ref[pick, 2] = 1.0
        ref[~pick, 0] = 1.0
        t1 = np.cross(ref, normal)
        n1 = np.linalg.norm(t1, axis=1, keepdims=True)
        t1 = t1 / np.where(n1 < 1e-12, 1.0, n1)
        t2 = np.cross(normal, t1)
        n2 = np.linalg.norm(t2, axis=1, keepdims=True)
        return t1, t2 / np.where(n2 < 1e-12, 1.0, n2)

    def _solve_contacts(self, body_a, body_b, point, normal, gap, track,
                        inv_inertia) -> None:
        count = int(body_a.shape[0])
        if count == 0:
            self._contact_count = 0
            return
        dt = self.dt

        dyn_b = body_b >= 0
        idx_b = np.where(dyn_b, body_b, 0)

        r_a = point - self.x[body_a]
        r_b = np.where(dyn_b[:, None], point - self.x[idx_b], 0.0)

        inv_m_a = self.inv_mass[body_a]
        inv_m_b = np.where(dyn_b, self.inv_mass[idx_b], 0.0)
        ii_a = inv_inertia[body_a]
        ii_b = np.where(dyn_b[:, None, None], inv_inertia[idx_b], 0.0)

        t1, t2 = self._tangents(normal)

        def eff_mass(direction):
            ca = np.cross(r_a, direction)
            cb = np.cross(r_b, direction)
            ta = np.einsum("cij,cj->ci", ii_a, ca)
            tb = np.einsum("cij,cj->ci", ii_b, cb)
            ang = np.sum(direction * (np.cross(ta, r_a) + np.cross(tb, r_b)),
                         axis=1)
            k = inv_m_a + inv_m_b + ang
            return np.where(k < 1e-12, 1e-12, k)

        k_n, k_t1, k_t2 = eff_mass(normal), eff_mass(t1), eff_mass(t2)

        def relative_velocity():
            va = self.v[body_a] + np.cross(self.w[body_a], r_a)
            vb = np.where(
                dyn_b[:, None],
                self.v[idx_b] + np.cross(self.w[idx_b], r_b),
                0.0,
            )
            return va - vb

        # Newton restitution uses the pre-solve approach speed
        vn_pre = np.sum(relative_velocity() * normal, axis=1)
        rest = np.where(vn_pre < -self.REST_THRESHOLD,
                        -self.restitution * vn_pre, 0.0)

        penetration = np.maximum(0.0, -(gap + self.SLOP))
        bias = np.minimum(self.BETA * penetration / dt, self.MAX_RECOVERY)
        # speculative contacts may close their remaining gap within one step
        approach_allowance = np.maximum(gap, 0.0) / dt
        target = bias + rest - approach_allowance

        lam_n = np.zeros(count)
        lam_t1 = np.zeros(count)
        lam_t2 = np.zeros(count)
        relax = self.RELAX

        for _ in range(self.ITERATIONS):
            v_rel = relative_velocity()

            vn = np.sum(v_rel * normal, axis=1)
            new_n = np.maximum(0.0, lam_n + relax * (target - vn) / k_n)
            d_n = new_n - lam_n
            lam_n = new_n

            vt1 = np.sum(v_rel * t1, axis=1)
            vt2 = np.sum(v_rel * t2, axis=1)
            cand1 = lam_t1 - relax * vt1 / k_t1
            cand2 = lam_t2 - relax * vt2 / k_t2
            limit = self.mu * lam_n
            mag = np.sqrt(cand1 * cand1 + cand2 * cand2)
            scale = np.where(mag > limit,
                             limit / np.where(mag < 1e-12, 1.0, mag), 1.0)
            cand1, cand2 = cand1 * scale, cand2 * scale
            d_t1, d_t2 = cand1 - lam_t1, cand2 - lam_t2
            lam_t1, lam_t2 = cand1, cand2

            impulse = (normal * d_n[:, None]
                       + t1 * d_t1[:, None]
                       + t2 * d_t2[:, None])
            impulse = np.nan_to_num(impulse, nan=0.0, posinf=0.0, neginf=0.0)

            np.add.at(self.v, body_a, impulse * inv_m_a[:, None])
            np.add.at(self.w, body_a,
                      np.einsum("cij,cj->ci", ii_a, np.cross(r_a, impulse)))
            if np.any(dyn_b):
                sel = dyn_b
                np.add.at(self.v, body_b[sel],
                          -impulse[sel] * inv_m_b[sel, None])
                np.add.at(self.w, body_b[sel],
                          -np.einsum("cij,cj->ci", ii_b[sel],
                                     np.cross(r_b[sel], impulse[sel])))

        total = (normal * lam_n[:, None]
                 + t1 * lam_t1[:, None]
                 + t2 * lam_t2[:, None])
        total = np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        np.add.at(self._impulse_accum, track, total)

        self._contact_count = count
        self._max_penetration = max(
            self._max_penetration, float(np.max(-np.minimum(gap, 0.0))))

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt

        # relax the calibrated initial interpenetration towards true geometry
        ramp = 1.0 - min(1.0, self.time / self.RELAX_TIME)
        self.pair_allow = self._allow0 * ramp

        # 1. external force: gravity, plus weak damping for numerical settling
        self.v += self.gravity[None, :] * dt
        self.v *= max(0.0, 1.0 - self.LIN_DAMPING * dt)
        self.w *= max(0.0, 1.0 - self.ANG_DAMPING * dt)

        # 2. collision detection against the frozen geometry
        rot = _quat_to_matrix(self.q)
        inv_inertia = np.einsum("nij,nj,nkj->nik", rot,
                                self.inv_inertia_body, rot)
        corners = self._world_corners(rot)
        body_a, body_b, point, normal, gap, track, raw = self._collide(
            rot, corners)
        self._signed_distance = raw

        # 3. frictional contact response
        self._solve_contacts(body_a, body_b, point, normal, gap, track,
                             inv_inertia)

        # 4. clamp for guaranteed finiteness, then integrate
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v *= np.where(speed > self.MAX_LIN,
                           self.MAX_LIN / np.maximum(speed, 1e-12), 1.0)
        spin = np.linalg.norm(self.w, axis=1, keepdims=True)
        self.w *= np.where(spin > self.MAX_ANG,
                           self.MAX_ANG / np.maximum(spin, 1e-12), 1.0)

        self.x = self.x + self.v * dt
        omega_q = np.concatenate([self.w, np.zeros((self.n, 1))], axis=1)
        self.q = _quat_normalize(self.q + 0.5 * dt * _quat_mul(omega_q, self.q))
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)

        self.step_index += 1
        self.time = self.step_index * dt
        self._accum_time += dt

    # ------------------------------------------------------------------
    # Solver contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        target = float(time)
        target_step = int(round(target / self.dt))
        self._impulse_accum[:] = 0.0
        self._accum_time = 0.0
        while self.step_index < target_step:
            self._step()

        if self._accum_time > 0.0:
            # impulse / elapsed time = average contact force over the interval
            interaction = self._impulse_accum / self._accum_time
        else:
            interaction = np.zeros((self.n_tracked, 3), dtype=float)
        interaction = np.clip(
            np.nan_to_num(interaction, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0e7, 1.0e7)

        signed = np.clip(
            np.nan_to_num(np.asarray(self._signed_distance, dtype=float),
                          nan=0.0, posinf=_FAR_GAP, neginf=-1.0),
            -1.0, _FAR_GAP)

        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.astype(np.int32),
            attributes={},
            fields={
                "external_force": self._external_force.copy(),
                "interaction_force": interaction,
                "contact_signed_distance": signed,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        rot = _quat_to_matrix(self.q)
        # local +z of an upright domino points up; cos of its tilt from vertical
        upright = float(np.sum(rot[:, 2, 2] > 0.7))
        return {
            "method": "semi_implicit_euler_pgs_impulse_contacts",
            "dt": float(self.dt),
            "steps": int(self.step_index),
            "solver_iterations": int(self.ITERATIONS),
            "bodies": int(self.n),
            "tracked_pairs": int(self.n_tracked),
            "friction": float(self.mu),
            "restitution": float(self.restitution),
            "contact_length_scale": float(self.length_scale),
            "floor_z": float(self.floor_z),
            "initial_overlap_relax_time": float(self.RELAX_TIME),
            "max_initial_pair_overlap": float(-np.min(self._allow0)),
            "max_penetration": float(self._max_penetration),
            "last_contact_count": int(self._contact_count),
            "upright_bodies_remaining": upright,
            "seed": int(self._seed),
        }
