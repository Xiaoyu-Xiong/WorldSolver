"""Rigid-body card-house solver.

Self-contained rigid multibody simulator (NumPy only):

* semi-implicit Euler time stepping at a fixed small sub-step,
* box-box and box-floor contact detection through vertex/box signed distances,
* velocity-level contact resolution with a relaxed projected Jacobi (mass-split)
  iteration: non-penetration, Baumgarte stabilization, Newton restitution and a
  two-directional Coulomb friction cone,
* deterministic export of the required raw StateFrame trajectory fields.

No scripted animation, keyframes or precomputed trajectories: every sample comes
from the integrated equations of motion.
"""
from __future__ import annotations

import os

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices for quaternions in xyzw order. q: [B,4] -> [B,3,3]."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n < _EPS, 1.0, n)
    x, y, z, w = (q / n).T
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


def _integrate_quat(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """q_{n+1} = normalize(q_n + 0.5*dt*omega (x) q_n), xyzw convention."""
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dq = np.empty_like(q)
    dq[:, 0] = qw * wx + wy * qz - wz * qy
    dq[:, 1] = qw * wy + wz * qx - wx * qz
    dq[:, 2] = qw * wz + wx * qy - wy * qx
    dq[:, 3] = -(wx * qx + wy * qy + wz * qz)
    out = q + 0.5 * dt * dq
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n = np.where(n < _EPS, 1.0, n)
    return out / n


_BOX_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=float,
)

# Frozen scene geometry fallback (matches scene_geometry.yaml half extents,
# ordered by state_binding body_id).
_FALLBACK_HALF_EXTENTS = [
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.45, 0.18, 0.015],
    [0.16, 0.16, 0.16],
    [0.16, 0.16, 0.16],
    [0.19, 0.19, 0.19],
]
_FALLBACK_FLOOR_TOP = 0.0


class BenchmarkSolver:
    """Frictional rigid-contact solver for the card-house block-impact scene."""

    # numerical parameters
    sub_dt = 8.0e-4
    solver_iterations = 26
    speculative_gap = 1.0e-2
    penetration_slop = 1.5e-3
    baumgarte = 0.2
    max_correction_velocity = 0.08
    restitution_threshold = 0.30
    linear_damping = 0.6
    angular_damping = 0.9
    freeze_linear = 1.0e-2
    freeze_angular = 4.0e-2
    max_linear_speed = 40.0
    max_angular_speed = 80.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        config = dict(getattr(scene, "config", {}) or {})
        meta = scene.metadata() if hasattr(scene, "metadata") else {}
        frame = scene.initial_state()

        self.gravity = np.asarray(
            meta.get("gravity", config.get("gravity", [0.0, 0.0, -9.81])), dtype=float
        ).reshape(3)
        self.mu = float(meta.get("friction_coefficient", 0.8))
        self.restitution = float(meta.get("restitution", 0.1))

        self.n = int(frame.position.shape[0])
        self.pos = np.array(frame.position, dtype=float)
        self.quat = (
            np.array(frame.orientation, dtype=float)
            if frame.orientation is not None
            else np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        )
        self.vel = (
            np.array(frame.velocity, dtype=float)
            if frame.velocity is not None
            else np.zeros((self.n, 3))
        )
        self.omega = (
            np.array(frame.angular_velocity, dtype=float)
            if frame.angular_velocity is not None
            else np.zeros((self.n, 3))
        )
        self.mass = (
            np.array(frame.mass, dtype=float)
            if frame.mass is not None
            else np.full(self.n, 0.1)
        )
        self.body_id = (
            np.array(frame.body_id, dtype=int)
            if frame.body_id is not None
            else np.arange(self.n, dtype=int)
        )

        attrs = dict(frame.attributes or {})
        if "release_time" in attrs:
            self.release_time = np.asarray(attrs["release_time"], dtype=float).reshape(self.n)
        else:
            self.release_time = np.zeros(self.n, dtype=float)
        if "is_impact_block" in attrs:
            self.is_block = np.asarray(attrs["is_impact_block"], dtype=float).reshape(self.n) > 0.5
        else:
            self.is_block = np.zeros(self.n, dtype=bool)
        # bodies that are held in place until their release time
        self.held_initially = self.is_block & (self.release_time > 0.0)

        self.half = self._load_half_extents()
        self.floor_top = self._load_floor_top()
        self.radius = np.linalg.norm(self.half, axis=1)
        self.local_verts = _BOX_SIGNS[None, :, :] * self.half[:, None, :]

        # body-frame principal inertia of a homogeneous box
        h2 = self.half ** 2
        self.inertia_body = (self.mass[:, None] / 3.0) * np.stack(
            [h2[:, 1] + h2[:, 2], h2[:, 0] + h2[:, 2], h2[:, 0] + h2[:, 1]], axis=1
        )
        self.inertia_body = np.maximum(self.inertia_body, 1e-9)

        # static pseudo-body index (floor / world) is self.n
        self.static_index = self.n
        self.nb = self.n + 1

        # fixed contact-pair slots: dynamic-dynamic pairs then body-floor pairs
        lut = np.full((self.nb, self.nb), -1, dtype=int)
        first = []
        second = []
        slot = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                lut[i, j] = lut[j, i] = slot
                first.append(i)
                second.append(j)
                slot += 1
        for i in range(self.n):
            lut[i, self.static_index] = lut[self.static_index, i] = slot
            first.append(i)
            second.append(self.static_index)
            slot += 1
        self.pair_lut = lut
        self.pair_first = np.asarray(first, dtype=int)
        self.pair_second = np.asarray(second, dtype=int)
        self.n_pairs = slot

        self.contact_length_scale = float(2.0 * np.min(self.half[:10, 2])) if self.n >= 10 else 0.03
        if not np.isfinite(self.contact_length_scale) or self.contact_length_scale <= 0.0:
            self.contact_length_scale = 0.03

        self.time = 0.0
        self.pair_impulse = np.zeros((self.n_pairs, 3), dtype=float)
        self.accum_time = 0.0
        self._steps = 0
        self._max_penetration = 0.0
        self._update_derived()

    def _load_half_extents(self) -> np.ndarray:
        half = np.array(_FALLBACK_HALF_EXTENTS, dtype=float)
        if half.shape[0] < self.n:
            pad = np.tile(half[-1], (self.n - half.shape[0], 1))
            half = np.vstack([half, pad])
        half = half[: self.n].copy()
        data = self._read_geometry_yaml()
        if data:
            for obj in data.get("objects", []) or []:
                binding = obj.get("state_binding") or {}
                if binding.get("selector") != "body_id":
                    continue
                idx = int(binding.get("value", -1))
                geom = obj.get("geometry") or {}
                he = geom.get("half_extents")
                if 0 <= idx < self.n and geom.get("type") == "box" and he:
                    half[idx] = np.asarray(he, dtype=float).reshape(3)
        return np.maximum(half, 1e-4)

    def _load_floor_top(self) -> float:
        data = self._read_geometry_yaml()
        if data:
            for obj in data.get("objects", []) or []:
                if obj.get("id") != "floor":
                    continue
                geom = obj.get("geometry") or {}
                he = geom.get("half_extents")
                tz = ((obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0])[2]
                if he:
                    return float(tz) + float(he[2])
        return _FALLBACK_FLOOR_TOP

    def _read_geometry_yaml(self):
        cached = getattr(self, "_geometry_yaml", "unset")
        if cached != "unset":
            return cached
        data = None
        try:
            import yaml  # type: ignore

            here = os.path.dirname(os.path.abspath(__file__))
            for candidate in (
                os.path.join(here, os.pardir, "scene_geometry.yaml"),
                os.path.join(os.getcwd(), "scene_geometry.yaml"),
            ):
                path = os.path.abspath(candidate)
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        data = yaml.safe_load(handle)
                    break
        except Exception:
            data = None
        self._geometry_yaml = data
        return data

    # -------------------------------------------------------------- utilities
    def _active_mask(self) -> np.ndarray:
        """Bodies currently free to move (blocks are held until release)."""
        return ~(self.held_initially & (self.time < self.release_time - 1e-12))

    def _update_derived(self) -> None:
        self.rot = _quat_to_matrix(self.quat)
        active = self._active_mask()
        inv_mass = np.zeros(self.nb, dtype=float)
        inv_mass[: self.n] = np.where(active, 1.0 / np.maximum(self.mass, 1e-9), 0.0)
        self.inv_mass = inv_mass
        inv_inertia = np.zeros((self.nb, 3, 3), dtype=float)
        inv_body = np.where(active[:, None], 1.0 / self.inertia_body, 0.0)
        # I^-1_world = R diag(inv_body) R^T
        inv_inertia[: self.n] = np.einsum(
            "bij,bj,bkj->bik", self.rot, inv_body, self.rot
        )
        self.inv_inertia = inv_inertia
        self.active = active

    def _world_vertices(self) -> np.ndarray:
        return self.pos[:, None, :] + np.einsum("bij,bkj->bki", self.rot, self.local_verts)

    # ------------------------------------------------------- contact assembly
    def _detect_contacts(self, verts: np.ndarray):
        """Return dict of contact arrays (indices, normals, points, gaps, slots)."""
        ia_l = []
        ib_l = []
        pt_l = []
        nrm_l = []
        gap_l = []

        thresh = self.speculative_gap

        # --- floor half-space (normal +z, surface at floor_top)
        vz = verts[:, :, 2] - self.floor_top
        rows, cols = np.nonzero(vz < thresh)
        if rows.size:
            keep = self.active[rows]
            rows, cols = rows[keep], cols[keep]
        if rows.size:
            pts = verts[rows, cols, :]
            ia_l.append(rows)
            ib_l.append(np.full(rows.shape, self.static_index, dtype=int))
            pt_l.append(pts)
            nrm_l.append(np.tile(np.array([0.0, 0.0, 1.0]), (rows.size, 1)))
            gap_l.append(vz[rows, cols])

        # --- pairwise boxes (broad phase on bounding spheres)
        centers = self.pos
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not (self.active[i] or self.active[j]):
                    continue
                d = centers[i] - centers[j]
                if float(d @ d) > (self.radius[i] + self.radius[j] + thresh) ** 2:
                    continue
                for a, b in ((i, j), (j, i)):
                    res = self._vertex_box_contacts(verts[a], b, thresh)
                    if res is None:
                        continue
                    pts, nrm, gap = res
                    cnt = gap.shape[0]
                    ia_l.append(np.full(cnt, a, dtype=int))
                    ib_l.append(np.full(cnt, b, dtype=int))
                    pt_l.append(pts)
                    nrm_l.append(nrm)
                    gap_l.append(gap)

        if not gap_l:
            return None
        ia = np.concatenate(ia_l)
        ib = np.concatenate(ib_l)
        pts = np.concatenate(pt_l, axis=0)
        nrm = np.concatenate(nrm_l, axis=0)
        gap = np.concatenate(gap_l)
        norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.where(norm < _EPS, 1.0, norm)
        slots = self.pair_lut[ia, ib]
        return {
            "ia": ia,
            "ib": ib,
            "point": pts,
            "normal": nrm,
            "gap": gap,
            "slot": slots,
        }

    def _vertex_box_contacts(self, vertices: np.ndarray, box: int, thresh: float):
        """Contacts of `vertices` (world) against box `box`.

        Normal points from the box surface toward the vertex.
        """
        rel = vertices - self.pos[box]
        local = rel @ self.rot[box]  # world -> local
        h = self.half[box]
        q = np.abs(local) - h
        inside = np.all(q < 0.0, axis=1)
        outside_gap = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        near = (~inside) & (outside_gap < thresh)
        sel = inside | near
        if not np.any(sel):
            return None
        idx = np.nonzero(sel)[0]
        q_s = q[idx]
        loc_s = local[idx]
        sign = np.where(loc_s >= 0.0, 1.0, -1.0)
        n_local = np.zeros_like(q_s)
        gap = np.empty(idx.size, dtype=float)

        ins = inside[idx]
        if np.any(ins):
            axis = np.argmax(q_s[ins], axis=1)
            rows = np.nonzero(ins)[0]
            n_local[rows, axis] = sign[rows, axis]
            gap[rows] = q_s[rows, axis]
        out = ~ins
        if np.any(out):
            rows = np.nonzero(out)[0]
            d = np.maximum(q_s[rows], 0.0) * sign[rows]
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            dn = np.where(dn < _EPS, 1.0, dn)
            n_local[rows] = d / dn
            gap[rows] = np.linalg.norm(np.maximum(q_s[rows], 0.0), axis=1)

        normal_world = n_local @ self.rot[box].T
        return vertices[idx], normal_world, gap

    def _pair_signed_distances(self, verts: np.ndarray) -> np.ndarray:
        """Approximate signed gap per fixed contact-pair slot."""
        out = np.empty(self.n_pairs, dtype=float)
        # floor slots
        floor_gap = np.min(verts[:, :, 2] - self.floor_top, axis=1)
        base = self.n * (self.n - 1) // 2
        out[base:] = floor_gap
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                out[k] = min(
                    self._box_sdf(verts[i], j), self._box_sdf(verts[j], i)
                )
                k += 1
        return out

    def _box_sdf(self, vertices: np.ndarray, box: int) -> float:
        local = (vertices - self.pos[box]) @ self.rot[box]
        q = np.abs(local) - self.half[box]
        inside = np.all(q < 0.0, axis=1)
        sd = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        if np.any(inside):
            sd = np.where(inside, np.max(q, axis=1), sd)
        return float(np.min(sd))

    # ---------------------------------------------------------------- solving
    def _solve_contacts(self, contacts, dt: float, record: bool) -> None:
        ia = contacts["ia"]
        ib = contacts["ib"]
        n = contacts["normal"]
        gap = contacts["gap"]
        pts = contacts["point"]
        c = gap.shape[0]

        ra = pts - self.pos_full[ia]
        rb = pts - self.pos_full[ib]

        inv_m_a = self.inv_mass[ia]
        inv_m_b = self.inv_mass[ib]
        Ia = self.inv_inertia[ia]
        Ib = self.inv_inertia[ib]

        # tangent basis
        helper = np.zeros((c, 3), dtype=float)
        pick = np.abs(n[:, 2]) < 0.9
        helper[pick] = np.array([0.0, 0.0, 1.0])
        helper[~pick] = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(helper, n)
        t1n = np.linalg.norm(t1, axis=1, keepdims=True)
        t1 = t1 / np.where(t1n < _EPS, 1.0, t1n)
        t2 = np.cross(n, t1)

        def eff_mass(direction):
            k = inv_m_a + inv_m_b
            aa = np.cross(ra, direction)
            bb = np.cross(rb, direction)
            ta = np.einsum("cij,cj->ci", Ia, aa)
            tb = np.einsum("cij,cj->ci", Ib, bb)
            k = k + np.sum(np.cross(ta, ra) * direction, axis=1)
            k = k + np.sum(np.cross(tb, rb) * direction, axis=1)
            return np.maximum(k, 1e-9)

        kn = eff_mass(n)
        kt1 = eff_mass(t1)
        kt2 = eff_mass(t2)

        # mass splitting keeps the Jacobi sweep non-overshooting
        counts = np.zeros(self.nb, dtype=float)
        np.add.at(counts, ia, 1.0)
        np.add.at(counts, ib, 1.0)
        counts[self.static_index] = 1.0
        counts = np.maximum(counts, 1.0)
        split = np.maximum(counts[ia], counts[ib])
        kn = kn * split
        kt1 = kt1 * split
        kt2 = kt2 * split

        # pre-step normal velocity for restitution
        v_rel0 = (
            self.vel_full[ia]
            + np.cross(self.omega_full[ia], ra)
            - self.vel_full[ib]
            - np.cross(self.omega_full[ib], rb)
        )
        vn0 = np.sum(v_rel0 * n, axis=1)
        rest = np.where(vn0 < -self.restitution_threshold, -self.restitution * vn0, 0.0)

        depth = np.maximum(-gap - self.penetration_slop, 0.0)
        bias = np.minimum(self.baumgarte * depth / dt, self.max_correction_velocity)
        spec = np.where(gap > 0.0, -gap / dt, 0.0)
        target = rest + bias + spec

        ln = np.zeros(c, dtype=float)
        lt1 = np.zeros(c, dtype=float)
        lt2 = np.zeros(c, dtype=float)
        total = np.zeros((c, 3), dtype=float)

        for _ in range(self.solver_iterations):
            v_rel = (
                self.vel_full[ia]
                + np.cross(self.omega_full[ia], ra)
                - self.vel_full[ib]
                - np.cross(self.omega_full[ib], rb)
            )
            vn = np.sum(v_rel * n, axis=1)
            d_ln = -(vn - target) / kn
            new_ln = np.maximum(ln + d_ln, 0.0)
            d_ln = new_ln - ln
            ln = new_ln

            vt1 = np.sum(v_rel * t1, axis=1)
            vt2 = np.sum(v_rel * t2, axis=1)
            cand1 = lt1 - vt1 / kt1
            cand2 = lt2 - vt2 / kt2
            limit = self.mu * ln
            mag = np.sqrt(cand1 * cand1 + cand2 * cand2)
            scale = np.where(mag > limit, limit / np.maximum(mag, 1e-12), 1.0)
            cand1 = cand1 * scale
            cand2 = cand2 * scale
            d_t1 = cand1 - lt1
            d_t2 = cand2 - lt2
            lt1, lt2 = cand1, cand2

            imp = d_ln[:, None] * n + d_t1[:, None] * t1 + d_t2[:, None] * t2
            total += imp
            self._apply_impulses(ia, ib, ra, rb, imp)

        if record:
            slots = contacts["slot"]
            signs = np.where(ia == self.pair_first[slots], 1.0, -1.0)
            np.add.at(self.pair_impulse, slots, total * signs[:, None])
        self._contact_count = c
        if c:
            self._max_penetration = max(self._max_penetration, float(np.max(-gap)))

    def _apply_impulses(self, ia, ib, ra, rb, imp) -> None:
        np.add.at(self.vel_full, ia, imp * self.inv_mass[ia][:, None])
        np.add.at(self.vel_full, ib, -imp * self.inv_mass[ib][:, None])
        wa = np.einsum("cij,cj->ci", self.inv_inertia[ia], np.cross(ra, imp))
        wb = np.einsum("cij,cj->ci", self.inv_inertia[ib], np.cross(rb, imp))
        np.add.at(self.omega_full, ia, wa)
        np.add.at(self.omega_full, ib, -wb)
        self.vel_full[self.static_index] = 0.0
        self.omega_full[self.static_index] = 0.0

    # -------------------------------------------------------------- time step
    def _step(self, dt: float, record: bool = True, commit: bool = True) -> None:
        self._update_derived()

        active = self.active
        vel = self.vel.copy()
        omega = self.omega.copy()
        # gravity acts only on released bodies; held blocks are kinematically fixed
        vel[active] += dt * self.gravity
        vel[~active] = 0.0
        omega[~active] = 0.0

        # padded arrays including the static world body
        self.pos_full = np.vstack([self.pos, np.zeros((1, 3))])
        self.vel_full = np.vstack([vel, np.zeros((1, 3))])
        self.omega_full = np.vstack([omega, np.zeros((1, 3))])

        verts = self._world_vertices()
        contacts = self._detect_contacts(verts)
        if contacts is not None:
            self._solve_contacts(contacts, dt, record)
        else:
            self._contact_count = 0

        vel = self.vel_full[: self.n]
        omega = self.omega_full[: self.n]

        # mild viscous damping (air drag) for numerical quiescence
        vel *= float(np.exp(-self.linear_damping * dt))
        omega *= float(np.exp(-self.angular_damping * dt))

        vel[~active] = 0.0
        omega[~active] = 0.0

        # clamp for guaranteed finiteness
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(sp > self.max_linear_speed, vel * (self.max_linear_speed / np.maximum(sp, _EPS)), vel)
        sw = np.linalg.norm(omega, axis=1, keepdims=True)
        omega = np.where(sw > self.max_angular_speed, omega * (self.max_angular_speed / np.maximum(sw, _EPS)), omega)

        # static-friction style quiescence: supported bodies moving slower than
        # the freeze thresholds are held (avoids Jacobi creep in equilibrium)
        supported = np.zeros(self.n, dtype=bool)
        if contacts is not None:
            touch = contacts["gap"] < self.speculative_gap
            for arr in (contacts["ia"], contacts["ib"]):
                sel = arr[touch]
                sel = sel[sel < self.n]
                supported[sel] = True
        slow = (
            supported
            & (np.linalg.norm(vel, axis=1) < self.freeze_linear)
            & (np.linalg.norm(omega, axis=1) < self.freeze_angular)
        )
        vel[slow] = 0.0
        omega[slow] = 0.0

        if commit:
            self.vel = vel
            self.omega = omega
            new_pos = self.pos + dt * vel
            new_quat = _integrate_quat(self.quat, omega, dt)
            self.pos = np.where(active[:, None], new_pos, self.pos)
            self.quat = np.where(active[:, None], new_quat, self.quat)
            self.time += dt
            self.accum_time += dt
            self._steps += 1
        self._update_derived()

    # ------------------------------------------------------------- public API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-9:
            raise ValueError("advance_to must be called with non-decreasing times")

        self.pair_impulse[:] = 0.0
        self.accum_time = 0.0

        guard = 0
        while self.time < target - 1e-9:
            dt = min(self.sub_dt, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt, record=True, commit=True)
            guard += 1
            if guard > 10_000_000:
                break
        if abs(self.time - target) < 1e-6:
            self.time = target

        if self.accum_time > 1e-12:
            interaction = self.pair_impulse / self.accum_time
        else:
            interaction = self._instantaneous_pair_forces()

        return self._make_frame(target, interaction)

    def _instantaneous_pair_forces(self) -> np.ndarray:
        """Contact forces for the current state without advancing (t = 0 sample)."""
        saved = (self.pos.copy(), self.quat.copy(), self.vel.copy(), self.omega.copy(),
                 self.time, self.pair_impulse.copy())
        self.pair_impulse[:] = 0.0
        dt = self.sub_dt
        try:
            self._step(dt, record=True, commit=False)
            forces = self.pair_impulse / dt
        except Exception:
            forces = np.zeros((self.n_pairs, 3), dtype=float)
        (self.pos, self.quat, self.vel, self.omega, self.time, self.pair_impulse) = saved
        self._update_derived()
        return forces

    def _make_frame(self, time: float, interaction: np.ndarray) -> StateFrame:
        self._update_derived()
        verts = self._world_vertices()
        signed = self._pair_signed_distances(verts)

        external = np.zeros((self.n, 3), dtype=float)
        external[self.active] = self.mass[self.active, None] * self.gravity[None, :]

        pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        omega = np.nan_to_num(self.omega, nan=0.0, posinf=0.0, neginf=0.0)
        quat = np.nan_to_num(self.quat, nan=0.0, posinf=0.0, neginf=0.0)
        qn = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / np.where(qn < _EPS, 1.0, qn)
        signed = np.nan_to_num(signed, nan=0.0, posinf=1e3, neginf=-1e3)
        interaction = np.nan_to_num(interaction, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(max(time, 0.0)),
            position=pos.astype(float),
            velocity=vel.astype(float),
            orientation=quat.astype(float),
            angular_velocity=omega.astype(float),
            mass=self.mass.astype(float),
            body_id=self.body_id.astype(int),
            attributes={
                "release_time": self.release_time.astype(float),
                "is_impact_block": self.is_block.astype(float),
                "is_active": self.active.astype(float),
            },
            fields={
                "contact_signed_distance": signed.astype(float),
                "external_force": external.astype(float),
                "interaction_force": interaction.astype(float),
                "contact_pair_bodies": np.stack(
                    [self.pair_first.astype(float), self.pair_second.astype(float)], axis=1
                ),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "friction_coefficient": float(self.mu),
                "restitution": float(self.restitution),
                "sub_dt": float(self.sub_dt),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "sub_dt": float(self.sub_dt),
            "solver_iterations": int(self.solver_iterations),
            "steps": int(self._steps),
            "bodies": int(self.n),
            "contact_pairs": int(self.n_pairs),
            "last_contact_count": int(getattr(self, "_contact_count", 0)),
            "max_penetration": float(self._max_penetration),
            "friction_coefficient": float(self.mu),
            "restitution": float(self.restitution),
        }
