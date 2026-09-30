"""Deterministic rigid-body solver for the rings-over-poles scene.

The frozen scene drops a large collection of non-convex rings onto a 3x3 array
of vertical poles and a receiving ground box.  Every ring is simulated as a
rigid body whose collision shape is a ring of contact spheres placed on its
mid-circle; that collocation reproduces the frozen axisymmetric ring geometry
(inner bore radius ~0.17 m, outer radius ~0.39 m, half height ~0.11 m) and is
what lets rings thread over the poles and interlock with each other.

Dynamics are the rigid-body Newton-Euler equations integrated with a fixed
5 ms sub-step.  Contacts (ring/ring sphere pairs, sphere vs. finite cylinder
poles, sphere vs. ground box) are resolved with a projected-impulse (Jacobi)
velocity solver including Coulomb friction, followed by a positional
projection pass that removes residual overlap without injecting energy.
Everything is deterministic and uses only NumPy.
"""
from __future__ import annotations

import time as _time

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical / numerical constants (SI units, z-up).
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
SUB_DT = 0.0025                    # integration sub-step (s)
N_SPHERE = 12                      # contact spheres per ring
RING_RADIUS = 0.28                 # mid-circle radius of the ring collocation
SPHERE_RADIUS = 0.12               # contact sphere radius
CONTACT_LENGTH_SCALE = 0.06        # frozen ring wall thickness (normalization)
# The frozen mesh revolves its [r, z] profile about the body-frame z axis with
# z in [0, 0.26]; the ring's centre of mass therefore sits at local z = 0.13.
BODY_OFFSET = np.array([0.0, 0.0, 0.13], dtype=float)

SOLVER_ITERATIONS = 40
POSITION_ITERATIONS = 4
POSITION_RELAXATION = 0.5
MAX_POSITION_CORRECTION = 0.01
BAUMGARTE = 0.2
POSITION_SLOP = 0.002
MAX_BIAS_VELOCITY = 0.5
LINEAR_DAMPING = 0.15
ANGULAR_DAMPING = 0.35

NEAR_CONTACT_SKIN = 0.01           # report contacts within 1 cm of touching
MU_RING = 0.6
MU_POLE = 0.45
MU_GROUND = 0.6
PAIR_CHUNK = 256

GROUND_MIN = np.array([-3.2, -3.2, -0.16], dtype=float)
GROUND_MAX = np.array([3.2, 3.2, 0.0], dtype=float)
POLE_RADIUS = 0.075
POLE_HEIGHT = 2.6
POLE_CENTERS = np.array(
    [
        [-0.9, -0.9], [0.0, -0.9], [0.9, -0.9],
        [-0.9, 0.0], [0.0, 0.0], [0.9, 0.0],
        [-0.9, 0.9], [0.0, 0.9], [0.9, 0.9],
    ],
    dtype=float,
)

CONTACT_KEYS = ("ci", "cj", "nvec", "pen", "point", "mu")


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body->world rotation matrices for xyzw quaternions, shape [M,3,3]."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((q.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)
    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)
    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def _tangent_basis(nvec: np.ndarray):
    """Deterministic orthonormal tangents for each contact normal."""
    ref = np.zeros_like(nvec)
    axis = np.argmin(np.abs(nvec), axis=1)
    ref[np.arange(nvec.shape[0]), axis] = 1.0
    t1 = np.cross(nvec, ref)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1e-12)[:, None]
    t2 = np.cross(nvec, t1)
    return t1, t2


class BenchmarkSolver:
    """Impulse-based rigid-body solver honouring the frozen scene geometry."""

    def __init__(self) -> None:
        self._ready = False
        self.n = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        self._seed = int(seed)
        self._sample_dt = float(config.get("sample_dt", 0.02))

        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float)
        n = int(pos.shape[0])
        self.n = n

        vel = frame.velocity
        vel = np.zeros((n, 3)) if vel is None else np.asarray(vel, dtype=float)[:, :3]

        ori = frame.orientation
        if ori is None:
            ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        ori = np.asarray(ori, dtype=float)[:, :4].copy()
        norms = np.linalg.norm(ori, axis=1)
        ori /= np.where(norms > 1e-12, norms, 1.0)[:, None]

        ang = frame.angular_velocity
        ang = np.zeros((n, 3)) if ang is None else np.asarray(ang, dtype=float)[:, :3]

        mass = frame.mass
        mass = np.ones(n) if mass is None else np.asarray(mass, dtype=float).reshape(-1)
        if mass.shape[0] != n or not np.isfinite(mass).all():
            mass = np.ones(n)
        self.mass = np.maximum(mass, 1e-6)

        # One extra slot (index n) represents the static world.  ``self.x``
        # stores centres of mass; the reported (renderer) position is the mesh
        # origin, i.e. the COM shifted by -R @ BODY_OFFSET.
        self.x = np.zeros((n + 1, 3))
        self.x[:n] = pos
        self.v = np.zeros((n + 1, 3))
        self.v[:n] = vel
        self.w = np.zeros((n + 1, 3))
        self.w[:n] = ang
        self.q = np.zeros((n + 1, 4))
        self.q[:n] = ori
        self.q[n] = np.array([0.0, 0.0, 0.0, 1.0])

        inv_m = np.zeros(n + 1)
        inv_m[:n] = 1.0 / self.mass
        self.inv_m = inv_m

        spin = 0.4 * SPHERE_RADIUS * SPHERE_RADIUS
        i_trans = self.mass * (0.5 * RING_RADIUS * RING_RADIUS + spin)
        i_axial = self.mass * (RING_RADIUS * RING_RADIUS + spin)
        self.inv_I = np.zeros((n + 1, 3))
        self.inv_I[:n, 0] = 1.0 / i_trans
        self.inv_I[:n, 1] = 1.0 / i_trans
        self.inv_I[:n, 2] = 1.0 / i_axial

        theta = 2.0 * np.pi * np.arange(N_SPHERE, dtype=float) / N_SPHERE
        self.s_off = np.stack(
            [RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta), np.zeros(N_SPHERE)],
            axis=1,
        )
        self._body_index = np.repeat(np.arange(n), N_SPHERE)

        self.steps_done = 0
        self._last_contacts = None
        self._last_impulses = None
        self._frame_seconds: list = []
        self._pen_counts: list = []
        self._min_signed: list = []
        self._max_pen = 0.0
        self._ready = True

        # Frozen initial positions address the mesh origin; convert to COM.
        rot0 = _quat_to_matrix(self.q[:n])
        self.x[:n] = pos + np.einsum("nij,j->ni", rot0, BODY_OFFSET)

        self._prepare_ext()
        self._last_contacts = self._build_contacts()

    # ------------------------------------------------------------------
    # kinematics helpers
    # ------------------------------------------------------------------
    def _prepare_ext(self) -> None:
        self.R_ext = _quat_to_matrix(self.q)

    def _sphere_points(self) -> np.ndarray:
        n = self.n
        pts = np.einsum("nij,sj->nsi", self.R_ext[:n], self.s_off) + self.x[:n, None, :]
        return pts.reshape(-1, 3)

    def _inv_inertia(self, idx: np.ndarray, vec: np.ndarray) -> np.ndarray:
        rot = self.R_ext[idx]
        local = np.einsum("kji,kj->ki", rot, vec)
        local = local * self.inv_I[idx]
        return np.einsum("kij,kj->ki", rot, local)

    def _eff_mass(self, ci, cj, ri, rj, dvec) -> np.ndarray:
        ai = self._inv_inertia(ci, np.cross(ri, dvec))
        aj = self._inv_inertia(cj, np.cross(rj, dvec))
        ti = np.einsum("ij,ij->i", np.cross(ai, ri), dvec)
        tj = np.einsum("ij,ij->i", np.cross(aj, rj), dvec)
        return self.inv_m[ci] + self.inv_m[cj] + ti + tj

    def _apply_impulse(self, ci, ri, cj, rj, imp) -> None:
        np.add.at(self.v, ci, -imp * self.inv_m[ci, None])
        np.add.at(self.w, ci, -self._inv_inertia(ci, np.cross(ri, imp)))
        np.add.at(self.v, cj, imp * self.inv_m[cj, None])
        np.add.at(self.w, cj, self._inv_inertia(cj, np.cross(rj, imp)))

    # ------------------------------------------------------------------
    # contact generation
    # ------------------------------------------------------------------
    def _empty_contacts(self) -> dict:
        z = np.zeros(0)
        zc = np.zeros((0, 3))
        return {
            "ci": np.zeros(0, dtype=np.int64),
            "cj": np.zeros(0, dtype=np.int64),
            "nvec": zc,
            "pen": z,
            "point": zc,
            "mu": z,
            "ri": zc,
            "rj": zc,
            "x0i": zc,
            "x0j": zc,
        }

    def _build_contacts(self) -> dict:
        pts = self._sphere_points()
        body = self._body_index
        parts = [
            self._ring_ring_contacts(pts),
            self._ground_contacts(pts, body),
            self._pole_contacts(pts, body),
        ]
        parts = [p for p in parts if p is not None and p["ci"].size]
        if not parts:
            return self._empty_contacts()
        merged = {k: np.concatenate([p[k] for p in parts]) for k in CONTACT_KEYS}
        x0i = self.x[merged["ci"]].copy()
        x0j = self.x[merged["cj"]].copy()
        merged["x0i"] = x0i
        merged["x0j"] = x0j
        merged["ri"] = merged["point"] - x0i
        merged["rj"] = merged["point"] - x0j
        return merged

    def _ring_ring_contacts(self, pts: np.ndarray):
        n = self.n
        if n < 2:
            return None
        ps = pts.reshape(n, N_SPHERE, 3)
        centers = self.x[:n]
        delta_c = centers[:, None, :] - centers[None, :, :]
        dist2_c = np.einsum("ijk,ijk->ij", delta_c, delta_c)
        limit = 2.0 * (RING_RADIUS + SPHERE_RADIUS) + NEAR_CONTACT_SKIN
        iu, ju = np.triu_indices(n, 1)
        sel = dist2_c[iu, ju] < limit * limit
        ii = iu[sel]
        jj = ju[sel]
        if ii.size == 0:
            return None
        rlim = 2.0 * SPHERE_RADIUS + NEAR_CONTACT_SKIN
        ci_parts, cj_parts, nv_parts, pen_parts, pt_parts = [], [], [], [], []
        for start in range(0, ii.size, PAIR_CHUNK):
            sl = slice(start, min(start + PAIR_CHUNK, ii.size))
            pi = ps[ii[sl]]
            pj = ps[jj[sl]]
            deltas = pj[:, None, :, :] - pi[:, :, None, :]
            d2 = np.einsum("...k,...k->...", deltas, deltas)
            hit = d2 < rlim * rlim
            if not hit.any():
                continue
            found = np.argwhere(hit)
            pair = found[:, 0]
            si = found[:, 1]
            sj = found[:, 2]
            pea = pi[pair, si]
            peb = pj[pair, sj]
            dvec = peb - pea
            dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
            pen = 2.0 * SPHERE_RADIUS - dist
            keep = pen > -NEAR_CONTACT_SKIN
            if not keep.any():
                continue
            ci_parts.append(ii[sl][pair][keep])
            cj_parts.append(jj[sl][pair][keep])
            nv_parts.append(dvec[keep] / dist[keep, None])
            pen_parts.append(pen[keep])
            pt_parts.append(0.5 * (pea[keep] + peb[keep]))
        if not ci_parts:
            return None
        ci = np.concatenate(ci_parts).astype(np.int64)
        cj = np.concatenate(cj_parts).astype(np.int64)
        return {
            "ci": ci,
            "cj": cj,
            "nvec": np.concatenate(nv_parts),
            "pen": np.concatenate(pen_parts),
            "point": np.concatenate(pt_parts),
            "mu": np.full(ci.size, MU_RING),
        }

    def _ground_contacts(self, pts: np.ndarray, body: np.ndarray):
        q = np.clip(pts, GROUND_MIN, GROUND_MAX)
        d = pts - q
        dist = np.linalg.norm(d, axis=1)
        nvec = np.zeros_like(pts)
        pen = np.full(pts.shape[0], -1e9)
        outside = dist > 1e-9
        if outside.any():
            nvec[outside] = -d[outside] / dist[outside, None]
            pen[outside] = SPHERE_RADIUS - dist[outside]
        inside = ~outside
        if inside.any():
            p = pts[inside]
            lo = p - GROUND_MIN
            hi = GROUND_MAX - p
            face = np.minimum(lo, hi)
            ax = np.argmin(face, axis=1)
            rows = np.arange(p.shape[0])
            outward = np.zeros_like(p)
            outward[rows, ax] = np.where(lo[rows, ax] <= hi[rows, ax], -1.0, 1.0)
            nvec[inside] = -outward
            pen[inside] = SPHERE_RADIUS + face[rows, ax]
            q[inside] = p
        keep = pen > -NEAR_CONTACT_SKIN
        if not keep.any():
            return None
        count = int(np.count_nonzero(keep))
        return {
            "ci": body[keep],
            "cj": np.full(count, self.n, dtype=np.int64),
            "nvec": nvec[keep],
            "pen": pen[keep],
            "point": q[keep],
            "mu": np.full(count, MU_GROUND),
        }

    def _pole_contacts(self, pts: np.ndarray, body: np.ndarray):
        outs = []
        for cx, cy in POLE_CENTERS:
            rel = pts[:, :2] - np.array([cx, cy])
            rad = np.linalg.norm(rel, axis=1)
            safe = np.maximum(rad, 1e-9)
            radial = rel / safe[:, None]
            radial[rad < 1e-9] = np.array([1.0, 0.0])
            qx = np.array([cx, cy]) + radial * np.minimum(rad, POLE_RADIUS)[:, None]
            qz = np.clip(pts[:, 2], 0.0, POLE_HEIGHT)
            q = np.stack([qx[:, 0], qx[:, 1], qz], axis=1)
            d = pts - q
            dist = np.linalg.norm(d, axis=1)
            nvec = np.zeros_like(pts)
            pen = np.full(pts.shape[0], -1e9)
            outside = dist > 1e-9
            if outside.any():
                nvec[outside] = -d[outside] / dist[outside, None]
                pen[outside] = SPHERE_RADIUS - dist[outside]
            inside = ~outside
            if inside.any():
                nvec[inside] = -radial[inside]
                pen[inside] = POLE_RADIUS + SPHERE_RADIUS - rad[inside]
                q[inside] = pts[inside]
            keep = pen > -NEAR_CONTACT_SKIN
            if not keep.any():
                continue
            count = int(np.count_nonzero(keep))
            outs.append(
                {
                    "ci": body[keep],
                    "cj": np.full(count, self.n, dtype=np.int64),
                    "nvec": nvec[keep],
                    "pen": pen[keep],
                    "point": q[keep],
                    "mu": np.full(count, MU_POLE),
                }
            )
        if not outs:
            return None
        return {k: np.concatenate([o[k] for o in outs]) for k in CONTACT_KEYS}

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _solve_velocity(self, c: dict, dt: float):
        ci = c["ci"]
        if ci.size == 0:
            return None
        cj = c["cj"]
        nvec, ri, rj = c["nvec"], c["ri"], c["rj"]
        pen, mu = c["pen"], c["mu"]

        kn = np.maximum(self._eff_mass(ci, cj, ri, rj, nvec), 1e-9)
        t1, t2 = _tangent_basis(nvec)
        k1 = np.maximum(self._eff_mass(ci, cj, ri, rj, t1), 1e-9)
        k2 = np.maximum(self._eff_mass(ci, cj, ri, rj, t2), 1e-9)
        bias = np.clip(
            BAUMGARTE * np.maximum(pen - POSITION_SLOP, 0.0) / dt, 0.0, MAX_BIAS_VELOCITY
        )
        pn = np.zeros_like(pen)
        pt1 = np.zeros_like(pen)
        pt2 = np.zeros_like(pen)

        for _ in range(SOLVER_ITERATIONS):
            vrel = (self.v[cj] + np.cross(self.w[cj], rj)) - (
                self.v[ci] + np.cross(self.w[ci], ri)
            )
            vn = np.einsum("ij,ij->i", vrel, nvec)
            lam = (bias - vn) / kn
            new_pn = np.maximum(pn + lam, 0.0)
            dlam = new_pn - pn
            pn = new_pn

            lim = mu * pn
            vt1 = np.einsum("ij,ij->i", vrel, t1)
            new_t1 = np.clip(pt1 - vt1 / k1, -lim, lim)
            d1 = new_t1 - pt1
            pt1 = new_t1
            vt2 = np.einsum("ij,ij->i", vrel, t2)
            new_t2 = np.clip(pt2 - vt2 / k2, -lim, lim)
            d2 = new_t2 - pt2
            pt2 = new_t2

            imp = dlam[:, None] * nvec + d1[:, None] * t1 + d2[:, None] * t2
            self._apply_impulse(ci, ri, cj, rj, imp)

        return {"pn": pn, "pt1": pt1, "pt2": pt2, "nvec": nvec, "t1": t1, "t2": t2}

    def _project_positions(self, c: dict) -> None:
        ci = c["ci"]
        if ci.size == 0:
            return
        cj = c["cj"]
        nvec, pen = c["nvec"], c["pen"]
        x0i, x0j = c["x0i"], c["x0j"]
        wi = self.inv_m[ci]
        wj = self.inv_m[cj]
        tot = wi + wj
        safe_tot = np.where(tot > 0.0, tot, 1.0)
        for _ in range(POSITION_ITERATIONS):
            di = self.x[ci] - x0i
            dj = self.x[cj] - x0j
            now = pen + np.einsum("ij,ij->i", di - dj, nvec)
            corr = np.clip(now - POSITION_SLOP, 0.0, MAX_POSITION_CORRECTION)
            corr = corr * POSITION_RELAXATION
            scale_i = np.where(tot > 0.0, corr * wi / safe_tot, 0.0)
            scale_j = np.where(tot > 0.0, corr * wj / safe_tot, 0.0)
            np.add.at(self.x, ci, -scale_i[:, None] * nvec)
            np.add.at(self.x, cj, scale_j[:, None] * nvec)

    def _integrate_orientation(self, dt: float) -> None:
        q = self.q
        w = self.w
        qv = q[:, :3]
        qw = q[:, 3:4]
        dw = -np.einsum("ij,ij->i", w, qv)
        dv = qw * w + np.cross(w, qv)
        q[:, :3] += 0.5 * dt * dv
        q[:, 3] += 0.5 * dt * dw
        norms = np.linalg.norm(q, axis=1)
        q /= np.where(norms > 1e-12, norms, 1.0)[:, None]

    def _step(self, dt: float) -> None:
        n = self.n
        self.v[:n] += GRAVITY * dt
        if LINEAR_DAMPING > 0.0:
            self.v[:n] *= max(0.0, 1.0 - LINEAR_DAMPING * dt)
        if ANGULAR_DAMPING > 0.0:
            self.w[:n] *= max(0.0, 1.0 - ANGULAR_DAMPING * dt)

        self._prepare_ext()
        contacts = self._build_contacts()
        self._last_impulses = self._solve_velocity(contacts, dt)

        self.x[:n] += self.v[:n] * dt
        self._integrate_orientation(dt)
        self._project_positions(contacts)
        self._last_contacts = contacts

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        started = _time.perf_counter()
        target = int(round(float(time) / SUB_DT))
        if target > self.steps_done:
            for _ in range(target - self.steps_done):
                self._step(SUB_DT)
            self.steps_done = target
        frame = self._make_frame(float(time))
        self._frame_seconds.append(_time.perf_counter() - started)
        return frame

    def _make_frame(self, time: float):
        n = self.n
        contacts = self._last_contacts
        pen = contacts["pen"] if contacts is not None else np.zeros(0)
        signed = -np.asarray(pen, dtype=float)
        impulses = self._last_impulses
        if impulses is not None and signed.size:
            force = (
                impulses["pn"][:, None] * impulses["nvec"]
                + impulses["pt1"][:, None] * impulses["t1"]
                + impulses["pt2"][:, None] * impulses["t2"]
            ) / SUB_DT
        else:
            force = np.zeros((signed.size, 3))

        external = np.zeros((n, 3))
        external[:, 2] = -9.81 * self.mass

        if signed.size:
            self._pen_counts.append(int(np.count_nonzero(signed < 0.0)))
            self._min_signed.append(float(signed.min()))
            self._max_pen = max(self._max_pen, float(-signed.min()))
        else:
            self._pen_counts.append(0)
            self._min_signed.append(0.0)

        rot = _quat_to_matrix(self.q[:n])
        origin_position = self.x[:n] + np.einsum("nij,j->ni", rot, -BODY_OFFSET)

        return StateFrame(
            time=float(time),
            position=origin_position,
            velocity=self.v[:n].copy(),
            orientation=self.q[:n].copy(),
            angular_velocity=self.w[:n].copy(),
            mass=self.mass.copy(),
            body_id=np.arange(n, dtype=np.int64),
            fields={
                "contact_signed_distance": signed,
                "external_force": external,
                "interaction_force": force,
            },
            scalars={"contact_length_scale": float(CONTACT_LENGTH_SCALE)},
        )

    def diagnostics(self) -> dict[str, object]:
        n = self.n
        if n == 0:
            return {}
        speed2 = np.einsum("ij,ij->i", self.v[:n], self.v[:n])
        ke_lin = float(0.5 * np.sum(self.mass * speed2))
        times = self._frame_seconds
        return {
            "seed": int(getattr(self, "_seed", 0)),
            "bodies": int(n),
            "sub_step": float(SUB_DT),
            "steps": int(self.steps_done),
            "settled_height": float(np.max(self.x[:n, 2]) - BODY_OFFSET[2]),
            "pile_height_mean": float(np.mean(self.x[:n, 2]) - BODY_OFFSET[2]),
            "residual_kinetic_energy": ke_lin,
            "max_penetration": float(self._max_pen),
            "final_penetration_count": int(self._pen_counts[-1]) if self._pen_counts else 0,
            "max_penetration_count": int(max(self._pen_counts)) if self._pen_counts else 0,
            "min_signed_distance": float(min(self._min_signed)) if self._min_signed else 0.0,
            "frames": len(times),
            "mean_frame_seconds": float(np.mean(times)) if times else 0.0,
            "max_frame_seconds": float(np.max(times)) if times else 0.0,
        }

