"""Physically-based solver for the knitted legwarmer deformation scene.

Physical model
--------------
The garment is a tubular knit sleeve discretised as a 17x17 triangular
membrane mesh (289 particle samples, 544 elements, periodic around the
tube).  Every element carries a genuine finite-strain deformation gradient
built from a global material parameterisation of the tube surface; the
per-element rest metric is measured once from the initial configuration, so
the relative right Cauchy-Green tensor is exactly the identity at rest.

Membrane response is a 2-D finite-strain hyperelastic (neo-Hookean) energy

    W = mu/2 (tr C - 2) - mu ln J + lambda/2 (ln J)^2,   J = sqrt(det C)

evaluated on the relative strain, whose first Piola stress P = F S gives
the exact per-element nodal forces.  A rest-length diagonal (hinge) spring
per quad supplies bending stiffness, and a global viscous term damps the
sheet.

The articulated leg is a set of kinematic colliders (two capsules, a knee
sphere, a foot capsule and a ground box) reconstructed from the frozen
scene geometry.  The lower limb and the foot are rigidly rotated about the
knee joint by a smooth C1 bend profile, so the sleeve is driven by real
kinematic contact motion: the inside of the joint compresses the sleeve
while the outside stretches.  Contact uses a penalty normal law plus
regularised Coulomb friction, i.e. the yarn friction that carries the
sleeve with the leg.

Integration is explicit symplectic Euler with a fixed deterministic
substep dt = sample_dt / 120, well inside the CFL limit of the mesh
(c dt / h ~ 0.16).  Nothing is scripted: every sample is obtained by
stepping the equations of motion from the previous sample.

Exported deformation gradient
-----------------------------
``attributes.deformation_gradient`` is the per-sample right stretch tensor
U = sqrt(C) of the area-weighted local material map, expressed in material
coordinates (2x2).  Its principal values are the principal stretches and
det(U) is the local area ratio; it is exactly the identity in the rest
state.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81])

# --- sleeve placement / discretisation (frozen configuration) -------------
_SLEEVE_RADIUS = 0.42
_SLEEVE_CENTER = np.array([0.0, 0.0, 1.10])

# --- material parameters (2-D moduli [N/m] for a light knit fabric) -------
_MU_2D = 80.0
_LAMBDA_2D = 160.0
_BEND_SPRING = 0.5
_DAMPING = 4.0          # global velocity damping [1/s]
_VELOCITY_LIMIT = 25.0  # numerical safeguard [m/s]
_KN = 4000.0            # contact penalty stiffness [N/m]
_CN = 0.20              # contact normal damping [N s/m]
_FRICTION = 0.9         # Coulomb yarn/collider friction coefficient
_CLEARANCE = 0.002      # initial separation between sleeve and colliders [m]

# --- driven bending profile ----------------------------------------------
_T_START, _T_FULL, _T_HOLD, _T_END = 0.6, 3.0, 4.2, 6.5
_A_MAX = 0.70           # peak joint flexion [rad]
_SUBSTEPS = 120

_KNEE_DEFAULT = np.array([0.0, 0.0, 1.10])
_ROTATING_IDS = ("limb_lower", "foot")

# Geometry fallback (mirrors the frozen scene_geometry.yaml).
_FALLBACK_OBJECTS = [
    ("ground", "box", None, None, [0.0, 0.0, -0.05], [2.4, 2.0, 0.05], None),
    ("limb_upper", "capsule", 0.34, 1.3, [-0.62, 0.0, 1.10], None,
     [0.0, 0.7071068, 0.0, 0.7071068]),
    ("limb_lower", "capsule", 0.31, 1.3, [0.62, 0.0, 1.10], None,
     [0.0, 0.7071068, 0.0, 0.7071068]),
    ("knee_joint", "sphere", 0.36, None, [0.0, 0.0, 1.10], None,
     [0.0, 0.0, 0.0, 1.0]),
    ("foot", "capsule", 0.25, 0.72, [1.55, -0.1, 0.92], None,
     [0.0, 0.7071068, 0.0, 0.7071068]),
]


def _smoothstep(u: float) -> float:
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_d(u: float, span: float) -> float:
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return 6.0 * u * (1.0 - u) / span


def _bend_drive(t: float) -> tuple[float, float]:
    """Return (flexion angle, angular rate) of the lower limb about the knee."""
    if t <= _T_START:
        return 0.0, 0.0
    if t < _T_FULL:
        u = (t - _T_START) / (_T_FULL - _T_START)
        return _A_MAX * _smoothstep(u), _A_MAX * _smoothstep_d(u, _T_FULL - _T_START)
    if t <= _T_HOLD:
        return _A_MAX, 0.0
    if t < _T_END:
        u = (t - _T_HOLD) / (_T_END - _T_HOLD)
        return _A_MAX * (1.0 - _smoothstep(u)), -_A_MAX * _smoothstep_d(u, _T_END - _T_HOLD)
    return 0.0, 0.0


def _quat_rotate(q, v):
    x, y, z, w = (float(c) for c in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.asarray(v, dtype=float).copy()
    x, y, z, w = x / n, y / n, z / n, w / n
    qv = np.array([x, y, z])
    u = np.asarray(v, dtype=float)
    return u + 2.0 * np.cross(qv, np.cross(qv, u) + w * u)


def _sym_sqrt_inv(C):
    """Inverse square root of a batched symmetric positive definite 2x2 array."""
    det = np.maximum(C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0], 1e-12)
    sq = np.sqrt(det)
    den = sq * np.sqrt(np.maximum(C[:, 0, 0] + C[:, 1, 1] + 2.0 * sq, 1e-12))
    A = np.empty_like(C)
    A[:, 0, 0] = (C[:, 0, 0] + sq) / den
    A[:, 1, 1] = (C[:, 1, 1] + sq) / den
    A[:, 0, 1] = C[:, 0, 1] / den
    A[:, 1, 0] = C[:, 1, 0] / den
    return A


def _sym_sqrt(C):
    """Square root of a batched symmetric positive definite 2x2 array."""
    det = np.maximum(C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0], 1e-12)
    sq = np.sqrt(det)
    den = np.sqrt(np.maximum(C[:, 0, 0] + C[:, 1, 1] + 2.0 * sq, 1e-12))
    U = np.empty_like(C)
    U[:, 0, 0] = (C[:, 0, 0] + sq) / den
    U[:, 1, 1] = (C[:, 1, 1] + sq) / den
    U[:, 0, 1] = C[:, 0, 1] / den
    U[:, 1, 0] = C[:, 1, 0] / den
    return U


class BenchmarkSolver:
    """Deterministic finite-strain membrane solver with kinematic contact."""

    # ------------------------------------------------------------- lifecycle
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}) or {})
        grid = list(cloth.get("grid_shape", [17, 17]))
        self._n_ang = int(grid[0])
        self._n_ax = int(grid[1])
        self._N = self._n_ax * self._n_ang
        sizes = list(cloth.get("size", [3.0, 2.5]))
        center = np.asarray(cloth.get("center", _SLEEVE_CENTER), dtype=float)
        radius = _SLEEVE_RADIUS
        self._thickness = float(cloth.get("thickness", 0.015))
        total_mass = float(cloth.get("total_mass", 0.32))

        self._build_colliders(cfg)

        # --- tube collar positions (scaffold-compatible ordering) ---------
        ax = np.linspace(-0.5 * float(sizes[0]), 0.5 * float(sizes[0]), self._n_ax)
        angs = np.linspace(0.0, 2.0 * np.pi, self._n_ang, endpoint=False)
        pos = np.asarray(
            [[x, radius * math.cos(a), radius * math.sin(a)] for x in ax for a in angs],
            dtype=float,
        ) + center[None, :]
        pos = self._resolve_overlaps(pos)

        # --- global material parameterisation -----------------------------
        du = float(sizes[0]) / max(self._n_ax - 1, 1)
        dv = 2.0 * radius * math.sin(math.pi / self._n_ang)
        mat = np.asarray(
            [[i * du, j * dv] for i in range(self._n_ax) for j in range(self._n_ang)],
            dtype=float,
        )

        ids = np.arange(self._N).reshape(self._n_ax, self._n_ang)
        jm = (np.arange(self._n_ang) + 1) % self._n_ang
        i0 = np.arange(self._n_ax - 1)[:, None]
        a = ids[i0, :].reshape(-1)
        b = ids[i0, jm].reshape(-1)
        c = ids[i0 + 1, jm].reshape(-1)
        d = ids[i0 + 1, :].reshape(-1)
        tri = np.concatenate(
            [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
        ).astype(np.int64)
        self._tri = tri

        M = np.stack([mat[b] - mat[a], mat[c] - mat[a]], axis=2)
        self._Minv = np.linalg.inv(M)

        xa = pos[a]
        X0 = np.stack([pos[b] - xa, pos[c] - xa], axis=2)          # (T,3,2)
        self._A0 = 0.5 * np.linalg.norm(np.cross(X0[:, :, 0], X0[:, :, 1]), axis=1)
        F0 = np.einsum("tkm,tml->tkl", X0, self._Minv)
        C0 = np.einsum("tkl,tkm->tlm", F0, F0)
        self._Asqrt = _sym_sqrt_inv(C0)                            # (T,2,2)

        # hinge/bending springs along the quad diagonals
        self._sa = b.copy()
        self._sb = d.copy()
        self._L0 = np.linalg.norm(pos[d] - pos[b], axis=1)

        self._mass = np.full(self._N, total_mass / self._N, dtype=float)
        self._x = pos.copy()
        self._v = np.zeros_like(pos)
        self._boundary = np.zeros(self._N, dtype=np.int32)
        self._boundary[: self._n_ang] = 1
        self._boundary[(self._n_ax - 1) * self._n_ang :] = 1

        self._dt = float(cfg.get("sample_dt", 1.0 / 30.0)) / _SUBSTEPS
        self._damp = math.exp(-_DAMPING * self._dt)
        self._t = 0.0
        self._steps = 0
        self._last_min_sd = float("inf")

    def _build_colliders(self, cfg: dict) -> None:
        raw = []
        geo = cfg.get("scene_geometry") or {}
        for item in geo.get("objects", []) or []:
            gtype = (item.get("geometry") or {}).get("type")
            if gtype not in ("capsule", "sphere", "box"):
                continue
            if item.get("semantic") == "cloth" or item.get("id") == "cloth":
                continue
            if not bool(item.get("collision_enabled", True)):
                continue
            raw.append(item)

        self._knee = None
        prims = []
        if raw:
            for item in raw:
                g = item["geometry"]
                tr = item.get("transform") or {}
                t = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                q = tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
                oid = str(item.get("id", ""))
                rot = oid in _ROTATING_IDS
                if gtype == "capsule":
                    axis = _quat_rotate(q, [0.0, 0.0, 1.0])
                    half = 0.5 * float(g["height"])
                    prims.append(("capsule", t - half * axis, t + half * axis,
                                  float(g["radius"]), rot))
                elif gtype == "sphere":
                    if oid == "knee_joint":
                        self._knee = t.copy()
                    prims.append(("sphere", t, t, float(g["radius"]), rot))
                else:
                    prims.append(("box", t, np.asarray(g["half_extents"], dtype=float),
                                  0.0, False))
        else:
            for oid, gtype, r, h, t, he, q in _FALLBACK_OBJECTS:
                t = np.asarray(t, dtype=float)
                rot = oid in _ROTATING_IDS
                if gtype == "capsule":
                    axis = _quat_rotate(q, [0.0, 0.0, 1.0])
                    half = 0.5 * float(h)
                    prims.append(("capsule", t - half * axis, t + half * axis,
                                  float(r), rot))
                elif gtype == "sphere":
                    if oid == "knee_joint":
                        self._knee = t.copy()
                    prims.append(("sphere", t, t, float(r), rot))
                else:
                    prims.append(("box", t, np.asarray(he, dtype=float), 0.0, False))
        if self._knee is None:
            self._knee = _KNEE_DEFAULT.copy()
        self._prims = prims

    def _resolve_overlaps(self, pos):
        """Push sleeve samples out of the frozen colliders before time zero."""
        p = pos.copy()
        ang = 0.0
        for _ in range(60):
            sd, cp, _ = self._proximity(p, ang, 0.0)
            gap = _CLEARANCE - sd
            bad = gap > 0.0
            if not np.any(bad):
                break
            delta = p - cp
            dist = np.linalg.norm(delta, axis=1)
            safe = np.maximum(dist, 1e-9)
            nrm = delta / safe[:, None]
            nrm[dist < 1e-9] = np.array([0.0, 0.0, 1.0])
            step = np.where(bad, gap, 0.0)
            p = p + step[:, None] * nrm
        return p

    # ------------------------------------------------------------- kinematics
    def _hinge(self, p, ca, sa):
        r = np.asarray(p, dtype=float) - self._knee
        return self._knee + np.array([r[0] * ca - r[2] * sa, r[1], r[0] * sa + r[2] * ca])

    def _proximity(self, x, ang, dang):
        """Nearest-collider signed gap, closest point and surface velocity."""
        ca, sa = math.cos(ang), math.sin(ang)
        n = x.shape[0]
        zero = np.zeros((n, 3))
        sd = np.full(n, np.inf)
        cp = np.zeros((n, 3))
        vs = np.zeros((n, 3))
        for prim in self._prims:
            kind = prim[0]
            if kind == "capsule":
                _, a0, b0, r, rot = prim
                if rot:
                    a, b = self._hinge(a0, ca, sa), self._hinge(b0, ca, sa)
                else:
                    a, b = a0, b0
                seg = b - a
                dd = float(seg @ seg)
                if dd < 1e-12:
                    c = np.repeat(a[None, :], n, axis=0)
                else:
                    tt = np.clip(((x - a) @ seg) / dd, 0.0, 1.0)
                    c = a[None, :] + tt[:, None] * seg[None, :]
                sdp = np.linalg.norm(x - c, axis=1) - r
                if rot:
                    rr = c - self._knee
                    vsp = np.stack([-dang * rr[:, 2], np.zeros(n), dang * rr[:, 0]], axis=1)
                else:
                    vsp = zero
            elif kind == "sphere":
                _, c0, _, r, rot = prim
                c = np.repeat(np.asarray(c0, dtype=float)[None, :], n, axis=0)
                sdp = np.linalg.norm(x - c, axis=1) - r
                vsp = zero
            else:
                _, ctr, he, _, _ = prim
                q = np.abs(x - ctr) - he
                outside = np.maximum(q, 0.0)
                sdp = np.linalg.norm(outside, axis=1) + np.minimum(np.max(q, axis=1), 0.0)
                c = np.clip(x, ctr - he, ctr + he)
                vsp = zero
            better = sdp < sd
            if np.any(better):
                sd = np.where(better, sdp, sd)
                cp = np.where(better[:, None], c, cp)
                vs = np.where(better[:, None], vsp, vs)
        return sd, cp, vs

    # ---------------------------------------------------------------- forces
    def _membrane_forces(self, x, f):
        tri = self._tri
        xa = x[tri[:, 0]]
        X = np.stack([x[tri[:, 1]] - xa, x[tri[:, 2]] - xa], axis=2)     # (T,3,2)
        F = np.einsum("tkm,tml->tkl", X, self._Minv)
        C = np.einsum("tkl,tkm->tlm", F, F)
        Cr = np.einsum("til,tlm,tmj->tij", self._Asqrt, C, self._Asqrt)
        c00, c01 = Cr[:, 0, 0], Cr[:, 0, 1]
        c10, c11 = Cr[:, 1, 0], Cr[:, 1, 1]
        det = np.maximum(c00 * c11 - c01 * c10, 1e-8)
        inv = 1.0 / det
        i00, i01 = c11 * inv, -c01 * inv
        i10, i11 = -c10 * inv, c00 * inv
        lnj = np.log(np.maximum(np.sqrt(det), 0.45))
        mu, lam = _MU_2D, _LAMBDA_2D
        r00 = mu * (1.0 - i00) + lam * lnj * i00
        r01 = mu * (-i01) + lam * lnj * i01
        r10 = mu * (-i10) + lam * lnj * i10
        r11 = mu * (1.0 - i11) + lam * lnj * i11
        # S = A S_rel A  (A symmetric)
        S = np.empty_like(Cr)
        A = self._Asqrt
        S[:, 0, 0] = (A[:, 0, 0] * (A[:, 0, 0] * r00 + A[:, 1, 0] * r10)
                      + A[:, 1, 0] * (A[:, 0, 0] * r01 + A[:, 1, 0] * r11))
        S[:, 1, 1] = (A[:, 0, 1] * (A[:, 0, 1] * r00 + A[:, 1, 1] * r10)
                      + A[:, 1, 1] * (A[:, 0, 1] * r01 + A[:, 1, 1] * r11))
        S[:, 0, 1] = (A[:, 0, 0] * (A[:, 0, 1] * r00 + A[:, 1, 1] * r10)
                      + A[:, 1, 0] * (A[:, 0, 1] * r01 + A[:, 1, 1] * r11))
        S[:, 1, 0] = S[:, 0, 1]
        P = np.empty_like(F)
        P[:, :, 0] = F[:, :, 0] * S[:, 0, 0][:, None] + F[:, :, 1] * S[:, 1, 0][:, None]
        P[:, :, 1] = F[:, :, 0] * S[:, 0, 1][:, None] + F[:, :, 1] * S[:, 1, 1][:, None]
        fb = -self._A0[:, None] * np.einsum("tkl,tl->tk", P, self._Minv[:, 1, :])
        fc = -self._A0[:, None] * np.einsum("tkl,tl->tk", P, self._Minv[:, 0, :])
        fa = -(fb + fc)
        idx = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        w = np.concatenate([fa, fb, fc], axis=0)
        for k in range(3):
            f[:, k] += np.bincount(idx, weights=w[:, k], minlength=self._N)

    def _bending_forces(self, x, f):
        dx = x[self._sb] - x[self._sa]
        L = np.maximum(np.linalg.norm(dx, axis=1), 1e-9)
        fs = _BEND_SPRING * ((L - self._L0) / L)[:, None] * dx
        for k in range(3):
            f[:, k] += np.bincount(self._sa, weights=fs[:, k], minlength=self._N)
            f[:, k] -= np.bincount(self._sb, weights=fs[:, k], minlength=self._N)

    def _contact_forces(self, x, v, ang, dang, f):
        sd, cp, vs = self._proximity(x, ang, dang)
        delta = x - cp
        dist = np.linalg.norm(delta, axis=1)
        pen = -sd
        act = pen > 0.0
        if np.any(act):
            safe = np.maximum(dist, 1e-9)
            nrm = delta / safe[:, None]
            degenerate = dist < 1e-9
            if np.any(degenerate):
                nrm[degenerate] = np.array([0.0, 0.0, 1.0])
            vrel = v - vs
            vn = np.einsum("ij,ij->i", vrel, nrm)
            fn = np.where(act, np.maximum(_KN * pen - _CN * np.minimum(vn, 0.0), 0.0), 0.0)
            vt = vrel - vn[:, None] * nrm
            vtm = np.linalg.norm(vt, axis=1)
            scale = np.where(act, _FRICTION * fn / np.maximum(vtm, 3e-3), 0.0)
            f += fn[:, None] * nrm - scale[:, None] * vt

    # -------------------------------------------------------------- stepping
    def _step(self):
        dt = self._dt
        x, v = self._x, self._v
        ang, dang = _bend_drive(self._t + 0.5 * dt)
        f = self._mass[:, None] * _GRAVITY
        self._membrane_forces(x, f)
        self._bending_forces(x, f)
        self._contact_forces(x, v, ang, dang, f)
        v += (f / self._mass[:, None]) * dt
        speed = np.linalg.norm(v, axis=1)
        too_fast = speed > _VELOCITY_LIMIT
        if np.any(too_fast):
            v[too_fast] *= (_VELOCITY_LIMIT / speed[too_fast])[:, None]
        v *= self._damp
        x += v * dt
        self._t += dt
        self._steps += 1

    # ---------------------------------------------------------------- output
    def _vertex_stretch(self, x):
        tri = self._tri
        xa = x[tri[:, 0]]
        X = np.stack([x[tri[:, 1]] - xa, x[tri[:, 2]] - xa], axis=2)
        F = np.einsum("tkm,tml->tkl", X, self._Minv)
        C = np.einsum("tkl,tkm->tlm", F, F)
        Cr = np.einsum("til,tlm,tmj->tij", self._Asqrt, C, self._Asqrt)
        num = np.zeros((self._N, 2, 2))
        idx = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        cw = np.concatenate([Cr * self._A0[:, None, None]] * 3, axis=0)
        for i in range(2):
            for j in range(2):
                num[:, i, j] = np.bincount(idx, weights=cw[:, i, j], minlength=self._N)
        den = np.maximum(
            np.bincount(idx, weights=np.concatenate([self._A0] * 3), minlength=self._N),
            1e-12,
        )
        return _sym_sqrt(num / den[:, None, None])

    def _make_frame(self) -> StateFrame:
        ang, _ = _bend_drive(self._t)
        sd, _, _ = self._proximity(self._x, ang, 0.0)
        self._last_min_sd = float(np.min(sd))
        frame = StateFrame(
            float(self._t),
            self._x.copy(),
            self._v.copy(),
            mass=self._mass.copy(),
            body_id=np.zeros(self._N, dtype=np.int32),
            attributes={
                "mass_quantity": self._mass.copy(),
                "deformation_gradient": self._vertex_stretch(self._x),
                "boundary_vertex": self._boundary.copy(),
            },
            fields={
                "contact_signed_distance": sd,
                "external_force": self._mass[:, None] * _GRAVITY,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self._thickness)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        dt = self._dt
        steps = int(round((target - self._t) / dt))
        if steps < 0:
            steps = 0
        for _ in range(steps):
            self._step()
        self._t = self._steps * dt
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps_per_sample": _SUBSTEPS,
            "dt": float(self._dt),
            "steps_taken": int(self._steps),
            "sample_count": int(self._N),
            "element_count": int(self._tri.shape[0]),
            "min_signed_distance": float(self._last_min_sd),
            "mu_2d": _MU_2D,
            "lambda_2d": _LAMBDA_2D,
            "friction": _FRICTION,
        }
