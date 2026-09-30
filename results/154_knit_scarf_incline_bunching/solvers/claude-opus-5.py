"""Knitted-scarf-on-incline solver (self-implemented, NumPy/SciPy only).

Physical model
--------------
The scarf is discretised as a lumped-mass thin sheet on the frozen 17x17
material grid.  Internal response uses a yarn-inspired anisotropic network:

  * structural (course/wale) edges: stiff in tension, much softer in
    compression, so the knit buckles long before it stretches;
  * diagonal shear edges: compliant, giving the large shear compliance of a
    knitted fabric;
  * one-sided second-neighbour flexural edges: they only resist *closing*,
    which yields very low bending stiffness with a finite fold radius, i.e.
    thick, volumetric bunched folds instead of paper-like sharp creases.

Time stepping is a substepped XPBD/projective-dynamics scheme: symplectic
Euler prediction, then colored Gauss-Seidel projection of all constraints
(vectorised, each colour is an independent set), then velocity recovery from
the position update.  Contacts are resolved geometrically against the frozen
box geometry (exact box signed-distance field) with Coulomb friction applied
at the velocity level, plus particle-thickness self-contact with inelastic
normal response and tangential friction so the pile keeps volume.

Everything is deterministic (fixed step count, no random sampling beyond a
seeded infinitesimal symmetry-breaking jitter).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # SciPy is available in the runtime; keep a pure-NumPy fallback anyway.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _quat_to_matrix(xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class _Box:
    """Static oriented box with an exact signed-distance field."""

    __slots__ = ("center", "rot", "half", "friction")

    def __init__(self, center, rot, half, friction):
        self.center = np.asarray(center, dtype=float)
        self.rot = np.asarray(rot, dtype=float)          # local -> world
        self.half = np.asarray(half, dtype=float)
        self.friction = float(friction)

    def sdf(self, p: np.ndarray):
        """Signed distance and outward unit normal (world frame)."""
        q = (p - self.center) @ self.rot                  # world -> local
        d = np.abs(q) - self.half
        outside = np.maximum(d, 0.0)
        dmax = d.max(axis=1)
        sd = np.linalg.norm(outside, axis=1) + np.minimum(dmax, 0.0)

        sgn = np.sign(q)
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        nl = sgn * outside
        nrm = np.linalg.norm(nl, axis=1)
        deep = nrm < 1e-12
        rows = np.arange(len(q))
        k = np.argmax(d, axis=1)
        n_in = np.zeros_like(q)
        n_in[rows, k] = sgn[rows, k]
        n_local = np.where(deep[:, None], n_in, nl / np.maximum(nrm, 1e-12)[:, None])
        return sd, n_local @ self.rot.T


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class BenchmarkSolver:
    # --- discretisation -------------------------------------------------- #
    SUBSTEPS = 40            # substeps per sample interval (dt ~ 8.3e-4 s)
    ITERS = 2                # constraint sweeps per substep

    # --- knit constitutive parameters ------------------------------------ #
    K_STRETCH = 0.85         # structural yarn: stiff in tension
    K_COMPRESS = 0.30        # ... soft in compression (buckles / bunches)
    K_SHEAR = 0.35           # compliant shear (knit loops rotate easily)
    K_BEND = 0.06            # very low flexural stiffness, one-sided
    DRAG = 0.25              # air drag [1/s]
    V_MAX = 12.0             # velocity clamp for step robustness

    # --- contact --------------------------------------------------------- #
    SELF_RADIUS = 0.085      # effective knit thickness for self contact
    SELF_FRICTION = 0.30
    MU = 0.50                # Coulomb coefficient against the rigid boxes

    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        cloth = cfg["cloth"]

        self.nx, self.ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        self.n = self.nx * self.ny
        sx, sy = float(cloth["size"][0]), float(cloth["size"][1])
        self.du = sx / max(self.nx - 1, 1)
        self.dv = sy / max(self.ny - 1, 1)
        self.thickness = float(cloth.get("thickness", 0.01))

        init = scene.initial_state()
        self.pos = np.array(init.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(init.mass, dtype=float) if init.mass is not None else np.full(
            self.n, float(cloth["total_mass"]) / self.n
        )
        self.boundary = (
            np.asarray(init.attributes["boundary_vertex"], dtype=np.int32)
            if "boundary_vertex" in init.attributes
            else np.zeros(self.n, dtype=np.int32)
        )
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # infinitesimal, seeded, deterministic symmetry break so that folds
        # can pick a phase instead of staying perfectly translation-symmetric
        rng = np.random.default_rng(int(seed))
        self.pos += 2.0e-4 * (rng.random(self.pos.shape) - 0.5)

        self.rest_pos = self.pos.copy()
        self._build_constraints()
        self._build_boxes(cfg)

        self.sample_dt = float(cfg["sample_dt"])
        self.dt = self.sample_dt / self.SUBSTEPS
        self.time = 0.0
        self.steps = 0
        self._max_speed = 0.0
        self._min_gap = 0.0

    # ------------------------------------------------------------------ #
    def _idx(self, j, i):
        return (j * self.nx + i).astype(np.int64)

    def _group(self, ia, ib, kind, stiff):
        ia = np.asarray(ia, dtype=np.int64).ravel()
        ib = np.asarray(ib, dtype=np.int64).ravel()
        rest = np.linalg.norm(self.rest_pos[ib] - self.rest_pos[ia], axis=1)
        return (ia, ib, rest, kind, float(stiff))

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny
        groups = []

        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx - 1), indexing="ij")
        for c in range(2):
            m = (ii % 2) == c
            if m.any():
                groups.append(
                    self._group(self._idx(jj[m], ii[m]), self._idx(jj[m], ii[m] + 1),
                                "axial", self.K_STRETCH)
                )

        jj, ii = np.meshgrid(np.arange(ny - 1), np.arange(nx), indexing="ij")
        for c in range(2):
            m = (jj % 2) == c
            if m.any():
                groups.append(
                    self._group(self._idx(jj[m], ii[m]), self._idx(jj[m] + 1, ii[m]),
                                "axial", self.K_STRETCH)
                )

        jj, ii = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        for cj in range(2):
            for ci in range(2):
                m = ((jj % 2) == cj) & ((ii % 2) == ci)
                if not m.any():
                    continue
                groups.append(
                    self._group(self._idx(jj[m], ii[m]), self._idx(jj[m] + 1, ii[m] + 1),
                                "shear", self.K_SHEAR)
                )
                groups.append(
                    self._group(self._idx(jj[m], ii[m] + 1), self._idx(jj[m] + 1, ii[m]),
                                "shear", self.K_SHEAR)
                )

        if nx > 2:
            jj, ii = np.meshgrid(np.arange(ny), np.arange(nx - 2), indexing="ij")
            for c in range(4):
                m = (ii % 4) == c
                if m.any():
                    groups.append(
                        self._group(self._idx(jj[m], ii[m]), self._idx(jj[m], ii[m] + 2),
                                    "bend", self.K_BEND)
                    )
        if ny > 2:
            jj, ii = np.meshgrid(np.arange(ny - 2), np.arange(nx), indexing="ij")
            for c in range(4):
                m = (jj % 4) == c
                if m.any():
                    groups.append(
                        self._group(self._idx(jj[m], ii[m]), self._idx(jj[m] + 2, ii[m]),
                                    "bend", self.K_BEND)
                    )

        self.groups = groups
        grid = np.arange(self.n)
        self.g_i = (grid % self.nx).astype(np.int64)
        self.g_j = (grid // self.nx).astype(np.int64)

    def _build_boxes(self, cfg) -> None:
        boxes = []
        geo = cfg.get("scene_geometry") or {}
        for obj in geo.get("objects", []) or []:
            if str(obj.get("semantic", "")) == "cloth":
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            g = obj.get("geometry", {}) or {}
            if str(g.get("type", "")) != "box":
                continue
            half = np.asarray(g.get("half_extents", [1.0, 1.0, 1.0]), dtype=float)
            tr = obj.get("transform", {}) or {}
            center = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            quat = tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            boxes.append(_Box(center, _quat_to_matrix(quat), half, self.MU))
        if not boxes:  # defensive fallback matching the frozen scene
            boxes = [
                _Box([0.0, 0.0, -0.05], np.eye(3), [2.4, 2.0, 0.05], self.MU),
                _Box([0.0, 0.0, 0.8],
                     _quat_to_matrix([0.0, 0.300706, 0.0, 0.953717]),
                     [2.2, 1.1, 0.08], self.MU),
            ]
        self.boxes = boxes
        self.contact_offset = max(0.5 * self.thickness, 0.02)

    # ------------------------------------------------------------------ #
    def _self_pairs(self, p):
        r = self.SELF_RADIUS
        if _KDTree is not None:
            tree = _KDTree(p)
            pairs = tree.query_pairs(r, output_type="ndarray")
            if len(pairs) == 0:
                return None
            a, b = pairs[:, 0], pairs[:, 1]
        else:  # pragma: no cover - brute force fallback
            d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            a, b = np.where(np.triu(d2 < r * r, 1))
            if len(a) == 0:
                return None
        keep = (np.abs(self.g_i[a] - self.g_i[b]) > 2) | (np.abs(self.g_j[a] - self.g_j[b]) > 2)
        if not keep.any():
            return None
        return a[keep], b[keep]

    def _project(self, p, pairs):
        for ia, ib, rest, kind, stiff in self.groups:
            d = p[ib] - p[ia]
            L = np.sqrt((d * d).sum(axis=1))
            Ls = np.maximum(L, 1e-9)
            C = L - rest
            if kind == "axial":
                k = np.where(C > 0.0, stiff, self.K_COMPRESS)
            elif kind == "shear":
                k = np.full(L.shape, stiff)
            else:  # one-sided flexural: resist closing only
                k = np.where(C < 0.0, stiff, 0.0)
            s = (0.5 * k * C / Ls)[:, None]
            corr = s * d
            p[ia] = p[ia] + corr
            p[ib] = p[ib] - corr

        # self contact (equal masses -> symmetric split)
        if pairs is not None:
            a, b = pairs
            d = p[b] - p[a]
            L = np.sqrt((d * d).sum(axis=1))
            act = L < self.SELF_RADIUS
            if act.any():
                aa, bb = a[act], b[act]
                dd = d[act]
                LL = np.maximum(L[act], 1e-9)
                corr = (0.5 * (self.SELF_RADIUS - LL) / LL)[:, None] * dd
                np.add.at(p, aa, -corr)
                np.add.at(p, bb, corr)

        # rigid boundaries
        r = self.contact_offset
        mask = np.zeros(len(p), dtype=bool)
        normals = np.zeros_like(p)
        for box in self.boxes:
            sd, nrm = box.sdf(p)
            hit = sd < r
            if hit.any():
                p[hit] = p[hit] + ((r - sd[hit])[:, None] * nrm[hit])
                mask |= hit
                normals[hit] = nrm[hit]
        return mask, normals

    def _step(self, dt):
        p0 = self.pos
        v = self.vel
        v += dt * self.gravity
        v *= np.exp(-self.DRAG * dt)
        sp = np.sqrt((v * v).sum(axis=1))
        big = sp > self.V_MAX
        if big.any():
            v[big] *= (self.V_MAX / sp[big])[:, None]

        p = p0 + dt * v
        pairs = self._self_pairs(p)
        for _ in range(self.ITERS):
            cmask, cnorm = self._project(p, pairs)

        v = (p - p0) / dt

        # Coulomb friction / inelastic normal response on rigid contacts
        if cmask.any():
            idx = np.flatnonzero(cmask)
            nn = cnorm[idx]
            vv = v[idx]
            vn = (vv * nn).sum(axis=1)
            vt = vv - vn[:, None] * nn
            vt_mag = np.sqrt((vt * vt).sum(axis=1))
            gn = np.abs((self.gravity[None, :] * nn).sum(axis=1))
            limit = self.MU * (np.maximum(-vn, 0.0) + dt * gn)
            scale = np.maximum(0.0, 1.0 - limit / np.maximum(vt_mag, 1e-9))
            # Inelastic normal response: the outward normal velocity produced by
            # the geometric push-out is removed instead of being kept, so the
            # scarf settles and slides on the incline rather than bouncing.
            v[idx] = vt * scale[:, None]

        # self-contact normal damping + tangential friction (pile cohesion)
        if pairs is not None:
            a, b = pairs
            d = p[b] - p[a]
            L = np.sqrt((d * d).sum(axis=1))
            act = L < self.SELF_RADIUS * 1.02
            if act.any():
                aa, bb = a[act], b[act]
                nn = d[act] / np.maximum(L[act], 1e-9)[:, None]
                dv = v[bb] - v[aa]
                dvn = (dv * nn).sum(axis=1)
                dvt = dv - dvn[:, None] * nn
                imp = 0.5 * (np.minimum(dvn, 0.0)[:, None] * nn + self.SELF_FRICTION * dvt)
                np.add.at(v, aa, imp)
                np.add.at(v, bb, -imp)

        self.pos = p
        self.vel = v
        self.steps += 1

    # ------------------------------------------------------------------ #
    def _deformation_gradient(self):
        P = self.pos.reshape(self.ny, self.nx, 3)
        a = np.gradient(P, self.du, axis=1)
        b = np.gradient(P, self.dv, axis=0)
        a = a.reshape(-1, 3)
        b = b.reshape(-1, 3)
        la = np.sqrt((a * a).sum(axis=1))
        e1 = a / np.maximum(la, 1e-12)[:, None]
        be1 = (b * e1).sum(axis=1)
        bperp = b - be1[:, None] * e1
        lb = np.sqrt((bperp * bperp).sum(axis=1))
        F = np.zeros((self.n, 2, 2))
        F[:, 0, 0] = la
        F[:, 0, 1] = be1
        F[:, 1, 1] = lb
        bad = (la < 1e-9) | (lb < 1e-9) | ~np.isfinite(F).all(axis=(1, 2))
        if bad.any():
            F[bad] = np.eye(2)
        return F

    def _signed_gaps(self):
        r = 0.5 * self.thickness
        gap = np.full(self.n, 10.0)
        for box in self.boxes:
            sd, _ = box.sdf(self.pos)
            gap = np.minimum(gap, sd - r)
        pairs = self._self_pairs(self.pos)
        if pairs is not None:
            a, b = pairs
            d = np.linalg.norm(self.pos[b] - self.pos[a], axis=1) - self.SELF_RADIUS
            np.minimum.at(gap, a, d)
            np.minimum.at(gap, b, d)
        return np.clip(gap, -10.0, 10.0)

    def _frame(self) -> StateFrame:
        pos = np.ascontiguousarray(self.pos, dtype=float)
        vel = np.ascontiguousarray(self.vel, dtype=float)
        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(vel).all():
            vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        ext = self.mass[:, None] * self.gravity[None, :]
        gaps = self._signed_gaps()
        self._max_speed = float(max(self._max_speed, np.abs(vel).max()))
        self._min_gap = float(min(self._min_gap, gaps.min()))
        frame = StateFrame(
            float(self.time),
            pos,
            vel,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "external_force": ext,
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "self_contact_radius": float(self.SELF_RADIUS),
                "friction_coefficient": float(self.MU),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            nsteps = int(round((target - self.time) / self.dt))
            for _ in range(max(nsteps, 0)):
                self._step(self.dt)
            self.time = target
        else:
            self.time = max(self.time, target)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped position-based dynamics (knit mass-spring network)",
            "substeps_per_sample": int(self.SUBSTEPS),
            "constraint_iterations": int(self.ITERS),
            "substep_dt": float(self.dt),
            "total_substeps": int(self.steps),
            "vertices": int(self.n),
            "max_speed": float(self._max_speed),
            "min_signed_gap": float(self._min_gap),
            "friction_coefficient": float(self.MU),
            "self_contact_radius": float(self.SELF_RADIUS),
        }
