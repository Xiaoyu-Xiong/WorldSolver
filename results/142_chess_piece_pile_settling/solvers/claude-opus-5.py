"""Self-implemented rigid-body solver for chess piece pile settling.

Physical model
--------------
Each chess piece is a rigid body (6 DOF: position of the body-frame origin and a
unit quaternion).  Collision geometry is a sphere-union proxy generated directly
from the frozen triangle meshes (rings of spheres per height slab), so the wide
flared base, thin stem and bulky head are all represented -- the proxy is
non-convex and therefore pieces can tumble and interlock.

Contacts (sphere/sphere between distinct bodies and sphere/floor-plane) are
resolved with a regularised compliant contact law:

    Fn = max(0, k * g + k2 * g^2 - c * d(gap)/dt)          (Hunt-Crossley style)
    Ft = -mu * Fn * tanh(|vt| / vs) * vt_hat               (regularised Coulomb)

The normal damping coefficient c is derived from the frozen restitution, and the
friction impulse is clamped so it can never reverse the tangential relative
velocity within a step (keeps the explicit scheme unconditionally bounded).

Time integration is a semi-implicit (symplectic) Euler / Newton-Euler update
with a fixed sub-step, so the trajectory is deterministic and finite.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_FALLBACK_ASSETS = [
    "pawn.obj", "rook.obj", "knight.obj", "bishop.obj",
    "queen.obj", "king.obj",
]


# ----------------------------------------------------------------------------- helpers
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """q: (N,4) xyzw unit quaternions -> (N,3,3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """First order quaternion update with renormalisation (xyzw)."""
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dq = np.empty_like(q)
    dq[:, 0] = 0.5 * (wx * qw + wy * qz - wz * qy)
    dq[:, 1] = 0.5 * (-wx * qz + wy * qw + wz * qx)
    dq[:, 2] = 0.5 * (wx * qy - wy * qx + wz * qw)
    dq[:, 3] = 0.5 * (-wx * qx - wy * qy - wz * qz)
    out = q + dt * dq
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return out / n


def _load_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError("no vertices in %s" % path)
    return np.asarray(verts, dtype=float)


def _sphere_proxy(verts: np.ndarray, k_slabs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a surface-of-revolution-like mesh by a union of spheres.

    Returns (centers (S,3), radii (S,)) in mesh-local coordinates.
    """
    z = verts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    height = max(zmax - zmin, 1e-6)
    h = height / k_slabs
    rad = np.hypot(verts[:, 0], verts[:, 1])
    centers: list[tuple[float, float, float]] = []
    radii: list[float] = []
    for i in range(k_slabs):
        z0 = zmin + i * h
        z1 = z0 + h
        m = (z >= z0 - 1e-9) & (z <= z1 + 1e-9)
        if not bool(m.any()):
            continue
        rmax = float(rad[m].max())
        rmax = max(rmax, 0.25 * h)
        rs = max(0.55 * h, 0.02)
        zc = 0.5 * (z0 + z1)
        if rmax <= 2.0 * rs:
            r = max(min(rmax, 1.6 * rs), 0.02)
            zz = zc
            if height > 2.0 * r:
                zz = min(max(zc, zmin + r), zmax - r)
            else:
                zz = 0.5 * (zmin + zmax)
            centers.append((0.0, 0.0, zz))
            radii.append(r)
        else:
            rr = rmax - rs
            n = int(np.clip(round(2.0 * math.pi * rr / (1.7 * rs)), 3, 6))
            zz = zc
            if height > 2.0 * rs:
                zz = min(max(zc, zmin + rs), zmax - rs)
            else:
                zz = 0.5 * (zmin + zmax)
            phase = 0.37 * i
            for j in range(n):
                a = 2.0 * math.pi * j / n + phase
                centers.append((rr * math.cos(a), rr * math.sin(a), zz))
                radii.append(rs)
            centers.append((0.0, 0.0, zz))
            radii.append(rs)
    if not centers:
        centers.append((0.0, 0.0, 0.5 * (zmin + zmax)))
        radii.append(0.5 * height)
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def _inertia_from_spheres(centers: np.ndarray, radii: np.ndarray, mass: float) -> np.ndarray:
    """Inertia tensor about the body-frame origin for a sphere-union body."""
    w = radii ** 3
    w = w / float(w.sum())
    m = w * mass
    I = np.zeros((3, 3), dtype=float)
    for mi, ri, ci in zip(m, radii, centers):
        I += (0.4 * mi * ri * ri) * np.eye(3)
        I += mi * (float(ci @ ci) * np.eye(3) - np.outer(ci, ci))
    # keep well conditioned
    I += np.eye(3) * (1e-4 * mass)
    return I


# ----------------------------------------------------------------------------- solver
class BenchmarkSolver:
    """Explicit compliant-contact rigid body solver (NumPy only)."""

    # numerical parameters
    SUBSTEPS = 25            # per 0.02 s sample -> dt = 8e-4 s
    K_N = 9.0e3              # contact normal stiffness [N/m]
    K_N2 = 6.0e5             # quadratic term, limits deep penetration [N/m^2]
    V_SLIP = 0.02            # friction regularisation velocity [m/s]
    V_DEPEN = 0.6            # max depenetration speed per contact [m/s]
    V_MAX = 25.0             # velocity clamp (safety) [m/s]
    W_MAX = 60.0             # angular velocity clamp (safety) [rad/s]

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        bodies = cfg["bodies"]
        self.n = len(bodies)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.5))
        self.restitution = float(np.clip(float(contact.get("restitution", 0.1)), 1e-3, 0.95))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.dt = self.sample_dt / self.SUBSTEPS

        # damping ratio consistent with the frozen restitution
        le = math.log(self.restitution)
        self.zeta = -le / math.sqrt(math.pi * math.pi + le * le)

        # --- rigid state -------------------------------------------------
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.x = np.asarray([b["position"] for b in bodies], dtype=float).copy()
        q0 = np.asarray([b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], dtype=float)
        self.q = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
        self.v = np.zeros((self.n, 3), dtype=float)
        self.w = np.zeros((self.n, 3), dtype=float)
        self.body_id = np.arange(self.n, dtype=np.int32)

        # deterministic symmetry-breaking tilt (seeded, few degrees)
        rng = np.random.default_rng(self.seed)
        axis = rng.normal(size=(self.n, 3))
        axis[:, 2] *= 0.15
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        ang = rng.uniform(0.02, 0.06, size=self.n)
        s = np.sin(0.5 * ang)[:, None]
        dq = np.concatenate([axis * s, np.cos(0.5 * ang)[:, None]], axis=1)
        self.q = self._qmul(dq, self.q)

        # --- collision proxies from the frozen meshes --------------------
        assets = self._resolve_assets()
        self.floor_z = self._resolve_floor_top()
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        loc_list, rad_list, sb_list = [], [], []
        self.inertia = np.zeros((self.n, 3, 3), dtype=float)
        self.body_radius = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            name = assets[i]
            if name not in cache:
                cache[name] = _sphere_proxy(_load_obj_vertices(os.path.join(_ROOT, name)))
            c, r = cache[name]
            loc_list.append(c)
            rad_list.append(r)
            sb_list.append(np.full(len(r), i, dtype=np.int32))
            self.inertia[i] = _inertia_from_spheres(c, r, self.mass[i])
            self.body_radius[i] = float(np.max(np.linalg.norm(c, axis=1) + r))
        self.sph_local = np.concatenate(loc_list, axis=0)
        self.sph_r = np.concatenate(rad_list, axis=0)
        self.sph_b = np.concatenate(sb_list, axis=0)
        self.n_sph = len(self.sph_r)
        self.inv_inertia = np.linalg.inv(self.inertia)

        # per-body sphere index blocks + cached sphere-pair indices per body pair
        self.body_sph = [np.flatnonzero(self.sph_b == i).astype(np.int64)
                         for i in range(self.n)]
        self._pair_idx: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for a in range(self.n):
            for b in range(a + 1, self.n):
                ia = self.body_sph[a]
                ib = self.body_sph[b]
                gi = np.repeat(ia, ib.size)
                gj = np.tile(ib, ia.size)
                self._pair_idx[(a, b)] = (gi, gj)
        self.bp_margin = 4.0 * self.length_scale + 0.05
        self._empty_i = np.zeros(0, dtype=np.int64)

        self.time = 0.0
        self._steps = 0
        self._max_pen = 0.0
        self._last_report = {
            "gaps": np.zeros(1, dtype=float),
            "forces": np.zeros((1, 3), dtype=float),
        }

    # ---------------------------------------------------------------- config lookup
    def _resolve_assets(self) -> list[str]:
        """Map body index -> mesh asset path (relative to workspace root)."""
        names: dict[int, str] = {}
        path = os.path.join(_ROOT, "scene_geometry.yaml")
        try:
            import yaml  # type: ignore

            with open(path, "r") as fh:
                geo = yaml.safe_load(fh)
            for obj in geo.get("objects", []):
                binding = obj.get("state_binding") or {}
                asset = ((obj.get("geometry") or {}).get("asset"))
                if asset and binding.get("selector") == "body_id":
                    names[int(binding["value"])] = str(asset)
        except Exception:
            names = {}
        out = []
        for i in range(self.n):
            rel = names.get(i)
            if rel is None or not os.path.exists(os.path.join(_ROOT, rel)):
                rel = os.path.join("assets", _FALLBACK_ASSETS[i % len(_FALLBACK_ASSETS)])
            out.append(rel)
        return out

    def _resolve_floor_top(self) -> float:
        path = os.path.join(_ROOT, "scene_geometry.yaml")
        try:
            import yaml  # type: ignore

            with open(path, "r") as fh:
                geo = yaml.safe_load(fh)
            for obj in geo.get("objects", []):
                if obj.get("semantic") == "boundary" or obj.get("id") == "floor":
                    g = obj.get("geometry") or {}
                    t = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
                    he = g.get("half_extents", [1.0, 1.0, 0.05])
                    return float(t[2]) + float(he[2])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        out = np.empty_like(a)
        out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
        out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
        out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
        out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n < 1e-12, 1.0, n)

    # ---------------------------------------------------------------- contact forces
    def _sphere_state(self):
        R = _quat_to_mat(self.q)
        cw = np.einsum("sij,sj->si", R[self.sph_b], self.sph_local) + self.x[self.sph_b]
        arm = cw - self.x[self.sph_b]
        vw = self.v[self.sph_b] + np.cross(self.w[self.sph_b], arm)
        return R, cw, vw

    def _contact_forces(self, collect: bool = False):
        """Accumulate contact forces/torques; optionally report contact data."""
        dt = self.dt
        R, cw, vw = self._sphere_state()
        F = self.mass[:, None] * self.gravity[None, :]
        T = np.zeros((self.n, 3), dtype=float)
        rep_gap: list[np.ndarray] = []
        rep_f: list[np.ndarray] = []
        margin = 2.0 * self.length_scale

        # ---------------- broadphase: which body pairs can touch ---------
        bi_list, bj_list = [], []
        for a in range(self.n):
            dx = self.x[a + 1:] - self.x[a]
            if dx.shape[0] == 0:
                continue
            dsq = np.einsum("ij,ij->i", dx, dx)
            lim = self.body_radius[a + 1:] + self.body_radius[a] + self.bp_margin
            near = np.flatnonzero(dsq < lim * lim)
            for k in near:
                gi, gj = self._pair_idx[(a, a + 1 + int(k))]
                bi_list.append(gi)
                bj_list.append(gj)
        if bi_list:
            cpi = np.concatenate(bi_list)
            cpj = np.concatenate(bj_list)
        else:
            cpi = self._empty_i
            cpj = self._empty_i

        # ---------------- body/body sphere contacts ----------------------
        d = cw[cpi] - cw[cpj]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        pen_all = (self.sph_r[cpi] + self.sph_r[cpj]) - dist
        sel = np.flatnonzero(pen_all > -margin)
        if sel.size:
            si = cpi[sel]
            sj = cpj[sel]
            bi = self.sph_b[si]
            bj = self.sph_b[sj]
            dd = np.maximum(dist[sel], 1e-9)
            nrm = d[sel] / dd[:, None]
            pen = pen_all[sel]
            act = pen > 0.0
            # contact points (mid-surface)
            cp = cw[si] - nrm * (self.sph_r[si] - 0.5 * pen)[:, None]
            v_i = self.v[bi] + np.cross(self.w[bi], cp - self.x[bi])
            v_j = self.v[bj] + np.cross(self.w[bj], cp - self.x[bj])
            vrel = v_i - v_j
            vn = np.einsum("ij,ij->i", vrel, nrm)          # >0 separating
            mi = self.mass[bi]
            mj = self.mass[bj]
            meff = mi * mj / (mi + mj)
            cdamp = 2.0 * self.zeta * np.sqrt(self.K_N * meff)
            pp = np.maximum(pen, 0.0)
            fn = self.K_N * pp + self.K_N2 * pp * pp - cdamp * vn
            fn = np.where(act, np.maximum(fn, 0.0), 0.0)
            fn = np.minimum(fn, meff * self.V_DEPEN / dt)
            vt = vrel - vn[:, None] * nrm
            vtm = np.linalg.norm(vt, axis=1)
            ft_mag = self.mu * fn * np.tanh(vtm / self.V_SLIP)
            cap = 0.3 * meff * vtm / dt
            ft_mag = np.minimum(ft_mag, cap)
            safe = np.where(vtm > 1e-12, vtm, 1.0)
            fvec = fn[:, None] * nrm - (ft_mag / safe)[:, None] * vt
            arm_i = cp - self.x[bi]
            arm_j = cp - self.x[bj]
            np.add.at(F, bi, fvec)
            np.add.at(F, bj, -fvec)
            np.add.at(T, bi, np.cross(arm_i, fvec))
            np.add.at(T, bj, np.cross(arm_j, -fvec))
            if pen.size:
                self._max_pen = max(self._max_pen, float(pen.max()))
            if collect:
                rep_gap.append(-pen)
                rep_f.append(fvec)

        # ---------------- sphere/floor plane contacts --------------------
        pen_f = self.sph_r - (cw[:, 2] - self.floor_z)
        selg = np.flatnonzero(pen_f > -margin)
        if selg.size:
            b = self.sph_b[selg]
            pen = pen_f[selg]
            act = pen > 0.0
            nrm = np.zeros((selg.size, 3), dtype=float)
            nrm[:, 2] = 1.0
            cp = cw[selg].copy()
            cp[:, 2] -= (self.sph_r[selg] - 0.5 * pen)
            vrel = self.v[b] + np.cross(self.w[b], cp - self.x[b])
            vn = vrel[:, 2]
            meff = self.mass[b]
            cdamp = 2.0 * self.zeta * np.sqrt(self.K_N * meff)
            pp = np.maximum(pen, 0.0)
            fn = self.K_N * pp + self.K_N2 * pp * pp - cdamp * vn
            fn = np.where(act, np.maximum(fn, 0.0), 0.0)
            fn = np.minimum(fn, meff * self.V_DEPEN / dt)
            vt = vrel.copy()
            vt[:, 2] = 0.0
            vtm = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(self.mu * fn * np.tanh(vtm / self.V_SLIP),
                                0.3 * meff * vtm / dt)
            safe = np.where(vtm > 1e-12, vtm, 1.0)
            fvec = fn[:, None] * nrm - (ft_mag / safe)[:, None] * vt
            np.add.at(F, b, fvec)
            np.add.at(T, b, np.cross(cp - self.x[b], fvec))
            if pen.size:
                self._max_pen = max(self._max_pen, float(pen.max()))
            if collect:
                rep_gap.append(-pen)
                rep_f.append(fvec)

        if collect:
            if rep_gap:
                gaps = np.concatenate(rep_gap).astype(float)
                forces = np.concatenate(rep_f, axis=0).astype(float)
            else:
                gaps = np.zeros(0, dtype=float)
                forces = np.zeros((0, 3), dtype=float)
            if gaps.size == 0:
                # keep the field non-empty: closest approach in the scene
                gmin = float(np.min(-pen_all)) if pen_all.size else margin
                gaps = np.asarray([gmin], dtype=float)
                forces = np.zeros((1, 3), dtype=float)
            self._last_report = {"gaps": gaps, "forces": forces}
        return F, T

    # ---------------------------------------------------------------- integration
    def _step(self, collect: bool = False) -> None:
        dt = self.dt
        F, T = self._contact_forces(collect=collect)
        self._ext_force = self.mass[:, None] * self.gravity[None, :]

        # linear: semi-implicit Euler
        self.v += dt * (F / self.mass[:, None])

        # angular: Newton-Euler in world frame
        R = _quat_to_mat(self.q)
        Iw = np.einsum("nij,njk,nlk->nil", R, self.inertia, R)
        gyro = np.cross(self.w, np.einsum("nij,nj->ni", Iw, self.w))
        dw = np.linalg.solve(Iw, (T - gyro)[:, :, None])[:, :, 0]
        self.w += dt * dw

        # safety clamps (guarantee finite states)
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        sv = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v *= np.minimum(1.0, self.V_MAX / np.maximum(sv, 1e-12))
        sw = np.linalg.norm(self.w, axis=1, keepdims=True)
        self.w *= np.minimum(1.0, self.W_MAX / np.maximum(sw, 1e-12))

        self.x += dt * self.v
        self.q = _quat_integrate(self.q, self.w, dt)
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        self._steps += 1

    # ---------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing times")
        n_steps = int(round((target - self.time) / self.dt))
        for k in range(n_steps):
            self._step(collect=(k == n_steps - 1))
        self.time = target
        if n_steps == 0:
            self._contact_forces(collect=True)
            self._ext_force = self.mass[:, None] * self.gravity[None, :]
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        gaps = np.asarray(self._last_report["gaps"], dtype=float)
        forces = np.asarray(self._last_report["forces"], dtype=float)
        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": gaps,
                "external_force": np.ascontiguousarray(self._ext_force, dtype=float),
                "interaction_force": forces,
            },
            scalars={
                "contact_length_scale": float(self.length_scale),
                "friction": float(self.mu),
                "restitution": float(self.restitution),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        ke = 0.5 * float(np.sum(self.mass[:, None] * self.v ** 2))
        return {
            "method": "sphere-union compliant contact, semi-implicit Euler",
            "substep_dt": float(self.dt),
            "steps": int(self._steps),
            "bodies": int(self.n),
            "proxy_spheres": int(self.n_sph),
            "max_penetration": float(self._max_pen),
            "kinetic_energy": ke,
        }
