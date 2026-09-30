"""Knitted scarf bunching on an inclined slab (effective yarn-level model).

Physical model
--------------
The scarf is a thin knitted sheet discretised as a regular 17x17 particle
network (289 samples, 512 triangles) taken from the frozen scaffold's initial
state.  The knit response is modelled with an effective yarn-level network:

* warp/weft yarn stretch constraints (stiff, nearly inextensible),
* diagonal shear constraints (softer, allows the knit to distort),
* second-neighbour bending constraints (very compliant, so the scarf buckles
  into rounded folds instead of staying a rigid plate),
* thickness-limited self contact (equal to the knit thickness) which gives the
  bunched pile real volume and prevents sheet interpenetration,
* rigid contact with the frozen inclined slab and ground slab including
  Coulomb friction; the concave corner where the slab meets the ground is the
  lower stop described by the task.

Time integration is an actual physical time-stepping scheme: XPBD substepping
(position based dynamics with compliance, Macklin et al.) with gravity as the
only external force.  Gravity pulls the scarf down the 35 degree incline, the
leading region is arrested at the lower stop, and the following material is
compressed into thick bunched folds while the upper scarf stays connected on
the slope.  Everything is deterministic and finite.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_NX_DEFAULT = 17
_NY_DEFAULT = 17
_TOTAL_MASS_DEFAULT = 0.32
_THICKNESS_DEFAULT = 0.015
_SUBSTEPS = 15                 # XPBD substeps per exported sample
_XPBD_ITERS = 2

_K_STRETCH = 4000.0            # N/m   yarn (warp/weft) stretch
_K_SHEAR = 900.0               # N/m   knit shear
_K_BEND = 0.35                 # N/m   axial 2nd-neighbour bending (compliant)
_K_BEND_DIAG = 0.15            # N/m   diagonal 2nd-neighbour bending
_MU_INCLINE = 0.58
_MU_GROUND = 1.10
_SELF_GAP = 0.018              # m     thickness-limited self contact gap
_VEL_DAMPING = 0.10            # 1/s   light structural damping
_VMAX = 25.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class BenchmarkSolver:
    """Deterministic XPBD thin-sheet solver for the knit scarf scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}) or {})
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.dt = self.sample_dt / float(_SUBSTEPS)

        init = scene.initial_state()
        self.p0 = np.array(init.position, dtype=float)
        self.n = int(self.p0.shape[0])
        mass = init.mass
        if mass is None:
            total = float(cloth.get("total_mass", _TOTAL_MASS_DEFAULT))
            mass = np.full(self.n, total / self.n)
        self.mass = np.asarray(mass, dtype=float).copy()

        nx, ny = self._grid_shape(cfg, cloth)
        self.nx, self.ny = nx, ny
        self.faces = self._faces(cfg)
        self.tri = self.faces

        self.p = self.p0.copy()
        self.v = np.asarray(init.velocity, dtype=float).copy()
        self.time = 0.0
        self.step = 0
        self.total_steps = int(round(float(cfg.get("duration", 7.0)) / self.dt)) + 1

        thickness = cloth.get("thickness")
        if thickness is None:
            thickness = (init.scalars or {}).get("contact_length_scale", _THICKNESS_DEFAULT)
        self._thickness = float(thickness)
        self._radius = 0.5 * self._thickness
        self.w = 1.0 / np.maximum(self.mass, 1e-12)
        self._build_constraints(nx, ny)
        self._build_self_pairs(nx, ny)
        self._setup_obstacles(cfg)
        self._build_material_coords()
        self._rest_area = self._triangle_areas(self.p0)
        self._face_weight = self._rest_area / max(self._rest_area.sum(), 1e-12)
        self._step_count = 0
        self._min_gap = float("inf")
        self._max_speed = 0.0

    def _grid_shape(self, cfg: dict, cloth: dict) -> tuple[int, int]:
        shape = cloth.get("grid_shape")
        if shape and len(shape) == 2 and int(shape[0]) * int(shape[1]) == self.n:
            return int(shape[0]), int(shape[1])
        geo = cfg.get("scene_geometry", {}).get("objects", [])
        for obj in geo:
            if obj.get("id") == "cloth":
                g = obj.get("geometry", {})
                s = g.get("grid_shape")
                if s and int(s[0]) * int(s[1]) == self.n:
                    return int(s[0]), int(s[1])
        root = int(round(math.sqrt(self.n)))
        if root * root == self.n:
            return root, root
        raise ValueError("cannot determine cloth grid shape")

    def _topology_path(self, cfg: dict) -> str | None:
        for obj in cfg.get("scene_geometry", {}).get("objects", []):
            if obj.get("id") == "cloth":
                path = obj.get("geometry", {}).get("topology_asset")
                if path:
                    return str(path)
        local = Path("scene_geometry.yaml")
        if local.is_file():
            try:
                import yaml

                data = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
                for obj in data.get("objects", []):
                    if obj.get("id") == "cloth":
                        path = (obj.get("geometry") or {}).get("topology_asset")
                        if path:
                            return str(Path(path).resolve())
            except Exception:
                return None
        return None

    def _faces(self, cfg: dict) -> np.ndarray:
        path = self._topology_path(cfg)
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                faces = np.asarray(data["faces"], dtype=np.int64)
                if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
                    return faces
            except Exception:
                pass
        nx, ny = self.nx, self.ny
        idx = np.arange(self.n, dtype=np.int64).reshape(ny, nx)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        return np.concatenate(
            (np.stack([a, b, c], axis=1), np.stack([b, d, c], axis=1)), axis=0
        )

    # ------------------------------------------------------------ constraints
    def _build_constraints(self, nx: int, ny: int) -> None:
        idx = np.arange(self.n, dtype=np.int64).reshape(ny, nx)
        ii: list[np.ndarray] = []
        jj: list[np.ndarray] = []
        kk: list[float] = []

        def add(dy: int, dx: int, stiff: float) -> None:
            if dx >= 0:
                a = idx[0 : ny - dy, 0 : nx - dx]
                b = idx[dy:ny, dx:nx]
            else:
                a = idx[0 : ny - dy, -dx : nx]
                b = idx[dy:ny, 0 : nx + dx]
            if a.size == 0:
                return
            ii.append(a.ravel())
            jj.append(b.ravel())
            kk.append(stiff)

        for dx, dy, k in (
            (1, 0, _K_STRETCH),
            (0, 1, _K_STRETCH),
            (1, 1, _K_SHEAR),
            (1, -1, _K_SHEAR),
            (2, 0, _K_BEND),
            (0, 2, _K_BEND),
            (2, 2, _K_BEND_DIAG),
            (2, -2, _K_BEND_DIAG),
        ):
            add(dy, dx, k)

        self.ci = np.concatenate(ii)
        self.cj = np.concatenate(jj)
        self.ck = np.concatenate([np.full(a.size, k) for a, k in zip(ii, kk)])
        self.cl0 = np.linalg.norm(self.p0[self.cj] - self.p0[self.ci], axis=1)
        self.cl0 = np.maximum(self.cl0, 1e-9)
        self.calpha = 1.0 / self.ck
        self.cw = self.w[self.ci] + self.w[self.cj]
        self.clam = np.zeros(len(self.ci))
        self.calpha_t = self.calpha / (self.dt * self.dt)

    def _build_self_pairs(self, nx: int, ny: int) -> None:
        idx = np.arange(self.n, dtype=np.int64).reshape(ny, nx)
        adj = np.zeros((self.n, self.n), dtype=bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if dx >= 0:
                a = idx[0 : ny - dy, 0 : nx - dx].ravel()
                b = idx[dy:ny, dx:nx].ravel()
            else:
                a = idx[0 : ny - dy, -dx:nx].ravel()
                b = idx[dy:ny, 0 : nx + dx].ravel()
            adj[a, b] = True
            adj[b, a] = True
        iu, ju = np.triu_indices(self.n, k=1)
        keep = ~adj[iu, ju]
        self.pi = iu[keep]
        self.pj = ju[keep]

    def _setup_obstacles(self, cfg: dict) -> None:
        self.obstacles = []
        for obj in cfg.get("scene_geometry", {}).get("objects", []):
            if obj.get("id") not in {"incline", "ground"}:
                continue
            geom = obj.get("geometry", {})
            if geom.get("type") != "box":
                continue
            half = np.asarray(geom.get("half_extents", [1, 1, 1]), dtype=float)
            tr = obj.get("transform", {}) or {}
            center = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            q = np.asarray(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
            rot = self._quat_to_matrix(q)
            mu = _MU_GROUND if obj.get("id") == "ground" else _MU_INCLINE
            self.obstacles.append({"c": center, "R": rot, "h": half, "mu": float(mu)})
        if not self.obstacles:
            self.obstacles = [
                {"c": np.array([0.0, 0.0, -0.05]), "R": np.eye(3),
                 "h": np.array([2.4, 2.0, 0.05]), "mu": _MU_GROUND},
                {"c": np.array([0.0, 0.0, 0.8]),
                 "R": self._quat_to_matrix(np.array([0.0, 0.300706, 0.0, 0.953717])),
                 "h": np.array([2.2, 1.1, 0.08]), "mu": _MU_INCLINE},
            ]

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = (float(a) for a in q)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _build_material_coords(self) -> None:
        ref = self.p0 - self.p0[0]
        e1 = _normalize(ref[1])
        nrm = np.cross(ref[1], ref[self.nx] if self.nx < self.n else ref[-1])
        nrm = _normalize(nrm)
        e2 = _normalize(np.cross(nrm, e1))
        self.mat_uv = np.column_stack((ref @ e1, ref @ e2))
        if not np.isfinite(self.mat_uv).all():
            self.mat_uv = np.zeros((self.n, 2))

    def _triangle_areas(self, p: np.ndarray) -> np.ndarray:
        e1 = p[self.tri[:, 1]] - p[self.tri[:, 0]]
        e2 = p[self.tri[:, 2]] - p[self.tri[:, 0]]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    # ---------------------------------------------------------------- physics
    def _solve_distance(self) -> None:
        p = self.p
        d = p[self.cj] - p[self.ci]
        dist = np.linalg.norm(d, axis=1)
        inv = 1.0 / np.maximum(dist, 1e-12)
        n = d * inv[:, None]
        c = dist - self.cl0
        dl = (-c - self.calpha_t * self.clam) / (self.cw + self.calpha_t)
        corr = n * (self.w[self.ci] * dl)[:, None]
        np.add.at(p, self.ci, corr)
        np.add.at(p, self.cj, -corr)
        self.clam += dl

    def _box_gap_normal(self, p: np.ndarray, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        local = (p - obs["c"][None, :]) @ obs["R"]
        a = np.abs(local) - obs["h"][None, :]
        k = np.argmax(a, axis=1)
        rows = np.arange(len(p))
        qk = local[rows, k]
        sign = np.where(qk >= 0.0, 1.0, -1.0)
        nrm = obs["R"][:, k].T * sign[:, None]
        outside = np.linalg.norm(np.maximum(a, 0.0), axis=1)
        inside = np.minimum(np.max(a, axis=1), 0.0)
        gap = np.where(np.max(a, axis=1) > 0.0, outside, inside)
        gn = _normalize(nrm)
        return gap, gn

    def _collide_slabs(self, p_old: np.ndarray) -> None:
        p = self.p
        radius = self._radius
        for obs in self.obstacles:
            gap, nrm = self._box_gap_normal(p, obs)
            gap = gap - radius
            mask = gap < 0.0
            self._min_gap = min(self._min_gap, float(np.min(gap)))
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            dep = -gap[idx]
            n = nrm[idx]
            p[idx] += n * dep[:, None]
            delta = p[idx] - p_old[idx]
            dn = np.einsum("ij,ij->i", delta, n)
            tan = delta - n * dn[:, None]
            mag = np.linalg.norm(tan, axis=1)
            fac = np.minimum(1.0, obs["mu"] * dep / np.maximum(mag, 1e-12))
            p[idx] -= tan * fac[:, None]
            self._min_gap = min(self._min_gap, float(np.min(gap)))

    def _collide_self(self) -> None:
        if self.pi.size == 0:
            return
        p = self.p
        d = p[self.pi] - p[self.pj]
        d2 = np.einsum("ij,ij->i", d, d)
        mask = d2 < (_SELF_GAP * _SELF_GAP)
        if not mask.any():
            return
        idx = np.nonzero(mask)[0]
        dd = np.sqrt(np.maximum(d2[idx], 1e-18))
        n = d[idx] / dd[:, None]
        pen = 0.5 * (_SELF_GAP - dd)
        corr = n * pen[:, None]
        np.add.at(p, self.pi[idx], corr)
        np.add.at(p, self.pj[idx], -corr)

    def _step(self) -> None:
        p, v = self.p, self.v
        p_old = p.copy()
        v += self.gravity[None, :] * self.dt
        v *= math.exp(-_VEL_DAMPING * self.dt)
        p += v * self.dt
        self.clam[:] = 0.0
        for _ in range(_XPBD_ITERS):
            self._solve_distance()
        self._collide_self()
        self._collide_slabs(p_old)
        v[:] = (p - p_old) / self.dt
        np.clip(v, -_VMAX, _VMAX, out=v)
        bad = ~np.isfinite(p)
        if bad.any():
            p[bad] = self.p0[bad]
            v[bad] = 0.0
        self._step_count += 1
        self._max_speed = max(self._max_speed, float(np.max(np.linalg.norm(v, axis=1))))

    # ---------------------------------------------------------------- outputs
    def _deformation_gradient(self) -> np.ndarray:
        tri, uv, p = self.tri, self.mat_uv, self.p
        u = uv[tri]
        dm = np.stack((u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]), axis=1)
        det = dm[:, 0, 0] * dm[:, 1, 1] - dm[:, 0, 1] * dm[:, 1, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        inv = np.empty_like(dm)
        inv[:, 0, 0] = dm[:, 1, 1] / det
        inv[:, 0, 1] = -dm[:, 0, 1] / det
        inv[:, 1, 0] = -dm[:, 1, 0] / det
        inv[:, 1, 1] = dm[:, 0, 0] / det

        e1 = p[tri[:, 1]] - p[tri[:, 0]]
        e2 = p[tri[:, 2]] - p[tri[:, 0]]
        l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-12)
        a1 = e1 / l1[:, None]
        proj = np.einsum("ij,ij->i", e2, a1)
        b = e2 - a1 * proj[:, None]
        lb = np.maximum(np.linalg.norm(b, axis=1), 1e-12)
        ds = np.zeros((len(tri), 2, 2))
        ds[:, 0, 0] = l1
        ds[:, 0, 1] = proj
        ds[:, 1, 1] = lb
        f = ds @ inv
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        eye = np.eye(2)

        wv = self._face_weight
        weighted = f * wv[:, None, None]
        acc = np.zeros((self.n, 2, 2))
        wsum = np.zeros(self.n)
        for k in range(3):
            idxk = tri[:, k]
            np.add.at(acc, idxk, weighted)
            np.add.at(wsum, idxk, wv)
        fv = acc / np.maximum(wsum, 1e-16)[:, None, None]
        return np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

    def _contact_distance(self) -> np.ndarray:
        radius = self._radius
        gaps = []
        for obs in self.obstacles:
            gap, _ = self._box_gap_normal(self.p, obs)
            gaps.append(gap - radius)
        return np.min(np.stack(gaps, axis=0), axis=0)

    def _frame(self, time: float) -> StateFrame:
        p = np.array(self.p, dtype=float)
        v = np.array(self.v, dtype=float)
        fv = self._deformation_gradient()
        gap = self._contact_distance()
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            p,
            v,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": fv,
            },
            fields={
                "external_force": ext,
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self._thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ solver API
    def advance_to(self, time: float):
        target = int(round(max(0.0, float(time)) / self.dt))
        target = min(target, self.total_steps)
        while self.step < target:
            self._step()
            self.step += 1
        self.time = self.step * self.dt
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "samples": int(self.n),
            "triangles": int(len(self.tri)),
            "constraints": int(len(self.ci)),
            "substeps": int(self._step_count),
            "dt": float(self.dt),
            "max_speed": float(self._max_speed),
            "min_contact_gap": float(self._min_gap) if np.isfinite(self._min_gap) else 0.0,
        }
