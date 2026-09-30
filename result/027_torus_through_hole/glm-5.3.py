"""XPBD deformable-solid solver for scene 027_torus_through_hole.

Physical model
--------------
The subject is a closed-surface elastic torus mesh.  It is simulated with a
substepped XPBD (extended position based dynamics) time integrator:

* per-vertex inertia (total subject mass distributed uniformly),
* gravitational body force,
* distance (spring) constraints on all mesh edges plus soft second-ring
  shear springs (edge stiffness taken from the frozen ``stiffness`` config),
* a global enclosed-volume constraint (soft near-incompressibility) built
  from the divergence theorem over the closed mesh,
* implicit-velocity damping from the frozen ``damping`` parameter,
* unilateral contact (full positional projection) against:
    - the ground box top plane,
    - the rigid hole plate (a solid torus fit analytically from the frozen
      ``hole_plate.obj`` mesh),
    - the kinematic pusher cylinder moving down at ``push_speed``,
* Coulomb friction and near-zero restitution at contacts, using the frozen
  ``friction`` / ``restitution`` parameters.

Per-vertex deformation gradients are recovered with a weighted
moving-least-squares fit against rest-space neighbourhoods.

No external physics engine, no keyframes, no precomputed motion: the state
at every sample time is produced by stepping the physics forward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional, only used to build neighbourhoods
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:  # only used to read the frozen scene geometry description
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _edges_from_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.sort(e, axis=1)
    key = e[:, 0].astype(np.int64) * 1000003 + e[:, 1]
    key = np.unique(key)
    a = key // 1000003
    b = key % 1000003
    return a.astype(np.int64), b.astype(np.int64)


def _load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"no vertices found in {path}")
    return np.asarray(verts, dtype=float)


def _bincount_scatter(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    out = np.empty((n, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=vals[:, c], minlength=n)
    return out


class BenchmarkSolver:
    """Deterministic XPBD solver for the torus-through-hole scene."""

    def __init__(self) -> None:
        self._initialized = False
        self.t = 0.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        root = Path(__file__).resolve().parents[1]

        # ---- subject mesh (frozen asset) --------------------------------
        subject_path = root / str(cfg.get("subject_asset", "assets/subject.npz"))
        with np.load(subject_path) as data:
            V = np.asarray(data["vertices"], dtype=float)
            faces = np.asarray(data["faces"], dtype=np.int64)
        self.n = len(V)
        self.x = V.copy()
        self.v = np.zeros_like(V)
        self.t = 0.0
        self.x0 = V.copy()

        total_mass = float(cfg.get("subject_mass", 1.0))
        self.m = np.full(self.n, total_mass / self.n, dtype=float)
        self.w = 1.0 / self.m
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # ---- spring network ---------------------------------------------
        ea, eb = _edges_from_faces(faces)
        rest0 = np.linalg.norm(V[ea] - V[eb], axis=1)
        mean_edge = float(rest0.mean())
        self.contact_scale = float(cfg.get("contact_length_scale", mean_edge))

        stiffness = float(cfg.get("stiffness", 10.0))
        edge_keys = ea.astype(np.int64) * 1000003 + eb

        extra_a: list[np.ndarray] = [ea]
        extra_b: list[np.ndarray] = [eb]
        extra_rest: list[np.ndarray] = [rest0]
        extra_alpha: list[np.ndarray] = [np.full(len(ea), 1.0)]
        if cKDTree is not None:
            tree = cKDTree(V)
            rmax = 2.0 * mean_edge
            pairs = tree.query_pairs(r=rmax, output_type="ndarray")
            if len(pairs):
                p = np.sort(pairs, axis=1)
                d = np.linalg.norm(V[p[:, 0]] - V[p[:, 1]], axis=1)
                keep = d > 1.05 * mean_edge
                p = p[keep]
                if len(p):
                    pk = p[:, 0].astype(np.int64) * 1000003 + p[:, 1]
                    mask = ~np.isin(pk, edge_keys)
                    p = p[mask]
                if len(p) > 12000:  # deterministic thinning
                    stride = int(np.ceil(len(p) / 12000.0))
                    p = p[::stride]
                if len(p):
                    d = np.linalg.norm(V[p[:, 0]] - V[p[:, 1]], axis=1)
                    extra_a.append(p[:, 0].astype(np.int64))
                    extra_b.append(p[:, 1].astype(np.int64))
                    extra_rest.append(d)
                    extra_alpha.append(np.full(len(p), 3.0))

        self.ea = np.concatenate(extra_a)
        self.eb = np.concatenate(extra_b)
        self.rest = np.concatenate(extra_rest)
        alpha = np.concatenate(extra_alpha) / max(stiffness, 1e-6)

        # ---- integrator parameters --------------------------------------
        self.h = 1.0 / 600.0
        self.nit = 5
        self.alpha_t = alpha / (self.h * self.h)
        self.lam = np.zeros(len(self.rest))
        damping = float(cfg.get("damping", 1.0))
        self.damping_rate = damping / max(total_mass, 1e-9)
        self.mu = float(cfg.get("friction", 0.5))
        self.restitution = float(cfg.get("restitution", 0.0))
        self.vcap = 40.0

        # ---- enclosed volume constraint ---------------------------------
        self.f0 = faces[:, 0].copy()
        self.f1 = faces[:, 1].copy()
        self.f2 = faces[:, 2].copy()
        self.V0, grad0 = self._volume_and_grad(self.x0)
        denom0 = float(np.sum(self.w * np.einsum("ij,ij->i", grad0, grad0)))
        self.alpha_vol_t = 0.25 * max(denom0, 1e-12)
        self.lam_v = 0.0

        # ---- obstacles ---------------------------------------------------
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.ground_half = 2.0
        self._setup_hole_plate(root, cfg)
        self._setup_pusher(root, cfg)

        # ---- deformation-gradient MLS fit --------------------------------
        self._setup_mls(V)

        self.substeps = 0
        self._initialized = True

    # ------------------------------------------------------------- obstacles
    def _setup_hole_plate(self, root: Path, cfg: dict) -> None:
        asset = "assets/hole_plate.obj"
        # try to read the frozen geometry description for the asset name
        geo_path = root / "scene_geometry.yaml"
        if geo_path.exists() and yaml is not None:
            try:
                with open(geo_path, "r") as fh:
                    geo = yaml.safe_load(fh) or {}
                for obj in geo.get("objects", []):
                    if obj.get("id") == "rigid_hole":
                        asset = obj.get("geometry", {}).get("asset", asset)
            except Exception:
                pass
        plate_path = root / asset
        zc = float(cfg.get("hole_z", 0.62))
        hole_r = float(cfg.get("hole_radius", 0.5))
        R, a = None, None
        if plate_path.exists():
            pv = _load_obj_vertices(plate_path)
            zc = 0.5 * (pv[:, 2].min() + pv[:, 2].max())
            mid = pv[np.abs(pv[:, 2] - zc) < 1e-4]
            a = 0.5 * (pv[:, 2].max() - pv[:, 2].min())
            if len(mid) >= 2:
                r = np.hypot(mid[:, 0], mid[:, 1])
                R = 0.5 * (r.min() + r.max())
                a = 0.5 * (r.max() - r.min())
            else:
                R = hole_r + a
        if R is None or a is None or a <= 0.0:
            a = 0.23
            R = hole_r + a
        self.plate_R = float(R)
        self.plate_a = float(a)
        self.plate_zc = float(zc)

    def _setup_pusher(self, root: Path, cfg: dict) -> None:
        rp, hp, z0 = 0.48, 0.09, 1.88
        geo_path = root / "scene_geometry.yaml"
        if geo_path.exists() and yaml is not None:
            try:
                with open(geo_path, "r") as fh:
                    geo = yaml.safe_load(fh) or {}
                for obj in geo.get("objects", []):
                    if obj.get("id") == "pusher":
                        g = obj.get("geometry", {})
                        rp = float(g.get("radius", rp))
                        hp = 0.5 * float(g.get("height", 2.0 * hp))
                        tr = obj.get("transform", {}).get("translation", [0, 0, z0])
                        z0 = float(tr[2])
            except Exception:
                pass
        self.pusher_r = rp
        self.pusher_h = hp
        self.pusher_z0 = z0
        self.pusher_stop_z = hp  # bottom cap rests on the ground plane
        self.push_speed = float(cfg.get("push_speed", 0.0))

    def _pusher_center(self, t: float) -> float:
        z = self.pusher_z0 - self.push_speed * t
        return float(max(z, self.pusher_stop_z))

    def _pusher_vel(self, t: float) -> np.ndarray:
        z = self.pusher_z0 - self.push_speed * t
        if z > self.pusher_stop_z:
            return np.array([0.0, 0.0, -self.push_speed])
        return np.zeros(3)

    # ------------------------------------------------------------------ MLS
    def _setup_mls(self, V: np.ndarray) -> None:
        k = 8
        nb = np.arange(self.n)[:, None]
        if cKDTree is not None:
            tree = cKDTree(V)
            dist, idx = tree.query(V, k=min(self.n, k + 2))
        else:  # brute force fallback
            d2 = np.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=2)
            idx = np.argsort(d2, axis=1)[:, : min(self.n, k + 2)]
            dist = np.take_along_axis(np.sqrt(d2), idx, axis=1)
        keep = dist > 1e-9
        nb_list = np.full((self.n, k), -1, dtype=np.int64)
        wts = np.zeros((self.n, k))
        D = np.zeros((self.n, k, 3))
        for j in range(idx.shape[1]):
            room = (nb_list < 0).any(axis=1)
            take = room & keep[:, j]
            if not take.any():
                continue
            slot = np.argmax(nb_list < 0, axis=1)
            rows = np.nonzero(take)[0]
            nb_list[rows, slot[rows]] = idx[rows, j]
        valid = nb_list >= 0
        rows, cols = np.nonzero(valid)
        D[rows, cols] = V[nb_list[rows, cols]] - V[rows]
        w = 1.0 / (np.linalg.norm(D, axis=2) ** 2 + 1e-8)
        wts[rows, cols] = w[rows, cols]
        # fill invalid slots with zero-weight self references (safe)
        bad = ~valid
        if bad.any():
            nb_list[bad] = np.arange(self.n)[:, None][bad]
        self.mls_nb = nb_list
        self.mls_w = wts
        self.mls_D = D
        # Tikhonov regularization: surface neighbourhoods are rank-deficient
        # along the surface normal, so use a relative ridge term.
        M = np.einsum("nk,nki,nkj->nij", self.mls_w, D, D)
        ridge = 1e-2 * (np.einsum("nii->n", M) / 3.0 + 1e-12)
        M += ridge[:, None, None] * np.eye(3)[None, :, :]
        self.mls_Minvinv = np.linalg.inv(M)

    # ------------------------------------------------------------- physics
    def _volume_and_grad(self, x: np.ndarray):
        a0 = x[self.f0]
        a1 = x[self.f1]
        a2 = x[self.f2]
        c12 = np.cross(a1, a2)
        c20 = np.cross(a2, a0)
        c01 = np.cross(a0, a1)
        vol = float(np.sum(np.einsum("ij,ij->i", a0, c12)) / 6.0)
        grad = _bincount_scatter(self.f0, c12 / 6.0, self.n)
        grad += _bincount_scatter(self.f1, c20 / 6.0, self.n)
        grad += _bincount_scatter(self.f2, c01 / 6.0, self.n)
        return vol, grad

    def _project_ground(self, x: np.ndarray, corr: np.ndarray) -> None:
        pen = self.ground_z - x[:, 2]
        inside = (
            (pen > 0.0)
            & (np.abs(x[:, 0]) < self.ground_half)
            & (np.abs(x[:, 1]) < self.ground_half)
        )
        if inside.any():
            d = pen[inside]
            x[inside, 2] += d
            corr[inside, 2] += d

    def _project_plate(self, x: np.ndarray, corr: np.ndarray) -> None:
        r = np.hypot(x[:, 0], x[:, 1])
        q1 = r - self.plate_R
        q2 = x[:, 2] - self.plate_zc
        L = np.sqrt(q1 * q1 + q2 * q2)
        sd = L - self.plate_a
        mask = (sd < 0.0) & (L > 1e-12)
        rows = np.nonzero(mask)[0]
        if len(rows) == 0:
            return
        rr = r[rows]
        rsafe = np.where(rr > 1e-9, rr, 1.0)
        nx = q1[rows] * x[rows, 0] / (rsafe * L[rows])
        ny = q1[rows] * x[rows, 1] / (rsafe * L[rows])
        nz = q2[rows] / L[rows]
        depth = -sd[rows]
        dx = depth * nx
        dy = depth * ny
        dz = depth * nz
        x[rows, 0] += dx
        x[rows, 1] += dy
        x[rows, 2] += dz
        corr[rows, 0] += dx
        corr[rows, 1] += dy
        corr[rows, 2] += dz

    def _project_pusher(self, x: np.ndarray, corr: np.ndarray, zc: float) -> None:
        rho = np.hypot(x[:, 0], x[:, 1])
        dz = x[:, 2] - zc
        adz = np.abs(dz)
        inside = (rho < self.pusher_r) & (adz < self.pusher_h)
        rows = np.nonzero(inside)[0]
        if len(rows) == 0:
            return
        rside = self.pusher_r - rho[rows]
        rcap = self.pusher_h - adz[rows]
        # side pushes
        side = (rside <= rcap) & (rho[rows] > 1e-9)
        if side.any():
            rs = rows[side]
            scale = self.pusher_r / np.maximum(rho[rs], 1e-12)
            newx = x[rs, 0] * scale
            newy = x[rs, 1] * scale
            corr[rs, 0] += newx - x[rs, 0]
            corr[rs, 1] += newy - x[rs, 1]
            x[rs, 0] = newx
            x[rs, 1] = newy
        # cap pushes
        cap = ~side
        if cap.any():
            rc = rows[cap]
            newz = zc + np.sign(dz[rc]) * self.pusher_h
            corr[rc, 2] += newz - x[rc, 2]
            x[rc, 2] = newz

    # signed distances (for reporting and contact tests)
    def _sd_ground(self, x: np.ndarray) -> np.ndarray:
        sd = x[:, 2] - self.ground_z
        outside = (np.abs(x[:, 0]) >= self.ground_half) | (
            np.abs(x[:, 1]) >= self.ground_half
        )
        sd = np.where(outside, np.maximum(sd, 0.0), sd)
        return sd

    def _sd_plate(self, x: np.ndarray) -> np.ndarray:
        r = np.hypot(x[:, 0], x[:, 1])
        q1 = r - self.plate_R
        q2 = x[:, 2] - self.plate_zc
        return np.sqrt(q1 * q1 + q2 * q2) - self.plate_a

    def _sd_pusher(self, x: np.ndarray, zc: float) -> np.ndarray:
        rho = np.hypot(x[:, 0], x[:, 1])
        dz = np.abs(x[:, 2] - zc)
        dr = rho - self.pusher_r
        dcap = dz - self.pusher_h
        inside = (rho < self.pusher_r) & (dz < self.pusher_h)
        sd_in = -np.minimum(self.pusher_r - rho, self.pusher_h - dz)
        sd_corner = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dcap, 0.0) ** 2)
        sd_out = np.maximum(dr, dcap)
        sd = np.where(inside, sd_in, np.where((dr > 0) & (dcap > 0), sd_corner, sd_out))
        return sd

    def _contact_normals(self, x: np.ndarray, which: str, zc: float):
        """Return (mask, normals, obstacle velocity) for near-contact points."""
        if which == "ground":
            sd = self._sd_ground(x)
            n = np.tile(np.array([0.0, 0.0, 1.0]), (self.n, 1))
            vobs = np.zeros(3)
        elif which == "plate":
            sd = self._sd_plate(x)
            r = np.hypot(x[:, 0], x[:, 1])
            q1 = r - self.plate_R
            q2 = x[:, 2] - self.plate_zc
            L = np.sqrt(q1 * q1 + q2 * q2)
            rsafe = np.where(r > 1e-9, r, 1.0)
            n = np.stack(
                [q1 * x[:, 0] / (rsafe * L), q1 * x[:, 1] / (rsafe * L), q2 / L],
                axis=1,
            )
            vobs = np.zeros(3)
        else:
            sd = self._sd_pusher(x, zc)
            rho = np.hypot(x[:, 0], x[:, 1])
            dz = x[:, 2] - zc
            adz = np.abs(dz)
            rside = self.pusher_r - rho
            rcap = self.pusher_h - adz
            n = np.zeros((self.n, 3))
            side = (rside <= rcap) & (rho > 1e-9)
            n[side, 0] = x[side, 0] / rho[side]
            n[side, 1] = x[side, 1] / rho[side]
            capm = ~side
            n[capm, 2] = np.sign(dz[capm])
            vobs = self._pusher_vel(self.t)
        mask = sd < 0.5 * self.contact_scale
        return mask, n, vobs

    def _apply_friction(self, x: np.ndarray, corr: np.ndarray, h: float) -> None:
        zc = self._pusher_center(self.t)
        for which in ("ground", "plate", "pusher"):
            mask, n, vobs = self._contact_normals(x, which, zc)
            rows = np.nonzero(mask)[0]
            if len(rows) == 0:
                continue
            vrel = self.v[rows] - vobs[None, :]
            vn = np.einsum("ij,ij->i", vrel, n[rows])
            vt = vrel - vn[:, None] * n[rows]
            vtn = np.linalg.norm(vt, axis=1)
            impn = np.abs(np.einsum("ij,ij->i", corr[rows], n[rows])) / h
            slip = np.maximum(vtn, 1e-9)
            scale = np.clip(1.0 - self.mu * impn / slip, 0.0, 1.0)
            vn_new = np.where(vn < 0.0, -self.restitution * vn, vn)
            self.v[rows] = vobs[None, :] + scale[:, None] * vt + vn_new[:, None] * n[rows]

    def _substep(self, h: float) -> None:
        # gravity + damping
        self.v += h * self.g[None, :]
        self.v /= 1.0 + self.damping_rate * h
        np.clip(self.v, -self.vcap, self.vcap, out=self.v)
        x_old = self.x.copy()
        x = self.x + h * self.v

        self.lam[:] = 0.0
        self.lam_v = 0.0
        corr = np.zeros_like(x)
        zc = self._pusher_center(self.t)

        wa = self.w[self.ea]
        wb = self.w[self.eb]
        for _ in range(self.nit):
            # --- distance constraints (XPBD) ---
            d = x[self.eb] - x[self.ea]
            L = np.linalg.norm(d, axis=1)
            Ls = np.maximum(L, 1e-9)
            C = L - self.rest
            denom = wa + wb + self.alpha_t
            dlam = (-C - self.alpha_t * self.lam) / denom
            self.lam += dlam
            step = (dlam / Ls)[:, None] * d
            x = x - _bincount_scatter(self.ea, wa[:, None] * step, self.n)
            x = x + _bincount_scatter(self.eb, wb[:, None] * step, self.n)

            # --- global volume constraint ---
            vol, grad = self._volume_and_grad(x)
            denomv = float(np.sum(self.w * np.einsum("ij,ij->i", grad, grad)))
            denomv += self.alpha_vol_t
            dlamv = (-(vol - self.V0) - self.alpha_vol_t * self.lam_v) / denomv
            self.lam_v += dlamv
            x = x + (self.w * dlamv)[:, None] * grad

            # --- collisions (unilateral projection) ---
            self._project_ground(x, corr)
            self._project_plate(x, corr)
            self._project_pusher(x, corr, zc)

        # velocity update
        self.v = (x - x_old) / h
        np.clip(self.v, -self.vcap, self.vcap, out=self.v)
        # friction / restitution at contacts
        self._apply_friction(x, corr, h)

        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.x = x
        self.t += h
        self.substeps += 1

    # ---------------------------------------------------------------- output
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        dY = x[self.mls_nb] - x[:, None, :]
        B = np.einsum("nk,nki,nkj->nij", self.mls_w, dY, self.mls_D)
        F = np.einsum("nij,njk->nik", B, self.mls_Minvinv)
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _contact_signed_distance(self, x: np.ndarray) -> np.ndarray:
        zc = self._pusher_center(self.t)
        sd = np.minimum(self._sd_ground(x), self._sd_plate(x))
        sd = np.minimum(sd, self._sd_pusher(x, zc))
        return sd

    def _frame(self, time: float) -> StateFrame:
        x = self.x
        F = self._deformation_gradient(x)
        external = self.m[:, None] * self.g[None, :]
        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
                "contact_signed_distance": self._contact_signed_distance(x),
            },
            scalars={"contact_length_scale": float(self.contact_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ api
    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("solver not initialized")
        target = float(time)
        remaining = target - self.t
        if remaining > 1e-12:
            n = int(np.ceil(remaining / self.h - 1e-9))
            n = max(1, n)
            h = remaining / n
            for _ in range(n):
                self._substep(h)
        return self._frame(max(target, self.t) if target >= 0 else self.t)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "substeps": int(self.substeps),
            "num_vertices": int(self.n),
            "num_springs": int(len(self.rest)),
            "rest_volume": float(self.V0),
            "plate_torus": [self.plate_R, self.plate_a, self.plate_zc],
            "pusher": [self.pusher_r, self.pusher_h, self.pusher_z0],
        }
