"""Physics solver for scene 034_noodles_in_bowl.

Six hundred twenty-five rod-modeled noodles (two nodes, one segment each,
length 0.4 m, radius 0.006 m) fall into a porcelain bowl, intertwine through
self-contact and settle with finite thickness.

The discretization is a position-based dynamics (PBD) scheme on the noodle
nodes:

* each noodle is an inextensible edge (distance constraint) between its two
  nodes -- the codimensional rod reduced to its stretch mode;
* gravity is integrated explicitly;
* noodle-bowl contact is a signed-distance projection against the exact
  axisymmetric inner surface of revolution (offset by the noodle radius);
* noodle-noodle contact is a finite-thickness (capsule) constraint: the
  centerline distance between two noodle segments must stay above the sum of
  radii (2r = 0.012 m), matching the C-IPC thickness/offset setting;
* contacts are resolved with position projection plus inelastic (normal
  velocity killing) and frictional velocity updates.

No animation, keyframes or precomputed trajectories are used: every frame is
the result of the explicit time stepping below. Only NumPy is required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBD solver for 625 noodles falling into an axisymmetric bowl."""

    def __init__(self) -> None:
        self.initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        n = self.x.shape[0]
        if frame0.mass is not None:
            self.mass = np.array(frame0.mass, dtype=float)
        else:
            self.mass = np.full(n, 0.00125)

        self.g = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # --- noodle topology: one edge (2 nodes) per noodle ---------------
        meta = scene.metadata() if callable(getattr(scene, "metadata", None)) else {}
        edges = None
        if isinstance(meta, dict):
            edges = meta.get("edges")
        if edges is None:
            m = int(cfg.get("noodle_count", n // 2))
            edges = [[2 * i, 2 * i + 1] for i in range(m)]
        self.edges = np.asarray(edges, dtype=np.int64)
        self.e0 = self.edges[:, 0]
        self.e1 = self.edges[:, 1]
        # rest lengths taken from the frozen initial state
        self.rest = np.linalg.norm(self.x[self.e1] - self.x[self.e0], axis=1)
        self.rest = np.maximum(self.rest, 1e-6)

        self.radius = float(cfg.get("radius", 0.006))
        self.contact_dist = 2.0 * self.radius  # finite thickness offset
        self.n_noodles = len(self.edges)

        # --- bowl inner profile -------------------------------------------
        profile = self._load_profile(cfg)
        # close the bottom with the disc from the axis to the profile foot
        pts = [(0.0, profile[0][1])] + [tuple(map(float, q)) for q in profile]
        pts = np.asarray(pts, dtype=float)
        # drop duplicate consecutive points
        keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-12]
        pts = pts[keep]
        self.prof_a = pts[:-1]
        self.prof_b = pts[1:]
        seg = self.prof_b - self.prof_a
        seg_len = np.linalg.norm(seg, axis=1)
        # cavity-side (left) normals of each profile segment in (rho, z)
        self.seg_nin = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / np.maximum(seg_len, 1e-12)[:, None]
        self.zknots = np.asarray([p[1] for p in profile], dtype=float)
        self.rknots = np.asarray([p[0] for p in profile], dtype=float)
        self.ztop = float(self.zknots[-1])

        # --- broadphase mask (upper triangle of noodle pairs) --------------
        m = self.n_noodles
        self.triu_mask = np.triu(np.ones((m, m), dtype=bool), 1)
        self.broad2 = (0.5 + self.contact_dist) ** 2  # half-lengths + thickness

        # --- time integration parameters -----------------------------------
        self.h = 1.0 / 400.0
        self.iters = 3
        self.drag = 0.6
        self.vmax = 30.0
        self.friction = 0.9  # tangential velocity kept per bowl-contact substep

        self.t = 0.0
        self.initialized = True
        self._steps = 0
        self._last_pairs = 0

    def _load_profile(self, cfg: dict) -> np.ndarray:
        default = [(0.12, 0.0), (0.38, 0.08), (0.66, 0.34), (0.72, 0.48)]
        try:
            import yaml  # type: ignore

            path = Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            for obj in data.get("objects", []):
                if obj.get("id") == "bowl":
                    prof = obj.get("geometry", {}).get("inner_profile")
                    if prof:
                        return np.asarray(prof, dtype=float)
        except Exception:
            pass
        bowl = cfg.get("bowl", {}) if isinstance(cfg, dict) else {}
        rim = float(bowl.get("rim_radius", 0.72))
        height = float(bowl.get("height", 0.48))
        if abs(rim - 0.72) < 1e-9 and abs(height - 0.48) < 1e-9:
            return np.asarray(default, dtype=float)
        return np.asarray(
            [(0.12, 0.0), (0.38, 0.08), (0.66, 0.34), (rim, height)], dtype=float
        )

    # --------------------------------------------------- bowl surface queries
    def _bowl_query(self, p: np.ndarray):
        """Distance of every node to the bowl's inner surface of revolution.

        Returns (signed_gap, inside, closest_rho, closest_z, dist, seg_idx).
        Positive gap = node inside the cavity; negative = inside the solid.
        """
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rho = np.hypot(x, y)
        qx, qz = rho, z

        a = self.prof_a
        d = self.prof_b - self.prof_a
        l2 = np.maximum((d * d).sum(axis=1), 1e-18)
        relx = qx[:, None] - a[None, :, 0]
        relz = qz[:, None] - a[None, :, 1]
        t = np.clip((relx * d[None, :, 0] + relz * d[None, :, 1]) / l2[None, :], 0.0, 1.0)
        cx = a[None, :, 0] + t * d[None, :, 0]
        cz = a[None, :, 1] + t * d[None, :, 1]
        dist2 = (qx[:, None] - cx) ** 2 + (qz[:, None] - cz) ** 2
        k = np.argmin(dist2, axis=1)
        idx = np.arange(len(p))
        cpx = cx[idx, k]
        cpz = cz[idx, k]
        dist = np.sqrt(dist2[idx, k])

        zc = np.clip(z, 0.0, self.ztop)
        rin = np.interp(zc, self.zknots, self.rknots)
        inside = ((z > 0.0) & (rho < rin)) | (z >= self.ztop)
        gap = np.where(inside, dist, -dist)
        return gap, inside, cpx, cpz, dist, k

    def _bowl_project(self, p: np.ndarray):
        """Project penetrating / near-wall nodes onto the offset inner surface."""
        gap, inside, cpx, cpz, dist, k = self._bowl_query(p)
        contact = gap < self.radius
        if not np.any(contact):
            return p, None
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rho = np.hypot(x, y)
        nin = self.seg_nin[k]
        eps = 1e-9
        # push direction (in the (rho,z) half plane)
        away_r = (rho - cpx) / np.maximum(dist, eps)
        away_z = (z - cpz) / np.maximum(dist, eps)
        use_in = (~inside) | (dist <= eps)
        dr = np.where(use_in, nin[:, 0], away_r)
        dz = np.where(use_in, nin[:, 1], away_z)
        new_r = np.maximum(cpx + self.radius * dr, 0.0)
        new_z = cpz + self.radius * dz
        ux = np.where(rho > eps, x / np.maximum(rho, eps), 0.0)
        uy = np.where(rho > eps, y / np.maximum(rho, eps), 0.0)
        out = p.copy()
        out[contact, 0] = new_r[contact] * ux[contact]
        out[contact, 1] = new_r[contact] * uy[contact]
        out[contact, 2] = new_z[contact]
        normals = np.zeros_like(p)
        normals[contact, 0] = dr[contact] * ux[contact]
        normals[contact, 1] = dr[contact] * uy[contact]
        normals[contact, 2] = dz[contact]
        return out, contact, normals

    # ------------------------------------------------ noodle-noodle contacts
    def _pair_geometry(self, p: np.ndarray, margin: float):
        """Closest-point geometry between all broadphase noodle pairs."""
        p1 = p[self.e0]
        p2 = p[self.e1]
        mid = 0.5 * (p1 + p2)
        diff_mid = mid[:, None, :] - mid[None, :, :]
        d2 = (diff_mid * diff_mid).sum(axis=-1)
        cand = self.triu_mask & (d2 < self.broad2)
        ci, cj = np.nonzero(cand)
        if len(ci) == 0:
            return None
        a1 = p1[ci]
        a2 = p2[ci]
        b1 = p1[cj]
        b2 = p2[cj]
        u = a2 - a1
        w = b2 - b1
        aa = np.maximum((u * u).sum(axis=1), 1e-12)
        ee = np.maximum((w * w).sum(axis=1), 1e-12)
        w0 = a1 - b1
        bb = (u * w).sum(axis=1)
        cc = (u * w0).sum(axis=1)
        dd = (w * w0).sum(axis=1)
        # alternating projection onto the two segments (both convex sets)
        s = np.full(len(ci), 0.5)
        t = np.full(len(ci), 0.5)
        for _ in range(8):
            s = np.clip((bb * t - cc) / aa, 0.0, 1.0)
            t = np.clip((bb * s + dd) / ee, 0.0, 1.0)
        pa = a1 + s[:, None] * u
        pb = b1 + t[:, None] * w
        diff = pa - pb
        dist = np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0))
        hit = dist < (self.contact_dist + margin)
        ci, cj = ci[hit], cj[hit]
        s, t = s[hit], t[hit]
        dist = dist[hit]
        n = diff[hit] / np.maximum(dist[:, None], 1e-12)
        return ci, cj, s, t, dist, n

    def _self_project(self, p: np.ndarray):
        geo = self._pair_geometry(p, 0.0)
        if geo is None:
            return None
        ci, cj, s, t, dist, n = geo
        pen = self.contact_dist - dist  # > 0 means violation
        ssum = (1.0 - s) ** 2 + s**2 + (1.0 - t) ** 2 + t**2
        lam = pen / np.maximum(ssum, 1e-9)
        out = p.copy()
        i1 = self.e0[ci]
        i2 = self.e1[ci]
        j1 = self.e0[cj]
        j2 = self.e1[cj]
        np.add.at(out, i1, (lam * (1.0 - s))[:, None] * n)
        np.add.at(out, i2, (lam * s)[:, None] * n)
        np.add.at(out, j1, -(lam * (1.0 - t))[:, None] * n)
        np.add.at(out, j2, -(lam * t)[:, None] * n)
        return out, (i1, i2, j1, j2, s, t, n)

    def _length_project(self, p: np.ndarray) -> np.ndarray:
        d = p[self.e1] - p[self.e0]
        ln = np.sqrt(np.maximum((d * d).sum(axis=1), 1e-18))
        corr = ((ln - self.rest) / ln) * 0.5
        out = p.copy()
        out[self.e0] += corr[:, None] * d
        out[self.e1] -= corr[:, None] * d
        return out

    # ------------------------------------------------------------ time step
    def _substep(self, h: float) -> None:
        xold = self.x
        v = self.v + self.g[None, :] * h
        drag = max(0.0, 1.0 - self.drag * h)
        v *= drag
        speed = np.sqrt((v * v).sum(axis=1))
        scale = np.minimum(1.0, self.vmax / np.maximum(speed, 1e-12))
        v = v * scale[:, None]
        p = xold + v * h

        bowl_info = None
        self_info = None
        for _ in range(self.iters):
            p = self._length_project(p)
            res = self._bowl_project(p)
            p = res[0]
            bowl_info = res
            res2 = self._self_project(p)
            if res2 is not None:
                p = res2[0]
                self_info = res2
            self._steps += 1

        v = (p - xold) / h

        # --- inelastic + frictional velocity update for bowl contacts -----
        if bowl_info is not None and bowl_info[1] is not None:
            contact, normals = bowl_info[1], bowl_info[2]
            vc = v[contact]
            nc = normals[contact]
            vn = (vc * nc).sum(axis=1)
            vc = vc - nc * np.minimum(vn, 0.0)[:, None]  # no approaching normal vel
            vn2 = (vc * nc).sum(axis=1)
            vt = vc - nc * vn2[:, None]
            vc = nc * vn2[:, None] + vt * self.friction
            v[contact] = vc

        # --- inelastic normal velocity for noodle-noodle contacts ---------
        if self_info is not None:
            i1, i2, j1, j2, s, t, n = self_info[1]
            va = (1.0 - s)[:, None] * v[i1] + s[:, None] * v[i2]
            vb = (1.0 - t)[:, None] * v[j1] + t[:, None] * v[j2]
            vrel = va - vb
            vn = (vrel * n).sum(axis=1)
            approaching = vn < 0.0
            if np.any(approaching):
                ssum = (1.0 - s) ** 2 + s**2 + (1.0 - t) ** 2 + t**2
                lam = np.where(approaching, -vn / np.maximum(ssum, 1e-9), 0.0)
                np.add.at(v, i1, (lam * (1.0 - s))[:, None] * n)
                np.add.at(v, i2, (lam * s)[:, None] * n)
                np.add.at(v, j1, -(lam * (1.0 - t))[:, None] * n)
                np.add.at(v, j2, -(lam * t)[:, None] * n)

        speed = np.sqrt((v * v).sum(axis=1))
        scale = np.minimum(1.0, self.vmax / np.maximum(speed, 1e-12))
        v = v * scale[:, None]

        self.x = p
        self.v = v

    # ------------------------------------------------------------- sampling
    def _contact_gaps(self, margin: float = 0.02) -> np.ndarray:
        gap, _, _, _, _, _ = self._bowl_query(self.x)
        gaps = [gap[gap < margin]]
        geo = self._pair_geometry(self.x, margin)
        if geo is not None:
            gaps.append(geo[4] - self.contact_dist)
        if gaps:
            return np.concatenate(gaps).astype(float)
        return np.zeros(0, dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        if not self.initialized:
            raise RuntimeError("solver not initialized")
        target = max(float(time), self.t)
        guard = 0
        while target - self.t > 1e-9:
            h = min(self.h, target - self.t)
            self._substep(h)
            self.t += h
            guard += 1
            if guard > 200000:
                break

        gaps = self._contact_gaps()
        external = self.mass[:, None] * self.g[None, :]
        return StateFrame(
            time=float(target),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
            },
            scalars={"contact_length_scale": float(self.radius)},
        )

    def diagnostics(self) -> dict:
        geo = self._pair_geometry(self.x, 0.0)
        ncontact = 0 if geo is None else len(geo[0])
        gap, _, _, _, _, _ = self._bowl_query(self.x)
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "noodle_noodle_contacts": int(ncontact),
            "bowl_contacts": int(np.count_nonzero(gap < self.radius)),
            "max_speed": float(np.sqrt((self.v * self.v).sum(axis=1)).max()),
            "max_z": float(self.x[:, 2].max()),
            "min_z": float(self.x[:, 2].min()),
        }
