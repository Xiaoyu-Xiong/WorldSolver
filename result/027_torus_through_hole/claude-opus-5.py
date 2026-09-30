"""Deterministic XPBD deformable-solid solver for 027_torus_through_hole.

Physical model (implemented from scratch with NumPy/SciPy):

* The subject torus is a closed triangle mesh; mass is lumped at the vertices.
* Elasticity is a compliant-constraint (XPBD) energy:
  - per-edge distance constraints        -> stretch / shear resistance
  - apex-apex hinge constraints          -> bending resistance
  - one global enclosed-volume constraint-> bulk (pressure) resistance
* Contact is unilateral vertex projection against analytic signed-distance
  fields of the static ground slab, the static rigid hole plate (a torus
  shaped rim whose aperture is smaller than the subject) and the kinematic
  pusher cylinder, followed by Coulomb friction and restitution at the
  velocity level.
* Self contact uses vertex-vertex repulsion at the frozen contact length
  scale (topological neighbours within two rings excluded).

Time integration is XPBD (Macklin et al. 2016): symplectic-Euler prediction
followed by compliant constraint projection with per-constraint Lagrange
multipliers, which is stable for the frozen step size and converges towards
the elastic equilibrium as the iteration count grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:  # SciPy is used for the self-contact broad phase only.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None

from worldsolver.contracts import StateFrame

_TINY = 1e-12


def _scatter3(count: int, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Deterministic scatter-add of [M,3] values into an [count,3] buffer."""
    out = np.zeros((count, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.bincount(idx, weights=vals[:, k], minlength=count)
    return out


def _vertex_normals(x: np.ndarray, faces: np.ndarray, count: int) -> np.ndarray:
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    acc = np.zeros((count, 3), dtype=float)
    for k in range(3):
        acc += _scatter3(count, faces[:, k], fn)
    nrm = np.linalg.norm(acc, axis=1)
    return acc / np.maximum(nrm, 1e-15)[:, None]


def _sdf_box(p: np.ndarray, c: np.ndarray, h: np.ndarray):
    """Signed distance and unit gradient for an axis-aligned box."""
    d = p - c
    q = np.abs(d) - h[None, :]
    aq = np.maximum(q, 0.0)
    out = np.linalg.norm(aq, axis=1)
    qmax = np.max(q, axis=1)
    dist = out + np.minimum(qmax, 0.0)
    grad = aq * np.sign(d)
    inside = out <= _TINY
    if np.any(inside):
        one = np.zeros((int(np.count_nonzero(inside)), 3), dtype=float)
        am = np.argmax(q[inside], axis=1)
        one[np.arange(one.shape[0]), am] = 1.0
        grad[inside] = one * np.sign(d[inside] + _TINY)
    nrm = np.linalg.norm(grad, axis=1)
    grad = grad / np.maximum(nrm, 1e-15)[:, None]
    return dist, grad


def _sdf_torus(p: np.ndarray, c: np.ndarray, major: float, minor: float):
    """Signed distance and unit gradient for a z-axis torus."""
    dx = p[:, 0] - c[0]
    dy = p[:, 1] - c[1]
    dz = p[:, 2] - c[2]
    rr = np.hypot(dx, dy)
    rs = np.maximum(rr, 1e-12)
    a = rr - major
    ell = np.hypot(a, dz)
    es = np.maximum(ell, 1e-12)
    dist = ell - minor
    gr = a / es
    grad = np.stack([gr * dx / rs, gr * dy / rs, dz / es], axis=1)
    nrm = np.linalg.norm(grad, axis=1)
    grad = grad / np.maximum(nrm, 1e-15)[:, None]
    return dist, grad


def _sdf_cylinder(p: np.ndarray, c: np.ndarray, radius: float, half_h: float):
    """Signed distance and unit gradient for a finite z-axis cylinder."""
    dx = p[:, 0] - c[0]
    dy = p[:, 1] - c[1]
    dz = p[:, 2] - c[2]
    rr = np.hypot(dx, dy)
    rs = np.maximum(rr, 1e-12)
    qr = rr - radius
    qz = np.abs(dz) - half_h
    ar = np.maximum(qr, 0.0)
    az = np.maximum(qz, 0.0)
    out = np.hypot(ar, az)
    dist = out + np.minimum(np.maximum(qr, qz), 0.0)
    sgn_z = np.where(dz >= 0.0, 1.0, -1.0)
    outside = out > _TINY
    g_r = np.where(outside, ar / np.maximum(out, 1e-15), np.where(qr > qz, 1.0, 0.0))
    g_z = np.where(outside, az / np.maximum(out, 1e-15), np.where(qr > qz, 0.0, 1.0))
    grad = np.stack([g_r * dx / rs, g_r * dy / rs, g_z * sgn_z], axis=1)
    nrm = np.linalg.norm(grad, axis=1)
    grad = grad / np.maximum(nrm, 1e-15)[:, None]
    return dist, grad


class BenchmarkSolver:
    """XPBD deformable-solid solver honouring the frozen scene geometry."""

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))

        base = Path.cwd()
        try:
            mod = sys.modules.get(scene.__class__.__module__)
            cand = Path(getattr(mod, "__file__", "")).resolve().parent
            if (cand / cfg["subject_asset"]).exists():
                base = cand
        except Exception:
            pass
        if not (base / cfg["subject_asset"]).exists():
            here = Path(__file__).resolve().parent.parent
            if (here / cfg["subject_asset"]).exists():
                base = here
        self.base = base

        with np.load(self.base / cfg["subject_asset"]) as data:
            self.x0 = np.asarray(data["vertices"], dtype=float)
            self.faces = np.asarray(data["faces"], dtype=np.int64)

        self.n = int(self.x0.shape[0])
        self.x = self.x0.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n, 1))

        self.total_mass = float(cfg["subject_mass"])
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)
        self.w = 1.0 / self.mass
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        self.contact_h = float(cfg["contact_length_scale"])
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.restitution = float(cfg.get("restitution", 0.0))
        self.friction = float(cfg.get("friction", 0.0))
        self.damping_rate = float(cfg.get("damping", 0.0))
        self.push_speed = float(cfg.get("push_speed", 0.0))

        # --- frozen obstacle geometry -----------------------------------
        self.hole_z = float(cfg.get("hole_z", 0.62))
        self.hole_radius = float(cfg.get("hole_radius", 0.5))
        minor = 0.23
        center_z = self.hole_z
        r_out = self.hole_radius + 2.0 * minor
        try:
            with np.load(self.base / "assets" / "hole_plate.npz") as plate:
                pv = np.asarray(plate["vertices"], dtype=float)
            z_lo, z_hi = float(pv[:, 2].min()), float(pv[:, 2].max())
            minor = 0.5 * (z_hi - z_lo)
            center_z = 0.5 * (z_lo + z_hi)
            r_out = float(np.hypot(pv[:, 0], pv[:, 1]).max())
        except Exception:
            pass
        self.plate_minor = float(minor)
        self.plate_major = self.hole_radius + self.plate_minor
        self.plate_center = np.array([0.0, 0.0, float(center_z)], dtype=float)
        self.plate_r_out = float(r_out)

        self.ground_half = np.array([2.0, 2.0, 0.04], dtype=float)
        self.ground_center = np.array([0.0, 0.0, -0.04], dtype=float)

        self.push_radius = 0.48
        self.push_half_h = 0.09
        self.push_start_z = 1.88
        # Descend until the pusher face has driven the torus bulk below the
        # rim, then hold: the body releases elastically and falls to ground.
        # The face stops just above the rim crown so the pusher wall never
        # enters the narrow aperture annulus (radius 0.48 vs hole 0.50).
        self.push_stop_z = self.hole_z + 0.04 + self.push_half_h
        self.push_center = np.array([0.0, 0.0, self.push_start_z], dtype=float)
        self.push_vel = np.zeros(3, dtype=float)

        self._build_topology()
        self._build_gradient_operator()

        # --- constraint compliances (XPBD) ------------------------------
        k = float(cfg.get("stiffness", 10.0))
        self.k_edge = 100.0 * k
        self.k_bend = 6.0 * k
        self.alpha_edge = 1.0 / self.k_edge
        self.alpha_bend = 1.0 / self.k_bend
        self.alpha_vol = 0.0  # near-incompressible bulk response
        self.vol_relax = 0.6
        self.self_thickness = 0.45 * float(self.rest_len.mean())

        self.rest_volume = self._volume(self.x0)
        self.substeps = 25
        self.iters = 6

        self.time = 0.0
        self.max_penetration = 0.0
        self.min_volume_ratio = 1.0
        self.max_det = 1.0
        self.min_det = 1.0
        self.last_residual = 0.0
        self.inverted_elements = 0
        self.steps_taken = 0

    # ------------------------------------------------------------------
    def _build_topology(self) -> None:
        f = self.faces
        pairs = np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0
        )
        srt = np.sort(pairs, axis=1)
        key = srt[:, 0] * self.n + srt[:, 1]
        uniq, inv = np.unique(key, return_inverse=True)
        self.edges = np.stack([uniq // self.n, uniq % self.n], axis=1).astype(np.int64)
        self.rest_len = np.linalg.norm(
            self.x0[self.edges[:, 0]] - self.x0[self.edges[:, 1]], axis=1
        )

        # Interior-edge hinges: the two vertices opposite a shared edge.
        opp = np.concatenate([f[:, 2], f[:, 0], f[:, 1]], axis=0)
        order = np.argsort(inv, kind="stable")
        inv_s = inv[order]
        opp_s = opp[order]
        starts = np.searchsorted(inv_s, np.arange(len(uniq)), side="left")
        ends = np.searchsorted(inv_s, np.arange(len(uniq)), side="right")
        two = (ends - starts) == 2
        b0 = opp_s[starts[two]]
        b1 = opp_s[starts[two] + 1]
        self.bend = np.stack([b0, b1], axis=1).astype(np.int64)
        self.rest_bend = np.linalg.norm(
            self.x0[self.bend[:, 0]] - self.x0[self.bend[:, 1]], axis=1
        )

        # one-ring adjacency (padded) for the deformation gradient
        allp = np.concatenate([self.edges, self.edges[:, ::-1]], axis=0)
        ordn = np.lexsort((allp[:, 1], allp[:, 0]))
        allp = allp[ordn]
        counts = np.bincount(allp[:, 0], minlength=self.n)
        vmax = int(counts.max())
        nb = np.repeat(np.arange(self.n, dtype=np.int64)[:, None], vmax, axis=1)
        mask = np.zeros((self.n, vmax), dtype=float)
        off = np.concatenate([[0], np.cumsum(counts)])
        for i in range(self.n):
            c = int(counts[i])
            if c:
                nb[i, :c] = allp[off[i]:off[i] + c, 1]
                mask[i, :c] = 1.0
        self.nbr = nb
        self.nbr_mask = mask

        # two-ring exclusion set for self contact
        excl = set()
        for i in range(self.n):
            ring1 = nb[i, : int(counts[i])]
            for j in ring1:
                excl.add((min(i, int(j)), max(i, int(j))))
                for k2 in nb[int(j), : int(counts[int(j)])]:
                    k2 = int(k2)
                    if k2 != i:
                        excl.add((min(i, k2), max(i, k2)))
        if excl:
            ex = np.array(sorted(excl), dtype=np.int64)
            self.excl_keys = np.sort(ex[:, 0] * self.n + ex[:, 1])
        else:  # pragma: no cover
            self.excl_keys = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    def _build_gradient_operator(self) -> None:
        """Least-squares per-vertex deformation gradient operator."""
        self.n0 = _vertex_normals(self.x0, self.faces, self.n)
        d0 = self.x0[self.nbr] - self.x0[:, None, :]
        wts = self.nbr_mask / np.maximum(np.sum(d0 * d0, axis=2), 1e-12)
        wts *= self.nbr_mask
        self.grad_w = wts
        self.d0 = d0
        amat = np.einsum("vk,vki,vkj->vij", wts, d0, d0)
        amat += np.einsum("vi,vj->vij", self.n0, self.n0)
        amat += np.eye(3)[None, :, :] * 1e-9
        self.a_inv = np.linalg.inv(amat)


    # ------------------------------------------------------------------
    def _volume(self, x: np.ndarray) -> float:
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        return float(np.sum(a * np.cross(b, c)) / 6.0)

    def _volume_gradient(self, x: np.ndarray) -> np.ndarray:
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        g = np.zeros((self.n, 3), dtype=float)
        g += _scatter3(self.n, self.faces[:, 0], np.cross(b, c) / 6.0)
        g += _scatter3(self.n, self.faces[:, 1], np.cross(c, a) / 6.0)
        g += _scatter3(self.n, self.faces[:, 2], np.cross(a, b) / 6.0)
        return g

    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        d = x[self.nbr] - x[:, None, :]
        nc = _vertex_normals(x, self.faces, self.n)
        bmat = np.einsum("vk,vki,vkj->vij", self.grad_w, d, self.d0)
        bmat += np.einsum("vi,vj->vij", nc, self.n0)
        f = np.einsum("vij,vjk->vik", bmat, self.a_inv)
        return np.where(np.isfinite(f), f, 0.0)

    # ------------------------------------------------------------------
    def _pusher_at(self, t: float):
        z = self.push_start_z - self.push_speed * t
        if z <= self.push_stop_z:
            return np.array([0.0, 0.0, self.push_stop_z]), np.zeros(3)
        return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, -self.push_speed])

    def _obstacles(self):
        """Active obstacle list: (kind, params, linear velocity)."""
        return (
            ("ground", (self.ground_center, self.ground_half), np.zeros(3)),
            ("plate", (self.plate_center, self.plate_major, self.plate_minor), np.zeros(3)),
            ("pusher", (self.push_center, self.push_radius, self.push_half_h), self.push_vel),
        )

    def _obstacle_distance(self, x: np.ndarray) -> np.ndarray:
        dmin = np.full(x.shape[0], 1e6, dtype=float)
        for kind, params, _ in self._obstacles():
            if kind == "ground":
                d, _g = _sdf_box(x, params[0], params[1])
            elif kind == "plate":
                d, _g = _sdf_torus(x, params[0], params[1], params[2])
            else:
                d, _g = _sdf_cylinder(x, params[0], params[1], params[2])
            dmin = np.minimum(dmin, d)
        return dmin


    # ------------------------------------------------------------------
    def _self_pairs(self, x: np.ndarray) -> np.ndarray:
        if _KDTree is None:
            return np.zeros((0, 2), dtype=np.int64)
        tree = _KDTree(x)
        pairs = tree.query_pairs(self.self_thickness, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.sort(pairs.astype(np.int64), axis=1)
        keys = pairs[:, 0] * self.n + pairs[:, 1]
        keep = ~np.isin(keys, self.excl_keys)
        return pairs[keep]

    def _project_contacts(self, x: np.ndarray, contact: dict) -> None:
        """Unilateral projection out of every collision obstacle."""
        for kind, params, ovel in self._obstacles():
            if kind == "ground":
                d, g = _sdf_box(x, params[0], params[1])
            elif kind == "plate":
                d, g = _sdf_torus(x, params[0], params[1], params[2])
            else:
                d, g = _sdf_cylinder(x, params[0], params[1], params[2])
            hit = d < 0.0
            if not np.any(hit):
                continue
            corr = (-d[hit])[:, None] * g[hit]
            x[hit] += corr
            depth = -d[hit]
            better = depth > contact["depth"][hit]
            idx = np.nonzero(hit)[0][better]
            contact["depth"][idx] = depth[better]
            contact["normal"][idx] = g[hit][better]
            contact["ovel"][idx] = ovel[None, :]
            contact["active"][idx] = True
            contact["pen"] = max(contact["pen"], float(depth.max()))

        pairs = self._self_pairs(x)
        if pairs.shape[0]:
            i, j = pairs[:, 0], pairs[:, 1]
            dvec = x[j] - x[i]
            dist = np.linalg.norm(dvec, axis=1)
            act = dist < self.self_thickness
            if np.any(act):
                i, j = i[act], j[act]
                dist = np.maximum(dist[act], 1e-9)
                nvec = dvec[act] / dist[:, None]
                pen = (self.self_thickness - dist) * 0.5
                wi, wj = self.w[i], self.w[j]
                ws = np.maximum(wi + wj, _TINY)
                x[i] -= ((2.0 * pen * wi / ws) * 0.5)[:, None] * nvec
                x[j] += ((2.0 * pen * wj / ws) * 0.5)[:, None] * nvec

    def _solve_constraints(self, x: np.ndarray, dt: float, contact: dict) -> float:
        e0, e1 = self.edges[:, 0], self.edges[:, 1]
        b0, b1 = self.bend[:, 0], self.bend[:, 1]
        idx_e = np.concatenate([e0, e1])
        idx_b = np.concatenate([b0, b1])
        w_e = np.maximum(self.w[e0] + self.w[e1], _TINY)
        w_b = np.maximum(self.w[b0] + self.w[b1], _TINY)
        ae = self.alpha_edge / (dt * dt)
        ab = self.alpha_bend / (dt * dt)
        lam_e = np.zeros(self.edges.shape[0], dtype=float)
        lam_b = np.zeros(self.bend.shape[0], dtype=float)
        lam_v = 0.0
        relax = 1.0 / 6.0
        residual = 0.0

        for it in range(self.iters):
            # ---- edge distance constraints
            dvec = x[e0] - x[e1]
            dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
            c = dist - self.rest_len
            dlam = (-c - ae * lam_e) / (w_e + ae)
            dlam *= relax
            lam_e += dlam
            grad = dvec / dist[:, None]
            imp = dlam[:, None] * grad
            x += _scatter3(
                self.n, idx_e,
                np.concatenate([self.w[e0][:, None] * imp, -self.w[e1][:, None] * imp]),
            )
            residual = float(np.abs(c).max())

            # ---- hinge (bending) constraints
            dvb = x[b0] - x[b1]
            dstb = np.maximum(np.linalg.norm(dvb, axis=1), 1e-12)
            cb = dstb - self.rest_bend
            dlb = (-cb - ab * lam_b) / (w_b + ab)
            dlb *= relax
            lam_b += dlb
            gb = dvb / dstb[:, None]
            impb = dlb[:, None] * gb
            x += _scatter3(
                self.n, idx_b,
                np.concatenate([self.w[b0][:, None] * impb, -self.w[b1][:, None] * impb]),
            )

            # ---- global volume (bulk) constraint
            vol = self._volume(x)
            gv = self._volume_gradient(x)
            cv = vol - self.rest_volume
            denom = float(np.sum(self.w[:, None] * gv * gv))
            av = self.alpha_vol / (dt * dt)
            if denom > _TINY:
                dlv = (-cv - av * lam_v) / (denom + av)
                dlv *= self.vol_relax
                lam_v += dlv
                x += dlv * self.w[:, None] * gv

            # ---- contact (last iterations, keeps penetration small)
            if it >= self.iters - 3:
                self._project_contacts(x, contact)

        return residual

    def _substep(self, dt: float) -> None:
        x_prev = self.x.copy()
        v_pred = self.v + dt * self.gravity[None, :]
        x = x_prev + dt * v_pred

        contact = {
            "active": np.zeros(self.n, dtype=bool),
            "normal": np.zeros((self.n, 3), dtype=float),
            "ovel": np.zeros((self.n, 3), dtype=float),
            "depth": np.zeros(self.n, dtype=float),
            "pen": 0.0,
        }
        self.last_residual = self._solve_constraints(x, dt, contact)

        v_new = (x - x_prev) / dt

        act = contact["active"]
        if np.any(act):
            nrm = contact["normal"][act]
            ovel = contact["ovel"][act]
            rel_pre = v_pred[act] - ovel
            rel_new = v_new[act] - ovel
            vn_pre = np.sum(rel_pre * nrm, axis=1)
            vn_new = np.sum(rel_new * nrm, axis=1)
            # restitution on approaching contacts
            bounce = np.where(vn_pre < 0.0, -self.restitution * vn_pre, 0.0)
            target_vn = np.maximum(vn_new, bounce)
            rel_new = rel_new + (target_vn - vn_new)[:, None] * nrm
            # Coulomb friction on the tangential part
            vt = rel_new - np.sum(rel_new * nrm, axis=1)[:, None] * nrm
            vt_mag = np.linalg.norm(vt, axis=1)
            dvn = np.abs(np.maximum(target_vn, 0.0) - np.minimum(vn_pre, 0.0))
            limit = self.friction * dvn
            scale = np.where(vt_mag > 1e-12, np.minimum(limit, vt_mag) / np.maximum(vt_mag, 1e-12), 0.0)
            rel_new = rel_new - scale[:, None] * vt
            v_new[act] = rel_new + ovel

        # viscous damping (material dissipation)
        v_new *= float(np.exp(-self.damping_rate * dt))

        # determinism / safety guards
        x = np.where(np.isfinite(x), x, x_prev)
        v_new = np.where(np.isfinite(v_new), v_new, 0.0)
        x[:, 0] = np.clip(x[:, 0], -4.0, 4.0)
        x[:, 1] = np.clip(x[:, 1], -4.0, 4.0)
        x[:, 2] = np.clip(x[:, 2], -1.0, 6.0)
        v_new = np.clip(v_new, -40.0, 40.0)

        self.x = x
        self.v = v_new
        self.max_penetration = max(self.max_penetration, contact["pen"])
        self.steps_taken += 1


    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing times")
        span = target - self.time
        if span > 1e-12:
            sample_dt = float(self.cfg["sample_dt"])
            nsub = max(1, int(round(self.substeps * span / sample_dt)))
            dt = span / nsub
            for _ in range(nsub):
                self.push_center, self.push_vel = self._pusher_at(self.time + dt)
                self._substep(dt)
                self.time += dt
            self.time = target
        else:
            self.push_center, self.push_vel = self._pusher_at(self.time)
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        fgrad = self._deformation_gradient(self.x)
        det = np.linalg.det(fgrad)
        det = np.where(np.isfinite(det), det, 1.0)
        self.max_det = max(self.max_det, float(det.max()))
        self.min_det = min(self.min_det, float(det.min()))
        self.inverted_elements = int(np.count_nonzero(det <= 0.0))

        gap = self._obstacle_distance(self.x)
        vol = self._volume(self.x)
        ratio = vol / max(self.rest_volume, _TINY)
        self.min_volume_ratio = min(self.min_volume_ratio, float(ratio))

        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(t),
            position=np.ascontiguousarray(self.x, dtype=float),
            velocity=np.ascontiguousarray(self.v, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=float),
                "deformation_gradient": np.ascontiguousarray(fgrad, dtype=float),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(ext_force, dtype=float),
                "contact_signed_distance": np.ascontiguousarray(gap, dtype=float),
            },
            scalars={
                "contact_length_scale": float(self.contact_h),
                "enclosed_volume": float(vol),
                "volume_ratio": float(ratio),
                "pusher_z": float(self.push_center[2]),
                "min_jacobian": float(det.min()),
                "constraint_residual": float(self.last_residual),
                "max_penetration": float(self.max_penetration),
                "total_mass": float(self.mass.sum()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "XPBD compliant-constraint deformable solid (self-implemented)",
            "vertices": int(self.n),
            "edges": int(self.edges.shape[0]),
            "hinges": int(self.bend.shape[0]),
            "substeps_per_sample": int(self.substeps),
            "iterations_per_substep": int(self.iters),
            "substeps_taken": int(self.steps_taken),
            "rest_volume": float(self.rest_volume),
            "min_volume_ratio": float(self.min_volume_ratio),
            "min_jacobian": float(self.min_det),
            "max_jacobian": float(self.max_det),
            "inverted_elements": int(self.inverted_elements),
            "max_penetration": float(self.max_penetration),
            "final_constraint_residual": float(self.last_residual),
            "seed": int(self.seed),
        }
