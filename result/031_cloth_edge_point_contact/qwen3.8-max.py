"""Physically-based cloth solver for 031_cloth_edge_point_contact.

XPBD mass-spring cloth (structural + shear + bending distance constraints)
with analytic signed-distance collision against the ground box, the sharp
edge box, and the sharp point cone, Coulomb friction on the contact
displacement, and KD-tree based vertex-vertex self-collision repulsion.
All motion is produced by gravity + constraints + contact; nothing scripted.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _unit(vec):
    """Row-wise unit vectors; degenerate rows map to +z."""
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    out = vec / np.maximum(n, 1e-9)
    bad = (n[:, 0] < 1e-9)
    if np.any(bad):
        out[bad] = np.array([0.0, 0.0, 1.0])
    return out


def _box_sdf(p, c, h):
    """Vectorized box SDF; returns (signed distance, closest point on box)."""
    q = p - c
    aq = np.abs(q)
    outside = np.any(aq > h, axis=1)
    d_out = np.linalg.norm(np.maximum(aq - h, 0.0), axis=1)
    d_in = np.max(aq - h, axis=1)  # negative when strictly inside
    d = np.where(outside, d_out, d_in)
    closest_out = c + np.clip(q, -h, h)
    # For inside points, the closest surface point is on the nearest face.
    ax = np.argmax(aq - h, axis=1)
    idx = np.arange(len(p))
    inner_closest = c + q.copy()
    inner_closest[idx, ax] = c[ax] + np.sign(q[idx, ax]) * h[ax]
    closest = np.where(outside[:, None], closest_out, inner_closest)
    return d, closest


def _cone_sdf_closest(p, apex, r, h):
    """Truncated cone: apex at `apex` (top), base disk radius r at z=apex_z-h.

    Returns (signed distance, closest surface point). The 2D profile in
    (rho, z) is the segment from (0, 0) (apex) to (r, -h) (rim) plus the
    base disk. Sign is negative strictly inside the cone solid.
    """
    d = p - apex
    rho = np.hypot(d[:, 0], d[:, 1])
    z = d[:, 2]
    L = np.hypot(r, h)
    # closest point on the lateral segment from (0,0) to (r,-h)
    tt = np.clip((rho * r - z * h) / (L * L), 0.0, 1.0)
    lat = np.stack([tt * r, -tt * h], axis=1)
    dlat = np.linalg.norm(np.stack([rho, z], axis=1) - lat, axis=1)
    # closest point on the base disk (z = -h, radius r)
    dbase = np.sqrt(np.maximum(rho - r, 0.0) ** 2 + (z + h) ** 2)
    use_lat = dlat <= dbase
    dist = np.where(use_lat, dlat, dbase)
    # inside test: point lies within the truncated cone solid
    s = r / h
    inside = (z <= 0.0) & (z >= -h) & (rho <= s * (-z))
    dist = np.where(inside, -dist, dist)
    tprof = np.where(use_lat[:, None], lat, np.stack([np.clip(rho, 0.0, r),
                                                      np.full_like(rho, -h)], axis=1))
    ang = np.arctan2(d[:, 1], np.where(rho < 1e-9, 1.0, d[:, 0]))
    closest = np.column_stack([apex[0] + tprof[:, 0] * np.cos(ang),
                               apex[1] + tprof[:, 0] * np.sin(ang),
                               apex[2] + tprof[:, 1]])
    return dist, closest


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg
        init = scene.initial_state()
        self.p = np.array(init.position, dtype=float)
        self.v = np.array(init.velocity, dtype=float)
        self.n = self.p.shape[0]
        nx, ny = map(int, cfg["cloth"]["grid_shape"])
        self.nx, self.ny = nx, ny
        total_mass = float(cfg["cloth"]["total_mass"])
        self.m = np.full(self.n, total_mass / self.n)
        self.w = 1.0 / self.m
        self.rest = np.array(init.position, dtype=float)
        self.uv = np.array(init.attributes["uv"], dtype=float)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt = float(cfg["sample_dt"])
        self.substeps = 8
        self.h = self.dt / self.substeps
        self.iters = 8
        self.mu = float(cfg["friction"]["coefficient"])
        self.thickness = float(cfg["cloth"]["thickness"])
        self.margin = float(cfg["contact"]["thickness_offset"])
        self.t = 0.0

        # Static obstacles
        self.ground_c = np.asarray(cfg["ground"]["center"], dtype=float)
        self.ground_h = np.asarray(cfg["ground"]["half_extents"], dtype=float)
        self.edge_c = np.asarray(cfg["sharp_edge"]["center"], dtype=float)
        self.edge_h = np.asarray(cfg["sharp_edge"]["half_extents"], dtype=float)
        sp = cfg["sharp_point"]
        self.cone_apex = np.asarray(sp["tip"], dtype=float)
        self.cone_r = float(sp["base_radius"])
        self.cone_h = float(sp["height"])

        self._build_constraints()
        self._build_exclusions()
        self._build_f_basis()

        self.dmin = 0.024          # self-collision rest separation
        self.dnear = 0.035         # self-collision broadphase radius
        self.signed = np.full(self.n, 1.0)
        self.pair_count = 0
        self.substep_count = 0

    # ---------------- setup ----------------
    def _build_constraints(self):
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        pairs = []
        comps = []
        # structural
        pairs.append(np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 2e-6))
        pairs.append(np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 2e-6))
        # shear (cell diagonals, matching triangle diagonals)
        pairs.append(np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 2e-6))
        pairs.append(np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 2e-6))
        # bending (two-neighbor springs, much softer)
        pairs.append(np.stack([idx[:, :-2].ravel(), idx[:, 2:].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 3e-4))
        pairs.append(np.stack([idx[:-2, :].ravel(), idx[2:, :].ravel()], axis=1))
        comps.append(np.full(pairs[-1].shape[0], 3e-4))
        self.ci = np.concatenate(pairs, axis=0)
        self.compliance = np.concatenate(comps)
        self.rest_len = np.linalg.norm(
            self.rest[self.ci[:, 0]] - self.rest[self.ci[:, 1]], axis=1)
        self.w_i = self.w[self.ci[:, 0]]
        self.w_j = self.w[self.ci[:, 1]]
        self.w_ij = self.w_i + self.w_j
        self.alpha = self.compliance / (self.h * self.h)
        # Jacobi relaxation: scale accumulated corrections by inverse valence
        val = np.bincount(self.ci.ravel(), minlength=self.n).astype(float)
        self.relax = (1.0 / np.maximum(val, 1.0))[:, None]

    def _build_exclusions(self):
        nx = self.nx
        offs = [dy * nx + dx for dy in range(-2, 3) for dx in range(-2, 3)]
        self.excl = np.array(sorted(set(int(o) for o in offs)), dtype=np.int64)

    def _build_f_basis(self):
        rest2 = self.rest[:, :2]
        nx, ny = self.nx, self.ny
        tri = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                b, c, d = a + 1, a + nx, a + nx + 1
                tri.append((a, b, d))
                tri.append((a, d, c))
        tri = np.asarray(tri, dtype=np.int64)
        self.tri = tri
        p0 = rest2[tri[:, 0]]
        e1 = rest2[tri[:, 1]] - p0
        e2 = rest2[tri[:, 2]] - p0
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        inv_det = 1.0 / det
        # Dm = [[e1x, e2x], [e1y, e2y]]  (columns e1, e2); store its inverse
        self.Dm_inv = np.stack([
            np.stack([e2[:, 1] * inv_det, -e2[:, 0] * inv_det], axis=1),
            np.stack([-e1[:, 1] * inv_det, e1[:, 0] * inv_det], axis=1),
        ], axis=1)  # [T,2,2]
        self.tri_area = 0.5 * np.abs(det)

    # ---------------- collision helpers ----------------
    def _obstacle_project(self, x):
        """Push points outside all obstacles by `margin`.

        Returns (x, signed_min_distance, outward_contact_normal).
        """
        margin = self.margin
        n = len(x)
        best_d = np.full(n, np.inf)
        best_cp = x.copy()
        for sdf in (
            lambda pp: _box_sdf(pp, self.ground_c, self.ground_h),
            lambda pp: _box_sdf(pp, self.edge_c, self.edge_h),
            lambda pp: _cone_sdf_closest(pp, self.cone_apex, self.cone_r, self.cone_h),
        ):
            d, cp = sdf(x)
            upd = d < best_d
            best_d = np.where(upd, d, best_d)
            best_cp = np.where(upd[:, None], cp, best_cp)
            bad = d < margin
            if np.any(bad):
                x[bad] = cp[bad] + margin * _unit(x[bad] - cp[bad])
        normal = _unit(x - best_cp)
        return x, best_d, normal

    # ---------------- stepping ----------------
    def _substep(self):
        h = self.h
        self.v += h * self.gravity
        pred = self.p + h * self.v
        # XPBD distance-constraint iterations (Jacobi-style accumulation)
        for _ in range(self.iters):
            diff = pred[self.ci[:, 0]] - pred[self.ci[:, 1]]
            dist = np.linalg.norm(diff, axis=1)
            safe = np.maximum(dist, 1e-9)
            C = dist - self.rest_len
            lam = -C / (self.w_ij + self.alpha)
            corr = (lam / safe)[:, None] * diff
            dx = np.zeros_like(pred)
            np.add.at(dx, self.ci[:, 0], self.w_i[:, None] * corr)
            np.add.at(dx, self.ci[:, 1], -self.w_j[:, None] * corr)
            pred += dx * self.relax * 2.0
        # cloth self-contact
        self._self_collision(pred)
        # obstacle contact: positional projection + Coulomb friction
        pred, signed, normal = self._obstacle_project(pred)
        # face-centroid pass: catches the thin blade slipping between vertices
        self._centroid_collide(pred)
        self.signed = signed
        pre_dp = pred - self.p
        contact = signed < self.margin + 1e-4
        dp = pre_dp.copy()
        if np.any(contact):
            dp_c = dp[contact]
            n_c = normal[contact]
            dn = np.einsum("ij,ij->i", dp_c, n_c)
            dt_vec = dp_c - dn[:, None] * n_c
            dt_len = np.linalg.norm(dt_vec, axis=1)
            scale = np.clip(self.mu * np.abs(dn) / np.maximum(dt_len, 1e-9), 0.0, 1.0)
            dp[contact] = dp_c - dt_vec * scale[:, None]
        self.p = pred
        self.v = dp / h
        self.v *= 0.999
        speed = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v *= np.minimum(1.0, 8.0 / np.maximum(speed, 1e-9))
        self.substep_count += 1

    def _centroid_collide(self, pred):
        tri = self.tri
        margin = self.margin
        for _ in range(2):
            cent = (pred[tri[:, 0]] + pred[tri[:, 1]] + pred[tri[:, 2]]) / 3.0
            any_bad = False
            for sdf in (
                lambda pp: _box_sdf(pp, self.ground_c, self.ground_h),
                lambda pp: _box_sdf(pp, self.edge_c, self.edge_h),
                lambda pp: _cone_sdf_closest(pp, self.cone_apex, self.cone_r, self.cone_h),
            ):
                d, cp = sdf(cent)
                bad = d < margin
                if not np.any(bad):
                    continue
                any_bad = True
                idx = np.nonzero(bad)[0]
                delta = (cp[bad] + margin * _unit(cent[bad] - cp[bad]) - cent[bad]) / 3.0
                delta = np.clip(delta, -0.02, 0.02)
                for k in range(3):
                    np.add.at(pred, tri[idx, k], delta)
            if not any_bad:
                break

    def _self_collision(self, pred):
        tree = cKDTree(pred)
        pairs = tree.query_pairs(self.dnear, output_type="ndarray")
        if len(pairs) == 0:
            self.pair_count = 0
            return
        off = pairs[:, 1] - pairs[:, 0]
        keep = ~np.isin(off, self.excl)
        pairs = pairs[keep]
        self.pair_count = len(pairs)
        if len(pairs) == 0:
            return
        alpha = 0.02 / (self.h * self.h)
        for _ in range(2):
            i = pairs[:, 0]
            j = pairs[:, 1]
            diff = pred[i] - pred[j]
            dist = np.linalg.norm(diff, axis=1)
            act = dist < self.dmin
            if not np.any(act):
                break
            i = i[act]; j = j[act]
            diff = diff[act]; dist = dist[act]
            safe = np.maximum(dist, 1e-9)
            C = dist - self.dmin
            wij = self.w[i] + self.w[j]
            lam = -C / (wij + alpha)
            corr = 0.5 * (lam / safe)[:, None] * diff
            dx = np.zeros_like(pred)
            np.add.at(dx, i, self.w[i, None] * corr)
            np.add.at(dx, j, -self.w[j, None] * corr)
            pred += np.clip(dx, -0.01, 0.01)

    # ---------------- public API ----------------
    def advance_to(self, time: float):
        target = float(time)
        nsteps = max(int(round((target - self.t) / self.h)), 0)
        for _ in range(nsteps):
            self._substep()
        self.t = target
        return self._frame(target)

    def _deformation_gradients(self):
        tri = self.tri
        p0 = self.p[tri[:, 0]]
        Ds = np.empty((len(tri), 3, 2))
        Ds[:, :, 0] = self.p[tri[:, 1]] - p0
        Ds[:, :, 1] = self.p[tri[:, 2]] - p0
        F2 = np.einsum("tij,tjk->tik", Ds, self.Dm_inv)  # [T,3,2]
        F = np.zeros((len(tri), 3, 3))
        F[:, :, :2] = F2
        F[:, 2, 2] = 1.0
        Fw = F * self.tri_area[:, None, None]
        Fv = np.zeros((self.n, 3, 3))
        wsum = np.zeros(self.n)
        for k in range(3):
            np.add.at(Fv, tri[:, k], Fw)
            np.add.at(wsum, tri[:, k], self.tri_area)
        Fv /= np.maximum(wsum, 1e-12)[:, None, None]
        return Fv

    def _frame(self, time: float):
        Fv = self._deformation_gradients()
        mass = self.m.copy()
        frame = StateFrame(
            float(time),
            self.p.copy(),
            self.v.copy(),
            mass=mass,
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": mass,
                "deformation_gradient": Fv,
            },
            fields={
                "contact_signed_distance": self.signed.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "substeps": self.substep_count,
            "last_self_contact_pairs": int(self.pair_count),
            "solver": "xpbd_mass_spring_cloth",
        }
