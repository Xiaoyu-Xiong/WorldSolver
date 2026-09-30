"""Physically-based cloth solver: mass-spring/PBD sheet through a rigid funnel.

Method
------
Position-based dynamics (PBD) on the 17x17 cloth grid:
  * structural / shear / bending distance constraints (Jacobi solves),
  * vertex self-contact repulsion over non-adjacent vertex pairs,
  * vertex-triangle collision against the static funnel mesh and an
    analytic ground plane, with simple Coulomb-style tangential friction,
  * symplectic substepping (10 substeps per output sample).

All required StateFrame fields are exported: per-vertex mass, position,
velocity, external force, right-Cauchy-Green surface deformation measure
(2x2 per vertex), signed contact distances, cumulative mass source (0),
and the frozen contact length scale (cloth thickness).
"""
from __future__ import annotations

import os
import numpy as np

from worldsolver.contracts import StateFrame

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _closest_points_on_triangles(p, a, b, c):
    """Vectorized closest point on triangle (Ericson, Real-Time Collision)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp_ = p - c
    d5 = np.einsum("ij,ij->i", ab, cp_)
    d6 = np.einsum("ij,ij->i", ac, cp_)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    closest = a.copy()
    un = np.ones(len(p), dtype=bool)

    m = un & (d1 <= 0.0) & (d2 <= 0.0)
    closest[m] = a[m]
    un &= ~m

    m = un & (d3 >= 0.0) & (d4 <= d3)
    closest[m] = b[m]
    un &= ~m

    m = un & (d6 >= 0.0) & (d5 <= d6)
    closest[m] = c[m]
    un &= ~m

    m = un & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    den = d1 - d3
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = d1 / den
    closest[m] = a[m] + ab[m] * t[m, None]
    un &= ~m

    m = un & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    den = d2 - d6
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = d2 / den
    closest[m] = a[m] + ac[m] * t[m, None]
    un &= ~m

    m = un & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    den = (d4 - d3) + (d5 - d6)
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (d4 - d3) / den
    closest[m] = b[m] + (c[m] - b[m]) * t[m, None]
    un &= ~m

    m = un.copy()
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = vb / den
    w = vc / den
    closest[m] = a[m] + ab[m] * v[m, None] + ac[m] * w[m, None]
    return closest


class BenchmarkSolver:
    # ---- tunables -------------------------------------------------------
    SUBSTEPS = 10          # physics substeps per output sample
    ITERATIONS = 6         # constraint iterations per substep
    K_STRETCH = 0.8
    K_SHEAR = 0.6
    K_BEND = 0.12
    K_SELF = 0.5           # self-contact relaxation
    SELF_RADIUS = 0.030    # ~2x cloth thickness
    FUNNEL_RADIUS = 0.028  # collision margin around funnel surface
    GROUND_RADIUS = 0.02
    FRICTION = 0.5         # tangential velocity damping at contacts
    V_MAX = 20.0

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        cloth = cfg["cloth"]
        self.thickness = float(cloth.get("thickness", 0.015))
        frame0 = scene.initial_state()
        self.x0 = np.asarray(frame0.position, dtype=float).copy()
        self.X = self.x0.copy()          # rest (material) positions
        n = len(self.X)
        self.n = n
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.inv_mass = np.where(self.mass > 0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)
        nx, ny = map(int, cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        self.boundary = np.asarray(frame0.attributes["boundary_vertex"]).copy()

        self.x = self.X.copy()
        self.v = np.zeros_like(self.X)
        self.t = 0.0
        self.f_ext = self.mass[:, None] * self.gravity[None, :]

        # ---- grid topology (matches scene.py ordering: idx = iy*nx + ix) ----
        ix, iy = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        ix = ix.ravel(); iy = iy.ravel()
        v0 = iy * nx + ix
        v1 = v0 + 1
        v2 = v0 + nx
        v3 = v2 + 1
        self.tris = np.concatenate([np.stack([v0, v2, v1], 1),
                                    np.stack([v1, v2, v3], 1)], axis=0)

        # ---- constraint sets ----
        def make(i, j, k):
            rest = np.linalg.norm(self.X[j] - self.X[i], axis=1)
            return i, j, rest, k

        ii = np.arange(n).reshape(ny, nx)
        # structural
        s_i = np.concatenate([ii[:, :-1].ravel(), ii[:-1, :].ravel()])
        s_j = np.concatenate([ii[:, 1:].ravel(), ii[1:, :].ravel()])
        self.c_struct = make(s_i, s_j, self.K_STRETCH)
        # shear
        d_i = np.concatenate([ii[:-1, :-1].ravel(), ii[:-1, 1:].ravel()])
        d_j = np.concatenate([ii[1:, 1:].ravel(), ii[1:, :-1].ravel()])
        self.c_shear = make(d_i, d_j, self.K_SHEAR)
        # bending (skip 2 in each direction)
        b_i = np.concatenate([ii[:, :-2].ravel(), ii[:-2, :].ravel()])
        b_j = np.concatenate([ii[:, 2:].ravel(), ii[2:, :].ravel()])
        self.c_bend = make(b_i, b_j, self.K_BEND)

        # ---- self-contact pairs (exclude near-neighbours in rest shape) ----
        hmin = min(float(cfg["cloth"]["size"][0]) / (nx - 1),
                   float(cfg["cloth"]["size"][1]) / (ny - 1))
        I, J = np.triu_indices(n, k=1)
        drest = np.linalg.norm(self.X[J] - self.X[I], axis=1)
        keep = drest > 1.6 * hmin
        self.self_i, self.self_j = I[keep], J[keep]
        self.self_drest2 = drest[keep]

        # ---- static obstacles ----
        self._load_funnel()
        self.ground_z = 0.0  # top of ground box (translation -0.05 + half 0.05)

        # ---- precomputed deformation-gradient data ----
        t0, t1, t2 = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        dX1 = self.X[t1] - self.X[t0]      # (T,3) but z=0
        dX2 = self.X[t2] - self.X[t0]
        m1 = dX1[:, :2]
        m2 = dX2[:, :2]
        det = m1[:, 0] * m2[:, 1] - m1[:, 1] * m2[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        self.invDm = np.empty((len(det), 2, 2))
        self.invDm[:, 0, 0] = m2[:, 1] / det
        self.invDm[:, 0, 1] = -m2[:, 0] / det
        self.invDm[:, 1, 0] = -m1[:, 1] / det
        self.invDm[:, 1, 1] = m1[:, 0] / det
        self.tri_area = 0.5 * np.abs(det)
        self._vert_area = np.zeros(n)
        np.add.at(self._vert_area, t0, self.tri_area)
        np.add.at(self._vert_area, t1, self.tri_area)
        np.add.at(self._vert_area, t2, self.tri_area)
        self._vert_area = np.maximum(self._vert_area, 1e-12)
        self.contact_length_scale = self.thickness

        self._diag = {"steps": 0, "min_contact": 0.0}

    # ------------------------------------------------------------------
    def _load_funnel(self) -> None:
        path = os.path.join(_HERE, "assets", "funnel.obj")
        verts, faces = [], []
        with open(path) as fh:
            for line in fh:
                tok = line.split()
                if not tok:
                    continue
                if tok[0] == "v":
                    verts.append([float(t) for t in tok[1:4]])
                elif tok[0] == "f":
                    idx = [int(t.split("/")[0]) - 1 for t in tok[1:4]]
                    faces.append(idx)
        fv = np.asarray(verts, dtype=float)
        ff = np.asarray(faces, dtype=np.int64)
        self.f_a = fv[ff[:, 0]]
        self.f_b = fv[ff[:, 1]]
        self.f_c = fv[ff[:, 2]]
        nrm = np.cross(self.f_b - self.f_a, self.f_c - self.f_a)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        self.f_n = nrm / np.maximum(ln, 1e-12)

    # ------------------------------------------------------------------
    def _solve_constraints(self, p):
        for (ci, cj, rest, k) in (self.c_struct, self.c_shear, self.c_bend):
            d = p[cj] - p[ci]
            L = np.linalg.norm(d, axis=1)
            ok = L > 1e-12
            diff = np.zeros_like(L)
            diff[ok] = k * (L[ok] - rest[ok]) / L[ok]
            corr = d * (0.5 * diff)[:, None]
            np.add.at(p, ci, corr)
            np.add.at(p, cj, -corr)

    def _solve_self_contact(self, p):
        d = p[self.self_j] - p[self.self_i]
        L2 = np.einsum("ij,ij->i", d, d)
        r2 = self.SELF_RADIUS ** 2
        hit = L2 < r2
        if not hit.any():
            return
        idx = np.nonzero(hit)[0]
        L = np.sqrt(np.maximum(L2[idx], 1e-16))
        dirv = d[idx] / L[:, None]
        corr = (0.5 * self.K_SELF * (self.SELF_RADIUS - L))[:, None] * dirv
        np.add.at(p, self.self_i[idx], corr)
        np.add.at(p, self.self_j[idx], -corr)

    def _collide(self, p, cn, cmask):
        nv = len(p)
        # funnel: vertex vs all triangles
        a = np.repeat(self.f_a, nv, 0)
        b = np.repeat(self.f_b, nv, 0)
        c = np.repeat(self.f_c, nv, 0)
        q = np.tile(p, (len(a) // nv, 1))
        cp = _closest_points_on_triangles(q, a, b, c)
        dv = q - cp
        dist = np.linalg.norm(dv, axis=1)
        r = self.FUNNEL_RADIUS
        hit = dist < r
        if hit.any():
            hidx = np.nonzero(hit)[0]
            vidx = hidx % nv
            dh = np.maximum(dist[hidx], 1e-9)
            nhat = dv[hidx] / dh[:, None]
            degen = dist[hidx] < 1e-6
            if degen.any():
                fn = np.repeat(self.f_n, nv, 0)
                nhat[degen] = fn[hidx[degen]]
            corr = ((r - dist[hidx])[:, None]) * nhat
            np.add.at(p, vidx, corr)
            np.add.at(cn, vidx, nhat)
            cmask[vidx] = True
        # ground plane
        zg = self.ground_z + self.GROUND_RADIUS
        low = p[:, 2] < zg
        if low.any():
            lidx = np.nonzero(low)[0]
            p[lidx, 2] = zg
            cn[lidx, 2] += 1.0
            cmask[lidx] = True

    def _substep(self, dt):
        self.v += dt * self.gravity[None, :]
        sp = np.linalg.norm(self.v, axis=1)
        over = sp > self.V_MAX
        if over.any():
            self.v[over] *= (self.V_MAX / sp[over])[:, None]
        p = self.x + dt * self.v
        cn = np.zeros_like(p)
        cmask = np.zeros(len(p), dtype=bool)
        for _ in range(self.ITERATIONS):
            self._solve_constraints(p)
            self._solve_self_contact(p)
            self._collide(p, cn, cmask)
        self.v = (p - self.x) / dt
        # friction at contacts: damp tangential velocity
        if cmask.any():
            nrm = cn[cmask]
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.maximum(ln, 1e-12)
            vc = self.v[cmask]
            vn = np.einsum("ij,ij->i", vc, nrm)[:, None] * nrm
            vt = vc - vn
            self.v[cmask] = vn + vt * (1.0 - self.FRICTION)
        self.x = p
        self.t += dt
        self._diag["steps"] += 1

    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        t0, t1, t2 = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        dx1 = self.x[t1] - self.x[t0]
        dx2 = self.x[t2] - self.x[t0]
        F = (dx1[:, :, None] * self.invDm[:, None, 0, :] +
             dx2[:, :, None] * self.invDm[:, None, 1, :])
        C = np.einsum("tai,taj->tij", F, F)  # right Cauchy-Green, (T,2,2)
        w = self.tri_area
        out = np.zeros((self.n, 4))
        Cf = (C * w[:, None, None]).reshape(len(C), 4)
        np.add.at(out, t0, Cf)
        np.add.at(out, t1, Cf)
        np.add.at(out, t2, Cf)
        out /= self._vert_area[:, None]
        return out.reshape(self.n, 2, 2)

    def _signed_contact_distance(self):
        nv = self.n
        a = np.repeat(self.f_a, nv, 0)
        b = np.repeat(self.f_b, nv, 0)
        c = np.repeat(self.f_c, nv, 0)
        q = np.tile(self.x, (len(a) // nv, 1))
        cp = _closest_points_on_triangles(q, a, b, c)
        dv = q - cp
        dist = np.linalg.norm(dv, axis=1)
        fn = np.repeat(self.f_n, nv, 0)
        sgn = np.where(np.einsum("ij,ij->i", dv, fn) < 0.0, -1.0, 1.0)
        sd = sgn * dist
        sd = sd.reshape(len(a) // nv, nv)
        k = np.argmin(np.abs(sd), axis=0)
        d_funnel = sd[k, np.arange(nv)]
        d_ground = self.x[:, 2] - self.ground_z
        out = np.where(np.abs(d_ground) < np.abs(d_funnel), d_ground, d_funnel)
        self._diag["min_contact"] = float(out.min()) if nv else 0.0
        return out

    def _frame(self):
        return StateFrame(
            float(self.t),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "contact_signed_distance": self._signed_contact_distance(),
                "external_force": self.f_ext.copy(),
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        if time < self.t - 1e-12:
            raise ValueError("advance_to called with time in the past")
        target = time
        dt_sub = (1.0 / 30.0) / self.SUBSTEPS  # nominal substep
        while self.t < target - 1e-9:
            remaining = target - self.t
            k = max(1, int(np.ceil(remaining / (dt_sub + 1e-12))))
            dt = remaining / k if remaining < dt_sub * 1.5 else dt_sub
            self._substep(dt)
        self.t = target
        return self._frame()

    def diagnostics(self) -> dict:
        return dict(self._diag)
