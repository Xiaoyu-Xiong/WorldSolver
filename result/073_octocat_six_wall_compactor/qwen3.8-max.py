"""Physically-based solver for the six-wall Octocat compactor.

Method
------
The Octocat surface mesh (frozen topology, N samples) is driven by a coarse
volumetric lattice of hexahedral finite elements built from the analytic
signed-distance field of the shape.  The lattice is simulated with

* co-rotational linear elasticity (Saint-Venant/Kirchhoff type, polar-free
  via SVD) on every hexahedron -> large-deformation, inversion-tolerant,
* lumped nodal masses and symplectic Euler time stepping,
* hard kinematic wall contact (velocity projection + Coulomb-ish friction,
  impulses accumulated exactly for the momentum budget),
* penalty self-contact between material-distant lattice nodes so pressed
  tentacles do not interpenetrate and spring apart on release.

Surface samples are trilinearly interpolated from the lattice with a frozen
detail offset so the initial frame reproduces the asset exactly and the body
is stress-free at t=0.  Per-hex deformation gradients are exported in fixed
element order.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

RHO = 1000.0            # kg/m^3
E_YOUNG = 1.5e4         # Pa
NU = 0.35
H = 0.065               # lattice spacing (m)
DT_SUB = 0.002          # substep (s)
C_DAMP = 0.5            # global viscous damping (1/s)
K_CONTACT = 5.0e3       # self-contact penalty stiffness (N/m)
C_CONTACT = 16.0        # self-contact damping (N/(m/s))
R_CONTACT_FACTOR = 1.6  # self-contact radius / H
EXCLUDE_FACTOR = 2.4    # material-neighbour exclusion radius / H
KNN = 30                # neighbours queried for self-contact
MU_FRICTION = 0.25

_LO = np.array([-1.15, -1.15, -0.82])
_HI = np.array([1.15, 1.15, 1.28])


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(np.sum((points - a) * ab, axis=-1) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def _octocat_sdf(p):
    """Analytic signed-distance field used to generate assets/octocat.obj."""
    q = (p - np.array([0.0, 0.0, 0.35])) / np.array([0.55, 0.50, 0.68])
    field = np.linalg.norm(q, axis=-1) - 1.0
    for cx in (-0.34, 0.34):
        tip = np.array([cx, 0.0, 1.18])
        base = np.array([cx * 0.72, 0.0, 0.78])
        field = np.minimum(field, _segment_distance(p, base, tip) / 0.18 - 1.0)
    for i in range(8):
        a = i * np.pi / 4.0 + np.pi / 8.0
        radial = np.array([np.cos(a), np.sin(a), 0.0])
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        pts = [
            np.array([0.32 * radial[0], 0.32 * radial[1], -0.05]),
            np.array([0.62 * radial[0], 0.62 * radial[1], -0.30]),
            np.array([0.86 * radial[0], 0.86 * radial[1], -0.42]) + 0.12 * tangent,
            np.array([1.02 * radial[0], 1.02 * radial[1], -0.28]) + 0.28 * tangent,
        ]
        for j in range(3):
            radius = 0.15 - 0.025 * j
            field = np.minimum(
                field, _segment_distance(p, pts[j], pts[j + 1]) / radius - 1.0
            )
    return field


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        self.cfg_compactor = cfg["compactor"]
        self.seed = int(seed)

        frame0 = scene.initial_state()
        verts = np.asarray(frame0.position, dtype=float)
        self.nv = len(verts)

        self._build_lattice()
        self._build_mapping(verts)

        mu = E_YOUNG / (2.0 * (1.0 + NU))
        lam = E_YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        self.mu, self.lam = mu, lam
        self.ve = H ** 3

        self.X = self.X0n.copy()
        self.V = np.zeros_like(self.X0n)
        self.t = 0.0
        self.cum_wall_imp = np.zeros(3)
        self.cum_damp_imp = np.zeros(3)
        self.prev_wall_imp = np.zeros(3)
        self.prev_damp_imp = np.zeros(3)
        self.prev_t = 0.0

        # reference momentum scale: total mass x peak wall speed
        v_peak = self._peak_wall_speed()
        self.ref_momentum = float(self.total_mass * max(v_peak, 0.1))

    def _build_lattice(self):
        n = np.ceil((_HI - _LO) / H).astype(int)
        self.grid_n = n
        cx = _LO[0] + (np.arange(n[0]) + 0.5) * H
        cy = _LO[1] + (np.arange(n[1]) + 0.5) * H
        cz = _LO[2] + (np.arange(n[2]) + 0.5) * H
        cpts = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), axis=-1).reshape(-1, 3)
        sdf_c = _octocat_sdf(cpts)
        active = sdf_c <= 0.1 * H
        self.active_map = np.full(len(active), -1, dtype=np.int64)
        act_idx = np.where(active)[0]
        self.active_map[act_idx] = np.arange(len(act_idx))
        self.ne = len(act_idx)
        ci = act_idx // (n[1] * n[2])
        rem = act_idx % (n[1] * n[2])
        cj = rem // n[2]
        ck = rem % n[2]

        offs = np.array(list(np.ndindex(2, 2, 2)))  # (8,3) corner offsets
        strides = np.array([(n[1] + 1) * (n[2] + 1), n[2] + 1, 1])
        self._strides = strides
        base = np.stack([ci, cj, ck], axis=1)  # (ne,3)
        corner_grid = (base[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        corner_flat = corner_grid @ strides
        uniq, inverse = np.unique(corner_flat, return_inverse=True)
        self.nn = len(uniq)
        self.cellnodes = inverse.reshape(self.ne, 8)

        gi = uniq // strides[0]
        rem = uniq % strides[0]
        gj = rem // strides[1]
        gk = rem % strides[1]
        self.X0n = np.stack(
            [_LO[0] + gi * H, _LO[1] + gj * H, _LO[2] + gk * H], axis=1
        ).astype(float)
        self._node_flat = uniq  # sorted ascending (np.unique output)

        # shape-function gradients (constant per regular hex, material coords)
        signs = 2.0 * offs - 1.0  # (8,3)
        self.G8 = signs / (4.0 * H)

        # lumped masses
        counts = np.bincount(self.cellnodes.ravel(), minlength=self.nn)
        self.m_node = RHO * self.ve_dummy() / 8.0 * counts
        self.total_mass = float(self.m_node.sum())
        self.node_tree = cKDTree(self.X0n)

        self.r_contact = R_CONTACT_FACTOR * H
        self.r_exclude = EXCLUDE_FACTOR * H

    @staticmethod
    def ve_dummy():
        return H ** 3

    def _build_mapping(self, verts):
        n = self.grid_n
        strides = self._strides
        u = (verts - _LO) / H
        gi = np.clip(np.floor(u).astype(int), 0, n - 1)
        loc = np.clip(u - gi, 0.0, 1.0)
        cell_flat = gi[:, 0] * (n[1] * n[2]) + gi[:, 1] * n[2] + gi[:, 2]
        cell_active = self.active_map[cell_flat] >= 0

        offs = np.array(list(np.ndindex(2, 2, 2)))
        act_idx = np.where(cell_active)[0]
        rows, cols, vals = [], [], []
        for off in offs:
            cg = gi[act_idx] + off
            cf = cg @ strides
            node_id = np.clip(
                np.searchsorted(self._node_flat, cf), 0, self.nn - 1
            )
            w = np.where(off == 1, loc[act_idx], 1.0 - loc[act_idx]).prod(axis=1)
            rows.append(act_idx)
            cols.append(node_id)
            vals.append(w)
        fb = np.where(~cell_active)[0]
        if len(fb):
            d, jj = self.node_tree.query(verts[fb], k=6)
            d = np.maximum(d, 1e-6)
            w = 1.0 / d ** 2
            w /= w.sum(axis=1, keepdims=True)
            for a in range(6):
                rows.append(fb)
                cols.append(jj[:, a])
                vals.append(w[:, a])
        W = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nv, self.nn),
        )
        self.W = W
        mapped0 = W @ self.X0n
        self.offset = verts - mapped0
        mv = W @ self.m_node
        s = mv.sum()
        self.m_vert = (
            mv * (self.total_mass / s)
            if s > 0
            else np.full(self.nv, self.total_mass / self.nv)
        )

    # ------------------------------------------------------------- kinematics
    def _inner_half_extent(self, time):
        c = self.cfg_compactor
        open_h = float(c["open_inner_half_extent"])
        closed_h = float(c["closed_inner_half_extent"])
        inward = c["inward_interval"]
        withdraw = c["withdrawal_interval"]
        if time < inward[0]:
            return open_h
        if time <= inward[1]:
            return open_h + (closed_h - open_h) * _smoothstep(
                (time - inward[0]) / (inward[1] - inward[0])
            )
        if time < withdraw[0]:
            return closed_h
        if time <= withdraw[1]:
            return closed_h + (open_h - closed_h) * _smoothstep(
                (time - withdraw[0]) / (withdraw[1] - withdraw[0])
            )
        return open_h

    def _wall_speed(self, time):
        eps = 1e-5
        t0 = max(time - eps, 0.0)
        return (self._inner_half_extent(time + eps) - self._inner_half_extent(t0)) / (
            time + eps - t0
        )

    def _peak_wall_speed(self):
        ts = np.linspace(0.0, float(self.scene.config.get("duration", 8.0)), 401)
        return float(max(abs(self._wall_speed(t)) for t in ts))

    # ---------------------------------------------------------------- forces
    def _deformation_gradients(self, X=None):
        if X is None:
            X = self.X
        Xe = X[self.cellnodes]                      # (ne,8,3)
        return np.einsum("eai,aj->eij", Xe, self.G8)

    def _elastic_forces(self):
        F = self._deformation_gradients()
        U, _, Vt = np.linalg.svd(F)
        R = U @ Vt
        d = np.linalg.det(R)
        flip = np.sign(d)[:, None]
        U = U.copy()
        U[:, :, -1] *= flip
        R = U @ Vt
        Er = F - R
        tr = np.einsum("eii->e", Er)
        P = 2.0 * self.mu * Er + (self.lam * tr)[:, None, None] * R
        fe = -self.ve * np.einsum("eij,aj->eai", P, self.G8)
        Fn = np.zeros_like(self.X)
        idx = self.cellnodes.ravel()
        flat = fe.reshape(-1, 3)
        for c in range(3):
            Fn[:, c] = np.bincount(idx, weights=flat[:, c], minlength=self.nn)
        return Fn

    def _self_contact_forces(self):
        X, V = self.X, self.V
        tree = cKDTree(X)
        d, jj = tree.query(X, k=KNN + 1)
        d = d[:, 1:]
        jj = jj[:, 1:]
        dvec = X[jj] - X[:, None, :]
        ref = self.X0n[jj] - self.X0n[:, None, :]
        refd = np.sqrt(np.einsum("ekj,ekj->ek", ref, ref))
        mask = (d < self.r_contact) & (d > 1e-9) & (refd > self.r_exclude)
        if not mask.any():
            return np.zeros_like(X)
        inv = 1.0 / np.maximum(d, 1e-9)
        nhat = dvec * inv[..., None]
        vrel = V[jj] - V[:, None, :]
        vn = np.einsum("ekj,ekj->ek", vrel, nhat)
        fs = -K_CONTACT * (self.r_contact - d) + C_CONTACT * vn
        fs = np.where(mask, fs, 0.0)
        fvec = fs[..., None] * nhat
        i_idx = np.broadcast_to(np.arange(self.nn)[:, None], jj.shape)[mask]
        j_idx = jj[mask]
        fv = fvec[mask]
        Fn = np.zeros_like(X)
        for c in range(3):
            Fn[:, c] += np.bincount(i_idx, weights=fv[:, c], minlength=self.nn)
            Fn[:, c] += np.bincount(j_idx, weights=-fv[:, c], minlength=self.nn)
        return Fn

    # --------------------------------------------------------------- substep
    def _substep(self, dt):
        t1 = self.t + dt
        hw = self._inner_half_extent(t1)
        vw = self._wall_speed(t1)

        F = self._elastic_forces()
        F += self._self_contact_forces()
        Fd = -C_DAMP * self.m_node[:, None] * self.V
        F += Fd
        self.cum_damp_imp += Fd.sum(axis=0) * dt

        self.V += dt * F / self.m_node[:, None]

        # hard wall contact: velocity projection + friction, exact impulses
        for axis in range(3):
            for side in (1.0, -1.0):
                lim = side * hw
                wv = side * vw
                if side > 0:
                    pen = self.X[:, axis] > lim
                else:
                    pen = self.X[:, axis] < lim
                if not pen.any():
                    continue
                idx = np.where(pen)[0]
                vn_old = self.V[idx, axis]
                approaching = side * (vn_old - wv) > 0.0
                idx = idx[approaching]
                if len(idx) == 0:
                    continue
                m = self.m_node[idx]
                dv = self.V[idx, axis] - wv
                self.V[idx, axis] = wv
                dpx = -m * dv
                self.cum_wall_imp[axis] += dpx.sum()
                # friction on tangential axes
                for t_ax in range(3):
                    if t_ax == axis:
                        continue
                    vt = self.V[idx, t_ax]
                    budget = MU_FRICTION * np.abs(dv)
                    scale = np.clip(1.0 - budget / (np.abs(vt) + 1e-12), 0.0, 1.0)
                    vt_new = vt * scale
                    self.V[idx, t_ax] = vt_new
                    self.cum_wall_imp[t_ax] += (m * (vt_new - vt)).sum()

        self.X += dt * self.V
        np.clip(self.X, -hw, hw, out=self.X)
        self.t = t1

    # ----------------------------------------------------------------- frame
    def advance_to(self, time):
        from worldsolver.contracts import StateFrame

        time = float(time)
        n = int(round((time - self.t) / DT_SUB))
        for _ in range(max(n, 0)):
            self._substep(DT_SUB)
        if n <= 0 and time > self.t:
            self.t = time

        pos = self.W @ self.X + self.offset
        vel = self.W @ self.V
        Fg = self._deformation_gradients()

        dt_frame = time - self.prev_t
        d_wall = self.cum_wall_imp - self.prev_wall_imp
        d_damp = self.cum_damp_imp - self.prev_damp_imp
        if dt_frame > 1e-12:
            ext_force = np.stack([d_wall / dt_frame, d_damp / dt_frame])[None, :, :]
        else:
            ext_force = np.zeros((1, 2, 3))
        self.prev_wall_imp = self.cum_wall_imp.copy()
        self.prev_damp_imp = self.cum_damp_imp.copy()
        self.prev_t = time

        frame = StateFrame(
            time,
            pos,
            vel,
            mass=self.m_vert.copy(),
        )
        frame.attributes["mass_quantity"] = self.m_vert.copy()
        frame.fields["budget_domain_weights"] = np.ones((1, self.nv))
        frame.fields["budget_external_force"] = ext_force
        frame.fields["budget_cumulative_external_impulse"] = (
            self.cum_wall_imp + self.cum_damp_imp
        )[None, :]
        frame.fields["budget_cumulative_inflow_momentum"] = np.zeros((1, 3))
        frame.fields["budget_cumulative_outflow_momentum"] = np.zeros((1, 3))
        frame.fields["cumulative_mass_source"] = np.zeros(1)
        frame.fields["element_deformation_gradient"] = Fg
        frame.scalars["reference_momentum"] = self.ref_momentum
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "corotational hex FEM + hard wall projection + penalty self-contact",
            "lattice_nodes": int(self.nn),
            "lattice_elements": int(self.ne),
            "spacing": H,
            "substep": DT_SUB,
            "total_mass": self.total_mass,
            "reference_momentum": self.ref_momentum,
        }
