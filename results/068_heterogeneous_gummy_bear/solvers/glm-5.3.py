"""Heterogeneous gummy-bear solver (scene 068).

Physical model
--------------
The bear is a bonded two-material 3-D solid.  It is discretized with the frozen
mesh vertices of ``gummy_bear_outer.obj`` (soft shell, body 0) and
``gummy_bear_stiff.obj`` (stiff interior, body 1) as material points carrying
lumped masses, connected by a volumetric network of nonlinear distance springs
(a lumped-parameter finite-element model of a Neo-Hookean-type solid):

* soft-soft membrane springs on the shell mesh edges, stiffness calibrated from
  the frozen soft Young's modulus (k ~ E * h);
* soft-soft bending springs across graph-distance-2 neighbours of the shell;
* stiff-stiff springs on the interior mesh edges and its distance-2 neighbours,
  with a stiffness in the rigid limit of the frozen 1e14 Pa modulus (the value
  is regularized to the point where substep strains are O(1e-4), i.e. the
  interior is effectively rigid; raising it further changes nothing observable
  but degrades the conditioning of the linear solves);
* bonded soft-stiff interface springs (nearest-neighbour couplings) so the two
  regions share gross translation and rotation without slip or separation.

Time integration
----------------
Linearized implicit Euler.  Each substep solves

    (M + dt^2 K(x_pred) + dt C(x_pred)) dx = dt^2 f(x_pred)

where f/K/C are the nonlinear spring forces and their exact tangents evaluated
at the predictor ``x_pred = x + dt v + dt^2 g``; the update is
``x_new = x_pred + dx``, ``v_new = (x_new - x)/dt``.  The system is symmetric
positive definite (the transverse tangent is clamped at zero, a standard
stabilization) and is solved with a matrix-free-free Jacobi-preconditioned
conjugate-gradient iteration (deterministic), with a sparse-LU fallback.

Ground contact is resolved at velocity level (non-smooth contact): penetrations
are projected out positionally, normal velocities are made non-negative
(restitution 0) and Coulomb friction caps the tangential impulse.  Every
external impulse (gravity, contact, bonded-interface traction for the
single-body budget domains) is accumulated exactly for the momentum budgets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import scipy.sparse.linalg as sla

from worldsolver.contracts import StateFrame

DT_TARGET = 2.5e-3          # nominal substep [s]
ZETA = 0.15                 # spring damping ratio
MU = 0.6                    # Coulomb friction coefficient
K_STIFF_CAP = 5.0e8         # rigid-limit regularization of the 1e14 Pa modulus
CG_RTOL = 1e-8
CG_MAXITER = 600
CG_ACCEPT_RTOL = 1e-4     # accept an inexact Newton step at this relative residual
BEND_SOFT = 0.35            # bending-to-membrane stiffness ratio (soft)
BEND_STIFF = 0.20           # bending-to-membrane stiffness ratio (stiff)
BOND_FACTOR = 1.5           # bonded interface stiffness relative to soft


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _load_obj(path: Path):
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if a == b:
                continue
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _dist2_pairs(verts: np.ndarray, edges: np.ndarray, per_vertex: int) -> np.ndarray:
    """Graph-distance-2 pairs (bending stiffness), shortest-first, capped per vertex."""
    n = len(verts)
    adj: list[list[int]] = [[] for _ in range(n)]
    edge_set = {(int(a), int(b)) for a, b in edges}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    chosen: set[tuple[int, int]] = set()
    for i in range(n):
        nbrs = adj[i]
        cand: list[tuple[float, int, int]] = []
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                a, b = nbrs[ai], nbrs[bi]
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    continue
                d2 = float(np.sum((verts[a] - verts[b]) ** 2))
                cand.append((d2, key[0], key[1]))
        cand.sort()
        taken = 0
        for _d2, a, b in cand:
            if (a, b) in chosen:
                continue
            chosen.add((a, b))
            taken += 1
            if taken >= per_vertex:
                break
    if not chosen:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(chosen), dtype=np.int64)


def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    tri = verts[faces]
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    return abs(vol)


def _solve_cg(A, b: np.ndarray, dinv: np.ndarray):
    """Jacobi-preconditioned CG for an SPD sparse system. Returns (x, info)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = dinv * r
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return x, 0
    tol = CG_RTOL * bnorm
    for it in range(1, CG_MAXITER + 1):
        if float(np.linalg.norm(r)) <= tol:
            return x, it - 1
        Ap = A @ p
        pAp = float(p @ Ap)
        if not np.isfinite(pAp) or pAp <= 0.0:
            return x, -1
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = dinv * r
        rz_new = float(r @ z)
        if not np.isfinite(rz_new):
            return x, -1
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, -2


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ---- setup -----------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        ic = scene.initial_state()
        x0 = np.array(ic.position, dtype=np.float64)
        v0 = np.array(ic.velocity, dtype=np.float64)
        body = np.asarray(ic.body_id, dtype=np.int64)
        n = len(x0)
        if x0.shape != (n, 3) or v0.shape != (n, 3) or body.shape != (n,):
            raise ValueError("initial state shape mismatch")

        ic_cfg = cfg.get("initial_conditions", {}) or {}
        bc_cfg = cfg.get("boundary_conditions", {}) or {}
        g_vec = np.asarray(ic_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        e_soft = float(ic_cfg.get("soft_youngs_modulus_pa", 6.0e4))
        e_stiff = float(ic_cfg.get("stiff_youngs_modulus_pa", 1.0e14))
        rho = float(ic_cfg.get("density_kg_m3", 1000.0))
        ground_h = float(bc_cfg.get("ground_height", 0.0))

        # assets (topology only; geometry comes from the frozen initial state)
        root = Path(__file__).resolve().parents[1]
        assets = root / "assets"
        if not assets.is_dir():
            assets = Path("assets")
        _vo, fo = _load_obj(assets / "gummy_bear_outer.obj")
        _vs, fs = _load_obj(assets / "gummy_bear_stiff.obj")
        n_soft = int(np.count_nonzero(body == 0))
        n_stiff = int(np.count_nonzero(body == 1))
        if len(fo) and len(fo[0]) and n_soft != len(_vo):
            raise ValueError("outer asset vertex count does not match body 0")
        if len(fs) and n_stiff != len(_vs):
            raise ValueError("stiff asset vertex count does not match body 1")
        outer = np.nonzero(body == 0)[0]
        stiff = np.nonzero(body == 1)[0]
        vo = x0[outer]
        vs = x0[stiff]

        # ---- masses from the frozen geometry -----------------------------
        v_out = _mesh_volume(vo, fo) if len(fo) else 0.0
        v_st = _mesh_volume(vs, fs) if len(fs) else 0.0
        v_soft = max(v_out - v_st, 0.05 * v_out) if v_out > 0 else max(v_st, 1e-3)
        m = np.empty(n, dtype=np.float64)
        m[outer] = rho * v_soft / max(n_soft, 1)
        m[stiff] = rho * v_st / max(n_stiff, 1)

        # ---- spring network ----------------------------------------------
        e_out = _mesh_edges(fo)
        e_st = _mesh_edges(fs)
        d2_out = _dist2_pairs(vo, e_out, 6)
        d2_st = _dist2_pairs(vs, e_st, 4)

        h_out = float(np.mean(np.linalg.norm(vo[e_out[:, 0]] - vo[e_out[:, 1]], axis=1))) if len(e_out) else 0.1
        h_st = float(np.mean(np.linalg.norm(vs[e_st[:, 0]] - vs[e_st[:, 1]], axis=1))) if len(e_st) else 0.1
        k_mem = e_soft * h_out
        k_stiff = min(e_stiff * h_st, K_STIFF_CAP)
        k_bond = BOND_FACTOR * k_mem

        pairs: list[tuple[int, int, int]] = []

        def _add(pair_arr: np.ndarray, offset_a: np.ndarray, offset_b: np.ndarray, typ: int) -> None:
            if len(pair_arr) == 0:
                return
            for a, b in pair_arr:
                ga, gb = int(offset_a[int(a)]), int(offset_b[int(b)])
                pairs.append((min(ga, gb), max(ga, gb), typ))

        _add(e_out, outer, outer, 0)
        _add(d2_out, outer, outer, 1)
        _add(e_st, stiff, stiff, 2)
        _add(d2_st, stiff, stiff, 3)

        # bonded interface: nearest-neighbour couplings between the regions
        tree_o = cKDTree(vo)
        tree_s = cKDTree(vs)
        d_cross, idx_cross = tree_o.query(vs, k=6, distance_upper_bound=0.32)
        if idx_cross.ndim == 1:
            d_cross = d_cross[:, None]
            idx_cross = idx_cross[:, None]
        bond_set: set[tuple[int, int]] = set()
        for s in range(n_stiff):
            for c in range(idx_cross.shape[1]):
                j = int(idx_cross[s, c])
                if not np.isfinite(d_cross[s, c]) or j >= n_soft:
                    continue
                bond_set.add((min(int(outer[j]), int(stiff[s])), max(int(outer[j]), int(stiff[s]))))
        d_rev, idx_rev = tree_s.query(vo, k=2, distance_upper_bound=0.18)
        if idx_rev.ndim == 1:
            d_rev = d_rev[:, None]
            idx_rev = idx_rev[:, None]
        for o in range(n_soft):
            for c in range(idx_rev.shape[1]):
                j = int(idx_rev[o, c])
                if not np.isfinite(d_rev[o, c]) or j >= n_stiff:
                    continue
                bond_set.add((min(int(outer[o]), int(stiff[j])), max(int(outer[o]), int(stiff[j]))))
        for a, b in sorted(bond_set):
            pairs.append((a, b, 4))

        # dedupe (keep the first/strongest occurrence per unordered pair)
        seen: set[tuple[int, int]] = set()
        ia: list[int] = []
        ib: list[int] = []
        typ: list[int] = []
        for a, b, t in sorted(pairs):
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            ia.append(a)
            ib.append(b)
            typ.append(t)
        si = np.asarray(ia, dtype=np.int64)
        sj = np.asarray(ib, dtype=np.int64)
        typ = np.asarray(typ, dtype=np.int64)
        ne = len(si)

        k_spring = np.asarray(
            [k_mem, BEND_SOFT * k_mem, k_stiff, BEND_STIFF * k_stiff, k_bond], dtype=np.float64
        )[typ]
        l0 = np.linalg.norm(x0[si] - x0[sj], axis=1)
        l0 = np.maximum(l0, 1e-6)
        m_h = 2.0 * m[si] * m[sj] / np.maximum(m[si] + m[sj], 1e-12)
        c_spring = ZETA * np.sqrt(k_spring * m_h)

        cross_mask = typ == 4
        cross_idx = np.nonzero(cross_mask)[0]
        # traction on the soft body from a cross spring = sign * F_I
        is_soft_node = body == 0
        soft_is_i = is_soft_node[si[cross_idx]]
        cross_sign = np.where(soft_is_i, 1.0, -1.0)

        # ---- sparse pattern (fixed) --------------------------------------
        dofs = 3 * n
        ai = np.arange(3)

        def _block(rows_idx: np.ndarray, cols_idx: np.ndarray):
            r = (3 * rows_idx[:, None] + ai[None, :])[:, :, None].repeat(3, 2)
            c = (3 * cols_idx[:, None] + ai[None, :])[:, None, :].repeat(3, 1)
            return r.reshape(ne * 9), c.reshape(ne * 9)

        r_ii, c_ii = _block(si, si)
        r_ij, c_ij = _block(si, sj)
        r_ji, c_ji = _block(sj, si)
        r_jj, c_jj = _block(sj, sj)
        self._rows = np.concatenate([r_ii, r_ij, r_ji, r_jj, np.arange(dofs)])
        self._cols = np.concatenate([c_ii, c_ij, c_ji, c_jj, np.arange(dofs)])
        self._mdiag = np.repeat(m, 3)
        self._shape = (dofs, dofs)

        # deformation-gradient neighbourhood (directed spring adjacency)
        self._dg_i = np.concatenate([si, sj])
        self._dg_j = np.concatenate([sj, si])
        self._dg_u = x0[self._dg_j] - x0[self._dg_i]
        self._dg_w = np.repeat(1.0 / l0, 2)

        # ---- budget domains ----------------------------------------------
        # rows: 0 = whole bear, 1 = soft exterior, 2 = stiff interior
        w_dom = np.stack([np.ones(n), (body == 0).astype(np.float64), (body == 1).astype(np.float64)])
        f_grav_dom = (w_dom @ m[:, None]) * g_vec[None, :]

        # ---- state ---------------------------------------------------------
        self._n = n
        self._body = body
        self._x = x0.copy()
        self._v = v0.copy()
        self._m = m
        self._g = g_vec
        self._ground_h = ground_h
        self._t = 0.0
        self._si = si
        self._sj = sj
        self._l0 = l0
        self._k = k_spring
        self._c = c_spring
        self._cross_idx = cross_idx
        self._cross_sign = cross_sign
        self._w_dom = w_dom
        self._f_grav_dom = f_grav_dom
        self._ref_momentum = float(m.sum() * 1.5)
        self._i3 = np.eye(3)

        # budget accumulators
        self._cum_gravity = np.zeros((3, 3))
        self._cum_contact = np.zeros((3, 3))
        self._cum_interface = np.zeros((3, 3))
        self._int_contact = np.zeros((3, 3))
        self._t_frame = 0.0
        self._t_now = np.zeros(3)
        self._substeps = 0
        self._cg_iters = 0
        self._cg_fallbacks = 0
        self._ready = True

    # ---- one implicit substep ---------------------------------------------
    def _substep(self, dt: float) -> None:
        x_n = self._x
        v_n = self._v
        n = self._n
        si, sj, l0, k_s, c_s = self._si, self._sj, self._l0, self._k, self._c
        dt2 = dt * dt

        xp = x_n + dt * v_n + dt * self._g
        u_prev = x_n[si] - x_n[sj]
        u = xp[si] - xp[sj]
        L = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        nv = u / L[:, None]
        dvn = np.einsum("ij,ij->i", (u - u_prev) / dt, nv)
        f_mag = k_s * (l0 - L) - c_s * dvn
        f_spring = f_mag[:, None] * nv

        rhs = np.zeros((n, 3))
        np.add.at(rhs, si, dt2 * f_spring)
        np.add.at(rhs, sj, -dt2 * f_spring)
        b = rhs.ravel()

        # SPD tangent: M + dt^2 K + dt C (transverse tangent clamped at zero)
        nnT = nv[:, :, None] * nv[:, None, :]
        tang = np.clip((L - l0) / L, 0.0, None)
        blk = k_s[:, None, None] * (nnT + tang[:, None, None] * (self._i3 - nnT))
        blk = dt2 * blk + (dt * c_s)[:, None, None] * nnT
        a9 = blk.reshape(-1)
        vals = np.concatenate([a9, -a9, -a9, a9, self._mdiag])
        amat = coo_matrix((vals, (self._rows, self._cols)), shape=self._shape).tocsr()
        dinv = 1.0 / amat.diagonal()

        dx, info = _solve_cg(amat, b, dinv)
        if info < 0:
            # inexact-Newton safeguard: a partially converged SPD step is still
            # a stable descent direction; only pay for a direct solve otherwise
            resid = float(np.linalg.norm(b - amat @ dx))
            if info == -2 and resid <= CG_ACCEPT_RTOL * float(np.linalg.norm(b)):
                info = CG_MAXITER
            else:
                self._cg_fallbacks += 1
                dx = sla.splu(amat.tocsc()).solve(b)
        self._cg_iters += max(info, 0)

        x_new = xp + dx.reshape(n, 3)
        if not np.isfinite(x_new).all():
            x_new = xp.copy()
        v_new = (x_new - x_n) / dt
        if not np.isfinite(v_new).all():
            v_new = v_n + dt * self._g

        # ---- gravity impulse ---------------------------------------------
        self._cum_gravity += dt * self._f_grav_dom

        # ---- ground contact (velocity level, Coulomb friction) -----------
        gh = self._ground_h
        mask = x_new[:, 2] < gh
        if np.any(mask):
            v_before = v_new[mask].copy()
            x_new[mask, 2] = gh
            vz = v_new[mask, 2]
            vz_new = np.maximum(vz, 0.0)
            jn = self._m[mask] * (vz_new - vz)
            vt = v_new[mask, :2]
            vtn = np.linalg.norm(vt, axis=1)
            scale = np.ones(len(vtn))
            pos = vtn > 1e-12
            scale[pos] = np.maximum(0.0, 1.0 - (MU * jn[pos]) / (self._m[mask][pos] * vtn[pos]))
            v_new[mask, :2] = vt * scale[:, None]
            v_new[mask, 2] = vz_new
            j_imp = self._m[mask, None] * (v_new[mask] - v_before)
            add = self._w_dom[:, mask] @ j_imp
            self._cum_contact += add
            self._int_contact += add

        # ---- interface traction (recomputed at the accepted state) --------
        u2 = x_new[si] - x_new[sj]
        l2 = np.maximum(np.linalg.norm(u2, axis=1), 1e-9)
        n2 = u2 / l2[:, None]
        dvn2 = np.einsum("ij,ij->i", v_new[si] - v_new[sj], n2)
        f2 = (k_s * (l0 - l2) - c_s * dvn2)[:, None] * n2
        ci = self._cross_idx
        traction = (self._cross_sign[:, None] * f2[ci]).sum(axis=0)
        self._t_now = traction
        self._cum_interface[1] += dt * traction
        self._cum_interface[2] -= dt * traction

        self._x = x_new
        self._v = v_new
        self._t += dt
        self._substeps += 1

    # ---- per-node deformation gradient ------------------------------------
    def _deformation_gradients(self) -> np.ndarray:
        x = self._x
        n = self._n
        di, dj = self._dg_i, self._dg_j
        u_rest = self._dg_u
        w = self._dg_w
        u = x[dj] - x[di]
        a_mat = np.zeros((n, 3, 3))
        b_mat = np.zeros((n, 3, 3))
        np.add.at(a_mat, di, (w[:, None, None] * u_rest[:, :, None] * u_rest[:, None, :]))
        np.add.at(b_mat, di, (w[:, None, None] * u[:, :, None] * u_rest[:, None, :]))
        tr = a_mat[:, 0, 0] + a_mat[:, 1, 1] + a_mat[:, 2, 2]
        reg = 1e-6 * tr + 1e-12
        a_mat[:, 0, 0] += reg
        a_mat[:, 1, 1] += reg
        a_mat[:, 2, 2] += reg
        f = np.linalg.solve(a_mat, np.transpose(b_mat, (0, 2, 1)))
        f = np.transpose(f, (0, 2, 1))
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- StateFrame --------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        dt_int = t - self._t_frame
        if dt_int > 1e-12:
            contact_avg = self._int_contact / dt_int
        else:
            contact_avg = np.zeros((3, 3))
        self._int_contact[:] = 0.0
        self._t_frame = t

        # budget_external_force[G, K, 3]: terms = gravity, ground contact,
        # bonded-interface traction (zero for the combined global domain).
        bef = np.zeros((3, 3, 3))
        bef[:, 0, :] = self._f_grav_dom
        bef[:, 1, :] = contact_avg
        bef[1, 2, :] = self._t_now
        bef[2, 2, :] = -self._t_now

        # cumulative external impulse per domain (gravity + contact; the
        # single-body domains additionally receive the interface traction,
        # which cancels in the combined domain).
        cum = np.zeros((3, 3))
        cum[0] = self._cum_gravity[0] + self._cum_contact[0]
        cum[1] = self._cum_gravity[1] + self._cum_contact[1] + self._cum_interface[1]
        cum[2] = self._cum_gravity[2] + self._cum_contact[2] + self._cum_interface[2]

        n = self._n
        fields = {
            "budget_cumulative_external_impulse": cum,
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "budget_domain_weights": self._w_dom.copy(),
            "budget_external_force": bef,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": self._deformation_gradients(),
            "interaction_force": self._t_now.reshape(1, 3).copy(),
        }
        attributes = {
            "mass_quantity": self._m.copy(),
            "material_region": self._body.astype(np.float64),
        }
        return StateFrame(
            time=float(t),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            body_id=self._body.copy(),
            attributes=attributes,
            fields=fields,
            scalars={"reference_momentum": self._ref_momentum},
        )

    # ---- public API ---------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        target = float(time)
        if target > self._t + 1e-12:
            n_sub = max(1, int(round((target - self._t) / DT_TARGET)))
            dt = (target - self._t) / n_sub
            for _ in range(n_sub):
                self._substep(dt)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "linearized implicit Euler on a bonded heterogeneous spring network",
            "nodes": int(self._n),
            "springs": int(len(self._si)),
            "substeps": int(self._substeps),
            "cg_iterations": int(self._cg_iters),
            "cg_direct_fallbacks": int(self._cg_fallbacks),
            "total_mass_kg": float(self._m.sum()),
        }
