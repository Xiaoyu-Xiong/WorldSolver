"""Incompressible soft ball drop — co-rotational linear FEM tetrahedral solver.

Physics:
  * 13-node volumetric mesh: 12 icosahedron surface nodes (matching the OBJ
    asset exactly, required by the deformed-mesh render binding) plus one
    internal center node; 20 tetrahedra (center + each surface face).
  * Nearly-incompressible co-rotational linear elasticity (Irving et al. 2007,
    "Volume Conserving Finite Element Simulations of Deformable Models"):
    W = mu*|eps|^2 + lam/2*tr(eps)^2 with eps = sym(R^T F) - I and R the
    polar rotation of F; lam >> mu enforces volume conservation.
  * Fully implicit backward-Euler integration with Newton iterations and the
    frozen-rotation consistent stiffness (unconditionally stable).
  * Penalty contact against the rigid floor plane z = 0, sampled at the 12
    nodes and at the 20 face centroids, with normal spring-damper plus smooth
    Coulomb friction. Center node carries zero lumped mass, so the exported
    12-node state carries the full 4 kg and the momentum budget closes.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --- frozen scene constants -------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
TOTAL_MASS = 4.0
BALL_CENTER0 = np.array([0.0, 0.0, 2.25])
FLOOR_Z = 0.0

# Faces of assets/ball_sphere.obj (converted to 0-based indices).
_FACES = np.array([
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
], dtype=int)


def _reference_vertices() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    pos = np.asarray(verts, dtype=float) / np.sqrt(1.0 + phi**2)
    pos[:, 2] += BALL_CENTER0[2]
    return pos


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)

        # ---- mesh: 13 nodes (12 surface + center), 20 tets ----------------
        surf = _reference_vertices()
        X = np.vstack([surf, BALL_CENTER0[None, :]])
        self.n_nodes = 13
        self.n_surf = 12
        self.center = 12
        tets = np.column_stack([np.full(len(_FACES), self.center), _FACES])  # [Ne,4]
        self.tets = tets
        self.ne = len(tets)

        # ---- reference tet geometry ---------------------------------------
        p0 = X[tets[:, 0]]
        dm = np.stack([X[tets[:, i + 1]] - p0 for i in range(3)], axis=-1)  # [Ne,3,3]
        det = np.linalg.det(dm)
        self.v0 = np.abs(det) / 6.0
        self.dm_inv = np.linalg.inv(dm)
        self.vol0 = float(self.v0.sum())

        # Shape-function gradients g_a with F = sum_a x_a (x) g_a:
        # g_j = row (j-1) of dm_inv for j = 1..3, g_0 = -sum_j g_j.
        g = np.zeros((self.ne, 4, 3))
        g[:, 1:, :] = self.dm_inv
        g[:, 0, :] = -self.dm_inv.sum(axis=1)
        self.g_ref = g

        # ---- material (nearly incompressible, soft) ------------------------
        young, nu = 400.0, 0.499
        self.mu = young / (2.0 * (1.0 + nu))
        self.lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        # Voigt elasticity matrix: eps_v = [xx, yy, zz, xy, yz, zx] (engineering shear)
        d = np.zeros((6, 6))
        d[:3, :3] = self.lam
        d[0, 0] = d[1, 1] = d[2, 2] = self.lam + 2.0 * self.mu
        d[3, 3] = d[4, 4] = d[5, 5] = self.mu
        # Fixed Voigt basis matrices H_k (B_a = sum_k ghat_a[k] H_k)
        h_basis = np.zeros((3, 6, 3))
        h_basis[0, 0, 0] = 1.0
        h_basis[1, 1, 1] = 1.0
        h_basis[2, 2, 2] = 1.0
        h_basis[0, 3, 1] = h_basis[1, 3, 0] = 1.0
        h_basis[1, 4, 2] = h_basis[2, 4, 1] = 1.0
        h_basis[2, 5, 0] = h_basis[0, 5, 2] = 1.0
        # A_kl[i,j] = (H_k^T D H_l)[i,j]
        self.a_kl = np.einsum('kip,pq,jq->klij', h_basis, d, h_basis)

        # ---- lumped masses (center node massless; all mass on 12 surface) --
        rho = TOTAL_MASS / self.vol0
        m = np.zeros(self.n_nodes)
        tet_m = rho * self.v0
        for j in range(1, 4):
            np.add.at(m, tets[:, j], tet_m / 3.0)
        m[self.center] = 0.0
        m[: self.n_surf] *= TOTAL_MASS / m[: self.n_surf].sum()
        self.mass = m
        self.m_diag = np.repeat(m, 3)  # [39]

        # ---- contact sampling weights (half nodal, half face centroids) ----
        e1 = surf[_FACES[:, 1]] - surf[_FACES[:, 0]]
        e2 = surf[_FACES[:, 2]] - surf[_FACES[:, 0]]
        face_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        node_area = np.zeros(self.n_surf)
        for j in range(3):
            np.add.at(node_area, _FACES[:, j], face_area / 3.0)
        self.w_node = 0.5 * node_area          # [12] per-node total weight
        self.w_face = 0.5 * face_area          # [Ne] per-face total weight
        self.k_contact = 4.0e4                 # N/m per m^2
        self.c_contact = 15.0                  # N/(m/s) per m^2
        self.mu_friction = 0.45
        self.v_slip = 0.03

        # ---- Rayleigh (mass-proportional) damping --------------------------
        self.beta = 1.0

        # ---- integration schedule -------------------------------------------
        self.h = 0.001
        self.newton_iters = 4

        # ---- state -----------------------------------------------------------
        self.x = X.copy()
        self.v = np.zeros_like(X)
        self.t = 0.0
        self.impulse = np.zeros(3)
        self._last_contact = (np.zeros(3), np.zeros(3))
        self._scene = scene

        # ---- frozen export scales --------------------------------------------
        self.constraint_scale = self.vol0
        lowest = float(np.min(surf[:, 2]))  # initial gap of lowest node above floor
        v_impact = np.sqrt(2.0 * 9.81 * max(lowest, 1e-6))
        self.reference_momentum = float(TOTAL_MASS * v_impact)

        # global index helpers for stiffness assembly, order (t,a,i,b,j)
        node_idx = tets  # [Ne,4]
        ii = (node_idx[:, :, None, None, None, None] * 3
              + np.arange(3)[None, None, :, None, None, None])
        jj = (node_idx[:, None, :, None, None, None] * 3
              + np.arange(3)[None, None, None, None, :, None])
        self._rows = np.broadcast_to(ii, (self.ne, 4, 3, 4, 3)).reshape(-1)
        self._cols = np.broadcast_to(jj, (self.ne, 4, 3, 4, 3)).reshape(-1)

    # ------------------------------------------------------------------ FEM --
    def _elastic_forces(self, x):
        """Return (f_int [n,3], K_blocks [Ne,4,4,3,3] = d2W/dxa dxb, F [Ne,3,3])."""
        tets = self.tets
        p0 = x[tets[:, 0]]
        ds = np.stack([x[tets[:, i + 1]] - p0 for i in range(3)], axis=-1)
        f_grad = ds @ self.dm_inv  # [Ne,3,3]
        u, _s, vt = np.linalg.svd(f_grad)
        rot = u @ vt
        flip = np.linalg.det(rot) < 0.0
        if np.any(flip):
            u = u.copy()
            u[flip, :, -1] *= -1.0
            rot = u @ vt
        rt_f = np.einsum('nji,njk->nik', rot, f_grad)  # R^T F
        eps = 0.5 * (rt_f + np.transpose(rt_f, (0, 2, 1))) - np.eye(3)[None]
        tr = np.trace(eps, axis1=1, axis2=2)
        sigma = 2.0 * self.mu * eps + self.lam * tr[:, None, None] * np.eye(3)[None]
        # dW/dDs = V0 * R sigma dm_inv^T ; columns map to nodes 1..3, f_int = -dW/dx
        gm = self.v0[:, None, None] * (rot @ sigma @ np.transpose(self.dm_inv, (0, 2, 1)))
        f_int = np.zeros((self.n_nodes, 3))
        np.add.at(f_int, tets[:, 1], -gm[:, :, 0])
        np.add.at(f_int, tets[:, 2], -gm[:, :, 1])
        np.add.at(f_int, tets[:, 3], -gm[:, :, 2])
        np.add.at(f_int, tets[:, 0], gm.sum(axis=2))
        # corotational stiffness: K_ab = V0 * B_a^T D B_b with B_a built from ghat_a = R g_a
        ghat = np.einsum('tik,tak->tai', rot, self.g_ref)
        k_blocks = self.v0[:, None, None, None, None] * np.einsum(
            'tak,tbl,klij->tabij', ghat, ghat, self.a_kl, optimize=True)
        return f_int, k_blocks, f_grad

    # -------------------------------------------------------------- contact --
    def _contact_forces(self, x, v, with_jacobian=False):
        f_c = np.zeros((self.n_nodes, 3))
        f_n_tot = np.zeros(3)
        f_t_tot = np.zeros(3)
        kz = np.zeros(self.n_nodes)
        cz = np.zeros(self.n_nodes)
        solver = self

        def apply(node_ids, weights, pos, vel):
            # node_ids: [S] or [S,P]; weights: [S] total sample weight
            pen = FLOOR_Z - pos[:, 2]
            act = pen > 0.0
            if not np.any(act):
                return
            ids = node_ids[act]
            w = weights[act]              # [k]
            d = pen[act]                  # [k]
            vz = vel[act, 2]              # [k]
            fn = solver.k_contact * w * d
            fn = fn + np.where(vz < 0.0, solver.c_contact * w * (-vz), 0.0)
            np.maximum(fn, 0.0, out=fn)
            f_n_tot[2] += fn.sum()
            vt = vel[act, :2]
            speed = np.linalg.norm(vt, axis=1)
            ft_mag = solver.mu_friction * fn * np.tanh(speed / solver.v_slip)
            ft = np.zeros_like(vt)
            good = speed > 1e-9
            ft[good] = -ft_mag[good, None] * (vt[good] / speed[good, None])
            f_t_tot[0] += ft[:, 0].sum()
            f_t_tot[1] += ft[:, 1].sum()
            p_count = ids.shape[1] if ids.ndim == 2 else 1
            share = 1.0 / p_count
            for p in range(p_count):
                rows = ids[:, p] if ids.ndim == 2 else ids
                np.add.at(f_c, rows, np.stack(
                    [ft[:, 0] * share, ft[:, 1] * share, fn * share], axis=1))
                if with_jacobian:
                    np.add.at(kz, rows, solver.k_contact * w * share * share)
                    np.add.at(cz, rows,
                              np.where(vz < 0.0, solver.c_contact * w, 0.0) * share * share)

        apply(np.arange(self.n_surf), self.w_node, x[: self.n_surf], v[: self.n_surf])
        tri = _FACES
        cen = x[tri].mean(axis=1)
        cen_v = v[tri].mean(axis=1)
        apply(tri, self.w_face, cen, cen_v)
        return f_c, f_n_tot, f_t_tot, kz, cz

    # ----------------------------------------------------------------- step --
    def _step(self, h):
        x0, v0 = self.x, self.v
        x_new = x0 + h * v0  # predictor
        f_grav = self.mass[:, None] * GRAVITY[None, :]
        ndof = 3 * self.n_nodes
        dof = np.arange(ndof)
        base_diag = self.m_diag / h ** 2 + self.beta * self.m_diag / h

        for _ in range(self.newton_iters):
            v_new = (x_new - x0) / h
            f_int, k_blocks, _ = self._elastic_forces(x_new)
            f_c, _, _, kz, cz = self._contact_forces(x_new, v_new, with_jacobian=True)
            f_damp = -self.beta * self.mass[:, None] * v_new
            f_tot = f_int + f_c + f_damp + f_grav
            resid = (self.m_diag * (x_new - x0).reshape(-1) / h ** 2
                     - self.m_diag * v0.reshape(-1) / h
                     - f_tot.reshape(-1))
            jacob = np.zeros((ndof, ndof))
            # k_blocks[t,a,b,i,j] -> assembly order (t,a,i,b,j)
            np.add.at(jacob, (self._rows, self._cols),
                      k_blocks.transpose(0, 1, 3, 2, 4).reshape(-1))
            jacob[dof, dof] += base_diag
            kzz_total = kz + cz / h
            nodes_act = kzz_total > 0.0
            if np.any(nodes_act):
                zz = np.flatnonzero(nodes_act) * 3 + 2
                jacob[zz, zz] += kzz_total[nodes_act]
            delta = np.linalg.solve(jacob, -resid)
            x_new = (x_new.reshape(-1) + delta).reshape(self.n_nodes, 3)

        v_new = (x_new - x0) / h
        f_c, f_n_vec, f_t_vec, _, _ = self._contact_forces(x_new, v_new)
        self.impulse += h * (f_grav.sum(axis=0) + f_n_vec + f_t_vec)
        self.x = x_new
        self.v = v_new
        self.t += h
        self._last_contact = (f_n_vec.copy(), f_t_vec.copy())

    # -------------------------------------------------------------- export --
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            self._step(min(self.h, time - self.t))
        if time <= 1e-12:
            self._last_contact = self._contact_forces(self.x, self.v)[1:3]
        _, _, f_grad = self._elastic_forces(self.x)
        vol = float(np.sum(self.v0 * np.linalg.det(f_grad)))
        f_grav_total = self.mass.sum() * GRAVITY
        f_n_vec, f_t_vec = self._last_contact
        ext_force = np.stack([f_grav_total, f_n_vec, f_t_vec])[None, :, :]  # [1,3,3]
        fields = {
            "budget_cumulative_external_impulse": self.impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n_surf)),
            "budget_external_force": ext_force,
            "constraint_value": np.array([vol]),
            "constraint_target": np.array([self.vol0]),
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": f_grad.copy(),
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
        }
        return StateFrame(
            time,
            self.x[: self.n_surf].copy(),
            self.v[: self.n_surf].copy(),
            mass=self.mass[: self.n_surf].copy(),
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene": "055_incompressible_ball_drop",
            "method": "implicit corotational tetrahedral FEM, near-incompressible penalty",
            "n_nodes_exported": self.n_surf,
            "n_tets": self.ne,
            "young": 400.0,
            "poisson": 0.499,
            "total_mass": float(self.mass.sum()),
            "reference_volume": float(self.vol0),
            "cumulative_external_impulse": self.impulse.tolist(),
            "sim_time": float(self.t),
        }
