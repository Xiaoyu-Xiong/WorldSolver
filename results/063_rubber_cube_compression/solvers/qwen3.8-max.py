"""Physically-based FEM solver for lateral bulging of a compressed rubber cube.

A compressible Neo-Hookean hyperelastic solid (strain energy
W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2, i.e. bulk modulus
kappa = lambda + 2 mu / 3) is discretized on the frozen tetrahedral mesh.
The bottom face is fully fixed (rigid support), and the top face carries a
dead uniform downward traction p(t) ramped from 0 to 50 kPa per the frozen
schedule. Each sample time is solved by Newton-Raphson on the nonlinear
equilibrium equations R = f_ext - f_int = 0 with a sparse direct solve, which
is appropriate for the slowly ramped, quasi-static compression. The result is
the classical barreling response: the cube shortens while the free side faces
bulge outward.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = scene.config
        bcs = cfg.get("boundary_conditions", {})
        load = bcs.get("top_compressive_traction", {})
        self._pressure = float(load.get("peak_magnitude_pa", 5.0e4))
        sched = load.get("schedule", [{"time": 0.0, "scale": 0.0}])
        self._sched_t = np.array([float(s["time"]) for s in sched])
        self._sched_s = np.array([float(s["scale"]) for s in sched])

        X = np.asarray(scene.vertices, dtype=float).copy()
        T = np.asarray(scene.tetrahedra, dtype=np.int64)
        F = np.asarray(scene.faces, dtype=np.int64)
        self._X = X
        self._T = T
        self._n = len(X)
        self._ne = len(T)

        # ---------------- material: soft nearly-incompressible rubber --------
        E = 3.0e5            # Pa (soft rubber; 50 kPa gives ~30% shortening)
        nu = 0.49            # nearly incompressible -> strong lateral bulge
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        rho = 1100.0         # kg/m^3
        self._rho = rho

        # ---------------- element reference geometry ------------------------
        Dm = X[T[:, 1:4]] - X[T[:, 0:1]]                    # [Ne,3,3]
        self._V0 = np.linalg.det(Dm) / 6.0                  # [Ne]
        Dm_inv = np.linalg.inv(Dm)                          # dX_i/d? layout
        # grad N_i (w.r.t. X), i = 0..3: rows sum to zero, grad N_{1..3} = Dm_inv^T
        G = np.empty((self._ne, 4, 3))
        G[:, 1:4, :] = np.transpose(Dm_inv, (0, 2, 1))      # [Ne,3(node),3(dir)]
        G[:, 0, :] = -G[:, 1:4, :].sum(axis=1)
        self._G = G

        # lumped masses (Voronoi-style: quarter of each tet volume per node)
        m = np.zeros(self._n)
        np.add.at(m, T.ravel(), np.repeat(self._V0 * rho / 4.0, 4))
        self._mass = m
        self._mass_total = float(m.sum())

        # ---------------- boundary conditions --------------------------------
        self._bottom = np.isclose(X[:, 2], 0.0)
        self._top_node = np.isclose(X[:, 2], X[:, 2].max())
        self._free = ~self._bottom
        self._free_dofs = np.nonzero(np.repeat(self._free, 3))[0]

        # top surface triangles -> dead-load nodal forces (downward, -z)
        top_tris = F[np.all(self._top_node[F], axis=1)]
        a = X[top_tris[:, 0]]
        b = X[top_tris[:, 1]]
        c = X[top_tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        w = np.repeat(areas / 3.0, 3)
        f_unit = np.zeros((len(top_tris) * 3, 3))
        f_unit[:, 2] = -1.0
        self._top_load_nodes = top_tris.ravel()
        self._top_load_weights = w[:, None] * f_unit        # per-unit-pressure
        self._top_area = float(areas.sum())

        # ---------------- state ---------------------------------------------
        self._x = X.copy()
        self._v = np.zeros_like(X)
        self._t = 0.0
        self._scale = 0.0
        self._reaction = np.zeros(3)
        self._F_degrad = np.tile(np.eye(3), (self._ne, 1, 1))

    # ------------------------------------------------------------- schedule
    def _load_scale(self, t: float) -> float:
        return float(np.interp(t, self._sched_t, self._sched_s))

    # ------------------------------------------------- constitutive + forces
    def _stress_tangent(self, x):
        """First Piola stress and material tangent for all elements."""
        mu, lam = self._mu, self._lam
        ds = x[self._T]                                     # [Ne,4,3]
        Fm = np.einsum('eaj,eai->eij', ds, self._G)         # F[i,j] = dx_i/dX_j
        Fm_inv = np.linalg.inv(Fm)
        J = np.linalg.det(Fm)
        lnJ = np.log(J)
        Finv_T = np.swapaxes(Fm_inv, 1, 2)
        # P = mu (F - F^-T) + lam lnJ F^-T
        P = mu * (Fm - Finv_T) + (lam * lnJ)[:, None, None] * Finv_T
        # dP_ij/dF_kl = mu d_ik d_jl + (lam lnJ + lam) F^-T_ij F^-T_kl
        #               - lam F^-T_kj F^-T_il
        I = np.eye(3)
        C = (mu * np.einsum('ik,jl->ijkl', I, I)
             + (lam * (lnJ + 1.0))[:, None, None, None, None]
             * np.einsum('eij,ekl->eijkl', Finv_T, Finv_T)
             - lam * np.einsum('ekj,eil->eijkl', Finv_T, Finv_T))
        return Fm, P, C, Finv_T, lnJ

    def _internal_force(self, P):
        # f_int[node_i] = V0 * P : grad N_i
        e = self._V0[:, None, None] * np.einsum('eij,eaj->eai', P, self._G)
        f = np.zeros((self._n, 3))
        for k in range(4):
            np.add.at(f, self._T[:, k], e[:, k, :])
        return f

    def _assemble_K(self, C):
        ne = self._ne
        T = self._T
        B = self._G                                         # [Ne,4,3] grad N
        # K[a,p,b,q] = V0 * B[a,i] C[i,j,k,l] B[b,k] delta? -> contract properly
        # K[a p, b q] = V0 * B[a,i] * C[i q k l]? use index: dP_ij/dF_kl,
        # dF_kl = B[b,k] du[l? ...]  => K[a i, b k] = V0 B[a,j]? Careful:
        # virtual work: dW = V0 dP_ij B[a,i]? Actually f[a,i] = V0 P_ij B[a,j].
        # df[a,i]/du[b,k] = V0 C_ijkl B[b,l] B[a,j].
        kel = np.einsum('eaj,eijkl,ebl->eakib', B, C, B, optimize=True) * self._V0[:, None, None, None, None]
        rows = np.repeat((T[:, :, None, None, None] * 3 +
                          np.arange(3)[None, None, :, None, None]), 4 * 3, axis=4)
        cols = np.repeat(T[:, None, None, :, None] * 3 + np.arange(3)[None, None, None, None, :], 3, axis=2)
        # simpler: build flat index arrays
        ia = (T[:, :, None, None, None] * 3 + np.arange(3)[None, None, :, None, None])
        ib = (T[:, None, None, :, None] * 3 + np.arange(3)[None, None, None, None, :])
        rows = np.broadcast_to(ia, (ne, 4, 3, 4, 3)).ravel()
        cols = np.broadcast_to(ib, (ne, 4, 3, 4, 3)).ravel()
        K = sp.coo_matrix((kel.ravel(), (rows, cols)),
                          shape=(3 * self._n, 3 * self._n)).tocsr()
        return K

    # ------------------------------------------------------------------ solve
    def _external_force(self, scale):
        f_ext = np.zeros((self._n, 3))
        np.add.at(f_ext, self._top_load_nodes,
                  self._pressure * scale * self._top_load_weights)
        return f_ext

    def _solve_equilibrium(self, target_scale, max_iter=30):
        x = self._x
        f_ext = self._external_force(target_scale)
        free = self._free_dofs
        fd = free
        for it in range(max_iter):
            Fm, P, C, _, _ = self._stress_tangent(x)
            f_int = self._internal_force(P)
            R = f_ext.ravel() - f_int.ravel()
            rn = np.linalg.norm(R[fd])
            if rn <= 1e-9 * max(1.0, np.linalg.norm(f_ext)):
                break
            K = self._assemble_K(C)
            Kff = K[fd][:, fd].tocsc()
            try:
                du = spla.spsolve(Kff, R[fd])
            except Exception:
                du = spla.lsqr(Kff.tocsc(), R[fd])[0]
            if not np.all(np.isfinite(du)):
                du = np.zeros_like(du)
            # line search on residual norm
            step = 1.0
            x_new = x.copy()
            for _ in range(8):
                x_new = x.copy()
                x_new.reshape(-1)[fd] += step * du
                _, Pn, _, _, _ = self._stress_tangent(x_new)
                Rn = f_ext.ravel() - self._internal_force(Pn).ravel()
                new_norm = np.linalg.norm(Rn[fd])
                if np.isfinite(new_norm) and (new_norm < rn or step < 0.05):
                    break
                step *= 0.5
            x = x_new
            if np.max(np.abs(step * du)) < 1e-10:
                break
        self._x = x
        self._scale = target_scale
        self._F_degrad = Fm
        # support reaction at fixed (bottom) dofs: R = f_int - f_ext there
        f_int = self._internal_force(P)
        self._reaction = (f_int - f_ext)[self._bottom].sum(axis=0)

    # ------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        time = float(time)
        dt = time - self._t
        target_scale = self._load_scale(time)
        x_prev = self._x.copy()
        if dt > 0.0:
            self._solve_equilibrium(target_scale)
            self._v = (self._x - x_prev) / dt
        self._t = time

        scale_now = self._scale
        f_ext = self._external_force(scale_now)
        load_total = f_ext[self._top_load_nodes].sum(axis=0) if len(self._top_load_nodes) else np.zeros(3)
        # budget: single domain (whole cube); internal pairs cancel.
        budget_force = np.zeros((1, 2, 3))
        budget_force[0, 0] = load_total                 # applied top traction
        budget_force[0, 1] = self._reaction             # support reaction
        budget_pos = np.zeros((1, 2, 3))
        top_z = float(np.mean(self._x[self._top_node][:, 2]))
        budget_pos[0, 0, 2] = top_z
        budget_pos[0, 1, 2] = 0.0

        frame = StateFrame(
            time=time,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=np.full(self._n, 1.0 / self._n),
            attributes={
                "rest_position": self._X.copy(),
                "bottom_face": self._bottom.astype(float),
                "top_face": self._top_node.astype(float),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "element_deformation_gradient": self._F_degrad.copy(),
                "cumulative_mass_source": np.zeros(1),
                "budget_external_force": budget_force,
                "budget_force_position": budget_pos,
                "external_force": f_ext,
            },
            scalars={
                "applied_traction_pa": self._pressure * scale_now,
                "reference_force": self._pressure * self._top_area,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "solver": "quasi-static nonlinear FEM (compressible Neo-Hookean)",
            "elements": int(self._ne),
            "nodes": int(self._n),
            "final_load_scale": float(self._scale),
            "top_face_z_final": float(np.mean(self._x[self._top_node][:, 2])) if hasattr(self, "_x") else 0.0,
        }
