"""Physically-based solver for the armadillo roller stick-slip scene.

Method
------
* The Armadillo surface mesh (24 vertices, closed low-poly proxy) is
  tetrahedralized once with a Delaunay triangulation of its rest vertices,
  giving a volumetric (3D solid) discretization.
* Elasticity: co-rotated Saint-Venant-Kirchhoff linear tetrahedra
  (E = 5e5 Pa, nu = 0.3), assembled with a frozen-rotation consistent
  tangent.  Integrated with backward Euler + Newton iterations
  (72-DOF dense solves) and small mass-proportional damping.
* Rollers: four kinematic cylinders (axis y, radius 0.22, centers
  (x,z) = (-0.2,+-0.3), (0.3,+-0.3)).  The configuration prescribes
  counter-rotation with |omega| = 5 rad/s and intended transport toward
  +x; the surface velocity of the upper roller at its lower contact is
  v = omega x r = (omega_y * dz, 0, -omega_y * dx), so transport in +x
  requires omega_y = -5 (upper) and +5 (lower).  Angular velocity is
  ramped linearly from t=0.35 over 0.25 s.
* Contact: node-vs-cylinder penalty contact (k = 1e5 N/m, unilateral,
  clamped to the roller axial extent |y| <= 0.36).  Regularized Coulomb
  friction (mu_s = mu_d = 0.5, regularization delta = 5e-3 m/s) acting on
  the tangential relative velocity node-vs-roller-surface.  Both normal
  and friction tangent stiffness enter the Newton matrix, so stick/slip
  emerges naturally from elasticity + friction: catch, stretch, slip,
  catch again.

All required trajectory fields are exported per frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------
# Scene constants (from frozen config / scene geometry)
# ----------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
TOTAL_MASS = 2.4                      # kg (from scene.initial_state)
E_YOUNG = 5.0e5                       # Pa (moderate stiffness reference)
NU = 0.3
RHO = None                            # computed from tet volumes

ROLLER_R = 0.22
ROLLER_HALFLEN = 0.72 / 2.0           # 0.36
# (x_c, z_c, omega_y_effective) -- signs chosen so that the roller surface
# velocity at the contact transports the body toward +x (intended direction).
ROLLERS = (
    (-0.2, 0.3, -5.0),   # upper_left
    (0.3, 0.3, -5.0),    # upper_right
    (-0.2, -0.3, 5.0),   # lower_left
    (0.3, -0.3, 5.0),    # lower_right
)
RAMP_T0 = 0.35
RAMP_DUR = 0.25

MU_S = 0.5
MU_D = 0.5
K_PEN = 1.0e5                         # N/m penalty stiffness per node
FRIC_DELTA = 5.0e-3                   # m/s regularization
BETA_DAMP = 2.0                       # 1/s mass-proportional damping

H_SUB = 1.0e-3                        # substep (s)
NEWTON_ITERS = 4

REF_VELOCITY = 5.0 * ROLLER_R         # 1.1 m/s roller surface speed
REF_FORCE = TOTAL_MASS * 9.81         # body weight scale
REF_MOMENTUM = TOTAL_MASS * REF_VELOCITY
STICK_TOL = 0.02                      # m/s stick-slip classification tol


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, dtype=float).copy()
        self.n_nodes = len(x0)

        # --- volumetric tetrahedralization of the rest configuration ---
        rest = np.asarray(frame0.attributes.get("rest_position", x0), dtype=float)
        try:
            tri = Delaunay(rest)
        except Exception:
            tri = Delaunay(rest, qhull_options=["QJ1e-8"])
        tets = tri.simplices.copy()
        # remove sliver / degenerate tets (coplanar corner quads can appear)
        vols_all = self._tet_volumes(rest, tets)
        keep = vols_all > 1e-9 * max(1.0, float(np.max(vols_all)))
        tets = tets[keep]
        vols = vols_all[keep]
        self.tets = tets
        self.ne = len(tets)

        vol_total = float(vols.sum())
        rho = TOTAL_MASS / vol_total
        # lumped nodal masses: quarter of each tet volume
        m = np.zeros(self.n_nodes)
        np.add.at(m, tets[:, 0], vols * rho / 4.0)
        np.add.at(m, tets[:, 1], vols * rho / 4.0)
        np.add.at(m, tets[:, 2], vols * rho / 4.0)
        np.add.at(m, tets[:, 3], vols * rho / 4.0)
        m *= TOTAL_MASS / m.sum()
        self.mass = m
        self.m_diag = np.repeat(m, 3)

        # --- material parameters (Lame) ---
        self.lam = E_YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        self.mu = E_YOUNG / (2.0 * (1.0 + NU))

        # --- per-tet rest shape-function gradients G (Ne,4,3):
        #     F = sum_a x_a outer G_a -----------------------------------
        G = np.zeros((self.ne, 4, 3))
        for t in range(self.ne):
            X = rest[tets[t]]
            Dm = np.stack([X[1] - X[0], X[2] - X[0], X[3] - X[0]], axis=1)  # cols
            Ei = np.linalg.inv(Dm)   # F = Ds @ Ei
            G[t, 1] = Ei[0]
            G[t, 2] = Ei[1]
            G[t, 3] = Ei[2]
            G[t, 0] = -(Ei[0] + Ei[1] + Ei[2])
        self.G = G
        self.V0 = vols
        self.GG = np.einsum("nai,nbi->nab", G, G)   # G_a . G_b

        # node index helper for dense Hessian scatter (single bincount pass)
        ar3 = np.arange(3)
        self._pairs = [(a, b) for a in range(4) for b in range(4)]
        rows_list, cols_list = [], []
        for (a, b) in self._pairs:
            rr = tets[:, a][:, None] * 3 + ar3
            cc = tets[:, b][:, None] * 3 + ar3
            rows_list.append(np.broadcast_to(rr[:, :, None], (self.ne, 3, 3)).ravel())
            cols_list.append(np.broadcast_to(cc[:, None, :], (self.ne, 3, 3)).ravel())
        rows_all = np.concatenate(rows_list)
        cols_all = np.concatenate(cols_list)
        self._kflat = rows_all * (3 * self.n_nodes) + cols_all
        self._kdof = 3 * self.n_nodes

        # --- state ---
        self.rest = rest
        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.t = 0.0
        self.impulse_cum = np.zeros(3)
        self._fext_prev = None
        self._diag = {"steps": 0}

    # ------------------------------------------------------------------
    @staticmethod
    def _tet_volumes(P, tets):
        X = P[tets]
        D = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)
        return np.abs(np.linalg.det(D)) / 6.0

    # ------------------------------------------------------------------
    def _ramp(self, t):
        return float(np.clip((t - RAMP_T0) / RAMP_DUR, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _contact_geometry(self, x, v, s_ramp):
        """Node contact against the four kinematic cylinders.

        Returns per-node arrays: pen, n (3), N (normal force), v_surf (3),
        and index of the deepest-penetrating roller.
        """
        nn = self.n_nodes
        best_pen = np.full(nn, -np.inf)
        best = np.zeros(nn, dtype=int)
        pen_r = np.zeros((4, nn))
        n_r = np.zeros((4, nn, 3))
        vs_r = np.zeros((4, nn, 3))
        for ri, (xc, zc, om0) in enumerate(ROLLERS):
            om = om0 * s_ramp
            dx = x[:, 0] - xc
            dz = x[:, 2] - zc
            r = np.sqrt(dx * dx + dz * dz) + 1e-12
            pen = ROLLER_R - r
            inside_ax = np.abs(x[:, 1]) <= ROLLER_HALFLEN + 1e-9
            pen_r[ri] = np.where(inside_ax, pen, -np.inf)
            n_r[ri, :, 0] = dx / r
            n_r[ri, :, 2] = dz / r
            vs_r[ri, :, 0] = om * dz
            vs_r[ri, :, 2] = -om * dx
        best = np.argmax(pen_r, axis=0)
        rows = np.arange(nn)
        best_pen = pen_r[best, rows]
        best_pen = np.where(np.isfinite(best_pen), best_pen, -1.0)
        n = n_r[best, rows]           # (nn,3) unit radial normals
        vs = vs_r[best, rows]         # (nn,3) roller surface velocities
        active = best_pen > 0.0
        Nforce = np.where(active, K_PEN * np.maximum(best_pen, 0.0), 0.0)
        return active, best_pen, n, Nforce, vs

    # ------------------------------------------------------------------
    def _assemble(self, x, v, s_ramp, h):
        """Total force F (n,3) and tangent stiffness K (3n,3n) = -dF/dx."""
        n = self.n_nodes
        F = np.zeros((n, 3))
        K = np.zeros((3 * n, 3 * n))

        # ---- co-rotational elasticity ----
        xt = x[self.tets]                              # (Ne,4,3)
        Fdef = np.einsum("nai,naj->nij", xt, self.G)   # (Ne,3,3)
        U, _S, Vt = np.linalg.svd(Fdef)
        R = U @ Vt
        det = np.linalg.det(R)
        flip = det < 0.0
        if np.any(flip):
            U[flip, :, 2] *= -1.0
            R = U @ Vt
        eye = np.eye(3)
        eps = 0.5 * (np.einsum("nji,njk->nik", R, Fdef)
                     + np.einsum("nij,nkj->nik", Fdef, R))
        eps -= eye[None]
        tr = np.einsum("nii->n", eps)
        Smat = self.lam * tr[:, None, None] * eye[None] + 2.0 * self.mu * eps
        RS = np.einsum("nij,njk->nik", R, Smat)
        fel = -self.V0[:, None, None] * np.einsum("nij,naj->nai", RS, self.G)
        np.add.at(F, self.tets[:, 0], fel[:, 0])
        np.add.at(F, self.tets[:, 1], fel[:, 1])
        np.add.at(F, self.tets[:, 2], fel[:, 2])
        np.add.at(F, self.tets[:, 3], fel[:, 3])

        # frozen-rotation tangent Hessian, scattered with one bincount pass
        RG = np.einsum("nij,naj->nai", R, self.G)      # (Ne,4,3)
        blocks = []
        for (a, b) in self._pairs:
            blocks.append(self.V0[:, None, None] * (
                self.lam * RG[:, a, :, None] * RG[:, b, None, :]
                + self.mu * self.GG[:, a, b, None, None] * eye[None]
                + self.mu * RG[:, b, :, None] * RG[:, a, None, :]
            ))
        vals = np.stack(blocks)                        # (16,Ne,3,3)
        K += np.bincount(self._kflat, weights=vals.ravel(),
                         minlength=self._kdof * self._kdof
                         ).reshape(self._kdof, self._kdof)

        # ---- gravity + damping ----
        F += self.mass[:, None] * GRAVITY[None]
        F -= BETA_DAMP * self.mass[:, None] * v

        # ---- roller contact: penalty normal + regularized Coulomb friction
        active, pen, nc, Nf, vs = self._contact_geometry(x, v, s_ramp)
        if np.any(active):
            ia = np.nonzero(active)[0]
            na = nc[ia]                       # (m,3)
            Nfa = Nf[ia]
            F[ia] += Nfa[:, None] * na        # normal push-out force

            vrel = v[ia] - vs[ia]
            vn = np.einsum("mi,mi->m", vrel, na)[:, None]
            vt = vrel - vn * na               # tangential relative velocity
            tt = np.linalg.norm(vt, axis=1)
            d = tt + FRIC_DELTA
            F[ia] += -MU_D * Nfa[:, None] * vt / d[:, None]

            # tangent contributions (normal + linearized friction, N lagged)
            Pm = eye[None] - na[:, :, None] * na[:, None, :]
            Kf = (MU_D * Nfa)[:, None, None] * Pm / (h * d)[:, None, None]
            big = tt > 1e-9
            if np.any(big):
                Pproj = np.einsum("mi,mik->mk", vt, Pm)      # (t^T P)_k
                corr = np.einsum("mj,mk->mjk", vt, Pproj)    # t t^T P
                coef = (MU_D * Nfa) / (h * d * d * np.maximum(tt, 1e-12))
                Kf[big] -= coef[big, None, None] * corr[big]
            for j, i in enumerate(ia):
                ni = na[j]
                K[3 * i:3 * i + 3, 3 * i:3 * i + 3] += (
                    K_PEN * np.outer(ni, ni) + Kf[j]
                )

        return F, K, active, pen, nc, Nf, vs

    # ------------------------------------------------------------------
    def _external_force(self, x, v, s_ramp):
        """Gravity + contact (normal + friction) forces, for the momentum
        budget.  Excludes internal elasticity and artificial damping."""
        n = self.n_nodes
        F = self.mass[:, None] * GRAVITY[None]
        active, pen, nc, Nf, vs = self._contact_geometry(x, v, s_ramp)
        if np.any(active):
            ia = np.nonzero(active)[0]
            na = nc[ia]
            F[ia] += Nf[ia, None] * na
            vrel = v[ia] - vs[ia]
            vn = np.einsum("mi,mi->m", vrel, na)[:, None]
            vt = vrel - vn * na
            tt = np.linalg.norm(vt, axis=1)
            F[ia] += -MU_D * Nf[ia, None] * vt / (tt + FRIC_DELTA)[:, None]
        return F

    # ------------------------------------------------------------------
    def _substep(self, h):
        t1 = self.t + h
        s = self._ramp(t1)
        x0 = self.x
        v0 = self.v
        x = x0 + h * v0
        Mh2 = self.m_diag / (h * h)
        A_diag_const = Mh2 + BETA_DAMP * self.m_diag / h
        for _ in range(NEWTON_ITERS):
            v = (x - x0) / h
            F, K, *_ = self._assemble(x, v, s, h)
            Ff = F.ravel()
            r = Mh2 * (x - x0 - h * v0).ravel() - Ff
            A = K.copy()
            A[np.diag_indices_from(A)] += A_diag_const
            try:
                dx = np.linalg.solve(A, -r)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)):
                break
            # safeguard against diverging Newton updates
            nrm = np.linalg.norm(dx)
            if nrm > 0.05:
                dx *= 0.05 / nrm
            x = x + dx.reshape(-1, 3)
            if np.linalg.norm(dx) < 1e-8:
                break
        self.v = (x - x0) / h
        self.x = x
        self.t = t1
        # cumulative external impulse (trapezoid on substep)
        fext1 = self._external_force(x, self.v, s)
        if self._fext_prev is None:
            self.impulse_cum += fext1.sum(axis=0) * h
        else:
            self.impulse_cum += 0.5 * (self._fext_prev + fext1).sum(axis=0) * h
        self._fext_prev = fext1
        self._diag["steps"] += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        if not hasattr(self, "_fext_prev"):
            self._fext_prev = self._external_force(self.x, self.v, self._ramp(self.t))
        target = float(time)
        while self.t < target - 1e-12:
            h = min(H_SUB, target - self.t)
            self._substep(h)
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, time):
        n = self.n_nodes
        x, v = self.x, self.v
        s = self._ramp(self.t)

        # deformation gradients
        xt = x[self.tets]
        Fdef = np.einsum("nai,naj->nij", xt, self.G)

        # contact state for friction fields (deepest roller per node)
        active, pen, nc, Nf, vs = self._contact_geometry(x, v, s)
        vrel = v - vs
        vn = np.einsum("ni,ni->n", vrel, nc)[:, None]
        vt = vrel - vn * nc
        tt = np.linalg.norm(vt, axis=1)
        ft = -MU_D * Nf[:, None] * vt / (tt + FRIC_DELTA)[:, None]
        ft[~active] = 0.0
        fn_force = np.where(active, Nf, 0.0)

        fext = self._external_force(x, v, s)

        G_domains = 1
        K_mats = 1
        C = n
        fields = {
            "budget_domain_weights": np.ones((G_domains, n)),
            "budget_external_force": fext.sum(axis=0)[None, None, :].copy(),
            "budget_cumulative_external_impulse": self.impulse_cum[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((G_domains, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((G_domains, 3)),
            "cumulative_mass_source": np.zeros(K_mats),
            "element_deformation_gradient": Fdef,
            "friction_active_mask": active.astype(float),
            "friction_static_coefficient": np.full(C, MU_S),
            "friction_dynamic_coefficient": np.full(C, MU_D),
            "friction_normal": nc,
            "friction_normal_force": fn_force,
            "friction_relative_velocity": vrel,
            "friction_tangential_force": ft,
        }
        scalars = {
            "friction_stick_speed_tolerance": STICK_TOL,
            "reference_force": REF_FORCE,
            "reference_momentum": REF_MOMENTUM,
            "reference_velocity": REF_VELOCITY,
            "roller_friction": MU_S,
            "reference_youngs_modulus_pa": E_YOUNG,
        }
        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            mass=self.mass.copy(),
            body_id=np.ones(n, dtype=np.int64),
            attributes={
                "rest_position": self.rest.copy(),
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        cx = float(self.x[:, 0].mean())
        return {
            "steps": self._diag.get("steps", 0),
            "time": self.t,
            "center_of_mass_x": cx,
            "max_speed": float(np.linalg.norm(self.v, axis=1).max()),
        }
