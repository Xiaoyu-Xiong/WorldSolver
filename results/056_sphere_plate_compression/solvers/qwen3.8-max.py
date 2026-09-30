"""Physically-based solver: incompressible elastic sphere compressed between plates.

Method
------
The unit icosahedron surface mesh (12 vertices, 20 faces) is turned into a
volumetric tetrahedral mesh by adding a central node (13 nodes, 20 tetrahedra).
The material is a nearly-incompressible neo-Hookean hyperelastic solid

    W(F) = mu/2 (tr(F^T F) - 3) - mu ln J + lam/2 (ln J)^2 ,   J = det F,

integrated with fully implicit backward Euler and Newton iterations (exact
material tangent).  The two rigid plates are kinematic boundaries: their inner
faces follow ``initial_inner_faces_z -/+ speed_each * clamp(t, start, end)`` as
declared by the frozen scene metadata.  Contact is enforced with a normal
penalty spring plus tangential (frictional) viscous damping on penetrating
nodes.  Gravity is zero per the configuration.

All contract fields are exported per frame: per-sample mass/velocity, element
deformation gradients, the volume-conservation constraint value/target, and
the momentum-budget fields for the single sphere domain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _commutation9() -> np.ndarray:
    """K with vec(H^T) = K vec(H) for 3x3 (column-major vec)."""
    K = np.zeros((9, 9))
    for k in range(3):
        for l in range(3):  # noqa: E741
            K[3 * k + l, 3 * l + k] = 1.0
    return K


def _dDs_selector() -> np.ndarray:
    """S with vec(dDs) = S u for tet nodes (u0..u3), dDs cols = x_k - x_0."""
    S = np.zeros((9, 12))
    for k in range(3):
        for i in range(3):
            S[3 * k + i, 3 * (k + 1) + i] = 1.0
            S[3 * k + i, i] = -1.0
    return S


def _load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(p) for p in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    return np.asarray(verts, float), np.asarray(faces, dtype=np.int64)


class BenchmarkSolver:
    # ---- material / numerical parameters ---------------------------------
    MU = 120.0            # shear modulus [Pa]
    LAMBDA = 24000.0      # volumetric penalty [Pa] (nu ~ 0.4975)
    K_CONTACT = 6.0e4     # normal contact penalty [N/m]
    C_FRICTION = 120.0    # tangential viscous friction when in contact [N s/m]
    C_DAMP = 15.0         # mass-proportional implicit damping [1/s]
    H_MAX = 0.004         # internal time step [s]
    NEWTON_MAX = 8
    V_CLAMP = 25.0        # safety velocity clamp [m/s]

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        sphere = cfg["sphere"]
        plates = cfg["plates"]

        asset = Path(__file__).resolve().parent.parent / "assets" / "soft_sphere.obj"
        verts, faces = _load_obj(asset)
        center = np.asarray(sphere["center"], dtype=float)
        verts = verts + center

        self.total_mass = float(sphere["mass"])
        self.rest_volume = float(sphere["rest_volume"])
        self.n_surf = len(verts)  # 12

        # ---- volumetric mesh: surface nodes + central node ----------------
        X0 = np.vstack([verts, center[None, :]])  # (13,3)
        tets = []
        for a, b, c in faces:
            nodes = [self.n_surf, int(a), int(b), int(c)]
            Dm = np.column_stack([X0[nodes[k]] - X0[nodes[0]] for k in (1, 2, 3)])
            if np.linalg.det(Dm) < 0.0:
                nodes[1], nodes[2] = nodes[2], nodes[1]
            tets.append(nodes)
        self.tets = np.asarray(tets, dtype=np.int64)  # (20,4)
        self.X0 = X0
        self.n_nodes = len(X0)

        # ---- per-tet precomputation ---------------------------------------
        rho = self.total_mass / self.rest_volume
        self.m_node = np.zeros(self.n_nodes)
        self.V0 = np.zeros(len(self.tets))
        self.Bmats = []
        self.Lmats = []
        self.dofs = []
        S = _dDs_selector()
        I3 = np.eye(3)
        for e, nodes in enumerate(self.tets):
            Dm = np.column_stack([X0[nodes[k]] - X0[nodes[0]] for k in (1, 2, 3)])
            v0 = np.linalg.det(Dm) / 6.0
            self.V0[e] = v0
            B = np.linalg.inv(Dm)
            self.Bmats.append(B)
            self.Lmats.append(np.kron(B.T, I3) @ S)
            dofs = np.empty(12, dtype=np.int64)
            for k, n in enumerate(nodes):
                dofs[3 * k:3 * k + 3] = 3 * n + np.arange(3)
            self.dofs.append(dofs)
            self.m_node[nodes] += rho * v0 / 4.0
        self.rho = rho

        self.K99 = _commutation9()
        self.I9 = np.eye(9)

        # ---- plate kinematics (frozen scene metadata) ---------------------
        self.z_top0 = float(plates["initial_inner_faces_z"][1])
        self.speed = float(plates["speed_each"])
        self.t_start = float(plates["motion_start"])
        self.t_end = float(plates["motion_end"])
        self.min_pos_gap_cfg = float(plates["minimum_positive_gap"])

        # ---- dynamic state -------------------------------------------------
        self.x = X0.copy()
        self.v = np.zeros_like(X0)
        self.t = 0.0
        m = np.maximum(self.m_node, 1e-6)
        self.m_diag = np.repeat(m, 3)          # (39,)
        self.Minv = 1.0 / self.m_diag
        self.Cd_diag = self.C_DAMP * self.m_diag

        self.impulse = np.zeros(3)             # cumulative external impulse
        self.f_ext_prev = np.zeros(3)          # contact force at previous step
        self.last_volume_err = 0.0
        self.steps_done = 0

    # ------------------------------------------------------------------
    def plate_inner_top(self, t: float) -> float:
        s = min(max(t - self.t_start, 0.0), self.t_end - self.t_start)
        return self.z_top0 - self.speed * s

    # ------------------------------------------------------------------
    def _elastic(self, x: np.ndarray):
        """Neo-Hookean gradient (39,) and Hessian (39,39)."""
        g = np.zeros(3 * self.n_nodes)
        H = np.zeros((3 * self.n_nodes, 3 * self.n_nodes))
        mu, lam = self.MU, self.LAMBDA
        for e, nodes in enumerate(self.tets):
            xs = x[nodes]
            Ds = (xs[1:] - xs[0]).T                      # cols x1-x0 ...
            B = self.Bmats[e]
            F = Ds @ B
            det = np.linalg.det(F)
            Jc = max(det, 1.0e-4)
            lnJ = np.log(Jc)
            Finv = np.linalg.inv(F)
            FinvT = Finv.T
            P = mu * (F - FinvT) + lam * lnJ * FinvT
            Gmat = self.V0[e] * (P @ B.T)                # col k -> grad node k+1
            dofs = self.dofs[e]
            g[dofs[0:3]] += -Gmat.sum(axis=1)
            for k in range(3):
                g[dofs[3 * (k + 1):3 * (k + 2)]] += Gmat[:, k]
            w = Finv.flatten()                           # vec(F^-T), col-major
            T9 = (mu * self.I9
                  + (mu - lam * lnJ) * (np.kron(Finv, FinvT) @ self.K99)
                  + lam * np.outer(w, w))
            L = self.Lmats[e]
            H[np.ix_(dofs, dofs)] += self.V0[e] * (L.T @ T9 @ L)
        return g, H

    def _contact(self, x: np.ndarray, v: np.ndarray, zt: float):
        """Penalty contact forces (n,3) and diagonal stiffness/damping adds."""
        f = np.zeros((self.n_nodes, 3))
        kdiag = np.zeros((self.n_nodes, 3))   # d(-f)/dx diagonal (for Jacobian)
        cdiag = np.zeros((self.n_nodes, 3))   # d(-f)/dv diagonal
        zb = -zt
        for i in range(self.n_nodes):
            pen = zb - x[i, 2]
            if pen > 0.0:
                f[i, 2] += self.K_CONTACT * pen
                f[i, 0] -= self.C_FRICTION * v[i, 0]
                f[i, 1] -= self.C_FRICTION * v[i, 1]
                kdiag[i, 2] += self.K_CONTACT
                cdiag[i, 0] += self.C_FRICTION
                cdiag[i, 1] += self.C_FRICTION
            pen = x[i, 2] - zt
            if pen > 0.0:
                f[i, 2] -= self.K_CONTACT * pen
                f[i, 0] -= self.C_FRICTION * v[i, 0]
                f[i, 1] -= self.C_FRICTION * v[i, 1]
                kdiag[i, 2] += self.K_CONTACT
                cdiag[i, 0] += self.C_FRICTION
                cdiag[i, 1] += self.C_FRICTION
        return f, kdiag, cdiag

    # ------------------------------------------------------------------
    def _substep(self, h: float, t_next: float) -> None:
        zt = self.plate_inner_top(t_next)
        v_old = self.v.ravel().copy()
        v_new = v_old.copy()
        Md = self.m_diag
        for _ in range(self.NEWTON_MAX):
            x_new = (self.x.ravel() + h * v_new).reshape(self.n_nodes, 3)
            g, H = self._elastic(x_new)
            fc, kdiag, cdiag = self._contact(x_new, v_new.reshape(self.n_nodes, 3), zt)
            fdamp = -self.Cd_diag * v_new
            f_total = -g + fc.ravel() + fdamp
            R = Md * (v_new - v_old) - h * f_total
            J = (
                np.diag(Md + h * (self.Cd_diag + cdiag.ravel()) + (h * h) * kdiag.ravel())
                + (h * h) * H
            )
            J[np.diag_indices_from(J)] += 1.0e-9 * max(np.trace(J), 1.0)
            try:
                dv = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dv = np.linalg.lstsq(J, -R, rcond=None)[0]
            if not np.isfinite(dv).all():
                break
            v_new = v_new + dv
            if np.max(np.abs(dv)) < 1.0e-8:
                break
        if not np.isfinite(v_new).all():
            v_new = 0.5 * v_old  # extreme fallback, keeps state finite
        speed = np.abs(v_new.reshape(self.n_nodes, 3)).max(axis=1)
        over = speed > self.V_CLAMP
        if over.any():
            v_new = v_new.reshape(self.n_nodes, 3)
            v_new[over] *= self.V_CLAMP / speed[over, None]
            v_new = v_new.ravel()
        self.x = (self.x.ravel() + h * v_new).reshape(self.n_nodes, 3)
        self.v = v_new.reshape(self.n_nodes, 3)
        # external force bookkeeping (contact tractions from plates)
        f_ext = self._contact(self.x, self.v, zt)[0].sum(axis=0)
        self.impulse += 0.5 * (self.f_ext_prev + f_ext) * h
        self.f_ext_prev = f_ext
        self.t = t_next
        self.steps_done += 1

    # ------------------------------------------------------------------
    def _frame(self, t: float) -> StateFrame:
        n = self.n_surf
        F_elems = np.zeros((len(self.tets), 3, 3))
        volume = 0.0
        for e, nodes in enumerate(self.tets):
            xs = self.x[nodes]
            Ds = (xs[1:] - xs[0]).T
            F = Ds @ self.Bmats[e]
            F_elems[e] = F
            volume += self.V0[e] * np.linalg.det(F)
        mass = np.full(n, self.total_mass / n)
        zt = self.plate_inner_top(t)
        f_ext = self._contact(self.x, self.v, zt)[0].sum(axis=0)
        vol_err = abs(volume - self.rest_volume) / self.rest_volume
        self.last_volume_err = vol_err
        frame = StateFrame(
            time=float(t),
            position=self.x[:n].copy(),
            velocity=self.v[:n].copy(),
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "rest_position": self.X0[:n].copy(),
            },
            fields={
                "element_deformation_gradient": F_elems,
                "constraint_value": np.array([volume]),
                "constraint_target": np.array([self.rest_volume]),
                "budget_domain_weights": np.ones((1, n)),
                "budget_external_force": f_ext[None, None, :].copy(),
                "budget_cumulative_external_impulse": self.impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            },
            scalars={
                "constraint_scale": self.rest_volume,
                "reference_momentum": self.total_mass * max(self.speed, 1.0e-3),
                "rest_volume": self.rest_volume,
                "initial_plate_gap": 2.0 * self.z_top0,
                "current_plate_gap": 2.0 * zt,
                "minimum_positive_gap": self.min_pos_gap_cfg,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        if time > self.t + 1.0e-12:
            remaining = time - self.t
            n_sub = max(1, int(np.ceil(remaining / self.H_MAX)))
            h = remaining / n_sub
            t_cur = self.t
            for k in range(n_sub):
                self._substep(h, t_cur + (k + 1) * h)
            self.t = time  # remove floating-point drift; frame times exact
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "implicit neo-Hookean tetrahedral FEM (20 tets, 13 nodes)",
            "substeps": self.steps_done,
            "simulated_time": self.t,
            "volume_error_last": float(
                abs(self._current_volume() - self.rest_volume) / self.rest_volume
            ),
        }

    def _current_volume(self) -> float:
        volume = 0.0
        for e, nodes in enumerate(self.tets):
            xs = self.x[nodes]
            Ds = (xs[1:] - xs[0]).T
            volume += self.V0[e] * np.linalg.det(Ds @ self.Bmats[e])
        return volume
