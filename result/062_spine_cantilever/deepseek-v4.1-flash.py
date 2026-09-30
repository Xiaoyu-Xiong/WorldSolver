"""Gravity sag of the volumetric segmented soft spine cantilever.

Physical model
--------------
The frozen scene mesh is a chain of 21 axis-aligned box segments (one root
block, ten tapering "vertebra" bodies and ten thin necks).  Each box is a
trilinear hexahedral finite element (8 nodes, 2x2x2 Gauss quadrature) with a
compressible neo-Hookean strain energy

    W = mu/2 (tr C - 3) - mu ln J + lambda/2 (ln J)^2,

mu and lambda taken from the frozen Young's modulus and Poisson ratio.  The
frozen surface mesh gives the boxes disjoint node sets, so the neck ends are
coupled to the neighbouring bodies with penalty tie constraints: every neck
end-face node is tied to the trilinear interpolation of the box that contains
it in the reference configuration.  That reproduces the volumetric union of
the frozen geometry while keeping the exported nodes exactly the frozen asset
vertices (so the renderer's deformed surface binding stays valid).

Time integration is implicit backward Euler with a full Newton-Raphson solve
of the nonlinear momentum balance (dense 3N x 3N system), which is stable for
the nearly incompressible material.  Motion is deterministic and finite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_SQRT1_3 = 1.0 / np.sqrt(3.0)

# Local hexahedron node coordinates in the generator's vertex order.
_HEX_A = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)


def _hex_shape(eps: np.ndarray):
    """Trilinear shape functions and local derivatives at `eps` (...,3)."""
    e = np.asarray(eps, dtype=float).reshape(-1, 3)
    ax = _HEX_A[None, :, 0]
    ay = _HEX_A[None, :, 1]
    az = _HEX_A[None, :, 2]
    x = e[:, 0:1]
    y = e[:, 1:2]
    z = e[:, 2:3]
    n = 0.125 * (1.0 + ax * x) * (1.0 + ay * y) * (1.0 + az * z)
    d = np.empty((e.shape[0], 8, 3), dtype=float)
    d[:, :, 0] = 0.125 * ax * (1.0 + ay * y) * (1.0 + az * z)
    d[:, :, 1] = 0.125 * ay * (1.0 + ax * x) * (1.0 + az * z)
    d[:, :, 2] = 0.125 * az * (1.0 + ax * x) * (1.0 + ay * y)
    return n, d


class BenchmarkSolver:
    """Deterministic three-dimensional solid solver for the spine cantilever."""

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)

        material = dict(cfg.get("material", {}))
        self.rho = float(material.get("density", 1070.0))
        youngs = float(material.get("youngs_modulus_pa", 150000.0))
        poisson = float(material.get("poisson_ratio", 0.4999))
        poisson = min(max(poisson, -0.999), 0.4999)
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        # ---------------- frozen reference geometry -------------------------
        initial = scene.initial_state()
        local = np.asarray(initial.position, dtype=float)
        translation, _asset = self._geometry_hint(cfg)
        x_ref = local + translation[None, :]

        n_nodes = int(x_ref.shape[0])
        if n_nodes % 8 != 0:
            raise ValueError("frozen spine mesh is not a chain of hexahedral boxes")
        self.n_nodes = n_nodes
        self.n_elems = n_nodes // 8
        self.conn = np.arange(n_nodes).reshape(self.n_elems, 8)
        self.x_ref = x_ref

        # ---------------- boundary condition ---------------------------------
        fixed = None
        attrs = getattr(initial, "attributes", {}) or {}
        if "fixed" in attrs:
            fixed = np.asarray(attrs["fixed"], dtype=float).reshape(-1) > 0.0
        if fixed is None or fixed.shape[0] != n_nodes:
            region = dict(cfg.get("boundary_conditions", {}).get("fixed_region", {}))
            x_max = float(region.get("x_max", 0.08))
            fixed = local[:, 0] <= x_max
        self.fixed = np.asarray(fixed, dtype=bool)
        dof_fixed = np.repeat(self.fixed, 3)
        self.free_idx = np.flatnonzero(~dof_fixed)

        # ---------------- quadrature / reference operators -------------------
        g = _SQRT1_3
        gauss = np.array(
            [[a, b, c] for a in (-g, g) for b in (-g, g) for c in (-g, g)], dtype=float
        )
        _n_gp, dN_gp = _hex_shape(gauss)  # (8,8), (8,8,3)
        xe_ref = x_ref[self.conn]  # (E,8,3)
        j0 = np.einsum("nai,gaj->ngij", xe_ref, dN_gp)  # (E,G,3,3)
        det0 = np.linalg.det(j0)
        if not np.all(np.isfinite(det0)) or np.any(det0 <= 0.0):
            raise ValueError("degenerate reference hexahedron in frozen mesh")
        self.det0 = det0
        j0_inv = np.linalg.inv(j0)
        self.grad_n = np.einsum("gaj,ngjk->ngak", dN_gp, j0_inv)  # dN/dX at Gauss points

        _n_c, dN_c = _hex_shape(np.zeros((1, 3)))
        self.dn_c = dN_c[0]  # (8,3)
        self.j0c_inv = np.linalg.inv(np.einsum("nai,aj->nij", xe_ref, self.dn_c))

        # ---------------- lumped nodal mass ----------------------------------
        volume = det0.sum(axis=1)
        self.elem_volume = volume
        mass = np.zeros(n_nodes, dtype=float)
        for e in range(self.n_elems):
            mass[self.conn[e]] += self.rho * volume[e] / 8.0
        self.mass = mass
        self.total_mass = float(mass.sum())

        # ---------------- tie constraints ------------------------------------
        self.tie_stiffness = 1.0e8
        self.ties = self._build_ties()
        self.tie_count = int(sum(len(hosts) for _n, hosts in self.ties))
        self.untied = int(sum(1 for _n, hosts in self.ties if not hosts))

        # ---------------- time integration -----------------------------------
        sample_dt = float(cfg.get("sample_dt", 0.02))
        self.dt = sample_dt / 4.0
        # No explicit viscous term: backward Euler already damps the fast
        # modes, and keeping the momentum budget free of dissipative forces
        # lets the reported external forces close it exactly.
        self.damping = 0.0
        self.max_iter = 20
        self.last_iterations = 0

        self.x = x_ref.copy()
        self.v = np.zeros_like(x_ref)
        self.t = 0.0
        self.f_int = np.zeros_like(x_ref)
        self.f_tie = np.zeros_like(x_ref)
        self.steps = 0

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _geometry_hint(cfg: dict):
        translation = np.array([0.0, 0.0, 0.42], dtype=float)
        asset = "assets/spine_arm.obj"
        geometry = cfg.get("scene_geometry")
        if not isinstance(geometry, dict):
            geometry = None
            path = Path(__file__).resolve().parents[1] / "scene_geometry.yaml"
            try:
                from worldsolver.config import load_yaml

                geometry = load_yaml(path)
            except Exception:
                geometry = None
        if isinstance(geometry, dict):
            for item in geometry.get("objects", []) or []:
                if item.get("id") == "spine_arm":
                    trans = (item.get("transform") or {}).get("translation")
                    if trans is not None:
                        translation = np.asarray(trans, dtype=float).reshape(3)
                    hint = (item.get("geometry") or {}).get("asset")
                    if hint:
                        asset = str(hint)
        return translation, asset

    def _build_ties(self):
        """Tie every neck end node to the box that contains it at rest."""
        x_ref = self.x_ref
        conn = self.conn
        ties = []
        for node in range(self.n_nodes):
            owners = np.flatnonzero((conn == node).any(axis=1))
            hosts = []
            for e in range(self.n_elems):
                if e in owners:
                    continue
                xi = np.zeros(3)
                ok = False
                for _ in range(25):
                    n_val, d_val = _hex_shape(xi)
                    resid = n_val[0] @ x_ref[conn[e]] - x_ref[node]
                    if float(np.max(np.abs(resid))) < 1e-13:
                        ok = True
                        break
                    jac = np.einsum("ai,aj->ij", x_ref[conn[e]], d_val[0])
                    try:
                        xi = xi - np.linalg.solve(jac, resid)
                    except np.linalg.LinAlgError:
                        break
                if ok and np.all(xi >= -1e-8) and np.all(xi <= 1.0 + 1e-8):
                    n_val, _d = _hex_shape(xi)
                    hosts.append((int(e), n_val[0].copy()))
            if not hosts:
                # Robust fallback: inverse-distance weights to the nearest box.
                gap = np.array(
                    [
                        float(
                            np.linalg.norm(
                                x_ref[conn[e]] - x_ref[node][None, :], axis=1
                            ).min()
                        )
                        for e in range(self.n_elems)
                    ],
                    dtype=float,
                )
                nearest = int(np.argmin(gap))
                d = np.linalg.norm(x_ref[conn[nearest]] - x_ref[node][None, :], axis=1)
                w = 1.0 / np.maximum(d, 1e-9)
                w = w / w.sum()
                hosts.append((nearest, w))
            ties.append((int(node), hosts))
        return ties

    # ------------------------------------------------------------- assembly
    def _internal_forces(self, x: np.ndarray, with_tangent: bool = True):
        grad_n = self.grad_n
        det0 = self.det0
        xe = x[self.conn]
        jac = np.einsum("nai,ngaj->ngij", xe, grad_n)
        f = np.einsum("ngij,ngjk->ngik", jac, self.j0c_inv)
        det_f = np.linalg.det(f)
        det_f = np.maximum(det_f, 1e-10)
        finv_t = np.swapaxes(np.linalg.inv(f), -1, -2)
        ln_j = np.log(det_f)
        p_stress = self.mu * (f - finv_t) + (self.lam * ln_j)[:, :, None, None] * finv_t

        f_el = -np.einsum("ngiJ,ngaJ,ng->nai", p_stress, grad_n, det0)
        f_int = np.zeros_like(x)
        np.add.at(f_int, self.conn.reshape(-1), f_el.reshape(-1, 3))
        if not with_tangent:
            return f_int, None

        t1 = np.einsum("ngiL,ngkJ->ngiJkL", finv_t, finv_t)
        t2 = np.einsum("ngiJ,ngkL->ngiJkL", finv_t, finv_t)
        eye = np.eye(3)
        i4 = np.einsum("ik,JL->iJkL", eye, eye)[None, None]
        tangent = (
            self.mu * i4
            + self.mu * t1
            + self.lam * t2
            - (self.lam * ln_j)[:, :, None, None, None, None] * t1
        )
        k_el = np.einsum("ngaJ,ngiJkL,ngbL,ng->naibk", grad_n, tangent, grad_n, det0)
        n3 = 3 * self.n_nodes
        k_int = np.zeros((n3, n3), dtype=float)
        dofs = 3 * self.conn[:, :, None] + np.arange(3)[None, None, :]
        k_loc = k_el.transpose(0, 1, 3, 2, 4).reshape(self.n_elems, 24, 24)
        for e in range(self.n_elems):
            ii = dofs[e].reshape(-1)
            k_int[np.ix_(ii, ii)] += k_loc[e]
        return f_int, k_int

    def _tie_forces(self, x: np.ndarray, with_tangent: bool = True):
        k = self.tie_stiffness
        eye = np.eye(3)
        f_tie = np.zeros_like(x)
        n3 = 3 * self.n_nodes
        k_tie = np.zeros((n3, n3), dtype=float) if with_tangent else None
        for node, hosts in self.ties:
            dn = 3 * node + np.arange(3)
            for e, w in hosts:
                cn = self.conn[e]
                gap = x[node] - w @ x[cn]
                f_tie[node] -= k * gap
                np.add.at(f_tie, cn, k * np.outer(w, gap))
                if not with_tangent:
                    continue
                db = (3 * cn[:, None] + np.arange(3)[None, :]).reshape(-1)
                # (3,24): b[i, a*3+j] = w_a delta_ij
                b = np.einsum("a,ij->iaj", w, eye).reshape(3, 24)
                k_tie[np.ix_(dn, dn)] += k * eye
                k_tie[np.ix_(dn, db)] -= k * b
                k_tie[np.ix_(db, dn)] -= k * b.T
                k_tie[np.ix_(db, db)] += k * np.einsum("a,b,il->aibl", w, w, eye).reshape(24, 24)
        return f_tie, k_tie

    def _residual(self, x, x_n, v_n, dt):
        f_int, k_int = self._internal_forces(x, with_tangent=True)
        f_tie, k_tie = self._tie_forces(x, with_tangent=True)
        res = (self.mass / dt**2)[:, None] * (x - x_n) - (self.mass / dt)[:, None] * v_n
        res = res + (self.damping * self.mass / dt)[:, None] * (x - x_n)
        res = res + f_int + f_tie - self.mass[:, None] * self.gravity[None, :]
        jac = k_int + k_tie
        diag = self.mass / dt**2 + self.damping * self.mass / dt
        jac[np.arange(3 * self.n_nodes), np.arange(3 * self.n_nodes)] += diag
        return res, jac, f_int, f_tie

    def _residual_only(self, x, x_n, v_n, dt):
        f_int, _k = self._internal_forces(x, with_tangent=False)
        f_tie, _k2 = self._tie_forces(x, with_tangent=False)
        res = (self.mass / dt**2)[:, None] * (x - x_n) - (self.mass / dt)[:, None] * v_n
        res = res + (self.damping * self.mass / dt)[:, None] * (x - x_n)
        res = res + f_int + f_tie - self.mass[:, None] * self.gravity[None, :]
        return res

    # ------------------------------------------------------------ time loop
    def _step(self, dt: float) -> None:
        x_n = self.x
        v_n = self.v
        idx = self.free_idx
        x = x_n + dt * v_n
        scale = max(float(np.abs(self.mass[:, None] * self.gravity[None, :]).sum()), 1.0)
        tol = 1e-8 * scale
        self.last_iterations = 0
        f_int = self.f_int
        f_tie = self.f_tie
        for _ in range(self.max_iter):
            self.last_iterations += 1
            res, jac, f_int, f_tie = self._residual(x, x_n, v_n, dt)
            r_vec = res.reshape(-1)[idx]
            r_norm = float(np.max(np.abs(r_vec)))
            if r_norm <= tol:
                break
            kff = jac[np.ix_(idx, idx)]
            try:
                dx = np.linalg.solve(kff, -r_vec)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(kff, -r_vec, rcond=None)[0]
            if not np.all(np.isfinite(dx)):
                break
            # Backtracking line search on the nonlinear residual; the full
            # Newton step is kept when no shorter step improves the residual.
            step = 1.0
            x_try = x.copy()
            x_try.reshape(-1)[idx] += dx
            best = r_norm
            for _ls in range(4):
                cand = x.copy()
                cand.reshape(-1)[idx] += step * dx
                r_cand = float(
                    np.max(np.abs(self._residual_only(cand, x_n, v_n, dt).reshape(-1)[idx]))
                )
                if r_cand < best:
                    best = r_cand
                    x_try = cand
                    break
                step *= 0.5
            x = x_try
            if best <= tol:
                break
        self.x = x
        self.v = (x - x_n) / dt
        # Refresh stored force terms at the accepted state (used for the
        # wall-traction term of the external-force budget).
        self.f_int, _ki = self._internal_forces(x, with_tangent=False)
        self.f_tie, _kt = self._tie_forces(x, with_tangent=False)
        self.t += dt
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 100000:
            dt = min(self.dt, target - self.t)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
        return self._frame(target)

    # -------------------------------------------------------------- outputs
    def _deformation_gradient(self) -> np.ndarray:
        xe = self.x[self.conn]
        jac = np.einsum("nai,aj->nij", xe, self.dn_c)
        return np.einsum("nij,njk->nik", jac, self.j0c_inv)

    def _wall_reaction(self) -> np.ndarray:
        """Total traction the wall applies through the clamped root samples.

        The residual internal force on a node is the negative of the elastic /
        tie force acting on it, so equilibrium at a clamped sample gives
        F_wall = f_int + f_tie - m g there.
        """
        fixed = self.fixed
        total = (self.f_int + self.f_tie)[fixed].sum(axis=0)
        total = total - (self.mass[fixed].sum()) * self.gravity
        return total

    def _frame(self, time: float) -> StateFrame:
        fixed_f = self.fixed.astype(float)
        ext = np.zeros((1, 2, 3), dtype=float)
        ext[0, 0] = self.total_mass * self.gravity
        ext[0, 1] = self._wall_reaction()
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "fixed": fixed_f,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "element_deformation_gradient": self._deformation_gradient(),
                "budget_external_force": ext,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_force": float(self.total_mass * float(np.linalg.norm(self.gravity))),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elems),
            "tie_constraints": int(self.tie_count),
            "untied_nodes": int(self.untied),
            "substep_dt": float(self.dt),
            "steps": int(self.steps),
            "last_newton_iterations": int(self.last_iterations),
            "total_mass": float(self.total_mass),
            "fixed_nodes": int(self.fixed.sum()),
            "shear_modulus": float(self.mu),
            "lambda": float(self.lam),
        }
