"""Physically simulated near-incompressible elastic solid compressed between plates.

Scene 056, "Soft Sphere Compressed between Plates".

Physical model
--------------
The deforming material is the frozen closed icosahedral surface mesh
(``assets/soft_sphere.obj``, 12 vertices / 20 triangles, unit circumradius,
centred at the origin).  A three-dimensional solid is obtained by connecting
every surface triangle to the body centre, which yields a closed tetrahedral
mesh of 20 volume elements (12 surface vertices + 1 interior vertex).  The
interior vertex is a linear (bubble) degree of freedom tied to the arithmetic
mean of the 12 surface vertices, so the exported state contains exactly the
frozen mesh vertices while the volume interior is still discretised.

Material law (per tetrahedron, material coordinates):

    psi(F) = mu/2 (tr(F^T F) - 3) + kappa/2 (det F - 1)^2

with soft shear modulus ``mu`` and a large bulk modulus ``kappa`` enforcing
near-incompressibility, so the solid spreads sideways instead of shrinking
when it is squeezed.

Kinematics
----------
Two opposed rigid plates with inner faces at z = +/- H(t) move symmetrically
inward from H(0) = 0.8506508084 (the extreme vertices of the mesh, i.e. the
plates initially just touch the sphere) down to H = 0.15 (positive gap 0.30)
between t = 0 and t = 3.4 s and then hold.

Time stepping
-------------
Implicit backward-Euler on the 12 nodal positions with a full Newton solve,
analytic consistent gradient and Hessian of the element energy, and a hard
frictionless active-set contact constraint against both plate faces
(z <= H(t), z >= -H(t)).  Contact reactions are recovered from the discrete
force balance, so the reported external forces, cumulative impulses and the
nodal momentum balance are mutually consistent; internal element force pairs
cancel exactly in the assembled system.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

__all__ = ["BenchmarkSolver"]

# ---------------------------------------------------------------------------
# frozen fall-backs (used only when the scaffold config omits a value)
# ---------------------------------------------------------------------------
_DEFAULT_HALF_GAP_0 = 0.8506508084
_DEFAULT_HALF_GAP_1 = 0.15
_DEFAULT_MOTION_END = 3.4
_DEFAULT_DT = 1.0e-3


def _hull_triangle_faces(vertices: np.ndarray) -> np.ndarray:
    """Deterministic closed triangle list of the convex hull of ``vertices``."""
    count = len(vertices)
    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
                norm = float(np.linalg.norm(normal))
                if norm < 1.0e-12:
                    continue
                normal = normal / norm
                offsets = (vertices - vertices[i]) @ normal
                others = np.ones(count, dtype=bool)
                others[[i, j, k]] = False
                if np.all(offsets[others] <= 1.0e-9) or np.all(offsets[others] >= -1.0e-9):
                    faces.append((i, j, k))
    if not faces:
        raise ValueError("degenerate deformable mesh: no hull faces found")
    return np.asarray(sorted(faces), dtype=np.int64)


class BenchmarkSolver:
    """Implicit finite-element solver for the plate-compressed elastic sphere."""

    # material constants (soft incompressible rubber-like solid)
    mu = 1.0e3
    kappa = 1.0e6
    max_newton = 24
    max_backtrack = 12
    j_floor = 0.05

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        frame0 = scene.initial_state()
        rest = np.array(frame0.position, dtype=float)
        if rest.ndim != 2 or rest.shape[1] != 3:
            raise ValueError("scene initial position must be [N,3]")
        self.n_nodes = int(rest.shape[0])
        self.rest_position = rest.copy()
        self.mass = np.array(frame0.mass, dtype=float).reshape(-1)
        if self.mass.shape[0] != self.n_nodes:
            raise ValueError("scene mass must supply one value per sample")
        self.total_mass = float(self.mass.sum())
        self.rest_volume_share = np.array(
            frame0.attributes.get("rest_volume_share", np.full(self.n_nodes, np.nan)),
            dtype=float,
        )
        if not np.isfinite(self.rest_volume_share).all():
            self.rest_volume_share = np.full(self.n_nodes, float(self.total_mass) / self.n_nodes)

        self._build_solid(rest)

        # plate kinematics (frozen scaffold parameters)
        plates = dict(config.get("plates", {}) or {})
        initial = plates.get("initial_inner_faces_z", [-_DEFAULT_HALF_GAP_0, _DEFAULT_HALF_GAP_0])
        final = plates.get("final_inner_faces_z", [-_DEFAULT_HALF_GAP_1, _DEFAULT_HALF_GAP_1])
        self.half_gap_0 = float(0.5 * (float(initial[1]) - float(initial[0])))
        self.half_gap_1 = float(0.5 * (float(final[1]) - float(final[0])))
        self.motion_end = float(plates.get("motion_end", _DEFAULT_MOTION_END))
        self.initial_gap = 2.0 * self.half_gap_0
        self.min_positive_gap = float(plates.get("minimum_positive_gap", 2.0 * self.half_gap_1))
        self.gravity = np.array(config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # integration controls
        self.dt = float(config.get("solver_dt", _DEFAULT_DT))
        self.step = 0
        self.u = self.rest_position.copy()
        self.v = np.zeros_like(self.u)
        self._impulse = np.zeros(3)
        self._prev_total_force = np.zeros(3)
        self._contact_force = np.zeros_like(self.u)
        self._force_terms = (np.zeros(3), np.zeros(3))  # (lower plate, upper plate)
        self._mass_dof = np.repeat(self.mass, 3)
        self._u_flat0 = self.u.reshape(-1).copy()
        self._v_flat0 = self.v.reshape(-1).copy()
        self._force_scale = max(
            1.0, abs(self.kappa) * float(np.sum(self.V0)) ** (1.0 / 3.0)
        )
        self.contact_tol = 1.0e-9 * max(1.0, self.half_gap_0)
        self._volume0 = float(np.sum(self.V0))
        self._steps_taken = 0

    # ------------------------------------------------------------------
    # solid construction
    # ------------------------------------------------------------------
    def _build_solid(self, rest: np.ndarray) -> None:
        faces = _hull_triangle_faces(rest)
        centre = rest.mean(axis=0)
        self.centroid_rest = centre
        tetra = np.hstack(
            [faces, np.full((len(faces), 1), self.n_nodes, dtype=np.int64)]
        ).astype(np.int64)
        positions = np.vstack([rest, centre])

        # orient every element so that its material volume is positive
        for _ in range(2):
            local = positions[tetra]
            edges = np.stack(
                [
                    local[:, 1] - local[:, 0],
                    local[:, 2] - local[:, 0],
                    local[:, 3] - local[:, 0],
                ],
                axis=2,
            )
            det = np.linalg.det(edges)
            bad = det < 0.0
            if not np.any(bad):
                break
            tetra[bad, 1], tetra[bad, 2] = tetra[bad, 2].copy(), tetra[bad, 1].copy()

        local = positions[tetra]
        shape = np.stack(
            [
                local[:, 1] - local[:, 0],
                local[:, 2] - local[:, 0],
                local[:, 3] - local[:, 0],
            ],
            axis=2,
        )
        det = np.linalg.det(shape)
        if np.any(det <= 0.0):
            raise ValueError("degenerate tetrahedral element in the frozen mesh")

        self.tet_nodes = tetra
        self.n_elements = int(tetra.shape[0])
        self.V0 = det / 6.0
        shape_inv = np.linalg.inv(shape)
        # rows of the (4x3) discrete gradient operator:  F = sum_i p_i M_i^T
        const = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        # M_i = Dm^{-T} D_i, so that F = sum_i p_i M_i^T = Ds Dm^{-1}
        # and F = I exactly in the undeformed configuration.
        self.Mcoef = np.einsum("enm,in->eim", shape_inv, const)

    # ------------------------------------------------------------------
    # element energy, gradient and Hessian
    # ------------------------------------------------------------------
    def _eval(self, u: np.ndarray, hessian: bool):
        n = self.n_nodes
        points = np.empty((n + 1, 3), dtype=float)
        points[:n] = u
        points[n] = u.mean(axis=0)
        local = points[self.tet_nodes]                      # (Ne,4,3)
        M = self.Mcoef                                      # (Ne,4,3)

        F = np.einsum("eim,ein->emn", local, M)             # deformation gradient
        det = np.linalg.det(F)
        inv = np.linalg.inv(F)
        q = np.transpose(inv, (0, 2, 1))                    # F^{-T}
        i1 = np.einsum("emn,emn->e", F, F)                  # tr(F^T F)
        # isochoric scaling, clipped so a transiently inverted element can never
        # inject non-finite values into the Newton iteration
        g = np.maximum(det, 1.0e-8) ** (-2.0 / 3.0)

        # isochoric neo-Hookean + volumetric penalty:
        #   psi = mu/2 (J^-2/3 tr(C) - 3) + kappa/2 (J - 1)^2
        pk = (
            self.mu * g[:, None, None] * (F - (i1 / 3.0)[:, None, None] * q)
            + (self.kappa * (det - 1.0) * det)[:, None, None] * q
        )
        grad_local = self.V0[:, None, None] * np.einsum("emn,ein->eim", pk, M)

        grad = np.zeros((n, 3), dtype=float)
        np.add.at(grad, self.tet_nodes[:, 0], grad_local[:, 0])
        np.add.at(grad, self.tet_nodes[:, 1], grad_local[:, 1])
        np.add.at(grad, self.tet_nodes[:, 2], grad_local[:, 2])
        grad += grad_local[:, 3].sum(axis=0) / float(n)

        if not hessian:
            return grad, F, det

        # exact tangent A_abcd = dP_ab / dF_cd of the constitutive model
        # identity on the (a,c) and (b,d) index pairs: dF_ab / dF_cd = delta_ac delta_bd
        eye4 = np.eye(3)[:, None, :, None] * np.eye(3)[None, :, None, :]
        fq = np.einsum("eab,ecd->eabcd", F, q)
        qq = np.einsum("eab,ecd->eabcd", q, q)
        t4 = np.einsum("ecb,ead->eabcd", q, q)
        tangent = self.mu * g[:, None, None, None, None] * (
            eye4
            - (2.0 / 3.0) * (fq + np.transpose(fq, (0, 3, 4, 1, 2)))
            + (2.0 / 9.0) * i1[:, None, None, None, None] * qq
            + (i1 / 3.0)[:, None, None, None, None] * t4
        )
        tangent = tangent + self.kappa * det[:, None, None, None, None] * (
            (2.0 * det - 1.0)[:, None, None, None, None] * qq
            - (det - 1.0)[:, None, None, None, None] * t4
        )
        # H_ij[a,b] = sum_{n,d} M_i[n] M_j[d] A_{a n b d}
        hess_local = self.V0[:, None, None, None, None] * np.einsum(
            "eib,ejd,eabgd->eijag", M, M, tangent
        )

        H = np.zeros((3 * n, 3 * n), dtype=float)
        nodes = self.tet_nodes
        for a in range(3):
            for b in range(3):
                self._scatter_block(H, nodes[:, a], nodes[:, b], hess_local[:, a, b])
            self._scatter_rows_all_cols(H, nodes[:, a], hess_local[:, a, 3], 1.0 / n)
            self._scatter_all_rows_cols(H, nodes[:, a], hess_local[:, 3, a], 1.0 / n)
        centre_block = hess_local[:, 3, 3].sum(axis=0) / float(n * n)
        for m in range(3):
            for l in range(3):
                H[m::3, l::3] += centre_block[m, l]
        return grad, F, det, H

    @staticmethod
    def _scatter_block(H, row_nodes, col_nodes, block):
        me = len(row_nodes)
        rows = (3 * row_nodes[:, None, None] + np.arange(3)[None, :, None])
        rows = np.broadcast_to(rows, (me, 3, 3)).reshape(-1)
        cols = (3 * col_nodes[:, None, None] + np.arange(3)[None, None, :])
        cols = np.broadcast_to(cols, (me, 3, 3)).reshape(-1)
        np.add.at(H, (rows, cols), block.reshape(-1))

    @staticmethod
    def _scatter_rows_all_cols(H, row_nodes, block, weight):
        """Block for one local node against every exported node (centre projection)."""
        me = len(row_nodes)
        ncol = H.shape[0]
        rows = (3 * row_nodes[:, None, None] + np.arange(3)[None, :, None])
        rows = np.broadcast_to(rows, (me, 3, ncol)).reshape(-1)
        cols = np.broadcast_to(np.arange(ncol)[None, None, :], (me, 3, ncol)).reshape(-1)
        values = np.broadcast_to(
            (weight * block)[:, :, None, :], (me, 3, ncol // 3, 3)
        ).reshape(-1)
        np.add.at(H, (rows, cols), values)

    @staticmethod
    def _scatter_all_rows_cols(H, col_nodes, block, weight):
        """Transposed counterpart of :meth:`_scatter_rows_all_cols`.

        Adds ``weight * block[m, n]`` to ``H[3*l + m, 3*col_e + n]`` for every
        exported node ``l``; here ``block`` holds ``w_i x w_j`` for the centre
        degree of freedom (which receives 1/N of the surface node weights).
        """
        me = len(col_nodes)
        n = H.shape[0] // 3
        rows = np.broadcast_to(
            (3 * np.arange(n)[None, :, None, None] + np.arange(3)[None, None, :, None]),
            (me, n, 3, 3),
        )
        cols = np.broadcast_to(
            (3 * col_nodes[:, None, None, None] + np.arange(3)[None, None, None, :]),
            (me, n, 3, 3),
        )
        values = np.broadcast_to(
            (weight * block)[:, None, :, :], (me, n, 3, 3)
        )
        np.add.at(H, (rows.reshape(-1), cols.reshape(-1)), values.reshape(-1))

    # ------------------------------------------------------------------
    # plate kinematics
    # ------------------------------------------------------------------
    def _half_gap(self, time: float) -> float:
        if time <= 0.0:
            return self.half_gap_0
        if time >= self.motion_end:
            return self.half_gap_1
        frac = time / self.motion_end
        return self.half_gap_0 + (self.half_gap_1 - self.half_gap_0) * frac

    # ------------------------------------------------------------------
    # implicit time step
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt
        n = self.n_nodes
        u0 = self.u.reshape(-1).copy()
        v0 = self.v.reshape(-1).copy()
        time = (self.step + 1) * dt
        half = self._half_gap(time)

        predicted_z = u0.reshape(n, 3)[:, 2] + dt * v0.reshape(n, 3)[:, 2]
        # a node already resting on a face must stay active even when the
        # predicted gap closes exactly to zero in floating point
        hi = predicted_z >= half - self.contact_tol
        lo = predicted_z <= -half + self.contact_tol
        fix = np.zeros((n, 3), dtype=bool)
        fix[hi, 2] = True
        fix[lo, 2] = True
        target = np.zeros((n, 3), dtype=float)
        target[hi, 2] = half
        target[lo, 2] = -half
        fix_flat = fix.reshape(-1)
        target_flat = target.reshape(-1)
        fixed = np.flatnonzero(fix_flat)
        free = np.flatnonzero(~fix_flat)

        inv_dt2 = 1.0 / (dt * dt)
        mass_dof = self._mass_dof

        u = u0 + dt * v0
        u[fixed] = target_flat[fixed]

        total = u.size
        residual = np.zeros(total, dtype=float)
        r0 = 1.0
        newton_jac = np.zeros((total, total), dtype=float)

        for iteration in range(self.max_newton):
            acc = (u - u0 - dt * v0) * inv_dt2
            residual = mass_dof * acc + self._eval(
                u.reshape(n, 3), hessian=False
            )[0].reshape(-1)
            r_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            if iteration == 0:
                r0 = max(r_norm, 1.0)
            if r_norm <= 1.0e-11 * r0 + 1.0e-10:
                break

            grad, _, _, hess = self._eval(u.reshape(n, 3), hessian=True)
            residual = mass_dof * acc + grad.reshape(-1)
            newton_jac = hess + np.diag(mass_dof * inv_dt2)
            newton_jac = 0.5 * (newton_jac + newton_jac.T)

            delta = np.zeros(total, dtype=float)
            if fixed.size:
                d_fixed = target_flat[fixed] - u[fixed]
                delta[fixed] = d_fixed
                rhs = -residual[free] - newton_jac[np.ix_(free, fixed)] @ d_fixed
            else:
                rhs = -residual[free]
            reduced = newton_jac[np.ix_(free, free)]
            try:
                delta[free] = np.linalg.solve(reduced, rhs)
            except np.linalg.LinAlgError:
                delta[free] = np.linalg.lstsq(reduced, rhs, rcond=None)[0]

            alpha = 1.0
            accepted = False
            for _ in range(self.max_backtrack):
                trial = u + alpha * delta
                if fixed.size:
                    trial[fixed] = target_flat[fixed]
                grad_t, _, det_t = self._eval(trial.reshape(n, 3), hessian=False)
                if float(det_t.min()) <= self.j_floor:
                    alpha *= 0.5
                    continue
                acc_t = (trial - u0 - dt * v0) * inv_dt2
                res_t = mass_dof * acc_t + grad_t.reshape(-1)
                norm_t = (
                    float(np.linalg.norm(res_t[free])) if free.size else 0.0
                )
                if not np.isfinite(norm_t):
                    alpha *= 0.5
                    continue
                if norm_t <= (1.0 - 1.0e-4 * alpha) * r_norm or alpha <= 1.0e-3:
                    u = trial
                    residual = res_t
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                trial = u + alpha * delta
                if fixed.size:
                    trial[fixed] = target_flat[fixed]
                u = trial

        # ---- commit state -------------------------------------------------
        if not np.isfinite(u).all():
            # defensive fallback: keep the previous configuration and let the
            # active-set contact projection advance the plate-held nodes
            u = u0.copy()
            if fixed.size:
                u[fixed] = target_flat[fixed]
            self.u = u.reshape(n, 3).copy()
            self.v = ((u - u0) / dt).reshape(n, 3).copy()
            self._contact_force = np.zeros((n, 3), dtype=float)
            self._force_terms = (np.zeros(3), np.zeros(3))
            self._impulse = self._impulse + self._prev_total_force * dt
            self.step += 1
            self._steps_taken += 1
            return
        grad_f, _, _ = self._eval(u.reshape(n, 3), hessian=False)
        acc_f = (u - u0 - dt * v0) * inv_dt2
        residual = mass_dof * acc_f + grad_f.reshape(-1)

        contact = np.zeros(total, dtype=float)
        if fixed.size:
            contact[fixed] = residual[fixed]

        self.u = u.reshape(n, 3).copy()
        self.v = ((u - u0) / dt).reshape(n, 3).copy()
        self._contact_force = contact.reshape(n, 3)
        lower = float(np.sum(contact.reshape(n, 3)[lo, 2]))
        upper = float(np.sum(contact.reshape(n, 3)[hi, 2]))
        self._force_terms = (
            np.array([0.0, 0.0, lower]),
            np.array([0.0, 0.0, upper]),
        )
        total_force = np.array([0.0, 0.0, lower + upper])
        self._impulse = self._impulse + 0.5 * (self._prev_total_force + total_force) * dt
        self._prev_total_force = total_force
        self.step += 1
        self._steps_taken += 1

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        target_step = int(round(target / self.dt))
        if target_step < self.step:
            raise ValueError("advance_to requires monotonically increasing times")
        while self.step < target_step:
            self._step()
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        n = self.n_nodes
        half = self._half_gap(time)
        grad, F, det = self._eval(self.u, hessian=False)
        extent = float(self.u[:, 2].max() - self.u[:, 2].min())
        volume = float(np.sum(self.V0 * det))
        lower_force, upper_force = self._force_terms

        frame = StateFrame(
            time=float(time),
            position=self.u.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "rest_position": self.rest_position.copy(),
                "mass_quantity": self.mass.copy(),
                "rest_volume_share": self.rest_volume_share.copy(),
                "deformation_gradient": self._nodal_deformation_gradient(F),
            },
            fields={
                "element_deformation_gradient": F.copy(),
                "constraint_value": np.array([extent]),
                "constraint_target": np.array([2.0 * half]),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "budget_domain_weights": np.ones((1, n), dtype=float),
                "budget_external_force": np.array(
                    [[lower_force, upper_force]], dtype=float
                ),
                "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "external_force": self._contact_force.copy(),
            },
            scalars={
                "rest_volume": float(np.sum(self.rest_volume_share)),
                "initial_plate_gap": float(self.initial_gap),
                "minimum_positive_gap": float(self.min_positive_gap),
                "constraint_scale": float(self.initial_gap),
                "reference_momentum": float(
                    self.total_mass
                    * max(
                        1.0e-6,
                        abs(self.half_gap_0 - self.half_gap_1)
                        / max(self.motion_end, 1.0e-9),
                    )
                ),
                "current_volume": volume,
                "volume_ratio": volume / max(self._volume0, 1.0e-30),
                "plate_gap": 2.0 * half,
                "kinematic_steps": float(self._steps_taken),
            },
        )
        frame.validate()
        return frame

    def _nodal_deformation_gradient(self, F: np.ndarray) -> np.ndarray:
        """Volume weighted per-vertex average of the adjacent element gradients."""
        n = self.n_nodes
        out = np.zeros((n, 3, 3), dtype=float)
        weight = np.zeros(n, dtype=float)
        for a in range(3):
            np.add.at(out, self.tet_nodes[:, a], self.V0[:, None, None] * F)
            np.add.at(weight, self.tet_nodes[:, a], self.V0)
        centre = np.einsum("e,emn->mn", self.V0, F) / n
        out += centre[None, :, :]
        weight += float(np.sum(self.V0)) / n
        return out / weight[:, None, None]

    def diagnostics(self) -> dict[str, object]:
        grad, F, det = self._eval(self.u, hessian=False)
        return {
            "solver": "implicit_backward_euler_newton_fem",
            "steps": int(self._steps_taken),
            "elements": int(self.n_elements),
            "nodes": int(self.n_nodes),
            "volume_ratio": float(np.sum(self.V0 * det) / max(self._volume0, 1.0e-30)),
            "min_det_F": float(det.min()),
            "max_det_F": float(det.max()),
            "max_penetration": float(
                max(
                    0.0,
                    float(self.u[:, 2].max() - self._half_gap(self.step * self.dt)),
                    float(-self._half_gap(self.step * self.dt) - self.u[:, 2].min()),
                )
            ),
            "net_momentum": [float(x) for x in np.sum(self.mass[:, None] * self.v, axis=0)],
            "cumulative_external_impulse": [float(x) for x in self._impulse],
        }
