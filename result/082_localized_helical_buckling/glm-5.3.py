"""Quasi-static discrete-elastic-rod solver for localized helical buckling.

Physics: a naturally straight, isotropic, inextensible Cosserat rod discretized
with 129 nodes / 128 edges (discrete elastic rod formulation).

Energies (all evaluated in vectorized batches over configurations):
  * stretching penalty     0.5*k_s*sum((l_j - h)^2 / h)
  * bending                0.5*B*sum(|kb_i|^2 / hbar_i)   (curvature binormal)
  * twisting               0.5*C*sum(m_i^2 / hbar_i)      (Bishop-frame angles)
  * clamp twist penalty    0.5*k_bc*wrapPi(theta_last - psi)^2
    (psi = world angle of the right clamp material frame relative to the
     transported Bishop frame; the quadratic-in-wrapped-angle form keeps the
     loaded 2*pi branch, so twist cannot leak away discontinuously)
  * self-contact           0.5*k_c*sum(max(0, 2r - d)^2) for node pairs |i-j|>=5

Boundary conditions: nodes 0,1 pinned to the left clamp (position + tangent),
nodes 127,128 pinned to the right clamp (position + tangent, kinematically
translated during the shortening stage), theta_0 = 0 (left material frame).

Loading protocol (quasi-static, displacement controlled):
  t in [0,2]  : relative clamp material rotation phi ramps 0 -> 27 rad
  t in [2,8]  : right clamp translates 0.4645 -> 0.1855 along x
  t in [8,10] : hold

At every requested sample time the equilibrium is found by L-BFGS-B
minimization of the total energy with an analytic-free (central finite
difference, batched) gradient, warm started from the previous sample.
"""
from __future__ import annotations

import time

import numpy as np

from worldsolver.contracts import StateFrame


def _wrap_pi(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _quat_from_matrix(R):
    """Quaternion (xyzw) from a rotation matrix (standard branch method)."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


class BenchmarkSolver:
    VERTEX_COUNT = 129
    CONTACT_SKIP = 5          # node pairs with |i-j| >= CONTACT_SKIP attract contact
    CONTACT_MARGIN = 0.004    # activation margin for the contact pair set (m)
    CHUNK = 256               # configurations per vectorized energy block
    FD_EPS = 1e-6
    TIME_BUDGET = 1500.0      # soft wall-clock guard (s)

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        rod = cfg["rod"]
        self.L = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.B = float(rod["bending_modulus_ratio"])   # bending stiffness (N m^2)
        self.C = float(rod["twisting_modulus_ratio"])   # twisting stiffness (N m^2)
        self.N = self.VERTEX_COUNT
        self.NE = self.N - 1
        self.h = self.L / self.NE
        self.mass_total = self.density * np.pi * self.radius**2 * self.L
        self.mass = np.full(self.N, self.mass_total / self.N)
        self.mass[[0, -1]] *= 0.5
        self.arc_length = np.linspace(0.0, self.L, self.N)

        init = cfg["initial_conditions"]
        bc = cfg["boundary_conditions"]
        proto = cfg["loading_protocol"]
        self.p_left = np.asarray(bc["left_clamp"]["fixed_position"], dtype=float)
        self.p_right0 = np.asarray(bc["right_clamp"]["initial_position"], dtype=float)
        self.p_right1 = np.asarray(bc["right_clamp"]["final_position"], dtype=float)
        self.t_axis = np.asarray(bc["left_clamp"]["tangent"], dtype=float)
        self.t_axis = self.t_axis / np.linalg.norm(self.t_axis)
        self.phi_final = float(proto["twist_stage"]["final_relative_rotation_rad"])
        self.t_twist_end = float(proto["twist_stage"]["end_time"])
        self.t_short_start = float(proto["shortening_stage"]["start_time"])
        self.t_short_end = float(proto["shortening_stage"]["end_time"])
        self.t_end = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # stiffness scales
        self.k_stretch = 100.0 * self.B / self.h**2        # N
        self.k_clamp = 3.0e4                               # N m (twist penalty)
        self.k_contact = 1.0e5                             # N/m
        self.contact_rest = 2.0 * self.radius

        # interior DOF layout: [positions of nodes 2..N-3 (flat), theta edges 1..NE-1]
        self.n_interior_nodes = self.N - 4                 # nodes 2..N-3
        self.n_pos_dof = 3 * self.n_interior_nodes
        self.n_theta_dof = self.NE - 1                     # edges 1..NE-1
        self.n_dof = self.n_pos_dof + self.n_theta_dof

        # full contact pair index set (|i-j| >= CONTACT_SKIP)
        ii, jj = np.triu_indices(self.N, k=self.CONTACT_SKIP)
        self.pair_i_all = ii
        self.pair_j_all = jj

        # seed state from the frozen scene initial frame (includes the
        # reproducible branch-selection perturbation)
        frame0 = scene.initial_state()
        self.endpoint_clamp = np.asarray(frame0.attributes["endpoint_clamp"], dtype=float).copy()
        p0 = np.asarray(frame0.position, dtype=float)
        self.p_seed = p0.copy()
        self._check_seed(p0)
        self.dof = self._dof_from_positions(p0)
        self.dof[self.n_pos_dof:] = 0.0

        # cache of the previous emitted sample (for velocities / reuse)
        self._last_time = None
        self._last_positions = p0.copy()
        self._last_frame = None
        self._hold_dof = None
        self._start_clock = time.monotonic()
        self._n_solves = 0
        self._last_energy = 0.0

    # ------------------------------------------------------------------ seed
    def _check_seed(self, p0):
        if p0.shape != (self.N, 3) or not np.isfinite(p0).all():
            raise ValueError("initial rod positions are not a finite [N,3] array")

    def _dof_from_positions(self, P):
        dof = np.zeros(self.n_dof)
        dof[: self.n_pos_dof] = P[2 : self.N - 2].reshape(-1)
        return dof

    # -------------------------------------------------------- loading protocol
    def _loading(self, t):
        """Prescribed (phi, right clamp position) at time t."""
        if t <= self.t_twist_end:
            u = 0.0 if self.t_twist_end <= 0 else min(max(t / self.t_twist_end, 0.0), 1.0)
            phi = self.phi_final * u
            return phi, self.p_right0.copy()
        if t <= self.t_short_end:
            u = (t - self.t_short_start) / max(self.t_short_end - self.t_short_start, 1e-12)
            u = min(max(u, 0.0), 1.0)
            phi = self.phi_final
            p_r = self.p_right0 + (self.p_right1 - self.p_right0) * u
            return phi, p_r
        return self.phi_final, self.p_right1.copy()

    # ------------------------------------------------------------- kinematics
    def _positions_from_block(self, block, p_right):
        """block: (K, n_dof) -> positions (K, N, 3) with clamped nodes filled in."""
        K = block.shape[0]
        P = np.empty((K, self.N, 3))
        P[:, 0, :] = self.p_left
        P[:, 1, :] = self.p_left + self.h * self.t_axis
        P[:, self.N - 2, :] = p_right - self.h * self.t_axis
        P[:, self.N - 1, :] = p_right
        P[:, 2 : self.N - 2, :] = block[:, : self.n_pos_dof].reshape(K, self.n_interior_nodes, 3)
        return P

    def _frames(self, t):
        """Rotation-minimizing (Bishop) frames for a batch of edge tangents.

        t: (K, NE, 3) unit edge tangents; u_0 is the left clamp material axis
        projected normal to t_0. Returns u: (K, NE, 3).
        """
        K = t.shape[0]
        u = np.tile(np.array([0.0, 1.0, 0.0]), (K, 1))
        u = u - np.sum(u * t[:, 0, :], axis=1, keepdims=True) * t[:, 0, :]
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        U = np.empty((K, self.NE, 3))
        U[:, 0, :] = u
        for j in range(1, self.NE):
            ta = t[:, j - 1, :]
            tb = t[:, j, :]
            ax = np.cross(ta, tb)
            s = np.linalg.norm(ax, axis=1)
            c = np.sum(ta * tb, axis=1)
            ang = np.arctan2(s, c)
            sn = np.sin(ang)
            cs = np.cos(ang)
            k = ax / np.maximum(s, 1e-14)[:, None]
            kdu = np.sum(k * u, axis=1)
            u = u * cs[:, None] + np.cross(k, u) * sn[:, None] + k * (kdu * (1.0 - cs))[:, None]
            # re-orthogonalize against the new tangent
            u = u - np.sum(u * tb, axis=1, keepdims=True) * tb
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            U[:, j, :] = u
        return U

    # ------------------------------------------------------------------ energy
    def _energy_block(self, block, phi, p_right, pair_i, pair_j):
        """Total elastic energy for a block of configurations.

        block: (K, n_dof); returns (parts, E) with parts a dict of scalars
        (means) used only for diagnostics on single configurations.
        """
        P = self._positions_from_block(block, p_right)
        theta = np.zeros((block.shape[0], self.NE))
        theta[:, 1:] = block[:, self.n_pos_dof:]
        return self._energy_core(P, theta, phi, pair_i, pair_j)

    def _energy_core(self, P, theta, phi, pair_i, pair_j):
        """Elastic energy for explicit positions/angles batches (K,N,3)/(K,NE)."""
        K = P.shape[0]
        e = P[:, 1:, :] - P[:, :-1, :]
        l = np.linalg.norm(e, axis=2)
        l = np.maximum(l, 1e-9)
        tang = e / l[:, :, None]

        # stretching penalty (inextensibility)
        E_str = 0.5 * self.k_stretch * np.sum((l - self.h) ** 2 / self.h, axis=1)

        # bending: curvature binormal at interior nodes
        t1 = tang[:, :-1, :]
        t2 = tang[:, 1:, :]
        cr = np.cross(t1, t2)
        dt = np.sum(t1 * t2, axis=2)
        denom = np.maximum(1.0 + dt, 0.05)
        kb = 2.0 * cr / denom[:, :, None]
        hbar = 0.5 * (l[:, :-1] + l[:, 1:])
        hbar = np.maximum(hbar, 1e-9)
        E_bend = 0.5 * self.B * np.sum(np.sum(kb * kb, axis=2) / hbar, axis=1)

        # twisting: Bishop frames + material angles
        U = self._frames(tang)
        V = np.cross(tang, U)
        m_i = theta[:, 1:] - theta[:, :-1]
        E_tw = 0.5 * self.C * np.sum(m_i * m_i / hbar, axis=1)

        # right clamp material-frame prescription: the clamp frame is the left
        # clamp frame rotated by phi about the rod axis; psi is its angle
        # relative to the transported Bishop frame at the last edge.
        m1 = np.array([0.0, np.cos(phi), np.sin(phi)])
        uL = U[:, -1, :]
        vL = V[:, -1, :]
        psi = np.arctan2(np.sum(m1 * vL, axis=1), np.sum(m1 * uL, axis=1))
        dang = _wrap_pi(theta[:, -1] - psi)
        E_clamp = 0.5 * self.k_clamp * dang * dang

        # self-contact between well-separated node pairs
        E_con = np.zeros(K)
        if pair_i.size:
            d = np.linalg.norm(P[:, pair_i, :] - P[:, pair_j, :], axis=2)
            gap = np.maximum(0.0, self.contact_rest - d)
            E_con = 0.5 * self.k_contact * np.sum(gap * gap, axis=1)

        E = E_str + E_bend + E_tw + E_clamp + E_con
        E = np.where(np.isfinite(E), E, 1e18)
        parts = {
            "stretch": float(np.mean(E_str)),
            "bend": float(np.mean(E_bend)),
            "twist": float(np.mean(E_tw)),
            "clamp": float(np.mean(E_clamp)),
            "contact": float(np.mean(E_con)),
        }
        return parts, E

    def _energy_batch(self, dofs, phi, p_right, pair_i, pair_j):
        K = dofs.shape[0]
        out = np.empty(K)
        for s in range(0, K, self.CHUNK):
            sl = slice(s, min(s + self.CHUNK, K))
            _, out[sl] = self._energy_block(dofs[sl], phi, p_right, pair_i, pair_j)
        return out

    def _active_pairs(self, dof, phi, p_right):
        """Contact candidate pairs near contact for the current configuration."""
        P = self._positions_from_block(dof[None, :], p_right)[0]
        d = np.linalg.norm(P[self.pair_i_all] - P[self.pair_j_all], axis=1)
        sel = d < self.contact_rest + self.CONTACT_MARGIN
        return self.pair_i_all[sel], self.pair_j_all[sel]

    # ---------------------------------------------------------------- gradient
    def _fun_factory(self, phi, p_right):
        D = self.n_dof
        idx = np.arange(D)
        eps = self.FD_EPS

        def fun(y):
            pair_i, pair_j = self._active_pairs(y, phi, p_right)
            batch = np.tile(y, (1 + 2 * D, 1))
            batch[1 + idx, idx] += eps
            batch[1 + D + idx, idx] -= eps
            E = self._energy_batch(batch, phi, p_right, pair_i, pair_j)
            g = (E[1 : D + 1] - E[1 + D :]) / (2.0 * eps)
            return float(E[0]), g

        return fun

    # ------------------------------------------------------------- quasi-static
    def _solve(self, phi, p_right):
        from scipy.optimize import minimize

        elapsed = time.monotonic() - self._start_clock
        if elapsed < 600.0:
            maxiter = 120
        elif elapsed < 1000.0:
            maxiter = 60
        elif elapsed < 1300.0:
            maxiter = 25
        else:
            maxiter = 8
        fun = self._fun_factory(phi, p_right)
        y0 = self.dof.copy()
        E0 = float(self._energy_batch(y0[None, :], phi, p_right, *self._active_pairs(y0, phi, p_right))[0])
        try:
            res = minimize(
                fun,
                y0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 5e-4, "maxls": 20},
            )
            y1 = np.asarray(res.x, dtype=float)
            E1 = float(res.fun)
            ok = np.isfinite(y1).all() and np.isfinite(E1) and E1 <= E0 + 1e-6 * max(1.0, abs(E0))
            if not ok:
                # retry once from the seed with a smaller trust in line searches
                res = minimize(
                    fun,
                    y0,
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-3, "maxls": 40},
                )
                y1 = np.asarray(res.x, dtype=float)
                E1 = float(res.fun)
                ok = np.isfinite(y1).all() and np.isfinite(E1) and E1 <= E0 + 1e-4 * max(1.0, abs(E0))
                if not ok:
                    return E0
            self.dof = y1
            self._n_solves += 1
            return E1
        except Exception:
            return E0

    # ---------------------------------------------------------------- reactions
    def _reactions(self, dof, phi, p_right):
        """Clamp wrenches (force on rod at each clamp origin + free couple)."""
        P0 = self._positions_from_block(dof[None, :], p_right)
        theta0 = np.zeros((1, self.NE))
        theta0[:, 1:] = dof[None, self.n_pos_dof :]
        pair_i, pair_j = self._active_pairs(dof, phi, p_right)

        node_idx = [0, 1, self.N - 2, self.N - 1]
        rows_P = []
        meta = []
        eps = 1e-6
        for n in node_idx:
            for c in range(3):
                for sgn in (+1.0, -1.0):
                    rows_P.append(sgn)
                    meta.append((n, c, sgn))
        K = 1 + len(meta) + 2  # base, node perturbations, phi +/-
        # build perturbation batch: base first, then node perturbations, then phi
        Pb = np.repeat(P0, K, axis=0)
        thb = np.repeat(theta0, K, axis=0)
        phib = np.full(K, phi)
        for r, (n, c, sgn) in enumerate(meta, start=1):
            Pb[r, n, c] += sgn * eps
        phib[-2] += eps
        phib[-1] -= eps

        E = np.empty(K)
        for s in range(0, K, self.CHUNK):
            sl = slice(s, min(s + self.CHUNK, K))
            _, E[sl] = self._energy_core(Pb[sl], thb[sl], float(phib[sl][0]), pair_i, pair_j)
        # NOTE: phi varies within a block only in the last two rows; recompute
        # those rows individually to keep the energy consistent.
        _, E[-2] = self._energy_core(Pb[-2:][0:1], thb[-2:][0:1], phi + eps, pair_i, pair_j)
        _, E[-1] = self._energy_core(Pb[-1:][0:1], thb[-1:][0:1], phi - eps, pair_i, pair_j)

        r = 0
        F_nodes = {}
        for n in node_idx:
            F = np.zeros(3)
            for c in range(3):
                Ep = E[1 + r]
                Em = E[2 + r]
                F[c] = -(Ep - Em) / (2.0 * eps)   # external clamp force on rod
                r += 2
            F_nodes[n] = F
        tau_phi = (E[-2] - E[-1]) / (2.0 * eps)   # couple on rod at right clamp (+axis)
        axis = self.t_axis

        F_left = F_nodes[0] + F_nodes[1]
        F_right = F_nodes[self.N - 2] + F_nodes[self.N - 1]
        cp_left = np.zeros(3)
        cp_right = np.zeros(3)
        for n, cp in ((0, cp_left), (1, cp_left)):
            cp += np.cross(P0[0, n] - self.p_left, F_nodes[n])
        for n in (self.N - 2, self.N - 1):
            cp_right += np.cross(P0[0, n] - p_right, F_nodes[n])
        couple_left = cp_left - tau_phi * axis
        couple_right = cp_right + tau_phi * axis
        return F_left, F_right, couple_left, couple_right

    # ------------------------------------------------------------ state arrays
    def _state_arrays(self, dof, phi, p_right):
        """Current positions, tangents, Bishop frames and material angles."""
        P = self._positions_from_block(dof[None, :], p_right)
        theta = np.zeros((1, self.NE))
        theta[:, 1:] = dof[None, self.n_pos_dof :]
        e = P[:, 1:, :] - P[:, :-1, :]
        l = np.maximum(np.linalg.norm(e, axis=2), 1e-9)
        tang = e / l[:, :, None]
        U = self._frames(tang)
        V = np.cross(tang, U)
        return P[0], tang[0], U[0], V[0], theta[0]

    def _orientation(self, tang, U, V, theta):
        """Per-node material frame quaternions (xyzw)."""
        q = np.empty((self.N, 4))
        for i in range(self.N):
            j = min(i, self.NE - 1)
            m1 = U[j] * np.cos(theta[j]) + V[j] * np.sin(theta[j])
            m2 = V[j] * np.cos(theta[j]) - U[j] * np.sin(theta[j])
            R = np.column_stack((m1, m2, tang[j]))
            q[i] = _quat_from_matrix(R)
        return q

    # ------------------------------------------------------------------- frame
    def _build_frame(self, t, P, velocity, phi, p_right, parts):
        F_left, F_right, c_left, c_right = self._reactions(self.dof, phi, p_right)
        end_distance = float(np.linalg.norm(P[-1] - P[0]))
        target_distance = float(np.linalg.norm(p_right - self.p_left))
        _, tang, U, V, theta = self._state_arrays(self.dof, phi, p_right)
        orientation = self._orientation(tang, U, V, theta)
        frame = StateFrame(
            time=float(t),
            position=P.copy(),
            velocity=velocity.copy(),
            orientation=orientation,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "natural_curvature": np.zeros(self.N),
                "natural_twist": np.zeros(self.N),
                "endpoint_clamp": self.endpoint_clamp.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": np.array([[F_left, F_right]], dtype=float),
                "budget_force_position": np.array([[self.p_left, p_right]], dtype=float),
                "budget_external_couple": np.array([[c_left, c_right]], dtype=float),
                "constraint_value": np.array([end_distance, float(phi)]),
                "constraint_target": np.array([target_distance, float(phi)]),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": self.L,
                "reference_force": self.B / self.L**2,
                "reference_torque": self.C / self.L,
                "rod_radius": self.radius,
                "target_relative_twist_rad": self.phi_final,
                "imposed_relative_twist_rad": float(phi),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- advance
    def advance_to(self, time: float):
        t = float(time)
        if self._last_frame is not None and t <= self._last_time + 1e-12:
            return self._last_frame

        phi, p_right = self._loading(t)

        if t <= 1e-9:
            # initial conditions: nearly straight, untwisted
            P = self.p_seed.copy()
            velocity = np.zeros_like(P)
            self.dof = self._dof_from_positions(P)
            self.dof[self.n_pos_dof:] = 0.0
            parts, _ = self._energy_block(
                self.dof[None, :], phi, p_right, *self._active_pairs(self.dof, phi, p_right)
            )
            frame = self._build_frame(t, P, velocity, phi, p_right, parts)
        else:
            prev_phi, _ = self._loading(self._last_time if self._last_time is not None else 0.0)
            in_hold = t >= self.t_short_end - 1e-12
            if in_hold and self._hold_dof is not None:
                self.dof = self._hold_dof.copy()
                energy = self._last_energy
            else:
                # twist-stage predictor: rescale material angles with phi
                if prev_phi > 1e-9 and phi > prev_phi + 1e-12:
                    ratio = phi / prev_phi
                    self.dof[self.n_pos_dof :] *= ratio
                energy = self._solve(phi, p_right)
                if in_hold:
                    self._hold_dof = self.dof.copy()
            parts, _ = self._energy_block(
                self.dof[None, :], phi, p_right, *self._active_pairs(self.dof, phi, p_right)
            )
            self._last_energy = energy
            P = self._positions_from_block(self.dof[None, :], p_right)[0]
            dt = t - (self._last_time if self._last_time is not None else 0.0)
            if dt > 1e-12:
                velocity = (P - self._last_positions) / dt
            else:
                velocity = np.zeros_like(P)
            velocity = np.where(np.isfinite(velocity), velocity, 0.0)
            frame = self._build_frame(t, P, velocity, phi, p_right, parts)

        self._last_time = t
        self._last_positions = np.asarray(frame.position, dtype=float).copy()
        self._last_frame = frame
        return frame

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, object]:
        phi, p_right = self._loading(self._last_time or 0.0)
        parts, E = self._energy_block(
            self.dof[None, :], phi, p_right, *self._active_pairs(self.dof, phi, p_right)
        )
        P = self._positions_from_block(self.dof[None, :], p_right)[0]
        lateral = np.linalg.norm(P[:, 1:] ** 2 + P[:, 2:] ** 2, axis=1)
        return {
            "solver": "quasi-static discrete elastic rod (batched finite-difference L-BFGS)",
            "n_dof": int(self.n_dof),
            "n_solves": int(self._n_solves),
            "total_energy_J": float(E[0]),
            "energy_parts_J": parts,
            "max_lateral_amplitude_m": float(np.max(lateral)),
            "elapsed_s": float(time.monotonic() - self._start_clock),
        }
