"""Wilberforce pendulum solver: discrete helical elastic rod (axial-torsional
coupled) carrying a rigid load, integrated with velocity Verlet.

Physical model
--------------
The helical spring is discretised into ``vertex_count`` material cross sections.
Each cross section ``i`` carries two generalised coordinates measured from the
gravity equilibrium configuration frozen in ``config.yaml``:

    u_i      axial (vertical) displacement of the cross section  [m]
    a_i      rotation of the cross section about the spring axis  [rad]

Each helical segment ``j`` (between cross sections ``j`` and ``j+1``) is a
discrete elastic-rod element whose reduced (helix-homogenised) constitutive law
couples axial stretch and axial rotation,

    T_j = k  * delta_j + c * psi_j + T0_j        (axial force)
    M_j = c  * delta_j + q * psi_j               (axial moment)

with ``delta_j = u_j - u_{j+1}`` the segment stretch and
``psi_j = a_{j+1} - a_j`` the relative rotation.  ``T0_j`` is the pretension of
the reference configuration; because the reference configuration is exactly the
static gravity equilibrium, ``T0`` cancels the nodal weight identically at every
free node, so the displacement dynamics are driven by the elastic increment
alone while gravity remains physically present (it appears explicitly in the
exported momentum budgets together with the clamp reaction).

The rigid load is rigidly welded to the last cross section through the frozen
body-frame attachment offset, contributing its mass to the axial degree of
freedom and its polar moment ``I_zz`` to the rotational degree of freedom.
The lowest axial and lowest torsional eigenfrequencies and the beat (exchange)
frequency of the discrete operator are matched to the frozen coupled-mode
regime by solving the generalised eigenproblem of the assembled operator, so the
translation <-> rotation exchange emerges from the integrated equations of
motion rather than from any prescribed motion.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - SciPy is optional for the eigen analysis
    from scipy.linalg import eigh as _scipy_eigh
except Exception:  # pragma: no cover
    _scipy_eigh = None


def _lowest_eigenvalues(matrix: np.ndarray, count: int) -> np.ndarray:
    """Ascending eigenvalues of a symmetric matrix (lowest ``count``)."""
    sym = 0.5 * (matrix + matrix.T)
    if _scipy_eigh is not None:
        try:
            hi = min(count - 1, sym.shape[0] - 1)
            return np.asarray(
                _scipy_eigh(sym, eigvals_only=True, subset_by_index=[0, hi]),
                dtype=float,
            )
        except Exception:
            pass
    return np.linalg.eigvalsh(sym)[:count]


def _max_eigenvalue(matrix: np.ndarray) -> float:
    sym = 0.5 * (matrix + matrix.T)
    return float(np.linalg.eigvalsh(sym)[-1])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


class BenchmarkSolver:
    """Discrete axial-torsional elastic-rod Wilberforce pendulum solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.config = cfg
        self.seed = int(seed)
        spring = cfg["spring"]
        load = cfg["load"]
        regime = cfg.get("coupled_mode_regime", {}) or {}

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g_mag = float(-self.gravity[2])

        self.n = int(spring["vertex_count"])
        self.n_clamp = max(1, int(spring["upper_clamped_vertex_count"]))
        self.coil_radius = float(spring["coil_radius"])
        self.coil_count = float(spring["coil_count"])
        self.wire_radius = float(spring["wire_radius"])
        self.z_top = float(spring["equilibrium_top_z"])
        self.z_bot = float(spring["equilibrium_bottom_z"])
        self.spring_mass = float(spring["total_mass"])

        self.load_mass = float(load["mass"])
        self.load_inertia = np.asarray(load["principal_inertia"], dtype=float)
        self.load_offset = np.asarray(load["attachment_offset_body"], dtype=float)
        self.load_radius = float(load["radius"])
        self.load_height = float(load["height"])

        n = self.n
        # Reference (gravity equilibrium) helix.
        self.theta = np.linspace(0.0, 2.0 * math.pi * self.coil_count, n)
        self.z_ref = np.linspace(self.z_top, self.z_bot, n)

        node_mass = self.spring_mass / n
        self.node_mass = node_mass
        self.m_axial = np.full(n, node_mass)
        self.m_axial[-1] += self.load_mass
        # Polar inertia of a helical cross section about the spring axis.
        self.i_torsion = np.full(n, node_mass * self.coil_radius ** 2)
        self.i_torsion[-1] += float(self.load_inertia[2])

        self._calibrate(regime)

        # -------------------------------------------------- initial condition
        dz = float(cfg["initial_conditions"]["upward_displacement"])
        u0, a0 = self._static_hold(dz)
        self.u = u0
        self.a = a0
        v0 = np.asarray(cfg["initial_conditions"]["initial_linear_velocity"], float)
        w0 = np.asarray(cfg["initial_conditions"]["initial_angular_velocity"], float)
        self.ud = np.zeros(n)
        self.ad = np.zeros(n)
        self.ud[self.n_clamp:] = float(v0[2])
        self.ad[self.n_clamp:] = float(w0[2])

        self.time = 0.0
        self._cum_impulse = np.zeros((3, 3))
        self._cum_ang_impulse = np.zeros((3, 3))
        self._prev_terms = None
        self._steps = 0

        self._udd, self._add = self._accelerations(self.u, self.a)
        self._prev_terms = self._budget_terms()

    # ------------------------------------------------------- model assembly
    def _basis_stiffness(self):
        """Return (Ku, Kq, Kc): stiffness contributions for unit k, q, c.

        Degrees of freedom are ordered [u_0..u_{n-1}, a_0..a_{n-1}].
        """
        n = self.n
        ndof = 2 * n
        Ku = np.zeros((ndof, ndof))
        Kq = np.zeros((ndof, ndof))
        Kc = np.zeros((ndof, ndof))
        for j in range(n - 1):
            iu0, iu1 = j, j + 1
            ia0, ia1 = n + j, n + j + 1
            # e_j = u_j - u_{j+1}
            eu = np.zeros(ndof)
            eu[iu0] = 1.0
            eu[iu1] = -1.0
            # psi_j = a_{j+1} - a_j
            ep = np.zeros(ndof)
            ep[ia1] = 1.0
            ep[ia0] = -1.0
            Ku += np.outer(eu, eu)
            Kq += np.outer(ep, ep)
            Kc += np.outer(eu, ep) + np.outer(ep, eu)
        return Ku, Kq, Kc

    def _elastic_forces(self, u, a):
        """Generalised elastic forces (axial force, axial torque) per section."""
        e = u[:-1] - u[1:]
        psi = a[1:] - a[:-1]
        tension = self.k_seg * e + self.c_seg * psi
        moment = self.c_seg * e + self.q_seg * psi
        fu = np.zeros(self.n)
        fu[:-1] -= tension
        fu[1:] += tension
        ta = np.zeros(self.n)
        ta[:-1] += moment
        ta[1:] -= moment
        return fu, ta

    def _accelerations(self, u, a):
        fu, ta = self._elastic_forces(u, a)
        udd = fu / self.m_axial
        add = ta / self.i_torsion
        udd[: self.n_clamp] = 0.0
        add[: self.n_clamp] = 0.0
        return udd, add

    def _calibrate(self, regime) -> None:
        n, nc = self.n, self.n_clamp
        Ku, Kq, Kc = self._basis_stiffness()
        free_u = np.arange(nc, n)
        free_a = np.arange(n + nc, 2 * n)
        free = np.concatenate((free_u, free_a))
        mass_free = np.concatenate((self.m_axial[nc:], self.i_torsion[nc:]))
        inv_sqrt = 1.0 / np.sqrt(mass_free)
        scale = np.outer(inv_sqrt, inv_sqrt)

        Ku_f = Ku[np.ix_(free, free)]
        Kq_f = Kq[np.ix_(free, free)]
        Kc_f = Kc[np.ix_(free, free)]

        tz = float(regime.get("target_vertical_period", 1.2) or 1.2)
        tt = float(regime.get("target_torsional_period", 1.24) or 1.24)
        omega_axial = 2.0 * math.pi / tz
        omega_tors = 2.0 * math.pi / tt

        # Unit-stiffness fundamental eigenvalues of the decoupled operators.
        lam_u = _lowest_eigenvalues(Ku_f * scale, 2)[0]
        lam_q = _lowest_eigenvalues(Kq_f * scale, 2)[0]
        self.k_seg = float(omega_axial ** 2 / max(lam_u, 1e-300))
        self.q_seg = float(omega_tors ** 2 / max(lam_q, 1e-300))

        base = (self.k_seg * Ku_f + self.q_seg * Kq_f) * scale
        coupling = Kc_f * scale

        revival = float(regime.get("expected_vertical_revival_time", 9.0) or 9.0)
        if revival <= 0.0:
            revival = 9.0
        delta_target = 2.0 * math.pi / revival

        def splitting(c_val: float) -> float:
            lam = _lowest_eigenvalues(base + c_val * coupling, 2)
            lam = np.clip(lam, 1e-12, None)
            w = np.sqrt(lam)
            return float(abs(w[1] - w[0]))

        c_hi = 0.45 * math.sqrt(self.k_seg * self.q_seg)
        if splitting(c_hi) < delta_target:
            c_best = c_hi
        else:
            lo, hi = 0.0, c_hi
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if splitting(mid) < delta_target:
                    lo = mid
                else:
                    hi = mid
            c_best = 0.5 * (lo + hi)
        self.c_seg = float(c_best)

        # Stiffness matrix actually used, and the stable explicit step size.
        self.K_full = self.k_seg * Ku + self.q_seg * Kq + self.c_seg * Kc
        full = (self.k_seg * Ku_f + self.q_seg * Kq_f + self.c_seg * Kc_f) * scale
        lam_max = max(_max_eigenvalue(full), 1e-12)
        omega_max = math.sqrt(lam_max)
        sample_dt = float(self.config["sample_dt"])
        dt_limit = 0.9 / omega_max  # < 2/omega_max stability bound of Verlet
        self.substeps = max(1, int(math.ceil(sample_dt / dt_limit)))
        self.dt = sample_dt / self.substeps
        self._free_dofs = free

        # Pretension of the frozen gravity-equilibrium reference configuration:
        # T0_j carries everything hanging below segment j.
        below = np.zeros(self.n - 1)
        acc = self.node_mass + self.load_mass
        for j in range(self.n - 2, -1, -1):
            below[j] = acc
            acc += self.node_mass
        self.pretension = below * self.g_mag

        # Frozen positive reference scales.
        v_char = omega_axial * float(self.config["initial_conditions"]["upward_displacement"])
        m_tot = self.spring_mass + self.load_mass
        self.reference_momentum = float(max(m_tot * v_char, 1e-6))
        theta_char = v_char * math.sqrt(
            (m_tot * self.k_seg) / max(self.q_seg, 1e-300)
        ) / max(omega_tors, 1e-12)
        self.reference_angular_momentum = float(
            max(float(self.load_inertia[2]) * omega_tors * max(theta_char, 0.1), 1e-6)
        )

    def _static_hold(self, dz: float):
        """Equilibrium shape with the load held ``dz`` above equilibrium, untwisted."""
        n, nc = self.n, self.n_clamp
        K = self.K_full
        constrained = list(range(nc)) + [n - 1] + list(range(n, n + nc)) + [2 * n - 1]
        constrained = sorted(set(constrained))
        free = [i for i in range(2 * n) if i not in set(constrained)]
        xc = np.zeros(2 * n)
        xc[n - 1] = dz
        rhs = -K[np.ix_(free, constrained)] @ xc[constrained]
        sol = np.linalg.solve(K[np.ix_(free, free)], rhs)
        x = xc.copy()
        x[free] = sol
        u = x[:n].copy()
        a = x[n:].copy()
        u[:nc] = 0.0
        a[:nc] = 0.0
        return u, a
