"""Cosserat (Simo-Reissner) elastic rod solver for the gravity-induced
three-dimensional curling of a naturally curved rod.

Physical model
--------------
The rod centerline is discretized into ``N`` nodes (``M = N-1`` edges).  Each
edge carries a material frame ``R_j = [d1 d2 d3]`` (columns, world
coordinates).  Degrees of freedom are the nodal positions ``x_i`` and the edge
frames ``R_j``, with nodal velocities ``v_i`` and edge angular velocities
``w_j`` (world frame).

Strain measures (Cosserat rod):

* shear / stretch      ``sigma_j = R_j^T (e_j / l0_j) - e3``,  ``e_j = x_{j+1}-x_j``
* bend / twist         ``kappa_j = rotvec(R_j^T R_{j+1}) / l_vor_j``

Elastic energy

    E = 1/2 sum_j l0_j sigma_j . S sigma_j
      + 1/2 sum_j l_vor_j (kappa_j - kappa_j^0) . B (kappa_j - kappa_j^0)

with ``S = diag(GA, GA, EA)`` and ``B = diag(EI, EI, GJ)``.  The natural
(reference) curvature ``kappa^0`` is measured directly from the frozen planar
circular arc of the scene, so the undeformed configuration is stress free and
carries zero natural twist, exactly as the task prescribes.

Nodal forces and edge torques are the analytic gradients of that energy:

    f_j      = R_j S sigma_j            (force +f_j on node j, -f_j on node j+1)
    tau_j   += e_j x f_j                (shear couple on edge j)
    Mw_j     = R_j B (kappa_j-kappa^0_j)/l_vor_j
    tau_j   += Mw_j ,  tau_{j+1} -= Mw_j (equal/opposite internal couple)

plus uniform gravity (ramped by the frozen load schedule) and viscous damping
(mass-proportional drag plus dashpots on the relative motion of neighbours),
which drives the rod along the quasi-static equilibrium branch.

Time stepping is semi-implicit (symplectic) Euler with a fixed sub-step chosen
from the stiffest discrete mode; edge frames are advanced with the exact
Rodrigues exponential map and re-orthonormalized periodically.

The root clamp freezes nodes 0 and 1 and the frame of edge 0 (root tangent
pointing down, no imposed rotation); the tip is free.  Out-of-plane motion is
seeded only by the frozen ``branch_selection_perturbation`` and then grows
through the physical bend-twist coupling.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew matrices of a stack of vectors, shape (M,3) -> (M,3,3)."""
    m = v.shape[0]
    k = np.zeros((m, 3, 3), dtype=float)
    k[:, 0, 1] = -v[:, 2]
    k[:, 0, 2] = v[:, 1]
    k[:, 1, 0] = v[:, 2]
    k[:, 1, 2] = -v[:, 0]
    k[:, 2, 0] = -v[:, 1]
    k[:, 2, 1] = v[:, 0]
    return k


def _exp_map(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map of rotation vectors, (M,3) -> (M,3,3)."""
    th2 = np.einsum("mi,mi->m", w, w)
    th = np.sqrt(th2)
    small = th < 1.0e-7
    a = np.where(small, 1.0 - th2 / 6.0, np.sin(np.where(small, 1.0, th)) / np.where(small, 1.0, th))
    b = np.where(
        small,
        0.5 - th2 / 24.0,
        (1.0 - np.cos(np.where(small, 1.0, th))) / np.where(small, 1.0, th2),
    )
    k = _skew(w)
    kk = np.einsum("mij,mjk->mik", k, k)
    out = np.zeros_like(k)
    out[:, 0, 0] = 1.0
    out[:, 1, 1] = 1.0
    out[:, 2, 2] = 1.0
    out += a[:, None, None] * k + b[:, None, None] * kk
    return out


def _rot_vec(rot: np.ndarray) -> np.ndarray:
    """Logarithm map of a stack of rotation matrices, (M,3,3) -> (M,3)."""
    v = 0.5 * np.stack(
        (
            rot[:, 2, 1] - rot[:, 1, 2],
            rot[:, 0, 2] - rot[:, 2, 0],
            rot[:, 1, 0] - rot[:, 0, 1],
        ),
        axis=-1,
    )
    s = np.sqrt(np.einsum("mi,mi->m", v, v))
    c = 0.5 * (rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2] - 1.0)
    theta = np.arctan2(s, c)
    scale = np.where(s < 1.0e-10, 1.0, theta / np.where(s < 1.0e-10, 1.0, s))
    return v * scale[:, None]


def _orthonormalize(rot: np.ndarray) -> np.ndarray:
    """Gram-Schmidt re-orthonormalization of frame columns."""
    c0 = rot[:, :, 0]
    c1 = rot[:, :, 1]
    c0 = c0 / np.maximum(np.linalg.norm(c0, axis=1, keepdims=True), _EPS)
    c1 = c1 - np.einsum("mi,mi->m", c1, c0)[:, None] * c0
    c1 = c1 / np.maximum(np.linalg.norm(c1, axis=1, keepdims=True), _EPS)
    c2 = np.cross(c0, c1)
    return np.stack((c0, c1, c2), axis=-1)


class BenchmarkSolver:
    """Deterministic Cosserat-rod solver, NumPy only."""

    # Material of the frozen ``rod_elastomer`` rod.  The scaffold freezes the
    # geometry, density and natural curvature; a soft elastomer modulus places
    # the 0.103 m snapshot on the documented three-dimensional branch of the
    # bend-twist test (gravito-bending number rho*A*g*L^3/EI ~ 34).
    YOUNG_MODULUS = 5.0e5
    POISSON_RATIO = 0.5

    # Damping (viscous, physical form) used to follow the quasi-static branch.
    DRAG_RATE = 10.0            # 1/s, mass proportional nodal drag
    SPIN_RATE = 10.0            # 1/s, inertia proportional edge spin drag
    PAIR_ZETA = 0.30            # damping ratio of the stiffest axial mode
    ROT_ZETA = 0.30             # damping ratio of the stiffest rotational mode
    STEP_SAFETY = 0.36          # fraction of the explicit stability limit

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.config = cfg
        rng = np.random.default_rng(int(seed))
        self._seed = int(seed)
        del rng  # deterministic solver; no stochastic terms are used

        rod = cfg["rod"]
        init = cfg["initial_conditions"]
        frame0 = scene.initial_state()

        self.length = float(rod["length"])
        self.rod_radius = float(rod["radius"])
        self.density = float(rod["density"])

        # ---------------------------------------------------------- geometry
        x0 = np.array(frame0.position, dtype=float)
        self.n_nodes = int(x0.shape[0])
        self.n_edges = self.n_nodes - 1
        self.arc_length = np.asarray(frame0.attributes["arc_length"], dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.total_mass = float(self.mass.sum())

        # Reference (natural) configuration: the frozen planar circular arc.
        e_ref = x0[1:] - x0[:-1]
        self.l0 = np.linalg.norm(e_ref, axis=1)
        t_ref = e_ref / self.l0[:, None]
        # Natural arc lies in the x-z plane -> the arc binormal (+y) is a
        # twist-free material director everywhere along the rest shape.
        d1_ref = np.tile(np.array([0.0, 1.0, 0.0]), (self.n_edges, 1))
        d1_ref = d1_ref - np.einsum("mi,mi->m", d1_ref, t_ref)[:, None] * t_ref
        d1_ref /= np.maximum(np.linalg.norm(d1_ref, axis=1, keepdims=True), _EPS)
        d2_ref = np.cross(t_ref, d1_ref)
        rot_ref = np.stack((d1_ref, d2_ref, t_ref), axis=-1)
        self.R = _orthonormalize(rot_ref.copy())
        self.R0_clamp = self.R[0].copy()

        # Reference bend/twist strain measured from the natural arc frames.
        self.l_vor = 0.5 * (self.l0[:-1] + self.l0[1:])
        rel_ref = np.einsum("mia,mib->mab", self.R[:-1], self.R[1:])
        self.u_rest = _rot_vec(rel_ref)

        # ------------------------------------------------------- constitutive
        area = np.pi * self.rod_radius**2
        inertia = 0.25 * np.pi * self.rod_radius**4
        young = float(self.YOUNG_MODULUS)
        shear = young / (2.0 * (1.0 + float(self.POISSON_RATIO)))
        self.EA = young * area
        self.GA = shear * area
        self.EI = young * inertia
        self.GJ = shear * 2.0 * inertia
        self.S_stiff = np.array([self.GA, self.GA, self.EA])
        self.B_stiff = np.array([self.EI, self.EI, self.GJ])

        m_edge = self.density * area * self.l0
        # Isotropic edge inertia (polar value, the larger one): rotational
        # inertia only affects transients, never the equilibrium branch.
        self.J_edge = m_edge * (0.5 * self.rod_radius**2)

        # ------------------------------------------------------------- state
        self.x = x0.copy()
        pert = np.asarray(init.get("branch_selection_perturbation", [0.0, 0.0, 0.0]), dtype=float)
        shape = (self.arc_length / self.length) ** 2
        shape[0] = 0.0
        shape[1] = 0.0
        self.x = self.x + shape[:, None] * pert[None, :]
        self.v = np.zeros_like(self.x)
        self.w = np.zeros((self.n_edges, 3), dtype=float)

        self.clamp_nodes = np.array([0, 1], dtype=int)
        self.clamp_target = self.x[self.clamp_nodes].copy()
        self.root_position = self.x[0].copy()

        # ------------------------------------------------------------ loading
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        sched = dict(cfg.get("load_schedule") or {})
        self.ramp_start = float(sched.get("start_time", 0.0))
        self.ramp_full = float(sched.get("full_load_time", self.ramp_start))

        # ------------------------------------------------------------- damping
        k_axial = self.EA / float(self.l0.min())
        # Smallest free (non-clamped) nodal mass bounds the explicit step; the
        # free tip node carries half the lumped mass of an interior node.
        m_min = float(np.min(self.mass[2:])) if self.n_nodes > 2 else float(self.mass.min())
        omega_axial = 2.0 * np.sqrt(k_axial / m_min)
        k_rot = float(np.max(self.B_stiff)) / float(self.l_vor.min()) + float(
            self.l0.max() * self.GA
        )
        omega_rot = 2.0 * np.sqrt(k_rot / float(self.J_edge.min()))
        self.c_pair = 0.5 * self.PAIR_ZETA * m_min * omega_axial
        self.c_rot = 0.5 * self.ROT_ZETA * float(self.J_edge.min()) * omega_rot

        omega_max = max(omega_axial, omega_rot)
        dt_limit = 2.0 / omega_max
        dt_damp = min(m_min / max(2.0 * self.c_pair, _EPS),
                      float(self.J_edge.min()) / max(2.0 * self.c_rot, _EPS))
        self.dt_target = float(min(self.STEP_SAFETY * dt_limit, 0.5 * dt_damp, 1.0e-4))

        self.time = 0.0
        self._steps = 0
        self._max_stretch = 0.0

        # frozen normalization scales
        weight = self.total_mass * float(np.linalg.norm(self.gravity))
        self.reference_force = float(max(weight, 1.0e-9))
        self.reference_torque = float(max(weight * self.length, 1.0e-12))
        self.constraint_scale = float(self.length)

    # ------------------------------------------------------------------ physics
    def _load_factor(self, t: float) -> float:
        if self.ramp_full <= self.ramp_start:
            return 1.0
        return float(np.clip((t - self.ramp_start) / (self.ramp_full - self.ramp_start), 0.0, 1.0))

    def _forces(self, t: float, damping: bool = True):
        """Nodal forces (N,3), edge torques (M,3) and the applied gravity."""
        x, v, w, rot = self.x, self.v, self.w, self.R
        force = np.zeros_like(x)
        torque = np.zeros((self.n_edges, 3), dtype=float)

        # --- shear / stretch -------------------------------------------------
        edge = x[1:] - x[:-1]
        eps = edge / self.l0[:, None]
        sigma = np.einsum("mik,mi->mk", rot, eps)
        sigma[:, 2] -= 1.0
        n_mat = sigma * self.S_stiff[None, :]
        f_edge = np.einsum("mik,mk->mi", rot, n_mat)
        force[:-1] += f_edge
        force[1:] -= f_edge
        torque += np.cross(edge, f_edge)

        # --- bend / twist ----------------------------------------------------
        rel = np.einsum("mia,mib->mab", rot[:-1], rot[1:])
        u = _rot_vec(rel)
        m_mat = (u - self.u_rest) * self.B_stiff[None, :] / self.l_vor[:, None]
        m_world = np.einsum("mik,mk->mi", rot[:-1], m_mat)
        torque[:-1] += m_world
        torque[1:] -= m_world

        # --- gravity ---------------------------------------------------------
        g_now = self.gravity * self._load_factor(t)
        f_grav = self.mass[:, None] * g_now[None, :]
        force += f_grav

        # --- viscous damping -------------------------------------------------
        if damping:
            force -= (self.DRAG_RATE * self.mass)[:, None] * v
            dv = v[1:] - v[:-1]
            fd = self.c_pair * dv
            force[:-1] += fd
            force[1:] -= fd
            torque -= (self.SPIN_RATE * self.J_edge)[:, None] * w
            dw = w[1:] - w[:-1]
            td = self.c_rot * dw
            torque[:-1] += td
            torque[1:] -= td

        self._max_stretch = max(self._max_stretch, float(np.max(np.abs(sigma[:, 2]))))
        return force, torque, f_grav

    def _step(self, dt: float) -> None:
        force, torque, _ = self._forces(self.time)
        self.v += dt * (force / self.mass[:, None])
        self.w += dt * (torque / self.J_edge[:, None])
        self.v[self.clamp_nodes] = 0.0
        self.w[0] = 0.0
        self.x += dt * self.v
        self.x[self.clamp_nodes] = self.clamp_target
        self.R = np.einsum("mij,mjk->mik", _exp_map(self.w * dt), self.R)
        self.R[0] = self.R0_clamp
        self._steps += 1
        if self._steps % 500 == 0:
            self.R = _orthonormalize(self.R)
        self.time += dt

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1.0e-15:
            span = target - self.time
            n_sub = max(1, int(np.ceil(span / self.dt_target)))
            dt = span / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        force, _, f_grav = self._forces(t, damping=True)

        # Actual accelerations: clamped nodes are kinematically fixed.
        accel = force / self.mass[:, None]
        accel[self.clamp_nodes] = 0.0
        m_a = self.mass[:, None] * accel

        # Clamp reaction closing linear and angular momentum balance of the rod
        # (internal elastic pairs cancel over the whole body).
        reaction = m_a.sum(axis=0) - f_grav.sum(axis=0)
        root = self.x[0]
        l_rate = np.cross(self.x, m_a).sum(axis=0)
        grav_moment = np.cross(self.x, f_grav).sum(axis=0)
        couple = l_rate - grav_moment - np.cross(root, reaction)

        n = self.n_nodes
        k_terms = n + 1
        b_force = np.zeros((1, k_terms, 3), dtype=float)
        b_pos = np.zeros((1, k_terms, 3), dtype=float)
        b_couple = np.zeros((1, k_terms, 3), dtype=float)
        b_force[0, :n, :] = f_grav
        b_pos[0, :n, :] = self.x
        b_force[0, n, :] = reaction
        b_pos[0, n, :] = root
        b_couple[0, n, :] = couple

        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_length": self.arc_length.copy(),
                "natural_curvature": np.full(n, 1.0 / float(self.config["rod"]["natural_radius"])),
                "natural_twist": np.zeros(n),
                "is_clamped_root": np.r_[1.0, 1.0, np.zeros(n - 2)],
            },
            fields={
                "budget_external_force": b_force,
                "budget_force_position": b_pos,
                "budget_external_couple": b_couple,
                "budget_domain_weights": np.ones((1, n), dtype=float),
                "constraint_value": self.x[self.clamp_nodes].copy(),
                "constraint_target": self.clamp_target.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": self.constraint_scale,
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
                "rod_radius": float(self.rod_radius),
                "load_factor": self._load_factor(t),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        tip = self.x[-1]
        return {
            "model": "cosserat_rod_semi_implicit_euler",
            "nodes": int(self.n_nodes),
            "sub_steps": int(self._steps),
            "dt": float(self.dt_target),
            "young_modulus": float(self.YOUNG_MODULUS),
            "bending_stiffness": float(self.EI),
            "torsional_stiffness": float(self.GJ),
            "max_axial_strain": float(self._max_stretch),
            "tip_out_of_plane_y": float(tip[1]),
            "max_abs_y": float(np.max(np.abs(self.x[:, 1]))),
            "seed": int(self._seed),
        }
