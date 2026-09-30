"""Discrete elastic rod solver for the twist-induced snap of an open trefoil knot.

Physical model (implemented from scratch with NumPy/SciPy only):

* Centerline discretised as ``N`` vertices / ``N-1`` edges carrying lumped mass.
* Inextensibility: stiff axial springs on every edge (rest lengths frozen from
  the initial knot geometry) with strain-rate damping.
* Bending: naturally straight, isotropic rod, discrete energy
  ``E_b = sum_i (alpha / l_i) * (1 - cos(theta_i))`` which is the second order
  form of ``int alpha/2 * kappa^2 ds``.
* Twist: per-edge reference directors are transported with the tangents
  (time-parallel transport), reference twist per interior vertex is unwrapped
  continuously, and the material frame angles are eliminated quasistatically
  (zero torsional inertia).  For an isotropic rod this yields a uniform
  transmitted axial moment ``tau = beta * Phi / sum_j l_j`` with
  ``Phi = phi_clamp(t) + sum_j refTwist_j`` (twist / writhe exchange).  The
  position gradient of the twist energy uses the curvature-binormal identity
  ``grad_{x_{i-1}} psi_i = kb_i / (2 |e_{i-1}|)``,
  ``grad_{x_{i+1}} psi_i = -kb_i / (2 |e_i|)``.
* Finite thickness self-contact: penalty springs plus normal damping between
  vertex pairs closer than ``2 * rod radius`` (broad phase via a KD-tree).
* Boundary conditions: the two vertices at each tail are pinned (position and
  tangent clamped); the driven clamp imposes the smoothstep twist ramp from the
  frozen configuration about its outward tail tangent.
* Time stepping: symplectic (semi-implicit) Euler with exponential viscous
  damping, at a fixed step chosen from the explicit stability limits of the
  stretching / bending / contact stiffnesses.

Everything is deterministic: no random numbers influence the dynamics.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional broad-phase acceleration
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - fallback path
    _KDTree = None


def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def _unit(a: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    n = _norm(a)
    return a / np.maximum(n, eps)[:, None]


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class BenchmarkSolver:
    # ---- material constants (SI) -------------------------------------
    BEND_STIFFNESS = 0.12      # alpha = E*I   [N m^2]
    TWIST_RATIO = 0.70         # beta / alpha for an isotropic rod
    STRETCH_STIFFNESS = 3000.0  # [N/m] per edge (near-inextensible)
    STRETCH_DAMPING = 0.03     # [N s/m] per edge
    CONTACT_STIFFNESS = 450.0  # [N/m]
    CONTACT_DAMPING = 0.06     # [N s/m]
    DRAG = 0.0025              # [N s/m] per vertex (ambient dissipation)
    # ---- numerics ----------------------------------------------------
    CONTACT_SKIP = 4           # ignore |i-j| <= CONTACT_SKIP
    REBUILD_EVERY = 25         # broad-phase rebuild interval (steps)
    BROAD_MARGIN = 0.05        # [m]
    MAX_STEPS = 260000
    DT_CAP = 1.5e-4

    # ==================================================================
    # setup
    # ==================================================================
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.config = cfg
        self.seed = int(seed)

        rod = cfg["rod"]
        clamps = cfg["clamps"]
        self.n = int(rod["vertex_count"])
        self.radius = float(rod["radius"])
        self.total_mass = float(rod["mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        # --- initial geometry, resampled to uniform arc length ---------
        init = scene.initial_state()
        p0 = np.asarray(init.position, dtype=float).copy()
        self.scene_attributes = {
            str(k): np.asarray(v, dtype=float).copy()
            for k, v in dict(init.attributes).items()
        }
        p0 = self._resample(p0)
        left = np.asarray(clamps["left"]["position"], dtype=float)
        right = np.asarray(clamps["right"]["position"], dtype=float)
        p0[0] = left
        p0[-1] = right

        self.x = p0
        self.v = np.zeros_like(p0)
        n = self.n
        self.mass = np.full(n, self.total_mass / n)
        self.inv_mass = 1.0 / self.mass

        # --- clamped (kinematic) vertices ------------------------------
        self.fixed_idx = np.array([0, 1, n - 2, n - 1], dtype=int)
        self.left_idx = np.array([0, 1], dtype=int)
        self.right_idx = np.array([n - 2, n - 1], dtype=int)
        self.fixed_pos = self.x[self.fixed_idx].copy()
        self.free_mask = np.ones(n, dtype=bool)
        self.free_mask[self.fixed_idx] = False

        # --- rest geometry ---------------------------------------------
        edges = self.x[1:] - self.x[:-1]
        self.rest_len = np.maximum(_norm(edges), 1e-9)
        self.total_length = float(self.rest_len.sum())
        # Voronoi lengths of interior vertices 1..n-2
        self.lbar = 0.5 * (self.rest_len[:-1] + self.rest_len[1:])
        self.lbar_sum = float(self.lbar.sum())
        self.bend_k = self.BEND_STIFFNESS / self.lbar
        self.twist_beta = self.TWIST_RATIO * self.BEND_STIFFNESS

        # --- clamp tail tangents (fixed by the clamps) -----------------
        t_left = self.x[1] - self.x[0]
        t_right = self.x[-1] - self.x[-2]
        self.tangent_left = t_left / max(float(np.linalg.norm(t_left)), 1e-12)
        self.tangent_right = t_right / max(float(np.linalg.norm(t_right)), 1e-12)
        self.twist_axis = self.tangent_right.copy()

        # --- reference directors (Bishop-like frame) --------------------
        tangents = _unit(edges)
        self.d1 = self._orthogonal_seed(tangents)
        self.ref_twist = np.zeros(n - 2)  # continuous, per interior vertex
        self.ref_twist_0 = 0.0  # frozen below, after the first evaluation

        # --- twist ramp ------------------------------------------------
        ramp = clamps["twist_ramp"]
        self.ramp_start = float(ramp["ramp_start"])
        self.ramp_end = float(ramp["ramp_end"])
        self.final_turns = float(ramp["final_relative_turns"])

        # --- time step from explicit stability limits -------------------
        mv = float(self.mass[0])
        lmin = float(min(self.lbar.min(), self.rest_len.min()))
        dt_bend = 0.2 * math.sqrt(mv * lmin ** 3 / self.BEND_STIFFNESS)
        dt_stretch = 0.25 * math.sqrt(mv / self.STRETCH_STIFFNESS)
        dt_contact = 0.25 * math.sqrt(mv / max(self.CONTACT_STIFFNESS, 1e-9))
        dt = min(self.DT_CAP, dt_bend, dt_stretch, dt_contact)
        if self.duration / dt > self.MAX_STEPS:
            dt = self.duration / self.MAX_STEPS
        self.dt = float(dt)
        self.vertex_mass = mv

        # --- contact broad phase ---------------------------------------
        self.contact_d0 = 2.0 * self.radius
        self.pair_a = np.zeros(0, dtype=int)
        self.pair_b = np.zeros(0, dtype=int)
        self._step_count = 0
        self._rebuild_pairs()

        # --- accumulators / diagnostics --------------------------------
        self.time = 0.0
        self.impulse = np.zeros(3)
        self.angular_impulse = np.zeros(3)
        self.tau = 0.0
        self.theta = np.zeros(n - 1)
        self.theta_prev = np.zeros(n - 1)
        self.theta_prev_time = 0.0
        self.total_twist = 0.0
        self.max_speed = 0.0
        self.max_stretch = 0.0
        self.contact_count = 0

        self._terms = np.zeros((4, 3))
        self._term_pos = np.tile(self.x.mean(axis=0), (4, 1))
        self._term_couple = np.zeros((4, 3))

        # frozen normalisation scales
        v_ref = math.sqrt(abs(self.gravity[2]) * self.total_length)
        self.reference_momentum = float(self.total_mass * v_ref)
        self.reference_angular_momentum = float(
            self.total_mass * v_ref * max(self.total_length / 6.0, 1e-3)
        )
        self.constraint_scale = float(self.radius)

        # Initial force evaluation so that frame 0 already reports the live
        # external wrench (statics of the clamped knot under gravity).
        self._evaluate(self.x, self.v, self.time)
        # The seeded reference frame carries an arbitrary holonomy; freeze it so
        # the initial knot is genuinely twist free (initial_material_twist_turns
        # is 0.0) and only clamp-driven twist plus writhe exchange loads the rod.
        self.ref_twist_0 = float(self.ref_twist.sum())
        f_int0, f_grav0 = self._evaluate(self.x, self.v, self.time)
        reac0 = -(f_int0[self.fixed_idx] + f_grav0[self.fixed_idx])
        self._update_budget_terms(self.x, f_grav0, np.zeros_like(f_grav0), reac0)

    # ------------------------------------------------------------------
    def _resample(self, p: np.ndarray) -> np.ndarray:
        """Uniform arc-length resampling of the frozen initial centerline."""
        seg = _norm(p[1:] - p[:-1])
        s = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(s[-1])
        if total <= 0.0:
            return p
        target = np.linspace(0.0, total, len(p))
        out = np.empty_like(p)
        for k in range(3):
            out[:, k] = np.interp(target, s, p[:, k])
        return out

    @staticmethod
    def _orthogonal_seed(tangents: np.ndarray) -> np.ndarray:
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(tangents), 1))
        bad = np.abs(np.einsum("ij,ij->i", tangents, ref)) > 0.9
        ref[bad] = np.array([1.0, 0.0, 0.0])
        d = ref - tangents * np.einsum("ij,ij->i", ref, tangents)[:, None]
        return _unit(d)

    # ==================================================================
    # kinematic driving
    # ==================================================================
    def _clamp_angle(self, t: float) -> float:
        span = max(self.ramp_end - self.ramp_start, 1e-9)
        u = (t - self.ramp_start) / span
        u = min(max(u, 0.0), 1.0)
        u = u * u * (3.0 - 2.0 * u)
        return 2.0 * math.pi * self.final_turns * u

    # ==================================================================
    # forces
    # ==================================================================
    def _rebuild_pairs(self) -> None:
        cutoff = self.contact_d0 + self.BROAD_MARGIN
        x = self.x
        if _KDTree is not None:
            tree = _KDTree(x)
            pairs = tree.query_pairs(cutoff, output_type="ndarray")
            if len(pairs):
                a = pairs[:, 0]
                b = pairs[:, 1]
            else:
                a = np.zeros(0, dtype=int)
                b = np.zeros(0, dtype=int)
        else:  # pragma: no cover
            diff = x[:, None, :] - x[None, :, :]
            dist = np.sqrt((diff * diff).sum(-1))
            a, b = np.nonzero(np.triu(dist < cutoff, 1))
        keep = np.abs(a - b) > self.CONTACT_SKIP
        self.pair_a = np.ascontiguousarray(a[keep])
        self.pair_b = np.ascontiguousarray(b[keep])

    def _internal_forces(self, x: np.ndarray, v: np.ndarray, t: float):
        n = self.n
        f = np.zeros((n, 3))

        edges = x[1:] - x[:-1]
        elen = np.maximum(_norm(edges), 1e-12)
        tangents = edges / elen[:, None]

        # ---- stretching (near inextensible) --------------------------
        strain = elen - self.rest_len
        dv = v[1:] - v[:-1]
        rate = np.einsum("ij,ij->i", dv, tangents)
        axial = (self.STRETCH_STIFFNESS * strain + self.STRETCH_DAMPING * rate)
        fs = axial[:, None] * tangents
        np.add.at(f, np.arange(n - 1), fs)
        np.add.at(f, np.arange(1, n), -fs)
        self.max_stretch = max(
            self.max_stretch, float(np.max(np.abs(strain) / self.rest_len))
        )

        a = edges[:-1]
        b = edges[1:]
        la = elen[:-1]
        lb = elen[1:]
        dot = np.einsum("ij,ij->i", a, b)
        lab = la * lb
        cosang = np.clip(dot / lab, -1.0, 1.0)

        # ---- bending: E = sum k_i (1 - cos theta_i), k_i = alpha / lbar
        inv_lab = 1.0 / lab
        dcda = b * inv_lab[:, None] - a * (cosang / (la * la))[:, None]
        dcdb = a * inv_lab[:, None] - b * (cosang / (lb * lb))[:, None]
        kb_ = self.bend_k[:, None]
        idx = np.arange(1, n - 1)
        np.add.at(f, idx - 1, -kb_ * dcda)
        np.add.at(f, idx, kb_ * (dcda - dcdb))
        np.add.at(f, idx + 1, kb_ * dcdb)

        # ---- reference twist (Bishop frame holonomy) ------------------
        d_prev = self.d1[:-1]
        d_cur = self.d1[1:]
        proj = d_prev - tangents[1:] * np.einsum(
            "ij,ij->i", d_prev, tangents[1:]
        )[:, None]
        proj = _unit(proj)
        cross = np.cross(proj, d_cur)
        sin_a = np.einsum("ij,ij->i", cross, tangents[1:])
        cos_a = np.einsum("ij,ij->i", proj, d_cur)
        ang = np.arctan2(sin_a, cos_a)
        self.ref_twist = self.ref_twist + _wrap_pi(ang - self.ref_twist)

        # ---- quasistatic material twist ------------------------------
        # ``ref_twist_0`` is the holonomy of the seeded reference frame on the
        # frozen initial knot; subtracting it makes the initial configuration
        # exactly twist-free (initial_material_twist_turns = 0) so that all
        # transmitted moment comes from the clamp ramp plus subsequent
        # writhe/twist exchange.
        phi = self._clamp_angle(t)
        total = phi + float(self.ref_twist.sum()) - self.ref_twist_0
        self.total_twist = total
        tau = self.twist_beta * total / max(self.lbar_sum, 1e-12)
        self.tau = float(tau)

        # curvature binormal
        denom = np.maximum(lab + dot, 1e-12)
        kb = 2.0 * np.cross(a, b) / denom[:, None]
        g_prev = kb / (2.0 * la)[:, None]
        g_next = -kb / (2.0 * lb)[:, None]
        np.add.at(f, idx - 1, -tau * g_prev)
        np.add.at(f, idx, tau * (g_prev + g_next))
        np.add.at(f, idx + 1, -tau * g_next)

        # ---- finite thickness self contact ---------------------------
        if len(self.pair_a):
            pa = self.pair_a
            pb = self.pair_b
            dvec = x[pa] - x[pb]
            dist = np.maximum(_norm(dvec), 1e-9)
            pen = self.contact_d0 - dist
            act = pen > 0.0
            if np.any(act):
                pa = pa[act]
                pb = pb[act]
                nrm = dvec[act] / dist[act][:, None]
                vn = np.einsum("ij,ij->i", v[pa] - v[pb], nrm)
                mag = self.CONTACT_STIFFNESS * pen[act] - self.CONTACT_DAMPING * vn
                mag = np.maximum(mag, 0.0)
                fc = mag[:, None] * nrm
                np.add.at(f, pa, fc)
                np.add.at(f, pb, -fc)
                self.contact_count = int(len(pa))
            else:
                self.contact_count = 0
        else:
            self.contact_count = 0

        # material frame angles (for reported angular velocity)
        m_i = total * (self.lbar / max(self.lbar_sum, 1e-12))
        self.theta = np.concatenate(([0.0], np.cumsum(m_i - self.ref_twist)))
        return f, tangents

    # ==================================================================
    # one explicit step
    # ==================================================================
    def _evaluate(self, x, v, t, record=False):
        f_int, tangents = self._internal_forces(x, v, t)
        self._tangents = tangents
        f_grav = self.mass[:, None] * self.gravity
        return f_int, f_grav

    def _update_budget_terms(self, x, f_grav, f_drag, reac):
        """Resolve the external wrench into per-term forces, points and couples.

        Four external terms act on the single rod domain: gravity, ambient drag,
        and the two clamp reactions.  Each clamp lumps its two pinned vertices
        into one force applied at the tail vertex; the residual free couple
        restores the exact wrench (this carries the transmitted axial twist
        moment of the driven clamp).  The torque of the lumped force itself is
        supplied by budget_force_position and is not counted twice.
        """
        f_grav_tot = f_grav.sum(axis=0)
        f_drag_tot = f_drag.sum(axis=0)
        r_l = reac[:2].sum(axis=0)
        r_r = reac[2:].sum(axis=0)

        com = (self.mass[:, None] * x).sum(axis=0) / self.total_mass
        pos_l = x[self.left_idx[0]]
        pos_r = x[self.right_idx[-1]]

        tq_grav = np.cross(x, f_grav).sum(axis=0)
        tq_drag = np.cross(x, f_drag).sum(axis=0)
        tq_l = np.cross(x[self.left_idx], reac[:2]).sum(axis=0)
        tq_r = np.cross(x[self.right_idx], reac[2:]).sum(axis=0)

        self._terms = np.array([f_grav_tot, f_drag_tot, r_l, r_r])
        self._term_pos = np.array([com, com, pos_l, pos_r])
        self._term_couple = np.array(
            [
                tq_grav - np.cross(com, f_grav_tot),
                tq_drag - np.cross(com, f_drag_tot),
                tq_l - np.cross(pos_l, r_l),
                tq_r - np.cross(pos_r, r_r),
            ]
        )

        # Internal elastic forces (stretch, bend, twist) and penalty contacts
        # come from rotation-invariant energies / equal-and-opposite pairs, so
        # they carry zero net force and zero net torque about the origin.
        f_ext_tot = f_grav_tot + f_drag_tot + r_l + r_r
        tq_ext_tot = tq_grav + tq_drag + tq_l + tq_r
        return f_ext_tot, tq_ext_tot

    def _step(self, dt: float) -> None:
        x = self.x
        v = self.v
        f_int, f_grav = self._evaluate(x, v, self.time)

        damp = math.exp(-self.DRAG * dt / self.vertex_mass)

        acc = (f_int + f_grav) * self.inv_mass[:, None]
        v_star = v + dt * acc
        v_new = v_star * damp
        f_drag = self.mass[:, None] * (damp - 1.0) * v_star / dt

        # clamped vertices: exact reaction forces, zero velocity
        f_drag[self.fixed_idx] = 0.0
        v_new[self.fixed_idx] = 0.0
        reac = -(f_int[self.fixed_idx] + f_grav[self.fixed_idx])

        x_new = x + dt * v_new
        x_new[self.fixed_idx] = self.fixed_pos

        f_ext_tot, tq_ext_tot = self._update_budget_terms(x, f_grav, f_drag, reac)
        self.impulse = self.impulse + dt * f_ext_tot
        self.angular_impulse = self.angular_impulse + dt * tq_ext_tot

        # ---- transport reference directors with the new tangents -----
        new_edges = x_new[1:] - x_new[:-1]
        new_t = _unit(new_edges)
        d = self.d1
        d = d - new_t * np.einsum("ij,ij->i", d, new_t)[:, None]
        nd = _norm(d)
        bad = nd < 1e-8
        if np.any(bad):
            seed = self._orthogonal_seed(new_t[bad])
            d[bad] = seed
            nd[bad] = 1.0
        self.d1 = d / np.maximum(nd, 1e-14)[:, None]

        self.x = x_new
        self.v = v_new
        self.time += dt
        self._step_count += 1
        self.max_speed = max(self.max_speed, float(np.abs(v_new).max()))
        if self._step_count % self.REBUILD_EVERY == 0:
            self._rebuild_pairs()

    # ==================================================================
    # public API
    # ==================================================================
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        limit = int(self.MAX_STEPS * 2 + 10)
        while self.time < target - 1e-12 and guard < limit:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-15:
                break
            self._step(dt)
            guard += 1
        return self._frame(max(target, 0.0))

    def _angular_velocity(self, t_now: float) -> np.ndarray:
        """Material frame angular velocity mapped to vertices."""
        x = self.x
        v = self.v
        edges = x[1:] - x[:-1]
        elen = np.maximum(_norm(edges), 1e-12)
        tang = edges / elen[:, None]
        dt_edge = (v[1:] - v[:-1]) / elen[:, None]
        dt_perp = dt_edge - tang * np.einsum("ij,ij->i", dt_edge, tang)[:, None]
        omega_edge = np.cross(tang, dt_perp)
        span = t_now - self.theta_prev_time
        if span > 1e-9:
            theta_rate = (self.theta - self.theta_prev) / span
            omega_edge = omega_edge + tang * theta_rate[:, None]
        self.theta_prev = self.theta.copy()
        self.theta_prev_time = t_now
        n = self.n
        omega = np.zeros((n, 3))
        omega[:-1] += 0.5 * omega_edge
        omega[1:] += 0.5 * omega_edge
        omega[0] = omega_edge[0]
        omega[-1] = omega_edge[-1]
        return omega

    def _frame(self, t_now: float) -> StateFrame:
        n = self.n
        x = np.ascontiguousarray(self.x, dtype=float)
        v = np.ascontiguousarray(self.v, dtype=float)
        omega = self._angular_velocity(t_now)

        edges = x[1:] - x[:-1]
        tang = _unit(edges)
        vert_t = np.zeros((n, 3))
        vert_t[:-1] += tang
        vert_t[1:] += tang
        vert_t = _unit(vert_t)
        # per-vertex material frame orientation quaternion (from directors)
        d1 = np.zeros((n, 3))
        d1[:-1] += self.d1
        d1[1:] += self.d1
        d1 = d1 - vert_t * np.einsum("ij,ij->i", d1, vert_t)[:, None]
        d1 = _unit(d1)
        theta_v = np.zeros(n)
        theta_v[:-1] += 0.5 * self.theta
        theta_v[1:] += 0.5 * self.theta
        theta_v[0] = self.theta[0]
        theta_v[-1] = self.theta[-1]
        m1 = d1 * np.cos(theta_v)[:, None] + np.cross(vert_t, d1) * np.sin(
            theta_v
        )[:, None]
        m2 = np.cross(vert_t, m1)
        quat = self._quat_from_frame(vert_t, m1, m2)

        attrs = dict(self.scene_attributes)
        attrs["inertia_tensor"] = np.zeros((n, 3, 3))
        attrs["mass_quantity"] = self.mass.copy()
        attrs["rod_radius"] = np.full(n, self.radius)
        attrs["contact_diameter"] = np.full(n, self.contact_d0)
        attrs["arc_fraction"] = np.linspace(0.0, 1.0, n)
        attrs["endpoint_controlled"] = np.isin(
            np.arange(n), self.fixed_idx
        ).astype(float)
        attrs["material_twist_turns"] = theta_v / (2.0 * np.pi)
        attrs["twist_moment"] = np.full(n, self.tau)

        phi = self._clamp_angle(t_now)
        left = np.asarray(self.config["clamps"]["left"]["position"], dtype=float)
        right = np.asarray(self.config["clamps"]["right"]["position"], dtype=float)
        constraint_value = np.array(
            [x[0], x[-1], [phi / (2.0 * np.pi), 0.0, 0.0]], dtype=float
        )
        constraint_target = np.array(
            [left, right, [self._clamp_angle(t_now) / (2.0 * np.pi), 0.0, 0.0]],
            dtype=float,
        )

        fields = {
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_force": self._terms.reshape(1, 4, 3).copy(),
            "budget_force_position": self._term_pos.reshape(1, 4, 3).copy(),
            "budget_external_couple": self._term_couple.reshape(1, 4, 3).copy(),
            "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
            "budget_cumulative_external_angular_impulse":
                self.angular_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "cumulative_mass_source": np.zeros(1),
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
            "reference_angular_momentum": float(self.reference_angular_momentum),
            "twist_turns_applied": float(phi / (2.0 * np.pi)),
            "total_twist_radians": float(self.total_twist),
            "rod_length": float(self.total_length),
        }

        frame = StateFrame(
            float(max(t_now, 0.0)),
            x.copy(),
            v.copy(),
            orientation=quat,
            angular_velocity=omega,
            mass=self.mass.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    @staticmethod
    def _quat_from_frame(t: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Rotation matrices [m1, m2, t] as columns -> xyzw quaternions."""
        n = len(t)
        r = np.stack([m1, m2, t], axis=2)  # columns
        tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
        q = np.zeros((n, 4))
        ok = tr > 0.0
        s = np.zeros(n)
        s[ok] = np.sqrt(np.maximum(tr[ok] + 1.0, 1e-12)) * 2.0
        if np.any(ok):
            q[ok, 0] = (r[ok, 2, 1] - r[ok, 1, 2]) / s[ok]
            q[ok, 1] = (r[ok, 0, 2] - r[ok, 2, 0]) / s[ok]
            q[ok, 2] = (r[ok, 1, 0] - r[ok, 0, 1]) / s[ok]
            q[ok, 3] = 0.25 * s[ok]
        rest = np.flatnonzero(~ok)
        for i in rest:
            m = r[i]
            k = int(np.argmax(np.array([m[0, 0], m[1, 1], m[2, 2]])))
            if k == 0:
                sk = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
                q[i] = [0.25 * sk, (m[0, 1] + m[1, 0]) / sk,
                        (m[0, 2] + m[2, 0]) / sk, (m[2, 1] - m[1, 2]) / sk]
            elif k == 1:
                sk = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
                q[i] = [(m[0, 1] + m[1, 0]) / sk, 0.25 * sk,
                        (m[1, 2] + m[2, 1]) / sk, (m[0, 2] - m[2, 0]) / sk]
            else:
                sk = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
                q[i] = [(m[0, 2] + m[2, 0]) / sk, (m[1, 2] + m[2, 1]) / sk,
                        0.25 * sk, (m[1, 0] - m[0, 1]) / sk]
        nq = np.sqrt(np.einsum("ij,ij->i", q, q))
        bad = nq < 1e-9
        q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        nq[bad] = 1.0
        return q / nq[:, None]

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete elastic rod (isotropic, quasistatic twist)",
            "dt": float(self.dt),
            "steps": int(self._step_count),
            "vertices": int(self.n),
            "rod_length": float(self.total_length),
            "twist_moment": float(self.tau),
            "total_twist_radians": float(self.total_twist),
            "max_relative_stretch": float(self.max_stretch),
            "max_speed": float(self.max_speed),
            "active_contacts": int(self.contact_count),
            "bend_stiffness": float(self.BEND_STIFFNESS),
            "twist_stiffness": float(self.twist_beta),
        }
