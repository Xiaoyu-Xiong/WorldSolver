"""Discrete elastic rod solver for Michell buckling of a pre-twisted closed ring.

Physical model (Bergou et al., Discrete Elastic Rods, in spirit, implemented from
scratch with NumPy):

  * Centerline: N vertices of a closed loop, x_i, with edges e^j = x_{j+1} - x_j
    (cyclic).  Point masses m_i carry the translational inertia.
  * Material frame: one scalar angle theta^j per edge measured from a reference
    (Bishop / parallel-transported) director a1^j.  Each edge carries the polar
    moment of inertia J so the twist degrees of freedom are dynamic.
  * Elastic energy of a naturally straight isotropic rod:
        E_bend  = sum_i (alpha / (2 lbar)) |kb_i|^2
        E_twist = sum_i (beta  / (2 lbar)) m_i^2
        E_stretch = sum_j (ks/2) (l_j - l0)^2      (near-inextensibility)
    with the discrete curvature binormal
        kb_i = 2 e^{i-1} x e^i / (|e^{i-1}||e^i| + e^{i-1}.e^i)
    and the discrete twist
        m_i = theta^i - theta^{i-1} + refTwist_i (+ 2*pi*Lk at the closure seam).
  * The reference directors are time-parallel-transported, so refTwist_i is the
    holonomy of the Bishop frame.  Its position gradient is the standard
        d m_i / d x_{i-1} = -kb_i / (2 l_{i-1}),
        d m_i / d x_{i+1} = +kb_i / (2 l_i),
        d m_i / d x_i     = -(sum of the two),
    which is exactly the geometric twist/writhe coupling that makes an
    over-twisted closed ring (Lk = 3, total twist 6*pi > 2*pi*sqrt(3)) unstable:
    the Michell instability.  Nothing about the buckled shape is prescribed; the
    only seed is the small mode-2 out-of-plane perturbation of the frozen scene.
  * Self-contact: penalty repulsion with normal dissipation between rod points
    closer than one rod diameter (excluding near-neighbours along arclength).
  * Time integration: symplectic (semi-implicit) Euler at a fixed substep with
    light viscous dissipation so the released ring relaxes into a persistent
    non-planar shape.  No external loads (gravity is zero, ring is free).
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _cross(a, b):
    return np.cross(a, b)


def _norm(a):
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def _parallel_transport(v, t0, t1):
    """Rotate v by the minimal rotation taking unit t0 onto unit t1 (row-wise)."""
    b = _cross(t0, t1)
    c = 1.0 + np.einsum("ij,ij->i", t0, t1)
    safe = c > 1.0e-10
    cs = np.where(safe, c, 1.0)[:, None]
    out = v + _cross(b, v) + _cross(b, _cross(b, v)) / cs
    return np.where(safe[:, None], out, v)


def _orthonormalize(v, t):
    v = v - np.einsum("ij,ij->i", v, t)[:, None] * t
    n = _norm(v)
    bad = n < 1.0e-9
    if np.any(bad):
        helper = np.zeros_like(v)
        helper[:, 2] = 1.0
        alt = helper - np.einsum("ij,ij->i", helper, t)[:, None] * t
        an = _norm(alt)
        deg = an < 1.0e-9
        if np.any(deg):
            alt[deg] = np.array([1.0, 0.0, 0.0])
            alt[deg] -= np.einsum("ij,ij->i", alt[deg], t[deg])[:, None] * t[deg]
            an = _norm(alt)
        v = np.where(bad[:, None], alt, v)
        n = np.where(bad, an, n)
    return v / n[:, None]


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class BenchmarkSolver:
    """Dynamic discrete elastic rod, closed loop, released from a compatible preload."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        self.config = config

        rod = config["rod"]
        initial = config["initial_conditions"]

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.m = np.array(frame0.mass, dtype=float)
        self.n = int(self.x.shape[0])
        n = self.n

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        self.rod_radius = float(rod["cross_section_radius"])
        self.density = float(rod["density"])
        self.total_length = float(rod["length"])
        self.l0 = self.total_length / n
        self.area = np.pi * self.rod_radius ** 2
        self.line_mass = self.density * self.area

        # Bending / twisting moduli of a naturally straight isotropic rod.
        self.alpha = float(rod.get("bending_stiffness_alpha", 1.0))
        self.beta = float(rod.get("twisting_stiffness_beta", 1.0))
        # Near-inextensibility: axial stiffness consistent with a circular section
        # (EA = 4*EI/r^2) so the rod stretches by only a tiny fraction.
        self.ks = 4.0 * self.alpha / self.rod_radius ** 2

        # Twist degrees of freedom: one material angle per edge.
        polar_moment = 0.5 * np.pi * self.rod_radius ** 4
        self.jj = self.density * polar_moment * self.l0
        self.theta = np.zeros(n, dtype=float)
        self.theta_dot = np.zeros(n, dtype=float)

        winding = int(initial["material_frame_winding_number"])
        stored = float(initial["stored_total_twist_rad"])
        self.stored_twist = stored
        # Single-valued angle representation of a closed material frame: the
        # 2*pi*Lk jump lives at one seam vertex and is a topological constant.
        self.seam = np.zeros(n, dtype=float)
        self.seam[0] = 2.0 * np.pi * winding
        self.theta[:] = (stored / n) * np.arange(n, dtype=float)

        # Internal (momentum-conserving) dashpots: all damping forces act along
        # the line joining the interacting pair, so linear and angular momentum
        # remain conserved and the external budget stays exactly zero.
        self.c_axial = 25.0
        self.c_chord = 5.0
        self.c_twist = 0.02

        # Self contact (penalty, normal direction only).
        self.contact_distance = 2.0 * self.rod_radius
        self.contact_stiffness = 2.0e3
        self.contact_damping = 5.0
        self.contact_exclude = 4
        index = np.arange(n)
        separation = np.abs(index[:, None] - index[None, :])
        cyclic = np.minimum(separation, n - separation)
        self._contact_mask = cyclic > self.contact_exclude

        self.dt = 5.0e-5
        self.time = 0.0
        self.substeps = 0

        # Reference (Bishop) directors and the holonomy they accumulate.
        e = np.roll(self.x, -1, axis=0) - self.x
        t = e / _norm(e)[:, None]
        d1 = np.zeros_like(e)
        seed_vec = np.array([0.0, 0.0, 1.0])
        d1[0] = _orthonormalize(seed_vec[None, :], t[0][None, :])[0]
        for j in range(1, n):
            transported = _parallel_transport(
                d1[j - 1][None, :], t[j - 1][None, :], t[j][None, :]
            )
            d1[j] = _orthonormalize(transported, t[j][None, :])[0]
        self.d1 = d1
        self.ref_twist = np.zeros(n, dtype=float)
        self.ref_twist = self._reference_twist(t, d1, self.ref_twist)

        total_mass = float(self.m.sum())
        v_char = np.sqrt(self.alpha / self.line_mass) / max(float(rod["radius"]), _EPS)
        self.reference_momentum = max(total_mass * v_char, 1.0e-6)
        self.reference_angular_momentum = max(
            total_mass * v_char * float(rod["radius"]), 1.0e-6
        )
        self.max_speed = 0.0
        self.max_out_of_plane = float(np.max(np.abs(self.x[:, 2])))

        # The twist force is driven by the position gradient of the Bishop-frame
        # holonomy (equivalently of the writhe).  Verify the orientation of the
        # analytic curvature-binormal gradient once, against a central finite
        # difference of the very same holonomy routine the time loop uses.
        self.twist_sign = 1.0
        self.twist_gradient_correlation = float("nan")
        self._calibrate_twist_sign()

    def _holonomy_sum(self, x, d1_reference, t_reference, previous):
        """Total reference twist after time-parallel transport onto perturbed x."""
        _, _, t = self._edge_geometry(x)
        moved = _orthonormalize(_parallel_transport(d1_reference, t_reference, t), t)
        return float(self._reference_twist(t, moved, previous).sum())

    def _calibrate_twist_sign(self) -> None:
        n = self.n
        radius = float(self.config["rod"]["radius"])
        phase = 2.0 * np.pi * np.arange(n) / n
        # A deliberately writhing probe shape: the planar circle has a vanishing
        # writhe gradient, so it cannot reveal the orientation.
        breathe = radius * (1.0 + 0.05 * np.cos(2.0 * phase))
        probe = np.column_stack((
            breathe * np.cos(phase),
            breathe * np.sin(phase),
            0.25 * np.sin(2.0 * phase) + 0.08 * np.sin(3.0 * phase),
        ))
        e, l, t = self._edge_geometry(probe)
        d1 = np.zeros_like(t)
        d1[0] = _orthonormalize(np.array([[0.0, 0.0, 1.0]]), t[0][None, :])[0]
        for j in range(1, n):
            d1[j] = _orthonormalize(
                _parallel_transport(d1[j - 1][None, :], t[j - 1][None, :], t[j][None, :]),
                t[j][None, :],
            )[0]
        base = self._reference_twist(t, d1, np.zeros(n))

        kb = self._curvature_binormal(e, l, t)
        l_prev = np.roll(l, 1, axis=0)
        p_term = -0.5 * kb / l_prev[:, None]
        q_term = 0.5 * kb / l[:, None]
        r_term = -(p_term + q_term)
        analytic = np.roll(p_term, -1, axis=0) + np.roll(q_term, 1, axis=0) + r_term

        step = 1.0e-6
        probes = np.arange(0, n, max(1, n // 8))
        numeric = np.zeros((probes.size, 3), dtype=float)
        for row, vertex in enumerate(probes):
            for axis in range(3):
                forward = probe.copy()
                forward[vertex, axis] += step
                backward = probe.copy()
                backward[vertex, axis] -= step
                numeric[row, axis] = (
                    self._holonomy_sum(forward, d1, t, base)
                    - self._holonomy_sum(backward, d1, t, base)
                ) / (2.0 * step)

        reference = analytic[probes].ravel()
        measured = numeric.ravel()
        scale = np.linalg.norm(reference) * np.linalg.norm(measured)
        if scale > 1.0e-12:
            correlation = float(reference @ measured) / scale
            self.twist_gradient_correlation = correlation
            if correlation < -0.5:
                self.twist_sign = -1.0

    # -------------------------------------------------------------- geometry
    def _reference_twist(self, t, d1, previous):
        """Holonomy of the parallel-transported reference frame at each vertex."""
        t_prev = np.roll(t, 1, axis=0)
        d_prev = np.roll(d1, 1, axis=0)
        transported = _parallel_transport(d_prev, t_prev, t)
        transported = _orthonormalize(transported, t)
        sin_part = np.einsum("ij,ij->i", _cross(transported, d1), t)
        cos_part = np.einsum("ij,ij->i", transported, d1)
        angle = np.arctan2(sin_part, cos_part)
        return previous + _wrap_pi(angle - previous)

    def _update_reference_frames(self, t_old, t_new):
        moved = _parallel_transport(self.d1, t_old, t_new)
        self.d1 = _orthonormalize(moved, t_new)

    def _edge_geometry(self, x):
        e = np.roll(x, -1, axis=0) - x
        l = _norm(e)
        l = np.maximum(l, _EPS)
        t = e / l[:, None]
        return e, l, t

    def _curvature_binormal(self, e, l, t):
        e_prev = np.roll(e, 1, axis=0)
        l_prev = np.roll(l, 1, axis=0)
        denominator = l_prev * l + np.einsum("ij,ij->i", e_prev, e)
        denominator = np.where(np.abs(denominator) < 1.0e-10, 1.0e-10, denominator)
        return 2.0 * _cross(e_prev, e) / denominator[:, None]

    def _contact_pairs(self, x):
        delta = x[:, None, :] - x[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        candidate = self._contact_mask & (squared < self.contact_distance ** 2)
        return np.nonzero(candidate)

    # ---------------------------------------------------------------- forces
    def _discrete_twist(self, ref_twist, theta):
        return theta - np.roll(theta, 1) + ref_twist + self.seam

    def _forces(self, x, v, theta, theta_dot, e, l, t, ref_twist):
        n = self.n
        lbar = self.l0
        force = np.zeros((n, 3), dtype=float)

        # --- bending: E = sum_i (alpha / 2 lbar) |t_i - t_{i-1}|^2 (naturally straight)
        laplacian = 2.0 * t - np.roll(t, 1, axis=0) - np.roll(t, -1, axis=0)
        gradient_t = (self.alpha / lbar) * laplacian
        projected = gradient_t - np.einsum("ij,ij->i", gradient_t, t)[:, None] * t
        g = projected / l[:, None]
        force += g - np.roll(g, 1, axis=0)

        # --- twist: E = sum_i (beta / 2 lbar) m_i^2 with m_i the discrete twist
        m = self._discrete_twist(ref_twist, theta)
        w = (self.beta / lbar) * m
        kb = self._curvature_binormal(e, l, t)
        l_prev = np.roll(l, 1, axis=0)
        p_term = -0.5 * kb / l_prev[:, None]
        q_term = 0.5 * kb / l[:, None]
        r_term = -(p_term + q_term)
        weight = self.twist_sign * w
        a_term = weight[:, None] * p_term
        b_term = weight[:, None] * q_term
        c_term = weight[:, None] * r_term
        force -= np.roll(a_term, -1, axis=0) + np.roll(b_term, 1, axis=0) + c_term
        torque = np.roll(w, -1) - w

        # --- near-inextensible axial springs with along-edge dissipation
        stretch = (self.ks / lbar) * (l - lbar)
        relative = np.roll(v, -1, axis=0) - v
        axial_rate = np.einsum("ij,ij->i", relative, t)
        edge_force = (stretch + self.c_axial * axial_rate)[:, None] * t
        force += edge_force - np.roll(edge_force, 1, axis=0)

        # --- second-neighbour dashpot: damps bending waves, acts along the chord
        chord = np.roll(x, -2, axis=0) - x
        chord_length = np.maximum(_norm(chord), _EPS)
        chord_dir = chord / chord_length[:, None]
        chord_rate = np.einsum(
            "ij,ij->i", np.roll(v, -2, axis=0) - v, chord_dir
        )
        chord_force = (self.c_chord * chord_rate)[:, None] * chord_dir
        force += chord_force - np.roll(chord_force, 2, axis=0)

        # --- self contact (penalty along the pair normal only)
        rows, columns = self._contact_pairs(x)
        if rows.size:
            delta = x[columns] - x[rows]
            distance = np.maximum(_norm(delta), _EPS)
            normal = delta / distance[:, None]
            penetration = self.contact_distance - distance
            approach = np.einsum("ij,ij->i", v[columns] - v[rows], normal)
            magnitude = self.contact_stiffness * penetration - self.contact_damping * approach
            magnitude = np.maximum(magnitude, 0.0)
            impulse = magnitude[:, None] * normal
            np.add.at(force, rows, -impulse)
            np.add.at(force, columns, impulse)

        # --- twist dissipation between neighbouring edges
        torque += self.c_twist * (
            np.roll(theta_dot, -1) + np.roll(theta_dot, 1) - 2.0 * theta_dot
        )
        return force, torque, m, kb

    # ------------------------------------------------------------- integrate
    def _step(self, dt):
        e, l, t = self._edge_geometry(self.x)
        force, torque, _, _ = self._forces(
            self.x, self.v, self.theta, self.theta_dot, e, l, t, self.ref_twist
        )
        # Gravity is zero for this free-relaxation scene, but honour the config.
        if np.any(self.gravity):
            force += self.m[:, None] * self.gravity[None, :]

        self.v += dt * force / self.m[:, None]
        self.theta_dot += dt * torque / self.jj
        self.x += dt * self.v
        self.theta += dt * self.theta_dot

        e_new, l_new, t_new = self._edge_geometry(self.x)
        self._update_reference_frames(t, t_new)
        self.ref_twist = self._reference_twist(t_new, self.d1, self.ref_twist)
        self.substeps += 1
        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all():
            raise FloatingPointError("rod state diverged")

    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1.0e-12:
            steps = int(round((target - self.time) / self.dt))
            steps = max(steps, 1)
            step_dt = (target - self.time) / steps
            for _ in range(steps):
                self._step(step_dt)
        self.time = target
        return self._frame(target)

    # ----------------------------------------------------------------- output
    def _frame(self, time):
        n = self.n
        x = self.x
        v = self.v
        e, l, t = self._edge_geometry(x)
        m = self._discrete_twist(self.ref_twist, self.theta)

        # Per-vertex tangent and its rate, from the two incident edges.
        t_prev = np.roll(t, 1, axis=0)
        raw = t + t_prev
        raw_length = np.maximum(_norm(raw), _EPS)
        tangent = raw / raw_length[:, None]

        e_prev = np.roll(e, 1, axis=0)
        l_prev = np.roll(l, 1, axis=0)
        de = np.roll(v, -1, axis=0) - v
        de_prev = np.roll(v, 1, axis=0) * -1.0 + v
        dt_edge = (de - np.einsum("ij,ij->i", de, t)[:, None] * t) / l[:, None]
        dt_prev = (
            de_prev - np.einsum("ij,ij->i", de_prev, t_prev)[:, None] * t_prev
        ) / l_prev[:, None]
        raw_rate = dt_edge + dt_prev
        tangent_rate = raw_rate - np.einsum("ij,ij->i", raw_rate, tangent)[:, None] * tangent
        tangent_rate /= raw_length[:, None]

        spin = 0.5 * (self.theta_dot + np.roll(self.theta_dot, 1))
        angular_velocity = _cross(tangent, tangent_rate) + spin[:, None] * tangent

        polar_moment = 0.5 * np.pi * self.rod_radius ** 4
        inertia_coefficient = self.density * polar_moment * self.l0
        inertia = inertia_coefficient * np.einsum("ij,ik->ijk", tangent, tangent)

        total_twist = float(m.sum())
        speed = float(np.max(_norm(v))) if n else 0.0
        self.max_speed = max(self.max_speed, speed)
        self.max_out_of_plane = max(self.max_out_of_plane, float(np.max(np.abs(x[:, 2]))))

        zeros_g3 = np.zeros((1, 3), dtype=float)
        center_of_mass = (self.m[:, None] * x).sum(axis=0) / self.m.sum()

        frame = StateFrame(
            time=float(time),
            position=np.array(x, dtype=float),
            velocity=np.array(v, dtype=float),
            angular_velocity=np.array(angular_velocity, dtype=float),
            mass=np.array(self.m, dtype=float),
            attributes={
                "inertia_tensor": inertia,
                "mass_quantity": np.array(self.m, dtype=float),
                "material_frame_angle": np.array(self.theta, dtype=float),
                "discrete_twist": np.array(m, dtype=float),
                "stored_twist_density": np.array(m / self.l0, dtype=float),
                "arc_length": self.l0 * np.arange(n, dtype=float),
                "closed_curve_index": np.arange(n, dtype=float),
                "edge_length": np.array(l, dtype=float),
            },
            fields={
                "budget_domain_weights": np.ones((1, n), dtype=float),
                "budget_external_force": np.zeros((1, 1, 3), dtype=float),
                "budget_force_position": center_of_mass.reshape(1, 1, 3).copy(),
                "budget_external_couple": np.zeros((1, 1, 3), dtype=float),
                "budget_cumulative_external_impulse": zeros_g3.copy(),
                "budget_cumulative_external_angular_impulse": zeros_g3.copy(),
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3.copy(),
                "budget_cumulative_inflow_angular_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_angular_momentum": zeros_g3.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "constraint_value": np.array(l, dtype=float),
                "constraint_target": np.full(n, self.l0, dtype=float),
                "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            },
            scalars={
                "constraint_scale": float(self.l0),
                "reference_momentum": float(self.reference_momentum),
                "reference_angular_momentum": float(self.reference_angular_momentum),
                "rod_radius": float(self.rod_radius),
                "total_twist_rad": total_twist,
                "closed_ring": 1.0,
                "material_frame_winding_number": float(
                    self.config["initial_conditions"]["material_frame_winding_number"]
                ),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        e, l, _ = self._edge_geometry(self.x)
        m = self._discrete_twist(self.ref_twist, self.theta)
        return {
            "method": "discrete_elastic_rod_symplectic_euler",
            "substeps": int(self.substeps),
            "substep_dt": float(self.dt),
            "vertex_count": int(self.n),
            "max_edge_strain": float(np.max(np.abs(l / self.l0 - 1.0))),
            "total_twist_rad": float(m.sum()),
            "max_out_of_plane": float(self.max_out_of_plane),
            "max_speed": float(self.max_speed),
        }
