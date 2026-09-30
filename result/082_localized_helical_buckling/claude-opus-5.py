"""Localized helical buckling of a twisted, clamped elastic rod.

Physical model
--------------
Discrete Kirchhoff (elastic-rod) model of a naturally straight, isotropic,
inextensible rod discretized into ``N-1`` straight segments:

* bending energy   ``sum_i B/(2 l_i) * |kappa_b,i|^2`` with the discrete
  curvature binormal magnitude ``|kappa_b| = 2 tan(phi/2)``,
* twist energy     ``beta/(2 L) * (Theta - Psi)^2``.  For an isotropic rod the
  material-frame rotation relaxes instantaneously so the twist rate is uniform;
  ``Theta`` is the clamp-prescribed relative material rotation and ``Psi`` is the
  accumulated holonomy (reference twist / writhe) of the space-parallel
  transported Bishop frame, integrated along the deformation path with the
  standard rod holonomy one-form ``grad Psi``,
* axial energy     ``sum_j EA/(2 h) (|e_j| - h)^2`` with ``EA = 4 B / r^2``,
  which enforces inextensibility (strains stay of order 1e-5),
* self-contact     smooth normal penalty between non-neighbouring centreline
  samples at a gap of one rod diameter.

Time stepping
-------------
The scene is quasi-static displacement control.  The clamps' prescribed
positions, tangents and relative material rotation are advanced in small
monotone increments (one increment per requested sample time), and at every
increment the rod is stepped to mechanical equilibrium of the frozen elastic
model (implicit static step, L-BFGS on the residual of the free degrees of
freedom, warm started from the previous increment).  Within one increment the
holonomy is linearized about the incoming configuration; after the increment it
is updated by the exact path integral of that linearization, which is the usual
explicit reference-twist update of discrete elastic rods.  The scheme is
deterministic and produces finite states at every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from worldsolver.contracts import StateFrame


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.config = cfg

        rod = cfg["rod"]
        self.L = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        # Frozen benchmark moduli (N*m^2).
        self.B = float(rod["bending_modulus_ratio"])
        self.beta = float(rod["twisting_modulus_ratio"])
        self.theta_final = float(rod["source_imposed_end_twist_rad"])

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.n = self.x.shape[0]
        self.mass = np.array(frame0.mass, dtype=float)
        self.attr0 = {k: np.asarray(v, dtype=float) for k, v in frame0.attributes.items()}

        self.h = self.L / (self.n - 1)
        # Axial stiffness of a circular cross-section consistent with B = E*I.
        self.EA = 4.0 * self.B / (self.radius ** 2)
        self.k_stretch = self.EA / self.h
        self.k_contact = 5.0e4
        self.contact_gap = 2.0 * self.radius
        idx = np.arange(self.n)
        self.contact_mask = np.abs(idx[:, None] - idx[None, :]) >= 4

        prot = cfg["loading_protocol"]
        self.t_twist0 = float(prot["twist_stage"]["start_time"])
        self.t_twist1 = float(prot["twist_stage"]["end_time"])
        self.theta_target_final = float(prot["twist_stage"]["final_relative_rotation_rad"])
        self.t_short0 = float(prot["shortening_stage"]["start_time"])
        self.t_short1 = float(prot["shortening_stage"]["end_time"])
        self.d_final = float(prot["shortening_stage"]["final_end_distance"])

        bc = cfg["boundary_conditions"]
        self.left_pos = np.array(bc["left_clamp"]["fixed_position"], dtype=float)
        self.left_tan = np.array(bc["left_clamp"]["tangent"], dtype=float)
        self.left_tan /= np.linalg.norm(self.left_tan)
        right_init = np.array(bc["right_clamp"]["initial_position"], dtype=float)
        self.right_tan = np.array(bc["right_clamp"]["tangent"], dtype=float)
        self.right_tan /= np.linalg.norm(self.right_tan)
        self.d_initial = float(np.linalg.norm(right_init - self.left_pos))

        grav = cfg.get("gravity", [0.0, 0.0, 0.0])
        self.gravity = np.array(grav, dtype=float)

        # Deterministic geometric imperfection of the rod (branch selection).  A
        # perfectly straight, perfectly symmetric rod is an exact equilibrium of the
        # discrete model, so buckling needs a physical imperfection; it is a smooth
        # low-mode transverse shape with zero displacement and slope at both clamps
        # and an amplitude frozen by the scene configuration.
        amp = float(cfg["initial_conditions"]["branch_selection_perturbation_amplitude"])
        rng = np.random.default_rng(self.seed)
        q = np.linspace(0.0, 1.0, self.n)
        window = np.sin(np.pi * q) ** 2
        imp = np.zeros((self.n, 3))
        for mode in (1, 2, 3):
            phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
            weight = 1.0 / mode
            imp[:, 1] += weight * window * np.sin(2.0 * np.pi * mode * q + phase[0])
            imp[:, 2] += weight * window * np.sin(2.0 * np.pi * mode * q + phase[1])
        scale = np.max(np.linalg.norm(imp, axis=1))
        self.imperfection = imp * (amp / max(scale, 1e-30))
        self.imperfection_amplitude = amp

        self.psi = 0.0                      # accumulated reference twist (holonomy)
        self.theta = 0.0                    # prescribed relative material rotation
        self.time = 0.0
        self.prev_x = self.x.copy()
        self.prev_time = 0.0
        self.velocity = np.zeros_like(self.x)
        self.x = self.x + self.imperfection
        self._apply_clamps(self.x, 0.0)
        self.x = self._project_lengths(self.x)
        self._apply_clamps(self.x, 0.0)
        self.prev_x = self.x.copy()
        self._iterations = 0
        self._frames = 0
        self._max_strain = 0.0

    # ------------------------------------------------------- loading protocol
    def _end_distance(self, t: float) -> float:
        if t <= self.t_short0:
            return self.d_initial
        u = (t - self.t_short0) / max(self.t_short1 - self.t_short0, 1e-12)
        return self.d_initial + (self.d_final - self.d_initial) * _smoothstep(u)

    def _end_twist(self, t: float) -> float:
        u = (t - self.t_twist0) / max(self.t_twist1 - self.t_twist0, 1e-12)
        return self.theta_target_final * _smoothstep(u)

    def _clamp_positions(self, t: float):
        d = self._end_distance(t)
        p0 = self.left_pos
        p1 = self.left_pos + d * self.right_tan
        return p0, p1

    def _apply_clamps(self, x: np.ndarray, t: float) -> None:
        p0, p1 = self._clamp_positions(t)
        x[0] = p0
        x[1] = p0 + self.h * self.left_tan
        x[-1] = p1
        x[-2] = p1 - self.h * self.right_tan

    # ---------------------------------------------------------- rod mechanics
    def _edges(self, x):
        e = x[1:] - x[:-1]
        l = np.linalg.norm(e, axis=1)
        l = np.maximum(l, 1e-12)
        t = e / l[:, None]
        return e, l, t

    def _holonomy_gradient(self, x):
        """Gradient of the accumulated reference twist (Bishop-frame holonomy)."""
        e, l, t = self._edges(x)
        e0, e1 = e[:-1], e[1:]
        l0, l1 = l[:-1], l[1:]
        cross = np.cross(e0, e1)
        denom = l0 * l1 + np.einsum("ij,ij->i", e0, e1)
        denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
        kb = 2.0 * cross / denom[:, None]
        g = np.zeros_like(x)
        gm = -kb / (2.0 * l0[:, None])
        gp = kb / (2.0 * l1[:, None])
        np.add.at(g, np.arange(0, self.n - 2), gm)
        np.add.at(g, np.arange(2, self.n), gp)
        np.add.at(g, np.arange(1, self.n - 1), -(gm + gp))
        return g

    def _energy_gradient(self, x, psi_ref, g_psi, x_ref, theta):
        e, l, t = self._edges(x)
        grad = np.zeros_like(x)
        energy = 0.0

        # --- bending: E = sum (2B/h) (1-c)/(1+c),  |kappa_b| = 2 tan(phi/2)
        t0, t1 = t[:-1], t[1:]
        c = np.einsum("ij,ij->i", t0, t1)
        c = np.clip(c, -0.985, 1.0)
        opc = 1.0 + c
        energy += float(np.sum((2.0 * self.B / self.h) * (1.0 - c) / opc))
        dEdc = -4.0 * self.B / (self.h * opc ** 2)
        g0 = (t1 - c[:, None] * t0) / l[:-1, None]
        g1 = (t0 - c[:, None] * t1) / l[1:, None]
        cm = dEdc[:, None] * g0
        cp = dEdc[:, None] * g1
        np.add.at(grad, np.arange(0, self.n - 2), -cm)
        np.add.at(grad, np.arange(2, self.n), cp)
        np.add.at(grad, np.arange(1, self.n - 1), cm - cp)

        # --- twist: uniform twist of an isotropic rod, holonomy linearized
        psi = psi_ref + float(np.einsum("ij,ij->", g_psi, x - x_ref))
        twist = theta - psi
        energy += 0.5 * self.beta / self.L * twist ** 2
        grad += -(self.beta / self.L) * twist * g_psi

        # --- axial (inextensibility penalty)
        ext = l - self.h
        energy += 0.5 * self.k_stretch * float(np.sum(ext ** 2))
        fs = (self.k_stretch * ext)[:, None] * t
        np.add.at(grad, np.arange(0, self.n - 1), -fs)
        np.add.at(grad, np.arange(1, self.n), fs)

        # --- self contact (smooth normal penalty at one rod diameter)
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-24)
        pen = np.where(self.contact_mask, self.contact_gap - dist, 0.0)
        active = pen > 0.0
        if np.any(active):
            pen = np.where(active, pen, 0.0)
            energy += 0.25 * self.k_contact * float(np.sum(pen ** 2))
            coef = np.where(active, -self.k_contact * pen / dist, 0.0)
            grad += np.einsum("ij,ijk->ik", coef, diff)
        return energy, grad

    def _project_lengths(self, x, passes: int = 60):
        """Red-black (vectorized) projection onto |e_j| = h, clamped vertices frozen."""
        movable = np.ones(self.n)
        movable[[0, 1, self.n - 2, self.n - 1]] = 0.0
        edges = np.arange(self.n - 1)
        groups = (edges[edges % 2 == 0], edges[edges % 2 == 1])
        for _ in range(passes):
            err = 0.0
            for a in groups:
                b = a + 1
                d = x[b] - x[a]
                ln = np.maximum(np.sqrt(np.sum(d * d, axis=1)), 1e-12)
                err = max(err, float(np.max(np.abs(ln - self.h))))
                wa = movable[a]
                wb = movable[b]
                wsum = wa + wb
                inv = np.where(wsum > 0.0, 1.0 / np.where(wsum > 0.0, wsum, 1.0), 0.0)
                shift = (((ln - self.h) / ln) * inv)[:, None] * d
                x[a] += wa[:, None] * shift
                x[b] -= wb[:, None] * shift
            if err < 1e-13:
                break
        return x

    def _solve_equilibrium(self, x0, psi_ref, g_psi, x_ref, theta):
        free = np.arange(2, self.n - 2)
        x = x0.copy()

        def pack(xx):
            return xx[free].reshape(-1)

        def unpack(vec):
            xx = x.copy()
            xx[free] = vec.reshape(-1, 3)
            return xx

        def fun(vec):
            xx = unpack(vec)
            e, g = self._energy_gradient(xx, psi_ref, g_psi, x_ref, theta)
            return e, g[free].reshape(-1)

        for outer in range(3):
            res = minimize(
                fun,
                pack(x),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 150, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10},
            )
            self._iterations += int(res.nit)
            cand = unpack(np.asarray(res.x, dtype=float))
            if not np.isfinite(cand).all():
                break
            x = cand
            if res.status == 0 and float(np.max(np.abs(res.jac))) < 1e-9:
                break
        self._apply_clamps(x, self.time)
        if not np.isfinite(x).all():
            x = x0.copy()
            self._apply_clamps(x, self.time)
        return x

    # ------------------------------------------------------------- stepping
    def _step_to(self, t_target: float, substeps: int) -> None:
        t0 = self.time
        for k in range(1, substeps + 1):
            t = t0 + (t_target - t0) * (k / substeps)
            self.time = t
            self.theta = self._end_twist(t)
            x_ref = self.x.copy()
            g_psi = self._holonomy_gradient(x_ref)
            x_new = self.x.copy()
            self._apply_clamps(x_new, t)
            x_new = self._solve_equilibrium(x_new, self.psi, g_psi, x_ref, self.theta)
            self.psi = self.psi + float(np.einsum("ij,ij->", g_psi, x_new - x_ref))
            self.x = x_new
        self.time = t_target

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        dt = time - self.prev_time
        if dt > 1e-12:
            substeps = 2 if time > self.t_short0 - 1e-9 else 1
            self._step_to(time, substeps)
            new_v = (self.x - self.prev_x) / dt
            accel = (new_v - self.velocity) / dt
            self.velocity = new_v
        else:
            self.time = time
            self.theta = self._end_twist(time)
            accel = np.zeros_like(self.x)
            self.velocity = np.zeros_like(self.x)
        self.prev_x = self.x.copy()
        self.prev_time = time
        self._frames += 1
        return self._make_frame(time, accel)

    # ---------------------------------------------------------- frame output
    def _make_frame(self, time: float, accel: np.ndarray) -> StateFrame:
        x = self.x
        g_psi = self._holonomy_gradient(x)
        _, grad = self._energy_gradient(x, self.psi, g_psi, x, self.theta)
        e, l, t = self._edges(x)
        self._max_strain = max(self._max_strain, float(np.max(np.abs(l - self.h)) / self.h))

        # Clamp reaction forces (boundary tractions) from the static residual.
        left_nodes = [0, 1]
        right_nodes = [self.n - 2, self.n - 1]
        r_nodes = grad + self.mass[:, None] * accel
        f_left = np.sum(r_nodes[left_nodes], axis=0)
        f_right = np.sum(r_nodes[right_nodes], axis=0)
        p0, p1 = self._clamp_positions(time)

        # Uniform twisting moment of the isotropic rod.
        m_twist = self.beta * (self.theta - self.psi) / self.L
        tan_l = (x[1] - x[0]) / max(np.linalg.norm(x[1] - x[0]), 1e-12)
        tan_r = (x[-1] - x[-2]) / max(np.linalg.norm(x[-1] - x[-2]), 1e-12)
        c_left = np.zeros(3)
        for i in left_nodes:
            c_left += np.cross(x[i] - p0, r_nodes[i])
        c_left -= m_twist * tan_l
        c_right = np.zeros(3)
        for i in right_nodes:
            c_right += np.cross(x[i] - p1, r_nodes[i])
        c_right += m_twist * tan_r

        total_mass = float(np.sum(self.mass))
        com = np.sum(self.mass[:, None] * x, axis=0) / total_mass
        f_gravity = total_mass * self.gravity

        budget_force = np.zeros((1, 3, 3))
        budget_force[0, 0] = f_left
        budget_force[0, 1] = f_right
        budget_force[0, 2] = f_gravity
        budget_pos = np.zeros((1, 3, 3))
        budget_pos[0, 0] = p0
        budget_pos[0, 1] = p1
        budget_pos[0, 2] = com
        budget_couple = np.zeros((1, 3, 3))
        budget_couple[0, 0] = c_left
        budget_couple[0, 1] = c_right

        # Prescribed kinematic / inextensibility constraints.
        d_now = float(np.linalg.norm(x[-1] - x[0]))
        constraint_value = np.zeros((7, 3))
        constraint_target = np.zeros((7, 3))
        constraint_value[0] = x[0]
        constraint_target[0] = p0
        constraint_value[1] = tan_l
        constraint_target[1] = self.left_tan
        constraint_value[2] = x[-1]
        constraint_target[2] = p1
        constraint_value[3] = tan_r
        constraint_target[3] = self.right_tan
        constraint_value[4, 0] = d_now
        constraint_target[4, 0] = self._end_distance(time)
        constraint_value[5, 0] = self.theta
        constraint_target[5, 0] = self._end_twist(time)
        constraint_value[6] = [float(np.min(l)), float(np.mean(l)), float(np.max(l))]
        constraint_target[6] = [self.h, self.h, self.h]

        attributes = {
            "arc_length": self.attr0["arc_length"],
            "natural_curvature": self.attr0["natural_curvature"],
            "natural_twist": self.attr0["natural_twist"],
            "endpoint_clamp": self.attr0["endpoint_clamp"],
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "budget_external_force": budget_force,
            "budget_force_position": budget_pos,
            "budget_external_couple": budget_couple,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "rod_radius": self.radius,
            "target_relative_twist_rad": self.theta_final,
            "relative_material_rotation_rad": float(self.theta),
            "reference_twist_rad": float(self.psi),
            "twisting_moment": float(m_twist),
            "end_distance": d_now,
            "constraint_scale": float(self.L),
            "reference_force": float(self.B / self.L ** 2),
            "reference_torque": float(self.B / self.L),
        }
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(x, dtype=float),
            velocity=np.ascontiguousarray(self.velocity, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete elastic rod, quasi-static displacement control",
            "vertices": int(self.n),
            "frames": int(self._frames),
            "lbfgs_iterations": int(self._iterations),
            "max_relative_axial_strain": float(self._max_strain),
            "final_reference_twist_rad": float(self.psi),
        }
