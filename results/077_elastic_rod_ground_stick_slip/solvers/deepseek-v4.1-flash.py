"""Explicit-dynamics finite-element solver for stick-slip of an elastic rod.

Physical model
--------------
The frozen scene mesh (``assets/rod_surface.obj``) is a hexagonal-prism
discretisation of a thin volumetric rod (10 rings x 6 sectors = 60 vertices,
height 1.45 m, radius 0.055 m).  Those 60 frozen vertices are used directly as
finite-element nodes; the rod volume is filled with 108 linear tetrahedra
(each of the 9 layers is split into 6 triangular prisms, each prism into three
tetrahedra), so every node carries real volumetric elastic stress.

Material: compressible neo-Hookean solid in a total-Lagrangian formulation,
    psi = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2,
    P   = mu (F - F^-T) + lambda ln J F^-T,
with F measured against the frozen initial configuration, so the exported
element deformation gradients and stresses are physical.

Dynamics: lumped-mass explicit central-difference (symplectic Euler) stepping
with a fixed substep chosen from the elastic wave speed, the smallest element
edge and the contact stiffness, keeping the scheme well inside its stability
limit.  A small, momentum-neutral element-local viscous damping removes
resolved-scale element chatter.

Contact: the rod is swept by node-centred spheres of the frozen rod radius
(0.055 m), which makes the frozen initial configuration exactly touching the
frozen ground plane (top surface z = 0).  Normal contact is a penalty spring
plus dashpot; dry friction is a regularised Coulomb law with a tangential
spring projected onto the friction cone (static = dynamic = 0.35).  The lower
end therefore sticks, the rod bows, and the base slips forward in repeating
cycles.

Boundary condition: vertices selected by the frozen ``driven_top`` attribute
follow the prescribed press (0-0.8 s, -0.05 m in z) and drag (0.8-8.0 s,
0.22 m/s in x) motion.  The reaction removed from the driven nodes by the
prescribed motion is measured from the unrestricted momentum update and
exported as an explicit external force term, so the reported discrete momentum
budget closes exactly, with zero inflow/outflow and zero mass source.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


class BenchmarkSolver:
    """Deterministic explicit-FEM stick-slip solver."""

    YOUNGS_MODULUS = 3.0e4      # Pa (rod buckles under the prescribed press)
    POISSON_RATIO = 0.30
    NORMAL_STIFFNESS = 3000.0   # N/m penalty spring (ground contact)
    NORMAL_DASHPOT = 0.30       # N s/m contact dashpot
    TANGENTIAL_STIFFNESS = 3000.0   # N/m regularised Coulomb friction spring
    ELEMENT_DAMPING = 3.0       # 1/s, element-local, momentum neutral
    STABILITY_SAFETY = 0.20
    MIN_SUBSTEPS = 32
    MAX_SUBSTEPS = 4000

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.scene = scene
        frame = scene.initial_state()
        config = getattr(scene, "config", {}) or {}

        x0 = np.asarray(frame.position, dtype=float).copy()
        self.n_nodes = int(x0.shape[0])
        self.reference_position = x0.copy()
        declared_mass = np.asarray(
            frame.mass if frame.mass is not None else np.full(self.n_nodes, 0.018 / self.n_nodes),
            dtype=float,
        )
        self.mass_total = float(np.sum(declared_mass))

        self.scalars_scene = dict(frame.scalars)
        self.radius = float(self.scalars_scene.get("rod_radius", 0.055))
        height = float(self.scalars_scene.get("rod_height", 1.45))

        young = self.YOUNGS_MODULUS
        poisson = self.POISSON_RATIO
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        contact_cfg = config.get("contact", {}) or {}
        self.mu_friction = float(contact_cfg.get("ground_friction_coefficient", 0.35))
        bc = (config.get("boundary_conditions", {}) or {}).get("driven_top", {})
        self.press_interval = np.asarray(bc.get("press_interval", [0.0, 0.8]), dtype=float)
        self.press_vector = np.asarray(
            bc.get("press_displacement", [0.0, 0.0, -0.05]), dtype=float
        )
        self.drag_interval = np.asarray(bc.get("drag_interval", [0.8, 8.0]), dtype=float)
        self.drag_velocity = np.asarray(bc.get("drag_velocity", [0.22, 0.0, 0.0]), dtype=float)

        self.tets = self._build_tets()
        self.n_elem = int(self.tets.shape[0])
        self.vol0, self.dminv = self._reference_data(x0)

        density = self.mass_total / float(np.sum(self.vol0))
        self.density = density
        nodal = np.zeros(self.n_nodes)
        share = density * self.vol0 / 4.0
        for local in range(4):
            np.add.at(nodal, self.tets[:, local], share)
        self.mass = np.maximum(nodal, 1e-12)

        driven_attr = frame.attributes.get("driven_top")
        if driven_attr is None:
            driven = x0[:, 2] > (height - 1e-9)
        else:
            driven = np.asarray(driven_attr, dtype=float).reshape(-1) > 0.5
        self.driven = driven.astype(bool)
        self.driven_position = x0[self.driven].copy()

        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.s_tan = np.zeros_like(x0)
        self.step_index = 0
        self.time = 0.0
        self.slip_events = 0
        self._was_sliding = False
        self.external_terms = np.zeros((4, 3))
        self.cum_impulse = np.zeros(3)

        sample_dt = float(config.get("sample_dt", 0.02))
        self.sample_dt = sample_dt
        wave_speed = float(np.sqrt((self.lam + 2.0 * self.mu) / max(density, 1e-12)))
        h_min = self._min_edge_length(x0)
        dt_elastic = self.STABILITY_SAFETY * h_min / max(wave_speed, 1e-12)
        m_min = float(np.min(self.mass))
        dt_contact = self.STABILITY_SAFETY * np.sqrt(
            m_min / (self.NORMAL_STIFFNESS + 2.0 * self.TANGENTIAL_STIFFNESS)
        )
        dt_limit = max(min(dt_elastic, dt_contact), 1e-9)
        n_sub = int(np.ceil(sample_dt / dt_limit))
        self.n_sub = int(min(max(n_sub, self.MIN_SUBSTEPS), self.MAX_SUBSTEPS))
        self.dt = sample_dt / float(self.n_sub)

        area = float(np.sum(self.vol0)) / max(height, 1e-9)
        press_mag = float(abs(self.press_vector[2]))
        self.reference_force = float(max(young * area * press_mag / max(height, 1e-9), 1e-6))
        drag_speed = float(np.linalg.norm(self.drag_velocity))
        self.reference_velocity = float(max(drag_speed, 1e-9))
        self.reference_momentum = float(max(self.mass_total * drag_speed, 1e-12))
        self.stick_speed_tolerance = float(max(0.05 * drag_speed, 1e-4))

        self.cum_inflow = np.zeros(3)
        self.cum_outflow = np.zeros(3)
    # ------------------------------------------------------------- mesh tools
    @staticmethod
    def _build_tets() -> np.ndarray:
        rings, sides = 10, 6
        fan = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5))
        tets = []
        for k in range(rings - 1):
            for i, j, l in fan:
                b0, b1, b2 = sides * k + i, sides * k + j, sides * k + l
                t0, t1, t2 = sides * (k + 1) + i, sides * (k + 1) + j, sides * (k + 1) + l
                tets.append([b0, b1, b2, t2])
                tets.append([b0, b1, t2, t1])
                tets.append([b0, t1, t2, t0])
        return np.asarray(tets, dtype=np.int64)

    def _reference_data(self, x0: np.ndarray):
        tets = self.tets.copy()
        for _ in range(3):
            dm = np.stack(
                [
                    x0[tets[:, 1]] - x0[tets[:, 0]],
                    x0[tets[:, 2]] - x0[tets[:, 0]],
                    x0[tets[:, 3]] - x0[tets[:, 0]],
                ],
                axis=2,
            )
            bad = np.linalg.det(dm) < 0.0
            if not np.any(bad):
                break
            swap = tets[bad, 1].copy()
            tets[bad, 1] = tets[bad, 2]
            tets[bad, 2] = swap
        dm = np.stack(
            [
                x0[tets[:, 1]] - x0[tets[:, 0]],
                x0[tets[:, 2]] - x0[tets[:, 0]],
                x0[tets[:, 3]] - x0[tets[:, 0]],
            ],
            axis=2,
        )
        vol0 = np.maximum(np.abs(np.linalg.det(dm)) / 6.0, 1e-14)
        inv = np.linalg.inv(dm)
        self.tets = tets
        return vol0, inv

    def _min_edge_length(self, x0: np.ndarray) -> float:
        t = self.tets
        best = np.inf
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            d = x0[t[:, a]] - x0[t[:, b]]
            best = min(best, float(np.min(np.sqrt(np.einsum("ij,ij->i", d, d)))))
        return float(best)

    # ---------------------------------------------------------- physics terms
    def _deformation(self, x: np.ndarray) -> np.ndarray:
        t = self.tets
        ds = np.stack(
            [
                x[t[:, 1]] - x[t[:, 0]],
                x[t[:, 2]] - x[t[:, 0]],
                x[t[:, 3]] - x[t[:, 0]],
            ],
            axis=2,
        )
        return ds @ self.dminv

    def _elastic_force(self, x: np.ndarray):
        t = self.tets
        deform = self._deformation(x)
        det = np.linalg.det(deform)
        bad = np.abs(det) < 1e-9
        if bool(np.any(bad)):
            patched = deform.copy()
            patched[bad] += 1e-6 * np.eye(3)
            deform = patched
            det = np.linalg.det(deform)
        jac = np.maximum(det, 1e-8)
        inv = np.linalg.inv(deform)
        inv_t = np.transpose(inv, (0, 2, 1))
        ln_j = np.log(jac)
        piola = self.mu * (deform - inv_t) + self.lam * ln_j[:, None, None] * inv_t
        mat = piola @ self.dminv  # column a -> local node a (a = 1, 2, 3)
        fe = np.zeros((self.n_elem, 4, 3))
        fe[:, 1:, :] = -self.vol0[:, None, None] * np.transpose(mat, (0, 2, 1))
        fe[:, 0, :] = -np.sum(fe[:, 1:, :], axis=1)
        force = np.zeros((self.n_nodes, 3))
        for local in range(4):
            idx = t[:, local]
            for comp in range(3):
                force[:, comp] += np.bincount(
                    idx, weights=fe[:, local, comp], minlength=self.n_nodes
                )
        return force, piola

    def _damping_force(self, v: np.ndarray) -> np.ndarray:
        t = self.tets
        ve = v[t].mean(axis=1)
        rel = v[t] - ve[:, None, :]
        share = (self.density * self.vol0 / 4.0)[:, None, None]
        fe = -self.ELEMENT_DAMPING * share * rel
        force = np.zeros((self.n_nodes, 3))
        for local in range(4):
            idx = t[:, local]
            for comp in range(3):
                force[:, comp] += np.bincount(
                    idx, weights=fe[:, local, comp], minlength=self.n_nodes
                )
        return force

    def _contact_force(self, x, v, s):
        pen = self.radius - x[:, 2]
        active = pen > 0.0
        normal = np.where(active, self.NORMAL_STIFFNESS * pen, 0.0)
        ft = self.TANGENTIAL_STIFFNESS * s
        mag = np.sqrt(np.einsum("ij,ij->i", ft, ft))
        limit = self.mu_friction * normal
        scale = np.ones(self.n_nodes)
        sliding = mag > limit
        scale[sliding] = limit[sliding] / np.maximum(mag[sliding], 1e-30)
        ft = ft * scale[:, None]
        force = ft.copy()
        fn = normal - self.NORMAL_DASHPOT * v[:, 2]
        force[:, 2] += np.where(active, np.maximum(fn, 0.0), 0.0)
        return force, normal, ft, active, sliding

    # ------------------------------------------------------ prescribed motion
    def _offset(self, time: float) -> np.ndarray:
        out = np.zeros(3)
        p0, p1 = float(self.press_interval[0]), float(self.press_interval[1])
        if p1 > p0:
            frac = min(max((time - p0) / (p1 - p0), 0.0), 1.0)
        else:
            frac = 1.0 if time >= p1 else 0.0
        out += frac * self.press_vector
        d0 = float(self.drag_interval[0])
        if time > d0:
            out += (time - d0) * self.drag_velocity
        return out

    def _offset_velocity(self, time: float) -> np.ndarray:
        out = np.zeros(3)
        p0, p1 = float(self.press_interval[0]), float(self.press_interval[1])
        if p1 > p0 and p0 < time < p1:
            out += self.press_vector / (p1 - p0)
        if time >= float(self.drag_interval[0]):
            out += self.drag_velocity
        return out

    # --------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        x, v = self.x, self.v
        f_elastic, _ = self._elastic_force(x)
        f_damp = self._damping_force(v)
        f_grav = self.mass[:, None] * GRAVITY
        f_contact, _, _, _, _ = self._contact_force(x, v, self.s_tan)

        f_tot = f_elastic + f_damp + f_grav + f_contact
        v_pred = v + dt * f_tot / self.mass[:, None]
        x_new = x + dt * v_pred
        v_new = v_pred

        f_constr = np.zeros((self.n_nodes, 3))
        if bool(np.any(self.driven)):
            t_new = self.time + dt
            off = self._offset(t_new)
            vel = self._offset_velocity(t_new)
            idx = self.driven
            x_new[idx] = self.driven_position + off[None, :]
            v_new[idx] = vel[None, :]
            f_constr[idx] = self.mass[idx][:, None] * (vel[None, :] - v_pred[idx]) / dt

        # regularised Coulomb friction spring, updated semi-implicitly
        vt = v_new.copy()
        vt[:, 2] = 0.0
        s = self.s_tan + dt * vt
        s[:, 2] = 0.0
        pen_new = self.radius - x_new[:, 2]
        active_new = pen_new > 0.0
        normal_new = np.where(active_new, self.NORMAL_STIFFNESS * pen_new, 0.0)
        ft = self.TANGENTIAL_STIFFNESS * s
        mag = np.sqrt(np.einsum("ij,ij->i", ft, ft))
        limit = self.mu_friction * normal_new
        scale = np.ones(self.n_nodes)
        sliding_new = mag > limit
        scale[sliding_new] = limit[sliding_new] / np.maximum(mag[sliding_new], 1e-30)
        s = s * scale[:, None]
        s[~active_new] = 0.0
        self.s_tan = s
        if bool(np.any(sliding_new)):
            if not self._was_sliding:
                self.slip_events += 1
            self._was_sliding = True
        else:
            self._was_sliding = False

        self.external_terms = np.stack(
            [
                f_grav.sum(axis=0),
                f_contact.sum(axis=0),
                f_constr.sum(axis=0),
                f_damp.sum(axis=0),
            ]
        )
        self.cum_impulse = self.cum_impulse + self.external_terms.sum(axis=0) * dt

        self.x = x_new
        self.v = v_new
        self.step_index += 1
        self.time = self.step_index * dt

    def advance_to(self, time: float):
        target = int(round(float(time) / self.sample_dt))
        limit = target * self.n_sub
        guard = 0
        while self.step_index < limit and guard < 20_000_000:
            self._step(self.dt)
            guard += 1
        return self._frame(float(time))

    # ------------------------------------------------------------- reporting
    def _frame(self, time: float) -> StateFrame:
        x, v = self.x, self.v
        n = self.n_nodes
        deform = self._deformation(x)
        f_elastic, piola = self._elastic_force(x)
        _, normal, ft, active, _ = self._contact_force(x, v, self.s_tan)
        frame = StateFrame(
            time,
            x.copy(),
            v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass[:, None].copy(),
                "driven_top": self.driven.astype(float),
                "rod_vertex": np.ones(n),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, n)),
                "budget_external_force": self.external_terms[None, :, :].copy(),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": deform.copy(),
                "friction_active_mask": active.astype(float),
                "friction_dynamic_coefficient": np.full(n, self.mu_friction),
                "friction_normal": np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
                "friction_normal_force": normal.copy(),
                "friction_relative_velocity": v.copy(),
                "friction_static_coefficient": np.full(n, self.mu_friction),
                "friction_tangential_force": ft.copy(),
                "gravity_acceleration": GRAVITY[None, :].copy(),
                "elastic_internal_force": f_elastic,
                "elastic_first_piola_stress": piola,
                "elastic_reference_position": self.reference_position.copy(),
                "elastic_active_mask": np.ones(self.n_elem),
            },
            scalars={
                "friction_stick_speed_tolerance": self.stick_speed_tolerance,
                "reference_force": self.reference_force,
                "reference_momentum": self.reference_momentum,
                "reference_velocity": self.reference_velocity,
                "contact_length_scale": float(self.radius),
                "characteristic_length": float(self.radius),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        pen = float(np.max(self.radius - self.x[:, 2]))
        return {
            "success": True,
            "solver": "explicit_neo_hookean_fem",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elem),
            "substeps_per_sample": int(self.n_sub),
            "dt": float(self.dt),
            "mass_total": float(self.mass_total),
            "min_height": float(np.min(self.x[:, 2])),
            "max_penetration": pen,
            "slip_events": int(self.slip_events),
            "cumulative_external_impulse": [float(a) for a in self.cum_impulse],
            "contact_nodes_last_frame": int(np.sum(self.radius - self.x[:, 2] > 0.0)),
            "finite_state": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
            "simulated_time": float(self.time),
        }
