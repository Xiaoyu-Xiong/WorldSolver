"""Translated elastic-rod coiling on a slowly moving belt (self-contained physics).

Physical model
--------------
A naturally straight, inextensible elastic rod is fed downward through a fixed
injector.  The rod is discretised into material nodes connected by inextensible
segments (hard distance constraints) with discrete Euler-Bernoulli bending
elasticity (angle constraints whose stiffness is ``EI / l0`` such that the
discrete bending energy is ``0.5 * EI * theta**2 / l0``).  The equations of
motion are integrated with an implicit sub-stepped scheme (XPBD: position based
dynamics with compliance, i.e. an implicit treatment of the elastic constraint
forces), which is stable for the stiff discretisation and whose static
equilibrium reproduces the exact discrete elastica force balance.

External actions
----------------
* uniform gravity ``-9.81 z``;
* a light viscous drag that places the process in the quasi-static regime of
  the source experiment (the deposition time scale is far longer than the
  elastic relaxation time);
* a lateral guide inside the injector barrel: material in the needle is held on
  the injection axis and can only slide along it;
* a sticking belt contact: material that reaches the belt surface adheres to
  the substrate and is thereafter advected with the belt velocity (the source
  "sticking substrate" condition).  Self contact of the deposited material is
  neglected exactly as in the source model.

Feeding
-------
New material enters at the injector outlet at the prescribed injection speed, so
the deployed rod length grows as ``injection_speed * t`` and the feed pushes the
suspended heel against the stuck deposit, which buckles and lays down nearly
circular turns whose centres are carried away by the belt.

All state is deterministic and finite; no trajectory is precomputed.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete elastica with a sticking belt contact driven by a rod injector."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        rod = cfg["rod"]
        inj = cfg["injector"]
        belt = cfg["belt"]

        self.duration = float(cfg["duration"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g_mag = float(np.linalg.norm(self.gravity)) or 9.81
        self.g_vec = np.asarray(
            [0.0, 0.0, -self.g_mag] if self.gravity[2] < 0 else self.gravity, dtype=float
        )

        # --- rod / material -------------------------------------------------
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.n_initial = int(rod["vertex_count"])
        self.length_initial = float(rod["initial_deployed_length"])
        self.l0 = self.length_initial / float(self.n_initial - 1)
        self.area = np.pi * self.radius**2
        self.line_density = self.density * self.area
        self.node_mass = self.line_density * self.l0

        # Bending rigidity: chosen so that the elastogravity length
        # ell = (EI / (lambda g))**(1/3) is a few centimetres, the regime in
        # which a hanging elastica buckles into centimetre scale coils.
        self.young = 1.0e6
        self.ei = self.young * np.pi * self.radius**4 / 4.0
        self.ell = (self.ei / (self.line_density * self.g_mag)) ** (1.0 / 3.0)
        self.bend_stiffness = self.ei / self.l0          # U = 0.5 * k * theta**2

        # --- injector / belt ------------------------------------------------
        self.outlet = np.asarray(inj["outlet"], dtype=float)
        self.inj_axis = np.asarray(inj["axis"], dtype=float)
        self.v_inj = float(inj["injection_speed"])
        self.feed_start = float(inj["feed_start"])
        self.feed_end = float(inj["feed_end"])
        self.surface_z = float(belt["surface_z"])
        self.contact_z = self.surface_z + self.radius
        self.v_belt = float(belt["speed"]) * np.asarray(belt["translation_axis"], dtype=float)
        self.guide_z = self.outlet[2] - 0.05      # needle length below the outlet
        self.belt_half = np.asarray(belt["half_extents"], dtype=float)

        # --- numerics -------------------------------------------------------
        self.dt = 5.0e-4                # implicit sub-step (stable, quasi-static)
        self.max_dt = self.dt
        self.n_iter = 4                 # constraint solver sweeps per sub-step
        self.drag_tau = 0.02            # viscous relaxation time (quasi-static)
        self.time = 0.0
        self.step_count = 0

        # --- initial state (frozen scene initial condition) ------------------
        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame.mass, dtype=float)
        n = self.x.shape[0]
        self.deposited = np.zeros(n, dtype=bool)
        self.dep_xy = np.zeros((n, 2), dtype=float)
        self.dep_time = np.zeros(n, dtype=float)
        self.arc = np.arange(n, dtype=float) * self.l0

        self.bias = float(cfg.get("initial_conditions", {}).get("lateral_branch_bias", 0.0))
        self.feed_len = 0.0
        self.inserted = 0

        # per frame budget accumulators
        self._reset_budget()
        self._last_time = 0.0

    # ---------------------------------------------------------------- helpers
    def _reset_budget(self) -> None:
        self.b_grav = np.zeros(3)
        self.b_drag = np.zeros(3)
        self.b_belt = np.zeros(3)
        self.b_guide = np.zeros(3)
        self.p_grav = np.zeros(3)
        self.p_drag = np.zeros(3)
        self.p_belt = np.zeros(3)
        self.p_guide = np.zeros(3)
        self.w_grav = 0.0
        self.w_drag = 0.0
        self.w_belt = 0.0
        self.w_guide = 0.0
        self.b_dep_impulse = np.zeros(3)

    def _guide_mask(self) -> np.ndarray:
        return (self.x[:, 2] >= self.guide_z) & (~self.deposited)

    def _insert_nodes(self) -> None:
        """Add material at the injector outlet, extending the deployed length."""
        feed_elapsed = float(np.clip(self.time - self.feed_start, 0.0,
                                     self.feed_end - self.feed_start))
        self.feed_len = self.v_inj * feed_elapsed
        target = self.n_initial + int(np.floor(self.feed_len / self.l0 + 1e-12))
        while self.x.shape[0] < target:
            top = self.x[-1]
            below = self.x[-2]
            direction = top - below
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                direction = -self.inj_axis
                norm = 1.0
            new_pos = top + direction / norm * self.l0
            if new_pos[2] >= self.guide_z:
                new_pos[0] = 0.0
                new_pos[1] = 0.0
            self.x = np.vstack([self.x, new_pos])
            self.v = np.vstack([self.v, self.v_inj * self.inj_axis])
            self.mass = np.append(self.mass, self.node_mass)
            self.deposited = np.append(self.deposited, False)
            self.dep_xy = np.vstack([self.dep_xy, np.zeros((1, 2))])
            self.dep_time = np.append(self.dep_time, 0.0)
            self.arc = np.append(self.arc, self.arc[-1] + self.l0)
            self.inserted += 1

    # ------------------------------------------------------------- mechanics
    def _solve_constraints(self, x: np.ndarray, dt: float):
        """XPBD sweeps: hard axial segments + compliant bending angles.

        Returns the accumulated per-node constraint force (multiplier based,
        valid for prescribed nodes as well).
        """
        n = x.shape[0]
        mass = self.mass
        inv_mass = np.where(self.deposited, 0.0, 1.0 / mass)
        fcon = np.zeros((n, 3))
        dt2 = dt * dt

        for _ in range(self.n_iter):
            # --- inextensible segments -------------------------------------
            d = x[1:] - x[:-1]
            length = np.linalg.norm(d, axis=1)
            length_safe = np.where(length < 1e-12, 1e-12, length)
            dhat = d / length_safe[:, None]
            c_ax = length - self.l0
            w_sum = inv_mass[:-1] + inv_mass[1:]
            denom = np.where(w_sum > 1e-12, w_sum, 1e-12)
            lam = np.where(w_sum > 1e-12, -c_ax / denom, 0.0)
            corr_lo = (-inv_mass[:-1] * lam)[:, None] * dhat
            corr_hi = (+inv_mass[1:] * lam)[:, None] * dhat
            x[:-1] += corr_lo
            x[1:] += corr_hi
            force = (lam / dt2)[:, None] * dhat
            fcon[:-1] -= force
            fcon[1:] += force

            # --- bending stiffness (angle between consecutive segments) ----
            if n > 2:
                u = x[1:-1] - x[:-2]
                w = x[2:] - x[1:-1]
                nu = np.linalg.norm(u, axis=1)
                nw = np.linalg.norm(w, axis=1)
                good = (nu > 1e-12) & (nw > 1e-12)
                nu = np.where(good, nu, 1.0)
                nw = np.where(good, nw, 1.0)
                uh = u / nu[:, None]
                wh = w / nw[:, None]
                cos_t = np.clip(np.sum(uh * wh, axis=1), -1.0, 1.0)
                theta = np.arccos(cos_t)
                sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
                active = good & (theta > 1e-7) & (sin_t > 1e-9)
                sin_safe = np.where(active, sin_t, 1.0)
                ga = (wh - cos_t[:, None] * uh) / (nu[:, None] * sin_safe[:, None])
                gc = -(uh - cos_t[:, None] * wh) / (nw[:, None] * sin_safe[:, None])
                gb = -(ga + gc)
                wa = inv_mass[:-2]
                wb = inv_mass[1:-1]
                wc = inv_mass[2:]
                denom_b = (
                    wa * np.sum(ga * ga, axis=1)
                    + wb * np.sum(gb * gb, axis=1)
                    + wc * np.sum(gc * gc, axis=1)
                    + 1.0 / (self.bend_stiffness * dt2)
                )
                denom_b = np.where(denom_b > 1e-12, denom_b, 1e-12)
                lam_b = np.where(active, -theta / denom_b, 0.0)
                x[:-2] += (wa * lam_b)[:, None] * ga
                x[1:-1] += (wb * lam_b)[:, None] * gb
                x[2:] += (wc * lam_b)[:, None] * gc
                fb = (lam_b / dt2)[:, None]
                fcon[:-2] += fb * ga
                fcon[1:-1] += fb * gb
                fcon[2:] += fb * gc

            # --- injector needle: lateral guide ----------------------------
            guide = self._guide_mask()
            if np.any(guide):
                x[guide, 0] = 0.0
                x[guide, 1] = 0.0

        return fcon

    def _substep(self, dt: float) -> None:
        self.time += dt
        self.step_count += 1
        self._insert_nodes()

        x_old = self.x.copy()
        dep = self.deposited
        mass = self.mass

        # prescribed (stuck) material is advected exactly with the belt
        if np.any(dep):
            idx = np.flatnonzero(dep)
            lag = self.time - self.dep_time[idx]
            self.x[idx, 0] = self.dep_xy[idx, 0] + self.v_belt[0] * lag
            self.x[idx, 1] = self.dep_xy[idx, 1] + self.v_belt[1] * lag
            self.x[idx, 2] = self.contact_z
            self.v[idx] = self.v_belt
            x_old[idx] = self.x[idx]

        # explicit accelerations: gravity + viscous drag (quasi-static limit)
        acc = np.zeros_like(self.x)
        free = ~dep
        acc[free] = self.g_vec[None, :] - self.v[free] / self.drag_tau
        v_expl = self.v + dt * acc
        v_expl[dep] = self.v_belt
        x_pred = self.x + dt * v_expl
        x_pred[dep] = self.x[dep]

        # implicit elastic constraints
        x = x_pred.copy()
        fcon = self._solve_constraints(x, dt)

        # ---- momentum budget of this sub-step ----------------------------
        f_grav = mass[:, None] * self.g_vec[None, :]
        f_drag = np.zeros_like(self.x)
        f_drag[free] = -mass[free, None] * self.v[free] / self.drag_tau
        self.b_grav += f_grav.sum(axis=0)
        self.p_grav += (mass[:, None] * x_old).sum(axis=0)
        self.w_grav += float(mass.sum())
        self.b_drag += f_drag.sum(axis=0)
        self.p_drag += (mass[free, None] * x_old[free]).sum(axis=0)
        self.w_drag += float(mass[free].sum())

        if np.any(dep):
            idx = np.flatnonzero(dep)
            belt_force = -(f_grav[idx] + fcon[idx])
            self.b_belt += belt_force.sum(axis=0)
            self.p_belt += (mass[idx, None] * x_old[idx]).sum(axis=0)
            self.w_belt += float(mass[idx].sum())

        guide = self._guide_mask()
        if np.any(guide):
            gforce = -(fcon[guide] + f_drag[guide])
            gforce[:, 2] = 0.0
            self.b_guide += gforce.sum(axis=0)
            self.p_guide += (mass[guide, None] * x_old[guide]).sum(axis=0)
            self.w_guide += float(mass[guide].sum())

        # velocities from the projection, then update state
        v_new = (x - x_old) / dt
        v_new[dep] = self.v_belt

        # ---- belt contact / sticking deposition ---------------------------
        self.x = x
        contact = (~dep) & (self.x[:, 2] <= self.contact_z)
        if np.any(contact):
            cidx = np.flatnonzero(contact)
            impulse = mass[cidx, None] * (self.v_belt[None, :] - v_new[cidx])
            self.b_belt += impulse.sum(axis=0) / dt
            self.p_belt += (mass[cidx, None] * self.x[cidx]).sum(axis=0)
            self.w_belt += float(mass[cidx].sum())
            self.dep_xy[cidx, 0] = self.x[cidx, 0]
            self.dep_xy[cidx, 1] = self.x[cidx, 1]
            self.dep_time[cidx] = self.time
            self.deposited[cidx] = True
            self.x[cidx, 2] = self.contact_z
            v_new[cidx] = self.v_belt

        self.v = v_new

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            target = self.time
        guard = 0
        while self.time < target - 1e-12:
            dt = min(self.max_dt, target - self.time)
            if dt <= 1e-12:
                break
            self._substep(dt)
            guard += 1
            if guard > 5_000_000:            # pragma: no cover - safety guard
                raise RuntimeError("solver exceeded sub-step guard")
        self._last_time = target
        return self._frame(target)

    def _elastica_state(self):
        d = self.x[1:] - self.x[:-1]
        seg = np.linalg.norm(d, axis=1)
        return seg

    def _frame(self, time: float):
        x = self.x
        n = x.shape[0]
        seg = self._elastica_state()
        strain = float(np.mean(seg) / self.l0) - 1.0

        feed_elapsed = float(np.clip(time - self.feed_start, 0.0,
                                     self.feed_end - self.feed_start))
        injected_length = self.v_inj * feed_elapsed
        deployed = self.length_initial + injected_length
        injected_mass = self.line_density * injected_length

        top = int(np.argmax(x[:, 2]))
        controlled = np.zeros(n)
        controlled[top] = 1.0

        f_ext = np.stack(
            [self.b_grav, self.b_drag, self.b_belt, self.b_guide], axis=0
        )[None, :, :]
        p_ext = np.stack(
            [self.p_grav, self.p_drag, self.p_belt, self.p_guide], axis=0
        )
        w_sum = np.array(
            [self.w_grav, self.w_drag, self.w_belt, self.w_guide]
        )
        fallback = x[int(np.argmax(x[:, 2]))]
        positions = np.empty((4, 3))
        for k in range(4):
            if w_sum[k] > 1e-15:
                positions[k] = p_ext[k] / w_sum[k]
            else:
                positions[k] = fallback
        positions = positions[None, :, :]

        frame = StateFrame(
            time=time,
            position=x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc.copy(),
                "natural_curvature": np.zeros(n),
                "natural_twist": np.zeros(n),
                "injector_controlled": controlled,
                "deposited_sticking": self.deposited.astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_couple": np.zeros((1, 4, 3)),
                "budget_external_force": f_ext,
                "budget_force_position": positions,
                "budget_domain_weights": np.ones((1, n)),
                "constraint_value": np.array(
                    [injected_length, strain, float(x[top, 0]), float(x[top, 1])]
                ),
                "constraint_target": np.array(
                    [injected_length, 0.0, 0.0, 0.0]
                ),
                "cumulative_mass_source": np.array([injected_mass]),
            },
            scalars={
                "constraint_scale": float(self.ell),
                "reference_force": float(self.ei / self.ell**2),
                "reference_torque": float(self.ei / self.ell),
                "rod_radius": float(self.radius),
                "injection_speed": float(self.v_inj),
                "belt_speed": float(np.linalg.norm(self.v_belt)),
                "elastic_modulus": float(self.young),
                "elastogravity_length": float(self.ell),
                "deployed_length": float(deployed),
                "injected_length": float(injected_length),
                "node_count": float(n),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "xpbd_discrete_elastica_sticking_belt",
            "youngs_modulus": self.young,
            "elastogravity_length": self.ell,
            "bending_stiffness_ei": self.ei,
            "rest_segment_length": self.l0,
            "sub_step": self.max_dt,
            "sub_steps": self.step_count,
            "nodes": int(self.x.shape[0]),
            "injected_length": self.feed_len,
            "deposited_nodes": int(self.deposited.sum()),
        }
