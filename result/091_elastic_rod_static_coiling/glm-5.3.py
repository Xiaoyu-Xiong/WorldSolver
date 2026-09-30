"""Quasi-static coiling of a naturally straight elastic rod on a rigid substrate.

Physical model (implemented from scratch, no external physics engine):

* The rod is a slender inextensible elastic rod discretized as a mass-lumped
  node chain ("discrete elastic rod").  Axial elasticity is enforced by stiff
  two-node springs using the frozen Young modulus; bending uses the discrete
  bending energy E_j = (1/2) * k_j * phi_j^2 with k_j = 2*EI/(l0+l1)
  (Bergou et al. form), which is exact for a naturally straight rod
  (natural_curvature = 0).

* Material is fed downward through the fixed injector outlet at the frozen
  injection speed.  Nodes are activated as material crosses the outlet; the
  topmost active node is kinematically prescribed on the vertical feed line
  with the feed velocity, which is the standard nozzle boundary condition for
  rod/strand feeding.

* The substrate is rigid, horizontal and stationary.  The frozen "sticking"
  deposition condition is modeled as perfectly inelastic, permanent sticking:
  any node that touches the substrate is deposited at the contact point and
  pinned thereafter, so the contact point of the suspended heel advances as
  new material is laid down.

* Uniform downward gravity acts on every active node.  A light repulsion
  between the swinging (free) nodes and already deposited material keeps the
  successive turns of the coil distinguishable (rod-rod contact at the
  physical rod diameter); no multilayer collision requirement is imposed.

Integration is semi-implicit (symplectic) Euler with mass-proportional
damping, per-substep velocity clamping and NaN guards, giving deterministic
finite states at every requested sample time.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic solver for the frozen static-coiling scene 091."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.metadata()
        self.seed = int(seed)

        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        rod = cfg.get("rod", {})
        inj = cfg.get("injector", {})
        sub = cfg.get("substrate", {})

        self.radius = float(rod.get("radius", 0.0016))
        self.density = float(rod.get("density", 1180.0))
        self.young_modulus = float(rod.get("young_modulus", 1.3e6))
        self.area = math.pi * self.radius ** 2
        self.rhoA = self.density * self.area
        self.inertia = math.pi * self.radius ** 4 / 4.0
        self.EI = self.young_modulus * self.inertia
        self.L_total = float(rod.get("total_available_length", 1.45))

        self.v_inj = float(inj.get("injection_speed", 0.005))
        self.t_start = float(inj.get("start_time", 0.0))
        self.t_end = float(inj.get("end_time", 180.0))
        self.outlet = np.asarray(inj.get("outlet_position", (0.0, 0.0, 0.5)), dtype=float)
        self.outlet_z = float(self.outlet[2])

        self.z_surface = float(sub.get("surface_z", 0.0))

        # ---------------- discretization ----------------
        self.h = 0.01
        L0_nominal = float(rod.get("initial_deployed_length", 0.5))
        n_seg0 = max(int(round((L0_nominal - self.radius) / self.h)), 4)
        d0 = (L0_nominal - self.radius) / n_seg0
        z_init = self.radius + d0 * np.arange(n_seg0 + 1)

        # Symmetry-breaking shape taken from the frozen initial state asset.
        init = scene.initial_state()
        p0 = np.asarray(init.position, dtype=float)
        order = np.argsort(p0[:, 2])
        x_break = np.interp(z_init, p0[order, 2], p0[order, 0])

        n_extra = int(math.ceil((self.L_total - L0_nominal) / self.h)) + 10
        self.N = (n_seg0 + 1) + n_extra

        self.x = np.zeros((self.N, 3))
        self.x[: n_seg0 + 1, 0] = x_break
        self.x[: n_seg0 + 1, 2] = z_init
        self.v = np.zeros((self.N, 3))

        self.rest = np.empty(self.N - 1)
        seg0 = np.linalg.norm(np.diff(self.x[: n_seg0 + 1], axis=0), axis=1)
        self.rest[:n_seg0] = seg0
        self.rest[n_seg0:] = self.h
        self.s = np.concatenate(([0.0], np.cumsum(self.rest)))
        self.L0 = float(self.s[n_seg0])

        lumped = np.empty(self.N)
        lumped[0] = 0.5 * self.rest[0]
        lumped[1:-1] = 0.5 * (self.rest[:-1] + self.rest[1:])
        lumped[-1] = 0.5 * self.rest[-1]
        self.m = self.rhoA * lumped

        # Per-joint bending stiffness k_j = 2*EI/(l0+l1); joint j (vertex at
        # node j) uses self.k_theta[j-1].
        self.k_theta = 2.0 * self.EI / (self.rest[:-1] + self.rest[1:])

        self.stuck = np.zeros(self.N, dtype=bool)
        self.stuck[0] = True  # frozen initial contact of the rod tip

        self.t = 0.0
        self.fed = 0.0
        self.L = self.L0
        self.i_top = n_seg0
        self.feed_rate = self.v_inj
        self.v[self.i_top] = (0.0, 0.0, -self.v_inj)
        self._last_F = np.zeros((self.N, 3))
        self._steps = 0

        # ---------------- numerics ----------------
        self.dt = 2.5e-4
        self.k_axial = 500.0        # N/m penalty enforcing near-inextensibility
        self.c_damp = 50.0          # 1/s mass-proportional damping
        self.k_self = 20.0          # N/m rod-rod repulsion stiffness
        self.d_self = 2.4 * self.radius  # rod-rod contact distance
        self.v_max = 3.0            # m/s velocity clamp (stability guard)
        self.near_substrate = 6.0 * self.radius

    # ------------------------------------------------------------------
    # injector feed: activate material nodes as they cross the outlet
    # ------------------------------------------------------------------
    def _update_feed(self) -> None:
        span = max(self.t_end - self.t_start, 0.0)
        feed_time = min(max(self.t - self.t_start, 0.0), span)
        fed = min(self.v_inj * feed_time, max(self.L_total - self.L0, 0.0))
        self.fed = fed
        self.L = self.L0 + fed
        self.feed_rate = 0.0
        if self.t > self.t_start and self.t < self.t_end and fed < self.L_total - self.L0 - 1e-12:
            self.feed_rate = self.v_inj
        itop = int(np.searchsorted(self.s, self.L, side="right") - 1)
        itop = min(itop, self.N - 1)
        if itop > self.i_top:
            for i in range(self.i_top + 1, itop + 1):
                zz = self.outlet_z - (self.L - self.s[i])
                self.x[i] = (self.outlet[0], self.outlet[1], zz)
                self.v[i] = (0.0, 0.0, -self.feed_rate)
                self.stuck[i] = False
            self.i_top = itop

    # ------------------------------------------------------------------
    # one physical substep (semi-implicit Euler)
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        self.t += dt
        self._update_feed()
        A = self.i_top + 1
        x, v, m = self.x, self.v, self.m

        F = np.zeros_like(x)
        F[:A] += m[:A, None] * self.gravity
        F[:A] -= self.c_damp * m[:A, None] * v[:A]

        # --- axial (stretching) forces ---
        d = x[1:A] - x[: A - 1]
        l = np.linalg.norm(d, axis=1)
        np.maximum(l, 1e-9, out=l)
        fax = (self.k_axial * (l - self.rest[: A - 1]))[:, None] * d / l[:, None]
        F[: A - 1] += fax
        F[1:A] -= fax

        # --- bending forces (angle between consecutive edges) ---
        # Joint j (vertex at node j) has u = x[j-1]-x[j] and w = x[j+1]-x[j];
        # the deflection angle is phi = pi - angle(u, w), zero when straight.
        if A >= 3:
            ua = x[: A - 2] - x[1 : A - 1]
            wa = x[2:A] - x[1 : A - 1]
            lu = np.linalg.norm(ua, axis=1)
            lw = np.linalg.norm(wa, axis=1)
            np.maximum(lu, 1e-9, out=lu)
            np.maximum(lw, 1e-9, out=lw)
            uh = ua / lu[:, None]
            wh = wa / lw[:, None]
            ct = np.clip(np.einsum("ij,ij->i", uh, wh), -1.0, 1.0)
            theta = np.arccos(ct)
            phi = math.pi - theta
            st = np.maximum(np.sqrt(np.maximum(1.0 - ct * ct, 0.0)), 1e-6)
            ga = (ct[:, None] * uh - wh) / (lu[:, None] * st[:, None])
            gc = (ct[:, None] * wh - uh) / (lw[:, None] * st[:, None])
            gb = -(ga + gc)
            fb = self.k_theta[: A - 2, None] * phi[:, None]
            F[: A - 2] += fb * ga
            F[2:A] += fb * gc
            F[1 : A - 1] += fb * gb

        # --- free vs deposited-node repulsion (near-substrate free nodes) ---
        stuck_idx = np.nonzero(self.stuck[:A])[0]
        free_idx = np.nonzero(~self.stuck[:A])[0]
        free_idx = free_idx[free_idx != self.i_top]
        if stuck_idx.size and free_idx.size:
            near = (x[free_idx, 2] - self.z_surface) < self.near_substrate
            if np.any(near):
                fn = free_idx[near]
                D = x[fn][:, None, :] - x[stuck_idx][None, :, :]
                dist = np.sqrt(np.einsum("ijk,ijk->ij", D, D))
                np.maximum(dist, 1e-6, out=dist)
                w = self.d_self - dist
                act = w > 0.0
                if np.any(act):
                    fmag = (self.k_self * w / self.d_self)[:, :, None] * D / dist[:, :, None]
                    fmag *= act[:, :, None]
                    np.add.at(F, fn, fmag.sum(axis=1))

        # --- integrate free nodes ---
        if free_idx.size:
            acc = F[free_idx] / m[free_idx][:, None]
            v[free_idx] += acc * dt
            sp = np.linalg.norm(v[free_idx], axis=1)
            over = sp > self.v_max
            if np.any(over):
                v[free_idx[over]] *= (self.v_max / sp[over])[:, None]
            x[free_idx] += v[free_idx] * dt
            if not np.isfinite(x[free_idx]).all():
                x[free_idx] = np.nan_to_num(x[free_idx])
                v[free_idx] = 0.0

        # --- kinematic injector boundary condition ---
        it = self.i_top
        x[it, 0] = self.outlet[0]
        x[it, 1] = self.outlet[1]
        x[it, 2] = self.outlet_z - (self.L - self.s[it])
        v[it, 0] = 0.0
        v[it, 1] = 0.0
        v[it, 2] = -self.feed_rate

        # --- sticking substrate deposition ---
        if free_idx.size:
            low = x[free_idx, 2] < self.z_surface + self.radius
            if np.any(low):
                sn = free_idx[low]
                self.stuck[sn] = True
                x[sn, 2] = self.z_surface + self.radius
                v[sn] = 0.0

        self._last_F = F
        self._steps += 1

    # ------------------------------------------------------------------
    # public stepping to a requested sample time
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = max(float(time), self.t)
        if target > self.t:
            n = int(math.ceil((target - self.t) / self.dt - 1e-12))
            n = max(n, 1)
            dt = (target - self.t) / n
            for _ in range(n):
                self._substep(dt)
            self.t = target
        return self._build_frame(float(target))

    # ------------------------------------------------------------------
    # StateFrame assembly with all required raw trajectory fields
    # ------------------------------------------------------------------
    def _build_frame(self, t: float):
        A = self.i_top + 1
        N = self.N
        active = np.zeros(N, dtype=bool)
        active[:A] = True

        pos = self.x.copy()
        pos[A:, 0] = self.outlet[0]
        pos[A:, 1] = self.outlet[1]
        pos[A:, 2] = self.outlet_z + 2.0e-4 * np.arange(1, N - A + 1)
        vel = self.v.copy()
        vel[A:] = 0.0

        mass = np.where(active, self.m, 0.0)
        orientation = np.zeros((N, 4))
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((N, 3))
        body_id = np.zeros(N, dtype=np.int64)

        m_act = self.m[:A]
        m_tot = float(m_act.sum())
        F = self._last_F

        # external-force budget, one global domain (G=1), K=3 terms:
        # gravity, substrate reaction, injector actuation.
        F_grav = m_tot * self.gravity
        p_grav = (m_act[:, None] * self.x[:A]).sum(axis=0) / m_tot

        stuck_idx = np.nonzero(self.stuck[:A])[0]
        if stuck_idx.size:
            R = -F[stuck_idx]  # support reactions at deposited (pinned) nodes
            F_sub = R.sum(axis=0)
            w = np.linalg.norm(R, axis=1)
            wsum = float(w.sum())
            if wsum > 1e-15:
                p_sub = (w[:, None] * self.x[stuck_idx]).sum(axis=0) / wsum
            else:
                p_sub = self.x[stuck_idx].mean(axis=0)
        else:
            F_sub = np.zeros(3)
            p_sub = np.array([0.0, 0.0, self.z_surface + self.radius])

        F_inj = -F[self.i_top]  # actuator force applied at the controlled node
        p_inj = self.x[self.i_top].copy()

        budget_force = np.zeros((1, 3, 3))
        budget_force[0, 0] = F_grav
        budget_force[0, 1] = F_sub
        budget_force[0, 2] = F_inj
        budget_position = np.zeros((1, 3, 3))
        budget_position[0, 0] = p_grav
        budget_position[0, 1] = p_sub
        budget_position[0, 2] = p_inj
        budget_couple = np.zeros((1, 3, 3))  # no free force couples applied

        # case-selected constraint: kinematic injector feed control.
        constraint_value = np.zeros((2, 3))
        constraint_target = np.zeros((2, 3))
        constraint_value[0] = self.x[self.i_top]
        constraint_target[0] = np.array(
            [self.outlet[0], self.outlet[1], self.outlet_z - (self.L - self.s[self.i_top])]
        )
        constraint_value[1] = (self.L, 0.0, 0.0)
        constraint_target[1] = (min(self.L0 + self.fed, self.L_total), 0.0, 0.0)

        cumulative_mass_source = np.array([self.rhoA * self.fed])

        deposited_arc = float(self.s[stuck_idx[-1]]) if stuck_idx.size else 0.0

        attributes = {
            "mass_quantity": mass,
            "arc_coordinate": self.s.copy(),
            "rod_radius": np.full(N, self.radius),
            "is_stuck": self.stuck.astype(float),
            "is_active": active.astype(float),
            "is_injector_controlled": (np.arange(N) == self.i_top).astype(float),
            "is_initial_contact": (np.arange(N) == 0).astype(float),
        }
        fields = {
            "budget_external_couple": budget_couple,
            "budget_external_force": budget_force,
            "budget_force_position": budget_position,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": cumulative_mass_source,
        }
        scalars = {
            "constraint_scale": float(max(self.L0, 1e-6)),
            "reference_force": float(self.rhoA * float(np.linalg.norm(self.gravity)) * self.L_total),
            "reference_torque": float(self.EI),
            "injection_speed": float(self.v_inj),
            "natural_curvature": 0.0,
            "substrate_speed": 0.0,
            "deployed_length": float(self.L),
            "deposited_length": deposited_arc,
        }

        return StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete elastic rod, semi-implicit Euler, sticking substrate",
            "substeps": int(self._steps),
            "node_count": int(self.N),
            "active_nodes": int(self.i_top + 1),
            "deposited_nodes": int(self.stuck.sum()),
            "fed_length": float(self.fed),
            "deployed_length": float(self.L),
        }
