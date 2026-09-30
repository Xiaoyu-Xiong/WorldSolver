"""Physically based solver for the coiling reversal of a naturally curved rod.

The rod is discretised as a Lagrangian chain of nodes (a discrete elastic rod).
Each node carries a position and a material twist angle.  The model invariants:

* inextensibility is enforced geometrically by projecting every edge back to its
  rest length after each explicit update (a constrained gradient flow);
* bending elasticity uses the discrete curvature binormal of Bergou et al. with
  a *material* natural curvature vector ``kappa0 * (t x d1)`` so that the natural
  curvature direction rotates with the material frame (this is the mechanism
  behind the coiling inversion);
* twist elasticity penalises the material frame rotation rate along the rod;
  the inlet material orientation is held fixed, which is the source boundary
  condition that drives the twist build up;
* uniform gravity acts on every node;
* the substrate is a rigid plane (box top at z = 0).  A node that reaches it
  sticks (deposited material is frozen in position *and* material orientation),
  exactly the source sticking-substrate condition; no self contact is modelled;
* the injector feeds material at a constant rate: new material points are
  inserted at the nozzle with the frozen inlet material orientation.

Time integration is a first order (overdamped / quasi-static) explicit scheme
with a small fixed step, which is the creeping-flow limit of the rod dynamics
requested by the ``source_static_coiling_inversion`` regime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_G = 9.81
_TINY = 1e-15


def _unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if not isinstance(cfg, dict):
            cfg = json.loads((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        self.cfg = cfg

        rod = cfg["rod"]
        inj = cfg["injector"]
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.young = float(rod["young_modulus"])
        self.poisson = float(rod["poisson_ratio"])
        self.kappa0 = float(rod["natural_curvature"])
        self.v_inj = float(inj["injection_speed"])
        self.t_start = float(inj.get("start_time", 0.0))
        self.t_end = float(inj.get("end_time", 1e9))
        self.nozzle = np.asarray(inj["outlet_position"], dtype=float)
        self.gvec = np.asarray(cfg["gravity"], dtype=float)
        self.seed = int(seed)

        r = self.radius
        self.area = math.pi * r * r
        self.inertia = math.pi * r ** 4 / 4.0
        self.polar = math.pi * r ** 4 / 2.0
        self.B = self.young * self.inertia                      # bending stiffness
        self.C = self.young / (2.0 * (1.0 + self.poisson)) * self.polar  # twisting stiffness

        self.l_gb = float(rod.get("gravito_bending_length", 0.0)) or float(
            (self.B / (self.density * self.area * _G)) ** (1.0 / 3.0)
        )
        self.reference_force = self.B / self.l_gb ** 2
        self.reference_torque = self.B / self.l_gb
        self.constraint_scale = float(rod.get("total_available_length", 1.0))
        self.total_length = float(rod.get("total_available_length", 1.75))

        sub = cfg["substrate"]
        self.surface_z = float(sub.get("surface_z", 0.0))
        self.half_x = 0.24
        self.half_y = 0.20
        geo = cfg.get("scene_geometry")
        if isinstance(geo, dict):
            for obj in geo.get("objects", []):
                if obj.get("id") == "substrate":
                    he = obj.get("geometry", {}).get("half_extents")
                    if he:
                        self.half_x = float(he[0])
                        self.half_y = float(he[1])

        # damping (creeping / quasi static limit); chosen so that the elastic and
        # twist relaxation times are well below the injection time scale.
        self.gamma = 0.25
        self.gamma_theta = 3.0e-5
        self.dt = 2.0e-3
        self.pbc_iterations = 40
        self.contact_z = self.surface_z + self.radius
        self.escape_z = self.surface_z - 0.15

        frame = scene.initial_state()
        self.r = np.array(frame.position, dtype=float)
        self.theta = np.zeros(len(self.r))
        deposited = np.asarray(frame.attributes["is_initially_deposited"], dtype=float)
        self.initially_deposited = deposited.copy()
        frozen = deposited > 0.5
        frozen |= self.r[:, 2] <= self.contact_z + 1e-9
        frozen |= self.r[:, 2] <= self.surface_z - 1e-6
        self.frozen = frozen
        self.r[self.frozen, 2] = np.maximum(self.r[self.frozen, 2], self.contact_z)

        self.rest = np.linalg.norm(np.diff(self.r, axis=0), axis=1)
        self.l0 = float(self.rest[0])
        self.initial_length = float(np.sum(self.rest))
        self.injected_total = 0.0
        self.pending_feed = 0.0
        self.arc = np.concatenate([[0.0], np.cumsum(self.rest)])
        self._update_mass()

        self.t = 0.0
        self.r_prev = self.r.copy()
        self.theta_prev = self.theta.copy()
        self.steps = 0
        self.last_contact_force = np.zeros(3)
        self.last_contact_point = np.zeros(3)
        self.last_feed_force = np.zeros(3)

    # ------------------------------------------------------------------ utils
    def _update_mass(self) -> None:
        n = len(self.r)
        lumped = np.empty(n)
        lumped[0] = self.rest[0] * 0.5
        lumped[-1] = self.rest[-1] * 0.5
        lumped[1:-1] = (self.rest[:-1] + self.rest[1:]) * 0.5
        self.mass = self.density * self.area * lumped

    # ------------------------------------------------------------- kinematics
    @staticmethod
    def _edges(r):
        return np.diff(r, axis=0)

    def _tangents(self, e):
        le = np.maximum(np.linalg.norm(e, axis=1), _TINY)
        th = e / le[:, None]
        n = len(th)
        t = np.empty((n + 1, 3))
        t[0] = th[0]
        t[-1] = th[-1]
        t[1:-1] = th[:-1] + th[1:]
        t = _unit(t)
        return le, th, t

    def _transport(self, tangents):
        """Parallel transport of the inlet material normal along the rod.

        ``tangents`` holds the unit tangent at every node; the returned
        rotation-minimising frame is per node as well, so that the per node
        twist angles measure the genuine material rotation about the tangent.
        """
        n = len(tangents)
        u = np.empty((n, 3))
        ref = np.array([1.0, 0.0, 0.0])
        t0 = tangents[0]
        if abs(float(np.dot(ref, t0))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u0 = ref - np.dot(ref, t0) * t0
        u[0] = u0 / max(float(np.linalg.norm(u0)), _TINY)
        for i in range(1, n):
            a = tangents[i - 1]
            b = tangents[i]
            axis = np.cross(a, b)
            s = float(np.linalg.norm(axis))
            c = float(np.dot(a, b))
            v = u[i - 1]
            if s < 1e-12:
                if c < 0.0:
                    # 180 degree flip: rotate about any axis normal to a
                    aux = np.array([0.0, 0.0, 1.0])
                    if abs(float(np.dot(aux, a))) > 0.9:
                        aux = np.array([0.0, 1.0, 0.0])
                    axis = _unit(np.cross(a, aux))
                    v = 2.0 * np.dot(axis, v) * axis - v
                # else: identity
            else:
                axis = axis / s
                ang = math.atan2(s, c)
                v = (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
                     + axis * float(np.dot(axis, v)) * (1.0 - math.cos(ang)))
            v = v - float(np.dot(v, b)) * b
            u[i] = v / max(float(np.linalg.norm(v)), _TINY)
        return u

    # ---------------------------------------------------------------- physics
    def _elastic_forces(self, r, theta):
        """Bending + twisting internal forces and generalized torques."""
        n = len(r)
        F = np.zeros((n, 3))
        Q = np.zeros(n)
        if n < 3:
            return F, Q
        e = self._edges(r)
        le, th, t = self._tangents(e)
        u = self._transport(t)
        v = np.cross(t, u)
        ct = np.cos(theta)[:, None]
        st = np.sin(theta)[:, None]
        d1 = ct * u + st * v
        d1p = -st * u + ct * v  # derivative wrt theta

        a = e[:-1]
        b = e[1:]
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        la = np.maximum(la, _TINY)
        lb = np.maximum(lb, _TINY)
        nvec = np.cross(a, b)
        D = la * lb + np.einsum("ij,ij->i", a, b)
        # Guard the nearly folded-back configuration: D -> 0 would make the
        # integrated curvature singular.  Capping it bounds the turning angle.
        D = np.maximum(D, 0.1 * la * lb)
        kb = 2.0 * nvec / D[:, None]

        lbar = np.maximum(0.5 * (la + lb), 1e-4)
        ti = t[1:-1]
        # ``kb`` is the dimensionless integrated curvature (turning angle) of the
        # vertex, so the natural curvature enters scaled by the local rest length.
        c_nat = (self.kappa0 * lbar)[:, None] * np.cross(ti, d1[1:-1])
        G = (self.B / lbar)[:, None] * (kb - c_nat)

        nG = np.einsum("ij,ij->i", nvec, G)
        two = 2.0 / D
        four = -2.0 / D ** 2 * nG
        wa = (lb / la)[:, None] * a + b
        wb = (la / lb)[:, None] * b + a
        Aa = two[:, None] * np.cross(b, G) + four[:, None] * wa
        Ab = -two[:, None] * np.cross(a, G) + four[:, None] * wb
        F[:-2] += Aa
        F[1:-1] += -Aa + Ab
        F[2:] += -Ab

        # bending contribution to the material twist torque
        dc = (self.kappa0 * lbar)[:, None] * np.cross(ti, d1p[1:-1])
        Q[1:-1] += np.einsum("ij,ij->i", G, dc)

        # twist energy: (C / 2 l) (theta_{i+1} - theta_i)^2
        dth = theta[1:] - theta[:-1]
        k = self.C / np.maximum(self.rest, _TINY)
        gtw = k * dth
        dEdth = np.zeros(n)
        dEdth[:-1] += -gtw
        dEdth[1:] += gtw
        Q += -dEdth
        return F, Q

    def _step(self, dt: float) -> None:
        F, Q = self._elastic_forces(self.r, self.theta)
        F[:, 2] += self.mass * self.gvec[2]
        F[:, 0] += self.mass * self.gvec[0]
        F[:, 1] += self.mass * self.gvec[1]

        free = ~self.frozen
        denom = np.where(free, self.gamma, 1.0)
        # contact reactions are the residual force on anchored nodes
        anchor = self.frozen
        self.last_contact_force = -np.sum(F[anchor], axis=0) if np.any(anchor) else np.zeros(3)
        if np.any(anchor):
            self.last_contact_point = np.mean(self.r[anchor], axis=0)
        else:
            self.last_contact_point = self.nozzle.copy()

        self.r[free] += (dt / denom[free, None]) * F[free]
        theta_old = self.theta.copy()
        dtheta = dt / self.gamma_theta * Q
        self.theta = theta_old + dtheta
        self.theta[0] = 0.0                       # inlet material orientation fixed
        self.theta[self.frozen] = theta_old[self.frozen]  # deposited material is stuck

        self._project_inextensible()
        self._resolve_contact()
        self.steps += 1

    def _project_inextensible(self) -> None:
        rest = self.rest
        frozen = self.frozen
        for _ in range(self.pbc_iterations):
            e = np.diff(self.r, axis=0)
            le = np.linalg.norm(e, axis=1)
            le = np.maximum(le, _TINY)
            corr = (e / le[:, None]) * ((le - rest) * 0.5)[:, None]
            dr = np.zeros_like(self.r)
            dr[:-1] += corr
            dr[1:] -= corr
            dr[frozen] = 0.0
            self.r += 0.9 * dr

    def _resolve_contact(self) -> None:
        z = self.r[:, 2]
        on_plate = (np.abs(self.r[:, 0]) <= self.half_x) & (np.abs(self.r[:, 1]) <= self.half_y)
        touch = (~self.frozen) & on_plate & (z <= self.contact_z + 1e-7)
        if np.any(touch):
            self.r[touch, 2] = self.contact_z
            self.frozen |= touch
        # safety: rod that leaves the plate is parked instead of falling forever
        lost = (~self.frozen) & (z < self.escape_z)
        if np.any(lost):
            self.frozen |= lost

    def feed_target(self, t: float) -> float:
        """Prescribed extruded length at time ``t`` (the injection constraint)."""
        if t <= self.t_start:
            return 0.0
        te = min(float(t), self.t_end)
        return self.v_inj * max(0.0, te - self.t_start)

    def _feed(self, dt: float) -> None:
        if not (self.t_start <= self.t <= self.t_end):
            return
        self.pending_feed += self.v_inj * dt
        while self.pending_feed >= self.l0:
            self.pending_feed -= self.l0
            self._insert_node()

    def _insert_node(self) -> None:
        r = self.r
        t0 = r[1] - r[0]
        t0 = t0 / max(float(np.linalg.norm(t0)), _TINY)
        p = r[0] - self.l0 * t0
        if float(np.linalg.norm(self.nozzle - r[0])) < self.l0:
            p = self.nozzle.copy()
        p = np.array(p, dtype=float)
        p[2] = min(p[2], self.nozzle[2])
        self.r = np.vstack([p[None, :], r])
        self.theta = np.concatenate([[0.0], self.theta])
        self.frozen = np.concatenate([[False], self.frozen])
        self.initially_deposited = np.concatenate([[0.0], self.initially_deposited])
        self.rest = np.concatenate([[self.l0], self.rest])
        self.arc = np.concatenate([[0.0], self.arc + self.l0])
        self.injected_total += self.l0
        self._update_mass()
        self.r_prev = np.vstack([p[None, :], self.r_prev])
        self.theta_prev = np.concatenate([[0.0], self.theta_prev])

    # ------------------------------------------------------------- trajectory
    def _orientation(self, r, theta):
        n = len(r)
        if n < 2:
            q = np.zeros((n, 4))
            q[:, 3] = 1.0
            return q
        e = self._edges(r)
        _, th, t = self._tangents(e)
        u = self._transport(t)
        v = np.cross(t, u)
        ct = np.cos(theta)[:, None]
        st = np.sin(theta)[:, None]
        d1 = _unit(ct * u + st * v)
        d2 = np.cross(t, d1)
        # rotation matrix, columns (d1, d2, d3=t)
        m00 = d1[:, 0]
        m11 = d2[:, 1]
        m22 = t[:, 2]
        tr = m00 + m11 + m22
        def _sgn(x):
            return np.where(x >= 0.0, 1.0, -1.0)

        q = np.zeros((n, 4))
        q[:, 3] = 0.5 * np.sqrt(np.maximum(1.0 + tr, 0.0))
        q[:, 0] = 0.5 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * _sgn(d2[:, 2] - t[:, 1])
        q[:, 1] = 0.5 * np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * _sgn(t[:, 0] - d1[:, 2])
        q[:, 2] = 0.5 * np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * _sgn(d1[:, 1] - d2[:, 0])
        nrm = np.linalg.norm(q, axis=1, keepdims=True)
        bad = nrm[:, 0] < 1e-9
        q[bad] = [0.0, 0.0, 0.0, 1.0]
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), _TINY)
        return q

    def _build_frame(self, t: float) -> StateFrame:
        r = self.r
        n = len(r)
        dt = 1.0 / 50.0
        vel = np.zeros((n, 3))
        free = ~self.frozen
        vel[free] = (r[free] - self.r_prev[free]) / dt
        e = self._edges(r) if n > 1 else np.zeros((0, 3))
        _, th, tangent = self._tangents(e) if n > 1 else (None, None, np.zeros((n, 3)))
        omega = np.zeros((n, 3))
        dtheta = (self.theta - self.theta_prev) / dt
        omega[free] = dtheta[free, None] * tangent[free]
        orientation = self._orientation(r, self.theta)

        gravity_term = float(np.sum(self.mass)) * self.gvec
        contact_term = self.last_contact_force
        feed_force = self.density * self.area * self.v_inj ** 2 * np.array([0.0, 0.0, -1.0])
        self.last_feed_force = feed_force
        com = np.sum(self.mass[:, None] * r, axis=0) / max(float(np.sum(self.mass)), _TINY)

        ext_force = np.stack([gravity_term, contact_term, feed_force], axis=0)[None, ...]
        ext_couple = np.zeros((1, 3, 3))
        force_pos = np.stack([com, self.last_contact_point, self.nozzle], axis=0)[None, ...]

        current_length = float(np.sum(self.rest))
        fields = {
            "budget_external_force": ext_force,
            "budget_external_couple": ext_couple,
            "budget_force_position": force_pos,
            "constraint_value": np.array([current_length]),
            "constraint_target": np.array([self.initial_length + self.feed_target(t)]),
            "cumulative_mass_source": np.array([self.density * self.area * self.injected_total]),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "arc_coordinate": self.arc.copy(),
            "is_sticking": self.frozen.astype(float),
            "is_initially_deposited": self.initially_deposited.copy(),
            "is_injector_controlled": (np.arange(n) == 0).astype(float),
            "rod_radius": np.full(n, self.radius),
            "natural_curvature": np.full(n, self.kappa0),
            "natural_curvature_direction": np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)),
        }
        scalars = {
            "injection_speed": self.v_inj,
            "natural_curvature": self.kappa0,
            "normalized_natural_curvature": float(self.kappa0 * self.l_gb),
            "substrate_speed": 0.0,
            "constraint_scale": self.constraint_scale,
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
        }
        frame = StateFrame(
            time=float(t),
            position=r.copy(),
            velocity=vel,
            orientation=orientation,
            angular_velocity=omega,
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.t:
            target = self.t
        if target > self.t:
            n_sub = int(round((target - self.t) / self.dt))
            if n_sub < 1:
                n_sub = 1
            h = (target - self.t) / n_sub
            self.r_prev = self.r.copy()
            self.theta_prev = self.theta.copy()
            for _ in range(n_sub):
                self._feed(h)
                self._step(h)
                self.t += h
            self.t = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "nodes": int(len(self.r)),
            "steps": int(self.steps),
            "injected_length": float(self.injected_total),
            "time": float(self.t),
        }
