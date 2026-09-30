"""Quasi-static discrete elastic rod solver for naturally curved rod coiling.

The rod is discretized as an inextensible chain of ``N`` vertices.  Vertices
are injected at the fixed outlet (vertex 0) by increasing the rest lengths of
the ``dynamic'' edges.  Vertices that touch the rigid substrate become
sticking and are frozen in place, so the deposited coil grows from the seed
coil provided in the initial topology.

The model is a Kirchhoff-style naturally-curved rod:
* gravity loads the chain;
* a discrete bending energy penalises deviations of the curvature binormal
  from the natural-curvature direction, which rotates with the material twist;
* a torsional energy couples twist along the rod;
* distance constraints enforce inextensibility;
* heavy damping makes the response quasi-static.

No Genesis / gs code is used.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.cfg = scene.config
        self.scene = scene

        # Pull initial frame from the scene.
        frame = scene.initial_state()
        self.N: int = frame.position.shape[0]
        self.x: NDArray[np.float64] = np.asarray(frame.position, dtype=float).copy()
        self.v: NDArray[np.float64] = np.asarray(frame.velocity, dtype=float).copy()
        if frame.orientation is not None:
            self.orientation = np.asarray(frame.orientation, dtype=float).copy()
        else:
            self.orientation = np.zeros((self.N, 4))
            self.orientation[:, 3] = 1.0
        if frame.angular_velocity is not None:
            self.angular_velocity = np.asarray(frame.angular_velocity, dtype=float).copy()
        else:
            self.angular_velocity = np.zeros((self.N, 3))
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.body_id = np.asarray(frame.body_id, dtype=int).copy() if frame.body_id is not None else np.zeros(self.N, dtype=int)

        # Rod parameters.
        rod = self.cfg["rod"]
        self.r = float(rod["radius"])
        self.rho = float(rod["density"])
        self.E = float(rod["young_modulus"])
        self.nu = float(rod["poisson_ratio"])
        self.kappa0 = float(rod["natural_curvature"])
        self.A = np.pi * self.r * self.r
        self.I = np.pi * self.r ** 4 / 4.0
        self.EI = self.E * self.I
        self.GJ = (self.E / (2.0 * (1.0 + self.nu))) * (np.pi * self.r ** 4 / 2.0)

        # Environment.
        self.grav = np.asarray(self.cfg["gravity"], dtype=float)
        self.surface_z = float(self.cfg["substrate"]["surface_z"])
        self.v_inject = float(self.cfg["injector"]["injection_speed"])
        self.t_total = float(self.cfg["duration"])
        self.sample_dt = float(self.cfg["sample_dt"])

        # State masks.
        self.stuck = np.asarray(frame.attributes["is_initially_deposited"], dtype=bool).copy()
        # Vertex 0 is the controlled outlet and must remain fixed in space.
        self.fixed = self.stuck.copy()
        self.fixed[0] = True

        # Rest lengths and total material length.
        self.rest_len = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self.L0 = float(np.sum(self.rest_len))
        self.injected_len = 0.0

        # Material-twist variables (one scalar per vertex, material rotation
        # about the local tangent).  Initialise them so that the natural
        # curvature direction matches the initial shape, giving a low-energy
        # start.
        self.theta = np.zeros(self.N)
        self.omega_theta = np.zeros(self.N)
        self._init_twist_from_shape()

        # Time-stepping.
        self.t = 0.0
        self.prev_time = 0.0
        self.n_sub = 4  # substeps per output sample (dt = sample_dt / n_sub)
        self.dt_sub = self.sample_dt / self.n_sub

        # Damping chosen to make the response quasi-static.
        mass0 = np.mean(self.mass)
        self.c_damp = 0.05 * mass0 * np.linalg.norm(self.grav) / 0.001  # ~ positional damping
        if self.c_damp < 1e-6:
            self.c_damp = 1e-3
        self.c_twist = 1e-7
        self.Ip = np.pi * self.r ** 4 / 2.0

        # Store initial attribute / scalar keys we must keep.
        self.base_attributes = dict(frame.attributes)
        self.base_scalars = dict(frame.scalars)

        self._diag = {"steps": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1e-12:
            dt = min(self.dt_sub, target - self.t)
            self._step(dt)
            self.t += dt
        return self._build_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return self._diag.copy()

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # Inject material: distribute the new length over all dynamic edges.
        self.injected_len += self.v_inject * dt
        dynamic_edge = ~self._both_stuck_edges()
        num_dyn = int(dynamic_edge.sum())
        if num_dyn > 0:
            self.rest_len[dynamic_edge] += (self.v_inject * dt) / num_dyn

        # Recompute per-node mass from current material distribution.
        self.mass = self._compute_mass()

        # Gravity.
        f_ext = self.mass[:, None] * self.grav[None, :]

        # Bending force from natural-curvature energy.
        f_bend = self._bending_force()
        f_ext += f_bend

        # Damping.
        f_ext -= self.c_damp * self.v

        # Integrate dynamic vertices.
        dynamic = ~self.fixed
        self.v[dynamic] += dt * f_ext[dynamic] / self.mass[dynamic, None]
        self.x[dynamic] += dt * self.v[dynamic]

        # Ground contact / sticking.
        self._handle_contact()

        # Enforce inextensibility via iterative projection.
        self._project_lengths()

        # Re-enforce contact after projection (projection can move vertices).
        self._handle_contact()

        # Twist dynamics.
        self._twist_step(dt)

        self._diag["steps"] += 1

    # ------------------------------------------------------------------
    # StateFrame builder
    # ------------------------------------------------------------------
    def _build_frame(self, t: float) -> StateFrame:
        # Arc coordinate from rest lengths.
        arc = np.zeros(self.N)
        arc[1:] = np.cumsum(self.rest_len)

        # Current natural curvature direction in world coordinates.
        _, u, v = self._bishop_frames(self.x)
        d = np.zeros_like(u)
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        for i in range(self.N):
            d[i] = c[i] * u[i] + s[i] * v[i]

        total_mass = float(np.sum(self.mass))
        weight = total_mass * np.linalg.norm(self.grav)

        # External force budget: total gravity on the rod.
        budget_force = np.sum(self.mass[:, None] * self.grav[None, :], axis=0)
        # External couple about the origin.
        com = np.sum(self.mass[:, None] * self.x, axis=0) / max(total_mass, 1e-18)
        budget_couple = np.sum(
            self.x * budget_force[None, :],
            axis=0,
        )
        # Use a simple cross-product torque for the budget couple.
        budget_couple = np.cross(self.x - com, self.mass[:, None] * self.grav[None, :]).sum(axis=0)

        scalars = dict(self.base_scalars)
        scalars["constraint_scale"] = float(np.mean(self.rest_len))
        scalars["reference_force"] = float(weight)
        scalars["reference_torque"] = float(weight / max(self.kappa0, 1e-12))

        edge_len = np.linalg.norm(np.diff(self.x, axis=0), axis=1)

        attributes = {
            "arc_coordinate": arc,
            "is_injector_controlled": (np.arange(self.N) == 0).astype(float),
            "is_initially_deposited": self.base_attributes["is_initially_deposited"].astype(float),
            "is_sticking": self.stuck.astype(float),
            "rod_radius": np.full(self.N, self.r),
            "natural_curvature": np.full(self.N, self.kappa0),
            "natural_curvature_direction": d,
            "mass_quantity": self.mass.copy(),
        }

        fields = {
            "budget_external_force": budget_force.reshape(1, 1, 3),
            "budget_external_couple": budget_couple.reshape(1, 1, 3),
            "budget_force_position": com.reshape(1, 1, 3),
            "constraint_value": edge_len,
            "constraint_target": self.rest_len.copy(),
            "cumulative_mass_source": np.array([self.injected_len * self.rho * self.A], dtype=float),
        }

        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # Geometry / frames
    # ------------------------------------------------------------------
    def _both_stuck_edges(self) -> NDArray[np.bool_]:
        return self.fixed[:-1] & self.fixed[1:]

    def _compute_mass(self) -> NDArray[np.float64]:
        # Per-node effective length = half of adjacent edge rest lengths.
        eff = np.zeros(self.N)
        eff[0] = self.rest_len[0] * 0.5
        eff[-1] = self.rest_len[-1] * 0.5
        eff[1:-1] = 0.5 * (self.rest_len[:-1] + self.rest_len[1:])
        return self.rho * self.A * eff

    def _node_tangents(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.zeros_like(x)
        e = np.diff(x, axis=0)
        le = np.linalg.norm(e, axis=1, keepdims=True)
        te = e / np.where(le > 0, le, 1.0)
        t[0] = te[0]
        t[-1] = te[-1]
        for i in range(1, self.N - 1):
            ti = te[i - 1] + te[i]
            n = np.linalg.norm(ti)
            if n > 1e-12:
                ti = ti / n
            else:
                ti = te[i]
            t[i] = ti
        return t

    def _bishop_frames(self, x: NDArray[np.float64]):
        """Return node tangents and two orthonormal material axes (u, v)."""
        t = self._node_tangents(x)
        u = np.zeros_like(t)
        v = np.zeros_like(t)

        # Outlet direction is the inlet natural-curvature direction.
        d0 = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(t[0], d0)) > 0.999:
            d0 = np.array([0.0, 1.0, 0.0], dtype=float)
        u0 = d0 - np.dot(d0, t[0]) * t[0]
        n0 = np.linalg.norm(u0)
        if n0 < 1e-12:
            u0 = np.array([1.0, 0.0, 0.0], dtype=float)
            u0 -= np.dot(u0, t[0]) * t[0]
            n0 = np.linalg.norm(u0)
        u0 /= n0
        u[0] = u0
        v[0] = np.cross(t[0], u[0])
        v[0] /= max(np.linalg.norm(v[0]), 1e-12)

        for i in range(1, self.N):
            ti = t[i - 1]
            tj = t[i]
            a = np.cross(ti, tj)
            sina = np.linalg.norm(a)
            cosa = np.dot(ti, tj)
            if sina < 1e-12:
                ui = u[i - 1]
            else:
                angle = np.arctan2(sina, cosa)
                k = a / sina
                ui = (
                    u[i - 1] * np.cos(angle)
                    + np.cross(k, u[i - 1]) * np.sin(angle)
                    + k * np.dot(k, u[i - 1]) * (1.0 - np.cos(angle))
                )
            # Re-orthogonalise.
            ui -= np.dot(ui, tj) * tj
            n = np.linalg.norm(ui)
            if n > 1e-12:
                ui = ui / n
            else:
                # Fallback: pick any perpendicular vector.
                if np.linalg.norm(tj - np.array([0.0, 0.0, 1.0])) < 1e-6:
                    ref = np.array([0.0, 1.0, 0.0])
                else:
                    ref = np.array([0.0, 0.0, 1.0])
                ui = ref - np.dot(ref, tj) * tj
                ui /= np.linalg.norm(ui)
            u[i] = ui
            v[i] = np.cross(tj, ui)
            v[i] /= max(np.linalg.norm(v[i]), 1e-12)
        return t, u, v

    def _curvature_binormals(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Discrete curvature binormal at each interior vertex."""
        kappa_b = np.zeros((self.N, 3))
        for i in range(1, self.N - 1):
            a = x[i] - x[i - 1]
            b = x[i + 1] - x[i]
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            if la < 1e-12 or lb < 1e-12:
                continue
            D = la * lb + np.dot(a, b)
            if D < 1e-12:
                continue
            kappa_b[i] = 2.0 * np.cross(a, b) / D
        return kappa_b

    def _bend_energy(self, x: NDArray[np.float64], target_dir: NDArray[np.float64]) -> float:
        # Edge lengths and Voronoi lengths.
        e = np.diff(x, axis=0)
        l = np.linalg.norm(e, axis=1)
        kappa_b = self._curvature_binormals(x)
        energy = 0.0
        for i in range(1, self.N - 1):
            lbar = 0.5 * (l[i - 1] + l[i])
            if lbar < 1e-12:
                continue
            target = self.kappa0 * lbar * target_dir[i]
            diff = kappa_b[i] - target
            energy += 0.5 * self.EI / lbar * np.dot(diff, diff)
        return energy

    def _bending_force(self) -> NDArray[np.float64]:
        """Negative gradient of bending energy w.r.t. positions (central FD)."""
        # Target direction and edge lengths fixed at the current configuration.
        _, u, v = self._bishop_frames(self.x)
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        target_dir = np.zeros_like(u)
        for i in range(self.N):
            target_dir[i] = c[i] * u[i] + s[i] * v[i]

        eps = 1e-7
        f = np.zeros_like(self.x)
        e0 = self._bend_energy(self.x, target_dir)
        for i in range(self.N):
            for j in range(3):
                x_plus = self.x.copy()
                x_plus[i, j] += eps
                e_plus = self._bend_energy(x_plus, target_dir)
                x_minus = self.x.copy()
                x_minus[i, j] -= eps
                e_minus = self._bend_energy(x_minus, target_dir)
                f[i, j] = -(e_plus - e_minus) / (2.0 * eps)
        return f

    def _init_twist_from_shape(self) -> None:
        """Set theta so the material natural-curvature direction matches
        the initial curvature binormal at each interior vertex."""
        _, u, v = self._bishop_frames(self.x)
        kappa_b = self._curvature_binormals(self.x)
        for i in range(1, self.N - 1):
            kb = kappa_b[i]
            if np.linalg.norm(kb) < 1e-12:
                self.theta[i] = 0.0
                continue
            # Project onto the local (u, v) plane.
            cu = np.dot(kb, u[i])
            cv = np.dot(kb, v[i])
            self.theta[i] = np.arctan2(cv, cu)
        self.theta[0] = 0.0
        self.theta[-1] = self.theta[-2]

    # ------------------------------------------------------------------
    # Constraints / contact
    # ------------------------------------------------------------------
    def _project_lengths(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            for i in range(self.N - 1):
                if self.fixed[i] and self.fixed[i + 1]:
                    continue
                p1 = self.x[i]
                p2 = self.x[i + 1]
                d = p2 - p1
                len_d = np.linalg.norm(d)
                if len_d < 1e-12:
                    continue
                err = (len_d - self.rest_len[i]) / len_d
                corr = 0.5 * err * d
                if not self.fixed[i]:
                    self.x[i] += corr
                if not self.fixed[i + 1]:
                    self.x[i + 1] -= corr

    def _handle_contact(self) -> None:
        # Dynamic vertices that have reached the substrate become sticking.
        for i in range(self.N):
            if self.fixed[i] or self.stuck[i]:
                continue
            if self.x[i, 2] <= self.surface_z + 1e-6:
                self.x[i, 2] = self.surface_z
                self.v[i] = 0.0
                self.stuck[i] = True
                self.fixed[i] = True

    # ------------------------------------------------------------------
    # Twist dynamics
    # ------------------------------------------------------------------
    def _twist_step(self, dt: float) -> None:
        _, u, v = self._bishop_frames(self.x)
        kappa_b = self._curvature_binormals(self.x)
        e = np.diff(self.x, axis=0)
        l_edge = np.linalg.norm(e, axis=1)
        lbar = np.zeros(self.N)
        lbar[0] = l_edge[0]
        lbar[-1] = l_edge[-1]
        lbar[1:-1] = 0.5 * (l_edge[:-1] + l_edge[1:])

        c = np.cos(self.theta)
        s = np.sin(self.theta)

        # Bending torque: derivative of bending energy wrt theta.
        tau = np.zeros(self.N)
        for i in range(1, self.N - 1):
            lb = max(lbar[i], 1e-12)
            target = self.kappa0 * lb * (c[i] * u[i] + s[i] * v[i])
            target_perp = self.kappa0 * lb * (-s[i] * u[i] + c[i] * v[i])
            diff = kappa_b[i] - target
            tau[i] = (self.EI / lb) * np.dot(diff, target_perp)

        # Torsional torque.
        for i in range(1, self.N):
            li = max(l_edge[i - 1], 1e-12)
            if i < self.N - 1:
                li_next = max(l_edge[i], 1e-12)
                tau[i] += self.GJ * (
                    (self.theta[i] - self.theta[i - 1]) / li
                    - (self.theta[i + 1] - self.theta[i]) / li_next
                )
            else:
                tau[i] += self.GJ * (self.theta[i] - self.theta[i - 1]) / li

        # Inlet twist is fixed.
        tau[0] = 0.0

        # Update twist velocities with damping.
        I_theta = self.rho * self.Ip * lbar
        I_theta = np.where(I_theta > 0, I_theta, 1e-12)
        self.omega_theta += dt * (tau - self.c_twist * self.omega_theta) / I_theta
        self.omega_theta[0] = 0.0
        self.theta += dt * self.omega_theta
        self.theta[0] = 0.0
