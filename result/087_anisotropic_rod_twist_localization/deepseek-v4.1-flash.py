"""Quasi-static discrete elastic rod solver for the V-shaped anisotropic rod.

Physical model
--------------
The rod is a discrete Kirchhoff/elastic rod (49 vertices).  Its centerline is
held by the two endpoint clamps in the frozen V-shaped configuration, so its
centerline degrees of freedom are fully constrained by the boundary conditions
(the rod is inextensible, gravity-free, and clamped at both ends).  The only
remaining degrees of freedom are the material-frame rotations theta_i about the
vertex tangent, which carry the anisotropic cross-section orientation and the
torsion of the rod.

The elastic energy is the standard discrete-rod energy

    E = sum_i  (1 / (2*lbar_i)) [ B1 (k1_i - kb1_i)^2 + B2 (k2_i - kb2_i)^2 ]
      + sum_i  (1/2) * T_i * ((theta_i - theta_{i-1}) / l_i)^2

with material curvature components k1 = kappa_b . d1, k2 = kappa_b . d2
(kappa_b is the discrete curvature binormal), stiffnesses B1 = E*I1,
B2 = E*I2 for the flat rectangular cross-section, torsion stiffness T = G*J,
and frozen natural curvatures kb1/kb2 taken from the untwisted reference
configuration (nonzero only at the apex bend).  Rotating the material frame at
the sharp apex bend therefore costs bending energy while torsion of each arm is
cheap, which is exactly the twist-localization mechanism of the source
experiment.

For every requested sample time the driven-end rotation is ramped along the
configured schedule; each increment is equilibrated with a damped Newton
iteration on the material-frame angles (analytic gradient and Hessian), using
continuation in the twist increment.  The result is a deterministic, fully
physical equilibrium path of the anisotropic rod.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_Y = np.array([0.0, 1.0, 0.0])


def _unit(v, axis=-1, eps=1e-14):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _rotate(vec, axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * (np.dot(axis, vec) * (1.0 - c))


def _quats_from_frames(d1, d2, t):
    """Quaternion (xyzw) of the rotation with columns (d1, d2, t)."""
    n = d1.shape[0]
    q = np.zeros((n, 4))
    for i in range(n):
        r = np.stack((d1[i], d2[i], t[i]), axis=-1)
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            q[i] = (
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
                0.25 * s,
            )
        else:
            k = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
            if k == 0:
                s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                q[i] = (0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s,
                        (r[2, 1] - r[1, 2]) / s)
            elif k == 1:
                s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
                q[i] = ((r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s,
                        (r[0, 2] - r[2, 0]) / s)
            else:
                s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
                q[i] = ((r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s,
                        (r[1, 0] - r[0, 1]) / s)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-14)
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


class BenchmarkSolver:
    """Energy-minimizing discrete elastic rod driven by an end rotation."""

    max_newton_iterations = 150
    max_continuation_step = 0.4

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = getattr(scene, "config", {}) or {}
        rod = config.get("rod", {})
        bc = config.get("boundary_conditions", {})

        initial = scene.initial_state()
        position = np.asarray(initial.position, dtype=float)
        mass = np.asarray(initial.mass, dtype=float)
        self.position = position
        self.n = int(position.shape[0])
        self.apex = int(rod.get("apex_vertex", self.n // 2))
        self.mass = mass

        edges = np.diff(position, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        edge_dir = edges / lengths[:, None]
        self.lengths = lengths
        node_length = 0.5 * (lengths[:-1] + lengths[1:])
        self.node_length = node_length
        self.inv_node_length = np.empty(self.n)
        self.inv_node_length[0] = 1.0 / node_length[0]
        self.inv_node_length[-1] = 1.0 / node_length[-1]
        self.inv_node_length[1:-1] = 1.0 / node_length

        tangent = np.empty_like(position)
        tangent[0] = edge_dir[0]
        tangent[-1] = edge_dir[-1]
        tangent[1:-1] = _unit(edge_dir[:-1] + edge_dir[1:])
        self.tangent = tangent

        # reference material frame: minimal-rotation transport of the initial
        # in-plane transverse direction (the wide face of the flat strip lies in
        # the plane of the V, so the ribbon is untwisted at theta = 0).
        ref1 = np.empty_like(position)
        ref1[0] = _unit(np.cross(_Y, tangent[0]))
        for i in range(1, self.n):
            cross = np.cross(tangent[i - 1], tangent[i])
            sine = np.linalg.norm(cross)
            cosine = float(np.clip(np.dot(tangent[i - 1], tangent[i]), -1.0, 1.0))
            if sine < 1e-12:
                carried = ref1[i - 1]
            else:
                carried = _rotate(ref1[i - 1], cross / sine, np.arctan2(sine, cosine))
            carried = carried - np.dot(carried, tangent[i]) * tangent[i]
            ref1[i] = _unit(carried)
        ref2 = np.cross(tangent, ref1)
        self.ref1 = ref1
        self.ref2 = ref2

        # discrete curvature binormal and frozen natural curvature
        kappa_b = np.zeros_like(position)
        for i in range(1, self.n - 1):
            cross = np.cross(tangent[i - 1], tangent[i])
            denom = (1.0 + float(np.dot(tangent[i - 1], tangent[i]))) * node_length[i - 1]
            if abs(denom) > 1e-14:
                kappa_b[i] = 2.0 * cross / denom
        self.kappa_b = kappa_b
        self.nat1 = np.einsum("ij,ij->i", kappa_b, ref1)
        self.nat2 = np.einsum("ij,ij->i", kappa_b, ref2)

        width = float(rod.get("cross_section_width", 0.026))
        thick = float(rod.get("cross_section_thickness", 0.006))
        density = float(rod.get("density", 1050.0))
        young = 2.0e9
        poisson = 0.35
        shear = young / (2.0 * (1.0 + poisson))
        inertia1 = width * thick**3 / 12.0
        inertia2 = thick * width**3 / 12.0
        torsion = (width * thick**3 / 3.0) * (1.0 - 0.63 * thick / width)
        self.B1 = young * inertia1
        self.B2 = young * inertia2
        self.T = shear * torsion
        self.width = width
        self.thick = thick
        self.density = density
        self.area = width * thick

        # driven-end rotation schedule (piecewise linear in time, held outside)
        schedule = np.asarray(
            bc.get("driven_end", {}).get("rotation_schedule", [[0.0, 0.0]]), dtype=float
        )
        self.schedule_time = schedule[:, 0]
        self.schedule_angle = schedule[:, 1]
        self.total_angle = float(self.schedule_angle[-1])
        self.fixed_vertex = int(bc.get("fixed_end", {}).get("vertex", self.n - 1))
        self.driven_vertex = int(bc.get("driven_end", {}).get("vertex", 0))

        self.reference_force = max(1e-9, self.B2 / float(np.sum(lengths)) ** 2)
        self.reference_torque = max(1e-12, self.T / float(np.sum(lengths)))
        self.constraint_scale = max(1e-9, abs(self.total_angle)) if self.total_angle else 1.0

        self.theta = np.zeros(self.n)
        self.time = 0.0
        self._last_gradient = np.zeros(self.n)
        self._solve_count = 0
        self._cache: dict[float, int] = {}

    # ---------------------------------------------------------------- helpers
    def driven_angle(self, time: float) -> float:
        return float(
            np.interp(
                float(time),
                self.schedule_time,
                self.schedule_angle,
                left=float(self.schedule_angle[0]),
                right=float(self.schedule_angle[-1]),
            )
        )

    def _frame(self, theta):
        c = np.cos(theta)[:, None]
        s = np.sin(theta)[:, None]
        d1 = c * self.ref1 + s * self.ref2
        d2 = -s * self.ref1 + c * self.ref2
        return d1, d2

    def _energy_gradient_hessian(self, theta):
        """Energy, gradient and Hessian of E over all material angles."""
        d1, d2 = self._frame(theta)
        k1 = np.einsum("ij,ij->i", self.kappa_b, d1)
        k2 = np.einsum("ij,ij->i", self.kappa_b, d2)
        inv = self.inv_node_length
        r1 = k1 - self.nat1
        r2 = k2 - self.nat2
        energy = float(np.sum(0.5 * inv * (self.B1 * r1**2 + self.B2 * r2**2)))

        grad = inv * (self.B1 * r1 * k2 - self.B2 * r2 * k1)
        diag = inv * (
            self.B1 * (k2**2 - k1**2 + self.nat1 * k1)
            + self.B2 * (k1**2 - k2**2 + self.nat2 * k2)
        )

        twist = np.diff(theta)
        tau = twist / self.lengths
        seg_torque = self.T * tau / self.lengths
        energy += float(np.sum(0.5 * self.T * tau**2))
        grad[1:] += seg_torque
        grad[:-1] -= seg_torque
        diag[1:] += self.T / self.lengths**2
        diag[:-1] += self.T / self.lengths**2
        off = -self.T / self.lengths**2
        return energy, grad, diag, off

    def _energy(self, theta) -> float:
        return self._energy_gradient_hessian(theta)[0]

    # ------------------------------------------------------------- equilibria
    def _newton(self, theta_start, driven_angle, max_iterations=None):
        """Damped Newton solve for the interior material angles."""
        theta = theta_start.copy()
        theta[0] = driven_angle
        theta[-1] = 0.0
        lo, hi = 1, self.n - 1
        iterations = self.max_newton_iterations if max_iterations is None else max_iterations
        damping = 0.0
        for _ in range(iterations):
            energy, grad, diag, off = self._energy_gradient_hessian(theta)
            interior_grad = grad[lo:hi]
            scale = np.max(np.abs(interior_grad)) if interior_grad.size else 0.0
            ref = max(1.0, abs(energy))
            if scale <= 1e-9 * self.reference_torque:
                return theta, True
            size = hi - lo
            matrix = np.diag(diag[lo:hi].copy())
            if size > 1:
                idx = np.arange(size - 1)
                matrix[idx, idx + 1] = off[lo : hi - 1]
                matrix[idx + 1, idx] = off[lo : hi - 1]
            step = None
            local_damping = damping
            for _attempt in range(12):
                try:
                    step = np.linalg.solve(
                        matrix + local_damping * np.eye(size), -interior_grad
                    )
                except np.linalg.LinAlgError:
                    local_damping = max(2.0 * local_damping, 1e-6)
                    continue
                if not np.all(np.isfinite(step)):
                    local_damping = max(2.0 * local_damping, 1e-6)
                    continue
                if step.size and np.max(np.abs(step)) > 3.0:
                    step = step * (3.0 / np.max(np.abs(step)))
                alpha = 1.0
                accepted = False
                for _backtrack in range(30):
                    trial = theta.copy()
                    trial[lo:hi] += alpha * step
                    if self._energy(trial) <= energy + 1e-14 * ref:
                        accepted = True
                        break
                    alpha *= 0.5
                if accepted:
                    break
                local_damping = max(2.0 * local_damping, 1e-6)
            if step is None or not accepted:
                return theta, False
            theta = trial
            damping = local_damping * 0.25
        _, grad, _, _ = self._energy_gradient_hessian(theta)
        scale = np.max(np.abs(grad[lo:hi])) if hi > lo else 0.0
        return theta, bool(scale <= 1e-7 * self.reference_torque)

    def _relax(self, theta_start, driven_angle, steps=4000):
        """Robust fallback: monotone gradient descent."""
        theta = theta_start.copy()
        theta[0] = driven_angle
        theta[-1] = 0.0
        lo, hi = 1, self.n - 1
        rate = 1e-3 / max(self.T, 1e-12)
        for _ in range(steps):
            energy, grad, diag, off = self._energy_gradient_hessian(theta)
            interior = grad[lo:hi]
            if (
                interior.size == 0
                or np.max(np.abs(interior)) <= 1e-9 * self.reference_torque
            ):
                break
            direction = -interior
            scale = float(np.max(np.abs(direction)))
            if scale > 0.0:
                direction = direction / scale
            alpha = rate * (1.0 + abs(energy))
            for _backtrack in range(40):
                trial = theta.copy()
                trial[lo:hi] += alpha * direction
                if self._energy(trial) <= energy:
                    break
                alpha *= 0.5
            theta = trial
        return theta

    def _advance_angle(self, target: float) -> None:
        """Equilibrate the rod from the current driven angle to ``target``."""
        current = float(self.theta[0])
        step = 0.08
        guard = 0
        while abs(target - current) > 1e-12:
            guard += 1
            if guard > 20000:
                # safety valve: relax directly to the requested angle
                self.theta = self._relax(self.theta, target, steps=20000)
                self._solve_count += 1
                break
            delta = target - current
            trial_angle = current + float(np.sign(delta)) * min(abs(delta), step)
            candidate, converged = self._newton(self.theta, trial_angle)
            if converged:
                self.theta = candidate
                current = trial_angle
                self._solve_count += 1
                step = min(self.max_continuation_step, step * 1.4)
                continue
            step *= 0.5
            if step < 2e-4:
                self.theta = self._relax(self.theta, trial_angle)
                current = trial_angle
                self._solve_count += 1
                step = 0.05
        _, grad, _, _ = self._energy_gradient_hessian(self.theta)
        self._last_gradient = grad

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        time = float(time)
        target = self.driven_angle(time)
        if abs(target - float(self.theta[0])) > 1e-12:
            self._advance_angle(target)
            self._solve_count += 1
        self.time = time
        return self._make_frame(time)

    def _make_frame(self, time: float, index: int | None = None) -> StateFrame:
        theta = self.theta
        d1, d2 = self._frame(theta)
        orientation = _quats_from_frames(d1, d2, self.tangent)
        n = self.n
        position = self.position
        velocity = np.zeros_like(position)
        angular_velocity = np.zeros_like(position)
        spin = np.zeros(3)
        if n > 1:
            dt = 1e-4
            rate = (self.driven_angle(time + dt) - self.driven_angle(time - dt)) / (2.0 * dt)
            spin = rate * self.tangent[self.driven_vertex]
        angular_velocity[self.driven_vertex] = spin

        # generalized material torques (conjugate to the frame angles)
        _, grad, _, _ = self._energy_gradient_hessian(theta)
        couple_driven = grad[self.driven_vertex] * self.tangent[self.driven_vertex]
        couple_fixed = grad[self.fixed_vertex] * self.tangent[self.fixed_vertex]

        mass_quantity = self.mass[:, None]
        external_force = np.zeros((2, 1, 3))
        force_position = np.stack(
            (
                position[self.driven_vertex][None, None, :],
                position[self.fixed_vertex][None, None, :],
            ),
            axis=0,
        )
        external_couple = np.stack(
            (couple_driven[None, None, :], couple_fixed[None, None, :]), axis=0
        )
        driven_target = position[self.driven_vertex].copy()
        position_error = float(
            np.linalg.norm(position[self.driven_vertex] - driven_target)
        )
        constraint_value = np.array([theta[self.driven_vertex], position_error, theta[self.fixed_vertex]])
        constraint_target = np.array([self.driven_angle(time), 0.0, 0.0])

        fields = {
            "budget_external_force": external_force,
            "budget_force_position": force_position,
            "budget_external_couple": external_couple,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": np.zeros(1),
        }
        attributes = {
            "arc_length": np.r_[0.0, np.cumsum(self.lengths)],
            "material_twist": theta.copy(),
            "cross_section_width": np.full(n, self.width),
            "cross_section_thickness": np.full(n, self.thick),
            "is_apex": (np.arange(n) == self.apex).astype(float),
            "is_driven_end": (np.arange(n) == self.driven_vertex).astype(float),
            "is_fixed_end": (np.arange(n) == self.fixed_vertex).astype(float),
            "mass_quantity": mass_quantity,
        }
        scalars = {
            "driven_end_angle": float(self.driven_angle(time)),
            "constraint_scale": float(self.constraint_scale),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
        }
        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        _, grad, _, _ = self._energy_gradient_hessian(self.theta)
        return {
            "model": "discrete_elastic_rod_quasi_static",
            "material_angles": self.theta.tolist(),
            "max_gradient_residual": float(np.max(np.abs(grad[1:-1]))) if self.n > 2 else 0.0,
            "equilibrium_solves": int(self._solve_count),
            "bend_stiffness": float(self.B1),
            "bend_stiffness_inplane": float(self.B2),
            "torsion_stiffness": float(self.T),
        }
