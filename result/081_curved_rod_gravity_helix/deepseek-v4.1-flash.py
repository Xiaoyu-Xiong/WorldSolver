"""Physical solver for scene 081: gravity-induced spatial curling of a curved rod.

Model
-----
A discrete Cosserat (Kirchhoff) rod:

* ``N`` material nodes carrying the mass, connected by ``N-1`` rigid-ish segments
  whose orientation is stored as a unit quaternion ``q[j]`` (material -> world).
* Elastic energy
    - unshearable/inextensible penalty  ``0.5 * ks * |e_j - l0 * d3_j|^2``,
    - bend/twist strain relative to the frozen natural shape
      ``0.5 * (EI1*du1^2 + EI2*du2^2 + GJ*du3^2) * l0`` with
      ``du = (k_j - k_j_ref)/l0`` and ``k_j = log(R_{j+1}^T R_j)``.
      The reference rotation vectors ``k_ref`` are measured on the *natural*
      circular arc with the same discrete operator, so the natural arc is a
      zero-strain configuration by construction (planar natural curvature,
      zero natural twist).
* Gravity is ramped with the frozen ``gravity_ramp`` schedule.
* Time stepping: explicit velocity-Verlet-type damped dynamics
  (semi-implicit viscous damping, unconditionally stable in the damping term),
  which relaxes the rod onto the quasi-static three-dimensional equilibrium
  branch selected by the frozen branch-selection perturbation.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen modelling constants (public configuration fixes geometry, density and
# gravity; the elastic modulus of the rod material is not part of the public
# configuration and is therefore selected here once, for all frames).
YOUNGS_MODULUS = 2.5e5  # Pa, soft elastomer rod
POISSON_RATIO = 0.5
DAMPING_RATE = 5.0  # 1/s, viscous drag (quasi-static relaxation)
SUBSTEPS_PER_SAMPLE = 1000


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (world <- material) for unit quaternions (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=float)
    m[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[:, 0, 1] = 2.0 * (x * y - w * z)
    m[:, 0, 2] = 2.0 * (x * z + w * y)
    m[:, 1, 0] = 2.0 * (x * y + w * z)
    m[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[:, 1, 2] = 2.0 * (y * z - w * x)
    m[:, 2, 0] = 2.0 * (x * z - w * y)
    m[:, 2, 1] = 2.0 * (y * z + w * x)
    m[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Unit quaternions (w,x,y,z) from rotation matrices (Shepperd's method)."""
    n = m.shape[0]
    q = np.empty((n, 4), dtype=float)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(n):
        mi = m[i]
        t = trace[i]
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            q[i, 0] = 0.25 * s
            q[i, 1] = (mi[2, 1] - mi[1, 2]) / s
            q[i, 2] = (mi[0, 2] - mi[2, 0]) / s
            q[i, 3] = (mi[1, 0] - mi[0, 1]) / s
        elif mi[0, 0] > mi[1, 1] and mi[0, 0] > mi[2, 2]:
            s = np.sqrt(1.0 + mi[0, 0] - mi[1, 1] - mi[2, 2]) * 2.0
            q[i, 0] = (mi[2, 1] - mi[1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (mi[0, 1] + mi[1, 0]) / s
            q[i, 3] = (mi[0, 2] + mi[2, 0]) / s
        elif mi[1, 1] > mi[2, 2]:
            s = np.sqrt(1.0 + mi[1, 1] - mi[0, 0] - mi[2, 2]) * 2.0
            q[i, 0] = (mi[0, 2] - mi[2, 0]) / s
            q[i, 1] = (mi[0, 1] + mi[1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (mi[1, 2] + mi[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mi[2, 2] - mi[0, 0] - mi[1, 1]) * 2.0
            q[i, 0] = (mi[1, 0] - mi[0, 1]) / s
            q[i, 1] = (mi[0, 2] + mi[2, 0]) / s
            q[i, 2] = (mi[1, 2] + mi[2, 1]) / s
            q[i, 3] = 0.25 * s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty_like(p)
    out[:, 0] = pw * qw - px * qx - py * qy - pz * qz
    out[:, 1] = pw * qx + px * qw + py * qz - pz * qy
    out[:, 2] = pw * qy - px * qz + py * qw + pz * qx
    out[:, 3] = pw * qz + px * qy - py * qx + pz * qw
    return out


def _relative_log(q: np.ndarray) -> np.ndarray:
    """Rotation vectors log(R_{j+1}^T R_j) in the material frame of segment j+1."""
    conj = q[1:].copy()
    conj[:, 1:] *= -1.0
    rel = _quat_mul(conj, q[:-1])
    neg = rel[:, 0] < 0.0
    rel[neg] *= -1.0
    v = rel[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    w = np.clip(rel[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arctan2(nv, w)
    scale = np.where(nv > 1e-12, angle / np.maximum(nv, 1e-12), 2.0)
    return v * scale[:, None]


class BenchmarkSolver:
    """Damped discrete-Cosserat-rod solver on the frozen benchmark scene."""

    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None)
        if config is None:
            raise ValueError("BenchmarkSolver requires the scene configuration")
        rod = config["rod"]
        initial = config["initial_conditions"]
        self.scene = scene
        self.seed = int(seed)

        self.length = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.kappa0 = float(rod["natural_curvature"])
        self.natural_radius = 1.0 / self.kappa0
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.root = np.asarray(initial["root_position"], dtype=float)
        perturbation = np.asarray(initial["branch_selection_perturbation"], dtype=float)

        schedule = config.get("load_schedule", {})
        self.ramp_start = float(schedule.get("start_time", 0.0))
        self.ramp_full = float(schedule.get("full_load_time", 1.0))

        self.node_count = int(getattr(scene, "vertex_count", 129))
        self.segment_count = self.node_count - 1
        self.l0 = self.length / self.segment_count

        # Material properties.
        self.area = np.pi * self.radius**2
        self.inertia_bending = np.pi * self.radius**4 / 4.0
        self.inertia_torsion = 2.0 * self.inertia_bending
        self.shear_modulus = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        self.EI = YOUNGS_MODULUS * self.inertia_bending
        self.GJ = self.shear_modulus * self.inertia_torsion
        self.k_bend = np.array([self.EI, self.EI, self.GJ])
        self.k_stretch = (YOUNGS_MODULUS * self.area) / self.l0
        self.node_mass = np.full(self.node_count, self.density * self.area * self.l0)
        self.node_mass[0] *= 0.5
        self.node_mass[-1] *= 0.5
        segment_mass = self.density * self.area * self.l0
        self.segment_inertia = segment_mass * (
            self.l0**2 / 12.0 + 0.5 * self.radius**2
        )
        self.gamma = DAMPING_RATE

        # Natural planar circular arc: x(s) = Rc (1 - cos(s/Rc)) e_x - Rc sin(s/Rc) e_z.
        s = np.linspace(0.0, self.length, self.node_count)
        angle = s / self.natural_radius
        position = np.column_stack(
            (
                self.natural_radius * (1.0 - np.cos(angle)),
                np.zeros_like(s),
                -self.natural_radius * np.sin(angle),
            )
        ) + self.root
        if np.any(perturbation != 0.0):
            shape = np.sin(np.pi * s / self.length)
            position = position + np.outer(shape, perturbation)

        # Natural material frames (d3 tangent, d2 curvature direction, d1 = d2 x d3).
        s_mid = 0.5 * (s[:-1] + s[1:])
        angle_mid = s_mid / self.natural_radius
        d3 = np.column_stack(
            (np.sin(angle_mid), np.zeros_like(angle_mid), -np.cos(angle_mid))
        )
        d2 = np.column_stack(
            (np.cos(angle_mid), np.zeros_like(angle_mid), np.sin(angle_mid))
        )
        d1 = np.cross(d2, d3)
        frames = np.stack((d1, d2, d3), axis=-1)
        self.q_ref = _mat_to_quat(frames)
        self.k_ref = _relative_log(self.q_ref)

        # State.
        self.time = 0.0
        self.position = position
        self.velocity = np.zeros_like(position)
        self.q = self.q_ref.copy()
        self.omega = np.zeros((self.segment_count, 3))
        self.reference_force = float(np.sum(self.node_mass) * np.linalg.norm(self.gravity))
        self.reference_torque = self.reference_force * self.length
        self.last_damping_force = np.zeros_like(position)

    # ------------------------------------------------------------------ forces
    def _ramp(self, time: float) -> float:
        span = max(self.ramp_full - self.ramp_start, 1e-12)
        u = (time - self.ramp_start) / span
        u = min(1.0, max(0.0, u))
        return u * u * (3.0 - 2.0 * u)

    def _elastic(self, x: np.ndarray, q: np.ndarray):
        """Nodal elastic forces, segment torques, stretch residual, bend vectors."""
        frames = _quat_to_mat(q)
        d3 = frames[:, :, 2]
        edge = x[1:] - x[:-1]
        residual = edge - self.l0 * d3

        force = np.zeros_like(x)
        force[:-1] -= self.k_stretch * residual
        force[1:] += self.k_stretch * residual

        torque = -self.k_stretch * self.l0 * np.cross(residual, d3)

        k = _relative_log(q)
        strain = (k - self.k_ref) * self.k_bend / self.l0
        # first-order inverse left Jacobian transpose: (I + 0.5 [k]x) strain
        world = strain + 0.5 * np.cross(k, strain)
        moment = np.einsum("nij,nj->ni", frames[1:], world)
        torque[:-1] -= moment
        torque[1:] += moment
        return force, torque, residual, k

    # --------------------------------------------------------------- stepping
    def _step(self, dt: float, ramp: float) -> None:
        x, v, q, w = self.position, self.velocity, self.q, self.omega
        force, torque, _, _ = self._elastic(x, q)

        gravity_force = self.node_mass[:, None] * (self.gravity * ramp)
        force = force + gravity_force

        v = (v + dt * force / self.node_mass[:, None]) / (1.0 + dt * self.gamma)
        v[0] = 0.0
        x = x + dt * v
        x[0] = self.root

        w = (w + dt * torque / self.segment_inertia) / (1.0 + dt * self.gamma)
        w[0] = 0.0
        half = 0.5 * dt * w
        norm = np.linalg.norm(half, axis=1)
        delta = np.empty_like(half)
        delta[:, 0] = np.cos(norm)
        delta[:, 1:] = half * np.where(norm > 1e-12, np.sin(norm) / np.maximum(norm, 1e-12), 1.0)[:, None]
        q = _quat_mul(delta, q)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q[0] = self.q_ref[0]

        self.position = x
        self.velocity = v
        self.q = q
        self.omega = w
        self.last_damping_force = -self.gamma * self.node_mass[:, None] * v
        self.time += dt

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing times")
        remaining = target - self.time
        if remaining > 0.0:
            steps = int(round(remaining / (0.02 / SUBSTEPS_PER_SAMPLE)))
            steps = max(steps, 1)
            dt = remaining / steps
            for step in range(steps):
                ramp = self._ramp(self.time + 0.5 * dt)
                self._step(dt, ramp)
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------ output
    def _node_frames(self):
        n = self.node_count
        idx = np.arange(n)
        left = self.q[np.clip(idx - 1, 0, self.segment_count - 1)]
        right = self.q[np.clip(idx, 0, self.segment_count - 1)]
        sign = np.sign(np.einsum("ij,ij->i", left, right))
        sign[sign == 0.0] = 1.0
        avg = left + sign[:, None] * right
        avg /= np.linalg.norm(avg, axis=1, keepdims=True)
        omega = self.omega[np.clip(idx, 0, self.segment_count - 1)]
        return avg, omega

    def _frame(self, time: float) -> StateFrame:
        x = self.position
        v = self.velocity
        q = self.q
        ramp = self._ramp(time)
        force_elastic, _, _, _ = self._elastic(x, q)
        gravity_force = self.node_mass[:, None] * (self.gravity * ramp)
        damping_force = -self.gamma * self.node_mass[:, None] * v

        # Clamp reaction: whatever the fixed root must supply so that the total
        # external force on the rod balances the rate of change of momentum.
        # F_clamp = sum over free nodes (elastic + damping) - weight of the
        # clamped boundary node, so that sum(external forces) = dP/dt exactly.
        clamp_force = (
            np.sum((force_elastic + damping_force)[1:], axis=0) - gravity_force[0]
        )

        budget_force = np.vstack((gravity_force, clamp_force[None, :]))
        budget_position = np.vstack((x, self.root[None, :]))
        budget_couple = np.zeros_like(budget_force)

        segment_lengths = np.linalg.norm(x[1:] - x[:-1], axis=1)
        constraint_value = np.array([float(np.sum(segment_lengths))])

        node_orientation, node_omega = self._node_frames()
        s = np.linspace(0.0, self.length, self.node_count)

        frame = StateFrame(
            time=time,
            position=x.copy(),
            velocity=v.copy(),
            orientation=np.column_stack(
                (node_orientation[:, 1], node_orientation[:, 2],
                 node_orientation[:, 3], node_orientation[:, 0])
            ),
            angular_velocity=node_omega.copy(),
            mass=self.node_mass.copy(),
            attributes={
                "mass_quantity": self.node_mass.copy(),
                "arc_length": s,
                "natural_curvature": np.full(self.node_count, self.kappa0),
                "natural_twist": np.zeros(self.node_count),
                "is_clamped_root": np.r_[1.0, np.zeros(self.node_count - 1)],
            },
            fields={
                "budget_external_force": budget_force,
                "budget_force_position": budget_position,
                "budget_external_couple": budget_couple,
                "constraint_value": constraint_value,
                "constraint_target": np.array([self.length]),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": self.length,
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
                "rod_radius": self.radius,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "success": True,
            "node_count": self.node_count,
            "time": self.time,
            "youngs_modulus": YOUNGS_MODULUS,
            "bending_stiffness": self.EI,
            "torsion_stiffness": self.GJ,
            "max_speed": float(np.max(np.linalg.norm(self.velocity[1:], axis=1))),
        }
