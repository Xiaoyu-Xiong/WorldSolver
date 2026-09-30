"""Quasi-static, inextensible elastic-rod solver for the cantilever scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rod = cfg["rod"]
        root_bc = cfg["boundary_conditions"]["root"]

        self.n = int(scene.vertex_count)
        self.ns = self.n - 1
        self.length = float(rod["length"])
        self.ds = self.length / self.ns
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.area = np.pi * self.radius**2
        self.total_mass = self.density * self.area * self.length
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.root = np.asarray(root_bc["position"], dtype=float)
        tangent = np.asarray(root_bc["tangent"], dtype=float)
        self.root_angle = float(np.arctan2(tangent[2], tangent[0]))
        self.full_load_time = float(cfg["load_schedule"]["full_load_time"])

        # Circular-section Euler-Bernoulli bending stiffness.  This elastomer
        # modulus produces a pronounced but smooth gravity-loaded cantilever.
        young_modulus = 2.0e6
        second_moment = np.pi * self.radius**4 / 4.0
        self.bending_stiffness = young_modulus * second_moment

        segment_mass = self.total_mass / self.ns
        self.mass = np.full(self.n, segment_mass, dtype=float)
        self.mass[[0, -1]] *= 0.5
        self.arc_length = np.linspace(0.0, self.length, self.n)
        self.theta = np.full(self.ns, self.root_angle, dtype=float)
        self.last_time = 0.0
        self.last_position = self._positions(self.theta)
        self._iterations = 0
        self._initialized = True

    def _positions(self, theta: np.ndarray) -> np.ndarray:
        """Reconstruct an exactly inextensible centerline from segment angles."""
        pos = np.empty((self.n, 3), dtype=float)
        pos[0] = self.root
        increments = np.zeros((self.ns, 3), dtype=float)
        increments[:, 0] = self.ds * np.cos(theta)
        increments[:, 2] = self.ds * np.sin(theta)
        pos[1:] = self.root + np.cumsum(increments, axis=0)
        return pos

    def _load_fraction(self, time: float) -> float:
        if self.full_load_time <= 0.0:
            return 1.0
        return float(np.clip(time / self.full_load_time, 0.0, 1.0))

    def _relax(self, load: float) -> None:
        """Damped pseudo-time stepping toward discrete-elastica equilibrium."""
        theta = self.theta.copy()
        theta[0] = self.root_angle
        k = self.bending_stiffness
        weight = self.total_mass * (-self.gravity[2]) * load
        seg_weight = weight / self.ns
        base_step = 0.18 * self.ds / max(k, 1.0e-12)
        downstream = np.arange(self.ns, 0, -1, dtype=float) - 0.5

        def energy_gradient(q: np.ndarray) -> tuple[float, np.ndarray]:
            bend = np.diff(q)
            energy = 0.5 * k / self.ds * float(np.dot(bend, bend))
            grad = np.zeros_like(q)
            grad[:-1] -= (k / self.ds) * bend
            grad[1:] += (k / self.ds) * bend
            energy += seg_weight * self.ds * float(np.dot(downstream, np.sin(q)))
            grad += seg_weight * self.ds * downstream * np.cos(q)
            grad[0] = 0.0
            return energy, grad

        energy, grad = energy_gradient(theta)
        iteration = 0
        for iteration in range(6000):
            if float(np.max(np.abs(grad[1:]))) < 2.0e-9:
                break
            step = base_step
            accepted = False
            for _ in range(14):
                trial = theta - step * grad
                trial[0] = self.root_angle
                trial = np.clip(trial, -1.55, 1.20)
                trial_energy, trial_grad = energy_gradient(trial)
                if trial_energy <= energy:
                    theta, energy, grad = trial, trial_energy, trial_grad
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        self.theta = theta
        self._iterations += iteration + 1

    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        time = float(time)
        if not np.isfinite(time) or time < self.last_time - 1.0e-12:
            raise ValueError("sample times must be finite and nondecreasing")

        old_position = self.last_position.copy()
        load = self._load_fraction(time)
        self._relax(load)
        position = self._positions(self.theta)
        dt = time - self.last_time
        velocity = ((position - old_position) / dt if dt > 0.0
                    else np.zeros_like(position))

        gravity_force = self.total_mass * self.gravity * load
        center_of_mass = np.sum(position * self.mass[:, None], axis=0) / self.total_mass
        reaction_force = -gravity_force
        reaction_moment = -np.cross(center_of_mass - self.root, gravity_force)

        # One rod budget domain; terms are distributed gravity and clamp reaction.
        external_force = np.zeros((1, 2, 3), dtype=float)
        external_force[0, 0] = gravity_force
        external_force[0, 1] = reaction_force
        force_position = np.zeros((1, 2, 3), dtype=float)
        force_position[0, 0] = center_of_mass
        force_position[0, 1] = self.root
        external_couple = np.zeros((1, 2, 3), dtype=float)
        external_couple[0, 1] = reaction_moment

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "is_clamped_root": np.r_[1.0, np.zeros(self.n - 1)],
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_couple": external_couple,
                "budget_external_force": external_force,
                "budget_force_position": force_position,
                "constraint_target": np.concatenate((self.root, [self.root_angle])),
                "constraint_value": np.concatenate((position[0], [self.theta[0]])),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "rod_radius": self.radius,
                "constraint_scale": self.length,
                "reference_force": self.total_mass * float(np.linalg.norm(self.gravity)),
                "reference_torque": self.total_mass * float(np.linalg.norm(self.gravity)) * self.length,
            },
        )
        frame.validate()
        self.last_time = time
        self.last_position = position
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "inextensible planar discrete elastica",
            "bending_stiffness": float(self.bending_stiffness),
            "relaxation_iterations": int(self._iterations),
            "tip_height": float(self.last_position[-1, 2]) if self._initialized else None,
        }
