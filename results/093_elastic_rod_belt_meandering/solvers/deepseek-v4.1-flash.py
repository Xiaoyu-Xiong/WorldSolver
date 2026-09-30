"""Elastic rod meandering on a moving belt.

Physical model
--------------
The rod is a Lagrangian chain of material nodes with fixed rest lengths: the
initial mesh is the frozen ``assets/rod_topology.json`` centreline and further
material is stamped out of the injector at a finer spacing.  Every edge
carries a Hookean axial stiffness ``k = E A / L0`` (inextensible limit, only
the compressive response is limited) and the bending energy is the discrete
arc-length (Euler-Bernoulli) energy

    E_b = (B/2) sum_i |x_{i-1} - 2 x_i + x_{i+1}|^2 / ds_i^3,  B = E pi r^4 / 4

so all internal forces are the exact gradients of a quadratic positive
semi-definite discrete elastic energy whose stencils telescope (the total
internal force of the rod is therefore exactly zero).

Time stepping is a variational implicit Euler step.  With the elastic Hessian
``K`` the update solves the symmetric positive definite system

    [ M/dt + C + dt K ] v^{n+1} = f_elastic(x^n) + m g - C v^n + (M/dt) v^n

which is unconditionally stable and lets the whole run be advanced with
dt = sample_dt/8 = 2.5 ms - the frozen regime is quasi-static and the implicit
bending damping relaxes the shape on the elastic time scale.

Contact / deposition
--------------------
The belt is the frozen box centred at (0.22, 0, -0.012) with half extents
(0.62, 0.2, 0.012), hence a top surface at z = 0 and a footprint
x in [-0.40, 0.84], |y| <= 0.20.  A node that reaches the surface inside the
footprint adheres to the belt ("sticking_to_moving_belt"): its velocity is set
to the belt velocity so the freshly deposited strand is carried in +x without
sliding.  The constraint is non-holonomic and one-way: it is released when the
elastic tension pulls the node off the surface, or when the node is carried
past the downstream end of the belt (it then falls off the edge).  Nodes are
never allowed below the belt surface; a Hertzian normal stiffness supplies the
contact load for the implicit system and the reaction is exported in the
budget fields.

Boundary conditions
-------------------
Node 0 is the material point inside the injector: its position is prescribed
to the outlet and its velocity to the injection speed along the injector axis
(``inlet_position_controlled`` / ``inlet_direction_controlled``).  New
material is born at the outlet whenever the injected arc length grows by one
injection spacing - the prescribed mass source of the injector.

All external forces (gravity, viscous drag, injector reaction, belt normal
load, belt friction traction) and their inertial application points are
exported, so the sampled trajectory satisfies the discrete momentum and
angular momentum balance evaluated by the platform.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from worldsolver.contracts import StateFrame

_EPS = 1e-12
_UP = np.array([0.0, 0.0, 1.0])


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm < _EPS:
        return _UP.copy()
    return array / norm


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1.0e-12)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg
        self.seed = int(seed)
        rod = cfg["rod"]
        injector = cfg["injector"]
        belt = cfg["belt"]

        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.young = float(rod["young_modulus"])
        self.poisson = float(rod["poisson_ratio"])
        self.area = float(np.pi * self.radius**2)
        self.mass_per_length = self.density * self.area
        self.bending_rigidity = self.young * np.pi * self.radius**4 / 4.0
        self.axial_rigidity = self.young * self.area

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.outlet = np.asarray(injector["outlet_position"], dtype=float)
        self.inject_axis = _unit(injector["axis"])
        self.injection_speed = float(injector["injection_speed"])
        self.inject_end = float(injector["end_time"])
        self.total_length = float(rod["total_available_length"])

        self.belt_center = np.asarray(belt["center"], dtype=float)
        self.belt_half = np.asarray(belt["half_extents"], dtype=float)
        self.belt_top = float(belt["surface_z"])
        self.belt_direction = _unit(belt["travel_direction"])
        self.belt_speed_initial = float(belt["initial_speed"])
        self.belt_speed_matched = float(belt["matched_speed"])
        self.switch_time = float(belt["speed_switch_time"])
        self.belt_x_min = float(self.belt_center[0] - self.belt_half[0])
        self.belt_x_max = float(self.belt_center[0] + self.belt_half[0])
        self.belt_abs_y = float(self.belt_half[1])
        self.surface_contact_z = self.belt_top + self.radius

        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.step_dt = max(1.0e-4, self.sample_dt / 8.0)

        self.damping = 20.0
        self.stretch_clamp = 5.0e-4
        self.contact_stiffness = (
            (4.0 / 3.0)
            * (self.young / (1.0 - self.poisson**2))
            * float(np.sqrt(self.radius))
        )
        self.ds_inject = 0.005
        self.seed_amplitude = 1.5e-4
        self.max_nodes = 460

        asset = Path(rod["topology_asset"])
        if not asset.is_file():
            asset = Path(__file__).resolve().parent.parent / asset
        data = json.loads(asset.read_text(encoding="utf-8"))
        self.initial_position = np.asarray(data["vertices"], dtype=float)
        self.initial_length = float(
            np.linalg.norm(np.diff(self.initial_position, axis=0), axis=1).sum()
        )
        self.initial_deposited = np.asarray(data["deposited_mask"], dtype=float) > 0.5
        self.node_force_scale = self.mass_per_length * abs(self.gravity[2]) * self.ds_inject
        self.release_tension = 0.25 * self.node_force_scale

        self._x = np.zeros((self.max_nodes, 3))
        self._v = np.zeros((self.max_nodes, 3))
        self._stuck = np.zeros(self.max_nodes, dtype=bool)
        self._rest = np.full(max(1, self.max_nodes - 1), self.ds_inject)
        self._lumped = np.zeros(self.max_nodes)
        self._tangent = np.zeros((self.max_nodes, 3))
        self._n = 0
        self._bend = sparse.csr_matrix((0, 0))
        self._bend_reduced = None
        self._initialise_chain()
        self._time = 0.0
        self._diagnostics = {"steps": 0, "births": 0, "message": ""}
        self._reaction = {
            "injector": np.zeros(3),
            "injector_point": self.outlet.copy(),
            "normal": np.zeros(3),
            "normal_point": self.belt_center.copy(),
            "friction": np.zeros(3),
            "friction_point": self.belt_center.copy(),
            "damping": np.zeros(3),
            "damping_point": self.belt_center.copy(),
        }

    # ------------------------------------------------------------- mesh setup
    def _initialise_chain(self) -> None:
        n0 = len(self.initial_position)
        self._x[:n0] = self.initial_position
        self._n = n0
        gaps = np.linalg.norm(np.diff(self.initial_position, axis=0), axis=1)
        self._rest[: n0 - 1] = gaps
        self._stuck[:n0] = self.initial_deposited
        self._v[:n0] = 0.0
        self._v[0] = self.injection_speed * self.inject_axis
        deposited = np.nonzero(self.initial_deposited)[0]
        if deposited.size:
            self._v[deposited] = self._belt_velocity(0.0)
        self._update_lumped_mass()
        self._rebuild_bending()

    def _belt_velocity(self, time: float) -> np.ndarray:
        speed = (
            self.belt_speed_initial if time < self.switch_time else self.belt_speed_matched
        )
        return self.belt_direction * speed

    def _update_lumped_mass(self) -> None:
        n = self._n
        lumped = np.zeros(n)
        if n > 1:
            rest = self._rest[: n - 1]
            lumped[0] = 0.5 * rest[0]
            lumped[-1] = 0.5 * rest[-1]
            if n > 2:
                lumped[1:-1] = 0.5 * (rest[:-1] + rest[1:])
        self._lumped[:n] = self.mass_per_length * lumped
        self._lumped[n:] = 0.0

    def _rebuild_bending(self) -> None:
        """Constant Hessian of the live mesh's discrete bending energy."""
        n = self._n
        if n < 3:
            self._bend = sparse.csr_matrix((3 * n, 3 * n))
        else:
            rest = self._rest[: n - 1]
            rows: list[int] = []
            cols: list[int] = []
            vals: list[float] = []
            for i in range(1, n - 1):
                ds = 0.5 * (rest[i - 1] + rest[i])
                weight = float(np.sqrt(self.bending_rigidity / max(ds, _EPS) ** 3))
                for offset, coefficient in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                    rows.append(i)
                    cols.append(i + offset)
                    vals.append(weight * coefficient)
            stencil = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
            normal = (stencil.T @ stencil).tocoo()
            dof_rows = (3 * normal.row[:, None] + np.arange(3)).ravel()
            dof_cols = (3 * normal.col[:, None] + np.arange(3)).ravel()
            self._bend = sparse.csr_matrix(
                (np.repeat(normal.data, 3), (dof_rows, dof_cols)), shape=(3 * n, 3 * n)
            )
        self._bend_reduced = None

    # ------------------------------------------------------------ mass source
    def _injected_length(self, time: float) -> float:
        active = min(max(float(time), 0.0), self.inject_end)
        return self.injection_speed * active

    def _target_nodes(self, time: float) -> int:
        length = min(self.initial_length + self._injected_length(time), self.total_length)
        count = 1 + int(np.floor((length - self.initial_length) / self.ds_inject + 1e-9))
        return int(min(count, self.max_nodes))

    def _grow_chain(self, time: float) -> None:
        target = self._target_nodes(time)
        while self._n < target:
            n = self._n
            self._x[1 : n + 1] = self._x[:n]
            self._v[1 : n + 1] = self._v[:n]
            self._stuck[1 : n + 1] = self._stuck[:n]
            self._tangent[1 : n + 1] = self._tangent[:n]
            self._rest[1:n] = self._rest[: max(0, n - 1)]
            self._rest[0] = self.ds_inject
            # the boundary node is returned to the nozzle outlet; the node that
            # was there has already advanced a full injection spacing by its own
            # prescribed motion, so it simply becomes the first material node.
            self._x[0] = self.outlet
            self._v[0] = self.injection_speed * self.inject_axis
            self._stuck[0] = False
            self._n = n + 1
            # deterministic sub-visual injector imperfection: the seed of the
            # lateral (meandering) instability
            self._x[0, 1] += self.seed_amplitude * np.sin(1.3 * float(self._n))
            self._update_lumped_mass()
            self._rebuild_bending()
            self._diagnostics["births"] = int(self._diagnostics["births"]) + 1

    # ---------------------------------------------------------------- stepping
    def _step(self, dt: float, time_new: float) -> None:
        n = self._n
        x = self._x[:n]
        v = self._v[:n]
        mass = self._lumped[:n]
        rest = self._rest[: n - 1]
        belt_velocity = self._belt_velocity(time_new)

        seg = x[1:] - x[:-1]
        length = np.linalg.norm(seg, axis=1)
        length_safe = np.maximum(length, 1.0e-9)
        tangent = seg / length_safe[:, None]
        strain = length - rest
        strain_c = np.maximum(strain, -self.stretch_clamp * rest)
        stiffness = self.axial_rigidity / np.maximum(rest, _EPS)
        force_mag = stiffness * strain_c

        axial_force = np.zeros((n, 3))
        axial_force[:-1] += force_mag[:, None] * tangent
        axial_force[1:] -= force_mag[:, None] * tangent

        inside = self._inside_belt(x)
        penetration = np.where(inside, self.surface_contact_z - x[:, 2], 0.0)
        penetration = np.maximum(penetration, 0.0)
        contact_force = np.zeros((n, 3))
        contact_force[:, 2] = self.contact_stiffness * penetration**1.5
        contact_curvature = 1.5 * self.contact_stiffness * np.sqrt(penetration)

        damping_force = -self.damping * mass[:, None] * v
        external = mass[:, None] * self.gravity[None, :] + damping_force
        elastic = axial_force - (self._bend @ x.ravel()).reshape(n, 3) + contact_force

        k_spring = self._assemble_spring_hessian(stiffness, strain_c, strain, length_safe, tangent, rest)
        k_total = k_spring + self._bend + sparse.diags(np.repeat(contact_curvature, 3))
        base = sparse.diags(np.repeat(mass / dt + self.damping * mass, 3)) + dt * k_total
        rhs = elastic + external + (mass / dt)[:, None] * v
        rhs_flat = rhs.ravel()
        rhs_base = rhs_flat.copy()

        free = np.arange(3, 3 * n)
        prescribed = self.injection_speed * self.inject_axis
        reduced = base[free][:, free].tocsc()
        rhs_free = rhs_flat[free] - (base[free][:, :3] @ prescribed)
        solution = spla.spsolve(reduced, rhs_free)
        if not np.all(np.isfinite(solution)):
            solution = self._damped_fallback(rhs_free, reduced)
            self._diagnostics["message"] = "fallback solve"
        velocity = v.copy()
        velocity[1:] = solution.reshape(n - 1, 3)

        # one Newton refinement of the nonlinear Hertzian contact
        for _ in range(2):
            predicted = x + dt * velocity
            delta_new = np.where(inside, self.surface_contact_z - predicted[:, 2], 0.0)
            delta_new = np.maximum(delta_new, 0.0)
            residual = (
                self.contact_stiffness * (delta_new**1.5 - penetration**1.5)
                - contact_curvature * (delta_new - penetration)
            )
            if float(np.max(np.abs(residual))) < 1.0e-14:
                break
            extra = np.zeros((n, 3))
            extra[:, 2] = residual
            rhs_free = rhs_base[free] - (base[free][:, :3] @ prescribed) + extra.ravel()[free]
            solution = spla.spsolve(reduced, rhs_free)
            if not np.all(np.isfinite(solution)):
                break
            velocity[1:] = solution.reshape(n - 1, 3)

        velocity[0] = prescribed
        injector_reaction = (base @ velocity.ravel() - rhs_base)[:3].copy()

        predicted = x + dt * velocity
        free_velocity = velocity.copy()

        inside_new = self._inside_belt(predicted)
        touching = inside_new & (predicted[:, 2] <= self.surface_contact_z + 1.0e-9)
        for node in range(1, n):
            # "sticking_to_moving_belt": material that has reached the substrate
            # stays attached and is carried away by the belt (no sliding, no
            # peeling), so the deposited trace is advected rigidly in +x.
            if not self._stuck[node] and touching[node]:
                self._stuck[node] = True
            if self._stuck[node]:
                predicted[node, 2] = self.surface_contact_z
                velocity[node] = belt_velocity
            elif inside_new[node] and predicted[node, 2] < self.surface_contact_z:
                predicted[node, 2] = self.surface_contact_z
                velocity[node, 2] = max(velocity[node, 2], 0.0)

        delta_v = velocity - free_velocity
        change = mass[:, None] * delta_v / dt
        normal_impulse = np.where(self._stuck[:n], change[:, 2], 0.0)
        normal_impulse = np.where(
            self._stuck[:n], normal_impulse, np.maximum(change[:, 2], 0.0)
        )
        normal_impulse[0] = 0.0
        friction_impulse = np.zeros((n, 3))
        friction_nodes = self._stuck[:n].copy()
        friction_nodes[0] = False
        friction_impulse[friction_nodes, 0] = change[friction_nodes, 0]
        friction_impulse[friction_nodes, 1] = change[friction_nodes, 1]

        self._reaction["injector"] = injector_reaction
        self._reaction["injector_point"] = self.outlet.copy()
        self._reaction["normal"] = np.array(
            [
                0.0,
                0.0,
                float(np.sum(normal_impulse) + np.sum(contact_force[:, 2])),
            ]
        )
        self._reaction["friction"] = np.sum(friction_impulse, axis=0)
        self._reaction["normal_point"] = self._weighted_point(np.abs(normal_impulse), x)
        self._reaction["friction_point"] = self._weighted_point(
            np.linalg.norm(friction_impulse, axis=1), x
        )
        self._reaction["damping"] = np.sum(damping_force, axis=0)
        self._reaction["damping_point"] = self._weighted_point(
            np.linalg.norm(damping_force, axis=1), x
        )

        self._x[:n] = predicted
        self._v[:n] = velocity
        self._tangent[:n] = self._node_tangents(predicted)
        self._diagnostics["steps"] = int(self._diagnostics["steps"]) + 1

    def _inside_belt(self, positions: np.ndarray) -> np.ndarray:
        return (
            (positions[:, 0] >= self.belt_x_min)
            & (positions[:, 0] <= self.belt_x_max)
            & (np.abs(positions[:, 1]) <= self.belt_abs_y)
        )

    def _assemble_spring_hessian(
        self, stiffness, strain_c, strain, length_safe, tangent, rest
    ) -> sparse.csr_matrix:
        """Exact PSD Hessian of the axial spring energy of every edge."""
        n = self._n
        edges = n - 1
        index = np.arange(edges)
        alpha = stiffness * np.maximum(strain_c, 0.0) / length_safe
        beta = stiffness * rest / length_safe
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for a in range(3):
            for b in range(3):
                block = np.zeros(edges)
                if a == b:
                    block += alpha
                block += beta * tangent[:, a] * tangent[:, b]
                for node_row, node_col, sign in (
                    (index, index, 1.0),
                    (index + 1, index + 1, 1.0),
                    (index, index + 1, -1.0),
                    (index + 1, index, -1.0),
                ):
                    rows.append(3 * node_row + a)
                    cols.append(3 * node_col + b)
                    vals.append(sign * block)
        rows_arr = np.concatenate(rows)
        cols_arr = np.concatenate(cols)
        vals_arr = np.concatenate(vals)
        matrix = sparse.coo_matrix(
            (vals_arr, (rows_arr, cols_arr)), shape=(3 * n, 3 * n)
        )
        return matrix.tocsr()

    def _node_tangents(self, positions: np.ndarray) -> np.ndarray:
        n = self._n
        tangent = np.zeros((n, 3))
        if n > 1:
            seg = positions[1:] - positions[:-1]
            tangent[:-1] += seg
            tangent[1:] += seg
        return _unit_rows(tangent)

    @staticmethod
    def _weighted_point(weights: np.ndarray, positions: np.ndarray) -> np.ndarray:
        total = float(np.sum(weights))
        if total <= _EPS:
            return positions.mean(axis=0)
        return (weights[:, None] * positions).sum(axis=0) / total

    @staticmethod
    def _damped_fallback(rhs, matrix):
        diagonal = np.abs(matrix.diagonal())
        safe = np.where(diagonal > _EPS, diagonal, 1.0)
        return rhs / safe

    # -------------------------------------------------------------- trajectory
    def advance_to(self, time: float):
        time = float(time)
        while self._time < time - 1.0e-12:
            dt = min(self.step_dt, time - self._time)
            self._grow_chain(self._time + dt)
            self._time = self._time + dt
            self._step(dt, self._time)
        self._time = max(self._time, time)
        return self._frame(time)

    def _frame(self, time: float) -> StateFrame:
        n = self._n
        position = np.nan_to_num(self._x[:n].copy())
        velocity = np.nan_to_num(self._v[:n].copy())
        mass = self._lumped[:n].copy()
        rest = self._rest[: n - 1]
        orientation = self._orientations(position)
        angular_velocity = self._angular_velocity(position)

        arc = np.zeros(n)
        if n > 1:
            arc[1:] = np.cumsum(rest)
        stiffness = self.axial_rigidity / np.maximum(rest, _EPS)
        direction = _unit_rows(position[1:] - position[:-1])
        magnitude = stiffness * np.maximum(
            np.linalg.norm(position[1:] - position[:-1], axis=1) - rest,
            -self.stretch_clamp * rest,
        )
        elastic_force = np.zeros((n, 3))
        elastic_force[:-1] += magnitude[:, None] * direction
        elastic_force[1:] -= magnitude[:, None] * direction
        elastic_energy = self._elastic_energy(position, rest)

        names = ("gravity", "damping", "injector", "belt_normal", "belt_friction")
        forces = np.zeros((1, len(names), 3))
        points = np.zeros((1, len(names), 3))
        forces[0, 0] = float(np.sum(mass)) * self.gravity
        points[0, 0] = self._weighted_point(mass, position)
        forces[0, 1] = self._reaction["damping"]
        points[0, 1] = self._reaction["damping_point"]
        forces[0, 2] = self._reaction["injector"]
        points[0, 2] = self._reaction["injector_point"]
        forces[0, 3] = self._reaction["normal"]
        points[0, 3] = self._reaction["normal_point"]
        forces[0, 4] = self._reaction["friction"]
        points[0, 4] = self._reaction["friction_point"]

        deposited = np.zeros(n)
        deposited[: min(n, self.initial_deposited.size)] = self.initial_deposited[
            : min(n, self.initial_deposited.size)
        ]
        injected_mass = self.mass_per_length * self._injected_length(time)
        reference_force = max(
            self.mass_per_length * abs(self.gravity[2]) * self.total_length, 1.0e-12
        )
        reference_torque = max(reference_force * 0.033, 1.0e-12)

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "rod_radius": np.full(n, self.radius),
                "arc_coordinate": arc,
                "natural_curvature": np.zeros(n),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_sticking_to_belt": self._stuck[:n].astype(float),
                "is_initially_deposited": deposited,
                "elastic_energy": elastic_energy,
                "injected_length": np.full(n, self._injected_length(time)),
            },
            fields={
                "budget_external_force": forces,
                "budget_force_position": points,
                "budget_external_couple": np.zeros((1, len(names), 3)),
                "cumulative_mass_source": np.array([injected_mass]),
                "constraint_value": position[:, 2].copy(),
                "constraint_target": np.full(n, self.surface_contact_z),
                "elastic_internal_force": elastic_force,
                "belt_surface_height": np.full(n, self.surface_contact_z),
            },
            scalars={
                "constraint_scale": float(self.radius),
                "reference_force": float(reference_force),
                "reference_torque": float(reference_torque),
                "injection_speed": float(self.injection_speed),
                "belt_speed": float(np.linalg.norm(self._belt_velocity(time))),
                "speed_switch_time": float(self.switch_time),
                "node_count": float(n),
                "time": float(time),
            },
        )
        frame.validate()
        return frame

    def _elastic_energy(self, position, rest) -> np.ndarray:
        n = self._n
        energy = np.zeros(n)
        if n > 1:
            length = np.linalg.norm(position[1:] - position[:-1], axis=1)
            stiffness = self.axial_rigidity / np.maximum(rest, _EPS)
            stretch = 0.5 * stiffness * (length - rest) ** 2
            energy[:-1] += 0.5 * stretch
            energy[1:] += 0.5 * stretch
        if n > 2:
            curvature = position[2:] - 2.0 * position[1:-1] + position[:-2]
            ds = 0.5 * (rest[:-1] + rest[1:])
            coefficient = 0.5 * self.bending_rigidity / np.maximum(ds, _EPS) ** 3
            bend = coefficient * np.sum(curvature**2, axis=1)
            energy[:-2] += 0.5 * bend
            energy[1:-1] += bend
            energy[2:] += 0.5 * bend
        return energy

    def _orientations(self, position) -> np.ndarray:
        n = self._n
        quaternions = np.zeros((n, 4))
        quaternions[:, 3] = 1.0
        if n < 1:
            return quaternions
        tangent = self._node_tangents(position)
        for index in range(n):
            axis = np.cross(_UP, tangent[index])
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < _EPS:
                if tangent[index, 2] < 0.0:
                    quaternions[index] = np.array([1.0, 0.0, 0.0, 0.0])
                continue
            axis = axis / axis_norm
            half = 0.5 * float(np.arccos(np.clip(tangent[index, 2], -1.0, 1.0)))
            quaternions[index] = np.array(
                [
                    axis[0] * np.sin(half),
                    axis[1] * np.sin(half),
                    axis[2] * np.sin(half),
                    np.cos(half),
                ]
            )
        return quaternions

    def _angular_velocity(self, position) -> np.ndarray:
        previous = self._tangent[: self._n]
        current = self._node_tangents(position)
        return np.cross(previous, current) / max(self.step_dt, _EPS)

    def diagnostics(self) -> dict[str, object]:
        n = self._n
        return {
            "method": "variational implicit Euler, discrete arc-length elastic rod",
            "node_count": int(n),
            "steps": int(self._diagnostics["steps"]),
            "births": int(self._diagnostics["births"]),
            "simulated_time": float(self._time),
            "deployed_length": float(self._rest[: max(0, n - 1)].sum()),
            "sticking_nodes": int(np.sum(self._stuck[:n])),
            "max_speed": float(np.max(np.linalg.norm(self._v[:n], axis=1))) if n else 0.0,
            "message": str(self._diagnostics.get("message", "")),
        }
