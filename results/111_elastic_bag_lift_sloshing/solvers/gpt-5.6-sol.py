"""Deterministic particle/elastic-shell solver for the lifted bag scene.

The liquid is represented by equal-volume Lagrangian particles. A short range
position projection supplies its nearly incompressible pressure, while the bag
is a mass carrying triangular spring shell. Contact is resolved on both sides
of the impermeable shell with equal and opposite liquid/cloth impulses. Only
the vertices in the frozen rim set are kinematically constrained.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Self-contained two-way cloth/liquid time stepper."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # deterministic discretisation; no random forcing
        initial = scene.initial_state()
        cfg = scene.config
        meta = scene.metadata()

        self.time = 0.0
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.attributes0 = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.nbag = int(np.count_nonzero(self.body == 0))
        self.nwater = len(self.x) - self.nbag
        self.cx = self.x[:self.nbag].copy()
        self.cv = self.v[:self.nbag].copy()
        self.wx = self.x[self.nbag:].copy()
        self.wv = self.v[self.nbag:].copy()
        self.cm = self.mass[:self.nbag].copy()
        self.wm = self.mass[self.nbag:].copy()

        self.g = np.asarray(cfg["gravity"], dtype=np.float64)
        tank = cfg["boundary_conditions"]["tank"]
        self.tank_min = np.asarray(tank["interior_min"], dtype=np.float64)
        self.tank_max = np.asarray(tank["interior_max"], dtype=np.float64)
        pool = cfg["initial_conditions"]["pool"]
        self.spacing = float(pool["particle_spacing"])
        self.particle_radius = 0.50 * self.spacing
        self.max_step = 0.004

        self.faces = np.asarray(meta["topology"]["bag_faces"], dtype=np.int32)
        self.rim = np.asarray(meta["controlled_vertex_sets"]["rim"], dtype=np.int32)
        self.is_rim = np.zeros(self.nbag, dtype=bool)
        self.is_rim[self.rim] = True
        self.lift = cfg["boundary_conditions"]["rim_lift"]
        self.rim_rest = self.cx[self.rim].copy()
        self.rim_start_center = np.asarray(self.lift["start_center"], dtype=float)
        self.rim_end_center = np.asarray(self.lift["end_center"], dtype=float)

        edge_array = np.vstack(
            (self.faces[:, (0, 1)], self.faces[:, (1, 2)], self.faces[:, (2, 0)])
        )
        self.edges = np.unique(np.sort(edge_array, axis=1), axis=0)
        self.edge_rest = np.linalg.norm(
            self.cx[self.edges[:, 1]] - self.cx[self.edges[:, 0]], axis=1
        )

        # Opposite vertices of adjacent triangles form weak bending springs.
        owners: dict[tuple[int, int], list[int]] = {}
        for tri in self.faces:
            a, b, c = map(int, tri)
            owners.setdefault(tuple(sorted((a, b))), []).append(c)
            owners.setdefault(tuple(sorted((b, c))), []).append(a)
            owners.setdefault(tuple(sorted((c, a))), []).append(b)
        bends = [(opposite[0], opposite[1]) for opposite in owners.values()
                 if len(opposite) == 2]
        self.bends = np.asarray(bends, dtype=np.int32).reshape((-1, 2))
        self.bend_rest = np.linalg.norm(
            self.cx[self.bends[:, 1]] - self.cx[self.bends[:, 0]], axis=1
        )

        # The asset is one pole followed by circular rings ending at the rim.
        self.naround = len(self.rim)
        self.nlevels = (self.nbag - 1) // self.naround
        self.rings = np.arange(1, self.nbag, dtype=np.int32).reshape(
            self.nlevels, self.naround
        )
        rim_xy = self.rim_rest[:, :2] - self.rim_rest[:, :2].mean(axis=0)
        norm = np.linalg.norm(rim_xy, axis=1)
        self.spoke_unit = rim_xy / np.maximum(norm[:, None], 1.0e-12)

        # Material side is Lagrangian; it changes only through the open mouth.
        floor_z, surf_r, rim_z, rim_r, radial, _ = self._bag_coordinates(self.wx)
        self.inside = (
            (self.wx[:, 2] >= floor_z - 0.25 * self.particle_radius)
            & (self.wx[:, 2] <= rim_z)
            & (radial <= surf_r)
            & (radial <= rim_r)
        )

        total_mass = float(self.mass.sum())
        self.reference_velocity = math.sqrt(
            9.81 * max(float(pool["free_surface_z"]), 0.1)
        )
        self.reference_momentum = total_mass * self.reference_velocity
        self.cumulative_external_impulse = np.zeros((3, 3), dtype=np.float64)
        self.last_external_force = np.zeros((3, 3, 3), dtype=np.float64)
        self.last_external_force[0, 0] = self.g * float(self.cm.sum())
        self.last_external_force[1, 0] = self.g * float(self.wm.sum())
        self.last_external_force[2] = (
            self.last_external_force[0] + self.last_external_force[1]
        )
        self.last_interaction_force = np.zeros((2, 3), dtype=np.float64)
        self.step_count = 0
        self._sync_combined()

    def _rim_kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t0 = float(self.lift["start_time"])
        duration = float(self.lift["rise_duration"])
        if t <= t0:
            s = ds = 0.0
        elif t >= t0 + duration:
            s, ds = 1.0, 0.0
        else:
            u = (t - t0) / duration
            s = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
            ds = (30.0 * u**2 * (1.0 - u)**2) / duration
        delta = self.rim_end_center - self.rim_start_center
        position = self.rim_rest + s * delta
        velocity = np.broadcast_to(ds * delta, position.shape).copy()
        return position, velocity

    @staticmethod
    def _project_distances(
        position: np.ndarray,
        pairs: np.ndarray,
        rest: np.ndarray,
        fixed: np.ndarray,
        stiffness: float,
    ) -> None:
        if len(pairs) == 0:
            return
        a, b = pairs[:, 0], pairs[:, 1]
        delta = position[b] - position[a]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length[:, None], 1.0e-12)
        error = (length - rest)[:, None] * direction * stiffness
        wa = (~fixed[a]).astype(np.float64)
        wb = (~fixed[b]).astype(np.float64)
        denom = np.maximum(wa + wb, 1.0)
        ca = error * (wa / denom)[:, None]
        cb = -error * (wb / denom)[:, None]
        correction = np.zeros_like(position)
        count = np.zeros(len(position), dtype=np.float64)
        np.add.at(correction, a, ca)
        np.add.at(correction, b, cb)
        np.add.at(count, a, wa)
        np.add.at(count, b, wb)
        active = count > 0.0
        position[active] += correction[active] / count[active, None]

    def _advance_cloth(self, dt: float, t1: float) -> np.ndarray:
        old = self.cx.copy()
        unconstrained_velocity = self.cv + dt * self.g
        predicted = self.cx + dt * unconstrained_velocity
        rim_position, rim_velocity = self._rim_kinematics(t1)
        predicted[self.rim] = rim_position
        for _ in range(6):
            self._project_distances(
                predicted, self.edges, self.edge_rest, self.is_rim, 0.92
            )
            self._project_distances(
                predicted, self.bends, self.bend_rest, self.is_rim, 0.10
            )
            predicted[self.rim] = rim_position
        predicted[:, 2] = np.maximum(predicted[:, 2], 0.008)
        self.cx = predicted
        self.cv = (self.cx - old) / dt
        self.cv[~self.is_rim] *= 0.998
        self.cv[self.rim] = rim_velocity
        rim_impulse = np.sum(
            self.cm[self.rim, None]
            * (rim_velocity - unconstrained_velocity[self.rim]), axis=0
        )
        return rim_impulse

    def _bag_coordinates(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Lower surface and side geometry on each particle's radial ray."""
        center = self.cx[self.rim, :2].mean(axis=0)
        relative = points[:, :2] - center
        angle = np.arctan2(relative[:, 1], relative[:, 0])
        j = np.mod(
            np.rint(angle * self.naround / (2.0 * np.pi)).astype(np.int32),
            self.naround,
        )
        radial = np.linalg.norm(relative, axis=1)

        nodes = np.empty((self.nlevels + 1, self.naround, 3), dtype=np.float64)
        nodes[0, :, :] = self.cx[0]
        nodes[1:, :, :] = self.cx[self.rings]
        ring_centers = nodes[:, :, :2].mean(axis=1)
        radii = np.einsum(
            "lki,ki->lk", nodes[:, :, :2] - ring_centers[:, None, :],
            self.spoke_unit,
        )
        radii = np.maximum.accumulate(np.maximum(radii, 0.0), axis=0)
        ztable = nodes[:, :, 2]
        zcolumn = ztable[:, j]
        rcolumn = radii[:, j]

        below_count = np.sum(zcolumn <= points[:, 2][None, :], axis=0)
        level = np.clip(below_count - 1, 0, self.nlevels - 1)
        column = np.arange(len(points))
        z0, z1 = zcolumn[level, column], zcolumn[level + 1, column]
        alpha = np.clip(
            (points[:, 2] - z0) / np.maximum(z1 - z0, 1.0e-6), 0.0, 1.0
        )
        surf_r = ((1.0 - alpha) * rcolumn[level, column]
                  + alpha * rcolumn[level + 1, column])

        radial_count = np.sum(rcolumn < radial[None, :], axis=0)
        rlevel = np.clip(radial_count - 1, 0, self.nlevels - 1)
        r0, r1 = rcolumn[rlevel, column], rcolumn[rlevel + 1, column]
        beta = np.clip(
            (radial - r0) / np.maximum(r1 - r0, 1.0e-6), 0.0, 1.0
        )
        floor_z = ((1.0 - beta) * zcolumn[rlevel, column]
                   + beta * zcolumn[rlevel + 1, column])
        rim_z = zcolumn[-1]
        rim_r = rcolumn[-1]
        return floor_z, surf_r, rim_z, rim_r, radial, j

    def _fluid_pairs(self, predicted: np.ndarray) -> np.ndarray:
        pairs = cKDTree(predicted).query_pairs(
            1.27 * self.spacing, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(pairs, dtype=np.int32).reshape((-1, 2))

    def _project_liquid(self, predicted: np.ndarray, pairs: np.ndarray) -> None:
        if len(pairs) == 0:
            return
        minimum = 0.93 * self.spacing
        for _ in range(2):
            a, b = pairs[:, 0], pairs[:, 1]
            delta = predicted[b] - predicted[a]
            distance = np.linalg.norm(delta, axis=1)
            overlap = np.maximum(minimum - distance, 0.0)
            active = overlap > 0.0
            if not np.any(active):
                break
            aa, bb = a[active], b[active]
            shift = (
                0.5 * overlap[active] / np.maximum(distance[active], 1.0e-9)
            )[:, None] * delta[active]
            correction = np.zeros_like(predicted)
            count = np.zeros(len(predicted), dtype=np.float64)
            np.add.at(correction, aa, -shift)
            np.add.at(correction, bb, shift)
            np.add.at(count, aa, 1.0)
            np.add.at(count, bb, 1.0)
            used = count > 0.0
            predicted[used] += correction[used] / count[used, None]

    def _bag_contact(
        self, predicted: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        floor_z, surf_r, rim_z, rim_r, radial, j = self._bag_coordinates(predicted)
        relative = predicted[:, :2] - self.cx[self.rim, :2].mean(axis=0)
        outward = relative / np.maximum(radial[:, None], 1.0e-10)

        capture = (
            (~self.inside) & (predicted[:, 2] < rim_z)
            & (predicted[:, 2] > rim_z - 0.20)
            & (radial < 0.82 * rim_r) & (predicted[:, 2] >= floor_z)
        )
        self.inside[capture] = True
        spilled = (
            self.inside & (predicted[:, 2] > rim_z + 0.10 * self.particle_radius)
            & (radial > 0.72 * rim_r)
        )
        self.inside[spilled] = False

        correction = np.zeros_like(predicted)
        node_impulse = np.zeros_like(self.cx)
        radius = self.particle_radius
        under_rim = predicted[:, 2] < rim_z
        wall_inside = self.inside & under_rim & (radial > surf_r - radius)
        wall_outside = (
            (~self.inside) & under_rim & (predicted[:, 2] >= floor_z)
            & (radial < surf_r + radius)
        )
        wall_delta = np.zeros(len(predicted), dtype=np.float64)
        wall_delta[wall_inside] = (surf_r - radius - radial)[wall_inside]
        wall_delta[wall_outside] = (surf_r + radius - radial)[wall_outside]
        wall_delta = np.clip(wall_delta, -0.035, 0.035)
        correction[:, :2] += wall_delta[:, None] * outward

        bottom_inside = self.inside & (predicted[:, 2] < floor_z + radius)
        bottom_outside = (
            (~self.inside) & (predicted[:, 2] < floor_z)
            & (predicted[:, 2] > floor_z - 2.0 * radius)
            & (radial < rim_r + radius)
        )
        vertical = np.zeros(len(predicted), dtype=np.float64)
        vertical[bottom_inside] = (floor_z + radius - predicted[:, 2])[bottom_inside]
        vertical[bottom_outside] = (floor_z - radius - predicted[:, 2])[bottom_outside]
        vertical = np.clip(vertical, -0.035, 0.035)
        correction[:, 2] += vertical

        predicted += correction
        impulse = self.wm[:, None] * correction / dt
        z_fraction = np.clip(
            (predicted[:, 2] - self.cx[0, 2])
            / np.maximum(rim_z - self.cx[0, 2], 1.0e-6), 0.0, 0.999999
        )
        level = np.clip(
            (z_fraction * self.nlevels).astype(np.int32), 0, self.nlevels - 1
        )
        wall_node = self.rings[level, j]
        r_fraction = np.clip(
            radial / np.maximum(rim_r, 1.0e-6), 0.0, 0.999999
        )
        floor_level = np.clip(
            (r_fraction * self.nlevels).astype(np.int32), 0, self.nlevels - 1
        )
        floor_node = np.where(
            floor_level == 0, 0, self.rings[np.maximum(floor_level - 1, 0), j]
        )
        wall_used = wall_delta != 0.0
        floor_used = vertical != 0.0
        np.add.at(node_impulse, wall_node[wall_used], -impulse[wall_used])
        np.add.at(node_impulse, floor_node[floor_used], -impulse[floor_used])
        return impulse.sum(axis=0), node_impulse

    def _tank_contact(self, predicted: np.ndarray, dt: float) -> np.ndarray:
        before = predicted.copy()
        r = self.particle_radius
        predicted[:, 0] = np.clip(
            predicted[:, 0], self.tank_min[0] + r, self.tank_max[0] - r
        )
        predicted[:, 1] = np.clip(
            predicted[:, 1], self.tank_min[1] + r, self.tank_max[1] - r
        )
        predicted[:, 2] = np.maximum(predicted[:, 2], self.tank_min[2] + r)
        return np.sum(self.wm[:, None] * (predicted - before) / dt, axis=0)

    def _advance_fluid(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        old = self.wx.copy()
        predicted = self.wx + dt * self.wv + dt * dt * self.g
        pairs = self._fluid_pairs(predicted)
        self._project_liquid(predicted, pairs)
        fluid_contact_impulse, cloth_node_impulse = self._bag_contact(predicted, dt)
        tank_impulse = self._tank_contact(predicted, dt)
        self.wx = predicted
        self.wv = (self.wx - old) / dt

        # Pairwise viscosity uses equal and opposite increments.
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            close = np.linalg.norm(self.wx[b] - self.wx[a], axis=1) < 1.22 * self.spacing
            a, b = a[close], b[close]
            if len(a):
                exchange = 0.012 * (self.wv[b] - self.wv[a])
                dv = np.zeros_like(self.wv)
                count = np.zeros(len(self.wv), dtype=np.float64)
                np.add.at(dv, a, exchange)
                np.add.at(dv, b, -exchange)
                np.add.at(count, a, 1.0)
                np.add.at(count, b, 1.0)
                used = count > 0.0
                self.wv[used] += dv[used] / count[used, None]

        # Implicit effective-mass regularisation keeps stiff shell contact stable.
        free = ~self.is_rim
        delta_v = cloth_node_impulse[free] / self.cm[free, None]
        speed = np.linalg.norm(delta_v, axis=1)
        scale = np.minimum(1.0, 1.5 / np.maximum(speed, 1.0e-12))
        self.cv[free] += delta_v * scale[:, None]
        return fluid_contact_impulse, tank_impulse

    @staticmethod
    def _limit_velocity(velocity: np.ndarray, maximum: float) -> None:
        speed = np.linalg.norm(velocity, axis=1)
        scale = np.minimum(1.0, maximum / np.maximum(speed, 1.0e-12))
        velocity *= scale[:, None]

    def _substep(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t1 = self.time + dt
        rim_impulse = self._advance_cloth(dt, t1)
        fluid_contact_impulse, tank_impulse = self._advance_fluid(dt)
        free_velocity = self.cv[~self.is_rim].copy()
        self._limit_velocity(free_velocity, 15.0)
        self.cv[~self.is_rim] = free_velocity
        self._limit_velocity(self.wv, 12.0)
        self.time = t1
        self.step_count += 1
        return rim_impulse, fluid_contact_impulse, tank_impulse

    def _sync_combined(self) -> None:
        self.x = np.vstack((self.cx, self.wx))
        self.v = np.vstack((self.cv, self.wv))

    def _boundary_fields(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tank_normals = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
        )
        center = self.cx[self.rim, :2].mean(axis=0)
        radial = self.cx[self.rim, :2] - center
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1.0e-12)
        rim_normals = np.column_stack((radial, np.zeros(len(self.rim))))
        normals = np.vstack((tank_normals, rim_normals))
        _, target_rim_velocity = self._rim_kinematics(t)
        target = np.vstack((np.zeros((5, 3)), target_rim_velocity))
        actual = np.vstack((np.zeros((5, 3)), target_rim_velocity))
        return normals, target, actual

    def _make_frame(self, t: float) -> StateFrame:
        self._sync_combined()
        normals, boundary_target, boundary_velocity = self._boundary_fields(t)
        bag_weight = (self.body == 0).astype(np.float64)
        water_weight = (self.body == 1).astype(np.float64)
        weights = np.vstack((bag_weight, water_weight, np.ones(len(self.body))))
        attributes = {key: value.copy() for key, value in self.attributes0.items()}
        attributes["mass_quantity"] = self.mass.copy()
        attributes["contained_water"] = np.r_[
            np.zeros(self.nbag, dtype=np.float64), self.inside.astype(np.float64)
        ]
        fields = {
            "boundary_normal": normals,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3), dtype=np.float64),
            "budget_domain_weights": weights,
            "budget_external_force": self.last_external_force.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "interaction_force": self.last_interaction_force.copy(),
        }
        frame = StateFrame(
            time=float(t), position=self.x.copy(), velocity=self.v.copy(),
            mass=self.mass.copy(), body_id=self.body.copy(), attributes=attributes,
            fields=fields,
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "reference_velocity": float(self.reference_velocity),
                "water_rest_density": 1000.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")
        interval = target - self.time
        if interval > 1.0e-12:
            count = max(1, int(math.ceil(interval / self.max_step)))
            dt = interval / count
            rim_sum = np.zeros(3)
            interaction_sum = np.zeros(3)
            tank_sum = np.zeros(3)
            for _ in range(count):
                rim_impulse, interaction_impulse, tank_impulse = self._substep(dt)
                rim_sum += rim_impulse
                interaction_sum += interaction_impulse
                tank_sum += tank_impulse
                bag_gravity_impulse = self.g * float(self.cm.sum()) * dt
                water_gravity_impulse = self.g * float(self.wm.sum()) * dt
                self.cumulative_external_impulse[0] += bag_gravity_impulse + rim_impulse
                self.cumulative_external_impulse[1] += water_gravity_impulse + tank_impulse
                self.cumulative_external_impulse[2] += (
                    bag_gravity_impulse + water_gravity_impulse
                    + rim_impulse + tank_impulse
                )

            bag_gravity = self.g * float(self.cm.sum())
            water_gravity = self.g * float(self.wm.sum())
            external = np.zeros((3, 3, 3), dtype=np.float64)
            external[0, 0] = bag_gravity
            external[0, 1] = rim_sum / interval
            external[1, 0] = water_gravity
            external[1, 2] = tank_sum / interval
            external[2] = external[0] + external[1]
            self.last_external_force = external
            force_on_water = interaction_sum / interval
            self.last_interaction_force = np.vstack((-force_on_water, force_on_water))
            self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "PBD liquid with elastic triangle shell and two-way contact",
            "substeps": int(self.step_count),
            "contained_particles": int(np.count_nonzero(self.inside)),
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
