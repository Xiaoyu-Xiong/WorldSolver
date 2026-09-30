"""Wrecking-ball brittle wall fracture.

Physical model
--------------
The brittle masonry wall is represented as a bonded particle lattice (a
bond-based peridynamic / discrete-element solid).  Neighbouring material
points are connected by axial springs; a bond carries a tensile strain only
up to a small brittle limit and then breaks irreversibly, which is what makes
cracks nucleate at the impact zone and propagate through the intact region.
Broken bonds leave free surfaces along which fragments separate.

The heavy ball is a prescribed (kinematic) pendulum on a rigid cable.  Its
equation of motion is integrated together with the wall: the impulse that the
ball delivers to the wall particles produces an equal and opposite reaction
torque about the pivot, so the ball physically loses energy while smashing
through the wall.  Gravity driven fragments then fall and collide with the
rigid floor.

Numerics: explicit symplectic (semi-implicit) Euler with a fixed substep that
divides the sampling interval exactly, penalty/projection contact handling,
and per-substep bond rupture tests.  All quantities are deterministic and
finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81])


class BenchmarkSolver:
    """Bonded-particle brittle wall struck by a swinging wrecking ball."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None) or {}
        sampling = config.get("sampling", {}) or {}
        self.spacing = float(sampling.get("spacing", 0.12))
        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.seed = int(seed)

        state0 = scene.initial_state()
        reference = np.asarray(state0.position, dtype=float).copy()
        mass = np.asarray(state0.mass, dtype=float).copy()
        count = int(reference.shape[0])
        self.reference = reference
        self.mass = mass
        self.count = count

        radii = state0.attributes.get("particle_radius")
        self.particle_radius = (
            np.asarray(radii, dtype=float).copy()
            if radii is not None
            else np.full(count, 0.45 * self.spacing)
        )
        labels = state0.body_id
        self.body_labels = (
            np.asarray(labels).copy()
            if labels is not None
            else np.zeros(count, dtype=np.int32)
        )
        scalars0 = dict(state0.scalars) if state0.scalars else {}
        self.contact_length_scale = float(
            scalars0.get("contact_length_scale", self.spacing)
        )
        self.characteristic_length = float(
            scalars0.get(
                "characteristic_length",
                max(float(np.ptp(reference, axis=0).max()), self.spacing),
            )
        )

        # ---------------------------------------------------------- lattice
        index_i, index_j = np.triu_indices(count, 1)
        reference_edges = reference[index_j] - reference[index_i]
        rest_length = np.linalg.norm(reference_edges, axis=1)
        cutoff = 1.75 * self.spacing
        keep = (rest_length > 1e-12) & (rest_length <= cutoff)
        self.bond_i = index_i[keep]
        self.bond_j = index_j[keep]
        self.bond_rest = rest_length[keep]
        edge = reference_edges[keep]
        self.bond_direction = edge / self.bond_rest[:, None]
        self.bond_alive = np.ones(self.bond_i.size, dtype=bool)

        shape_tensor = np.zeros((count, 3, 3))
        outer = edge[:, :, None] * edge[:, None, :]
        np.add.at(shape_tensor, self.bond_i, outer)
        np.add.at(shape_tensor, self.bond_j, outer)
        scale = np.maximum(np.trace(shape_tensor, axis1=1, axis2=2) / 3.0, 1e-12)
        shape_tensor += np.eye(3)[None] * (1e-9 * scale)[:, None, None]
        self.shape_inverse = np.linalg.inv(shape_tensor)

        pair_i, pair_j = np.triu_indices(count, 1)
        self.pair_i = pair_i
        self.pair_j = pair_j

        # ------------------------------------------------------ ball / cable
        # Kinematic wrecking ball on a rigid cable, released from rest at the
        # frozen initial pose (-1.75, 0, 1.35), displaced to one side of the
        # wall.  The pivot is chosen so the resulting arc carries the ball
        # horizontally through the masonry at the height of the wall body.
        self.ball_radius = 0.48
        self.ball_mass = 400.0
        self.ball_length = 1.125
        self.ball_start = np.array([-1.75, 0.0, 1.35])
        self.ball_theta = -np.arcsin(0.8)  # released position on the near side
        self.ball_omega = 0.0
        self.pivot = self.ball_start + self.ball_length * np.array(
            [-np.sin(self.ball_theta), 0.0, np.cos(self.ball_theta)]
        )

        # -------------------------------------------------------- material
        self.bond_stiffness = 1.0e5
        self.bond_damping = 20.0
        self.bond_force_cap = 2.0e4
        self.tensile_limit = 0.02
        self.particle_stiffness = 2.0e4
        self.particle_damping = 60.0
        self.ground_stiffness = 5.0e4
        self.ground_damping = 250.0
        self.ground_restitution = 0.15
        self.friction = 0.6

        self.substeps = 64
        self.dt = self.sample_dt / self.substeps

        # ------------------------------------------------------------- state
        self.position = reference.copy()
        self.velocity = np.zeros_like(self.position)
        self.ball_contact_impulse = np.zeros((count, 3))
        self.ground_contact_impulse = np.zeros((count, 3))
        self.steps_taken = 0
        self.frames_emitted = 0
        self._ready = True

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        tolerance = 1e-9 * max(1.0, abs(time))
        while self.steps_taken * self.dt < time - tolerance:
            self._step()
        return self._build_frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        alive = int(self.bond_alive.sum())
        return {
            "particle_count": int(self.count),
            "bond_count": int(self.bond_i.size),
            "intact_bonds": alive,
            "broken_bonds": int(self.bond_i.size - alive),
            "time_step": float(self.dt),
            "steps": int(self.steps_taken),
            "frames": int(self.frames_emitted),
            "ball_theta": float(self.ball_theta),
            "ball_omega": float(self.ball_omega),
        }

    # ------------------------------------------------------------- dynamics
    def _ball_state(self):
        theta = self.ball_theta
        center = self.pivot + self.ball_length * np.array(
            [np.sin(theta), 0.0, -np.cos(theta)]
        )
        velocity = self.ball_omega * self.ball_length * np.array(
            [np.cos(theta), 0.0, np.sin(theta)]
        )
        return center, velocity

    def _step(self) -> None:
        dt = self.dt
        position = self.position
        velocity = self.velocity
        mass = self.mass
        ball_center, ball_velocity = self._ball_state()
        ball_impulse = np.zeros((self.count, 3))
        floor_impulse = np.zeros((self.count, 3))

        acceleration = np.tile(_GRAVITY_VECTOR, (self.count, 1))

        # ---------------------------------------------------- lattice bonds
        if self.bond_i.size:
            delta = position[self.bond_j] - position[self.bond_i]
            distance = np.linalg.norm(delta, axis=1)
            distance = np.maximum(distance, 1e-9)
            direction = delta / distance[:, None]
            extension = distance - self.bond_rest
            relative = velocity[self.bond_j] - velocity[self.bond_i]
            closing = np.einsum("ij,ij->i", relative, direction)
            magnitude = self.bond_stiffness * extension + self.bond_damping * closing
            magnitude = np.clip(magnitude, -self.bond_force_cap, self.bond_force_cap)
            magnitude = np.where(self.bond_alive, magnitude, 0.0)
            force = magnitude[:, None] * direction
            np.add.at(acceleration, self.bond_i, force / mass[self.bond_i, None])
            np.add.at(acceleration, self.bond_j, -force / mass[self.bond_j, None])

        # ------------------------------------------------- particle contacts
        delta = position[self.pair_i] - position[self.pair_j]
        distance = np.linalg.norm(delta, axis=1)
        distance = np.maximum(distance, 1e-9)
        touching = 2.0 * self.particle_radius[self.pair_i] - distance
        active = touching > 0.0
        if np.any(active):
            direction = delta / distance[:, None]
            relative = velocity[self.pair_i] - velocity[self.pair_j]
            approach = np.einsum("ij,ij->i", relative, direction)
            overlap = np.minimum(touching, 0.05)
            normal = self.particle_stiffness * overlap - self.particle_damping * approach
            normal = np.where(active, np.maximum(normal, 0.0), 0.0)
            force = normal[:, None] * direction
            np.add.at(acceleration, self.pair_i, force / mass[self.pair_i, None])
            np.add.at(acceleration, self.pair_j, -force / mass[self.pair_j, None])

        # ------------------------------------------------------------- floor
        floor_overlap = self.particle_radius - position[:, 2]
        on_floor = floor_overlap > 0.0
        if np.any(on_floor):
            normal = (
                self.ground_stiffness * np.minimum(floor_overlap, 0.05)
                - self.ground_damping * np.minimum(velocity[:, 2], 0.0)
            )
            normal = np.where(on_floor, np.maximum(normal, 0.0), 0.0)
            acceleration[:, 2] += normal / mass

        # -------------------------------------------------------- integrate
        velocity += dt * acceleration
        position += dt * velocity

        # ------------------------------------------------------ floor contact
        floor_overlap = self.particle_radius - position[:, 2]
        on_floor = floor_overlap > 0.0
        if np.any(on_floor):
            position[on_floor, 2] = self.particle_radius[on_floor]
            normal_velocity = velocity[on_floor, 2]
            rebound = np.where(
                normal_velocity < 0.0,
                -self.ground_restitution * normal_velocity,
                0.0,
            )
            correction = rebound - normal_velocity
            velocity[on_floor, 2] = rebound
            tangential = velocity[on_floor, :2]
            speed = np.linalg.norm(tangential, axis=1)
            friction_drop = np.minimum(speed, self.friction * 9.81 * dt)
            scale = np.where(speed > 1e-12, (speed - friction_drop) / np.maximum(speed, 1e-12), 0.0)
            velocity[on_floor, :2] = tangential * scale[:, None]
            impulse = np.zeros((int(on_floor.sum()), 3))
            impulse[:, 2] = mass[on_floor] * correction
            impulse[:, :2] = mass[on_floor, None] * (
                tangential * scale[:, None] - tangential
            )
            np.add.at(floor_impulse, np.flatnonzero(on_floor), impulse)

        # ------------------------------------------------------- ball contact
        offset = position - ball_center
        distance = np.linalg.norm(offset, axis=1)
        distance = np.maximum(distance, 1e-9)
        normal = offset / distance[:, None]
        reach = self.ball_radius + self.particle_radius
        penetration = reach - distance
        struck = penetration > 0.0
        if np.any(struck):
            index = np.flatnonzero(struck)
            normal_i = normal[index]
            position[index] = ball_center + normal_i * reach[index, None]
            surface_velocity = np.einsum("ij,j->i", normal_i, ball_velocity)
            normal_speed = np.einsum("ij,ij->i", velocity[index], normal_i)
            target = np.maximum(normal_speed, surface_velocity)
            new_velocity = velocity[index] + (target - normal_speed)[:, None] * normal_i
            tangential = new_velocity - np.einsum(
                "ij,ij->i", new_velocity, normal_i
            )[:, None] * normal_i
            new_velocity -= 0.15 * tangential
            change = new_velocity - velocity[index]
            velocity[index] = new_velocity
            np.add.at(ball_impulse, index, mass[index, None] * change)

        # --------------------------------------------------------- safeguard
        np.clip(velocity, -40.0, 40.0, out=velocity)

        # ------------------------------------------------------ bond rupture
        if self.bond_i.size:
            delta = position[self.bond_j] - position[self.bond_i]
            distance = np.linalg.norm(delta, axis=1)
            strain = (distance - self.bond_rest) / self.bond_rest
            self.bond_alive &= strain <= self.tensile_limit

        # --------------------------------------------------- ball pendulum
        reaction = -ball_impulse.sum(axis=0)
        torque_y = (
            (ball_center[2] - self.pivot[2]) * reaction[0]
            - (ball_center[0] - self.pivot[0]) * reaction[2]
        )
        inertia = self.ball_mass * self.ball_length ** 2
        angular = -9.81 * np.sin(self.ball_theta) / self.ball_length
        angular += -torque_y / inertia
        self.ball_omega += dt * angular
        self.ball_omega = float(np.clip(self.ball_omega, -12.0, 12.0))
        self.ball_theta += dt * self.ball_omega

        self.ball_contact_impulse += ball_impulse
        self.ground_contact_impulse += floor_impulse
        self.steps_taken += 1

    # ---------------------------------------------------------------- frames
    def _deformation_gradient(self) -> np.ndarray:
        edge = self.position[self.bond_j] - self.position[self.bond_i]
        outer = edge[:, :, None] * self.bond_direction[:, None, :]
        tensor = np.zeros((self.count, 3, 3))
        np.add.at(tensor, self.bond_i, outer)
        np.add.at(tensor, self.bond_j, outer)
        gradient = np.einsum("nij,njk->nik", tensor, self.shape_inverse)
        bad = ~np.isfinite(gradient).all(axis=(1, 2))
        if np.any(bad):
            gradient[bad] = np.eye(3)
        return gradient

    def _build_frame(self, time: float) -> StateFrame:
        ball_center, _ = self._ball_state()
        gap_ball = (
            np.linalg.norm(self.position - ball_center, axis=1)
            - (self.ball_radius + self.particle_radius)
        )
        gap_floor = self.position[:, 2] - self.particle_radius
        contact_gap = np.concatenate((gap_ball, gap_floor))

        external = self.mass[:, None] * _GRAVITY_VECTOR[None, :]
        external = external + (
            self.ball_contact_impulse + self.ground_contact_impulse
        ) / self.sample_dt
        self.ball_contact_impulse = np.zeros_like(self.ball_contact_impulse)
        self.ground_contact_impulse = np.zeros_like(self.ground_contact_impulse)

        frame = StateFrame(
            time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_labels.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.body_labels.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
                "contact_signed_distance": contact_gap,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": 1.0,
            },
        )
        self.frames_emitted += 1
        frame.validate()
        return frame
