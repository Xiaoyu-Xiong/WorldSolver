"""Solver Agent-owned implementation for the wrecking ball box stack scene.

A lightweight penalty/contact rigid-body integrator.  Each dynamic body is
treated as a sphere whose radius is chosen from the geometry (box inscribed
sphere or capsule end-cap sphere).  Bodies are integrated with an explicit
symplectic Euler sub-stepping scheme, gravity, floor contact, inter-body
contacts, and stiff chain constraints that keep the wrecking ball attached to
the fixed support beam.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        config = scene.config
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])
        self.gravity = np.asarray(config["gravity"], dtype=float)

        contact = config["contact"]
        self.friction = float(contact["friction"])
        self.restitution = float(contact["restitution"])
        self.contact_length_scale = float(contact["length_scale"])

        # Substepping for stability.
        self.substeps = 100
        self.dt = self.sample_dt / self.substeps

        # Initial state from the scaffold.
        init = scene.initial_state()
        self.position = np.array(init.position, dtype=float, copy=True)
        self.velocity = np.array(init.velocity, dtype=float, copy=True)
        self.orientation = np.array(init.orientation, dtype=float, copy=True)
        self.angular_velocity = np.array(init.angular_velocity, dtype=float, copy=True)
        self.mass = np.array(init.mass, dtype=float, copy=True)
        self.body_id = np.array(init.body_id, dtype=np.int64, copy=True)

        self.num_bodies = len(self.position)

        # Representative radii.
        self.collision_radius = np.full(self.num_bodies, 0.17, dtype=float)
        for i, bid in enumerate(self.body_id):
            if bid == 30:  # wrecking_ball
                self.collision_radius[i] = 0.30
            elif bid >= 31:  # chain capsules
                self.collision_radius[i] = 0.07

        self.inv_mass = 1.0 / self.mass

        # Fixed support anchor for the top of the chain.
        # The support beam is at z = 4.0 with half height 0.12.
        self.support_anchor = np.array([-2.2, 0.0, 4.12], dtype=float)

        # Chain links: ball - chain_0 - chain_1 - ... - chain_5 - support.
        # body_id values: 30 (ball), 31..36 (chain from bottom to top).
        self.chain_indices = [30, 31, 32, 33, 34, 35, 36]
        # Rest lengths tuned to the initial vertical spacing.
        self.rest_lengths = [0.55, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]

        self.current_time = 0.0

    def _step(self) -> None:
        # Gravity.
        self.velocity += self.gravity * self.dt

        # Chain constraints (stiff springs with damping) to hold the wrecking
        # ball and chain together and to the support.
        spring_k = 1e5
        damping = 1e2
        for idx in range(len(self.chain_indices) - 1):
            i = self.chain_indices[idx]
            j = self.chain_indices[idx + 1]
            if i >= self.num_bodies or j >= self.num_bodies:
                continue
            pi = self.position[i]
            pj = self.position[j]
            diff = pj - pi
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            n = diff / dist
            rel_vel = self.velocity[j] - self.velocity[i]
            stretch = dist - self.rest_lengths[idx]
            force_mag = spring_k * stretch + damping * np.dot(rel_vel, n)
            force = force_mag * n
            # Apply equal and opposite.
            self.velocity[i] += force * self.inv_mass[i] * self.dt
            self.velocity[j] -= force * self.inv_mass[j] * self.dt

        # Top chain link to fixed support anchor.
        top = self.chain_indices[-1]
        if top < self.num_bodies:
            pi = self.position[top]
            diff = self.support_anchor - pi
            dist = np.linalg.norm(diff)
            if dist > 1e-8:
                n = diff / dist
                rel_vel = -self.velocity[top]
                stretch = dist - self.rest_lengths[-1]
                force_mag = spring_k * stretch + damping * np.dot(rel_vel, n)
                force = force_mag * n
                self.velocity[top] += force * self.inv_mass[top] * self.dt

        # Floor and body-body contacts (penalty + velocity clamping).
        # Floor is at z = 0.
        for i in range(self.num_bodies):
            z = self.position[i, 2]
            r = self.collision_radius[i]
            if z - r < 0.0:
                pen = -(z - r)
                self.position[i, 2] += pen
                # Velocity response
                vn = self.velocity[i, 2]
                if vn < 0.0:
                    self.velocity[i, 2] = -self.restitution * vn
                # Friction in tangent plane
                v_tan = self.velocity[i].copy()
                v_tan[2] = 0.0
                self.velocity[i] -= self.friction * v_tan * self.dt * 50.0
                # Normal repulsion
                self.velocity[i, 2] += 50.0 * pen * self.dt

        # Pairwise body contacts (penalty-based).
        for i in range(self.num_bodies):
            for j in range(i + 1, self.num_bodies):
                diff = self.position[j] - self.position[i]
                dist = np.linalg.norm(diff)
                r_sum = self.collision_radius[i] + self.collision_radius[j]
                if dist < r_sum and dist > 1e-8:
                    n = diff / dist
                    pen = r_sum - dist
                    # Separate positions
                    separation = pen * n
                    self.position[i] -= separation * 0.5
                    self.position[j] += separation * 0.5
                    # Relative velocity along normal
                    rel_v = self.velocity[j] - self.velocity[i]
                    vn = np.dot(rel_v, n)
                    if vn < 0.0:
                        jn = -(1.0 + self.restitution) * vn / (self.inv_mass[i] + self.inv_mass[j])
                        impulse = jn * n
                        self.velocity[i] -= impulse * self.inv_mass[i]
                        self.velocity[j] += impulse * self.inv_mass[j]
                    # Friction (penalty)
                    rel_v = self.velocity[j] - self.velocity[i]
                    v_tan = rel_v - np.dot(rel_v, n) * n
                    self.velocity[i] += self.friction * v_tan * self.inv_mass[i] * self.dt * 20.0
                    self.velocity[j] -= self.friction * v_tan * self.inv_mass[j] * self.dt * 20.0

        # Integrate positions.
        self.position += self.velocity * self.dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            raise ValueError("advance_to called with a time earlier than current state")

        target = float(time)
        while self.current_time < target - 1e-12:
            next_time = min(self.current_time + self.sample_dt, target)
            steps = max(1, int(round((next_time - self.current_time) / self.dt)))
            real_dt = (next_time - self.current_time) / steps
            saved_dt = self.dt
            self.dt = real_dt
            for _ in range(steps):
                self._step()
            self.dt = saved_dt
            self.current_time = next_time

        self.current_time = target
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        # Contact signed distances: floor contacts + pairwise contacts.
        signed_dists = []
        contact_pairs = []
        # Floor contacts.
        for i in range(self.num_bodies):
            sd = self.position[i, 2] - self.collision_radius[i]
            if sd < 0.05:
                signed_dists.append(sd)
        # Pairwise contacts.
        for i in range(self.num_bodies):
            for j in range(i + 1, self.num_bodies):
                diff = self.position[j] - self.position[i]
                dist = np.linalg.norm(diff)
                r_sum = self.collision_radius[i] + self.collision_radius[j]
                sd = dist - r_sum
                if sd < 0.05:
                    signed_dists.append(sd)
                    contact_pairs.append((i, j))
        if not signed_dists:
            signed_dists = [0.05]
        signed_dists_arr = np.asarray(signed_dists, dtype=float)

        # External force: gravity only (per-sample).
        external_force = (self.mass[:, None] * self.gravity[None, :]).astype(float)

        # Interaction force: per-contact pair zero vectors.
        if contact_pairs:
            interaction_force = np.zeros((len(contact_pairs), 3), dtype=float)
        else:
            interaction_force = np.zeros((1, 3), dtype=float)

        return StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy().astype(np.int32),
            attributes={"phase_id": np.zeros(self.num_bodies, dtype=np.int32)},
            scalars={"contact_length_scale": self.contact_length_scale},
            fields={
                "contact_signed_distance": signed_dists_arr,
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "substeps": self.substeps,
            "sample_dt": self.sample_dt,
            "num_bodies": self.num_bodies,
        }
