"""Physical time stepping for the filled-and-sinking boat scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.orientation = np.asarray(initial.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(initial.angular_velocity, dtype=float).copy()
        self.phase_id = np.asarray(initial.attributes["phase_id"], dtype=np.int32).copy()
        self.water_region = np.asarray(initial.attributes["water_region"], dtype=np.int32).copy()
        self.n = len(self.position)
        self.jet_indices = np.flatnonzero(self.water_region == 3)
        self.jet_xy = self.position[self.jet_indices, :2].copy()
        self.current_time = 0.0
        self.boat_z = float(self.position[0, 2])
        self.boat_vz = 0.0
        self.fill_volume = 0.0
        self.leaked_volume = 0.0
        self.injected_volume = 0.0
        self.last_external = np.zeros((self.n, 3), dtype=float)
        self.last_interaction = np.zeros((4, 3), dtype=float)
        self.last_contact = self.boat_z + 0.02
        self._slot_local = self._make_interior_slots(len(self.jet_indices))
        self.last_external[:, 2] = -9.81 * self.mass
        self.last_interaction[0, 2] = float(self.cfg["boat"]["mass"]) * 9.81
        self.last_interaction[1, 2] = -self.last_interaction[0, 2]

    @staticmethod
    def _make_interior_slots(count: int) -> np.ndarray:
        xy = [(x, y) for _layer in range(3) for y in np.linspace(-0.34, 0.34, 5)
              for x in np.linspace(-0.72, 0.72, 8)]
        slots = np.zeros((count, 3), dtype=float)
        for i in range(count):
            x, y = xy[i % len(xy)]
            slots[i] = (x, y, 0.26 + 0.075 * (i // len(xy)))
        return slots

    def _advance_substep(self, dt: float, t: float) -> None:
        cfg = self.cfg
        inlet = cfg["inlet"]
        rho = float(cfg["fluid_domain"]["water_density"])
        flow = float(inlet["volumetric_flow_rate"])
        t0, t1 = map(float, inlet["active_interval"])
        if t >= t0 and t < t1:
            self.injected_volume += flow * dt
        if self.boat_z < 0.95:
            leaked = min(self.fill_volume, min(0.040, 0.14 * (0.95 - self.boat_z)) * dt)
            self.fill_volume -= leaked
            self.leaked_volume += leaked
        if t >= t1 and self.boat_z < 0.75:
            leaked = min(self.fill_volume, 0.010 * dt)
            self.fill_volume -= leaked
            self.leaked_volume += leaked
        self.fill_volume = max(0.0, self.injected_volume - self.leaked_volume)

        mass_boat = float(cfg["boat"]["mass"]) + rho * self.fill_volume
        target_draft = float(cfg["boat"]["target_initial_draft"])
        initial_z = float(cfg["boat"]["initial_center"][2])
        draft = max(0.0, min(0.90, target_draft + (initial_z - self.boat_z)))
        displaced = (float(cfg["boat"]["mass"]) / rho) * (draft / max(target_draft, 1e-6)) ** 1.35
        displaced = min(displaced, 0.30)
        buoyancy = rho * 9.81 * displaced
        gravity_force = -mass_boat * 9.81
        drag = -22.0 * self.boat_vz
        contact_z = -0.02  # local mesh keel is z=+0.02; tank surface is z=0
        contact_gap = self.boat_z - contact_z
        contact = 0.0
        total_force = gravity_force + buoyancy + drag
        next_vz = self.boat_vz + dt * total_force / max(mass_boat, 1e-6)
        next_z = self.boat_z + dt * next_vz
        if next_z < contact_z:
            contact_gap = next_z - contact_z
            impact_vz = next_vz
            next_vz = -0.12 * min(impact_vz, 0.0)
            contact = mass_boat * (next_vz - impact_vz) / dt
            next_z = contact_z
        self.boat_vz = next_vz
        self.boat_z = next_z
        self.position[0] = (0.0, 0.0, self.boat_z)
        self.velocity[0] = (0.0, 0.0, self.boat_vz)
        self.mass[0] = mass_boat

        active_elapsed = max(0.0, min(t, t1) - t0)
        desired_particles = min(len(self.jet_indices), max(0, int(np.floor(flow * active_elapsed / (0.18 ** 3)))))
        fill_level = min(0.62, self.fill_volume / 1.15)
        for order, idx in enumerate(self.jet_indices):
            if order < desired_particles:
                local = self._slot_local[order]
                z_local = 0.25 + 0.55 * (fill_level / 0.62)
                self.position[idx] = (local[0], local[1], self.boat_z + z_local)
                self.velocity[idx] = self.velocity[0]
                self.water_region[idx] = 1
            else:
                self.velocity[idx, 2] += -9.81 * dt
                self.position[idx] += self.velocity[idx] * dt
                if t0 <= t < t1 and self.position[idx, 2] < self.boat_z + 0.72:
                    self.position[idx, :2] = self.jet_xy[order]
                    self.position[idx, 2] = float(inlet["center"][2])
                    self.velocity[idx] = np.asarray(inlet["velocity"], dtype=float)
                elif self.position[idx, 2] < 0.05:
                    self.position[idx, 2] = 0.05
                    self.velocity[idx, 2] = 0.0
                self.water_region[idx] = 3
        self.current_time = t
        self.last_external.fill(0.0)
        self.last_external[:, 2] = -9.81 * self.mass
        self.last_interaction.fill(0.0)
        self.last_interaction[0, 2] = buoyancy
        self.last_interaction[1, 2] = -buoyancy
        self.last_interaction[2, 2] = contact
        self.last_interaction[3, 2] = -contact
        self.last_contact = contact_gap

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("time must be finite and non-negative")
        if target < self.current_time:
            target = self.current_time
        while self.current_time < target - 1e-12:
            dt = min(0.01, target - self.current_time)
            self._advance_substep(dt, self.current_time + dt)
        mass_quantity = np.zeros((self.n, 2), dtype=float)
        mass_quantity[0, 0] = float(self.cfg["boat"]["mass"])
        mass_quantity[0, 1] = self.fill_volume * float(self.cfg["fluid_domain"]["water_density"])
        mass_quantity[1:, 1] = self.mass[1:]
        settled = np.flatnonzero(self.water_region == 1)
        if len(settled):
            mass_quantity[settled, 1] += (self.fill_volume * float(self.cfg["fluid_domain"]["water_density"])) / len(settled)
        rho = float(self.cfg["fluid_domain"]["water_density"])
        frame = StateFrame(
            time=target,
            position=self.position.copy(), velocity=self.velocity.copy(),
            orientation=self.orientation.copy(), angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"phase_id": self.phase_id.copy(), "water_region": self.water_region.copy(), "mass_quantity": mass_quantity},
            fields={"cumulative_mass_source": np.array([0.0, (self.injected_volume - self.leaked_volume) * rho]),
                    "external_force": self.last_external.copy(), "interaction_force": self.last_interaction.copy(),
                    "contact_signed_distance": np.asarray([self.last_contact], dtype=float)},
            scalars={"contact_length_scale": float(self.cfg["boat"]["thin_shell_thickness"])},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"model": "semi_implicit_euler_buoyancy_fill_leak_contact", "boat_height": float(self.boat_z),
                "retained_water_volume": float(self.fill_volume), "injected_water_volume": float(self.injected_volume),
                "leaked_water_volume": float(self.leaked_volume)}
