"""Reduced-order coupled fluid/rigid solver for the overloaded boat scene."""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.n_water = int(np.count_nonzero(self.body_id == 0))

        ic = self.cfg["initial_conditions"]
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.rho = float(self.cfg["fluid"]["density"])
        self.depth0 = float(self.cfg["fluid"]["initial_water_depth"])
        self.radius = float(ic["sphere_radius"])
        self.release = np.array([-math.inf, float(ic["light_release_time"]),
                                 float(ic["heavy_release_time"])])

        # A finite-difference shallow-water column for each x-y particle stack.
        wp = self.pos[:self.n_water]
        self.xc, self.yc = np.unique(wp[:, 0]), np.unique(wp[:, 1])
        self.nx, self.ny = len(self.xc), len(self.yc)
        self.dx = float(np.median(np.diff(self.xc)))
        self.dy = float(np.median(np.diff(self.yc)))
        self.eta = np.zeros((self.nx, self.ny))
        self.u = np.zeros_like(self.eta)
        self.v = np.zeros_like(self.eta)
        self.eta_rate = np.zeros_like(self.eta)
        self.water_ix = np.searchsorted(self.xc, wp[:, 0])
        self.water_iy = np.searchsorted(self.yc, wp[:, 1])
        self.layer_fraction = wp[:, 2] / self.depth0
        self.water_xy = wp[:, :2].copy()

        self.rpos = self.pos[self.n_water:].copy()
        self.rvel = self.vel[self.n_water:].copy()
        self.initial_rpos = self.rpos.copy()
        self.boat_length = float(ic["hull_length"])
        self.boat_width = float(ic["hull_width"])
        self.boat_mass = float(ic["boat_mass"])
        self.sphere_volume = 4.0 * math.pi * self.radius**3 / 3.0

        # Effective trapped-volume hull.  Boat plus dense cargo exceeds its
        # reserve buoyancy, while the empty and lightly loaded hull floats.
        self.hull_area = 0.90
        self.hull_keel_offset = -0.25
        self.hull_capacity = 0.150
        self.hull_shell_volume = 0.012
        self.dry_fraction = 1.0
        self.prev_displaced = np.zeros(3)

        self.external_impulse = np.zeros((5, 3))
        self.last_external = np.zeros((5, 3, 3))
        self.last_interaction = np.zeros((6, 3))
        self.steps = 0
        self.min_depth_seen = self.depth0
        surfaces = self._surface_at(self.rpos[:, 0], self.rpos[:, 1])
        self.prev_displaced[0] = self._boat_displacement(surfaces[0])
        self.prev_displaced[1] = self._sphere_volume(self.rpos[1, 2], surfaces[1])
        self.prev_displaced[2] = self._sphere_volume(self.rpos[2, 2], surfaces[2])

    def _sphere_volume(self, center_z: float, surface_z: float) -> float:
        h = float(np.clip(surface_z - (center_z - self.radius), 0.0, 2.0 * self.radius))
        return math.pi * h * h * (self.radius - h / 3.0)

    def _boat_displacement(self, surface_z: float) -> float:
        draft = max(0.0, surface_z - (self.rpos[0, 2] + self.hull_keel_offset))
        intact = min(self.hull_area * draft, self.hull_capacity)
        return self.hull_shell_volume + self.dry_fraction * max(
            0.0, intact - self.hull_shell_volume)

    def _grid_index(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.clip(round((x - self.xc[0]) / self.dx), 0, self.nx - 1))
        j = int(np.clip(round((y - self.yc[0]) / self.dy), 0, self.ny - 1))
        return i, j

    def _surface_at(self, xs, ys) -> np.ndarray:
        out = []
        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
            i, j = self._grid_index(float(x), float(y))
            out.append(self.depth0 + self.eta[i, j])
        return np.asarray(out)

    def _deposit(self, array: np.ndarray, x: float, y: float, value: float,
                 radius_cells: int = 2) -> None:
        ci, cj = self._grid_index(x, y)
        points, total = [], 0.0
        sigma2 = max(1.0, 0.5 * radius_cells * radius_cells)
        for i in range(max(0, ci-radius_cells), min(self.nx, ci+radius_cells+1)):
            for j in range(max(0, cj-radius_cells), min(self.ny, cj+radius_cells+1)):
                w = math.exp(-((i-ci)**2 + (j-cj)**2) / sigma2)
                points.append((i, j, w)); total += w
        for i, j, w in points:
            array[i, j] += value * w / total

    def _fluid_step(self, dt: float) -> None:
        # Linear shallow-water equations with reflective solid walls.
        e = np.pad(self.eta, ((1, 1), (1, 1)), mode="edge")
        gx = (e[2:, 1:-1] - e[:-2, 1:-1]) / (2.0 * self.dx)
        gy = (e[1:-1, 2:] - e[1:-1, :-2]) / (2.0 * self.dy)
        self.u += dt * (-9.81 * gx - 0.32 * self.u)
        self.v += dt * (-9.81 * gy - 0.32 * self.v)
        self.u[0, :] = self.u[-1, :] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0
        up = np.pad(self.u, ((1, 1), (0, 0)), mode="edge")
        vp = np.pad(self.v, ((0, 0), (1, 1)), mode="edge")
        rate = -self.depth0 * ((up[2:] - up[:-2]) / (2.0*self.dx)
                              + (vp[:, 2:] - vp[:, :-2]) / (2.0*self.dy))
        self.eta += dt * rate
        ep = np.pad(self.eta, ((1, 1), (1, 1)), mode="edge")
        lap = (ep[2:, 1:-1] + ep[:-2, 1:-1] + ep[1:-1, 2:]
               + ep[1:-1, :-2] - 4.0*self.eta)
        self.eta += 0.015 * lap
        self.eta -= np.mean(self.eta)
        self.eta = np.clip(self.eta, -0.50, 0.48)
        self.eta_rate = rate
        self.min_depth_seen = min(self.min_depth_seen,
                                  float(np.min(self.depth0 + self.eta)))

    def _contact(self, a: int, b: int, normal: np.ndarray, penetration: float,
                 stiffness: float, damping: float) -> np.ndarray:
        if penetration <= 0.0:
            return np.zeros(3)
        relative_speed = float(np.dot(self.rvel[a] - self.rvel[b], normal))
        return max(0.0, stiffness*penetration - damping*relative_speed) * normal

    def _rigid_step(self, dt: float) -> None:
        surfaces = self._surface_at(self.rpos[:, 0], self.rpos[:, 1])
        masses = np.array([self.boat_mass, self.mass[-2], self.mass[-1]])
        gravity_forces = masses[:, None] * self.gravity
        forces = gravity_forces.copy()
        interactions = np.zeros((6, 3))
        displaced = np.array([
            self._boat_displacement(surfaces[0]),
            self._sphere_volume(self.rpos[1, 2], surfaces[1]),
            self._sphere_volume(self.rpos[2, 2], surfaces[2]),
        ])

        # Pressure and drag couple each rigid to the fluid.  Horizontal reaction
        # impulse is deposited into the water momentum field.
        for q in range(3):
            i, j = self._grid_index(self.rpos[q, 0], self.rpos[q, 1])
            water_v = np.array([self.u[i, j], self.v[i, j], 0.0])
            buoyancy = np.array([0.0, 0.0, self.rho*9.81*displaced[q]])
            scale = self.hull_capacity if q == 0 else self.sphere_volume
            wet = np.clip(displaced[q] / scale, 0.0, 1.0)
            drag = (95.0 if q == 0 else 24.0) * wet * (water_v-self.rvel[q])
            hydro = buoyancy + drag
            forces[q] += hydro; interactions[q] = hydro
            cell_mass = self.rho * self.depth0 * self.dx * self.dy
            self._deposit(self.u, self.rpos[q, 0], self.rpos[q, 1],
                          -dt*hydro[0]/cell_mass)
            self._deposit(self.v, self.rpos[q, 0], self.rpos[q, 1],
                          -dt*hydro[1]/cell_mass)

        # Cargo interacts with the moving interior floor but is not attached.
        floor_z = self.rpos[0, 2] - 0.21
        half_l = 0.5*self.boat_length - 0.45*self.radius
        half_w = 0.5*self.boat_width - 0.45*self.radius
        for sphere, row in ((1, 3), (2, 4)):
            inside = (abs(self.rpos[sphere, 0]-self.rpos[0, 0]) < half_l and
                      abs(self.rpos[sphere, 1]-self.rpos[0, 1]) < half_w)
            if inside:
                penetration = floor_z + self.radius - self.rpos[sphere, 2]
                f = self._contact(sphere, 0, np.array([0., 0., 1.]),
                                  penetration, 6200.0, 210.0)
                forces[sphere] += f; forces[0] -= f; interactions[row] = f

        delta = self.rpos[1] - self.rpos[2]
        distance = float(np.linalg.norm(delta))
        if 1e-9 < distance < 2.0*self.radius:
            f = self._contact(1, 2, delta/distance, 2.0*self.radius-distance,
                              7000.0, 160.0)
            forces[1] += f; forces[2] -= f; interactions[5] = f

        support = np.zeros((3, 3))
        for q in (1, 2):
            if self.time < self.release[q]:
                support[q] = -forces[q]
                forces[q] = 0.0
                self.rvel[q] = 0.0
                self.rpos[q] = self.initial_rpos[q]

        self.rvel += dt * forces / masses[:, None]
        self.rvel[0, :2] *= math.exp(-0.45*dt)
        self.rvel = np.clip(self.rvel, -8.0, 8.0)
        self.rpos += dt * self.rvel

        # Inelastic impacts against the pool walls and bottom.
        for q in range(3):
            rad = 0.42 if q == 0 else self.radius
            for axis, (lo, hi) in enumerate(((-2.30, 2.30), (-1.30, 1.30))):
                if self.rpos[q, axis] < lo+rad:
                    self.rpos[q, axis] = lo+rad
                    self.rvel[q, axis] = abs(self.rvel[q, axis])*0.25
                elif self.rpos[q, axis] > hi-rad:
                    self.rpos[q, axis] = hi-rad
                    self.rvel[q, axis] = -abs(self.rvel[q, axis])*0.25
            bottom = 0.18 if q == 0 else self.radius
            if self.rpos[q, 2] < bottom:
                self.rpos[q, 2] = bottom
                self.rvel[q, 2] = max(0.0, -0.18*self.rvel[q, 2])

        # The open hull floods once the local free surface passes its rim.
        local_surface = self._surface_at([self.rpos[0, 0]], [self.rpos[0, 1]])[0]
        overtop = local_surface - self.rpos[0, 2]
        if overtop > 0.0:
            self.dry_fraction *= math.exp(-dt*(1.7 + 7.0*min(overtop, 0.35)))

        # Displaced-volume changes are incompressibility sources for the free
        # surface, completing the two-way coupling.
        for q in range(3):
            dv = displaced[q] - self.prev_displaced[q]
            self._deposit(self.eta, self.rpos[q, 0], self.rpos[q, 1],
                          dv/(self.dx*self.dy), 3 if q == 0 else 2)
        self.eta -= np.mean(self.eta)
        self.prev_displaced[:] = displaced
        self.last_interaction = interactions

        ext = np.zeros((5, 3, 3))
        ext[1, 0] = self.mass[:self.n_water].sum() * self.gravity
        ext[2:, 0] = gravity_forces
        ext[3:, 1] = support[1:]
        ext[0] = np.sum(ext[1:], axis=0)
        self.last_external = ext
        self.external_impulse += dt * np.sum(ext, axis=1)

    def _step(self, dt: float) -> None:
        self._fluid_step(dt)
        self._rigid_step(dt)
        ui = self.u[self.water_ix, self.water_iy]
        vi = self.v[self.water_ix, self.water_iy]
        self.water_xy[:, 0] = np.clip(self.water_xy[:, 0] + dt*ui, -2.30, 2.30)
        self.water_xy[:, 1] = np.clip(self.water_xy[:, 1] + dt*vi, -1.30, 1.30)
        self.time += dt
        self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1e-12:
            self._step(min(0.005, target-self.time))
        self.time = target

        water_h = np.maximum(0.04, self.depth0+self.eta[self.water_ix, self.water_iy])
        self.pos[:self.n_water, :2] = self.water_xy
        self.pos[:self.n_water, 2] = self.layer_fraction * water_h
        self.vel[:self.n_water, 0] = self.u[self.water_ix, self.water_iy]
        self.vel[:self.n_water, 1] = self.v[self.water_ix, self.water_iy]
        self.vel[:self.n_water, 2] = self.layer_fraction * self.eta_rate[
            self.water_ix, self.water_iy]
        self.pos[self.n_water:] = self.rpos
        self.vel[self.n_water:] = self.rvel

        n = len(self.mass)
        weights = np.zeros((5, n))
        weights[0] = 1.0
        weights[1, :self.n_water] = 1.0
        for q in range(3):
            weights[q+2, self.n_water+q] = 1.0
        normals = np.array([[0., 0., 1.], [1., 0., 0.], [-1., 0., 0.],
                            [0., 1., 0.], [0., -1., 0.]])
        fields = {
            "boundary_normal": normals,
            "boundary_target_velocity": np.zeros_like(normals),
            "boundary_velocity": np.zeros_like(normals),
            "budget_cumulative_external_impulse": self.external_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((5, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((5, 3)),
            "budget_domain_weights": weights,
            "budget_external_force": self.last_external.copy(),
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": self.last_interaction.copy(),
        }
        ref_v = math.sqrt(9.81*self.depth0)
        frame = StateFrame(
            target, self.pos.copy(), self.vel.copy(), mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"phase": self.body_id.copy(),
                        "mass_quantity": self.mass.copy()},
            fields=fields,
            scalars={"reference_velocity": ref_v,
                     "reference_momentum": float(np.sum(self.mass)*ref_v)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "finite_difference_shallow_water_rigid_coupling",
                "steps": self.steps,
                "minimum_water_depth": self.min_depth_seen,
                "boat_dry_fraction": self.dry_fraction}
