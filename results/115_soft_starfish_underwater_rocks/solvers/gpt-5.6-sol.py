"""Deterministic reduced fluid--deformable-body solver for this scene.

The starfish is integrated as a triangulated mass-spring shell. Water is a
finite-volume shallow-water sheet whose columns are represented by the water
samples supplied by the scene. Buoyancy and drag act on submerged shell
nodes; equal-and-opposite drag and displaced volume are deposited in water.
"""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A small, self-contained physical solver using only NumPy."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.scene = scene
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.rest_position = self.position.copy()
        self.star_idx = np.flatnonzero(self.body_id == 1)
        self.water_idx = np.flatnonzero(self.body_id == 2)
        self.ns, self.nw = len(self.star_idx), len(self.water_idx)
        self.gravity = np.asarray(scene.config.get("gravity", [0, 0, -9.81]), dtype=float)
        self.rho_water = float(scene.config["water"]["density"])
        self.water_level = float(scene.config["water"]["initial_free_surface_z"])

        from scene import starfish_surface
        _, faces = starfish_surface()
        tri_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        tri_edges.sort(axis=1)
        edges = np.unique(tri_edges, axis=0)
        self.edge_i, self.edge_j = edges[:, 0], edges[:, 1]
        d = self.position[self.edge_j] - self.position[self.edge_i]
        self.edge_rest = np.linalg.norm(d, axis=1)

        # Two-ring chords provide bending resistance; top-bottom ties preserve
        # thickness while allowing every arm to move freely.
        bend = []
        n_theta, n_radial = 100, 8
        for ring in range(n_radial):
            base = 2 + 2 * n_theta * ring
            for a in range(n_theta):
                for side in (0, 1):
                    bend.append((base + 2 * a + side,
                                 base + 2 * ((a + 2) % n_theta) + side))
                bend.append((base + 2 * a, base + 2 * a + 1))
        for ring in range(n_radial - 2):
            b0, b2 = 2 + 2 * n_theta * ring, 2 + 2 * n_theta * (ring + 2)
            for a in range(n_theta):
                bend.append((b0 + 2 * a, b2 + 2 * a))
                bend.append((b0 + 2 * a + 1, b2 + 2 * a + 1))
        self.bend = np.asarray(bend, dtype=np.int32)
        bd = self.position[self.bend[:, 1]] - self.position[self.bend[:, 0]]
        self.bend_rest = np.linalg.norm(bd, axis=1)

        rock_data = [
            (-.34, -.25, .10, .150, .0984, .0936),
            (-.10, -.31, .11, .126, .0819, .11025),
            (.18, -.30, .09, .15525, .0897, .0828),
            (.39, -.22, .12, .0858, .1232, .1155),
            (-.39, .00, .12, .1035, .1472, .115),
            (-.16, -.04, .13, .14375, .110, .1325),
            (.09, -.06, .10, .128, .085, .082),
            (.34, .03, .14, .1144, .1456, .143),
            (-.31, .27, .10, .143, .0902, .0858),
            (-.05, .25, .15, .1056, .144, .144),
            (.20, .25, .09, .1491, .0819, .07875),
            (.42, .29, .11, .0897, .1288, .1058),
        ]
        rocks = np.asarray(rock_data, dtype=float)
        self.rock_center, self.rock_radii = rocks[:, :3], rocks[:, 3:]

        wp = self.position[self.water_idx]
        self.grid_x, self.grid_y, self.layer_z = (np.unique(wp[:, k]) for k in range(3))
        self.nx, self.ny, self.nz = len(self.grid_x), len(self.grid_y), len(self.layer_z)
        self.dx, self.dy = float(np.mean(np.diff(self.grid_x))), float(np.mean(np.diff(self.grid_y)))
        self.cell_area = self.dx * self.dy
        self.eta = np.zeros((self.nx, self.ny))
        self.u = np.zeros_like(self.eta)
        self.v = np.zeros_like(self.eta)
        self.disp_x = np.zeros_like(self.eta)
        self.disp_y = np.zeros_like(self.eta)
        self.eta_rate = np.zeros_like(self.eta)
        self.layer_alpha = (self.layer_z - 0.08) / (self.water_level - 0.08)

        self.star_volume = 0.0115
        self.node_volume = self.star_volume / self.ns
        self.submerged = np.zeros(self.ns)
        self.last_fluid_force = np.zeros(3)
        self.last_contact_force = np.zeros(3)
        self.last_budget_force = np.zeros((3, 5, 3))
        self.cumulative_impulse = np.zeros((3, 3))
        self.step_count = 0
        self.boundary_normal = np.asarray([[0., 0., 1.], [1., 0., 0.],
                                           [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        self.boundary_zero = np.zeros_like(self.boundary_normal)
        self.domain_weights = np.vstack((np.ones(len(self.mass)),
                                         (self.body_id == 1).astype(float),
                                         (self.body_id == 2).astype(float)))
        self.reference_velocity = math.sqrt(abs(self.gravity[2]) * 0.58)
        self.reference_momentum = float(self.mass[self.star_idx].sum() * self.reference_velocity)
        star_gravity = self.mass[self.star_idx].sum() * self.gravity
        water_gravity = self.mass[self.water_idx].sum() * self.gravity
        self.last_budget_force[1, 0] = star_gravity
        self.last_budget_force[2, 0] = water_gravity
        self.last_budget_force[2, 1] = -water_gravity
        self.last_budget_force[0] = self.last_budget_force[1] + self.last_budget_force[2]

    @staticmethod
    def _deposit(array, ix, iy, values) -> None:
        np.add.at(array, (ix, iy), values)

    def _grid_indices(self, xy):
        ix = np.rint((xy[:, 0] - self.grid_x[0]) / self.dx).astype(np.int32)
        iy = np.rint((xy[:, 1] - self.grid_y[0]) / self.dy).astype(np.int32)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1)

    def _spring_forces(self, p, vel):
        force = np.zeros_like(p)
        d = p[self.edge_j] - p[self.edge_i]
        length = np.maximum(np.linalg.norm(d, axis=1), 1.e-9)
        direction = d / length[:, None]
        rate = np.einsum("ij,ij->i", vel[self.edge_j] - vel[self.edge_i], direction)
        ef = (18.0 * (length - self.edge_rest) + .075 * rate)[:, None] * direction
        np.add.at(force, self.edge_i, ef)
        np.add.at(force, self.edge_j, -ef)
        i, j = self.bend[:, 0], self.bend[:, 1]
        d = p[j] - p[i]
        length = np.maximum(np.linalg.norm(d, axis=1), 1.e-9)
        direction = d / length[:, None]
        rate = np.einsum("ij,ij->i", vel[j] - vel[i], direction)
        stiffness = np.where(self.bend_rest < .075, 9.0, 1.2)
        bf = (stiffness * (length - self.bend_rest) + .035 * rate)[:, None] * direction
        np.add.at(force, i, bf)
        np.add.at(force, j, -bf)
        return force

    def _contact_forces(self, p, vel):
        total = np.zeros_like(p)
        k_contact, damping, skin = 720.0, 3.6, .003
        for center, radii in zip(self.rock_center, self.rock_radii):
            rel = p - center
            q = rel / radii
            s = np.maximum(np.linalg.norm(q, axis=1), 1.e-10)
            grad = (q / radii) / s[:, None]
            grad_norm = np.maximum(np.linalg.norm(grad, axis=1), 1.e-10)
            normal = grad / grad_norm[:, None]
            penetration = skin - (s - 1.0) / grad_norm
            active = penetration > 0.0
            if not np.any(active):
                continue
            vn = np.einsum("ij,ij->i", vel, normal)
            normal_mag = (k_contact * np.maximum(penetration, 0.0)
                          + damping * np.maximum(-vn, 0.0)) * active
            fc = normal_mag[:, None] * normal
            tangential = vel - vn[:, None] * normal
            speed = np.linalg.norm(tangential, axis=1)
            friction_scale = np.minimum(1.25, .48 * normal_mag / np.maximum(speed, 1.e-8))
            fc -= active[:, None] * friction_scale[:, None] * tangential
            total += fc
        pen = .006 - p[:, 2]
        active = pen > 0.0
        total[:, 2] += (k_contact * np.maximum(pen, 0.0)
                        + damping * np.maximum(-vel[:, 2], 0.0)) * active
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                distance = .495 - sign * p[:, axis]
                active = distance < skin
                inward = -sign
                vn = inward * vel[:, axis]
                mag = (k_contact * np.maximum(skin - distance, 0.0)
                       + damping * np.maximum(-vn, 0.0)) * active
                total[:, axis] += inward * mag
        return total

    def _step(self, dt):
        p = self.position[self.star_idx]
        vel = self.velocity[self.star_idx]
        m = self.mass[self.star_idx]
        ix, iy = self._grid_indices(p[:, :2])
        local_surface = self.water_level + self.eta[ix, iy]
        submerged = np.clip((local_surface + .018 - p[:, 2]) / .036, 0.0, 1.0)
        fluid_velocity = np.column_stack((self.u[ix, iy], self.v[ix, iy], self.eta_rate[ix, iy]))
        relative = vel - fluid_velocity
        buoyancy = np.zeros_like(p)
        buoyancy[:, 2] = self.rho_water * self.node_volume * (-self.gravity[2]) * submerged
        speed = np.linalg.norm(relative, axis=1)
        drag = -.030 * submerged[:, None] * relative
        drag += -.010 * submerged[:, None] * speed[:, None] * relative
        fluid_force = buoyancy + drag
        elastic = self._spring_forces(p, vel)
        contact = self._contact_forces(p, vel)
        gravity_force = m[:, None] * self.gravity
        air_damping = -.012 * (1.0 - submerged)[:, None] * vel
        force = elastic + contact + fluid_force + gravity_force + air_damping
        vel += dt * force / m[:, None]
        np.clip(vel, -5.0, 5.0, out=vel)
        p += dt * vel
        self.position[self.star_idx] = p
        self.velocity[self.star_idx] = vel

        displaced_delta = self.node_volume * (submerged - self.submerged)
        volume_source = np.zeros_like(self.eta)
        self._deposit(volume_source, ix, iy, displaced_delta)
        self.eta += volume_source / self.cell_area
        self.submerged = submerged
        fx, fy = np.zeros_like(self.eta), np.zeros_like(self.eta)
        self._deposit(fx, ix, iy, -fluid_force[:, 0])
        self._deposit(fy, ix, iy, -fluid_force[:, 1])
        depth = np.maximum(.12, self.water_level + self.eta)
        column_mass = self.rho_water * self.cell_area * depth
        ep = np.pad(self.eta, 1, mode="edge")
        grad_x = (ep[2:, 1:-1] - ep[:-2, 1:-1]) / (2.0 * self.dx)
        grad_y = (ep[1:-1, 2:] - ep[1:-1, :-2]) / (2.0 * self.dy)
        up, vp = np.pad(self.u, 1, mode="edge"), np.pad(self.v, 1, mode="edge")
        lap_u = ((up[2:, 1:-1] + up[:-2, 1:-1] - 2*self.u) / self.dx**2
                 + (up[1:-1, 2:] + up[1:-1, :-2] - 2*self.u) / self.dy**2)
        lap_v = ((vp[2:, 1:-1] + vp[:-2, 1:-1] - 2*self.v) / self.dx**2
                 + (vp[1:-1, 2:] + vp[1:-1, :-2] - 2*self.v) / self.dy**2)
        self.u += dt * (-9.81*grad_x + fx/column_mass + .0015*lap_u - .12*self.u)
        self.v += dt * (-9.81*grad_y + fy/column_mass + .0015*lap_v - .12*self.v)
        self.u[0, :] = self.u[-1, :] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0
        np.clip(self.u, -1.5, 1.5, out=self.u)
        np.clip(self.v, -1.5, 1.5, out=self.v)
        flux_x, flux_y = depth*self.u, depth*self.v
        xp, yp = np.pad(flux_x, 1, mode="edge"), np.pad(flux_y, 1, mode="edge")
        divergence = ((xp[2:, 1:-1] - xp[:-2, 1:-1]) / (2*self.dx)
                      + (yp[1:-1, 2:] - yp[1:-1, :-2]) / (2*self.dy))
        self.eta_rate = -divergence
        self.eta += dt * self.eta_rate
        np.clip(self.eta, -.10, .14, out=self.eta)
        self.disp_x += dt*self.u
        self.disp_y += dt*self.v
        np.clip(self.disp_x, -.025, .025, out=self.disp_x)
        np.clip(self.disp_y, -.025, .025, out=self.disp_y)

        fluid_sum, contact_sum = fluid_force.sum(axis=0), contact.sum(axis=0)
        budget = np.zeros((3, 5, 3))
        budget[1, 0] = gravity_force.sum(axis=0)
        budget[1, 2] = contact_sum
        budget[1, 3] = fluid_sum
        budget[1, 4] = air_damping.sum(axis=0)
        water_gravity = self.mass[self.water_idx].sum() * self.gravity
        budget[2, 0], budget[2, 1], budget[2, 3] = water_gravity, -water_gravity, -fluid_sum
        budget[0] = budget[1] + budget[2]
        self.cumulative_impulse += dt * budget.sum(axis=1)
        self.last_budget_force = budget
        self.last_fluid_force, self.last_contact_force = fluid_sum, contact_sum
        self.step_count += 1

    def _update_water_samples(self):
        height = np.maximum(.10, self.water_level + self.eta)
        x = self.grid_x[:, None, None] + self.disp_x[:, :, None]
        y = self.grid_y[None, :, None] + self.disp_y[:, :, None]
        z = .08 + (height[:, :, None] - .08) * self.layer_alpha[None, None, :]
        wp = np.empty((self.nx, self.ny, self.nz, 3))
        wp[..., 0] = np.broadcast_to(x, wp.shape[:-1])
        wp[..., 1] = np.broadcast_to(y, wp.shape[:-1])
        wp[..., 2] = z
        wv = np.empty_like(wp)
        wv[..., 0] = self.u[:, :, None]
        wv[..., 1] = self.v[:, :, None]
        wv[..., 2] = self.eta_rate[:, :, None] * self.layer_alpha[None, None, :]
        self.position[self.water_idx] = wp.reshape(-1, 3)
        self.velocity[self.water_idx] = wv.reshape(-1, 3)

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.e-12:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.e-12:
            dt = min(.002, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        self._update_water_samples()
        interaction = np.vstack((self.last_fluid_force, -self.last_fluid_force,
                                 self.last_contact_force, -self.last_contact_force))
        frame = StateFrame(
            time=target, position=self.position.copy(), velocity=self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"rest_position": self.rest_position.copy(),
                        "mass_quantity": self.mass.copy()},
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_zero.copy(),
                "boundary_velocity": self.boundary_zero.copy(),
                "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.last_budget_force.copy(),
                "cumulative_mass_source": np.zeros(1),
                "interaction_force": interaction,
            },
            scalars={"water_free_surface_z": self.water_level,
                     "reference_momentum": self.reference_momentum,
                     "reference_velocity": self.reference_velocity},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "mass_spring_shell_coupled_to_finite_volume_shallow_water",
                "internal_steps": self.step_count, "star_nodes": self.ns,
                "water_columns": self.nx*self.ny, "two_way_coupling": True}
