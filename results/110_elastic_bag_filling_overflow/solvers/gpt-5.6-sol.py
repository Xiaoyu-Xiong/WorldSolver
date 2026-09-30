"""Deterministic two-way coupled elastic bag and particle-water solver.

The liquid is represented by finite-mass Lagrangian parcels. Short-range
incompressibility contacts keep parcels separated, while contact with the
spring shell is resolved by equal and opposite impulses.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Coupled mass-spring shell / particle-liquid time integrator."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.config = scene.config
        self.rng = np.random.default_rng(int(seed))
        self.time = 0.0
        self.nb = int(np.count_nonzero(np.asarray(initial.body_id) == 0))
        self.xb = np.asarray(initial.position[:self.nb], float).copy()
        self.vb = np.asarray(initial.velocity[:self.nb], float).copy()
        self.rest = self.xb.copy()
        self.bag_mass = float(scene.config["initial_conditions"]["bag"]["total_mass"])
        self.mb = np.full(self.nb, self.bag_mass / self.nb)
        self.pinned = np.asarray(initial.attributes["pinned"][:self.nb]) > 0.5
        self.pin_position = self.rest.copy()
        faces = np.asarray(scene._bag_faces, np.int32)
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.rest_length = np.linalg.norm(
            self.rest[self.edges[:, 1]] - self.rest[self.edges[:, 0]], axis=1)

        # Constant pool: unemitted samples have zero mass and stay out of view.
        self.nw = 3300
        self.xw = np.zeros((self.nw, 3)); self.xw[:] = (0.0, 0.0, -8.0)
        self.vw = np.zeros_like(self.xw)
        self.mw = np.zeros(self.nw)
        self.outside = np.zeros(self.nw, dtype=bool)
        initial_water = np.asarray(initial.position[self.nb:], float)
        ni = min(len(initial_water), self.nw)
        self.xw[:ni] = initial_water[:ni]
        self.vw[:ni] = np.asarray(initial.velocity[self.nb:self.nb + ni], float)
        self.parcel_mass = (float(scene.config["initial_conditions"]["water"]["rest_density"])
                            * 0.055 ** 3)
        self.mw[:ni] = np.asarray(initial.mass[self.nb:self.nb + ni], float)
        self.next_free = ni

        inflow = scene.config["boundary_conditions"]["nozzle_inflow"]
        self.nozzle_center = np.asarray(inflow["center"], float)
        self.nozzle_direction = np.asarray(inflow["direction"], float)
        self.nozzle_speed = float(inflow["speed"])
        self.nozzle_radius = float(inflow["radius"])
        self.inflow_start, self.inflow_end = map(float, inflow["active_interval"])
        self.emit_rate = 430.0
        self.emit_credit = 0.0
        self.emit_index = 0
        self.gravity = np.asarray(scene.config["gravity"], float)
        self.floor_z = float(scene.config["boundary_conditions"]["exterior"]["catch_floor_z"])
        self.ring_size = 32
        self.ring_count = (self.nb - 1) // self.ring_size
        self.rim_indices = np.flatnonzero(self.pinned)
        self.cumulative_source = 0.0
        self.cumulative_inflow_p = np.zeros(3)
        self.cumulative_outflow_p = np.zeros(3)
        self.cumulative_external = np.zeros((3, 3))
        self.last_interaction = np.zeros(3)
        self.last_support = np.zeros(3)
        self.last_floor_force = np.zeros(3)
        self.step_count = 0

    def _ring_geometry(self):
        radii = np.empty(self.ring_count + 1)
        heights = np.empty(self.ring_count + 1)
        radii[0], heights[0] = 0.0, self.xb[0, 2]
        for k in range(1, self.ring_count + 1):
            sl = slice(1 + (k - 1) * self.ring_size, 1 + k * self.ring_size)
            radii[k] = np.mean(np.linalg.norm(self.xb[sl, :2], axis=1))
            heights[k] = np.mean(self.xb[sl, 2])
        heights = np.maximum.accumulate(heights + np.arange(len(heights)) * 1.e-6)
        return radii, heights

    def _emit(self, dt):
        overlap = max(0.0, min(self.time + dt, self.inflow_end)
                      - max(self.time, self.inflow_start))
        self.emit_credit += self.emit_rate * overlap
        count = min(int(self.emit_credit), self.nw - self.next_free)
        if count <= 0:
            return
        self.emit_credit -= count
        ids = np.arange(self.next_free, self.next_free + count)
        q = np.arange(self.emit_index, self.emit_index + count, dtype=float)
        rad = self.nozzle_radius * .82 * np.sqrt((q * .7548776662466927) % 1.0)
        ang = 2 * np.pi * ((q * .6180339887498949) % 1.0)
        self.xw[ids, 0] = self.nozzle_center[0] + rad * np.cos(ang)
        self.xw[ids, 1] = self.nozzle_center[1] + rad * np.sin(ang)
        self.xw[ids, 2] = self.nozzle_center[2] - .012 * (q % 3)
        self.vw[ids] = self.nozzle_direction * self.nozzle_speed
        self.mw[ids] = self.parcel_mass
        self.outside[ids] = False
        added = self.parcel_mass * count
        self.cumulative_source += added
        self.cumulative_inflow_p += added * self.nozzle_direction * self.nozzle_speed
        self.next_free += count
        self.emit_index += count

    def _cloth_substep(self, dt):
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = self.xb[j] - self.xb[i]
        length = np.linalg.norm(d, axis=1)
        normal = d / np.maximum(length[:, None], 1.e-9)
        rel = np.sum((self.vb[j] - self.vb[i]) * normal, axis=1)
        edge_force = (105.0 * (length - self.rest_length) + .055 * rel)[:, None] * normal
        force = self.mb[:, None] * self.gravity
        np.add.at(force, i, edge_force); np.add.at(force, j, -edge_force)
        force -= .018 * self.vb
        self.last_support = -np.sum(force[self.pinned], axis=0)
        free = ~self.pinned
        self.vb[free] += dt * force[free] / self.mb[free, None]
        speed = np.linalg.norm(self.vb[free], axis=1)
        self.vb[free] *= np.minimum(1.0, 5.0 / np.maximum(speed, 1.e-12))[:, None]
        self.xb[free] += dt * self.vb[free]
        self.xb[self.pinned] = self.pin_position[self.pinned]
        self.vb[self.pinned] = 0.0

    def _liquid_contacts(self, active):
        ids = np.flatnonzero(active)
        if len(ids) < 2:
            return
        separation = .057
        for _ in range(2):
            pairs = cKDTree(self.xw[ids]).query_pairs(separation, output_type="ndarray")
            if len(pairs) == 0: break
            ia, ib = ids[pairs[:, 0]], ids[pairs[:, 1]]
            delta = self.xw[ib] - self.xw[ia]
            dist = np.linalg.norm(delta, axis=1)
            good = dist > 1.e-8
            ia, ib, delta, dist = ia[good], ib[good], delta[good], dist[good]
            if len(ia) == 0: break
            normal = delta / dist[:, None]
            corr = .46 * (separation - dist)[:, None] * normal
            accum = np.zeros_like(self.xw); count = np.zeros(self.nw)
            np.add.at(accum, ia, -corr); np.add.at(accum, ib, corr)
            np.add.at(count, ia, 1.0); np.add.at(count, ib, 1.0)
            touched = count > 0
            self.xw[touched] += accum[touched] / count[touched, None]
            rv = np.sum((self.vw[ib] - self.vw[ia]) * normal, axis=1)
            closing = rv < 0
            impulse = (-.52 * rv[closing])[:, None] * normal[closing]
            np.add.at(self.vw, ia[closing], -impulse)
            np.add.at(self.vw, ib[closing], impulse)

    def _give_cloth_impulse(self, impulse, z, theta):
        if len(impulse) == 0: return
        _, heights = self._ring_geometry()
        ring = np.clip(np.searchsorted(heights, z), 1, self.ring_count)
        sector = np.mod(np.rint(theta * self.ring_size / (2*np.pi)).astype(int), self.ring_size)
        nodes = 1 + (ring - 1) * self.ring_size + sector
        np.add.at(self.vb, nodes, impulse / self.mb[nodes, None])

    def _bag_collision(self, active, dt):
        ids = np.flatnonzero(active); total = np.zeros(3)
        if len(ids) == 0: return total
        radii, heights = self._ring_geometry()
        p = self.xw[ids]; r = np.linalg.norm(p[:, :2], axis=1)
        theta = np.arctan2(p[:, 1], p[:, 0])
        rim_z, rim_r, radius = heights[-1], radii[-1], .0275
        newly = (~self.outside[ids]) & (p[:, 2] > rim_z-.018) & (r > .82*rim_r)
        self.outside[ids[newly]] = True
        inside = ~self.outside[ids]
        side = (p[:, 2] < rim_z) & (p[:, 2] > heights[0]-radius)
        rb = np.interp(np.clip(p[:, 2], heights[0], rim_z), heights, radii)

        hit = inside & side & (r > rb-radius)
        if np.any(hit):
            q = ids[hit]; rr = np.maximum(r[hit], 1.e-9)
            outward = np.column_stack((p[hit, 0]/rr, p[hit, 1]/rr, np.zeros(len(q))))
            penetration = r[hit] - (rb[hit]-radius)
            self.xw[q, :2] -= penetration[:, None] * outward[:, :2]
            vn = np.sum(self.vw[q]*outward, axis=1)
            mag = self.mw[q] * (np.maximum(vn, 0)*1.08 + .18*penetration/dt)
            impulse = mag[:, None]*outward
            self.vw[q] -= impulse/self.mw[q, None]
            self._give_cloth_impulse(impulse, p[hit, 2], theta[hit]); total += impulse.sum(0)

        bottom = heights[0] + radius + .10*(r/max(rim_r, 1.e-6))**2
        hit = inside & (p[:, 2] < bottom) & (r < rb+.09)
        if np.any(hit):
            q = ids[hit]; penetration = bottom[hit]-p[hit, 2]
            self.xw[q, 2] += penetration
            mag = self.mw[q]*(np.maximum(-self.vw[q, 2], 0)*1.06 + .16*penetration/dt)
            impulse = np.zeros((len(q), 3)); impulse[:, 2] = -mag
            self.vw[q, 2] += mag/self.mw[q]
            nodes = np.r_[0, 1 + np.arange(0, self.ring_size, 4)]
            share = impulse.sum(0)/len(nodes)
            self.vb[nodes] += share/self.mb[nodes, None]; total += impulse.sum(0)

        hit = self.outside[ids] & side & (r < rb+radius)
        if np.any(hit):
            q = ids[hit]; rr = np.maximum(r[hit], 1.e-9)
            outward = np.column_stack((p[hit, 0]/rr, p[hit, 1]/rr, np.zeros(len(q))))
            penetration = rb[hit]+radius-r[hit]
            self.xw[q, :2] += penetration[:, None]*outward[:, :2]
            vn = np.sum(self.vw[q]*outward, axis=1)
            mag = self.mw[q]*(np.maximum(-vn, 0)*1.03 + .15*penetration/dt)
            impulse = -mag[:, None]*outward
            self.vw[q] += mag[:, None]*outward/self.mw[q, None]
            self.vw[q, :2] *= .985
            self._give_cloth_impulse(impulse, p[hit, 2], theta[hit]); total += impulse.sum(0)
        return total

    def _step(self, dt):
        self._emit(dt)
        active = self.mw > 0
        self.last_floor_force[:] = 0; self.last_interaction[:] = 0
        self.vw[active] += dt*self.gravity; self.xw[active] += dt*self.vw[active]
        self._liquid_contacts(active)
        self.last_interaction = self._bag_collision(active, dt)/dt
        ids = np.flatnonzero(active & (self.xw[:, 2] < self.floor_z+.0275))
        if len(ids):
            old = self.vw[ids, 2].copy(); self.xw[ids, 2] = self.floor_z+.0275
            self.vw[ids, 2] = np.maximum(0, -.08*old); self.vw[ids, :2] *= .91
            self.last_floor_force[2] = np.sum(self.mw[ids]*(self.vw[ids, 2]-old))/dt
        leaving = active & (np.linalg.norm(self.xw[:, :2], axis=1) > 2.4)
        if np.any(leaving):
            self.cumulative_outflow_p += np.sum(self.mw[leaving, None]*self.vw[leaving], axis=0)
            self.cumulative_source -= float(np.sum(self.mw[leaving]))
            self.mw[leaving] = 0; self.vw[leaving] = 0; self.xw[leaving] = (0, 0, -8)
        for _ in range(4): self._cloth_substep(dt*.25)
        bag_g = self.bag_mass*self.gravity
        water_g = float(np.sum(self.mw))*self.gravity
        self.cumulative_external[0] += dt*(bag_g+water_g+self.last_support+self.last_floor_force)
        self.cumulative_external[1] += dt*(bag_g+self.last_support)
        self.cumulative_external[2] += dt*(water_g+self.last_floor_force)
        self.time += dt; self.step_count += 1

    def _frame(self):
        position = np.vstack((self.xb, self.xw)); velocity = np.vstack((self.vb, self.vw))
        mass = np.r_[self.mb, self.mw]
        body_id = np.r_[np.zeros(self.nb, np.int32), np.ones(self.nw, np.int32)]
        n = len(mass)
        weights = np.zeros((3, n)); weights[0, mass > 0] = 1
        weights[1, :self.nb] = 1; weights[2, self.nb:] = (self.mw > 0)
        ext = np.zeros((3, 3, 3)); bg = self.bag_mass*self.gravity
        wg = float(np.sum(self.mw))*self.gravity
        ext[0, 0] = bg+wg; ext[0, 1] = self.last_support+self.last_floor_force
        ext[1, 0] = bg; ext[1, 1] = self.last_support
        ext[2, 0] = wg; ext[2, 1] = self.last_floor_force
        inflow = np.zeros((3, 3)); inflow[[0, 2]] = self.cumulative_inflow_p
        outflow = np.zeros((3, 3)); outflow[[0, 2]] = self.cumulative_outflow_p
        rim = self.xb[self.rim_indices]; rr = np.linalg.norm(rim[:, :2], axis=1)
        normals = np.vstack((np.column_stack((-rim[:, 0]/np.maximum(rr, 1.e-9),
                                               -rim[:, 1]/np.maximum(rr, 1.e-9),
                                               np.zeros(len(rim)))), [[0, 0, 1]]))
        bvel = np.vstack((self.vb[self.rim_indices], np.zeros((1, 3))))
        frame = StateFrame(
            time=float(self.time), position=position, velocity=velocity, mass=mass,
            body_id=body_id,
            attributes={"pinned": np.r_[self.pinned.astype(float), np.zeros(self.nw)],
                        "material_phase": body_id.astype(float),
                        "active": np.r_[np.ones(self.nb), (self.mw > 0).astype(float)],
                        "wetness": np.r_[np.zeros(self.nb), (self.mw > 0).astype(float)],
                        "mass_quantity": mass.copy()},
            fields={"boundary_normal": normals,
                    "boundary_target_velocity": np.zeros_like(normals),
                    "boundary_velocity": bvel,
                    "budget_cumulative_external_impulse": self.cumulative_external.copy(),
                    "budget_cumulative_inflow_momentum": inflow,
                    "budget_cumulative_outflow_momentum": outflow,
                    "budget_domain_weights": weights,
                    "budget_external_force": ext,
                    "cumulative_mass_source": np.array([0.0, self.cumulative_source]),
                    "interaction_force": np.vstack((self.last_interaction, -self.last_interaction))},
            scalars={"reference_momentum": 100.0,
                     "reference_velocity": self.nozzle_speed,
                     "water_rest_density": float(self.config["initial_conditions"]["water"]["rest_density"]),
                     "bag_total_mass": self.bag_mass,
                     "bag_thickness": float(self.config["initial_conditions"]["bag"]["thickness"])})
        frame.validate(); return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time-1.e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target-1.e-12:
            self._step(min(.01, target-self.time))
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"method": "spring shell coupled to conservative contact particles",
                "steps": int(self.step_count),
                "active_water_parcels": int(np.count_nonzero(self.mw)),
                "overflow_parcels": int(np.count_nonzero(self.outside & (self.mw > 0))),
                "cumulative_water_source_kg": float(self.cumulative_source)}
