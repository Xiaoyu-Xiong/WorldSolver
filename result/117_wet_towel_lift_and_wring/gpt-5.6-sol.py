"""Deterministic wet porous-cloth / receiving-water solver.

The model uses a geometrically nonlinear spring shell for the towel, Lagrangian
water parcels for the receiving water, and a sub-grid pore reservoir at each
cloth node. Pore liquid transfers conservatively to Lagrangian parcels when
gravity and deformation drive drainage. Cloth and receiving water exchange
equal and opposite drag/buoyancy forces. Only the two grasp constraints supplied
by the frozen scene are imposed.
"""
from __future__ import annotations

import math

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit fixed-step multiphysics solver for the frozen scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.body = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.nt = int(np.count_nonzero(self.body == 0))
        self.n = len(self.x)
        self.nw = self.n - self.nt
        self.time = 0.0
        self.steps = 0

        towel = self.cfg["towel"]
        self.nx, self.ny = map(int, towel["grid_shape"])
        area = float(towel["size"][0]) * float(towel["size"][1])
        self.dry_node_mass = float(towel["dry_areal_density"]) * area / self.nt
        capacity = float(towel["liquid_storage_capacity_kg_m2"]) * area / self.nt
        self.capacity = np.full(self.nt, capacity, dtype=float)
        self.pore_mass = self.capacity * float(towel["initial_saturation"])
        self.water_mass = np.asarray(initial.mass[self.nt :], dtype=float).copy()
        self.rest = self.x[: self.nt].copy()

        # Nonlinear membrane, shear and bending spring network. Pair forces are
        # symmetric and therefore conserve momentum.
        edges: list[tuple[int, int, float, float]] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                a = iy * self.nx + ix
                if ix + 1 < self.nx:
                    edges.append((a, a + 1, 24.0, 0.040))
                if iy + 1 < self.ny:
                    edges.append((a, a + self.nx, 15.0, 0.034))
                if ix + 1 < self.nx and iy + 1 < self.ny:
                    edges.append((a, a + self.nx + 1, 8.0, 0.024))
                    edges.append((a + 1, a + self.nx, 8.0, 0.024))
                if ix + 2 < self.nx:
                    edges.append((a, a + 2, 1.15, 0.012))
                if iy + 2 < self.ny:
                    edges.append((a, a + 2 * self.nx, 0.75, 0.010))
        data = np.asarray(edges, dtype=float)
        self.ei = data[:, 0].astype(np.int32)
        self.ej = data[:, 1].astype(np.int32)
        self.ek = data[:, 2]
        self.ec = data[:, 3]
        self.erest = np.linalg.norm(self.rest[self.ej] - self.rest[self.ei], axis=1)

        grasp = float(towel["grasp_length"])
        self.left = np.flatnonzero(self.rest[:, 0] <= self.rest[:, 0].min() + grasp)
        self.right = np.flatnonzero(self.rest[:, 0] >= self.rest[:, 0].max() - grasp)
        self.grasp_mask = np.zeros(self.nt, dtype=bool)
        self.grasp_mask[self.left] = True
        self.grasp_mask[self.right] = True

        # Reserve samples as initially massless pore-outflow parcels. Their
        # displaced pool mass is put into the other pool samples, preserving the
        # configured initial receiving-water mass exactly.
        rng = np.random.default_rng(int(seed))
        carrier_count = min(280, max(80, self.nw // 12))
        candidate = np.arange(self.nw, dtype=np.int32)
        self.carriers = candidate[np.linspace(0, self.nw - 1, carrier_count, dtype=int)]
        is_carrier = np.zeros(self.nw, dtype=bool)
        is_carrier[self.carriers] = True
        displaced = float(np.sum(self.water_mass[self.carriers]))
        self.water_mass[self.carriers] = 0.0
        self.water_mass[~is_carrier] += displaced / np.count_nonzero(~is_carrier)
        self.carrier_released = np.zeros(carrier_count, dtype=bool)
        self.carrier_threshold = rng.uniform(0.0018, 0.0038, carrier_count)

        # Material-space pore outlets are spread over only the free towel.
        free_nodes = np.flatnonzero(~self.grasp_mask)
        pick = np.linspace(0, len(free_nodes) - 1, carrier_count, dtype=int)
        self.carrier_owner = free_nodes[pick]
        uv = self.rest[:, :2]
        outlet_uv = uv[self.carrier_owner]
        d2 = np.sum((uv[:, None, :] - outlet_uv[None, :, :]) ** 2, axis=2)
        self.node_outlet = np.argmin(d2, axis=1).astype(np.int32)
        offsets = np.zeros((carrier_count, 3), dtype=float)
        offsets[:, 1] = ((np.arange(carrier_count) % 5) - 2.0) * 0.004
        offsets[:, 2] = 0.009
        self.carrier_offset = offsets
        wc = self.nt + self.carriers
        self.x[wc] = self.x[self.carrier_owner] + offsets
        self.v[wc] = self.v[self.carrier_owner]

        # Fixed receiving-water neighbors provide a local immersed/porous
        # coupling stencil. The fluid samples themselves remain dynamic.
        pool_local = np.flatnonzero(~is_carrier)
        wp = self.x[self.nt + pool_local]
        nearest = np.empty(self.nt, dtype=np.int32)
        for start in range(0, self.nt, 64):
            q = self.x[start : min(start + 64, self.nt)]
            dist = np.sum((q[:, None, :] - wp[None, :, :]) ** 2, axis=2)
            nearest[start : start + len(q)] = pool_local[np.argmin(dist, axis=1)]
        self.near_water = nearest

        self.initial_attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.free_surface = float(self.cfg["water"]["free_surface_z"])
        self.basin_x = float(self.cfg["basin"]["interior_half_extents"][0])
        self.basin_y = float(self.cfg["basin"]["interior_half_extents"][1])
        self.max_h = 0.004

        # Rows: combined, towel, receiving water. Terms: gravity, grasp,
        # basin/ambient traction, and paired coupling.
        self.cum_external = np.zeros((3, 3), dtype=float)
        self.cum_inflow = np.zeros((3, 3), dtype=float)
        self.cum_outflow = np.zeros((3, 3), dtype=float)
        self.last_budget_force = np.zeros((3, 4, 3), dtype=float)
        self.last_interaction = np.zeros((2, 3), dtype=float)
        total_mass = float(np.sum(initial.mass))
        self.reference_momentum = max(1.0e-6, total_mass * math.sqrt(9.81 * 2.5))

    def _grasp_targets(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        poses = self.scene.grip_kinematics(float(t))
        targets = np.empty((len(self.left) + len(self.right), 3), dtype=float)
        ids = np.concatenate((self.left, self.right))
        p = 0
        for nodes, key, x0 in ((self.left, "grip_left", -1.25),
                               (self.right, "grip_right", 1.25)):
            pose = poses[key]
            ang = float(pose["rotation_about_x_radians"])
            c, s = math.cos(ang), math.sin(ang)
            trans = np.asarray(pose["translation"], dtype=float)
            rel = self.rest[nodes] - np.array([x0, 0.0, 0.72])
            out = np.empty_like(rel)
            out[:, 0] = trans[0] + rel[:, 0]
            out[:, 1] = trans[1] + c * rel[:, 1] - s * rel[:, 2]
            out[:, 2] = trans[2] + s * rel[:, 1] + c * rel[:, 2]
            targets[p : p + len(nodes)] = out
            p += len(nodes)
        return ids, targets

    @staticmethod
    def _add_pairs(force: np.ndarray, i: np.ndarray, j: np.ndarray,
                   pair_force: np.ndarray) -> None:
        np.add.at(force, i, pair_force)
        np.add.at(force, j, -pair_force)

    def _substep(self, h: float) -> None:
        grip_ids, previous_grip_target = self._grasp_targets(self.time)
        xt = self.x[: self.nt]
        vt = self.v[: self.nt]
        mt = self.dry_node_mass + self.pore_mass
        ft = mt[:, None] * self.gravity

        delta = xt[self.ej] - xt[self.ei]
        length = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-9)
        direction = delta / length[:, None]
        relative = np.sum((vt[self.ej] - vt[self.ei]) * direction, axis=1)
        scalar = self.ek * (length - self.erest) + self.ec * relative
        pair = scalar[:, None] * direction
        self._add_pairs(ft, self.ei, self.ej, pair)

        # Current deformation, rather than a control envelope, drives wringing.
        edge_strain = np.abs(length / self.erest - 1.0)
        strain_sum = np.zeros(self.nt, dtype=float)
        strain_count = np.zeros(self.nt, dtype=float)
        np.add.at(strain_sum, self.ei, edge_strain)
        np.add.at(strain_sum, self.ej, edge_strain)
        np.add.at(strain_count, self.ei, 1.0)
        np.add.at(strain_count, self.ej, 1.0)
        strain = strain_sum / np.maximum(strain_count, 1.0)

        # Porous immersed coupling permits through-flow: it is drag plus the
        # buoyancy of solid fibers, not a no-penetration contact constraint.
        submerged = np.clip((self.free_surface + 0.04 - xt[:, 2]) / 0.10, 0.0, 1.0)
        wf_idx = self.nt + self.near_water
        relv = self.v[wf_idx] - vt
        node_area = (float(self.cfg["towel"]["size"][0]) *
                     float(self.cfg["towel"]["size"][1]) / self.nt)
        fiber_volume = (node_area * float(self.cfg["towel"]["thickness"]) *
                        (1.0 - float(self.cfg["towel"]["porosity"])))
        buoy = np.zeros_like(ft)
        buoy[:, 2] = (float(self.cfg["water"]["density"]) * 9.81 *
                      fiber_volume * submerged)
        drag = (0.010 * submerged)[:, None] * relv
        coupling = buoy + drag
        ft += coupling

        fw = self.water_mass[:, None] * self.gravity
        # Hydrostatic pressure balances gravity in quiescent submerged parcels
        # and vanishes above the free surface, leaving drops ballistic.
        wz = self.x[self.nt :, 2]
        hydro = np.clip((self.free_surface + 0.025 - wz) / 0.075, 0.0, 1.0)
        ambient = np.zeros_like(fw)
        ambient[:, 2] = -self.water_mass * self.gravity[2] * hydro
        ambient -= (0.55 * hydro * self.water_mass)[:, None] * self.v[self.nt :]
        fw += ambient
        np.add.at(fw, self.near_water, -coupling)

        # Semi-implicit Euler for all unconstrained degrees of freedom.
        self.v[: self.nt] += h * ft / mt[:, None]
        positive = self.water_mass > 1.0e-12
        self.v[self.nt :][positive] += h * fw[positive] / self.water_mass[positive, None]
        self.x[: self.nt] += h * self.v[: self.nt]
        self.x[self.nt :][positive] += h * self.v[self.nt :][positive]

        boundary_impulse_t = np.zeros(3, dtype=float)
        boundary_impulse_w = np.zeros(3, dtype=float)
        # Basin collision projections and their measured impulses.
        for offset, count, masses, accumulator in (
            (0, self.nt, mt, boundary_impulse_t),
            (self.nt, self.nw, self.water_mass, boundary_impulse_w),
        ):
            px = self.x[offset : offset + count]
            pv = self.v[offset : offset + count]
            active = masses > 1.0e-12
            for axis, lo, hi in ((0, -self.basin_x, self.basin_x),
                                 (1, -self.basin_y, self.basin_y)):
                low = active & (px[:, axis] < lo)
                high = active & (px[:, axis] > hi)
                for mask, bound, inward in ((low, lo, 1.0), (high, hi, -1.0)):
                    if np.any(mask):
                        old = pv[mask].copy()
                        px[mask, axis] = bound
                        pv[mask, axis] = (inward *
                                          np.maximum(inward * pv[mask, axis], 0.0) * 0.12)
                        tangent = [q for q in range(3) if q != axis]
                        pv[np.ix_(mask, tangent)] *= 0.82
                        accumulator += np.sum(masses[mask, None] *
                                              (pv[mask] - old), axis=0)
            floor = active & (px[:, 2] < 0.002)
            if np.any(floor):
                old = pv[floor].copy()
                px[floor, 2] = 0.002
                pv[floor, 2] = np.maximum(-0.10 * pv[floor, 2], 0.0)
                pv[floor, :2] *= 0.78
                accumulator += np.sum(masses[floor, None] *
                                      (pv[floor] - old), axis=0)

        # Enforce only labeled grasp nodes, measuring the required impulse.
        _, grip_target = self._grasp_targets(self.time + h)
        old_grip_v = self.v[grip_ids].copy()
        grip_v = (grip_target - previous_grip_target) / h
        self.x[grip_ids] = grip_target
        self.v[grip_ids] = grip_v
        control_impulse = np.sum(mt[grip_ids, None] *
                                 (grip_v - old_grip_v), axis=0)

        # Gravity drainage plus strain-generated pore pressure. Quadratic
        # Forchheimer inversion makes fast wringing explicitly nonlinear.
        exposed = np.clip((self.x[: self.nt, 2] - self.free_surface) / 0.30, 0.0, 1.0)
        pressure = 1800.0 * strain * strain + 120.0 * exposed
        a, b = 1.0, 0.012
        through_speed = (np.sqrt(a * a + 4.0 * b * pressure) - a) / (2.0 * b)
        saturation = self.pore_mass / np.maximum(self.capacity, 1.0e-12)
        rate = saturation * exposed * (0.006 + 0.0017 * through_speed)
        rate = np.minimum(rate, 0.42)
        dm = np.minimum(self.pore_mass, self.capacity * rate * h)
        dm[self.grasp_mask] *= 0.25
        self.pore_mass -= dm
        parcel_dm = np.zeros(len(self.carriers), dtype=float)
        np.add.at(parcel_dm, self.node_outlet, dm)
        self.water_mass[self.carriers] += parcel_dm

        transfer_velocity = self.v[self.carrier_owner]
        transfer_momentum = np.sum(parcel_dm[:, None] * transfer_velocity, axis=0)
        self.cum_outflow[1] += transfer_momentum
        self.cum_inflow[2] += transfer_momentum

        wc = self.nt + self.carriers
        newly_released = ((~self.carrier_released) &
                          (self.water_mass[self.carriers] >= self.carrier_threshold))
        if np.any(newly_released):
            self.carrier_released[newly_released] = True
            self.v[wc[newly_released]] = self.v[self.carrier_owner[newly_released]]
        held = ~self.carrier_released
        self.x[wc[held]] = self.x[self.carrier_owner[held]] + self.carrier_offset[held]
        self.v[wc[held]] = self.v[self.carrier_owner[held]]

        gravity_t = np.sum(mt[:, None] * self.gravity, axis=0)
        gravity_w = np.sum(self.water_mass[:, None] * self.gravity, axis=0)
        boundary_t = boundary_impulse_t / h
        boundary_w = np.sum(ambient, axis=0) + boundary_impulse_w / h
        control = control_impulse / h
        couple_t = np.sum(coupling, axis=0)
        couple_w = -couple_t
        bf = np.zeros((3, 4, 3), dtype=float)
        bf[1, 0], bf[2, 0] = gravity_t, gravity_w
        bf[1, 1] = control
        bf[1, 2], bf[2, 2] = boundary_t, boundary_w
        bf[1, 3], bf[2, 3] = couple_t, couple_w
        bf[0] = bf[1] + bf[2]
        self.last_budget_force = bf
        self.last_interaction[0] = couple_t
        self.last_interaction[1] = couple_w
        self.cum_external += np.sum(bf, axis=1) * h
        self.time += h
        self.steps += 1

    def _frame(self) -> StateFrame:
        mass = np.concatenate((self.dry_node_mass + self.pore_mass, self.water_mass))
        mass_quantity = np.zeros((self.n, 2), dtype=float)
        mass_quantity[: self.nt, 0] = self.dry_node_mass
        mass_quantity[: self.nt, 1] = self.pore_mass
        mass_quantity[self.nt :, 1] = self.water_mass
        attrs = {key: value.copy() for key, value in self.initial_attributes.items()}
        attrs["saturation"] = np.concatenate((
            self.pore_mass / np.maximum(self.capacity, 1.0e-12),
            np.ones(self.nw, dtype=float),
        ))
        attrs["mass_quantity"] = mass_quantity
        weights = np.zeros((3, self.n), dtype=float)
        weights[0, :] = 1.0
        weights[1, : self.nt] = 1.0
        weights[2, self.nt :] = 1.0
        frame = StateFrame(
            time=float(self.time), position=self.x.copy(), velocity=self.v.copy(),
            mass=mass, body_id=self.body.copy(), attributes=attrs,
            fields={
                "budget_cumulative_external_impulse": self.cum_external.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "budget_domain_weights": weights,
                "budget_external_force": self.last_budget_force.copy(),
                "cumulative_mass_source": np.zeros(2, dtype=float),
                "interaction_force": self.last_interaction.copy(),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "water_density": float(self.cfg["water"]["density"]),
                "water_free_surface_z": float(self.free_surface),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time + 1.0e-12 < target:
            self._substep(min(self.max_h, target - self.time))
        self.time = target
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            raise FloatingPointError("non-finite coupled state")
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "nonlinear spring shell + Lagrangian porous water parcels",
            "substeps": int(self.steps),
            "remaining_pore_water_kg": float(np.sum(self.pore_mass)),
            "released_droplet_parcels": int(np.count_nonzero(self.carrier_released)),
        }
