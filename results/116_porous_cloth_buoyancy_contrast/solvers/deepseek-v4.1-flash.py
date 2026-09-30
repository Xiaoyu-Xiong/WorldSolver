"""Physically coupled porous-cloth / liquid solver.

Model
-----
* Liquid: particle domain integrated with a weakly-compressible position-based
  fluid (PBF, Macklin et al.) projection.  Density is estimated with a poly6
  kernel over a uniform *rest lattice* calibration, so the frozen water block is
  the reference incompressible state.  Only non-negative pressure (compression)
  is resolved, which is the physical free-surface behaviour.
* Cloth: three deformable triangular grids integrated with XPBD distance
  constraints (structural, shear, bending) and per-node masses taken from the
  frozen scene.  Low bending stiffness lets the sheets fold and bend.
* Coupling (two-way): every cloth node samples the surrounding liquid particles
  and receives a hydrostatic/pressure (buoyancy) force based on its *solid*
  volume (thickness * element area, never a bounding box), an inertial drag
  force, a Darcy (permeability) drag and an added-mass inertia.  The exact
  opposite impulse is distributed back onto the sampled liquid particles, so
  momentum is exchanged both ways.  The cloth imposes no impermeable interface:
  the density projection is applied to the liquid alone and liquid particles may
  freely cross the sheet, as required for a porous fabric.

All state is advanced by real time stepping (dt = 5 ms) and sampled at the
requested times.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Two-way coupled porous-cloth buoyancy contrast solver."""

    # ---- solver choices (numerics / material law, not scene scripting) ----
    substep = 5.0e-3          # internal physical time step [s]
    kernel_scale = 2.0        # PBF smoothing length / particle spacing
    pbf_iterations = 2
    pbf_eps = 1.0e-9
    xsph = 0.08               # XSPH viscosity strength
    xpbd_iterations = 3
    cd_inertial = 0.6         # inertial drag coefficient of the fabric
    added_mass_coeff = 0.5    # entrained liquid mass per solid volume
    dynamics_viscosity = 1.0e-3
    cloth_velocity_damping = 0.5   # 1/s
    substep_max_disp = 0.25   # fraction of smoothing length per PBF iteration
    surface_ramp = 0.04       # [m] immersion ramp for the pressure force
    max_speed = 20.0

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        meta = scene.metadata()
        self._meta = meta
        self._g = np.asarray(meta["gravity"], dtype=float).reshape(3)
        tank = meta["tank"]
        self._lo = np.asarray(tank["interior_bounds_min"], dtype=float).reshape(3)
        self._hi = np.asarray(tank["interior_bounds_max"], dtype=float).reshape(3)
        cloth = meta["cloth"]
        self._nx, self._ny = (int(v) for v in cloth["grid_shape"])
        size = np.asarray(cloth["size"], dtype=float).reshape(2)
        self._thickness = float(cloth["thickness"])
        self._porosity = float(cloth["porosity"])
        self._permeability = float(cloth["permeability_m2"])
        self._area_per_vertex = float(np.prod(size)) / (2.0 * (self._nx - 1) * (self._ny - 1))
        # Solid (fiber) volume of one node's tributary area: the physical volume
        # that can displace water.  A cloth bounding box is never used.
        self._solid_volume = self._thickness * self._area_per_vertex
        self._body_ids = tuple(int(cloth["regimes"][name]["body_id"]) for name in ("light", "neutral", "heavy"))

        frame = scene.initial_state()
        position = np.asarray(frame.position, dtype=float)
        velocity = np.asarray(frame.velocity, dtype=float)
        mass = np.asarray(frame.mass, dtype=float)
        body = np.asarray(frame.body_id)
        iw = np.flatnonzero(body == 0)
        ic = np.flatnonzero(body != 0)
        self._nw = int(iw.size)
        self._nc = int(ic.size)
        self._wx = np.array(position[iw], dtype=float)
        self._wv = np.array(velocity[iw], dtype=float)
        self._wm_vec = np.array(mass[iw], dtype=float)
        self._wm = float(self._wm_vec[0])
        self._cx = np.array(position[ic], dtype=float)
        self._cv = np.array(velocity[ic], dtype=float)
        self._cm = np.array(mass[ic], dtype=float)
        self._body_c = np.asarray(body[ic], dtype=np.int32)
        self._water_density = float(meta["water"]["density"])

        self._attr_water: dict[str, np.ndarray] = {}
        self._attr_cloth: dict[str, np.ndarray] = {}
        for key, value in frame.attributes.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim >= 1 and arr.shape[0] == len(position):
                self._attr_water[key] = np.array(arr[iw])
                self._attr_cloth[key] = np.array(arr[ic])

        # Body index groups.
        self._bidx = {b: np.flatnonzero(self._body_c == b) for b in self._body_ids}

        # --- liquid kernel calibration from the frozen lattice -------------
        water_bounds = np.asarray(frame.fields["water_initial_bounds"], dtype=float)
        volume = float(np.prod(water_bounds[1] - water_bounds[0]))
        spacing = (volume / max(self._nw, 1)) ** (1.0 / 3.0)
        self._w_mass = self._wm
        self._h = self.kernel_scale * spacing
        self._h2 = self._h * self._h
        self._poly6 = 315.0 / (64.0 * np.pi * self._h ** 9)
        self._grad6 = -945.0 / (32.0 * np.pi * self._h ** 9)
        self._hs = 1.7 * spacing
        center = 0.5 * (water_bounds[0] + water_bounds[1])
        seed_particle = int(np.argmin(np.einsum("ij,ij->i", self._wx - center, self._wx - center)))
        rho0 = self._density(self._wx)[seed_particle]
        self._rho0 = float(rho0)
        self._delta_q = 0.2 * self._h
        self._w_dq = self._poly6 * (self._h2 - self._delta_q ** 2) ** 3

        # --- cloth topology and rest lengths --------------------------------
        self._build_cloth_constraints()

        # --- cached per-node derived quantities ------------------------------
        self._added_mass = np.full(self._nc, self.added_mass_coeff * self._water_density * self._solid_volume)
        self._m_tot = self._cm + self._added_mass
        self._drag_coeff = 0.5 * self.cd_inertial * self._water_density * self._area_per_vertex
        self._darcy_coeff = (self.dynamics_viscosity / max(self._permeability, 1e-30)) * self._solid_volume

        self._sub = np.ones(self._nc)
        self._vw_local = np.zeros((self._nc, 3))
        self._last_fluid_force = np.zeros((self._nc, 3))

        # --- budget bookkeeping ---------------------------------------------
        self._domains = 5  # 0: global, 1: liquid, 2..4: cloth bodies
        self._terms = 3    # 0: gravity, 1: fluid-solid coupling, 2: boundary contact
        self._dom_weights = np.zeros((self._domains, self._nw + self._nc))
        self._dom_weights[0, :] = 1.0
        self._dom_weights[1, : self._nw] = 1.0
        for row, b in enumerate(self._body_ids):
            self._dom_weights[2 + row, self._nw + self._bidx[b]] = 1.0
        self._cum_ext_impulse = np.zeros((self._domains, 3))
        self._cum_inflow = np.zeros((self._domains, 3))
        self._cum_outflow = np.zeros((self._domains, 3))
        self._interaction = np.zeros((self._terms, 3))
        self._cum_mass_source = np.zeros(4)
        self._ext_force = np.zeros((self._domains, self._terms, 3))

        total_mass = float(np.sum(self._wm_vec) + np.sum(self._cm))
        self._reference_momentum = float(total_mass * np.sqrt(abs(self._g[2]) * max(self._hi[2], 1e-3)) * 0.05)

        self._t = 0.0
        self._steps = 0

    # ------------------------------------------------------------------ setup
    def _build_cloth_constraints(self) -> None:
        nx, ny, nbody = self._nx, self._ny, len(self._body_ids)
        edges: list[tuple[int, int]] = []
        kinds: list[int] = []
        for b in range(nbody):
            off = b * nx * ny
            for iy in range(ny):
                for ix in range(nx):
                    a = off + iy * nx + ix
                    if ix + 1 < nx:
                        edges.append((a, a + 1)); kinds.append(0)
                    if iy + 1 < ny:
                        edges.append((a, a + nx)); kinds.append(0)
                    if ix + 1 < nx and iy + 1 < ny:
                        edges.append((a, a + nx + 1)); kinds.append(1)
                        edges.append((a + 1, a + nx)); kinds.append(1)
                    if ix + 2 < nx:
                        edges.append((a, a + 2)); kinds.append(2)
                    if iy + 2 < ny:
                        edges.append((a, a + 2 * nx)); kinds.append(2)
        self._edges = np.asarray(edges, dtype=np.int64)
        kinds = np.asarray(kinds, dtype=np.int64)
        d = self._cx[self._edges[:, 0]] - self._cx[self._edges[:, 1]]
        self._L0 = np.sqrt(np.einsum("ij,ij->i", d, d))
        stiffness = np.where(kinds == 0, 20000.0, np.where(kinds == 1, 8000.0, 20.0))
        self._alpha = 1.0 / stiffness
        self._edge_w = 1.0 / self._cm
        self._inv_mass_c = 1.0 / self._cm

    # ------------------------------------------------------------- kernels
    def _density(self, pos: np.ndarray) -> np.ndarray:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self._h, output_type="ndarray")
        rho = np.full(len(pos), self._poly6 * self._h2 ** 3 * self._w_mass)
        if pairs.shape[0]:
            d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            r2 = np.einsum("ij,ij->i", d, d)
            keep = r2 < self._h2
            i, j, r2 = pairs[keep, 0], pairs[keep, 1], r2[keep]
            w = self._poly6 * (self._h2 - r2) ** 3 * self._w_mass
            np.add.at(rho, i, w)
            np.add.at(rho, j, w)
        return rho

    # ------------------------------------------------------------ simulation
    def _coupling(self) -> None:
        """Fluid forces on cloth nodes and the equal/opposite liquid reaction."""
        tree = cKDTree(self._wx)
        neigh = tree.query_ball_point(self._cx, self._hs)
        z_surf = np.full(self._nc, -np.inf)
        self._vw_local[:] = 0.0
        weights: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        hs2 = self._hs * self._hs
        for k in range(self._nc):
            nb = np.asarray(neigh[k], dtype=np.int64)
            if nb.size == 0:
                weights.append(nb); indices.append(nb)
                continue
            d = self._wx[nb] - self._cx[k]
            r2 = np.einsum("ij,ij->i", d, d)
            keep = r2 < hs2
            nb = nb[keep]; r2 = r2[keep]
            if nb.size == 0:
                weights.append(nb); indices.append(nb)
                continue
            w = (hs2 - r2) ** 3
            wsum = float(np.sum(w))
            self._vw_local[k] = (w[:, None] * self._wv[nb]).sum(axis=0) / wsum
            z_surf[k] = float(np.max(self._wx[nb, 2]))
            weights.append(w / wsum)
            indices.append(nb)

        zc = self._cx[:, 2]
        subm = np.clip(0.5 + (z_surf - zc) / self.surface_ramp, 0.0, 1.0)
        subm[~np.isfinite(z_surf)] = np.where(zc < self._hi[2], 0.0, 0.0)
        self._sub = subm

        up = -self._g  # buoyancy acts opposite to gravity
        f_buoy = self._water_density * self._solid_volume * subm[:, None] * up[None, :]
        vrel = self._cv - self._vw_local
        speed = np.linalg.norm(vrel, axis=1, keepdims=True)
        kd = self._drag_coeff * speed + self._darcy_coeff
        f_fluid = f_buoy - kd * vrel
        # Explicit drag stability guard: never remove more than half the
        # relative velocity in a single substep.
        dv = self.substep * kd / self._m_tot[:, None]
        cap = np.where(dv > 0.5, 0.5 / np.maximum(dv, 1e-30), 1.0)
        f_fluid = f_buoy - kd * vrel * cap
        self._last_fluid_force = f_fluid

        # Equal and opposite reaction, distributed by the same kernel weights.
        reaction = np.zeros((self._nw, 3))
        for k in range(self._nc):
            nb = indices[k]
            if nb.size == 0:
                continue
            np.add.at(reaction, nb, -(weights[k][:, None] * f_fluid[k][None, :]))
        return reaction

    def _pbf(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self._w_mass
        for _ in range(self.pbf_iterations):
            tree = cKDTree(pos)
            pairs = tree.query_pairs(self._h, output_type="ndarray")
            if pairs.shape[0] == 0:
                break
            d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            r2 = np.einsum("ij,ij->i", d, d)
            keep = r2 < self._h2
            i, j, d, r2 = pairs[keep, 0], pairs[keep, 1], d[keep], r2[keep]
            diff = self._h2 - r2
            w = self._poly6 * diff ** 3
            gw = self._grad6 * diff ** 2  # (1/r) dW/dr , negative
            rho = np.full(len(pos), self._poly6 * self._h2 ** 3 * m)
            np.add.at(rho, i, w * m)
            np.add.at(rho, j, w * m)
            c = np.maximum(rho / self._rho0 - 1.0, 0.0)  # pressure cannot be tensile
            gci = (m / self._rho0) * gw[:, None] * d
            g2 = np.zeros(len(pos))
            np.add.at(g2, i, np.einsum("ij,ij->i", gci, gci))
            np.add.at(g2, j, np.einsum("ij,ij->i", gci, gci))
            lam = -c / (g2 + self.pbf_eps)
            s_corr = -0.1 * (w / self._w_dq) ** 4
            scale = (lam[i] + lam[j] + s_corr)[:, None] * gci
            delta = np.zeros_like(pos)
            np.add.at(delta, i, scale)
            np.add.at(delta, j, -scale)
            mag = np.linalg.norm(delta, axis=1, keepdims=True)
            lim = self.substep_max_disp * self._h
            delta = np.where(mag > lim, delta * (lim / np.maximum(mag, 1e-30)), delta)
            pos = pos + delta
        # XSPH viscosity keeps the free surface coherent.
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self._h, output_type="ndarray")
        if pairs.shape[0] and self.xsph > 0.0:
            d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            r2 = np.einsum("ij,ij->i", d, d)
            keep = r2 < self._h2
            i, j, r2 = pairs[keep, 0], pairs[keep, 1], r2[keep]
            w = self._poly6 * (self._h2 - r2) ** 3
            dv = (self.xsph * w / max(self._rho0, 1e-12))[:, None] * (vel[j] - vel[i])
            acc = np.zeros_like(vel)
            np.add.at(acc, i, dv)
            np.add.at(acc, j, -dv)
            vel = vel + acc
        return pos, vel

    def _xpbd_cloth(self) -> None:
        a, b = self._edges[:, 0], self._edges[:, 1]
        wa, wb = self._edge_w[a], self._edge_w[b]
        alpha_t = self._alpha / (self.substep * self.substep)
        lam = np.zeros(len(a))
        for _ in range(self.xpbd_iterations):
            d = self._cx[a] - self._cx[b]
            length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            c = length - self._L0
            dl = (-c - alpha_t * lam) / (wa + wb + alpha_t)
            lam = lam + dl
            corr = (dl / length)[:, None] * d
            np.add.at(self._cx, a, wa[:, None] * corr)
            np.add.at(self._cx, b, -wb[:, None] * corr)

    def _apply_bounds(self, pos: np.ndarray, old: np.ndarray, mass: np.ndarray,
                      margin: float, friction: float, dt: float):
        """Clamp samples inside the frozen tank; return velocities and contact impulse."""
        v_before = (pos - old) / dt
        for ax in range(3):
            low = self._lo[ax] + margin
            high = self._hi[ax] - margin
            np.clip(pos[:, ax], low, high, out=pos[:, ax])
        v_after = (pos - old) / dt
        impulse = mass[:, None] * (v_after - v_before)
        if friction > 0.0:
            floor = self._lo[2] + margin
            touching = pos[:, 2] <= floor + 1e-6
            if np.any(touching):
                damp = np.clip(1.0 - friction * dt * 20.0, 0.0, 1.0)
                v_after[touching, :2] *= damp
                impulse[touching] = mass[touching, None] * (v_after[touching] - v_before[touching])
        return v_after, impulse

    def _record_budget(self, reaction: np.ndarray, jc: np.ndarray, jw: np.ndarray, dt: float) -> None:
        ext = np.zeros((self._domains, self._terms, 3))
        gw = self._wm_vec.sum() * self._g
        ext[0, 0] += gw
        ext[1, 0] += gw
        fluid_sum = self._last_fluid_force.sum(axis=0)
        ext[1, 1] += -fluid_sum
        contact_w = jw.sum(axis=0) / dt
        ext[1, 2] += contact_w
        ext[0, 2] += contact_w + jc.sum(axis=0) / dt
        ext[0, 0] += self._cm.sum() * self._g
        for row, b in enumerate(self._body_ids):
            idx = self._bidx[b]
            ext[2 + row, 0] += self._cm[idx].sum() * self._g
            ext[2 + row, 1] += self._last_fluid_force[idx].sum(axis=0)
            ext[2 + row, 2] += jc[idx].sum(axis=0) / dt
        self._cum_ext_impulse += ext.sum(axis=1) * dt

        # Momentum crossing the liquid/fabric interfaces (mass transfer and
        # permeable flow).  Internal to the global domain, so it cancels there.
        jw_coupling = -fluid_sum * dt
        self._cum_inflow[1] += np.maximum(jw_coupling, 0.0)
        self._cum_outflow[1] += np.maximum(-jw_coupling, 0.0)
        for row, b in enumerate(self._body_ids):
            jb = self._last_fluid_force[self._bidx[b]].sum(axis=0) * dt
            self._cum_inflow[2 + row] += np.maximum(jb, 0.0)
            self._cum_outflow[2 + row] += np.maximum(-jb, 0.0)

        self._interaction[0] = fluid_sum
        self._interaction[1] = -fluid_sum
        self._interaction[2] = jc.sum(axis=0) / dt
        self._ext_force = ext

    def _step(self, dt: float) -> None:
        reaction = self._coupling()
        # --- integrate cloth velocities (gravity + coupled fluid force) -----
        self._cv += dt * (self._g[None, :] + self._last_fluid_force / self._m_tot[:, None])
        speed = np.linalg.norm(self._cv, axis=1)
        big = speed > self.max_speed
        if np.any(big):
            self._cv[big] *= (self.max_speed / speed[big])[:, None]
        self._cv *= max(0.0, 1.0 - self.cloth_velocity_damping * dt)
        # --- integrate liquid velocities ------------------------------------
        self._wv += dt * (self._g[None, :] + reaction / self._wm)
        wspeed = np.linalg.norm(self._wv, axis=1)
        bigw = wspeed > self.max_speed
        if np.any(bigw):
            self._wv[bigw] *= (self.max_speed / wspeed[bigw])[:, None]

        # --- predict ---------------------------------------------------------
        cx_old = self._cx.copy()
        wx_old = self._wx.copy()
        self._cx = self._cx + dt * self._cv
        self._wx = self._wx + dt * self._wv

        # --- constraint projections -----------------------------------------
        self._xpbd_cloth()
        self._wx, self._wv = self._pbf(self._wx, self._wv)

        # --- boundaries and contact impulses ---------------------------------
        v_after_c, jc = self._apply_bounds(self._cx, cx_old, self._cm, 0.004, 0.6, dt)
        self._cv = v_after_c
        v_after_w, jw = self._apply_bounds(self._wx, wx_old, self._wm_vec, 0.01, 0.0, dt)
        self._wv = v_after_w

        self._record_budget(reaction, jc, jw, dt)
        self._attr_cloth["saturation"] = np.array(self._sub)
        self._t += dt
        self._steps += 1

    # ------------------------------------------------------------ interface
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self._t < target - 1e-12 and guard < 10_000_000:
            dt = min(self.substep, target - self._t)
            if dt <= 1e-9:
                break
            self._step(dt)
            guard += 1
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        n = self._nw + self._nc
        position = np.vstack((self._wx, self._cx))
        velocity = np.vstack((self._wv, self._cv))
        mass = np.concatenate((self._wm_vec, self._cm))
        body_id = np.concatenate((np.zeros(self._nw, dtype=np.int32), self._body_c))

        mass_quantity = np.zeros((n, 4))
        mass_quantity[: self._nw, 0] = self._wm
        for col, b in enumerate(self._body_ids):
            idx = self._nw + self._bidx[b]
            mass_quantity[idx, 1 + col] = self._cm[self._bidx[b]]

        attributes: dict[str, np.ndarray] = {}
        for key in self._attr_water:
            w_val = np.asarray(self._attr_water[key], dtype=float)
            c_val = np.asarray(self._attr_cloth.get(key, np.zeros((self._nc, 1))), dtype=float)
            if w_val.shape[0] != self._nw or c_val.shape[0] != self._nc:
                continue
            attributes[key] = np.concatenate((w_val, c_val))
        attributes["mass_quantity"] = mass_quantity

        fields = {
            "budget_cumulative_external_impulse": np.array(self._cum_ext_impulse),
            "budget_cumulative_inflow_momentum": np.array(self._cum_inflow),
            "budget_cumulative_outflow_momentum": np.array(self._cum_outflow),
            "budget_domain_weights": np.array(self._dom_weights),
            "budget_external_force": np.array(self._ext_force),
            "cumulative_mass_source": np.array(self._cum_mass_source),
            "interaction_force": np.array(self._interaction),
        }
        scalars = {
            "reference_momentum": float(self._reference_momentum),
            "water_density": float(self._water_density),
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, Any]:
        return {
            "substeps": int(self._steps),
            "time": float(self._t),
            "water_particles": int(self._nw),
            "cloth_nodes": int(self._nc),
            "rest_density": float(self._rho0),
            "reference_momentum": float(self._reference_momentum),
        }
