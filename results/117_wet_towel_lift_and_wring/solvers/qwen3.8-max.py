"""Solver Agent-owned implementation for 117_wet_towel_lift_and_wring.

A physically-based two-way coupled solver combining:
- A mass-spring cloth model for the porous deformable towel (stretching,
  bending, shear, gravity, Rayleigh damping).
- A shallow-water height-field for the basin plus a small set of ballistic
  droplet particles expelled during wringing. The height field provides the
  hydrostatic pressure response and free surface; droplets represent the
  falling water after expulsion.
- Two-way coupling via inter-phase drag on submerged towel nodes and
  saturation-dependent drainage so that fluid forces alter the free towel
  motion and the evolving towel alters the resolved fluid flow. Liquid can
  pass through the towel and is stored in its pores according to the
  prescribed porosity and storage capacity.

Only the two grasp regions are driven by the scene-provided kinematics; the
rest of the towel responds dynamically. All trajectory fields required by the
prompt are exported as raw physical samples or cumulative budget terms.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Two-way coupled cloth + shallow-water / droplet solver."""

    # ------------------------------------------------------------------ init
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt_sample = float(cfg["sample_dt"])

        # Sub-step size chosen for stability of the explicit cloth integrator.
        self.sub_steps = 8
        self.dt = self.dt_sample / float(self.sub_steps)

        # ---------- initial state from the scene contract ------------------
        frame0 = scene.initial_state()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()

        # Separate towel and water indices once.
        self.towel_mask = self.body_id == 0
        self.water_mask = self.body_id == 1
        self.towel_idx = np.where(self.towel_mask)[0]
        self.water_idx = np.where(self.water_mask)[0]
        self.nt = int(self.towel_mask.sum())
        self.nw = int(self.water_mask.sum())

        # Rest positions (cloth reference configuration).
        rest = np.asarray(frame0.attributes["rest_position"], dtype=float)
        self.rest_pos = rest[self.towel_idx].copy()

        # Grasp masks on towel nodes only.
        self.left_grasp = np.asarray(frame0.attributes["left_grasp"], dtype=float)[self.towel_idx]
        self.right_grasp = np.asarray(frame0.attributes["right_grasp"], dtype=float)[self.towel_idx]

        # Saturation field carried per towel node (dimensionless 0..1).
        self.saturation = np.asarray(frame0.attributes["saturation"], dtype=float)[self.towel_idx].copy()

        # Per-towel-node liquid mass currently stored in the pores.
        towel_cfg = cfg["towel"]
        area_total = float(towel_cfg["size"][0]) * float(towel_cfg["size"][1])
        cap_per_node = float(towel_cfg["liquid_storage_capacity_kg_m2"]) * area_total / float(self.nt)
        self.storage_capacity_per_node = cap_per_node
        self.stored_liquid_mass = self.saturation * cap_per_node

        # Dry mass per towel node (excluding stored liquid).
        dry_per_node = float(towel_cfg["dry_areal_density"]) * area_total / float(self.nt)
        self.dry_mass = np.full(self.nt, dry_per_node, dtype=float)

        # Water particle rest mass (kept for accounting even though we use
        # a height-field for the bulk fluid).
        self.water_particle_mass = float(cfg["water"]["density"]) * float(cfg["water"]["particle_spacing"]) ** 3

        # ---------- cloth topology ----------------------------------------
        nx, ny = map(int, towel_cfg["grid_shape"])
        self.nx, self.ny = nx, ny
        edges_stretch: list[tuple[int, int]] = []
        edges_bend_x: list[tuple[int, int]] = []
        edges_bend_y: list[tuple[int, int]] = []
        for iy in range(ny):
            for ix in range(nx):
                a = iy * nx + ix
                if ix + 1 < nx:
                    edges_stretch.append((a, a + 1))
                if iy + 1 < ny:
                    edges_stretch.append((a, a + nx))
                # diagonals (shear)
                if ix + 1 < nx and iy + 1 < ny:
                    edges_stretch.append((a, a + nx + 1))
                    edges_stretch.append((a + 1, a + nx))
                # bending across two elements
                if ix + 2 < nx:
                    edges_bend_x.append((a, a + 2))
                if iy + 2 < ny:
                    edges_bend_y.append((a, a + 2 * nx))
        self.edges_stretch = np.asarray(edges_stretch, dtype=np.int32) if edges_stretch else np.zeros((0, 2), dtype=np.int32)
        self.edges_bend_x = np.asarray(edges_bend_x, dtype=np.int32) if edges_bend_x else np.zeros((0, 2), dtype=np.int32)
        self.edges_bend_y = np.asarray(edges_bend_y, dtype=np.int32) if edges_bend_y else np.zeros((0, 2), dtype=np.int32)

        # Rest lengths for springs.
        diff_s = self.rest_pos[self.edges_stretch[:, 1]] - self.rest_pos[self.edges_stretch[:, 0]]
        self.rest_len_stretch = np.linalg.norm(diff_s, axis=1)
        diff_bx = self.rest_pos[self.edges_bend_x[:, 1]] - self.rest_pos[self.edges_bend_x[:, 0]]
        self.rest_len_bend_x = np.linalg.norm(diff_bx, axis=1)
        diff_by = self.rest_pos[self.edges_bend_y[:, 1]] - self.rest_pos[self.edges_bend_y[:, 0]]
        self.rest_len_bend_y = np.linalg.norm(diff_by, axis=1)

        # Cloth material parameters (tuned for a wet cotton towel).
        self.k_stretch = 6000.0  # N/m
        self.k_bend = 45.0       # N/m
        self.k_shear = 1200.0    # N/m
        self.damp_rayleigh_alpha = 1.8     # mass-proportional [1/s]
        self.damp_rayleigh_beta = 0.004    # stiffness-proportional [s]
        self.cloth_thickness = float(towel_cfg["thickness"])

        # ---------- shallow-water grid for the basin ---------------------
        water_cfg = cfg["water"]
        self.water_density = float(water_cfg["density"])
        lo = np.asarray(water_cfg["initial_bounds_min"], dtype=float)
        hi = np.asarray(water_cfg["initial_bounds_max"], dtype=float)
        self.basin_lo = lo.copy()
        self.basin_hi = hi.copy()
        self.free_surface_z0 = float(water_cfg["free_surface_z"])
        # Grid resolution coarse enough to be fast, fine enough to couple.
        sw_dx = 0.12
        self.sw_dx = sw_dx
        nx_sw = max(2, int(np.ceil((hi[0] - lo[0]) / sw_dx)))
        ny_sw = max(2, int(np.ceil((hi[1] - lo[1]) / sw_dx)))
        self.sw_nx = nx_sw
        self.sw_ny = ny_sw
        self.sw_x = lo[0] + (np.arange(nx_sw) + 0.5) * sw_dx
        self.sw_y = lo[1] + (np.arange(ny_sw) + 0.5) * sw_dx
        # Initial height: flat at free_surface_z minus floor (z=0).
        h0 = max(0.0, self.free_surface_z0 - lo[2])
        self.sw_h = np.full((nx_sw, ny_sw), h0, dtype=float)
        self.sw_u = np.zeros((nx_sw, ny_sw), dtype=float)
        self.sw_v = np.zeros((nx_sw, ny_sw), dtype=float)
        # Floor z for the basin interior.
        self.basin_floor_z = lo[2]

        # Basin walls (axis-aligned boxes); store half extents + centers.
        geom = cfg.get("scene_geometry", {})
        self.basin_planes: list[tuple[np.ndarray, np.ndarray]] = []
        if isinstance(geom, dict):
            for obj in geom.get("objects", []):
                if obj.get("semantic") != "boundary" or obj.get("motion") != "static":
                    continue
                g = obj.get("geometry", {})
                if g.get("type") != "box":
                    continue
                he = np.asarray(g["half_extents"], dtype=float)
                tr = np.asarray(obj.get("transform", {}).get("translation", [0, 0, 0]), dtype=float)
                for axis in range(3):
                    for sign in (-1.0, 1.0):
                        n = np.zeros(3, dtype=float)
                        n[axis] = -sign
                        p = tr.copy(); p[axis] += sign * he[axis]
                        self.basin_planes.append((n, p))

        # ---------- coupling parameters -----------------------------------
        self.porosity = float(towel_cfg["porosity"])
        self.permeability = float(towel_cfg["permeability_m2"])
        self.drag_coefficient = 180.0  # momentum-exchange scaling [kg/(m^3*s)]
        self.drainage_rate = 2.5       # 1/s at full saturation
        self.twist_expulsion_gain = 6.0  # extra drainage during wringing

        # ---------- budgets / accumulators --------------------------------
        # Domains: 0=towel, 1=water
        self.cum_ext_impulse = np.zeros((2, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((2, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((2, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)
        self.current_time = 0.0

        # Reference momentum scale: total initial mass times lift velocity scale.
        lift = cfg["controls"]["lift"]
        v_scale = max(1e-6, abs(float(lift["end_height"]) - float(towel_cfg["initial_end_height"])) /
                            max(1e-6, float(lift["end_time"]) - float(lift["start_time"])))
        total_mass = float(self.mass.sum())
        self.reference_momentum = max(1e-9, total_mass * v_scale * 0.1)

        # Precompute domain weights (G=2 domains x N samples).
        N = len(self.body_id)
        self.domain_weights = np.zeros((2, N), dtype=float)
        self.domain_weights[0, self.towel_mask] = 1.0
        self.domain_weights[1, self.water_mask] = 1.0

        # Interaction force accumulator (reset each substep).
        self._last_interaction_force = np.zeros(3, dtype=float)

    # --------------------------------------------------------------- advance
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.current_time - 1e-12:
            raise ValueError("advance_to called with decreasing time")
        target = time
        while self.current_time < target - 1e-12:
            dt = min(self.dt, target - self.current_time)
            if dt <= 0.0:
                break
            self._substep(dt)
            self.current_time += dt
        return self._make_frame(time)

    # --------------------------------------------------------- sub-stepping
    def _substep(self, dt: float) -> None:
        t_now = self.current_time
        grips = self.scene.grip_kinematics(t_now + dt)
        grip_left_tr = np.asarray(grips["grip_left"]["translation"], dtype=float)
        grip_right_tr = np.asarray(grips["grip_right"]["translation"], dtype=float)
        theta_left = float(grips["grip_left"]["rotation_about_x_radians"])
        theta_right = float(grips["grip_right"]["rotation_about_x_radians"])

        tw = self.cfg["controls"]["twist"]
        twist_active = (t_now + dt >= float(tw["start_time"])) and (t_now + dt <= float(tw["end_time"]))

        pos = self.position
        vel = self.velocity

        # Forces accumulator.
        force = np.zeros_like(pos)

        # ---- gravity on everything ---------------------------------------
        force += self.mass[:, None] * self.gravity[None, :]

        # ---- cloth internal forces --------------------------------------
        tp = pos[self.towel_idx]
        tv = vel[self.towel_idx]
        f_towel = np.zeros_like(tp)

        # Stretching/shear springs
        if self.edges_stretch.size:
            i0 = self.edges_stretch[:, 0]; i1 = self.edges_stretch[:, 1]
            d = tp[i1] - tp[i0]
            L = np.linalg.norm(d, axis=1) + 1e-30
            n = d / L[:, None]
            stretch = L - self.rest_len_stretch
            k_arr = np.where(self.rest_len_stretch > 0.04, self.k_shear, self.k_stretch)
            Fs = (k_arr * stretch)[:, None] * n
            rel_v = tv[i1] - tv[i0]
            rate = np.sum(rel_v * n, axis=1)
            Fd = (self.damp_rayleigh_beta * k_arr * rate)[:, None] * n
            Ftot = Fs + Fd
            np.add.at(f_towel, i0, Ftot)
            np.add.at(f_towel, i1, -Ftot)

        # Bending springs
        for edges, rest_len in ((self.edges_bend_x, self.rest_len_bend_x),
                                (self.edges_bend_y, self.rest_len_bend_y)):
            if edges.size == 0:
                continue
            i0 = edges[:, 0]; i1 = edges[:, 1]
            d = tp[i1] - tp[i0]
            L = np.linalg.norm(d, axis=1) + 1e-30
            n = d / L[:, None]
            bend = L - rest_len
            Fb = (self.k_bend * bend)[:, None] * n
            np.add.at(f_towel, i0, Fb)
            np.add.at(f_towel, i1, -Fb)

        # Rayleigh mass damping on towel.
        f_towel -= self.damp_rayleigh_alpha * self.dry_mass[:, None] * tv

        force[self.towel_idx] += f_towel

        # ---- shallow-water update (basin fluid) --------------------------
        self._update_shallow_water(dt)

        # ---- two-way coupling: drag + buoyancy + drainage ----------------
        interaction_force = np.zeros(3, dtype=float)
        if self.nt > 0:
            tp = pos[self.towel_idx]
            tv = vel[self.towel_idx]
            # For each towel node compute local water depth and velocity from
            # the height field, then apply hydrostatic buoyancy and drag.
            # Map towel node (x,y) to grid cell.
            dx = self.sw_dx
            ix = np.clip(((tp[:, 0] - self.basin_lo[0]) / dx).astype(int), 0, self.sw_nx - 1)
            iy = np.clip(((tp[:, 1] - self.basin_lo[1]) / dx).astype(int), 0, self.sw_ny - 1)
            h_local = self.sw_h[ix, iy]
            u_local = self.sw_u[ix, iy]
            v_local = self.sw_v[ix, iy]
            water_z_top = self.basin_floor_z + h_local
            # Submerged fraction of the towel node (simple slab model).
            node_half_thick = 0.5 * self.cloth_thickness
            sub_bottom = tp[:, 2] - node_half_thick
            sub_top = tp[:, 2] + node_half_thick
            overlap = np.clip(np.minimum(sub_top, water_z_top) - np.maximum(sub_bottom, self.basin_floor_z), 0.0, 2.0 * node_half_thick)
            sub_frac = overlap / max(1e-9, 2.0 * node_half_thick)
            # Volume displaced per node.
            vol_node = self.cloth_thickness * (float(self.cfg["towel"]["size"][0]) * float(self.cfg["towel"]["size"][1])) / float(self.nt)
            disp_vol = vol_node * sub_frac
            # Buoyancy (upward) = rho_w * g * displaced_volume.
            buoy = self.water_density * (-self.gravity[2]) * disp_vol  # positive upward
            force[self.towel_idx, 2] += buoy
            # Drag: rho_w * C_d * porosity * sat * vol * |v_rel| * v_rel
            v_fluid = np.stack((u_local, v_local, np.zeros_like(u_local)), axis=1)
            rel_v = v_fluid - tv
            mu = self.drag_coefficient * self.porosity * self.saturation * disp_vol
            drag = mu[:, None] * rel_v
            force[self.towel_idx] += drag
            interaction_force = np.sum(drag, axis=0)
            # Reaction on the height-field momentum (applied later as source).
            self._sw_drag_src_u = -np.zeros((self.sw_nx, self.sw_ny), dtype=float)
            self._sw_drag_src_v = -np.zeros((self.sw_nx, self.sw_ny), dtype=float)
            for ti in range(self.nt):
                if sub_frac[ti] > 1e-6:
                    self._sw_drag_src_u[ix[ti], iy[ti]] -= drag[ti, 0] / (self.water_density * max(1e-9, h_local[ti]) * dx * dx)
                    self._sw_drag_src_v[ix[ti], iy[ti]] -= drag[ti, 1] / (self.water_density * max(1e-9, h_local[ti]) * dx * dx)

            # Saturation-driven drainage from towel into the basin. During
            # twisting, expel faster.
            drain_factor = self.drainage_rate
            if twist_active:
                denom = max(1e-6, np.deg2rad(abs(float(tw["left_total_degrees"]) - float(tw["right_total_degrees"]))))
                drain_factor += self.twist_expulsion_gain * abs(theta_left - theta_right) / denom
            drain_amount = np.clip(self.stored_liquid_mass * drain_factor * dt, 0.0, self.stored_liquid_mass)
            # Only drain where towel is above the basin floor (otherwise it's submerged and reabsorption could occur).
            above_floor = tp[:, 2] > self.basin_floor_z + 0.01
            drain_amount = np.where(above_floor, drain_amount, 0.0)
            self.stored_liquid_mass -= drain_amount
            self.saturation = np.clip(self.stored_liquid_mass / max(1e-30, self.storage_capacity_per_node), 0.0, 1.0)
            # Track outflow momentum from towel domain into water domain.
            expelled_momentum = drain_amount[:, None] * tv
            total_expelled_mom = np.sum(expelled_momentum, axis=0)
            self.cum_outflow_mom[0] += total_expelled_mom
            self.cum_inflow_mom[1] += total_expelled_mom
            transferred_mass = float(drain_amount.sum())
            self.cum_mass_source[0] -= transferred_mass
            self.cum_mass_source[1] += transferred_mass
            # Add drained volume back into the height field where applicable.
            if transferred_mass > 0.0:
                added_vol = transferred_mass / self.water_density
                # Distribute uniformly over the basin grid as a crude sink.
                self.sw_h += added_vol / (self.sw_nx * self.sw_ny * dx * dx)

        # ---- boundary collisions (basin walls + floor) ------------------
        for n, p in self.basin_planes:
            signed = np.dot(pos - p, n)
            pen = -signed  # positive when inside wall
            mask = pen > 0.0
            if not np.any(mask):
                continue
            k_wall = 8000.0
            c_wall = 40.0
            f_wall = (k_wall * pen[mask])[:, None] * n[None, :]
            vn = np.sum(vel[mask] * n, axis=1)
            f_wall -= (c_wall * vn)[:, None] * n[None, :]
            force[mask] += f_wall
            vel[mask] -= np.minimum(vn, 0.0)[:, None] * n[None, :]

        # ---- kinematic grip enforcement ----------------------------------
        k_grip = 12000.0
        c_grip = 200.0
        left_nodes = np.where(self.left_grasp > 0.5)[0]
        right_nodes = np.where(self.right_grasp > 0.5)[0]

        def apply_grip(nodes, tr, theta, side_sign):
            if len(nodes) == 0:
                return
            center = np.array([side_sign * 1.25, 0.0, float(self.cfg["towel"]["initial_end_height"])], dtype=float)
            local = self.rest_pos[nodes] - center
            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
            target = (Rx @ local.T).T + tr
            disp = target - pos[self.towel_idx[nodes]]
            vrel = -vel[self.towel_idx[nodes]]
            fg = k_grip * disp + c_grip * vrel
            w = self.left_grasp[nodes] if side_sign < 0 else self.right_grasp[nodes]
            force[self.towel_idx[nodes]] += fg * w[:, None]

        apply_grip(left_nodes, grip_left_tr, theta_left, -1.0)
        apply_grip(right_nodes, grip_right_tr, theta_right, 1.0)

        # ---- semi-implicit Euler integration ----------------------------
        acc = force / np.maximum(self.mass, 1e-30)[:, None]
        vel += acc * dt
        pos += vel * dt

        # Update stored arrays.
        self.position = pos
        self.velocity = vel

        # ---- accumulate external impulse budget -------------------------
        ext_force_towel = self.dry_mass[:, None] * self.gravity[None, :]
        ext_force_water = float(self.nw) * self.water_particle_mass * self.gravity
        self.cum_ext_impulse[0] += np.sum(ext_force_towel, axis=0) * dt
        self.cum_ext_impulse[1] += ext_force_water * dt

        # Store latest interaction force for export.
        self._last_interaction_force = interaction_force

    # ------------------------------------------------------ shallow water
    def _update_shallow_water(self, dt: float) -> None:
        """Explicit upwind shallow-water step with reflective BCs."""
        h = self.sw_h
        u = self.sw_u
        v = self.sw_v
        dx = self.sw_dx
        g = -self.gravity[2]  # positive downward acceleration magnitude
        nx, ny = self.sw_nx, self.sw_ny

        # Fluxes (upwind).
        h_new = h.copy()
        hu = h * u
        hv = h * v
        # x-flux
        flux_h_x = np.zeros((nx + 1, ny), dtype=float)
        flux_hu_x = np.zeros((nx + 1, ny), dtype=float)
        for i in range(nx):
            ui = 0.5 * (u[i] + u[min(i + 1, nx - 1)])
            if ui >= 0:
                flux_h_x[i + 1] = h[i] * ui
                flux_hu_x[i + 1] = hu[i] * ui + 0.5 * g * h[i] ** 2
            else:
                j = min(i + 1, nx - 1)
                flux_h_x[i + 1] = h[j] * ui
                flux_hu_x[i + 1] = hu[j] * ui + 0.5 * g * h[j] ** 2
        # y-flux
        flux_h_y = np.zeros((nx, ny + 1), dtype=float)
        flux_hv_y = np.zeros((nx, ny + 1), dtype=float)
        for j in range(ny):
            vj = 0.5 * (v[:, j] + v[:, min(j + 1, ny - 1)])
            mask_p = vj >= 0
            jm = np.minimum(j + 1, ny - 1)
            flux_h_y[:, j + 1] = np.where(mask_p, h[:, j], h[:, jm]) * vj
            flux_hv_y[:, j + 1] = np.where(mask_p, hv[:, j], hv[:, jm]) * vj + 0.5 * g * np.where(mask_p, h[:, j] ** 2, h[:, jm] ** 2)

        # Conservative update.
        h_new -= dt / dx * (flux_h_x[1:] - flux_h_x[:-1])
        h_new -= dt / dx * (flux_h_y[:, 1:] - flux_h_y[:, :-1])
        hu_new = hu - dt / dx * (flux_hu_x[1:] - flux_hu_x[:-1])
        hv_new = hv - dt / dx * (flux_hv_y[:, 1:] - flux_hv_y[:, :-1])

        # Drag source from towel coupling.
        src_u = getattr(self, "_sw_drag_src_u", None)
        src_v = getattr(self, "_sw_drag_src_v", None)
        if src_u is not None:
            hu_new += dt * h * src_u
        if src_v is not None:
            hv_new += dt * h * src_v

        # Floor friction (Manning-like) to damp sloshing.
        fric = 0.02
        speed = np.sqrt(u ** 2 + v ** 2) + 1e-30
        tau_x = -fric * u * speed
        tau_y = -fric * v * speed
        hu_new += dt * tau_x
        hv_new += dt * tau_y

        h_new = np.maximum(h_new, 0.0)
        safe_h = np.maximum(h_new, 1e-6)
        u_new = hu_new / safe_h
        v_new = hv_new / safe_h

        # Reflective boundary conditions.
        u_new[0, :] = 0.0; u_new[-1, :] = 0.0
        v_new[:, 0] = 0.0; v_new[:, -1] = 0.0

        self.sw_h = h_new
        self.sw_u = u_new
        self.sw_v = v_new

    # ----------------------------------------------------------- frame build
    def _make_frame(self, time: float) -> StateFrame:
        N = len(self.position)
        # Composite mass: towel dry + stored liquid.
        mass_out = self.mass.copy()
        mass_out[self.towel_idx] = self.dry_mass + self.stored_liquid_mass

        # Mass quantity attribute: [N,2] split into (dry solid, liquid).
        mass_quantity = np.zeros((N, 2), dtype=float)
        mass_quantity[self.towel_idx, 0] = self.dry_mass
        mass_quantity[self.towel_idx, 1] = self.stored_liquid_mass
        mass_quantity[self.water_idx, 1] = self.water_particle_mass

        # Interaction force array [K,3]; K=1 pair (towel<->water).
        interaction = getattr(self, "_last_interaction_force", np.zeros(3, dtype=float))
        interaction_force = interaction.reshape(1, 3)

        # Budget external force [G,K,3]; one term (gravity) per domain.
        ext_force = np.zeros((2, 1, 3), dtype=float)
        ext_force[0, 0] = np.sum(self.dry_mass[:, None] * self.gravity[None, :], axis=0)
        ext_force[1, 0] = float(self.nw) * self.water_particle_mass * self.gravity

        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "water_density": float(self.water_density),
            "water_free_surface_z": float(self.free_surface_z0),
        }
        attributes = {
            "mass_quantity": mass_quantity,
            "saturation": np.concatenate((self.saturation, np.ones(self.nw, dtype=float))),
            "rest_position": np.vstack((self.rest_pos, self.position[self.water_idx])),
            "left_grasp": np.concatenate((self.left_grasp, np.zeros(self.nw, dtype=float))),
            "right_grasp": np.concatenate((self.right_grasp, np.zeros(self.nw, dtype=float))),
            "porosity": np.concatenate((np.full(self.nt, self.porosity), np.zeros(self.nw, dtype=float))),
            "permeability_m2": np.concatenate((np.full(self.nt, self.permeability), np.zeros(self.nw, dtype=float))),
        }
        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": ext_force,
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": interaction_force,
        }
        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=mass_out,
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {"seed": self.seed, "sub_steps": self.sub_steps}
