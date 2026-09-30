"""Two-way coupled water-sheet-on-cloth solver.

Implements a simplified but physically-motivated model:
- Cloth: mass-spring network integrated with semi-implicit Euler / PBD constraints.
- Water: weakly-compressible SPH-style particles with spatial hashing.
- Coupling: repulsive contact force between water and cloth triangles; reaction
  applied to cloth vertices by barycentric weighting. Impermeability enforced by
  reflecting water velocity across the local cloth normal when approaching from the
  supported side.

The goal is a qualitatively correct recreation of Figure 8 (Coupling Water and
Smoke to Thin Deformable and Rigid Shells, Guendelman et al., SIGGRAPH 2005):
water lands on a four-corner suspended cloth, spreads, sags it, then leaves as
thin sheets over the free edges into a receiving basin below.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Spatial hash for neighbor search
# ---------------------------------------------------------------------------

class SpatialHash:
    """Uniform-grid spatial hash for fast neighbor queries."""

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self.inv = 1.0 / self.cell_size
        self.buckets: dict[tuple[int, int, int], list[int]] = {}

    def clear(self) -> None:
        self.buckets.clear()

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (int(math.floor(x * self.inv)),
                int(math.floor(y * self.inv)),
                int(math.floor(z * self.inv)))

    def insert(self, idx: int, pos: NDArray[np.floating]) -> None:
        k = self._key(float(pos[0]), float(pos[1]), float(pos[2]))
        self.buckets.setdefault(k, []).append(idx)

    def build(self, positions: NDArray[np.floating]) -> None:
        self.clear()
        for i in range(len(positions)):
            self.insert(i, positions[i])

    def query_ball(self, center: NDArray[np.floating], radius: float) -> list[int]:
        out: list[int] = []
        cx = int(math.floor(float(center[0]) * self.inv))
        cy = int(math.floor(float(center[1]) * self.inv))
        cz = int(math.floor(float(center[2]) * self.inv))
        r_cells = max(1, int(math.ceil(radius * self.inv)))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    bucket = self.buckets.get((cx + dx, cy + dy, cz + dz))
                    if bucket is not None:
                        out.extend(bucket)
        return out


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    """Agent-owned solver implementing two-way water/cloth coupling."""

    # Tunables chosen for stability at sample_dt=0.02 with sub-stepping.
    SUB_STEPS = 8
    CLOTH_STRETCH_STIFFNESS = 600.0
    CLOTH_BEND_STIFFNESS = 4.0
    CLOTH_DAMPING = 0.98
    WATER_REST_DENSITY = 1000.0
    WATER_PRESSURE_STIFFNESS = 2000.0
    WATER_VISCOSITY = 0.05
    WATER_GRAVITY_SCALE = 1.0
    COUPLING_STIFFNESS = 8000.0
    COUPLING_RADIUS = 0.12
    BASIN_RESTITUTION = 0.2
    BASIN_FRICTION = 0.4

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # ---- Cloth setup --------------------------------------------------
        cloth_cfg = self.cfg["cloth"]
        topo_path = Path(__file__).resolve().parent.parent / "assets" / "suspended_cloth_topology.json"
        with open(topo_path) as f:
            topo = json.load(f)
        self.cloth_pos = np.array(topo["vertices"], dtype=np.float64)
        self.cloth_faces = np.array(topo["faces"], dtype=np.int32)
        nx, ny = map(int, cloth_cfg["grid_shape"])
        self.cloth_nx = nx
        self.cloth_ny = ny
        self.n_cloth = len(self.cloth_pos)
        total_mass = float(cloth_cfg["total_mass"])
        self.cloth_mass = np.full(self.n_cloth, total_mass / self.n_cloth, dtype=np.float64)
        self.cloth_vel = np.zeros_like(self.cloth_pos)
        pinned_idx = np.asarray(cloth_cfg["pinned_vertices"], dtype=int)
        self.pinned_mask = np.zeros(self.n_cloth, dtype=bool)
        self.pinned_mask[pinned_idx] = True
        self.cloth_pinned_pos = self.cloth_pos[self.pinned_mask].copy()

        # Build edge springs (structural + shear)
        edges_set: set[tuple[int, int]] = set()
        for j in range(ny):
            for i in range(nx):
                v = j * nx + i
                if i + 1 < nx:
                    edges_set.add((v, v + 1))
                if j + 1 < ny:
                    edges_set.add((v, v + nx))
                if i + 1 < nx and j + 1 < ny:
                    edges_set.add((v, v + nx + 1))
                    edges_set.add((v + 1, v + nx))
        self.cloth_edges = np.array(sorted(edges_set), dtype=np.int32)
        p0 = self.cloth_pos[self.cloth_edges[:, 0]]
        p1 = self.cloth_pos[self.cloth_edges[:, 1]]
        self.cloth_rest_len = np.linalg.norm(p1 - p0, axis=1)

        # Vertex-triangle adjacency for coupling force distribution
        self.vert_tris: list[list[int]] = [[] for _ in range(self.n_cloth)]
        for fi, tri in enumerate(self.cloth_faces):
            for vi in tri:
                self.vert_tris[vi].append(fi)

        # ---- Water setup --------------------------------------------------
        frame0 = scene.initial_state()
        body_id = np.asarray(frame0.body_id, dtype=int)
        water_mask = body_id == 1
        self.water_pos = np.array(frame0.position[water_mask], dtype=np.float64)
        self.water_vel = np.array(frame0.velocity[water_mask], dtype=np.float64)
        self.water_mass = np.array(frame0.mass[water_mask], dtype=np.float64)
        self.n_water = len(self.water_pos)
        self.water_density = float(self.cfg["water"]["density"])
        self.spacing = float(self.cfg["water"]["initial_particle_spacing"])
        self.h = self.spacing * 2.2  # smoothing radius
        self.grid = SpatialHash(self.h)

        # Precompute particle rest volume from initial spacing
        self.particle_volume = float(self.spacing ** 3)

        # Inflow bookkeeping
        inflow = self.cfg["inflow"]
        self.inflow_center = np.asarray(inflow["center"], dtype=np.float64)
        self.inflow_axis = np.asarray(inflow["axis"], dtype=np.float64)
        self.inflow_speed = float(inflow["speed"])
        self.inflow_radius = float(inflow["radius"])
        self.inflow_t0, self.inflow_t1 = map(float, inflow["active_interval"])
        self.inflow_initial_bottom = float(inflow["initial_column_bottom"])
        self.cumulative_inflow_mass = 0.0

        # Basin boundaries (closed floor + side walls, open top)
        self.basin_min = np.array([-2.05, -1.8, 0.0], dtype=np.float64)
        self.basin_max = np.array([2.05, 1.8, 3.5], dtype=np.float64)

        # Budget accumulators
        self.gravity = np.asarray(self.cfg["gravity"], dtype=np.float64)
        self.cum_ext_impulse = np.zeros(3, dtype=np.float64)
        self.cum_inflow_mom = np.zeros(3, dtype=np.float64)
        self.cum_outflow_mom = np.zeros(3, dtype=np.float64)

        # Simulation time tracking
        self.current_time = 0.0
        self.prev_water_count = self.n_water

        # Reference scales (frozen)
        self.ref_velocity = abs(self.inflow_speed)
        self.ref_momentum = float(self.water_density * self.spacing ** 3 * self.ref_velocity)

    # --------------------------------------------------------------------- #
    #                              Physics step                             #
    # --------------------------------------------------------------------- #

    def _substep(self, dt: float) -> None:
        """Advance one sub-step of length dt."""
        g = self.gravity

        # --- Cloth forces (stretch + bend + damping + gravity) ---------------
        cloth_force = np.zeros_like(self.cloth_pos)
        cloth_force += self.cloth_mass[:, None] * g

        # Stretch springs
        p0 = self.cloth_pos[self.cloth_edges[:, 0]]
        p1 = self.cloth_pos[self.cloth_edges[:, 1]]
        diff = p1 - p0
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        stretch = (dist - self.cloth_rest_len) / self.cloth_rest_len
        f_mag = self.CLOTH_STRETCH_STIFFNESS * stretch
        f_vec = (diff / dist[:, None]) * f_mag[:, None]
        np.add.at(cloth_force, self.cloth_edges[:, 0], f_vec)
        np.add.at(cloth_force, self.cloth_edges[:, 1], -f_vec)

        # Simple bending via Laplacian smoothing towards neighbors
        if self.CLOTH_BEND_STIFFNESS > 0:
            lap = np.zeros_like(self.cloth_pos)
            cnt = np.zeros(self.n_cloth, dtype=np.float64)
            np.add.at(lap, self.cloth_edges[:, 0], self.cloth_pos[self.cloth_edges[:, 1]])
            np.add.at(lap, self.cloth_edges[:, 1], self.cloth_pos[self.cloth_edges[:, 0]])
            np.add.at(cnt, self.cloth_edges[:, 0], 1.0)
            np.add.at(cnt, self.cloth_edges[:, 1], 1.0)
            cnt = np.maximum(cnt, 1.0)
            avg = lap / cnt[:, None]
            bend_f = self.CLOTH_BEND_STIFFNESS * (avg - self.cloth_pos)
            cloth_force += bend_f

        # Damping
        cloth_force -= (1.0 - self.CLOTH_DAMPING) / dt * self.cloth_mass[:, None] * self.cloth_vel

        # Semi-implicit Euler for cloth
        acc = cloth_force / self.cloth_mass[:, None]
        self.cloth_vel += acc * dt
        self.cloth_pos += self.cloth_vel * dt

        # Pin corners
        self.cloth_pos[self.pinned_mask] = self.cloth_pinned_pos
        self.cloth_vel[self.pinned_mask] = 0.0

        # --- Water: build spatial hash, compute density & forces -------------
        self.grid.build(self.water_pos)

        n_w = len(self.water_pos)
        if n_w == 0:
            return

        h = self.h
        h2 = h * h
        poly6_coeff = 315.0 / (64.0 * math.pi * h ** 9)
        spiky_coeff = -45.0 / (math.pi * h ** 6)
        visc_lap_coeff = 45.0 / (math.pi * h ** 6)

        densities = np.empty(n_w, dtype=np.float64)
        pressures = np.empty(n_w, dtype=np.float64)
        w_forces = np.zeros((n_w, 3), dtype=np.float64)
        w_forces += self.water_mass[:, None] * g * self.WATER_GRAVITY_SCALE

        for i in range(n_w):
            nbrs = self.grid.query_ball(self.water_pos[i], h)
            dens = 0.0
            pi = self.water_pos[i]
            for j in nbrs:
                r2 = float(np.sum((pi - self.water_pos[j]) ** 2))
                if r2 < h2:
                    dens += self.water_mass[j] * poly6_coeff * ((h2 - r2) ** 3)
            densities[i] = max(dens, 1.0)
            pressures[i] = self.WATER_PRESSURE_STIFFNESS * (densities[i] - self.WATER_REST_DENSITY)

        # Pressure + viscosity forces
        for i in range(n_w):
            nbrs = self.grid.query_ball(self.water_pos[i], h)
            pi = self.water_pos[i]
            vi = self.water_vel[i]
            rho_i = densities[i]
            p_i = pressures[i]
            fp = np.zeros(3, dtype=np.float64)
            fv = np.zeros(3, dtype=np.float64)
            for j in nbrs:
                if j == i:
                    continue
                rij = pi - self.water_pos[j]
                r = float(np.sqrt(np.sum(rij ** 2))) + 1e-12
                if r >= h:
                    continue
                grad_w = spiky_coeff * ((h - r) ** 2) * (rij / r)
                p_term = -self.water_mass[j] * (p_i + pressures[j]) / (2.0 * rho_i * densities[j]) * grad_w
                fp += p_term
                lap_w = visc_lap_coeff * (h - r)
                fv += self.WATER_VISCOSITY * self.water_mass[j] * (self.water_vel[j] - vi) / densities[j] * lap_w
            w_forces[i] += fp + fv

        # --- Two-way coupling: water <-> cloth -------------------------------
        interaction_force_total = np.zeros(3, dtype=np.float64)
        coupling_force_on_cloth = np.zeros_like(self.cloth_pos)

        cloth_z_mean = float(np.mean(self.cloth_pos[:, 2]))
        candidates = np.where(np.abs(self.water_pos[:, 2] - cloth_z_mean) < self.COUPLING_RADIUS * 2.5)[0]
        for wi in candidates:
            wp = self.water_pos[wi]
            best_d2 = 1e30
            best_bary = None
            best_tri = -1
            for fi in range(len(self.cloth_faces)):
                tri = self.cloth_faces[fi]
                a = self.cloth_pos[tri[0]]
                b = self.cloth_pos[tri[1]]
                c = self.cloth_pos[tri[2]]
                ab = b - a
                ac = c - a
                ap = wp - a
                d1 = float(np.dot(ab, ap))
                d2_ = float(np.dot(ac, ap))
                d3 = float(np.dot(ab, ab))
                d4 = float(np.dot(ab, ac))
                d5 = float(np.dot(ac, ac))
                denom = d3 * d5 - d4 * d4 + 1e-30
                v = (d5 * d1 - d4 * d2_) / denom
                w = (d3 * d2_ - d4 * d1) / denom
                u = 1.0 - v - w
                if u < 0 or v < 0 or w < 0:
                    continue
                proj = u * a + v * b + w * c
                dd = float(np.sum((wp - proj) ** 2))
                if dd < best_d2:
                    best_d2 = dd
                    best_bary = (u, v, w)
                    best_tri = fi

            if best_tri < 0:
                continue
            dist = math.sqrt(best_d2)
            if dist >= self.COUPLING_RADIUS:
                continue

            tri = self.cloth_faces[best_tri]
            a = self.cloth_pos[tri[0]]
            b = self.cloth_pos[tri[1]]
            c = self.cloth_pos[tri[2]]
            nrm = np.cross(b - a, c - a)
            nl = np.linalg.norm(nrm) + 1e-30
            nrm = nrm / nl
            if nrm[2] < 0:
                nrm = -nrm

            proj_pt = best_bary[0] * a + best_bary[1] * b + best_bary[2] * c
            signed = float(np.dot(wp - proj_pt, nrm))

            penetration = self.COUPLING_RADIUS - abs(signed)
            if penetration > 0:
                f_mag = self.COUPLING_STIFFNESS * penetration
                f_dir = nrm if signed >= 0 else -nrm
                f_on_water = f_dir * f_mag
                w_forces[wi] += f_on_water

                u_, v_, w_ = best_bary
                reaction = -f_on_water
                coupling_force_on_cloth[tri[0]] += u_ * reaction
                coupling_force_on_cloth[tri[1]] += v_ * reaction
                coupling_force_on_cloth[tri[2]] += w_ * reaction
                interaction_force_total += f_on_water

                vn = float(np.dot(self.water_vel[wi], nrm))
                if signed >= 0 and vn < 0:
                    self.water_vel[wi] -= 1.5 * vn * nrm
                elif signed < 0 and vn > 0:
                    self.water_vel[wi] -= 1.5 * vn * nrm

        # Apply coupling force to cloth
        cloth_acc_coupling = coupling_force_on_cloth / self.cloth_mass[:, None]
        self.cloth_vel += cloth_acc_coupling * dt
        self.cloth_pos += cloth_acc_coupling * dt * dt
        self.cloth_pos[self.pinned_mask] = self.cloth_pinned_pos
        self.cloth_vel[self.pinned_mask] = 0.0

        # --- Integrate water ------------------------------------------------
        w_acc = w_forces / self.water_mass[:, None]
        self.water_vel += w_acc * dt
        self.water_pos += self.water_vel * dt

        # --- Basin boundary collisions --------------------------------------
        floor_z = 0.0
        below = self.water_pos[:, 2] < floor_z
        if np.any(below):
            self.water_pos[below, 2] = floor_z
            self.water_vel[below, 2] *= -self.BASIN_RESTITUTION
            self.water_vel[below, :2] *= (1.0 - self.BASIN_FRICTION)

        for ax, lo, hi in [(0, -2.05, 2.05), (1, -1.8, 1.8)]:
            too_lo = self.water_pos[:, ax] < lo
            if np.any(too_lo):
                self.water_pos[too_lo, ax] = lo
                self.water_vel[too_lo, ax] *= -self.BASIN_RESTITUTION
            too_hi = self.water_pos[:, ax] > hi
            if np.any(too_hi):
                self.water_pos[too_hi, ax] = hi
                self.water_vel[too_hi, ax] *= -self.BASIN_RESTITUTION

        # Track external impulse from gravity on both domains this sub-step
        total_mass = float(np.sum(self.cloth_mass) + np.sum(self.water_mass))
        self.cum_ext_impulse += g * total_mass * dt

    def _inject_inflow(self, dt: float) -> None:
        """Add new water particles from the nozzle during active interval."""
        if not (self.inflow_t0 <= self.current_time < self.inflow_t1):
            return
        spacing = self.spacing
        r = self.inflow_radius
        n_rings = max(1, int(round(r / spacing)))
        new_pos = []
        new_vel = []
        for ring in range(n_rings + 1):
            rr = r * ring / max(n_rings, 1)
            n_pts = max(1, int(round(2 * math.pi * rr / spacing))) if ring > 0 else 1
            for k in range(n_pts):
                theta = 2 * math.pi * k / n_pts
                x = self.inflow_center[0] + rr * math.cos(theta)
                y = self.inflow_center[1] + rr * math.sin(theta)
                z = self.inflow_center[2]
                new_pos.append([x, y, z])
                new_vel.append(self.inflow_axis * self.inflow_speed)
        if not new_pos:
            return
        new_pos = np.array(new_pos, dtype=np.float64)
        new_vel = np.array(new_vel, dtype=np.float64)
        single_mass = float(self.water_density * self.particle_volume)
        new_mass = np.full(len(new_pos), single_mass, dtype=np.float64)

        self.water_pos = np.vstack((self.water_pos, new_pos))
        self.water_vel = np.vstack((self.water_vel, new_vel))
        self.water_mass = np.concatenate((self.water_mass, new_mass))
        added_mass = float(np.sum(new_mass))
        self.cumulative_inflow_mass += added_mass
        self.cum_inflow_mom += new_vel.sum(axis=0) * single_mass

    def _remove_outflow(self) -> None:
        """Remove particles that fell far below the basin to keep arrays bounded."""
        z_lo = -0.5
        keep = self.water_pos[:, 2] >= z_lo
        if not np.all(keep):
            removed_mom = np.sum(self.water_vel[~keep] * self.water_mass[~keep, None], axis=0)
            self.cum_outflow_mom += removed_mom
            self.water_pos = self.water_pos[keep]
            self.water_vel = self.water_vel[keep]
            self.water_mass = self.water_mass[keep]

    # --------------------------------------------------------------------- #
    #                          Public API                                   #
    # --------------------------------------------------------------------- #

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.current_time:
            return self._build_frame(self.current_time)

        sample_dt = float(self.cfg["sample_dt"])
        sub_dt = sample_dt / self.SUB_STEPS

        while self.current_time < target - 1e-12:
            remaining = target - self.current_time
            dt = min(sub_dt, remaining)
            self._inject_inflow(dt)
            self._substep(dt)
            self._remove_outflow()
            self.current_time += dt

        return self._build_frame(target)

    def _build_frame(self, time: float) -> StateFrame:
        n_c = self.n_cloth
        n_w = len(self.water_pos)
        n = n_c + n_w

        position = np.vstack((self.cloth_pos, self.water_pos)) if n_w > 0 else self.cloth_pos.copy()
        velocity = np.vstack((self.cloth_vel, self.water_vel)) if n_w > 0 else self.cloth_vel.copy()
        mass = np.concatenate((self.cloth_mass, self.water_mass)) if n_w > 0 else self.cloth_mass.copy()
        body_id = np.concatenate((np.zeros(n_c, dtype=np.int32),
                                  np.ones(n_w, dtype=np.int32))) if n_w > 0 else np.zeros(n_c, dtype=np.int32)

        # Attributes
        uv_cloth = np.column_stack((
            np.tile(np.linspace(0.0, 1.0, self.cloth_nx), self.cloth_ny),
            np.repeat(np.linspace(0.0, 1.0, self.cloth_ny), self.cloth_nx),
        ))
        uv = np.vstack((uv_cloth, np.zeros((n_w, 2)))) if n_w > 0 else uv_cloth
        pinned_attr = np.zeros(n, dtype=np.float64)
        pinned_attr[self.pinned_mask] = 1.0
        phase = body_id.astype(np.float64)

        attributes = {
            "uv": uv,
            "pinned": pinned_attr,
            "material_phase": phase,
            "mass_quantity": mass.copy(),
        }

        # Fields - boundary samples (basin walls + cloth vertex normals)
        basin_normals = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
        ], dtype=np.float64)
        cloth_vnorm = np.zeros_like(self.cloth_pos)
        for fi in range(len(self.cloth_faces)):
            tri = self.cloth_faces[fi]
            a = self.cloth_pos[tri[0]]
            b = self.cloth_pos[tri[1]]
            c = self.cloth_pos[tri[2]]
            nrm = np.cross(b - a, c - a)
            for vi in tri:
                cloth_vnorm[vi] += nrm
        norms = np.linalg.norm(cloth_vnorm, axis=1, keepdims=True) + 1e-30
        cloth_vnorm /= norms
        boundary_normal = np.vstack((basin_normals, cloth_vnorm))
        n_b = len(boundary_normal)
        boundary_velocity = np.zeros((n_b, 3), dtype=np.float64)
        boundary_velocity[5:] = self.cloth_vel
        boundary_target_velocity = np.zeros((n_b, 3), dtype=np.float64)

        # Budget fields - G=2 domains: [0]=cloth, [1]=water
        G = 2
        budget_domain_weights = np.zeros((G, n), dtype=np.float64)
        budget_domain_weights[0, :n_c] = 1.0
        if n_w > 0:
            budget_domain_weights[1, n_c:] = 1.0

        K = 1
        budget_external_force = np.zeros((G, K, 3), dtype=np.float64)
        budget_external_force[0, 0] = np.sum(self.cloth_mass) * self.gravity
        if n_w > 0:
            budget_external_force[1, 0] = np.sum(self.water_mass) * self.gravity

        m_tot = max(float(np.sum(mass)), 1e-30)
        m_c = float(np.sum(self.cloth_mass))
        m_w = float(np.sum(self.water_mass))
        budget_cum_ext_impulse = np.stack((
            self.cum_ext_impulse * m_c / m_tot,
            self.cum_ext_impulse * m_w / m_tot,
        ))

        interaction_force = np.zeros((K, 3), dtype=np.float64)

        cumulative_mass_source = np.array([self.cumulative_inflow_mass], dtype=np.float64)

        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_velocity,
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_impulse": budget_cum_ext_impulse,
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3).repeat(G, axis=0),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3).repeat(G, axis=0),
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": cumulative_mass_source,
            "interaction_force": interaction_force,
        }

        scalars = {
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
            "water_density": self.water_density,
            "cloth_thickness": float(self.cfg["cloth"]["thickness"]),
            "nominal_inflow_speed": float(self.cfg["inflow"]["speed"]),
            "nozzle_radius": float(self.cfg["inflow"]["radius"]),
        }

        frame = StateFrame(
            time=time,
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

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_cloth": int(self.n_cloth),
            "n_water": int(len(self.water_pos)),
            "current_time": float(self.current_time),
            "cumulative_inflow_mass": float(self.cumulative_inflow_mass),
        }

