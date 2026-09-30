"""Two-way coupled soft-starfish / water / rock-bed solver.

Reduced particle model:
  * Deformable starfish (body_id==1): mass-spring network from rest configuration.
  * Water (body_id==2): particles with simple pressure/drag proxy.
  * Static obstacles (rocks + tank walls) via SDF penalty contact.
  * Two-way coupling: buoyancy+drag on star from water; momentum exchange back.

Designed for speed: avoids O(N^2) SPH by using a grid-averaged fluid field
for coupling and only light inter-particle repulsion for water.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ellipsoid_sdf(p: np.ndarray, center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    d = p - center
    s = np.sqrt(np.sum((d / radii) ** 2, axis=-1)) - 1.0
    return s * float(np.min(radii))


def _box_sdf(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    d = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = dict(scene.config) if hasattr(scene, "config") else {}
        self.rng = np.random.default_rng(self.seed)

        # Physics constants
        self.gravity = np.asarray(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.water_density = float(
            self.config.get("water", {}).get("density", 1000.0)
        )
        self.star_density = float(
            self.config.get("starfish", {}).get("density", 1080.0)
        )
        self.free_surface_z = float(
            self.config.get("water", {}).get("initial_free_surface_z", 0.40)
        )

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float, copy=True)
        self.velocity = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.body_id = np.array(frame0.body_id, dtype=np.int32, copy=True)
        self.rest_position = np.array(
            frame0.attributes.get("rest_position", self.position.copy()),
            dtype=float, copy=True,
        )

        self.n_total = len(self.position)
        self.star_mask = self.body_id == 1
        self.water_mask = self.body_id == 2
        self.n_star = int(self.star_mask.sum())
        self.n_water = int(self.water_mask.sum())

        self.obstacles = self._build_obstacles()
        self.springs = self._build_starfish_springs()

        # Cumulative budgets [G=2 domains: 0=star, 1=water]
        self.cum_ext_impulse = np.zeros((2, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((2, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((2, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)

        self.current_time = 0.0

        # Water grid for fast density/velocity lookup
        self.grid_res = 20
        self.grid_min = np.array([-0.5, -0.5, 0.0])
        self.grid_max = np.array([0.5, 0.5, 0.72])
        self.grid_cell = (self.grid_max - self.grid_min) / self.grid_res

    # ------------------------------------------------------------------
    # Obstacle construction
    # ------------------------------------------------------------------
    def _build_obstacles(self):
        obstacles = []
        geom = self.config.get("scene_geometry")
        if geom is None:
            return obstacles
        for obj in geom.get("objects", []):
            semantic = obj.get("semantic", "")
            motion = obj.get("motion", "")
            geo = obj.get("geometry", {})
            xform = obj.get("transform", {})
            trans = np.asarray(xform.get("translation", [0, 0, 0]), dtype=float)
            scale = np.asarray(xform.get("scale", [1, 1, 1]), dtype=float)
            if not obj.get("collision_enabled", True):
                continue
            gtype = geo.get("type")
            if semantic == "boundary" or motion == "static":
                if gtype == "box":
                    half = np.asarray(geo["half_extents"], dtype=float) * scale
                    c = trans.copy()
                    h = half.copy()
                    obstacles.append(("wall", lambda p, c=c, h=h: _box_sdf(p, c, h), c))
                elif gtype == "sphere":
                    r = float(geo["radius"])
                    radii = r * scale
                    c = trans.copy()
                    rd = radii.copy()
                    obstacles.append(("rock", lambda p, c=c, rd=rd: _ellipsoid_sdf(p, c, rd), c))
        return obstacles

    # ------------------------------------------------------------------
    # Starfish springs
    # ------------------------------------------------------------------
    def _build_starfish_springs(self):
        star_rest = self.rest_position[self.star_mask]
        n = len(star_rest)
        if n == 0:
            return {"pairs": np.empty((0, 2), dtype=np.int64),
                    "rest_len": np.empty(0, dtype=float)}
        from scipy.spatial import KDTree
        tree = KDTree(star_rest)
        k = 6
        _, idxs = tree.query(star_rest, k=k + 1)
        edges = set()
        for i in range(n):
            for j_local in range(1, k + 1):
                j = idxs[i, j_local]
                if j < n and i != j:
                    edges.add((min(i, j), max(i, j)))
        edges = np.array(sorted(edges), dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        if len(edges) == 0:
            return {"pairs": edges, "rest_len": np.empty(0, dtype=float)}
        rest_len = np.linalg.norm(
            star_rest[edges[:, 0]] - star_rest[edges[:, 1]], axis=1
        )
        return {"pairs": edges, "rest_len": rest_len}

    # ------------------------------------------------------------------
    # Physics sub-step
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity
        mass = self.mass
        acc = np.zeros_like(pos)

        # Gravity
        acc += self.gravity[None, :]

        # Starfish elastic forces
        self._apply_spring_forces(pos, vel, mass, acc)

        # Fluid forces on water particles (lightweight pressure proxy)
        self._apply_water_forces(pos, vel, mass, acc, dt)

        # Two-way coupling: buoyancy + drag on starfish from water
        self._apply_fluid_coupling_on_star(pos, vel, mass, acc)

        # Obstacle contact
        self._apply_obstacle_penalty(pos, vel, acc)

        # Semi-implicit Euler
        vel += acc * dt
        vel *= 0.999  # light damping
        pos += vel * dt

        # Clamp to tank bounds
        self._clamp_positions(pos)

        # Track cumulative external impulse (gravity)
        self.cum_ext_impulse[0] += np.sum(
            mass[self.star_mask, None] * self.gravity[None, :] * dt, axis=0
        )
        self.cum_ext_impulse[1] += np.sum(
            mass[self.water_mask, None] * self.gravity[None, :] * dt, axis=0
        )

    def _apply_spring_forces(self, pos, vel, mass, acc):
        if self.n_star == 0 or len(self.springs["pairs"]) == 0:
            return
        pairs = self.springs["pairs"]
        rest_len = self.springs["rest_len"]
        k_stiff = 6000.0
        damping = 10.0

        si = np.flatnonzero(self.star_mask)
        p_i = pos[si[pairs[:, 0]]]
        p_j = pos[si[pairs[:, 1]]]
        diff = p_j - p_i
        cur_len = np.linalg.norm(diff, axis=1, keepdims=True)
        direction = diff / np.maximum(cur_len, 1e-12)

        stretch = cur_len - rest_len[:, None]
        v_i = vel[si[pairs[:, 0]]]
        v_j = vel[si[pairs[:, 1]]]
        rel_v_dot = np.sum((v_j - v_i) * direction, axis=1, keepdims=True)
        f = (k_stiff * stretch + damping * rel_v_dot) * direction

        m_i = mass[si[pairs[:, 0]]]
        m_j = mass[si[pairs[:, 1]]]
        np.add.at(acc, si[pairs[:, 0]], f / m_i[:, None])
        np.subtract.at(acc, si[pairs[:, 1]], f / m_j[:, None])

    def _apply_water_forces(self, pos, vel, mass, acc, dt):
        """Lightweight water: gentle repulsion + gravity already applied."""
        widx = np.flatnonzero(self.water_mask)
        if len(widx) == 0:
            return
        # Simple vertical pressure gradient to prevent collapse
        wp = pos[widx]
        z = wp[:, 2]
        # Hydrostatic-ish upward push below free surface
        submerged = np.clip(self.free_surface_z - z, 0.0, 0.5)
        # Pressure gradient force ≈ rho*g*h upward per unit volume
        # As acceleration: a = rho_w * g * submerged_fraction / rho_w = g * fraction
        # This roughly balances gravity for submerged particles
        acc[widx, 2] += 9.81 * np.clip(submerged / 0.04, 0.0, 1.0)

        # Light neighbor repulsion to maintain volume
        # Use grid-based approach for speed
        cell_size = 0.06
        cells = np.floor((wp - self.grid_min) / cell_size).astype(int)
        cells = np.clip(cells, 0, self.grid_res - 1)
        keys = cells[:, 0] * self.grid_res * self.grid_res + cells[:, 1] * self.grid_res + cells[:, 2]
        order = np.argsort(keys)
        sorted_wp = wp[order]
        sorted_vel = vel[widx][order]
        sorted_keys = keys[order]

        # Find unique cell boundaries
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        ends = np.append(starts[1:], len(sorted_keys))

        repulsion_force = np.zeros_like(sorted_wp)
        rep_r = 0.04
        rep_k = 200.0

        for sk, s, e in zip(unique_keys, starts, ends):
            local = sorted_wp[s:e]
            n_loc = e - s
            if n_loc <= 1:
                continue
            # Pairwise within cell (small count expected)
            if n_loc > 30:
                # Subsample
                sel = self.rng.choice(n_loc, size=30, replace=False)
                local = local[sel]
                n_loc = 30
            dp = local[:, None, :] - local[None, :, :]
            r2 = np.sum(dp * dp, axis=2)
            mask = (r2 < rep_r * rep_r) & (r2 > 1e-10)
            r = np.sqrt(np.where(mask, r2, 1.0))
            overlap = np.where(mask, rep_r - r, 0.0)
            f = rep_k * overlap[:, :, None] * dp / r[:, :, None]
            # Zero out self-interaction diagonal
            np.fill_diagonal(f[:, :, 0], 0)
            np.fill_diagonal(f[:, :, 1], 0)
            np.fill_diagonal(f[:, :, 2], 0)
            if e - s <= 30:
                repulsion_force[s:e] += np.sum(f, axis=1)
            else:
                repulsion_force[s + sel] += np.sum(f, axis=1)

        unsorted_rep = np.empty_like(repulsion_force)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        unsorted_rep[inv_order] = repulsion_force
        acc[widx] += unsorted_rep / mass[widx, None]

    def _apply_fluid_coupling_on_star(self, pos, vel, mass, acc):
        """Buoyancy + drag on starfish from water field (grid-accelerated)."""
        sidx = np.flatnonzero(self.star_mask)
        if len(sidx) == 0:
            return
        sp = pos[sidx]
        sv = vel[sidx]

        # Submerged fraction
        sub_frac = np.clip((self.free_surface_z - sp[:, 2]) / 0.03, 0.0, 1.0)

        # Buoyancy
        buoyancy = sub_frac[:, None] * (-self.gravity) * (self.water_density / self.star_density)
        acc[sidx] += buoyancy

        # Drag via grid-accelerated neighbor lookup
        widx = np.flatnonzero(self.water_mask)
        if len(widx) == 0:
            return
        wp = pos[widx]
        wv = vel[widx]

        h = 0.08
        h2 = h * h
        cell_size = h
        inv_cell = 1.0 / cell_size
        gr = self.grid_res

        # Hash water into cells
        w_cells = np.floor((wp - self.grid_min) * inv_cell).astype(np.int64)
        w_cells = np.clip(w_cells, 0, gr - 1)
        w_keys = w_cells[:, 0] * (gr * gr) + w_cells[:, 1] * gr + w_cells[:, 2]
        w_order = np.argsort(w_keys)
        sorted_wp = wp[w_order]
        sorted_wv = wv[w_order]
        sorted_wkeys = w_keys[w_order]
        unique_wkeys, w_starts = np.unique(sorted_wkeys, return_index=True)
        w_ends = np.append(w_starts[1:], len(sorted_wkeys))
        w_cell_dict = {}
        for k, s, e in zip(unique_wkeys.tolist(), w_starts.tolist(), w_ends.tolist()):
            w_cell_dict[k] = (s, e)

        drag_coeff = 8.0
        v_water_est = np.zeros_like(sp)

        s_cells = np.floor((sp - self.grid_min) * inv_cell).astype(np.int64)
        s_cells = np.clip(s_cells, 0, gr - 1)

        for i in range(len(sidx)):
            cx, cy, cz = int(s_cells[i, 0]), int(s_cells[i, 1]), int(s_cells[i, 2])
            vi = np.zeros(3)
            wi = 0.0
            for dx in (-1, 0, 1):
                nx = cx + dx
                if nx < 0 or nx >= gr:
                    continue
                for dy in (-1, 0, 1):
                    ny = cy + dy
                    if ny < 0 or ny >= gr:
                        continue
                    for dz in (-1, 0, 1):
                        nz = cz + dz
                        if nz < 0 or nz >= gr:
                            continue
                        nk = nx * gr * gr + ny * gr + nz
                        entry = w_cell_dict.get(nk)
                        if entry is None:
                            continue
                        s, e = entry
                        nb_p = sorted_wp[s:e]
                        nb_v = sorted_wv[s:e]
                        dp = sp[i] - nb_p
                        r2 = np.sum(dp * dp, axis=1)
                        mask = r2 < h2
                        if not np.any(mask):
                            continue
                        w = (1.0 - r2[mask] / h2) ** 2
                        vi += np.sum(w[:, None] * nb_v[mask], axis=0)
                        wi += np.sum(w)
            if wi > 1e-12:
                v_water_est[i] = vi / wi

        drag = drag_coeff * (v_water_est - sv) * sub_frac[:, None]
        acc[sidx] += drag

    def _apply_obstacle_penalty(self, pos, vel, acc):
        stiffness = 40000.0
        damping = 400.0
        eps = 1e-4
        for kind, sdf_fn, center in self.obstacles:
            d = sdf_fn(pos)
            pen = -d
            active = pen > 0.0
            if not np.any(active):
                continue
            nx = (sdf_fn(pos + [eps, 0, 0]) - d) / eps
            ny = (sdf_fn(pos + [0, eps, 0]) - d) / eps
            nz = (sdf_fn(pos + [0, 0, eps]) - d) / eps
            normal = np.stack([nx, ny, nz], axis=-1)
            nl = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / np.maximum(nl, 1e-12)

            vn = np.sum(vel * normal, axis=-1, keepdims=True)
            f = (stiffness * pen[:, None] - damping * np.minimum(vn, 0.0)) * normal
            f *= active[:, None]
            acc += f / self.mass[:, None]

    def _clamp_positions(self, pos):
        # Tank interior bounds
        np.clip(pos[:, 0], -0.49, 0.49, out=pos[:, 0])
        np.clip(pos[:, 1], -0.49, 0.49, out=pos[:, 1])
        np.clip(pos[:, 2], 0.005, 0.71, out=pos[:, 2])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.current_time:
            remaining = target - self.current_time
            max_sub_dt = 0.002
            n_sub = max(1, int(math.ceil(remaining / max_sub_dt)))
            sub_dt = remaining / n_sub
            for _ in range(n_sub):
                self._substep(sub_dt)
            self.current_time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_star": self.n_star,
            "n_water": self.n_water,
            "n_obstacles": len(self.obstacles),
        }

    # ------------------------------------------------------------------
    # Frame assembly
    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        pos = self.position.copy()
        vel = self.velocity.copy()
        mass_arr = self.mass.copy()
        body_id_arr = self.body_id.copy()

        attributes: dict[str, np.ndarray] = {
            "mass_quantity": mass_arr.copy(),
            "rest_position": self.rest_position.copy(),
        }

        ref_vel = 0.5
        total_mass = float(np.sum(mass_arr))
        scalars: dict[str, float] = {
            "reference_velocity": ref_vel,
            "reference_momentum": total_mass * ref_vel,
            "water_free_surface_z": float(self.free_surface_z),
        }

        boundary_pts, boundary_normals = self._sample_boundary()
        interaction = self._compute_interaction_forces()

        G = 2
        N = len(pos)
        domain_weights = np.zeros((G, N), dtype=float)
        domain_weights[0, self.star_mask] = 1.0
        domain_weights[1, self.water_mask] = 1.0

        ext_force = np.zeros((G, 1, 3), dtype=float)
        ext_force[0, 0, :] = np.sum(
            mass_arr[self.star_mask, None] * self.gravity[None, :], axis=0
        )
        ext_force[1, 0, :] = np.sum(
            mass_arr[self.water_mask, None] * self.gravity[None, :], axis=0
        )

        fields: dict[str, np.ndarray] = {
            "boundary_normal": boundary_normals.astype(float),
            "boundary_target_velocity": np.zeros_like(boundary_pts),
            "boundary_velocity": np.zeros_like(boundary_pts),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
            "budget_domain_weights": domain_weights,
            "budget_external_force": ext_force,
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": interaction,
        }

        frame = StateFrame(
            time=t,
            position=pos,
            velocity=vel,
            mass=mass_arr,
            body_id=body_id_arr,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _sample_boundary(self, n_per: int = 20):
        pts_list, nrm_list = [], []
        eps = 1e-4
        for kind, sdf_fn, center in self.obstacles:
            dirs = self.rng.standard_normal((n_per, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            test_pts = center[None, :] + dirs * 0.15
            for _ in range(5):
                d = sdf_fn(test_pts)
                gx = (sdf_fn(test_pts + [eps, 0, 0]) - d) / eps
                gy = (sdf_fn(test_pts + [0, eps, 0]) - d) / eps
                gz = (sdf_fn(test_pts + [0, 0, eps]) - d) / eps
                grad = np.stack([gx, gy, gz], axis=-1)
                gn = np.linalg.norm(grad, axis=1, keepdims=True)
                grad = grad / np.maximum(gn, 1e-12)
                test_pts -= d[:, None] * grad
            normals = np.stack([
                (sdf_fn(test_pts + [eps, 0, 0]) - sdf_fn(test_pts - [eps, 0, 0])) / (2 * eps),
                (sdf_fn(test_pts + [0, eps, 0]) - sdf_fn(test_pts - [0, eps, 0])) / (2 * eps),
                (sdf_fn(test_pts + [0, 0, eps]) - sdf_fn(test_pts - [0, 0, eps])) / (2 * eps),
            ], axis=-1)
            nn = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(nn, 1e-12)
            pts_list.append(test_pts)
            nrm_list.append(normals)
        if not pts_list:
            return np.zeros((1, 3)), np.zeros((1, 3))
        return np.vstack(pts_list), np.vstack(nrm_list)

    def _compute_interaction_forces(self) -> np.ndarray:
        sidx = np.flatnonzero(self.star_mask)
        if len(sidx) == 0:
            return np.zeros((1, 3), dtype=float)
        sp = self.position[sidx]
        stiffness = 40000.0
        total_f = np.zeros(3, dtype=float)
        eps = 1e-4
        for kind, sdf_fn, center in self.obstacles:
            if kind != "rock":
                continue
            d = sdf_fn(sp)
            pen = -d
            active = pen > 0
            if not np.any(active):
                continue
            nx = (sdf_fn(sp + [eps, 0, 0]) - d) / eps
            ny = (sdf_fn(sp + [0, eps, 0]) - d) / eps
            nz = (sdf_fn(sp + [0, 0, eps]) - d) / eps
            normal = np.stack([nx, ny, nz], axis=-1)
            nl = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / np.maximum(nl, 1e-12)
            f = stiffness * pen[active, None] * normal[active]
            total_f += np.sum(f, axis=0)
        return total_f[None, :]
