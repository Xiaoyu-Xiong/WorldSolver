"""Physically-based particle solver for the water-on-cloth-curtain scene.

The solver advances a coupled cloth + water particle system with:
* mass-spring cloth network (pinned top row)
* weakly compressible particle-particle repulsion for the water
* two-way cloth-water contact via nearest-face penalty forces
* tank boundary penalties and continuous nozzle inflow

Integration uses explicit sub-stepping (symplectic Euler) with deterministic
sampling at the requested output times.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.scene = scene

        # ----- initial state from the frozen scene ---------------------------------
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.pinned = np.asarray(frame0.attributes["pinned"], dtype=np.float64).copy()
        self.material_phase = np.asarray(frame0.attributes["material_phase"], dtype=np.float64).copy()

        # bookkeeping
        self.cloth_mask = self.body_id == 0
        self.water_mask = self.body_id == 1
        self.n_cloth = int(self.cloth_mask.sum())
        self.cloth_idx0 = 0
        self.cloth_idx1 = self.n_cloth

        # ----- physical constants / parameters ------------------------------------
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.dt_sub = 1.0e-4  # fixed physics sub-step (seconds)
        self.max_substeps = 50_000_000  # safety guard (far above needed count)

        # water equation-of-state repulsion
        self.water_cutoff = 0.12  # m
        self.water_k = 80.0       # N/m scale for central repulsion
        self.water_c = 0.5        # damping along the contact line

        # cloth-water contact
        self.contact_h = 0.10     # contact margin (m)
        self.contact_k = 800.0    # penalty stiffness (N/m)
        self.contact_c = 5.0      # contact damping coefficient

        # tank boundaries (inner half-spaces, normal points into the tank)
        self.walls = [
            (np.array([1.0, 0.0, 0.0]), np.array([-1.58, 0.0, 0.0]), 0.05),   # left
            (np.array([-1.0, 0.0, 0.0]), np.array([2.08, 0.0, 0.0]), 0.05),   # right
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.38, 0.0]), 0.05),   # front
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.38, 0.0]), 0.05),   # back
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.00]), 0.05),    # floor
        ]
        self.boundary_k = 4000.0
        self.boundary_c = 10.0

        # cloth spring network
        self.cloth_k = 1.0e3   # structural spring stiffness (N/m)
        self.cloth_c = 5.0     # spring damping
        self.cloth_drag = 0.05
        self._build_cloth_topology()

        # inflow
        meta = scene.metadata()
        inflow = meta.get("inflow", {})
        self.nozzle_center = np.asarray(inflow.get("center", [-1.255, 0.0, 1.72]), dtype=np.float64)
        self.nozzle_radius = float(inflow.get("radius", 0.20))
        self.nozzle_velocity = np.asarray(inflow.get("velocity", [3.2, 0.0, 0.0]), dtype=np.float64)
        self.inflow_active = np.asarray(inflow.get("active_interval", [0.0, 4.0]), dtype=np.float64)

        # each water particle has this mass in the initial scene
        if self.water_mask.any():
            self.m_water = float(self.mass[self.water_mask][0])
        else:
            self.m_water = 1000.0 * 0.075 ** 3
        self.inflow_bank = 0.0
        self.cum_mass_source = np.zeros(2, dtype=np.float64)  # [cloth, water]
        self.contact_length_scale = self.water_cutoff

        # store rest positions for pinned vertices
        self.rest_position = self.position.copy()
        self.current_time = 0.0
        self._contact_sd = np.array([], dtype=np.float64)

    # -------------------------------------------------------------------------
    def _build_cloth_topology(self):
        """Generate the cloth quad grid topology from the frozen scene dimensions."""
        n_cloth = self.n_cloth
        ny = 31
        nz = 41
        self.cloth_ny = ny
        self.cloth_nz = nz

        # faces (two triangles per quad)
        faces = []
        for z in range(nz - 1):
            for y in range(ny - 1):
                a = z * ny + y
                b = a + 1
                c = a + ny
                d = c + 1
                faces.append((a, b, d))
                faces.append((a, d, c))
        self.cloth_faces = np.asarray(faces, dtype=np.int32)

        # spring edges: every immediate neighbor within the grid (8-connectivity)
        edges = []
        for z in range(nz):
            for y in range(ny):
                i = z * ny + y
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dz == 0 and dy == 0:
                            continue
                        nz2, ny2 = z + dz, y + dy
                        if 0 <= nz2 < nz and 0 <= ny2 < ny:
                            j = nz2 * ny + ny2
                            if i < j:
                                edges.append((i, j))
        self.cloth_edges = np.asarray(edges, dtype=np.int32)
        rest = self.position[self.cloth_edges[:, 1]] - self.position[self.cloth_edges[:, 0]]
        self.cloth_rest_len = np.linalg.norm(rest, axis=1)

    # -------------------------------------------------------------------------
    def _add_inflow(self, dt: float) -> int:
        """Spawn new water particles at the nozzle for this sub-step."""
        t = self.current_time
        if t < self.inflow_active[0] or t > self.inflow_active[1]:
            return 0
        if not (self.inflow_active[0] <= t <= self.inflow_active[1]):
            return 0

        # prescribed volume flux -> mass flux
        volume_flux = np.pi * self.nozzle_radius ** 2 * self.nozzle_velocity[0]
        mass_flux = 1000.0 * volume_flux
        self.inflow_bank += mass_flux * dt
        n_spawn = int(self.inflow_bank // self.m_water)
        if n_spawn <= 0:
            return 0

        new_pos = []
        new_vel = []
        for _ in range(n_spawn):
            # uniform disk in the nozzle cross-section (y-z plane)
            r = self.nozzle_radius * np.sqrt(self.rng.random())
            theta = 2.0 * np.pi * self.rng.random()
            offset = np.array([
                0.0,
                r * np.cos(theta),
                r * np.sin(theta),
            ], dtype=np.float64)
            pos = self.nozzle_center + offset
            pos[0] = -1.28  # nozzle exit plane
            # small random perturbation for a natural stream
            vel = self.nozzle_velocity + self.rng.normal(0.0, 0.10, size=3)
            vel[0] = max(vel[0], 0.5)
            new_pos.append(pos)
            new_vel.append(vel)

        new_pos = np.asarray(new_pos, dtype=np.float64)
        new_vel = np.asarray(new_vel, dtype=np.float64)
        n_new = new_pos.shape[0]

        # grow state arrays
        self.position = np.vstack([self.position, new_pos])
        self.velocity = np.vstack([self.velocity, new_vel])
        self.mass = np.concatenate([self.mass, np.full(n_new, self.m_water, dtype=np.float64)])
        self.body_id = np.concatenate([self.body_id, np.full(n_new, 1, dtype=np.int32)])
        self.pinned = np.concatenate([self.pinned, np.zeros(n_new, dtype=np.float64)])
        self.material_phase = np.concatenate([self.material_phase, np.ones(n_new, dtype=np.float64)])
        self.rest_position = np.vstack([self.rest_position, new_pos])  # not really used for water

        self.inflow_bank -= n_spawn * self.m_water
        self.cum_mass_source[1] += n_spawn * self.m_water

        self.cloth_mask = self.body_id == 0
        self.water_mask = self.body_id == 1
        return n_new

    # -------------------------------------------------------------------------
    def _cloth_face_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current cloth face centers and face velocities."""
        verts = self.position[: self.n_cloth]
        vel = self.velocity[: self.n_cloth]
        fc = verts[self.cloth_faces].mean(axis=1)
        fv = vel[self.cloth_faces].mean(axis=1)
        return fc, fv

    # -------------------------------------------------------------------------
    def _step(self) -> None:
        """Advance one physics sub-step."""
        # 1) inflow
        self._add_inflow(self.dt_sub)

        n = self.position.shape[0]
        pos = self.position
        vel = self.velocity
        mass = self.mass
        n_cloth = self.n_cloth

        # force accumulator
        f = np.zeros((n, 3), dtype=np.float64)

        # 2) external force: gravity
        f += mass[:, None] * self.gravity[None, :]

        # 3) cloth internal spring forces
        if n_cloth > 1:
            i = self.cloth_edges[:, 0]
            j = self.cloth_edges[:, 1]
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dir_ = diff / (dist + 1e-12)
            rel_v = vel[i] - vel[j]
            rel_dot = np.sum(rel_v * dir_, axis=1, keepdims=True)
            spring_mag = self.cloth_k * (dist[:, 0] - self.cloth_rest_len)[:, None] + self.cloth_c * rel_dot
            spring_force = -spring_mag * dir_

            # global cloth damping
            f[:n_cloth] -= self.cloth_drag * vel[:n_cloth] * mass[:n_cloth, None]

            # accumulate per vertex
            f_cloth = np.zeros((n_cloth, 3), dtype=np.float64)
            np.add.at(f_cloth, i, spring_force[:, 0])
            np.add.at(f_cloth, j, -spring_force[:, 0])
            f[:n_cloth] += f_cloth

        # 4) water-water repulsion
        water_idx = np.where(self.water_mask)[0]
        if water_idx.size > 0:
            wpos = pos[water_idx]
            wvel = vel[water_idx]
            tree = cKDTree(wpos)
            pairs = tree.sparse_distance_matrix(tree, self.water_cutoff, output_type="ndarray")
            if pairs.size:
                i = pairs["i"]
                j = pairs["j"]
                dists = pairs["v"]
                # keep unique unordered pairs, discard self
                mask = (i < j) & (dists > 1e-12)
                i = i[mask]
                j = j[mask]
                dists = dists[mask]
                if i.size:
                    diff = wpos[i] - wpos[j]
                    r = dists[:, None]
                    dir_ij = diff / r
                    q = r / self.water_cutoff
                    # central repulsion
                    f_mag = self.water_k * (1.0 - q)
                    # damping along the line of centers
                    vrel = wvel[i] - wvel[j]
                    vrel_n = np.sum(vrel * dir_ij, axis=1, keepdims=True)
                    pair_force = (f_mag + self.water_c * (1.0 - q) * vrel_n) * dir_ij

                    # map back to global indices
                    fg_w = np.zeros((water_idx.size, 3), dtype=np.float64)
                    np.add.at(fg_w, i, pair_force[:, 0])
                    np.add.at(fg_w, j, -pair_force[:, 0])
                    f[water_idx] += fg_w

        # 5) cloth-water contact (two-way)
        if water_idx.size > 0 and n_cloth > 0:
            fc, fv = self._cloth_face_data()
            tree = cKDTree(fc)
            dists, fi = tree.query(pos[water_idx], k=1)
            dists = dists.astype(np.float64)
            contact_dir = pos[water_idx] - fc[fi]
            r = np.linalg.norm(contact_dir, axis=1, keepdims=True)
            r = np.where(r < 1e-12, 1.0, r)
            dir_ = contact_dir / r
            vrel = vel[water_idx] - fv[fi]
            vrel_n = np.sum(vrel * dir_, axis=1, keepdims=True)
            penetration = self.contact_h - r[:, 0]
            active = penetration > 0.0
            if active.any():
                f_mag = self.contact_k * penetration[active] - self.contact_c * vrel_n[active, 0]
                force_on_water = f_mag[:, None] * dir_[active]

                # signed distance for diagnostics
                self._contact_sd = (r[active, 0] - self.contact_h).astype(np.float64)

                # water feels the push
                f_water_contact = np.zeros((water_idx.size, 3), dtype=np.float64)
                np.add.at(f_water_contact, np.where(active)[0], force_on_water)
                f[water_idx] += f_water_contact

                # cloth vertices feel the equal-and-opposite reaction (distributed)
                face_verts = self.cloth_faces[fi[active]]  # (A,3) per active particle
                reaction = -force_on_water / 3.0
                np.add.at(f[:n_cloth], face_verts[:, 0], reaction)
                np.add.at(f[:n_cloth], face_verts[:, 1], reaction)
                np.add.at(f[:n_cloth], face_verts[:, 2], reaction)
            else:
                self._contact_sd = np.array([], dtype=np.float64)
        else:
            self._contact_sd = np.array([], dtype=np.float64)

        # 6) tank boundary penalty (applies to both water and cloth)
        for n_vec, p0, margin in self.walls:
            sd = np.dot(pos - p0[None, :], n_vec)
            near = sd < margin
            if np.any(near):
                vn = np.sum(vel * n_vec[None, :], axis=1)
                pen = margin - sd
                mag = self.boundary_k * pen + self.boundary_c * (-vn)
                mag = np.where(near, mag, 0.0)
                f += (mag[:, None] * n_vec[None, :])

        # 7) integrate (semi-implicit Euler)
        a = f / mass[:, None]
        vel += a * self.dt_sub
        # pinned vertices do not move
        vel[:n_cloth][self.pinned[:n_cloth] > 0.5] = 0.0
        pos += vel * self.dt_sub
        # enforce pinned positions exactly
        pinned_cloth = self.pinned[:n_cloth] > 0.5
        pos[:n_cloth][pinned_cloth] = self.rest_position[:n_cloth][pinned_cloth]

        self.position = pos
        self.velocity = vel
        self.current_time += self.dt_sub

    # -------------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        """Build the required StateFrame for the current state."""
        n = self.position.shape[0]

        # attributes.mass_quantity: per-sample mass split by material
        mass_quantity = np.zeros((n, 2), dtype=np.float64)
        cloth_local = self.cloth_mask
        water_local = self.water_mask
        mass_quantity[cloth_local, 0] = self.mass[cloth_local]
        mass_quantity[water_local, 1] = self.mass[water_local]

        # external force: gravity only
        external_force = self.mass[:, None] * self.gravity[None, :]

        # interaction force: sum of all contact forces on each material.
        # Recompute a lightweight contact snapshot for diagnostics.
        interaction = np.zeros((2, 3), dtype=np.float64)
        # cloth-water contact contribution
        water_local_bool = self.water_mask
        if water_local_bool.any() and self.n_cloth > 0:
            fc, _ = self._cloth_face_data()
            wpos = self.position[water_local_bool]
            tree = cKDTree(fc)
            dists, fi = tree.query(wpos, k=1)
            contact_dir = wpos - fc[fi]
            r = np.linalg.norm(contact_dir, axis=1)
            dir_ = contact_dir / (r[:, None] + 1e-12)
            pen = self.contact_h - r
            active = pen > 0.0
            if active.any():
                f_mag = self.contact_k * pen[active]
                force_on_water = f_mag[:, None] * dir_[active]
                # net on water body (1) from cloth
                interaction[1] += force_on_water.sum(axis=0)
                # equal-and-opposite on cloth body (0)
                interaction[0] -= force_on_water.sum(axis=0)

        # tank boundary contact: sum on both bodies
        for b in (0, 1):
            idx = np.where(self.body_id == b)[0]
            if idx.size == 0:
                continue
            pos_b = self.position[idx]
            vel_b = self.velocity[idx]
            for n_vec, p0, margin in self.walls:
                sd = np.dot(pos_b - p0[None, :], n_vec)
                near = sd < margin
                if np.any(near):
                    vn = np.sum(vel_b[near] * n_vec[None, :], axis=1)
                    pen = margin - sd[near]
                    mag = self.boundary_k * pen + self.boundary_c * (-vn)
                    interaction[b] += (mag[:, None] * n_vec[None, :]).sum(axis=0)

        # contact signed distance: collect active water-cloth and boundary samples
        contact_sd_parts = [self._contact_sd]
        for n_vec, p0, margin in self.walls:
            sd = np.dot(self.position - p0[None, :], n_vec)
            near = sd < margin
            if np.any(near):
                contact_sd_parts.append(sd[near])
        contact_signed_distance = np.concatenate(contact_sd_parts) if contact_sd_parts else np.array([], dtype=np.float64)

        attributes = {
            "pinned": self.pinned,
            "material_phase": self.material_phase,
            "mass_quantity": mass_quantity,
        }
        fields = {
            "contact_signed_distance": contact_signed_distance,
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "external_force": external_force,
            "interaction_force": interaction,
        }
        scalars = {"contact_length_scale": self.contact_length_scale}

        return StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # -------------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            raise ValueError("advance_to time must be non-decreasing")
        if time == self.current_time:
            return self._frame(time)

        # guard against pathological long gaps
        n_steps = int(round((time - self.current_time) / self.dt_sub))
        n_steps = min(n_steps, self.max_substeps)
        # account for any rounding residual with a final partial sub-step
        dt_full = time - self.current_time
        for _ in range(n_steps):
            self._step()
        # account for any rounding residual with a final partial sub-step
        remaining = time - self.current_time
        if remaining > 1e-12:
            # use a temporary smaller dt for the partial step without changing self.dt_sub
            original_dt = self.dt_sub
            self.dt_sub = remaining
            self.current_time -= original_dt  # _step will add self.dt_sub
            self._step()
            self.dt_sub = original_dt

        return self._frame(time)

    # -------------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "dt_sub": self.dt_sub,
            "n_particles": int(self.position.shape[0]),
            "n_cloth": self.n_cloth,
            "inflow_cum_mass": float(self.cum_mass_source[1]),
        }
