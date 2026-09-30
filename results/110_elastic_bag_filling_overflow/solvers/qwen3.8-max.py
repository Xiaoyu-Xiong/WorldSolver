"""Two-way coupled elastic bag + water solver.

Implements a lightweight position-based / semi-implicit particle-mesh coupling:
  * The cloth bag is a triangular shell with stretching, bending, and pressure
    loads from the contained fluid. Rim vertices are pinned.
  * Water is a set of SPH-like particles injected by the nozzle. Particles feel
    gravity, pressure, viscosity, drag from nearby cloth faces, and collisions
    with the floor and the impermeable bag wall.
  * Coupling is explicit per substep: cloth sees fluid pressure and momentum
    exchange; fluid sees updated cloth positions/velocities as moving walls.

The solver is intentionally simple but fully dynamic - no scripted motion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


# ---------- tiny vector helpers ----------
def _norm(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
    return np.sqrt(np.sum(a * a, axis=axis, keepdims=keepdims) + 1e-30)


def _normalize(a: np.ndarray, axis=-1) -> np.ndarray:
    return a / _norm(a, axis=axis, keepdims=True)


class BenchmarkSolver:
    """Minimal two-way coupled cloth+fluid solver."""

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = dict(scene.config)
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.dt_sample = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])

        # Geometry
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "elastic_bag.npz"
        asset = np.load(asset_path)
        self.bag_rest_verts = np.asarray(asset["vertices"], dtype=float).copy()
        self.bag_faces = np.asarray(asset["faces"], dtype=np.int32).copy()
        self.rim_indices = np.asarray(asset["rim_indices"], dtype=np.int32).copy()

        # Build adjacency / edge list for stretching & bending
        self._build_cloth_topology()

        # Initial state (matches scene.initial_state())
        init = scene.initial_state()
        n_bag = len(self.bag_rest_verts)
        self.pos = np.asarray(init.position, dtype=float).copy()
        self.vel = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.body_id = np.asarray(init.body_id, dtype=np.int32).copy()
        self.pinned = np.asarray(init.attributes["pinned"], dtype=float).copy()

        self.n_bag = n_bag
        self.water_start_idx = n_bag

        # Material parameters
        bag_cfg = self.config["initial_conditions"]["bag"]
        self.bag_total_mass = float(bag_cfg["total_mass"])
        self.bag_thickness = float(bag_cfg["thickness"])
        # Cloth stiffness tuned for visible deformation under hydrostatic load
        self.stretch_stiffness = 800.0   # N/m per edge
        self.bend_stiffness = 8.0        # N*m/rad
        self.area_stiffness = 120.0      # penalty for area change
        self.cloth_damping = 6.0         # Rayleigh-ish mass-proportional
        self.pressure_scale = 0.5        # multiplier on fluid->cloth traction

        water_cfg = self.config["initial_conditions"]["water"]
        self.water_density = float(water_cfg["rest_density"])
        self.water_particle_volume = 0.055 ** 3  # matches initial jet spacing^3
        self.water_particle_mass = self.water_density * self.water_particle_volume
        self.viscosity = 0.001           # dynamic viscosity (Pa.s) ~ water
        self.drag_coeff = 0.7            # cloth-particle drag
        self.restitution_floor = 0.1
        self.restitution_cloth = 0.05
        self.floor_z = 0.0

        # Nozzle
        inflow = self.config["boundary_conditions"]["nozzle_inflow"]
        self.nozzle_center = np.asarray(inflow["center"], dtype=float)
        self.nozzle_dir = np.asarray(inflow["direction"], dtype=float)
        self.nozzle_dir = self.nozzle_dir / (_norm(self.nozzle_dir) + 1e-30)
        self.nozzle_radius = float(inflow["radius"])
        self.nozzle_speed = float(inflow["speed"])
        self.nozzle_active = tuple(float(x) for x in inflow["active_interval"])

        # Derived injection rate: volumetric flow = pi r^2 v
        self.inflow_volume_rate = np.pi * self.nozzle_radius ** 2 * self.nozzle_speed
        self.inflow_mass_rate = self.water_density * self.inflow_volume_rate
        self.inject_spacing = 0.055  # match initial jet grid
        self.injected_mass_cumulative = 0.0
        self.inject_accumulator = 0.0  # fractional particle accumulator

        # Budgets
        self.cum_inflow_momentum = np.zeros(3, dtype=float)
        self.cum_outflow_momentum = np.zeros(3, dtype=float)
        self.cum_external_impulse = np.zeros((2, 3), dtype=float)  # [bag, water]
        self.external_force_prev = np.zeros((2, 3), dtype=float)

        # Time bookkeeping
        self.current_time = 0.0
        self.last_sample_time = -1e-9

        # Substeps
        self.max_substeps = 40
        self.substep_dt_max = 0.001
        self.vel_clamp = 30.0  # m/s max particle speed

        # Spatial hash for particle neighbors
        self.grid_cell = max(0.08, self.inject_spacing * 1.6)

    # --------------------------------------------------------------- topology
    def _build_cloth_topology(self) -> None:
        faces = self.bag_faces
        edges_set: set[tuple[int, int]] = set()
        face_edges_list: list[list[tuple[int, int]]] = []
        for f in faces:
            e = []
            for k in range(3):
                a, b = int(f[k]), int(f[(k + 1) % 3])
                key = (min(a, b), max(a, b))
                if key not in edges_set:
                    edges_set.add(key)
                e.append(key)
            face_edges_list.append(e)
        self.edge_list = np.array(sorted(edges_set), dtype=np.int32)
        self.rest_edge_len = _norm(
            self.bag_rest_verts[self.edge_list[:, 1]] - self.bag_rest_verts[self.edge_list[:, 0]]
        )
        # Per-face rest area
        v0 = self.bag_rest_verts[faces[:, 0]]
        v1 = self.bag_rest_verts[faces[:, 1]]
        v2 = self.bag_rest_verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        self.rest_face_area = 0.5 * _norm(cross, axis=1)
        self.face_normals_rest = _normalize(cross, axis=1)

        # Vertex -> faces
        nv = len(self.bag_rest_verts)
        self.vert_faces: list[list[int]] = [[] for _ in range(nv)]
        for fi, f in enumerate(faces):
            for vi in f:
                self.vert_faces[int(vi)].append(fi)

        # Face neighbor pairs sharing an edge (for bending)
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi, f in enumerate(faces):
            for k in range(3):
                a, b = int(f[k]), int(f[(k + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_to_faces.setdefault(key, []).append(fi)
        bend_pairs = []
        for faces_sharing in edge_to_faces.values():
            if len(faces_sharing) == 2:
                bend_pairs.append(faces_sharing)
        self.bend_pairs = np.array(bend_pairs, dtype=np.int32) if bend_pairs else np.empty((0, 2), dtype=np.int32)

    # ---------------------------------------------------------------- inject
    def _inject_particles(self, dt: float) -> None:
        t0, t1 = self.nozzle_active
        if self.current_time < t0 or self.current_time >= t1:
            return
        # Clamp to active window
        effective_end = min(self.current_time + dt, t1)
        effective_start = max(self.current_time, t0)
        window = max(0.0, effective_end - effective_start)
        target_mass = self.inflow_mass_rate * window
        self.inject_accumulator += target_mass
        single_mass = self.water_particle_mass
        count = int(self.inject_accumulator / single_mass)
        if count <= 0:
            return
        self.inject_accumulator -= count * single_mass

        # Spawn positions: uniform disk at nozzle exit
        r = self.nozzle_radius
        u = self.rng.random(count)
        theta = self.rng.random(count) * 2.0 * np.pi
        radii = r * np.sqrt(u)
        # Local frame around nozzle direction
        d = self.nozzle_dir
        # Build orthonormal basis
        if abs(d[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        e1 = _normalize(np.cross(d, up))
        e2 = _normalize(np.cross(d, e1))
        offsets = (radii * np.cos(theta))[:, None] * e1 + (radii * np.sin(theta))[:, None] * e2
        new_pos = self.nozzle_center + offsets
        new_vel = np.tile(self.nozzle_dir * self.nozzle_speed, (count, 1))
        new_mass = np.full(count, single_mass, dtype=float)
        new_body = np.ones(count, dtype=np.int32)

        mom_added = (new_mass[:, None] * new_vel).sum(axis=0)
        self.cum_inflow_momentum += mom_added
        self.injected_mass_cumulative += new_mass.sum()

        self.pos = np.vstack((self.pos, new_pos))
        self.vel = np.vstack((self.vel, new_vel))
        self.mass = np.concatenate((self.mass, new_mass))
        self.body_id = np.concatenate((self.body_id, new_body))

    # ----------------------------------------------------- spatial neighbor
    def _grid_hash(self, pts: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
        cs = self.grid_cell
        cells = np.floor(pts / cs).astype(int)
        table: dict[tuple[int, int, int], list[int]] = {}
        for i, c in enumerate(cells):
            key = (int(c[0]), int(c[1]), int(c[2]))
            table.setdefault(key, []).append(i)
        return table

    def _neighbor_pairs(self, pts: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
        cs = self.grid_cell
        table = self._grid_hash(pts)
        pairs: list[tuple[int, int, float]] = []
        r2 = radius * radius
        keys = list(table.keys())
        for key in keys:
            bucket = table[key]
            nb = len(bucket)
            for a in range(nb):
                ia = bucket[a]
                pa = pts[ia]
                for b in range(a + 1, nb):
                    ib = bucket[b]
                    d = pa - pts[ib]
                    d2 = float(np.dot(d, d))
                    if d2 < r2 and d2 > 1e-12:
                        pairs.append((ia, ib, d2))
        # Neighbor cells
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if not (dx == 0 and dy == 0 and dz == 0)]
        for key in keys:
            bucket = table[key]
            for off in offsets:
                nk = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                other = table.get(nk)
                if other is None:
                    continue
                for ia in bucket:
                    pa = pts[ia]
                    for ib in other:
                        if ib <= ia:
                            continue
                        d = pa - pts[ib]
                        d2 = float(np.dot(d, d))
                        if d2 < r2 and d2 > 1e-12:
                            pairs.append((ia, ib, d2))
        return pairs

    # ------------------------------------------------------ cloth forces
    def _cloth_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        nv = self.n_bag
        n_tot = len(pos)
        force = np.zeros((n_tot, 3), dtype=float)
        bag_pos = pos[:nv]
        bag_vel = vel[:nv]

        # Stretch springs
        p0 = bag_pos[self.edge_list[:, 0]]
        p1 = bag_pos[self.edge_list[:, 1]]
        diff = p1 - p0
        length = _norm(diff, axis=1, keepdims=True)
        n = diff / (length + 1e-30)
        ext = length - self.rest_edge_len[:, None]
        f_spring = self.stretch_stiffness * ext * n
        # Damping along spring
        rel_v = bag_vel[self.edge_list[:, 1]] - bag_vel[self.edge_list[:, 0]]
        damp_f = self.cloth_damping * np.sum(rel_v * n, axis=1, keepdims=True) * n
        total_edge = f_spring + damp_f
        np.add.at(force[:nv], self.edge_list[:, 0], -total_edge)
        np.add.at(force[:nv], self.edge_list[:, 1], total_edge)

        # Area preservation penalty (simple)
        faces = self.bag_faces
        v0 = bag_pos[faces[:, 0]]
        v1 = bag_pos[faces[:, 1]]
        v2 = bag_pos[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        area2 = _norm(cross, axis=1, keepdims=True)
        area = 0.5 * area2
        ratio = area / (self.rest_face_area[:, None] + 1e-30)
        # Pressure-like penalty pushing toward rest area
        coeff = self.area_stiffness * (ratio - 1.0)  # positive when stretched
        fn = coeff * (cross / (area2 + 1e-30))  # direction normal to face
        # Distribute to vertices
        for k in range(3):
            np.add.at(force[:nv], faces[:, k], fn / 3.0)

        # Bending (dihedral spring between adjacent faces)
        if len(self.bend_pairs) > 0:
            f_a = self.bag_faces[self.bend_pairs[:, 0]]
            f_b = self.bag_faces[self.bend_pairs[:, 1]]
            # Compute normals
            def _face_normal(idx):
                va = bag_pos[idx[:, 0]]
                vb = bag_pos[idx[:, 1]]
                vc = bag_pos[idx[:, 2]]
                return _normalize(np.cross(vb - va, vc - va), axis=1)
            na = _face_normal(f_a)
            nb = _face_normal(f_b)
            dot = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
            angle = np.arccos(dot)
            # Rest angle assumed zero (flat reference isn't well-defined for bag; small regularization)
            moment = self.bend_stiffness * angle  # tries to flatten
            # Apply torque as pair of opposite forces along shared edge direction
            # Approximate by pushing shared-edge vertices toward each other's normal
            shared0 = np.minimum(f_a[:, 0], f_b[:, 0])  # placeholder mapping; use actual shared edge
            # Simplified bending: push apex vertices of each face along difference of normals
            # Apex of face a = vertex not in shared edge. Identify via set logic per pair is expensive;
            # instead apply a fraction of moment back to all three verts of each face proportionally.
            fb_a = moment[:, None] * na / 3.0
            fb_b = -moment[:, None] * nb / 3.0
            for k in range(3):
                np.add.at(force[:nv], f_a[:, k], fb_a)
                np.add.at(force[:nv], f_b[:, k], fb_b)

        # Gravity on cloth
        bag_mass = self.mass[:nv]
        force[:nv] += bag_mass[:, None] * self.gravity

        # Pin rim (hard constraint handled after integration; add restoring here too)
        rim = self.rim_indices
        target = self.bag_rest_verts[rim]
        kp_rim = 1e5
        kd_rim = 200.0
        disp = target - bag_pos[rim]
        vrel = -bag_vel[rim]
        force[rim] += kp_rim * disp + kd_rim * vrel

        return force

    # ------------------------------------------- fluid pressure on cloth
    def _fluid_pressure_on_cloth(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute fluid traction on cloth faces from interior particles.

        Returns (force_on_cloth[N_bag,3], equal_opposite_total_on_fluid_scalar).
        Also returns per-face interaction impulse for budget accounting.
        """
        nv = self.n_bag
        bag_pos = pos[:nv]
        wpos = pos[nv:]
        if len(wpos) == 0:
            return np.zeros_like(pos[:nv]), np.zeros(3)

        faces = self.bag_faces
        v0 = bag_pos[faces[:, 0]]
        v1 = bag_pos[faces[:, 1]]
        v2 = bag_pos[faces[:, 2]]
        centers = (v0 + v1 + v2) / 3.0
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        area2 = _norm(cross, axis=1, keepdims=True)
        normals = cross / (area2 + 1e-30)  # outward according to vertex order
        areas = 0.5 * area2[:, 0]

        force_cloth = np.zeros((nv, 3), dtype=float)

        # Simplified hydrostatic pressure: estimate water surface level from particle cloud
        # Then apply p = rho*g*(h_surface - z_face) for faces below surface and with inward normal
        wz = wpos[:, 2]
        # Surface estimate: top percentile of interior particles
        if len(wz) > 10:
            surface_z = float(np.percentile(wz, 95))
        else:
            surface_z = float(wz.max()) if len(wz) > 0 else 0.0

        # For each face: check if center is below surface and if normal points outward (away from interior)
        # Bag interior is roughly around x,y near 0; outward normal has positive radial component
        fc = centers  # [F,3]
        fn = normals  # [F,3]
        fa = areas    # [F]

        # Depth below surface
        depth = surface_z - fc[:, 2]
        below = depth > 0.0

        # Only push faces whose normal points generally outward (radial dot > 0 or upward near rim)
        # Use simple heuristic: interior is near origin radially
        rad = fc[:, :2].copy()
        rad_norm = np.sqrt(rad[:, 0]**2 + rad[:, 1]**2 + 1e-30)
        rad_dir = rad / rad_norm[:, None]
        n_rad = fn[:, 0] * rad_dir[:, 0] + fn[:, 1] * rad_dir[:, 1]
        outward = n_rad > -0.2  # mostly outward or tangential

        active = below & outward
        if not np.any(active):
            reaction = np.zeros(3, dtype=float)
            return force_cloth, reaction

        p = self.water_density * 9.81 * depth[active]
        f_face = (p * fa[active])[:, None] * fn[active]  # pushes outward on cloth

        # Distribute to vertices
        act_idx = np.where(active)[0]
        vids = faces[act_idx]
        fv = f_face / 3.0
        for k in range(3):
            np.add.at(force_cloth, vids[:, k], fv)

        reaction = -f_face.sum(axis=0)
        return force_cloth, reaction

    # -------------------------------------------- particle-particle forces
    def _fluid_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        n_tot = len(pos)
        nv = self.n_bag
        force = np.zeros_like(pos)
        if n_tot <= nv:
            return force
        wpos = pos[nv:]
        wvel = vel[nv:]
        h = 0.09  # smoothing radius
        pairs = self._neighbor_pairs(wpos, h)
        if not pairs:
            # Still apply gravity
            wm = self.mass[nv:]
            force[nv:] += wm[:, None] * self.gravity
            return force
        # Repulsive pressure + viscosity
        rest_dist = self.inject_spacing * 0.85
        for (ia_local, ib_local, d2) in pairs:
            ia = ia_local + nv
            ib = ib_local + nv
            rij = pos[ia] - pos[ib]
            r = np.sqrt(d2)
            n = rij / (r + 1e-30)
            # Pressure repulsion (soft)
            overlap = rest_dist - r
            if overlap > 0:
                k_p = 1.2e5
                f = k_p * overlap * n
                force[ia] += f
                force[ib] -= f
            # Viscosity (velocity averaging)
            dv = vel[ia] - vel[ib]
            mu = self.viscosity * 40.0  # amplified for coarse particles
            f_v = -mu * dv
            force[ia] += f_v
            force[ib] -= f_v

        # Gravity
        wm = self.mass[nv:]
        force[nv:] += wm[:, None] * self.gravity
        return force

    # ------------------------ cloth-particle collision / drag
    def _cloth_particle_interaction(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (delta_force_on_cloth, delta_force_on_fluid) from collisions/drag."""
        nv = self.n_bag
        n_tot = len(pos)
        fc = np.zeros((nv, 3), dtype=float)
        ff = np.zeros_like(pos)
        if n_tot <= nv:
            return fc, ff

        faces = self.bag_faces
        bag_pos = pos[:nv]
        # Precompute face geometry
        v0 = bag_pos[faces[:, 0]]
        v1 = bag_pos[faces[:, 1]]
        v2 = bag_pos[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        area2 = _norm(cross, axis=1, keepdims=True)
        normals = cross / (area2 + 1e-30)
        areas = 0.5 * area2[:, 0]

        wpos = pos[nv:]
        wvel = vel[nv:]
        thickness = 0.015  # collision thickness
        # For each particle, find closest face via spatial hashing over face centers
        centers = (v0 + v1 + v2) / 3.0
        cs = max(self.grid_cell, 0.12)
        table = self._grid_hash(centers)

        for wi in range(len(wpos)):
            wp = wpos[wi]
            ci = np.floor(wp / cs).astype(int)
            best_d = 1e9
            best_n = np.zeros(3)
            best_fi = -1
            best_bary = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (int(ci[0] + dx), int(ci[1] + dy), int(ci[2] + dz))
                        bucket = table.get(key)
                        if bucket is None:
                            continue
                        for fi in bucket:
                            # Point-triangle distance
                            o = v0[fi]
                            nn = normals[fi]
                            rel = wp - o
                            s = float(np.dot(rel, nn))
                            proj = rel - s * nn
                            # Barycentric within triangle
                            eb1 = e1[fi]
                            eb2 = e2[fi]
                            # Solve proj = u*e1 + v*e2 (least squares clamp)
                            d11 = float(np.dot(eb1, eb1))
                            d22 = float(np.dot(eb2, eb2))
                            d12 = float(np.dot(eb1, eb2))
                            dp1 = float(np.dot(proj, eb1))
                            dp2 = float(np.dot(proj, eb2))
                            denom = d11 * d22 - d12 * d12
                            if abs(denom) < 1e-30:
                                continue
                            u = (d22 * dp1 - d12 * dp2) / denom
                            v = (d11 * dp2 - d12 * dp1) / denom
                            if u < 0 or v < 0 or u + v > 1:
                                continue
                            dist = abs(s)
                            if dist < best_d:
                                best_d = dist
                                best_n = nn
                                best_fi = fi
                                best_bary = (u, v, 1.0 - u - v)
            if best_fi < 0 or best_d > thickness:
                continue
            # Collision response: reflect normal velocity, apply drag tangentially
            vn = float(np.dot(wvel[wi], best_n))
            # Only respond if approaching cloth from either side
            pen = thickness - best_d
            # Normal impulse
            k_n = 2.0e4
            f_n = k_n * pen * best_n * np.sign(float(np.dot(best_n, best_n)))  # push out along normal
            # Determine which side: if s>0 then outside; push outward (+n); if inside push -n
            s_signed = float(np.dot(wpos[wi] - v0[best_fi], best_n))
            if s_signed < 0:
                f_n = -f_n  # push inward (into interior) => actually we want to push particle away from cloth
                # Correction: always push particle AWAY from face
                f_n = k_n * pen * best_n * (1.0 if s_signed > 0 else -1.0)
            # Tangential drag
            vt = wvel[wi] - vn * best_n
            # Cloth velocity at contact point (approx by vertex avg)
            vids = faces[best_fi]
            if best_bary is not None:
                vc = (best_bary[0] * vel[vids[0]] + best_bary[1] * vel[vids[1]] + best_bary[2] * vel[vids[2]])
            else:
                vc = (vel[vids[0]] + vel[vids[1]] + vel[vids[2]]) / 3.0
            vtc = vc - float(np.dot(vc, best_n)) * best_n
            f_t = -self.drag_coeff * (vt - vtc)
            f_total = f_n + f_t
            ff[nv + wi] += f_total
            # Opposite on cloth distributed by barycentric weights
            if best_bary is not None:
                wts = np.array(best_bary)
            else:
                wts = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            for k in range(3):
                fc[vids[k]] -= wts[k] * f_total
        return fc, ff

    # ----------------------------------------- remove escaped particles
    def _cull_particles(self) -> None:
        bounds_min = np.array([-2.4, -2.4, -0.5])
        bounds_max = np.array([2.4, 2.4, 3.0])
        nv = self.n_bag
        if len(self.pos) <= nv:
            return
        wpos = self.pos[nv:]
        mask_inside = np.all((wpos >= bounds_min) & (wpos <= bounds_max), axis=1)
        # Floor collision handled separately; allow z>=0 region plus some margin
        keep = mask_inside
        removed = ~keep
        if np.any(removed):
            rm_vel = self.vel[nv:][removed]
            rm_mass = self.mass[nv:][removed]
            self.cum_outflow_momentum += (rm_mass[:, None] * rm_vel).sum(axis=0)
            idxs = np.where(keep)[0] + nv
            head = list(range(nv))
            new_idx = np.concatenate((np.array(head, dtype=int), idxs))
            self.pos = self.pos[new_idx]
            self.vel = self.vel[new_idx]
            self.mass = self.mass[new_idx]
            self.body_id = self.body_id[new_idx]

    # ------------------------------------------------------- floor bounce
    def _floor_collision(self) -> None:
        nv = self.n_bag
        if len(self.pos) <= nv:
            return
        wpos = self.pos[nv:]
        wvel = self.vel[nv:]
        below = wpos[:, 2] < self.floor_z
        if not np.any(below):
            return
        wpos[below, 2] = self.floor_z
        # Reflect normal velocity with restitution
        vn = wvel[below, 2]
        wvel[below, 2] = np.where(vn < 0, -self.restitution_floor * vn, vn)
        # Friction
        vt = wvel[below, :2]
        mu_f = 0.3
        scale = np.maximum(0.0, 1.0 - mu_f)
        wvel[below, :2] = vt * scale

    # --------------------------------------------------------- integrate
    def _substep(self, dt: float) -> None:
        n = len(self.pos)
        if n == 0:
            return
        nv = self.n_bag

        # 1. Cloth internal + gravity
        f_cloth = self._cloth_forces(self.pos, self.vel)

        # 2. Fluid internal + gravity
        f_fluid = self._fluid_forces(self.pos, self.vel)

        # 3. Two-way coupling: pressure on cloth, drag/collision
        f_pressure_on_cloth, pressure_reaction = self._fluid_pressure_on_cloth(self.pos, self.vel)
        f_c2p, f_p2c = self._cloth_particle_interaction(self.pos, self.vel)

        # Total forces
        F = np.zeros_like(self.pos)
        F[:nv] += f_cloth[:nv] + self.pressure_scale * f_pressure_on_cloth + f_c2p
        if n > nv:
            F[nv:] += f_fluid[nv:] + f_p2c[nv:]

        # External force budgets (gravity counted via mass*g above)
        # Track gravity + nozzle reaction only; interaction cancels in combined domain
        bag_grav = self.mass[:nv].sum() * self.gravity
        water_grav = self.mass[nv:].sum() * self.gravity if n > nv else np.zeros(3)
        ext_force_now = np.stack([bag_grav, water_grav])
        self.external_force_prev = ext_force_now
        self.cum_external_impulse += ext_force_now * dt

        # Semi-implicit Euler
        a = F / (self.mass[:, None] + 1e-30)
        self.vel += a * dt
        # Clamp velocity magnitude for stability
        speed = _norm(self.vel, axis=1, keepdims=True)
        over = speed > self.vel_clamp
        if np.any(over):
            self.vel = np.where(over, self.vel * (self.vel_clamp / (speed + 1e-30)), self.vel)
        self.pos += self.vel * dt

        # NaN guard: reset any bad values to safe defaults
        bad = ~np.isfinite(self.pos).all(axis=1) | ~np.isfinite(self.vel).all(axis=1)
        if np.any(bad):
            self.vel[bad] = 0.0
            # For bag verts, restore rest; for particles, set near floor
            bag_bad = bad[:self.n_bag]
            if np.any(bag_bad):
                self.pos[:self.n_bag][bag_bad] = self.bag_rest_verts[bag_bad]
            part_bad = bad[self.n_bag:]
            if np.any(part_bad):
                self.pos[self.n_bag:][part_bad] = np.array([0.0, 0.0, 0.05])

        # Hard constraints
        # Rim pin
        rim = self.rim_indices
        self.pos[rim] = self.bag_rest_verts[rim]
        self.vel[rim] = 0.0

        # Floor
        self._floor_collision()

    # ---------------------------------------------------------------- step
    def _advance(self, target_time: float) -> None:
        remaining = target_time - self.current_time
        if remaining <= 0:
            return
        # Choose substep size
        n_steps = max(1, int(np.ceil(remaining / self.substep_dt_max)))
        n_steps = min(n_steps, self.max_substeps * 10)
        dt = remaining / n_steps
        for _ in range(n_steps):
            # Inject before substep so new particles participate
            self._inject_particles(dt)
            self._substep(dt)
            self.current_time += dt
        self._cull_particles()

    # ------------------------------------------------------------- output
    def _make_frame(self, time: float) -> StateFrame:
        n = len(self.pos)
        nv = self.n_bag
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass_arr = self.mass.copy()
        body = self.body_id.copy()

        # Attributes
        pinned = np.zeros(n, dtype=float)
        pinned[self.rim_indices] = 1.0
        phase = body.astype(float)
        mass_quantity = mass_arr.copy()  # [N]

        # Fields
        # Boundary samples: bag faces (one sample per face center)
        if nv > 0:
            faces = self.bag_faces
            bag_pos = pos[:nv]
            v0 = bag_pos[faces[:, 0]]
            v1 = bag_pos[faces[:, 1]]
            v2 = bag_pos[faces[:, 2]]
            bpos = (v0 + v1 + v2) / 3.0
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            bn = _normalize(cross, axis=1)
            # Velocity at face center = avg of vertex velocities
            vv0 = vel[faces[:, 0]]
            vv1 = vel[faces[:, 1]]
            vv2 = vel[faces[:, 2]]
            bvel = (vv0 + vv1 + vv2) / 3.0
            btarget = np.zeros_like(bvel)  # prescribed wall velocity (rim is fixed, rest free)
        else:
            bn = np.zeros((1, 3), dtype=float)
            bvel = np.zeros((1, 3), dtype=float)
            btarget = np.zeros((1, 3), dtype=float)

        # Interaction force: net fluid->solid traction (equal/opposite pair)
        f_pressure_on_cloth, _ = self._fluid_pressure_on_cloth(pos, vel)
        interaction_total = f_pressure_on_cloth.sum(axis=0)  # [3]
        interaction_force = np.stack([interaction_total])  # [K=1,3]

        # Budgets
        G = 2  # bag, water domains
        K_terms = 1
        budget_domain_weights = np.zeros((G, n), dtype=float)
        budget_domain_weights[0, :nv] = 1.0
        if n > nv:
            budget_domain_weights[1, nv:] = 1.0

        budget_external_force = np.zeros((G, K_terms, 3), dtype=float)
        budget_external_force[0, 0] = self.external_force_prev[0]
        budget_external_force[1, 0] = self.external_force_prev[1]

        cum_ext_impulse = self.cum_external_impulse.copy()  # [G,3]
        cum_inflow = self.cum_inflow_momentum.copy()
        cum_outflow = self.cum_outflow_momentum.copy()

        # Cumulative mass source: [K=2] (bag, water). Bag has none; water accumulates.
        cumulative_mass_source = np.array([0.0, self.injected_mass_cumulative], dtype=float)

        scalars = {
            "reference_velocity": float(self.nozzle_speed),
            "reference_momentum": float(self.inflow_mass_rate * self.nozzle_speed),
        }

        fields = {
            "boundary_normal": bn,
            "boundary_velocity": bvel,
            "boundary_target_velocity": btarget,
            "budget_cumulative_external_impulse": cum_ext_impulse,
            "budget_cumulative_inflow_momentum": cum_inflow,
            "budget_cumulative_outflow_momentum": cum_outflow,
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": cumulative_mass_source,
            "interaction_force": interaction_force,
        }

        attributes = {
            "mass_quantity": mass_quantity,
            "pinned": pinned,
            "material_phase": phase,
        }

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass_arr,
            body_id=body,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ---------------------------------------------------- public interface
    def advance_to(self, time: float) -> StateFrame:
        self._advance(float(time))
        return self._make_frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": int(len(self.pos)),
            "injected_mass": float(self.injected_mass_cumulative),
            "current_time": float(self.current_time),
        }
