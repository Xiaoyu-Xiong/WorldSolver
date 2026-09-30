"""Self-contained two-way cloth/fluid solver for the elastic-bag overflow scene.

The implementation uses a particle representation for both the elastic bag
(mass-point triangle network with PBD-style distance constraints) and the water
(weakly-compressible SPH).  Coupling is realised through position corrections
between water particles and the cloth surface; no keyframes or precomputed
motions are used.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic explicit SPH + PBD-cloth solver."""

    # --- numerical settings -------------------------------------------------
    SUBSTEPS: int = 25                 # integration substeps per sample_dt
    SPH_SUPPORT: float = 0.16          # cubic-spline support radius
    SPH_K: float = 300.0               # pressure stiffness
    SPH_RHO0: float = 1000.0           # fluid rest density
    SPH_ALPHA: float = 0.05           # artificial viscosity coefficient
    CLOTH_STRETCH_STIFFNESS: float = 0.45
    CLOTH_ITERATIONS: int = 5
    COLLISION_MARGIN: float = 0.025
    COLLISION_LAMBDA: float = 0.35
    COLLISION_STIFFNESS: float = 5.0   # not used directly; position corrections dominate
    FLOOR_MARGIN: float = 0.01
    GRAVITY: NDArray[np.floating] = np.array([0.0, 0.0, -9.81], dtype=float)

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        frame0 = scene.initial_state()
        self._validate_initial(frame0)

        self.t: float = 0.0
        self.x: NDArray[np.floating] = np.asarray(frame0.position, dtype=float).copy()
        self.v: NDArray[np.floating] = np.asarray(frame0.velocity, dtype=float).copy()
        self.m: NDArray[np.floating] = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id: NDArray[np.integer] = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.pinned: NDArray[np.floating] = np.asarray(
            frame0.attributes["pinned"], dtype=float
        ).copy()

        self.bag_count: int = int(np.sum(self.body_id == 0))
        self.bag_faces: NDArray[np.integer] = np.asarray(
            scene.metadata()["topology"]["bag_faces"], dtype=np.int32
        )
        self.rim_indices: NDArray[np.integer] = np.asarray(
            scene.metadata()["fixed_vertex_sets"]["rim"], dtype=np.int32
        )
        self.x_ref: NDArray[np.floating] = self.x[: self.bag_count].copy()

        # cloth edge network
        edges = []
        for f in self.bag_faces:
            edges.append((f[0], f[1]))
            edges.append((f[1], f[2]))
            edges.append((f[2], f[0]))
        edges = np.asarray(sorted({tuple(sorted(e)) for e in edges}), dtype=np.int32)
        self.cloth_edges = edges
        self.cloth_l0 = np.linalg.norm(
            self.x_ref[edges[:, 1]] - self.x_ref[edges[:, 0]], axis=1
        )

        # face outward normals for collision
        self._face_normals()

        # inflow parameters
        inflow = scene.metadata()["inflow"]
        self.inflow_center = np.asarray(inflow["center"], dtype=float)
        self.inflow_dir = np.asarray(inflow["direction"], dtype=float)
        self.inflow_radius = float(inflow["radius"])
        self.inflow_speed = float(inflow["speed"])
        self.inflow_active = [float(x) for x in inflow["active_interval"]]
        self._inflow_frac = 0.0
        self._water_particle_mass = float(
            np.mean(self.m[self.body_id == 1]) if np.any(self.body_id == 1) else 0.166
        )

        # static boundary samples (floor + nozzle) for required trajectory fields
        self._boundary_samples()

        # budget accumulators
        self._cum_external_impulse = np.zeros((2, 3), dtype=float)
        self._cum_inflow_momentum = np.zeros((2, 3), dtype=float)
        self._cum_outflow_momentum = np.zeros((2, 3), dtype=float)
        self._cum_mass_source = 0.0
        self._interaction_impulse_interval = np.zeros(3, dtype=float)
        self._dt_accumulated = 0.0

        # reference scalars
        full_water_mass = (
            self.SPH_RHO0 * self.inflow_speed * np.pi * self.inflow_radius**2
        ) * (self.inflow_active[1] - self.inflow_active[0])
        self.reference_velocity = float(self.inflow_speed)
        self.reference_momentum = float(
            max(full_water_mass * self.inflow_speed, 100.0)
        )

        # scalar carry-over from the scene
        self._scalars = dict(frame0.scalars)
        self._scalars["reference_velocity"] = self.reference_velocity
        self._scalars["reference_momentum"] = self.reference_momentum

        self.sample_dt = float(scene.config["sample_dt"])
        self.dt = self.sample_dt / self.SUBSTEPS
        self.duration = float(scene.config["duration"])

    # ----------------------------------------------------------------------
    def _validate_initial(self, frame0: StateFrame) -> None:
        frame0.validate()
        if frame0.body_id is None or frame0.mass is None:
            raise ValueError("Initial state must include body_id and mass")

    def _face_normals(self) -> None:
        a = self.x_ref[self.bag_faces[:, 0]]
        b = self.x_ref[self.bag_faces[:, 1]]
        c = self.x_ref[self.bag_faces[:, 2]]
        n = np.cross(b - a, c - a)
        area = np.linalg.norm(n, axis=1)
        area[area == 0] = 1e-12
        n = n / area[:, None]
        # point outward: away from bag centre
        centre = self.x_ref.mean(axis=0)
        fc = (a + b + c) / 3.0
        outward = fc - centre
        outward /= np.linalg.norm(outward, axis=1)[:, None] + 1e-12
        flip = np.einsum("ij,ij->i", n, outward) < 0
        n[flip] = -n[flip]
        self.face_normal = n
        self.face_area = area * 0.5

    def _boundary_samples(self) -> None:
        # floor grid
        grid = np.linspace(-2.4, 2.4, 13)
        xx, yy = np.meshgrid(grid, grid)
        floor_pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        floor_n = np.tile(np.array([0.0, 0.0, 1.0]), (floor_pos.shape[0], 1))

        # nozzle cylinder side (radius 0.19, height 0.72, centre z=2.78)
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        zlev = np.linspace(2.42, 3.14, 6)
        tt, zz = np.meshgrid(theta, zlev)
        nozzle_pos = np.column_stack(
            (
                0.19 * np.cos(tt.ravel()),
                0.19 * np.sin(tt.ravel()),
                zz.ravel(),
            )
        )
        nozzle_n = np.column_stack(
            (
                np.cos(tt.ravel()),
                np.sin(tt.ravel()),
                np.zeros(tt.size),
            )
        )

        pos = np.vstack((floor_pos, nozzle_pos))
        n = np.vstack((floor_n, nozzle_n))
        v = np.zeros_like(pos)
        self.boundary_pos = pos
        self.boundary_normal = n
        self.boundary_vel = v
        self.boundary_target = v.copy()

    # ----------------------------------------------------------------------
    def _kernel(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        # cubic-spline f(q); q = r / support
        w = np.zeros_like(q)
        m1 = q <= 1.0
        m2 = (q > 1.0) & (q <= 2.0)
        w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        w[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return w

    def _kernel_grad_factor(
        self, q: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        # (1/(pi support^5)) * f'(q)/q
        f = np.zeros_like(q)
        m1 = q <= 1.0
        m2 = (q > 1.0) & (q <= 2.0)
        f[m1] = -3.0 + 2.25 * q[m1]
        f[m2] = -0.75 * (2.0 - q[m2]) ** 2 / q[m2]
        return f / (np.pi * self.SPH_SUPPORT**5)

    def _sph_forces(self) -> NDArray[np.floating]:
        n = len(self.x)
        acc = np.zeros_like(self.v)
        fluid_mask = self.body_id == 1
        fluid_idx = np.flatnonzero(fluid_mask)
        nf = fluid_idx.size
        if nf == 0:
            return acc

        xf = self.x[fluid_idx]
        vf = self.v[fluid_idx]
        mf = self.m[fluid_idx]
        tree = cKDTree(xf)
        pairs = tree.sparse_distance_matrix(
            tree, max_distance=self.SPH_SUPPORT, output_type="ndarray"
        )
        if pairs.size == 0:
            return acc

        i = pairs["i"]
        j = pairs["j"]
        r = pairs["v"]
        dx = xf[i] - xf[j]
        q = r / self.SPH_SUPPORT

        # density
        w = self._kernel(q)
        W = w / (np.pi * self.SPH_SUPPORT**3)
        rho = mf * (1.0 / (np.pi * self.SPH_SUPPORT**3))
        np.add.at(rho, i, mf[j] * W)
        np.add.at(rho, j, mf[i] * W)
        rho = np.maximum(rho, 0.1 * self.SPH_RHO0)

        # pressure
        p = self.SPH_K * np.maximum(rho / self.SPH_RHO0 - 1.0, 0.0)

        # pressure force
        gf = self._kernel_grad_factor(q)
        grad = gf[:, None] * dx
        term = mf[i] * mf[j] * (
            p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2
        )
        fpair = -term[:, None] * grad

        fbuf = np.zeros((nf, 3), dtype=float)
        np.add.at(fbuf, i, fpair)
        np.add.at(fbuf, j, -fpair)

        # artificial viscosity
        v_ij = vf[j] - vf[i]
        vr = np.einsum("ij,ij->i", v_ij, dx)
        cs = np.sqrt(self.SPH_K / self.SPH_RHO0)
        h_sph = 0.5 * self.SPH_SUPPORT
        rho_bar = 0.5 * (rho[i] + rho[j])
        eps = 1e-5
        mu = h_sph * vr / (r**2 + eps)
        pi_ij = np.where(vr < 0, -self.SPH_ALPHA * cs * mu / rho_bar, 0.0)
        fvisc = -mf[i] * mf[j] * pi_ij[:, None] * grad
        np.add.at(fbuf, i, fvisc)
        np.add.at(fbuf, j, -fvisc)

        acc[fluid_idx] += fbuf / mf[:, None]
        return acc

    def _cloth_constraints(self) -> None:
        # PBD distance constraints on the bag edge network
        for _ in range(self.CLOTH_ITERATIONS):
            dx = self.x[self.cloth_edges[:, 1]] - self.x[self.cloth_edges[:, 0]]
            l = np.linalg.norm(dx, axis=1)
            l[l == 0] = 1e-12
            n = dx / l[:, None]
            delta = (l - self.cloth_l0) * self.CLOTH_STRETCH_STIFFNESS * 0.5
            corr = delta[:, None] * n
            np.add.at(self.x, self.cloth_edges[:, 0], corr)
            np.add.at(self.x, self.cloth_edges[:, 1], -corr)

        # fixed rim
        self.x[self.rim_indices] = self.x_ref[self.rim_indices]

    def _closest_on_triangles(
        self,
        p: NDArray[np.floating],
        tri0: NDArray[np.floating],
        tri1: NDArray[np.floating],
        tri2: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        v0 = tri1 - tri0
        v1 = tri2 - tri0
        v2 = p - tri0
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denom = d00 * d11 - d01 * d01
        denom = np.where(denom == 0, 1e-12, denom)
        inv = 1.0 / denom
        u = (d11 * d20 - d01 * d21) * inv
        v = (d00 * d21 - d01 * d20) * inv
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        uv = u + v
        scale = np.where(uv > 1.0, 1.0 / uv, 1.0)
        u = u * scale
        v = v * scale
        closest = tri0 + u[:, None] * v0 + v[:, None] * v1
        return closest, u, v

    def _cloth_fluid_collision(self) -> NDArray[np.floating]:
        """Apply two-sided cloth-fluid contact corrections.

        Returns the impulse applied to water by the cloth during this step.
        """
        fluid_mask = self.body_id == 1
        fluid_idx = np.flatnonzero(fluid_mask)
        if fluid_idx.size == 0 or self.bag_faces.shape[0] == 0:
            return np.zeros(3, dtype=float)

        bag_idx = np.arange(self.bag_count)
        centroids = (
            self.x[self.bag_faces[:, 0]]
            + self.x[self.bag_faces[:, 1]]
            + self.x[self.bag_faces[:, 2]]
        ) / 3.0
        face_tree = cKDTree(centroids)
        fluid_tree = cKDTree(self.x[fluid_idx])
        radius = self.COLLISION_MARGIN + 0.05
        pairs = face_tree.sparse_distance_matrix(
            fluid_tree, max_distance=radius, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.zeros(3, dtype=float)

        f_idx = pairs["i"]
        p_loc = pairs["j"]
        p_glob = fluid_idx[p_loc]

        # current candidate face geometry
        tri0 = self.x[self.bag_faces[f_idx, 0]]
        tri1 = self.x[self.bag_faces[f_idx, 1]]
        tri2 = self.x[self.bag_faces[f_idx, 2]]
        n = self._current_face_normals(f_idx)

        closest, u, v = self._closest_on_triangles(
            self.x[p_glob], tri0, tri1, tri2
        )
        d_vec = self.x[p_glob] - closest
        d = np.einsum("ij,ij->i", d_vec, n)
        absd = np.abs(d)
        active = absd < self.COLLISION_MARGIN
        if not np.any(active):
            return np.zeros(3, dtype=float)

        f_idx = f_idx[active]
        p_glob = p_glob[active]
        u = u[active]
        v = v[active]
        d = d[active]
        n = n[active]
        absd = absd[active]
        penetration = self.COLLISION_MARGIN - absd
        sign = np.sign(d)
        sign[sign == 0] = 1.0

        dir_p = sign[:, None] * n  # move particle away from the surface
        corr = dir_p * penetration[:, None] * self.COLLISION_LAMBDA

        v0 = self.bag_faces[f_idx, 0]
        v1 = self.bag_faces[f_idx, 1]
        v2 = self.bag_faces[f_idx, 2]
        m0 = self.m[v0]
        m1 = self.m[v1]
        m2 = self.m[v2]
        b0 = 1.0 - u - v
        b1 = u
        b2 = v
        m_tri = b0 * m0 + b1 * m1 + b2 * m2
        m_p = self.m[p_glob]
        total = m_p + m_tri
        total[total == 0] = 1e-12

        w_particle = m_tri / total
        w_cloth = -m_p / total

        dp = corr * w_particle[:, None]
        dc = corr * w_cloth[:, None]

        x_before = self.x.copy()
        np.add.at(self.x, p_glob, dp)
        np.add.at(
            self.x,
            v0,
            dc * (b0 * m0 / m_tri)[:, None],
        )
        np.add.at(
            self.x,
            v1,
            dc * (b1 * m1 / m_tri)[:, None],
        )
        np.add.at(
            self.x,
            v2,
            dc * (b2 * m2 / m_tri)[:, None],
        )

        # impulse delivered to water by cloth
        delta_w = self.x[p_glob] - x_before[p_glob]
        impulse = np.sum(self.m[p_glob][:, None] * delta_w, axis=0) / self.dt
        return impulse

    def _current_face_normals(self, f_idx: NDArray[np.integer]) -> NDArray[np.floating]:
        a = self.x[self.bag_faces[f_idx, 0]]
        b = self.x[self.bag_faces[f_idx, 1]]
        c = self.x[self.bag_faces[f_idx, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        ln[ln == 0] = 1e-12
        n = n / ln[:, None]
        # keep the original outward orientation
        flip = np.einsum("ij,ij->i", n, self.face_normal[f_idx]) < 0
        n[flip] = -n[flip]
        return n

    def _emit_inflow(self) -> None:
        if self.t < self.inflow_active[0] or self.t >= self.inflow_active[1]:
            return
        rate = (
            self.SPH_RHO0
            * self.inflow_speed
            * np.pi
            * self.inflow_radius**2
        )
        self._inflow_frac += rate * self.dt / self._water_particle_mass
        n_new = int(self._inflow_frac)
        if n_new <= 0:
            return
        self._inflow_frac -= n_new

        # deterministic jittered disk samples
        r = self.inflow_radius * np.sqrt(np.random.rand(n_new))
        theta = 2.0 * np.pi * np.random.rand(n_new)
        pos = np.empty((n_new, 3), dtype=float)
        pos[:, 0] = r * np.cos(theta)
        pos[:, 1] = r * np.sin(theta)
        pos[:, 2] = self.inflow_center[2]
        vel = np.tile(self.inflow_dir * self.inflow_speed, (n_new, 1))

        mass = np.full(n_new, self._water_particle_mass, dtype=float)
        bids = np.ones(n_new, dtype=np.int32)
        pinned = np.zeros(n_new, dtype=float)

        inflow_mom = np.sum(mass[:, None] * vel, axis=0)
        self._cum_inflow_momentum[1] += inflow_mom
        self._cum_mass_source += float(np.sum(mass))

        self.x = np.vstack((self.x, pos))
        self.v = np.vstack((self.v, vel))
        self.m = np.concatenate((self.m, mass))
        self.body_id = np.concatenate((self.body_id, bids))
        self.pinned = np.concatenate((self.pinned, pinned))

    def _delete_out_of_bounds(self) -> None:
        x_out = (
            (self.x[:, 0] < -2.4)
            | (self.x[:, 0] > 2.4)
            | (self.x[:, 1] < -2.4)
            | (self.x[:, 1] > 2.4)
            | (self.x[:, 2] > 2.5)
        )
        # only delete water particles; keep the bag
        x_out &= self.body_id == 1
        if not np.any(x_out):
            return
        out_mom = np.sum(self.m[x_out][:, None] * self.v[x_out], axis=0)
        self._cum_outflow_momentum[1] += out_mom
        keep = ~x_out
        self.x = self.x[keep]
        self.v = self.v[keep]
        self.m = self.m[keep]
        self.body_id = self.body_id[keep]
        self.pinned = self.pinned[keep]

    def _step(self) -> None:
        x_old = self.x.copy()
        v_old = self.v.copy()

        # accelerations: gravity + SPH
        acc = np.zeros_like(self.v)
        acc += self.GRAVITY[None, :]
        acc += self._sph_forces()

        # explicit Euler predictor
        self.v = v_old + acc * self.dt
        self.x = x_old + self.v * self.dt

        # cloth distance constraints (elastic response)
        self._cloth_constraints()

        # cloth-fluid coupling
        impulse = self._cloth_fluid_collision()
        self._interaction_impulse_interval += impulse

        # floor contact (catch floor at z=0)
        x_before_floor = self.x.copy()
        below = self.x[:, 2] < self.FLOOR_MARGIN
        if np.any(below):
            self.x[below, 2] = self.FLOOR_MARGIN
            self.x[below, 0] = x_before_floor[below, 0]
            self.x[below, 1] = x_before_floor[below, 1]
            # simple friction/damping on tangential motion by not changing x,y
            floor_impulse = (
                self.m[below][:, None]
                * (self.x[below] - x_before_floor[below])
                / self.dt
            )
            for bid in (0, 1):
                mask = self.body_id[below] == bid
                if np.any(mask):
                    self._cum_external_impulse[bid] += np.sum(floor_impulse[mask], axis=0)

        # delete escaped water
        self._delete_out_of_bounds()

        # fixed rim support impulse
        x_before_rim = self.x.copy()
        self.x[self.rim_indices] = self.x_ref[self.rim_indices]
        rim_impulse = (
            self.m[self.rim_indices][:, None]
            * (self.x[self.rim_indices] - x_before_rim[self.rim_indices])
            / self.dt
        )
        self._cum_external_impulse[0] += np.sum(rim_impulse, axis=0)

        # derive velocities from corrected displacements
        self.v = (self.x - x_old) / self.dt
        self.v[self.rim_indices] = 0.0

        # inflow at the end of the substep
        self._emit_inflow()

        self.t += self.dt
        self._dt_accumulated += self.dt

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.t - 1e-9:
            raise ValueError("advance_to time must be non-decreasing")
        n_steps = int(round((time - self.t) / self.dt))
        for _ in range(n_steps):
            self._step()
        # guard against floating drift
        self.t = time
        return self._build_frame(time)

    def _build_frame(self, time: float) -> StateFrame:
        n = len(self.x)
        bag_mask = self.body_id == 0
        water_mask = self.body_id == 1

        # gravity external force per domain (single term K=1)
        bag_gravity = np.sum(self.m[bag_mask][:, None] * self.GRAVITY, axis=0)
        water_gravity = np.sum(self.m[water_mask][:, None] * self.GRAVITY, axis=0)
        budget_external_force = np.zeros((2, 1, 3), dtype=float)
        budget_external_force[0, 0] = bag_gravity
        budget_external_force[1, 0] = water_gravity

        weights = np.zeros((2, n), dtype=float)
        weights[0, bag_mask] = 1.0
        weights[1, water_mask] = 1.0

        dt_int = self._dt_accumulated if self._dt_accumulated > 0 else self.sample_dt
        interaction_force = self._interaction_impulse_interval / dt_int
        self._interaction_impulse_interval.fill(0.0)
        self._dt_accumulated = 0.0

        fields = {
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target.copy(),
            "boundary_velocity": self.boundary_vel.copy(),
            "budget_cumulative_external_impulse": self._cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self._cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self._cum_outflow_momentum.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": np.array([self._cum_mass_source], dtype=float),
            "interaction_force": interaction_force.reshape(1, 3),
        }

        attributes = {
            "pinned": self.pinned.astype(float),
            "material_phase": self.body_id.astype(float),
            "mass_quantity": self.m[:, None].copy(),
        }

        scalars = dict(self._scalars)

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver_SPH_PBD",
            "substeps": self.SUBSTEPS,
            "sph_support": self.SPH_SUPPORT,
            "particle_count": len(self.x),
            "bag_particles": int(np.sum(self.body_id == 0)),
            "water_particles": int(np.sum(self.body_id == 1)),
        }
