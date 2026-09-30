"""Two-way coupled smoke / hanging impermeable cloth curtain solver.

Physical model
--------------
* Incompressible constant-density air on the frozen uniform 126x70x70
  cell-centred grid.  Each step uses a stable semi-Lagrangian advection of the
  velocity, a no-penetration penalisation of the moving cloth, and a Jacobi
  pressure projection with open (zero-gradient) boundaries.
* Smoke is a passive tracer advected by the projected velocity and injected by
  the jet nozzle with the prescribed start/ramp schedule.
* The curtain is a mass-spring shell (structural, shear and bending springs)
  with only its two top corners pinned.  Local aerodynamic drag from the fluid
  velocity sampled at the vertices pushes it downstream; the moving sheet is fed
  back into the fluid as a no-penetration force so the smoke must bend around
  its free edges.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        cloth = cfg["cloth"]
        jet = cfg["jet"]
        dom = cfg["domain"]
        self.gravity = np.asarray(cfg["initial_conditions"]["gravity"], float)

        self.shape = tuple(int(s) for s in dom["grid_shape"])
        self.low = np.asarray(dom["bounds_min"], float)
        self.high = np.asarray(dom["bounds_max"], float)
        self.dvec = (self.high - self.low) / np.asarray(self.shape, float)
        self.dx = float(self.dvec[0])
        self.origin = self.low + 0.5 * self.dvec
        axes = [self.origin[k] + np.arange(self.shape[k]) * self.dvec[k] for k in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.cell = np.stack((X, Y, Z), axis=-1)
        self.cellvol = float(np.prod(self.dvec))

        frame = scene.initial_state()
        self.pos = np.array(frame.position, float)
        self.vel = np.array(frame.velocity, float)
        self.mass = np.array(frame.mass, float)
        self.ncl = self.pos.shape[0]
        self.mass0 = self.mass.copy()
        self.pos0 = self.pos.copy()
        self.vel0 = self.vel.copy()
        self.uv = np.array(frame.attributes["uv"], float)
        self.pinned = np.asarray(cfg["cloth"]["pinned_vertices"], int)
        self.pin_mask = np.zeros(self.ncl, bool)
        self.pin_mask[self.pinned] = True

        faces = scene.metadata()["topology"]["faces"]
        self.faces = np.asarray(faces, int)

        self.u = np.array(frame.fields["fluid_velocity"], float)
        if self.u.ndim == 4 and self.u.shape[0] == 3:  # tolerate channel-first layout
            self.u = np.moveaxis(self.u, 0, -1)
        self.smoke = np.array(frame.fields["smoke_density"], float)
        self.pressure = np.zeros(self.shape, float)

        # ---- cloth discretisation -----------------------------------------
        nx, nz = map(int, cloth["grid_shape"])
        self.nx, self.nz = nx, nz
        width, height = float(cloth["width"]), float(cloth["height"])
        self.width, self.height = width, height
        self.dy = width / (nx - 1)
        self.dz = height / (nz - 1)
        self.diag = float(np.hypot(self.dy, self.dz))
        ea, eb, rest, stiff = [], [], [], []
        for j in range(nz):
            for i in range(nx):
                a = j * nx + i
                if i + 1 < nx:
                    ea.append(a); eb.append(a + 1); rest.append(self.dy); stiff.append(4000.0)
                if j + 1 < nz:
                    ea.append(a); eb.append(a + nx); rest.append(self.dz); stiff.append(4000.0)
                if i + 1 < nx and j + 1 < nz:
                    ea.append(a); eb.append(a + nx + 1); rest.append(self.diag); stiff.append(1200.0)
                    ea.append(a + 1); eb.append(a + nx); rest.append(self.diag); stiff.append(1200.0)
                if i + 2 < nx:
                    ea.append(a); eb.append(a + 2); rest.append(2.0 * self.dy); stiff.append(150.0)
                if j + 2 < nz:
                    ea.append(a); eb.append(a + 2 * nx); rest.append(2.0 * self.dz); stiff.append(150.0)
        self.ea = np.asarray(ea, int)
        self.eb = np.asarray(eb, int)
        self.rest = np.asarray(rest, float)
        self.stiff = np.asarray(stiff, float)
        self.edamp = 0.25 * 2.0 * np.sqrt(self.stiff * (self.mass.mean()))
        self.air_damp = 0.7

        # ---- jet / forcing regions ----------------------------------------
        self.jet_center = np.asarray(jet["center"], float)
        self.jet_speed = float(jet["speed"])
        self.jet_radius = float(jet["radius"])
        self.jet_density = float(jet["smoke_density"])
        self.jet_start = float(jet["start_time"])
        self.jet_ramp = float(jet["ramp_duration"])
        self.jet_until = float(jet["active_until"])
        yc, zc = self.jet_center[1], self.jet_center[2]
        cyl = (self.cell[..., 1] - yc) ** 2 + (self.cell[..., 2] - zc) ** 2 <= self.jet_radius ** 2
        self.jet_region = cyl & (self.cell[..., 0] <= self.jet_center[0] + 0.18) & (
            self.cell[..., 0] >= self.low[0]
        )
        self.jet_region.setflags(write=False)

        # ---- aerodynamic coupling constants -------------------------------
        self.vertex_area = (width * height) / self.ncl
        self.aero = 0.5 * 1.2 * 1.2 * self.vertex_area

        # ---- state bookkeeping --------------------------------------------
        self.time = 0.0
        self.n_pressure = 22
        self.n_sub = 16
        self._pin_force = np.zeros(3)
        self._drag_total = np.zeros(3)
        self._u_at_cloth = np.zeros((self.ncl, 3))
        self._boundary_normal = self._update_vertex_normals(self.pos)
        self.inflow_momentum = np.zeros(3)
        self.outflow_momentum = np.zeros(3)
        self.smoke_injected = 0.0
        self._tree = None
        self._tree_pts = None
        self._tree_n = None
        self._tree_v = None
        self.reference_velocity = self.jet_speed
        self.reference_momentum = float(self.mass.sum()) * self.jet_speed

    # ------------------------------------------------------------ utilities
    def _update_vertex_normals(self, pos: np.ndarray) -> np.ndarray:
        faces = self.faces
        pv = pos[faces]
        fn = np.cross(pv[:, 1] - pv[:, 0], pv[:, 2] - pv[:, 0])
        vn = np.zeros_like(pos)
        for c in range(3):
            np.add.at(vn, faces[:, c], fn)
        length = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(length, 1e-12)

    def _sample_vec(self, field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        idx = (pts - self.origin) / self.dvec
        coords = [idx[:, 0], idx[:, 1], idx[:, 2]]
        out = np.empty((pts.shape[0], 3), float)
        for c in range(3):
            out[:, c] = map_coordinates(field[..., c], coords, order=1, mode="nearest")
        return out

    # ------------------------------------------------------------- fluid ops
    def _advect_vec(self, field: np.ndarray, dt: float) -> np.ndarray:
        out = np.empty_like(field)
        for c in range(3):
            back = self.cell - dt * field[..., c][..., None]
            coords = [(back[..., k] - self.origin[k]) / self.dvec[k] for k in range(3)]
            out[..., c] = map_coordinates(field[..., c], coords, order=1, mode="nearest")
        return out

    def _advect_scalar(self, field: np.ndarray, dt: float) -> np.ndarray:
        back = self.cell - dt * self.u
        coords = [(back[..., k] - self.origin[k]) / self.dvec[k] for k in range(3)]
        return map_coordinates(field, coords, order=1, mode="nearest")

    def _project(self, u: np.ndarray, dt: float) -> np.ndarray:
        dx, dy, dz = self.dvec
        div = (
            np.gradient(u[..., 0], dx, axis=0)
            + np.gradient(u[..., 1], dy, axis=1)
            + np.gradient(u[..., 2], dz, axis=2)
        )
        f = div / dt
        p = self.pressure
        d2 = self.dx * self.dx
        for _ in range(self.n_pressure):
            pp = np.pad(p, 1, mode="edge")
            p = (
                pp[2:, 1:-1, 1:-1] + pp[:-2, 1:-1, 1:-1]
                + pp[1:-1, 2:, 1:-1] + pp[1:-1, :-2, 1:-1]
                + pp[1:-1, 1:-1, 2:] + pp[1:-1, 1:-1, :-2]
                - d2 * f
            ) / 6.0
        self.pressure = p
        out = u.copy()
        out[..., 0] -= dt * np.gradient(p, dx, axis=0)
        out[..., 1] -= dt * np.gradient(p, dy, axis=1)
        out[..., 2] -= dt * np.gradient(p, dz, axis=2)
        return out

    def _rebuild_tree(self) -> None:
        pos, faces = self.pos, self.faces
        pv = pos[faces]
        vface = self.vel[faces].mean(axis=1)
        nface = np.cross(pv[:, 1] - pv[:, 0], pv[:, 2] - pv[:, 0])
        nlen = np.linalg.norm(nface, axis=1, keepdims=True)
        nface = nface / np.maximum(nlen, 1e-12)
        centroid = pv.mean(axis=1)
        pts = np.concatenate((pv.reshape(-1, 3), centroid), axis=0)
        vels = np.concatenate((self.vel[faces].reshape(-1, 3), vface), axis=0)
        normals = np.concatenate(
            (np.repeat(nface, 3, axis=0), nface), axis=0
        )
        self._tree = cKDTree(pts)
        self._tree_pts = pts
        self._tree_n = normals
        self._tree_v = vels

    def _penalise_cloth(self, u: np.ndarray, alpha: float = 0.55, h: float = 0.057) -> None:
        if self._tree is None:
            return
        lo = self.pos.min(axis=0) - 3.0 * self.dx
        hi = self.pos.max(axis=0) + 3.0 * self.dx
        i0 = int(np.clip(np.floor((lo[0] - self.origin[0]) / self.dvec[0]), 0, self.shape[0] - 1))
        i1 = int(np.clip(np.ceil((hi[0] - self.origin[0]) / self.dvec[0]) + 1, 0, self.shape[0]))
        j0 = int(np.clip(np.floor((lo[1] - self.origin[1]) / self.dvec[1]), 0, self.shape[1] - 1))
        j1 = int(np.clip(np.ceil((hi[1] - self.origin[1]) / self.dvec[1]) + 1, 0, self.shape[1]))
        k0 = int(np.clip(np.floor((lo[2] - self.origin[2]) / self.dvec[2]), 0, self.shape[2] - 1))
        k1 = int(np.clip(np.ceil((hi[2] - self.origin[2]) / self.dvec[2]) + 1, 0, self.shape[2]))
        if i1 - i0 < 1 or j1 - j0 < 1 or k1 - k0 < 1:
            return
        sub = u[i0:i1, j0:j1, k0:k1]
        flat = np.ascontiguousarray(sub).reshape(-1, 3)
        dist, idx = self._tree.query(flat, k=1)
        mask = np.clip(1.0 - (dist - 0.5 * self.dx) / (1.5 * self.dx), 0.0, 1.0)
        active = mask > 1e-6
        if np.any(active):
            n = self._tree_n[idx]
            vc = self._tree_v[idx]
            rel = flat - vc
            un = np.einsum("ij,ij->i", rel, n)
            corr = (alpha * mask * un)[:, None] * n
            flat -= corr
            sub[...] = flat.reshape(sub.shape)

    def _budget_flux(self, dt: float, rho: float = 1.0) -> None:
        dA = self.dvec
        specs = [
            (0, 0, -1, dA[1] * dA[2]),
            (0, -1, 1, dA[1] * dA[2]),
            (1, 0, -1, dA[0] * dA[2]),
            (1, -1, 1, dA[0] * dA[2]),
            (2, 0, -1, dA[0] * dA[1]),
            (2, -1, 1, dA[0] * dA[1]),
        ]
        for axis, plane, sign, area in specs:
            take = [slice(None)] * 3
            take[axis] = plane
            take = tuple(take)
            un = sign * self.u[..., axis][take]
            uvec = self.u[take]
            flux = rho * un[..., None] * uvec * area * dt
            entering = un < 0.0
            leaving = un > 0.0
            if np.any(entering):
                self.inflow_momentum -= flux[entering].sum(axis=0)
            if np.any(leaving):
                self.outflow_momentum += flux[leaving].sum(axis=0)

    # ------------------------------------------------------------- cloth ops
    def _spring_force(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        dr = pos[self.ea] - pos[self.eb]
        length = np.linalg.norm(dr, axis=1, keepdims=True)
        direction = dr / np.maximum(length, 1e-12)
        f = (self.stiff * (length[:, 0] - self.rest))[:, None] * direction
        rv = vel[self.ea] - vel[self.eb]
        fd = (self.edamp * np.einsum("ij,ij->i", rv, direction))[:, None] * direction
        f = f + fd
        out = np.zeros_like(pos)
        np.add.at(out, self.ea, f)
        np.add.at(out, self.eb, -f)
        return out

    def _drag_force(self, vel: np.ndarray, u_at_cloth: np.ndarray) -> np.ndarray:
        vrel = vel - u_at_cloth
        speed = np.linalg.norm(vrel, axis=1, keepdims=True)
        return -self.aero * speed * vrel

    def _advance_cloth(self, dt: float, u_at_cloth: np.ndarray) -> None:
        h = dt / self.n_sub
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        for _ in range(self.n_sub):
            applied = self._spring_force(self.pos, self.vel) + gravity_force
            applied = applied + self._drag_force(self.vel, u_at_cloth)
            self._pin_force = -applied[self.pin_mask].sum(axis=0)
            applied[self.pin_mask] = 0.0
            self.vel += h * applied / self.mass[:, None]
            self.vel *= max(0.0, 1.0 - self.air_damp * h)
            np.clip(self.vel, -12.0, 12.0, out=self.vel)
            self.pos += h * self.vel
            self.pos[self.pin_mask] = self.pos0[self.pin_mask]
            self.vel[self.pin_mask] = 0.0

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        sample_dt = 0.02
        nsteps = int(round((target - self.time) / sample_dt))
        if nsteps <= 0:
            return self._make_frame(self.time)
        dt = (target - self.time) / nsteps
        for _ in range(nsteps):
            self._step(dt, self.time)
        self.time = target
        return self._make_frame(target)

    def _step(self, dt: float, t: float) -> None:
        ramp = 0.0
        if self.jet_start <= t < self.jet_until:
            ramp = float(np.clip((t - self.jet_start) / max(self.jet_ramp, 1e-9), 0.0, 1.0))

        # 1. jet forcing
        if ramp > 0.0:
            beta = 0.85 * ramp
            region = self.jet_region
            vx = self.u[..., 0]
            vy = self.u[..., 1]
            vz = self.u[..., 2]
            vx[region] = (1.0 - beta) * vx[region] + beta * self.jet_speed
            vy[region] = (1.0 - beta) * vy[region]
            vz[region] = (1.0 - beta) * vz[region]

        # 2. buoyant smoke
        self.u[..., 2] += dt * 0.35 * self.smoke

        # 3. advect velocity
        self.u = self._advect_vec(self.u, dt)
        np.clip(self.u, -12.0, 12.0, out=self.u)

        # 4. fluid -> cloth coupling (sampled fluid velocity at the vertices)
        self._u_at_cloth = self._sample_vec(self.u, self.pos)

        # 5. cloth -> fluid coupling (moving no-penetration boundary)
        self._rebuild_tree()
        self._penalise_cloth(self.u, alpha=0.55)

        # 6. incompressible projection
        self.u = self._project(self.u, dt)

        # 7. smoke transport + prescribed injection
        self.smoke = self._advect_scalar(self.smoke, dt)
        np.clip(self.smoke, 0.0, 4.0, out=self.smoke)
        if ramp > 0.0:
            region = self.jet_region
            cur = self.smoke[region]
            target = ramp * self.jet_density
            added = np.maximum(target - cur, 0.0)
            self.smoke_injected += float(added.sum()) * self.cellvol
            self.smoke[region] = np.maximum(cur, target)

        # 8. momentum budgets
        self._budget_flux(dt)

        # 9. cloth dynamics with frozen fluid sample over the step
        self._advance_cloth(dt, self._u_at_cloth)
        self.time += dt
        self._boundary_normal = self._update_vertex_normals(self.pos)

    # ------------------------------------------------------------ frame dump
    def _make_frame(self, t: float):
        u_at_cloth = self._sample_vec(self.u, self.pos)
        self._drag_total = self._drag_force(self.vel, u_at_cloth).sum(axis=0)
        gravity_total = (self.mass[:, None] * self.gravity[None, :]).sum(axis=0)
        impulse = (self.mass[:, None] * (self.vel - self.vel0)).sum(axis=0)

        external_force = np.stack(
            (gravity_total, self._drag_total, self._pin_force), axis=0
        )[None, ...]
        interaction = np.stack((self._drag_total, -self._drag_total), axis=0)

        frame = StateFrame(
            time=t,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.ones(self.ncl, dtype=np.int32),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pin_mask.astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "smoke_density": self.smoke.astype(float).copy(),
                "fluid_velocity": self.u.astype(float).copy(),
                "boundary_normal": self._boundary_normal.copy(),
                "boundary_target_velocity": self.vel.copy(),
                "boundary_velocity": u_at_cloth.copy(),
                "budget_domain_weights": np.ones((1, self.ncl), float),
                "budget_external_force": external_force.astype(float),
                "budget_cumulative_external_impulse": impulse[None, :].astype(float),
                "budget_cumulative_inflow_momentum": self.inflow_momentum[None, :].astype(float),
                "budget_cumulative_outflow_momentum": self.outflow_momentum[None, :].astype(float),
                "interaction_force": interaction.astype(float),
                "cumulative_mass_source": np.array([0.0, self.smoke_injected], float),
            },
            scalars={
                "reference_velocity": float(self.reference_velocity),
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "smoke_injected": float(self.smoke_injected),
            "max_speed": float(np.linalg.norm(self.u, axis=-1).max()),
            "cloth_displacement": float(
                np.linalg.norm(self.pos - self.pos0, axis=1).max()
            ),
        }
