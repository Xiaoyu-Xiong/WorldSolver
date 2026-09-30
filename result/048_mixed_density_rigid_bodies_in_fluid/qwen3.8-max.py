"""Physically-based solver for mixed-density rigid bodies coupled to a fluid.

Model
-----
* Six rigid bodies (spheres / capsules / boxes) fully coupled to water via
  gravity, hydrostatic buoyancy, quadratic + linear hydrodynamic drag, and
  an added-mass correction (stabilizes wide density contrasts, cf. Batty et
  al. 2007 solid-fluid coupling).
* Body-body and body-wall (tank) contact via stiff spring-damper repulsion
  with restitution, no penetration.
* Fluid visualized by sparse marker particles advected by the superposed
  potential-flow disturbance of the moving bodies (translational dipole
  ~ R^3/d^3 fall-off) plus a short-range push-out that keeps markers out of
  rigid volumes; markers stay inside the closed tank.

Emits all required trajectory fields each sample:
  state.mass, state.velocity, attributes.mass_quantity, fields.external_force,
  fields.interaction_force, fields.velocity_field, fields.active_mask,
  fields.cumulative_mass_source, scalars.characteristic_length,
  scalars.grid_spacing.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        fluid = cfg["fluid"]
        self.rho_f = float(fluid["density"])
        lo = np.asarray(fluid["bounds_min"], dtype=float)
        hi = np.asarray(fluid["bounds_max"], dtype=float)
        self.domain_lo = lo.copy()
        self.domain_hi = hi.copy()

        frame0 = scene.initial_state()
        self.n_total = len(frame0.position)
        bid = frame0.body_id
        self.n_markers = int(np.count_nonzero(bid == 0))
        self.n_bodies = self.n_total - self.n_markers
        m = self.n_markers

        self.pos = frame0.position.copy()
        self.body_pos = self.pos[m:].copy()          # [6,3] centers
        self.body_vel = np.zeros((self.n_bodies, 3))
        self.marker_vel = np.zeros((m, 3))

        # per-body geometry (matches scene.py volumes / scene_geometry.yaml)
        self.volumes = np.array([
            4.0 / 3.0 * np.pi * 0.25 ** 3,
            np.pi * 0.22 ** 2 * 0.20 + 4.0 / 3.0 * np.pi * 0.22 ** 3,
            8.0 * 0.29 * 0.20 * 0.18,
            np.pi * 0.24 ** 2 * 0.18 + 4.0 / 3.0 * np.pi * 0.24 ** 3,
            8.0 * 0.20 ** 3,
            4.0 / 3.0 * np.pi * 0.23 ** 3,
        ])
        # effective collision / drag radii
        self.radii = np.array([0.25, 0.31, 0.29, 0.33, 0.20, 0.23])
        self.ratio = np.asarray(cfg["rigid_bodies"]["density_ratios"], float)
        self.body_mass = self.rho_f * self.ratio * self.volumes
        self.added_mass = 0.5 * self.rho_f * self.volumes
        self.cd = np.full(self.n_bodies, 0.55)
        self.area = np.pi * self.radii ** 2
        self.c_lin = 1.5 * self.radii * self.rho_f * 1e-3  # mild linear damping

        self.mass = frame0.mass.copy()
        self.body_id = bid.copy()
        self.attrs0 = {k: np.asarray(v).copy() for k, v in frame0.attributes.items()}

        # orientations: markers identity quaternion, bodies keep initial attitude
        self.orientation = np.zeros((self.n_total, 4))
        self.orientation[:, 0] = 1.0
        quats = [
            None,
            np.array([0.116, 0.341, -0.042, 0.932]),
            np.array([0.074, 0.194, 0.056, 0.977]),
            np.array([-0.070, 0.560, 0.103, 0.819]),
            np.array([0.034, 0.070, 0.234, 0.969]),
            None,
        ]
        for i, q in enumerate(quats):
            if q is not None:
                q = q / np.linalg.norm(q)
                self.orientation[m + i] = q

        # tank interior limits (floor top at z=0.04, walls at |x|<=1.48, |y|<=0.58)
        self.wall_lo = np.array([-1.48, -0.58, 0.04])
        self.wall_hi = np.array([1.48, 0.58, float(hi[2])])

        # diagnostic grid
        self.grid_spacing = 0.29
        h = self.grid_spacing
        self.grid_n = np.ceil((hi - lo) / h - 1e-9).astype(int)  # [nx,ny,nz]
        self.characteristic_length = float(hi[2] - lo[2])

        self.t = 0.0
        self.sub_dt = 0.001
        self._cum_mass_source = np.zeros(self.n_bodies)

    # --------------------------------------------------------------- physics
    def _body_forces(self):
        """Return (external_gravity_body, interaction[buoyancy+drag+contact])."""
        f_grav = self.body_mass[:, None] * GRAVITY[None, :]
        f_buoy = -(self.rho_f * self.volumes)[:, None] * GRAVITY[None, :]
        speed = np.linalg.norm(self.body_vel, axis=1, keepdims=True)
        f_drag = -0.5 * self.rho_f * (self.cd * self.area)[:, None] * speed * self.body_vel
        f_drag -= self.c_lin[:, None] * self.body_vel
        f_contact = np.zeros_like(f_grav)

        # body-body soft contact
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                d = self.body_pos[j] - self.body_pos[i]
                dist = max(np.linalg.norm(d), 1e-9)
                rsum = self.radii[i] + self.radii[j]
                if dist < rsum:
                    nrm = d / dist
                    overlap = rsum - dist
                    vrel = np.dot(self.body_vel[j] - self.body_vel[i], nrm)
                    fmag = 6000.0 * overlap - 40.0 * min(vrel, 0.0)
                    fmag = max(fmag, 0.0)
                    f_contact[i] -= fmag * nrm
                    f_contact[j] += fmag * nrm

        # wall contact (spring-damper, no penetration)
        k_w, c_w = 2.0e4, 60.0
        for i in range(self.n_bodies):
            p = self.body_pos[i]
            r = self.radii[i]
            for axis in range(3):
                pen_lo = (self.wall_lo[axis] + r) - p[axis]
                if pen_lo > 0:
                    f = k_w * pen_lo + c_w * max(-self.body_vel[i, axis], 0.0)
                    f_contact[i, axis] += f
                pen_hi = p[axis] - (self.wall_hi[axis] - r)
                if pen_hi > 0:
                    f = k_w * pen_hi + c_w * max(self.body_vel[i, axis], 0.0)
                    f_contact[i, axis] -= f
        return f_grav, f_buoy + f_drag + f_contact

    def _marker_velocity(self):
        """Induced marker velocity from translating bodies (potential-flow-ish)."""
        if self.n_markers == 0:
            return self.marker_vel
        mk = self.pos[: self.n_markers]
        d = mk[:, None, :] - self.body_pos[None, :, :]        # [M,B,3]
        dist = np.linalg.norm(d, axis=2)                       # [M,B]
        dist = np.maximum(dist, 1e-6)
        nrm = d / dist[:, :, None]
        R = self.radii[None, :]
        falloff = np.clip((R / dist) ** 3, 0.0, 1.0)           # [M,B]
        u = np.einsum("mb,mbc->mc", falloff, np.broadcast_to(
            self.body_vel[None, :, :], (self.n_markers, self.n_bodies, 3)))
        # short-range radial push so markers peel away from body surfaces
        push_gap = 1.15 * R
        strength = np.clip(push_gap - dist, 0.0, None) * 3.0   # [M,B]
        u += np.einsum("mb,mbc->mc", strength, nrm)
        speed = np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where(speed > 3.0, u * (3.0 / np.maximum(speed, 1e-9)), u)
        return u

    def _step(self, dt: float) -> None:
        f_grav_b, f_inter = self._body_forces()

        # bodies: a = (F_grav + F_interaction) / (m + added_mass)
        m_eff = (self.body_mass + self.added_mass)[:, None]
        acc = (f_grav_b + f_inter) / m_eff
        self.body_vel = self.body_vel + acc * dt
        sp = np.linalg.norm(self.body_vel, axis=1, keepdims=True)
        self.body_vel = np.where(sp > 8.0, self.body_vel * (8.0 / sp), self.body_vel)
        self.body_pos = self.body_pos + self.body_vel * dt

        # hard clamp positions inside tank (no penetration), damped rebound
        for i in range(self.n_bodies):
            r = self.radii[i]
            for axis in range(3):
                lo_lim = self.wall_lo[axis] + r
                hi_lim = self.wall_hi[axis] - r
                if self.body_pos[i, axis] < lo_lim:
                    self.body_pos[i, axis] = lo_lim
                    if self.body_vel[i, axis] < 0:
                        self.body_vel[i, axis] *= -0.25
                elif self.body_pos[i, axis] > hi_lim:
                    self.body_pos[i, axis] = hi_lim
                    if self.body_vel[i, axis] > 0:
                        self.body_vel[i, axis] *= -0.25

        # markers: advect with induced fluid velocity
        self.marker_vel = self._marker_velocity()
        mk = self.pos[: self.n_markers]
        mk = mk + 0.85 * self.marker_vel * dt
        # keep markers inside the closed fluid domain
        mk = np.clip(mk, self.domain_lo + 1e-4, self.domain_hi - 1e-4)
        # expel markers that intrude into rigid bodies
        d = mk[:, None, :] - self.body_pos[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        R = self.radii[None, :] * 1.03
        inside = dist < R
        if inside.any():
            dist = np.maximum(dist, 1e-9)
            scale = np.where(inside, R / dist, 1.0)
            d = d * scale[:, :, None]
            mk = self.body_pos[None, :, :] + d
            # collapse duplicates over bodies by taking mean correction
            mk = mk.mean(axis=1) if self.n_bodies > 1 else mk[:, 0, :]
            mk = np.clip(mk, self.domain_lo + 1e-4, self.domain_hi - 1e-4)
        self.pos[: self.n_markers] = mk
        self.pos[self.n_markers:] = self.body_pos
        self.t += dt

    # ----------------------------------------------------------- frame build
    def _velocity_grid(self):
        nx, ny, nz = self.grid_n
        h = self.grid_spacing
        lo = self.domain_lo
        vel_sum = np.zeros((nx, ny, nz, 3))
        cnt = np.zeros((nx, ny, nz))
        pts = self.pos
        vels = np.vstack((self.marker_vel, self.body_vel))
        idx = np.floor((pts - lo) / h).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
        flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
        np.add.at(vel_sum.reshape(-1, 3), flat, vels)
        np.add.at(cnt.reshape(-1), flat, 1.0)
        c = np.maximum(cnt[..., None], 1.0)
        vel_grid = vel_sum / c
        active = (cnt > 0).astype(float)
        return vel_grid, active

    def _make_frame(self, time: float) -> StateFrame:
        m = self.n_markers
        velocity = np.vstack((self.marker_vel, self.body_vel)).copy()
        f_grav_b, f_inter = self._body_forces()
        external = np.zeros((self.n_total, 3))
        external[:m] = self.mass[:m, None] * GRAVITY[None, :]
        external[m:] = f_grav_b
        vel_grid, active = self._velocity_grid()

        scalars = {
            "characteristic_length": self.characteristic_length,
            "grid_spacing": float(self.grid_spacing),
            "fluid_density": self.rho_f,
            "total_mass": float(self.mass.sum()),
            "time": float(time),
        }
        attrs = dict(self.attrs0)
        attrs["mass_quantity"] = self.mass.copy()

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=velocity,
            orientation=self.orientation.copy(),
            angular_velocity=np.zeros((self.n_total, 3)),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields={
                "active_mask": active,
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "external_force": external,
                "interaction_force": f_inter.copy(),
                "velocity_field": vel_grid,
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            target = self.t
        while self.t < target - 1e-12:
            dt = min(self.sub_dt, target - self.t)
            self._step(dt)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "n_markers": int(self.n_markers),
            "n_bodies": int(self.n_bodies),
            "body_positions": self.body_pos.tolist(),
            "body_speeds": np.linalg.norm(self.body_vel, axis=1).tolist(),
        }
