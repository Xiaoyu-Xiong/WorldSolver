"""Deterministic soft-sphere DEM solver for the 2D thin-slab hourglass scene.

Physical model
--------------
The granular charge is integrated as a three-dimensional soft-sphere
discrete-element system confined to a thin slab (thickness
``2 * slab_half_depth``) whose mid-plane carries the axisymmetric hourglass
profile.  Contacts use a linear spring/dashpot in the normal direction with a
regularised Coulomb limit in the tangential direction.  The glass body is
kinematic: it rotates about the world ``y`` axis through its own centre by
180 degrees at t=6 s and again at t=12 s (smooth-step profile with zero
angular velocity at both ends).  Grains are kept inside the glass by
non-penetration impulses evaluated in the rotating body frame; those impulses
also carry the wall friction that drives the tumbling during the flips.

Everything is advanced with a fixed-step semi-implicit Euler scheme at
``dt = 1.25e-4`` s, which resolves the contact period of the stiffest spring
used here with a large margin.  The integration is fully deterministic: no
random numbers are drawn after initialisation.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
_DEFAULT_PROFILE = (
    (0.46, 0.02),
    (0.44, 0.14),
    (0.32, 0.36),
    (0.12, 0.60),
    (0.038, 0.72),
    (0.12, 0.84),
    (0.32, 1.08),
    (0.44, 1.30),
    (0.46, 1.42),
)

# Half of the 27-cell neighbourhood so every unordered pair is visited exactly
# once by the vectorised cell-list builder.
_OFFSETS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
)

_FLIP_TIMES = (6.0, 12.0)
_FLIP_DURATION = 0.8


def _flip_state(time: float):
    """Cumulative rotation angle (rad) and angular speed (rad/s) about y."""
    theta = 0.0
    omega = 0.0
    for t0 in _FLIP_TIMES:
        u = (time - t0) / _FLIP_DURATION
        if u <= 0.0:
            continue
        if u >= 1.0:
            theta += np.pi
            continue
        theta += np.pi * (u * u * (3.0 - 2.0 * u))
        omega += np.pi * (6.0 * u * (1.0 - u)) / _FLIP_DURATION
    return theta, omega


class BenchmarkSolver:
    """Soft-sphere DEM with a kinematic rotating hourglass shell."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        frame = scene.initial_state()

        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.n = int(self.pos.shape[0])
        self.mass = np.array(frame.mass, dtype=float).reshape(self.n)
        rad_attr = frame.attributes.get("particle_radius")
        self.radius = float(np.asarray(rad_attr).reshape(-1)[0])

        # --- gravity -----------------------------------------------------
        grav = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.gravity = grav

        # --- frozen shell geometry --------------------------------------
        profile = _DEFAULT_PROFILE
        center = np.array([0.0, 0.0, 0.08], dtype=float)
        geom = cfg.get("scene_geometry") or {}
        for obj in geom.get("objects", []):
            if obj.get("id") != "hourglass_shell":
                continue
            inner = obj.get("geometry", {}).get("inner_profile")
            if inner:
                profile = tuple((float(p[0]), float(p[1])) for p in inner)
            trans = obj.get("transform", {}) or {}
            if trans.get("translation") is not None:
                center = np.asarray(trans["translation"], dtype=float).reshape(3)
        prof = np.asarray(profile, dtype=float)
        order = np.argsort(prof[:, 1])
        self.prof_z = prof[order, 1].copy()
        self.prof_r = prof[order, 0].copy()
        self.prof_z_bot = float(self.prof_z[0])
        self.prof_z_top = float(self.prof_z[-1])
        self.center = center

        bc = cfg.get("boundary_conditions", {}) or {}
        self.slab_half = float(bc.get("slab_half_depth", 0.055))
        self.floor_top = 0.0
        self.floor_hx = 0.72
        self.floor_hy = 0.18

        # --- numerics ----------------------------------------------------
        self.dt = 1.25e-4
        self.k = 1.0e4          # normal contact stiffness [N/m]
        self.zeta = 0.3         # normal damping ratio
        self.mu = 0.35          # grain/grain friction coefficient
        self.mu_wall = 0.30     # grain/glass friction coefficient
        self.beta_wall = 0.35   # Baumgarte position bias for wall constraints
        self.vbias_max = 3.0
        self.max_overlap = 0.5 * self.radius
        self.f_max = 60.0
        self.skin = 1.5 * self.radius
        self.rebuild_every = 4
        self.n_pair_ops = 0

        # --- cell grid for neighbour search ------------------------------
        self.grid_lo = np.array([-0.75, -0.12, -0.10], dtype=float)
        self.grid_hi = np.array([0.75, 0.12, 1.75], dtype=float)
        cell = 2.0 * self.radius + self.skin
        span = np.maximum(self.grid_hi - self.grid_lo, cell)
        self.nx = int(np.ceil(span[0] / cell))
        self.ny = int(np.ceil(span[1] / cell))
        self.nz = int(np.ceil(span[2] / cell))
        self.pair_i = np.zeros(0, dtype=np.int64)
        self.pair_j = np.zeros(0, dtype=np.int64)
        self.rebuild_countdown = 0

        grain_volume = 4.0 * np.pi * self.radius ** 3 / 3.0
        self.grain_volume = grain_volume
        self.volume = np.full(self.n, grain_volume, dtype=float)
        self.radius_attr = np.full(self.n, self.radius, dtype=float)

        # --- place the charge inside the glass and settle it -------------
        self._rotate_cache = (0.0, 0.0)
        self.theta = 0.0
        self.omega = 0.0
        self._project_inside()
        self._step_index = 0
        self.time = 0.0
        self._wall_gaps = np.zeros(5 * self.n, dtype=float)
        self._ext_force = self.mass[:, None] * self.gravity
        settle_steps = int(round(0.30 / self.dt))
        for _ in range(settle_steps):
            self._step(vmax=1.5)
        self._step_index = 0
        self.time = 0.0
        self._ready = True

    def _project_inside(self) -> None:
        """Move any grain that starts outside the glass interior back inside."""
        r = self.radius
        hmax = max(self.slab_half - r, 0.0)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -hmax, hmax)
        z_local = np.clip(
            self.pos[:, 2] - self.center[2],
            self.prof_z_bot + r,
            self.prof_z_top - r,
        )
        r_here = np.interp(z_local, self.prof_z, self.prof_r)
        xmax = np.maximum(r_here - r, 0.0)
        self.pos[:, 0] = np.clip(self.pos[:, 0], -xmax, xmax)
        self.pos[:, 2] = z_local + self.center[2]

    # ------------------------------------------------------- neighbour list
    def _rebuild_pairs(self) -> None:
        n = self.n
        cell = 2.0 * self.radius + self.skin
        idx = np.floor((self.pos - self.grid_lo) / cell).astype(np.int64)
        np.clip(idx[:, 0], 0, self.nx - 1, out=idx[:, 0])
        np.clip(idx[:, 1], 0, self.ny - 1, out=idx[:, 1])
        np.clip(idx[:, 2], 0, self.nz - 1, out=idx[:, 2])
        cid = (idx[:, 0] * self.ny + idx[:, 1]) * self.nz + idx[:, 2]
        order = np.argsort(cid, kind="stable")
        scid = cid[order]

        ii_parts = []
        jj_parts = []
        for dx, dy, dz in _OFFSETS:
            tx = idx[:, 0] + dx
            ty = idx[:, 1] + dy
            tz = idx[:, 2] + dz
            valid = (
                (tx >= 0) & (tx < self.nx)
                & (ty >= 0) & (ty < self.ny)
                & (tz >= 0) & (tz < self.nz)
            )
            if not valid.any():
                continue
            tcid = (tx * self.ny + ty) * self.nz + tz
            tcid = np.where(valid, tcid, -1)
            starts = np.searchsorted(scid, tcid)
            ends = np.searchsorted(scid, tcid, side="right")
            counts = (ends - starts).astype(np.int64)
            total = int(counts.sum())
            if total == 0:
                continue
            i_arr = np.repeat(np.arange(n, dtype=np.int64), counts)
            csum = np.cumsum(counts)
            offs = np.arange(total, dtype=np.int64) - np.repeat(csum - counts, counts)
            j_arr = order[np.repeat(starts, counts) + offs]
            if dx == 0 and dy == 0 and dz == 0:
                keep = i_arr < j_arr
                i_arr = i_arr[keep]
                j_arr = j_arr[keep]
            if i_arr.size:
                ii_parts.append(i_arr)
                jj_parts.append(j_arr)

        if ii_parts:
            self.pair_i = np.concatenate(ii_parts)
            self.pair_j = np.concatenate(jj_parts)
        else:
            self.pair_i = np.zeros(0, dtype=np.int64)
            self.pair_j = np.zeros(0, dtype=np.int64)
        self.n_pair_ops = self.pair_i.size

    # ------------------------------------------------------ contact forces
    def _contact_forces(self):
        n = self.n
        force = np.zeros((n, 3), dtype=float)
        if self.pair_i.size == 0:
            return force
        dp = self.pos[self.pair_j] - self.pos[self.pair_i]
        d2 = np.einsum("ij,ij->i", dp, dp)
        cutoff = 2.0 * self.radius + self.skin
        active = d2 < cutoff * cutoff
        if not active.any():
            return force
        i_idx = self.pair_i[active]
        j_idx = self.pair_j[active]
        dp = dp[active]
        d = np.sqrt(d2[active])
        inv_d = 1.0 / np.maximum(d, 1e-12)
        nvec = dp * inv_d[:, None]
        delta = 2.0 * self.radius - d
        delta_eff = np.minimum(delta, self.max_overlap)

        vrel = self.vel[j_idx] - self.vel[i_idx]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        m_eff = 0.5 * self.mass[i_idx]
        c_n = 2.0 * self.zeta * np.sqrt(self.k * m_eff)
        f_n = np.minimum(self.k * delta_eff - c_n * vn, self.f_max)
        f_n = np.maximum(f_n, 0.0)

        vt = vrel - vn[:, None] * nvec
        vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        c_t = 2.0 * np.sqrt(self.k * m_eff)
        f_t = np.minimum(self.mu * f_n, c_t * vt_mag)
        vt_hat = vt / np.maximum(vt_mag, 1e-12)[:, None]

        f_on_i = -f_n[:, None] * nvec - f_t[:, None] * vt_hat
        for axis in range(3):
            force[:, axis] += np.bincount(
                i_idx, weights=f_on_i[:, axis], minlength=n
            )
            force[:, axis] -= np.bincount(
                j_idx, weights=f_on_i[:, axis], minlength=n
            )
        return force

    # ------------------------------------------------------ wall handling
    def _prof_radius(self, z_local):
        return np.interp(z_local, self.prof_z, self.prof_r)

    @staticmethod
    def _rot_y(theta: float):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _wall_pass(self, dt: float) -> None:
        """Non-penetration impulses + friction against the rotating glass."""
        r = self.radius
        n = self.n
        theta, omega = _flip_state(self.time)
        self.theta = theta
        self.omega = omega
        rot = self._rot_y(theta)
        rel = self.pos - self.center
        q = rel @ rot                      # body-frame coordinates (rows)
        omega_vec = np.array([0.0, omega, 0.0])
        v_wall = np.cross(np.broadcast_to(omega_vec, (n, 3)), rel)

        qz = q[:, 2]
        r_here = self._prof_radius(qz)
        pen_profile = r - (r_here - np.abs(q[:, 0]))
        pen_bottom = (self.prof_z_bot + r) - qz
        pen_top = qz - (self.prof_z_top - r)
        pen_slab = np.abs(self.pos[:, 1]) - (self.slab_half - r)
        inside_floor = (
            (np.abs(self.pos[:, 0]) < self.floor_hx)
            & (np.abs(self.pos[:, 1]) < self.floor_hy)
        )
        pen_floor = np.where(inside_floor, r - self.pos[:, 2], -1.0)

        sgnx = np.sign(q[:, 0])
        sgny = np.sign(self.pos[:, 1])
        normals_obj = (
            np.column_stack((-sgnx, np.zeros(n), np.zeros(n))),
            np.column_stack((np.zeros(n), np.zeros(n), np.ones(n))),
            np.column_stack((np.zeros(n), np.zeros(n), -np.ones(n))),
            np.column_stack((np.zeros(n), -sgny, np.zeros(n))),
            np.column_stack((np.zeros(n), np.zeros(n), np.ones(n))),
        )
        pens = (pen_profile, pen_bottom, pen_top, pen_slab, pen_floor)
        gaps = (
            (r_here - np.abs(q[:, 0])) - r,
            qz - (self.prof_z_bot + r),
            (self.prof_z_top - r) - qz,
            (self.slab_half - r) - np.abs(self.pos[:, 1]),
            self.pos[:, 2] - r,
        )
        self._wall_gaps = np.concatenate([np.asarray(g, dtype=float) for g in gaps])

        mass = self.mass
        ext = mass[:, None] * self.gravity
        for _ in range(2):
            for pen, n_obj in zip(pens, normals_obj):
                act = pen > 0.0
                if not act.any():
                    continue
                n_world = n_obj[act] @ rot.T
                vr = self.vel[act] - v_wall[act]
                vn = np.einsum("ij,ij->i", vr, n_world)
                bias = np.minimum(self.beta_wall * pen[act] / dt, self.vbias_max)
                dvn = np.where(vn < bias, bias - vn, 0.0)
                if dvn.any():
                    self.vel[act] += dvn[:, None] * n_world
                    jn = mass[act] * dvn
                    ext[act] += (jn[:, None] * n_world) / dt
                elif not np.any(vn < bias):
                    continue
                jn = mass[act] * dvn
                vt = vr - vn[:, None] * n_world
                vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                jt = np.minimum(self.mu_wall * jn, mass[act] * vt_mag)
                vt_hat = vt / np.maximum(vt_mag, 1e-12)[:, None]
                dv_t = -(jt / mass[act])[:, None] * vt_hat
                self.vel[act] += dv_t
                ext[act] += (mass[act][:, None] * dv_t) / dt
        self._ext_force = ext

    # ------------------------------------------------------------ stepping
    def _step(self, vmax: float | None = None) -> None:
        dt = self.dt
        if self.rebuild_countdown <= 0:
            self._rebuild_pairs()
            self.rebuild_countdown = self.rebuild_every
        self.rebuild_countdown -= 1

        force = self._contact_forces() + self.mass[:, None] * self.gravity
        self.vel += (force / self.mass[:, None]) * dt
        if vmax is not None:
            speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
            scale = np.minimum(1.0, vmax / np.maximum(speed, 1e-12))
            self.vel *= scale[:, None]
        self._wall_pass(dt)
        self.pos += self.vel * dt
        self._step_index += 1
        self.time = self._step_index * dt

    # ------------------------------------------------------------- frames
    def _report_gaps(self) -> np.ndarray:
        r = self.radius
        parts = []
        if self.pair_i.size:
            dp = self.pos[self.pair_j] - self.pos[self.pair_i]
            d = np.sqrt(np.einsum("ij,ij->i", dp, dp))
            gap = d - 2.0 * r
            near = gap < 0.5 * r
            if near.any():
                parts.append(gap[near])
        parts.append(np.asarray(self._wall_gaps, dtype=float))
        out = np.concatenate(parts) if parts else np.zeros(1, dtype=float)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _frame(self, time: float) -> StateFrame:
        gaps = self._report_gaps()
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "particle_radius": self.radius_attr.copy(),
                "mass_quantity": self.mass.copy(),
                "volume_quantity": self.volume.copy(),
            },
            fields={
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self._ext_force.copy(),
            },
            scalars={
                "contact_length_scale": float(2.0 * self.radius),
                "particle_radius": float(self.radius),
                "particle_count": float(self.n),
                "bulk_particle_volume": float(self.n * self.grain_volume),
                "hourglass_rotation_angle": float(self.theta),
                "hourglass_rotation_rate": float(self.omega),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self._step_index < target:
            self._step()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        return {
            "particle_count": int(self.n),
            "total_mass": float(self.mass.sum()),
            "bulk_particle_volume": float(self.n * self.grain_volume),
            "steps": int(self._step_index),
            "final_time": float(self.time),
            "max_speed": float(speed.max()) if self.n else 0.0,
            "neighbour_pairs": int(self.n_pair_ops),
            "rotation_angle": float(self.theta),
        }
