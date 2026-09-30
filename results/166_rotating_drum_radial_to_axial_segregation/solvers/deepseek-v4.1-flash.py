"""Discrete-element solver for a long horizontal rotating drum.

Physical model
--------------
The scene is a horizontal cylindrical drum (axis +x, radius R, length L) that
spins about its own axis with angular velocity ``omega`` inside a uniform
gravitational field g = (0, 0, -9.81) m/s^2.  The drum is partially filled with
two species of spherical, frictional grains (identical density, different
radius) that start well mixed.

Each grain is treated as a soft sphere with:
  * normal contact: linear spring-dashpot  F_n = k*delta + c*v_n   (F_n >= 0)
  * tangential contact: Mindlin-like tangential spring with Coulomb slip cap
    |F_t| <= mu*F_n, history tracked per persistent contact (grain-grain) and
    per particle (grain-wall, one wall contact point per particle).
The drum wall is kinematic: every wall material point moves with
v = omega x (p - axis_point), so both the cylindrical shell and the flat end
caps drag the grains.  Gravity is the only external (body) force.

Time integration is symplectic (semi-implicit) Euler at a fixed step
dt = sample_dt / 200, small enough to resolve the stiffest contact mode.  The
contact law stiffness is ramped smoothly over the first 50 ms so that the
small geometric overlaps present in the frozen initial packing relax gently
instead of being released as an impulsive kick.

Nothing here is scripted or precomputed: every frame is the result of the
actual time-stepping of this force model.
"""
from __future__ import annotations

import numpy as np

try:  # fast neighbour search; deterministic fallback below if unavailable.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# Frozen scene constants (used only as fall-backs if the scene does not expose
# them through its configuration dictionary).
# ----------------------------------------------------------------------------
_FALLBACK = {
    "radius": 0.92,
    "length": 4.8,
    "center_height": 1.05,
    "angular_velocity_rad_s": 5.5,
    "spacing": 0.115,
    "particle_density": 1600.0,
    "species_count": 2,
    "field_grid_shape": (40, 18, 18),
    "sample_dt": 0.04,
    "duration": 14.0,
}

_GRAVITY = np.array([0.0, 0.0, -9.81])

# Contact model parameters (physical, species independent).
_STIFFNESS = 6.0e5          # normal linear spring constant [N/m]
_DAMPING_RATIO = 0.25       # normal dashpot ratio (restitution ~ 0.45)
_TANGENT_RATIO = 0.3        # k_t / k_n
_MU_GRAIN = 0.50            # grain-grain friction coefficient
_MU_WALL = 0.45             # grain-wall friction coefficient
_OMEGA_VEC = np.zeros(3)    # filled in initialize

_MAX_SPEED = 25.0           # hard velocity clamp [m/s], safety only
_MAX_PENETRATION = 3.0e-3   # confinement projection threshold [m]
_NEAR_CONTACT_GAP = 5.0e-3  # reported near-contact band [m]


def _pairwise_within(pos: np.ndarray, cutoff: float):
    """Return an (M, 2) int64 array of candidate index pairs within ``cutoff``."""
    n = pos.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    if cKDTree is not None:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(float(cutoff), output_type="ndarray")
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.asarray(pairs, dtype=np.int64)
    else:  # pragma: no cover - deterministic chunked fallback
        out = []
        step = 512
        for start in range(0, n, step):
            block = pos[start:start + step]
            d = block[:, None, :] - pos[None, :, :]
            r2 = np.einsum("ijk,ijk->ij", d, d)
            ii, jj = np.nonzero(r2 <= cutoff * cutoff)
            keep = (start + ii) < jj
            out.append(np.stack([start + ii[keep], jj[keep]], axis=1))
        pairs = np.concatenate(out, axis=0) if out else np.zeros((0, 2), np.int64)
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    key = lo.astype(np.int64) * np.int64(n) + hi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    keep = np.ones(key.shape[0], dtype=bool)
    if key.shape[0] > 1:
        keep[1:] = key[1:] != key[:-1]      # unique, deterministic ordering
    return np.stack([lo[order][keep], hi[order][keep]], axis=1)


class BenchmarkSolver:
    """DEM solver for the radial-to-axial granular segregation scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        cfg = getattr(scene, "config", None)
        cfg = dict(cfg) if isinstance(cfg, dict) else {}
        c = dict(cfg.get("centrifuge", {})) if isinstance(cfg.get("centrifuge"), dict) else {}
        p = {k: c.get(k, v) for k, v in _FALLBACK.items()}

        self.radius_drum = float(p["radius"])
        self.length = float(p["length"])
        self.center = np.array([0.0, 0.0, float(p["center_height"])], dtype=float)
        self.omega = float(p["angular_velocity_rad_s"])
        self.spacing = float(p["spacing"])
        self.species_count = int(p["species_count"])
        self.grid_shape = tuple(int(v) for v in p["field_grid_shape"])
        sample_dt = float(cfg.get("sample_dt", c.get("sample_dt", _FALLBACK["sample_dt"])))

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.n = int(self.pos.shape[0])

        attrs = dict(frame.attributes or {})
        if "species_id" in attrs:
            self.species = np.asarray(attrs["species_id"]).astype(np.int64).ravel()
        else:
            self.species = np.zeros(self.n, dtype=np.int64)
        if "particle_radius" in attrs:
            self.radius = np.asarray(attrs["particle_radius"], dtype=float).ravel()
        else:
            self.radius = np.full(self.n, 0.5 * self.spacing)
        self.species = np.clip(self.species, 0, max(self.species_count - 1, 0))
        self.radius = np.maximum(self.radius, 1.0e-4)
        self.inv_mass = 1.0 / np.maximum(self.mass, 1.0e-12)

        # integration parameters
        self.dt = sample_dt / 200.0
        self.time = 0.0
        self.step_index = 0
        self.ramp_time = 0.05
        self.ramp_floor = 0.1

        # contact search
        self.r_max = float(self.radius.max())
        self.cutoff = 2.0 * self.r_max + 8.0e-3

        # contact history
        self._prev_keys = np.zeros(0, dtype=np.int64)
        self._prev_xi = np.zeros((0, 3), dtype=float)
        self._xi_cyl = np.zeros((self.n, 3), dtype=float)
        self._xi_cap = np.zeros((self.n, 3), dtype=float)

        # diagnostics accumulators
        self._max_pen_seen = 0.0
        self._kinetic = 0.0
        self._contact_count_seen = 0
        self._radial_index = 0.0
        self._axial_variance = 0.0

    # ------------------------------------------------------------------
    # force evaluation
    # ------------------------------------------------------------------
    def _stiffness_scale(self) -> float:
        if self.time >= self.ramp_time:
            return 1.0
        frac = max(self.time, 0.0) / self.ramp_time
        return self.ramp_floor + (1.0 - self.ramp_floor) * frac

    def _grain_forces(self, dt: float, k_scale: float):
        """Particle-particle contacts. Returns acceleration contributions."""
        n = self.n
        pairs = _pairwise_within(self.pos, self.cutoff)
        if pairs.shape[0] == 0:
            self._prev_keys = np.zeros(0, dtype=np.int64)
            self._prev_xi = np.zeros((0, 3), dtype=float)
            return np.zeros((n, 3), dtype=float), 0

        i = pairs[:, 0]
        j = pairs[:, 1]
        key = i * np.int64(n) + j

        dvec = self.pos[j] - self.pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
        dist = np.maximum(dist, 1.0e-12)
        nvec = dvec / dist[:, None]

        sum_r = self.radius[i] + self.radius[j]
        delta = sum_r - dist

        # --- persistent tangential history (world-frame tangential spring) ---
        xi = np.zeros((i.shape[0], 3), dtype=float)
        if self._prev_keys.shape[0]:
            pos = np.searchsorted(self._prev_keys, key)
            pos_c = np.minimum(pos, self._prev_keys.shape[0] - 1)
            hit = self._prev_keys[pos_c] == key
            if np.any(hit):
                xi[hit] = self._prev_xi[pos_c[hit]]

        vrel = self.vel[j] - self.vel[i]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        delta_p = np.maximum(delta, 0.0)

        m_red = (self.mass[i] * self.mass[j]) / np.maximum(self.mass[i] + self.mass[j], 1e-12)
        k_n = _STIFFNESS * k_scale
        # linear dashpot: c = 2 * zeta * sqrt(k * m_reduced)
        c_n = 2.0 * _DAMPING_RATIO * np.sqrt(k_n * np.maximum(m_red, 1e-12))

        f_n = k_n * delta_p + c_n * vn
        f_n = np.where(delta > 0.0, np.maximum(f_n, 0.0), 0.0)

        vt = vrel - vn[:, None] * nvec
        xi = xi + vt * dt
        xi -= np.einsum("ij,ij->i", xi, nvec)[:, None] * nvec
        k_t = _TANGENT_RATIO * k_n
        ft = -k_t[:, None] * xi
        ft_mag = np.sqrt(np.einsum("ij,ij->i", ft, ft))
        cap = _MU_GRAIN * f_n
        over = ft_mag > cap
        safe_mag = np.maximum(ft_mag, 1.0e-12)
        shrink = np.where(over, cap / safe_mag, 1.0)
        ft = ft * shrink[:, None]
        xi = xi * shrink[:, None]

        self._prev_keys = key
        self._prev_xi = xi

        fpair = f_n[:, None] * nvec + ft
        acc = np.zeros((n, 3), dtype=float)
        for axis in range(3):
            w = fpair[:, axis]
            acc[:, axis] = (
                np.bincount(j, weights=w, minlength=n)
                - np.bincount(i, weights=w, minlength=n)
            ) * self.inv_mass
        return acc, int(np.count_nonzero(f_n > 0.0))

    def _wall_forces(self, dt: float, k_scale: float):
        """Cylindrical shell + flat end caps (kinematic rotation)."""
        n = self.n
        pos = self.pos
        vel = self.vel
        omega_vec = np.array([self.omega, 0.0, 0.0])
        k_n = _STIFFNESS * k_scale
        c_n = 2.0 * _DAMPING_RATIO * np.sqrt(k_n * np.maximum(self.mass, 1e-12))
        k_t = _TANGENT_RATIO * k_n

        acc = np.zeros((n, 3), dtype=float)
        contacts = 0

        # ---- cylindrical shell ----
        q = pos[:, 1:3] - self.center[1:3]
        d = np.sqrt(np.einsum("ij,ij->i", q, q))
        d = np.maximum(d, 1.0e-12)
        q3 = np.zeros((n, 3), dtype=float)
        q3[:, 1:3] = q
        nrm = -q3 / d[:, None]                      # inward normal (wall -> grain)
        delta = self.radius_drum - d - self.radius  # >0 : penetration
        wall_pt = q3 * (self.radius_drum / d)[:, None]
        v_wall = np.cross(np.broadcast_to(omega_vec, (n, 3)), wall_pt)
        vrel = vel - v_wall
        vn = np.einsum("ij,ij->i", vrel, nrm)
        f_n = np.where(delta > 0.0, np.maximum(k_n * np.maximum(delta, 0.0) + c_n * vn, 0.0), 0.0)
        contacts += int(np.count_nonzero(f_n > 0.0))

        vt = vrel - vn[:, None] * nrm
        act = f_n > 0.0
        self._xi_cyl = np.where(act[:, None], self._xi_cyl + vt * dt, 0.0)
        self._xi_cyl -= np.einsum("ij,ij->i", self._xi_cyl, nrm)[:, None] * nrm
        self._xi_cyl = np.where(act[:, None], self._xi_cyl, 0.0)
        f_t = -k_t[:, None] * self._xi_cyl
        mag = np.sqrt(np.einsum("ij,ij->i", f_t, f_t))
        cap = _MU_WALL * f_n
        safe = np.maximum(mag, 1e-12)
        shrink = np.where(mag > cap, cap / safe, 1.0)
        f_t = f_t * shrink[:, None]
        self._xi_cyl = self._xi_cyl * shrink[:, None]
        acc += (f_n[:, None] * nrm + f_t) * self.inv_mass[:, None]

        # ---- end caps ----
        ax = pos[:, 0]
        half = 0.5 * self.length
        sgn = np.where(ax >= 0.0, 1.0, -1.0)
        delta_c = np.abs(ax) + self.radius - half
        nrm_c = np.zeros((n, 3), dtype=float)
        nrm_c[:, 0] = -sgn
        cap_pt = np.zeros((n, 3), dtype=float)
        cap_pt[:, 0] = sgn * half
        cap_pt[:, 1:3] = pos[:, 1:3]
        v_wall_c = np.cross(np.broadcast_to(omega_vec, (n, 3)), cap_pt - self.center)
        vrel_c = vel - v_wall_c
        vn_c = np.einsum("ij,ij->i", vrel_c, nrm_c)
        f_n_c = np.where(delta_c > 0.0, np.maximum(k_n * np.maximum(delta_c, 0.0) + c_n * vn_c, 0.0), 0.0)
        contacts += int(np.count_nonzero(f_n_c > 0.0))

        vt_c = vrel_c - vn_c[:, None] * nrm_c
        act_c = f_n_c > 0.0
        self._xi_cap = np.where(act_c[:, None], self._xi_cap + vt_c * dt, 0.0)
        self._xi_cap -= np.einsum("ij,ij->i", self._xi_cap, nrm_c)[:, None] * nrm_c
        self._xi_cap = np.where(act_c[:, None], self._xi_cap, 0.0)
        f_t_c = -k_t[:, None] * self._xi_cap
        mag_c = np.sqrt(np.einsum("ij,ij->i", f_t_c, f_t_c))
        cap_c = _MU_WALL * f_n_c
        safe_c = np.maximum(mag_c, 1e-12)
        shrink_c = np.where(mag_c > cap_c, cap_c / safe_c, 1.0)
        f_t_c = f_t_c * shrink_c[:, None]
        self._xi_cap = self._xi_cap * shrink_c[:, None]
        acc += (f_n_c[:, None] * nrm_c + f_t_c) * self.inv_mass[:, None]

        return acc, contacts

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _confine(self) -> None:
        """Safety projection: never let a grain leave the container."""
        q = self.pos[:, 1:3] - self.center[1:3]
        d = np.sqrt(np.einsum("ij,ij->i", q, q))
        d = np.maximum(d, 1.0e-12)
        pen = d + self.radius - self.radius_drum
        bad = pen > _MAX_PENETRATION
        if np.any(bad):
            scale = (self.radius_drum - self.radius[bad] - 1.0e-6) / d[bad]
            self.pos[np.nonzero(bad)[0], 1:3] = self.center[1:3] + q[bad] * scale[:, None]
            qb = q[bad] / d[bad, None]
            v_out = np.einsum("ij,ij->i", self.vel[bad, 1:3], qb)
            v_out = np.maximum(v_out, 0.0)
            self.vel[np.nonzero(bad)[0], 1:3] -= v_out[:, None] * qb

        half = 0.5 * self.length
        xpen = np.abs(self.pos[:, 0]) + self.radius - half
        badx = xpen > _MAX_PENETRATION
        if np.any(badx):
            sgn = np.sign(self.pos[badx, 0])
            sgn = np.where(sgn == 0.0, 1.0, sgn)
            self.pos[badx, 0] = sgn * (half - self.radius[badx] - 1.0e-6)
            vx = self.vel[badx, 0] * sgn
            vx = np.maximum(vx, 0.0)
            self.vel[badx, 0] -= vx * sgn

    def _step(self) -> None:
        dt = self.dt
        k_scale = self._stiffness_scale()
        acc, n_c = self._grain_forces(dt, k_scale)
        accw, n_cw = self._wall_forces(dt, k_scale)
        acc += accw
        acc += _GRAVITY[None, :]
        self.vel += acc * dt
        speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        fast = speed > _MAX_SPEED
        if np.any(fast):
            self.vel[fast] *= (_MAX_SPEED / speed[fast])[:, None]
        self.pos += self.vel * dt
        self._confine()
        self.time += dt
        self.step_index += 1
        self._contact_count_seen = n_c + n_cw

    def _advance_to_time(self, target: float) -> None:
        dt = self.dt
        n_target = int(round(target / dt))
        guard = 0
        while self.step_index < n_target:
            self._step()
            guard += 1
            if guard > 5_000_000:  # pragma: no cover - unreachable safety net
                break

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _contact_report(self):
        """Signed gaps of active and near contacts (negative = penetration)."""
        pos = self.pos
        pairs = _pairwise_within(pos, self.cutoff + _NEAR_CONTACT_GAP)
        gaps = []
        max_pen = 0.0
        count = 0
        if pairs.shape[0]:
            dvec = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            gap_pp = dist - (self.radius[pairs[:, 0]] + self.radius[pairs[:, 1]])
            keep = gap_pp <= _NEAR_CONTACT_GAP
            gaps.append(gap_pp[keep])
            count += int(np.count_nonzero(gap_pp < 0.0))
            if keep.any():
                max_pen = max(max_pen, float(-gap_pp[keep].min()))
        q = pos[:, 1:3] - self.center[1:3]
        d = np.sqrt(np.einsum("ij,ij->i", q, q))
        gap_cyl = self.radius_drum - d - self.radius
        keep_c = gap_cyl <= _NEAR_CONTACT_GAP
        gaps.append(gap_cyl[keep_c])
        count += int(np.count_nonzero(gap_cyl < 0.0))
        if keep_c.any():
            max_pen = max(max_pen, float(-gap_cyl[keep_c].min()))
        half = 0.5 * self.length
        gap_cap = half - np.abs(pos[:, 0]) - self.radius
        keep_e = gap_cap <= _NEAR_CONTACT_GAP
        gaps.append(gap_cap[keep_e])
        count += int(np.count_nonzero(gap_cap < 0.0))
        if keep_e.any():
            max_pen = max(max_pen, float(-gap_cap[keep_e].min()))
        values = np.concatenate(gaps) if gaps else np.zeros(0, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            values = np.array([_NEAR_CONTACT_GAP], dtype=float)
        return values, count, max_pen

    def _velocity_field(self):
        """Coarse diagnostic velocity grid (same shape as the frozen scene)."""
        shape = self.grid_shape
        nx, ny, nz = shape
        half = 0.5 * self.length
        x = np.clip((self.pos[:, 0] + half) / (2.0 * half) * nx, 0, nx - 1e-9)
        y = np.clip((self.pos[:, 1] + self.radius_drum) / (2.0 * self.radius_drum) * ny, 0, ny - 1e-9)
        z = np.clip((self.pos[:, 2] - self.center[2] + self.radius_drum) / (2.0 * self.radius_drum) * nz,
                    0, nz - 1e-9)
        ix = x.astype(np.int64)
        iy = y.astype(np.int64)
        iz = z.astype(np.int64)
        flat = (ix * ny + iy) * nz + iz
        size = nx * ny * nz
        cnt = np.bincount(flat, minlength=size).astype(float)
        out = np.zeros((size, 3), dtype=float)
        for axis in range(3):
            out[:, axis] = np.bincount(flat, weights=self.vel[:, axis], minlength=size)
        nz_mask = cnt > 0
        out[nz_mask] /= cnt[nz_mask, None]
        return out.reshape(shape + (3,)), nz_mask.reshape(shape)

    def _build_frame(self, time: float) -> StateFrame:
        n = self.n
        k = self.species_count
        mass_quantity = np.zeros((n, k), dtype=float)
        mass_quantity[np.arange(n), self.species] = self.mass
        external_force = self.mass[:, None] * _GRAVITY[None, :]
        gaps, contact_count, max_pen = self._contact_report()
        vfield, active = self._velocity_field()
        self._max_pen_seen = max(self._max_pen_seen, max_pen)
        speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        self._kinetic = float(0.5 * np.sum(self.mass * speed ** 2))
        radial = np.sqrt(np.einsum("ij,ij->i", self.pos[:, 1:3] - self.center[1:3],
                                   self.pos[:, 1:3] - self.center[1:3]))
        mean_radius = np.zeros(k, dtype=float)
        for sid in range(k):
            sel = self.species == sid
            if np.any(sel):
                mean_radius[sid] = float(self.radius[sel].mean())
        small = self.species == int(np.argmin(mean_radius))
        if np.any(small) and np.any(~small):
            self._radial_index = float(np.mean(radial[small]) - np.mean(radial[~small]))
            nb = 20
            edges = np.linspace(-0.5 * self.length, 0.5 * self.length, nb + 1)
            idx = np.clip(np.digitize(self.pos[:, 0], edges) - 1, 0, nb - 1)
            frac = np.zeros(nb)
            cnt = np.zeros(nb)
            np.add.at(frac, idx[small], 1.0)
            np.add.at(cnt, idx, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                comp = np.where(cnt > 0, frac / np.maximum(cnt, 1.0), 0.0)
            self._axial_variance = float(np.var(comp))

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "species_id": self.species.astype(np.int32),
                "particle_radius": self.radius.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(k, dtype=float),
                "external_force": external_force,
                "velocity_field": vfield,
                "active_mask": active,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(2.0 * self.radius_drum),
                "kinetic_energy": self._kinetic,
                "max_penetration": float(self._max_pen_seen),
                "contact_count": float(contact_count),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = max(float(time), 0.0)
        if target > self.time + 1.0e-12:
            self._advance_to_time(target)
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        return {
            "solver": "soft_sphere_dem_symplectic_euler",
            "dt": float(self.dt),
            "steps": int(self.step_index),
            "particle_count": int(self.n),
            "species_count": int(self.species_count),
            "final_time": float(self.time),
            "kinetic_energy": float(0.5 * np.sum(self.mass * speed ** 2)),
            "max_speed": float(speed.max()) if self.n else 0.0,
            "max_penetration_seen": float(self._max_pen_seen),
            "radial_segregation_index": float(self._radial_index),
            "axial_composition_variance": float(self._axial_variance),
            "mass_total": float(self.mass.sum()),
        }
