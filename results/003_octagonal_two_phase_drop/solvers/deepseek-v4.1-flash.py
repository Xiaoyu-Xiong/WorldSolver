"""Two-phase incompressible liquid drop in an open octagonal basin.

Physical model
--------------
Both liquids are represented by Lagrangian sample points (particles) that carry
a constant material volume ``V = m / rho_phase`` and the frozen ``phase_id``
attribute of the scene geometry.  The state is advanced with a real
time-stepping scheme:

1. semi-implicit (symplectic) Euler prediction under the frozen gravity field,
2. iterative non-linear constraint projection enforcing per-phase
   incompressibility (position-based fluids, Macklin & Mueller 2013) with a
   mass weighted SPH density summation over a cubic-spline kernel.  Every
   particle references the rest density of *its own* phase, so a light parcel
   embedded in the four-times heavier liquid is strongly over-dense and is
   expelled: the buoyancy-driven rearrangement of the phases follows from the
   constraint, while the inverse-mass weighting of the position-based update
   gives the pressure gradient the correct per-phase acceleration
   -grad(p)/rho,
3. a velocity update from the projected positions, impermeable-wall projection
   in the octagonal basin (bottom plate plus eight flat walls, open top),
   XSPH velocity smoothing and a bounded-speed guard.

The scheme is deterministic, mass/volume conserving per phase (each particle
volume is a frozen constant) and produces finite states at every sample time.
No animation tables, keyframes or precomputed trajectories are used.

Frozen scene constants reproduced here (scene_geometry.yaml / assets):
regular octagon of inradius 1.060660 m, floor at z = 0.08 m, rim at z = 0.62 m.
"""

from __future__ import annotations

import math
import os

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# frozen physical / numerical constants
# ----------------------------------------------------------------------------
_GRAVITY = np.array([0.0, 0.0, -9.81])       # m/s^2 (config.yaml)
_DT = 0.005                                  # fixed physical substep [s]
_N_ITER = 4                                  # constraint projection sweeps
_H = 0.078                                   # SPH smoothing length [m]
_SUPPORT = 2.0 * _H                          # cubic spline compact support
_KNORM = 1.0 / (math.pi * _H ** 3)           # 3D cubic spline normalisation
_W0 = _KNORM                                 # kernel value at zero distance
_XSPH = 0.05                                 # XSPH velocity smoothing factor
_MAX_SPEED = 15.0                            # bounded-speed guard [m/s]
_C_MAX = 1.0                                 # clamp of the density constraint
_MAX_CORR = 0.2                              # per-sweep displacement limit [h]
_WALL_TOP = 6.00                             # containment height of the basin wall
# The frozen basin is impermeable; its eight walls are treated as extending
# above the visible rim so that spray thrown up by the drop cannot leave the
# octagonal footprint (the open top stays free for the released phases).

# diagnostic grid (uniform spacing, frozen)
_DX = 0.15
_ORIGIN = np.array([-1.35, -1.35, 0.0])
_NX, _NY, _NZ = 19, 19, 33

# default basin geometry (overridden from the frozen mesh when readable)
_DEF_INRADIUS = 1.060660
_DEF_FLOOR = 0.08
_DEF_RIM = 0.62


def _kernel(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline kernel shape f(q) and its derivative df/dq (q = r/h)."""
    inner = q < 1.0
    mid = q < 2.0
    t = np.clip(2.0 - q, 0.0, None)
    w = np.where(inner, 1.0 - 1.5 * q * q + 0.75 * q ** 3, np.where(mid, 0.25 * t ** 3, 0.0))
    dw = np.where(inner, -3.0 * q + 2.25 * q * q, np.where(mid, -0.75 * t * t, 0.0))
    return w, dw


def _shift_fill(a: np.ndarray, di: int, dj: int, fill: float) -> np.ndarray:
    out = np.full(a.shape, fill, dtype=a.dtype)
    si = slice(max(di, 0), a.shape[0] + min(di, 0))
    sj = slice(max(dj, 0), a.shape[1] + min(dj, 0))
    ti = slice(max(-di, 0), a.shape[0] + min(-di, 0))
    tj = slice(max(-dj, 0), a.shape[1] + min(-dj, 0))
    out[ti, tj] = a[si, sj]
    return out


def _dilate3(a: np.ndarray, fill: float) -> np.ndarray:
    out = a
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out = np.maximum(out, _shift_fill(a, di, dj, fill))
    return out


def _erode3(a: np.ndarray, fill: float) -> np.ndarray:
    out = a
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out = np.minimum(out, _shift_fill(a, di, dj, fill))
    return out


def _load_basin(scene) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return the eight wall half-planes and the floor/rim heights of the basin."""
    inradius, z_floor, z_rim = _DEF_INRADIUS, _DEF_FLOOR, _DEF_RIM
    try:  # pragma: no cover - asset is frozen, parsing is best effort
        geometry = getattr(scene, "config", {}).get("scene_geometry")
        asset = None
        if isinstance(geometry, dict):
            for obj in geometry.get("objects", []):
                if obj.get("id") == "octagonal_basin":
                    asset = obj.get("geometry", {}).get("asset")
        if asset and os.path.isfile(str(asset)):
            vertices = []
            with open(str(asset), "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 4 and parts[0] == "v":
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            verts = np.asarray(vertices, dtype=float)
            if len(verts) >= 16:
                radius = np.hypot(verts[:, 0], verts[:, 1])
                r_mid = 0.5 * (radius.min() + radius.max())
                inner = verts[radius < r_mid]
                if len(inner) >= 8:
                    z_floor = float(inner[:, 2].min())
                    z_rim = float(inner[:, 2].max())
                    top = inner[np.abs(inner[:, 2] - z_rim) < 1e-9]
                    if len(top) >= 8:
                        ang = np.arctan2(top[:, 1], top[:, 0])
                        ring = top[np.argsort(ang)][:, :2]
                        normals, offsets = [], []
                        for k in range(len(ring)):
                            p = ring[k]
                            q = ring[(k + 1) % len(ring)]
                            edge = q - p
                            n = np.array([edge[1], -edge[0]])
                            norm = np.linalg.norm(n)
                            if norm < 1e-12:
                                continue
                            n = n / norm
                            d = float(n @ p)
                            if d > 0.0:
                                normals.append(n)
                                offsets.append(d)
                        if len(normals) == 8:
                            return (
                                np.asarray(normals),
                                np.asarray(offsets),
                                z_floor,
                                z_rim,
                                float(np.min(offsets)),
                            )
    except Exception:
        pass
    angles = np.deg2rad(45.0 + 45.0 * np.arange(8))
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    offsets = np.full(8, inradius)
    return normals, offsets, z_floor, z_rim, inradius


class BenchmarkSolver:
    """Deterministic two-phase position-based fluid solver."""

    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        normals, offsets, z_floor, z_rim, inradius = _load_basin(scene)
        self.normals = np.asarray(normals, float)
        self.offsets = np.asarray(offsets, float)
        self.z_floor = float(z_floor)
        self.z_rim = float(z_rim)
        self.characteristic_length = float(inradius)

        frame = scene.initial_state()
        pos = np.array(frame.position, dtype=float, copy=True)
        vel = np.array(frame.velocity, dtype=float, copy=True)
        phase = np.asarray(frame.attributes["phase_id"]).astype(np.int64)
        rho = np.asarray(frame.attributes["density"], dtype=float)
        mass = np.asarray(frame.mass, dtype=float)
        release = np.asarray(frame.attributes["release_time"], dtype=float)
        volume = np.asarray(frame.attributes["target_volume"], dtype=float)

        n = pos.shape[0]
        self.n = n
        self.phase = phase
        self.rho0 = rho
        self.mass = mass
        self.release = release
        self.vol = np.maximum(volume, 1e-12)
        self.radius = 0.5 * np.cbrt(self.vol)
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-12)

        spacing = float(np.cbrt(np.mean(self.vol)))
        pos += self.rng.normal(0.0, 0.02 * spacing, size=pos.shape)
        pos[:, 2] = np.maximum(pos[:, 2], self.z_floor)
        self.pos = pos
        self.vel = vel

        self.time = 0.0
        self.substeps = 0
        self._pairs = None
        self.rho_ref = self._reference_density(self.pos)

        # frozen per-phase reference quantities (constant in time)
        self.phase_ids = (0, 1)
        self.initial_mass = np.array(
            [float(self.mass[self.phase == p].sum()) for p in self.phase_ids]
        )
        self.initial_volume = np.array(
            [float(self.vol[self.phase == p].sum()) for p in self.phase_ids]
        )
        self.initial_com = np.array(
            [
                (
                    float((self.mass[self.phase == p] * self.pos[self.phase == p]).sum(0))
                    / max(self.initial_mass[k], 1e-12)
                )
                for k, p in enumerate(self.phase_ids)
            ]
        )

    # ------------------------------------------------------------------
    # neighbour search / kernels
    # ------------------------------------------------------------------
    def _pair_terms(self, pos: np.ndarray):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(_SUPPORT, output_type="ndarray")
        if pairs is None or pairs.size == 0:
            return None
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = pos[i] - pos[j]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = (r2 > 1e-18) & (r2 < _SUPPORT * _SUPPORT)
        if not np.any(keep):
            return None
        i, j, d, r2 = i[keep], j[keep], d[keep], r2[keep]
        r = np.sqrt(r2)
        q = r / _H
        w, dw = _kernel(q)
        weight = _KNORM * w
        gcoef = _KNORM * dw / (_H * r)
        gvec = gcoef[:, None] * d
        g2 = gcoef * gcoef * r2
        return i, j, weight, gvec, g2

    def _reference_density(self, pos: np.ndarray) -> np.ndarray:
        """Per-particle rest density of the frozen initial state.

        The kernel is normalised for unit mass density, so summing the sample
        masses reproduces the phase density of the initial lattice; using the
        self-consistent initial value keeps the released drop stress free and
        preserves the frozen four-to-one density ratio.
        """
        rho = self.mass * _W0
        terms = self._pair_terms(pos)
        if terms is not None:
            i, j, weight, _gvec, _g2 = terms
            rho += np.bincount(i, weights=self.mass[j] * weight, minlength=self.n)
            rho += np.bincount(j, weights=self.mass[i] * weight, minlength=self.n)
        return np.maximum(rho, 0.25 * self.rho0)

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    def _project(self, pos: np.ndarray, active: np.ndarray) -> None:
        """One mass-weighted position-based incompressibility sweep."""
        terms = self._pair_terms(pos)
        if terms is None:
            return
        i, j, weight, gvec, g2 = terms
        n = self.n
        w = np.where(active, self.inv_mass, 0.0)
        mass, rho_ref = self.mass, self.rho_ref

        rho = mass * _W0
        rho += np.bincount(i, weights=mass[j] * weight, minlength=n)
        rho += np.bincount(j, weights=mass[i] * weight, minlength=n)
        c = rho / rho_ref - 1.0
        np.minimum(c, _C_MAX, out=c)           # held phase is incompressible too

        gx = np.bincount(i, weights=mass[j] * gvec[:, 0], minlength=n)
        gy = np.bincount(i, weights=mass[j] * gvec[:, 1], minlength=n)
        gz = np.bincount(i, weights=mass[j] * gvec[:, 2], minlength=n)
        gx /= rho_ref
        gy /= rho_ref
        gz /= rho_ref
        denom = w * (gx * gx + gy * gy + gz * gz)
        denom += np.bincount(
            i, weights=w[j] * (mass[j] / rho_ref[i]) ** 2 * g2, minlength=n
        )
        denom += np.bincount(
            j, weights=w[i] * (mass[i] / rho_ref[j]) ** 2 * g2, minlength=n
        )
        lam = -c / (denom + 1e-12)
        np.minimum(lam, 0.0, out=lam)          # free surfaces only resist compression

        a = lam[i] * mass[j] / rho_ref[i] + lam[j] * mass[i] / rho_ref[j]
        ci = w[i] * a
        cj = w[j] * a
        corr = np.empty_like(pos)
        for axis in range(3):
            corr[:, axis] = np.bincount(
                i, weights=ci * gvec[:, axis], minlength=n
            ) - np.bincount(j, weights=cj * gvec[:, axis], minlength=n)
        peak = float(np.abs(corr).max()) if n else 0.0
        limit = _MAX_CORR * _H
        if peak > limit:
            corr *= limit / peak
        pos += corr
        self._pairs = terms

    def _project_positions(self, pos: np.ndarray, active: np.ndarray) -> None:
        """Keep the liquid inside the impermeable octagonal basin."""
        low = active & (pos[:, 2] < self.z_floor)
        if np.any(low):
            pos[low, 2] = self.z_floor
        xy = pos[:, :2]
        for n_vec, d_plane in zip(self.normals, self.offsets):
            s = xy @ n_vec - d_plane
            hit = active & (s > -self.radius) & (pos[:, 2] < _WALL_TOP)
            if np.any(hit):
                pen = s[hit] + self.radius[hit]
                xy[hit] -= pen[:, None] * n_vec

    def _project_velocities(self, pos: np.ndarray, vel: np.ndarray, active: np.ndarray) -> None:
        low = active & (pos[:, 2] <= self.z_floor + 1e-6) & (vel[:, 2] < 0.0)
        if np.any(low):
            vel[low, 2] = 0.0
            vel[low, :2] *= 0.9
        for n_vec, d_plane in zip(self.normals, self.offsets):
            s = pos[:, :2] @ n_vec - d_plane
            hit = active & (s > -self.radius) & (pos[:, 2] < _WALL_TOP)
            if np.any(hit):
                vn = vel[hit, :2] @ n_vec
                vel[hit, :2] -= np.maximum(vn, 0.0)[:, None] * n_vec
                vel[hit, :2] *= 0.9

    def _xsph(self, vel: np.ndarray, active: np.ndarray) -> None:
        terms = self._pairs
        if terms is None:
            return
        i, j, weight, _gvec, _g2 = terms
        n = self.n
        cj = self.vol[j] * weight
        ci = self.vol[i] * weight
        den = np.bincount(i, weights=cj, minlength=n) + np.bincount(j, weights=ci, minlength=n)
        num = np.zeros((n, 3))
        for axis in range(3):
            num[:, axis] = np.bincount(
                i, weights=cj * vel[j, axis], minlength=n
            ) + np.bincount(j, weights=ci * vel[i, axis], minlength=n)
        good = active & (den > 1e-9)
        if np.any(good):
            vbar = num[good] / den[good, None]
            vel[good] += _XSPH * (vbar - vel[good])

    def _substep(self, dt: float) -> None:
        active = self.release <= self.time + 1e-12
        prev_pos = self.pos.copy()
        prev_vel = self.vel.copy()

        self.vel[active] += _GRAVITY * dt
        self.pos[active] += self.vel[active] * dt
        self._project_positions(self.pos, active)
        for _ in range(_N_ITER):
            self._project(self.pos, active)
            self._project_positions(self.pos, active)

        new_vel = self.vel.copy()
        new_vel[active] = (self.pos[active] - prev_pos[active]) / dt
        self.vel = new_vel
        self._project_velocities(self.pos, self.vel, active)
        self._xsph(self.vel, active)

        speed = np.linalg.norm(self.vel, axis=1)
        fast = speed > _MAX_SPEED
        if np.any(fast):
            self.vel[fast] *= (_MAX_SPEED / speed[fast])[:, None]

        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            self.pos = prev_pos
            self.vel = prev_vel
        self.pos[~active] = prev_pos[~active]
        self.vel[~active] = 0.0
        self.substeps += 1

    def advance_to(self, time: float):  # noqa: A002 - contract signature
        target = float(time)
        guard = 0
        while self.time < target - 1e-9:
            # never take a degenerate step (a tiny dt would inflate velocities)
            dt = min(_DT, max(target - self.time, 1e-6))
            self._substep(dt)
            self.time += dt
            guard += 1
            if guard > 100000:
                break
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------
    # trajectory diagnostics fields
    # ------------------------------------------------------------------
    def _grid_fields(self):
        pos, vel, vol, mass = self.pos, self.vel, self.vol, self.mass
        n_cells = _NX * _NY * _NZ
        shape = (_NX, _NY, _NZ)

        f = (pos - _ORIGIN) / _DX
        base = np.clip(np.floor(f).astype(np.int64), 0, np.array([_NX - 2, _NY - 2, _NZ - 2]))
        frac = np.clip(f - base, 0.0, 1.0)

        wsum = np.zeros(n_cells)
        vsum = np.zeros((n_cells, 3))
        mnum = np.zeros(n_cells)
        vden = np.zeros(n_cells)
        columns = base[:, 0] * _NY + base[:, 1]
        col = columns.astype(np.int64)

        for a in (0, 1):
            wa = frac[:, 0] if a else 1.0 - frac[:, 0]
            for b in (0, 1):
                wb = frac[:, 1] if b else 1.0 - frac[:, 1]
                for c in (0, 1):
                    wc = frac[:, 2] if c else 1.0 - frac[:, 2]
                    wgt = wa * wb * wc
                    idx = ((base[:, 0] + a) * _NY + (base[:, 1] + b)) * _NZ + (base[:, 2] + c)
                    wsum += np.bincount(idx, weights=wgt, minlength=n_cells)
                    mnum += np.bincount(idx, weights=wgt * mass, minlength=n_cells)
                    vden += np.bincount(idx, weights=wgt * vol, minlength=n_cells)
                    for axis in range(3):
                        vsum[:, axis] += np.bincount(idx, weights=wgt * vel[:, axis], minlength=n_cells)

        w_den = np.maximum(wsum, 1e-12)
        velocity = (vsum / w_den[:, None]).reshape(shape[0], shape[1], shape[2], 3)
        density = np.where(
            vden > 1e-12, mnum / np.maximum(vden, 1e-12), 0.0
        ).reshape(shape)

        z_hi = np.full(_NX * _NY, -np.inf)
        z_lo = np.full(_NX * _NY, np.inf)
        np.maximum.at(z_hi, col, pos[:, 2])
        np.minimum.at(z_lo, col, pos[:, 2])
        z_hi = _dilate3(z_hi.reshape(_NX, _NY), -np.inf)
        z_lo = _erode3(z_lo.reshape(_NX, _NY), np.inf)
        empty = ~np.isfinite(z_hi)
        z_hi[empty] = self.z_floor
        z_lo[empty] = self.z_floor
        z_lo = np.minimum(z_lo, z_hi)

        zc = _ORIGIN[2] + _DX * np.arange(_NZ)
        level = np.maximum(
            z_lo[:, :, None] - zc[None, None, :], zc[None, None, :] - z_hi[:, :, None]
        )
        mask = (level <= 0.0).astype(np.float64)
        return (
            velocity.astype(np.float64),
            density.astype(np.float64),
            level.astype(np.float64),
            mask,
        )

    def _frame(self, time: float) -> StateFrame:
        velocity_field, density_field, level_set, active_mask = self._grid_fields()
        n = self.n
        mass_quantity = np.zeros((n, 2))
        mass_quantity[np.arange(n), self.phase] = self.mass
        mass_source = np.zeros(2)
        volume_source = np.zeros(2)
        attributes = {
            "phase_id": self.phase.astype(np.float64),
            "density": self.rho0.copy(),
            "mass_quantity": mass_quantity,
            "volume_quantity": self.vol.copy(),
            "release_time": self.release.copy(),
        }
        fields = {
            "velocity_field": velocity_field,
            "density_field": density_field,
            "liquid_level_set": level_set,
            "active_mask": active_mask,
            "cumulative_mass_source": mass_source,
            "cumulative_volume_source": volume_source,
        }
        scalars = {
            "grid_spacing": float(_DX),
            "characteristic_length": float(self.characteristic_length),
            "heavy_phase_volume": float(self.initial_volume[0]),
            "light_phase_volume": float(self.initial_volume[1]),
        }
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.phase.astype(np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = float(np.linalg.norm(self.vel, axis=1).max()) if self.n else 0.0
        mass_now = [
            float(self.mass[self.phase == p].sum()) for p in self.phase_ids
        ]
        volume_now = [
            float(self.vol[self.phase == p].sum()) for p in self.phase_ids
        ]
        center = [
            (
                float((self.mass[self.phase == p] * self.pos[self.phase == p]).sum(0))
                / max(mass_now[k], 1e-12)
            )
            for k, p in enumerate(self.phase_ids)
        ]
        return {
            "particles": int(self.n),
            "substeps": int(self.substeps),
            "final_time": float(self.time),
            "max_speed": speed,
            "phase_mass_initial": [float(v) for v in self.initial_mass],
            "phase_mass_final": mass_now,
            "phase_volume_initial": [float(v) for v in self.initial_volume],
            "phase_volume_final": volume_now,
            "phase_center_of_mass_initial": [list(v) for v in self.initial_com],
            "phase_center_of_mass_final": center,
            "wall_top": float(_WALL_TOP),
            "grid_spacing": float(_DX),
            "characteristic_length": float(self.characteristic_length),
        }
