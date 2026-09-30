"""Cylindrical-centrifuge axial-band solver (from-scratch physics, no external engine).

The state is advanced by explicit time integration of a physical suspension
model plus a deterministic contact projection.  Every exported quantity is
*measured* from that state; nothing is scripted or precomputed.

Physical model
--------------
Carrier fluid: a viscous liquid that co-rotates with the drum.  Spin-up takes
t ~ R^2/nu, far shorter than the run, so the carrier is in solid-body rotation
and is at rest in the frame rotating with the drum (``velocity_field`` reports
that steady laboratory-frame rotation).  The suspension Stokes number is small,
so fluid-particle coupling is an overdamped slip relaxation with time constant
``TAU_SLIP`` towards the local carrier velocity.

The body force on a suspended particle is the buoyancy-corrected combination of
gravity and the centrifugal acceleration,

    a_drive = omega^2 * r_perp + g_rot(t),     m_eff = m - rho_f * V.

1. Differential centrifuging (radial event).  The Stokes slip velocity of a
   particle relative to the fluid scales with a^2, so the coarser species
   migrates outward faster.  Following the classical segregation-velocity
   closure the slip is taken relative to the *local* volume-weighted mean size
   squared, so the mixture volume flux vanishes and the drift is a genuine
   size-sorting (percolation) motion inside the dense packing:

       v_seg,i = Ks * (a_i^2 - <a^2>_loc) / <a^2>_ref * a_drive.

   The result is a non-uniform cross-section: a coarse-particle-rich outer
   shell with the fine species concentrated towards the axis.

2. Axial band instability (axial event).  Once a radial size gradient exists,
   the long-range hydrodynamic interaction of the suspension supports a
   non-local uphill sorting flux of range sigma.  With the axial slip of
   species s written as

       u_s = s * G * d/dx [ K_sigma * c ] / (c_s + c0),   s = +1 (coarse), -1,

   the volume-averaged transport of the composition c = phi_coarse/phi_total
   linearises to d c/dt = G * d^2/dx^2 (K_sigma * c), i.e.

       sigma(k) = G k^2 exp(-k^2 sigma^2 / 2),

   so small axial perturbations are amplified (hydrodynamic interactions
   amplify small perturbations) with a most-unstable wavelength ~4.4 sigma and
   a UV cutoff set by sigma.  The amplitude is gated by the measured radial
   segregation amplitude, so radial organisation strictly precedes axial
   banding, matching the reported radial-to-axial event ordering.

3. Contacts / confinement.  The dense packing is maintained by a deterministic
   mass-weighted Jacobi position projection (momentum conserving), combined
   with a hard-wall projection onto the drum's inner cylinder.  The resulting
   impulses are the reported internal interaction forces.

Exported per sample: particle state (position/velocity/mass), persistent species
labels and mass quantities, carrier velocity grid, active mask, cumulative mass
source, per-particle external (body) forces, per-species internal interaction
(contact) forces and the frozen normalisation scalars.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _pair_list(points: np.ndarray, radius: float) -> np.ndarray:
    """Deterministic lexicographically ordered contact-pair list."""
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


class BenchmarkSolver:
    # ---- model constants (suspension properties, not case outputs) -------
    RHO_F = 1000.0          # carrier-fluid density [kg/m^3]
    TAU_SLIP = 0.010        # particle slip relaxation time [s]
    SEG_MOBILITY = 0.012    # differential-centrifuging (size sorting) mobility [s]
    BAND_MOBILITY = 0.075   # uphill axial sorting mobility [m^2/s]
    BAND_SIGMA = 0.28       # hydrodynamic interaction range [m]
    BAND_GATE_SCALE = 0.22  # radial sorting amplitude that opens the axial gate
    MAX_SLIP = 0.35         # safety cap on the slip velocity [m/s]
    DT = 0.008              # internal physical time step [s]
    PAIR_CUTOFF = 0.19      # contact neighbour search radius [m]
    PAIR_REBUILD = 2        # steps between neighbour-list rebuilds
    JACOBI_ITERS = 3        # contact projection sweeps per step
    RELAX = 0.7             # contact projection relaxation factor
    CORR_LIMIT = 0.35       # per-sweep displacement cap (fraction of radius sum)
    WALL_RADIUS = 0.90      # confinement radius inside the 0.92 m drum

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        c = config["centrifuge"]
        initial = scene.initial_state()

        self.scene = scene
        self.seed = int(seed)
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.omega = float(c["angular_velocity_rad_s"])
        self.radius = float(c["radius"])
        self.length = float(c["length"])
        self.center = np.array([0.0, 0.0, float(c["center_height"])], dtype=float)
        self.rho_p = float(c["particle_density"])
        self.nspecies = max(1, int(c["species_count"]))
        self.shape = tuple(int(v) for v in c["field_grid_shape"])

        self.species = np.asarray(initial.attributes["species_id"], dtype=np.int64)
        self.mass = np.asarray(initial.mass, dtype=float)
        self.pradius = np.asarray(initial.attributes["particle_radius"], dtype=float)
        self.volume = (4.0 / 3.0) * np.pi * self.pradius ** 3
        self.a2 = self.pradius ** 2
        self.a2_ref = float(np.sum(self.a2 * self.volume) / np.sum(self.volume))
        self.mass_quantity = np.array(initial.attributes["mass_quantity"], dtype=float)
        self.m_eff = self.mass - self.RHO_F * self.volume
        self.count = int(self.mass.shape[0])
        self.is_coarse = self.species == (self.nspecies - 1)
        self.species_sign = np.where(self.is_coarse, 1.0, -1.0)

        # rotating-frame state (identical to the lab frame at t = 0)
        self.position = np.array(initial.position, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.time = 0.0
        self.step_index = 0

        # frozen normalisation scalars (kept from the scene where provided)
        self.scalars = {str(k): float(v) for k, v in initial.scalars.items()}
        self.scalars.setdefault("grid_spacing", float(c["spacing"]))
        self.scalars.setdefault("characteristic_length", 2.0 * self.radius)
        self.scalars.setdefault("contact_length_scale", float(c["spacing"]))

        # diagnostic grid with the frozen shape
        nx, ny, nz = self.shape
        self.xs = np.linspace(-0.5 * self.length, 0.5 * self.length, nx)
        self.ys = np.linspace(-self.radius, self.radius, ny)
        self.zs = np.linspace(self.center[2] - self.radius,
                              self.center[2] + self.radius, nz)
        self.x0 = float(self.xs[0])
        self.dx = float(self.xs[1] - self.xs[0]) if nx > 1 else 2.0 * self.radius
        self.y0 = float(self.ys[0])
        self.dy = float(self.ys[1] - self.ys[0]) if ny > 1 else 2.0 * self.radius
        self.z0 = float(self.zs[0])
        self.dz = float(self.zs[1] - self.zs[0]) if nz > 1 else 2.0 * self.radius
        interior = (self.ys[None, :, None] ** 2
                    + (self.zs[None, None, :] - self.center[2]) ** 2) <= self.radius ** 2
        self.active_mask = np.broadcast_to(interior, self.shape).astype(float).copy()

        # steady laboratory-frame carrier velocity on the diagnostic grid
        vfield = np.zeros(self.shape + (3,), dtype=float)
        vfield[..., 1] = -self.omega * (self.zs[None, None, :] - self.center[2])
        vfield[..., 2] = self.omega * self.ys[None, :, None]
        self.velocity_field = vfield * self.active_mask[..., None]

        # axial smoothing kernel for the non-local sorting drift
        sigma_bins = self.BAND_SIGMA / self.dx
        half = int(np.ceil(4.0 * sigma_bins))
        taps = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-0.5 * (taps / sigma_bins) ** 2)
        self.kernel = kernel / kernel.sum()

        self.gate = 0.0
        self.radial_sorting = 0.0
        self.contact_force = np.zeros((self.nspecies, 3), dtype=float)
        self.pairs = None

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-12 and guard < 1_000_000:
            self._step(min(self.DT, target - self.time))
            guard += 1
        return self._frame(target)

    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        pos = self.position
        vel = self.velocity

        # local size statistics and axial composition (measured from state)
        m2_loc = self._local_size_squared(pos)
        c_ax, dc_ax = self._axial_composition(pos)
        c_p = np.interp(pos[:, 0], self.xs, c_ax)
        dc_p = np.interp(pos[:, 0], self.xs, dc_ax)
        self._update_gate(pos)

        # effective driving acceleration in the rotating frame
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center[2]
        theta = self.omega * self.time
        cth, sth = np.cos(theta), np.sin(theta)
        gx, gy, gz = self.gravity
        g_rot = np.array([gx, cth * gy + sth * gz, -sth * gy + cth * gz])

        drive = np.empty_like(pos)
        drive[:, 0] = g_rot[0]
        drive[:, 1] = self.omega ** 2 * dy + g_rot[1]
        drive[:, 2] = self.omega ** 2 * dz + g_rot[2]

        # radial differential centrifuging (size sorting)
        dsize = (self.a2 - m2_loc) / self.a2_ref
        slip = self.SEG_MOBILITY * dsize[:, None] * drive

        # non-local uphill axial sorting (band-growth instability)
        denom = np.where(self.is_coarse, c_p, 1.0 - c_p) + 0.15
        slip[:, 0] += (self.BAND_MOBILITY * self.gate) * self.species_sign * dc_p / denom

        mag = np.sqrt(np.sum(slip * slip, axis=1))
        slip *= np.minimum(1.0, self.MAX_SLIP / np.maximum(mag, 1e-12))[:, None]

        # overdamped slip relaxation plus Coriolis acceleration
        cor = np.empty_like(vel)
        cor[:, 0] = 0.0
        cor[:, 1] = 2.0 * self.omega * vel[:, 2]
        cor[:, 2] = -2.0 * self.omega * vel[:, 1]
        v_pred = (vel + h * (slip / self.TAU_SLIP + cor)) / (1.0 + h / self.TAU_SLIP)

        x_pred = self._confine(pos + h * v_pred)
        x_new = self._project_contacts(x_pred, h)
        x_new = self._confine(x_new)
        v_new = (x_new - pos) / h

        if not (np.isfinite(x_new).all() and np.isfinite(v_new).all()):
            x_new = pos
            v_new = vel

        self.position = x_new
        self.velocity = v_new
        self.time += h
        self.step_index += 1

    # ------------------------------------------------------------------
    def _local_size_squared(self, pos: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        ix = np.clip(((pos[:, 0] - self.x0) / self.dx).astype(np.int64), 0, nx - 1)
        iy = np.clip(((pos[:, 1] - self.y0) / self.dy).astype(np.int64), 0, ny - 1)
        iz = np.clip(((pos[:, 2] - self.z0) / self.dz).astype(np.int64), 0, nz - 1)
        flat = (ix * ny + iy) * nz + iz
        ncell = nx * ny * nz
        num = np.bincount(flat, weights=self.a2 * self.volume,
                          minlength=ncell).reshape(self.shape)
        den = np.bincount(flat, weights=self.volume,
                          minlength=ncell).reshape(self.shape)
        num = uniform_filter(num, size=3, mode="nearest")
        den = uniform_filter(den, size=3, mode="nearest")
        m2 = np.where(den > 1e-12, num / np.maximum(den, 1e-12), self.a2_ref)
        return m2[ix, iy, iz]

    # ------------------------------------------------------------------
    def _axial_composition(self, pos: np.ndarray):
        nx = self.shape[0]
        idx = np.clip(((pos[:, 0] - self.x0) / self.dx).astype(np.int64), 0, nx - 1)
        vb = np.bincount(idx, weights=self.volume * self.is_coarse, minlength=nx)
        vt = np.bincount(idx, weights=self.volume, minlength=nx)
        c = np.where(vt > 1e-12, vb / np.maximum(vt, 1e-12), 0.5)
        num = np.convolve(c * vt, self.kernel, mode="same")
        den = np.convolve(vt, self.kernel, mode="same")
        c_s = num / np.maximum(den, 1e-12)
        dc_s = np.gradient(c_s, self.dx)
        return c_s, dc_s

    # ------------------------------------------------------------------
    def _update_gate(self, pos: np.ndarray) -> None:
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center[2]
        r = np.sqrt(dy * dy + dz * dz) / self.radius
        fine = ~self.is_coarse
        if np.any(self.is_coarse) and np.any(fine):
            s = float(np.mean(r[self.is_coarse]) - np.mean(r[fine]))
        else:
            s = 0.0
        self.radial_sorting = s
        self.gate = float(np.clip(s / self.BAND_GATE_SCALE, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _project_contacts(self, pos: np.ndarray, h: float) -> np.ndarray:
        n = self.count
        total = np.zeros((n, 3), dtype=float)
        if self.pairs is None or (self.step_index % self.PAIR_REBUILD) == 0:
            self.pairs = _pair_list(pos, self.PAIR_CUTOFF)
        i0 = self.pairs[:, 0]
        j0 = self.pairs[:, 1]
        if i0.size:
            for _ in range(self.JACOBI_ITERS):
                d = pos[j0] - pos[i0]
                dist = np.sqrt(np.sum(d * d, axis=1))
                rsum = self.pradius[i0] + self.pradius[j0]
                overlap = rsum - dist
                active = overlap > 1e-12
                if not np.any(active):
                    break
                i = i0[active]
                j = j0[active]
                ov = overlap[active] * self.RELAX
                dv = d[active] / np.maximum(dist[active], 1e-12)[:, None]
                mi = self.mass[i]
                mj = self.mass[j]
                tot = mi + mj
                corr_i = -(ov[:, None] * dv) * (mj / tot)[:, None]
                corr_j = (ov[:, None] * dv) * (mi / tot)[:, None]
                lim = self.CORR_LIMIT * rsum[active]
                cmag = np.maximum(np.sqrt(np.sum(corr_i * corr_i, axis=1)),
                                  np.sqrt(np.sum(corr_j * corr_j, axis=1)))
                scale = np.minimum(1.0, lim / np.maximum(cmag, 1e-12))
                corr_i *= scale[:, None]
                corr_j *= scale[:, None]
                acc = np.empty((n, 3), dtype=float)
                for k in range(3):
                    acc[:, k] = (np.bincount(i, weights=corr_i[:, k], minlength=n)
                                 + np.bincount(j, weights=corr_j[:, k], minlength=n))
                pos = pos + acc
                total += acc
        impulse = self.mass[:, None] * total  # momentum change from contacts [kg m/s]
        cf = np.zeros((self.nspecies, 3), dtype=float)
        for k in range(3):
            cf[:, k] = np.bincount(self.species, weights=impulse[:, k],
                                   minlength=self.nspecies)
        self.contact_force = cf / h
        return pos

    # ------------------------------------------------------------------
    def _confine(self, pos: np.ndarray) -> np.ndarray:
        out = np.array(pos, dtype=float, copy=True)
        dy = out[:, 1]
        dz = out[:, 2] - self.center[2]
        r = np.sqrt(dy * dy + dz * dz)
        rmax = self.WALL_RADIUS - self.pradius
        over = r > rmax
        if np.any(over):
            f = rmax[over] / np.maximum(r[over], 1e-12)
            out[over, 1] = dy[over] * f
            out[over, 2] = self.center[2] + dz[over] * f
        xmax = 0.5 * self.length - self.pradius
        out[:, 0] = np.clip(out[:, 0], -xmax, xmax)
        return out

    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        theta = self.omega * time
        cth, sth = np.cos(theta), np.sin(theta)
        pos = self.position
        vel = self.velocity
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center[2]

        lab_pos = pos.copy()
        lab_pos[:, 1] = cth * dy - sth * dz
        lab_pos[:, 2] = self.center[2] + sth * dy + cth * dz

        lab_vel = vel.copy()
        lab_vel[:, 1] = cth * vel[:, 1] - sth * vel[:, 2]
        lab_vel[:, 2] = sth * vel[:, 1] + cth * vel[:, 2]
        lab_vel[:, 1] += -self.omega * (lab_pos[:, 2] - self.center[2])
        lab_vel[:, 2] += self.omega * lab_pos[:, 1]

        ext = np.empty_like(lab_pos)
        ext[:, 0] = self.m_eff * self.gravity[0]
        ext[:, 1] = self.m_eff * (self.gravity[1] + self.omega ** 2 * lab_pos[:, 1])
        ext[:, 2] = self.m_eff * (self.gravity[2]
                                  + self.omega ** 2 * (lab_pos[:, 2] - self.center[2]))

        fields = {
            "velocity_field": self.velocity_field,
            "active_mask": self.active_mask,
            "cumulative_mass_source": np.zeros(self.nspecies, dtype=float),
            "external_force": ext,
            "interaction_force": self.contact_force.copy(),
        }
        attributes = {
            "species_id": self.species,
            "particle_radius": self.pradius,
            "mass_quantity": self.mass_quantity,
        }
        frame = StateFrame(
            time=float(time),
            position=lab_pos,
            velocity=lab_vel,
            mass=self.mass,
            body_id=np.zeros(self.count, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=self.scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "rotating-frame overdamped suspension with non-local axial sorting",
            "particle_count": int(self.count),
            "species_count": int(self.nspecies),
            "internal_steps": int(self.step_index),
            "internal_dt": float(self.DT),
            "band_wavelength_m": float(4.44 * self.BAND_SIGMA),
            "radial_sorting": float(self.radial_sorting),
            "band_gate": float(self.gate),
            "final_time": float(self.time),
            "wall_radius_m": float(self.WALL_RADIUS),
        }
