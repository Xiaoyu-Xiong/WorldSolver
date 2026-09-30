"""Deterministic rotating-centrifuge slurry axial-banding solver.

Physical model
--------------
A horizontal transparent cylinder (axis ``x``, inner radius ``R``, length ``L``,
axis height ``z_c``) rotates kinematically about its axis with the prescribed
angular velocity ``omega(t)`` (with a short centrifuge spin-up ramp).  The
partially filled drum contains a dense slurry of two particle-size species in a
carrier liquid.

The granular phase is carried by Lagrangian samples (the frozen scaffold's
particle set) advanced with an explicit symplectic (semi-implicit) Euler
time-stepping scheme.  Forces on every sample:

* body force: gravity reduced by hydrostatic buoyancy of the carrier liquid,
  ``m * g * (1 - rho_l/rho_p)``;
* particle-particle contact: linear spring-dashpot normal force plus a
  viscous-Coulomb regularised tangential friction force, evaluated on a Verlet
  neighbour list rebuilt from a k-d tree (true Newton's-third-law pair forces);
* particle-wall contact: the same law against the inner bore of the rotating
  cylinder and its two end caps; the tangential friction is measured against
  the wall surface velocity ``omega x r``, which spins the bed up;
* particle-fluid coupling: Schiller-Naumann drag toward the local carrier
  liquid velocity ``omega x r``.

Two mixture-theory drift velocities are superposed on the momentum velocity
(drift-flux split, standard for disperse two-phase flow):

* radial size segregation (percolation): samples larger than their local
  radial-shell mean migrate outward and smaller ones inward, with a velocity
  proportional to the local centrifugal acceleration ``omega^2 rho``; the
  deviation-from-local-mean form makes the flux self-limiting;
* axial banding: the axial composition field ``c(x)`` obeys a Cahn-Hilliard
  free energy ``f(c) = W c^2 (1-c)^2`` with gradient penalty ``kappa``.  The
  conservative flux ``J = -M d(mu)/dx`` produces uphill (negative-diffusion)
  transport of the fine species at the spinodal compositions, which is the
  classical continuum mechanism for alternating, persistent
  species-rich bands.  Its mobility is gated by the degree of radial
  segregation, so the radial event precedes the axial event as observed.

All steps are plain arithmetic on NumPy arrays with a fixed seed, hence
deterministic; positions are projected back inside the drum every step, so the
material stays confined.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
_G = 9.81
_LIQUID_DENSITY = 1150.0          # carrier slurry density [kg/m^3]
_LIQUID_VISCOSITY = 0.5           # effective slurry dynamic viscosity [Pa s]
_CONTACT_K = 1.0e5                # particle-particle normal stiffness [N/m]
_WALL_K = 1.2e5                   # particle-wall normal stiffness [N/m]
_ZETA = 0.45                      # contact normal damping ratio
_MU_PARTICLE = 0.5                # particle-particle friction coefficient
_MU_WALL = 0.45                   # particle-wall friction coefficient
_DT = 3.0e-4                      # internal timestep [s]
_SKIN = 0.015                     # Verlet-list skin [m]
_REBUILD_EVERY = 15               # neighbour-list rebuild interval [steps]
_SPINUP = 1.2                     # centrifuge angular-velocity ramp [s]

# segregation / banding closures
_K_SEG = 0.15                     # percolation drift scale [m/s]
_SEG_RAMP = 1.6                   # radial segregation ramp time [s]
_SEG_FULL = 0.05                  # radial-segregation degree for full axial gate
_BAND_M = 0.02                    # Cahn-Hilliard mobility
_BAND_KAPPA = 0.0062              # gradient penalty [m^2]
_BAND_W = 1.0                     # free-energy scale
_BAND_EVERY = 16                  # banding update interval [steps]
_BAND_START = 0.8                 # banding gate onset [s]
_BAND_RAMP = 1.5                  # banding gate ramp [s]
_DRIFT_MAX = 0.15                 # axial drift flux limiter [m/s]
_V_MAX = 8.0                      # velocity safety clamp [m/s]


def _smooth_neumann(values: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for _ in range(passes):
        left = np.concatenate((out[:1], out[:-1]))
        right = np.concatenate((out[1:], out[-1:]))
        out = 0.25 * left + 0.5 * out + 0.25 * right
    return out


class BenchmarkSolver:
    """Rotating-drum two-species slurry solver."""

    # ------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None)
        centrifuge = config.get("centrifuge", {}) if isinstance(config, dict) else {}

        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=float)
        self.velocity = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.species = np.array(frame.attributes["species_id"]).astype(np.int64)
        self.radius = np.array(frame.attributes["particle_radius"], dtype=float)
        self.body_id = np.zeros(self.position.shape[0], dtype=np.int32)
        self.n = int(self.position.shape[0])
        self.species_count = int(max(2, centrifuge.get("species_count", 2)))

        self.wall_radius = float(centrifuge.get("radius", 0.92))
        self.length = float(centrifuge.get("length", 4.8))
        self.z_axis = float(centrifuge.get("center_height", 1.05))
        self.omega_nominal = float(centrifuge.get("angular_velocity_rad_s", 5.5))
        self.fill_fraction = float(centrifuge.get("fill_fraction", 0.68))
        self.particle_density = float(centrifuge.get("particle_density", 1450.0))

        shape = centrifuge.get("field_grid_shape", None)
        if shape is None:
            shape = (40, 18, 18)
        self.grid_shape = tuple(int(v) for v in shape)
        self.h = float(frame.scalars.get("grid_spacing", centrifuge.get("spacing", 0.12)))

        # buoyancy-reduced gravity (carrier liquid displaces the particle volume)
        self.g_eff = np.array(
            [0.0, 0.0, -_G * (1.0 - _LIQUID_DENSITY / max(self.particle_density, 1.0))]
        )

        self.rng = np.random.default_rng(int(seed))
        self.time = 0.0
        self.step_index = 0
        self.pairs = np.zeros((0, 2), dtype=np.int64)
        self.seg_degree = 0.0
        self.band_gate = 0.0

        nx, ny, nz = self.grid_shape
        h = self.h
        self.x_centers = -0.5 * nx * h + (np.arange(nx) + 0.5) * h
        self.y_centers = -0.5 * ny * h + (np.arange(ny) + 0.5) * h
        self.z_centers = -0.5 * nz * h + (np.arange(nz) + 0.5) * h
        self.x_edges = -0.5 * nx * h + np.arange(nx + 1) * h
        self.y_edges = -0.5 * ny * h + np.arange(ny + 1) * h
        self.z_edges = -0.5 * nz * h + np.arange(nz + 1) * h

        # radial shell bins used by the self-limiting percolation closure
        self.radial_bins = 14
        self.radial_edges = np.linspace(0.0, self.wall_radius, self.radial_bins + 1)[1:]

        # axial Cahn-Hilliard working arrays
        self.band_flux = np.zeros(nx + 1, dtype=float)
        self.band_concentration = np.full(nx, 0.5, dtype=float)
        self.axial_drift = np.zeros(self.n, dtype=float)

        # cached force diagnostics (populated by the first force evaluation)
        self._external_force = self.mass[:, None] * self.g_eff[None, :]
        self._interaction_force = np.zeros((self.species_count, 3), dtype=float)

        self._build_pairs()
        self.diagnostics_payload = {
            "samples": self.n,
            "timestep": _DT,
            "grid_spacing": self.h,
            "wall_radius": self.wall_radius,
        }

    # ------------------------------------------------------- small helpers
    def _omega(self, t: float) -> float:
        return self.omega_nominal * min(1.0, max(0.0, t / _SPINUP))

    def _cylindrical(self):
        y = self.position[:, 1]
        z = self.position[:, 2] - self.z_axis
        rho = np.sqrt(y * y + z * z)
        return y, z, rho

    def _particle_cells(self):
        nx, ny, nz = self.grid_shape
        h = self.h
        ix = np.clip(((self.position[:, 0] + 0.5 * nx * h) / h).astype(np.int64), 0, nx - 1)
        iy = np.clip(((self.position[:, 1] + 0.5 * ny * h) / h).astype(np.int64), 0, ny - 1)
        iz = np.clip(
            ((self.position[:, 2] - self.z_axis + 0.5 * nz * h) / h).astype(np.int64), 0, nz - 1
        )
        return ix, iy, iz

    # ------------------------------------------------------- neighbour list
    def _build_pairs(self) -> None:
        if self.n < 2:
            self.pairs = np.zeros((0, 2), dtype=np.int64)
            return
        cutoff = 2.0 * float(self.radius.max()) + _SKIN
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if len(pairs):
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int64)
        self.pairs = pairs

    # ------------------------------------------------------------- forces
    def _pair_forces(self):
        """Return (per-sample particle contact force, per-sample fluid drag)."""
        n = self.n
        contact = np.zeros((n, 3), dtype=float)
        drag = np.zeros((n, 3), dtype=float)
        pairs = self.pairs
        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            d = self.position[i] - self.position[j]
            dist2 = np.einsum("ij,ij->i", d, d)
            rsum = self.radius[i] + self.radius[j]
            dist = np.sqrt(dist2)
            overlap = rsum - dist
            active = overlap > 0.0
            if np.any(active):
                i = i[active]
                j = j[active]
                dist = dist[active]
                overlap = overlap[active]
                normal = d[active] / dist[:, None]
                m_eff = (self.mass[i] * self.mass[j]) / (self.mass[i] + self.mass[j])
                gamma = 2.0 * _ZETA * np.sqrt(_CONTACT_K * m_eff)
                gamma_t = 0.5 * gamma
                rel = self.velocity[i] - self.velocity[j]
                v_n = np.einsum("ij,ij->i", rel, normal)
                f_n = np.maximum(_CONTACT_K * overlap - gamma * v_n, 0.0)
                tangential = rel - v_n[:, None] * normal
                t_mag = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
                safe = np.maximum(t_mag, 1e-12)
                f_t_mag = np.minimum(_MU_PARTICLE * f_n, gamma_t * t_mag)
                force = f_n[:, None] * normal - (f_t_mag / safe)[:, None] * tangential
                for axis in range(3):
                    contact[:, axis] = np.bincount(
                        i, weights=force[:, axis], minlength=n
                    ) - np.bincount(j, weights=force[:, axis], minlength=n)

        # particle-fluid coupling: Schiller-Naumann drag toward omega x r
        omega = self._omega(self.time)
        y = self.position[:, 1]
        z = self.position[:, 2] - self.z_axis
        u_f = np.empty((n, 3), dtype=float)
        u_f[:, 0] = 0.0
        u_f[:, 1] = -omega * z
        u_f[:, 2] = omega * y
        rel = self.velocity - u_f
        speed = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        reynolds = (
            2.0 * self.radius * speed * _LIQUID_DENSITY / max(_LIQUID_VISCOSITY, 1e-9)
        )
        factor = np.minimum(1.0 + 0.15 * np.power(reynolds, 0.687), 12.0)
        coeff = 6.0 * np.pi * _LIQUID_VISCOSITY * self.radius * factor
        drag = -coeff[:, None] * rel
        return contact, drag

    def _wall_forces(self, omega: float):
        """Contact and friction force from the rotating bore and the end caps."""
        n = self.n
        force = np.zeros((n, 3), dtype=float)
        y, z, rho = self._cylindrical()
        rho_safe = np.maximum(rho, 1e-12)
        penetration = rho + self.radius - self.wall_radius
        active = penetration > 0.0
        if np.any(active):
            idx = np.nonzero(active)[0]
            inward = np.empty((idx.size, 3), dtype=float)
            inward[:, 0] = 0.0
            inward[:, 1] = -y[idx] / rho_safe[idx]
            inward[:, 2] = -z[idx] / rho_safe[idx]
            gamma = 2.0 * _ZETA * np.sqrt(_WALL_K * self.mass[idx])
            gamma_t = 0.5 * gamma
            v_n = np.einsum("ij,ij->i", self.velocity[idx], inward)
            f_n = np.maximum(_WALL_K * penetration[idx] - gamma * v_n, 0.0)
            surface = np.empty((idx.size, 3), dtype=float)
            surface[:, 0] = 0.0
            surface[:, 1] = -omega * z[idx]
            surface[:, 2] = omega * y[idx]
            slip = self.velocity[idx] - surface
            slip_n = np.einsum("ij,ij->i", slip, inward)
            tangential = slip - slip_n[:, None] * inward
            t_mag = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
            safe = np.maximum(t_mag, 1e-12)
            f_t = np.minimum(_MU_WALL * f_n, gamma_t * t_mag)
            force[idx] = f_n[:, None] * inward - (f_t / safe)[:, None] * tangential

        half = 0.5 * self.length
        cap_pen = np.abs(self.position[:, 0]) + self.radius - half
        cap_active = cap_pen > 0.0
        if np.any(cap_active):
            idx = np.nonzero(cap_active)[0]
            sign = np.sign(self.position[idx, 0])
            sign[sign == 0.0] = 1.0
            inward = np.zeros((idx.size, 3), dtype=float)
            inward[:, 0] = -sign
            gamma = 2.0 * _ZETA * np.sqrt(_WALL_K * self.mass[idx])
            gamma_t = 0.5 * gamma
            v_n = np.einsum("ij,ij->i", self.velocity[idx], inward)
            f_n = np.maximum(_WALL_K * cap_pen[idx] - gamma * v_n, 0.0)
            y_c = self.position[idx, 1]
            z_c = self.position[idx, 2] - self.z_axis
            surface = np.zeros((idx.size, 3), dtype=float)
            surface[:, 1] = -omega * z_c
            surface[:, 2] = omega * y_c
            slip = self.velocity[idx] - surface
            slip_n = np.einsum("ij,ij->i", slip, inward)
            tangential = slip - slip_n[:, None] * inward
            t_mag = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
            safe = np.maximum(t_mag, 1e-12)
            f_t = np.minimum(_MU_WALL * f_n, gamma_t * t_mag)
            force[idx] += f_n[:, None] * inward - (f_t / safe)[:, None] * tangential
        return force

    # ------------------------------------------------- radial segregation
    def _radial_drift(self, dt: float) -> None:
        y, z, rho = self._cylindrical()
        speed = self._seg_ramp_now()
        if speed <= 0.0:
            self.seg_degree = 0.0
            return
        idx = np.clip(
            np.digitize(rho, self.radial_edges) - 1, 0, self.radial_bins - 1
        )
        count = np.bincount(idx, minlength=self.radial_bins).astype(float)
        total = np.bincount(
            idx, weights=self.radius, minlength=self.radial_bins
        )
        local_mean = total / np.maximum(count, 1.0)
        deviation = (self.radius - local_mean[idx]) / np.maximum(local_mean[idx], 1e-9)
        magnitude = (
            _K_SEG
            * speed
            * np.clip(deviation, -1.0, 1.0)
            * (rho / max(self.wall_radius, 1e-9))
        )
        magnitude = np.clip(magnitude, -0.2, 0.2)
        rho_safe = np.maximum(rho, 1e-12)
        self.position[:, 1] += dt * magnitude * y / rho_safe
        self.position[:, 2] += dt * magnitude * z / rho_safe
        self.seg_degree = self._measure_segregation()

    def _seg_ramp_now(self) -> float:
        return min(1.0, max(0.0, (self.time - _SPINUP) / _SEG_RAMP))

    def _measure_segregation(self) -> float:
        _, _, rho = self._cylindrical()
        small = self.species == 0
        if not np.any(small) or np.all(small):
            return 0.0
        difference = abs(float(rho[~small].mean()) - float(rho[small].mean()))
        return difference / max(self.wall_radius, 1e-9)

    # -------------------------------------------------- axial banding (CH)
    def _composition_profile(self) -> np.ndarray:
        nx = self.grid_shape[0]
        ix = self._particle_cells()[0]
        count = np.bincount(ix, minlength=nx).astype(float)
        fine = np.bincount(ix[self.species == 0], minlength=nx).astype(float)
        concentration = np.where(count > 0, fine / np.maximum(count, 1.0), 0.5)
        self.band_concentration = _smooth_neumann(concentration, passes=2)
        return self.band_concentration

    def _band_update(self, dt_band: float) -> None:
        nx = self.grid_shape[0]
        h = self.h
        c = self._composition_profile()
        gate = min(
            1.0,
            max(
                0.0,
                ((self.time - _BAND_START) / _BAND_RAMP)
                * min(1.0, self.seg_degree / _SEG_FULL),
            ),
        )
        self.band_gate = gate
        if gate <= 0.0:
            self.axial_drift[:] = 0.0
            self.band_flux[:] = 0.0
            return
        f_prime = 2.0 * _BAND_W * c * (1.0 - c) * (1.0 - 2.0 * c)
        laplacian = np.empty_like(c)
        laplacian[1:-1] = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (h * h)
        laplacian[0] = (c[1] - c[0]) / (h * h)
        laplacian[-1] = (c[-2] - c[-1]) / (h * h)
        chemical_potential = f_prime - _BAND_KAPPA * laplacian
        flux = np.zeros(nx + 1, dtype=float)
        flux[1:-1] = -_BAND_M * (chemical_potential[1:] - chemical_potential[:-1]) / h
        self.band_flux = flux
        ix = self._particle_cells()[0]
        face = 0.5 * (flux[ix] + flux[ix + 1])
        fine = self.species == 0
        c_cell = c[ix]
        drift = np.where(
            fine,
            face / np.maximum(c_cell, 0.05),
            -face / np.maximum(1.0 - c_cell, 0.05),
        )
        self.axial_drift = np.clip(drift, -_DRIFT_MAX, _DRIFT_MAX)

    # ------------------------------------------------------ confinement
    def _confine(self) -> None:
        y, z, rho = self._cylindrical()
        limit = self.wall_radius - self.radius
        outside = rho > limit
        if np.any(outside):
            idx = np.nonzero(outside)[0]
            rho_safe = np.maximum(rho[idx], 1e-12)
            scale = limit[idx] / rho_safe
            self.position[idx, 1] = y[idx] * scale
            self.position[idx, 2] = self.z_axis + z[idx] * scale
            radial = np.empty((idx.size, 3), dtype=float)
            radial[:, 0] = 0.0
            radial[:, 1] = y[idx] / rho_safe
            radial[:, 2] = z[idx] / rho_safe
            outward = np.einsum("ij,ij->i", self.velocity[idx], radial)
            positive = outward > 0.0
            self.velocity[idx[positive]] -= (
                outward[positive, None] * radial[positive]
            )
        half = 0.5 * self.length - self.radius
        np.clip(self.position[:, 0], -half, half, out=self.position[:, 0])
        bad = ~np.isfinite(self.position).all(axis=1)
        if np.any(bad):
            self.position[bad] = 0.0
            self.position[bad, 2] = self.z_axis
            self.velocity[bad] = 0.0
        bad = ~np.isfinite(self.velocity).all(axis=1)
        if np.any(bad):
            self.velocity[bad] = 0.0

    # --------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        self.time += dt
        self.step_index += 1
        omega = self._omega(self.time)
        wall_force = self._wall_forces(omega)
        contact_force, drag_force = self._pair_forces()
        external = self.mass[:, None] * self.g_eff[None, :] + wall_force
        total = external + contact_force + drag_force
        self.velocity += dt * total / self.mass[:, None]
        np.clip(self.velocity, -_V_MAX, _V_MAX, out=self.velocity)
        self.position += dt * self.velocity

        ramp = self._seg_ramp_now()
        if ramp > 0.0:
            self._radial_drift(dt)
        self.position[:, 0] += dt * self.axial_drift
        self._confine()

        if self.step_index % _REBUILD_EVERY == 0:
            self._build_pairs()
        if self.step_index % _BAND_EVERY == 0:
            self._band_update(dt * _BAND_EVERY)

        self._external_force = external
        self._interaction_force = self._species_forces(contact_force + drag_force)

    def _species_forces(self, per_sample: np.ndarray) -> np.ndarray:
        out = np.zeros((self.species_count, 3), dtype=float)
        for value in range(self.species_count):
            mask = self.species == value
            if np.any(mask):
                out[value] = per_sample[mask].sum(axis=0)
        return out

    # ------------------------------------------------------------ outputs
    def _velocity_grid(self, omega: float) -> np.ndarray:
        nx, ny, nz = self.grid_shape
        cells = nx * ny * nz
        ix, iy, iz = self._particle_cells()
        flat = (ix * ny + iy) * nz + iz
        count = np.bincount(flat, minlength=cells).astype(float)
        accumulated = np.zeros((cells, 3), dtype=float)
        for axis in range(3):
            accumulated[:, axis] = np.bincount(
                flat, weights=self.velocity[:, axis], minlength=cells
            )
        yy, zz = np.meshgrid(self.y_centers, self.z_centers, indexing="ij")  # (ny, nz)
        fluid = np.zeros((nx, ny, nz, 3), dtype=float)
        fluid[..., 1] = -omega * zz[None, :, :]
        fluid[..., 2] = omega * yy[None, :, :]
        flat_fluid = fluid.reshape(cells, 3)
        weight = np.clip(count / 4.0, 0.0, 1.0)[:, None]
        mixture = accumulated / np.maximum(count, 1.0)[:, None]
        grid = flat_fluid * (1.0 - weight) + mixture * weight
        return grid.reshape(nx, ny, nz, 3)

    def _active_mask(self) -> np.ndarray:
        nx, ny, nz = self.grid_shape
        cells = nx * ny * nz
        ix, iy, iz = self._particle_cells()
        flat = (ix * ny + iy) * nz + iz
        count = np.bincount(flat, minlength=cells).reshape(nx, ny, nz)
        occupied = count > 0
        padded = np.pad(occupied, 1, mode="constant")
        grown = occupied.copy()
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                for dz in (0, 1, 2):
                    shifted = padded[dx : dx + nx, dy : dy + ny, dz : dz + nz]
                    grown |= shifted
        y = self.y_centers[None, :, None]
        z = self.z_centers[None, None, :] - self.z_axis
        inside = (y * y + z * z) <= (self.wall_radius - 0.5 * self.h) ** 2
        return grown & inside

    def _frame(self) -> StateFrame:
        mass_quantity = np.zeros((self.n, self.species_count), dtype=float)
        mass_quantity[np.arange(self.n), self.species] = self.mass
        omega = self._omega(self.time)
        fields = {
            "cumulative_mass_source": np.zeros(self.species_count, dtype=float),
            "velocity_field": self._velocity_grid(omega),
            "active_mask": self._active_mask(),
            "external_force": np.array(self._external_force, dtype=float),
            "interaction_force": np.array(self._interaction_force, dtype=float),
        }
        scalars = {
            "grid_spacing": float(self.h),
            "characteristic_length": float(2.0 * self.wall_radius),
            "contact_length_scale": float(self.h),
            "reference_velocity": float(omega * self.wall_radius),
        }
        attributes = {
            "species_id": self.species.astype(np.int32),
            "particle_radius": self.radius.astype(float),
            "mass_quantity": mass_quantity,
        }
        frame = StateFrame(
            float(self.time),
            np.array(self.position, dtype=float),
            np.array(self.velocity, dtype=float),
            mass=np.array(self.mass, dtype=float),
            body_id=self.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ contract
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        guard = 0
        while self.time < target - 1e-12:
            dt = min(_DT, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("time stepping failed to reach the requested time")
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        _, _, rho = self._cylindrical()
        payload = dict(self.diagnostics_payload)
        payload.update(
            {
                "time": float(self.time),
                "steps": int(self.step_index),
                "samples": int(self.n),
                "max_speed": float(np.abs(self.velocity).max()) if self.n else 0.0,
                "max_radius": float(rho.max()) if self.n else 0.0,
                "wall_radius": float(self.wall_radius),
                "segregation_degree": float(self.seg_degree),
                "band_gate": float(self.band_gate),
                "contact_pairs": int(len(self.pairs)),
            }
        )
        return payload

