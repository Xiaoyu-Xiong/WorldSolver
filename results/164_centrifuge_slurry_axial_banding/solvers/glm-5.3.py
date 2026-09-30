"""Centrifuge slurry axial banding solver (scene 164).

Physically based time-stepping model (no precomputed motion):

* Rigid transparent drum rotating at prescribed angular velocity about the
  horizontal x axis; handled as a hard constraint with tangential (Coulomb-like
  viscous) wall friction so the rotating wall carries the slurry around.
* Carrier liquid modelled as a film in near solid-body rotation with the drum,
  locally retarded by the particle loading of the axial station (two-way
  momentum coupling).
* Per-particle forces: gravity, hydrostatic + centrifugal buoyancy, Stokes drag
  toward the local liquid velocity with Krieger-type hindrance by the local
  solids fraction.  Size-dependent drag relaxation time makes the large species
  migrate outward through the liquid faster than the small species, which
  produces the rotation-driven radial segregation.
* Soft-sphere (spring-dashpot) particle contacts keep the slurry confined and
  coherent; contact pressure saturates the segregation.
* Axial banding: shear-induced migration in the rotating suspension.  The local
  film shear rate is reduced where a species has accumulated (effective
  viscosity closure), so each species drifts axially toward its own
  particle-rich, low-shear stations while being expelled from stations rich in
  the other species.  Particle contact pressure balances this anti-diffusive
  drift and freezes a finite-wavelength, high-contrast banded pattern.

Integration: semi-implicit (symplectic Euler) with fixed substep, hard wall
projection, and a displacement-bounded neighbour rebuild.  Deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """DEM + suspension-coupling solver for the rotating slurry centrifuge."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        self.scene_id = cfg.get("scene_id", "164_centrifuge_slurry_axial_banding")
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.zc = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.h = float(c["spacing"])
        self.rho_p = float(c["particle_density"])
        self.g = np.asarray(cfg["gravity"], dtype=float)
        self.grid_shape = tuple(int(v) for v in c["field_grid_shape"])
        self.K = int(c["species_count"])

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.species = np.asarray(frame0.attributes["species_id"]).astype(np.int32)
        self.radius = np.array(frame0.attributes["particle_radius"], dtype=float)
        self.mass_quantity = np.array(frame0.attributes["mass_quantity"], dtype=float)
        self.volume = self.mass / self.rho_p
        self.N = self.pos.shape[0]
        self.body_id = np.zeros(self.N, dtype=np.int32)

        # ---- numerical / physical parameters -------------------------------
        self.dt = 4.0e-4                 # substep [s]
        self.rho_l = 1000.0              # carrier liquid density [kg/m^3]
        self.mu0 = 25.0                  # base effective slurry viscosity [Pa s]
        self.phi_max = 0.63              # close-packing solids fraction
        self.phi_clip = 0.75             # hindrance / shear closure clip
        self.kappa = 1.0                 # loading retardation of liquid film
        self.gamma0 = 30.0               # film shear-rate scale [1/s]
        self.C_mig = 4.0                 # shear-induced migration coefficient
        self.chi = 0.8                   # cross-species shear competition
        self.F_mig_max = 6.0             # migration force clamp [N]
        self.k_contact = 2.0e4           # contact normal stiffness [N/m]
        self.zeta_contact = 0.35         # contact normal damping ratio
        self.c_tan = 8.0                 # contact tangential viscous coeff
        self.wall_beta = 0.45            # wall tangential friction factor
        self.v_max = 25.0                # safety velocity clamp [m/s]
        self.neigh_radius = 0.17         # pair search radius [m]
        self.rebuild_disp = 0.010        # neighbour rebuild displacement [m]

        # ---- axial stations (loading / liquid coupling) --------------------
        self.nb = 48
        self.dbx = self.L / self.nb
        self.bin_volume = np.pi * self.R**2 * self.dbx

        # ---- diagnostic grid (frozen field grid) ---------------------------
        bmin = np.array([-self.L / 2.0, -self.R, -self.R + self.zc])
        bmax = np.array([self.L / 2.0, self.R, self.R + self.zc])
        ngrid = self.grid_shape
        self.cell_size = (bmax - bmin) / np.asarray(ngrid, dtype=float)
        centers = [
            bmin[d] + (np.arange(ngrid[d]) + 0.5) * self.cell_size[d] for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
        self.gx, self.gy, self.gz = gx, gy, gz
        rad = np.sqrt(gy**2 + (gz - self.zc) ** 2)
        self.active_mask = (rad < self.R - 0.5 * self.cell_size[1]) & (
            np.abs(gx) <= self.L / 2.0
        )
        # liquid velocity on grid centres (solid-body rotation)
        self.liquid_grid = np.stack(
            [np.zeros_like(gx), -self.omega * (gz - self.zc), self.omega * gy], axis=-1
        )
        self.ncell = int(np.prod(ngrid))

        # ---- neighbour cache ------------------------------------------------
        self._pi = np.zeros(0, dtype=np.int64)
        self._pj = np.zeros(0, dtype=np.int64)
        self._build_pos = None
        self._rebuilds = 0

        # ---- diagnostics of the last substep -------------------------------
        self._last_ext = self.mass[:, None] * self.g[None, :]
        self._last_int = np.zeros((self.K, 3))
        self._step = 0
        self.max_speed_seen = 0.0

    # ------------------------------------------------------------------ utils
    def _station(self, x):
        b = np.floor((x + self.L / 2.0) / self.dbx).astype(np.int64)
        return np.clip(b, 0, self.nb - 1)

    @staticmethod
    def _smooth(phi):
        p = np.pad(phi, 1, mode="edge")
        return 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]

    # --------------------------------------------------------------- contacts
    def _ensure_pairs(self):
        if self._build_pos is None:
            rebuild = True
        else:
            disp = np.abs(self.pos - self._build_pos).max()
            rebuild = disp > self.rebuild_disp
        if rebuild:
            tree = cKDTree(self.pos)
            try:
                pairs = tree.query_pairs(r=self.neigh_radius, output_type="ndarray")
            except TypeError:  # older scipy
                raw = sorted(tree.query_pairs(r=self.neigh_radius))
                if raw:
                    pairs = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
                else:
                    pairs = np.zeros((0, 2), dtype=np.int64)
            if pairs.size:
                self._pi = pairs[:, 0].astype(np.int64)
                self._pj = pairs[:, 1].astype(np.int64)
            else:
                self._pi = np.zeros(0, dtype=np.int64)
                self._pj = np.zeros(0, dtype=np.int64)
            self._build_pos = self.pos.copy()
            self._rebuilds += 1

    def _contact_forces(self):
        """Spring-dashpot normal + viscous tangential contact forces."""
        Fc = np.zeros((self.N, 3))
        if self._pi.size == 0:
            return Fc
        i, j = self._pi, self._pj
        d = self.pos[j] - self.pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        good = dist > 1e-9
        i, j, d, dist = i[good], j[good], d[good], dist[good]
        overlap = self.radius[i] + self.radius[j] - dist
        act = overlap > 0.0
        if not np.any(act):
            return Fc
        i, j, d, dist, overlap = i[act], j[act], d[act], dist[act], overlap[act]
        n = d / dist[:, None]
        vrel = self.vel[j] - self.vel[i]
        vn = np.einsum("ij,ij->i", vrel, n)
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_n = 2.0 * self.zeta_contact * np.sqrt(self.k_contact * m_eff)
        fn = self.k_contact * overlap - c_n * vn
        vt = vrel - vn[:, None] * n
        ft = -self.c_tan * vt
        fv = fn[:, None] * n + ft
        for comp in range(3):
            Fi = np.bincount(i, weights=-fv[:, comp], minlength=self.N)
            Fj = np.bincount(j, weights=fv[:, comp], minlength=self.N)
            Fc[:, comp] += Fi + Fj
        return Fc

    # ---------------------------------------------------------------- stepping
    def _substep(self):
        p, v = self.pos, self.vel
        m, r = self.mass, self.radius
        x = p[:, 0]
        y = p[:, 1]
        zp = p[:, 2] - self.zc

        # axial stations: total and per-species solids loading
        b = self._station(x)
        vol_all = np.bincount(b, weights=self.volume, minlength=self.nb)
        vol_s = []
        for s in range(self.K):
            sel = self.species == s
            vol_s.append(np.bincount(b[sel], weights=self.volume[sel], minlength=self.nb))
        phi_tot = self._smooth(vol_all / self.bin_volume)
        phi_s = [self._smooth(vs / self.bin_volume) for vs in vol_s]

        # hindered viscosity and retarded liquid rotation
        phi_c = np.clip(phi_tot / self.phi_max, 0.0, self.phi_clip)
        mu_loc = self.mu0 / (1.0 - phi_c) ** 2
        omega_l = self.omega / (1.0 + self.kappa * phi_tot)

        # species-wise film shear rate (self-clustering, cross-competition)
        gam = []
        for s in range(self.K):
            own = np.clip(phi_s[s] / self.phi_max, 0.0, self.phi_clip)
            other = 0.0
            for t in range(self.K):
                if t != s:
                    other += np.clip(phi_s[t] / self.phi_max, 0.0, self.phi_clip)
            gam.append(self.gamma0 * (1.0 - own) ** 2 * (1.0 + self.chi * other))
        dgam = [np.gradient(g, self.dbx) for g in gam]

        # per-particle liquid velocity (solid-body rotation, local rate)
        wl = omega_l[b]
        vly = -wl * zp
        vlz = wl * y

        # Stokes drag with hindrance
        mu_i = mu_loc[b]
        drag_c = 6.0 * np.pi * mu_i * r
        F_drag = -drag_c[:, None] * np.stack(
            [v[:, 0] - np.zeros_like(vly), v[:, 1] - vly, v[:, 2] - vlz], axis=-1
        )

        # buoyancy: hydrostatic + centrifugal (liquid pressure field)
        F_buoy = -self.rho_l * self.volume[:, None] * (
            self.g[None, :]
            + np.stack(
                [np.zeros_like(y), wl**2 * y, wl**2 * zp], axis=-1
            )
        )

        # shear-induced axial migration
        F_mig = np.zeros((self.N, 3))
        for s in range(self.K):
            sel = self.species == s
            if not np.any(sel):
                continue
            f = -self.C_mig * r[sel] ** 2 * mu_i[sel] * dgam[s][b[sel]]
            f = np.clip(f, -self.F_mig_max, self.F_mig_max)
            F_mig[sel, 0] = f

        F_contact = self._contact_forces()

        F_grav = m[:, None] * self.g[None, :]
        F_tot = F_grav + F_buoy + F_drag + F_mig + F_contact

        # integrate (semi-implicit Euler)
        v += self.dt * F_tot / m[:, None]
        p += self.dt * v

        # ---- rotating drum wall: projection + tangential friction ----------
        y = p[:, 1]
        zp = p[:, 2] - self.zc
        rad = np.sqrt(y * y + zp * zp)
        limit = self.R - r
        pen = rad > limit
        if np.any(pen):
            v_before = v[pen].copy()
            rad_p = rad[pen]
            n_out = np.stack(
                [np.zeros_like(rad_p), y[pen] / rad_p, zp[pen] / rad_p], axis=-1
            )
            # positional correction onto the wall
            p[pen, 1] -= (rad_p - limit[pen]) * n_out[:, 1]
            p[pen, 2] -= (rad_p - limit[pen]) * n_out[:, 2]
            # remove outward normal velocity (wall is rigid)
            vn = np.einsum("ij,ij->i", v[pen], n_out)
            vn_out = vn > 0.0
            idx = np.where(pen)[0][vn_out]
            if idx.size:
                v[idx] -= vn[vn_out][:, None] * n_out[vn_out]
            # tangential friction toward the wall velocity (no-slip-like)
            vw = np.stack(
                [np.zeros_like(rad_p), -self.omega * zp[pen], self.omega * y[pen]], axis=-1
            )
            dv = v[pen] - vw
            dvn = np.einsum("ij,ij->i", dv, n_out)
            dv_t = dv - dvn[:, None] * n_out
            v[pen] -= self.wall_beta * dv_t
            # wall impulse -> force estimate for diagnostics
            wall_force = (m[pen] * (v[pen] - v_before).T).T / self.dt
            self._wall_force = wall_force
            self._wall_pen = pen
        else:
            self._wall_force = np.zeros((0, 3))
            self._wall_pen = pen

        # ---- end caps -------------------------------------------------------
        xlim = self.L / 2.0 - r
        hi = p[:, 0] > xlim
        if np.any(hi):
            p[hi, 0] = xlim[hi]
            v[hi, 0] = np.minimum(v[hi, 0], 0.0)
        lo = p[:, 0] < -xlim
        if np.any(lo):
            p[lo, 0] = -xlim[lo]
            v[lo, 0] = np.maximum(v[lo, 0], 0.0)

        # safety clamps (determinism / finiteness)
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        self.max_speed_seen = max(self.max_speed_seen, float(speed.max()))
        fast = speed > self.v_max
        if np.any(fast):
            v[fast] *= (self.v_max / speed[fast])[:, None]
        if not (np.isfinite(p).all() and np.isfinite(v).all()):
            raise RuntimeError("non-finite state encountered")

        # ---- diagnostics of this substep -------------------------------------
        self._last_ext = F_grav
        F_coupling = F_drag + F_buoy + F_mig + F_contact
        if self._wall_force.size:
            F_coupling = F_coupling.copy()
            F_coupling[self._wall_pen] += self._wall_force
        for s in range(self.K):
            sel = self.species == s
            self._last_int[s] = F_coupling[sel].sum(axis=0)

        self._step += 1

    # ------------------------------------------------------------------ frame
    def _build_frame(self, time: float) -> StateFrame:
        p, v = self.pos, self.vel
        ngrid = self.grid_shape
        cs = self.cell_size
        ix = np.clip(((p[:, 0] + self.L / 2.0) / cs[0]).astype(np.int64), 0, ngrid[0] - 1)
        iy = np.clip(((p[:, 1] + self.R) / cs[1]).astype(np.int64), 0, ngrid[1] - 1)
        iz = np.clip(((p[:, 2] - self.zc + self.R) / cs[2]).astype(np.int64), 0, ngrid[2] - 1)
        lin = (ix * ngrid[1] + iy) * ngrid[2] + iz
        counts = np.bincount(lin, minlength=self.ncell).astype(float)
        vf = self.liquid_grid.copy()
        for comp in range(3):
            s = np.bincount(lin, weights=v[:, comp], minlength=self.ncell)
            occupied = counts > 0
            vf[..., comp][occupied] = s[occupied] / counts[occupied]

        cum_source = np.zeros(self.K)
        frame = StateFrame(
            time,
            p.copy(),
            v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radius.copy(),
                "mass_quantity": self.mass_quantity.copy(),
                "volume_quantity": self.volume.copy(),
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "cumulative_mass_source": cum_source,
                "external_force": self._last_ext.copy(),
                "interaction_force": self._last_int.copy(),
                "velocity_field": vf,
            },
            scalars={
                "grid_spacing": float(self.h),
                "characteristic_length": float(2.0 * self.R),
                "contact_length_scale": float(self.h),
                "reference_velocity": float(self.omega * self.R),
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        target = int(round(t / self.dt))
        while self._step < target:
            self._substep()
        return self._build_frame(t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "dem_slurry_suspension_coupling",
            "substeps": int(self._step),
            "dt": float(self.dt),
            "n_particles": int(self.N),
            "species_counts": [
                int(np.sum(self.species == s)) for s in range(self.K)
            ],
            "neighbour_rebuilds": int(self._rebuilds),
            "max_speed_seen": float(self.max_speed_seen),
            "drum_omega_rad_s": float(self.omega),
        }
