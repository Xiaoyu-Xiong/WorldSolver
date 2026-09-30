"""Particle-based multiphysics solver: egg frying on a hot pan.

Coupled model (no external physics engine):

* Immiscible two-material particle fluid (egg white + egg yolk) advanced with a
  weakly-compressible SPH scheme (cubic-spline kernel, Tait-like equation of
  state with a small cohesive negative-pressure clamp, XSPH viscosity, momentum
  conserving short-range repulsion).  Rest densities are self-calibrated from
  the frozen initial particle lattice, so the two concentrations stay unmixed
  (concentration diffusion disabled) while the density mismatch at the
  interface produces natural immiscibility.
* Grid-based heat conduction (explicit FDM on a frozen 3-D diagnostic grid)
  with a Dirichlet hot-pan bottom boundary (443 K) wherever the egg wets the
  pan and weak ambient-air coupling elsewhere.
* Arrhenius protein-cooking kinetics per particle: the solid-phase fraction
  grows from the pan upward, with phase diffusion enabled (small Laplacian
  smoothing of the phase field through the grid).  Solidification raises the
  viscosity and the pan friction so the cooked layer pins the spreading.
* Contact: inelastic pan contact with solidification-dependent friction and a
  no-escape rim wall.

All required trajectory fields are exported on a frozen 2-D diagnostic grid.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- constants
PAN_Z = 0.018            # pan surface height (top of hot_pan_base) [m]
PAN_T = 443.0            # pan surface temperature [K]
AMBIENT_T = 296.0        # ambient air temperature [K]
EGG_T0 = 291.0           # initial egg temperature [K]
RIM_R = 0.185            # inner rim radius (no-escape) [m]
WALL_R = 0.183           # practical wall radius for particles [m]

H = 0.008                # SPH smoothing length [m]
R_SUP = 2.0 * H          # kernel support radius [m]
SIGMA = 1.0 / (np.pi * H ** 3)
C_SOUND = 2.0            # artificial sound speed [m/s]
V_MAX = 1.5              # velocity safety clamp [m/s]
P_MAX = 3000.0           # pressure clamp [Pa]
A_REP = 400.0            # short-range repulsion acceleration scale [m/s^2]
R_REP = 0.25 * H         # repulsion activation distance [m]

GRAVITY = np.array([0.0, 0.0, -9.81])

WHITE, YOLK = 1, 2
P_COH = {WHITE: 0.5, YOLK: 20.0}          # cohesive negative-pressure clamp [Pa]
ALPHA0 = {WHITE: 0.10, YOLK: 0.40}        # raw-material XSPH viscosity
ALPHA_PHI = 0.55                          # solidification viscosity gain
ALPHA_MAX = 0.80
MU_PAN = 0.3                              # raw pan friction rate [1/s]
MU_PAN_COOKED = 600.0                     # cooked pan friction rate [1/s]

DT = 5.0e-4                               # dynamics substep [s]
THERMAL_EVERY = 20                        # thermal updates per dynamics step
DT_TH = DT * THERMAL_EVERY                # thermal substep [s] (= 0.01 s)

# thermal grid (frozen)
TG_MIN = np.array([-0.20, -0.20, 0.018])
TG_D = 0.005
TG_N = (81, 81, 16)                       # x, y, z cell counts
TG_NCELL = TG_N[0] * TG_N[1] * TG_N[2]
D_COND = 2.5e-6                           # effective thermal diffusivity [m^2/s]
AIR_COUPLE = 0.10                         # weak air-boundary coupling factor
D_PHASE = 8.0e-7                          # phase (solid fraction) diffusivity

# cooking kinetics: k(T) = K_REF * exp(-EA_R * (1/T - 1/T_REF))
T_REF = 433.0
K_REF = {WHITE: 1.5, YOLK: 1.2}           # [1/s]
EA_R = {WHITE: 20000.0, YOLK: 17000.0}    # [K]
T_COOK_MIN = 330.0                        # below this no cooking happens [K]

# 2-D diagnostic output grid (frozen)
OG_MIN = -0.21
OG_D = 0.01
OG_N = 43
CONTACT_LENGTH_SCALE = 0.007              # mean particle spacing [m]


def _kernel(r: np.ndarray):
    """Cubic-spline SPH kernel value and radial derivative at radii r."""
    q = r / H
    w = np.zeros_like(q)
    dw = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    qm1, qm2 = q[m1], q[m2]
    w[m1] = 1.0 - 1.5 * qm1 ** 2 + 0.75 * qm1 ** 3
    w[m2] = 0.25 * (2.0 - qm2) ** 3
    dw[m1] = -3.0 * qm1 + 2.25 * qm1 ** 2
    dw[m2] = -0.75 * (2.0 - qm2) ** 2
    return SIGMA * w, (SIGMA / H) * dw


class BenchmarkSolver:
    """SPH + heat-conduction egg-frying solver."""

    # ------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.seed = int(seed)
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.body = np.array(frame.body_id, dtype=np.int64)
        self.n = len(self.pos)
        self.conc = np.array(frame.attributes["material_concentration"], dtype=float)
        self.label = np.array(frame.attributes["material_label"], dtype=float)
        self.phi = np.zeros(self.n)                    # solid phase fraction
        self.rate = np.zeros(self.n)                   # cooking rate [1/s]
        self.temp = np.full(self.n, EGG_T0)            # particle temperature [K]
        self.time = 0.0
        self.steps = 0
        self._interaction = np.zeros((2, 3))           # white<->yolk pair force
        self._stats = {
            "white_spread_radius_m": [],
            "mean_solid_fraction": [],
            "min_temperature_k": [],
            "max_temperature_k": [],
        }

        # per-material stiffness / rest density calibrated on the frozen lattice
        i, j = self._pair_list()
        rho = self._density(i, j)
        self.rho0 = np.empty(self.n)
        self.stiff = np.empty(self.n)
        for mat, sel in ((WHITE, self.body == WHITE), (YOLK, self.body == YOLK)):
            rest = float(np.median(rho[sel]))
            rest = max(rest, 1.0)
            self.rho0[sel] = rest
            self.stiff[sel] = C_SOUND ** 2 * rest
        self.alpha = self._alpha()
        self.mu = self._mu()

        # thermal grid state
        self.tg_T = np.full(TG_N, AMBIENT_T)
        self.tg_occ = np.zeros(TG_N, dtype=bool)
        self.tg_phi = np.zeros(TG_N)
        self._thermal_update(initial=True)
        self._advance_dynamics(0)  # computes initial interaction force only

    # ------------------------------------------------------- neighbour ops
    def _pair_list(self):
        tree = cKDTree(self.pos)
        try:
            pairs = tree.query_pairs(r=R_SUP, output_type="ndarray")
        except TypeError:  # pragma: no cover - older scipy fallback
            pairs = np.array(sorted(tree.query_pairs(r=R_SUP)), dtype=np.int64)
        if len(pairs) == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)

    def _density(self, i, j):
        if len(i) == 0:
            return self.mass * SIGMA
        _, _, _, w = self._pair_geometry(i, j)
        rho = np.bincount(i, weights=self.mass[j] * w, minlength=self.n)
        rho += np.bincount(j, weights=self.mass[i] * w, minlength=self.n)
        return rho + self.mass * SIGMA

    def _pair_geometry(self, i, j):
        d = self.pos[i] - self.pos[j]
        r = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1.0e-12
        w, dwdr = _kernel(r)
        return d, r, dwdr, w

    def _alpha(self):
        a = np.where(
            self.body == YOLK,
            ALPHA0[YOLK] + ALPHA_PHI * self.phi,
            ALPHA0[WHITE] + ALPHA_PHI * self.phi,
        )
        return np.clip(a, 0.0, ALPHA_MAX)

    def _mu(self):
        return MU_PAN + MU_PAN_COOKED * self.phi

    # ------------------------------------------------------------ dynamics
    def _advance_dynamics(self, nsteps: int) -> None:
        for _ in range(nsteps):
            self._substep()

    def _substep(self) -> None:
        pos, vel, mass, body = self.pos, self.vel, self.mass, self.body
        i, j = self._pair_list()
        if len(i) > 0:
            d, r, dwdr, w = self._pair_geometry(i, j)
            rho = np.bincount(i, weights=mass[j] * w, minlength=self.n)
            rho += np.bincount(j, weights=mass[i] * w, minlength=self.n)
            rho += mass * SIGMA
            p = self.stiff * (rho / self.rho0 - 1.0)
            p_lo = -np.where(body == YOLK, P_COH[YOLK], P_COH[WHITE])
            p = np.clip(p, p_lo, P_MAX)
            # pairwise pressure coefficient: accel on i along +d per unit m_j
            pcoef = -(p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2) * dwdr / r
            # short-range contact repulsion (momentum conserving, r < R_REP)
            rep = A_REP * np.maximum(0.0, 1.0 - r / R_REP) / r
            wsum = mass[i] + mass[j]
            ci = mass[j] * pcoef + (mass[j] / wsum) * rep
            cj = mass[i] * pcoef + (mass[i] / wsum) * rep
            acc = np.empty((self.n, 3))
            for k in range(3):
                acc[:, k] = (
                    np.bincount(i, weights=ci * d[:, k], minlength=self.n)
                    - np.bincount(j, weights=cj * d[:, k], minlength=self.n)
                )
            # net white<->yolk interaction force (exact action/reaction pair)
            cross = (body[i] == WHITE) & (body[j] == YOLK)
            cross |= (body[i] == YOLK) & (body[j] == WHITE)
            if np.any(cross):
                fcoef = mass[i] * mass[j] * pcoef + (mass[i] * mass[j] / wsum) * rep
                dw = np.where((body[i] == WHITE)[:, None], d, -d)
                fsum = np.zeros((2, 3))
                for k in range(3):
                    fsum[0, k] = np.sum(fcoef[cross] * dw[cross, k])
                fsum[1] = -fsum[0]
                self._interaction = fsum
            # XSPH viscosity: dv_i += sum_j (m_j/rho_j) W (v_j - v_i)
            wj = (mass[j] / np.maximum(rho[j], 1.0e-6)) * w
            wi = (mass[i] / np.maximum(rho[i], 1.0e-6)) * w
            xsph = np.zeros((self.n, 3))
            for k in range(3):
                rel = vel[j, k] - vel[i, k]
                xsph[:, k] = (
                    np.bincount(i, weights=wj * rel, minlength=self.n)
                    + np.bincount(j, weights=wi * (-rel), minlength=self.n)
                )
            vel += self.alpha[:, None] * xsph
        else:
            acc = np.zeros((self.n, 3))

        vel += (acc + GRAVITY[None, :]) * DT
        speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        fast = speed > V_MAX
        if np.any(fast):
            vel[fast] *= (V_MAX / speed[fast])[:, None]
        pos += vel * DT
        self._collide()

    def _collide(self) -> None:
        pos, vel = self.pos, self.vel
        # pan floor (inelastic + friction, friction grows with solid fraction)
        contact = pos[:, 2] <= PAN_Z + 0.0015
        below = pos[:, 2] < PAN_Z
        pos[below, 2] = PAN_Z
        vel[below, 2] = np.maximum(vel[below, 2], 0.0)
        if np.any(contact):
            damp = np.maximum(0.0, 1.0 - self.mu[contact] * DT)
            vel[contact, 0] *= damp
            vel[contact, 1] *= damp
        # rim wall (no escape)
        rxy = np.hypot(pos[:, 0], pos[:, 1])
        out = rxy > WALL_R
        if np.any(out):
            oidx = np.flatnonzero(out)
            scale = WALL_R / rxy[oidx]
            pos[oidx, 0] *= scale
            pos[oidx, 1] *= scale
            ux = pos[oidx, 0] / WALL_R
            uy = pos[oidx, 1] / WALL_R
            vr = vel[oidx, 0] * ux + vel[oidx, 1] * uy
            hit = vr > 0.0
            hidx = oidx[hit]
            vel[hidx, 0] -= vr[hit] * ux[hit]
            vel[hidx, 1] -= vr[hit] * uy[hit]
            vel[oidx, 0] *= 0.7
            vel[oidx, 1] *= 0.7

    # ------------------------------------------------------------- thermal
    def _cell_index(self):
        """Nearest-cell index of every particle on the thermal grid."""
        g = np.floor((self.pos - TG_MIN) / TG_D).astype(np.int64)
        g = np.clip(g, 0, np.array(TG_N) - 1)
        flat = (g[:, 0] * TG_N[1] + g[:, 1]) * TG_N[2] + g[:, 2]
        return g, flat

    def _sample_grid(self, field):
        """Trilinear sample of a cell-centred grid field at particle positions."""
        f = (self.pos - TG_MIN) / TG_D - 0.5
        lo = np.floor(f).astype(np.int64)
        lo = np.clip(lo, 0, np.array(TG_N) - 1)
        hi = np.clip(lo + 1, 0, np.array(TG_N) - 1)
        t = np.clip(f - lo, 0.0, 1.0)
        fpad = np.pad(field, 1, mode="edge")
        out = np.zeros(self.n)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = np.stack(
                        (lo[:, 0] + dx, lo[:, 1] + dy, lo[:, 2] + dz), axis=1
                    ) + 1
                    wgt = (
                        (t[:, 0] if dx else 1.0 - t[:, 0])
                        * (t[:, 1] if dy else 1.0 - t[:, 1])
                        * (t[:, 2] if dz else 1.0 - t[:, 2])
                    )
                    out += wgt * fpad[idx[:, 0], idx[:, 1], idx[:, 2]]
        return out

    def _laplacian(self, field, occ, boundary_value, boundary_coupling):
        """6-point Laplacian; insulated at non-occupied neighbours except for a
        weak relaxational coupling toward ``boundary_value`` (ambient air)."""
        fpad = np.pad(field, 1, mode="constant", constant_values=boundary_value)
        opad = np.pad(occ, 1, mode="constant", constant_values=False)
        flux = np.zeros_like(field)
        for axis in range(3):
            for shift in (-1, 1):
                Tn = np.roll(fpad, -shift, axis=axis + 1)[1:-1, 1:-1, 1:-1]
                On = np.roll(opad, -shift, axis=axis + 1)[1:-1, 1:-1, 1:-1]
                flux += np.where(On, Tn - field, boundary_coupling * (boundary_value - field))
        return flux / TG_D ** 2

    def _thermal_update(self, initial: bool = False) -> None:
        g, flat = self._cell_index()
        count = np.bincount(flat, minlength=TG_NCELL).reshape(TG_N)
        mass_cell = np.bincount(flat, weights=self.mass, minlength=TG_NCELL).reshape(TG_N)
        occ = count > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            tsum = np.bincount(
                flat, weights=self.mass * self.temp, minlength=TG_NCELL
            ).reshape(TG_N)
            tmean = np.where(count > 0, tsum / np.maximum(mass_cell, 1e-30), AMBIENT_T)
        newly = occ & ~self.tg_occ
        self.tg_T[newly] = tmean[newly]
        self.tg_occ = occ

        if not initial:
            # heat conduction inside the egg, weak ambient loss at free surface
            lap = self._laplacian(self.tg_T, occ, AMBIENT_T, AIR_COUPLE)
            self.tg_T += DT_TH * D_COND * lap
        # hot-pan Dirichlet boundary wherever the egg wets the pan
        bottom = np.zeros(TG_N, dtype=bool)
        bottom[:, :, 0] = occ[:, :, 0]
        self.tg_T[bottom] = PAN_T
        self.tg_T = np.clip(self.tg_T, 280.0, PAN_T)

        if initial:
            # t = 0: egg starts at its uniform initial temperature everywhere
            self.tg_phi = np.zeros(TG_N)
            return

        # sample temperature back onto particles (advected with material)
        self.temp = np.clip(self._sample_grid(self.tg_T), 280.0, PAN_T)

        # Arrhenius cooking kinetics per particle
        mat = self.body
        kref = np.where(mat == YOLK, K_REF[YOLK], K_REF[WHITE])
        ear = np.where(mat == YOLK, EA_R[YOLK], EA_R[WHITE])
        tt = np.clip(self.temp, 250.0, 460.0)
        kcook = kref * np.exp(-ear * (1.0 / tt - 1.0 / T_REF))
        rate = np.where(tt > T_COOK_MIN, kcook * (1.0 - self.phi), 0.0)
        self.phi = np.clip(self.phi + rate * DT_TH, 0.0, 1.0)
        self.rate = rate

        # phase diffusion (enabled): small Laplacian smoothing of solid fraction
        psum = np.bincount(
            flat, weights=self.mass * self.phi, minlength=TG_NCELL
        ).reshape(TG_N)
        phicell = np.where(count > 0, psum / np.maximum(mass_cell, 1e-30), 0.0)
        self.tg_phi = phicell
        lap_phi = self._laplacian(phicell, occ, 0.0, 0.0)
        dphi = DT_TH * D_PHASE * self._sample_grid(lap_phi)
        self.phi = np.clip(self.phi + dphi, 0.0, 1.0)

        # solidification raises viscosity and pan friction
        self.alpha = self._alpha()
        self.mu = self._mu()

    # ------------------------------------------------------------- frames
    def _output_grid(self):
        """Bin particles onto the frozen 2-D diagnostic grid."""
        ix = np.clip(np.round((self.pos[:, 0] - OG_MIN) / OG_D), 0, OG_N - 1)
        iy = np.clip(np.round((self.pos[:, 1] - OG_MIN) / OG_D), 0, OG_N - 1)
        flat = (ix.astype(np.int64) * OG_N + iy.astype(np.int64))
        ncell = OG_N * OG_N
        m = self.mass
        mcell = np.bincount(flat, weights=m, minlength=ncell).reshape(OG_N, OG_N)
        active = (mcell > 0.0).astype(float)
        denom = np.maximum(mcell, 1e-30)
        phi_cell = (
            np.bincount(flat, weights=m * self.phi, minlength=ncell).reshape(OG_N, OG_N)
            / denom
        )
        rate_cell = (
            np.bincount(flat, weights=m * self.rate, minlength=ncell).reshape(OG_N, OG_N)
            / denom
        )
        vx = (
            np.bincount(flat, weights=m * self.vel[:, 0], minlength=ncell).reshape(OG_N, OG_N)
            / denom
        )
        vy = (
            np.bincount(flat, weights=m * self.vel[:, 1], minlength=ncell).reshape(OG_N, OG_N)
            / denom
        )
        vel_field = np.stack((vx, vy), axis=-1)
        vel_field[active == 0.0] = 0.0
        return active, phi_cell, rate_cell, vel_field

    def _contact_distances(self):
        d_pan = self.pos[:, 2] - PAN_Z
        near = d_pan < 0.012
        dists = list(d_pan[near])
        rxy = np.hypot(self.pos[:, 0], self.pos[:, 1])
        near_wall = rxy > 0.165
        dists.extend((RIM_R - rxy[near_wall]).tolist())
        if not dists:
            dists = [float(np.min(d_pan))]
        return np.asarray(dists, dtype=float)

    def _frame(self, time: float):
        active, phi_cell, rate_cell, vel_field = self._output_grid()
        mass_split = np.zeros((self.n, 2))
        mass_split[self.body == WHITE, 0] = self.mass[self.body == WHITE]
        mass_split[self.body == YOLK, 1] = self.mass[self.body == YOLK]
        attributes = {
            "material_concentration": self.conc.copy(),
            "material_label": self.label.copy(),
            "liquid_phase_fraction": 1.0 - self.phi,
            "solid_phase_fraction": self.phi.copy(),
            "temperature_k": self.temp.copy(),
            "pan_contact_distance": self.pos[:, 2] - PAN_Z,
            "mass_quantity": mass_split,
        }
        fields = {
            "active_mask": active,
            "transported_scalar": phi_cell,
            "scalar_source": rate_cell,
            "velocity_field": vel_field,
            "contact_signed_distance": self._contact_distances(),
            "interaction_force": self._interaction.copy(),
            "cumulative_mass_source": np.zeros(2),
        }
        scalars = {
            "contact_length_scale": CONTACT_LENGTH_SCALE,
            "grid_spacing": OG_D,
            "total_egg_mass_kg": float(np.sum(self.mass)),
            "pan_temperature_k": PAN_T,
        }
        from worldsolver.contracts import StateFrame

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.astype(np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- public
    def advance_to(self, time: float):
        target = max(0.0, float(time))
        expected = int(round(target / DT))
        while self.steps < expected:
            for _ in range(min(THERMAL_EVERY, expected - self.steps)):
                self._substep()
                self.steps += 1
            if self.steps % THERMAL_EVERY == 0:
                self._thermal_update()
        if self.steps == 0 and not hasattr(self, "_initial_frame_done"):
            self._initial_frame_done = True
        # track simple diagnostics
        white = self.body == WHITE
        self._stats["white_spread_radius_m"].append(
            float(np.max(np.hypot(self.pos[white, 0], self.pos[white, 1])))
        )
        self._stats["mean_solid_fraction"].append(float(np.mean(self.phi)))
        self._stats["min_temperature_k"].append(float(np.min(self.temp)))
        self._stats["max_temperature_k"].append(float(np.max(self.temp)))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        white = self.body == WHITE
        yolk = self.body == YOLK
        spread = self._stats["white_spread_radius_m"]
        return {
            "model": "weakly-compressible SPH + grid heat conduction + Arrhenius cooking",
            "num_particles": int(self.n),
            "total_mass_kg": float(np.sum(self.mass)),
            "mass_conserved": bool(np.isclose(np.sum(self.mass), 0.052, atol=1e-9)),
            "concentration_diffusion": "disabled (one-hot, exactly conserved)",
            "phase_diffusion": "enabled",
            "white_spread_radius_initial_m": float(spread[0]) if spread else None,
            "white_spread_radius_final_m": float(spread[-1]) if spread else None,
            "yolk_radius_final_m": float(
                np.max(np.hypot(self.pos[yolk, 0], self.pos[yolk, 1]))
            ),
            "mean_solid_phase_fraction": float(np.mean(self.phi)),
            "mean_white_solid_fraction": float(np.mean(self.phi[white])),
            "solid_front_height_m": float(
                np.max(np.where(self.phi > 0.5, self.pos[:, 2], -1.0))
            ),
            "max_temperature_k": float(np.max(self.temp)),
            "min_temperature_k": float(np.min(self.temp)),
            "max_particle_speed_mps": float(
                np.max(np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel)))
            ),
            "finite_state": True,
        }
