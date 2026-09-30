"""Wettable, insoluble sand + pouring water: a self-contained particle solver.

Everything is implemented from scratch with NumPy/SciPy only:

* Granular sand -- position based dynamics (PBD) with frictional
  non-penetration contacts between spherical grains, gravity, and contacts
  against the frozen tray boxes.  Sand mass never changes (no dissolution).
* Water -- position based fluid (PBF, Macklin & Mueller style density
  constraints) with XSPH viscosity.  Sand grains contribute to the fluid
  density estimate and exchange momentum with the fluid through mass
  weighted contact projections, which produces splash, run-down over the
  pile flanks and pile deformation at the impact craters.
* Wetting -- a water-concentration transport model for insoluble sand.
  Water touching a grain is absorbed at a finite rate towards a saturation
  capacity of 0.05 kg water per kg sand; the absorbed water then percolates
  downwards under gravity and diffuses between grains, so a wetting front
  starts exactly at the stream impact points and penetrates the pile.
  Absorbed mass is taken out of the donor water particles, so the total
  water mass is conserved up to the prescribed emitter source.

Time stepping is a fixed-step symplectic-Euler predictor followed by
Gauss/Jacobi constraint projection, five substeps per exported sample.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_TINY = 1e-12


def _scatter3(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Scatter-add rows of ``values`` [P,3] into an [n,3] accumulator."""
    out = np.zeros((n, 3), dtype=float)
    if len(index):
        for c in range(3):
            out[:, c] = np.bincount(index, weights=values[:, c], minlength=n)
    return out


def _scatter1(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    if not len(index):
        return np.zeros(n, dtype=float)
    return np.bincount(index, weights=values, minlength=n)

class BenchmarkSolver:
    """Deterministic multiphysics particle solver for wettable sand + water."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.duration = float(config.get("duration", 6.0))
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)

        self.substeps = 5
        self.dt = self.sample_dt / self.substeps
        self.iterations = 3

        # ------------------------------------------------ frozen tray boxes
        # (center, half_extents) of the impermeable floor and the low walls.
        self.boxes = [
            (np.array([0.0, 0.0, -0.04]), np.array([1.55, 1.15, 0.04])),
            (np.array([0.0, 1.11, 0.14]), np.array([1.55, 0.04, 0.14])),
            (np.array([-1.51, 0.0, 0.14]), np.array([0.04, 1.15, 0.14])),
            (np.array([1.51, 0.0, 0.14]), np.array([0.04, 1.15, 0.14])),
        ]
        # Front rim (y = -1.15) keeps the runoff retained and visible in tray.
        self.boxes.append((np.array([0.0, -1.11, 0.14]), np.array([1.55, 0.04, 0.14])))

        # ------------------------------------------------ particle geometry
        self.sand_spacing = 0.075
        self.r_sand = 0.5 * self.sand_spacing
        self.water_spacing = 0.06
        self.r_water = 0.5 * self.water_spacing
        self.h = 2.0 * self.water_spacing          # PBF kernel radius
        self.search_r = max(self.h, self.sand_spacing + self.r_water)

        self.mu_sand = 0.62      # dry grain-grain friction
        self.mu_wall = 0.55
        self.contact_relax = 0.85

        # wetting model constants
        self.sat = 0.05          # kg water per kg sand at saturation
        self.k_absorb = 4.0      # 1/s absorption rate on water contact
        self.k_perc = 1.5        # 1/s gravity driven percolation rate
        self.k_diff = 0.8        # 1/s isotropic capillary diffusion rate

        self._build_initial_state()
        self._build_emitters()
        self._build_grid()

        self.step_index = 0
        self.time = 0.0
        self._frame_cache: dict[int, StateFrame] = {}
        self._diag = {
            "method": "PBD granular sand + position based fluid water",
            "substeps_per_sample": self.substeps,
            "dt": self.dt,
        }

    def _build_initial_state(self) -> None:
        frame = self.scene.initial_state()
        pos = np.asarray(frame.position, dtype=float).copy()
        vel = np.asarray(frame.velocity, dtype=float).copy()
        body = np.asarray(frame.body_id, dtype=np.int32).copy()
        mass = np.asarray(frame.mass, dtype=float).copy()

        self.pos = pos
        self.vel = vel
        self.body = body
        self.is_water = body == 2
        self.dry_mass = np.where(self.is_water, 0.0, mass)
        self.wat_mass = np.where(self.is_water, mass, 0.0)
        self.radius = np.where(self.is_water, self.r_water, self.r_sand)
        self.wet_rate = np.zeros(len(pos), dtype=float)
        self.frozen = np.zeros(len(pos), dtype=bool)

        self.sand_mass_ref = float(self.dry_mass[~self.is_water].sum())
        self.water_mass_initial = float(self.wat_mass[self.is_water].sum())
        self.m_water_particle = float(
            np.mean(self.wat_mass[self.is_water]) if self.is_water.any() else 2.75e-4
        )
        self.emitted_water_mass = 0.0

        # PBF rest density from a perfect lattice kernel sum (consistent with
        # the discretisation instead of an analytic continuum value).
        offs = np.array(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        ) * self.water_spacing
        d = np.linalg.norm(offs, axis=1)
        self.rho0 = float(self.m_water_particle * self._poly6(d).sum())
        self.eps_relax = 1.0e-2
        self.v_max = 12.0

    def _build_emitters(self) -> None:
        """Two overhead spouts continue the frozen initial streams."""
        self.emitters = []
        for side in (-1.0, 1.0):
            origin = np.array([side * 0.56, 0.08, 1.945])
            vel = np.array([-side * 0.38, 0.0, -1.85])
            n = vel / (np.linalg.norm(vel) + _TINY)
            e1 = np.array([0.0, 1.0, 0.0])
            e2 = np.cross(n, e1)
            e2 /= np.linalg.norm(e2) + _TINY
            offsets = []
            rad = 0.085
            for a in (-1.0, 0.0, 1.0):
                for b in (-1.0, 0.0, 1.0):
                    off = a * self.water_spacing * e1 + b * self.water_spacing * e2
                    if np.linalg.norm(off) <= rad + 1e-9:
                        offsets.append(off)
            self.emitters.append(
                (origin, vel, np.asarray(offsets, dtype=float))
            )
        speed = 1.85
        self.emit_every = max(1, int(round(self.water_spacing / (speed * self.dt))))

    def _build_grid(self) -> None:
        """Uniform collocated diagnostic grid over the tray + pour column."""
        self.grid_spacing = 0.10
        self.grid_min = np.array([-1.55, -1.15, 0.0])
        self.grid_max = np.array([1.55, 1.15, 2.0])
        extent = self.grid_max - self.grid_min
        self.grid_shape = tuple(
            int(max(2, round(extent[c] / self.grid_spacing))) for c in range(3)
        )

    def _emit(self) -> None:
        new_pos, new_vel = [], []
        for origin, vel, offsets in self.emitters:
            jitter = self.rng.normal(0.0, 4.0e-4, size=offsets.shape)
            new_pos.append(origin[None, :] + offsets + jitter)
            new_vel.append(np.repeat(vel[None, :], len(offsets), axis=0))
        p = np.vstack(new_pos)
        v = np.vstack(new_vel)
        k = len(p)
        m = np.full(k, self.m_water_particle)

        self.pos = np.vstack((self.pos, p))
        self.vel = np.vstack((self.vel, v))
        self.body = np.concatenate((self.body, np.full(k, 2, dtype=np.int32)))
        self.is_water = np.concatenate((self.is_water, np.ones(k, dtype=bool)))
        self.dry_mass = np.concatenate((self.dry_mass, np.zeros(k)))
        self.wat_mass = np.concatenate((self.wat_mass, m))
        self.radius = np.concatenate((self.radius, np.full(k, self.r_water)))
        self.wet_rate = np.concatenate((self.wet_rate, np.zeros(k)))
        self.frozen = np.concatenate((self.frozen, np.zeros(k, dtype=bool)))
        self.emitted_water_mass += float(m.sum())

    # -------------------------------------------------------------- kernels
    def _poly6(self, r: np.ndarray) -> np.ndarray:
        h = self.h
        coef = 315.0 / (64.0 * np.pi * h**9)
        q = np.clip(h * h - r * r, 0.0, None)
        return coef * q**3

    def _spiky_grad_mag(self, r: np.ndarray) -> np.ndarray:
        """Magnitude of dW/dr for the spiky kernel (negative slope removed)."""
        h = self.h
        coef = -45.0 / (np.pi * h**6)
        q = np.clip(h - r, 0.0, None)
        return coef * q * q

    # ---------------------------------------------------------- time stepping
    def _substep(self) -> None:
        dt = self.dt
        n = len(self.pos)
        mass_tot = self.dry_mass + self.wat_mass
        inv_mass = np.where(self.frozen, 0.0, 1.0 / np.maximum(mass_tot, _TINY))

        # ---- predictor (symplectic Euler with mild aerodynamic damping)
        vel = self.vel.copy()
        vel += self.gravity[None, :] * dt
        vel *= np.where(self.is_water, 1.0 - 0.02 * dt, 1.0 - 0.10 * dt)[:, None]
        vel[self.frozen] = 0.0
        speed = np.linalg.norm(vel, axis=1)
        scale = np.where(speed > self.v_max, self.v_max / np.maximum(speed, _TINY), 1.0)
        vel *= scale[:, None]
        pred = self.pos + vel * dt
        pred[self.frozen] = self.pos[self.frozen]

        # ---- neighbour pairs on the predicted configuration
        tree = cKDTree(pred)
        pairs = tree.query_pairs(self.search_r, output_type="ndarray")
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        pairs = np.ascontiguousarray(pairs, dtype=np.int64)
        ia, ib = pairs[:, 0], pairs[:, 1]
        wa, wb = self.is_water[ia], self.is_water[ib]
        m_ww = wa & wb
        m_ss = (~wa) & (~wb)
        m_sw = ~(m_ww | m_ss)

        # water-water and sand-water pairs feed the fluid density estimate
        fluid_pairs = np.nonzero(m_ww | m_sw)[0]
        contact_pairs = np.nonzero(m_ss | m_sw)[0]

        # symmetric double listing for cheap scatter-adds
        f_i = np.concatenate((ia[fluid_pairs], ib[fluid_pairs]))
        f_j = np.concatenate((ib[fluid_pairs], ia[fluid_pairs]))
        c_i = ia[contact_pairs]
        c_j = ib[contact_pairs]
        c_d0 = np.where(
            m_ss[contact_pairs],
            2.0 * self.r_sand,
            self.r_sand + 0.55 * self.r_water,
        )
        # Wet grains hold a steeper slope: friction grows with saturation.
        satur = np.clip(
            self.wat_mass / np.maximum(self.sat * self.dry_mass, _TINY), 0.0, 1.0
        )
        satur = np.where(self.is_water, 0.0, satur)
        mu_pair = self.mu_sand * (1.0 + 0.5 * 0.5 * (satur[c_i] + satur[c_j]))
        c_mu = np.where(m_ss[contact_pairs], mu_pair, 0.12)

        start = pred.copy()
        for _ in range(self.iterations):
            pred = self._project_fluid(pred, f_i, f_j, mass_tot, inv_mass)
            pred = self._project_contacts(
                pred, start, c_i, c_j, c_d0, c_mu, inv_mass
            )
            pred = self._project_boundaries(pred, start, inv_mass)

        # ---- velocity update from the projected positions
        vel = (pred - self.pos) / dt
        vel[self.frozen] = 0.0
        speed = np.linalg.norm(vel, axis=1)
        scale = np.where(speed > self.v_max, self.v_max / np.maximum(speed, _TINY), 1.0)
        vel *= scale[:, None]

        # ---- XSPH viscosity on the water phase only
        ww = np.nonzero(m_ww)[0]
        if len(ww):
            wi = np.concatenate((ia[ww], ib[ww]))
            wj = np.concatenate((ib[ww], ia[ww]))
            d = np.linalg.norm(pred[wi] - pred[wj], axis=1)
            w = self._poly6(d)
            dv = (vel[wj] - vel[wi]) * (w * self.m_water_particle / self.rho0)[:, None]
            vel += 0.12 * _scatter3(wi, dv, len(pred))

        # ---- granular dissipation
        vel *= np.where(self.is_water, 0.9995, 0.985)[:, None]
        vel[self.frozen] = 0.0

        self.pos = pred
        self.vel = vel

        # ---- wetting concentration transport (no sand mass change)
        self._update_wetting(c_i, c_j, m_ss[contact_pairs], dt)

        # ---- park anything that leaves the tray far below the floor
        escaped = (self.pos[:, 2] < -0.30) & (~self.frozen)
        if escaped.any():
            self.frozen[escaped] = True
            self.vel[escaped] = 0.0

    def _project_fluid(self, pred, f_i, f_j, mass_tot, inv_mass):
        """PBF density constraint; sand acts as a heavy static fluid boundary."""
        n = len(pred)
        if not len(f_i):
            return pred
        dvec = pred[f_i] - pred[f_j]
        r = np.linalg.norm(dvec, axis=1)
        w = self._poly6(r)
        rho = self._poly6(np.zeros(n)) * mass_tot
        rho = rho + _scatter1(f_i, w * mass_tot[f_j], n)

        gmag = self._spiky_grad_mag(r)
        nvec = dvec / np.maximum(r, _TINY)[:, None]
        # grad_j C_i = -m_j / rho0 * dW/dr * n_ij  (points from j towards i)
        gj = (-(mass_tot[f_j] / self.rho0) * gmag)[:, None] * nvec
        grad_i = _scatter3(f_i, -gj, n)
        sum_g2 = _scatter1(f_i, np.sum(gj * gj, axis=1), n)
        sum_g2 = sum_g2 + np.sum(grad_i * grad_i, axis=1)

        C = rho / self.rho0 - 1.0
        lam = np.where(
            self.is_water & (~self.frozen),
            -C / (sum_g2 + self.eps_relax),
            0.0,
        )
        lam = np.minimum(lam, 0.0)          # pressure only, no cohesion pull

        contrib = ((lam[f_i] + lam[f_j]) * (mass_tot[f_j] / self.rho0) * gmag)[
            :, None
        ] * nvec
        dp = _scatter3(f_i, contrib, n)
        dp[~self.is_water] = 0.0
        dp[self.frozen] = 0.0
        lim = 0.3 * self.h
        mag = np.linalg.norm(dp, axis=1)
        clamp = np.where(mag > lim, lim / np.maximum(mag, _TINY), 1.0)
        return pred + dp * clamp[:, None]

    def _project_contacts(self, pred, start, c_i, c_j, c_d0, c_mu, inv_mass):
        """Frictional non-penetration for sand-sand and sand-water pairs."""
        n = len(pred)
        if not len(c_i):
            return pred
        dvec = pred[c_i] - pred[c_j]
        dist = np.linalg.norm(dvec, axis=1)
        pen = c_d0 - dist
        act = pen > 0.0
        if not np.any(act):
            return pred
        ai, aj = c_i[act], c_j[act]
        nvec = dvec[act] / np.maximum(dist[act], _TINY)[:, None]
        pen_a = pen[act]
        wi, wj = inv_mass[ai], inv_mass[aj]
        wsum = np.maximum(wi + wj, _TINY)
        corr = self.contact_relax * pen_a / wsum
        di = (wi * corr)[:, None] * nvec
        dj = -(wj * corr)[:, None] * nvec

        # Coulomb friction on the tangential relative displacement.
        rel = (pred[ai] - start[ai]) - (pred[aj] - start[aj])
        tang = rel - np.sum(rel * nvec, axis=1)[:, None] * nvec
        tmag = np.linalg.norm(tang, axis=1)
        limit = c_mu[act] * pen_a
        fscale = np.where(tmag > _TINY, np.minimum(1.0, limit / np.maximum(tmag, _TINY)), 0.0)
        ft = -tang * fscale[:, None]
        di = di + (wi / wsum)[:, None] * ft
        dj = dj - (wj / wsum)[:, None] * ft

        dp = _scatter3(ai, di, n) + _scatter3(aj, dj, n)
        dp[self.frozen] = 0.0
        return pred + dp

    def _project_boundaries(self, pred, start, inv_mass):
        """Impermeable tray floor and low walls with wall friction."""
        out = pred.copy()
        rad = self.radius
        for center, half in self.boxes:
            local = out - center[None, :]
            ext = half[None, :] + rad[:, None]
            inside = np.all(np.abs(local) < ext, axis=1) & (~self.frozen)
            idx = np.nonzero(inside)[0]
            if not len(idx):
                continue
            pen = ext[idx] - np.abs(local[idx])
            axis = np.argmin(pen, axis=1)
            rows = np.arange(len(idx))
            depth = pen[rows, axis]
            sign = np.sign(local[idx, :][rows, axis])
            sign = np.where(sign == 0.0, 1.0, sign)
            nvec = np.zeros((len(idx), 3))
            nvec[rows, axis] = sign
            out[idx] += nvec * depth[:, None]

            mu = np.where(self.is_water[idx], 0.02, self.mu_wall)
            rel = out[idx] - start[idx]
            tang = rel - np.sum(rel * nvec, axis=1)[:, None] * nvec
            tmag = np.linalg.norm(tang, axis=1)
            limit = mu * depth
            fscale = np.where(
                tmag > _TINY, np.minimum(1.0, limit / np.maximum(tmag, _TINY)), 0.0
            )
            out[idx] -= tang * fscale[:, None]
        return out

    # ------------------------------------------------------------- wetting
    def _update_wetting(self, c_i, c_j, is_ss, dt) -> None:
        """Insoluble sand: only the water concentration moves, never the sand.

        Absorption at wet contacts, gravity driven percolation and capillary
        diffusion between grains.  Every transfer is antisymmetric, so the
        total water mass is conserved exactly.
        """
        n = len(self.pos)
        cap = self.sat * self.dry_mass
        self.wet_rate = np.zeros(n, dtype=float)

        # ---- absorption at sand/water contacts (the wetting front source)
        sw = ~is_ss
        if len(c_i) and np.any(sw):
            a, b = c_i[sw], c_j[sw]
            a_is_w = self.is_water[a]
            s_idx = np.where(a_is_w, b, a)
            w_idx = np.where(a_is_w, a, b)
            remaining = np.clip(cap - self.wat_mass, 0.0, None)
            cnt_s = _scatter1(s_idx, np.ones(len(s_idx)), n)
            demand = np.minimum(self.k_absorb * dt * remaining, remaining)
            demand = np.where(cnt_s > 0.0, demand, 0.0)
            share = demand[s_idx] / np.maximum(cnt_s[s_idx], 1.0)
            avail = np.clip(self.wat_mass - 0.02 * self.m_water_particle, 0.0, None)
            cnt_w = _scatter1(w_idx, np.ones(len(w_idx)), n)
            donor = avail[w_idx] / np.maximum(cnt_w[w_idx], 1.0)
            take = np.clip(np.minimum(share, donor), 0.0, None)
            gain = _scatter1(s_idx, take, n)
            loss = _scatter1(w_idx, take, n)
            self.wat_mass = np.clip(self.wat_mass + gain - loss, 0.0, None)
            self.wet_rate = np.where(
                self.is_water, 0.0, gain / np.maximum(self.dry_mass, _TINY) / dt
            )

        # ---- percolation + capillary diffusion inside the pile
        ss = is_ss
        if len(c_i) and np.any(ss):
            a, b = c_i[ss], c_j[ss]
            cap_a = np.maximum(cap[a], _TINY)
            cap_b = np.maximum(cap[b], _TINY)
            sat_a = self.wat_mass[a] / cap_a
            sat_b = self.wat_mass[b] / cap_b
            cap_m = 0.5 * (cap_a + cap_b)
            flux_d = self.k_diff * dt * (sat_a - sat_b) * cap_m

            dz = self.pos[a, 2] - self.pos[b, 2]
            up = np.where(dz > 0.0, a, b)
            lo = np.where(dz > 0.0, b, a)
            dist = np.maximum(np.linalg.norm(self.pos[a] - self.pos[b], axis=1), _TINY)
            vert = np.abs(dz) / dist
            room = np.clip(1.0 - self.wat_mass[lo] / np.maximum(cap[lo], _TINY), 0.0, 1.0)
            flux_p = self.k_perc * dt * vert * self.wat_mass[up] * room

            idx = np.concatenate((a, b, up, lo))
            val = np.concatenate((-flux_d, flux_d, -flux_p, flux_p))
            self.wat_mass = np.clip(self.wat_mass + _scatter1(idx, val, n), 0.0, None)

    # ------------------------------------------------------- diagnostic grid
    def _grid_fields(self):
        nx, ny, nz = self.grid_shape
        ncell = nx * ny * nz
        mass_tot = self.dry_mass + self.wat_mass
        wetness = np.where(
            self.is_water, 0.0, self.wat_mass / np.maximum(self.dry_mass, _TINY)
        )

        idx = np.floor((self.pos - self.grid_min[None, :]) / self.grid_spacing)
        idx = idx.astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array([nx, ny, nz])[None, :]), axis=1)
        flat = np.where(
            ok, idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2], 0
        )
        wgt = np.where(ok, 1.0, 0.0)

        m_cell = np.bincount(flat, weights=mass_tot * wgt, minlength=ncell)
        mom = np.zeros((ncell, 3))
        for c in range(3):
            mom[:, c] = np.bincount(
                flat, weights=mass_tot * self.vel[:, c] * wgt, minlength=ncell
            )
        vel_field = mom / np.maximum(m_cell, _TINY)[:, None]

        sand_w = np.where(self.is_water, 0.0, 1.0) * wgt
        sand_m = np.bincount(flat, weights=self.dry_mass * sand_w, minlength=ncell)
        sand_c = np.bincount(flat, weights=sand_w, minlength=ncell)
        wet_sum = np.bincount(
            flat, weights=wetness * self.dry_mass * sand_w, minlength=ncell
        )
        src_sum = np.bincount(
            flat, weights=self.wet_rate * self.dry_mass * sand_w, minlength=ncell
        )
        transported = wet_sum / np.maximum(sand_m, _TINY)
        source = src_sum / np.maximum(sand_m, _TINY)
        active = (sand_c > 0.0).astype(np.float64)

        return (
            vel_field.reshape(nx, ny, nz, 3).astype(np.float32),
            transported.reshape(nx, ny, nz).astype(np.float32),
            source.reshape(nx, ny, nz).astype(np.float32),
            active.reshape(nx, ny, nz).astype(np.float32),
        )

    # ---------------------------------------------------- contact diagnostics
    def _contact_diagnostics(self, max_contacts: int = 800):
        """Signed gaps and paired interaction forces at near-contact samples."""
        mass_tot = self.dry_mass + self.wat_mass
        rmax = 2.0 * self.r_sand
        near = rmax * 1.25
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(near, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros(1, dtype=float), np.zeros((1, 3), dtype=float)
        pairs = np.ascontiguousarray(pairs, dtype=np.int64)
        ia, ib = pairs[:, 0], pairs[:, 1]
        wa, wb = self.is_water[ia], self.is_water[ib]
        keep = ~(wa & wb)
        ia, ib = ia[keep], ib[keep]
        if not len(ia):
            return np.zeros(1, dtype=float), np.zeros((1, 3), dtype=float)

        d0 = np.where(
            self.is_water[ia] | self.is_water[ib],
            self.r_sand + 0.55 * self.r_water,
            2.0 * self.r_sand,
        )
        dvec = self.pos[ia] - self.pos[ib]
        dist = np.linalg.norm(dvec, axis=1)
        gap = dist - d0
        order = np.argsort(gap)[:max_contacts]
        ia, ib, gap = ia[order], ib[order], gap[order]
        dvec = dvec[order]
        dist = np.maximum(dist[order], _TINY)
        nvec = dvec / dist[:, None]

        m_i = np.maximum(mass_tot[ia], _TINY)
        m_j = np.maximum(mass_tot[ib], _TINY)
        m_eff = m_i * m_j / (m_i + m_j)
        pen = np.clip(-gap, 0.0, None)
        # impulse implied by the projected non-penetration correction
        f_mag = self.contact_relax * m_eff * pen / (self.dt * self.dt)
        f_on_i = f_mag[:, None] * nvec
        forces = np.vstack((f_on_i, -f_on_i))
        gaps = np.concatenate((gap, gap))
        return gaps.astype(float), forces.astype(float)

    # ------------------------------------------------------- frame assembly
    def _frame(self, time: float) -> StateFrame:
        n = len(self.pos)
        mass_tot = self.dry_mass + self.wat_mass
        wetness_raw = np.where(
            self.is_water, 0.0, self.wat_mass / np.maximum(self.dry_mass, _TINY)
        )
        wetness = np.where(
            self.is_water, 1.0, np.clip(wetness_raw, 0.0, self.sat)
        )
        concentration = np.where(
            self.is_water, 1.0, np.clip(wetness_raw / self.sat, 0.0, 1.0)
        )
        solid = np.where(self.is_water, 0.0, 1.0)

        mass_quantity = np.column_stack((self.dry_mass, self.wat_mass))
        cumulative_source = np.array([0.0, self.emitted_water_mass], dtype=float)
        vel_field, transported, source, active = self._grid_fields()
        gaps, forces = self._contact_diagnostics()

        sand = ~self.is_water
        wet_sel = sand & (wetness_raw > 0.2 * self.sat)
        if np.any(wet_sel):
            wet_zmin = float(self.pos[wet_sel, 2].min())
            wet_zmax = float(self.pos[wet_sel, 2].max())
        else:
            wet_zmin, wet_zmax = 0.0, 0.0
        pile_top = float(self.pos[sand, 2].max()) if np.any(sand) else 0.0
        radial = np.linalg.norm(
            self.pos[:, :2] - np.array([0.0, 0.12])[None, :], axis=1
        )
        runoff = float(
            self.wat_mass[self.is_water & (radial > 0.80) & (self.pos[:, 2] < 0.25)].sum()
        )

        scalars = {
            "grid_spacing": float(self.grid_spacing),
            "contact_length_scale": float(2.0 * self.r_sand),
            "characteristic_length": float(2.0 * self.r_sand),
            "reference_velocity": 1.85,
            "liquid_phase_id": 2.0,
            "sand_mass": float(self.dry_mass[sand].sum()),
            "water_mass_total": float(self.wat_mass.sum()),
            "water_mass_emitted": float(self.emitted_water_mass),
            "water_mass_in_sand": float(self.wat_mass[sand].sum()),
            "mean_sand_wetness": float(np.mean(wetness_raw[sand])) if np.any(sand) else 0.0,
            "max_sand_wetness": float(np.max(wetness_raw[sand])) if np.any(sand) else 0.0,
            "wetted_sand_fraction": float(np.mean(wet_sel[sand])) if np.any(sand) else 0.0,
            "wetting_front_min_z": wet_zmin,
            "wetting_front_max_z": wet_zmax,
            "wetting_penetration_depth": float(max(0.0, wet_zmax - wet_zmin)),
            "pile_apex_height": pile_top,
            "runoff_mass": runoff,
            "particle_count": float(n),
        }

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=mass_tot.copy(),
            body_id=self.body.copy(),
            attributes={
                "material_id": self.body.copy(),
                "phase_id": self.body.copy(),
                "mass_quantity": mass_quantity,
                "solid_phase_fraction": solid,
                "water_concentration": concentration,
                "wetness": wetness,
            },
            fields={
                "velocity_field": vel_field,
                "transported_scalar": transported,
                "scalar_source": source,
                "active_mask": active,
                "contact_signed_distance": gaps,
                "interaction_force": forces,
                "cumulative_mass_source": cumulative_source,
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_index < target:
            if self.step_index > 0 and self.step_index % self.emit_every == 0:
                self._emit()
            self._substep()
            self.step_index += 1
            self.time = self.step_index * self.dt
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diag)
        out.update(
            {
                "final_particle_count": int(len(self.pos)),
                "sand_mass_drift": float(
                    self.dry_mass[~self.is_water].sum() - self.sand_mass_ref
                ),
                "water_mass_balance": float(
                    self.wat_mass.sum()
                    - (self.water_mass_initial + self.emitted_water_mass)
                ),
                "rest_density": float(self.rho0),
            }
        )
        return out
