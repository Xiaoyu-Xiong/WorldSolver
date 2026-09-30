"""Foam tendril pull-apart solver (self-implemented, NumPy/SciPy only).

Model
-----
* Weakly compressible SPH (cubic-spline kernel) supplies the volumetric foam
  response: pressure resists compaction, a bounded suction branch supplies the
  cohesive stress that lets a stretched neck contract radially, and Galilean
  invariant pair dashpots supply the shear viscosity of shaving foam.
* A peridynamic-style bond network supplies tensile cohesion with viscoplastic
  (Herschel-Bulkley-like) yielding: bonds beyond the yield strain relax their
  rest length (plastic drawing), soften through a damage ramp, and break
  irreversibly once plastic stretch or bond length passes the tearing limits.
  Explicit tearing is therefore emergent irreversible material failure.
* Boundaries follow the frozen geometry: penalty contact with regularised
  Coulomb friction against the cylinder floor top and the box plate underside;
  the foam clump adhered under the plate is held by damped adhesion anchors
  (a Dirichlet-type adhesive boundary condition).
* Integration is symplectic Euler with a fixed sub-step derived from the
  acoustic / contact / bond stability limits, landing exactly on sample times.
"""
from __future__ import annotations

import time as _wall

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_EYE3 = np.eye(3, dtype=np.float64)


class BenchmarkSolver:
    """Cohesive SPH + breakable bond-network foam solver."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._t0_wall = _wall.perf_counter()
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        ic = dict(cfg.get("initial_conditions", {}) or {})
        self.gravity = np.asarray(
            cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        ).reshape(3)
        self.spacing = float(ic.get("particle_spacing", 0.045))
        self.p_rad_nom = float(ic.get("nominal_particle_radius", 0.026))
        self.rho_bulk = float(ic.get("bulk_density", 180.0))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        zone = ic.get("intended_tear_zone_z", [0.78, 1.12]) or [0.78, 1.12]
        self.tear_zone = (float(zone[0]), float(zone[1]))
        self._read_geometry(cfg)
        self._read_initial_state(scene)
        self._setup_parameters()
        self._setup_bonds()
        self._setup_anchors()
        self.time = 0.0
        self.step_count = 0
        self.broken_bonds = 0
        self.tear_time = -1.0
        self.min_neck_diameter = float(self._neck_diameter())
        self._rebuild_pairs()
    # ------------------------------------------------------------------
    def _read_geometry(self, cfg: dict) -> None:
        """Pull the frozen boundary geometry (cylinder floor, box plate)."""
        self.floor_top = 0.0
        self.floor_radius = 1.35
        self.plate_bottom = 1.84
        self.plate_half = np.array([0.78, 0.62, 0.08], dtype=np.float64)
        self.plate_center = np.array([0.0, 0.0, 1.92], dtype=np.float64)
        geometry = cfg.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            geo = dict(obj.get("geometry", {}) or {})
            tr = dict(obj.get("transform", {}) or {})
            trans = np.asarray(
                tr.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
            ).reshape(3)
            oid = str(obj.get("id", ""))
            if oid == "contact_floor" and geo.get("type") == "cylinder":
                self.floor_radius = float(geo.get("radius", 1.35))
                self.floor_top = float(trans[2]) + 0.5 * float(geo.get("height", 0.08))
            elif oid == "upper_plate" and geo.get("type") == "box":
                self.plate_half = np.asarray(
                    geo.get("half_extents", [0.78, 0.62, 0.08]), dtype=np.float64
                ).reshape(3)
                self.plate_center = trans.copy()
                self.plate_bottom = float(trans[2]) - float(self.plate_half[2])

    def _read_initial_state(self, scene) -> None:
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64, copy=True)
        self.vel = np.array(frame0.velocity, dtype=np.float64, copy=True)
        self.n = int(self.pos.shape[0])
        if frame0.mass is not None:
            self.mass = np.array(frame0.mass, dtype=np.float64, copy=True)
        else:
            self.mass = np.full(self.n, self.rho_bulk * self.spacing**3, np.float64)
        self.mass = np.maximum(self.mass, 1e-12)
        self.inv_mass = 1.0 / self.mass
        self.volume = self.mass / self.rho_bulk
        if frame0.body_id is not None:
            self.body_id = np.array(frame0.body_id, dtype=np.int32, copy=True)
        else:
            self.body_id = np.zeros(self.n, dtype=np.int32)
        attrs = dict(frame0.attributes or {})
        rad = attrs.get("particle_radius")
        self.p_radius = (
            np.array(rad, dtype=np.float64, copy=True).reshape(self.n)
            if rad is not None
            else np.full(self.n, self.p_rad_nom, np.float64)
        )
        mat = attrs.get("foam_material")
        self.foam_material = (
            np.array(mat, dtype=np.float64, copy=True).reshape(self.n)
            if mat is not None
            else np.ones(self.n, np.float64)
        )
        tc = attrs.get("tear_candidate")
        if tc is not None:
            self.tear_candidate = np.array(tc, dtype=np.float64, copy=True).reshape(self.n)
        else:
            z = self.pos[:, 2]
            self.tear_candidate = (
                (z >= self.tear_zone[0]) & (z <= self.tear_zone[1])
            ).astype(np.float64)
        self.pos0 = self.pos.copy()
        # Per-particle deformation gradient, advected by the SPH velocity gradient.
        self.F = np.tile(_EYE3, (self.n, 1, 1))
        self.mass_total0 = float(self.mass.sum())

    # ------------------------------------------------------------------
    def _setup_parameters(self) -> None:
        dx = self.spacing
        self.h = 1.1 * dx                      # smoothing length
        self.support = 2.0 * self.h            # cubic-spline support radius
        self.skin = 0.3 * self.h               # verlet skin
        self.c_snd = 12.0                      # artificial sound speed [m/s]
        self.gamma_relax = 0.045               # pair-dashpot relaxation time [s]
        # bond network (peridynamic-style, viscoplastic + damage)
        self.bond_dist = 1.7 * dx
        self.bond_break_len = min(1.35 * self.bond_dist, 0.985 * self.support)
        self.E_bond = 7000.0                   # bond-scale Young modulus [Pa]
        self.eps_yield = 0.06                  # tensile yield strain
        self.tau_plastic = 0.05                # viscoplastic relaxation time [s]
        self.lam_damage = 1.5                  # plastic stretch where damage starts
        self.lam_break = 2.4                   # plastic stretch at rupture
        self.stretch_rupture = 0.55            # brittle elastic-strain rupture limit
        self.zeta_bond = 0.25
        # contact / adhesion
        self.mu_friction = 0.6
        self.contact_omega = 200.0
        self.contact_zeta = 0.7
        self.anchor_omega = 55.0
        self.anchor_zeta = 1.0
        self.adhesion_pressure = 400.0         # plate adhesion traction [Pa]
        self.adhesion_range = 1.0 * dx
        self.v_clamp = 25.0
        # SPH reference density from the frozen lattice so t=0 is nearly stress free
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        rho_est = np.full(self.n, 0.0, np.float64)
        w_self = float(self._kernel_w(np.zeros(1))[0])
        self.w_self = w_self
        rho_est += self.mass * w_self
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.linalg.norm(self.pos[i] - self.pos[j], axis=1)
            w = self._kernel_w(d)
            rho_est += np.bincount(i, weights=self.mass[j] * w, minlength=self.n)
            rho_est += np.bincount(j, weights=self.mass[i] * w, minlength=self.n)
        self.rho0 = float(np.percentile(rho_est, 85.0))
        if not np.isfinite(self.rho0) or self.rho0 <= 1e-6:
            self.rho0 = self.rho_bulk
        self.K_eos = self.rho0 * self.c_snd**2
        self.p_suction_min = -0.12 * self.K_eos
        # dashpot coefficient normalised by the mean kernel sum per particle
        if len(pairs):
            wsum = np.bincount(i, weights=w, minlength=self.n)
            wsum += np.bincount(j, weights=w, minlength=self.n)
            s_ref = float(np.median(wsum[wsum > 0.0])) if np.any(wsum > 0.0) else w_self
        else:
            s_ref = w_self
        self.pair_damp_coeff = 1.0 / max(self.gamma_relax * s_ref, 1e-9)
        # sub-step count from acoustic / contact stability
        dt_cfl = 0.25 * self.h / max(self.c_snd, 1e-6)
        self.substeps_nominal = max(24, int(np.ceil(self.sample_dt / dt_cfl)))
        self.substeps = self.substeps_nominal
    def _setup_bonds(self) -> None:
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.bond_dist, output_type="ndarray")
        if len(pairs) == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        order = np.lexsort((pairs[:, 1], pairs[:, 0])) if len(pairs) else np.arange(0)
        pairs = pairs[order]
        self.bi = pairs[:, 0].astype(np.int64)
        self.bj = pairs[:, 1].astype(np.int64)
        d0 = np.linalg.norm(self.pos[self.bi] - self.pos[self.bj], axis=1)
        self.bond_len0 = np.maximum(d0, 1e-9)
        self.bond_rest = self.bond_len0.copy()      # plastically drawn rest length
        self.bond_count0 = int(len(self.bi))
        # peridynamic micro-modulus -> pair force = k_bond * elastic_stretch
        nu = 0.25
        bulk = self.E_bond / (3.0 * (1.0 - 2.0 * nu))
        c_micro = 18.0 * bulk / (np.pi * self.bond_dist**4)
        vol = float(np.median(self.volume))
        self.k_bond = c_micro * vol * vol
        m_ref = float(np.median(self.mass))
        self.c_bond = 2.0 * self.zeta_bond * np.sqrt(
            max(self.k_bond, 1e-9) / max(self.spacing, 1e-9) * m_ref
        )

    def _setup_anchors(self) -> None:
        """Adhesive Dirichlet band for the foam clump stuck under the plate."""
        depth = self.plate_bottom - self.pos[:, 2]
        inside_xy = (np.abs(self.pos[:, 0] - self.plate_center[0]) <= self.plate_half[0]) & (
            np.abs(self.pos[:, 1] - self.plate_center[1]) <= self.plate_half[1]
        )
        d_full = 0.30
        d_zero = 0.52
        w = np.clip((d_zero - depth) / max(d_zero - d_full, 1e-9), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)                      # smooth taper
        w = np.where(inside_xy & (depth >= -1e-9), w, 0.0)
        self.anchor_w = w.astype(np.float64)
        self.anchor_idx = np.flatnonzero(self.anchor_w > 1e-6).astype(np.int64)
        self.anchor_target = self.pos0[self.anchor_idx].copy()
        # The lattice sampling leaves the topmost foam layer a fraction of a
        # spacing below the plate; adhesion draws the wetted clump into contact.
        if self.anchor_idx.size:
            top_surface = float(
                np.max(self.pos0[self.anchor_idx, 2] + self.p_radius[self.anchor_idx])
            )
            lift = float(np.clip(self.plate_bottom - top_surface, 0.0, self.spacing))
            self.anchor_target[:, 2] += lift * self.anchor_w[self.anchor_idx]
            self.anchor_lift = lift
        else:
            self.anchor_lift = 0.0
        m_ref = float(np.median(self.mass))
        self.k_anchor = m_ref * self.anchor_omega**2
        self.c_anchor = 2.0 * self.anchor_zeta * m_ref * self.anchor_omega
        self.k_contact = m_ref * self.contact_omega**2
        self.c_contact = 2.0 * self.contact_zeta * m_ref * self.contact_omega

    # ------------------------------------------------------------------
    # SPH kernel (cubic spline, 3D)
    # ------------------------------------------------------------------
    def _kernel_w(self, r: np.ndarray) -> np.ndarray:
        h = self.h
        q = np.asarray(r, dtype=np.float64) / h
        sigma = 1.0 / (np.pi * h**3)
        w = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        w[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return sigma * w

    def _kernel_dw(self, r: np.ndarray) -> np.ndarray:
        """dW/dr (negative inside the support)."""
        h = self.h
        q = np.asarray(r, dtype=np.float64) / h
        sigma = 1.0 / (np.pi * h**3)
        dw = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        dw[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        dw[m2] = -0.75 * (2.0 - q[m2]) ** 2
        return sigma * dw / h

    # ------------------------------------------------------------------
    def _rebuild_pairs(self) -> None:
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.support + self.skin, output_type="ndarray")
        if len(pairs) == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        self.pi = np.ascontiguousarray(pairs[:, 0].astype(np.int64))
        self.pj = np.ascontiguousarray(pairs[:, 1].astype(np.int64))
        self.pos_at_rebuild = self.pos.copy()

    def _maybe_rebuild(self) -> None:
        drift = np.max(np.abs(self.pos - self.pos_at_rebuild)) if self.n else 0.0
        if 2.0 * float(drift) >= self.skin:
            self._rebuild_pairs()

    @staticmethod
    def _scatter3(idx_i, idx_j, vec, n, out):
        """Accumulate +vec on i and -vec on j (momentum conserving)."""
        for k in range(3):
            out[:, k] += np.bincount(idx_i, weights=vec[:, k], minlength=n)
            out[:, k] -= np.bincount(idx_j, weights=vec[:, k], minlength=n)
        return out
    # ------------------------------------------------------------------
    # one explicit sub-step
    # ------------------------------------------------------------------
    def _step(self, dt: float, update_F: bool, f_dt: float) -> None:
        self._maybe_rebuild()
        n = self.n
        pos = self.pos
        vel = self.vel
        force = self.mass[:, None] * self.gravity[None, :]
        force = np.ascontiguousarray(force)

        # ---------------- weakly compressible SPH ----------------
        i = self.pi
        j = self.pj
        rho = np.full(n, self.w_self, np.float64) * self.mass
        if i.size:
            dx = pos[i] - pos[j]
            r = np.sqrt(np.einsum("ij,ij->i", dx, dx))
            keep = (r < self.support) & (r > 1e-12)
            i = i[keep]
            j = j[keep]
            dx = dx[keep]
            r = r[keep]
        if i.size:
            w = self._kernel_w(r)
            rho = rho + np.bincount(i, weights=self.mass[j] * w, minlength=n)
            rho = rho + np.bincount(j, weights=self.mass[i] * w, minlength=n)
        rho = np.maximum(rho, 1e-6)
        ratio = rho / self.rho0
        press = self.K_eos * (ratio - 1.0)
        # bounded suction branch; switched off in dilute (thread-like) regions
        dens_factor = np.clip((ratio - 0.55) / 0.25, 0.0, 1.0)
        press = np.where(
            press < 0.0, np.maximum(press, self.p_suction_min) * dens_factor, press
        )
        if i.size:
            dw = self._kernel_dw(r)
            e_ij = dx / r[:, None]
            grad_w = dw[:, None] * e_ij
            p_over = press / (rho * rho)
            pfac = -(self.mass[i] * self.mass[j]) * (p_over[i] + p_over[j]) * dw
            pair_vec = pfac[:, None] * e_ij
            dv = vel[i] - vel[j]
            m_red = 2.0 * self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
            pair_vec -= (self.pair_damp_coeff * w * m_red)[:, None] * dv
            self._scatter3(i, j, pair_vec, n, force)
            if update_F:
                self._advect_F(i, j, grad_w, rho, f_dt)

        # ---------------- cohesive bond network ----------------
        self._bond_forces(dt, force)

        # ---------------- boundaries ----------------
        self._boundary_forces(force)

        # ---------------- symplectic Euler ----------------
        vel = vel + dt * (force * self.inv_mass[:, None])
        vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        over = speed > self.v_clamp
        if np.any(over):
            vel[over] *= (self.v_clamp / speed[over])[:, None]
        pos = pos + dt * vel
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        # domain guard: freeze anything that escapes the simulation box
        stray = (
            (np.abs(pos[:, 0]) > 6.0)
            | (np.abs(pos[:, 1]) > 6.0)
            | (pos[:, 2] < -3.0)
            | (pos[:, 2] > 6.0)
        )
        if np.any(stray):
            pos[stray, 0] = np.clip(pos[stray, 0], -6.0, 6.0)
            pos[stray, 1] = np.clip(pos[stray, 1], -6.0, 6.0)
            pos[stray, 2] = np.clip(pos[stray, 2], -3.0, 6.0)
            vel[stray] = 0.0
        self.pos = pos
        self.vel = vel
        self.rho = rho
        self.step_count += 1

    # ------------------------------------------------------------------
    def _advect_F(self, i, j, grad_w, rho, f_dt: float) -> None:
        n = self.n
        dv_ji = self.vel[j] - self.vel[i]
        tens = dv_ji[:, :, None] * grad_w[:, None, :]
        wi = (self.mass[j] / rho[j])[:, None, None] * tens
        wj = (self.mass[i] / rho[i])[:, None, None] * tens
        grad_v = np.zeros((n, 3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                grad_v[:, a, b] = np.bincount(
                    i, weights=wi[:, a, b], minlength=n
                ) + np.bincount(j, weights=wj[:, a, b], minlength=n)
        self.F = self.F + f_dt * np.einsum("nab,nbc->nac", grad_v, self.F)
        self.F = np.nan_to_num(self.F, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.F, -25.0, 25.0, out=self.F)
        det = np.linalg.det(self.F)
        bad = ~np.isfinite(det) | (np.abs(det) < 0.02) | (np.abs(det) > 60.0)
        if np.any(bad):
            self.F[bad] = _EYE3
    # ------------------------------------------------------------------
    def _bond_forces(self, dt: float, force: np.ndarray) -> None:
        """Viscoplastic cohesive bonds with damage softening and rupture."""
        if self.bi.size == 0:
            return
        bi = self.bi
        bj = self.bj
        dx = self.pos[bi] - self.pos[bj]
        length = np.sqrt(np.einsum("ij,ij->i", dx, dx))
        length = np.maximum(length, 1e-9)
        e_ij = dx / length[:, None]
        rest = self.bond_rest
        lam_p = rest / self.bond_len0                      # plastic stretch
        damage = np.clip(
            (self.lam_break - lam_p) / max(self.lam_break - self.lam_damage, 1e-9),
            0.0,
            1.0,
        )
        strain = (length - rest) / rest                    # elastic stretch
        tension = self.k_bond * damage * np.maximum(strain, 0.0)
        dv_n = np.einsum("ij,ij->i", self.vel[bi] - self.vel[bj], e_ij)
        f_mag = tension - self.c_bond * damage * dv_n
        f_mag = np.maximum(f_mag, 0.0)                     # tension only
        pair_vec = (-f_mag)[:, None] * e_ij                # pulls i toward j
        self._scatter3(bi, bj, pair_vec, self.n, force)

        # viscoplastic drawing: rest length creeps toward the current length
        eps_y = self.eps_yield * np.maximum(damage, 0.05)
        excess = strain - eps_y
        flowing = excess > 0.0
        if np.any(flowing):
            rate = np.zeros_like(rest)
            rate[flowing] = np.minimum(
                excess[flowing] * (dt / self.tau_plastic), 0.05
            )
            self.bond_rest = np.maximum(rest * (1.0 + rate), rest)
            rest = self.bond_rest
            lam_p = rest / self.bond_len0

        # irreversible rupture: drawn out, over-strained, or beyond kernel reach
        broke = (
            (lam_p >= self.lam_break)
            | (length >= self.bond_break_len)
            | (strain >= self.stretch_rupture)
        )
        if np.any(broke):
            alive = ~broke
            self.broken_bonds += int(np.count_nonzero(broke))
            self.bi = bi[alive]
            self.bj = bj[alive]
            self.bond_len0 = self.bond_len0[alive]
            self.bond_rest = self.bond_rest[alive]

    # ------------------------------------------------------------------
    def _boundary_forces(self, force: np.ndarray) -> None:
        pos = self.pos
        vel = self.vel
        rad = self.p_radius
        # --- cylinder floor top ---
        gap = (pos[:, 2] - rad) - self.floor_top
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        hit = (gap < 0.0) & (radial <= self.floor_radius)
        if np.any(hit):
            pen = -gap[hit]
            vn = vel[hit, 2]
            fn = self.k_contact * pen - self.c_contact * vn
            fn = np.maximum(fn, 0.0)
            force[hit, 2] += fn
            vt = vel[hit, :2]
            vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt)) + 1e-9
            ft = -self.mu_friction * fn * np.tanh(vt_mag / 0.02) / vt_mag
            force[hit, 0] += ft * vt[:, 0]
            force[hit, 1] += ft * vt[:, 1]
        # --- plate underside ---
        gap_p = self.plate_bottom - (pos[:, 2] + rad)
        under = (
            (np.abs(pos[:, 0] - self.plate_center[0]) <= self.plate_half[0])
            & (np.abs(pos[:, 1] - self.plate_center[1]) <= self.plate_half[1])
            & (gap_p < 0.0)
            & (pos[:, 2] < self.plate_center[2])
        )
        if np.any(under):
            pen = -gap_p[under]
            vn = -vel[under, 2]
            fn = np.maximum(self.k_contact * pen - self.c_contact * vn, 0.0)
            force[under, 2] -= fn
            vt = vel[under, :2]
            vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt)) + 1e-9
            ft = -self.mu_friction * fn * np.tanh(vt_mag / 0.02) / vt_mag
            force[under, 0] += ft * vt[:, 0]
            force[under, 1] += ft * vt[:, 1]
        # --- adhesive traction of foam wetting the plate underside ---
        area = self.spacing * self.spacing
        near = (
            (np.abs(pos[:, 0] - self.plate_center[0]) <= self.plate_half[0])
            & (np.abs(pos[:, 1] - self.plate_center[1]) <= self.plate_half[1])
            & (gap_p >= 0.0)
            & (gap_p <= self.adhesion_range)
        )
        if np.any(near):
            taper = 1.0 - gap_p[near] / self.adhesion_range
            force[near, 2] += self.adhesion_pressure * area * taper
        # --- adhesion anchors holding the clump stuck under the plate ---
        idx = self.anchor_idx
        if idx.size:
            w = self.anchor_w[idx][:, None]
            delta = self.anchor_target - pos[idx]
            f_anchor = w * (self.k_anchor * delta - self.c_anchor * vel[idx])
            force[idx] += f_anchor
    # ------------------------------------------------------------------
    # diagnostics helpers
    # ------------------------------------------------------------------
    def _neck_diameter(self) -> float:
        z0, z1 = self.tear_zone
        z = self.pos[:, 2]
        band = (z >= z0) & (z <= z1)
        if not np.any(band):
            return 0.0
        radial = np.sqrt(self.pos[band, 0] ** 2 + self.pos[band, 1] ** 2)
        zb = z[band]
        edges = np.linspace(z0, z1, 9)
        best = np.inf
        for k in range(len(edges) - 1):
            sel = (zb >= edges[k]) & (zb < edges[k + 1])
            cnt = int(np.count_nonzero(sel))
            if cnt < 3:
                return 0.0
            d = 2.0 * float(np.percentile(radial[sel], 95.0))
            best = min(best, d)
        return 0.0 if not np.isfinite(best) else float(best)

    def _fragment_labels(self) -> tuple[np.ndarray, int]:
        n = self.n
        if self.bi.size:
            data = np.ones(self.bi.size, dtype=np.float64)
            graph = coo_matrix((data, (self.bi, self.bj)), shape=(n, n)).tocsr()
        else:
            graph = coo_matrix((n, n)).tocsr()
        count, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=count)
        big = int(np.count_nonzero(sizes >= 20))
        return labels.astype(np.float64), big

    def _contact_gaps(self) -> np.ndarray:
        pos = self.pos
        rad = self.p_radius
        gap_floor = (pos[:, 2] - rad) - self.floor_top
        gap_plate = self.plate_bottom - (pos[:, 2] + rad)
        band = 3.0 * self.spacing
        gaps = []
        sel_f = gap_floor < band
        if np.any(sel_f):
            gaps.append(gap_floor[sel_f])
        in_xy = (
            np.abs(pos[:, 0] - self.plate_center[0]) <= self.plate_half[0]
        ) & (np.abs(pos[:, 1] - self.plate_center[1]) <= self.plate_half[1])
        sel_p = in_xy & (gap_plate < band) & (pos[:, 2] < self.plate_center[2])
        if np.any(sel_p):
            gaps.append(gap_plate[sel_p])
        if gaps:
            out = np.concatenate(gaps)
        else:
            out = np.array([float(min(gap_floor.min(), gap_plate.min()))], np.float64)
        return np.nan_to_num(out.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        labels, n_frag = self._fragment_labels()
        neck_d = self._neck_diameter()
        if t > 0.05:
            self.min_neck_diameter = min(self.min_neck_diameter, neck_d)
        zone_gone = self._tear_zone_severed()
        if zone_gone and self.tear_time < 0.0:
            self.tear_time = float(t)
            self.tear_diameter = float(self.min_neck_diameter)
        pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        ext_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(t),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": np.ascontiguousarray(self.F.copy()),
                "mass_quantity": self.mass.copy(),
                "volume_quantity": self.volume.copy(),
                "particle_radius": self.p_radius.copy(),
                "foam_material": self.foam_material.copy(),
                "tear_candidate": self.tear_candidate.copy(),
                "fragment_id": labels,
            },
            fields={
                "contact_signed_distance": self._contact_gaps(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.ascontiguousarray(ext_force),
            },
            scalars={
                "contact_length_scale": float(self.p_rad_nom),
                "explicit_tearing_enabled": 1.0,
                "initial_particle_count": float(self.n),
                "fragment_count": float(n_frag),
                "bond_fraction_alive": float(
                    self.bi.size / max(self.bond_count0, 1)
                ),
                "broken_bond_count": float(self.broken_bonds),
                "tendril_min_diameter": float(self.min_neck_diameter),
                "tendril_current_diameter": float(neck_d),
                "tear_time": float(self.tear_time),
                "total_mass": float(self.mass.sum()),
                "sub_steps": float(self.substeps),
            },
        )
        frame.validate()
        return frame

    def _tear_zone_severed(self) -> bool:
        """True once no surviving bond crosses the mid-plane of the tear zone."""
        if self.bi.size == 0:
            return True
        z_mid = 0.5 * (self.tear_zone[0] + self.tear_zone[1])
        zi = self.pos[self.bi, 2]
        zj = self.pos[self.bj, 2]
        crossing = (zi - z_mid) * (zj - z_mid) <= 0.0
        return not bool(np.any(crossing))

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target <= self.time + 1e-12:
            return self._make_frame(max(target, 0.0))
        elapsed = _wall.perf_counter() - self._t0_wall
        if elapsed > 1200.0:
            self.substeps = max(24, self.substeps_nominal // 2)
        while self.time < target - 1e-12:
            span = min(self.sample_dt, target - self.time)
            steps = max(1, int(round(self.substeps * span / self.sample_dt)))
            dt = span / steps
            for k in range(steps):
                do_F = (k % 2) == 0
                self._step(dt, do_F, 2.0 * dt)
            self.time += span
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        labels, n_frag = self._fragment_labels()
        return {
            "method": "cohesive_SPH_with_breakable_viscoplastic_bond_network",
            "particles": int(self.n),
            "sub_steps_per_sample": int(self.substeps),
            "total_steps": int(self.step_count),
            "bonds_initial": int(self.bond_count0),
            "bonds_alive": int(self.bi.size),
            "bonds_broken": int(self.broken_bonds),
            "fragment_count": int(n_frag),
            "tear_time": float(self.tear_time),
            "tendril_min_diameter": float(self.min_neck_diameter),
            "reference_density": float(self.rho0),
            "sound_speed": float(self.c_snd),
            "bond_stiffness": float(self.k_bond),
            "wall_seconds": float(_wall.perf_counter() - self._t0_wall),
        }






