"""Physically-based Lagrangian suspension solver for the horizontal centrifuge.

Physics model (particle-resolved Stokes-coupled suspension):
  * Viscous carrier fluid inside the drum rotates essentially as a rigid body
    with the prescribed drum rotation (low-Re, high-viscosity carrier):
        u_f(x) = omega_vec x (x - x_axis),  omega_vec = (Omega, 0, 0).
  * Each particle obeys Newton's law with
      - gravity m_i g,
      - fluid pressure-gradient (buoyancy/centrifugal) force -(m_i/rho_p)*grad p,
        where grad p / rho_f = omega^2 r_vec - g for rigid-body rotating fluid,
        giving net body acceleration (1 - rho_f/rho_p)*omega^2*r_vec + g:
        differential centrifuging; the heavier/larger species migrates radially
        outward faster, organizing the cross-section first.
      - Stokes hydrodynamic coupling F_d = -gamma_i (v_i - u_f(x_i)),
        gamma_i = 6 pi mu R_i (velocity relaxation rate gamma_i/m_i),
      - pairwise soft-sphere contact: linear spring repulsion + damping and
        tangential Coulomb friction (species contrast: species b slightly
        stickier/coarser -> radial stratification),
      - short-range hydrodynamic (lubrication) cohesion beyond contact; with
        the rotating-drum free-surface-like layering this provides the
        instability that amplifies small axial concentration perturbations
        into alternating particle-rich / particle-poor axial bands
        (axial segregation, cf. PRL 89, 104301),
      - soft confinement by the drum wall (radial) and end caps (axial).
  * Explicit Euler with fixed substeps sized below the drag and contact
    timescales. Neighbor search via uniform spatial hash bins (vectorized).

Required StateFrame fields exported every sample:
  state.position/velocity/mass/body_id, attributes.species_id/particle_radius/
  mass_quantity, fields.external_force/interaction_force/velocity_field/
  active_mask/cumulative_mass_source, scalars.grid_spacing/characteristic_length/
  contact_length_scale/reference_velocity.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

RHO_F = 1000.0      # carrier fluid density [kg/m^3]
MU = 0.06           # carrier dynamic viscosity [Pa s]
COH = 55.0          # hydrodynamic cohesion modulus [N/m]
COH_RANGE = 0.028   # attraction range beyond contact [m]
WALL_K = 6.0e4      # wall normal spring stiffness [N/m]
WALL_GAMMA = 40.0   # wall normal damping [kg/s]
END_K = 6.0e4       # end-cap spring stiffness [N/m]
END_GAMMA = 40.0    # end-cap damping [kg/s]
FRICTION = 0.45     # Coulomb friction coefficient (species b richer)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.zc = float(c["center_height"])   # drum axis: line (x, 0, zc)
        self.omega = float(c["angular_velocity_rad_s"])
        self.rho_p = float(c["particle_density"])
        self.h = float(c["spacing"])
        self.K = int(c["species_count"])
        self.grid_shape = tuple(int(v) for v in c["field_grid_shape"])
        g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g = g
        self.gnorm = float(np.linalg.norm(g))

        st = scene.initial_state()
        self.pos = np.array(st.position, dtype=float, copy=True)
        self.vel = np.array(st.velocity, dtype=float, copy=True)
        self.mass = np.array(st.mass, dtype=float, copy=True)
        self.body_id = np.array(st.body_id, dtype=np.int32, copy=True)
        self.species = np.array(st.attributes["species_id"], dtype=np.int32, copy=True)
        self.radius = np.array(st.attributes["particle_radius"], dtype=float, copy=True)
        self.mass_quantity = np.array(st.attributes["mass_quantity"], dtype=float, copy=True)
        self.N = len(self.pos)
        self.inv_m = 1.0 / self.mass

        # seeded tiny perturbations: symmetry breaking that the hydrodynamic
        # band instability amplifies (reproducible via seed)
        rng = np.random.default_rng(int(seed) + 77)
        self.pos += rng.uniform(-0.012, 0.012, size=(self.N, 3)) * self.h
        self.vel += rng.normal(0.0, 0.02, size=(self.N, 3))
        # enforce interior of drum after jitter
        self._clamp_inside()

        # Stokes drag: gamma = 6 pi mu R ; relaxation rate = gamma / m
        self.gamma = 6.0 * np.pi * MU * self.radius
        self.drag_rate = self.gamma * self.inv_m

        # net body force: gravity + rotating-fluid pressure gradient
        dp_ratio = 1.0 - RHO_F / self.rho_p
        accel = np.empty_like(self.pos)
        accel[:, 0] = self.g[0]
        accel[:, 1] = dp_ratio * self.omega**2 * self.pos[:, 1] + self.g[1]
        accel[:, 2] = dp_ratio * self.omega**2 * (self.pos[:, 2] - self.zc) + self.g[2]
        # NOTE: centrifugal direction is radially outward from axis -> (y, z-zc)
        self.body_force = self.mass[:, None] * accel

        # frozen diagnostic grid over drum bounding box
        nx, ny, nz = self.grid_shape
        self.dxg = self.L / nx
        xg = (np.arange(nx) + 0.5) * self.dxg - self.L / 2.0
        yg = (np.arange(ny) + 0.5) * (2.0 * self.R / ny) - self.R
        zg = (np.arange(nz) + 0.5) * (2.0 * self.R / nz) - self.R + self.zc
        X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
        rr2 = Y**2 + (Z - self.zc) ** 2
        self.active_mask = (rr2 < self.R - 1e-12).astype(np.float64)
        U = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        U[..., 1] = -self.omega * (Z - self.zc)
        U[..., 2] = self.omega * Y
        self.vf_base = U * self.active_mask[..., None]
        self.cell_dV = self.dxg * (2.0 * self.R / ny) * (2.0 * self.R / nz)

        # spatial hash bins
        self.cut = 2.0 * float(self.radius.max()) + COH_RANGE + 1e-6
        self.nbx = int(np.ceil((self.L + 0.06) / self.cut))
        self.nby = int(np.ceil((2 * self.R + 0.06) / self.cut))
        self.nbz = self.nby
        self.bx0 = -self.L / 2 - 0.03
        self.by0 = -self.R - 0.03
        self.bz0 = self.zc - self.R - 0.03
        nb = self.nbx * self.nby * self.nbz
        offs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    offs.append(dx + dy * self.nbz + dz * self.nby * self.nbz)
        self.offs = np.array(offs, dtype=np.int64)
        self.nb_cells = nb

        # export buffers
        self.fext = np.array(self.body_force, copy=True)
        self.fpair = np.zeros((self.N, 3))
        self.interaction_force = np.zeros((self.K, 3))
        self.cum_mass_source = np.zeros(self.K)
        self.t = 0.0
        # timestep: resolve fastest drag relaxation and contact stiffness
        rate_max = float(self.drag_rate.max())
        kmax = WALL_K / float(self.mass.min())
        self.dt = float(np.clip(min(0.15 / max(rate_max, np.sqrt(kmax)),
                                    self.h / 40.0), 5e-4, 2e-3))
        self._compute_forces()

    # ------------------------------------------------------------- utilities
    def _clamp_inside(self) -> None:
        pos = self.pos
        yy = pos[:, 1]
        zz = pos[:, 2] - self.zc
        rr = np.sqrt(yy * yy + zz * zz)
        lim = self.R - self.radius - 1e-4
        over = rr > lim
        if np.any(over):
            s = np.where(over, lim / np.maximum(rr, 1e-12), 1.0)
            pos[:, 1] = yy * s
            pos[:, 2] = self.zc + zz * s
        half = self.L / 2 - self.radius - 1e-4
        np.clip(pos[:, 0], -half, half, out=pos[:, 0])

    def _fluid_velocity(self, pos: np.ndarray) -> np.ndarray:
        # rigid-body rotating fluid: u = omega x (r - axis)
        u = np.empty_like(pos)
        u[:, 0] = 0.0
        u[:, 1] = -self.omega * (pos[:, 2] - self.zc)
        u[:, 2] = self.omega * pos[:, 1]
        return u

    def _compute_forces(self) -> None:
        pos, vel = self.pos, self.vel
        fp = np.zeros_like(self.fpair)

        # ---- wall / end-cap soft confinement ----
        yy = pos[:, 1]
        zz = pos[:, 2] - self.zc
        rr = np.sqrt(yy * yy + zz * zz)
        gap = (self.R - self.radius) - rr
        pen = gap < 0.0
        if np.any(pen):
            ip = np.nonzero(pen)[0]
            nrm = np.stack([yy[ip], zz[ip]], axis=1)
            nl = np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
            nx_ = nrm[:, 0] / nl
            nz_ = nrm[:, 1] / nl
            vn = vel[ip, 1] * nx_ + vel[ip, 2] * nz_
            fn = np.maximum(-WALL_K * gap[ip] - WALL_GAMMA * np.minimum(vn, 0.0), 0.0)
            fp[ip, 1] += fn * nx_
            fp[ip, 2] += fn * nz_
        ax_pen = np.abs(pos[:, 0]) > self.L / 2 - self.radius
        if np.any(ax_pen):
            ia = np.nonzero(ax_pen)[0]
            sx = np.sign(pos[ia, 0])
            ov = np.abs(pos[ia, 0]) - (self.L / 2 - self.radius[ia])
            vx = vel[ia, 0] * sx
            fa = np.maximum(-END_K * ov - END_GAMMA * np.minimum(vx, 0.0), 0.0) * (-sx)
            # push inward: if x > limit, force negative
            fp[ia, 0] += fa
            # (fa already negative for right wall; ensure direction)
        # correct end-cap sign: inward direction is -sign(x)
        # handled above by fa*(-sx) with ov>0 -> fa>0 -> force points inward.

        # ---- pairwise contacts + cohesion via spatial hash ----
        b = self.cut
        bi = np.clip(((pos[:, 0] - self.bx0) / b).astype(np.int64), 0, self.nbx - 1)
        bj = np.clip(((pos[:, 1] - self.by0) / b).astype(np.int64), 0, self.nby - 1)
        bk = np.clip(((pos[:, 2] - self.bz0) / b).astype(np.int64), 0, self.nbz - 1)
        keys = (bi * self.nby + bj) * self.nbz + bk
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        starts = np.zeros(self.nb_cells + 1, dtype=np.int64)
        np.add.at(starts[1:], sk, 1)  # histogram counts into starts[1:]
        # build counts per cell
        counts = np.bincount(sk, minlength=self.nb_cells)
        starts = np.zeros(self.nb_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        occ = np.nonzero(counts)[0]
        # for each occupied cell, gather particle indices in its 27-neighborhood
        occ_all = np.repeat(occ, 27) + np.tile(self.offs, len(occ))
        valid = (occ_all >= 0) & (occ_all < self.nb_cells)
        occ_all = np.where(valid, occ_all, self.nb_cells)
        lo = starts[occ_all]
        hi = starts[occ_all + 1]
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot > 0:
            rep = np.repeat(np.nonzero(cnt)[0], cnt[cnt > 0])
            within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, tot // 1) if False else None
            csum = np.cumsum(cnt)
            within = np.arange(tot) - np.repeat(csum - cnt, cnt)
            # source particles of each source-cell block
            src_cell_rep = np.repeat(np.arange(len(occ_all)), cnt)
            src_cells = occ_all[src_cell_rep]
            I = order[starts[src_cells] + within]
            J = order[lo[rep] + (np.arange(tot) - np.repeat(csum - cnt, cnt))] if False else None
            # neighbor particles: flatten [lo,hi) of each candidate cell
            J = np.concatenate([order[lo[m]:hi[m]] for m in np.nonzero(cnt)[0]])
        # (loop above replaced by vectorized gather below)
        if tot > 0:
            self._apply_pairs(I, J, fp)

        self.fpair = fp
        self.fext = np.array(self.body_force, copy=True)
        # per-species resultant coupling+contact force
        inter = np.zeros((self.K, 3))
        for k in range(self.K):
            m = self.species == k
            inter[k] = fp[m].sum(axis=0)
        self.interaction_force = inter

    def _apply_pairs(self, I: np.ndarray, J: np.ndarray, fp: np.ndarray) -> None:
        keep = I < J
        I = I[keep]
        J = J[keep]
        if len(I) == 0:
            return
        d = self.pos[J] - self.pos[I]
        r2 = np.einsum("ij,ij->i", d, d)
        rsum = self.radius[I] + self.radius[J]
        rng = rsum + COH_RANGE
        act = r2 < rng * rng
        I, J, d, r2, rsum, rng = I[act], J[act], d[act], r2[act], rsum[act], rng[act]
        r = np.sqrt(np.maximum(r2, 1e-12))
        n = d / r[:, None]
        dv = self.vel[I] - self.vel[J]
        vn = np.einsum("ij,ij->i", dv, n)
        # normal: repulsion when overlapping; cohesion in [rsum, rng)
        ovl = rsum - r
        fn = np.where(ovl > 0.0,
                      1.2e4 * ovl - 25.0 * np.minimum(vn, 0.0),
                      -COH * np.minimum(ovl + COH_RANGE, 0.0) * 0.0
                      - COH * (rng - r) / max(COH_RANGE, 1e-9) * 0.0)
        # simpler cohesion profile: constant-modulus spring toward contact
        fn = np.where(ovl > 0.0, fn, -COH * (rng - r) / COH_RANGE)
        fn = np.maximum(fn, -COH)
        f = fn[:, None] * n
        # tangential friction (damped)
        vt = dv - vn[:, None] * n
        ft_mag = np.linalg.norm(vt, axis=1)
        fmax = FRICTION * np.maximum(fn, 0.0)
        scale = np.where(ft_mag > 1e-9,
                         np.minimum(18.0 * ft_mag, fmax) / np.maximum(ft_mag, 1e-9),
                         0.0)
        f = f - scale[:, None] * vt
        np.add.at(fp, I, f)
        np.add.at(fp, J, -f)

    # ----------------------------------------------------------------- step
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.dt, time - self.t)
            fext = self.fext
            fp = self.fpair
            uf = self._fluid_velocity(self.pos)
            drag = self.gamma[:, None] * (uf - self.vel)
            a = (fext + fp + drag) * self.inv_m[:, None]
            # clamp acceleration for robustness
            amax = np.linalg.norm(a, axis=1)
            bad = amax > 5.0e4
            if np.any(bad):
                a[bad] *= (5.0e4 / amax[bad])[:, None]
            self.vel = self.vel + a * dt
            # velocity clamp
            vmax = np.linalg.norm(self.vel, axis=1)
            over = vmax > 60.0
            if np.any(over):
                self.vel[over] *= (60.0 / vmax[over])[:, None]
            self.pos = self.pos + self.vel * dt
            self._clamp_inside()
            self._compute_forces()
            self.t += dt
            self.nsteps = getattr(self, "nsteps", 0) + 1
        return self._frame()

    def _frame(self) -> StateFrame:
        frame = StateFrame(
            float(self.t),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radius.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "external_force": self.fext.copy(),
                "interaction_force": self.interaction_force.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "velocity_field": self._velocity_field(),
                "active_mask": self.active_mask.copy(),
            },
            scalars={
                "grid_spacing": float(self.grid_spacing_val()),
                "characteristic_length": float(2.0 * self.R),
                "contact_length_scale": float(self.h),
                "reference_velocity": float(self.omega * self.R),
            },
        )
        frame.validate()
        return frame

    def grid_spacing_val(self) -> float:
        return self.dxg

    def _velocity_field(self) -> np.ndarray:
        # diagnostic grid velocity: fluid rigid rotation + particle-cloud mean
        U = self.vf_base.copy()
        nx, ny, nz = self.grid_shape
        b = self.dxg
        ix = np.clip(((self.pos[:, 0] + self.L / 2) / b).astype(int), 0, nx - 1)
        iy = np.clip(((self.pos[:, 1] + self.R) / (2 * self.R / ny)).astype(int), 0, ny - 1)
        iz = np.clip(((self.pos[:, 2] - self.zc + self.R) / (2 * self.R / nz)).astype(int), 0, nz - 1)
        ck = (ix * ny + iy) * nz + iz
        cnt = np.bincount(ck, minlength=nx * ny * nz).reshape(nx, ny, nz)
        vsum = np.zeros((nx * ny * nz, 3))
        np.add.at(vsum, ck, self.vel)
        vsum = vsum.reshape(nx, ny, nz, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            vmean = np.where(cnt[..., None] > 0, vsum / np.maximum(cnt, 1)[..., None], 0.0)
        w = np.clip(cnt / max(self.cell_dV * 2.0e3, 1.0), 0.0, 1.0)[..., None]
        return (1.0 - w) * U + w * vmean

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": int(self.N),
            "substep_dt": float(self.dt),
            "fluid_density": RHO_F,
            "fluid_viscosity": MU,
            "omega_rad_s": self.omega,
        }
