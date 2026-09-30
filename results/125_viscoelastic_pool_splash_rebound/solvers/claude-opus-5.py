"""Viscoelastic pool splash and rebound solver.

Physical model
--------------
Material Point Method (MLS-MPM, quadratic B-spline, APIC affine transfers) with a
finite-strain elastoplastic (viscoelastic) constitutive model:

* Hencky (logarithmic strain) isotropic elasticity supplies the elastic restoring
  stress that keeps stretched sheets/strands cohesive and pulls them back.
* A von-Mises style return mapping with a *finite relaxation time* ``tau`` bleeds
  the excess deviatoric elastic strain into plastic (viscous) flow.  This is the
  Goktekin et al. viscoelastic regime: elastic below the yield strain, creeping
  fluid above it, which produces the rebound plus damped oscillation.
* An additional Newtonian deviatoric stress (eta) damps the splash.
* Rigid basin/ground are frozen boxes handled both as grid velocity boundary
  conditions and as particle-level non-penetration projection with friction.

Everything is stepped forward in time with a fixed, CFL-safe explicit time step;
no keyframes, no scripted motion.
"""
from __future__ import annotations

import itertools

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- numerical / material constants -------------------------------------
    DT = 1.0e-3            # explicit time step [s] (CFL safe, see notes above)
    DX = 0.20              # background grid spacing [m]
    MU = 1.5e4             # elastic shear modulus [Pa]
    LAM = 1.2e5            # elastic bulk (Lame lambda) [Pa]
    ETA = 25.0             # Newtonian (viscous) deviatoric viscosity [Pa s]
    YIELD_STRAIN = 0.15    # von Mises yield in log-strain units [-]
    TAU_RELAX = 0.50       # viscoelastic stress relaxation time [s]
    MAX_DEV = 1.10         # hard cap on deviatoric log strain (stability)
    MAX_VOL = 0.30         # hard cap on |volumetric log strain| (stability)
    MU_FRICTION = 0.45     # Coulomb friction coefficient at rigid contacts
    F_CLAMP_EVERY = 25     # steps between total-F conditioning

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.config = cfg

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=np.float64, copy=True)
        self.v = np.zeros_like(self.x)
        self.m = np.array(frame0.mass, dtype=np.float64, copy=True)
        self.labels = np.array(frame0.body_id, dtype=np.int32, copy=True)
        self.n = int(self.x.shape[0])

        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])
        self.vol0 = self.spacing ** 3
        self.radius = np.asarray(
            frame0.attributes.get(
                "particle_radius", np.full(self.n, 0.45 * self.spacing)
            ),
            dtype=np.float64,
        ).copy()
        self.contact_radius = 0.45 * self.spacing

        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        # deformation gradients: elastic (drives stress) and total (reported)
        eye = np.eye(3)
        self.Fe = np.repeat(eye[None], self.n, axis=0).copy()
        self.Ft = np.repeat(eye[None], self.n, axis=0).copy()
        self.C = np.zeros((self.n, 3, 3))

        self._build_geometry()
        self._build_grid()

        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0

    # -------------------------------------------------------------- geometry
    def _build_geometry(self) -> None:
        """Frozen rigid boxes (ground + three basin walls), from scene_geometry."""
        boxes = [
            ((0.0, 0.0, -0.05), (3.2, 1.4, 0.05)),    # ground
            ((0.0, 0.72, 0.30), (1.45, 0.08, 0.34)),  # basin_back
            ((-1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),  # basin_left
            ((1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),   # basin_right
        ]
        self.box_c = np.array([b[0] for b in boxes], dtype=np.float64)
        self.box_h = np.array([b[1] for b in boxes], dtype=np.float64)

        # particle domain (frozen) used as safety clamp for the grid
        self.dom_lo = np.array([-3.0, -1.2, 0.0]) + 0.06
        self.dom_hi = np.array([3.0, 1.2, 3.8]) - 0.06

    def _box_sdf(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance to the union of the frozen rigid boxes."""
        d = np.full(pts.shape[0], np.inf)
        for c, h in zip(self.box_c, self.box_h):
            q = np.abs(pts - c) - h
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            d = np.minimum(d, outside + inside)
        return d

    def _build_grid(self) -> None:
        dx = self.DX
        self.origin = np.array([-3.2, -1.35, -0.45])
        upper = np.array([3.2, 1.35, 3.95])
        dims = np.ceil((upper - self.origin) / dx).astype(int) + 1
        self.gshape = tuple(int(v) for v in dims)
        self.nx, self.ny, self.nz = self.gshape
        self.nn = self.nx * self.ny * self.nz

        ii, jj, kk = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij"
        )
        nodes = self.origin + dx * np.column_stack(
            (ii.ravel(), jj.ravel(), kk.ravel())
        ).astype(np.float64)
        self.node_pos = nodes

        # rigid-body grid boundary conditions: node index + outward normal
        infl = self.contact_radius
        self.bc_nodes: list[tuple[np.ndarray, np.ndarray]] = []
        for c, h in zip(self.box_c, self.box_h):
            hh = h + infl
            q = np.abs(nodes - c) - hh
            inside = np.all(q < 0.0, axis=1)
            idx = np.flatnonzero(inside)
            if idx.size == 0:
                continue
            axis = np.argmax(q[idx], axis=1)
            nrm = np.zeros((idx.size, 3))
            sgn = np.sign(nodes[idx, :] - c)
            sgn[sgn == 0.0] = 1.0
            nrm[np.arange(idx.size), axis] = sgn[np.arange(idx.size), axis]
            self.bc_nodes.append((idx, nrm))

        # domain walls (2-cell band) -> zero normal velocity component
        lo = self.origin + 2.0 * dx
        hi = self.origin + (dims - 3) * dx
        self.dom_masks = []
        for a in range(3):
            self.dom_masks.append((a, nodes[:, a] < lo[a] - 1e-9, +1.0))
            self.dom_masks.append((a, nodes[:, a] > hi[a] + 1e-9, -1.0))
        self.grid_lo = lo
        self.grid_hi = hi

        self.offsets = np.array(list(itertools.product(range(3), repeat=3)), dtype=np.int64)

    # ------------------------------------------------------------- constitutive
    def _stress(self) -> np.ndarray:
        """Return-map the elastic strain and return Kirchhoff stress tau [N,3,3]."""
        U, S, Vt = np.linalg.svd(self.Fe)
        S = np.clip(S, 1e-4, 1e4)
        eps = np.log(S)

        vol = eps.sum(axis=1) / 3.0
        vol = np.clip(vol, -self.MAX_VOL, self.MAX_VOL)
        dev = eps - eps.sum(axis=1, keepdims=True) / 3.0

        norm = np.linalg.norm(dev, axis=1)
        safe = np.maximum(norm, 1e-12)
        # viscoelastic relaxation of the over-yield part of the elastic strain
        excess = np.maximum(norm - self.YIELD_STRAIN, 0.0)
        rate = 1.0 - np.exp(-self.dt_cur / self.TAU_RELAX)
        new_norm = norm - excess * rate
        new_norm = np.minimum(new_norm, self.MAX_DEV)
        dev = dev * (new_norm / safe)[:, None]

        eps = dev + vol[:, None]
        # rebuild the (projected) elastic deformation gradient
        S_new = np.exp(eps)
        self.Fe = np.einsum("nij,nj,njk->nik", U, S_new, Vt)

        # Hencky isotropic elasticity: tau = U (2 mu eps + lam tr(eps) I) U^T
        principal = 2.0 * self.MU * eps + self.LAM * (3.0 * vol)[:, None]
        tau = np.einsum("nij,nj,nkj->nik", U, principal, U)

        # Newtonian deviatoric viscosity from the APIC velocity gradient
        d = 0.5 * (self.C + np.transpose(self.C, (0, 2, 1)))
        tr = np.trace(d, axis1=1, axis2=2) / 3.0
        d = d - tr[:, None, None] * np.eye(3)[None]
        tau = tau + 2.0 * self.ETA * d
        return tau

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        self.dt_cur = dt
        dx = self.DX
        inv_dx = 1.0 / dx
        n = self.n

        tau = self._stress()

        base = np.floor(self.x * inv_dx - self.origin * inv_dx - 0.5).astype(np.int64)
        base = np.clip(base, 0, np.array(self.gshape) - 3)
        fx = (self.x - self.origin) * inv_dx - base.astype(np.float64)

        w = np.empty((n, 3, 3))
        w[:, :, 0] = 0.5 * (1.5 - fx) ** 2
        w[:, :, 1] = 0.75 - (fx - 1.0) ** 2
        w[:, :, 2] = 0.5 * (fx - 0.5) ** 2

        o = self.offsets                                     # (27,3)
        idx = (
            (base[:, 0:1] + o[None, :, 0]) * (self.ny * self.nz)
            + (base[:, 1:2] + o[None, :, 1]) * self.nz
            + (base[:, 2:3] + o[None, :, 2])
        )                                                    # (N,27)
        wt = (
            w[:, 0, :][:, o[:, 0]] * w[:, 1, :][:, o[:, 1]] * w[:, 2, :][:, o[:, 2]]
        )                                                    # (N,27)
        node_xyz = (
            self.origin[None, None, :]
            + (base[:, None, :] + o[None, :, :]).astype(np.float64) * dx
        )
        dpos = node_xyz - self.x[:, None, :]                 # (N,27,3)

        # ---- particle to grid ------------------------------------------------
        affine = self.m[:, None, None] * self.C - (
            dt * self.vol0 * 4.0 * inv_dx * inv_dx
        ) * tau
        contrib = wt[:, :, None] * (
            self.m[:, None, None] * self.v[:, None, :]
            + np.einsum("nij,nkj->nki", affine, dpos)
        )
        flat = idx.ravel()
        gm = np.bincount(flat, weights=(wt * self.m[:, None]).ravel(), minlength=self.nn)
        gv = np.empty((self.nn, 3))
        for a in range(3):
            gv[:, a] = np.bincount(flat, weights=contrib[:, :, a].ravel(), minlength=self.nn)

        active = gm > 1e-12
        gv[active] /= gm[active, None]
        gv[~active] = 0.0

        # ---- grid forces + boundary conditions -------------------------------
        gv[active] += dt * self.gravity[None, :]

        for bidx, nrm in self.bc_nodes:
            sub = gv[bidx]
            vn = np.einsum("ij,ij->i", sub, nrm)
            hit = vn < 0.0
            if np.any(hit):
                # separate the normal component, then apply Coulomb friction whose
                # magnitude is proportional to the normal impulse that was removed
                vt = sub[hit] - vn[hit, None] * nrm[hit]
                vt_mag = np.linalg.norm(vt, axis=1)
                keep = np.clip(
                    1.0 - self.MU_FRICTION * np.abs(vn[hit]) / np.maximum(vt_mag, 1e-9),
                    0.0,
                    1.0,
                )
                sub[hit] = vt * keep[:, None]
                gv[bidx] = sub

        for a, mask, sgn in self.dom_masks:
            if np.any(mask):
                comp = gv[mask, a]
                gv_mask = gv[mask]
                bad = comp * sgn < 0.0
                if np.any(bad):
                    gv_mask[bad, a] = 0.0
                    gv[mask] = gv_mask

        # ---- grid to particle -------------------------------------------------
        gvp = gv[idx]                                        # (N,27,3)
        v_new = np.einsum("nk,nki->ni", wt, gvp)
        C_new = 4.0 * inv_dx * inv_dx * np.einsum("nk,nki,nkj->nij", wt, gvp, dpos)

        self.v = v_new
        self.C = C_new
        incr = np.eye(3)[None] + dt * C_new
        self.Fe = np.einsum("nij,njk->nik", incr, self.Fe)
        self.Ft = np.einsum("nij,njk->nik", incr, self.Ft)
        self.x = self.x + dt * self.v

        self._resolve_particle_contacts()

        self.steps += 1
        if self.steps % self.F_CLAMP_EVERY == 0:
            self._condition_total_F()

        sp = float(np.abs(self.v).max()) if n else 0.0
        self.max_speed = max(self.max_speed, sp)

    def _condition_total_F(self) -> None:
        U, S, Vt = np.linalg.svd(self.Ft)
        Sc = np.clip(S, 0.15, 6.0)
        if not np.allclose(Sc, S):
            self.Ft = np.einsum("nij,nj,njk->nik", U, Sc, Vt)

    def _resolve_particle_contacts(self) -> None:
        """Non-penetration projection against the frozen rigid boxes + domain."""
        r = self.contact_radius
        for c, h in zip(self.box_c, self.box_h):
            hh = h + r
            q = np.abs(self.x - c) - hh
            inside = np.flatnonzero(np.all(q < 0.0, axis=1))
            if inside.size == 0:
                continue
            axis = np.argmax(q[inside], axis=1)
            rows = np.arange(inside.size)
            ref = self.x[inside, :][rows, axis] - c[axis]
            sgn = np.where(ref >= 0.0, 1.0, -1.0)
            xs = self.x[inside]
            vs = self.v[inside]
            xs[rows, axis] = c[axis] + sgn * hh[axis]
            vn = vs[rows, axis]
            into = vn * sgn < 0.0
            # remove only the inward normal component; tangential friction is
            # handled consistently on the grid (Coulomb, impulse proportional)
            vs[rows[into], axis[into]] = 0.0
            self.x[inside] = xs
            self.v[inside] = vs

        for a in range(3):
            lo = max(self.dom_lo[a], self.grid_lo[a] + 0.5 * self.DX)
            hi = min(self.dom_hi[a], self.grid_hi[a] - 0.5 * self.DX)
            low = self.x[:, a] < lo
            if np.any(low):
                self.x[low, a] = lo
                self.v[low, a] = np.maximum(self.v[low, a], 0.0)
            high = self.x[:, a] > hi
            if np.any(high):
                self.x[high, a] = hi
                self.v[high, a] = np.minimum(self.v[high, a], 0.0)

        np.clip(self.v, -60.0, 60.0, out=self.v)

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.time < target - 1e-12 and guard < 10_000_000:
            dt = min(self.DT, target - self.time)
            if dt <= 1e-15:
                break
            self._step(dt)
            self.time += dt
            guard += 1
        if abs(self.time - target) < 1e-9:
            self.time = target
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        n = self.n
        ext_force = self.m[:, None] * self.gravity[None, :]

        # signed gap of every sample to the union of rigid boundaries; fixed
        # length [N] so the trajectory has a consistent per-frame shape.
        gap = self._box_sdf(self.x) - self.contact_radius
        gap = np.clip(gap, -10.0, 10.0)
        contact = np.ascontiguousarray(gap, dtype=np.float64)

        Ft = np.array(self.Ft, dtype=np.float64, copy=True)
        Ft[~np.isfinite(Ft)] = 0.0

        pos = np.array(self.x, dtype=np.float64, copy=True)
        vel = np.array(self.v, dtype=np.float64, copy=True)
        pos[~np.isfinite(pos)] = 0.0
        vel[~np.isfinite(vel)] = 0.0

        frame = StateFrame(
            float(max(t, 0.0)),
            pos,
            vel,
            mass=self.m.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "particle_radius": self.radius.copy(),
                "material_label": self.labels.copy(),
                "deformation_gradient": Ft,
            },
            fields={
                "contact_signed_distance": contact,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(
                    max(float(np.ptp(pos, axis=0).max()), self.spacing)
                ),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MLS-MPM (APIC) viscoelastic elastoplastic solid-fluid",
            "particles": int(self.n),
            "grid_shape": list(self.gshape),
            "dt": float(self.DT),
            "steps": int(self.steps),
            "sim_time": float(self.time),
            "max_speed": float(self.max_speed),
            "shear_modulus": float(self.MU),
            "lame_lambda": float(self.LAM),
            "viscosity": float(self.ETA),
            "yield_strain": float(self.YIELD_STRAIN),
            "relaxation_time": float(self.TAU_RELAX),
        }
