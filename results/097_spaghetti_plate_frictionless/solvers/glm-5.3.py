"""Frictionless spreading of 31 spaghetti strands in a plate (scene 097).

Physical model
--------------
Each strand is a discrete elastic rod (33 point masses) with

  * linearly elastic stretching along the rod plus Kelvin-Voigt damping,
  * a tangent-difference bending energy  E = kb/2 * |t_{j+1} - t_j|^2  that
    resists deviation from the naturally straight rest shape,
  * viscous damping of the transverse relative velocity of neighbours
    (material damping of the pasta).

Collisions are frictionless by construction: strand-strand and strand-plate
reactions act only along the contact normal (penalty stiffness with normal
viscous damping, i.e. perfectly inelastic normal impacts, zero tangential
force, static and kinetic friction coefficients exactly zero).  The plate is
a static axisymmetric container; its interior surface is the frozen
piecewise-linear profile (r, z) and collision uses a height-field test with
the finite strand radius.  Outside the plate rim the (implicit) table at
z = 0 supports anything that escapes, so states stay finite.

Time integration is fixed-step semi-implicit (symplectic) Euler with
dt = 5e-5 s (400 substeps per 0.02 s sample), stable for the stiffest
bending / contact modes of this discretisation.  Every sample is produced
by explicit physical time stepping from the previous one; no keyframes,
animation tables or scripted motion are used anywhere.
"""
from __future__ import annotations

import numpy as np

# Half of the 3x3x3 neighbourhood: the zero offset plus all offsets that are
# lexicographically positive.  Using this half-set yields every unordered
# cross-cell pair exactly once.
_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) == (0, 0, 0) or (dx, dy, dz) > (0, 0, 0)
)


class BenchmarkSolver:
    """Explicit frictionless elastic-rod solver for the spaghetti plate."""

    DT = 5.0e-5           # physical substep (s)
    REFRESH = 8           # contact-pair re-detection interval (substeps)
    CELL = 0.024          # hash cell size (m)
    CUTOFF = 0.024        # candidate-pair distance cutoff (m)
    VMAX = 30.0           # emergency velocity clamp (m/s), never reached normally

    # ---- material / numerics parameters -------------------------------
    EI = 0.25             # bending stiffness (N m^2)
    K_STRETCH = 500.0     # stretching stiffness (N/m)
    C_STRETCH = 0.15      # stretching damping (N s/m)
    C_PERP = 0.10         # transverse neighbour damping (N s/m)
    C_AIR = 0.25          # global drag (1/s)
    K_PLATE = 2500.0      # strand-plate penalty stiffness (N/m)
    C_PLATE = 0.35        # strand-plate normal damping (N s/m)
    K_SS = 2500.0         # strand-strand penalty stiffness (N/m)
    C_SS = 0.30           # strand-strand normal damping (N s/m)

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()

        self.p = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.m = np.array(frame0.mass, dtype=float)
        self.body = np.array(frame0.body_id, dtype=np.int64)
        self.n = self.p.shape[0]
        self._inv_m = 1.0 / self.m

        self.S = int(cfg["strand_count"])
        self.V = int(cfg["strand"]["vertex_count"])
        self.R = float(cfg["strand"]["radius"])
        self.d_contact = 2.0 * self.R
        self.g = np.array(cfg["gravity"], dtype=float)

        # --- rod topology (strands are contiguous vertex blocks) --------
        base = (np.arange(self.S) * self.V)[:, None, None]
        el = np.stack([np.arange(self.V - 1), np.arange(1, self.V)], axis=1)
        self.edges = (el[None, :, :] + base).reshape(-1, 2)
        tl = np.stack(
            [np.arange(self.V - 2), np.arange(1, self.V - 1), np.arange(2, self.V)],
            axis=1,
        )
        self.triples = (tl[None, :, :] + base).reshape(-1, 3)
        self.e0 = self.edges[:, 0]
        self.e1 = self.edges[:, 1]
        self.L0 = np.linalg.norm(self.p[self.e1] - self.p[self.e0], axis=1)
        self.h = float(np.mean(self.L0))
        self.kb = self.EI / self.h  # discrete bending coefficient (N m/rad)

        # --- collision points: vertices + edge midpoints ---------------
        # Midpoint samples give a continuous envelope of spheres of radius R
        # along each strand, so side-by-side contacts are detected for any
        # axial offset between neighbouring strands.
        self.nP = self.n + self.edges.shape[0]
        self.ci = np.concatenate(
            [np.arange(self.n), self.e0, self.e1]
        ).astype(np.int64)
        self.cpnt = np.concatenate(
            [np.arange(self.n), np.arange(self.n, self.nP), np.arange(self.n, self.nP)]
        ).astype(np.int64)
        self.cw = np.concatenate(
            [np.ones(self.n), np.full(self.edges.shape[0], 0.5), np.full(self.edges.shape[0], 0.5)]
        )
        self.body_pt = np.concatenate([self.body, self.body[self.e0]])
        self._q = np.zeros((self.nP, 3))
        self._vq = np.zeros((self.nP, 3))

        # --- plate interior height field (frozen profile) ---------------
        # segments: flat centre, shallow floor, inner slope, rim, rim top,
        # vertical outer wall, table outside the plate.
        self.pr = np.array([0.0, 0.10, 0.88, 1.05, 1.14, 1.195, 1.196, 1.0e9])
        self.pz = np.array([0.075, 0.075, 0.08, 0.135, 0.22, 0.22, 0.0, 0.0])

        # --- constant forces / budget bookkeeping ------------------------
        self.Fg = self.m[:, None] * self.g[None, :]
        mass_per_strand = np.bincount(self.body, weights=self.m, minlength=self.S)
        self.imp_strand = np.zeros((self.S, 3))     # accumulated contact impulse per strand
        self.imp_global = np.zeros(3)               # plate impulse on the whole bundle
        self.G = self.S + 1
        weights = np.zeros((self.G, self.n))
        for s in range(self.S):
            weights[s, self.body == s] = 1.0
        weights[self.S, :] = 1.0
        self.budget_weights = weights
        gf = mass_per_strand[:, None] * self.g[None, :]
        self.budget_gravity = np.concatenate(
            [gf, gf.sum(axis=0)[None, :]], axis=0
        )[:, None, :]  # (G, K=1, 3)

        # --- initial attributes / scalars to echo every frame ------------
        self.attrs0 = {k: np.array(v) for k, v in frame0.attributes.items()}
        total_mass = float(self.m.sum())
        self.scalars0 = dict(frame0.scalars)
        self.scalars0["friction_stick_speed_tolerance"] = 5.0e-3
        self.scalars0["reference_force"] = float(total_mass * 9.81)
        self.scalars0["reference_velocity"] = 1.0
        self.scalars0["reference_momentum"] = float(total_mass * 1.0)

        self.pair_a = np.zeros(0, dtype=np.int64)
        self.pair_b = np.zeros(0, dtype=np.int64)
        self._step = 0
        self._good_p = self.p.copy()
        self._good_v = self.v.copy()

    # ------------------------------------------------------------------
    # plate height field
    # ------------------------------------------------------------------
    def _plate_height(self, r):
        i = np.clip(np.searchsorted(self.pr, r, side="right") - 1, 0, len(self.pr) - 2)
        dr = self.pr[i + 1] - self.pr[i]
        t = (r - self.pr[i]) / dr
        h = self.pz[i] + t * (self.pz[i + 1] - self.pz[i])
        hp = (self.pz[i + 1] - self.pz[i]) / dr
        return h, hp

    def _points_kinematics(self):
        q = self._q
        vq = self._vq
        q[: self.n] = self.p
        np.add(self.p[self.e0], self.p[self.e1], out=q[self.n :])
        q[self.n :] *= 0.5
        vq[: self.n] = self.v
        np.add(self.v[self.e0], self.v[self.e1], out=vq[self.n :])
        vq[self.n :] *= 0.5
        return q, vq

    # ------------------------------------------------------------------
    # contact forces (frictionless: normal direction only)
    # ------------------------------------------------------------------
    def _plate_force(self, q, vq):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        r = np.hypot(x, y)
        h, hp = self._plate_height(r)
        pen = h + self.R - z
        rr = np.maximum(r, 1.0e-9)
        nx = -hp * x / rr
        ny = -hp * y / rr
        nz = np.ones_like(hp)
        nn = np.sqrt(nx * nx + ny * ny + 1.0)
        nx /= nn
        ny /= nn
        nz /= nn
        vn = vq[:, 0] * nx + vq[:, 1] * ny + vq[:, 2] * nz
        fn = self.K_PLATE * pen - self.C_PLATE * vn
        fn = np.where(pen > 0.0, np.maximum(fn, 0.0), 0.0)
        Fp = np.stack([fn * nx, fn * ny, fn * nz], axis=1)
        normals = np.stack([nx, ny, nz], axis=1)
        return Fp, pen, normals, fn

    def _ss_force(self, q, vq):
        Fp = np.zeros((self.nP, 3))
        a, b = self.pair_a, self.pair_b
        if a.size == 0:
            return Fp
        d = q[a] - q[b]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        msk = dist < self.d_contact
        if not msk.any():
            return Fp
        idx = np.nonzero(msk)[0]
        dd = np.maximum(dist[idx], 1.0e-9)
        n = d[idx] / dd[:, None]
        pen = self.d_contact - dd
        vn = np.einsum("ij,ij->i", vq[a[idx]] - vq[b[idx]], n)
        fn = np.maximum(self.K_SS * pen - self.C_SS * vn, 0.0)
        aa = a[idx]
        bb = b[idx]
        for c in range(3):
            Fp[:, c] += np.bincount(aa, weights=fn * n[:, c], minlength=self.nP)
            Fp[:, c] -= np.bincount(bb, weights=fn * n[:, c], minlength=self.nP)
        return Fp

    def _to_vertices(self, Fp):
        contrib = self.cw[:, None] * Fp[self.cpnt]
        Fv = np.empty((self.n, 3))
        for c in range(3):
            Fv[:, c] = np.bincount(self.ci, weights=contrib[:, c], minlength=self.n)
        return Fv

    # ------------------------------------------------------------------
    # internal rod forces (stretching, bending, damping)
    # ------------------------------------------------------------------
    def _internal(self, p, v):
        F = np.zeros((self.n, 3))
        a, b = self.e0, self.e1
        d = p[b] - p[a]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        Ls = np.maximum(L, 1.0e-9)
        t = d / Ls[:, None]
        rel = v[b] - v[a]
        rate = np.einsum("ij,ij->i", rel, t)
        fmag = self.K_STRETCH * (L - self.L0) + self.C_STRETCH * rate
        stretch_a = fmag[:, None] * t
        vperp = rel - rate[:, None] * t
        perp_a = self.C_PERP * vperp
        F[a] += stretch_a + perp_a
        F[b] -= stretch_a + perp_a

        ia, ib, ic = self.triples[:, 0], self.triples[:, 1], self.triples[:, 2]
        e1 = p[ib] - p[ia]
        e2 = p[ic] - p[ib]
        L1 = np.maximum(np.sqrt(np.einsum("ij,ij->i", e1, e1)), 1.0e-9)
        L2 = np.maximum(np.sqrt(np.einsum("ij,ij->i", e2, e2)), 1.0e-9)
        t1 = e1 / L1[:, None]
        t2 = e2 / L2[:, None]
        dvec = t2 - t1
        # P1 d and P2 d with P = (I - t t^T)/L
        P1d = (dvec - t1 * np.einsum("ij,ij->i", t1, dvec)[:, None]) / L1[:, None]
        P2d = (dvec - t2 * np.einsum("ij,ij->i", t2, dvec)[:, None]) / L2[:, None]
        F[ia] -= self.kb * P1d
        F[ib] += self.kb * (P1d + P2d)
        F[ic] -= self.kb * P2d
        return F

    # ------------------------------------------------------------------
    # broad phase: spatial hash over collision points
    # ------------------------------------------------------------------
    def _detect(self, q):
        nP = self.nP
        ic = np.floor(q / self.CELL).astype(np.int64)
        np.clip(ic, -500, 499, out=ic)
        ic += 500
        key = ic[:, 0] * 4194304 + ic[:, 1] * 2048 + ic[:, 2]
        order = np.argsort(key, kind="stable")
        sk = key[order]
        uniq, starts, counts = np.unique(sk, return_index=True, return_counts=True)
        nu = uniq.shape[0]
        ar = np.arange(nP)
        out_a, out_b = [], []
        for off in _OFFSETS:
            ox, oy, oz = off
            okey = (ic[:, 0] + ox) * 4194304 + (ic[:, 1] + oy) * 2048 + (ic[:, 2] + oz)
            pos = np.searchsorted(uniq, okey)
            valid = pos < nu
            posc = np.where(valid, pos, 0)
            valid &= uniq[posc] == okey
            cnt = np.where(valid, counts[posc], 0)
            tot = int(cnt.sum())
            if tot == 0:
                continue
            qa = np.repeat(ar, cnt)
            base = np.repeat(starts[posc], cnt)
            offq = np.repeat(np.cumsum(cnt) - cnt, cnt)
            tb = order[np.arange(tot) - offq + base]
            if off == (0, 0, 0):
                m = qa < tb
                qa, tb = qa[m], tb[m]
            out_a.append(qa)
            out_b.append(tb)
        if not out_a:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        a = np.concatenate(out_a)
        b = np.concatenate(out_b)
        m = self.body_pt[a] != self.body_pt[b]
        a, b = a[m], b[m]
        d = q[a] - q[b]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        m = dist < self.CUTOFF
        return a[m].astype(np.int64), b[m].astype(np.int64)

    # ------------------------------------------------------------------
    # one physical substep
    # ------------------------------------------------------------------
    def _substep(self):
        q, vq = self._points_kinematics()
        if self._step % self.REFRESH == 0:
            self.pair_a, self.pair_b = self._detect(q)

        Fp_plate, _, _, _ = self._plate_force(q, vq)
        Fp_ss = self._ss_force(q, vq)
        Fv_plate = self._to_vertices(Fp_plate)
        Fv_ss = self._to_vertices(Fp_ss)

        F = self.Fg - (self.C_AIR * self.m)[:, None] * self.v
        F += Fv_plate
        F += Fv_ss
        F += self._internal(self.p, self.v)

        self.v += (F * self._inv_m[:, None]) * self.DT
        np.clip(self.v, -self.VMAX, self.VMAX, out=self.v)
        self.p += self.v * self.DT

        # exact contact-impulse bookkeeping (gravity stays in the force channel)
        dt = self.DT
        fc = Fv_plate + Fv_ss
        for c in range(3):
            self.imp_strand[:, c] += np.bincount(
                self.body, weights=fc[:, c], minlength=self.S
            ) * dt
        self.imp_global += Fv_plate.sum(axis=0) * dt

        self._step += 1
        if self._step % 1000 == 0:
            if np.isfinite(self.p).all() and np.isfinite(self.v).all():
                self._good_p = self.p.copy()
                self._good_v = self.v.copy()
            else:  # emergency: restore last finite state, damp it out
                self.p[:] = self._good_p
                self.v[:] = 0.5 * self._good_v
                self.pair_a = np.zeros(0, dtype=np.int64)
                self.pair_b = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # StateFrame assembly
    # ------------------------------------------------------------------
    def _make_frame(self, time: float):
        q, vq = self._points_kinematics()

        # plate contacts at this instant
        _, pen_p, normals_p, fn_p = self._plate_force(q, vq)
        msk_p = pen_p > 1.0e-7
        n_plate = normals_p[msk_p]
        f_plate = fn_p[msk_p]
        v_plate = vq[msk_p]

        # strand-strand contacts at this instant
        n_list = [n_plate]
        f_list = [f_plate]
        v_list = [v_plate]
        a, b = self.pair_a, self.pair_b
        if a.size:
            d = q[a] - q[b]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            msk = dist < self.d_contact
            if msk.any():
                idx = np.nonzero(msk)[0]
                dd = np.maximum(dist[idx], 1.0e-9)
                nrm = d[idx] / dd[:, None]
                pen = self.d_contact - dd
                vrel = vq[a[idx]] - vq[b[idx]]
                vn = np.einsum("ij,ij->i", vrel, nrm)
                fss = np.maximum(self.K_SS * pen - self.C_SS * vn, 0.0)
                n_list.append(nrm)
                f_list.append(fss)
                v_list.append(vrel)

        normals = np.concatenate(n_list) if n_list else np.zeros((0, 3))
        fnormal = np.concatenate(f_list) if f_list else np.zeros(0)
        relvel = np.concatenate(v_list) if v_list else np.zeros((0, 3))
        C = normals.shape[0]

        imp = np.concatenate([self.imp_strand, self.imp_global[None, :]], axis=0)
        fields = {
            "budget_cumulative_external_impulse": imp,
            "budget_cumulative_inflow_momentum": np.zeros((self.G, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((self.G, 3)),
            "budget_domain_weights": self.budget_weights,
            "budget_external_force": self.budget_gravity,
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": np.ones(C),
            "friction_dynamic_coefficient": np.zeros(C),
            "friction_normal": normals,
            "friction_normal_force": fnormal,
            "friction_relative_velocity": relvel,
            "friction_static_coefficient": np.zeros(C),
            "friction_tangential_force": np.zeros((C, 3)),
        }

        attributes = {k: w.copy() for k, w in self.attrs0.items()}
        attributes["mass_quantity"] = self.m.copy()

        from worldsolver.contracts import StateFrame

        return StateFrame(
            time=float(time),
            position=self.p.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=dict(self.scalars0),
        )

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = int(round(float(time) / self.DT))
        while self._step < target:
            self._substep()
        return self._make_frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        speed = np.linalg.norm(self.v, axis=1)
        q, _ = self._points_kinematics()
        r = np.hypot(q[:, 0], q[:, 1])
        h, _ = self._plate_height(r)
        plate_contacts = int(np.count_nonzero(h + self.R - q[:, 2] > 1.0e-7))
        return {
            "method": "semi-implicit Euler elastic rods, frictionless penalty contacts",
            "substep_dt": self.DT,
            "substeps_executed": int(self._step),
            "bending_stiffness_EI": self.EI,
            "strand_strand_candidate_pairs": int(self.pair_a.size),
            "final_max_speed": float(speed.max()),
            "final_plate_contact_points": plate_contacts,
            "final_bundle_center_z": float(self.p[:, 2].mean()),
            "deterministic": True,
        }
