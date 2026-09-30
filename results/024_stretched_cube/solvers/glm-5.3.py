"""Physical solver for scene 024_stretched_cube (Stretched Cube).

A deformable cube is clamped between two kinematic grips.  The grips translate
apart along +/-x with a smooth ramp, stretching the cube by ``stretch_ratio``
by time ``stretch_end``; they hold the stretch and then apply a small
compressive pulse at ``pulse_time`` (smooth sin^2 bump), producing a stable
large-deformation response and a damped subsequent oscillation.

Physics model (implemented from scratch with NumPy/SciPy only):

* The deformable body is the frozen surface triangle mesh.  Its elasticity is
  represented by a damped spring network:
  - structural springs on every mesh edge (surface elasticity), and
  - three families of through-body bracing springs pairing each vertex with
    its mirror image through the cube centre along each axis (volumetric
    bracing that keeps the body from folding through itself).
* Near-incompressibility is enforced with a volumetric pressure term
  (penalty on enclosed-volume change plus volumetric rate damping), which
  produces the lateral necking expected when a rubbery cube is stretched.
* Time integration is damped semi-implicit (symplectic) Euler with fixed
  deterministic substeps small enough for stability of the stiffest local
  mode; spring damping uses the frozen ``damping`` coefficient.
* The grips are kinematic boundary conditions: the cube's end faces are
  rigidly attached to the grips and follow the analytic loading script.
* Collisions with the ground plane and the (moving) grip boxes are resolved
  by projection with Coulomb friction and zero restitution.

Per-vertex deformation gradients are obtained from a weighted least-squares
fit of the local rest->deformed map using the surface neighbours plus the
through-body mirror neighbours (so the fit is full rank in 3D).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# Frozen geometry constants from scene_geometry.yaml (workspace scaffold).
GRIP_HALF = np.array([0.12, 0.55, 0.55])  # grip box half extents
GRIP_X0 = 0.72  # initial grip centre offset along x
GRIP_Z = 0.86   # grip centre height


class BenchmarkSolver:
    """Damped spring-network elastodynamics solver for the stretched cube."""

    SUB_DT = 2.0e-4        # integration substep (s)
    PULSE_T = 0.30         # compressive pulse duration (s)
    PULSE_A = 0.07         # compressive pulse amplitude per grip (m)
    V_MAX = 60.0           # velocity safety clamp (m/s)

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}))
        self.cfg = cfg
        self.seed = int(seed)
        self._steps = 0

        root = Path(__file__).resolve().parents[1]
        verts = None
        faces = None
        asset = root / cfg.get("subject_asset", "assets/subject.npz")
        try:
            with np.load(asset) as data:
                if "vertices" in data.files:
                    verts = np.asarray(data["vertices"], dtype=float)
                if "faces" in data.files:
                    faces = np.asarray(data["faces"], dtype=np.int64)
        except (FileNotFoundError, KeyError):
            verts = None
            faces = None
        if verts is None and hasattr(scene, "vertices"):
            verts = np.asarray(scene.vertices, dtype=float)
        if faces is None:
            # The cube surface is convex; fall back to its convex hull.
            from scipy.spatial import ConvexHull

            faces = np.asarray(ConvexHull(verts).simplices, dtype=np.int64)

        self.q = np.array(verts, dtype=float)          # rest (material) positions
        self.faces = faces
        n = self.q.shape[0]
        self.n = n

        stiffness = float(cfg.get("stiffness", 28.0))
        damping = float(cfg.get("damping", 0.42))
        self.mu = float(cfg.get("friction", 0.5))
        self.e_rest = float(cfg.get("restitution", 0.0))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        total_mass = float(cfg.get("subject_mass", 1.3))
        self.mass = np.full(n, total_mass / n, dtype=float)
        m_node = total_mass / n

        # Kinematic loading-script parameters.
        self.stretch_ratio = float(cfg.get("stretch_ratio", 2.15))
        self.stretch_end = float(cfg.get("stretch_end", 1.25))
        self.pulse_time = float(cfg.get("pulse_time", 1.55))

        lo = self.q.min(axis=0)
        hi = self.q.max(axis=0)
        self.center = 0.5 * (lo + hi)
        self.half = 0.5 * (hi - lo)

        # ---------------- spring network -------------------------------------
        tri_edges = np.sort(
            faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1
        )
        edges = np.unique(tri_edges, axis=0)

        # Through-body mirror pairs (one family per axis).
        tree = cKDTree(self.q)
        ids = np.arange(n)
        mirror_lists = []
        for ax in range(3):
            refl = self.q.copy()
            refl[:, ax] = 2.0 * self.center[ax] - refl[:, ax]
            dist, idx = tree.query(refl)
            keep = (dist < 1e-6) & (idx != ids) & (ids < idx)
            mirror_lists.append(np.stack([ids[keep], idx[keep]], axis=1))
        mirrors = (
            np.concatenate(mirror_lists, axis=0)
            if mirror_lists
            else np.zeros((0, 2), dtype=np.int64)
        )

        self.ia = np.concatenate([edges[:, 0], mirrors[:, 0]]).astype(np.int64)
        self.ib = np.concatenate([edges[:, 1], mirrors[:, 1]]).astype(np.int64)
        self.l0 = np.linalg.norm(self.q[self.ib] - self.q[self.ia], axis=1)
        n_edge = len(edges)
        k_edge = 16.0 * stiffness
        k_mir = 4.0 * stiffness
        self.k = np.concatenate(
            [np.full(n_edge, k_edge), np.full(len(mirrors), k_mir)]
        )
        self.b = (2.0 * damping) * np.sqrt(0.5 * self.k * m_node)

        # Sparse incidence for force accumulation: f = G.T @ (fs * n).
        s_count = len(self.ia)
        rows = np.concatenate([np.arange(s_count), np.arange(s_count)])
        cols = np.concatenate([self.ia, self.ib])
        vals = np.concatenate([np.ones(s_count), -np.ones(s_count)])
        g_mat = sp.coo_matrix((vals, (rows, cols)), shape=(s_count, n)).tocsr()
        self.gt = g_mat.T.tocsr()

        # Sparse face->vertex accumulation for area-weighted normals.
        nf = faces.shape[0]
        w_rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        w_cols = np.tile(np.arange(nf), 3)
        self.wt = sp.coo_matrix(
            (np.full(3 * nf, 0.5), (w_rows, w_cols)), shape=(n, nf)
        ).tocsr()

        p0 = self.q[faces[:, 0]]
        p1 = self.q[faces[:, 1]]
        p2 = self.q[faces[:, 2]]
        vol0_signed = float(np.einsum("ij,ij->", p0, np.cross(p1, p2)) / 6.0)
        self.flip = 1.0 if vol0_signed >= 0.0 else -1.0
        self.vol0 = abs(vol0_signed)

        self.kp = 500.0 * stiffness   # volumetric penalty stiffness
        self.cp = 30.0 * stiffness    # volumetric rate damping

        # ---------------- grip attachment ------------------------------------
        thr = self.half[0] - 0.03
        cx = self.center[0]
        self.att_r = self.q[:, 0] > cx + thr
        self.att_l = self.q[:, 0] < cx - thr
        self.att = self.att_r | self.att_l
        self.dyn = ~self.att

        # ---------------- deformation-gradient fit ---------------------------
        di = np.concatenate(
            [edges[:, 0], edges[:, 1], mirrors[:, 0], mirrors[:, 1]]
        ).astype(np.int64)
        dj = np.concatenate(
            [edges[:, 1], edges[:, 0], mirrors[:, 1], mirrors[:, 0]]
        ).astype(np.int64)
        qd = self.q[dj] - self.q[di]
        ld = np.linalg.norm(qd, axis=1)
        wgt = 1.0 / np.maximum(ld, 1e-9) ** 2
        r_sum = np.zeros((n, 3, 3))
        np.add.at(r_sum, di, (wgt[:, None] * qd)[:, :, None] * qd[:, None, :])
        r_sum += 1e-3 * np.eye(3)[None, :, :]
        self.rinv = np.linalg.inv(r_sum)
        self.fit_i = di
        self.fit_j = dj
        self.fit_q = qd
        self.fit_w = wgt

        # ---------------- dynamic state --------------------------------------
        self.t = 0.0
        self.p = self.q.copy()
        self.v = np.zeros_like(self.q)

    # ------------------------------------------------------- loading script
    def _grip(self, t: float):
        """Grip offset d(t) >= 0 from its initial position and its derivative.

        Smooth ramp to the full stretch by ``stretch_end``, hold, then a small
        compressive sin^2 pulse at ``pulse_time`` that returns to zero.
        """
        d_str = self.half[0] * (self.stretch_ratio - 1.0)
        te = max(self.stretch_end, 1e-9)
        if t <= 0.0:
            d = 0.0
            dv = 0.0
        elif t < te:
            u = t / te
            d = d_str * u * u * (3.0 - 2.0 * u)
            dv = d_str * 6.0 * u * (1.0 - u) / te
        else:
            d = d_str
            dv = 0.0
        t0 = self.pulse_time
        tt = self.PULSE_T
        if t0 <= t <= t0 + tt:
            ph = np.pi * (t - t0) / tt
            d -= self.PULSE_A * np.sin(ph) ** 2
            dv -= self.PULSE_A * (np.pi / tt) * np.sin(2.0 * ph)
        return float(d), float(dv)

    # ------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        q, p, v = self.q, self.p, self.v
        d1, dv1 = self._grip(self.t + dt)

        # Spring forces (structural edges + through-body bracing).
        e = p[self.ib] - p[self.ia]
        ln = np.sqrt(np.einsum("ij,ij->i", e, e))
        np.maximum(ln, 1e-12, out=ln)
        nrm = e / ln[:, None]
        rv = v[self.ib] - v[self.ia]
        vn = np.einsum("ij,ij->i", rv, nrm)
        fs = self.k * (ln - self.l0) + self.b * vn
        f = self.gt @ (fs[:, None] * nrm)

        # Volumetric pressure (near-incompressible penalty + rate damping).
        fa = self.faces
        a0 = p[fa[:, 0]]
        a1 = p[fa[:, 1]]
        a2 = p[fa[:, 2]]
        fn = np.cross(a1 - a0, a2 - a0) * self.flip
        vol = self.flip * float(np.einsum("ij,ij->", a0, np.cross(a1, a2)) / 6.0)
        w = self.wt @ fn
        dvol = float(np.einsum("ij,ij->", v, w)) / 3.0
        press = self.kp * (self.vol0 - vol) / self.vol0
        press -= self.cp * dvol / self.vol0
        f = f + (press / 3.0) * w

        # Body forces (gravity; frozen at zero for this scene).
        f = f + self.mass[:, None] * self.gravity[None, :]

        # Semi-implicit Euler on the free (dynamic) nodes.
        dyn = self.dyn
        v[dyn] += (dt / self.mass[dyn])[:, None] * f[dyn]
        np.clip(v, -self.V_MAX, self.V_MAX, out=v)
        p[dyn] += dt * v[dyn]

        # Prescribed grip-attached nodes follow the kinematic script exactly.
        if self.att_r.any():
            p[self.att_r] = q[self.att_r] + np.array([d1, 0.0, 0.0])
            v[self.att_r] = np.array([dv1, 0.0, 0.0])
        if self.att_l.any():
            p[self.att_l] = q[self.att_l] - np.array([d1, 0.0, 0.0])
            v[self.att_l] = np.array([-dv1, 0.0, 0.0])

        self._collide(d1)
        self.t += dt
        self._steps += 1

    # ------------------------------------------------------------ collisions
    def _collide(self, d: float) -> None:
        p, v = self.p, self.v
        dyn = self.dyn

        # Ground plane (top surface at ground_z).
        mask = dyn & (p[:, 2] < self.ground_z)
        if mask.any():
            approach = np.maximum(-v[mask, 2], 0.0)
            p[mask, 2] = self.ground_z
            v[mask, 2] = -self.e_rest * v[mask, 2]
            self._friction(v, mask, approach)

        # Moving grip boxes.
        for sgn in (1.0, -1.0):
            centre = np.array([sgn * (GRIP_X0 + d), 0.0, GRIP_Z])
            loc = p - centre
            aloc = np.abs(loc)
            inside = (aloc <= GRIP_HALF + 1e-12).all(axis=1)
            mask = dyn & inside
            if not mask.any():
                continue
            idxs = np.nonzero(mask)[0]
            pen = GRIP_HALF - aloc[idxs]
            axis = np.argmin(pen, axis=1)
            for ax in range(3):
                sel = axis == ax
                if not sel.any():
                    continue
                ii = idxs[sel]
                sgn_a = np.where(loc[ii, ax] >= 0.0, 1.0, -1.0)
                p[ii, ax] = centre[ax] + sgn_a * GRIP_HALF[ax]
                approach = np.maximum(-v[ii, ax] * sgn_a, 0.0)
                v[ii, ax] = 0.0
                self._friction(v, ii, approach)

    def _friction(self, v, mask, normal_impulse):
        """Coulomb-style tangential velocity reduction for contacting nodes."""
        if np.count_nonzero(mask) == 0:
            return
        speeds = np.linalg.norm(v[mask], axis=1)
        drop = self.mu * np.maximum(normal_impulse, 0.0)
        fac = np.maximum(0.0, 1.0 - drop / np.maximum(speeds, 1e-12))
        v[mask] = v[mask] * fac[:, None]

    # ----------------------------------------------------------- def. gradient
    def _deformation_gradient(self):
        pd = self.p[self.fit_j] - self.p[self.fit_i]
        vals = (self.fit_w[:, None] * pd)[:, :, None] * self.fit_q[:, None, :]
        s_sum = np.zeros((self.n, 3, 3))
        np.add.at(s_sum, self.fit_i, vals)
        return np.einsum("nij,njk->nik", s_sum, self.rinv)

    # ------------------------------------------------------------- distances
    def _contact_distances(self, t: float):
        p = self.p
        d, _ = self._grip(t)
        gaps = [p[:, 2] - self.ground_z]
        for sgn in (1.0, -1.0):
            centre = np.array([sgn * (GRIP_X0 + d), 0.0, GRIP_Z])
            loc = np.abs(p - centre)
            outside = np.maximum(loc - GRIP_HALF, 0.0)
            sd = np.linalg.norm(outside, axis=1)
            sd += np.minimum(np.max(GRIP_HALF - loc, axis=1), 0.0)
            gaps.append(sd)
        return np.min(np.stack(gaps, axis=0), axis=0)

    # ----------------------------------------------------------------- frame
    def _frame(self, t: float):
        frame = StateFrame(
            time=float(t),
            position=self.p.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.tile(
                    self.mass[:, None] * self.gravity[None, :], (1, 1)
                ),
                "contact_signed_distance": self._contact_distances(t),
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- public API
    def advance_to(self, time: float):
        target = max(float(time), self.t)
        while self.t < target - 1e-12:
            h = target - self.t
            n_sub = int(np.ceil(h / self.SUB_DT))
            dt = h / n_sub
            for _ in range(n_sub):
                self._step(dt)
        return self._frame(target)

    def diagnostics(self) -> dict:
        v = self.v
        e = self.p[self.ib] - self.p[self.ia]
        ln = np.sqrt(np.einsum("ij,ij->i", e, e))
        ke = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", v, v)))
        pe_spring = 0.5 * float(np.sum(self.k * (ln - self.l0) ** 2))
        fa = self.faces
        a0 = self.p[fa[:, 0]]
        a1 = self.p[fa[:, 1]]
        a2 = self.p[fa[:, 2]]
        vol = self.flip * float(np.einsum("ij,ij->", a0, np.cross(a1, a2)) / 6.0)
        pe_vol = 0.5 * self.kp * (vol - self.vol0) ** 2 / self.vol0
        d, _ = self._grip(self.t)
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "kinetic_energy": ke,
            "elastic_energy": pe_spring + pe_vol,
            "volume": vol,
            "grip_offset": d,
            "samples": int(self.n),
        }
