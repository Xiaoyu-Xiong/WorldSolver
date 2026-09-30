"""Physics solver: notched rubber sheet tear (scene 160).

Physical model
--------------
The rubber sheet is discretized with the frozen 17x17 vertex grid and simulated
as a stretchable membrane made of structural, shear and bending elastic edges
(calibrated against a rubber-like Young's modulus E and the frozen thickness).
Tearing is modeled as irreversible edge failure driven by the local tensile
strain: once an edge's stretch exceeds a critical strain it is permanently
removed from the force network, so stress re-concentrates at the new crack tip
and the tear advances dynamically.  The opposing grips are kinematic boxes that
separate with a smooth prescribed displacement; gripped vertices follow the
grip motion exactly.  Box obstacles (ground + grips) are handled with a signed
distance based penalty contact.  Time integration is semi-implicit (symplectic)
Euler on fixed sub-steps, fully deterministic, with edge damping and a global
Rayleigh damping term for stability.

No animation, keyframes or precomputed trajectories are used: every frame is
the result of the explicit time stepping procedure below.
"""
from __future__ import annotations

import math

import numpy as np
from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- parameters
E_MODULUS = 1.0e5          # rubber-like Young's modulus [Pa]
K_STRUCT = E_MODULUS * 1.0  # scaled by thickness below
K_SHEAR_RATIO = 0.5
K_BEND_RATIO = 0.04
ZETA_EDGE = 0.12           # per-edge damping ratio
GAMMA_GLOBAL = 0.5         # global Rayleigh damping [1/s]
TEAR_STRAIN = 0.55         # critical tensile strain for edge failure
GRIP_DISPLACEMENT = 0.7    # per-grip separation distance [m]
GRIP_RAMP_TIME = 6.0       # time at which the grips stop [s]
NSUB = 160                 # sub-steps per sample interval
VMAX = 20.0                # velocity clamp [m/s]
K_CONTACT = 1.0e4          # contact penalty stiffness [N/m]
C_CONTACT = 5.0            # contact damping [N s/m]


def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray):
    """Signed distance and outward normal of an axis-aligned box."""
    q = points - center[None, :]
    h = half[None, :]
    s = np.abs(q) - h
    out = np.maximum(s, 0.0)
    dist = np.linalg.norm(out, axis=1) + np.minimum(np.max(s, axis=1), 0.0)
    n = np.zeros_like(points)
    outside = np.linalg.norm(out, axis=1) > 1e-12
    if outside.any():
        cl = np.clip(q[outside], -h, h)
        nn = points[outside] - (center[None, :] + cl)
        norm = np.linalg.norm(nn, axis=1)
        norm[norm < 1e-12] = 1.0
        n[outside] = nn / norm[:, None]
    inside = ~outside
    if inside.any():
        si = s[inside]
        ax = np.argmax(si, axis=1)          # face with the least penetration
        rows = np.arange(len(ax))
        sg = np.sign(q[inside][rows, ax])
        sg[sg == 0.0] = 1.0
        ni = np.zeros((len(ax), points.shape[1]))
        ni[rows, ax] = sg
        n[inside] = ni
    return dist, n


class BenchmarkSolver:
    """Explicit membrane-with-tearing solver for the notched rubber sheet."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.thickness = float(cloth["thickness"])
        self.dt = self.sample_dt / NSUB
        self.steps = 0
        self.t = 0.0

        frame0 = scene.initial_state()
        p0 = np.asarray(frame0.position, dtype=float)
        self.n = len(p0)
        self.p = p0.copy()
        self.v = np.zeros_like(self.p)
        self.p0 = p0.copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.bmask = np.asarray(
            frame0.attributes.get("boundary_vertex", np.zeros(self.n)), dtype=float
        ).astype(bool).copy()
        self.inv_mass = np.where(self.bmask, 0.0, 1.0 / self.mass)

        # ---- grid layout derived from rest positions (robust to ordering)
        nx, ny = (int(v) for v in cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        xs = np.unique(np.round(p0[:, 0], 9))
        zs = np.unique(np.round(p0[:, 2], 9))
        self.i_of = np.searchsorted(xs, np.round(p0[:, 0], 9))
        self.j_of = np.searchsorted(zs, np.round(p0[:, 2], 9))
        grid = np.full((ny, nx), -1, dtype=np.int64)
        grid[self.j_of, self.i_of] = np.arange(self.n)
        if (grid < 0).any():
            raise ValueError("cloth vertices do not form a regular grid")
        self.grid = grid.ravel()  # slot (j*nx+i) -> vertex id

        def vid(i: int, j: int) -> int:
            return self.grid[j * nx + i]

        # ---- elastic edge network ------------------------------------
        edges = []  # (a, b, type) type: 0 structural, 1 shear, 2 bending
        for j in range(ny):
            for i in range(nx):
                a = vid(i, j)
                if i + 1 < nx:
                    edges.append((a, vid(i + 1, j), 0))
                if j + 1 < ny:
                    edges.append((a, vid(i, j + 1), 0))
                if i + 1 < nx and j + 1 < ny:
                    edges.append((a, vid(i + 1, j + 1), 1))
                    edges.append((vid(i + 1, j), vid(i, j + 1), 1))
                if i + 2 < nx:
                    edges.append((a, vid(i + 2, j), 2))
                if j + 2 < ny:
                    edges.append((a, vid(i, j + 2), 2))
        e = np.asarray(edges, dtype=np.int64)
        self.ea, self.eb = e[:, 0], e[:, 1]
        self.etype = e[:, 2]
        self.rest = np.linalg.norm(self.p0[self.ea] - self.p0[self.eb], axis=1)
        k_struct = E_MODULUS * self.thickness
        k_by_type = np.array([k_struct, K_SHEAR_RATIO * k_struct,
                              K_BEND_RATIO * k_struct])
        self.k = k_by_type[self.etype]
        m_node = float(np.mean(self.mass))
        self.c = 2.0 * ZETA_EDGE * np.sqrt(self.k * m_node)
        self.broken = np.zeros(len(self.ea), dtype=bool)

        self._edge_lookup = {}
        for idx in range(len(self.ea)):
            key = (min(self.ea[idx], self.eb[idx]), max(self.ea[idx], self.eb[idx]))
            self._edge_lookup[key] = idx

        # ---- triangles (for deformation gradients) --------------------
        tris, tri_edges = [], []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v00, v10, v01, v11 = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
                s = self._edge_lookup
                tris.append((v00, v10, v01))
                tri_edges.append((s[(min(v00, v10), max(v00, v10))],
                                  s[(min(v00, v01), max(v00, v01))],
                                  s[(min(v10, v01), max(v10, v01))]))
                tris.append((v10, v11, v01))
                tri_edges.append((s[(min(v10, v11), max(v10, v11))],
                                  s[(min(v11, v01), max(v11, v01))],
                                  s[(min(v10, v01), max(v10, v01))]))
        self.tris = np.asarray(tris, dtype=np.int64)
        self.tri_e = np.asarray(tri_edges, dtype=np.int64)

        # ---- initial edge notch (slit from the top edge) --------------
        i_split = int(np.argmin(np.abs(xs)))          # column closest to x=0
        j_notch = ny - 4                              # slit depth ~3 cells
        cut = set()
        for j in range(j_notch, ny):
            a, b = vid(i_split, j), vid(i_split + 1, j)
            cut.add((min(a, b), max(a, b)))
        for j in range(j_notch, ny - 1):
            for a, b in ((vid(i_split, j), vid(i_split + 1, j + 1)),
                         (vid(i_split + 1, j), vid(i_split, j + 1))):
                cut.add((min(a, b), max(a, b)))
        for j in range(j_notch, ny):
            for a, b in ((vid(i_split - 1, j), vid(i_split + 1, j)),
                         (vid(i_split, j), vid(i_split + 2, j))):
                cut.add((min(a, b), max(a, b)))
        for key in cut:
            idx = self._edge_lookup.get(key)
            if idx is not None:
                self.broken[idx] = True
        self._refresh_active()

        # ---- obstacles -------------------------------------------------
        self.ground_center = np.array([0.0, 0.0, -0.05])
        self.ground_half = np.array([2.4, 2.0, 0.05])
        self.grip_half = np.array([0.12, 0.18, 1.25])
        x_left = float(xs[0])
        x_right = float(xs[-1])
        self.grip_anchor = np.array([x_left, x_right], dtype=float)
        self.grip_zc = float(np.mean(p0[:, 2]))
        self.damp_factor = math.exp(-GAMMA_GLOBAL * self.dt)
        self.n_broken0 = int(self.broken.sum())

    # ---------------------------------------------------------------- utils
    def _refresh_active(self) -> None:
        self.active = np.flatnonzero(~self.broken)

    def _grip_disp(self, t: float) -> float:
        s = min(max(t / GRIP_RAMP_TIME, 0.0), 1.0)
        return 0.5 * GRIP_DISPLACEMENT * (1.0 - math.cos(math.pi * s))

    def _grip_vel(self, t: float) -> float:
        if t <= 0.0 or t >= GRIP_RAMP_TIME:
            return 0.0
        return (GRIP_DISPLACEMENT * math.pi / (2.0 * GRIP_RAMP_TIME)
                * math.sin(math.pi * t / GRIP_RAMP_TIME))

    def _grip_centers(self, d: float):
        left = np.array([self.grip_anchor[0] - d, 0.0, self.grip_zc])
        right = np.array([self.grip_anchor[1] + d, 0.0, self.grip_zc])
        return left, right

    # ------------------------------------------------------------- dynamics
    def _contact_forces(self, f: np.ndarray, d: float) -> None:
        left, right = self._grip_centers(d)
        boxes = ((self.ground_center, self.ground_half),
                 (left, self.grip_half), (right, self.grip_half))
        free_idx = np.flatnonzero(~self.bmask)
        if free_idx.size == 0:
            return
        pf = self.p[free_idx]
        vf = self.v[free_idx]
        for center, half in boxes:
            dist, nrm = _box_sdf(pf, center, half)
            pen = dist < 0.0
            if not pen.any():
                continue
            vn = np.einsum("ij,ij->i", vf[pen], nrm[pen])
            fmag = -K_CONTACT * dist[pen] - C_CONTACT * np.minimum(vn, 0.0)
            f[free_idx[pen]] += fmag[:, None] * nrm[pen]

    def _step(self, dt: float) -> None:
        t = self.t
        d = self._grip_disp(t)
        dg = self._grip_vel(t)
        p, v = self.p, self.v
        prev_p = p.copy()

        f = self.mass[:, None] * self.gravity[None, :]

        # elastic edges (structural + shear + bend) with damping
        act = self.active
        if act.size:
            ia, ib = self.ea[act], self.eb[act]
            delta = p[ia] - p[ib]
            ln = np.linalg.norm(delta, axis=1)
            ln = np.maximum(ln, 1e-9)
            n = delta / ln[:, None]
            stretch = ln - self.rest[act]
            vrel = np.einsum("ij,ij->i", v[ia] - v[ib], n)
            fm = self.k[act] * stretch + self.c[act] * vrel
            fvec = -fm[:, None] * n
            np.add.at(f, ia, fvec)
            np.add.at(f, ib, -fvec)

            # strain-driven irreversible edge failure (tearing)
            strain = ln / self.rest[act] - 1.0
            fail = act[strain > TEAR_STRAIN]
            if fail.size:
                self.broken[fail] = True
                self._refresh_active()

        # obstacle contact
        self._contact_forces(f, d)

        # semi-implicit Euler
        v += f * self.inv_mass[:, None] * dt
        v *= self.damp_factor
        np.clip(v, -VMAX, VMAX, out=v)
        p += v * dt

        # kinematic grip constraint
        if self.bmask.any():
            bidx = np.flatnonzero(self.bmask)
            side = np.sign(self.p0[bidx, 0])
            side[side == 0.0] = 1.0
            p[bidx] = self.p0[bidx]
            p[bidx, 0] += side * d
            v[bidx] = 0.0
            v[bidx, 0] = side * dg

        # finite-state guard
        if not (np.isfinite(p).all() and np.isfinite(v).all()):
            p[:] = prev_p
            v[:] = 0.0

        self.t += dt

    # ------------------------------------------------------------ sampling
    def _deformation_gradient(self) -> np.ndarray:
        X0 = self.p0[:, [0, 2]]
        Xc = self.p[:, [0, 2]]
        tri = self.tris
        bad = self.broken[self.tri_e].any(axis=1)
        F = np.repeat(np.eye(2)[None], self.n, axis=0)
        ok = np.flatnonzero(~bad)
        if ok.size == 0:
            return F
        a, b, c = tri[ok, 0], tri[ok, 1], tri[ok, 2]
        Dm = np.stack([X0[b] - X0[a], X0[c] - X0[a]], axis=2)
        Ds = np.stack([Xc[b] - Xc[a], Xc[c] - Xc[a]], axis=2)
        det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
        good = np.abs(det) > 1e-14
        if not good.any():
            return F
        a, b, c = a[good], b[good], c[good]
        Dm, Ds, det = Dm[good], Ds[good], det[good]
        inv = np.empty_like(Dm)
        inv[:, 0, 0] = Dm[:, 1, 1]
        inv[:, 1, 1] = Dm[:, 0, 0]
        inv[:, 0, 1] = -Dm[:, 0, 1]
        inv[:, 1, 0] = -Dm[:, 1, 0]
        inv /= det[:, None, None]
        Ft = Ds @ inv
        w = 0.5 * np.abs(det)
        wsum = np.zeros(self.n)
        np.add.at(F, a, Ft * w[:, None, None])
        np.add.at(F, b, Ft * w[:, None, None])
        np.add.at(F, c, Ft * w[:, None, None])
        np.add.at(wsum, a, w)
        np.add.at(wsum, b, w)
        np.add.at(wsum, c, w)
        has = wsum > 1e-18
        F[has] /= wsum[has][:, None, None]
        F[~has] = np.eye(2)
        return F

    def _contact_field(self, d: float) -> np.ndarray:
        left, right = self._grip_centers(d)
        dist, _ = _box_sdf(self.p, self.ground_center, self.ground_half)
        sd = dist
        for center in (left, right):
            dg, _ = _box_sdf(self.p, center, self.grip_half)
            sd = np.minimum(sd, dg)
        return sd

    def _frame(self, time: float) -> StateFrame:
        d = self._grip_disp(self.t)
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            self.p.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": self.bmask.astype(np.int32),
            },
            fields={
                "contact_signed_distance": self._contact_field(d),
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        target = float(time)
        while self.steps * self.dt < target - 1e-9:
            self._step(self.dt)
            self.steps += 1
        return self._frame(target)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "steps": int(self.steps),
            "edges_total": int(len(self.ea)),
            "edges_broken": int(self.broken.sum()),
            "edges_failed_dynamically": int(self.broken.sum()) - self.n_broken0,
        }
