"""Physically-based solver for the knitted legwarmer deformation scene.

A tubular knitted sleeve (mass-spring network with structural / shear /
bending elements, viscous yarn damping and hoop pre-tension) is integrated
with a fully implicit Euler scheme around a kinematic articulated leg.  The
leg bends at the knee during the run; implicit penalty contact (normal
stiffness/damping + Coulomb-clamped friction) against the thigh, shin, knee,
foot capsules and the ground keeps the sleeve outside the body.  A weak
explicit vertex-vertex repulsion reduces cloth self-intersection.

All required trajectory fields are exported every sample frame.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
DT = 1.0 / 240.0            # internal time step
THICKNESS = 0.015

K_STRUCT = 650.0            # structural springs (knit is extensible)
K_SHEAR = 160.0
K_BEND = 28.0               # soft bending -> knit folds / bunches easily
HOOP_PRESTRAIN = 0.92       # yarn tension gripping the leg

C_AXIAL = 0.9               # implicit viscous damping per spring
DRAG = 0.02                 # implicit linear air drag per vertex (kg/s)

K_CONTACT = 1.6e4           # contact penalty stiffness (N/m), implicit
C_CONTACT = 14.0            # normal contact damping, implicit
MU_FRICTION = 0.65          # yarn / skin friction
C_FRIC = 10.0               # tangential damping (implicit, Coulomb-clamped)

K_SELF = 100.0              # explicit self-contact repulsion (N/m)
SELF_MARGIN = 0.030

KNEE = np.array([0.0, 0.0, 1.1])
T0, T1, THETA_MAX = 1.0, 5.0, 1.15   # bend schedule (rad)

CONTACT_MARGIN = 0.05       # clip for reported signed distances
TUBE_RADIUS = 0.42
TUBE_LENGTH = 3.0


def _bend_theta(t):
    """Smooth bend angle and rate of the lower leg about the knee."""
    if t <= T0:
        return 0.0, 0.0
    if t >= T1:
        return THETA_MAX, 0.0
    u = (t - T0) / (T1 - T0)
    s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)            # smootherstep
    ds = 30.0 * u * u * (u - 1.0) * (u - 1.0) / (T1 - T0)    # d/dt
    return THETA_MAX * s, THETA_MAX * ds


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        meta = scene.metadata()
        self.gravity = np.asarray(meta["gravity"], dtype=float)
        frame0 = scene.initial_state()
        x0 = np.array(frame0.position, dtype=float)
        self.n = len(x0)
        self.mass = np.array(frame0.mass, dtype=float)
        nx = int(meta["grid_shape"][0])   # circumferential vertex count
        self.nx = nx
        self.ny = self.n // nx

        # ---- reference material coordinates (axial arclength, hoop arclength)
        ds_ = TUBE_LENGTH / (self.ny - 1)
        darc = 2.0 * np.pi * TUBE_RADIUS / nx
        self.X = np.empty((self.n, 2))
        for i in range(self.ny):
            for j in range(nx):
                self.X[i * nx + j] = (i * ds_, j * darc)

        # ---- triangles (matches assets/cloth_topology.json)
        faces = []
        for i in range(self.ny - 1):
            for j in range(nx):
                j1 = (j + 1) % nx
                a, b = i * nx + j, i * nx + j1
                c, d = (i + 1) * nx + j1, (i + 1) * nx + j
                faces.append((a, b, c))
                faces.append((a, c, d))
        self.faces = np.array(faces, dtype=np.int64)

        # unwrapped material coords per face (copies - do not alias self.X)
        L = nx * darc
        self.Xf = np.empty((len(self.faces), 3, 2))
        for f, (a, b, c) in enumerate(self.faces):
            pa = self.X[a].copy()
            pb = self.X[b].copy()
            pc = self.X[c].copy()
            for arr in (pb, pc):
                while arr[1] - pa[1] > L / 2:
                    arr[1] -= L
                while arr[1] - pa[1] < -L / 2:
                    arr[1] += L
            self.Xf[f, 0], self.Xf[f, 1], self.Xf[f, 2] = pa, pb, pc

        # ---- springs ---------------------------------------------------
        si, sj, sk, rest, hoop = [], [], [], [], []
        for i in range(self.ny):
            for j in range(nx):
                a = i * nx + j
                b = i * nx + (j + 1) % nx                 # hoop (periodic)
                si.append(a); sj.append(b); sk.append(K_STRUCT); hoop.append(True)
                if i < self.ny - 1:
                    d = (i + 1) * nx + j                  # axial
                    si.append(a); sj.append(d); sk.append(K_STRUCT); hoop.append(False)
                    c2 = (i + 1) * nx + (j + 1) % nx      # shear diagonals
                    si.append(a); sj.append(c2); sk.append(K_SHEAR); hoop.append(False)
                    si.append(b); sj.append(c2); sk.append(K_SHEAR); hoop.append(False)
                if i < self.ny - 2:                       # axial bending
                    e = (i + 2) * nx + j
                    si.append(a); sj.append(e); sk.append(K_BEND); hoop.append(False)
                j2 = (j + 2) % nx                         # hoop bending
                si.append(a); sj.append(i * nx + j2); sk.append(K_BEND); hoop.append(True)
        self.si = np.array(si); self.sj = np.array(sj)
        self.sk = np.array(sk)
        hoop = np.array(hoop, dtype=bool)
        r0 = np.linalg.norm(x0[self.si] - x0[self.sj], axis=1)
        self.rest = np.where(hoop, r0 * HOOP_PRESTRAIN, r0)
        self.ns = len(r0)
        self.sc = np.full(self.ns, C_AXIAL)

        # sparse index blocks for the 3x3 spring Hessians
        I3 = np.arange(3)
        ii = (3 * self.si[:, None, None] + I3[None, :, None]).repeat(3, axis=2)
        jj = (3 * self.sj[:, None, None] + I3[None, None, :]).repeat(3, axis=1)
        self._rows = np.concatenate([ii.ravel(), ii.ravel(), jj.ravel(), jj.ravel()])
        self._cols = np.concatenate([jj.ravel(), jj.ravel(), ii.ravel(), ii.ravel()])

        # ---- colliders ---------------------------------------------------
        def cap_half(h, r):
            return max(h / 2.0 - r, 0.01)
        hu = cap_half(1.3, 0.34)
        hl = cap_half(1.3, 0.31)
        hf = cap_half(0.72, 0.25)
        self.cap_static = [
            (np.array([-0.62 - hu, 0.0, 1.1]), np.array([-0.62 + hu, 0.0, 1.1]), 0.34),
        ]
        self.cap_moving0 = [
            (np.array([0.62 - hl, 0.0, 1.1]), np.array([0.62 + hl, 0.0, 1.1]), 0.31),
            (np.array([1.55 - hf, -0.1, 0.92]), np.array([1.55 + hf, -0.1, 0.92]), 0.25),
        ]

        # ---- state --------------------------------------------------------
        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.t = 0.0
        self.f_ext = self.mass[:, None] * self.gravity[None, :]

        # ---- brief pre-relaxation so initial cloth/foot overlap settles ---
        t_end = 0.12
        while self.t < t_end - 1e-12:
            self._step(min(DT, t_end - self.t))
        self.t = 0.0
        self.v[:] = 0.0

    # ------------------------------------------------------------- kinematics
    def _moving_colliders(self, t):
        th, thd = _bend_theta(t)
        c, s = np.cos(th), np.sin(th)
        # rotation about y by -theta: x' = c*x - s*z ; z' = s*x + c*z
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        caps = [(KNEE + R @ (p0 - KNEE), KNEE + R @ (p1 - KNEE), r)
                for p0, p1, r in self.cap_moving0]
        omega = np.array([0.0, -thd, 0.0])
        return caps, omega

    # ---------------------------------------------------------------- contact
    def _signed_distance(self):
        """Minimum signed gap from every vertex to any collider surface."""
        caps, _ = self._moving_colliders(self.t)
        x = self.x
        dmin = np.full(self.n, np.inf)
        for p0, p1, r in list(self.cap_static) + list(caps):
            ab = p1 - p0
            tt = np.clip(((x - p0) @ ab) / (ab @ ab), 0.0, 1.0)
            q = p0 + tt[:, None] * ab
            diff = x - q
            dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
            dmin = np.minimum(dmin, dist - r)
        dmin = np.minimum(dmin, np.linalg.norm(x - KNEE, axis=1) - 0.36)
        return np.minimum(dmin, x[:, 2])

    def _contact_terms(self, t):
        """Position-based normal forces + implicit contact Hessian contributions."""
        caps, omega = self._moving_colliders(t)
        x, v = self.x, self.v
        f = np.zeros_like(x)
        Hc = np.zeros((self.n, 3, 3))
        eye = np.eye(3)
        all_caps = [(p0, p1, r, None) for p0, p1, r in self.cap_static] + \
                   [(p0, p1, r, omega) for p0, p1, r in caps] + \
                   [(KNEE, KNEE, 0.36, None)]
        for p0, p1, r, om in all_caps:
            ab = p1 - p0
            L2 = ab @ ab
            if L2 < 1e-12:
                tt = np.zeros(self.n)
                q = np.tile(p0, (self.n, 1))
            else:
                tt = np.clip(((x - p0) @ ab) / L2, 0.0, 1.0)
                q = p0 + tt[:, None] * ab
            diff = x - q
            dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
            act = (dist - r) < 0.0
            if not act.any():
                continue
            nn = diff[act] / dist[act, None]
            dd = r - dist[act]                       # penetration depth > 0
            fn = K_CONTACT * dd                      # position-based normal force
            f[act] += fn[:, None] * nn
            nnt = nn[:, :, None] * nn[:, None, :]
            if om is not None:
                vsurf = np.cross(om, q[act] - KNEE)
            else:
                vsurf = 0.0
            vrel = v[act] - vsurf
            vt = vrel - np.einsum('ij,ij->i', vrel, nn)[:, None] * nn
            vtn = np.linalg.norm(vt, axis=1)
            ceff = np.minimum(C_FRIC, MU_FRICTION * fn / np.maximum(vtn, 1e-4))
            Pt = eye[None] - nnt
            Hc[act] += (K_CONTACT + C_CONTACT / DT)[:, None, None] * nnt \
                       + (ceff / DT)[:, None, None] * Pt
        # ground plane z = 0
        act = x[:, 2] < 0.0
        if act.any():
            fn = K_CONTACT * (-x[act, 2])
            f[act, 2] += fn
            vrel = v[act].copy()
            vtn = np.maximum(np.linalg.norm(vrel[:, :2], axis=1), 1e-9)
            ceff = np.minimum(C_FRIC, MU_FRICTION * fn / vtn)
            blk = np.zeros((act.sum(), 3, 3))
            blk[:, 2, 2] = K_CONTACT + C_CONTACT / DT
            blk[:, 0, 0] = ceff / DT
            blk[:, 1, 1] = ceff / DT
            Hc[act] += blk
        return f, Hc

    def _self_contact(self):
        """Weak explicit vertex-vertex repulsion (reduces self-intersection)."""
        x = self.x
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        ai, aj = np.nonzero(d2 < SELF_MARGIN * SELF_MARGIN)
        f = np.zeros_like(x)
        if ai.size:
            dd = diff[ai, aj]
            dist = np.maximum(np.sqrt(d2[ai, aj]), 1e-6)
            fn = (K_SELF * (SELF_MARGIN - dist) * 0.5)[:, None] * (dd / dist[:, None])
            np.add.at(f, ai, fn)
            np.add.at(f, aj, -fn)
        return f

    # ------------------------------------------------------------------- step
    def _spring_terms(self, dt):
        """Elastic forces and implicit spring Hessian blocks (axial form)."""
        x = self.x
        diff = x[self.sj] - x[self.si]
        L = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
        d = diff / L[:, None]
        fmag = self.sk * (L - self.rest)
        fs = fmag[:, None] * d                       # pulls i toward j if stretched
        f = np.zeros_like(x)
        np.add.at(f, self.si, fs)
        np.add.at(f, self.sj, -fs)
        coef = (self.sk + self.sc / dt)[:, None, None]
        ddt = d[:, :, None] * d[:, None, :]
        Hflat = (coef * ddt).ravel()
        return f, Hflat

    def _step(self, dt):
        f_spring, Hflat = self._spring_terms(dt)
        f_contact, Hc = self._contact_terms(self.t)
        f_self = self._self_contact()
        f_total = f_spring + f_contact + f_self + self.f_ext

        n3 = 3 * self.n
        # A = M/dt^2 + DRAG/dt - K - C/dt :
        # spring blocks (i,i)+H (i,j)-H (j,i)-H (j,j)+H
        vals = np.concatenate([-Hflat, Hflat, -Hflat, Hflat])
        A = sp.coo_matrix((vals, (self._rows, self._cols)), shape=(n3, n3))
        # contact blocks on the diagonal
        idx = np.arange(self.n)
        rows = (3 * idx[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2)
        cols = (3 * idx[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1)
        A = A + sp.coo_matrix((Hc.ravel(), (rows.ravel(), cols.ravel())), shape=(n3, n3))
        diag = np.repeat(self.mass / dt**2 + DRAG / dt, 3)
        A = (A + sp.diags(diag)).tocsc()

        rhs = f_total.ravel() + (self.v * (self.mass / dt)[:, None]).ravel()
        dx = spla.splu(A).solve(rhs).reshape(self.n, 3)
        self.x = self.x + dx
        self.v = dx / dt
        self.t += dt

    # ---------------------------------------------------------------- outputs
    def _deformation_gradients(self):
        f = self.faces
        x = self.x
        g0 = x[f[:, 0]]
        G3 = np.stack([x[f[:, 1]] - g0, x[f[:, 2]] - g0], axis=2)   # (F,3,2)
        X0 = self.Xf[:, 0]
        E1 = self.Xf[:, 1] - X0
        E2 = self.Xf[:, 2] - X0
        det = E1[:, 0] * E2[:, 1] - E1[:, 1] * E2[:, 0]
        inv = np.empty((len(f), 2, 2))
        inv[:, 0, 0] = E2[:, 1] / det
        inv[:, 0, 1] = -E2[:, 0] / det
        inv[:, 1, 0] = -E1[:, 1] / det
        inv[:, 1, 1] = E1[:, 0] / det
        Fm = G3 @ inv                                                # (F,3,2)
        C = np.einsum('fij,fik->fjk', Fm, Fm)                        # right Cauchy-Green
        tr = C[:, 0, 0] + C[:, 1, 1]
        dtm = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
        t = np.sqrt(np.maximum(dtm, 0.0))
        s = np.sqrt(np.maximum(tr + 2.0 * t, 1e-12))
        U = np.empty_like(C)
        U[:, 0, 0] = (C[:, 0, 0] + t) / s
        U[:, 0, 1] = C[:, 0, 1] / s
        U[:, 1, 0] = C[:, 1, 0] / s
        U[:, 1, 1] = (C[:, 1, 1] + t) / s
        Fv = np.zeros((self.n, 2, 2))
        cnt = np.zeros(self.n)
        for k in range(3):
            np.add.at(Fv, f[:, k], U)
            np.add.at(cnt, f[:, k], 1.0)
        Fv /= np.maximum(cnt, 1.0)[:, None, None]
        return Fv

    def _make_frame(self, time):
        dmin = np.minimum(self._signed_distance(), CONTACT_MARGIN)
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "contact_signed_distance": dmin.astype(float),
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": self.f_ext.copy(),
            },
            scalars={"contact_length_scale": THICKNESS},
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        eps = 1e-9
        while self.t < time - eps:
            self._step(min(DT, time - self.t))
        return self._make_frame(time)

    def diagnostics(self):
        return {"t": self.t, "n": self.n, "springs": self.ns}
