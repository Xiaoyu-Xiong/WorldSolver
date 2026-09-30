"""Physically-based solver for the knitted-scarf-on-incline bunching scene.

Model
-----
* The scarf is a 17x17 mass-spring sheet (structural + shear + bending
  springs) with soft bending stiffness to mimic knit compliance, integrated
  with linearly-implicit Euler (spring forces linearized along their axes,
  giving an SPD sparse system solved every substep).
* Rigid obstacles (incline box, ground box) are handled with penalty
  signed-distance contact, viscous contact damping and clipped Coulomb
  friction (yarn-on-plane mu ~ 0.5 so tan(35deg) > mu and the scarf slides).
* Self-contact (all node pairs beyond a 2-ring material neighborhood) uses a
  penalty repulsion at a few cloth thicknesses, giving the bunched pile
  volume like the yarn-level knit result of Kaldor et al., Figure 11.

The scarf starts on the incline (per the frozen scaffold pose), slides
downhill under gravity, its leading edge meets the ground stop at the foot
of the incline, and following material compresses into thick bunched folds
while the upper part remains connected on the slope.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from worldsolver.contracts import StateFrame


def _quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class _Box:
    """Static oriented box with vectorized signed distance / outward normal."""

    def __init__(self, center, half, R):
        self.c = np.asarray(center, dtype=float)
        self.h = np.asarray(half, dtype=float)
        self.R = np.asarray(R, dtype=float)

    def sdf(self, P):
        q = (P - self.c) @ self.R                 # world -> local
        qa = np.abs(q)
        exc = qa - self.h
        clamped = np.clip(q, -self.h, self.h)
        dv = q - clamped
        d_out = np.sqrt((dv * dv).sum(1))
        d_in = exc.max(1)
        d = d_out + np.minimum(d_in, 0.0)
        # outward normal (local frame)
        n_out = dv / np.maximum(d_out, 1e-12)[:, None]
        axis = np.argmax(exc, 1)
        n_in = np.zeros_like(q)
        rows = np.arange(len(q))
        n_in[rows, axis] = np.sign(q[rows, axis]) + (q[rows, axis] == 0)
        n_loc = np.where((d_in < 0.0)[:, None], n_in, n_out)
        n = n_loc @ self.R.T                       # local -> world
        return d, n


class BenchmarkSolver:
    # ---- model parameters -------------------------------------------------
    H = 1.0e-3              # physics substep
    K_STRUCT = 800.0        # structural spring stiffness (N/m)
    K_SHEAR = 200.0
    K_BEND = 6.0            # soft knit-like bending springs
    K_CONTACT = 2000.0      # obstacle penalty stiffness
    EPS_C = 0.02            # obstacle contact activation range
    C_CONTACT = 0.25        # contact normal damping (kg/s)
    MU_INCLINE = 0.50       # < tan(35deg)=0.70 so the scarf slides
    MU_GROUND = 0.60
    D_SELF = 0.06           # self-contact range (~4x thickness)
    K_SELF = 800.0
    C_VEL = 8.0e-4          # weak global viscous damping

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.config = scene.config
        cloth = self.config["cloth"]
        frame0 = scene.initial_state()
        self.X = np.array(frame0.position, dtype=float)
        self.V = np.array(frame0.velocity, dtype=float)
        self.M = np.array(frame0.mass, dtype=float)
        self.N = len(self.X)
        self.t = 0.0
        self.nsteps = 0
        self.g = np.asarray(self.config["gravity"], dtype=float)
        self.thickness = float(cloth["thickness"])

        nx, ny = (int(v) for v in cloth["grid_shape"])
        self.nx, self.ny = nx, ny

        # ---- springs -------------------------------------------------------
        si, sj, sk = [], [], []

        def add_grid(di, dj, k):
            for iy in range(ny):
                for ix in range(nx):
                    jy, jx = iy + dj, ix + di
                    if 0 <= jy < ny and 0 <= jx < nx:
                        si.append(iy * nx + ix)
                        sj.append(jy * nx + jx)
                        sk.append(k)

        add_grid(1, 0, self.K_STRUCT)
        add_grid(0, 1, self.K_STRUCT)
        add_grid(1, 1, self.K_SHEAR)
        add_grid(1, -1, self.K_SHEAR)
        add_grid(2, 0, self.K_BEND)
        add_grid(0, 2, self.K_BEND)
        self.SI = np.asarray(si, dtype=np.int64)
        self.SJ = np.asarray(sj, dtype=np.int64)
        self.SK = np.asarray(sk, dtype=float)
        d0 = self.X[self.SJ] - self.X[self.SI]
        self.SL0 = np.linalg.norm(d0, axis=1)
        self.nspr = len(self.SI)

        # precomputed sparse-pattern index arrays for the 3x3 axis blocks
        a3 = np.arange(3)[None, :]
        ra = (3 * self.SI[:, None] + a3)          # (S,3)
        rb = (3 * self.SJ[:, None] + a3)
        r_Ia = np.repeat(ra, 3, axis=1)           # (S,9)
        c_Jb = np.tile(rb, (1, 3))
        r_Ja = np.repeat(rb, 3, axis=1)
        c_Ib = np.tile(ra, (1, 3))
        self.H_rows = np.concatenate([r_Ia, r_Ia, r_Ja, r_Ja]).ravel()
        self.H_cols = np.concatenate([c_Jb, c_Ib, c_Ib, c_Jb]).ravel()
        self.diag_rows = np.arange(3 * self.N)

        # ---- self-contact exclusion mask (material 2-ring neighborhood) -----
        iy = np.arange(self.N) // nx
        ix = np.arange(self.N) % nx
        cheb = np.maximum(np.abs(iy[:, None] - iy[None, :]),
                          np.abs(ix[:, None] - ix[None, :]))
        self.self_ok = cheb > 2                   # allow self contact beyond 2-ring

        # ---- rigid obstacles (from frozen scene_geometry.yaml) --------------
        self.boxes = [
            (_Box([0.0, 0.0, -0.05], [2.4, 2.0, 0.05], np.eye(3)), self.MU_GROUND),
            (_Box([0.0, 0.0, 0.8], [2.2, 1.1, 0.08],
                  _quat_to_R([0.0, 0.300706, 0.0, 0.953717])), self.MU_INCLINE),
        ]

        # ---- deformation-gradient stencil (material frame of initial pose) --
        tilt = np.deg2rad(float(cloth.get("tilt_about_y_deg", 0.0)))
        self.e1 = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
        self.e2 = np.array([0.0, 1.0, 0.0])
        Xg = self.X.reshape(ny, nx, 3)
        s = Xg[0, :, 0] * self.e1[0] + Xg[0, :, 2] * self.e1[2]
        tt = Xg[:, 0, 1]
        self.den_s = np.empty(nx); self.den_t = np.empty(ny)
        ip = np.minimum(np.arange(nx) + 1, nx - 1)
        im = np.maximum(np.arange(nx) - 1, 0)
        self.den_s = s[ip] - s[im]
        jp = np.minimum(np.arange(ny) + 1, ny - 1)
        jm = np.maximum(np.arange(ny) - 1, 0)
        self.den_t = tt[jp] - tt[jm]
        self.ix_p, self.ix_m = ip, im
        self.iy_p, self.iy_m = jp, jm

        bnd = ((np.abs(np.arange(self.N) % nx == 0) |
                np.abs(np.arange(self.N) % nx == nx - 1) |
                (np.arange(self.N) // nx == 0) |
                (np.arange(self.N) // nx == ny - 1))).astype(np.int32)
        self.boundary = bnd
        self.contact_len = float(self.thickness)

        self._diag_min = self.M / self.H + self.C_VEL

    # ------------------------------------------------------------------ physics
    def _box_sdf(self, P):
        d = np.full(len(P), np.inf)
        n = np.zeros_like(P)
        for box, _mu in self.boxes:
            db, nb = box.sdf(P)
            upd = db < d
            d = np.where(upd, db, d)
            n[upd] = nb[upd]
        return d, n

    def _contact_terms(self, P, V, F, diag_extra):
        m = self.M
        for box, mu in self.boxes:
            d, n = box.sdf(P)
            act = d < self.EPS_C
            if not act.any():
                continue
            fn_mag = np.where(act, self.K_CONTACT * (self.EPS_C - d), 0.0)
            vn = (V * n).sum(1)
            # normal penalty + normal damping
            F += np.where(act, fn_mag - self.C_CONTACT * vn, 0.0)[:, None] * n
            # clipped Coulomb friction on tangential velocity
            vt = V - vn[:, None] * n
            vtn = np.sqrt((vt * vt).sum(1))
            f_max = np.minimum(mu * fn_mag, m * vtn / self.H)
            coef = np.where((act & (vtn > 1e-9)), f_max / np.maximum(vtn, 1e-9), 0.0)
            F -= coef[:, None] * vt
            diag_extra += np.where(act, self.K_CONTACT, 0.0)

    def _self_contact(self, P, F, diag_extra):
        G = (P * P).sum(1)
        D2 = np.maximum(G[:, None] + G[None, :] - 2.0 * (P @ P.T), 0.0)
        D = np.sqrt(D2 + 1e-16)
        Dsafe = np.maximum(D, 0.2 * self.D_SELF)
        act = (D < self.D_SELF) & self.self_ok
        W = np.where(act, self.K_SELF * (self.D_SELF - D) / Dsafe, 0.0)
        Wsum = W.sum(1)
        F += P * Wsum[:, None] - W @ P
        diag_extra += Wsum
        return D, act

    def _substep(self):
        h = self.H
        X, V, M = self.X, self.V, self.M
        N = self.N
        F = self.M[:, None] * self.g[None, :]
        diag_extra = np.zeros(N)

        # spring forces (full nonlinear) and axis-linearized Hessian blocks
        d = X[self.SJ] - X[self.SI]
        L = np.sqrt((d * d).sum(1))
        Lsafe = np.maximum(L, 1e-12)
        u = d / Lsafe[:, None]
        fmag = self.SK * (L - self.SL0)
        fI = fmag[:, None] * u
        np.add.at(F, self.SI, fI)
        np.add.at(F, self.SJ, -fI)
        T = (h * self.SK)[:, None, None] * u[:, :, None] * u[:, None, :]
        vals = np.concatenate([-T.ravel(), T.ravel(), T.ravel(), -T.ravel()])

        # contacts (obstacles + self)
        self._contact_terms(X, V, F, diag_extra)
        self._self_contact(X, F, diag_extra)

        # assemble SPD system  (M/h + C + h*K) v' = M v/h + f
        diag_vals = self._diag_min + h * diag_extra
        rows = np.concatenate([self.H_rows, self.diag_rows])
        cols = np.concatenate([self.H_cols, self.diag_rows])
        dv3 = np.empty(3 * N)
        dv3[0::3] = diag_vals
        dv3[1::3] = diag_vals
        dv3[2::3] = diag_vals
        data = np.concatenate([vals, dv3])
        A = sp.coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()
        b = ((M[:, None] * V) / h + F).ravel()
        Vn = spsolve(A, b)
        Vn = np.nan_to_num(Vn, nan=0.0, posinf=0.0, neginf=0.0).reshape(N, 3)
        speed = np.sqrt((Vn * Vn).sum(1))
        cap = np.minimum(1.0, 50.0 / np.maximum(speed, 1e-12))
        Vn *= cap[:, None]
        self.V = Vn
        self.X = X + h * Vn
        self.t += h
        self.nsteps += 1

    # ------------------------------------------------------------------ output
    def _deformation_gradients(self):
        nx, ny = self.nx, self.ny
        Xg = self.X.reshape(ny, nx, 3)
        dxds = (Xg[:, self.ix_p, :] - Xg[:, self.ix_m, :]) / self.den_s[None, :, None]
        dxdt = (Xg[self.iy_p, :, :] - Xg[self.iy_m, :, :]) / self.den_t[:, None, None]
        F = np.empty((ny, nx, 2, 2))
        F[..., 0, 0] = dxds @ self.e1
        F[..., 1, 0] = dxds @ self.e2
        F[..., 0, 1] = dxdt @ self.e1
        F[..., 1, 1] = dxdt @ self.e2
        return F.reshape(-1, 2, 2)

    def _make_frame(self, time):
        d_box, _ = self._box_sdf(self.X)
        G = (self.X * self.X).sum(1)
        D2 = np.maximum(G[:, None] + G[None, :] - 2.0 * (self.X @ self.X.T), 0.0)
        D = np.sqrt(D2 + 1e-16)
        Dself = D - self.D_SELF
        Dself[~self.self_ok] = np.inf
        self_gap = Dself.min(1)
        contact_sd = np.minimum(d_box, self_gap)

        Fext = self.M[:, None] * self.g[None, :]
        mass = self.M.copy()
        return StateFrame(
            float(time),
            self.X.copy(),
            self.V.copy(),
            mass=mass,
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "contact_signed_distance": contact_sd.astype(float),
                "external_force": Fext.astype(float),
            },
            scalars={"contact_length_scale": self.contact_len},
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        h = self.H
        n = int(round((target - self.t) / h))
        n = max(n, 0)
        for _ in range(n):
            self._substep()
        self.t = target
        frame = self._make_frame(target)
        frame.validate()
        self._last_min_contact = float(np.min(frame.fields["contact_signed_distance"]))
        self._last_max_speed = float(np.max(np.linalg.norm(self.V, axis=1)))
        return frame

    def diagnostics(self) -> dict:
        return {
            "substeps": int(self.nsteps),
            "time": float(self.t),
            "min_contact_signed_distance": float(getattr(self, "_last_min_contact", np.nan)),
            "max_speed": float(getattr(self, "_last_max_speed", np.nan)),
        }
