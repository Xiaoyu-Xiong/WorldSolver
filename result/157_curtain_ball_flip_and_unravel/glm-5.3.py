"""Physics solver: 157_curtain_ball_flip_and_unravel.

A 17x17 cloth curtain is pinned along its top edge (at the rod) and its lower
portion is wound around a rigid ball, forming a double-layer wrap over the top
of the ball.  The wrap is released at t=0 and gravity drives the classic
curtain "ball flip and unravel": the free hem slides around the ball, the fold
flips over the top of the ball while the two layers slide over each other
(self-contact with friction), and finally the cloth unwraps and hangs freely
again, draped against the ball.

Numerics: semi-implicit (symplectic) Euler on a spring network (structural,
shear and bending edges) with per-substep strain limiting, positional
collision projection against the sphere and the ground box, and
vertex-triangle self-collision with a broad phase.  All computations are
deterministic (no randomness).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen scene geometry (scene_geometry.yaml) -------------------------------
BALL_CENTER = np.array([0.0, 0.0, 0.68])
BALL_RADIUS = 0.62
GROUND_TOP_Z = 0.0
GROUND_HALF_X = 2.4
GROUND_HALF_Y = 2.0

GRAVITY = np.array([0.0, 0.0, -9.81])

# Wrap construction constants ----------------------------------------------
WRAP_THETA0 = 0.45       # wrap starts on the back-upper quadrant of the ball
WRAP_LAYER_GAP = 0.05    # radial gap between the two wound layers
WRAP_TH2_FRAC = 0.85     # back-span of the outer layer relative to fold span

# Numerics ------------------------------------------------------------------
DT_SUB = 1.0 / 1200.0
K_STRUCT = 600.0
C_STRUCT = 0.15
K_SHEAR = 120.0
C_SHEAR = 0.03
K_BEND = 2.0
C_BEND = 0.02
GLOBAL_DRAG = 0.3        # 1/s
STRETCH_LIMIT = 1.02     # strain limiting threshold
STRAIN_FACTOR = 0.25
SELF_D0 = 0.032          # self-contact distance
V_MAX = 40.0


def _closest_point_on_triangles(p, a, b, c):
    """Vectorized closest point on triangles (a,b,c) to points p (Ericson)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    q = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    m = (d1 <= 0) & (d2 <= 0)
    q[m] = a[m]
    done |= m

    m = (d3 >= 0) & (d4 <= d3) & ~done
    q[m] = b[m]
    done |= m

    m = (d6 >= 0) & (d5 <= d6) & ~done
    q[m] = c[m]
    done |= m

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~done
    if m.any():
        denom = np.where(np.abs(d1[m] - d3[m]) < 1e-12, 1e-12, d1[m] - d3[m])
        t = (d1[m] / denom)[:, None]
        q[m] = a[m] + t * ab[m]
    done |= m

    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~done
    if m.any():
        denom = np.where(np.abs(d2[m] - d6[m]) < 1e-12, 1e-12, d2[m] - d6[m])
        t = (d2[m] / denom)[:, None]
        q[m] = a[m] + t * ac[m]
    done |= m

    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~done
    if m.any():
        num = (d4 - d3)[m]
        den = (d4 - d3)[m] - (d5 - d6)[m]
        denom = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t = (num / denom)[:, None]
        q[m] = b[m] + t * (c[m] - b[m])
    done |= m

    m = ~done
    if m.any():
        denom = va[m] + vb[m] + vc[m]
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        v = vb[m] / denom
        w = vc[m] / denom
        q[m] = a[m] + v[:, None] * ab[m] + w[:, None] * ac[m]
    return q


class BenchmarkSolver:
    """Cloth solver producing the curtain ball flip-and-unravel trajectory."""

    def __init__(self) -> None:
        self.t = 0.0
        self._steps = 0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        self.thickness = float(cloth["thickness"])
        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, dtype=float)
        n = len(x0)
        self.n = n
        self.mass = np.full(n, float(cloth["total_mass"]) / n)
        self.body_id = np.zeros(n, dtype=np.int32)
        self.boundary_attr = np.asarray(frame0.attributes["boundary_vertex"]).astype(np.int32)

        nx, ny = self.nx, self.ny
        self.i_of = np.arange(n) // nx
        self.j_of = np.arange(n) % nx
        self.xcol = x0[:, 0].reshape(ny, nx)[0].copy()
        self.srow = x0[:, 2].reshape(ny, nx)[:, 0].copy()

        # faces (grid triangulation, matches assets/cloth_topology.json)
        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                a = i * nx + j
                b = a + 1
                c = a + nx + 1
                d = a + nx
                faces.append((a, b, c))
                faces.append((a, c, d))
        self.tri = np.asarray(faces, dtype=np.int64)

        # ---- initial wrapped configuration --------------------------------
        x = self._build_wrapped_initial(x0)

        # pinned top edge (row at the rod)
        self.pin_idx = np.flatnonzero(self.i_of == ny - 1).astype(np.int64)
        self.pin_pos = x[self.pin_idx].copy()

        # ---- edges ---------------------------------------------------------
        def edge(stiff, damp):
            ea, eb, lengths = [], [], []
            for i in range(ny):
                for j in range(nx):
                    a = i * nx + j
                    if j + 1 < nx:
                        ea.append(a); eb.append(a + 1)
                        lengths.append(np.linalg.norm(x0[a] - x0[a + 1]))
                    if i + 1 < ny:
                        ea.append(a); eb.append(a + nx)
                        lengths.append(np.linalg.norm(x0[a] - x0[a + nx]))
            return ea, eb, lengths

        def diag_edges():
            ea, eb, lengths = [], [], []
            for i in range(ny - 1):
                for j in range(nx - 1):
                    a = i * nx + j
                    ea.append(a); eb.append(a + nx + 1)
                    lengths.append(np.linalg.norm(x0[a] - x0[a + nx + 1]))
                    a2 = a + 1
                    ea.append(a2); eb.append(a2 + nx)
                    lengths.append(np.linalg.norm(x0[a2] - x0[a2 + nx]))
            return ea, eb, lengths

        def bend_edges():
            ea, eb, lengths = [], [], []
            for i in range(ny):
                for j in range(nx):
                    a = i * nx + j
                    if j + 2 < nx:
                        ea.append(a); eb.append(a + 2)
                        lengths.append(np.linalg.norm(x0[a] - x0[a + 2]))
                    if i + 2 < ny:
                        ea.append(a); eb.append(a + 2 * nx)
                        lengths.append(np.linalg.norm(x0[a] - x0[a + 2 * nx]))
            return ea, eb, lengths

        parts = []
        for (ea, eb, L), k, c in (
            (edge(0, 0), K_STRUCT, C_STRUCT),
            (diag_edges(), K_SHEAR, C_SHEAR),
            (bend_edges(), K_BEND, C_BEND),
        ):
            parts.append((np.asarray(ea, dtype=np.int64), np.asarray(eb, dtype=np.int64),
                          np.asarray(L, dtype=float),
                          np.full(len(L), k), np.full(len(L), c)))
        self.ea = np.concatenate([p[0] for p in parts])
        self.eb = np.concatenate([p[1] for p in parts])
        self.eL = np.concatenate([p[2] for p in parts])
        self.ek = np.concatenate([p[3] for p in parts])
        self.ec = np.concatenate([p[4] for p in parts])
        # structural-only set for strain limiting
        s = edge(0, 0)
        self.ea_s = np.asarray(s[0], dtype=np.int64)
        self.eb_s = np.asarray(s[1], dtype=np.int64)
        self.L_s = np.asarray(s[2], dtype=float)

        # ---- self collision exclusion mask ---------------------------------
        self.tri_center = x0[self.tri].mean(axis=1)  # placeholder, updated per call
        self.excl = np.zeros((n, len(self.tri)), dtype=bool)
        for tidx in range(len(self.tri)):
            for v in self.tri[tidx]:
                vi, vj = self.i_of[v], self.j_of[v]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii, jj = vi + di, vj + dj
                        if 0 <= ii < ny and 0 <= jj < nx:
                            self.excl[ii * nx + jj, tidx] = True

        # ---- deformation gradient neighborhoods ----------------------------
        nbrs = []
        for i in range(ny):
            for j in range(nx):
                a = i * nx + j
                lst = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ii, jj = i + di, j + dj
                        if 0 <= ii < ny and 0 <= jj < nx:
                            b = ii * nx + jj
                            lst.append((b, (self.xcol[jj] - self.xcol[j],
                                            self.srow[ii] - self.srow[i])))
                nbrs.append(lst)
        self.gnbr = [np.asarray([e[0] for e in lst], dtype=np.int64) for lst in nbrs]
        self.gdq = [np.asarray([e[1] for e in lst], dtype=float) for lst in nbrs]
        # neighbor along +x (or -x at the right border) for tangent frames
        self.gx = np.empty(n, dtype=np.int64)
        for i in range(ny):
            for j in range(nx):
                a = i * nx + j
                self.gx[a] = a + 1 if j + 1 < nx else a - 1

        # ---- state ----------------------------------------------------------
        self.x = x
        self.v = np.zeros((n, 3))
        self._cn = np.zeros((n, 3))
        self._cf = np.ones(n)
        self._contact = np.zeros(n, dtype=bool)

        # relaxation cleanup of the initial wrap (positions only)
        for _ in range(60):
            self._collide_static(x, record=False)
            self._self_collide(x, iters=1)
            x[self.pin_idx] = self.pin_pos
        self.x = x
        self.t = 0.0

    # ------------------------------------------------- initial configuration
    def _build_wrapped_initial(self, x0):
        """Wound curtain: hang from the rod, wrap around the ball (2 layers)."""
        n, nx, ny = self.n, self.nx, self.ny
        rho1 = BALL_RADIUS + self.thickness
        rho2 = rho1 + WRAP_LAYER_GAP
        z_ax = BALL_CENTER[2]

        # hang chord from pin (x,0,2.8) to wrap start on the back of the circle
        start = np.array([-rho1 * np.sin(WRAP_THETA0), 0.0, z_ax + rho1 * np.cos(WRAP_THETA0)])
        pin = np.array([0.0, 0.0, 2.8])
        l_hang = float(np.linalg.norm(pin - start))
        s_a = 2.8 - l_hang                 # material coordinate of wrap start
        w_tot = s_a - self.srow[0]         # material wound around the ball
        th1 = (w_tot - WRAP_THETA0 * rho1) / (rho1 + (1.0 + WRAP_TH2_FRAC) * rho2)
        th2 = WRAP_TH2_FRAC * th1
        l1 = (th1 + WRAP_THETA0) * rho1    # material in the inner layer

        x = np.empty_like(x0)
        s = x0[:, 2]                        # material coordinate (rest height)
        u = s_a - s                         # wound material coordinate
        on_hang = u <= 0.0
        # hanging part: straight chord pin -> start
        tpar = np.clip((2.8 - s) / l_hang, 0.0, 1.0)
        hang_pos = pin + tpar[:, None] * (start - pin)
        x[on_hang] = hang_pos[on_hang]

        wrap = ~on_hang
        inner = wrap & (u <= l1)
        outer = wrap & (u > l1)
        theta = np.empty(n)
        theta[inner] = -WRAP_THETA0 + u[inner] / rho1
        theta[outer] = th1 - (u[outer] - l1) / rho2
        radius = np.empty(n)
        radius[inner] = rho1
        radius[outer] = rho2
        x[wrap, 0] = x0[wrap, 0]
        x[wrap, 1] = radius[wrap] * np.sin(theta[wrap])
        x[wrap, 2] = z_ax + radius[wrap] * np.cos(theta[wrap])
        return x

    # ------------------------------------------------------------- dynamics
    def _forces(self, x, v):
        n = self.n
        f = self.mass[:, None] * GRAVITY[None, :]
        f -= GLOBAL_DRAG * self.mass[:, None] * v
        d = x[self.eb] - x[self.ea]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        dist = np.maximum(dist, 1e-9)
        dirn = d / dist[:, None]
        relv = v[self.eb] - v[self.ea]
        vproj = np.einsum("ij,ij->i", relv, dirn)
        fs = self.ek * (dist - self.eL) - self.ec * vproj
        fv = fs[:, None] * dirn
        for k in range(3):
            f[:, k] += np.bincount(self.ea, weights=fv[:, k], minlength=n)
            f[:, k] -= np.bincount(self.eb, weights=fv[:, k], minlength=n)
        return f

    def _strain_limit(self, x):
        d = x[self.eb_s] - x[self.ea_s]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        dist = np.maximum(dist, 1e-9)
        limit = self.L_s * STRETCH_LIMIT
        viol = dist > limit
        if not np.any(viol):
            return
        idx = np.flatnonzero(viol)
        dirn = d[idx] / dist[idx, None]
        mag = STRAIN_FACTOR * (dist[idx] - limit[idx])
        corr = dirn * mag[:, None]
        n = self.n
        for k in range(3):
            # dirn points from ea to eb: pull ea toward eb and eb toward ea
            x[:, k] += np.bincount(self.ea_s[idx], weights=corr[:, k], minlength=n)
            x[:, k] -= np.bincount(self.eb_s[idx], weights=corr[:, k], minlength=n)

    def _collide_static(self, x, record=True):
        thr = BALL_RADIUS + self.thickness
        d = x - BALL_CENTER[None, :]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        m = dist < thr
        if np.any(m):
            nn = d[m] / np.maximum(dist[m], 1e-12)[:, None]
            x[m] = BALL_CENTER[None, :] + thr * nn
            if record:
                self._cn[m] = nn
                self._cf[m] = 0.999
                self._contact[m] = True
        gm = (x[:, 2] < GROUND_TOP_Z + self.thickness) & (np.abs(x[:, 0]) <= GROUND_HALF_X + 0.05) \
            & (np.abs(x[:, 1]) <= GROUND_HALF_Y + 0.05)
        if np.any(gm):
            x[gm, 2] = GROUND_TOP_Z + self.thickness
            if record:
                self._cn[gm] = np.array([0.0, 0.0, 1.0])
                self._cf[gm] = 0.995
                self._contact[gm] = True

    def _self_collide(self, x, iters=1):
        n = self.n
        d0 = SELF_D0
        tri = self.tri
        tc = x[tri].mean(axis=1)
        tc2 = np.einsum("ij,ij->i", tc, tc)
        x2 = np.einsum("ij,ij->i", x, x)
        for _ in range(iters):
            d2 = x2[:, None] + tc2[None, :] - 2.0 * (x @ tc.T)
            mask = d2 < (0.19 + d0) ** 2
            mask &= ~self.excl
            pairs = np.argwhere(mask)
            if len(pairs) == 0:
                return
            vi = pairs[:, 0]
            ti = pairs[:, 1]
            a = x[tri[ti, 0]]
            b = x[tri[ti, 1]]
            c = x[tri[ti, 2]]
            p = x[vi]
            q = _closest_point_on_triangles(p, a, b, c)
            diff = p - q
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            pen = dist < d0
            if not np.any(pen):
                return
            vi = vi[pen]
            ti = ti[pen]
            diff = diff[pen]
            dist = np.maximum(dist[pen], 1e-9)
            dirn = diff / dist[:, None]
            corr = 0.5 * (d0 - dist[:, None]) * dirn
            for k in range(3):
                x[:, k] += np.bincount(vi, weights=corr[:, k], minlength=n)
                x[:, k] -= np.bincount(tri[ti, 0], weights=corr[:, k] / 3.0, minlength=n)
                x[:, k] -= np.bincount(tri[ti, 1], weights=corr[:, k] / 3.0, minlength=n)
                x[:, k] -= np.bincount(tri[ti, 2], weights=corr[:, k] / 3.0, minlength=n)
            # refresh for next iteration
            x2 = np.einsum("ij,ij->i", x, x)

    def _substep(self, dt):
        x, v = self.x, self.v
        f = self._forces(x, v)
        v += dt * f / self.mass[:, None]
        xp = x.copy()
        x += dt * v
        self._strain_limit(x)
        self._cn[:] = 0.0
        self._cf[:] = 1.0
        self._contact[:] = False
        self._collide_static(x)
        self._self_collide(x, iters=1)
        # recover velocity from projected positions (consistent contact response)
        v[:] = (x - xp) / dt
        # friction at contacts
        m = self._contact
        if np.any(m):
            nn = self._cn[m]
            vv = v[m]
            vn = np.einsum("ij,ij->i", vv, nn)
            vt = vv - vn[:, None] * nn
            vn_new = np.maximum(vn, 0.0)
            v[m] = vn_new[:, None] * nn + vt * self._cf[m][:, None]
        # safety: keep finite and bounded
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            x[:] = xp
            v[:] = 0.0
        np.clip(v, -V_MAX, V_MAX, out=v)
        x[self.pin_idx] = self.pin_pos
        v[self.pin_idx] = 0.0
        self.t += dt
        self._steps += 1

    # ---------------------------------------------------------------- frames
    def _vertex_normals(self, x):
        tri = self.tri
        a, b, c = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
        fn = np.cross(b - a, c - a)
        n = np.zeros((self.n, 3))
        for k in range(3):
            n[:, k] = np.bincount(tri[:, 0], weights=fn[:, k], minlength=self.n)
            n[:, k] += np.bincount(tri[:, 1], weights=fn[:, k], minlength=self.n)
            n[:, k] += np.bincount(tri[:, 2], weights=fn[:, k], minlength=self.n)
        ln = np.sqrt(np.einsum("ij,ij->i", n, n))
        bad = ln < 1e-12
        n[bad] = np.array([0.0, 1.0, 0.0])
        n[~bad] /= ln[~bad, None]
        return n

    def _deformation_gradient(self, x):
        nrm = self._vertex_normals(x)
        fout = np.zeros((self.n, 2, 2))
        for i in range(self.n):
            nbr = self.gnbr[i]
            dq = self.gdq[i]
            dp = x[nbr] - x[i]
            nv = nrm[i]
            e = x[self.gx[i]] - x[i]
            e = e - np.dot(e, nv) * nv
            le = np.linalg.norm(e)
            t1 = e / le if le > 1e-12 else np.array([1.0, 0.0, 0.0])
            t2 = np.cross(t1, nv)
            g = dq.T @ dq + 1e-9 * np.eye(2)
            m2 = np.column_stack((dp @ t1, dp @ t2))  # (K,2)
            fout[i] = np.linalg.solve(g, (m2.T @ dq).T).T
        return fout

    def _contact_distances(self, x):
        out = []
        d = np.sqrt(np.einsum("ij,ij->i", x - BALL_CENTER[None, :], x - BALL_CENTER[None, :]))
        ball_gap = d - (BALL_RADIUS + self.thickness)
        out.append(np.array([float(np.min(ball_gap))]))
        near = ball_gap < 0.1
        if np.any(near):
            out.append(ball_gap[near])
        in_foot = (np.abs(x[:, 0]) <= GROUND_HALF_X + 0.05) & (np.abs(x[:, 1]) <= GROUND_HALF_Y + 0.05)
        gz = x[:, 2] - (GROUND_TOP_Z + self.thickness)
        gn = in_foot & (gz < 0.1)
        if np.any(gn):
            out.append(gz[gn])
        # self-contact gaps
        tri = self.tri
        tc = x[tri].mean(axis=1)
        d2 = (np.einsum("ij,ij->i", x, x))[:, None] + np.einsum("ij,ij->i", tc, tc)[None, :] \
            - 2.0 * (x @ tc.T)
        mask = d2 < (0.06 + SELF_D0) ** 2
        mask &= ~self.excl
        pairs = np.argwhere(mask)
        if len(pairs):
            vi, ti = pairs[:, 0], pairs[:, 1]
            q = _closest_point_on_triangles(x[vi], x[tri[ti, 0]], x[tri[ti, 1]], x[tri[ti, 2]])
            dist = np.sqrt(np.einsum("ij,ij->i", x[vi] - q, x[vi] - q))
            sel = dist < SELF_D0 + 0.05
            if np.any(sel):
                out.append(dist[sel] - SELF_D0)
        return np.concatenate(out)

    def _build_frame(self, time):
        x = self.x.copy()
        v = self.v.copy()
        frame = StateFrame(
            float(time),
            x,
            v,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(x),
                "boundary_vertex": self.boundary_attr.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_distances(x),
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": self.mass[:, None] * GRAVITY[None, :],
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------------- api
    def advance_to(self, time):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9 and guard < 100000:
            dt = min(DT_SUB, target - self.t)
            self._substep(dt)
            guard += 1
        return self._build_frame(target if target >= self.t else self.t)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "dt": DT_SUB,
            "n_particles": int(self.n),
            "pinned": int(len(self.pin_idx)),
        }
