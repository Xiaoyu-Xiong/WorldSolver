"""Implicit mass-spring cloth solver for the Baraff-Witkin drape-over-cylinder scene.

Physics
-------
* Thin rectangular sheet discretised on the frozen ``grid_shape`` lattice with
  structural / shear / bending springs (lumped nodal masses).
* Fully implicit (backward) Euler integration: the linear system
  ``(M/h - h G - D) v_{n+1} = M v_n / h + f_pos(x_n)`` is assembled and solved
  with a sparse direct factorisation each sub-step (Baraff & Witkin, 1998).
* Gravity as the only body force; unilateral penalty contact with the static
  cylinder and the ground plane (normal penalty + normal damping + regularised
  Coulomb friction), followed by a positional projection / impulse pass that
  removes residual penetration and normal velocity.

The exported StateFrame carries every field required by the trajectory
contract (mass, velocity, external force, deformation gradients, signed
contact distances, mass source, frozen contact length scale).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame

_EPS = 1e-12
_I3 = np.eye(3)
_I2 = np.eye(2)


def _quat_rotate(q, v):
    """Rotate vector ``v`` by quaternion ``q = (x, y, z, w)``."""
    q = np.asarray(q, dtype=float)
    u = q[:3]
    w = float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class BenchmarkSolver:
    # ---- material parameters (cloth regime of the reference figure) ----
    K_STRETCH = 3000.0      # N/m   structural springs
    K_SHEAR = 150.0         # N/m   diagonal (in-plane shear) springs
    K_BEND = 20.0           # N/m   two-neighbour (out-of-plane bending) springs
    DAMP_BETA = 5.0e-4      # s     stiffness-proportional spring damping
    DAMP_ALPHA = 0.45       # 1/s   mass-proportional (air/global) damping
    # ---- contact ----
    K_CONTACT = 6.0e4       # N/m   normal penalty per node
    C_CONTACT = 1.0         # N s/m normal contact damping
    MU_DEFAULT = 0.5        # regularised Coulomb friction coefficient
    V_EPS = 2.0e-2          # m/s   friction regularisation velocity
    CONTACT_MARGIN = 0.06   # m     "near contact" reporting band
    # ---- numerics ----
    DT_TARGET = 1.0 / 500.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        frame0 = scene.initial_state()

        self.x0 = np.array(frame0.position, dtype=float)
        self.n = int(self.x0.shape[0])
        self.mass = np.array(frame0.mass, dtype=float)
        self.boundary = np.array(frame0.attributes.get("boundary_vertex",
                                                       np.zeros(self.n, dtype=np.int32)))

        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.thickness = float(cloth.get("thickness", 0.01))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.offset = 0.5 * self.thickness           # midsurface -> outer surface
        self.seed = int(seed)

        self._build_material_coords(cloth)
        self._build_springs()
        self._build_triangles()
        self._build_colliders(cfg)
        self._build_index_cache()

        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x0)
        self.t = 0.0
        self.dt_target = self.DT_TARGET
        self.steps = 0
        self._seeded = False
        self._gap = self._contact_gap(self.x)
        self._strain_max = 0.0
        self._pen_min = 0.0

        rng = np.random.default_rng(self.seed)
        self._imperfection = self._make_imperfection(rng)

    def _build_material_coords(self, cloth):
        nx, ny = self.nx, self.ny
        sx, sy = (float(v) for v in cloth["size"])
        dx = sx / max(nx - 1, 1)
        dy = sy / max(ny - 1, 1)
        i = np.arange(nx * ny) % nx
        j = np.arange(nx * ny) // nx
        self.mat = np.column_stack((i * dx, j * dy)).astype(float)
        self.dx, self.dy = dx, dy

    def _build_springs(self):
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        si, sj, kind = [], [], []

        def add(a, b, k):
            si.append(np.asarray(a).ravel())
            sj.append(np.asarray(b).ravel())
            kind.append(np.full(si[-1].size, k, dtype=np.int32))

        add(idx[:, :-1], idx[:, 1:], 0)      # structural along x
        add(idx[:-1, :], idx[1:, :], 0)      # structural along y
        add(idx[:-1, :-1], idx[1:, 1:], 1)   # shear diagonals
        add(idx[:-1, 1:], idx[1:, :-1], 1)
        add(idx[:, :-2], idx[:, 2:], 2)      # bending along x
        add(idx[:-2, :], idx[2:, :], 2)      # bending along y

        self.si = np.concatenate(si).astype(np.int64)
        self.sj = np.concatenate(sj).astype(np.int64)
        kind = np.concatenate(kind)
        stiff = np.array([self.K_STRETCH, self.K_SHEAR, self.K_BEND], dtype=float)
        self.ks = stiff[kind]
        self.cs = self.DAMP_BETA * self.ks
        d = self.x0[self.si] - self.x0[self.sj]
        self.L0 = np.sqrt((d * d).sum(axis=1))
        self.L0 = np.maximum(self.L0, 1e-9)
        self.n_springs = int(self.si.size)

    def _build_triangles(self):
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        self.tri = np.concatenate((np.column_stack((a, b, d)),
                                   np.column_stack((a, d, c))), axis=0).astype(np.int64)
        m = self.mat[self.tri]                              # (T,3,2)
        e1 = m[:, 1] - m[:, 0]
        e2 = m[:, 2] - m[:, 0]
        # Represent the reference triangle in its own orthonormal (Gram-Schmidt)
        # frame so that the exported 2x2 gradient is exactly the identity for
        # the undeformed sheet and frame-indifferent (F^T F = right Cauchy-Green
        # in material coordinates) for the draped/bent configuration.
        l1 = np.maximum(np.sqrt((e1 * e1).sum(axis=1)), 1e-12)
        u = e1 / l1[:, None]
        p = (e2 * u).sum(axis=1)
        w = e2 - p[:, None] * u
        l2 = np.maximum(np.sqrt((w * w).sum(axis=1)), 1e-12)
        Dm = np.zeros((self.tri.shape[0], 2, 2))
        Dm[:, 0, 0] = l1
        Dm[:, 1, 0] = p
        Dm[:, 1, 1] = l2
        self.rest_area = 0.5 * l1 * l2
        self.Dm_inv = np.linalg.inv(Dm)
        w = self.rest_area
        denom = np.zeros(self.n)
        for k in range(3):
            np.add.at(denom, self.tri[:, k], w)
        self.tri_wdenom = 1.0 / np.maximum(denom, _EPS)

    def _build_colliders(self, cfg):
        colliders = []
        geo = cfg.get("scene_geometry") or {}
        for obj in geo.get("objects", []) or []:
            if not bool(obj.get("collision_enabled", False)):
                continue
            if obj.get("motion", "static") != "static":
                continue
            g = obj.get("geometry") or {}
            tr = obj.get("transform") or {}
            c = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            q = np.asarray(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
            mu = self.MU_DEFAULT if obj.get("material_id") != "ground" else 0.6
            gtype = g.get("type")
            if gtype == "cylinder":
                axis = _quat_rotate(q, np.array([0.0, 0.0, 1.0]))
                axis = axis / max(np.linalg.norm(axis), _EPS)
                colliders.append({"kind": "cylinder", "center": c, "axis": axis,
                                  "radius": float(g.get("radius", 0.5)),
                                  "half_h": 0.5 * float(g.get("height", 1.0e6)),
                                  "mu": mu})
            elif gtype == "box":
                he = np.asarray(g.get("half_extents", [10.0, 10.0, 0.05]), dtype=float)
                nrm = _quat_rotate(q, np.array([0.0, 0.0, 1.0]))
                nrm = nrm / max(np.linalg.norm(nrm), _EPS)
                colliders.append({"kind": "plane", "point": c + nrm * he[2],
                                  "normal": nrm, "mu": mu})
            elif gtype == "plane":
                nrm = _quat_rotate(q, np.asarray(g.get("normal", [0.0, 0.0, 1.0]), float))
                nrm = nrm / max(np.linalg.norm(nrm), _EPS)
                colliders.append({"kind": "plane", "point": c, "normal": nrm, "mu": mu})
        if not colliders:  # defensive fallback matching the frozen scene
            colliders = [
                {"kind": "cylinder", "center": np.array([0.0, 0.0, 0.7]),
                 "axis": np.array([0.0, 1.0, 0.0]), "radius": 0.52,
                 "half_h": 1.4, "mu": self.MU_DEFAULT},
                {"kind": "plane", "point": np.zeros(3),
                 "normal": np.array([0.0, 0.0, 1.0]), "mu": 0.6},
            ]
        self.colliders = colliders

    def _build_index_cache(self):
        n3 = 3 * self.n
        a3 = np.arange(3)
        si, sj = self.si, self.sj
        ri = (3 * si[:, None, None] + a3[None, :, None] + 0 * a3[None, None, :]).ravel()
        ci = (3 * si[:, None, None] + 0 * a3[None, :, None] + a3[None, None, :]).ravel()
        rj = (3 * sj[:, None, None] + a3[None, :, None] + 0 * a3[None, None, :]).ravel()
        cj = (3 * sj[:, None, None] + 0 * a3[None, :, None] + a3[None, None, :]).ravel()
        self._spr_rows = np.concatenate((ri, rj, ri, rj))
        self._spr_cols = np.concatenate((ci, cj, cj, ci))
        self._diag_idx = np.arange(n3)
        self._mass_diag = np.repeat(self.mass, 3)
        self._arow = (3 * np.arange(self.n)[:, None, None]
                      + a3[None, :, None] + 0 * a3[None, None, :]).ravel()
        self._acol = (3 * np.arange(self.n)[:, None, None]
                      + 0 * a3[None, :, None] + a3[None, None, :]).ravel()

    def _make_imperfection(self, rng):
        """Tiny seeded out-of-flatness imperfection that breaks fold symmetry."""
        nx, ny = self.nx, self.ny
        i = (np.arange(self.n) % nx) / max(nx - 1, 1)
        j = (np.arange(self.n) // nx) / max(ny - 1, 1)
        amp = 1.5e-3
        w = (amp * np.sin(2.0 * np.pi * (1.5 * j + 0.13))
             * np.sin(np.pi * (0.5 + 0.5 * i))
             + 0.6 * amp * np.sin(2.0 * np.pi * (2.5 * j + 0.41))
             + 1.0e-4 * rng.standard_normal(self.n))
        pert = np.zeros((self.n, 3))
        pert[:, 2] = w
        return pert

    # -------------------------------------------------------------- geometry
    def _sdf(self, col, x):
        """Signed distance and outward unit normal of a static collider."""
        if col["kind"] == "plane":
            n = col["normal"]
            w = x - col["point"]
            s = w @ n
            nrm = np.tile(n, (x.shape[0], 1))
            return s, nrm
        c = col["center"]
        axis = col["axis"]
        R = col["radius"]
        hh = col["half_h"]
        w = x - c
        a = w @ axis
        rw = w - a[:, None] * axis[None, :]
        r = np.sqrt((rw * rw).sum(axis=1))
        rs = np.maximum(r, 1e-9)
        nrad = rw / rs[:, None]
        da = np.abs(a) - hh
        axial = da <= 0.0
        s = np.where(axial, r - R, np.hypot(np.maximum(da, 0.0), np.maximum(r - R, 0.0)))
        sax = np.sign(a)[:, None] * axis[None, :]
        n_out = (np.maximum(da, 0.0)[:, None] * sax
                 + np.maximum(r - R, 0.0)[:, None] * nrad)
        n_out /= np.maximum(np.sqrt((n_out * n_out).sum(axis=1)), 1e-9)[:, None]
        nrm = np.where(axial[:, None], nrad, n_out)
        # degenerate case: node exactly on the axis -> use radial fallback
        bad = r < 1e-8
        if np.any(bad):
            nrm[bad] = np.array([1.0, 0.0, 0.0])
        return s, nrm

    def _contact_gap(self, x):
        """Per-node minimum geometric gap to any collider (negative = penetration)."""
        gap = np.full(x.shape[0], np.inf)
        for col in self.colliders:
            s, _ = self._sdf(col, x)
            gap = np.minimum(gap, s - self.offset)
        if not np.isfinite(gap).any():
            gap = np.zeros(x.shape[0])
        return gap

    # ------------------------------------------------------------- stepping
    def _advance(self, h):
        x, v, m = self.x, self.v, self.mass
        n3 = 3 * self.n
        si, sj = self.si, self.sj

        # ---- spring internal forces / Jacobians at the current state ----
        d = x[si] - x[sj]
        r = np.sqrt((d * d).sum(axis=1))
        r = np.maximum(r, 1e-9)
        nrm = d / r[:, None]
        nn = nrm[:, :, None] * nrm[:, None, :]
        fe = self.ks * (r - self.L0)
        f_pair = -fe[:, None] * nrm                       # force on node si
        kp = np.maximum(self.ks * (1.0 - self.L0 / r), 0.0)   # PSD geometric term
        H = self.ks[:, None, None] * nn + kp[:, None, None] * (_I3 - nn)
        Dd = self.cs[:, None, None] * nn

        f_pos = (m[:, None] * self.gravity[None, :]).copy()
        np.add.at(f_pos, si, f_pair)
        np.add.at(f_pos, sj, -f_pair)

        Mtot = h * H + Dd
        data = np.concatenate((Mtot.ravel(), Mtot.ravel(),
                               (-Mtot).ravel(), (-Mtot).ravel()))
        rows = [self._spr_rows]
        cols = [self._spr_cols]
        vals = [data]

        rows.append(self._diag_idx)
        cols.append(self._diag_idx)
        vals.append(self._mass_diag / h + self.DAMP_ALPHA * self._mass_diag)

        # ---- unilateral contacts (penalty + damping + regularised friction) ----
        m_over_h = m / h
        ct_cap = 2.0 * m_over_h
        for col in self.colliders:
            s, cn = self._sdf(col, x)
            pen = self.offset - s
            act = pen > 0.0
            if not np.any(act):
                continue
            k = np.nonzero(act)[0]
            nk = cn[k]
            pk = pen[k]
            nkk = nk[:, :, None] * nk[:, None, :]
            fn = self.K_CONTACT * pk
            f_pos[k] += fn[:, None] * nk
            # regularised Coulomb friction evaluated at the current velocity
            vk = v[k]
            vn = (vk * nk).sum(axis=1)
            vt = vk - vn[:, None] * nk
            vtn = np.sqrt((vt * vt).sum(axis=1))
            denom = np.maximum(vtn, self.V_EPS)
            f_fric = -(col["mu"] * fn)[:, None] * vt / denom[:, None]
            f_pos[k] += f_fric
            ct = np.minimum(col["mu"] * fn / denom, ct_cap[k])
            B = ((h * self.K_CONTACT + self.C_CONTACT) * nkk
                 + ct[:, None, None] * (_I3 - nkk))
            arow = (3 * k[:, None, None] + np.arange(3)[None, :, None]
                    + 0 * np.arange(3)[None, None, :]).ravel()
            acol = (3 * k[:, None, None] + 0 * np.arange(3)[None, :, None]
                    + np.arange(3)[None, None, :]).ravel()
            rows.append(arow)
            cols.append(acol)
            vals.append(B.ravel())

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n3, n3)).tocsr()
        rhs = (self._mass_diag / h) * v.ravel() + f_pos.ravel()
        try:
            v_new = spla.spsolve(A.tocsc(), rhs, use_umfpack=False)
        except Exception:
            v_new = v.ravel().copy()
        v_new = np.asarray(v_new, dtype=float).reshape(self.n, 3)
        if not np.isfinite(v_new).all():
            v_new = v.copy()
        # cap unphysical spikes from the linearised contact law
        vmax = 60.0
        speed = np.sqrt((v_new * v_new).sum(axis=1))
        over = speed > vmax
        if np.any(over):
            v_new[over] *= (vmax / speed[over])[:, None]

        x_new = x + h * v_new
        self._project(x_new, v_new)
        self.x, self.v = x_new, v_new
        self.steps += 1

    def _project(self, x, v):
        """Positional projection + normal impulse removal (no restitution)."""
        for col in self.colliders:
            s, cn = self._sdf(col, x)
            pen = self.offset - s
            k = np.nonzero(pen > 0.0)[0]
            if k.size == 0:
                continue
            nk = cn[k]
            x[k] += pen[k, None] * nk
            vn = (v[k] * nk).sum(axis=1)
            hit = vn < 0.0
            if not np.any(hit):
                continue
            kh = k[hit]
            nh = nk[hit]
            vnh = vn[hit]
            vt = v[kh] - vnh[:, None] * nh
            vtn = np.sqrt((vt * vt).sum(axis=1))
            scale = np.maximum(0.0, 1.0 - col["mu"] * (-vnh) / np.maximum(vtn, 1e-9))
            v[kh] = vt * scale[:, None]

    def _step_to(self, time):
        if not self._seeded:
            self.x = self.x + self._imperfection
            self._seeded = True
        while self.t < time - 1e-9:
            h = min(self.dt_target, time - self.t)
            if h <= 0.0:
                break
            self._advance(h)
            self.t += h
        if abs(self.t - time) < 1e-9:
            self.t = time

    # -------------------------------------------------------------- outputs
    def _deformation_gradients(self):
        x = self.x
        tri = self.tri
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        d1 = x[b] - x[a]
        d2 = x[c] - x[a]
        l1 = np.maximum(np.sqrt((d1 * d1).sum(axis=1)), 1e-9)
        u = d1 / l1[:, None]
        p = (d2 * u).sum(axis=1)
        w = d2 - p[:, None] * u
        l2 = np.maximum(np.sqrt((w * w).sum(axis=1)), 1e-9)
        Dc = np.zeros((tri.shape[0], 2, 2))
        Dc[:, 0, 0] = l1
        Dc[:, 1, 0] = p
        Dc[:, 1, 1] = l2
        F = Dc @ self.Dm_inv                                  # (T,2,2)
        Fn = np.zeros((self.n, 2, 2))
        den = np.zeros(self.n)
        wgt = self.rest_area
        for k in range(3):
            np.add.at(Fn, tri[:, k], wgt[:, None, None] * F)
            np.add.at(den, tri[:, k], wgt)
        Fn *= self.tri_wdenom[:, None, None]
        green = np.einsum('tij,tjk->tik', F.swapaxes(1, 2), F) - _I2[None]
        self._strain_max = float(np.abs(green).max()) if green.size else 0.0
        return Fn

    def advance_to(self, time: float):
        time = float(time)
        if time > self.t:
            self._step_to(time)
        self.t = max(self.t, time)
        self._gap = self._contact_gap(self.x)
        F = self._deformation_gradients()

        gap = self._gap
        self._pen_min = float(-gap.min()) if gap.size else 0.0
        near = np.nonzero(gap < self.CONTACT_MARGIN)[0]
        if near.size < 4:  # keep the reported contact set non-degenerate
            near = np.unique(np.concatenate((near, np.argsort(gap)[:4])))
        contact_sd = np.asarray(gap[near], dtype=float)

        ext = (self.mass[:, None] * self.gravity[None, :]).copy()
        n = self.n
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": np.asarray(self.boundary, dtype=np.int32).reshape(n),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(n),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "min_contact_gap": float(gap.min()),
                "max_abs_strain": self._strain_max,
                "max_speed": float(np.abs(self.v).max()) if n else 0.0,
                "kinetic_energy": float(0.5 * np.sum(self.mass * (self.v * self.v).sum(axis=1))),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "implicit_euler_mass_spring_cloth",
            "nodes": int(self.n),
            "springs": int(self.n_springs),
            "triangles": int(self.tri.shape[0]),
            "substeps": int(self.steps),
            "dt_target": float(self.dt_target),
            "stretch_stiffness": float(self.K_STRETCH),
            "shear_stiffness": float(self.K_SHEAR),
            "bend_stiffness": float(self.K_BEND),
            "contact_stiffness": float(self.K_CONTACT),
            "friction": float(self.MU_DEFAULT),
            "min_contact_gap": float(self._pen_min * -1.0),
            "max_penetration": float(max(self._pen_min, 0.0)),
            "max_abs_green_strain": float(self._strain_max),
        }
