"""Yarn-level-inspired knit patch uniaxial stretch solver.

Physics
-------
The knitted patch is a mass-spring lattice on the frozen 17x17 grid.  The
knit-specific response of Kaldor et al., "Simulating Knitted Cloth at the Yarn
Level" (Fig. 8) is captured with a homogenised loop/yarn constitutive law:

* wale (x) and course (y) springs are very compliant at small strain -- loops
  rotate and uncrimp -- and stiffen quadratically once the yarn is straight;
* the course rest length shrinks as the local wale strain grows (loops tighten
  and pull yarn transversely), producing the large directional extension with
  strong transverse contraction that distinguishes knits from wovens;
* shear and bending springs keep the sheet coherent and let it wrinkle/fold.

Integration is a linearly-implicit (velocity) Euler step using a frozen
rest-configuration tangent stiffness evaluated at the stiffened yarn tangent,
lumped masses and Rayleigh damping: unconditionally stable for the whole
stretch even though the yarn law stiffens by an order of magnitude.

Boundary conditions: the left edge is clamped (fixed grip), the right edge
follows a prescribed smooth ramp / hold / partial unload (moving grip).
Ground and both grip boxes are treated with signed-distance penalty contact
and their gaps are exported.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame

# ------------------------------------------------------------- material -----
E2D_WALE_SOFT = 150.0      # N/m   loop-dominated small-strain stiffness (x)
E2D_WALE_STIFF = 3000.0    # N/m   straightened-yarn tangent stiffness (x)
E2D_COURSE_SOFT = 250.0    # N/m   (y)
E2D_COURSE_STIFF = 4000.0  # N/m   (y)
UNCrimp_WALE = 0.12        # strain where wale loops are pulled straight
UNCrimp_COURSE = 0.08
K_SHEAR = 4.0              # N per shear spring
K_BEND = 0.6               # N per bending (skip-one) spring
COUP_Y = 0.85              # course rest-length shrink per unit wale strain
COUP_X = 0.25              # wale rest-length shrink per unit course strain
COUP_CAP = 0.32
COMP_RATIO = 0.5           # compression/tension stiffness ratio (buckling)
STRAIN_MAX = 0.30          # strain at which the implicit operator is frozen
C_MASS = 2.0               # 1/s mass-proportional damping
C_STIFF = 4.0e-4           # s   stiffness-proportional damping
SUBSTEPS = 16              # substeps per exported sample interval

# ------------------------------------------------- prescribed grip motion ---
RAMP_T0, RAMP_T1 = 0.40, 4.40   # smooth extension
HOLD_T1 = 6.00                  # hold under tension
UNLOAD_T1 = 7.00                # partial elastic unload
GRIP_TRAVEL = 0.75              # ~25 % engineering strain on the 3.0 m patch
UNLOAD_FRACTION = 0.25

GROUND_Z = 0.0
GRIP_CENTERS = (np.array([-1.5, 0.0, 2.15]), np.array([1.5, 0.0, 2.15]))
GRIP_HALF = np.array([0.12, 1.25, 0.12])
K_CONTACT = 400.0
C_CONTACT = 0.6


def _smoothstep(tau: float) -> float:
    return tau * tau * (3.0 - 2.0 * tau)


def _smoothstep_dot(tau: float) -> float:
    return 6.0 * tau * (1.0 - tau)


class BenchmarkSolver:
    """Knitted patch pulled in uniaxial tension between two grips."""

    # ------------------------------------------------------------- set-up ---
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        nx, ny = (int(v) for v in cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        self.seed = int(seed)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.dt = self.sample_dt / SUBSTEPS
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.n = len(self.x0)

        # deterministic low-frequency out-of-plane seed (buckling symmetry break)
        i = np.arange(nx) / max(nx - 1, 1)
        j = np.arange(ny) / max(ny - 1, 1)
        gi, gj = np.meshgrid(i, j, indexing="xy")
        wig = (np.sin(2.0 * np.pi * gi * 1.5) * np.cos(2.0 * np.pi * gj)
               + 0.5 * np.sin(2.0 * np.pi * gj * 2.0 + 0.7))
        self.x0[:, 2] += 1.5e-3 * (wig * np.sin(np.pi * np.clip(gi, 0.0, 1.0))).ravel()
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x0)

        # grid spacings (node ordering is idx = j*nx + i, matching the topology asset)
        self.hx = float(abs(self.x0[1, 0] - self.x0[0, 0]))
        self.hy = float(abs(self.x0[nx, 0] - self.x0[0, 0]))

        self._build_springs()
        self._build_constraints()
        self._build_operator()
        self._build_triangles()

        self.step_index = 0
        self.time = 0.0
        self.min_contact = 1.0e9
        self.max_strain = 0.0
        self.max_contact_force = 0.0
        self._diag = {"model": "knit_mass_spring_lattice",
                      "substeps_per_sample": SUBSTEPS,
                      "spring_count": int(self._spring_count)}

    def _idx(self, i, j):
        return j * self.nx + i

    def _pairs(self, di, dj):
        nx, ny = self.nx, self.ny
        a, b = [], []
        for j in range(ny):
            for i in range(nx):
                i2, j2 = i + di, j + dj
                if 0 <= i2 < nx and 0 <= j2 < ny:
                    a.append(self._idx(i, j))
                    b.append(self._idx(i2, j2))
        return np.asarray(a, dtype=int), np.asarray(b, dtype=int)

    def _rest_len(self, a, b):
        return np.linalg.norm(self.x0[b] - self.x0[a], axis=1)

    def _nonlinear_group(self, a, b, e_soft, e_stiff, s0, spacing):
        return {
            "a": a, "b": b, "L0": self._rest_len(a, b),
            "k1": e_soft * spacing,
            "k2": (e_stiff - e_soft) * spacing / max(1e-9, 2.0 * (STRAIN_MAX - s0)),
            "s0": s0,
            "kbuild": e_stiff * spacing,
        }

    def _build_springs(self):
        nx, ny = self.nx, self.ny
        ax, bx = self._pairs(1, 0)
        ay, by = self._pairs(0, 1)
        self.g_x = self._nonlinear_group(ax, bx, E2D_WALE_SOFT, E2D_WALE_STIFF,
                                         UNCrimp_WALE, self.hy)
        self.g_y = self._nonlinear_group(ay, by, E2D_COURSE_SOFT, E2D_COURSE_STIFF,
                                         UNCrimp_COURSE, self.hx)

        self.shear = []
        for a, b in (self._pairs(1, 1), self._pairs(1, -1)):
            self.shear.append({"a": a, "b": b, "L0": self._rest_len(a, b),
                               "k": K_SHEAR, "kbuild": K_SHEAR})
        self.bend = []
        for di, dj in ((2, 0), (0, 2)):
            a, b = self._pairs(di, dj)
            self.bend.append({"a": a, "b": b, "L0": self._rest_len(a, b),
                              "k": K_BEND, "kbuild": K_BEND})

        # spring ordering: wale index = j*(nx-1)+i ; course index = j*nx+i
        self.xs_index_of = {(i, j): j * (nx - 1) + i
                            for j in range(ny) for i in range(nx - 1)}
        self.ys_index_of = {(i, j): j * nx + i
                            for j in range(ny - 1) for i in range(nx)}

        # transverse-neighbour tables implementing loop tightening
        tab_y = []   # for each course spring: neighbouring wale springs
        for j in range(ny - 1):
            for i in range(nx):
                il, ir = max(i - 1, 0), min(i, nx - 2)
                tab_y.append((self.xs_index_of[(il, j)], self.xs_index_of[(ir, j)],
                              self.xs_index_of[(il, j + 1)], self.xs_index_of[(ir, j + 1)]))
        tab_x = []   # for each wale spring: neighbouring course springs
        for j in range(ny):
            for i in range(nx - 1):
                jb, jt = max(j - 1, 0), min(j, ny - 2)
                tab_x.append((self.ys_index_of[(i, jb)], self.ys_index_of[(i, jt)],
                              self.ys_index_of[(i + 1, jb)], self.ys_index_of[(i + 1, jt)]))
        self.g_y["nb"] = np.asarray(tab_y, dtype=int)
        self.g_x["nb"] = np.asarray(tab_x, dtype=int)

        self._spring_count = (len(self.g_x["a"]) + len(self.g_y["a"])
                              + sum(len(g["a"]) for g in self.shear)
                              + sum(len(g["a"]) for g in self.bend))

    # -------------------------------------------------------- constraints ---
    def _build_constraints(self):
        nx, ny = self.nx, self.ny
        self.fixed_nodes = np.asarray([self._idx(0, j) for j in range(ny)])
        self.moving_nodes = np.asarray([self._idx(nx - 1, j) for j in range(ny)])
        self.gripped = np.zeros(self.n, dtype=bool)
        self.gripped[self.fixed_nodes] = True
        self.gripped[self.moving_nodes] = True
        free = np.ones(3 * self.n, dtype=bool)
        for node in self.fixed_nodes:
            free[3 * node:3 * node + 3] = False
        for node in self.moving_nodes:
            free[3 * node:3 * node + 3] = False
        self.free = np.flatnonzero(free)
        self.con = np.flatnonzero(~free)
        self.clamped_order = np.concatenate((self.fixed_nodes, self.moving_nodes))
        # patch outline (all four edges), as declared by the frozen scaffold
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        edge = (gx == 0) | (gx == nx - 1) | (gy == 0) | (gy == ny - 1)
        self.boundary = edge.ravel().astype(np.int32)

    def _grip_disp(self, t):
        """Prescribed uniaxial extension: (displacement, velocity) of moving grip."""
        if t <= RAMP_T0:
            s, sd = 0.0, 0.0
        elif t <= RAMP_T1:
            tau = (t - RAMP_T0) / (RAMP_T1 - RAMP_T0)
            s = _smoothstep(tau)
            sd = _smoothstep_dot(tau) / (RAMP_T1 - RAMP_T0)
        elif t <= HOLD_T1:
            s, sd = 1.0, 0.0
        else:
            span = max(1e-9, UNLOAD_T1 - HOLD_T1)
            tau = min(1.0, (t - HOLD_T1) / span)
            s = 1.0 - UNLOAD_FRACTION * _smoothstep(tau)
            sd = -UNLOAD_FRACTION * _smoothstep_dot(tau) / span
        return GRIP_TRAVEL * s, GRIP_TRAVEL * sd

    def _targets(self, t):
        disp, vel = self._grip_disp(t)
        pos = self.x0[self.clamped_order].copy()
        vel3 = np.zeros_like(pos)
        n_move = len(self.moving_nodes)
        pos[-n_move:, 0] += disp
        vel3[-n_move:, 0] = vel
        return pos, vel3

    # ---------------------------------------------------------- operator ----
    def _build_operator(self):
        rows, cols, vals = [], [], []
        for g in [self.g_x, self.g_y] + self.shear + self.bend:
            a, b = g["a"], g["b"]
            kb = np.full(len(a), float(g["kbuild"]))
            d = self.x0[b] - self.x0[a]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            n = d / L[:, None]
            blk = kb[:, None, None] * (n[:, :, None] * n[:, None, :])
            ri, rj, rv = [], [], []
            for k in range(len(a)):
                ia, ib = 3 * int(a[k]), 3 * int(b[k])
                for p in range(3):
                    for q in range(3):
                        val = float(blk[k, p, q])
                        if val == 0.0:
                            continue
                        ri += [ia + p, ib + q, ia + p, ib + q]
                        rj += [ia + q, ib + p, ib + q, ia + p]
                        rv += [val, val, -val, -val]
            rows.append(np.asarray(ri)); cols.append(np.asarray(rj)); vals.append(np.asarray(rv))
        K = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(3 * self.n, 3 * self.n),
        ).tocsr()

        m = np.repeat(self.mass, 3)
        A = ((1.0 / self.dt + C_MASS) * sp.diags(m) + (self.dt + C_STIFF) * K).tocsr()
        self.A_ff = splu(A[self.free][:, self.free].tocsc())
        self.A_fc = A[self.free][:, self.con].tocsr()
        self.m_over_dt = m / self.dt

    # --------------------------------------------------------- triangles ----
    def _build_triangles(self):
        nx, ny = self.nx, self.ny
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n00, n10 = self._idx(i, j), self._idx(i + 1, j)
                n11, n01 = self._idx(i + 1, j + 1), self._idx(i, j + 1)
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
        self.tris = np.asarray(tris, dtype=int)
        p0 = self.x0[self.tris[:, 0]][:, :2]
        Dm = np.stack((self.x0[self.tris[:, 1]][:, :2] - p0,
                       self.x0[self.tris[:, 2]][:, :2] - p0), axis=2)
        self.invDm = np.linalg.inv(Dm)
        self.tri_flat = self.tris.ravel()
        self.tri_counts = np.bincount(self.tri_flat, minlength=self.n).astype(float)
        self.tri_counts[self.tri_counts == 0] = 1.0

    # ------------------------------------------------------------ forces ----
    def _strain(self, x, g):
        d = x[g["b"]] - x[g["a"]]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        return d, L, (L - g["L0"]) / g["L0"]

    def _apply_spring(self, g, x, f, d, L, s, L0eff=None):
        s = np.clip(s, -0.9, 0.75)
        if L0eff is not None:
            s = np.clip((L - L0eff) / L0eff, -0.9, 0.75)
        if "k2" in g:
            stiff = g["k2"] * np.maximum(s - g["s0"], 0.0) ** 2
            T = np.where(s >= 0.0, g["k1"] * s + stiff, COMP_RATIO * g["k1"] * s)
        else:
            T = g["k"] * s
        fdir = (T / L)[:, None] * d
        np.add.at(f, g["a"], fdir)
        np.add.at(f, g["b"], -fdir)

    def _internal_forces(self, x):
        f = np.zeros_like(x)
        dx, Lx, sx = self._strain(x, self.g_x)
        dy, Ly, sy = self._strain(x, self.g_y)
        # loop tightening: rest lengths contract with the transverse strain
        L0x_eff = self.g_x["L0"] * (1.0 - COUP_X * np.clip(sy[self.g_x["nb"]].mean(1), 0.0, COUP_CAP))
        L0y_eff = self.g_y["L0"] * (1.0 - COUP_Y * np.clip(sx[self.g_y["nb"]].mean(1), 0.0, COUP_CAP))
        self._apply_spring(self.g_x, x, f, dx, Lx, sx, L0x_eff)
        self._apply_spring(self.g_y, x, f, dy, Ly, sy, L0y_eff)
        for g in self.shear:
            d, L, s = self._strain(x, g)
            self._apply_spring(g, x, f, d, L, s)
        for g in self.bend:
            d, L, s = self._strain(x, g)
            self._apply_spring(g, x, f, d, L, s)
        return f, sx

    def _contact(self, x, v, with_force=True):
        """Signed gaps to ground/grips and penalty force on penetration."""
        f = np.zeros_like(x)
        half = 0.5 * self.thickness
        d_ground = x[:, 2] - GROUND_Z - half
        dist = d_ground.copy()
        disp, _ = self._grip_disp(self.time)
        free_nodes = ~self.gripped
        for k, center in enumerate(GRIP_CENTERS):
            c = center.copy()
            if k == 1:
                c[0] += disp                      # moving grip travels with the edge
            q = np.abs(x - c) - GRIP_HALF
            db = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
            dist[free_nodes] = np.minimum(dist[free_nodes], db[free_nodes])
            if with_force:
                sel = free_nodes & (db < 0.0)
                if np.any(sel):
                    axis = np.argmax(q[sel], axis=1)
                    sgn = np.sign(x[sel] - c)[np.arange(len(axis)), axis]
                    sgn[sgn == 0.0] = 1.0
                    nvec = np.zeros((int(sel.sum()), 3))
                    nvec[np.arange(len(axis)), axis] = sgn
                    depth = -db[sel]
                    vn = np.einsum("ij,ij->i", v[sel], nvec)
                    push = K_CONTACT * depth - C_CONTACT * np.minimum(vn, 0.0)
                    f[sel] += nvec * push[:, None]
                    self.max_contact_force = max(self.max_contact_force, float(np.max(np.abs(push))))
        if with_force:
            pen = np.minimum(d_ground, 0.0)
            f[:, 2] += -K_CONTACT * pen - C_CONTACT * np.where(pen < 0.0, np.minimum(v[:, 2], 0.0), 0.0)
        return f, dist

    # --------------------------------------------------------------- step ---
    def _step(self):
        t1 = (self.step_index + 1) * self.dt
        tpos, tvel = self._targets(t1)
        nodes = self.clamped_order
        self.x[nodes] = tpos
        self.v[nodes] = tvel

        f_int, sx = self._internal_forces(self.x)
        f_ext = self.mass[:, None] * self.gravity[None, :]
        f_con, dist = self._contact(self.x, self.v)
        self.min_contact = min(self.min_contact, float(np.min(dist)))
        self.max_strain = max(self.max_strain, float(np.max(sx)))

        rhs = self.m_over_dt * self.v.reshape(-1) + (f_ext + f_int + f_con).reshape(-1)
        vflat = self.v.reshape(-1)                 # constrained entries hold tvel
        vc = vflat[self.con].copy()
        vf = self.A_ff.solve(rhs[self.free] - self.A_fc @ vc)
        v_new = vflat.copy()
        v_new[self.free] = vf
        v_new[self.con] = vc
        self.v = v_new.reshape(-1, 3)
        x_new = self.x + self.dt * self.v
        x_new[nodes] = tpos
        self.x = x_new
        self.time = t1
        self.step_index += 1

    # -------------------------------------------------------------- frame ---
    def _deformation_gradient(self):
        p = self.x[:, :2]
        p0 = p[self.tris[:, 0]]
        D = np.stack((p[self.tris[:, 1]] - p0, p[self.tris[:, 2]] - p0), axis=2)
        F = D @ self.invDm
        Fn = np.zeros((self.n, 2, 2))
        np.add.at(Fn, self.tri_flat, np.repeat(F, 3, axis=0))
        return Fn / self.tri_counts[:, None, None]

    def _frame(self, time):
        f_ext = self.mass[:, None] * self.gravity[None, :]
        _, dist = self._contact(self.x, self.v, with_force=False)
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": f_ext,
                "contact_signed_distance": dist,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.step_index < target:
            self._step()
        return self._frame(float(time))

    def diagnostics(self):
        d = dict(self._diag)
        d.update({
            "steps": int(self.step_index),
            "final_time": float(self.time),
            "max_wale_strain": float(self.max_strain),
            "max_contact_force": float(self.max_contact_force),
            "min_contact_signed_distance": (float(self.min_contact)
                                            if np.isfinite(self.min_contact) else None),
        })
        return d
