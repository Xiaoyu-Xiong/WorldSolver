"""Physics solver for the tightly twisted seven-rod bundle (072).

Model
-----
Each rod is a thin volumetric elastic solid whose closed triangulated surface
(49 rings x 10 nodes + 2 axis nodes per rod) is discretised as a nonlinear
mass-spring network calibrated to a continuum with Young's modulus E:
every mesh edge is a linear spring with stiffness k = E * (pi r^2 / 10) / l0,
which reproduces the axial stiffness EA and (through the surface node lever
arm) the bending stiffness EI ~ E * pi r^4 / 4 of a solid circular rod, while
the triangulated diagonals and hoops provide shear/torsional resistance.

Time integration is a linearly-implicit (frozen-material-stiffness) Euler
scheme with a single prefactorised sparse SPD matrix, mass-proportional and
pairwise (rest-direction) damping.  Rod-rod and rod-self contact is enforced
by a penalty repulsion between every surface node and the centreline station
points of other (sufficiently distant) rods, which reproduces exactly the
non-penetration condition of two cylinders of radius r for straight
centrelines (node vs. opposite axis distance < r  <=>  axes closer than 2r).

Boundary conditions: nodes inside the driven end regions (|x| >= L/2 - 0.38)
are kinematically prescribed -- rotated about the bundle x-axis at
-72 deg/s (left) and +72 deg/s (right), axial position fixed -- which winds
the bundle, makes the rods wrap around one another, and drives the helical
bundle into repeated tight buckled coils with close rod-rod contact.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

# --- numerical / material parameters -------------------------------------
DT = 2.5e-3            # substep (s)
E_MOD = 1.0e6          # Young's modulus (Pa); density given by config
ALPHA_D = 2.0          # mass-proportional damping (1/s)
ZETA = 0.10            # pairwise spring damping ratio factor
KH_ISO = 0.05          # isotropic augmentation of frozen stiffness
CONTACT_K = 8.0e4      # penalty stiffness (N/m)
CONTACT_R = 0.11       # node-vs-centreline activation radius (m)
SEARCH_R = 0.16        # neighbour search radius (m)
SEARCH_EVERY = 4       # substeps between contact searches
MIN_SELF_DI = 2        # min ring-index gap for same-rod self contact

NRING = 49
NTH = 10
NROD = 7
NPER = NRING * NTH + 2     # 492 nodes per rod
NSEG = 48


def _rod_edges(length: float, radius: float):
    """Unique rest edges of one rod surface (local coords, rod along +x)."""
    x = np.linspace(-0.5 * length, 0.5 * length, NRING)
    theta = 2.0 * np.pi * np.arange(NTH) / NTH
    verts = np.array([[xx, radius * np.cos(t), radius * np.sin(t)]
                      for xx in x for t in theta])
    verts = np.vstack((verts, [-0.5 * length, 0.0, 0.0],
                       [0.5 * length, 0.0, 0.0]))
    faces = []
    for i in range(NSEG):
        for j in range(NTH):
            a, b = i * NTH + j, i * NTH + (j + 1) % NTH
            c, d = (i + 1) * NTH + j, (i + 1) * NTH + (j + 1) % NTH
            faces.extend(((a, c, d), (a, d, b)))
    left, right = NSEG * NTH + 0, NSEG * NTH + 1
    for j in range(NTH):
        faces.extend(((left, (j + 1) % NTH, j),
                      (right, NSEG * NTH + j, NSEG * NTH + (j + 1) % NTH)))
    eset = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            eset.add((min(u, v), max(u, v)))
    e = np.array(sorted(eset), dtype=np.int64)
    l0 = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    return verts, e, l0


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        self.scene = scene
        g, bc = cfg["geometry"], cfg["boundary_conditions"]
        self.length = float(g["rod_length"])
        self.radius = float(g["rod_radius"])
        self.ring_r = float(g["bundle_ring_radius"])
        self.omega = (np.deg2rad(float(bc["left_end_angular_speed_deg_s"])),
                      np.deg2rad(float(bc["right_end_angular_speed_deg_s"])))
        self.t_start = float(bc["start_time"])
        self.t_stop = float(bc["stop_time"])

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int64)
        self.rest = np.array(frame0.attributes["rest_position"], dtype=float)
        self.left = np.asarray(frame0.attributes["driven_left"]).astype(bool)
        self.right = np.asarray(frame0.attributes["driven_right"]).astype(bool)
        self.n = len(self.x)
        self.t = 0.0

        driven = self.left | self.right
        self.driven = driven
        self.free = ~driven
        self.fi = np.flatnonzero(self.free)
        self.di = np.flatnonzero(driven)

        # ---- spring network -------------------------------------------
        _, e_loc, l0_loc = _rod_edges(self.length, self.radius)
        area_share = np.pi * self.radius ** 2 / NTH
        k_loc = E_MOD * area_share / l0_loc
        offs = (np.arange(NROD, dtype=np.int64) * NPER)[:, None, None]
        self.e0 = (e_loc[None, :, 0] + offs).ravel()
        self.e1 = (e_loc[None, :, 1] + offs).ravel()
        self.l0 = np.tile(l0_loc, NROD)
        self.k = np.tile(k_loc, NROD)
        dvec = self.rest[self.e0] - self.rest[self.e1]
        dlen = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
        self.u0 = dvec / dlen[:, None]                       # rest directions
        mbar = 2.0 * self.mass[self.e0] * self.mass[self.e1] / (
            self.mass[self.e0] + self.mass[self.e1])
        self.gamma = ZETA * np.sqrt(self.k * mbar)           # pairwise damping

        # ---- frozen SPD matrix A = M/dt + D + dt*Kh ---------------------
        rows, cols, vals = [], [], []
        dt = DT
        for sgn0 in (0, 1):
            for sgn1 in (0, 1):
                sgn = 1.0 if sgn0 == sgn1 else -1.0
                na = self.e0 if sgn0 == 0 else self.e1
                nb = self.e0 if sgn1 == 0 else self.e1
                blk = sgn * ((self.gamma + dt * self.k)[:, None, None]
                             * (self.u0[:, :, None] * self.u0[:, None, :])
                             + (dt * KH_ISO * self.k)[:, None, None]
                             * np.eye(3)[None])
                for a in range(3):
                    for b in range(3):
                        rows.append(3 * na + a)
                        cols.append(3 * nb + b)
                        vals.append(blk[:, a, b])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        ndof = 3 * self.n
        diag = np.array([self.mass[i] / dt + ALPHA_D * self.mass[i // 3]
                         for i in range(ndof)])
        dm = sp.coo_matrix((diag, (np.arange(ndof), np.arange(ndof))),
                           shape=(ndof, ndof))
        A = (dm + sp.coo_matrix((vals, (rows, cols)),
                                shape=(ndof, ndof))).tocsr()
        fi3 = np.concatenate([3 * self.fi + a for a in range(3)])
        di3 = np.concatenate([3 * self.di + a for a in range(3)])
        self.fi3 = fi3
        self.Afd = A[fi3][:, di3].tocsr()
        self.lu = splu(A[fi3][:, fi3].tocsc())

        # ---- contact bookkeeping ----------------------------------------
        st = np.arange(NROD * NRING)
        self.RING_NODES = ((st // NRING) * NPER
                           + (st % NRING) * NTH)[:, None] + np.arange(NTH)
        self.ring_idx = self.RING_NODES.ravel()              # 3430 nodes
        sa = np.arange(len(self.ring_idx))
        self.node_rod = sa // NRING
        self.node_ring = sa % NRING
        self.cand_n = np.zeros(0, dtype=np.int64)
        self.cand_s = np.zeros(0, dtype=np.int64)
        self.fc = np.zeros((self.n, 3))
        self.f_int = np.zeros((self.n, 3))

        # ---- budget domains: 7 rods + global ----------------------------
        self.G = NROD + 1
        self.KT = 2                                            # drive, contact
        w = np.zeros((self.G, self.n))
        for b in range(NROD):
            w[b, self.body_id == b] = 1.0
        w[-1] = 1.0
        self.domain_weights = w
        self.impulse = np.zeros((self.G, self.KT, 3))
        self.last_force = np.zeros((self.G, self.KT, 3))
        m_tot = float(self.mass.sum())
        self.p_ref = m_tot * abs(self.omega[1]) * self.ring_r

        # ---- deformation-gradient quadrature (fixed per segment) --------
        theta = 2.0 * np.pi * np.arange(NTH) / NTH
        h = self.length / NSEG
        ring = np.stack((np.zeros(NTH), self.radius * np.cos(theta),
                         self.radius * np.sin(theta)), axis=1)
        X = np.vstack((ring, ring + [h, 0.0, 0.0]))          # (20,3)
        Gm = X.T @ X + 1e-12 * np.eye(3)
        self.W = X @ np.linalg.inv(Gm)                       # (20,3)

        self._steps = 0
        self._search_contact(force=True)

    # ------------------------------------------------------------------ #
    def _kinematics(self, t):
        tc = np.clip(t, self.t_start, self.t_stop) - self.t_start
        th_l, th_r = self.omega[0] * tc, self.omega[1] * tc
        pos = self.x.copy()
        vel = self.v.copy()
        for mask, th, om in ((self.left, th_l, self.omega[0]),
                             (self.right, th_r, self.omega[1])):
            r0 = self.rest[mask]
            c, s = np.cos(th), np.sin(th)
            y = r0[:, 1] * c - r0[:, 2] * s
            z = r0[:, 1] * s + r0[:, 2] * c
            pos[mask, 0] = r0[:, 0]
            pos[mask, 1] = y
            pos[mask, 2] = z
            vel[mask, 0] = 0.0
            vel[mask, 1] = -om * z
            vel[mask, 2] = om * y
        return pos, vel

    def _stations(self, x):
        return x[self.RING_NODES].mean(1)

    def _search_contact(self, force=False):
        if not force and self._steps % SEARCH_EVERY != 0:
            return
        S = self._stations(self.x)
        tree = cKDTree(S)
        q = self.x[self.ring_idx]
        hits = tree.query_ball_point(q, SEARCH_R)
        ns, ss = [], []
        for a, hl in enumerate(hits):
            if not hl:
                continue
            st = np.asarray(hl, dtype=np.int64)
            b_n, i_n = self.node_rod[a], self.node_ring[a]
            b_s = st // NRING
            i_s = st % NRING
            keep = (b_s != b_n) | (np.abs(i_s - i_n) >= MIN_SELF_DI)
            if keep.any():
                ns.append(np.full(int(keep.sum()), self.ring_idx[a]))
                ss.append(st[keep])
        if ns:
            self.cand_n = np.concatenate(ns)
            self.cand_s = np.concatenate(ss)
        else:
            self.cand_n = np.zeros(0, dtype=np.int64)
            self.cand_s = np.zeros(0, dtype=np.int64)

    def _contact_forces(self):
        self.fc[:] = 0.0
        cn, cs = self.cand_n, self.cand_s
        if len(cn) == 0:
            return
        S = self._stations(self.x)
        d = self.x[cn] - S[cs]
        dist = np.linalg.norm(d, axis=1)
        act = dist < CONTACT_R
        act &= dist > 1e-9
        if not act.any():
            return
        cn_a, cs_a = cn[act], cs[act]
        d_a, dist_a = d[act], dist[act]
        fmag = CONTACT_K * (CONTACT_R - dist_a)
        fvec = fmag[:, None] * (d_a / dist_a[:, None])
        np.add.at(self.fc, cn_a, fvec)
        ring = self.RING_NODES[cs_a].ravel()
        np.add.at(self.fc, ring, np.repeat(-0.1 * fvec, NTH, axis=0))

    def _spring_forces(self):
        self.f_int[:] = 0.0
        d = self.x[self.e0] - self.x[self.e1]
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-9)
        fmag = self.k * (dist - self.l0)
        fvec = (fmag / dist)[:, None] * d
        np.add.at(self.f_int, self.e0, fvec)
        np.add.at(self.f_int, self.e1, -fvec)

    def _step(self, dt):
        self._search_contact()
        self._spring_forces()
        self._contact_forces()
        v_old = self.v.copy()
        rhs_full = (self.mass[:, None] * v_old / dt) + self.f_int + self.fc
        pos_k, vel_k = self._kinematics(self.t + dt)
        vd3 = np.concatenate([3 * self.di + a for a in range(3)], axis=0)
        vdof = np.empty(3 * self.n)
        vdof[0::3], vdof[1::3], vdof[2::3] = vel_k[:, 0], vel_k[:, 1], vel_k[:, 2]
        rhs = rhs_full[self.fi] - (self.Afd @ vdof[vd3]).reshape(-1, 3)
        dv = self.lu.solve(rhs.ravel()).reshape(-1, 3)
        self.v = vel_k.copy()
        self.v[self.fi] = dv
        self.x = pos_k.copy()
        self.x[self.fi] += dt * dv
        # drive reactions (term 0) and net contact per rod (term 1)
        react = np.zeros((self.n, 3))
        mdt = (self.mass[self.di] / dt)[:, None]
        react[self.di] = (mdt * (self.v[self.di] - v_old[self.di])
                          - self.f_int[self.di] - self.fc[self.di])
        F = np.zeros((self.G, self.KT, 3))
        for b in range(NROD):
            sl = slice(b * NPER, (b + 1) * NPER)
            F[b, 0] = react[sl][self.left[sl] | self.right[sl]].sum(0)
            F[b, 1] = self.fc[sl].sum(0)
        F[-1, 0] = react.sum(0)
        F[-1, 1] = 0.0
        self.impulse += F * dt
        self.last_force = F
        self.t += dt
        self._steps += 1

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame
        target = float(time)
        while self.t < target - 1e-12:
            self._step(min(DT, target - self.t))
        pos_k, vel_k = self._kinematics(self.t)
        # re-assert exact kinematics (already applied in _step)
        self.x[self.di] = pos_k[self.di]
        self.v[self.di] = vel_k[self.di]

        # element deformation gradients (336 segments, fixed order)
        P = self.x[:NROD * NSEG * NTH].reshape(NROD, NRING, NTH, 3)
        Q = np.concatenate((P[:, :-1], P[:, 1:]), axis=2)       # (7,48,20,3)
        Q = Q - Q[:, :, :NTH].mean(2, keepdims=True)
        F = np.einsum('bpnc,nd->bpcd', Q, self.W)               # (7,48,3,3)
        F = F.reshape(NROD * NSEG, 3, 3)

        mass_q = self.mass[:, None].copy()
        frame = StateFrame(
            self.t, self.x.copy(), self.v.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"rest_position": self.rest.copy(),
                        "driven_left": self.left.astype(float),
                        "driven_right": self.right.astype(float),
                        "mass_quantity": mass_q},
            fields={
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.last_force.copy(),
                "budget_cumulative_external_impulse":
                    self.impulse.sum(1).copy(),
                "budget_cumulative_inflow_momentum":
                    np.zeros((self.G, 3)),
                "budget_cumulative_outflow_momentum":
                    np.zeros((self.G, 3)),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F,
            },
            scalars={"reference_momentum": float(self.p_ref)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        ke = 0.5 * float((self.mass[:, None] * self.v ** 2).sum())
        return {"time": self.t, "steps": self._steps,
                "kinetic_energy": ke,
                "contact_pairs": int(len(self.cand_n)),
                "reference_momentum": float(self.p_ref)}
