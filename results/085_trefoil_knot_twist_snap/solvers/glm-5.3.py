"""Twist-induced snap of an open trefoil knot: discrete Cosserat rod solver.

Physical model (implemented from scratch with NumPy only)
---------------------------------------------------------
- Naturally straight, isotropic, nearly inextensible elastic rod discretised
  as a vertex chain (181 vertices) with one material frame per edge.
- Bending energy (rotation-invariant discrete elastica, small turning angle
  per edge):  E_b = sum_j (B/2) |x_j - 2 x_{j+1} + x_{j+2}|^2 / h_j^3.
- Twist energy: E_t = sum_j (beta/2) phi_j^2 / l_j with phi_j the signed
  angle between the material frame of edge j parallel-transported onto edge
  j+1 and the material frame of edge j+1.
- Stretch: stiff linear springs with tangential damping (inextensible limit).
- Self-contact: finite-thickness (2 * rod radius) penalty + damping between
  non-adjacent vertices, with periodic candidate-pair refresh.
- Material frames are Lagrangian: each substep they follow the minimal
  rotation that carries the old edge tangent to the new one, plus a dynamic
  twist rate Omega about the tangent.  Twist and writhe are therefore
  exchanged geometrically, which is the mechanism of the knot snap.
- Clamps: vertices 0 and 180 are fixed at the clamp positions; the material
  frame at the left end is pinned to the (fixed) left clamp grip axes, the
  right end frame follows the driven clamp rotation about the outward tail
  tangent (smoothstep ramp to +3 turns).  Initial frames are pure parallel
  transport, so the initial material twist is exactly zero.
- Time integration: symplectic (semi-implicit) Euler with a fixed substep,
  fully deterministic.

Budget bookkeeping: one domain (the whole rod), three external force terms
(gravity at the centre of mass, left clamp reaction, right clamp reaction)
and two free couples (the clamp twist torques at the ends).  Contact forces
are internal to the single domain and cancel.  No mass crosses the domain
boundary, so inflow/outflow terms are zero.
"""
from __future__ import annotations

import numpy as np


def _rotate_between(ta: np.ndarray, tb: np.ndarray, u: np.ndarray):
    """Rotate vectors ``u`` by the minimal rotation taking ``ta`` to ``tb``.

    ``ta``/``tb``/``u`` are (k,3) arrays.  Returns (u_rotated, qw, qv) with
    qw/qv the scalar/vector parts of the rotation quaternions (xyzw order
    uses only qv here).
    """
    c = np.einsum('ij,ij->i', ta, tb)
    v = np.cross(ta, tb)
    w = np.sqrt(np.maximum(1.0 + c, 0.0) * 0.5)
    denom = 2.0 * np.maximum(w, 1e-9)
    qv = v / denom[:, None]
    cr = np.cross(qv, u)
    u_rot = u + 2.0 * (w[:, None] * cr + np.cross(qv, cr))
    return u_rot, w, qv


def _mats_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (n,3,3) -> unit quaternions (n,4) in xyzw order."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    s = 2.0 * np.sqrt(np.maximum(1.0 + tr, 0.0))
    s = np.maximum(s, 1e-12)
    qa = np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s,
                   0.25 * s], axis=1)

    s = 2.0 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0))
    s = np.maximum(s, 1e-12)
    qb = np.stack([0.25 * s, (m01 + m10) / s, (m02 + m20) / s,
                   (m21 - m12) / s], axis=1)

    s = 2.0 * np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0))
    s = np.maximum(s, 1e-12)
    qc = np.stack([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s,
                   (m02 - m20) / s], axis=1)

    s = 2.0 * np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0))
    s = np.maximum(s, 1e-12)
    qd = np.stack([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s,
                   (m10 - m01) / s], axis=1)

    q = np.where((tr > 0.0)[:, None], qa,
                 np.where(((m00 > m11) & (m00 > m22))[:, None], qb,
                          np.where((m11 > m22)[:, None], qc, qd)))
    q /= np.maximum(np.linalg.norm(q, axis=1), 1e-12)[:, None]
    return q


class BenchmarkSolver:
    # --- numerical / material parameters -----------------------------------
    DT = 2.0e-5            # fixed substep (sample dt 0.02 -> 1000 substeps)
    B = 0.05               # bending stiffness  [N m^2]
    BETA = 0.038           # twist stiffness    [N m^2]  (~0.76 B, isotropic)
    K_STRETCH = 1000.0     # edge spring stiffness [N/m]
    C_STRETCH = 0.3        # tangential stretch damping [N s/m]
    K_CONTACT = 200.0      # self-contact penalty stiffness [N/m]
    C_CONTACT = 0.6        # self-contact damping [N s/m]
    GAMMA = 1.2            # global velocity damping [1/s]
    GAMMA_TW = 10.0        # twist-rate damping [1/s]
    CONTACT_MARGIN = 0.015  # candidate refresh margin [m]
    CONTACT_REFRESH = 16   # refresh candidate pairs every N substeps
    V_MAX = 50.0           # velocity guard [m/s]
    OM_MAX = 500.0         # twist-rate guard [rad/s]

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        f0 = scene.initial_state()

        self.p = np.array(f0.position, dtype=float)
        self.v = np.zeros_like(self.p)
        self.n = len(self.p)
        self.ne = self.n - 1
        self.mv = np.array(f0.mass, dtype=float)

        self.g = np.array(self.config['gravity'], dtype=float)
        self.Fg_total = self.mv.sum() * self.g

        e = self.p[1:] - self.p[:-1]
        self.h = np.sqrt(np.maximum(np.einsum('ij,ij->i', e, e), 1e-24))
        self.t = e / self.h[:, None]           # current edge tangents
        self.rho = self.mv.sum() / self.h.sum()

        r = float(self.config['rod']['radius'])
        self.radius = r
        self.rc = 2.0 * r                       # contact diameter
        self.Ip = np.pi * r ** 4 / 2.0          # polar second moment
        self.Ib = np.pi * r ** 4 / 4.0          # bending second moment
        self.Je = self.rho * self.Ip * self.h   # per-edge polar inertia
        self.lbar = 0.5 * (self.h[:-1] + self.h[1:])

        cl = self.config['clamps']
        self.p_left = np.array(cl['left']['position'], dtype=float)
        self.p_right = np.array(cl['right']['position'], dtype=float)
        self.ramp = cl['twist_ramp']
        self.axis = np.array(cl['right']['rotation_axis'], dtype=float)
        self.axis = self.axis / max(np.linalg.norm(self.axis), 1e-12)

        # Lagrangian material frames: pure parallel transport (zero twist).
        self.mf = self._init_frames()
        self.a_left = self.mf[0].copy()         # left clamp grip axis
        self.a_right0 = self.mf[-1].copy()      # right clamp grip axis at t=0

        self.t_old = self.t.copy()
        self.Om = np.zeros(self.ne)
        self.omega_e = np.zeros((self.ne, 3))
        self.phi = np.zeros(max(self.ne - 1, 0))

        # budget accumulators
        self.imp_F = np.zeros(3)
        self.imp_L = np.zeros(3)
        self._reset_accumulators()

        self.t = 0.0
        self.step = 0
        self.ci = np.array([], dtype=int)
        self.cj = np.array([], dtype=int)

    # ------------------------------------------------------------- helpers
    def _init_frames(self) -> np.ndarray:
        t = self.t
        m = np.empty((self.ne, 3))
        # deterministic seed normal: least-aligned world axis with t[0]
        seed = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(seed, t[0])) > 0.9:
            seed = np.array([0.0, 0.0, 1.0])
        m[0] = seed - t[0] * np.dot(seed, t[0])
        m[0] /= max(np.linalg.norm(m[0]), 1e-12)
        for i in range(1, self.ne):
            u = m[i - 1:i]
            ur, _, _ = _rotate_between(t[i - 1:i], t[i:i + 1], u)
            m[i] = ur[0]
            m[i] -= t[i] * np.dot(m[i], t[i])
            m[i] /= max(np.linalg.norm(m[i]), 1e-12)
        return m

    def _clamp_angle(self, time: float):
        r = self.ramp
        t0, t1 = float(r['ramp_start']), float(r['ramp_end'])
        turns = float(r['final_relative_turns'])
        span = t1 - t0
        u = np.clip((time - t0) / span, 0.0, 1.0)
        th = 2.0 * np.pi * turns * u * u * (3.0 - 2.0 * u)
        thd = 2.0 * np.pi * turns * 6.0 * u * (1.0 - u) / span
        return float(th), float(thd)

    def _twist_angles(self, t: np.ndarray, m: np.ndarray) -> np.ndarray:
        if self.ne < 2:
            return np.zeros(0)
        a, _, _ = _rotate_between(t[:-1], t[1:], m[:-1])
        b = m[1:]
        sin = np.einsum('ij,ij->i', t[1:], np.cross(a, b))
        cos = np.einsum('ij,ij->i', a, b)
        return np.arctan2(sin, cos)

    def _refresh_contact(self) -> None:
        p = self.p
        d2 = np.einsum('ij,ij->i', p, p)
        D2 = d2[:, None] + d2[None, :] - 2.0 * (p @ p.T)
        iu, ju = np.triu_indices(self.n, k=3)
        lim = (self.rc + self.CONTACT_MARGIN) ** 2
        sel = D2[iu, ju] < lim
        self.ci, self.cj = iu[sel], ju[sel]

    def _contact_force(self) -> np.ndarray:
        F = np.zeros_like(self.p)
        i, j = self.ci, self.cj
        if i.size == 0:
            return F
        dx = self.p[i] - self.p[j]
        dist = np.sqrt(np.maximum(np.einsum('ij,ij->i', dx, dx), 1e-24))
        nv = dx / dist[:, None]
        overlap = self.rc - dist
        vn = np.einsum('ij,ij->i', self.v[i] - self.v[j], nv)
        fmag = np.where(overlap > 0.0,
                        self.K_CONTACT * overlap - self.C_CONTACT * vn, 0.0)
        fmag = np.maximum(fmag, 0.0)
        f = fmag[:, None] * nv
        np.add.at(F, i, f)
        np.add.at(F, j, -f)
        return F

    def _reset_accumulators(self) -> None:
        self.acc_F = np.zeros((3, 3))   # gravity, left clamp, right clamp
        self.acc_C = np.zeros((3, 3))   # zero, left couple, right couple
        self.acc_dt = 0.0

    # -------------------------------------------------------------- substep
    def _substep(self) -> None:
        p, v, mv = self.p, self.v, self.mv
        dt = self.DT

        # ---- forces on vertices -------------------------------------------
        e = p[1:] - p[:-1]
        le = np.sqrt(np.maximum(np.einsum('ij,ij->i', e, e), 1e-24))
        t = e / le[:, None]

        F = np.zeros_like(p)
        # stretch (inextensible limit) + tangential damping
        fs = self.K_STRETCH * (le - self.h)
        dv = v[1:] - v[:-1]
        vt = np.einsum('ij,ij->i', dv, t)
        ft = fs + self.C_STRETCH * vt
        F[:-1] -= ft[:, None] * t
        F[1:] += ft[:, None] * t
        # bending (naturally straight)
        d = p[:-2] - 2.0 * p[1:-1] + p[2:]
        wj = self.B / self.lbar ** 3
        F[:-2] -= wj[:, None] * d
        F[1:-1] += 2.0 * wj[:, None] * d
        F[2:] -= wj[:, None] * d
        # gravity and global damping
        F += mv[:, None] * self.g[None, :]
        F -= self.GAMMA * mv[:, None] * v
        # finite-thickness self-contact
        if self.step % self.CONTACT_REFRESH == 0:
            self._refresh_contact()
        F += self._contact_force()

        # clamp reactions (vertices are held kinematically fixed)
        Rl = -F[0].copy()
        Rr = -F[-1].copy()

        # ---- integrate vertices -------------------------------------------
        v += dt * F / mv[:, None]
        v[0] = 0.0
        v[-1] = 0.0
        np.clip(v, -self.V_MAX, self.V_MAX, out=v)
        p += dt * v
        p[0] = self.p_left
        p[-1] = self.p_right

        # ---- material frames ----------------------------------------------
        m = self.mf
        m, qw, qv = _rotate_between(self.t_old, t, m)   # follow edge rotation
        m -= np.einsum('ij,ij->i', m, t)[:, None] * t   # re-orthogonalise
        m /= np.maximum(np.sqrt(np.einsum('ij,ij->i', m, m)),
                        1e-12)[:, None]

        # twist-rate update from the previous substep's twist angles
        c = self.BETA * self.phi / self.lbar
        Q = np.zeros(self.ne)
        if self.ne > 1:
            Q[1:-1] = c[1:] - c[:-1]
            Q[0] = c[0]
            Q[-1] = -c[-1]
        Om = self.Om + dt * (Q / self.Je - self.GAMMA_TW * self.Om)
        np.clip(Om, -self.OM_MAX, self.OM_MAX, out=Om)

        th, thd = self._clamp_angle(self.t)
        Om[0] = 0.0
        Om[-1] = thd * t[-1, 0]

        # dynamic twist about the tangent (Rodrigues; m stays ⊥ t)
        ang = Om * dt
        m = m * np.cos(ang)[:, None] + np.cross(t, m) * np.sin(ang)[:, None]

        # clamped end frames: left grip fixed, right grip rotated by th
        a0 = self.a_left
        m[0] = a0 - t[0] * np.dot(a0, t[0])
        m[0] /= max(np.linalg.norm(m[0]), 1e-12)
        ct, st = np.cos(th), np.sin(th)
        ax, ay, az = self.axis
        ar = np.array([ax, ay * ct - az * st, ay * st + az * ct])
        m[-1] = ar - t[-1] * np.dot(ar, t[-1])
        m[-1] /= max(np.linalg.norm(m[-1]), 1e-12)

        # store state
        self.mf = m
        self.t_old = t
        self.Om = Om
        self.omega_e = 2.0 * qv / dt + Om[:, None] * t
        self.phi = self._twist_angles(t, m)

        # ---- budget bookkeeping -------------------------------------------
        cl = -Q[0] * t[0]
        cr = -Q[-1] * t[-1]
        com = (mv[:, None] * p).sum(axis=0) / mv.sum()
        self.acc_dt += dt
        self.acc_F[0] += self.Fg_total * dt
        self.acc_F[1] += Rl * dt
        self.acc_F[2] += Rr * dt
        self.acc_C[1] += cl * dt
        self.acc_C[2] += cr * dt
        self.imp_F += (self.Fg_total + Rl + Rr) * dt
        self.imp_L += (np.cross(com, self.Fg_total)
                       + np.cross(self.p_left, Rl)
                       + np.cross(self.p_right, Rr)
                       + cl + cr) * dt
        self.step += 1

    # ---------------------------------------------------------- frame output
    def _build_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        p, v, mv = self.p, self.v, self.mv
        n, ne = self.n, self.ne
        t, m = self.t_old, self.mf

        # per-vertex averaged frame
        tv = np.zeros((n, 3))
        tv[:-1] += 0.5 * t
        tv[1:] += 0.5 * t
        tv /= np.maximum(np.linalg.norm(tv, axis=1), 1e-12)[:, None]
        mvf = np.zeros((n, 3))
        mvf[:-1] += 0.5 * m
        mvf[1:] += 0.5 * m
        mvf -= np.einsum('ij,ij->i', mvf, tv)[:, None] * tv
        mvf /= np.maximum(np.linalg.norm(mvf, axis=1), 1e-12)[:, None]
        bv = np.cross(tv, mvf)

        av = np.zeros((n, 3))
        if ne > 0:
            av[:-1] += 0.5 * self.omega_e
            av[1:] += 0.5 * self.omega_e
            av[0] = self.omega_e[0]
            av[-1] = self.omega_e[-1]

        R = np.stack([mvf, bv, tv], axis=2)      # columns: m, b, t
        quat = _mats_to_quat(R)

        # per-vertex inertia of the lumped cross-section slices (world frame)
        Ib, Ip = self.Ib, self.Ip
        inertia = (Ib * np.eye(3)[None, :, :]
                   + (Ip - Ib) * tv[:, :, None] * tv[:, None, :])
        inertia *= mv[:, None, None]

        phi_cum = np.concatenate(([0.0], np.cumsum(self.phi))) / (2.0 * np.pi)

        th, _ = self._clamp_angle(time)

        # external force terms averaged over the last sample interval
        if self.acc_dt > 0.0:
            avgF = self.acc_F / self.acc_dt
            avgC = self.acc_C / self.acc_dt
        else:
            avgF = np.stack([self.Fg_total,
                             np.zeros(3), np.zeros(3)])
            avgC = np.zeros((3, 3))
        com = (mv[:, None] * p).sum(axis=0) / mv.sum()
        fpos = np.stack([com, self.p_left, self.p_right])

        arc = np.linspace(0.0, 1.0, n)
        endpoint = np.zeros(n)
        endpoint[0] = 1.0
        endpoint[-1] = 1.0

        frame = StateFrame(
            float(time),
            p.copy(), v.copy(),
            orientation=quat,
            angular_velocity=av,
            mass=mv.copy(),
            attributes={
                'mass_quantity': mv.copy(),
                'arc_fraction': arc,
                'rod_radius': np.full(n, self.radius),
                'contact_diameter': np.full(n, self.rc),
                'endpoint_controlled': endpoint,
                'material_twist_turns': phi_cum,
                'clamp_twist_angle': np.full(n, th),
            },
            fields={
                'budget_domain_weights': np.ones((1, n)),
                'budget_external_force': avgF[None, :, :],
                'budget_force_position': fpos[None, :, :],
                'budget_external_couple': avgC[None, :, :],
                'budget_cumulative_external_impulse': self.imp_F[None, :],
                'budget_cumulative_external_angular_impulse':
                    self.imp_L[None, :],
                'budget_cumulative_inflow_momentum': np.zeros((1, 3)),
                'budget_cumulative_outflow_momentum': np.zeros((1, 3)),
                'budget_cumulative_inflow_angular_momentum':
                    np.zeros((1, 3)),
                'budget_cumulative_outflow_angular_momentum':
                    np.zeros((1, 3)),
                'cumulative_mass_source': np.zeros((1,)),
                'constraint_value': np.array([th]),
                'constraint_target': np.array([th]),
            },
            scalars={
                'constraint_scale':
                    2.0 * np.pi * float(self.ramp['final_relative_turns']),
                'reference_momentum':
                    float(self.mv.sum() * np.sqrt(9.81 * self.h.sum())),
                'reference_angular_momentum':
                    float(self.mv.sum() * np.sqrt(9.81 * self.h.sum()) * 1.0),
            },
        )
        self._reset_accumulators()
        frame.validate()
        return frame

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float):
        time = float(time)
        if time > self.t + 1e-12:
            nsub = int(round((time - self.t) / self.DT))
            for _ in range(nsub):
                self.t += self.DT
                self._substep()
            self.t = time
        out_time = time if time >= self.t - 1e-9 else self.t
        return self._build_frame(out_time)

    def diagnostics(self) -> dict[str, object]:
        phi_cum = (np.sum(self.phi) / (2.0 * np.pi)
                   if self.phi.size else 0.0)
        return {
            'substeps': int(self.step),
            'dt': self.DT,
            'bending_stiffness': self.B,
            'twist_stiffness': self.BETA,
            'rod_length': float(self.h.sum()),
            'linear_density': float(self.rho),
            'total_twist_turns': float(phi_cum),
            'max_vertex_speed': float(np.max(np.linalg.norm(self.v, axis=1)))
            if self.n else 0.0,
            'active_contact_pairs': int(self.ci.size),
        }
