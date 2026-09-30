"""Physics solver: plectoneme formation in a hanging elastic rod.

Discrete Cosserat-rod model (own implementation, no external physics engine):

- centerline nodes (positions/velocities) coupled to per-segment material
  frames (unit quaternions + angular velocities);
- elastic energies: edge stretching (inextensibility), frame/tangent shear
  coupling, discrete bending + twisting between adjacent segment frames,
  finite-thickness vertex/edge self-contact with normal damping and viscous
  friction;
- loads: gravity, ambient drag, position-controlled endpoints and
  servo-driven end twist about the end tangents;
- integration: semi-implicit (symplectic) Euler on nodes, over-damped
  implicit-damping update on segment rotations, fixed substep.

All trajectory/budget fields are accumulated from the actual discrete
impulses used by the integrator, so the momentum/angular-momentum budgets
are consistent with the reported state by construction.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# ----------------------------------------------------------------------
# quaternion helpers (w, x, y, z), Hamilton product, body->world columns
# ----------------------------------------------------------------------
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(a)
    out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def qconj(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[:, 1:] *= -1.0
    return out


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation-matrix stack (n,3,3) -> unit quaternions (n,4), w first."""
    n = len(R)
    q = np.empty((n, 4))
    m00 = R[:, 0, 0]; m01 = R[:, 0, 1]; m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]; m11 = R[:, 1, 1]; m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]; m21 = R[:, 2, 1]; m22 = R[:, 2, 2]
    tr = m00 + m11 + m22
    for i in range(n):
        if tr[i] > 0.0:
            s = np.sqrt(tr[i] + 1.0) * 2.0
            w = 0.25 * s
            x = (m21[i] - m12[i]) / s
            y = (m02[i] - m20[i]) / s
            z = (m10[i] - m01[i]) / s
        elif m00[i] > m11[i] and m00[i] > m22[i]:
            s = np.sqrt(1.0 + m00[i] - m11[i] - m22[i]) * 2.0
            w = (m21[i] - m12[i]) / s
            x = 0.25 * s
            y = (m01[i] + m10[i]) / s
            z = (m02[i] + m20[i]) / s
        elif m11[i] > m22[i]:
            s = np.sqrt(1.0 + m11[i] - m00[i] - m22[i]) * 2.0
            w = (m02[i] - m20[i]) / s
            x = (m01[i] + m10[i]) / s
            y = 0.25 * s
            z = (m12[i] + m21[i]) / s
        else:
            s = np.sqrt(1.0 + m22[i] - m00[i] - m11[i]) * 2.0
            w = (m10[i] - m01[i]) / s
            x = (m02[i] + m20[i]) / s
            y = (m12[i] + m21[i]) / s
            z = 0.25 * s
        q[i] = (w, x, y, z)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def quat_log_rotvec(q: np.ndarray) -> np.ndarray:
    """Unit quaternions -> rotation vectors (angle * axis)."""
    w = np.clip(q[:, 0], -1.0, 1.0)
    v = q[:, 1:]
    s = np.linalg.norm(v, axis=1)
    factor = np.where(s > 1e-12, 2.0 * np.arctan2(s, w) / np.maximum(s, 1e-12), 2.0)
    return v * factor[:, None]


class BenchmarkSolver:
    """Discrete Cosserat-rod solver for the hanging-rod plectoneme scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        rod = cfg["rod"]
        self.N = int(rod["vertex_count"])
        self.S = self.N - 1
        self.L = float(rod["total_length"])
        self.h = self.L / self.S
        self.radius = float(rod["radius"])
        self.M = float(rod["mass"])

        # --- initial state from the frozen scene -------------------------
        frame0 = scene.initial_state()
        x = np.array(frame0.position, dtype=float)
        m_node = np.array(frame0.mass, dtype=float)
        self.m_node = m_node
        self.mu = self.M / self.L

        # deterministic, tiny out-of-plane imperfection (buckling seed);
        # endpoints are kept exactly at their controlled positions.
        rng = np.random.default_rng(self.seed)
        noise = 2.0e-4 * rng.standard_normal(self.N)
        noise[0] = noise[-1] = 0.0
        x[:, 1] += noise
        x[0] = np.asarray(cfg["end_controls"]["left"]["position"], float)
        x[-1] = np.asarray(cfg["end_controls"]["right"]["position"], float)
        self.x = x
        self.v = np.zeros_like(x)

        # --- material parameters -----------------------------------------
        self.g = np.asarray(cfg["gravity"], dtype=float)
        # bending / twisting moduli of the naturally straight isotropic rod
        self.B = 0.03          # N*m^2 bending
        self.C = 0.03          # N*m^2 twisting (isropic rod: C = B)
        self.ks = 1800.0       # N/m edge stretch stiffness (inextensible)
        self.kc = 8.0          # N/m frame/tangent (shear) coupling stiffness
        self.k_tw = 6.0        # N*m/rad end-twist servo stiffness
        self.k_contact = 400.0  # N/m self-contact penalty
        self.c_n = 0.3         # N*s/m contact normal damping
        self.mu_f = 0.05       # N*s/m contact viscous friction
        drag_cfg = cfg.get("ambient_drag", {})
        drag_coeff = float(drag_cfg.get("coefficient", 0.035))
        self.c_lin = drag_coeff * self.h      # per-node ambient drag (per length)
        self.c_ang = 8.0e-3                   # per-segment angular drag (over-damped frames)

        # numerically lumped segment inertias (rotational sub-dynamics is
        # strongly damped; statics/energy unaffected by this lumping)
        self.I_t = 4.0e-6      # kg*m^2 transverse (per segment)
        self.I_a = 2.0e-6      # kg*m^2 axial (per segment)
        self.Ib = np.tile(np.array([self.I_t, self.I_t, self.I_a]), (self.S, 1))

        self.dt = 2.5e-4
        self.cd = 2.0 * self.radius
        self.cd2 = self.cd * self.cd

        # --- initial material frames: twist-free planar ------------------
        # d3 = edge tangent, d1 = global y (rod lies in the x-z plane).
        e = x[1:] - x[:-1]
        el = np.linalg.norm(e, axis=1)
        t0 = e / el[:, None]
        d1 = np.zeros_like(t0)
        d1[:, 1] = 1.0
        d3 = t0
        d2 = np.cross(d3, d1)
        R0 = np.stack((d1, d2, d3), axis=2)   # columns are the directors
        self.q = R_to_quat(R0)
        self.W = np.zeros((self.S, 3))
        self.phi = np.zeros(self.S)           # accumulated material spin (rad)

        # --- self-contact candidate pairs (vertex i vs edge j, non-adjacent)
        ii, jj = np.meshgrid(np.arange(self.N), np.arange(self.S), indexing="ij")
        mask = (jj <= ii - 2) | (jj >= ii + 1)
        self.IA = ii[mask].astype(np.int64)
        self.JE = jj[mask].astype(np.int64)
        self.contact_F = np.zeros_like(self.x)

        # --- twist ramp ---------------------------------------------------
        ramp = cfg["end_controls"]["twist_ramp"]
        self.ramp_start = float(ramp["ramp_start"])
        self.ramp_end = float(ramp["ramp_end"])
        self.final_turns = float(ramp["final_relative_turns"])
        self.left_pos = np.asarray(cfg["end_controls"]["left"]["position"], float)
        self.right_pos = np.asarray(cfg["end_controls"]["right"]["position"], float)

        # --- budget accumulators ------------------------------------------
        self.P_ext = np.zeros(3)   # cumulative external impulse
        self.A_ext = np.zeros(3)   # cumulative external angular impulse
        self.t = 0.0
        self.step_count = 0
        self.diverged = False
        self.min_contact_dist = float("inf")

        com = (m_node[:, None] * x).sum(0) / m_node.sum()
        self.inst = {
            "f_grav": (m_node[:, None] * self.g).sum(0),
            "f_drag": np.zeros(3),
            "f_con": np.zeros(3),
            "c_tw": np.zeros(3),
            "c_ad": np.zeros(3),
            "p_grav": com.copy(),
            "p_drag": com.copy(),
            "p_con": 0.5 * (self.left_pos + self.right_pos),
        }

        # scalars frozen for scoring
        gnorm = float(np.linalg.norm(self.g))
        self.scal_ref_mom = self.M * np.sqrt(max(gnorm, 1e-9) * self.L)
        self.scal_ref_ang = 0.5 * self.M * max(gnorm, 1e-9) * self.L
        self.constraint_scale = 2.0 * np.pi

        # backup for divergence safety
        self._save_backup()

    # ------------------------------------------------------------------
    def _save_backup(self) -> None:
        self.bk = (self.x.copy(), self.v.copy(), self.q.copy(), self.W.copy(),
                   self.phi.copy(), self.P_ext.copy(), self.A_ext.copy(), self.t)

    def _restore_backup(self) -> None:
        (self.x, self.v, self.q, self.W, self.phi,
         self.P_ext, self.A_ext, self.t) = (
            self.bk[0].copy(), self.bk[1].copy(), self.bk[2].copy(),
            self.bk[3].copy(), self.bk[4].copy(), self.bk[5].copy(),
            self.bk[6].copy(), self.bk[7])

    def _twist_cmd(self, time: float) -> tuple[float, float]:
        """Commanded end twist angles (left, right) about the end tangents."""
        span = max(self.ramp_end - self.ramp_start, 1e-9)
        u = np.clip((time - self.ramp_start) / span, 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        rel = 2.0 * np.pi * self.final_turns * u
        return -0.5 * rel, 0.5 * rel

    # ------------------------------------------------------------------
    def _contact_forces(self) -> None:
        """Finite-thickness self-contact: vertex vs edge penalty + friction."""
        x, v = self.x, self.v
        F = np.zeros_like(x)
        pi = x[self.IA]
        pj = x[self.JE]
        pk = x[self.JE + 1]
        e = pk - pj
        el2 = np.einsum("ij,ij->i", e, e)
        w = np.einsum("ij,ij->i", pi - pj, e) / el2
        np.clip(w, 0.0, 1.0, out=w)
        c = pj + w[:, None] * e
        d = pi - c
        d2 = np.einsum("ij,ij->i", d, d)
        act = d2 < self.cd2
        if np.any(act):
            ia = self.IA[act]
            je = self.JE[act]
            wj = w[act]
            dd = d[act]
            dist = np.sqrt(d2[act])
            self.min_contact_dist = min(self.min_contact_dist, float(dist.min()))
            dist = np.maximum(dist, 1e-9)
            n = dd / dist[:, None]
            pen = self.cd - dist
            vi = v[ia]
            vj = v[je]
            vk = v[je + 1]
            vrel = vi - ((1.0 - wj)[:, None] * vj + wj[:, None] * vk)
            vn = np.einsum("ij,ij->i", vrel, n)
            vt = vrel - vn[:, None] * n
            Fi = (self.k_contact * pen - self.c_n * vn)[:, None] * n - self.mu_f * vt
            np.add.at(F, ia, Fi)
            np.add.at(F, je, -(1.0 - wj)[:, None] * Fi)
            np.add.at(F, je + 1, -wj[:, None] * Fi)
        self.contact_F = F

    # ------------------------------------------------------------------
    def _substep(self) -> None:
        dt = self.dt
        self._save_backup()
        x, v, q, W = self.x, self.v, self.q, self.W
        N, S = self.N, self.S
        m_node = self.m_node

        # ---- kinematics ---------------------------------------------------
        e = x[1:] - x[:-1]
        el = np.sqrt(np.einsum("ij,ij->i", e, e))
        el = np.maximum(el, 1e-9)
        t = e / el[:, None]
        R = quat_to_R(q)
        d3 = R[:, :, 2]

        # ---- external impulses (evaluated at the pre-update configuration)
        Jg = m_node[:, None] * self.g[None, :] * dt
        Jd = -self.c_lin * v * dt
        # end-twist servo couples about the end tangents
        phiL, phiR = self._twist_cmd(self.t)
        cL = self.k_tw * (phiL - self.phi[0]) * d3[0]
        cR = self.k_tw * (phiR - self.phi[S - 1]) * d3[S - 1]

        # ---- node forces ---------------------------------------------------
        F = np.zeros_like(x)
        # stretching (inextensibility)
        fs = self.ks * (el - self.h)
        F[1:] += fs[:, None] * t
        F[:-1] -= fs[:, None] * t
        # frame/tangent coupling (transmits bending & twist to the centerline)
        dc = d3 - t
        dcn = dc - np.einsum("ij,ij->i", dc, t)[:, None] * t
        fc = (self.kc / el)[:, None] * dcn
        F[1:] += fc
        F[:-1] -= fc
        # gravity and ambient drag
        F += m_node[:, None] * self.g[None, :]
        F -= self.c_lin * v
        # self-contact (refreshed every second substep)
        self.step_count += 1
        if self.step_count % 2 == 1:
            self._contact_forces()
        F += self.contact_F

        # ---- segment torques ------------------------------------------------
        tau = self.kc * np.cross(d3, t)
        # discrete bending/twist from relative rotation of adjacent frames
        qr = qmul(qconj(q[:-1]), q[1:])
        psi = quat_log_rotvec(qr)                     # in frame of segment j-1
        Kpsi = np.stack((self.B * psi[:, 0], self.B * psi[:, 1],
                         self.C * psi[:, 2]), axis=1)
        A = np.einsum("nij,nj->ni", R[:-1], Kpsi) / self.h
        tau[1:] -= A
        tau[:-1] += A
        tau[0] += cL
        tau[S - 1] += cR

        # ---- integrate nodes (semi-implicit Euler) ---------------------------
        v = v + dt * F / m_node[:, None]
        # position-controlled endpoints: zero out their free velocity
        vfree = v[[0, N - 1]].copy()
        v[0] = 0.0
        v[N - 1] = 0.0
        Jc0 = -m_node[0] * vfree[0]
        JcN = -m_node[N - 1] * vfree[1]
        x = x + dt * v

        # ---- integrate segment rotations (over-damped update) ----------------
        tb = np.einsum("nji,nj->ni", R, tau)          # body-frame torques
        Wb = np.einsum("nji,nj->ni", R, W)            # body-frame ang. velocity
        Wb_new = (Wb + dt * tb / self.Ib) / (1.0 + dt * self.c_ang / self.Ib)
        cdrag_body = -dt * self.c_ang * Wb_new        # drag couple impulse (body)
        W_new = np.einsum("nij,nj->ni", R, Wb_new)
        # quaternion update: rotate by dt * W_new in world coordinates
        half = 0.5 * dt * W_new
        a = np.linalg.norm(half, axis=1)
        s = np.where(a > 1e-12, np.sin(np.maximum(a, 1e-12)) / np.maximum(a, 1e-12), 1.0)
        qs = np.empty_like(q)
        qs[:, 0] = np.cos(a)
        qs[:, 1:] = s[:, None] * half
        q = qmul(qs, q)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        # accumulated material spin about the (pre-update) body axis
        phi = self.phi + dt * np.einsum("ij,ij->i", W_new, d3)

        # ---- external budget accumulation -------------------------------------
        P = Jg.sum(0) + Jd.sum(0) + Jc0 + JcN
        A_acc = (np.cross(self.x, Jg).sum(0)
                 + np.cross(self.x, Jd).sum(0)
                 + np.cross(self.x[0], Jc0) + np.cross(self.x[N - 1], JcN)
                 + dt * (cL + cR)
                 + np.einsum("nij,nj->ni", R, cdrag_body).sum(0))
        self.P_ext = self.P_ext + P
        self.A_ext = self.A_ext + A_acc

        # ---- instantaneous per-term external forces for sampling --------------
        fd = -self.c_lin * self.v
        fdrag = fd.sum(0)
        wdrag = np.linalg.norm(fd, axis=1)
        sw = wdrag.sum()
        if sw > 1e-12:
            p_drag = (wdrag[:, None] * self.x).sum(0) / sw
        else:
            p_drag = self.inst["p_grav"]
        wc = np.linalg.norm(np.stack((Jc0, JcN)), axis=1).sum()
        if wc > 1e-15:
            p_con = (np.linalg.norm(Jc0) * self.x[0]
                     + np.linalg.norm(JcN) * self.x[N - 1]) / wc
        else:
            p_con = 0.5 * (self.left_pos + self.right_pos)
        com = (m_node[:, None] * self.x).sum(0) / m_node.sum()
        self.inst = {
            "f_grav": (m_node[:, None] * self.g).sum(0),
            "f_drag": fdrag,
            "f_con": (Jc0 + JcN) / dt,
            "c_tw": cL + cR,
            "c_ad": np.einsum("nij,nj->ni", R, cdrag_body).sum(0) / dt,
            "p_grav": com,
            "p_drag": p_drag,
            "p_con": p_con,
        }

        # ---- commit ------------------------------------------------------------
        self.x, self.v, self.q, self.W, self.phi = x, v, q, W_new, phi
        self.t = self.t + dt

        # ---- divergence guard ----------------------------------------------------
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()
                and np.isfinite(self.q).all() and np.isfinite(self.W).all()):
            self._restore_backup()
            self.diverged = True

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = max(float(time), 0.0)
        if not self.diverged:
            # number of whole substeps that fit before the requested time
            while self.t < target - 0.5 * self.dt:
                if self.diverged:
                    break
                self._substep()
        return self._build_frame(target)

    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        N, S = self.N, self.S
        x, v, q, W = self.x, self.v, self.q, self.W
        R = quat_to_R(q)

        # per-vertex mapping: vertex j carries segment j (j = 0..S-1);
        # the last vertex carries no rotational state.
        inertia = np.zeros((N, 3, 3))
        inertia[:S] = np.einsum("nij,nj,nkj->nik", R, self.Ib, R)
        ang_vel = np.zeros((N, 3))
        ang_vel[:S] = W
        quat_v = np.zeros((N, 4))
        quat_v[:S] = q
        quat_v[N - 1] = (0.0, 0.0, 0.0, 1.0)

        phiL, phiR = self._twist_cmd(time)
        turns = np.zeros(N)
        turns[:S] = self.phi / (2.0 * np.pi)

        inst = self.inst
        f_ext = np.stack((inst["f_grav"], inst["f_drag"], inst["f_con"],
                          np.zeros(3), np.zeros(3)))[None, :, :]
        c_ext = np.stack((np.zeros(3), np.zeros(3), np.zeros(3),
                          inst["c_tw"], inst["c_ad"]))[None, :, :]
        p_ext = np.stack((inst["p_grav"], inst["p_drag"], inst["p_con"],
                          0.5 * (self.left_pos + self.right_pos),
                          inst["p_grav"]))[None, :, :]

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            orientation=quat_v,
            angular_velocity=ang_vel,
            mass=self.m_node.copy(),
            attributes={
                "inertia_tensor": inertia,
                "mass_quantity": self.m_node.copy(),
                "arc_fraction": np.linspace(0.0, 1.0, N),
                "rod_radius": np.full(N, self.radius),
                "contact_diameter": np.full(N, self.cd),
                "endpoint_controlled": np.where(
                    (np.arange(N) == 0) | (np.arange(N) == N - 1), 1.0, 0.0),
                "material_twist_turns": turns,
            },
            fields={
                "budget_cumulative_external_impulse": self.P_ext[None, :].copy(),
                "budget_cumulative_external_angular_impulse": self.A_ext[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, N)),
                "budget_external_force": f_ext,
                "budget_external_couple": c_ext,
                "budget_force_position": p_ext,
                "constraint_value": np.array([[self.phi[0]], [self.phi[S - 1]]]),
                "constraint_target": np.array([[phiL], [phiR]]),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "reference_momentum": float(self.scal_ref_mom),
                "reference_angular_momentum": float(self.scal_ref_ang),
                "rest_length": float(self.L),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "substeps": int(self.step_count),
            "diverged": bool(self.diverged),
            "min_self_contact_distance": (
                None if self.min_contact_dist == float("inf")
                else float(self.min_contact_dist)),
            "end_spin_left_rad": float(self.phi[0]),
            "end_spin_right_rad": float(self.phi[self.S - 1]),
            "cumulative_external_impulse": self.P_ext.tolist(),
            "cumulative_external_angular_impulse": self.A_ext.tolist(),
        }
