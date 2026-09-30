"""Rigid-body bicycle-chain-on-sprockets solver.

Planar (x-z) constrained multibody dynamics, integrated with a
velocity-impulse time-stepping scheme (semi-implicit Euler + exact
constraint projection).  No keyframes or scripted motion: every sample is
produced by numerically stepping the physics.

Model
-----
* 20 rigid chain links form a closed kinematic loop through 20 revolute
  (pin) joints located at the chain pitch points (anchor points at
  +/- half the chain pitch along each link axis).
* Two sprockets spin about fixed axles (rotation-only degrees of freedom).
  The right sprocket receives a prescribed drive torque ramping up at
  t = 0; the left sprocket is a free follower with bearing drag.
* Unilateral contact keeps every link roller center outside the sprocket
  engagement circle (and above the floor); normal impulses are solved
  with an active-set complementarity loop.
* A link seated on a sprocket is captured by a tooth pocket: a no-slip
  engagement constraint locks the link's angular rate around the sprocket
  to the sprocket's spin.  This transmits the drive torque to the chain
  and drags the follower sprocket around coherently (no slip, no
  derailment), exactly like tooth engagement in a real chain drive.
* Joint position drift is removed every substep with a SHAKE-style
  projection, so the chain never stretches.

The geometry (link loop, sprocket centers, engagement radius, pitch,
tooth count, inertias from the frozen meshes) is derived from the frozen
configuration and scene geometry files.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TWO_PI = 2.0 * np.pi

# constraint row kinds
K_JOINT = 0
K_SPK_N = 1
K_ENGAGE = 2
K_FLOOR = 3

DOF_THL = 60  # left sprocket spin (psi-rate, CCW in the x-z plane)
DOF_THR = 61  # right sprocket spin
NDOF = 62


def _wrap_pm(value, period):
    """Wrap scalar/array to [-period/2, period/2)."""
    return np.mod(value + 0.5 * period, period) - 0.5 * period


def _read_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError(f"no vertices found in {path}")
    return np.asarray(verts, dtype=float)


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if cfg is None:  # defensive fallback
            meta = scene.metadata()
            cfg = {
                "bodies": [],
                "gravity": meta.get("gravity", [0.0, 0.0, -9.81]),
                "contact": {"friction": 0.5, "length_scale": 0.02},
            }
        bodies = list(cfg["bodies"])
        ids = [str(b["id"]) for b in bodies]
        link_rows = sorted(
            (i for i, name in enumerate(ids) if name.startswith("chain_")),
            key=lambda i: int(ids[i].rsplit("_", 1)[1]),
        )
        row_left = ids.index("sprocket_left")
        row_right = ids.index("sprocket_right")

        self.NL = NL = len(link_rows)
        n_bodies = NL + 2
        self.n_bodies = n_bodies

        # per-body output arrays (order: links by id, then left, right)
        mass = np.zeros(n_bodies)
        mass[:NL] = [float(bodies[i]["mass"]) for i in link_rows]
        mass[NL] = float(bodies[row_left]["mass"])
        mass[NL + 1] = float(bodies[row_right]["mass"])
        self.mass_out = mass
        self.body_id_out = np.arange(n_bodies, dtype=np.int32)

        # initial link centers in the x-z plane
        P = np.asarray(
            [
                [float(bodies[i]["position"][0]), float(bodies[i]["position"][2])]
                for i in link_rows
            ],
            dtype=float,
        )
        self.CL = np.asarray(
            [bodies[row_left]["position"][0], bodies[row_left]["position"][2]], dtype=float
        )
        self.CR = np.asarray(
            [bodies[row_right]["position"][0], bodies[row_right]["position"][2]], dtype=float
        )
        self.C_spk = np.stack([self.CL, self.CR])  # index 0 = left, 1 = right

        # frozen scene parameters
        self.g = float(cfg.get("gravity", [0.0, 0.0, -9.81])[2])
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.5))
        self.contact_length_scale = float(contact.get("length_scale", 0.02))

        # ---- mesh-derived rigid properties (frozen geometry assets) -----
        root = Path(__file__).resolve().parents[1]
        link_obj = root / "assets" / "chain_link.obj"
        spk_obj = root / "assets" / "sprocket.obj"
        if not link_obj.is_file():
            link_obj = Path("assets") / "chain_link.obj"
        if not spk_obj.is_file():
            spk_obj = Path("assets") / "sprocket.obj"
        lv = _read_obj_vertices(link_obj)
        sv = _read_obj_vertices(spk_obj)
        lx = float(lv[:, 0].max() - lv[:, 0].min())
        lz = float(lv[:, 2].max() - lv[:, 2].min())
        r_tip = float(np.max(np.hypot(sv[:, 0], sv[:, 2])))

        m_l = mass[:NL].copy()
        I_l = m_l / 12.0 * (lx * lx + lz * lz)  # slender link about y
        m_s = mass[NL:].copy()
        I_s = 0.5 * m_s * r_tip * r_tip  # disk sprocket about its axle
        self.m_l, self.I_l, self.m_s, self.I_s = m_l, I_l, m_s, I_s
        self.im = 1.0 / m_l
        self.iI = 1.0 / I_l
        self.iIs = 1.0 / I_s

        # ---- chain kinematics -------------------------------------------
        # pitch from the frozen initial loop: closed loop of NL links
        nxt = (np.arange(NL) + 1) % NL
        seg = P[nxt] - P
        loop_len = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        self.p = pitch = loop_len / NL
        self.h = 0.5 * pitch
        center_dist = float(np.hypot(*(self.CR - self.CL)))
        # taut wrap radius implied by the frozen loop length and axle span
        R_c = (loop_len - 2.0 * center_dist) / TWO_PI
        self.R_c = float(np.clip(R_c, 0.3, 0.8))
        # tooth count so that the tooth pitch matches the chain pitch
        self.Nt = int(np.clip(round(TWO_PI * self.R_c / pitch), 4, 24))
        self.tooth = TWO_PI / self.Nt

        # initial link orientations: along the local loop tangent
        prev = (np.arange(NL) - 1) % NL
        tang = P[nxt] - P[prev]
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        phi = np.arctan2(-tang[:, 1], tang[:, 0])

        # tooth pocket phase offsets aligned with the initially seated links
        off = np.zeros(2)
        for s_i in (0, 1):
            d = P - self.C_spk[s_i]
            rho = np.hypot(d[:, 0], d[:, 1])
            seated = (rho - self.R_c < 0.05) & (rho > self.R_c - 0.08)
            if np.any(seated):
                psi = np.arctan2(d[seated, 1], d[seated, 0])
                z = np.mean(np.exp(1j * TWO_PI * psi / self.tooth))
                off[s_i] = float(np.angle(z)) * self.tooth / TWO_PI
        self.off = off

        # ---- state -------------------------------------------------------
        self.px = P[:, 0].copy()
        self.pz = P[:, 1].copy()
        self.phi = phi
        self.th = np.zeros(2)  # sprocket spins (psi-rates integrated)
        self.u = np.zeros(NDOF)

        Minv = np.zeros(NDOF)
        Minv[0:NL] = self.im
        Minv[NL:2 * NL] = self.im
        Minv[2 * NL:3 * NL] = self.iI
        Minv[DOF_THL] = self.iIs[0]
        Minv[DOF_THR] = self.iIs[1]
        self.Minv = Minv

        # ---- numerics / physical parameters ------------------------------
        self.dt = 1.0e-3
        self.c_link = 0.5        # per-link viscous loss (N s/m)
        self.c_rot = 0.003       # link spin loss (N m s)
        self.b_spk = 0.5         # sprocket bearing drag (N m s)
        self.tau0 = 18.0         # prescribed drive torque (N m)
        self.ramp = 0.8          # drive engagement ramp (s)
        self.beta_j, self.vj_max = 0.25, 2.0
        self.beta_c, self.vc_max = 0.30, 0.5
        self.beta_e, self.ve_max = 0.15, 0.7
        self.seat_margin = 0.05
        self.h_floor = 0.5 * lz  # link half height (mesh)
        self.lam_cap = 20.0
        self.v_cap = 8.0
        self.om_cap = 60.0
        self.thd_cap = 30.0

        self.t = 0.0
        self._steps = 0
        self._rec = None
        self._ready = True

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _substep(self, dt: float, t: float) -> None:
        NL = self.NL
        u = self.u
        px, pz, phi, th = self.px, self.pz, self.phi, self.th
        backup = (px.copy(), pz.copy(), phi.copy(), th.copy(), u.copy())

        # ---- external forces and drive torque (semi-implicit) -----------
        u[0:NL] += dt * (-self.c_link * u[0:NL]) * self.im
        u[NL:2 * NL] += dt * (-self.m_l * self.g - self.c_link * u[NL:2 * NL]) * self.im
        u[2 * NL:3 * NL] += dt * (-self.c_rot * u[2 * NL:3 * NL]) * self.iI
        s = min(1.0, max(0.0, t / self.ramp))
        s = s * s * (3.0 - 2.0 * s)
        u[DOF_THR] += dt * (self.tau0 * s - self.b_spk * u[DOF_THR]) * self.iIs[1]
        u[DOF_THL] += dt * (-self.b_spk * u[DOF_THL]) * self.iIs[0]

        # ---- assemble constraint rows ------------------------------------
        r = np.arange(NL)
        B = (r + 1) % NL
        h = self.h
        cA = np.cos(phi)
        sA = np.sin(phi)
        cB = cA[B]
        sB = sA[B]

        J = np.zeros((2 * NL, NDOF))
        # x rows: vx_A - vx_B - h sA wA - h sB wB
        J[r, r] = 1.0
        J[r, B] = -1.0
        J[r, 2 * NL + r] = -h * sA
        J[r, 2 * NL + B] = -h * sB
        # z rows: vz_A - vz_B - h cA wA - h cB wB
        J[NL + r, NL + r] = 1.0
        J[NL + r, NL + B] = -1.0
        J[NL + r, 2 * NL + r] = -h * cA
        J[NL + r, 2 * NL + B] = -h * cB
        Cx = (px[r] + h * cA) - (px[B] - h * cB)
        Cz = (pz[r] - h * sA) - (pz[B] + h * sB)
        tgt = np.empty(2 * NL)
        tgt[r] = -np.clip(self.beta_j * Cx / dt, -self.vj_max, self.vj_max)
        tgt[NL + r] = -np.clip(self.beta_j * Cz / dt, -self.vj_max, self.vj_max)
        kind = np.zeros(2 * NL, dtype=np.int8)

        row_blocks = [J]
        tgt_blocks = [tgt]
        kind_blocks = [kind]
        meta = []  # (kind, link, sprocket, normal2d, aux)

        for s_i in (0, 1):
            C = self.C_spk[s_i]
            dof = DOF_THL if s_i == 0 else DOF_THR
            dx = px - C[0]
            dz = pz - C[1]
            rho = np.hypot(dx, dz)
            gap = rho - self.R_c
            # unilateral normal contact (cannot penetrate the sprocket)
            for i in np.nonzero(gap < 0.0)[0]:
                row = np.zeros(NDOF)
                row[i] = dx[i] / rho[i]
                row[NL + i] = dz[i] / rho[i]
                t_i = float(np.clip(-self.beta_c * gap[i] / dt, 0.0, self.vc_max))
                row_blocks.append(row[None, :])
                tgt_blocks.append(np.asarray([t_i]))
                kind_blocks.append(np.asarray([K_SPK_N], dtype=np.int8))
                meta.append((K_SPK_N, int(i), s_i, (dx[i] / rho[i], dz[i] / rho[i]), 0.0))
            # tooth-pocket engagement (no slip while seated)
            seated = (gap < self.seat_margin) & (rho > self.R_c - 0.08)
            for i in np.nonzero(seated)[0]:
                row = np.zeros(NDOF)
                row[i] = -dz[i] / (rho[i] * rho[i])
                row[NL + i] = dx[i] / (rho[i] * rho[i])
                row[dof] = -1.0
                psi = math.atan2(dz[i], dx[i])
                delta = float(_wrap_pm(psi - th[s_i] - self.off[s_i], self.tooth))
                t_i = float(np.clip(-self.beta_e * delta / dt, -self.ve_max, self.ve_max))
                row_blocks.append(row[None, :])
                tgt_blocks.append(np.asarray([t_i]))
                kind_blocks.append(np.asarray([K_ENGAGE, ], dtype=np.int8))
                meta.append((K_ENGAGE, int(i), s_i, (-dz[i] / rho[i], dx[i] / rho[i]), delta))
        # floor contact
        for i in np.nonzero(pz < self.h_floor)[0]:
            row = np.zeros(NDOF)
            row[NL + i] = 1.0
            t_i = float(np.clip(-self.beta_c * (pz[i] - self.h_floor) / dt, 0.0, self.vc_max))
            row_blocks.append(row[None, :])
            tgt_blocks.append(np.asarray([t_i]))
            kind_blocks.append(np.asarray([K_FLOOR], dtype=np.int8))
            meta.append((K_FLOOR, int(i), -1, (0.0, 1.0), 0.0))

        J_all = np.vstack(row_blocks)
        b_all = np.concatenate(tgt_blocks)
        kind_all = np.concatenate(kind_blocks)
        n_rows = len(b_all)
        unilateral = kind_all >= K_SPK_N
        unilateral &= kind_all != K_ENGAGE

        # ---- active-set impulse solve ------------------------------------
        active = np.ones(n_rows, dtype=bool)
        lam_full = np.zeros(n_rows)
        for _ in range(8):
            idx = np.nonzero(active)[0]
            Js = J_all[idx]
            S = (Js * self.Minv) @ Js.T
            eps = 1e-7 * max(1.0, float(np.max(np.abs(S))) * max(1, len(idx)))
            S[np.diag_indices_from(S)] += eps
            rhs = b_all[idx] - Js @ u
            try:
                lami = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                lami = np.linalg.lstsq(S, rhs, rcond=None)[0]
            lami = np.clip(lami, -self.lam_cap, self.lam_cap)
            bad = unilateral[idx] & (lami < -1e-9)
            if np.any(bad):
                active[idx[bad]] = False
                continue
            lam_full[:] = 0.0
            lam_full[idx] = lami
            break
        else:
            idx = np.nonzero(active)[0]
            Js = J_all[idx]
            S = (Js * self.Minv) @ Js.T
            eps = 1e-7 * max(1.0, float(np.max(np.abs(S))) * max(1, len(idx)))
            S[np.diag_indices_from(S)] += eps
            rhs = b_all[idx] - Js @ u
            lami = np.clip(np.linalg.solve(S, rhs), -self.lam_cap, self.lam_cap)
            lam_full[:] = 0.0
            lam_full[idx] = lami

        u += self.Minv * (J_all.T @ lam_full)

        # ---- Coulomb friction for floor contacts -------------------------
        for j, (kd, i, s_i, nrm, aux) in enumerate(meta):
            if kd == K_FLOOR and lam_full[2 * NL + j] > 0.0:
                ln = lam_full[2 * NL + j]
                want = -self.m_l[i] * u[i]
                cap = self.mu * ln
                u[i] += np.clip(want, -cap, cap) / self.m_l[i]

        # ---- integrate positions -----------------------------------------
        px += dt * u[0:NL]
        pz += dt * u[NL:2 * NL]
        phi += dt * u[2 * NL:3 * NL]
        th += dt * u[DOF_THL:DOF_THR + 1]

        # ---- position projection -----------------------------------------
        for _ in range(3):
            # push links out of the sprocket engagement circles / floor
            for s_i in (0, 1):
                C = self.C_spk[s_i]
                dx = px - C[0]
                dz = pz - C[1]
                rho = np.hypot(dx, dz)
                bad = rho < self.R_c
                if np.any(bad):
                    px[bad] = C[0] + dx[bad] * (self.R_c / rho[bad])
                    pz[bad] = C[1] + dz[bad] * (self.R_c / rho[bad])
            low = pz < self.h_floor
            if np.any(low):
                pz[low] = self.h_floor
            # SHAKE projection of the pin joints (keeps the loop inextensible)
            for _ in range(6):
                cA = np.cos(phi)
                sA = np.sin(phi)
                cB = cA[B]
                sB = sA[B]
                Cx = (px[r] + h * cA) - (px[B] - h * cB)
                Cz = (pz[r] - h * sA) - (pz[B] + h * sB)
                im = self.im
                iI = self.iI
                a11 = im[r] + im[B] + (h * h) * (sA * sA * iI[r] + sB * sB * iI[B])
                a12 = (h * h) * (sA * cA * iI[r] + sB * cB * iI[B])
                a22 = im[r] + im[B] + (h * h) * (cA * cA * iI[r] + cB * cB * iI[B])
                det = a11 * a22 - a12 * a12
                det = np.where(np.abs(det) < 1e-12, 1e-12, det)
                lx_ = (a22 * Cx - a12 * Cz) / det
                lz_ = (a11 * Cz - a12 * Cx) / det
                rel = 0.9
                px[r] -= rel * im[r] * lx_
                pz[r] -= rel * im[r] * lz_
                phi[r] += rel * iI[r] * h * (sA * lx_ + cA * lz_)
                px[B] += rel * im[B] * lx_
                pz[B] += rel * im[B] * lz_
                phi[B] += rel * iI[B] * h * (sB * lx_ + cB * lz_)

        # ---- safety clamps ------------------------------------------------
        u[0:2 * NL] = np.clip(u[0:2 * NL], -self.v_cap, self.v_cap)
        u[2 * NL:3 * NL] = np.clip(u[2 * NL:3 * NL], -self.om_cap, self.om_cap)
        u[DOF_THL:DOF_THR + 1] = np.clip(u[DOF_THL:DOF_THR + 1], -self.thd_cap, self.thd_cap)
        if not (np.isfinite(u).all() and np.isfinite(px).all()
                and np.isfinite(pz).all() and np.isfinite(phi).all()
                and np.isfinite(th).all()):
            (self.px, self.pz, self.phi, self.th, self.u) = backup
            self.u[:] = 0.0
            return

        # ---- record impulse data for trajectory export ---------------------
        joint_lam = np.stack([lam_full[r], lam_full[NL + r]], axis=1)
        contacts = []
        for j, (kd, i, s_i, nrm, aux) in enumerate(meta):
            lam_j = lam_full[2 * NL + j]
            contacts.append((kd, i, s_i, nrm, aux, lam_j))
        self._rec = {"dt": dt, "joint_lam": joint_lam, "contacts": contacts}
        self._steps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver not initialized")
        target = max(float(time), 0.0)
        if target < self.t:
            target = self.t
        while target - self.t > 1e-9:
            remaining = target - self.t
            n_full = int(math.floor(remaining / self.dt + 1e-9))
            if n_full > 0:
                for _ in range(n_full):
                    self._substep(self.dt, self.t)
                    self.t += self.dt
                continue
            self._substep(remaining, self.t)
            self.t += remaining
        if abs(self.t - target) < 1e-9:
            self.t = target
        return self._frame(self.t)

    # ------------------------------------------------------------------
    # frame export
    # ------------------------------------------------------------------
    def _frame(self, time: float):
        NL = self.NL
        n = self.n_bodies
        px, pz, phi, th, u = self.px, self.pz, self.phi, self.th, self.u

        position = np.zeros((n, 3))
        position[:NL, 0] = px
        position[:NL, 2] = pz
        position[NL, 0] = self.CL[0]
        position[NL, 2] = self.CL[1]
        position[NL + 1, 0] = self.CR[0]
        position[NL + 1, 2] = self.CR[1]

        velocity = np.zeros((n, 3))
        velocity[:NL, 0] = u[0:NL]
        velocity[:NL, 2] = u[NL:2 * NL]

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0
        orientation[:NL, 1] = np.sin(0.5 * phi)
        orientation[:NL, 3] = np.cos(0.5 * phi)
        orientation[NL, 1] = -np.sin(0.5 * th[0])
        orientation[NL, 3] = np.cos(0.5 * th[0])
        orientation[NL + 1, 1] = -np.sin(0.5 * th[1])
        orientation[NL + 1, 3] = np.cos(0.5 * th[1])
        orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)

        angular_velocity = np.zeros((n, 3))
        angular_velocity[:NL, 1] = u[2 * NL:3 * NL]
        angular_velocity[NL, 1] = -u[DOF_THL]
        angular_velocity[NL + 1, 1] = -u[DOF_THR]

        # external forces (gravity + viscous losses); drive torque acts on
        # the right sprocket spin and is not a body force.
        external = np.zeros((n, 3))
        external[:NL, 0] = -self.c_link * u[0:NL]
        external[:NL, 2] = -self.m_l * self.g - self.c_link * u[NL:2 * NL]
        external[NL:, 2] = -self.m_s * self.g

        # paired interaction forces (pin joints + contacts + engagement)
        forces = []
        rec = self._rec
        if rec is None:
            for _ in range(NL):
                forces.append((0.0, 0.0, 0.0))
        else:
            dtl = rec["dt"] if rec["dt"] > 0 else self.dt
            jl = rec["joint_lam"]
            for k in range(NL):
                forces.append((jl[k, 0] / dtl, 0.0, jl[k, 1] / dtl))
            for (kd, i, s_i, nrm, aux, lam_j) in rec["contacts"]:
                if kd == K_SPK_N:
                    forces.append((lam_j * nrm[0] / dtl, 0.0, lam_j * nrm[1] / dtl))
                elif kd == K_ENGAGE:
                    # tangential tooth force on the link (J^T lambda / dt)
                    rho = max(self.R_c, 1e-6)
                    fx = lam_j * nrm[0] / rho / dtl
                    fz = lam_j * nrm[1] / rho / dtl
                    forces.append((fx, 0.0, fz))
                elif kd == K_FLOOR:
                    forces.append((0.0, 0.0, lam_j / dtl))
        interaction = np.asarray(forces, dtype=float).reshape(-1, 3)

        # signed contact gaps (sprocket engagement circles + floor)
        gaps = []
        for s_i in (0, 1):
            C = self.C_spk[s_i]
            dx = px - C[0]
            dz = pz - C[1]
            gap = np.hypot(dx, dz) - self.R_c
            for i in np.nonzero(gap < 0.1)[0]:
                gaps.append(float(gap[i]))
        fgap = pz - self.h_floor
        for i in np.nonzero(fgap < 0.1)[0]:
            gaps.append(float(fgap[i]))
        if not gaps:
            gaps = [0.0]
        contact_signed_distance = np.asarray(gaps, dtype=float)

        frame_fields = {
            "external_force": external,
            "interaction_force": interaction,
            "contact_signed_distance": contact_signed_distance,
        }
        from worldsolver.contracts import StateFrame

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass_out,
            body_id=self.body_id_out,
            attributes={"release_time": np.zeros(n)},
            fields=frame_fields,
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        info: dict[str, object] = {
            "method": "planar constrained multibody impulse dynamics",
            "substeps": int(self._steps),
        }
        if self._ready:
            speed = float(np.mean(np.hypot(self.u[0:self.NL], self.u[self.NL:2 * self.NL])))
            info["mean_chain_speed"] = speed
            info["sprocket_spins_rad"] = [float(self.th[0]), float(self.th[1])]
            info["chain_pitch"] = float(self.p)
            info["engagement_radius"] = float(self.R_c)
            info["tooth_count"] = int(self.Nt)
        return info
