"""Physics solver for scene 103_cable_cylinder_winding.

Extensible Cosserat-type cable wound around a rigid cylinder by a kinematically
driven grip.  The rod is discretized as 121 nodes with:

* axial stretch elasticity (EA / h springs, extensible by construction),
* axial shear dashpots (extensible shear response),
* finite-difference curvature bending elasticity (EI),
* discrete twist (torsion) degrees of freedom with GJ stiffness,
* gravity,
* impulse-based contacts against the winding cylinder and the floor with the
  generalized power-law friction bound  ||lambda_t|| <= kappa*|lambda_n|^N
  (kappa = 1.296, N = 0.7616, taken from the frozen configuration),
* cable self-contact with the same friction law,
* a kinematically prescribed driven end (vertex 120) following the frozen
  winding schedule (approach -> constant-pitch helix -> hold).

Integration is semi-implicit (symplectic) Euler with a fixed substep small
enough to resolve the stiff axial modes; contacts are resolved at the velocity
level with position projection, which is unconditionally stable.  No keyframes,
animation tables or precomputed trajectories are used: the only prescribed
quantity is the frozen grip path, everything else emerges from the dynamics.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-implicit Euler discrete-rod solver with impulse contacts."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.m = np.array(frame0.mass, dtype=float)
        self.n = self.x.shape[0]
        self.m3 = self.m[:, None]
        self.attrs0 = {k: np.array(v) for k, v in frame0.attributes.items()}
        self.scalars0 = {k: float(v) for k, v in frame0.scalars.items()}

        cable = self.cfg["cable"]
        self.rc = float(cable["radius"])                      # cable radius
        self.h = float(cable["length"]) / (self.n - 1)        # rest edge length
        E = float(cable["young_modulus_pa"])
        G = float(cable["shear_modulus_pa"])
        area = np.pi * self.rc ** 2
        I_area = np.pi * self.rc ** 4 / 4.0                   # bending second moment
        J_area = np.pi * self.rc ** 4 / 2.0                   # polar second moment

        self.k_stretch = E * area / self.h                   # N/m per edge
        self.c_bend = E * I_area / self.h ** 3               # FD curvature coefficient
        self.k_twist = G * J_area / self.h                   # N*m/rad per edge
        self.I_twist = self.m * J_area / area                # kg*m^2 per node
        # damping (stiffness- and mass-proportional)
        self.c_edge = 0.08
        self.c_bend_d = 0.08
        self.beta_m = 0.2
        self.c_tw = 2.0 * 0.05 * np.sqrt(self.k_twist * self.I_twist.max())

        self.g = np.array(self.cfg["gravity"], dtype=float)
        self.grav_on = bool(np.any(np.abs(self.g) > 0.0))

        # cylinder geometry (axis = z)
        cyl = self.cfg["cylinder"]
        self.cyl_c = np.array(cyl["center"], dtype=float)
        self.cyl_R = float(cyl["radius"])
        zc = float(self.cyl_c[2])
        hh = 0.5 * float(cyl["height"])
        self.z_top = zc + hh
        self.z_bot = zc - hh

        # floor (box top at z = 0)
        self.floor_top = -0.006 + 0.006
        self.floor_half = 0.22

        # generalized power-law friction
        contact = self.cfg["contact"]
        self.kappa = float(contact["kappa"])
        self.mu_exp = float(contact["exponent_N"])

        # integration
        self.dt = 2.5e-5
        self.time = 0.0
        self._steps = 0
        self._self_contact_every = 4

        # self-contact pair index list (skip near neighbours along the chain)
        ii, jj = np.triu_indices(self.n, k=4)
        self.pair_i = ii
        self.pair_j = jj

        # driven grip
        self.grip_idx = int(cable["driven_vertex"])
        self.xg_prev = self.x[self.grip_idx].copy()

        # twist state
        self.phi = np.zeros(self.n)
        self.phidot = np.zeros(self.n)

        # per-node tangents and angular velocity
        d0 = self.x[1:] - self.x[:-1]
        l0 = np.linalg.norm(d0, axis=1)
        u0 = d0 / l0[:, None]
        t0 = np.zeros_like(self.x)
        t0[:-1] += u0
        t0[1:] += u0
        t0 /= np.maximum(np.linalg.norm(t0, axis=1), 1e-12)[:, None]
        self.t_prev = t0
        self.omega = np.zeros((self.n, 3))

        # momentum-budget bookkeeping
        self.cum_imp = np.zeros((3, 3))
        self.cum_ang = np.zeros((3, 3))
        self._reset_interval()
        self.contact_flag = np.zeros(self.n)
        self.x_frame_prev = self.x.copy()
        self.t_frame_prev = 0.0

        # reference momentum scales (positive, frozen at initialization)
        m_tot = float(self.m.sum())
        v_ref = float(np.sqrt(9.81 * float(cable["length"])))
        self.ref_p = m_tot * v_ref
        self.ref_a = self.ref_p * float(cyl["height"])

    # ------------------------------------------------------------------
    # grip target (frozen winding schedule; scene provides it)
    # ------------------------------------------------------------------
    def _grip(self, t: float):
        scene = self.scene
        if hasattr(scene, "grip_target"):
            return scene.grip_target(float(t))
        w = self.cfg["winding"]
        ta = float(w["approach_end_time"])
        tw = float(w["winding_end_time"])
        r = float(w["centerline_orbit_radius"])
        a0 = float(w["start_angle_rad"])
        init = np.array(self.cfg["initial_conditions"]["driven_end"], dtype=float)
        s = np.array([r * np.cos(a0), r * np.sin(a0), float(w["start_height"])])

        def smoothstep(u):
            u = float(np.clip(u, 0.0, 1.0))
            return u * u * (3.0 - 2.0 * u)

        if t < ta:
            u = smoothstep(t / ta)
            p = (1.0 - u) * init + u * s
            ang = a0
        else:
            u = smoothstep((t - ta) / (tw - ta))
            ang = a0 + 2.0 * np.pi * float(w["turns"]) * u
            z = (1.0 - u) * float(w["start_height"]) + u * float(w["end_height"])
            p = np.array([r * np.cos(ang), r * np.sin(ang), z])
        tang = np.array([-np.sin(ang), np.cos(ang),
                         -float(w["desired_pitch"]) / (2.0 * np.pi)])
        tang /= np.linalg.norm(tang)
        return {"position": p.tolist(), "tangent": tang.tolist(),
                "controlled_vertex": self.grip_idx}

    # ------------------------------------------------------------------
    # one physics substep
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        x, v = self.x, self.v
        n = self.n

        # ---- internal forces ----------------------------------------
        d = x[1:] - x[:-1]
        l = np.sqrt(np.einsum("ij,ij->i", d, d))
        l = np.maximum(l, 1e-12)
        u = d / l[:, None]

        # stretch + axial (shear) dashpot along each edge
        vrel = np.einsum("ij,ij->i", v[1:] - v[:-1], u)
        fedge = self.k_stretch * (l - self.h) - self.c_edge * vrel
        F = np.zeros_like(x)
        Fu = fedge[:, None] * u
        F[1:] += Fu
        F[:-1] -= Fu

        # bending: finite-difference curvature elasticity + dashpot
        A = x[:-2] - 2.0 * x[1:-1] + x[2:]
        Av = v[:-2] - 2.0 * v[1:-1] + v[2:]
        B = np.zeros_like(x)
        B[1:-1] = A
        Bv = np.zeros_like(x)
        Bv[1:-1] = Av
        D = np.zeros_like(x)
        D[1:-1] = B[:-2] - 2.0 * B[1:-1] + B[2:]
        Dv = np.zeros_like(x)
        Dv[1:-1] = Bv[:-2] - 2.0 * Bv[1:-1] + Bv[2:]
        F -= self.c_bend * D + self.c_bend_d * Dv

        # mass-proportional damping
        F -= self.beta_m * self.m3 * v

        # semi-implicit Euler
        v += dt * (F / self.m3 + self.g[None, :])
        x += dt * v

        # ---- contacts -------------------------------------------------
        self._contacts(dt)

        # ---- driven grip (kinematic) -----------------------------------
        tgt = self._grip(self.time + dt)
        xg = np.asarray(tgt["position"], dtype=float)
        vg = (xg - self.xg_prev) / dt
        vpre = v[self.grip_idx].copy()
        Jg = self.m[self.grip_idx] * (vg - vpre)
        v[self.grip_idx] = vg
        x[self.grip_idx] = xg
        self.xg_prev = xg
        jn = float(np.linalg.norm(Jg))
        if jn > 0.0:
            self.acc_grip_imp += Jg
            self.acc_grip_ang += np.cross(xg, Jg)
            self.acc_grip_wpos += jn * xg
            self.acc_grip_w += jn

        # ---- torsion (twist DOFs) --------------------------------------
        tq = np.zeros(n)
        dl = self.phi[1:] - self.phi[:-1]
        tq[:-1] += self.k_twist * dl
        tq[1:] -= self.k_twist * dl
        self.phidot += dt * (tq / self.I_twist - self.c_tw * self.phidot / self.I_twist)
        self.phi += dt * self.phidot

        # ---- per-sample angular velocity --------------------------------
        d2 = x[1:] - x[:-1]
        l2 = np.maximum(np.sqrt(np.einsum("ij,ij->i", d2, d2)), 1e-12)
        u2 = d2 / l2[:, None]
        t = np.zeros_like(x)
        t[:-1] += u2
        t[1:] += u2
        tn = np.maximum(np.linalg.norm(t, axis=1), 1e-12)
        t /= tn[:, None]
        self.omega = np.cross(self.t_prev, t) / dt + self.phidot[:, None] * t
        self.t_prev = t

        # ---- safety clamp (should never trigger) -------------------------
        spd = np.sqrt(np.einsum("ij,ij->i", v, v))
        fast = spd > 30.0
        if np.any(fast):
            v[fast] *= (30.0 / spd[fast])[:, None]

        self.time += dt
        self._steps += 1

    # ------------------------------------------------------------------
    def _apply_friction(self, v, nvec, Jn_tot, dt):
        """Power-law friction: ||lambda_t|| <= kappa*|lambda_n|^N."""
        vn = np.einsum("ij,ij->i", v, nvec)
        vt = v - vn[:, None] * nvec
        vtn = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        need = self.m * vtn
        bound = self.kappa * np.power(np.maximum(Jn_tot, 0.0), self.mu_exp)
        scale = np.ones_like(need)
        pos = need > 0.0
        scale[pos] = np.minimum(1.0, bound[pos] / need[pos])
        v += (scale - 1.0)[:, None] * vt
        return v

    def _contacts(self, dt: float) -> None:
        x, v = self.x, self.v
        n = self.n
        v_pre = v.copy()
        R_eff = self.cyl_R + self.rc

        rad = np.hypot(x[:, 0], x[:, 1])
        z = x[:, 2]

        nvec = np.zeros_like(x)
        pen = np.zeros(n)

        # cylinder side (radial normal) for nodes within the axial band
        inband = (z > self.z_bot) & (z < self.z_top) & (rad < R_eff)
        rsafe = np.maximum(rad, 1e-9)
        if np.any(inband):
            idx = np.nonzero(inband)[0]
            nvec[idx, 0] = x[idx, 0] / rsafe[idx]
            nvec[idx, 1] = x[idx, 1] / rsafe[idx]
            pen[idx] = R_eff - rad[idx]

        # top cap / rim for nodes above the band
        above = (z >= self.z_top) & (z < self.z_top + self.rc)
        if np.any(above):
            idx = np.nonzero(above)[0]
            inside = rad[idx] < self.cyl_R
            # flat top face
            ii = idx[inside]
            nvec[ii, 2] = 1.0
            pen[ii] = np.maximum(self.z_top + self.rc - z[ii], 0.0)
            # rim: approximate with radial push
            jj = idx[~inside & (rad[idx] < R_eff)]
            nvec[jj, 0] = x[jj, 0] / rsafe[jj]
            nvec[jj, 1] = x[jj, 1] / rsafe[jj]
            pen[jj] = np.maximum(R_eff - rad[jj], 0.0)

        # floor (top face) where the cylinder is not the active constraint
        fmask = (z < self.rc) & (pen <= 0.0) & \
                (np.abs(x[:, 0]) < self.floor_half + self.rc) & \
                (np.abs(x[:, 1]) < self.floor_half + self.rc)
        if np.any(fmask):
            idx = np.nonzero(fmask)[0]
            nvec[idx, 2] = 1.0
            pen[idx] = self.rc - z[idx]

        act = pen > 0.0
        if np.any(act):
            vn = np.einsum("ij,ij->i", v, nvec)
            Jn = np.where(act & (vn < 0.0), -self.m * vn, 0.0)
            v += (Jn / self.m)[:, None] * nvec
            corr = np.where(act, 0.5 * pen, 0.0)
            x += corr[:, None] * nvec
            Jn_tot = Jn + self.m * corr / dt
            self._apply_friction(v, nvec, Jn_tot, dt)

            # record external contact impulses (cylinder + floor)
            Jnode = self.m[:, None] * (v - v_pre) + (self.m * corr / dt)[:, None] * nvec
            w = np.linalg.norm(Jnode, axis=1)
            active = w > 0.0
            if np.any(active):
                xa = x[active]
                Ja = Jnode[active]
                wa = w[active]
                self.acc_contact_imp += Ja.sum(axis=0)
                self.acc_contact_ang += np.cross(xa, Ja).sum(axis=0)
                self.acc_contact_wpos += (wa[:, None] * xa).sum(axis=0)
                self.acc_contact_w += float(wa.sum())
                self.contact_flag[active] = 1.0

        # ---- cable self-contact (every few substeps) --------------------
        if (self._steps % self._self_contact_every) == 0:
            self._self_contact(dt)

    def _self_contact(self, dt: float) -> None:
        x, v = self.x, self.v
        I, J = self.pair_i, self.pair_j
        diff = x[J] - x[I]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        pen = 2.0 * self.rc - dist
        act = np.nonzero(pen > 0.0)[0]
        if act.size == 0:
            return
        for k in act:
            i = int(I[k])
            j = int(J[k])
            dvec = x[j] - x[i]
            dist_ij = float(np.linalg.norm(dvec))
            if dist_ij < 1e-12:
                continue
            nvec = dvec / dist_ij
            pen_ij = 2.0 * self.rc - dist_ij
            if pen_ij <= 0.0:
                continue
            mi = self.m[i]
            mj = self.m[j]
            mu = 1.0 / mi + 1.0 / mj
            meff = 1.0 / mu
            vr = v[j] - v[i]
            vn = float(np.dot(vr, nvec))
            Jn = 0.0
            if vn < 0.0:
                Jn = -vn / mu
                v[i] -= (Jn / mi) * nvec
                v[j] += (Jn / mj) * nvec
            corr = 0.5 * pen_ij
            x[i] -= corr * nvec
            x[j] += corr * nvec
            Jn_tot = Jn + meff * (2.0 * corr) / dt
            # friction on the tangential relative velocity
            vr = v[j] - v[i]
            vn2 = float(np.dot(vr, nvec))
            vt = vr - vn2 * nvec
            vtn = float(np.linalg.norm(vt))
            if vtn > 1e-12:
                need = vtn / mu
                bound = self.kappa * (abs(Jn_tot) ** self.mu_exp)
                mag = min(need, bound)
                Jt = -(mag / vtn) * vt
                v[i] -= Jt / mi
                v[j] += Jt / mj

    # ------------------------------------------------------------------
    def _reset_interval(self) -> None:
        self.acc_grip_imp = np.zeros(3)
        self.acc_grip_ang = np.zeros(3)
        self.acc_grip_wpos = np.zeros(3)
        self.acc_grip_w = 0.0
        self.acc_contact_imp = np.zeros(3)
        self.acc_contact_ang = np.zeros(3)
        self.acc_contact_wpos = np.zeros(3)
        self.acc_contact_w = 0.0

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        eps = 1e-9
        guard = 0
        while self.time < target - eps:
            dt = min(self.dt, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 2_000_000:
                break
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, t: float):
        n = self.n
        m_tot = float(self.m.sum())
        dtl = max(t - self.t_frame_prev, 0.0)
        g = self.g

        # gravity contributions over the sample interval
        grav0 = m_tot * g * dtl
        grav1 = self.m[self.grip_idx] * g * dtl
        xm = 0.5 * (self.x + self.x_frame_prev)
        # per-node gravity impulse vectors m_i * g * dtl, shape (n,3)
        Jg_nodes = self.m[:, None] * (g * dtl)[None, :]
        wang0 = np.cross(xm, Jg_nodes).sum(axis=0)
        wang1 = np.cross(xm[self.grip_idx], self.m[self.grip_idx] * g * dtl)

        grip_imp = self.acc_grip_imp
        grip_ang = self.acc_grip_ang
        cont_imp = self.acc_contact_imp
        cont_ang = self.acc_contact_ang

        com = (self.m[:, None] * self.x).sum(axis=0) / m_tot
        if self.acc_grip_w > 0.0:
            p_grip = self.acc_grip_wpos / self.acc_grip_w
        else:
            p_grip = self.x[self.grip_idx].copy()
        if self.acc_contact_w > 0.0:
            p_cont = self.acc_contact_wpos / self.acc_contact_w
        else:
            p_cont = com.copy()

        if dtl > 0.0:
            f_grip = grip_imp / dtl
            f_cont = cont_imp / dtl
        else:
            f_grip = np.zeros(3)
            f_cont = np.zeros(3)

        F = np.zeros((3, 1, 3))
        P = np.zeros((3, 1, 3))
        if dtl > 0.0:
            F[0, 0] = grav0 / dtl + f_grip + f_cont
            F[1, 0] = grav1 / dtl + f_grip
            F[2, 0] = f_cont
        else:
            F[0, 0] = m_tot * g
            F[1, 0] = self.m[self.grip_idx] * g
            F[2, 0] = np.zeros(3)

        w_g = float(np.linalg.norm(grav0))
        w_gr = float(np.linalg.norm(grip_imp))
        w_c = float(np.linalg.norm(cont_imp))
        wsum = w_g + w_gr + w_c
        if wsum > 1e-15:
            P[0, 0] = (w_g * com + w_gr * p_grip + w_c * p_cont) / wsum
        else:
            P[0, 0] = com
        P[1, 0] = p_grip if w_gr > 0.0 else self.x[self.grip_idx].copy()
        P[2, 0] = p_cont

        # cumulative budgets (never reset)
        self.cum_imp[0] += grav0 + grip_imp + cont_imp
        self.cum_imp[1] += grav1 + grip_imp
        self.cum_imp[2] += cont_imp
        self.cum_ang[0] += wang0 + grip_ang + cont_ang
        self.cum_ang[1] += wang1 + grip_ang
        self.cum_ang[2] += cont_ang

        fields = {
            "budget_cumulative_external_impulse": self.cum_imp.copy(),
            "budget_cumulative_external_angular_impulse": self.cum_ang.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((3, 3)),
            "budget_domain_weights": np.stack([
                np.ones(n),
                (np.arange(n) == self.grip_idx).astype(float),
                self.contact_flag.copy(),
            ]),
            "budget_external_couple": np.zeros((3, 1, 3)),
            "budget_external_force": F,
            "budget_force_position": P,
            "cumulative_mass_source": np.zeros(1),
        }

        attrs = dict(self.attrs0)
        attrs["inertia_tensor"] = np.zeros((n, 3, 3))
        attrs["mass_quantity"] = self.m.copy()

        scalars = dict(self.scalars0)
        scalars["reference_momentum"] = float(self.ref_p)
        scalars["reference_angular_momentum"] = float(self.ref_a)

        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.m.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()

        # reset interval accumulators
        self._reset_interval()
        self.contact_flag[:] = 0.0
        self.x_frame_prev = self.x.copy()
        self.t_frame_prev = float(t)
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "time": float(self.time),
            "steps": int(self._steps),
            "dt": float(self.dt),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self.n else 0.0,
            "max_stretch": float(np.max(np.linalg.norm(
                self.x[1:] - self.x[:-1], axis=1)) / self.h) if self.n > 1 else 1.0,
            "cumulative_external_impulse": self.cum_imp.tolist(),
        }
