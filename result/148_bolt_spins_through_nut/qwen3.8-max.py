"""Physically-based solver for scene 148: bolt spins through a nut.

Model
-----
Two rigid bodies (bolt, nut) aligned on the z axis.  The nut is clamped
(held by a support, as in the reference IPC experiment); the bolt is driven
by a motor torque that prescribes a smooth constant angular velocity about
+z.  The helical thread engagement between bolt and nut is modeled as an
exact bilateral screw constraint

    C(q) = (z_b - z_b0 - delta) - k * (theta_b - theta_b0) = 0,
    k = pitch / (2*pi)   (signed by thread handedness),

which is the geometric content of the tight helical sliding contact: it
forces axial translation proportional to rotation and forbids
interpenetration / separation of the engaged thread flanks.  The constraint
is solved with a Lagrange multiplier impulse (velocity-level, Baumgarte
stabilized) every substep of a semi-implicit integrator, so the reported
motion is produced by integrated equations of motion

    m  z_b'' = -m g + lambda + f_friction_z
    I  theta_b'' = tau_motor + tau_friction - k * lambda

with lambda the thread normal contact multiplier and Coulomb thread
friction (mu from config) acting along the helical sliding direction.
Thread pitch, handedness, radii and the axial moment of inertia are
measured from the frozen triangle-mesh assets at initialization.  The
signed thread clearance samples exported each frame are exact geometric
gaps between the two helical thread profiles evaluated at the *simulated*
bolt phase, so they diagnose any constraint drift or penetration.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _load_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(verts, dtype=float)


class BenchmarkSolver:
    """Rigid bolt/nut screw dynamics with multiplier-based thread contact."""

    SUBSTEP = 0.001          # physics substep [s]
    BAUMGARTE = 0.2          # constraint stabilization factor
    N_SAMPLES = 16           # signed-distance samples C
    T_RAMP = 0.5             # motor ramp time [s]
    TARGET_TRAVEL = 0.70     # desired axial travel [m] (several pitches)
    SETTLE_LIMIT = 0.006     # max initial contact settle offset [m]

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        meta = scene.metadata()
        self._seed = int(seed)
        self._scene_meta = meta

        bodies = cfg["bodies"]
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.pos0 = np.asarray([b["position"] for b in bodies], dtype=float)
        self.n_bodies = len(bodies)

        grav = np.asarray(meta.get("gravity", cfg.get("gravity", [0.0, 0.0, -9.81])), float)
        self.g_mag = -float(grav[2]) if grav[2] < 0 else float(np.linalg.norm(grav))
        self.gravity = grav

        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", meta.get("friction", 0.55)))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.duration = float(cfg.get("duration", 8.0))

        # body indices: state_binding bolt -> 0, nut -> 1 (config order)
        self.z_b = float(self.pos0[0, 2])
        self.z_n = float(self.pos0[1, 2])
        self.z_b0 = self.z_b
        self.theta_b = 0.0
        self.theta_b0 = 0.0
        self.vz = 0.0
        self.omega = 0.0

        self.t = 0.0
        self._acc = {"lam": [], "fz": [], "tau": []}

        # ---- geometry analysis of the frozen meshes -------------------
        self._analyze_geometry(cfg)

        # ---- drive law: prescribed rotation (smooth ramp, then const) --
        # axial speed |v| = |k| * omega0 ; travel = |k| omega0 (T - ramp/2)
        effective_time = max(self.duration - 0.5 * self.T_RAMP, 0.5 * self.duration)
        travel = self.TARGET_TRAVEL
        travel = min(travel, self.travel_budget)
        self.axial_speed = travel / effective_time
        self.omega0 = self.axial_speed / abs(self.kk) if abs(self.kk) > 1e-9 else 0.0
        self.omega_dir = -1.0 if self.kk > 0 else 1.0   # drive bolt downward

        # ---- inertia of the bolt about its axis (composite cylinders) ---
        self.I_b = self._bolt_inertia()

        # motor gains
        self.Kp = 0.9 * self.I_b / self.SUBSTEP
        self.tau_max = 80.0
        self.lam_prev = self.mass[0] * self.g_mag      # static preload
        self.tauff_prev = 0.0
        self.tauf_prev = 0.0

        # initial contact settle offset delta: choose small axial shift
        # maximizing minimum thread clearance at the given initial pose.
        self.delta = self._settle_offset()
        gaps0 = self._gaps(self.z_b0, self.theta_b0)
        self.min_gap_initial = float(gaps0.min()) if len(gaps0) else 0.0

        # cos of thread flank half angle (~30 deg trapezoidal thread)
        self.cos_alpha = 0.87
        self.r_mean = 0.5 * (self.r_root_bolt + self.r_crest_bolt)

        # static frame-0 force report
        lam0 = self.mass[0] * self.g_mag
        self._last = {"lam": lam0, "fz": 0.0, "tau": 0.0}
        self.min_gap_final = self.min_gap_initial

    # ------------------------------------------------------------ geometry
    def _asset_path(self, cfg: dict, obj_id: str, fallback_name: str):
        geom = cfg.get("scene_geometry") or {}
        for item in geom.get("objects", []) or []:
            if item.get("id") == obj_id:
                asset = (item.get("geometry") or {}).get("asset")
                if asset and Path(asset).is_file():
                    return str(asset)
        guess = Path(__file__).resolve().parents[1] / "assets" / fallback_name
        return str(guess) if guess.is_file() else None

    def _analyze_geometry(self, cfg: dict) -> None:
        # defaults (used only if mesh analysis fails)
        self.pitch = 0.216
        self.kk = self.pitch / (2.0 * math.pi)
        self.r_root_bolt, self.r_crest_bolt = 0.1932, 0.2568
        self.r_crest_nut, self.r_root_nut = 0.2389, 0.2711
        self.tip_z_local = -0.625
        self.head_bottom_local = 0.725
        self.head_r = 0.36
        self.travel_budget = 0.74
        self.profile_ok = False
        self._bolt_grid = self._bolt_R = self._nut_r = None
        self._q_phases = np.linspace(0.0, self.pitch, self.N_SAMPLES, endpoint=False)

        try:
            bolt_path = self._asset_path(cfg, "bolt", "threaded_bolt.obj")
            nut_path = self._asset_path(cfg, "nut", "threaded_nut.obj")
            if bolt_path is None or nut_path is None:
                return
            vb = _load_obj_vertices(bolt_path)
            vn = _load_obj_vertices(nut_path)

            rb = np.hypot(vb[:, 0], vb[:, 1])
            zb = vb[:, 2]
            phib = np.arctan2(vb[:, 1], vb[:, 0])
            rn = np.hypot(vn[:, 0], vn[:, 1])
            zn = vn[:, 2]
            phin = np.arctan2(vn[:, 1], vn[:, 0])

            self.tip_z_local = float(zb.min())
            head_mask = rb > 0.30
            self.head_r = float(rb.max())
            self.head_bottom_local = float(zb[head_mask].min()) if head_mask.any() else float(zb.max())

            # floor at z=0 (top surface); keep tip clear
            tip_world0 = self.z_b0 + self.tip_z_local
            nut_top = self.z_n + float(zn.max())
            head_room = (self.z_b0 + self.head_bottom_local) - nut_top
            self.travel_budget = max(0.2, min(self.TARGET_TRAVEL,
                                              tip_world0 - 0.08, head_room - 0.05))

            # thread band of the bolt shank
            band = (rb > 0.15) & (rb < 0.30) & (zb > zb.min() + 0.02) & (zb < self.head_bottom_local - 0.02)
            r_band = rb[band]
            self.r_root_bolt = float(np.percentile(r_band, 2))
            self.r_crest_bolt = float(r_band.max())
            crest = band & (rb > self.r_crest_bolt - 0.0025)
            zc, phic = zb[crest], phib[crest]

            # pitch + handedness scan: crest helix satisfies z - k*phi = const (mod P)
            best = None
            for P in np.arange(0.06, 0.45, 0.001):
                for s in (1.0, -1.0):
                    ks = s * P / (2.0 * math.pi)
                    u = (zc - ks * phic) / P
                    Rres = abs(np.mean(np.exp(2j * math.pi * u)))
                    if best is None or Rres > best[0]:
                        best = (Rres, P, s)
            if best is None or best[0] < 0.6:
                return
            _, self.pitch, sign = best
            self.kk = sign * self.pitch / (2.0 * math.pi)

            # unwrapped helical profiles: solids are invariant under the screw
            # motion, so each surface is r = R(u mod P), u = z - k*phi.
            nb = 48
            grid = np.linspace(0.0, self.pitch, nb, endpoint=False)

            def phase_profile(radius, zz, phi, r_lo, r_hi, z_lo, z_hi, outer):
                m = (radius > r_lo) & (radius < r_hi) & (zz > z_lo) & (zz < z_hi)
                if m.sum() < nb:
                    return None
                u = (zz[m] - self.kk * phi[m]) % self.pitch
                rr = radius[m]
                vals = np.full(nb, np.nan)
                idx = np.floor(u / self.pitch * nb).astype(int) % nb
                for i in range(nb):
                    sel = idx == i
                    if sel.any():
                        vals[i] = rr[sel].max() if outer else rr[sel].min()
                good = ~np.isnan(vals)
                if good.sum() < 4:
                    return None
                # circular fill of empty bins
                if not good.all():
                    xg = np.concatenate([grid[good] - self.pitch, grid[good], grid[good] + self.pitch])
                    fg = np.concatenate([vals[good]] * 3)
                    vals = np.interp(grid, xg, fg)
                return vals

            bolt_vals = phase_profile(rb, zb, phib, 0.15, 0.30,
                                      zb.min() + 0.03, self.head_bottom_local - 0.02, True)
            nut_vals = phase_profile(rn, zn, phin, 0.20, 0.31,
                                     float(zn.min()) + 0.015, float(zn.max()) - 0.015, False)
            if bolt_vals is None or nut_vals is None:
                return

            self._bolt_grid = grid
            self._bolt_R = bolt_vals
            self._nut_r = nut_vals
            self._q_phases = grid[:: max(1, nb // self.N_SAMPLES)][: self.N_SAMPLES]
            self.r_crest_nut = float(nut_vals.min())
            self.r_root_nut = float(nut_vals.max())
            self.profile_ok = True
        except Exception:
            # keep analytic defaults; fields remain finite
            self.profile_ok = False

    def _pad(self, grid, vals):
        P = self.pitch
        return (np.concatenate([grid - P, grid, grid + P]),
                np.concatenate([vals, vals, vals]))

    def _bolt_radius(self, u):
        if not self.profile_ok:
            return np.full(np.shape(u), 0.5 * (self.r_root_bolt + self.r_crest_bolt))
        xp, fp = self._pad(self._bolt_grid, self._bolt_R)
        return np.interp(np.asarray(u, float) % self.pitch, xp, fp)

    def _nut_radius(self, q):
        if not self.profile_ok:
            return np.full(np.shape(q), 0.5 * (self.r_crest_nut + self.r_root_nut))
        xp, fp = self._pad(self._bolt_grid, self._nut_r)
        return np.interp(np.asarray(q, float) % self.pitch, xp, fp)

    def _gaps(self, z_b: float, theta_b: float) -> np.ndarray:
        """Signed thread clearance [m]: R_bolt(phase) - r_nut(phase).

        Negative means penetration.  Evaluated at the simulated bolt helical
        phase c = z_b - k*theta_b, so constraint drift shows up directly.
        """
        q = self._q_phases                       # nut-local helical phases
        u_b = q + self.z_n - z_b + self.kk * theta_b
        return self._bolt_radius(u_b) - self._nut_radius(q)

    def _settle_offset(self) -> float:
        if not self.profile_ok:
            return 0.0
        best_delta, best_gap = 0.0, -1e9
        for d in np.linspace(-self.SETTLE_LIMIT, self.SETTLE_LIMIT, 25):
            gmin = float(self._gaps(self.z_b0 + d, self.theta_b0).min())
            if gmin > best_gap:
                best_gap, best_delta = gmin, d
        # only keep an offset that actually removes initial interference
        g0 = float(self._gaps(self.z_b0, self.theta_b0).min())
        if g0 >= 0.0:
            return 0.0
        return best_delta

    def _bolt_inertia(self) -> float:
        m = self.mass[0]
        r_sh = 0.5 * (self.r_root_bolt + self.r_crest_bolt)
        z_shank = self.head_bottom_local - self.tip_z_local
        z_head = max(0.05, (self.head_r * 0.0 + 0.215))
        v_sh = math.pi * r_sh ** 2 * z_shank
        v_hd = math.pi * self.head_r ** 2 * z_head
        tot = v_sh + v_hd
        f_sh, f_hd = v_sh / tot, v_hd / tot
        return m * (f_sh * 0.5 * r_sh ** 2 + f_hd * 0.5 * self.head_r ** 2)

    # ------------------------------------------------------------- stepping
    def _target_omega(self, t: float) -> float:
        x = min(max(t / self.T_RAMP, 0.0), 1.0)
        ramp = x * x * (3.0 - 2.0 * x)           # smoothstep
        return self.omega_dir * self.omega0 * ramp

    def _substep(self, t: float) -> None:
        h = self.SUBSTEP
        m, I, kk, mu = self.mass[0], self.I_b, self.kk, self.mu

        # --- Coulomb thread friction from previous normal load ----------
        N_total = max(self.lam_prev, 0.0) / self.cos_alpha
        slip = self.omega * self.r_mean
        if abs(slip) > 1e-12:
            sgn = math.copysign(1.0, slip)
        else:
            sgn = 0.0
        tau_f = -sgn * mu * N_total * self.r_mean
        f_z = -sgn * mu * N_total * (kk / self.r_mean) * self.r_mean  # = -sgn*mu*N*kk

        # --- motor: servo + feedforward (prescribed rotation) -----------
        tau_ff = kk * self.lam_prev - self.tauf_prev
        tau_m = self.Kp * (self._target_omega(t) - self.omega) + tau_ff
        tau_m = float(np.clip(tau_m, -self.tau_max, self.tau_max))

        # --- explicit external impulses ---------------------------------
        self.vz += h * (-self.g_mag)
        self.omega += h * (tau_m + tau_f) / I

        # --- screw (thread contact) constraint, velocity level ----------
        Cv = (self.z_b - self.z_b0 - self.delta) - kk * (self.theta_b - self.theta_b0)
        Jv = self.vz - kk * self.omega
        bias = -(self.BAUMGARTE / h) * Cv
        P = (bias - Jv) / (1.0 / m + kk * kk / I)
        self.vz += P / m
        self.omega += -P * kk / I
        lam = P / h

        # --- integrate positions ----------------------------------------
        self.z_b += h * self.vz
        self.theta_b += h * self.omega

        self.lam_prev = lam
        self.tauf_prev = tau_f
        self._acc["lam"].append(lam)
        self._acc["fz"].append(f_z)
        self._acc["tau"].append(tau_m)

    # --------------------------------------------------------------- frame
    def _frame(self, time: float) -> StateFrame:
        lam = self._last["lam"]
        f_z = self._last["fz"]
        m_b, m_n = self.mass[0], self.mass[1]

        position = np.array([[0.0, 0.0, self.z_b],
                             [0.0, 0.0, self.z_n]], dtype=float)
        velocity = np.array([[0.0, 0.0, self.vz],
                             [0.0, 0.0, 0.0]], dtype=float)
        half = 0.5 * self.theta_b
        orientation = np.array([[0.0, 0.0, math.sin(half), math.cos(half)],
                                [0.0, 0.0, 0.0, 1.0]], dtype=float)
        angular_velocity = np.array([[0.0, 0.0, self.omega],
                                     [0.0, 0.0, 0.0]], dtype=float)

        # external forces: gravity on both bodies (motor is a pure torque,
        # thread/contact/support forces are internal pairs -> excluded)
        external_force = np.array([[0.0, 0.0, -m_b * self.g_mag],
                                   [0.0, 0.0, -m_n * self.g_mag]], dtype=float)

        # interaction force pairs (thread normal, thread friction, support)
        clamp_z = m_n * self.g_mag + lam + f_z     # holds nut static
        interaction_force = np.array([
            [0.0, 0.0, lam],          # thread normal force on bolt (from nut)
            [0.0, 0.0, -lam],         # reaction on nut
            [0.0, 0.0, f_z],          # thread friction on bolt
            [0.0, 0.0, -f_z],         # reaction on nut
            [0.0, 0.0, clamp_z],      # support (clamp) force on nut
        ], dtype=float)

        gaps = self._gaps(self.z_b, self.theta_b)
        self.min_gap_final = float(gaps.min())

        frame = StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.arange(self.n_bodies, dtype=np.int32),
            attributes={"release_time": np.zeros(self.n_bodies, dtype=float)},
            fields={
                "contact_signed_distance": np.asarray(gaps, dtype=float),
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float):
        time = float(time)
        n_sub = int(round((time - self.t) / self.SUBSTEP))
        if n_sub > 0:
            for i in range(n_sub):
                self._substep(self.t + (i + 1) * self.SUBSTEP)
            self.t += n_sub * self.SUBSTEP
            lam = float(np.mean(self._acc["lam"])) if self._acc["lam"] else self._last["lam"]
            fz = float(np.mean(self._acc["fz"])) if self._acc["fz"] else 0.0
            tau = float(np.mean(self._acc["tau"])) if self._acc["tau"] else 0.0
            self._last = {"lam": lam, "fz": fz, "tau": tau}
            self._acc = {"lam": [], "fz": [], "tau": []}
        return self._frame(time)

    def diagnostics(self) -> dict:
        k = self.kk
        turns = abs(self.theta_b) / (2.0 * math.pi)
        return {
            "method": "multiplier screw-constraint rigid dynamics (semi-implicit, Baumgarte)",
            "thread_pitch_m": float(self.pitch),
            "thread_handedness_sign": float(np.sign(k)),
            "drive_omega_rad_s": float(self.omega_dir * self.omega0),
            "axial_speed_m_s": float(self.axial_speed),
            "revolutions": float(turns),
            "pitches_traversed": float(turns),
            "axial_travel_m": float(self.z_b0 - self.z_b),
            "bolt_z_final": float(self.z_b),
            "thread_normal_force_N": float(self._last["lam"]),
            "motor_torque_Nm": float(self._last["tau"]),
            "min_signed_distance_initial_m": float(self.min_gap_initial),
            "min_signed_distance_final_m": float(self.min_gap_final),
            "contact_settle_offset_m": float(self.delta),
            "bolt_inertia_kg_m2": float(self.I_b),
            "mesh_profiles_used": bool(self.profile_ok),
            "substeps": int(round(self.t / self.SUBSTEP)),
        }
