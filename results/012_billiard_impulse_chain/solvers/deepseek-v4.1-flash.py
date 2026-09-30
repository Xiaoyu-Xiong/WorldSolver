"""Billiard impulse chain: penalty-contact rigid-sphere solver.

Physical model
--------------
Six rigid spheres (radius 0.2 m, mass 1 kg) rest on a static ground box whose
top face is the plane z = 0.  The incoming ball travels along +x and strikes the
near end of a line of five stationary balls separated by 1 cm gaps.  The contact
impulse propagates ball to ball; the far ball is launched and then, through
sliding friction with the cloth, spins up until it rolls, after which rolling
resistance slowly decays its motion.

Discretisation
--------------
Semi-implicit (symplectic) Euler with a fixed micro time step.  Contacts use a
Kelvin-Voigt penalty law (stiffness k, damping set by the target coefficient of
restitution).  Friction is an impulse-clamped Coulomb model: the tangential
impulse that would exactly cancel contact-point slip is applied when it lies
inside the Coulomb cone, otherwise sliding friction at the cone boundary is
applied.  Sphere inertia is that of a solid sphere (I = 2/5 m R^2); quaternions
carry orientation.  Ground rolling resistance applies the torque
mu_r * F_n * R opposing the spin.

Everything is deterministic: fixed sub-step count per requested sample interval,
no randomness, no precomputed motion tables.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, row-wise."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


class BenchmarkSolver:
    """Deterministic rigid-sphere contact solver for the billiard chain."""

    # -- public contract -------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        raw_config = getattr(scene, "config", None)
        config = dict(raw_config) if isinstance(raw_config, dict) else {}
        self._config = config

        self._gravity = np.asarray(
            config.get("gravity", (0.0, 0.0, -9.81)), dtype=float
        ).reshape(3)
        self._mu_roll = float(config.get("rolling_friction", 0.015))
        self._mu_slide = 0.2          # sphere / cloth sliding friction
        self._mu_ball = 0.06          # sphere / sphere sliding friction
        self._restitution = 0.93      # inter-ball restitution (energy loss)
        self._stiffness = 1.0e6       # penalty contact stiffness [N/m]
        e = float(self._restitution)
        self._zeta_ball = float(-np.log(e) / np.sqrt(np.pi * np.pi + np.log(e) ** 2))
        self._zeta_ground = 1.0       # critically damped resting contact

        self._substeps = 300
        sample_dt = float(config.get("sample_dt", 1.0 / 60.0))
        if not np.isfinite(sample_dt) or sample_dt <= 0.0:
            sample_dt = 1.0 / 60.0
        self._dt_base = sample_dt / self._substeps
        self._last_dt = self._dt_base

        ground_top = 0.0
        ground_hx = np.inf
        ground_hy = np.inf
        radii: dict[int, float] = {}
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            geom = obj.get("geometry", {}) or {}
            if obj.get("semantic") == "boundary" or obj.get("id") == "ground":
                half = geom.get("half_extents")
                trans = (obj.get("transform") or {}).get("translation", (0.0, 0.0, 0.0))
                if half is not None and len(half) == 3:
                    ground_top = float(trans[2]) + float(half[2])
                    ground_hx = float(half[0])
                    ground_hy = float(half[1])
            if geom.get("type") == "sphere":
                binding = obj.get("state_binding") or {}
                if binding.get("selector") == "body_id" and binding.get("value") is not None:
                    radii[int(binding["value"])] = float(geom.get("radius", 0.2))
        self._ground_top = ground_top
        self._ground_hx = ground_hx
        self._ground_hy = ground_hy

        state = None
        try:
            state = scene.initial_state()
        except Exception:
            state = None

        if state is not None:
            try:
                position = np.array(state.position, dtype=float)
                velocity = np.array(state.velocity, dtype=float)
                n = position.shape[0]
                if position.shape != (n, 3) or velocity.shape != (n, 3):
                    raise ValueError("bad initial state shape")
                mass = (
                    np.array(state.mass, dtype=float)
                    if getattr(state, "mass", None) is not None
                    else np.ones(n)
                )
                body_id = (
                    np.array(state.body_id, dtype=int)
                    if getattr(state, "body_id", None) is not None
                    else np.arange(n)
                )
                orientation = (
                    np.array(state.orientation, dtype=float)
                    if getattr(state, "orientation", None) is not None
                    else np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
                )
                angular = (
                    np.array(state.angular_velocity, dtype=float)
                    if getattr(state, "angular_velocity", None) is not None
                    else np.zeros((n, 3))
                )
                t0 = float(getattr(state, "time", 0.0) or 0.0)
            except Exception:
                state = None

        if state is None:
            entries = []
            for obj in geometry.get("objects", []) or []:
                if obj.get("motion") != "dynamic":
                    continue
                binding = obj.get("state_binding") or {}
                value = binding.get("value")
                key = int(value) if value is not None else len(entries)
                trans = (obj.get("transform") or {}).get("translation", (0.0, 0.0, 0.0))
                entries.append((key, np.asarray(trans, dtype=float)))
            entries.sort(key=lambda item: item[0])
            n = len(entries)
            position = (
                np.vstack([item[1] for item in entries]) if n else np.zeros((0, 3))
            )
            velocity = np.zeros((n, 3))
            mass = np.ones(n)
            body_id = np.array([item[0] for item in entries], dtype=int)
            orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
            angular = np.zeros((n, 3))
            t0 = 0.0

        n = position.shape[0]
        self._n = n
        self._P = np.ascontiguousarray(position, dtype=float)
        self._V = np.ascontiguousarray(velocity, dtype=float)
        self._M = np.ascontiguousarray(mass, dtype=float)
        self._gravity = np.asarray(self._gravity, dtype=float).reshape(3)
        self._body_id = np.asarray(body_id, dtype=int).reshape(n)
        self._R = np.array(
            [float(radii.get(int(b), 0.2)) for b in self._body_id], dtype=float
        ).reshape(n)
        self._I = 0.4 * self._M * self._R * self._R
        q = np.asarray(orientation, dtype=float).reshape(n, 4)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        self._Q = q / np.maximum(norms, 1e-12)
        w = np.asarray(angular, dtype=float).reshape(n, 3)
        zero = np.zeros((n, 3))
        w = np.where(np.isfinite(w), w, zero)
        self._W = w
        self._t = float(t0) if np.isfinite(t0) and t0 >= 0.0 else 0.0
        self._step_count = 0

        i_idx, j_idx = np.triu_indices(n, 1)
        self._pi = i_idx.astype(np.int64)
        self._pj = j_idx.astype(np.int64)

    def advance_to(self, time: float):
        t = float(time)
        if not np.isfinite(t):
            t = self._t
        if t > self._t:
            remaining = t - self._t
            n_sub = int(round(remaining / self._dt_base))
            n_sub = max(1, min(n_sub, 2000000))
            dt = remaining / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self._t = t
        return self._frame(self._t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "penalty_contact_semi_implicit_euler",
            "bodies": int(self._n),
            "substeps_per_sample": int(self._substeps),
            "micro_dt": float(self._dt_base),
            "steps_taken": int(self._step_count),
            "stiffness": float(self._stiffness),
            "sliding_friction": float(self._mu_slide),
            "ball_friction": float(self._mu_ball),
            "rolling_friction": float(self._mu_roll),
            "restitution": float(self._restitution),
            "gravity": self._gravity.tolist(),
        }

    # -- numerics --------------------------------------------------------
    def _scatter(self, acc: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
        for d in range(3):
            acc[:, d] += np.bincount(idx, weights=vals[:, d], minlength=self._n)

    def _contacts(self, dt: float):
        """Penalty normal forces plus impulse-clamped Coulomb friction."""
        P = self._P
        V = self._V
        W = self._W
        R = self._R
        I = self._I
        M = self._M
        n_bodies = self._n
        force = np.zeros((n_bodies, 3))
        torque = np.zeros((n_bodies, 3))

        # --- sphere / sphere pairs ---
        ii = self._pi
        jj = self._pj
        rel = P[jj] - P[ii]
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        inv_dist = 1.0 / np.maximum(dist, 1e-12)
        normal = rel * inv_dist[:, None]
        Ri = R[ii]
        Rj = R[jj]
        gap_ss = dist - (Ri + Rj)
        pen = np.maximum(0.0, -gap_ss)
        r_i = normal * Ri[:, None]
        r_j = -normal * Rj[:, None]
        vc_i = V[ii] + np.cross(W[ii], r_i)
        vc_j = V[jj] + np.cross(W[jj], r_j)
        v_rel = vc_i - vc_j
        v_n = np.einsum("ij,ij->i", v_rel, normal)
        m_eff_n = 1.0 / (1.0 / M[ii] + 1.0 / M[jj])
        c_n = 2.0 * self._zeta_ball * np.sqrt(self._stiffness * m_eff_n)
        f_n = self._stiffness * pen - c_n * v_n
        f_n = np.where(pen > 0.0, np.maximum(f_n, 0.0), 0.0)

        v_t = v_rel - v_n[:, None] * normal
        v_t_mag = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        t_dir = np.zeros_like(v_t)
        moving = v_t_mag > 1e-12
        t_dir[moving] = v_t[moving] / v_t_mag[moving, None]
        inv_m_t = 1.0 / M[ii] + 1.0 / M[jj] + Ri * Ri / I[ii] + Rj * Rj / I[jj]
        m_eff_t = 1.0 / inv_m_t
        j_n = f_n * dt
        j_stop = m_eff_t * v_t_mag
        f_t_mag = np.where(
            j_stop <= self._mu_ball * j_n,
            m_eff_t * v_t_mag / dt,
            self._mu_ball * f_n,
        )
        f_t = -f_t_mag[:, None] * t_dir
        pair_ss = f_n[:, None] * normal + f_t

        self._scatter(force, jj, pair_ss)
        self._scatter(force, ii, -pair_ss)
        self._scatter(torque, jj, np.cross(r_j, pair_ss))
        self._scatter(torque, ii, np.cross(r_i, -pair_ss))

        # --- sphere / ground contacts (top face of the ground box) ---
        height = P[:, 2] - self._ground_top
        gap_g = height - R
        inside = (np.abs(P[:, 0]) <= self._ground_hx) & (
            np.abs(P[:, 1]) <= self._ground_hy
        )
        pen_g = np.maximum(0.0, -gap_g)
        up = np.zeros((n_bodies, 3))
        up[:, 2] = 1.0
        r_g = -R[:, None] * up
        vc_g = V + np.cross(W, r_g)
        v_n_g = vc_g[:, 2]
        c_g = 2.0 * self._zeta_ground * np.sqrt(self._stiffness * M)
        f_n_g = np.where(
            inside & (pen_g > 0.0),
            np.maximum(self._stiffness * pen_g - c_g * v_n_g, 0.0),
            0.0,
        )
        v_t_g = vc_g - v_n_g[:, None] * up
        v_t_g_mag = np.sqrt(np.einsum("ij,ij->i", v_t_g, v_t_g))
        t_dir_g = np.zeros_like(v_t_g)
        moving_g = v_t_g_mag > 1e-12
        t_dir_g[moving_g] = v_t_g[moving_g] / v_t_g_mag[moving_g, None]
        inv_m_t_g = 1.0 / M + R * R / I
        m_eff_t_g = 1.0 / inv_m_t_g
        j_n_g = f_n_g * dt
        j_stop_g = m_eff_t_g * v_t_g_mag
        f_t_g_mag = np.where(
            j_stop_g <= self._mu_slide * j_n_g,
            m_eff_t_g * v_t_g_mag / dt,
            self._mu_slide * f_n_g,
        )
        f_t_g = -f_t_g_mag[:, None] * t_dir_g
        pair_g = f_n_g[:, None] * up + f_t_g

        force += pair_g
        torque += np.cross(r_g, pair_g)

        # rolling resistance: torque opposing spin, magnitude mu_r * F_n * R
        w_mag = np.sqrt(np.einsum("ij,ij->i", W, W))
        spin = w_mag > 1e-9
        tau_scale = np.zeros(n_bodies)
        tau_scale[spin] = -self._mu_roll * f_n_g[spin] * R[spin] / w_mag[spin]
        rolling = tau_scale[:, None] * W
        rolling[f_n_g <= 0.0] = 0.0
        torque += rolling

        gaps = np.concatenate([gap_ss, gap_g])
        pair_forces = np.vstack([pair_ss, pair_g])
        return force, torque, gaps, pair_forces

    def _step(self, dt: float) -> None:
        force, torque, _gaps, _pairs = self._contacts(dt)
        self._V = self._V + dt * (force / self._M[:, None] + self._gravity[None, :])
        self._P = self._P + dt * self._V
        self._W = self._W + dt * (torque / self._I[:, None])
        omega = np.column_stack((self._W, np.zeros(self._n)))
        q = self._Q + (0.5 * dt) * _quat_mul(omega, self._Q)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        self._Q = q / np.maximum(norms, 1e-12)
        self._last_dt = dt
        self._step_count += 1

    def _frame(self, time: float) -> StateFrame:
        _force, _torque, gaps, pair_forces = self._contacts(self._last_dt)
        external = self._M[:, None] * self._gravity[None, :]
        return StateFrame(
            time=float(time),
            position=self._P.copy(),
            velocity=self._V.copy(),
            orientation=self._Q.copy(),
            angular_velocity=self._W.copy(),
            mass=self._M.copy(),
            body_id=self._body_id.copy(),
            fields={
                "contact_signed_distance": np.asarray(gaps, dtype=float),
                "external_force": np.asarray(external, dtype=float),
                "interaction_force": np.asarray(pair_forces, dtype=float),
            },
            scalars={"contact_length_scale": float(np.mean(self._R))},
        )
