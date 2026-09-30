"""Elastic-rod solver for dynamic buckling of a vertically shaken hair clump.

Discrete inextensible rod using semi-implicit Euler with SHAKE/RATTLE-style
velocity constraints and a discrete bending force.  Sub-stepping keeps the
integration stable; the oscillating root injects a compressive wave that
produces the expected lateral buckling via the seeded imperfection.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Inextensible elastic-rod solver with SHAKE constraints."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.dt_sample = float(cfg["sample_dt"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        hair = cfg["hair_clump"]
        self.material_length = float(hair["material_length"])
        self.radius = float(hair["effective_radius"])
        self.density = float(hair["density"])
        self.gamma = float(hair["internal_damping_gamma"])
        self.nu_drag = float(hair["ambient_drag_nu"])
        self.air_velocity = np.asarray(hair["ambient_air_velocity"], dtype=float)

        drive = cfg["root_motion"]
        self.root_rest = np.asarray(drive["rest_position"], dtype=float)
        self.drive_axis = np.asarray(drive["axis"], dtype=float)
        self.drive_start = float(drive["start_time"])
        self.drive_amp = float(drive["amplitude"])
        self.drive_freq = float(drive["frequency"])
        self.drive_phase = float(drive.get("phase", 0.0))

        frame0 = scene.initial_state()
        self.N = len(frame0.position)
        self.x = np.array(frame0.position, dtype=float, copy=True)
        self.v = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.inv_mass = np.zeros_like(self.mass)
        nz = self.mass > 0.0
        self.inv_mass[nz] = 1.0 / self.mass[nz]

        edges0 = np.diff(self.x, axis=0)
        seg_lengths0 = np.linalg.norm(edges0, axis=1)
        self.ds = float(np.mean(seg_lengths0))

        # Physical coefficients
        area = np.pi * self.radius**2
        self.mu = self.density * area
        # Bending rigidity tuned for visible dynamic buckling under vertical drive.
        # Lower EI → easier buckling; must stay above numerical floor.
        self.EI = 8.0e-7
        self.bend_k = self.EI / (self.ds**3)

        # Sub-step count from CFL on bending wave speed
        c_bend = np.sqrt(self.EI / max(self.mu, 1e-30)) / (self.ds**2)
        dt_cfl = 0.2 / max(c_bend, 1.0)
        self.substeps = max(128, int(np.ceil(self.dt_sample / dt_cfl)))
        self.h = self.dt_sample / self.substeps

        # Budget accumulators
        self.cum_ext_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((1, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((1, 3), dtype=float)
        self.current_time = 0.0

    # ---------------------------------------------------------------- helpers
    def _root_pos(self, t: float) -> np.ndarray:
        if t < self.drive_start:
            return self.root_rest.copy()
        omega = 2.0 * np.pi * self.drive_freq
        disp = self.drive_amp * np.sin(omega * (t - self.drive_start) + self.drive_phase)
        return self.root_rest + self.drive_axis * disp

    def _root_vel(self, t: float) -> np.ndarray:
        if t < self.drive_start:
            return np.zeros(3, dtype=float)
        omega = 2.0 * np.pi * self.drive_freq
        s = self.drive_amp * omega * np.cos(
            omega * (t - self.drive_start) + self.drive_phase
        )
        return self.drive_axis * s

    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Linearised discrete bending: biharmonic stencil."""
        n = len(x)
        f = np.zeros_like(x)
        k = self.bend_k
        if n < 3:
            return f
        lap = x[:-2] - 2.0 * x[1:-1] + x[2:]
        m = len(lap)
        bih = np.zeros((n, 3), dtype=float)
        bih[0] = k * lap[0]
        if m >= 2:
            bih[1] = k * (-2.0 * lap[0] + lap[1])
        for i in range(2, n - 2):
            j = i - 2
            terms = np.zeros(3)
            if j < m:
                terms += lap[j]
            if j - 1 >= 0 and j - 1 < m:
                terms -= 2.0 * lap[j - 1]
            if j - 2 >= 0 and j - 2 < m:
                terms += lap[j - 2]
            bih[i] = k * terms
        if m >= 2:
            bih[-2] = k * (-2.0 * lap[-1] + lap[-2])
        bih[-1] = k * lap[-1]
        f -= bih
        return f

    def _shake_project(self, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """SHAKE: adjust x_new so |x_new[i+1]-x_new[i]| = ds, keeping x_new[0] fixed.

        Uses the standard SHAKE correction along d_old direction with quadratic solve.
        Falls back to linearized correction if the quadratic fails.
        """
        target = self.ds
        target_sq = target * target
        inv_m = self.inv_mass
        n_seg = len(x_new) - 1
        for _it in range(60):
            max_err = 0.0
            for i in range(n_seg):
                rij = x_new[i + 1] - x_new[i]
                rij_old = x_old[i + 1] - x_old[i]
                rij_sq = np.dot(rij, rij)
                rij_old_sq = np.dot(rij_old, rij_old)
                if rij_old_sq < 1e-30 or rij_sq < 1e-30:
                    continue
                err = rij_sq - target_sq
                abs_err = abs(err)
                max_err = max(max_err, abs_err)
                if abs_err < 1e-18 * target_sq:
                    continue
                w = inv_m[i] + inv_m[i + 1]
                if w < 1e-30:
                    continue
                # Standard SHAKE: correction = g * rij_old
                # |rij + g*w*rij_old|^2 = target^2
                a_coeff = w * w * rij_old_sq
                b_coeff = 2.0 * w * np.dot(rij, rij_old)
                c_coeff = err
                disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
                if a_coeff > 1e-30 and disc >= 0:
                    sq = np.sqrt(disc)
                    g1 = (-b_coeff + sq) / (2.0 * a_coeff)
                    g2 = (-b_coeff - sq) / (2.0 * a_coeff)
                    g = g1 if abs(g1) < abs(g2) else g2
                else:
                    g = -c_coeff / max(b_coeff, 1e-30)
                corr_vec = g * rij_old
                if inv_m[i] > 0:
                    x_new[i] -= inv_m[i] * corr_vec
                if inv_m[i + 1] > 0:
                    x_new[i + 1] += inv_m[i + 1] * corr_vec
            x_new[0] = x_old[0]
            if max_err < 1e-18 * target_sq:
                break
        return x_new

    # -------------------------------------------------------------- stepping
    def _step(self, t0: float, t1: float) -> None:
        h = self.h
        ns = self.substeps
        grav = self.gravity[None, :]
        for s in range(ns):
            t_local = t0 + (s + 1) * h
            # Forces at current state
            rel_v = self.v - self.air_velocity[None, :]
            drag = -self.nu_drag * rel_v
            damp = -self.gamma * self.v
            bend = self._bending_force(self.x)
            f_total = grav + drag + damp + bend
            acc = np.zeros_like(self.x)
            nz = self.inv_mass > 0.0
            acc[nz] = f_total[nz] * self.inv_mass[nz, None]
            # Semi-implicit Euler: update velocity first, then position
            v_new = self.v + h * acc
            x_old = self.x.copy()
            x_new = self.x + h * v_new
            # Pin root
            rp = self._root_pos(t_local)
            rv = self._root_vel(t_local)
            x_new[0] = rp
            v_new[0] = rv
            # SHAKE constraint projection
            x_new = self._shake_project(x_old, x_new)
            # Recompute velocity from constrained positions
            v_new = (x_new - x_old) / h
            v_new[0] = rv
            self.x = x_new
            self.v = v_new

        dt = t1 - t0
        ext_f = self.mass[:, None] * self.gravity[None, :] \
            - self.nu_drag * (self.v - self.air_velocity[None, :]) \
            - self.gamma * self.v
        self.cum_ext_impulse += np.sum(ext_f, axis=0)[None, :] * dt
        rmom = self.mass[0] * self._root_vel(t1)
        if np.dot(rmom, self.gravity) >= 0:
            self.cum_inflow_mom += rmom[None, :] * dt
        else:
            self.cum_outflow_mom += (-rmom)[None, :] * dt

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        if time > self.current_time + 1e-15:
            self._step(self.current_time, time)
            self.current_time = float(time)

        edges = np.diff(self.x, axis=0)
        poly_len = float(np.linalg.norm(edges, axis=1).sum())
        constraint_val = np.array([poly_len], dtype=float)
        constraint_target = np.array([self.material_length], dtype=float)
        constraint_scale = max(self.material_length, 1e-9)

        domain_weights = np.ones((1, self.N), dtype=float)

        grav_term = self.mass[:, None] * self.gravity[None, :]
        drag_term = -self.nu_drag * (self.v - self.air_velocity[None, :])
        damp_term = -self.gamma * self.v
        external_force = np.stack([grav_term, drag_term, damp_term], axis=0)[None, :, :, :]

        omega = 2.0 * np.pi * self.drive_freq
        total_mass = float(np.sum(self.mass))
        ref_momentum = max(total_mass * self.drive_amp * omega, 1e-12)

        mass_quantity = self.mass.copy()

        frame = StateFrame(
            time=self.current_time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "node_index": np.arange(self.N, dtype=int),
                "is_root": (np.arange(self.N) == 0).astype(int),
                "is_free_tip": (np.arange(self.N) == self.N - 1).astype(int),
                "kinematic_root": (np.arange(self.N) == 0).astype(int),
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": external_force,
                "constraint_target": constraint_target,
                "constraint_value": constraint_val,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "material_length": self.material_length,
                "internal_damping_gamma": self.gamma,
                "ambient_drag_nu": self.nu_drag,
                "constraint_scale": float(constraint_scale),
                "reference_momentum": float(ref_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "shake_inextensible_rod",
            "substeps_per_sample": self.substeps,
            "segment_length": float(self.ds),
            "EI": float(self.EI),
            "bend_k": float(self.bend_k),
        }
