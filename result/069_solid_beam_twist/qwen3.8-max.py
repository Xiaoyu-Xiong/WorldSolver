"""Physically based solver for scene 069_solid_beam_twist.

Model
-----
A three-dimensional solid beam discretised by the frozen structured hexahedral
node grid (49 x 7 x 7 = 2401 material samples, 48 x 6 x 6 = 1728 eight-node
brick elements).  Each element is integrated with the full 2x2x2 Gauss rule
and a compressible Neo-Hookean hyperelastic law

    W(F) = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2 ,   J = det F

so that large end rotations are handled without artificial stiffening.  The
two end face bands are kinematically driven: every sample in a band follows a
rigid rotation about the frozen rotation axis through its end centre, with the
angle schedule (smoothstep interpolated) read from the scene configuration.
The remaining samples obey momentum balance

    m_i a_i = f_int_i + f_damp_i + f_reac_i + m_i g

integrated with a symplectic-Euler scheme using sub-critical time steps.  A
mass proportional (visco-elastic drag) term keeps the quasi-static twisting
path smooth instead of ringing; it is exported as an external force term so
that the discrete momentum budget closes exactly for every budget domain.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - platform provides the contract package
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None


class BenchmarkSolver:
    """Distributed twist of a volumetric elastic solid beam."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        beam = cfg["beam"]
        self.nx, self.ny, self.nz = (int(v) for v in beam["volumetric_grid_shape"])
        self.length, self.width, self.height = (float(v) for v in beam["dimensions"])
        self.center = np.asarray(beam["center"], dtype=float)
        self.rho = float(beam["density"])
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        bc = cfg["boundary_conditions"]
        self.band = float(bc["end_face_band"])
        axis = np.asarray(bc["rotation_axis"], dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.interpolation = str(bc.get("interpolation", "smoothstep"))
        self.left_center = np.asarray(bc["left_end"]["center"], dtype=float)
        self.right_center = np.asarray(bc["right_end"]["center"], dtype=float)
        self.left_schedule = [
            (float(p["time"]), float(p["angle"]))
            for p in bc["left_end"]["prescribed_rotation_degrees"]
        ]
        self.right_schedule = [
            (float(p["time"]), float(p["angle"]))
            for p in bc["right_end"]["prescribed_rotation_degrees"]
        ]
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # ---- material (soft rubber-like solid; displacement driven problem)
        young = 4.0e4
        poisson = 0.30
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.damping_coeff = 2.5  # 1/s, mass proportional (visco-elastic drag)

        # ---- frozen initial state
        frame0 = scene.initial_state()
        self.rest = np.array(frame0.attributes["rest_position"], dtype=float)
        self.end_region = np.array(frame0.attributes["end_region"], dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.n_nodes = int(self.rest.shape[0])
        self.position = self.rest.copy()
        self.velocity = np.zeros_like(self.rest)
        self.time = 0.0

        self.left_nodes = np.flatnonzero(self.end_region < 0.0)
        self.right_nodes = np.flatnonzero(self.end_region > 0.0)
        self.free_nodes = np.flatnonzero(self.end_region == 0.0)
        self.left_r = self.rest[self.left_nodes] - self.left_center
        self.right_r = self.rest[self.right_nodes] - self.right_center

        self._build_grid()
        self._build_budget_domains()

        # ---- stable explicit sub step (CFL on the smallest element edge)
        h_min = min(
            self.length / (self.nx - 1),
            self.width / (self.ny - 1),
            self.height / (self.nz - 1),
        )
        c_p = np.sqrt((self.lam + 2.0 * self.mu) / self.rho)
        dt_crit = 0.35 * h_min / c_p
        self.dt_sub = float(min(2.0e-3, dt_crit))

        # ---- cumulative budget accumulators
        self.total_mass = float(self.mass.sum())
        self.cum_impulse = np.zeros((self.n_domains, 3))
        self.cum_inflow = np.zeros((self.n_domains, 3))
        self.cum_outflow = np.zeros((self.n_domains, 3))
        self.cum_mass_source = np.zeros(1)

        self.f_int = np.zeros((self.n_nodes, 3))
        self.f_damp = np.zeros((self.n_nodes, 3))
        self.f_reac = np.zeros((self.n_nodes, 3))
        self.Fbar = np.tile(np.eye(3), (self.n_elem, 1, 1))

        # ---- frozen reference momentum scale (never the initial net momentum)
        omega_max = max(self._max_rate(self.left_schedule), self._max_rate(self.right_schedule))
        r_max = 0.5 * np.hypot(self.width, self.height)
        self.reference_momentum = float(max(self.total_mass * omega_max * r_max, 1.0e-6))

    # ------------------------------------------------------------- topology
    def _build_grid(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        hx = self.length / (nx - 1)
        hy = self.width / (ny - 1)
        hz = self.height / (nz - 1)
        self.h = np.array([hx, hy, hz])
        self.v_gauss = hx * hy * hz / 8.0  # volume weight per Gauss point

        ei, ej, ek = np.meshgrid(
            np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
        )
        ei = ei.ravel()
        ej = ej.ravel()
        ek = ek.ravel()
        self.n_elem = int(ei.size)
        # local corner offsets matching the trilinear corner ordering below
        offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=int,
        )
        base = (ei * ny + ej) * nz + ek
        self.elem_nodes = (
            base[:, None]
            + (offsets[None, :, 0] * ny + offsets[None, :, 1]) * nz
            + offsets[None, :, 2]
        ).astype(np.int64)
        self.elem_flat = self.elem_nodes.ravel()

        # Gauss points (2x2x2) and trilinear shape-function material gradients
        q = 1.0 / np.sqrt(3.0)
        gauss = np.array(
            [
                [-q, -q, -q],
                [q, -q, -q],
                [q, q, -q],
                [-q, q, -q],
                [-q, -q, q],
                [q, -q, q],
                [q, q, q],
                [-q, q, q],
            ]
        )
        corners = 2.0 * offsets - 1.0
        dNdxi = np.zeros((8, 8, 3))
        for g in range(8):
            xi = gauss[g]
            for l in range(8):
                c = corners[l]
                dNdxi[g, l, 0] = 0.125 * c[0] * (1.0 + xi[1] * c[1]) * (1.0 + xi[2] * c[2])
                dNdxi[g, l, 1] = 0.125 * (1.0 + xi[0] * c[0]) * c[1] * (1.0 + xi[2] * c[2])
                dNdxi[g, l, 2] = 0.125 * (1.0 + xi[0] * c[0]) * (1.0 + xi[1] * c[1]) * c[2]
        self.G = dNdxi * (2.0 / self.h)[None, None, :]  # (8gp, 8node, 3)
        self.Gt = np.ascontiguousarray(self.G.transpose(0, 2, 1).reshape(24, 8))
        self.GtT = np.ascontiguousarray(self.Gt.T)
        self.Fbuf = np.zeros((self.n_elem, 8, 3, 3))

    def _build_budget_domains(self) -> None:
        n = self.n_nodes
        w = np.zeros((4, n))
        w[0, :] = 1.0
        w[1, self.left_nodes] = 1.0
        w[2, self.right_nodes] = 1.0
        w[3, self.free_nodes] = 1.0
        self.domain_weights = w
        self.n_domains = 4
        self.n_force_terms = 4  # reactions, gravity, viscous drag, boundary tractions
        self.rows = [
            np.arange(n),
            self.left_nodes,
            self.right_nodes,
            self.free_nodes,
        ]

    # ------------------------------------------------------- loading schedule
    @staticmethod
    def _schedule_value(schedule, t: float, mode: str):
        times = [p[0] for p in schedule]
        angles = [p[1] for p in schedule]
        if t <= times[0]:
            return angles[0], 0.0
        if t >= times[-1]:
            return angles[-1], 0.0
        for k in range(len(times) - 1):
            if times[k] <= t <= times[k + 1]:
                t0, t1 = times[k], times[k + 1]
                a0, a1 = angles[k], angles[k + 1]
                span = t1 - t0
                tau = (t - t0) / span
                if mode == "linear":
                    frac, dfrac = tau, 1.0
                else:  # smoothstep
                    frac = tau * tau * (3.0 - 2.0 * tau)
                    dfrac = 6.0 * tau * (1.0 - tau)
                return a0 + (a1 - a0) * frac, (a1 - a0) * dfrac / span
        return angles[-1], 0.0

    def _max_rate(self, schedule) -> float:
        """Peak |d(theta)/dt| of a schedule in rad/s (smoothstep peaks at 1.5x)."""
        rates = []
        peak = 1.0 if self.interpolation == "linear" else 1.5
        for k in range(len(schedule) - 1):
            span = schedule[k + 1][0] - schedule[k][0]
            if span > 0.0:
                delta = schedule[k + 1][1] - schedule[k][1]
                rates.append(abs(delta) * peak / span)
        return float(max(rates) if rates else 0.0) * (np.pi / 180.0)

    def _end_kinematics(self, t: float):
        a_l, w_l = self._schedule_value(self.left_schedule, t, self.interpolation)
        a_r, w_r = self._schedule_value(self.right_schedule, t, self.interpolation)
        return (
            np.deg2rad(a_l),
            np.deg2rad(w_l),
            np.deg2rad(a_r),
            np.deg2rad(w_r),
            a_r - a_l,
        )

    def _rotate(self, r: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 1.0e-14:
            return r.copy()
        n = self.axis
        par = np.outer(r @ n, n)
        perp = r - par
        cross = np.cross(n, perp)
        return par + np.cos(angle) * perp + np.sin(angle) * cross

    def _prescribed_state(self, t: float):
        th_l, om_l, th_r, om_r, rel = self._end_kinematics(t)
        pos = self.position.copy()
        vel = self.velocity.copy()
        pl = self.left_center + self._rotate(self.left_r, th_l)
        pr = self.right_center + self._rotate(self.right_r, th_r)
        pos[self.left_nodes] = pl
        pos[self.right_nodes] = pr
        vel[self.left_nodes] = om_l * np.cross(self.axis, pl - self.left_center)
        vel[self.right_nodes] = om_r * np.cross(self.axis, pr - self.right_center)
        return pos, vel, rel

    # ------------------------------------------------------------- elasticity
    def _internal_forces(self, pos: np.ndarray) -> np.ndarray:
        xe = pos[self.elem_nodes]                                   # (Ne,8,3)
        F = self.Fbuf
        np.matmul(xe.transpose(0, 2, 1)[:, None, :, :], self.G[None], out=F)
        f00 = F[..., 0, 0]; f01 = F[..., 0, 1]; f02 = F[..., 0, 2]
        f10 = F[..., 1, 0]; f11 = F[..., 1, 1]; f12 = F[..., 1, 2]
        f20 = F[..., 2, 0]; f21 = F[..., 2, 1]; f22 = F[..., 2, 2]
        det = (
            f00 * (f11 * f22 - f12 * f21)
            - f01 * (f10 * f22 - f12 * f20)
            + f02 * (f10 * f21 - f11 * f20)
        )
        det_safe = np.where(np.abs(det) < 1.0e-6, 1.0e-6, det)
        inv = np.empty_like(F)
        inv[..., 0, 0] = (f11 * f22 - f12 * f21) / det_safe
        inv[..., 0, 1] = (f02 * f21 - f01 * f22) / det_safe
        inv[..., 0, 2] = (f01 * f12 - f02 * f11) / det_safe
        inv[..., 1, 0] = (f12 * f20 - f10 * f22) / det_safe
        inv[..., 1, 1] = (f00 * f22 - f02 * f20) / det_safe
        inv[..., 1, 2] = (f02 * f10 - f00 * f12) / det_safe
        inv[..., 2, 0] = (f10 * f21 - f11 * f20) / det_safe
        inv[..., 2, 1] = (f01 * f20 - f00 * f21) / det_safe
        inv[..., 2, 2] = (f00 * f11 - f01 * f10) / det_safe
        logJ = np.log(np.maximum(det_safe, 1.0e-9))
        invT = np.swapaxes(inv, -1, -2)
        P = self.mu * (F - invT) + (self.lam * logJ)[..., None, None] * invT
        pk = np.ascontiguousarray(P.transpose(0, 1, 3, 2).reshape(self.n_elem, 24, 3))
        fe = -self.v_gauss * np.matmul(self.GtT, pk)               # (Ne,8,3)
        f = np.zeros((self.n_nodes, 3))
        flat = fe.reshape(-1, 3)
        for c in range(3):
            f[:, c] = np.bincount(self.elem_flat, weights=flat[:, c], minlength=self.n_nodes)
        self.Fbar = F.mean(axis=1)
        self._last_det = det
        return f

    # --------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        t0 = self.time
        t1 = t0 + dt
        f_int = self._internal_forces(self.position)
        f_damp = -self.damping_coeff * self.mass[:, None] * self.velocity
        f_grav = self.mass[:, None] * self.gravity[None, :]

        acc = (f_int + f_damp + f_grav) / self.mass[:, None]
        vel_new = self.velocity + dt * acc
        pos_new = self.position + dt * vel_new

        # kinematically driven end bands
        pres_pos, pres_vel, _ = self._prescribed_state(t1)
        driven = np.concatenate([self.left_nodes, self.right_nodes])
        reac = np.zeros_like(f_int)
        dv = pres_vel[driven] - self.velocity[driven]
        reac[driven] = (
            self.mass[driven, None] * dv / dt
            - f_int[driven]
            - f_damp[driven]
            - f_grav[driven]
        )
        vel_new[driven] = pres_vel[driven]
        pos_new[driven] = pres_pos[driven]
        # free samples keep the elastic update, driven samples follow kinematics
        pos_new[self.free_nodes] = self.position[self.free_nodes] + dt * vel_new[self.free_nodes]

        # ---- discrete momentum budget (exact for every domain row)
        terms = np.zeros((self.n_domains, self.n_force_terms, 3))
        for g, idx in enumerate(self.rows):
            terms[g, 0] = reac[idx].sum(axis=0)
            terms[g, 1] = f_grav[idx].sum(axis=0)
            terms[g, 2] = f_damp[idx].sum(axis=0)
            terms[g, 3] = f_int[idx].sum(axis=0)
        self.cum_impulse += dt * terms.sum(axis=1)
        # no mass transfer across domain boundaries in a closed solid
        self.budget_force = terms

        self.position = pos_new
        self.velocity = vel_new
        self.f_int = f_int
        self.f_damp = f_damp
        self.f_reac = reac
        self.time = t1

    # ------------------------------------------------------------------ frame
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to called with a time earlier than the current state")
        while self.time < target - 1.0e-12:
            dt = min(self.dt_sub, target - self.time)
            self._step(dt)
        return self._make_frame(target)

    def _make_frame(self, time: float):
        _th_l, _om_l, _th_r, _om_r, rel = self._end_kinematics(time)
        ext = self.f_reac + self.f_damp + self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=np.array(self.position, dtype=float),
            velocity=np.array(self.velocity, dtype=float),
            mass=np.array(self.mass, dtype=float),
            attributes={
                "end_region": self.end_region.copy(),
                "rest_position": self.rest.copy(),
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields={
                "element_deformation_gradient": np.array(self.Fbar, dtype=float),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": np.array(
                    getattr(self, "budget_force", np.zeros((self.n_domains, self.n_force_terms, 3))),
                    dtype=float,
                ),
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "external_force": ext,
                "gravity_acceleration": self.gravity[None, :].copy(),
            },
            scalars={
                "relative_end_rotation_degrees": float(rel),
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        det = getattr(self, "_last_det", np.array([1.0]))
        mu, lam = float(self.mu), float(self.lam)
        young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        poisson = lam / (2.0 * (lam + mu))
        return {
            "model": "compressible_neohookean_hex8_2x2x2_gauss",
            "young_modulus_pa": float(young),
            "poisson_ratio": float(poisson),
            "shear_modulus_pa": mu,
            "first_lame_pa": float(self.lam),
            "damping_coefficient_per_s": float(self.damping_coeff),
            "n_samples": int(self.n_nodes),
            "n_elements": int(self.n_elem),
            "substep_dt": float(self.dt_sub),
            "final_time": float(self.time),
            "min_element_jacobian": float(np.min(det)) if np.size(det) else 1.0,
            "max_element_jacobian": float(np.max(det)) if np.size(det) else 1.0,
            "total_mass_kg": float(self.total_mass),
            "reference_momentum": float(self.reference_momentum),
        }
