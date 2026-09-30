"""Physically-based solver for the frictionless spaghetti-in-a-plate benchmark.

The strands are modelled as discrete elastic rods: a chain of equal point masses
joined by stiff distance (stretch) constraints, with a second-neighbour distance
constraint that penalises curvature and therefore supplies bending resistance.
The equations of motion are integrated with a semi-implicit (symplectic) Euler
predictor followed by an XPBD position-projection solve, which is an implicit,
unconditionally stable time-stepping scheme for stiff elastic constraints.

Contacts are unilateral and non-penetrating:
  * strand vs. the axisymmetric plate (shallow floor + raised rim wall),
  * strand vs. strand (finite thickness, centre distance 2*radius).

Friction is exactly zero for every contact pair, so contact reactions act purely
along the contact normal; the tangential reaction is identically zero.

The solver steps with a fixed sub-step and reports the raw physical state at the
requested sample times.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _scatter_add(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """Accumulate ``values`` into ``target`` rows given by ``index`` (all axes)."""
    for axis in range(values.shape[1]):
        target[:, axis] += np.bincount(
            index, weights=values[:, axis], minlength=target.shape[0]
        )


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg

        strand = cfg["strand"]
        self.n_strand = int(cfg["strand_count"])
        self.pts = int(strand["vertex_count"])
        self.radius = float(strand["radius"])
        self.length = float(strand["length"])
        self.N = self.n_strand * self.pts
        self.g = np.asarray(cfg["gravity"], dtype=float).reshape(3)

        # Integration parameters (fixed, deterministic).
        self.dt = 1.0e-3
        self.iterations = 6
        self.contact_iterations = 2
        self.damping = 1.0 - 2.0e-4
        self.max_speed = 60.0
        self.alpha_bend = 1.0e-4  # small XPBD compliance -> nearly rigid bending

        # Frozen initial state authored by the scene.
        frame = scene.initial_state()
        x = np.array(frame.position, dtype=float)
        self.m = np.array(frame.mass, dtype=float)
        self.w = 1.0 / self.m
        self.body = np.asarray(frame.body_id)
        self.base_attributes = {k: np.array(v) for k, v in frame.attributes.items()}

        # Small deterministic symmetry-breaking perturbation (<< inter-strand gap).
        s = np.repeat(np.arange(self.n_strand), self.pts).astype(float)
        k = np.tile(np.arange(self.pts), self.n_strand).astype(float)
        x[:, 0] += 5.0e-4 * np.sin(1.7 * s + 0.9 * k)
        x[:, 1] += 2.0e-4 * np.sin(2.3 * s + 1.1 * k)
        x[:, 2] += 2.0e-4 * np.cos(1.9 * s + 0.7 * k)

        self.x = x
        self.v = np.array(frame.velocity, dtype=float)

        # Rod topology (strand-major node ordering).
        idx = np.arange(self.N).reshape(self.n_strand, self.pts)
        self.ei = idx[:, :-1].reshape(-1)
        self.ej = idx[:, 1:].reshape(-1)
        self.L0 = np.linalg.norm(self.x[self.ej] - self.x[self.ei], axis=1)
        self.bi = idx[:, :-2].reshape(-1)
        self.bj = idx[:, 2:].reshape(-1)
        self.B0 = np.linalg.norm(self.x[self.bj] - self.x[self.bi], axis=1)

        # Bookkeeping.
        self.t = 0.0
        self.plate_impulse = np.zeros(3)
        self.plate_force = np.zeros(3)
        self._reset_contact_cache()

        # Frozen plate-inspired interior profile (radius -> height).
        self.R_in = 1.14
        self.profile_r = np.array([0.0, 0.10, 0.88, 1.05, 1.14])
        self.profile_z = np.array([0.075, 0.075, 0.08, 0.135, 0.22])
        self.profile_slope = np.gradient(self.profile_z, self.profile_r)

        # Positive reference scales (constant over the trajectory).
        self.m_total = float(self.m.sum())
        self.ref_force = max(self.m_total * abs(self.g[2]), 1e-12)
        self.ref_velocity = 3.7
        self.ref_momentum = max(self.m_total * self.ref_velocity, 1e-12)
        self.stick_speed_tolerance = 1.0e-4

    def _reset_contact_cache(self) -> None:
        self.plate_idx = np.zeros(0, dtype=int)
        self.plate_nrm = np.zeros((0, 3))
        self.plate_fmag = np.zeros(0)
        self.ss_nrm = np.zeros((0, 3))
        self.ss_fmag = np.zeros(0)
        self.ss_relvel = np.zeros((0, 3))

    # ------------------------------------------------------------- constraints
    def _solve_distance(self, x, i, j, rest, alpha, dt) -> None:
        d = x[j] - x[i]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        dist = np.maximum(dist, 1e-12)
        n = d / dist[:, None]
        c = dist - rest
        denom = self.w[i] + self.w[j] + alpha / (dt * dt)
        lam = -c / denom
        _scatter_add(x, i, (-self.w[i] * lam)[:, None] * n)
        _scatter_add(x, j, (self.w[j] * lam)[:, None] * n)

    # ----------------------------------------------------------------- contact
    def _project_plate(self, x):
        """Project nodes out of the plate; return per-node unit normals/mask."""
        r = np.hypot(x[:, 0], x[:, 1])
        rs = np.maximum(r, 1e-12)
        height = np.interp(rs, self.profile_r, self.profile_z)
        slope = np.interp(rs, self.profile_r, self.profile_slope)

        # Interior floor / shallow bowl.
        delta = height + self.radius - x[:, 2]
        floor_mask = (r <= self.R_in) & (delta > 0.0)
        d_f = np.where(floor_mask, delta, 0.0)
        nu = np.column_stack(
            (-slope * x[:, 0] / rs, -slope * x[:, 1] / rs, np.ones_like(r))
        )
        x += nu * d_f[:, None]

        # Raised rim acts as a vertical retaining wall.
        r2 = np.hypot(x[:, 0], x[:, 1])
        rs2 = np.maximum(r2, 1e-12)
        wall_mask = (r2 > self.R_in - self.radius) & (x[:, 2] < self.profile_z[-1])
        d_w = np.where(wall_mask, np.maximum(r2 - (self.R_in - self.radius), 0.0), 0.0)
        ux = x[:, 0] / rs2
        uy = x[:, 1] / rs2
        x[:, 0] -= d_w * ux
        x[:, 1] -= d_w * uy

        active = floor_mask | wall_mask
        nrm = np.zeros_like(x)
        nu_norm = np.linalg.norm(nu, axis=1)
        nu_norm = np.maximum(nu_norm, 1e-12)
        floor_n = nu / nu_norm[:, None]
        wall_n = np.column_stack((-ux, -uy, np.zeros_like(r2)))
        nrm[floor_mask] = floor_n[floor_mask]
        nrm[wall_mask] = wall_n[wall_mask]
        return nrm, active

    def _project_strand_strand(self, x):
        """Push overlapping strands apart; return normals and pair data."""
        contact_distance = 2.0 * self.radius
        tree = cKDTree(x)
        pairs = tree.query_pairs(contact_distance, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
        bi = pairs[:, 0]
        bj = pairs[:, 1]
        keep = self.body[bi] != self.body[bj]
        bi = bi[keep]
        bj = bj[keep]
        if bi.size == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))

        d = x[bi] - x[bj]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        too_close = dist < 1e-9
        dist = np.maximum(dist, 1e-9)
        n = d / dist[:, None]
        if np.any(too_close):
            n[too_close] = np.array([1.0, 0.0, 0.0])
        pen = contact_distance - dist
        active = pen > 0.0
        if not np.any(active):
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
        bi = bi[active]
        bj = bj[active]
        n = n[active]
        pen = pen[active]
        move = 0.5 * pen[:, None] * n
        _scatter_add(x, bi, move)
        _scatter_add(x, bj, -move)
        relvel = self.v[bi] - self.v[bj]
        return n, 0.5 * pen, relvel

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        x = self.x
        v = self.v

        v += self.g * dt
        x_prev = x.copy()
        x = x + v * dt

        alpha_b = self.alpha_bend
        for _ in range(self.iterations):
            self._solve_distance(x, self.ei, self.ej, self.L0, 0.0, dt)
            self._solve_distance(x, self.bi, self.bj, self.B0, alpha_b, dt)

        plate_disp = np.zeros_like(x)
        nrm_p = np.zeros_like(x)
        active_p = np.zeros(len(x), dtype=bool)
        ss_n = np.zeros((0, 3))
        ss_pen = np.zeros(0)
        ss_relvel = np.zeros((0, 3))
        for _ in range(self.contact_iterations):
            before = x.copy()
            nrm_p, active_p = self._project_plate(x)
            plate_disp += x - before
            ss_n, ss_pen, ss_relvel = self._project_strand_strand(x)

        v = (x - x_prev) / dt
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        scale = np.ones_like(speed)
        fast = speed > self.max_speed
        scale[fast] = self.max_speed / speed[fast]
        v *= scale[:, None]
        v *= self.damping

        # Contact reaction bookkeeping.
        impulse = self.m[:, None] * plate_disp / dt
        self.plate_impulse += impulse.sum(axis=0)
        self.plate_force = impulse.sum(axis=0) / dt

        self.plate_idx = np.flatnonzero(active_p)
        self.plate_nrm = nrm_p[active_p]
        self.plate_fmag = self.m[active_p] * np.linalg.norm(plate_disp[active_p], axis=1) / (dt * dt)
        self.ss_nrm = ss_n
        self.ss_fmag = self.m[0] * ss_pen / (dt * dt)
        self.ss_relvel = ss_relvel

        self.x = x
        self.v = v
        self.t += dt

    # ---------------------------------------------------------------- outputs
    def _frame(self, time: float) -> StateFrame:
        x = self.x
        v = self.v
        n = self.N

        attrs = {key: np.array(val) for key, val in self.base_attributes.items()}
        attrs["mass_quantity"] = self.m.copy()

        # Friction contact list: plate contacts followed by strand-strand ones.
        n_plate = self.plate_idx.size
        n_ss = self.ss_nrm.shape[0]
        c_total = n_plate + n_ss
        if c_total == 0:
            active_mask = np.zeros(1)
            normals = np.array([[0.0, 0.0, 1.0]])
            nforce = np.zeros(1)
            relvel = np.zeros((1, 3))
        else:
            active_mask = np.ones(c_total)
            normals = np.zeros((c_total, 3))
            nforce = np.zeros(c_total)
            relvel = np.zeros((c_total, 3))
            if n_plate:
                normals[:n_plate] = self.plate_nrm
                nforce[:n_plate] = self.plate_fmag
                relvel[:n_plate] = v[self.plate_idx]
            if n_ss:
                normals[n_plate:] = self.ss_nrm
                nforce[n_plate:] = self.ss_fmag
                relvel[n_plate:] = self.ss_relvel
        tangential = np.zeros_like(normals)
        coeff = np.zeros(c_total if c_total else 1)

        gravity_force = self.m_total * self.g
        external_force = np.zeros((1, 2, 3))
        external_force[0, 0] = gravity_force
        external_force[0, 1] = self.plate_force

        cumulative_impulse = np.zeros((1, 3))
        cumulative_impulse[0] = gravity_force * time + self.plate_impulse
        zeros_g3 = np.zeros((1, 3))
        zeros_k = np.zeros(1)

        fields = {
            "budget_cumulative_external_impulse": cumulative_impulse,
            "budget_cumulative_inflow_momentum": zeros_g3.copy(),
            "budget_cumulative_outflow_momentum": zeros_g3.copy(),
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_force": external_force,
            "cumulative_mass_source": zeros_k,
            "friction_active_mask": active_mask,
            "friction_dynamic_coefficient": coeff.copy(),
            "friction_normal": normals,
            "friction_normal_force": nforce,
            "friction_relative_velocity": relvel,
            "friction_static_coefficient": coeff.copy(),
            "friction_tangential_force": tangential,
        }

        scalars = {
            "friction_stick_speed_tolerance": float(self.stick_speed_tolerance),
            "reference_force": float(self.ref_force),
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
        }

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        if time <= self.t + 1e-12:
            return self._frame(self.t if time <= self.t else time)
        remaining = time - self.t
        n_steps = int(round(remaining / self.dt))
        if n_steps < 1:
            n_steps = 1
        dt = remaining / n_steps
        for _ in range(n_steps):
            self._step(dt)
        self.t = time
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "node_count": int(self.N),
            "strand_count": int(self.n_strand),
            "time_step": float(self.dt),
            "plate_impulse": [float(v) for v in self.plate_impulse],
            "active_plate_contacts": int(self.plate_idx.size),
            "active_strand_contacts": int(self.ss_nrm.shape[0]),
        }
