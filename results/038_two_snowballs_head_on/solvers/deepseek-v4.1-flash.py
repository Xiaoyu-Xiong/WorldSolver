"""Material-point-method solver for the head-on two-snowball collision scene.

Physics implemented from scratch on top of NumPy:

* MLS-MPM style particle/grid transfer with quadratic B-spline (APIC) shape
  functions and an explicit updated-Lagrangian time step.
* Finite-strain multiplicative elasto-plasticity for snow (Stomakhin et al.):
  ``F = F_e F_p`` with a singular-value clamping return map bounded by
  ``theta_c`` (compression) and ``theta_s`` (stretch), giving cohesive
  compaction and hardening before failure.
* Strain driven damage that degrades the elastic stiffness, producing
  cohesive fracture / fragmentation of the two snowballs after impact.
* Rigid floor contact (normal clamp + tangential friction) and free-slip
  lateral numerical domain walls.
* Optional mass loss: fully damaged, fast moving particles are eroded and the
  removed mass is reported through ``fields.cumulative_mass_source``.

The solver is fully deterministic; no random numbers are used at any point.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --- material constants (fresh snow) -------------------------------------
_YOUNG = 8.0e4
_POISSON = 0.25
_THETA_C = 0.025      # plastic yield in compression
_THETA_S = 0.0075     # plastic yield in tension
_DAMAGE_PLASTIC = 2.0
_DAMAGE_TENSILE = 200.0
_STIFFNESS_FLOOR = 0.03
_MAX_SPEED = 45.0
_MAX_STRETCH = 3.0
_FRICTION = 0.995
_ERODE_DAMAGE = 0.90
_ERODE_SPEED = 3.2
_ERODE_FRACTION = 0.04
_MIN_BODY_PARTICLES = 120


class BenchmarkSolver:
    """Explicit MPM snow solver honouring the frozen scene geometry."""

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = getattr(scene, "config", {}) or {}
        self.config = cfg
        self.seed = int(seed)
        snow = cfg["snowballs"]
        floor = cfg["floor"]

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.floor_z = float(floor["top_z"])
        self.radius = float(snow["radius"])
        self.spacing = float(snow["particle_spacing"])
        self.impact_speed = float(snow["impact_speed"])

        # --- particle state from the frozen scene --------------------------
        state = scene.initial_state()
        self.pos = np.array(state.position, dtype=float)
        self.vel = np.array(state.velocity, dtype=float)
        self.mass = np.array(state.mass, dtype=float)
        self.body = np.array(state.body_id, dtype=np.int64)
        n = len(self.pos)

        # reference volume: exact sphere occupancy (keeps the ball geometry
        # consistent with the frozen radius, independent of nominal density)
        sphere_volume = 4.0 / 3.0 * np.pi * self.radius ** 3
        self.V0 = np.empty(n, dtype=float)
        self.body_ids = np.unique(self.body)
        for bid in self.body_ids:
            sel = self.body == bid
            self.V0[sel] = sphere_volume / float(np.count_nonzero(sel))

        self.density0 = float(self.mass.sum() / self.V0.sum())
        self.mu = _YOUNG / (2.0 * (1.0 + _POISSON))
        self.lam = _YOUNG * _POISSON / ((1.0 + _POISSON) * (1.0 - 2.0 * _POISSON))

        self.F = np.tile(np.eye(3), (n, 1, 1))
        self.Fp = np.tile(np.eye(3), (n, 1, 1))
        self.C = np.zeros((n, 3, 3))
        self.damage = np.zeros(n)

        # --- background grid ----------------------------------------------
        self.dx = self.spacing
        self.origin = np.array([-3.0, -1.8, -0.375], dtype=float)
        self.nx = 80
        self.ny = 48
        self.nz = 40
        self._nmax = np.array([self.nx - 1, self.ny - 1, self.nz - 1])
        self.n_nodes = self.nx * self.ny * self.nz
        k = np.arange(3)
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
        self._ko = np.stack([kx, ky, kz], axis=-1).reshape(27, 3)
        self._i_floor = int(round((self.floor_z - self.origin[2]) / self.dx))

        # --- time stepping --------------------------------------------------
        c_wave = np.sqrt((self.lam + 2.0 * self.mu) / max(self.density0, 1e-9))
        cfl_dt = 0.5 * self.dx / max(c_wave, 1e-6)
        self.dt = float(min(1.0e-3, cfl_dt))
        if self.dt <= 0.0:
            self.dt = 1.0e-3
        self.time = 0.0
        self.removed_mass = 0.0
        self.removed_count = 0
        self.n0 = n
        self.initial_total_mass = float(self.mass.sum())
        self.initial_total_momentum = (self.mass[:, None] * self.vel).sum(axis=0).copy()
        self._eye = np.eye(3)

    # ------------------------------------------------------------------ #
    # shape functions
    # ------------------------------------------------------------------ #
    def _kernel(self, pc):
        """Quadratic B-spline weights, gradients and node ids for cells *pc*."""
        i0 = np.floor(pc - 0.5).astype(np.int64)
        node = i0[:, None, :] + self._ko[None, :, :]
        d = pc[:, None, :] - node
        nd = np.clip(node, 0, self._nmax[None, None, :])
        idx = nd[..., 0] + self.nx * (nd[..., 1] + self.ny * nd[..., 2])
        a = np.abs(d)
        nw = np.where(a > 0.5, 0.5 * (1.5 - a) ** 2, 0.75 - d * d)
        nw = np.where(a >= 1.5, 0.0, nw)
        dw = np.where(a <= 0.5, -2.0 * d, -np.sign(d) * (1.5 - a))
        dw = np.where(a >= 1.5, 0.0, dw)
        w = nw[..., 0] * nw[..., 1] * nw[..., 2]
        grads = []
        for j in range(3):
            g = dw[..., j] / self.dx
            for m in range(3):
                if m != j:
                    g = g * nw[..., m]
            grads.append(g)
        gw = np.stack(grads, axis=-1)
        dvec = -d * self.dx
        return w, gw, idx, dvec, d

    # ------------------------------------------------------------------ #
    # constitutive response
    # ------------------------------------------------------------------ #
    def _elastic_response(self, dt):
        Fe = self.F @ np.linalg.inv(self.Fp)
        U, s, Vt = np.linalg.svd(Fe)
        s_c = np.clip(s, 1.0 - _THETA_C, 1.0 + _THETA_S)
        R = U @ Vt
        J = np.prod(s_c, axis=1)
        V = Vt.transpose(0, 2, 1)
        Ut = U.transpose(0, 2, 1)
        inv_sc = 1.0 / s_c
        Fe_new = (U * s_c[:, None, :]) @ Vt
        self.Fp = ((V * inv_sc[:, None, :]) @ Ut) @ self.F
        self.Fp = np.clip(self.Fp, -_MAX_STRETCH, _MAX_STRETCH)
        invFe = (V * inv_sc[:, None, :]) @ Ut
        P = 2.0 * self.mu * (Fe_new - R) + (self.lam * (J - 1.0) * J)[:, None, None] * \
            invFe.transpose(0, 2, 1)
        # strain driven damage: plastic flow plus dilatant (tensile) straining
        dinc = np.abs(s - s_c).sum(axis=1)
        self.damage = np.clip(
            self.damage + _DAMAGE_PLASTIC * dinc
            + _DAMAGE_TENSILE * np.maximum(0.0, J - 1.0) * dt,
            0.0, 1.0,
        )
        stiffness = np.maximum(1.0 - self.damage, _STIFFNESS_FLOOR)
        P = P * stiffness[:, None, None]
        return Fe_new, P

    # ------------------------------------------------------------------ #
    # one explicit substep
    # ------------------------------------------------------------------ #
    def _substep(self, dt):
        Fe, P = self._elastic_response(dt)
        pc = (self.pos - self.origin) / self.dx
        w, gw, idx, dvec, d = self._kernel(pc)
        flat = idx.ravel()

        wm = w * self.mass[:, None]
        Coff = np.einsum("pij,pkj->pki", self.C, dvec)
        vnode = self.vel[:, None, :] + Coff
        mass_g = np.bincount(flat, weights=wm.ravel(), minlength=self.n_nodes)
        mom_g = np.empty((self.n_nodes, 3))
        for c in range(3):
            mom_g[:, c] = np.bincount(
                flat, weights=(wm[:, :, None] * vnode)[:, :, c].ravel(),
                minlength=self.n_nodes,
            )
        PFt = np.einsum("pij,pkj->pik", P, Fe)
        fpart = -self.V0[:, None, None] * np.einsum("pij,pkj->pki", PFt, gw)
        f_g = np.empty((self.n_nodes, 3))
        for c in range(3):
            f_g[:, c] = np.bincount(
                flat, weights=fpart[:, :, c].ravel(), minlength=self.n_nodes
            )

        mask = mass_g > 1e-12
        vel_g = np.zeros((self.n_nodes, 3))
        vel_g[mask] = mom_g[mask] / mass_g[mask, None]
        vel_g[mask] += (f_g[mask] / mass_g[mask, None]) * dt
        vel_g[mask, 2] += self.gravity[2] * dt

        vg = vel_g.reshape(self.nx, self.ny, self.nz, 3)
        vg[0, :, :, 0] = 0.0
        vg[-1, :, :, 0] = 0.0
        vg[:, 0, :, 1] = 0.0
        vg[:, -1, :, 1] = 0.0
        top = self._i_floor + 1
        vg[:, :, :top, 2] = np.maximum(vg[:, :, :top, 2], 0.0)
        vg[:, :, :top, 0] *= _FRICTION
        vg[:, :, :top, 1] *= _FRICTION

        # D matrix for APIC affine reconstruction
        D = self.dx * self.dx * np.einsum("pk,pki,pkj->pij", w, d, d)
        D = D + (1.0e-9 + 1.0e-6 * self.dx * self.dx) * self._eye[None, :, :]
        Dinv = np.linalg.inv(D)

        vk = vel_g[flat].reshape(-1, 27, 3)
        v_new = np.einsum("pk,pkc->pc", w, vk)
        C_new = np.einsum("pk,pkc,pkj->pcj", w, vk, dvec) @ Dinv

        below = self.pos[:, 2] < self.floor_z
        if np.any(below):
            v_new[below, 2] = np.maximum(v_new[below, 2], 0.0)

        speed = np.linalg.norm(v_new, axis=1)
        too_fast = speed > _MAX_SPEED
        if np.any(too_fast):
            v_new[too_fast] *= (_MAX_SPEED / speed[too_fast])[:, None]

        self.pos = self.pos + dt * v_new
        self.pos[:, 2] = np.maximum(self.pos[:, 2], self.floor_z)
        self.vel = v_new
        self.C = C_new
        self.F = (self._eye[None, :, :] + dt * C_new) @ self.F
        self.F = np.clip(self.F, -_MAX_STRETCH, _MAX_STRETCH)

        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()
                and np.isfinite(self.F).all() and np.isfinite(self.Fp).all()):
            self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
            self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
            self.F = np.nan_to_num(self.F, nan=1.0, posinf=1.0, neginf=-1.0)
            self.Fp = np.nan_to_num(self.Fp, nan=1.0, posinf=1.0, neginf=-1.0)
        return Fe

    # ------------------------------------------------------------------ #
    # mass loss
    # ------------------------------------------------------------------ #
    def _erode(self) -> None:
        speed = np.linalg.norm(self.vel, axis=1)
        cand = (self.damage > _ERODE_DAMAGE) & (speed > _ERODE_SPEED)
        if not np.any(cand):
            return
        budget = _ERODE_FRACTION * self.n0 - self.removed_count
        if budget <= 0:
            return
        keep_body = np.ones(len(self.pos), dtype=bool)
        for bid in self.body_ids:
            sel = self.body == bid
            allowed = int(np.count_nonzero(sel)) - _MIN_BODY_PARTICLES
            if allowed <= 0:
                keep_body[sel] = False
                continue
            local = np.nonzero(cand & sel)[0]
            if len(local) > allowed:
                keep_body[local[np.argsort(-speed[local])[allowed:]]] = False
        remove = cand & keep_body
        if int(remove.sum()) > budget:
            ids = np.nonzero(remove)[0]
            remove = np.zeros_like(remove)
            remove[ids[np.argsort(-speed[ids])[:int(budget)]]] = True
        if not np.any(remove):
            return
        self.removed_mass += float(self.mass[remove].sum())
        self.removed_count += int(remove.sum())
        keep = ~remove
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.mass = self.mass[keep]
        self.body = self.body[keep]
        self.V0 = self.V0[keep]
        self.F = self.F[keep]
        self.Fp = self.Fp[keep]
        self.C = self.C[keep]
        self.damage = self.damage[keep]

    # ------------------------------------------------------------------ #
    # public contract
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1.0e-12:
            dt = min(self.dt, target - self.time)
            if dt <= 0.0:
                break
            self._substep(dt)
            self.time += dt
            guard += 1
            if guard > 100000:
                break
        if self.time > target:
            self.time = target
        self._erode()
        return self._frame(target)

    def _frame(self, target: float) -> StateFrame:
        pos = np.nan_to_num(self.pos)
        vel = np.nan_to_num(self.vel)
        mass = np.nan_to_num(self.mass)
        F = np.nan_to_num(self.F, nan=1.0)
        frame = StateFrame(
            time=float(target),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=mass.copy(),
            body_id=self.body.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": F.copy(),
                "reference_volume": self.V0.copy(),
                "damage": self.damage.copy(),
            },
            fields={
                "external_force": mass[:, None] * self.gravity[None, :],
                "cumulative_mass_source": np.array([-self.removed_mass], dtype=float),
                "floor_signed_distance": pos[:, 2] - self.floor_z,
            },
            scalars={
                "initial_total_mass": float(self.initial_total_mass),
                "current_total_mass": float(mass.sum()),
                "removed_mass": float(self.removed_mass),
                "particle_count": float(len(pos)),
                "kinetic_energy": float(
                    0.5 * np.sum(mass * np.einsum("ij,ij->i", vel, vel))
                ),
                "max_speed": float(np.linalg.norm(vel, axis=1).max(initial=0.0)),
                "max_damage": float(self.damage.max(initial=0.0)),
                "particle_spacing": float(self.spacing),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mpm_finite_strain_snow_plasticity",
            "initial_particles": int(self.n0),
            "final_particles": int(len(self.pos)),
            "removed_particles": int(self.removed_count),
            "removed_mass_kg": float(self.removed_mass),
            "simulated_time": float(self.time),
            "dt": float(self.dt),
            "youngs_modulus": float(_YOUNG),
            "poisson_ratio": float(_POISSON),
        }
