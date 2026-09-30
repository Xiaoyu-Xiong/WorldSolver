"""Discrete-element (soft-sphere) solver for the pouring-sand scene.

Physical model
--------------
Every sand grain is a sphere of nominal radius ``R`` and mass taken from the
frozen initial state.  Grain dynamics follow classical Newtonian mechanics
integrated with a semi-implicit (symplectic) Euler scheme at a fixed time step
that resolves the contact stiffness (Rayleigh criterion).

Forces on each grain:

* gravity ``m * g``;
* grain-grain contacts: linear elastic normal spring ``k_n * delta`` plus a
  viscous normal dashpot, and Coulomb sliding friction regularised with a
  ``tanh`` of the relative tangential velocity (this yields true static
  friction at rest, hence a stable angle of repose);
* boundary contacts (floor, axisymmetric hopper walls, lateral containment of
  the declared particle domain) with the same normal spring-dashpot law and
  their own friction coefficients.

Neighbour search uses a :class:`scipy.spatial.cKDTree` rebuilt every step, so
contact topology (stream breakup, pile growth, runout) is fully emergent.

The hopper inner surface is the axisymmetric piecewise-linear profile declared
in ``scene_geometry.yaml``; it is open at the neck (outlet) and at the top.
"""
from __future__ import annotations

import time as _time

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Soft-sphere DEM solver reproducing hopper discharge and pile formation."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()
        self.rng = np.random.default_rng(int(seed))

        self.pos = np.ascontiguousarray(frame0.position, dtype=np.float64)
        self.vel = np.ascontiguousarray(frame0.velocity, dtype=np.float64)
        self.mass = np.ascontiguousarray(frame0.mass, dtype=np.float64)
        self.n = int(self.pos.shape[0])
        self.inv_mass = 1.0 / self.mass

        gravity = config.get("gravity", (0.0, 0.0, -9.81))
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
        initial = config.get("initial_conditions", {}) or {}
        self.radius = float(initial.get("nominal_grain_radius", 0.013))
        self.spacing = float(initial.get("particle_spacing", 2.0 * self.radius))

        # ---------------- numerics -------------------------------------
        self.dt = 2.5e-4                     # << 2*sqrt(m_eff/k_n) (Rayleigh)
        self.t = 0.0
        self.step_count = 0
        self.wall_time = 0.0

        # ---------------- contact material -----------------------------
        self.kn = 5.0e4                      # normal stiffness  [N/m]
        self.kt_ratio = 0.4                  # tangential/normal stiffness
        self.zeta = 0.35                     # contact damping ratio
        m_eff = 0.5 * float(np.mean(self.mass))          # grain-grain
        self.cn_pp = 2.0 * self.zeta * np.sqrt(self.kn * m_eff)
        self.ct_pp = 2.0 * self.zeta * np.sqrt(self.kn * self.kt_ratio * m_eff)
        self.m_eff = m_eff
        self.m_mean = float(np.mean(self.mass))
        m_wall = self.m_mean                               # grain-wall
        self.cn_w = 2.0 * self.zeta * np.sqrt(self.kn * m_wall)
        self.ct_w = 2.0 * self.zeta * np.sqrt(self.kn * self.kt_ratio * m_wall)

        self.mu_grain = 0.45                 # intergranular friction (sand)
        self.mu_floor = 0.55
        self.mu_hopper = 0.15
        self.mu_side = 0.10
        self.v_eps = 5.0e-4                  # friction regularisation velocity
        # A friction impulse may never reverse the sliding direction, so the
        # regularised Coulomb force is capped by the impulse that stops the
        # relative tangential motion within one step (keeps it unconditionally
        # stable while preserving static-friction-like stick at rest).
        self.friction_cap = 0.8

        self.delta_cap = 0.25 * self.radius  # force-law overlap cap (safety net)
        self.contact_d = 2.0 * self.radius   # interaction distance
        self.search_d = 1.03 * self.contact_d

        # ---------------- frozen boundaries ----------------------------
        self.floor_z = 0.0
        self.floor_hx, self.floor_hy = 1.65, 1.35
        self.side_x, self.side_y = 1.55, 1.25
        # axisymmetric hopper inner profile [radius, z] (scene_geometry.yaml)
        prof = np.asarray(
            [[0.075, 1.42], [0.075, 1.52], [0.34, 1.72], [0.34, 2.28]], dtype=float
        )
        self.prof_z = prof[:, 1].copy()
        self.prof_r = prof[:, 0].copy()
        self.hopper_z0 = float(self.prof_z[0])
        self.hopper_z1 = float(self.prof_z[-1])
        # dR/dz on the conical transition (zero on vertical/straight parts)
        self.cone_slope = float(
            (self.prof_r[2] - self.prof_r[1]) / (self.prof_z[2] - self.prof_z[1])
        )
        self.cone_z0 = float(self.prof_z[1])
        self.cone_z1 = float(self.prof_z[2])

        self.total_mass0 = float(self.mass.sum())
        self.nominal_radius = np.full(self.n, self.radius, dtype=np.float64)
        self.material_phase = np.zeros(self.n, dtype=np.float64)
        self.contact_length_scale = self.contact_d

        self._gaps = np.zeros(1, dtype=np.float64)
        self._ext = np.zeros((self.n, 3), dtype=np.float64)
        self._diag = {}

        # The seeded scene concatenates three lattices (reservoir, funnel and
        # stream) whose interlaced samples can be strongly interpenetrating.
        # Resolve that geometrically so the first physical step starts from a
        # feasible, force-free granular packing (mass and grain count kept).
        self.initial_overlap = self._relax_initial_packing()

    # ------------------------------------------------------- packing repair
    def _project_boundaries(self) -> None:
        """Project grains back into the admissible domain (walls/floor)."""
        R = self.radius
        self.pos[:, 2] = np.maximum(self.pos[:, 2], R + 1.0e-6)
        z = self.pos[:, 2]
        r_xy = np.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)
        inner = np.interp(z, self.prof_z, self.prof_r)
        band = (z >= self.hopper_z0) & (z <= self.hopper_z1)
        need = np.nonzero(band & (r_xy > inner - R) & (r_xy > 1.0e-9))[0]
        if need.size:
            scale = np.maximum(inner[need] - R, 0.0) / r_xy[need]
            self.pos[need, 0] *= scale
            self.pos[need, 1] *= scale
        np.clip(self.pos[:, 0], -(self.side_x - R), self.side_x - R, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], -(self.side_y - R), self.side_y - R, out=self.pos[:, 1])

    def _relax_initial_packing(self, iterations: int = 400) -> float:
        """Separate interpenetrating seed grains; returns the worst residual overlap."""
        target = 1.01 * self.contact_d
        n = self.n
        worst = 0.0
        for _ in range(iterations):
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(target, output_type="ndarray")
            if pairs.size == 0:
                worst = 0.0
                break
            i = pairs[:, 0]
            j = pairs[:, 1]
            dvec = self.pos[j] - self.pos[i]
            dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            np.maximum(dist, 1.0e-9, out=dist)
            ov = target - dist
            act = ov > 1.0e-6
            worst = float(ov.max())
            if not np.any(act):
                break
            i = i[act]
            j = j[act]
            ov = ov[act]
            disp = (0.25 * ov / dist[act])[:, None] * dvec[act]
            for axis in range(3):
                w = disp[:, axis]
                self.pos[:, axis] += np.bincount(j, weights=w, minlength=n)
                self.pos[:, axis] -= np.bincount(i, weights=w, minlength=n)
            self._project_boundaries()
        return worst

    # -------------------------------------------------------------- contacts
    def _pair_forces(self, force: np.ndarray, gaps: list) -> None:
        pos, vel = self.pos, self.vel
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.search_d, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        dvec = pos[j] - pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
        np.maximum(dist, 1.0e-12, out=dist)
        gaps.append(dist - self.contact_d)
        delta = self.contact_d - dist
        np.minimum(delta, self.delta_cap, out=delta)
        act = delta > 0.0
        if not np.any(act):
            return
        i = i[act]
        j = j[act]
        dist = dist[act]
        delta = delta[act]
        nhat = dvec[act] / dist[:, None]

        vrel = vel[i] - vel[j]
        vn = np.einsum("ij,ij->i", vrel, nhat)          # >0 approaching
        fn = self.kn * self._soft_start() * delta + self.cn_pp * vn
        np.maximum(fn, 0.0, out=fn)

        vt_vec = vrel - vn[:, None] * nhat
        vt = np.sqrt(np.einsum("ij,ij->i", vt_vec, vt_vec))
        fr = self.mu_grain * fn * np.tanh(vt / self.v_eps)
        np.minimum(fr, self.friction_cap * self.m_eff * vt / self.dt, out=fr)
        scale = np.divide(fr, vt, out=np.zeros_like(fr), where=vt > 1.0e-12)
        fvec = -(fn[:, None]) * nhat - scale[:, None] * vt_vec

        n = self.n
        for axis in range(3):
            w = fvec[:, axis]
            force[:, axis] += np.bincount(i, weights=w, minlength=n)
            force[:, axis] -= np.bincount(j, weights=w, minlength=n)

    def _wall_forces(self, force: np.ndarray, ext: np.ndarray, gaps: list) -> None:
        pos, vel = self.pos, self.vel
        R = self.radius
        dt_step = self.dt
        knw = self.kn * self._soft_start()

        # ---- receiving floor (top surface z = 0) ----
        pen = R - pos[:, 2]
        sel = np.nonzero(
            (pen > 0.0)
            & (np.abs(pos[:, 0]) <= self.floor_hx)
            & (np.abs(pos[:, 1]) <= self.floor_hy)
        )[0]
        if sel.size:
            p = np.minimum(pen[sel], self.delta_cap)
            fn = knw * p - self.cn_w * vel[sel, 2]
            np.maximum(fn, 0.0, out=fn)
            f = np.zeros((sel.size, 3))
            f[:, 2] = fn
            vt = vel[sel, :2]
            vtm = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            fr = self.mu_floor * fn * np.tanh(vtm / self.v_eps)
            np.minimum(fr, self.friction_cap * self.m_mean * vtm / dt_step, out=fr)
            scale = np.divide(fr, vtm, out=np.zeros_like(fr), where=vtm > 1.0e-12)
            f[:, 0] -= scale * vt[:, 0]
            f[:, 1] -= scale * vt[:, 1]
            force[sel] += f
            ext[sel] += f
            gaps.append(pos[sel, 2] - R)

        # ---- axisymmetric hopper inner wall ----
        z = pos[:, 2]
        r_xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        inner = np.interp(z, self.prof_z, self.prof_r)
        pen = r_xy + R - inner
        in_band = (z >= self.hopper_z0) & (z <= self.hopper_z1)
        sel = np.nonzero((pen > 0.0) & in_band & (r_xy > 1.0e-9))[0]
        if sel.size:
            slope = np.where(
                (z[sel] > self.cone_z0) & (z[sel] < self.cone_z1), self.cone_slope, 0.0
            )
            norm = np.sqrt(1.0 + slope**2)
            n_r = -1.0 / norm                 # inward radial component
            n_z = slope / norm                # inward axial component
            er_x = pos[sel, 0] / r_xy[sel]
            er_y = pos[sel, 1] / r_xy[sel]
            nx = n_r * er_x
            ny = n_r * er_y
            nz = np.full(sel.size, n_z)
            p = np.minimum(pen[sel], self.delta_cap)
            vn = vel[sel, 0] * nx + vel[sel, 1] * ny + vel[sel, 2] * nz
            fn = knw * p - self.cn_w * vn
            np.maximum(fn, 0.0, out=fn)
            f = np.empty((sel.size, 3))
            f[:, 0] = fn * nx
            f[:, 1] = fn * ny
            f[:, 2] = fn * nz
            vt = vel[sel] - vn[:, None] * np.stack((nx, ny, nz), axis=1)
            vtm = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            fr = self.mu_hopper * fn * np.tanh(vtm / self.v_eps)
            np.minimum(fr, self.friction_cap * self.m_mean * vtm / dt_step, out=fr)
            scale = np.divide(fr, vtm, out=np.zeros_like(fr), where=vtm > 1.0e-12)
            f -= scale[:, None] * vt
            force[sel] += f
            ext[sel] += f
            gaps.append(inner[sel] - r_xy[sel] - R)

        # ---- lateral containment of the declared particle domain ----
        for axis, limit in ((0, self.side_x), (1, self.side_y)):
            for sign in (-1.0, 1.0):
                coord = sign * pos[:, axis]
                pen = coord + R - limit
                sel = np.nonzero(pen > 0.0)[0]
                if sel.size == 0:
                    continue
                nvec = np.zeros((sel.size, 3))
                nvec[:, axis] = -sign
                vn = np.einsum("ij,ij->i", vel[sel], nvec)
                fn = knw * np.minimum(pen[sel], self.delta_cap) - self.cn_w * vn
                np.maximum(fn, 0.0, out=fn)
                f = fn[:, None] * nvec
                vt = vel[sel] - vn[:, None] * nvec
                vtm = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                fr = self.mu_side * fn * np.tanh(vtm / self.v_eps)
                np.minimum(fr, self.friction_cap * self.m_mean * vtm / dt_step, out=fr)
                scale = np.divide(fr, vtm, out=np.zeros_like(fr), where=vtm > 1.0e-12)
                f -= scale[:, None] * vt
                force[sel] += f
                ext[sel] += f
                gaps.append(limit - coord[sel] - R)

    def _compute_forces(self) -> np.ndarray:
        force = np.empty((self.n, 3))
        force[:, 0] = 0.0
        force[:, 1] = 0.0
        force[:, 2] = self.mass * self.gravity[2]
        ext = force.copy()
        gaps: list = []
        self._pair_forces(force, gaps)
        self._wall_forces(force, ext, gaps)
        self._ext = ext
        if gaps:
            self._gaps = np.concatenate(gaps)
        else:
            self._gaps = np.zeros(1, dtype=np.float64)
        return force

    def _soft_start(self) -> float:
        """Relax the seeded lattice gently during the first 20 ms."""
        ramp = 0.02
        if self.t >= ramp:
            return 1.0
        return 0.3 + 0.7 * (self.t / ramp)

    # ----------------------------------------------------------------- step
    def _step(self) -> None:
        dt = self.dt
        force = self._compute_forces()
        acc = force * self.inv_mass[:, None]
        self.vel += acc * dt
        self.pos += self.vel * dt
        # numerical safety net: recover any grain that escaped the domain
        bad = ~np.isfinite(self.pos).all(axis=1)
        if np.any(bad):
            self.pos[bad, 0] = 0.0
            self.pos[bad, 1] = 0.0
            self.pos[bad, 2] = self.radius + 1.0e-4
            self.vel[bad] = 0.0
        self.t += dt
        self.step_count += 1

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        t0 = _time.perf_counter()
        target = float(time)
        if target > self.t + 1.0e-12:
            n_steps = int(round((target - self.t) / self.dt))
            n_steps = max(n_steps, 1)
            for _ in range(n_steps):
                self._step()
            self.t = target
        self._compute_forces()          # contacts/external forces at sample time
        self._update_diagnostics()
        self.wall_time += _time.perf_counter() - t0
        return StateFrame(
            time=target,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "nominal_radius": self.nominal_radius.copy(),
                "material_phase": self.material_phase.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": np.asarray(self._gaps, dtype=np.float64).copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self._ext.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "initial_total_mass": self.total_mass0,
                "current_total_mass": float(self.mass.sum()),
                "hopper_inventory_mass": float(self._diag.get("hopper_mass", 0.0)),
                "pile_height": float(self._diag.get("pile_height", 0.0)),
                "pile_runout_radius": float(self._diag.get("runout", 0.0)),
                "mean_speed": float(self._diag.get("mean_speed", 0.0)),
            },
        )

    # ----------------------------------------------------------- diagnostics
    def _update_diagnostics(self) -> None:
        pos, vel = self.pos, self.vel
        r_all = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        in_hopper = pos[:, 2] > self.hopper_z0
        in_stream = (pos[:, 2] > 0.80) & ~in_hopper
        # deposited material: below the free-fall stream zone
        settled = pos[:, 2] < 0.80
        speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        runout = 0.0
        height = 0.0
        if np.any(settled):
            r = np.sqrt(pos[settled, 0] ** 2 + pos[settled, 1] ** 2)
            mobile = settled & (speed > 1.0e-2)
            still = settled & ~mobile
            if np.any(still):
                r_still = np.sqrt(pos[still, 0] ** 2 + pos[still, 1] ** 2)
                runout = float(np.percentile(r_still, 99.0))
                height = float(np.max(pos[still, 2]))
            else:
                runout = float(np.max(r))
                height = float(np.max(pos[settled, 2]))
        repose = 0.0
        if runout > 1.0e-6 and height > 1.0e-6:
            repose = float(np.degrees(np.arctan(height / runout)))
        self._diag = {
            "hopper_mass": float(self.mass[in_hopper].sum()),
            "pile_height": height,
            "runout": runout,
            "repose_deg": repose,
            "mean_speed": float(speed.mean()),
            "grains_below_neck": int(np.sum(~in_hopper)),
            "grains_in_stream": int(np.sum(in_stream)),
            "max_deposit_radius": float(r_all[settled].max()) if np.any(settled) else 0.0,
        }

    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diag)
        out.update(
            {
                "solver": "soft_sphere_dem",
                "n_particles": int(self.n),
                "dt": float(self.dt),
                "steps": int(self.step_count),
                "simulated_time": float(self.t),
                "wall_seconds": float(self.wall_time),
                "total_mass": float(self.mass.sum()),
                "contact_stiffness": float(self.kn),
                "friction_grain": float(self.mu_grain),
                "contact_length_scale": float(self.contact_length_scale),
            }
        )
        return out
