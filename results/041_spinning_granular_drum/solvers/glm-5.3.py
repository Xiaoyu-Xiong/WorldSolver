"""Granular rotating-drum solver (spheres + PBD-style frictional contacts).

Physics: discrete spherical grains (from the frozen scene initial state)
integrated with velocity Verlet-like prediction and XPBD position-level
contact + Coulomb friction constraints against each other, the rotating
cylindrical drum wall, and the front/back drum faces.  The wall is
kinematic and rotates at the prescribed angular velocity about the drum
axis, so wall friction physically lifts grains until the free surface
avalanches.  No keyframes or scripted motion are used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GRAVITY = np.array([0.0, 0.0, -9.81])


class BenchmarkSolver:
    def __init__(self):
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        drum = cfg["initial_conditions"]["drum"]
        grains = cfg["initial_conditions"]["grains"]

        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=np.float64)
        self.v0 = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.n = len(self.x0)
        self.radius = float(np.asarray(frame0.attributes["radius"])[0])

        # keep original attributes for re-export
        self.attr = {k: np.array(v) for k, v in frame0.attributes.items()}

        self.center = np.asarray(drum["center"], dtype=np.float64)
        self.R_wall = float(drum["radius"])          # 0.82
        self.half_depth = 0.5 * float(drum["depth"])  # 0.21
        self.omega = float(drum["angular_velocity_rad_s"])

        # contact params
        self.mu_pp = 0.55     # grain-grain friction
        self.mu_pw = 0.65     # grain-wall friction
        self.iterations = 4
        self.dt = 1.0e-3
        self.d = 2.0 * self.radius
        self.pair_cutoff = self.d + 0.02
        self.wall_tol = 0.03

        self.t = 0.0
        self.x = self.x0.copy()
        self.v = self.v0.copy()
        self._seed = int(seed)
        self._initialized = True

    # ------------------------------------------------------------- wall model
    def _wall_velocity(self, pos):
        """Velocity of the rotating drum material at world points ``pos``."""
        rel = pos - self.center
        om = np.zeros_like(rel)
        om[:, 1] = self.omega                       # rotation about +y
        return np.cross(om, rel)

    # ---------------------------------------------------------------- stepping
    def _step(self):
        dt = self.dt
        x0 = self.x.copy()
        v = self.v

        # gravity + mild damping, semi-implicit prediction
        v = v + GRAVITY * dt
        v *= 1.0 - 2.0e-3
        x = self.x + v * dt

        # neighbor list (fixed pair set for this step)
        tree = cKDTree(x)
        pairs = tree.query_pairs(self.pair_cutoff, output_type="ndarray")
        if pairs.size:
            pi = pairs[:, 0]
            pj = pairs[:, 1]
        else:
            pi = np.zeros(0, dtype=np.int64)
            pj = np.zeros(0, dtype=np.int64)

        cx, cz = self.center[0], self.center[2]
        Reff = self.R_wall - self.radius

        for _ in range(self.iterations):
            # ---- grain-grain normal + friction --------------------------
            if pi.size:
                dvec = x[pi] - x[pj]
                dist = np.linalg.norm(dvec, axis=1)
                act = dist < self.d
                if np.any(act):
                    ia = pi[act]
                    ja = pj[act]
                    dv = dvec[act]
                    ds = dist[act]
                    n = dv / ds[:, None]
                    ov = self.d - ds
                    corr = 0.5 * ov[:, None] * n
                    x[ia] += corr
                    x[ja] -= corr
                    # Coulomb friction on tangential relative displacement
                    rel_disp = (x[ia] - x0[ia]) - (x[ja] - x0[ja])
                    st = rel_disp - (rel_disp * n).sum(1)[:, None] * n
                    stn = np.linalg.norm(st, axis=1)
                    scale = np.ones_like(stn)
                    mask = stn > 1e-12
                    scale[mask] = np.minimum(
                        1.0, (self.mu_pp * ov[mask]) / stn[mask]
                    )
                    fcorr = 0.5 * st * (1.0 - scale)[:, None]
                    x[ia] -= fcorr
                    x[ja] += fcorr

            # ---- cylindrical drum wall (radial, x-z plane) --------------
            rel = x - self.center
            rad = np.hypot(rel[:, 0], rel[:, 2])
            hit = rad > Reff
            if np.any(hit):
                idx = np.nonzero(hit)[0]
                n = rel[idx, [0, 1, 2]].copy()
                n[:, 1] = 0.0
                rn = np.linalg.norm(n, axis=1)
                n = n / rn[:, None]
                pen = rn - Reff
                x[idx] -= pen[:, None] * n
                # wall friction: relative displacement vs. moving wall
                wall_disp = self._wall_velocity(x0[idx]) * dt
                rel_disp = (x[idx] - x0[idx]) - wall_disp
                st = rel_disp - (rel_disp * n).sum(1)[:, None] * n
                stn = np.linalg.norm(st, axis=1)
                scale = np.ones_like(stn)
                mask = stn > 1e-12
                scale[mask] = np.minimum(
                    1.0, (self.mu_pw * pen[mask]) / stn[mask]
                )
                x[idx] -= st * (1.0 - scale)[:, None]

            # ---- front/back drum faces (y direction) --------------------
            ylo = self.center[1] - self.half_depth + self.radius
            yhi = self.center[1] + self.half_depth - self.radius
            for bound, sign in ((yhi, +1.0), (ylo, -1.0)):
                hit = (x[:, 1] - bound) * sign > 0.0
                if np.any(hit):
                    idx = np.nonzero(hit)[0]
                    pen = (x[idx, 1] - bound) * sign
                    x[idx, 1] -= sign * pen
                    n = np.zeros((len(idx), 3))
                    n[:, 1] = sign
                    wall_disp = self._wall_velocity(x0[idx]) * dt
                    rel_disp = (x[idx] - x0[idx]) - wall_disp
                    st = rel_disp - (rel_disp * n).sum(1)[:, None] * n
                    stn = np.linalg.norm(st, axis=1)
                    scale = np.ones_like(stn)
                    m2 = stn > 1e-12
                    scale[m2] = np.minimum(
                        1.0, (self.mu_pw * pen[m2]) / stn[m2]
                    )
                    x[idx] -= st * (1.0 - scale)[:, None]

        self.v = (x - x0) / dt
        self.x = x
        self.t += dt

    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("initialize() must be called first")
        target = float(time)
        # deterministic sub-stepping to the requested sample time
        n_steps = int(round((target - self.t) / self.dt))
        for _ in range(max(0, n_steps)):
            self._step()
        # snap exactly (floating step accumulation)
        self.t = target
        return self._make_frame(target)

    # ------------------------------------------------------------------ export
    def _make_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        x = self.x
        v = self.v

        # contact signed distances: grain-grain near contacts + wall gaps
        gaps = []
        if self.n > 1:
            tree = cKDTree(x)
            pairs = tree.query_pairs(self.d + 0.01, output_type="ndarray")
            if pairs.size:
                dvec = x[pairs[:, 0]] - x[pairs[:, 1]]
                dist = np.linalg.norm(dvec, axis=1)
                gaps.append(dist - self.d)
        rel = x - self.center
        rad = np.hypot(rel[:, 0], rel[:, 2])
        wall_gap = (self.R_wall - self.radius) - rad
        near = wall_gap < self.wall_tol
        if np.any(near):
            gaps.append(wall_gap[near])
        if gaps:
            csd = np.concatenate(gaps)
        else:
            csd = np.zeros(1)
        csd = np.asarray(csd, dtype=np.float64)

        ext_force = self.mass[:, None] * GRAVITY[None, :]

        attributes = {k: np.array(v) for k, v in self.attr.items()}
        attributes["mass_quantity"] = self.mass.copy()

        scalars = {
            "contact_length_scale": float(self.d),
            "characteristic_length": float(self.d),
            "drum_angle": float(self.omega * time),
            "drum_angular_velocity_rad_s": float(self.omega),
            "escaped_particle_count": 0.0,
        }
        # simple diagnostics-derived scalars
        z = x[:, 2]
        top = z > np.quantile(z, 0.92)
        if np.count_nonzero(top) > 2:
            xt = x[top, 0]
            zt = z[top]
            A = np.vstack([xt, np.ones_like(xt)]).T
            coef, *_ = np.linalg.lstsq(A, zt, rcond=None)
            scalars["free_surface_slope"] = float(abs(coef[0]))
        else:
            scalars["free_surface_slope"] = 0.0
        scalars["lifted_grain_height"] = float(np.max(z) - self.center[2])
        speed = np.linalg.norm(v, axis=1)
        scalars["avalanche_layer_speed"] = (
            float(np.mean(speed[top])) if np.any(top) else 0.0
        )

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            mass=self.mass.copy(),
            body_id=np.ones(self.n, dtype=np.int32),
            attributes=attributes,
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext_force,
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        z = self.x[:, 2]
        speed = np.linalg.norm(self.v, axis=1)
        return {
            "time": self.t,
            "drum_angle": float(self.omega * self.t),
            "lifted_grain_height": float(np.max(z) - self.center[2]),
            "mean_speed": float(np.mean(speed)),
            "escaped_particle_count": 0,
            "particle_count": int(self.n),
        }
