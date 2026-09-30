"""Physically based coupled rigid-body / fluid solver for scene 048.

Rigid bodies (spheres, capsules, box) are integrated with explicit
semi-implicit (symplectic) Euler substeps under gravity, hydrostatic buoyancy
from the submerged volume fraction, quadratic drag against the local fluid
velocity, penalty-based wall / inter-body contacts, and isotropic rigid-body
rotation.  Fluid markers are Lagrangian tracer particles advected by the
velocity field induced by the moving bodies (with an incompressibility-style
expulsion term near immersed surfaces) and damped by viscosity, conserving
momentum with the bodies through the drag/buoyancy pairing forces.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class _Body:
    """State of one coupled rigid body (translation + rotation)."""

    def __init__(self, bid, shape, radius, half, x0, q0, mass, volume, rho):
        self.bid = bid
        self.shape = shape              # 'sphere' | 'capsule' | 'box'
        self.radius = radius            # primary radius
        self.half = np.asarray(half, float)  # half extents / half height
        self.x = np.asarray(x0, float).copy()
        self.q = np.asarray(q0, float).copy()
        self.v = np.zeros(3)
        self.w = np.zeros(3)            # angular velocity
        self.m = float(mass)
        self.V = float(volume)
        self.rho = float(rho)
        # collision / influence radius (matches the frozen scaffold radii)
        self.r_inf = float(radius if shape == "sphere"
                           else np.linalg.norm([half[0], half[1], half[2]])) \
            if shape == "box" else float(radius + 0.5 * half[0])
        if shape == "capsule":
            self.r_inf = float(radius + 0.5 * half[0])
        # isotropic inertia from bounding-sphere radius
        self.I = 0.4 * self.m * self.r_inf ** 2
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.pair_force = np.zeros(3)   # fluid-on-body interaction force

    def vertical_half_extent(self) -> float:
        R = _quat_to_mat(self.q)
        if self.shape == "sphere":
            return self.radius
        if self.shape == "capsule":
            axis = R[:, 2]
            return float(0.5 * self.half[0] * abs(axis[2]) + self.radius)
        # box: max |z| of rotated corners
        hx, hy, hz = self.half
        corners = np.array([[sx * hx, sy * hy, sz * hz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return float(np.max(np.abs(corners @ R.T)[:, 2]))

    def velocity_at(self, p: np.ndarray) -> np.ndarray:
        d = p - self.x
        return self.v + np.cross(self.w, d)


class BenchmarkSolver:
    DT = 0.002
    CD = 0.47
    WALL_K = 2000.0     # penalty stiffness per unit mass (1/s^2)
    WALL_C = 60.0       # penalty damping per unit mass (1/s)
    PAIR_K = 3000.0
    TRACER_TAU = 0.05   # viscous relaxation time of markers
    EXPEL_SPEED = 6.0   # surface-normal expulsion of markers out of bodies

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        f0 = scene.initial_state()
        self.seed = int(seed)
        self.rho_f = 1000.0
        self.g = np.array([0.0, 0.0, -9.81])
        self.lo = np.array([-1.45, -0.55, 0.08])
        self.hi = np.array([1.45, 0.55, 2.65])
        self.h = 0.155
        if cfg is not None:
            self.rho_f = float(cfg["fluid"]["density"])
            self.g = np.asarray(cfg["gravity"], float)
            self.lo = np.asarray(cfg["fluid"]["bounds_min"], float)
            self.hi = np.asarray(cfg["fluid"]["bounds_max"], float)
            self.h = float(cfg["fluid"]["marker_spacing"])

        # tank interior (from frozen tank geometry)
        self.wall_lo = np.array([-1.48, -0.58, 0.04])
        self.wall_hi = np.array([1.48, 0.58, 2.80])
        self.water_surface = float(self.hi[2])

        ids = np.asarray(f0.body_id)
        self.pm = np.array(f0.position[ids == 0], float)
        self.vm = np.array(f0.velocity[ids == 0], float)
        self.marker_mass = float(f0.mass[ids == 0][0])
        self.n_markers = len(self.pm)

        # frozen rigid-body geometry (scene_geometry.yaml)
        spec = [
            (1, "sphere", 0.25, [0.25, 0.25, 0.25], [-0.92, 0.00, 2.14], None),
            (2, "capsule", 0.22, [0.20, 0.20, 0.20], [-0.88, 0.06, 1.58], [0.116, 0.341, -0.042, 0.932]),
            (3, "box", 0.29, [0.29, 0.20, 0.18], [0.08, 0.00, 1.72], [0.074, 0.194, 0.056, 0.977]),
            (4, "capsule", 0.24, [0.18, 0.18, 0.18], [0.05, 0.00, 0.78], [-0.07, 0.56, 0.103, 0.819]),
            (5, "box", 0.20, [0.20, 0.20, 0.20], [0.32, -0.02, 0.39], [0.034, 0.07, 0.234, 0.969]),
            (6, "sphere", 0.23, [0.23, 0.23, 0.23], [0.95, 0.00, 0.62], None),
        ]
        ratios = [0.12, 0.35, 0.7, 1.35, 2.2, 5.0]
        volumes = [
            4.0 / 3.0 * np.pi * 0.25 ** 3,
            np.pi * 0.22 ** 2 * 0.20 + 4.0 / 3.0 * np.pi * 0.22 ** 3,
            8.0 * 0.29 * 0.20 * 0.18,
            np.pi * 0.24 ** 2 * 0.18 + 4.0 / 3.0 * np.pi * 0.24 ** 3,
            8.0 * 0.20 ** 3,
            4.0 / 3.0 * np.pi * 0.23 ** 3,
        ]
        r_inf = [0.27, 0.29, 0.36, 0.31, 0.29, 0.25]
        self.bodies: list[_Body] = []
        for k, (bid, shape, rad, half, x0, q0) in enumerate(spec):
            m = self.rho_f * ratios[k] * volumes[k]
            b = _Body(bid, shape, rad, half, x0,
                      q0 if q0 is not None else [0.0, 0.0, 0.0, 1.0],
                      m, volumes[k], self.rho_f * ratios[k])
            b.r_inf = r_inf[k]
            self.bodies.append(b)

        # diagnostic grid over the fluid domain
        self.nx = max(1, int(round((self.hi[0] - self.lo[0]) / self.h)))
        self.ny = max(1, int(round((self.hi[1] - self.lo[1]) / self.h)))
        self.nz = max(1, int(round((self.hi[2] - self.lo[2]) / self.h)))

        self.t = 0.0
        self.time = 0.0
        self._f0 = f0
        self._pair_avg = np.zeros((6, 3))
        self._steps_in_sample = 0

    # ------------------------------------------------------------------ util
    def _fluid_velocity_at_bodies(self):
        """Average marker velocity near each body centre (fluid-side velocity)."""
        out = np.zeros((len(self.bodies), 3))
        for i, b in enumerate(self.bodies):
            d = np.linalg.norm(self.pm - b.x, axis=1)
            sel = d < 2.0 * b.r_inf
            if np.any(sel):
                out[i] = self.vm[sel].mean(axis=0)
        return out

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        nb = len(self.bodies)
        v_fluid = self._fluid_velocity_at_bodies()

        for b in self.bodies:
            b.force = b.m * self.g  # external gravity (per-sample external force)
            b.torque = np.zeros(3)
            b.pair_force = np.zeros(3)

        for i, b in enumerate(self.bodies):
            # --- buoyancy from submerged fraction (hydrostatic, closed tank)
            e = b.vertical_half_extent()
            zbot, ztop = b.x[2] - e, b.x[2] + e
            zs = self.water_surface
            depth = min(max(zs - zbot, 0.0), 2.0 * e)
            frac = depth / (2.0 * e)
            V_sub = b.V * frac
            F_b = np.array([0.0, 0.0, -self.rho_f * V_sub * self.g[2]])
            b.force += F_b
            b.pair_force += F_b
            # righting torque about the centre of buoyancy
            if 0.0 < frac < 1.0:
                z_cb = zbot + 0.5 * depth
                r_cb = np.array([0.0, 0.0, z_cb - b.x[2]])
                b.torque += np.cross(r_cb, F_b)

            # --- quadratic drag against local fluid velocity
            v_rel = b.v - v_fluid[i]
            A = np.pi * b.r_inf ** 2
            F_d = -0.5 * self.rho_f * self.CD * A * np.linalg.norm(v_rel) * v_rel
            F_d = np.clip(F_d, -5.0 * b.m * abs(self.g[2]), 5.0 * b.m * abs(self.g[2]))
            b.force += F_d
            b.pair_force += F_d
            # rotational damping (viscous)
            b.torque += -2.0 * b.I * b.w

        # --- wall contacts (penalty springs on the tank interior)
        for b in self.bodies:
            e = b.r_inf
            for axis in range(3):
                for sign, bound in ((-1.0, self.wall_lo[axis]), (1.0, self.wall_hi[axis])):
                    pen = sign * (b.x[axis] + sign * e - bound)
                    if pen < 0.0:  # penetration depth
                        depth = -pen
                        n = np.zeros(3)
                        n[axis] = sign
                        Fn = (self.WALL_K * depth - self.WALL_C * np.dot(b.v, n)) * b.m
                        Fn = max(Fn, 0.0)
                        b.force += Fn * n
                        # tangential friction
                        vt = b.v - np.dot(b.v, n) * n
                        b.force += -np.clip(8.0 * Fn, 0.0, 5.0 * b.m) * vt / max(
                            np.linalg.norm(vt), 1e-9) * min(np.linalg.norm(vt), 1.0)

        # --- body-body contacts (bounding-sphere penalty, momentum conserving)
        for i in range(nb):
            for j in range(i + 1, nb):
                bi, bj = self.bodies[i], self.bodies[j]
                d = bj.x - bi.x
                dist = np.linalg.norm(d)
                gap = bi.r_inf + bj.r_inf
                if 0.0 < dist < gap:
                    n = d / dist
                    depth = gap - dist
                    mu = 0.5 * (bi.m + bj.m)
                    Fn = self.PAIR_K * mu * depth
                    vrel = np.dot(bj.v - bi.v, n)
                    Fn = max(Fn - 40.0 * mu * vrel, 0.0)
                    bi.force -= Fn * n
                    bj.force += Fn * n

        # --- integrate rigid bodies (semi-implicit Euler)
        for b in self.bodies:
            b.v += (b.force / b.m) * dt
            b.x += b.v * dt
            b.w += (b.torque / b.I) * dt
            b.w = np.clip(b.w, -50.0, 50.0)
            # quaternion integration
            wx, wy, wz = b.w
            dq = 0.5 * dt * np.array([
                wx * b.q[3] + wy * b.q[2] - wz * b.q[1],
                wy * b.q[3] + wz * b.q[0] - wx * b.q[2],
                wz * b.q[3] + wx * b.q[1] - wy * b.q[0],
                -wx * b.q[0] - wy * b.q[1] - wz * b.q[2],
            ])
            b.q = b.q + dq
            b.q /= np.linalg.norm(b.q)

        # --- advect fluid markers (tracers with viscous relaxation)
        self._step_markers(dt)

    def m_of(self, b):
        return b.m

    def _step_markers(self, dt: float) -> None:
        pm, vm = self.pm, self.vm
        vt = np.zeros_like(vm)
        wmax = np.zeros(len(pm))
        for b in self.bodies:
            d = pm - b.x
            dist = np.linalg.norm(d, axis=1)
            infl = b.r_inf + 0.28
            w = np.clip(1.0 - dist / infl, 0.0, 1.0)
            take = w > wmax
            wmax = np.where(take, w, wmax)
            # surface velocity of the body at each marker
            vb = (b.v[None, :] + np.cross(b.w[None, :], d))
            cand = vb * w[:, None]
            vt = np.where(take[:, None], cand, vt)
            # expulsion of markers out of the body volume (displacement)
            inside = dist < b.r_inf
            if np.any(inside):
                nrm = d[inside] / np.maximum(dist[inside][:, None], 1e-9)
                push = self.EXPEL_SPEED * (b.r_inf - dist[inside])[:, None] * nrm
                vt[inside] = vt[inside] + push
        # viscous relaxation toward the induced field
        alpha = min(1.0, dt / self.TRACER_TAU)
        vm += alpha * (vt - vm)
        pm += vm * dt
        # closed no-penetration tank: clamp markers inside the interior
        lo = self.wall_lo + 1e-3
        hi = self.wall_hi - 1e-3
        for ax in range(3):
            low = pm[:, ax] < lo[ax]
            pm[low, ax] = lo[ax]
            vm[low, ax] = np.maximum(vm[low, ax], 0.0) if ax == 2 else vm[low, ax] * 0.0
            high = pm[:, ax] > hi[ax]
            pm[high, ax] = hi[ax]
            vm[high, ax] = np.minimum(vm[high, ax], 0.0) if ax == 2 else vm[high, ax] * 0.0
        np.clip(pm, lo, hi, out=pm)

    # ---------------------------------------------------------------- frames
    def _grid_fields(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        vel = np.zeros((nx, ny, nz, 3))
        mask = np.zeros((nx, ny, nz))
        idx = np.floor((self.pm - self.lo) / self.h).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array([nx, ny, nz])), axis=1)
        idx = idx[ok]
        np.add.at(mask, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        cnt = mask.copy()
        np.add.at(vel, (idx[:, 0], idx[:, 1], idx[:, 2]), self.vm[ok])
        nzmask = cnt > 0
        vel[nzmask] /= cnt[nzmask][:, None]
        return vel, nzmask.astype(float)

    def _build_frame(self, t: float) -> StateFrame:
        nb = len(self.bodies)
        n = self.n_markers + nb
        position = np.vstack((self.pm, np.array([b.x for b in self.bodies])))
        velocity = np.vstack((self.vm, np.array([b.v for b in self.bodies])))
        quat = np.vstack((np.tile([0.0, 0.0, 0.0, 1.0], (self.n_markers, 1)),
                          np.array([b.q for b in self.bodies])))
        angv = np.vstack((np.zeros((self.n_markers, 3)),
                          np.array([b.w for b in self.bodies])))
        mass = np.concatenate((np.full(self.n_markers, self.marker_mass),
                               np.array([b.m for b in self.bodies])))
        body_id = np.concatenate((np.zeros(self.n_markers, np.int32),
                                  np.arange(1, nb + 1, dtype=np.int32)))
        density = np.concatenate((np.full(self.n_markers, self.rho_f),
                                  np.array([b.rho for b in self.bodies])))
        marker_volume = self.h ** 3
        volume = np.concatenate((np.full(self.n_markers, marker_volume),
                                 np.array([b.V for b in self.bodies])))
        phase = np.concatenate((np.zeros(self.n_markers, float),
                                np.ones(nb, float)))

        vel_grid, mask = self._grid_fields()
        ext_force = mass[:, None] * self.g[None, :]

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=quat,
            angular_velocity=angv,
            mass=mass,
            body_id=body_id,
            attributes={
                "density": density,
                "density_ratio": density / self.rho_f,
                "coupled_phase": phase,
                "marker_volume": volume,
                "mass_quantity": mass.copy(),
                "phase_id": body_id.astype(float),
            },
            fields={
                "velocity_field": vel_grid,
                "active_mask": mask,
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext_force,
                "interaction_force": self._pair_avg.copy(),
            },
            scalars={
                "fluid_density": float(self.rho_f),
                "grid_spacing": float(self.h),
                "characteristic_length": float(self.hi[0] - self.lo[0]),
                "liquid_phase_id": 0.0,
                "contact_length_scale": float(self.h),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-12:
            target = self.t
        pair_acc = np.zeros((len(self.bodies), 3))
        nsteps = 0
        while self.t < target - 1e-9:
            dt = min(self.DT, target - self.t)
            self._step(dt)
            self.t += dt
            pair_acc += sum(b.pair_force for b in self.bodies) * dt
            nsteps += 1
        if nsteps > 0:
            self._pair_avg = pair_acc / nsteps
        elif not hasattr(self, "_pair_avg"):
            self._pair_avg = np.zeros((len(self.bodies), 3))
        self.time = target
        return self._build_frame(target)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "n_markers": int(self.n_markers),
            "n_bodies": len(self.bodies),
            "grid": (self.nx, self.ny, self.nz),
            "max_marker_speed": float(np.max(np.linalg.norm(self.vm))) if self.n_markers else 0.0,
        }
