"""Closed cloth belt over three prescribed kinematic cylinders.

Physical model
-------------
The belt is a discrete shell (120 x 9 node ring, closed in the ring direction)
integrated with a substepped XPBD scheme:

  * gravity + viscous drag prediction,
  * stretch springs along ring and width directions (with the frozen prestrain
    baked into the rest lengths), shear springs on the quad diagonals and weak
    second-neighbour bending springs,
  * unilateral contact against the three analytic cylinders with position-level
    Coulomb friction (the standard PBD friction limit mu * |normal correction|,
    which is exactly the Coulomb impulse bound for a substep),
  * velocities recovered from the position change of the substep.

The three cylinders are prescribed boundaries: the top one is lifted, the two
outer ones spin forward one revolution, pause, then reverse one revolution as
frozen in ``config.yaml``.  Nothing about the cloth motion is scripted.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = scene.config
        self.config = config

        belt = config["belt"]
        cyl = config["cylinders"]
        self.timeline = config["timeline"]

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.mu = float(config["friction"]["coefficient"])
        self.sample_dt = float(config["sample_dt"])
        self.duration = float(config["duration"])

        # ---------------------------------------------------------- geometry
        initial = scene.initial_state()
        self.n_total = int(len(initial.position))
        self.ring = 120
        self.wide = int(belt["width_nodes"])
        self.n_cloth = self.ring * self.wide
        self.faces = np.asarray(scene.metadata()["topology"]["faces"], dtype=np.int32)

        self.x0 = np.array(initial.position[: self.n_cloth], dtype=float)
        self.x = self.x0.copy()
        self.v = np.array(belt["initial_velocity"], dtype=float)[None, :] * np.ones(
            (self.n_cloth, 1)
        )
        self.uv = np.array(initial.attributes["uv"], dtype=float)

        self.total_mass = float(belt["total_mass"])
        self.node_mass = self.total_mass / self.n_cloth
        self.w = np.full(self.n_cloth, 1.0 / self.node_mass)
        self.thickness = float(belt["thickness"])

        # cylinders
        self.cyl_radius = float(cyl["radius"])
        self.cyl_length = float(cyl["length"])
        self.contact_radius = self.cyl_radius + float(belt["centerline_clearance"])
        self.init_centers = np.array(
            [[x, 0.0, z] for x, z in cyl["initial_centers_xz"]], dtype=float
        )
        self.final_centers = np.array(
            [[x, 0.0, z] for x, z in cyl["final_centers_xz"]], dtype=float
        )
        self.spin_signs = np.array(
            [float(cyl["outer_rotation_signs"][0]), float(cyl["outer_rotation_signs"][1]), 0.0],
            dtype=float,
        )
        self.spin_speed = float(cyl["outer_angular_speed"])
        self.axis = np.asarray(cyl["axis"], dtype=float)
        self.axis = self.axis / max(np.linalg.norm(self.axis), 1e-12)
        self.base_quat = np.array([0.707106781, 0.0, 0.0, 0.707106781], dtype=float)

        self._build_constraints(belt)

        # ------------------------------------------------------- integration
        self.n_sub = 10
        self.n_iter = 8
        self.dt = self.sample_dt / self.n_sub
        self.k_stretch = 300.0
        self.k_shear = 60.0
        self.k_bend = 8.0
        self.drag = 0.0015          # per-substep velocity drag fraction
        self.vel_smooth = 0.06      # Jacobi velocity diffusion (numerical damping)
        self.v_max = 40.0

        compliance = np.empty(self.n_edges)
        compliance[self.grp_stretch] = 1.0 / self.k_stretch
        compliance[self.grp_shear] = 1.0 / self.k_shear
        compliance[self.grp_bend] = 1.0 / self.k_bend
        self.compliance = compliance
        self.alpha_tilde = compliance / (self.dt * self.dt)

        self.time = 0.0
        self._steps = 0
        self._max_penetration = 0.0

    # -------------------------------------------------------- constraint sets
    def _build_constraints(self, belt: dict) -> None:
        ring, wide = self.ring, self.wide
        idx = np.arange(ring * wide, dtype=np.int64).reshape(ring, wide)
        prestrain = float(belt["prestrain"])

        batches: list[dict] = []

        def add(a: np.ndarray, b: np.ndarray, kind: str, colors: np.ndarray, scale: float) -> None:
            a = np.ascontiguousarray(a.ravel())
            b = np.ascontiguousarray(b.ravel())
            colors = np.ascontiguousarray(colors.ravel())
            rest = np.linalg.norm(self.x0[a] - self.x0[b], axis=1) * scale
            for c in np.unique(colors):
                m = colors == c
                batches.append(
                    {
                        "e0": a[m],
                        "e1": b[m],
                        "rest": np.maximum(rest[m], 1e-9),
                        "kind": kind,
                        "one_sided": kind == "bend",
                    }
                )

        ii = np.arange(ring)[:, None] * np.ones((1, wide), dtype=np.int64)
        jj = np.ones((ring, 1), dtype=np.int64) * np.arange(wide)[None, :]

        # ring-direction stretch (closed loop), prestrained rest length
        add(idx, np.roll(idx, -1, axis=0), "stretch", ii % 2, 1.0 / (1.0 + prestrain))
        # width-direction stretch
        add(idx[:, :-1], idx[:, 1:], "stretch", jj[:, :-1] % 2, 1.0)
        # shear diagonals
        add(idx[:, :-1], np.roll(idx, -1, axis=0)[:, 1:], "shear", ii[:, :-1] % 2, 1.0)
        add(idx[:, 1:], np.roll(idx, -1, axis=0)[:, :-1], "shear", ii[:, 1:] % 2, 1.0)
        # second-neighbour bending (one-sided: resists folding, not stretching)
        add(idx, np.roll(idx, -2, axis=0), "bend", ii % 4, 1.0)
        if wide > 2:
            add(idx[:, :-2], idx[:, 2:], "bend", jj[:, :-2] % 4, 1.0)

        self.batches = batches
        self.n_edges = int(sum(len(b["e0"]) for b in batches))
        offset = 0
        stretch, shear, bend = [], [], []
        for b in batches:
            span = np.arange(offset, offset + len(b["e0"]))
            b["slice"] = slice(offset, offset + len(b["e0"]))
            {"stretch": stretch, "shear": shear, "bend": bend}[b["kind"]].append(span)
            offset += len(b["e0"])
        self.grp_stretch = np.concatenate(stretch)
        self.grp_shear = np.concatenate(shear)
        self.grp_bend = np.concatenate(bend)

        # rest metric for the deformation gradient (material arc lengths)
        du = np.linalg.norm(self.x0.reshape(ring, wide, 3) - np.roll(
            self.x0.reshape(ring, wide, 3), -1, axis=0), axis=2) / (1.0 + prestrain)
        self.rest_u = np.maximum(du, 1e-9)
        xg = self.x0.reshape(ring, wide, 3)
        self.rest_v = np.maximum(np.linalg.norm(xg[:, 1:] - xg[:, :-1], axis=2), 1e-9)

        # neighbour lists used for velocity diffusion
        sm0 = np.concatenate([idx.ravel(), idx[:, :-1].ravel()])
        sm1 = np.concatenate([np.roll(idx, -1, axis=0).ravel(), idx[:, 1:].ravel()])
        self.sm0, self.sm1 = sm0, sm1
        count = np.bincount(sm0, minlength=self.n_cloth) + np.bincount(
            sm1, minlength=self.n_cloth
        )
        self.sm_count = np.maximum(count, 1).astype(float)

    # --------------------------------------------------- prescribed kinematics
    def _cylinder_state(self, t: float):
        """Frozen boundary motion: top lift, then forward / pause / reverse spin."""
        lift0, lift1 = [float(v) for v in self.timeline["lift"]]
        fwd0, fwd1 = [float(v) for v in self.timeline["forward"]]
        rev0, rev1 = [float(v) for v in self.timeline["reverse"]]

        vel = np.zeros((3, 3))
        span = max(lift1 - lift0, 1e-9)
        u = min(max((t - lift0) / span, 0.0), 1.0)
        delta = self.final_centers - self.init_centers
        centers = self.init_centers + delta * _smoothstep(u)
        if 0.0 < u < 1.0:
            vel = delta * (6.0 * u * (1.0 - u) / span)

        w = self.spin_speed
        if t <= fwd0:
            angle = 0.0
            omega = 0.0
        elif t <= fwd1:
            angle = w * (t - fwd0)
            omega = w
        elif t <= rev0:
            angle = w * (fwd1 - fwd0)
            omega = 0.0
        elif t <= rev1:
            angle = w * (fwd1 - fwd0) - w * (t - rev0)
            omega = -w
        else:
            angle = w * (fwd1 - fwd0) - w * (rev1 - rev0)
            omega = 0.0
        angles = self.spin_signs * angle
        omegas = self.spin_signs * omega
        return centers, vel, angles, omegas

    # ---------------------------------------------------------- contact model
    def _project_contacts(self, centers: np.ndarray, normal_acc: np.ndarray) -> None:
        half = 0.5 * self.cyl_length
        for k in range(3):
            r = self.x - centers[k][None, :]
            axial = r @ self.axis
            rp = r - axial[:, None] * self.axis[None, :]
            dist = np.linalg.norm(rp, axis=1)
            safe = np.maximum(dist, 1e-9)
            gap = dist - self.contact_radius
            active = (gap < 0.0) & (np.abs(axial) <= half)
            if not np.any(active):
                continue
            corr = -gap[active]
            nrm = rp[active] / safe[active, None]
            self.x[active] += nrm * corr[:, None]
            normal_acc[active] += corr
            if corr.size:
                self._max_penetration = max(self._max_penetration, float(corr.max()))

    def _apply_friction(
        self,
        centers: np.ndarray,
        cyl_vel: np.ndarray,
        omegas: np.ndarray,
        x_start: np.ndarray,
        normal_acc: np.ndarray,
        dt: float,
    ) -> None:
        """Position-level Coulomb friction: |dx_t| <= mu * |dx_n|."""
        half = 0.5 * self.cyl_length
        tol = 1e-4
        for k in range(3):
            r = self.x - centers[k][None, :]
            axial = r @ self.axis
            rp = r - axial[:, None] * self.axis[None, :]
            dist = np.linalg.norm(rp, axis=1)
            active = (
                (dist < self.contact_radius + tol)
                & (np.abs(axial) <= half)
                & (normal_acc > 0.0)
            )
            if not np.any(active):
                continue
            nrm = rp[active] / np.maximum(dist[active], 1e-9)[:, None]
            omega_vec = omegas[k] * self.axis
            # surface displacement over the substep: translation + rotation
            dx_surf = cyl_vel[k][None, :] * dt + np.cross(
                omega_vec[None, :], rp[active]
            ) * dt
            rel = (self.x[active] - x_start[active]) - dx_surf
            rel_n = np.sum(rel * nrm, axis=1)
            rel_t = rel - rel_n[:, None] * nrm
            mag = np.linalg.norm(rel_t, axis=1)
            limit = self.mu * normal_acc[active]
            scale = np.where(mag <= limit, 1.0, np.divide(limit, np.maximum(mag, 1e-12)))
            corr = -rel_t * scale[:, None]
            idx = np.flatnonzero(active)
            self.x[idx] += corr

    # ----------------------------------------------------------- time stepping
    def _substep(self, t0: float, dt: float) -> None:
        t1 = t0 + dt
        c1, cyl_vel, _, omegas = self._cylinder_state(t1)

        x_start = self.x.copy()
        # explicit prediction with gravity and viscous drag
        self.v += self.gravity[None, :] * dt
        self.v *= 1.0 - self.drag
        self.x = self.x + self.v * dt

        lam = np.zeros(self.n_edges)
        normal_acc = np.zeros(self.n_cloth)
        w_sum = 2.0 / self.node_mass

        for _ in range(self.n_iter):
            for b in self.batches:
                i0, i1, rest = b["e0"], b["e1"], b["rest"]
                sl = b["slice"]
                d = self.x[i0] - self.x[i1]
                length = np.linalg.norm(d, axis=1)
                nrm = d / np.maximum(length, 1e-12)[:, None]
                c = length - rest
                if b["one_sided"]:
                    c = np.minimum(c, 0.0)
                alpha = self.alpha_tilde[sl]
                dlam = (-c - alpha * lam[sl]) / (w_sum + alpha)
                lam[sl] += dlam
                step = (dlam / self.node_mass)[:, None] * nrm
                self.x[i0] += step
                self.x[i1] -= step
            self._project_contacts(c1, normal_acc)

        self._apply_friction(c1, cyl_vel, omegas, x_start, normal_acc, dt)
        self._project_contacts(c1, normal_acc)

        self.v = (self.x - x_start) / dt
        # Jacobi velocity diffusion (numerical viscosity, unconditionally stable)
        acc = np.zeros((self.n_cloth, 3))
        for comp in range(3):
            acc[:, comp] = np.bincount(
                self.sm0, weights=self.v[self.sm1, comp], minlength=self.n_cloth
            ) + np.bincount(self.sm1, weights=self.v[self.sm0, comp], minlength=self.n_cloth)
        self.v += self.vel_smooth * (acc / self.sm_count[:, None] - self.v)
        speed = np.linalg.norm(self.v, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.v[over] *= (self.v_max / speed[over])[:, None]
        self._steps += 1

    # ------------------------------------------------------------- observables
    def _deformation_gradient(self) -> np.ndarray:
        ring, wide = self.ring, self.wide
        xg = self.x.reshape(ring, wide, 3)

        den_u = (self.rest_u + np.roll(self.rest_u, 1, axis=0))[:, :, None]
        du = (np.roll(xg, -1, axis=0) - np.roll(xg, 1, axis=0)) / den_u

        dv = np.empty_like(xg)
        if wide > 2:
            den_v = (self.rest_v[:, 1:] + self.rest_v[:, :-1])[:, :, None]
            dv[:, 1:-1] = (xg[:, 2:] - xg[:, :-2]) / den_v
        dv[:, 0] = (xg[:, 1] - xg[:, 0]) / self.rest_v[:, 0][:, None]
        dv[:, -1] = (xg[:, -1] - xg[:, -2]) / self.rest_v[:, -1][:, None]

        nvec = np.cross(du, dv)
        norm = np.linalg.norm(nvec, axis=2)
        nvec = np.where(norm[:, :, None] > 1e-12, nvec / np.maximum(norm, 1e-12)[:, :, None],
                        np.array([0.0, 0.0, 1.0]))

        f_cloth = np.empty((ring, wide, 3, 3))
        f_cloth[:, :, :, 0] = du
        f_cloth[:, :, :, 1] = dv
        f_cloth[:, :, :, 2] = nvec
        f_cloth = f_cloth.reshape(self.n_cloth, 3, 3)
        return np.concatenate([f_cloth, np.tile(np.eye(3), (3, 1, 1))], axis=0)

    def _signed_distance(self, centers: np.ndarray) -> np.ndarray:
        gaps = np.empty((3, self.n_cloth))
        for k in range(3):
            r = self.x - centers[k][None, :]
            axial = r @ self.axis
            rp = r - axial[:, None] * self.axis[None, :]
            gaps[k] = np.linalg.norm(rp, axis=1) - self.contact_radius
        return np.min(gaps, axis=0)

    def _frame(self, time: float) -> StateFrame:
        centers, cyl_vel, angles, omegas = self._cylinder_state(time)

        position = np.vstack([self.x, centers])
        velocity = np.vstack([self.v, cyl_vel])

        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (self.n_total, 1))
        for k in range(3):
            half = 0.5 * float(angles[k])
            spin = np.array(
                [
                    self.axis[0] * np.sin(half),
                    self.axis[1] * np.sin(half),
                    self.axis[2] * np.sin(half),
                    np.cos(half),
                ]
            )
            q = _quat_mul(spin, self.base_quat)
            orientation[self.n_cloth + k] = q / max(np.linalg.norm(q), 1e-12)

        angular_velocity = np.zeros((self.n_total, 3))
        for k in range(3):
            angular_velocity[self.n_cloth + k] = omegas[k] * self.axis

        mass = np.r_[np.full(self.n_cloth, self.node_mass), np.zeros(3)]
        body_id = np.r_[np.zeros(self.n_cloth, dtype=int), np.asarray([1, 2, 3], dtype=int)]

        frame = StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "uv": self.uv,
                "mass_quantity": mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._signed_distance(centers),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = max(float(time), 0.0)
        n_target = int(round(target / self.dt))
        while self._steps < n_target:
            self._substep(self._steps * self.dt, self.dt)
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped XPBD shell with Coulomb frictional cylinder contact",
            "substeps": int(self._steps),
            "substep_dt": float(self.dt),
            "iterations_per_substep": int(self.n_iter),
            "cloth_nodes": int(self.n_cloth),
            "constraint_count": int(self.n_edges),
            "max_penetration_correction": float(self._max_penetration),
            "friction_coefficient": float(self.mu),
            "final_speed_max": float(np.linalg.norm(self.v, axis=1).max()),
        }
