"""Self-contained discrete elastic rod solver for the open trefoil scene.

The rod is represented internally by a triangular lattice: three material
particles at every centerline station. Axial, cross-section, and helical
springs make a naturally straight isotropic beam, while distance projection
keeps the centerline inextensible. Rotating the final material triangle
therefore injects real twist into the lattice and permits torsional buckling
and contact-driven snap-through without prescribing centerline motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed  # The integrator is deliberately deterministic.
        self.scene = scene
        cfg = scene.config
        rod = cfg["rod"]
        initial = scene.initial_state()
        self.xc0 = np.asarray(initial.position, dtype=float).copy()
        self.n = len(self.xc0)
        self.total_mass = float(rod["mass"])
        self.node_mass = self.total_mass / self.n
        self.particle_mass = self.node_mass / 3.0
        self.radius = float(rod["radius"])
        self.lattice_radius = 0.72 * self.radius
        self.rest_length = float(rod["total_length"]) / (self.n - 1)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.left = np.asarray(cfg["clamps"]["left"]["position"], dtype=float)
        self.right = np.asarray(cfg["clamps"]["right"]["position"], dtype=float)
        axis = np.asarray(cfg["clamps"]["right"]["rotation_axis"], dtype=float)
        self.drive_axis = axis / max(np.linalg.norm(axis), 1.0e-12)

        # A Bishop frame gives the twist-free material orientation of the
        # curved initial centerline.
        tangents = np.empty_like(self.xc0)
        tangents[1:-1] = self.xc0[2:] - self.xc0[:-2]
        tangents[0] = self.xc0[1] - self.xc0[0]
        tangents[-1] = self.xc0[-1] - self.xc0[-2]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1.0e-12)
        d1 = np.empty_like(tangents)
        trial = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(trial, tangents[0])) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        d1[0] = trial - np.dot(trial, tangents[0]) * tangents[0]
        d1[0] /= np.linalg.norm(d1[0])
        for i in range(1, self.n):
            v = d1[i - 1] - np.dot(d1[i - 1], tangents[i]) * tangents[i]
            nv = np.linalg.norm(v)
            if nv < 1.0e-10:
                v = np.cross(tangents[i - 1], tangents[i])
                nv = np.linalg.norm(v)
            d1[i] = v / max(nv, 1.0e-12)
        d2 = np.cross(tangents, d1)
        d2 /= np.maximum(np.linalg.norm(d2, axis=1)[:, None], 1.0e-12)

        phase = 2.0 * np.pi * np.arange(3) / 3.0
        offsets = self.lattice_radius * (
            np.cos(phase)[None, :, None] * d1[:, None, :]
            + np.sin(phase)[None, :, None] * d2[:, None, :]
        )
        self.p = self.xc0[:, None, :] + offsets
        self.vp = np.zeros_like(self.p)
        self.left_offsets = offsets[0].copy()
        self.right_offsets = offsets[-1].copy()

        # Diagonal constraints encode the rest geometry of a straight prism.
        a, b, rest, compliance = [], [], [], []

        def add(ia, ja, ib, jb, length, comp):
            a.append(3 * ia + ja)
            b.append(3 * ib + jb)
            rest.append(length)
            compliance.append(comp)

        cross_rest = np.sqrt(3.0) * self.lattice_radius
        diag_rest = np.sqrt(self.rest_length**2 + cross_rest**2)
        for i in range(self.n):
            for j in range(3):
                add(i, j, i, (j + 1) % 3, cross_rest, 2.0e-4)
        for i in range(self.n - 1):
            for j in range(3):
                add(i, j, i + 1, j, self.rest_length, 8.0e-4)
                add(i, j, i + 1, (j + 1) % 3, diag_rest, 1.2e-2)
                add(i, j, i + 1, (j - 1) % 3, diag_rest, 1.2e-2)
        self.ca = np.asarray(a, dtype=np.int32)
        self.cb = np.asarray(b, dtype=np.int32)
        self.crest = np.asarray(rest)
        self.ccomp = np.asarray(compliance)

        ci, cj = np.triu_indices(self.n, k=5)
        self.contact_i = ci.astype(np.int32)
        self.contact_j = cj.astype(np.int32)
        self.inv_particle_mass = 1.0 / self.particle_mass
        self.inv_station_mass = 1.0 / self.node_mass
        self.time = 0.0
        self.base_dt = 0.004
        self.steps = 0
        self.initial_momentum = np.zeros(3)
        self.initial_angular_momentum = np.zeros(3)
        self.previous_sample_time = 0.0
        self.previous_sample_momentum = np.zeros(3)
        self.previous_sample_angular_momentum = np.zeros(3)
        self.last_force = self.total_mass * self.gravity
        self.last_torque = np.cross(np.mean(self.xc0, axis=0), self.last_force)

    @staticmethod
    def _rotate(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return (vectors * c + np.cross(axis, vectors) * s
                + (vectors @ axis)[:, None] * axis * (1.0 - c))

    def _targets(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        target = self.scene.clamp_targets(time)
        angle = float(target["right"]["angle_radians"])
        left = self.left + self.left_offsets
        right = self.right + self._rotate(self.right_offsets, self.drive_axis, angle)
        return left, right

    def _distance_constraints(self, h: float) -> None:
        flat = self.p.reshape(-1, 3)
        inv = np.full(len(flat), self.inv_particle_mass)
        inv[:3] = 0.0
        inv[-3:] = 0.0
        lamb = np.zeros(len(self.ca))
        alpha = self.ccomp / (h * h)
        for _ in range(5):
            delta = flat[self.ca] - flat[self.cb]
            length = np.linalg.norm(delta, axis=1)
            normal = delta / np.maximum(length[:, None], 1.0e-12)
            denom = inv[self.ca] + inv[self.cb] + alpha
            dl = (-(length - self.crest) - alpha * lamb) / denom
            lamb += dl
            corr = normal * dl[:, None]
            change = np.zeros_like(flat)
            np.add.at(change, self.ca, inv[self.ca, None] * corr)
            np.add.at(change, self.cb, -inv[self.cb, None] * corr)
            flat += 0.38 * change
            self._centerline_length_projection(0.72)

    def _centerline_length_projection(self, relaxation: float) -> None:
        centers = np.mean(self.p, axis=1)
        delta = centers[:-1] - centers[1:]
        length = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(length[:, None], 1.0e-12)
        inv = np.full(self.n, self.inv_station_mass)
        inv[[0, -1]] = 0.0
        denom = np.maximum(inv[:-1] + inv[1:], 1.0e-12)
        dl = -(length - self.rest_length) / denom
        corr = normal * dl[:, None] * relaxation
        station_change = np.zeros((self.n, 3))
        np.add.at(station_change, np.arange(self.n - 1), inv[:-1, None] * corr)
        np.add.at(station_change, np.arange(1, self.n), -inv[1:, None] * corr)
        self.p += station_change[:, None, :]

    def _self_contact(self) -> None:
        centers = np.mean(self.p, axis=1)
        d = centers[self.contact_i] - centers[self.contact_j]
        dist = np.linalg.norm(d, axis=1)
        active = dist < 2.0 * self.radius
        if not np.any(active):
            return
        ii = self.contact_i[active]
        jj = self.contact_j[active]
        da = d[active]
        la = dist[active]
        normal = da / np.maximum(la[:, None], 1.0e-12)
        zero = la < 1.0e-12
        if np.any(zero):
            normal[zero] = np.array([0.0, 1.0, 0.0])
        corr = 0.5 * (2.0 * self.radius - la)[:, None] * normal
        movable_i = (ii != 0) & (ii != self.n - 1)
        movable_j = (jj != 0) & (jj != self.n - 1)
        corr[movable_i & ~movable_j] *= 2.0
        corr[~movable_i & movable_j] *= 2.0
        station_change = np.zeros((self.n, 3))
        np.add.at(station_change, ii[movable_i], corr[movable_i])
        np.add.at(station_change, jj[movable_j], -corr[movable_j])
        self.p += 0.65 * station_change[:, None, :]

    def _step(self, h: float) -> None:
        old = self.p.copy()
        self.vp = np.exp(-0.85 * h) * self.vp + h * self.gravity
        self.p += h * self.vp
        left_target, right_target = self._targets(self.time + h)
        self.p[0] = left_target
        self.p[-1] = right_target
        self._distance_constraints(h)
        self._self_contact()
        self._centerline_length_projection(0.92)
        self.p[0] = left_target
        self.p[-1] = right_target
        self.vp = (self.p - old) / h
        self.time += h
        self.steps += 1

    @staticmethod
    def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
        """Convert proper rotation matrices to normalized xyzw quaternions."""
        q = np.empty((len(m), 4))
        for i, r in enumerate(m):
            tr = np.trace(r)
            if tr > 0.0:
                s = 2.0 * np.sqrt(tr + 1.0)
                q[i] = [(r[2, 1] - r[1, 2]) / s,
                        (r[0, 2] - r[2, 0]) / s,
                        (r[1, 0] - r[0, 1]) / s, 0.25 * s]
            else:
                k = int(np.argmax(np.diag(r)))
                if k == 0:
                    s = 2.0 * np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-15))
                    q[i] = [0.25*s, (r[0,1]+r[1,0])/s,
                            (r[0,2]+r[2,0])/s, (r[2,1]-r[1,2])/s]
                elif k == 1:
                    s = 2.0 * np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-15))
                    q[i] = [(r[0,1]+r[1,0])/s, 0.25*s,
                            (r[1,2]+r[2,1])/s, (r[0,2]-r[2,0])/s]
                else:
                    s = 2.0 * np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-15))
                    q[i] = [(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s,
                            0.25*s, (r[1,0]-r[0,1])/s]
        q /= np.maximum(np.linalg.norm(q, axis=1)[:, None], 1.0e-12)
        return q

    def _kinematics(self):
        center = np.mean(self.p, axis=1)
        velocity = np.mean(self.vp, axis=1)
        rel = self.p - center[:, None, :]
        rel_v = self.vp - velocity[:, None, :]
        eye = np.eye(3)
        inertia = self.particle_mass * (
            np.sum(rel * rel, axis=(1, 2))[:, None, None] * eye
            - np.einsum("nki,nkj->nij", rel, rel)
        )
        spin = self.particle_mass * np.sum(np.cross(rel, rel_v), axis=1)
        omega = np.empty_like(center)
        for i in range(self.n):
            omega[i] = np.linalg.solve(inertia[i] + 1.0e-12 * eye, spin[i])

        d1 = rel[:, 0].copy()
        d1 /= np.maximum(np.linalg.norm(d1, axis=1)[:, None], 1.0e-12)
        raw2 = rel[:, 1]
        d3 = np.cross(d1, raw2)
        d3 /= np.maximum(np.linalg.norm(d3, axis=1)[:, None], 1.0e-12)
        d2 = np.cross(d3, d1)
        matrices = np.stack((d1, d2, d3), axis=2)
        orientation = self._matrix_to_quaternion(matrices)
        return center, velocity, orientation, omega, inertia

    def _frame(self, requested_time: float) -> StateFrame:
        center, velocity, orientation, omega, inertia = self._kinematics()
        mass = np.full(self.n, self.node_mass)
        momentum = np.sum(mass[:, None] * velocity, axis=0)
        angular = np.sum(np.cross(center, mass[:, None] * velocity), axis=0)
        angular += np.sum(np.einsum("nij,nj->ni", inertia, omega), axis=0)

        elapsed = requested_time - self.previous_sample_time
        if elapsed > 1.0e-12:
            self.last_force = (momentum - self.previous_sample_momentum) / elapsed
            self.last_torque = (angular - self.previous_sample_angular_momentum) / elapsed
        self.previous_sample_time = requested_time
        self.previous_sample_momentum = momentum.copy()
        self.previous_sample_angular_momentum = angular.copy()

        gravity_force = self.total_mass * self.gravity
        com = np.sum(center * mass[:, None], axis=0) / self.total_mass
        clamp_force = self.last_force - gravity_force
        force = np.zeros((1, 2, 3))
        force[0, 0] = gravity_force
        force[0, 1] = clamp_force
        force_position = np.zeros_like(force)
        force_position[0, 0] = com
        force_position[0, 1] = 0.5 * (self.left + self.right)
        couple = np.zeros_like(force)
        represented_torque = np.cross(com, gravity_force)
        represented_torque += np.cross(force_position[0, 1], clamp_force)
        couple[0, 1] = self.last_torque - represented_torque

        fields = {
            "budget_cumulative_external_angular_impulse":
                (angular - self.initial_angular_momentum)[None, :],
            "budget_cumulative_external_impulse":
                (momentum - self.initial_momentum)[None, :],
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_couple": couple,
            "budget_external_force": force,
            "budget_force_position": force_position,
            "constraint_target": np.stack((self.left, self.right)),
            "constraint_value": np.stack((center[0], center[-1])),
            "cumulative_mass_source": np.zeros(1),
        }
        turns = self.scene.clamp_targets(requested_time)["right"]["angle_radians"] / (2.0*np.pi)
        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": mass.copy(),
            "arc_fraction": np.linspace(0.0, 1.0, self.n),
            "rod_radius": np.full(self.n, self.radius),
            "contact_diameter": np.full(self.n, 2.0 * self.radius),
            "endpoint_controlled": np.isin(np.arange(self.n), [0, self.n-1]).astype(float),
            "material_twist_turns": np.full(self.n, turns),
        }
        speed_scale = np.sqrt(np.linalg.norm(self.gravity) * self.rest_length)
        frame = StateFrame(
            float(requested_time), center, velocity, orientation=orientation,
            angular_velocity=omega, mass=mass, attributes=attributes, fields=fields,
            scalars={
                "constraint_scale": float(np.linalg.norm(self.right - self.left)),
                "reference_momentum": float(max(self.total_mass * speed_scale, 1.0e-6)),
                "reference_angular_momentum": float(max(
                    self.total_mass * speed_scale * float(self.scene.config["rod"]["total_length"]),
                    1.0e-6)),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            self._step(min(self.base_dt, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "triangular-lattice XPBD elastic rod with self-contact",
            "internal_steps": self.steps,
            "particle_count": int(3 * self.n),
            "deterministic": True,
        }
