"""Deterministic six-DOF rigid-body solver for the rings-over-poles scene.

Contact uses the actual axisymmetric wall profile as a swept capsule, preserving
the central hole. A damped penalty/Coulomb model and semi-implicit quaternion
integration advance the bodies without an external physics engine.
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


def _rotations(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from xyzw quaternions."""
    x, y, z, w = q.T
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1-2*(yy+zz); r[:, 0, 1] = 2*(xy-wz); r[:, 0, 2] = 2*(xz+wy)
    r[:, 1, 0] = 2*(xy+wz); r[:, 1, 1] = 1-2*(xx+zz); r[:, 1, 2] = 2*(yz-wx)
    r[:, 2, 0] = 2*(xz-wy); r[:, 2, 1] = 2*(yz+wx); r[:, 2, 2] = 1-2*(xx+yy)
    return r


def _quat_step(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    angle = np.linalg.norm(omega, axis=1)*dt
    half = 0.5*angle
    scale = np.empty_like(angle)
    small = angle < 1e-10
    scale[small] = 0.5*dt
    scale[~small] = np.sin(half[~small])/np.linalg.norm(omega[~small], axis=1)
    av, aw = omega*scale[:, None], np.cos(half)
    bv, bw = q[:, :3], q[:, 3]
    out = np.empty_like(q)
    out[:, :3] = aw[:, None]*bv + bw[:, None]*av + np.cross(av, bv)
    out[:, 3] = aw*bw - np.einsum("ij,ij->i", av, bv)
    out /= np.linalg.norm(out, axis=1)[:, None]
    return out


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.n = len(initial.position)
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.orientation = (np.tile([0., 0., 0., 1.], (self.n, 1)) if
                            initial.orientation is None else
                            np.asarray(initial.orientation, dtype=float).copy())
        self.orientation /= np.linalg.norm(self.orientation, axis=1)[:, None]
        self.omega = (np.zeros((self.n, 3)) if initial.angular_velocity is None else
                      np.asarray(initial.angular_velocity, dtype=float).copy())

        objects = scene.config.get("scene_geometry", {}).get("objects", [])
        ring = next((o for o in objects if o.get("id") == "ring_collection"), None)
        if ring is None:
            raise ValueError("ring_collection geometry is required")
        geo = ring["geometry"]
        profile = np.asarray(geo["inner_profile"], dtype=float)
        self.wall = float(geo["wall_thickness"])
        self.half_wall = 0.5*self.wall
        self.com_offset = 0.5*(profile[0, 1]+profile[-1, 1])
        self.profile = profile.copy()
        self.profile[:, 0] += self.half_wall
        self.profile[:, 1] -= self.com_offset
        segments = self.profile[1:]-self.profile[:-1]
        self.seg_len_sq = np.einsum("ij,ij->i", segments, segments)

        angles = np.linspace(0., 2.*math.pi, 12, endpoint=False)
        self.samples_local = np.asarray([
            [radius*math.cos(a), radius*math.sin(a), z]
            for radius, z in self.profile for a in angles
        ])
        rot = _rotations(self.orientation)
        self.position = (np.asarray(initial.position, dtype=float) +
                         np.einsum("nij,j->ni", rot, [0., 0., self.com_offset]))

        r2 = float(np.mean(self.profile[:, 0]**2))
        z2 = float(np.mean(self.profile[:, 1]**2))
        self.inertia_body = np.column_stack((self.mass*(0.5*r2+z2),
                                              self.mass*(0.5*r2+z2),
                                              self.mass*r2))
        self.inertia_body = np.maximum(self.inertia_body, 1e-4)

        self.poles = []
        for obj in objects:
            g = obj.get("geometry", {})
            if obj.get("motion") == "static" and g.get("type") == "cylinder":
                tr = np.asarray(obj.get("transform", {}).get("translation", [0, 0, 0]), float)
                h = float(g["height"])
                self.poles.append((tr[:2].copy(), tr[2]-0.5*h, tr[2]+0.5*h,
                                   float(g["radius"])))

        self.time = 0.0
        self.base_dt = 0.0025
        self.gravity = np.array([0., 0., -9.81])
        self.body_id = np.arange(self.n, dtype=np.int64)
        self.steps = 0
        self._last_gaps = np.empty(0)
        self._last_interactions = np.empty((0, 3))
        self._last_external = self.mass[:, None]*self.gravity
        self._compute_forces(self.base_dt)

    def _world_geometry(self):
        rot = _rotations(self.orientation)
        samples = self.position[:, None, :] + np.einsum(
            "nij,mj->nmi", rot, self.samples_local)
        return rot, samples

    def _profile_query(self, points: np.ndarray, center: np.ndarray,
                       rotation: np.ndarray):
        local = (points-center) @ rotation
        radial = np.linalg.norm(local[:, :2], axis=1)
        query = np.column_stack((radial, local[:, 2]))
        start = self.profile[:-1]
        segment = self.profile[1:]-start
        delta = query[:, None, :]-start[None, :, :]
        fraction = np.einsum("msd,sd->ms", delta, segment)/self.seg_len_sq[None, :]
        np.clip(fraction, 0., 1., out=fraction)
        closest = start[None, :, :]+fraction[:, :, None]*segment[None, :, :]
        diff = query[:, None, :]-closest
        d2 = np.einsum("msd,msd->ms", diff, diff)
        pi, si = np.unravel_index(int(np.argmin(d2)), d2.shape)
        cp = closest[pi, si]
        if radial[pi] > 1e-12:
            xy = local[pi, :2]*(cp[0]/radial[pi])
        else:
            xy = np.array([cp[0], 0.])
        closest_world = center + rotation @ np.array([xy[0], xy[1], cp[1]])
        return math.sqrt(max(float(d2[pi, si]), 0.))-self.wall, points[pi], closest_world

    @staticmethod
    def _cylinder_sdf(points: np.ndarray, pole):
        xy0, lower, upper, radius = pole
        rel = points[:, :2]-xy0[None, :]
        radial = np.linalg.norm(rel, axis=1)
        qr = radial-radius
        mid, hh = 0.5*(lower+upper), 0.5*(upper-lower)
        dz = points[:, 2]-mid
        qz = np.abs(dz)-hh
        outr, outz = np.maximum(qr, 0.), np.maximum(qz, 0.)
        sdf = np.hypot(outr, outz)+np.minimum(np.maximum(qr, qz), 0.)
        nr = np.zeros((len(points), 3))
        valid = radial > 1e-12
        nr[valid, :2] = rel[valid]/radial[valid, None]
        nr[~valid, 0] = 1.
        nz = np.zeros_like(nr); nz[:, 2] = np.where(dz >= 0., 1., -1.)
        normal = np.empty_like(nr)
        corner = (qr > 0.) & (qz > 0.)
        normal[corner] = ((nr[corner]*outr[corner, None]+nz[corner]*outz[corner, None]) /
                          np.hypot(outr[corner], outz[corner])[:, None])
        use_r = (~corner) & (qr >= qz)
        normal[use_r] = nr[use_r]
        normal[(~corner) & (~use_r)] = nz[(~corner) & (~use_r)]
        return sdf, normal

    def _add_contact(self, force, torque, i, j, point_i, point_j, normal,
                     gap, stiffness, dt, gaps, interactions):
        gaps.append(float(gap))
        vi = self.velocity[i]+np.cross(self.omega[i], point_i-self.position[i])
        if j is None:
            vj = np.zeros(3); effective_mass = self.mass[i]
        else:
            vj = self.velocity[j]+np.cross(self.omega[j], point_j-self.position[j])
            effective_mass = self.mass[i]*self.mass[j]/(self.mass[i]+self.mass[j])
        relative = vi-vj
        vn = float(np.dot(relative, normal))
        fn = 0.
        if gap < 0.:
            damping = 1.5*math.sqrt(stiffness*effective_mass)
            fn = min(max(0., -stiffness*gap-damping*vn), 3000.)
        tangent = relative-vn*normal
        tangent_speed = float(np.linalg.norm(tangent))
        friction = np.zeros(3)
        if fn > 0. and tangent_speed > 1e-10:
            mag = min(0.58*fn, effective_mass*tangent_speed/max(dt, 1e-8))
            friction = -mag*tangent/tangent_speed
        contact_force = fn*normal+friction
        force[i] += contact_force
        torque[i] += np.cross(point_i-self.position[i], contact_force)
        if j is not None:
            force[j] -= contact_force
            torque[j] += np.cross(point_j-self.position[j], -contact_force)
        interactions.append(contact_force.copy())

    def _compute_forces(self, dt: float):
        rotations, samples = self._world_geometry()
        external = self.mass[:, None]*self.gravity-0.10*self.mass[:, None]*self.velocity
        force = external.copy()
        torque = -0.08*self.omega
        gaps, interactions = [], []
        near = 0.035

        for i in range(self.n):
            sg = samples[i, :, 2]-self.half_wall
            minimum = float(np.min(sg))
            if minimum < near:
                active = np.flatnonzero(sg <= minimum+0.0015)
                stiffness = 10000./max(len(active), 1)
                for index in active:
                    pi = samples[i, index].copy(); pi[2] -= self.half_wall
                    pj = pi.copy(); pj[2] = 0.
                    self._add_contact(force, torque, i, None, pi, pj,
                                      np.array([0., 0., 1.]), float(sg[index]),
                                      stiffness, dt, gaps, interactions)

        for i in range(self.n):
            for pole in self.poles:
                sdf, normals = self._cylinder_sdf(samples[i], pole)
                sg = sdf-self.half_wall
                index = int(np.argmin(sg)); gap = float(sg[index])
                if gap < near:
                    normal = normals[index]
                    pi = samples[i, index]-self.half_wall*normal
                    pj = samples[i, index]-float(sdf[index])*normal
                    self._add_contact(force, torque, i, None, pi, pj, normal,
                                      gap, 9000., dt, gaps, interactions)

        delta = self.position[:, None, :]-self.position[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        cand_i, cand_j = np.where(np.triu(d2 < 0.82**2, k=1))
        for i, j in zip(cand_i.tolist(), cand_j.tolist()):
            gap1, pi1, pj1 = self._profile_query(samples[i], self.position[j], rotations[j])
            gap2, pj2, pi2 = self._profile_query(samples[j], self.position[i], rotations[i])
            if gap1 <= gap2:
                gap, pi, pj = gap1, pi1, pj1
            else:
                gap, pi, pj = gap2, pi2, pj2
            if gap >= near:
                continue
            direction = pi-pj
            length = float(np.linalg.norm(direction))
            if length > 1e-10:
                normal = direction/length
            else:
                direction = self.position[i]-self.position[j]
                length = float(np.linalg.norm(direction))
                normal = direction/length if length > 1e-10 else np.array([1., 0., 0.])
            self._add_contact(force, torque, i, j, pi, pj, normal, float(gap),
                              7500., dt, gaps, interactions)

        self._last_gaps = np.asarray(gaps, dtype=float)
        self._last_interactions = (np.asarray(interactions, dtype=float).reshape(-1, 3)
                                   if interactions else np.empty((0, 3)))
        self._last_external = external
        return force, torque

    def _step(self, dt: float) -> None:
        force, torque = self._compute_forces(dt)
        self.velocity += dt*force/self.mass[:, None]
        rot = _rotations(self.orientation)
        inertia = np.einsum("nij,nj,nkj->nik", rot, self.inertia_body, rot)
        momentum = np.einsum("nij,nj->ni", inertia, self.omega)
        rhs = torque-np.cross(self.omega, momentum)
        self.omega += dt*np.linalg.solve(inertia, rhs[..., None])[..., 0]
        speed = np.linalg.norm(self.velocity, axis=1)
        mask = speed > 25.; self.velocity[mask] *= (25./speed[mask])[:, None]
        angular_speed = np.linalg.norm(self.omega, axis=1)
        mask = angular_speed > 45.; self.omega[mask] *= (45./angular_speed[mask])[:, None]
        self.position += dt*self.velocity
        self.orientation = _quat_step(self.orientation, self.omega, dt)
        self.time += dt
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time-1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target-1e-12:
            self._step(min(self.base_dt, target-self.time))
        self.time = target
        self._compute_forces(self.base_dt)
        rot = _rotations(self.orientation)
        anchor = self.position-np.einsum("nij,j->ni", rot, [0., 0., self.com_offset])
        return StateFrame(
            time=target, position=anchor.copy(), velocity=self.velocity.copy(),
            orientation=self.orientation.copy(), angular_velocity=self.omega.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            fields={"contact_signed_distance": self._last_gaps.copy(),
                    "external_force": self._last_external.copy(),
                    "interaction_force": self._last_interactions.copy()},
            scalars={"contact_length_scale": self.wall})

    def diagnostics(self) -> dict[str, object]:
        kinetic = 0.5*np.sum(self.mass[:, None]*self.velocity**2)
        return {"integrator": "semi_implicit_rigid_penalty",
                "internal_steps": int(self.steps),
                "final_translational_kinetic_energy": float(kinetic)}
