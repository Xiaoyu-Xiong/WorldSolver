"""Rigid Card House Block Impacts Physics Solver.

Simulates 3D rigid body dynamics with contact, friction, and restitution
for a card house subjected to falling block impacts.
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n > 1e-12:
        x, y, z, w = x / n, y / n, z / n, w / n
    else:
        return np.eye(3, dtype=float)

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate xyzw quaternion with angular velocity omega over dt."""
    w_len = np.linalg.norm(omega)
    if w_len > 1e-10:
        axis = omega / w_len
        angle = w_len * dt
        dq = np.array([
            axis[0] * math.sin(angle * 0.5),
            axis[1] * math.sin(angle * 0.5),
            axis[2] * math.sin(angle * 0.5),
            math.cos(angle * 0.5)
        ], dtype=float)
        # Quaternion multiplication: dq * q
        x1, y1, z1, w1 = dq
        x2, y2, z2, w2 = q
        new_q = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dtype=float)
    else:
        new_q = q.copy()

    n = np.linalg.norm(new_q)
    if n > 1e-12:
        new_q /= n
    return new_q


class RigidBody:
    def __init__(self, body_id: int, mass: float, half_extents: np.ndarray,
                 pos: np.ndarray, quat: np.ndarray, release_time: float, is_block: bool):
        self.body_id = body_id
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        self.half_extents = np.array(half_extents, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.quat = np.array(quat, dtype=float)  # xyzw
        self.vel = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.release_time = release_time
        self.is_block = is_block
        self.active = (release_time <= 0.0)

        # Principal moments of inertia for a box
        lx, ly, lz = 2.0 * self.half_extents
        Ixx = (1.0 / 12.0) * mass * (ly * ly + lz * lz)
        Iyy = (1.0 / 12.0) * mass * (lx * lx + lz * lz)
        Izz = (1.0 / 12.0) * mass * (lx * lx + ly * ly)
        self.I_body = np.diag([Ixx, Iyy, Izz])
        self.inv_I_body = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz])

        # Local corner vertices of the box (8 corners)
        self.local_vertices = []
        for sx in [-1.0, 1.0]:
            for sy in [-1.0, 1.0]:
                for sz in [-1.0, 1.0]:
                    self.local_vertices.append(np.array([
                        sx * self.half_extents[0],
                        sy * self.half_extents[1],
                        sz * self.half_extents[2]
                    ], dtype=float))
        self.local_vertices = np.array(self.local_vertices, dtype=float)

    def rot(self) -> np.ndarray:
        return quat_to_rot(self.quat)

    def inv_inertia_world(self) -> np.ndarray:
        R = self.rot()
        return R @ self.inv_I_body @ R.T

    def get_world_vertices(self) -> np.ndarray:
        R = self.rot()
        return (self.local_vertices @ R.T) + self.pos


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.gravity = np.array(self.config["gravity"], dtype=float)
        self.mu = float(self.config["contact_parameters"]["friction_coefficient"])
        self.restitution = float(self.config["contact_parameters"]["restitution"])
        self.release_times = list(self.config["initial_conditions"]["impact_block_release_times"])

        init_frame = scene.initial_state()
        self.N = 13
        self.bodies: list[RigidBody] = []

        half_extents_list = [
            [0.45, 0.18, 0.015],  # 0: card_lower_left_outer
            [0.45, 0.18, 0.015],  # 1: card_lower_left_inner
            [0.45, 0.18, 0.015],  # 2: card_lower_right_inner
            [0.45, 0.18, 0.015],  # 3: card_lower_right_outer
            [0.45, 0.18, 0.015],  # 4: card_deck_left
            [0.45, 0.18, 0.015],  # 5: card_deck_right
            [0.45, 0.18, 0.015],  # 6: card_upper_left_outer
            [0.45, 0.18, 0.015],  # 7: card_upper_left_inner
            [0.45, 0.18, 0.015],  # 8: card_upper_right_inner
            [0.45, 0.18, 0.015],  # 9: card_upper_right_outer
            [0.16, 0.16, 0.16],   # 10: impact_block_left
            [0.16, 0.16, 0.16],   # 11: impact_block_right
            [0.19, 0.19, 0.19],   # 12: impact_block_center
        ]

        masses = init_frame.mass
        positions = init_frame.position
        orientations = init_frame.orientation
        rel_times = init_frame.attributes["release_time"]
        is_blocks = init_frame.attributes["is_impact_block"]

        for i in range(self.N):
            body = RigidBody(
                body_id=i,
                mass=float(masses[i]),
                half_extents=np.array(half_extents_list[i], dtype=float),
                pos=positions[i].copy(),
                quat=orientations[i].copy(),
                release_time=float(rel_times[i]),
                is_block=bool(is_blocks[i] > 0.5)
            )
            self.bodies.append(body)

        self.current_time = 0.0
        self.sub_dt = 0.001  # 1ms substepping for stability
        self.contact_length_scale = 0.03  # Card thickness scale

        # For tracking contact forces and gaps during frames
        self.last_contact_distances = np.zeros(0, dtype=float)
        self.last_interaction_forces = np.zeros((0, 3), dtype=float)
        self.last_external_forces = np.zeros((self.N, 3), dtype=float)

    def _detect_contacts(self):
        """Find contact points between all dynamic bodies and floor, and between dynamic pairs."""
        contacts = []  # list of dicts: body_a, body_b, r_a, r_b, normal, dist

        # 1. Body vs Floor (z = 0, normal = [0, 0, 1])
        floor_n = np.array([0.0, 0.0, 1.0], dtype=float)
        for b in self.bodies:
            if not b.active:
                continue
            w_verts = b.get_world_vertices()
            for v in w_verts:
                dist = v[2]  # floor is at z = 0
                if dist < 0.05:  # contact threshold (includes active/near-contact)
                    contacts.append({
                        'body_a': b,
                        'body_b': None,
                        'point': v.copy(),
                        'r_a': v - b.pos,
                        'r_b': np.zeros(3),
                        'normal': floor_n.copy(),
                        'dist': dist
                    })

        # 2. Body vs Body
        for i in range(self.N):
            b_a = self.bodies[i]
            if not b_a.active:
                continue
            R_a = b_a.rot()
            inv_R_a = R_a.T
            h_a = b_a.half_extents

            for j in range(i + 1, self.N):
                b_b = self.bodies[j]
                if not b_b.active:
                    continue

                # Broadphase sphere test
                rad_a = np.linalg.norm(b_a.half_extents)
                rad_b = np.linalg.norm(b_b.half_extents)
                d_ab = np.linalg.norm(b_a.pos - b_b.pos)
                if d_ab > rad_a + rad_b + 0.05:
                    continue

                R_b = b_b.rot()
                inv_R_b = R_b.T
                h_b = b_b.half_extents

                # Test vertices of b_a against box b_b
                w_verts_a = b_a.get_world_vertices()
                for v in w_verts_a:
                    v_local = inv_R_b @ (v - b_b.pos)
                    # Signed distances to 6 faces of b_b
                    d = np.abs(v_local) - h_b
                    max_d = np.max(d)
                    if max_d < 0.03:  # near contact or penetration
                        # Find deepest penetrating face or closest face
                        axis = np.argmax(d)
                        sign = 1.0 if v_local[axis] >= 0.0 else -1.0
                        local_n = np.zeros(3)
                        local_n[axis] = sign
                        world_n = R_b @ local_n
                        contacts.append({
                            'body_a': b_a,
                            'body_b': b_b,
                            'point': v.copy(),
                            'r_a': v - b_a.pos,
                            'r_b': v - b_b.pos,
                            'normal': -world_n,  # points from B to A
                            'dist': max_d
                        })

                # Test vertices of b_b against box b_a
                w_verts_b = b_b.get_world_vertices()
                for v in w_verts_b:
                    v_local = inv_R_a @ (v - b_a.pos)
                    d = np.abs(v_local) - h_a
                    max_d = np.max(d)
                    if max_d < 0.03:
                        axis = np.argmax(d)
                        sign = 1.0 if v_local[axis] >= 0.0 else -1.0
                        local_n = np.zeros(3)
                        local_n[axis] = sign
                        world_n = R_a @ local_n
                        contacts.append({
                            'body_a': b_a,
                            'body_b': b_b,
                            'point': v.copy(),
                            'r_a': v - b_a.pos,
                            'r_b': v - b_b.pos,
                            'normal': world_n,  # points from B to A
                            'dist': max_d
                        })

        return contacts

    def _step(self, dt: float):
        """Single physics substep with sequential impulse contact solver."""
        # Check releases
        for b in self.bodies:
            if not b.active and self.current_time >= b.release_time - 1e-7:
                b.active = True

        # Apply external forces (gravity)
        for b in self.bodies:
            if b.active:
                b.vel += self.gravity * dt
                # Apply small damping for physical stability
                b.vel *= (1.0 - 0.01 * dt)
                b.omega *= (1.0 - 0.02 * dt)

        # Contact detection
        contacts = self._detect_contacts()

        # Prepare contact data for solver
        # Solver iterations for sequential impulses
        num_iterations = 15

        # We also store impulses for interaction force output
        contact_impulses = [0.0] * len(contacts)
        contact_tangent_impulses = [np.zeros(3) for _ in contacts]

        for it in range(num_iterations):
            for idx, c in enumerate(contacts):
                b_a = c['body_a']
                b_b = c['body_b']
                n = c['normal']
                r_a = c['r_a']
                r_b = c['r_b']
                dist = c['dist']

                # Velocity at contact point
                v_a = b_a.vel + np.cross(b_a.omega, r_a)
                if b_b is not None:
                    v_b = b_b.vel + np.cross(b_b.omega, r_b)
                else:
                    v_b = np.zeros(3)

                v_rel = v_a - v_b
                vn = np.dot(v_rel, n)

                # Baumgarte stabilization for penetration recovery
                bias = 0.0
                if dist < 0.0:
                    bias = -0.2 * dist / dt  # push out penetration

                # Effective normal mass
                inv_m_a = b_a.inv_mass
                inv_I_a = b_a.inv_inertia_world()
                rn_a = np.cross(r_a, n)
                k_norm = inv_m_a + np.dot(rn_a, inv_I_a @ rn_a)

                if b_b is not None:
                    inv_m_b = b_b.inv_mass
                    inv_I_b = b_b.inv_inertia_world()
                    rn_b = np.cross(r_b, n)
                    k_norm += inv_m_b + np.dot(rn_b, inv_I_b @ rn_b)

                if k_norm < 1e-10:
                    continue

                # Normal impulse delta
                target_vn = 0.0
                if it == 0 and vn < -0.1:
                    target_vn = -self.restitution * vn

                delta_vn = target_vn - vn + bias
                impulse_mag = delta_vn / k_norm

                # Clamp accumulated normal impulse to >= 0
                old_impulse = contact_impulses[idx]
                new_impulse = max(0.0, old_impulse + impulse_mag)
                delta_p_norm = new_impulse - old_impulse
                contact_impulses[idx] = new_impulse

                # Apply normal impulse
                P_norm = delta_p_norm * n
                b_a.vel += inv_m_a * P_norm
                b_a.omega += inv_I_a @ np.cross(r_a, P_norm)
                if b_b is not None:
                    b_b.vel -= inv_m_b * P_norm
                    b_b.omega -= inv_I_b @ np.cross(r_b, P_norm)

                # Friction impulse
                v_a = b_a.vel + np.cross(b_a.omega, r_a)
                if b_b is not None:
                    v_b = b_b.vel + np.cross(b_b.omega, r_b)
                v_rel = v_a - v_b
                v_t = v_rel - np.dot(v_rel, n) * n
                vt_len = np.linalg.norm(v_t)

                if vt_len > 1e-6:
                    t_dir = v_t / vt_len
                    rt_a = np.cross(r_a, t_dir)
                    k_tangent = inv_m_a + np.dot(rt_a, inv_I_a @ rt_a)
                    if b_b is not None:
                        rt_b = np.cross(r_b, t_dir)
                        k_tangent += inv_m_b + np.dot(rt_b, inv_I_b @ rt_b)

                    if k_tangent > 1e-10:
                        delta_vt = -vt_len
                        tangent_mag = delta_vt / k_tangent
                        # Friction cone limit: |P_t| <= mu * P_n
                        max_fric = self.mu * new_impulse
                        old_fric = np.dot(contact_tangent_impulses[idx], t_dir)
                        new_fric = np.clip(old_fric + tangent_mag, -max_fric, max_fric)
                        delta_p_fric = new_fric - old_fric
                        contact_tangent_impulses[idx] += delta_p_fric * t_dir

                        P_fric = delta_p_fric * t_dir
                        b_a.vel += inv_m_a * P_fric
                        b_a.omega += inv_I_a @ np.cross(r_a, P_fric)
                        if b_b is not None:
                            b_b.vel -= inv_m_b * P_fric
                            b_b.omega -= inv_I_b @ np.cross(r_b, P_fric)

        # Integrate positions and orientations
        for b in self.bodies:
            if b.active:
                b.pos += b.vel * dt
                b.quat = quat_integrate(b.quat, b.omega, dt)

        # Record contact distances and interaction forces
        if len(contacts) > 0:
            self.last_contact_distances = np.array([c['dist'] for c in contacts], dtype=float)
            forces = []
            for idx, c in enumerate(contacts):
                # Total force exerted by contact pair = total impulse / dt
                f_norm = (contact_impulses[idx] / dt) * c['normal']
                f_fric = (contact_tangent_impulses[idx] / dt)
                forces.append(f_norm + f_fric)
            self.last_interaction_forces = np.array(forces, dtype=float)
        else:
            self.last_contact_distances = np.zeros(0, dtype=float)
            self.last_interaction_forces = np.zeros((0, 3), dtype=float)

        # External forces = gravity for active bodies
        ext_forces = np.zeros((self.N, 3), dtype=float)
        for i, b in enumerate(self.bodies):
            if b.active:
                ext_forces[i] = b.mass * self.gravity
        self.last_external_forces = ext_forces

        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance simulation to target_time and export StateFrame."""
        while self.current_time < target_time - 1e-7:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)

        # Build StateFrame
        positions = np.array([b.pos for b in self.bodies], dtype=float)
        velocities = np.array([b.vel for b in self.bodies], dtype=float)
        orientations = np.array([b.quat for b in self.bodies], dtype=float)
        angular_velocities = np.array([b.omega for b in self.bodies], dtype=float)
        masses = np.array([b.mass for b in self.bodies], dtype=float)
        body_ids = np.array([b.body_id for b in self.bodies], dtype=int)

        frame = StateFrame(
            time=float(target_time),
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=angular_velocities,
            mass=masses,
            body_id=body_ids,
            attributes={
                "release_time": np.array([b.release_time for b in self.bodies], dtype=float),
                "is_impact_block": np.array([1.0 if b.is_block else 0.0 for b in self.bodies], dtype=float),
            },
            scalars={
                "friction_coefficient": self.mu,
                "restitution": self.restitution,
                "contact_length_scale": float(self.contact_length_scale),
            },
            fields={
                "contact_signed_distance": self.last_contact_distances.copy(),
                "external_force": self.last_external_forces.copy(),
                "interaction_force": self.last_interaction_forces.copy(),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_bodies": self.N,
        }

