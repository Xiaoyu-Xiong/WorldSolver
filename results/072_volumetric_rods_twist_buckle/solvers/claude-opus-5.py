"""Quasi-static Cosserat-rod solver for twist-induced buckling of a seven-rod bundle.

Physical model (implemented from scratch with NumPy only):

* Each of the seven volumetric rods is discretized as a Cosserat (Simo-Reissner)
  rod: ``nx+1`` centerline nodes carrying translational mass and ``nx`` material
  frames (unit quaternions) carrying rotational inertia.
* Elastic energy per element: shear/stretch ``0.5*l0*eps.S.eps`` with
  ``eps = R^T e / l0 - e3`` and ``S = diag(GA, GA, EA)``; between adjacent frames a
  bend/twist energy ``0.5*(EI*alpha^2 + GJ*gamma^2)/l0`` built from the tangent
  turning angle ``alpha`` and the parallel-transport twist angle ``gamma``.  The
  twist angle is unwrapped across steps so that the prescribed forty end turns are
  represented without any 2*pi ambiguity.
* Rod-rod and rod-self contact is a penalty (spring + normal dashpot) between
  centerline node spheres of radius ``rod_radius``, evaluated on a neighbour list.
* Both driven end regions (``driven_end_length``) are kinematic: node positions are
  the exact rigid rotation of their rest positions about the x axis at the frozen
  angular speeds, and the enclosed material frames are rotated by the same angle.
  Axial position is fixed, so the twist-induced shortening buckles the bundle into
  repeated coils that press against each other.
* Time stepping is a symplectic (semi-implicit) Euler scheme with a fixed step of
  1 ms and viscous drag, which keeps the driven evolution quasi-static and stable.

Surface samples exported to the trajectory are generated from the rod kinematics
(``x_i + r*(cos(theta) d1_i + sin(theta) d2_i)`` plus the two cap centers), matching
the frozen ``assets/rod_surface.obj`` vertex ordering exactly.  Node masses are the
exact lumping of the surface-sample masses, so the exported per-sample momentum is
identical to the simulated nodal momentum and the exported impulse ledger closes.
"""
from __future__ import annotations

import time as _walltime

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _quat_dirs(q: np.ndarray):
    """Columns (d1, d2, d3) of the rotation matrices of quaternions (w, x, y, z)."""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    d1 = np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)), axis=-1)
    d2 = np.stack((2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)), axis=-1)
    d3 = np.stack((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), axis=-1)
    return d1, d2, d3


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_exp(vec: np.ndarray) -> np.ndarray:
    """Unit quaternion of the rotation vector ``vec``."""
    angle = np.linalg.norm(vec, axis=-1)
    half = 0.5 * angle
    small = angle < 1.0e-9
    safe = np.where(small, 1.0, angle)
    scale = np.where(small, 0.5, np.sin(half) / safe)
    return np.concatenate((np.cos(half)[..., None], vec * scale[..., None]), axis=-1)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def _unit(vec: np.ndarray, fallback: np.ndarray | None = None):
    norm = np.linalg.norm(vec, axis=-1)
    ok = norm > 1.0e-12
    safe = np.where(ok, norm, 1.0)[..., None]
    out = vec / safe
    if fallback is not None:
        out = np.where(ok[..., None], out, fallback)
    else:
        out = np.where(ok[..., None], out, 0.0)
    return out, ok


class BenchmarkSolver:
    """Cosserat-rod solver for ``072_volumetric_rods_twist_buckle``."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        geom = cfg["geometry"]
        bcs = cfg["boundary_conditions"]
        mat = cfg["material"]

        self.n_rod = int(geom["rod_count"])
        self.length = float(geom["rod_length"])
        self.radius = float(geom["rod_radius"])
        self.nx = int(geom["axial_segments"])
        self.nt = int(geom["radial_segments"])
        self.ring_radius = float(geom["bundle_ring_radius"])
        self.density = float(mat["density"])

        self.dt = 1.0e-3
        self.sim_time = 0.0
        self.step_count = 0
        self.rng_seed = int(seed)

        nx, nt, n_rod = self.nx, self.nt, self.n_rod
        self.n_node = nx + 1
        self.n_edge = nx
        self.l0 = self.length / nx

        # ---- reference (frozen) trajectory sample layout -------------------
        frame0 = scene.initial_state()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.mass_sample = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.n_sample = int(self.mass_sample.size)
        self.per_rod_samples = nt * (nx + 1) + 2
        m_vertex = float(self.mass_sample[0])

        # ---- rest configuration -------------------------------------------
        xs = np.linspace(-0.5 * self.length, 0.5 * self.length, nx + 1)
        offsets = [(0.0, 0.0)] + [
            (
                self.ring_radius * np.cos(k * np.pi / 3.0),
                self.ring_radius * np.sin(k * np.pi / 3.0),
            )
            for k in range(n_rod - 1)
        ]
        rest = np.zeros((n_rod, nx + 1, 3), dtype=np.float64)
        rest[:, :, 0] = xs[None, :]
        for r, (oy, oz) in enumerate(offsets[:n_rod]):
            rest[r, :, 1] = oy
            rest[r, :, 2] = oz
        self.rest = rest

        # ---- lumped nodal masses that reproduce the surface-sample momentum
        node_mass = np.full((n_rod, nx + 1), nt * m_vertex, dtype=np.float64)
        node_mass[:, 0] += m_vertex
        node_mass[:, -1] += m_vertex
        self.node_mass = node_mass
        self.total_mass = float(node_mass.sum())

        # ---- material properties -------------------------------------------
        area = np.pi * self.radius ** 2
        inertia = 0.25 * np.pi * self.radius ** 4
        young = 1.0e5
        shear = 0.5 * young
        self.EA = young * area
        self.GA = shear * area
        self.EI = young * inertia
        self.GJ = shear * (2.0 * inertia)
        elem_mass = self.density * area * self.l0
        # Isotropic (nearly exact for this slenderness) frame inertia.
        i_axial = elem_mass * 0.5 * self.radius ** 2
        i_trans = elem_mass * (0.25 * self.radius ** 2 + self.l0 ** 2 / 12.0)
        self.I_rot = float((2.0 * i_trans + i_axial) / 3.0)

        # Strong viscous drag keeps the driven evolution quasi-static and
        # suppresses jitter from the stiff contact penalty (dt*drag = 0.025).
        self.drag_lin = 25.0
        self.drag_rot = 25.0
        self.contact_d0 = 2.0 * self.radius
        self.contact_k = 1.0e5
        self.contact_c = 0.25 * np.sqrt(self.contact_k * float(node_mass[0, 1]))
        self.contact_skin = 0.10
        self.contact_cut2 = (self.contact_d0 + self.contact_skin) ** 2
        self.rebuild_every = 40

        # ---- boundary conditions -------------------------------------------
        end_len = float(bcs["driven_end_length"])
        axis = np.asarray(bcs["rotation_axis"], dtype=np.float64)
        self.axis = axis / max(float(np.linalg.norm(axis)), _EPS)
        self.omega_left = float(np.deg2rad(float(bcs["left_end_angular_speed_deg_s"])))
        self.omega_right = float(np.deg2rad(float(bcs["right_end_angular_speed_deg_s"])))
        self.t_start = float(bcs["start_time"])
        self.t_stop = float(bcs["stop_time"])

        left_node = xs <= (-0.5 * self.length + end_len + 1.0e-9)
        right_node = xs >= (0.5 * self.length - end_len - 1.0e-9)
        self.left_node = left_node
        self.right_node = right_node
        self.free_node = ~(left_node | right_node)
        self.left_edge = left_node[:-1] & left_node[1:]
        self.right_edge = right_node[:-1] & right_node[1:]
        self.driven_edge = self.left_edge | self.right_edge

        # ---- state ----------------------------------------------------------
        self.X = rest.copy()
        rng = np.random.default_rng(self.rng_seed)
        perturb = 3.0e-4 * rng.standard_normal((n_rod, nx + 1, 3))
        perturb[:, :, 0] = 0.0
        perturb[:, ~self.free_node, :] = 0.0
        self.X += perturb
        self.V = np.zeros_like(self.X)

        # Rest material frame: d1 = +y, d2 = +z, d3 = +x  (120 deg about (1,1,1)).
        self.q_rest = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (n_rod, self.n_edge, 1))
        self.q = self.q_rest.copy()
        self.W = np.zeros((n_rod, self.n_edge, 3), dtype=np.float64)

        self.gamma_prev = np.zeros((n_rod, self.n_edge - 1), dtype=np.float64)
        self.gamma_turns = np.zeros((n_rod, self.n_edge - 1), dtype=np.float64)

        # ---- contact bookkeeping -------------------------------------------
        nn = n_rod * (nx + 1)
        self.n_flat = nn
        self.node_body = np.repeat(np.arange(n_rod), nx + 1)
        station = np.tile(np.arange(nx + 1), n_rod)
        same_rod = self.node_body[:, None] == self.node_body[None, :]
        close_arc = np.abs(station[:, None] - station[None, :]) <= 1
        allowed = np.triu(np.ones((nn, nn), dtype=bool), 1)
        allowed &= ~(same_rod & close_arc)
        self.pair_allowed = allowed
        self.pair_a = np.zeros(0, dtype=np.int64)
        self.pair_b = np.zeros(0, dtype=np.int64)
        self.pair_inter = np.zeros(0, dtype=bool)
        self._rebuild_pairs()

        # ---- momentum budget ledger -----------------------------------------
        # domains: one row per rod body, one global row.  terms: gravity, drag,
        # driven-end reaction, inter-body contact.
        self.n_domain = n_rod + 1
        self.n_term = 4
        self.gravity = np.zeros(3, dtype=np.float64)
        self.cum_impulse = np.zeros((n_rod, self.n_term, 3), dtype=np.float64)
        self.cum_impulse_prev = self.cum_impulse.copy()
        self.prev_sample_time = 0.0
        self.mean_force = np.zeros((n_rod, self.n_term, 3), dtype=np.float64)

        weights = np.zeros((self.n_domain, self.n_sample), dtype=np.float32)
        for b in range(n_rod):
            weights[b] = (self.body_id == b).astype(np.float32)
        weights[n_rod] = 1.0
        self.domain_weights = weights

        omega_ref = max(abs(self.omega_left), abs(self.omega_right), 1.0e-3)
        self.reference_momentum = float(
            self.total_mass * omega_ref * (self.ring_radius + self.radius)
        )

        # ---- surface sampling helpers ---------------------------------------
        theta = 2.0 * np.pi * np.arange(nt) / nt
        self.cos_t = np.cos(theta)
        self.sin_t = np.sin(theta)

        self.max_penetration = 0.0
        self.max_speed = 0.0
        self.wall_start = _walltime.perf_counter()

    # ------------------------------------------------------------ kinematics
    def _end_angles(self, t: float):
        tc = min(max(float(t), self.t_start), self.t_stop) - self.t_start
        return self.omega_left * tc, self.omega_right * tc

    def _rotate_about_axis(self, points: np.ndarray, angle: float) -> np.ndarray:
        a = self.axis
        c, s = np.cos(angle), np.sin(angle)
        dot = points @ a
        return (
            points * c
            + np.cross(np.broadcast_to(a, points.shape), points) * s
            + a[None, :] * (dot * (1.0 - c))[:, None]
        )

    def _apply_driven(self, t: float) -> None:
        """Snap driven nodes/frames to the exact prescribed rigid rotation."""
        ang_l, ang_r = self._end_angles(t)
        for mask_n, mask_e, ang, omega in (
            (self.left_node, self.left_edge, ang_l, self.omega_left),
            (self.right_node, self.right_edge, ang_r, self.omega_right),
        ):
            rest = self.rest[:, mask_n, :].reshape(-1, 3)
            rotated = self._rotate_about_axis(rest, ang)
            shape = (self.n_rod, int(mask_n.sum()), 3)
            self.X[:, mask_n, :] = rotated.reshape(shape)
            omega_vec = omega * self.axis
            self.V[:, mask_n, :] = np.cross(
                np.broadcast_to(omega_vec, rotated.shape), rotated
            ).reshape(shape)
            drive = _quat_exp(np.asarray(ang * self.axis, dtype=np.float64))
            q_target = _quat_mul(
                np.broadcast_to(drive, self.q_rest[:, mask_e, :].shape),
                self.q_rest[:, mask_e, :],
            )
            self.q[:, mask_e, :] = _normalize_quat(q_target)
            self.W[:, mask_e, :] = omega_vec

    def _driven_target_velocity(self, t: float):
        ang_l, ang_r = self._end_angles(t)
        out = np.zeros_like(self.V)
        for mask_n, ang, omega in (
            (self.left_node, ang_l, self.omega_left),
            (self.right_node, ang_r, self.omega_right),
        ):
            rest = self.rest[:, mask_n, :].reshape(-1, 3)
            rotated = self._rotate_about_axis(rest, ang)
            omega_vec = omega * self.axis
            vel = np.cross(np.broadcast_to(omega_vec, rotated.shape), rotated)
            out[:, mask_n, :] = vel.reshape(self.n_rod, int(mask_n.sum()), 3)
        return out

    # -------------------------------------------------------------- contacts
    def _rebuild_pairs(self) -> None:
        flat = self.X.reshape(-1, 3)
        diff = flat[:, None, :] - flat[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        sel = (dist2 < self.contact_cut2) & self.pair_allowed
        a, b = np.nonzero(sel)
        self.pair_a = a
        self.pair_b = b
        self.pair_inter = self.node_body[a] != self.node_body[b]

    def _contact_forces(self):
        force = np.zeros((self.n_flat, 3), dtype=np.float64)
        inter = np.zeros((self.n_flat, 3), dtype=np.float64)
        if self.pair_a.size == 0:
            return force, inter
        flat_x = self.X.reshape(-1, 3)
        flat_v = self.V.reshape(-1, 3)
        a, b = self.pair_a, self.pair_b
        dvec = flat_x[b] - flat_x[a]
        dist = np.linalg.norm(dvec, axis=1)
        pen = self.contact_d0 - dist
        active = np.nonzero((pen > 0.0) & (dist > 1.0e-9))[0]
        if active.size == 0:
            return force, inter
        ia, ib = a[active], b[active]
        normal = dvec[active] / dist[active][:, None]
        rel = flat_v[ib] - flat_v[ia]
        vn = np.einsum("ij,ij->i", rel, normal)
        mag = np.maximum(self.contact_k * pen[active] - self.contact_c * vn, 0.0)
        fb = mag[:, None] * normal
        np.add.at(force, ib, fb)
        np.add.at(force, ia, -fb)
        mask = self.pair_inter[active]
        if np.any(mask):
            np.add.at(inter, ib[mask], fb[mask])
            np.add.at(inter, ia[mask], -fb[mask])
        pmax = float(pen[active].max())
        if pmax > self.max_penetration:
            self.max_penetration = pmax
        return force, inter

    # ------------------------------------------------------------ elasticity
    def _elastic(self):
        d1, d2, d3 = _quat_dirs(self.q)
        edge = self.X[:, 1:, :] - self.X[:, :-1, :]
        inv_l0 = 1.0 / self.l0
        u1 = np.einsum("rej,rej->re", d1, edge) * inv_l0
        u2 = np.einsum("rej,rej->re", d2, edge) * inv_l0
        u3 = np.einsum("rej,rej->re", d3, edge) * inv_l0

        n1 = self.GA * u1
        n2 = self.GA * u2
        n3 = self.EA * (u3 - 1.0)
        n_world = n1[..., None] * d1 + n2[..., None] * d2 + n3[..., None] * d3

        force = np.zeros_like(self.X)
        force[:, :-1, :] += n_world
        force[:, 1:, :] -= n_world

        # couple of the shear/stretch resultant: tau_local = l0 * (u x (S eps))
        c1 = self.l0 * (u2 * n3 - u3 * n2)
        c2 = self.l0 * (u3 * n1 - u1 * n3)
        c3 = self.l0 * (u1 * n2 - u2 * n1)
        torque = c1[..., None] * d1 + c2[..., None] * d2 + c3[..., None] * d3

        # --- bend / twist between adjacent material frames -------------------
        ta = d3[:, :-1, :]
        tb = d3[:, 1:, :]
        bvec = np.cross(ta, tb)
        sin_a = np.linalg.norm(bvec, axis=-1)
        cos_a = np.einsum("rej,rej->re", ta, tb)
        alpha = np.arctan2(sin_a, cos_a)
        axis_b, _ = _unit(bvec)
        moment = (self.EI * inv_l0 * alpha)[..., None] * axis_b

        d1a = d1[:, :-1, :]
        d1b = d1[:, 1:, :]
        ca = np.cos(alpha)[..., None]
        sa = np.sin(alpha)[..., None]
        proj = np.einsum("rej,rej->re", axis_b, d1a)[..., None]
        transported = d1a * ca + np.cross(axis_b, d1a) * sa + axis_b * proj * (1.0 - ca)
        cos_g = np.einsum("rej,rej->re", transported, d1b)
        sin_g = np.einsum("rej,rej->re", np.cross(transported, d1b), tb)
        gamma = np.arctan2(sin_g, cos_g)

        delta = gamma - self.gamma_prev
        self.gamma_turns -= (delta > np.pi).astype(np.float64)
        self.gamma_turns += (delta < -np.pi).astype(np.float64)
        self.gamma_prev = gamma
        gamma_total = gamma + 2.0 * np.pi * self.gamma_turns

        t_mid, _ = _unit(ta + tb, fallback=ta)
        moment = moment + (self.GJ * inv_l0 * gamma_total)[..., None] * t_mid
        torque[:, :-1, :] += moment
        torque[:, 1:, :] -= moment

        self._last_u3 = u3
        self._last_dirs = (d1, d2, d3)
        self._last_edge = edge
        return force, torque

    # --------------------------------------------------------------- stepping
    def _advance_one_step(self) -> None:
        dt = self.dt
        t_next = self.sim_time + dt

        force, torque = self._elastic()
        contact, contact_inter = self._contact_forces()
        force += contact.reshape(self.X.shape)

        grav = self.gravity[None, None, :] * self.node_mass[..., None]
        drag = -self.drag_lin * self.node_mass[..., None] * self.V
        force += grav + drag
        torque -= self.drag_rot * self.I_rot * self.W

        v_free = self.V + dt * force / self.node_mass[..., None]
        self.W = self.W + dt * torque / self.I_rot

        v_target = self._driven_target_velocity(t_next)
        driven = ~self.free_node
        reaction = np.zeros_like(self.V)
        reaction[:, driven, :] = (
            self.node_mass[:, driven, None] * (v_target[:, driven, :] - v_free[:, driven, :])
        )
        self.V = v_free
        self.V[:, driven, :] = v_target[:, driven, :]

        # ledger: exactly the impulses that were applied this step
        self.cum_impulse[:, 0, :] += grav.sum(axis=1) * dt
        self.cum_impulse[:, 1, :] += drag.sum(axis=1) * dt
        self.cum_impulse[:, 2, :] += reaction.sum(axis=1)
        self.cum_impulse[:, 3, :] += (
            contact_inter.reshape(self.n_rod, self.n_node, 3).sum(axis=1) * dt
        )

        self.X = self.X + dt * self.V
        self.q = _normalize_quat(_quat_mul(_quat_exp(self.W * dt), self.q))

        self._apply_driven(t_next)

        self.sim_time = t_next
        self.step_count += 1
        if self.step_count % self.rebuild_every == 0:
            self._rebuild_pairs()
        speed = float(np.abs(self.V).max())
        if speed > self.max_speed:
            self.max_speed = speed

    # ------------------------------------------------------------ sample data
    def _node_frames(self):
        q_edge = self.q
        q_node = np.empty((self.n_rod, self.n_node, 4), dtype=np.float64)
        q_node[:, 0, :] = q_edge[:, 0, :]
        q_node[:, -1, :] = q_edge[:, -1, :]
        left = q_edge[:, :-1, :]
        right = q_edge[:, 1:, :]
        sign = np.where(np.einsum("rej,rej->re", left, right) < 0.0, -1.0, 1.0)
        q_node[:, 1:-1, :] = left + sign[..., None] * right
        q_node = _normalize_quat(q_node)
        d1, d2, _ = _quat_dirs(q_node)
        w_node = np.empty((self.n_rod, self.n_node, 3), dtype=np.float64)
        w_node[:, 0, :] = self.W[:, 0, :]
        w_node[:, -1, :] = self.W[:, -1, :]
        w_node[:, 1:-1, :] = 0.5 * (self.W[:, :-1, :] + self.W[:, 1:, :])
        return d1, d2, w_node

    def _surface_samples(self):
        d1, d2, w_node = self._node_frames()
        r = self.radius
        offset = r * (
            self.cos_t[None, None, :, None] * d1[:, :, None, :]
            + self.sin_t[None, None, :, None] * d2[:, :, None, :]
        )
        pos_ring = self.X[:, :, None, :] + offset
        vel_ring = self.V[:, :, None, :] + np.cross(
            np.broadcast_to(w_node[:, :, None, :], offset.shape), offset
        )
        n_ring = self.nt * self.n_node
        pos = np.empty((self.n_rod, self.per_rod_samples, 3), dtype=np.float64)
        vel = np.empty((self.n_rod, self.per_rod_samples, 3), dtype=np.float64)
        pos[:, :n_ring, :] = pos_ring.reshape(self.n_rod, n_ring, 3)
        vel[:, :n_ring, :] = vel_ring.reshape(self.n_rod, n_ring, 3)
        pos[:, n_ring, :] = self.X[:, 0, :]
        pos[:, n_ring + 1, :] = self.X[:, -1, :]
        vel[:, n_ring, :] = self.V[:, 0, :]
        vel[:, n_ring + 1, :] = self.V[:, -1, :]
        return pos.reshape(-1, 3), vel.reshape(-1, 3)

    def _deformation_gradients(self):
        d1, d2, _ = self._last_dirs
        edge = self._last_edge
        u3 = np.maximum(self._last_u3, 0.05)
        lam_t = 1.0 / np.sqrt(u3)
        col_x = edge / self.l0
        col_y = lam_t[..., None] * d1
        col_z = lam_t[..., None] * d2
        grad = np.stack((col_x, col_y, col_z), axis=-1)
        return grad.reshape(-1, 3, 3)

    def _budget_fields(self, time: float):
        n_rod = self.n_rod
        cum = np.zeros((self.n_domain, 3), dtype=np.float64)
        cum[:n_rod, :] = self.cum_impulse.sum(axis=1)
        cum[n_rod, :] = cum[:n_rod, :].sum(axis=0)
        forces = np.zeros((self.n_domain, self.n_term, 3), dtype=np.float64)
        forces[:n_rod, :, :] = self.mean_force
        forces[n_rod, :, :] = self.mean_force.sum(axis=0)
        return cum, forces

    def _make_frame(self, time: float) -> StateFrame:
        if not hasattr(self, "_last_dirs"):
            self._elastic()
        pos, vel = self._surface_samples()
        cum, forces = self._budget_fields(time)
        zeros3 = np.zeros((self.n_domain, 3), dtype=np.float64)
        fields = {
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": forces,
            "budget_cumulative_external_impulse": cum,
            "budget_cumulative_inflow_momentum": zeros3,
            "budget_cumulative_outflow_momentum": zeros3.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": self._deformation_gradients(),
            "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
        }
        frame = StateFrame(
            time=float(time),
            position=pos.astype(np.float32),
            velocity=vel,
            mass=self.mass_sample.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass_sample.copy()},
            fields=fields,
            scalars={
                "reference_momentum": self.reference_momentum,
                "characteristic_length": float(self.length),
                "contact_length_scale": float(self.contact_d0),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        n_steps = int(round((target - self.sim_time) / self.dt))
        for _ in range(max(n_steps, 0)):
            self._advance_one_step()
        # Land exactly on the requested sample time.
        self.sim_time = target
        span = target - self.prev_sample_time
        if span > 0.0:
            self.mean_force = (self.cum_impulse - self.cum_impulse_prev) / span
        else:
            self.mean_force = np.zeros_like(self.mean_force)
        self.cum_impulse_prev = self.cum_impulse.copy()
        self.prev_sample_time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "cosserat_rod_symplectic_euler_penalty_contact",
            "time_step": self.dt,
            "steps": int(self.step_count),
            "rods": int(self.n_rod),
            "nodes_per_rod": int(self.n_node),
            "young_modulus_pa": float(self.EA / (np.pi * self.radius ** 2)),
            "contact_stiffness": float(self.contact_k),
            "max_penetration_m": float(self.max_penetration),
            "max_node_speed_m_s": float(self.max_speed),
            "twist_turns_total": float(
                abs(self.omega_left - self.omega_right) * (self.t_stop - self.t_start) / (2.0 * np.pi)
            ),
            "wall_seconds": float(_walltime.perf_counter() - self.wall_start),
        }
