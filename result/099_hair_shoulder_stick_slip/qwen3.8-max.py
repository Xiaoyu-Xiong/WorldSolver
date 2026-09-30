"""Solver Agent-owned implementation for 099_hair_shoulder_stick_slip.

Implements an elastic-rod hair simulation with Coulomb friction contacts
against kinematic body geometry (head, neck, shoulder_bar, torso) and
hair-hair self-contact. Uses semi-implicit Euler integration with
constraint projection for stick-slip friction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_to_matrix(xyzw):
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = xyzw
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _capsule_sdf(p, center, half_h, radius, rot_mat, scale):
    """Signed distance from point p to capsule.
    
    Capsule is centered at `center`, oriented along local Z axis,
    with segment half-length `half_h` and sphere-cap radius `radius`.
    Scale is applied in local frame before distance computation.
    """
    # Transform to local frame
    local = rot_mat.T @ (p - center)
    # Apply inverse scale
    inv_scale = 1.0 / scale
    local_scaled = local * inv_scale
    # Clamp to segment
    z_clamped = np.clip(local_scaled[2], -half_h, half_h)
    closest_on_axis = np.array([0.0, 0.0, z_clamped])
    diff = local_scaled - closest_on_axis
    dist_local = np.linalg.norm(diff)
    # Scale back the distance approximately (use average scale for normal)
    avg_scale = np.mean(scale)
    sdf = (dist_local - radius) * avg_scale
    # Normal in world frame
    if dist_local > 1e-12:
        n_local = diff / dist_local
    else:
        n_local = np.array([0.0, 0.0, 1.0])
    n_world = rot_mat @ (n_local * inv_scale)
    nn = np.linalg.norm(n_world)
    if nn > 1e-12:
        n_world = n_world / nn
    else:
        n_world = np.array([0.0, 0.0, 1.0])
    return sdf, n_world


def _sphere_sdf(p, center, radius, rot_mat, scale):
    """Signed distance from point p to scaled sphere."""
    local = rot_mat.T @ (p - center)
    inv_scale = 1.0 / scale
    local_scaled = local * inv_scale
    r = np.linalg.norm(local_scaled)
    sdf = (r - radius) * np.mean(scale)
    if r > 1e-12:
        n_local = local_scaled / r
    else:
        n_local = np.array([0.0, 0.0, 1.0])
    n_world = rot_mat @ (n_local * inv_scale)
    nn = np.linalg.norm(n_world)
    if nn > 1e-12:
        n_world = n_world / nn
    else:
        n_world = np.array([0.0, 0.0, 1.0])
    return sdf, n_world


class KinematicBody:
    """Represents a kinematic collision body."""
    def __init__(self, obj_dict):
        self.id = obj_dict["id"]
        self.geo_type = obj_dict["geometry"]["type"]
        self.radius = float(obj_dict["geometry"].get("radius", 0.0))
        self.height = float(obj_dict["geometry"].get("height", 0.0))
        t = obj_dict["transform"]
        self.base_translation = np.array(t["translation"], dtype=float)
        self.rotation_xyzw = np.array(t["rotation_xyzw"], dtype=float)
        self.scale = np.array(t["scale"], dtype=float)
        self.rot_mat = _quat_to_matrix(self.rotation_xyzw)
        # For capsules, half-height of the cylindrical segment
        if self.geo_type == "capsule":
            self.half_h = self.height / 2.0
        else:
            self.half_h = 0.0
        # Current position (updated each step)
        self.translation = self.base_translation.copy()

    def update_position(self, displacement_vec):
        """Apply rigid translation to base position."""
        self.translation = self.base_translation + displacement_vec

    def sdf_and_normal(self, p):
        """Compute signed distance and outward normal at point p."""
        if self.geo_type == "capsule":
            return _capsule_sdf(p, self.translation, self.half_h, self.radius,
                                self.rot_mat, self.scale)
        elif self.geo_type == "sphere":
            return _sphere_sdf(p, self.translation, self.radius,
                               self.rot_mat, self.scale)
        else:
            # Fallback: large positive SDF (no collision)
            return 1e6, np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        config = scene.config
        meta = scene.metadata()

        # Physics constants
        self.gravity = np.array(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu_static = 0.3
        self.mu_kinetic = 0.3
        self.stick_tol = 1e-4  # m/s tolerance for stick/slip

        # Head motion parameters
        hm = config["head_motion"]
        self.motion_axis = np.array(hm["axis"], dtype=float)
        self.motion_disp = float(hm["displacement"])
        self.t_start = float(hm["start_time"])
        self.t_stop = float(hm["stop_time"])
        self.t_settle = float(hm["settle_end_time"])

        # Get initial state
        init_frame = scene.initial_state()
        self.N = len(init_frame.position)
        self.pos = init_frame.position.copy()
        self.vel = init_frame.velocity.copy()
        self.mass = init_frame.mass.copy()
        self.body_id = init_frame.body_id.copy()
        self.strand_id = init_frame.attributes["strand_id"].copy()
        self.node_index = init_frame.attributes["node_index"].copy()
        self.is_root = init_frame.attributes["is_root"].copy().astype(bool)

        # Strand topology
        self.n_strands = meta["strand_count"]
        self.nodes_per_strand = meta["nodes_per_strand"]
        self.strand_radius = meta["strand_radius"]

        # Build rest lengths per strand edge
        self.rest_lengths = np.zeros(self.N, dtype=float)
        for s in range(self.n_strands):
            sl = slice(s * self.nodes_per_strand, (s + 1) * self.nodes_per_strand)
            pos_s = self.pos[sl]
            for i in range(1, self.nodes_per_strand):
                self.rest_lengths[s * self.nodes_per_strand + i] = np.linalg.norm(
                    pos_s[i] - pos_s[i - 1]
                )

        # Elastic stiffness
        # Hair Young's modulus ~5 GPa, area = pi*r^2
        hair_area = np.pi * self.strand_radius ** 2
        E_modulus = 5.0e9  # Pa
        self.stretch_stiffness = E_modulus * hair_area  # N
        # Bending stiffness EI
        I_area = np.pi * self.strand_radius ** 4 / 4.0
        self.bend_stiffness = E_modulus * I_area  # N*m^2

        # Damping
        self.damping = 0.02  # Rayleigh mass-proportional damping coefficient

        # Build kinematic bodies from scene_geometry
        geom_config = config.get("scene_geometry", {})
        objects = geom_config.get("objects", [])
        self.kinematic_bodies = []
        for obj in objects:
            if obj.get("motion") == "kinematic" and obj.get("collision_enabled", True):
                self.kinematic_bodies.append(KinematicBody(obj))

        # Time tracking
        self.current_time = 0.0

        # Cumulative budget accumulators
        self.cum_ext_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((1, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((1, 3), dtype=float)

        # Reference scales
        total_mass = np.sum(self.mass)
        self.ref_velocity = 1.0
        self.ref_force = max(total_mass * 9.81, 1e-6)
        self.ref_momentum = max(total_mass * self.ref_velocity, 1e-9)

        # Load topology edges for reference
        topo_path = Path(__file__).resolve().parent.parent / "assets" / "hair_topology.json"
        if topo_path.is_file():
            with open(topo_path) as f:
                topo = json.load(f)
            self.topo_edges = np.array(topo["edges"], dtype=int)
        else:
            # Build linear chain edges
            edges = []
            for s in range(self.n_strands):
                base = s * self.nodes_per_strand
                for i in range(self.nodes_per_strand - 1):
                    edges.append([base + i, base + i + 1])
            self.topo_edges = np.array(edges, dtype=int)

    def _head_displacement(self, t: float) -> np.ndarray:
        """Smooth lateral displacement of head/neck/shoulder/torso."""
        if t <= self.t_start:
            frac = 0.0
        elif t >= self.t_stop:
            frac = 1.0
        else:
            # Smoothstep
            x = (t - self.t_start) / (self.t_stop - self.t_start)
            frac = x * x * (3.0 - 2.0 * x)
        return self.motion_axis * self.motion_disp * frac

    def _head_velocity(self, t: float) -> np.ndarray:
        """Velocity of kinematic bodies."""
        if t <= self.t_start or t >= self.t_stop:
            return np.zeros(3)
        dt_eps = 1e-6
        d1 = self._head_displacement(t)
        d0 = self._head_displacement(t - dt_eps)
        return (d1 - d0) / dt_eps

    def _compute_elastic_forces(self, pos):
        """Stretch + bending forces for all strands."""
        force = np.zeros_like(pos)
        nps = self.nodes_per_strand
        ns = self.n_strands

        for s in range(ns):
            base = s * nps
            ps = pos[base:base + nps]

            # Stretch forces (spring between consecutive nodes)
            for i in range(1, nps):
                idx0 = base + i - 1
                idx1 = base + i
                diff = pos[idx1] - pos[idx0]
                length = np.linalg.norm(diff)
                if length < 1e-15:
                    continue
                rest = self.rest_lengths[idx1]
                if rest < 1e-15:
                    continue
                strain = (length - rest) / rest
                f_mag = self.stretch_stiffness * strain
                direction = diff / length
                force[idx0] += f_mag * direction
                force[idx1] -= f_mag * direction

            # Bending forces (discrete Laplacian)
            for i in range(1, nps - 1):
                idx_prev = base + i - 1
                idx_curr = base + i
                idx_next = base + i + 1
                h = self.rest_lengths[idx_curr]
                if h < 1e-15:
                    continue
                # Discrete second derivative
                bend = (pos[idx_prev] - 2.0 * pos[idx_curr] + pos[idx_next])
                kappa_vec = bend / (h * h)
                fb = self.bend_stiffness * kappa_vec / h
                force[idx_prev] += fb
                force[idx_curr] -= 2.0 * fb
                force[idx_next] += fb

        return force

    def _detect_contacts(self, pos, vel, dt):
        """Detect hair-body and hair-hair contacts. Returns contact lists."""
        contacts_body = []  # (node_idx, normal, sdf, body_vel_at_contact)
        contacts_hair = []  # (i, j, normal, gap)

        body_v = self._head_velocity(self.current_time)

        # Hair-body contacts
        for b in self.kinematic_bodies:
            for i in range(self.N):
                if self.is_root[i]:
                    continue
                sdf, normal = b.sdf_and_normal(pos[i])
                penetration = -sdf + self.strand_radius
                if penetration > 0:
                    contacts_body.append((i, normal, penetration, body_v))

        # Hair-hair contacts (spatial proximity within nearby strands)
        # Only check strands that are spatially close (simple grid or brute for small N)
        contact_dist = 2.5 * self.strand_radius
        nps = self.nodes_per_strand
        ns = self.n_strands
        
        # Simple O(N^2) but only non-root, limited by early distance check
        # For performance, only check every other node
        for s1 in range(ns):
            b1 = s1 * nps
            for i_local in range(1, nps, 2):
                i = b1 + i_local
                pi = pos[i]
                for s2 in range(s1, ns):
                    b2 = s2 * nps
                    start_j = (i_local + 2) if s2 == s1 else 1
                    for j_local in range(start_j, nps, 2):
                        j = b2 + j_local
                        diff = pi - pos[j]
                        dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                        if dist_sq < contact_dist * contact_dist and dist_sq > 1e-20:
                            dist = np.sqrt(dist_sq)
                            gap = dist - 2.0 * self.strand_radius
                            if gap < 0:
                                normal = diff / dist
                                contacts_hair.append((i, j, normal, -gap))

        return contacts_body, contacts_hair

    def _apply_friction_and_contact(self, pos, vel, force, dt):
        """Apply contact constraints with Coulomb friction. Modifies force and vel."""
        contacts_body, contacts_hair = self._detect_contacts(pos, vel, dt)

        n_contacts = len(contacts_body) + len(contacts_hair)
        
        # Storage for output fields
        fric_normals = []
        fric_fn = []
        fric_ft = []
        fric_relv = []
        fric_mu_s = []
        fric_mu_k = []
        fric_active = []

        body_v = self._head_velocity(self.current_time)

        # Hair-body contacts
        for (idx, normal, pen, bv) in contacts_body:
            # Relative velocity of hair node vs body surface
            v_rel = vel[idx] - bv
            vn = np.dot(v_rel, normal)
            
            # Normal force (penalty spring)
            k_contact = self.stretch_stiffness * 0.5
            fn_mag = max(k_contact * pen, 0.0)
            
            # Tangential relative velocity
            v_tan = v_rel - vn * normal
            v_tan_mag = np.linalg.norm(v_tan)

            # Coulomb friction
            if v_tan_mag < self.stick_tol:
                # Stick regime: tangential force opposes impending slip
                ft = -self.mu_static * fn_mag * (v_tan / max(v_tan_mag, 1e-15))
                ft_mag = np.linalg.norm(ft)
                if ft_mag > self.mu_static * fn_mag:
                    ft = ft * (self.mu_static * fn_mag / max(ft_mag, 1e-15))
            else:
                # Slip regime
                ft = -self.mu_kinetic * fn_mag * (v_tan / v_tan_mag)

            # Apply contact force
            f_contact = fn_mag * normal + ft
            force[idx] += f_contact

            # Record for output
            fric_normals.append(normal)
            fric_fn.append(fn_mag)
            fric_ft.append(ft)
            fric_relv.append(v_rel)
            fric_mu_s.append(self.mu_static)
            fric_mu_k.append(self.mu_kinetic)
            fric_active.append(1.0)

        # Hair-hair contacts
        for (i, j, normal, pen) in contacts_hair:
            v_rel = vel[i] - vel[j]
            vn = np.dot(v_rel, normal)

            k_contact = self.stretch_stiffness * 0.3
            fn_mag = max(k_contact * pen, 0.0)

            v_tan = v_rel - vn * normal
            v_tan_mag = np.linalg.norm(v_tan)

            if v_tan_mag < self.stick_tol:
                ft = -self.mu_static * fn_mag * (v_tan / max(v_tan_mag, 1e-15))
                ft_mag = np.linalg.norm(ft)
                if ft_mag > self.mu_static * fn_mag:
                    ft = ft * (self.mu_static * fn_mag / max(ft_mag, 1e-15))
            else:
                ft = -self.mu_kinetic * fn_mag * (v_tan / v_tan_mag)

            f_contact_i = fn_mag * normal + ft
            force[i] += f_contact_i
            force[j] -= f_contact_i

            fric_normals.append(normal)
            fric_fn.append(fn_mag)
            fric_ft.append(ft)
            fric_relv.append(v_rel)
            fric_mu_s.append(self.mu_static)
            fric_mu_k.append(self.mu_kinetic)
            fric_active.append(1.0)

        # Pad arrays
        C = max(n_contacts, 1)
        if n_contacts == 0:
            fric_normals_arr = np.zeros((1, 3))
            fric_fn_arr = np.zeros(1)
            fric_ft_arr = np.zeros((1, 3))
            fric_relv_arr = np.zeros((1, 3))
            fric_mu_s_arr = np.array([self.mu_static])
            fric_mu_k_arr = np.array([self.mu_kinetic])
            fric_active_arr = np.zeros(1)
        else:
            fric_normals_arr = np.array(fric_normals)
            fric_fn_arr = np.array(fric_fn)
            fric_ft_arr = np.array(fric_ft)
            fric_relv_arr = np.array(fric_relv)
            fric_mu_s_arr = np.array(fric_mu_s)
            fric_mu_k_arr = np.array(fric_mu_k)
            fric_active_arr = np.array(fric_active)

        return (fric_normals_arr, fric_fn_arr, fric_ft_arr, fric_relv_arr,
                fric_mu_s_arr, fric_mu_k_arr, fric_active_arr)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self.current_time:
            return self._make_frame(target)

        # Sub-step for stability
        max_dt = 0.0005
        total_dt = target - self.current_time
        n_sub = max(int(np.ceil(total_dt / max_dt)), 1)
        dt = total_dt / n_sub

        for _ in range(n_sub):
            self._step(dt)
            self.current_time += dt

        return self._make_frame(target)

    def _step(self, dt: float):
        """One sub-step of semi-implicit Euler with contact."""
        # Update kinematic body positions
        disp = self._head_displacement(self.current_time + dt)
        for b in self.kinematic_bodies:
            b.update_position(disp)

        # Enforce root attachment to head
        root_disp = disp
        nps = self.nodes_per_strand
        ns = self.n_strands
        for s in range(ns):
            root_idx = s * nps
            # Move root with head
            self.pos[root_idx] = self.pos[root_idx]  # roots already tracked
            # Actually we need to move roots incrementally
            prev_disp = self._head_displacement(self.current_time)
            delta_disp = disp - prev_disp
            self.pos[root_idx] += delta_disp
            self.vel[root_idx] = self._head_velocity(self.current_time + dt)

        # Compute elastic forces
        f_elastic = self._compute_elastic_forces(self.pos)

        # Gravity
        f_gravity = self.mass[:, None] * self.gravity[None, :]

        # Damping
        f_damp = -self.damping * self.mass[:, None] * self.vel

        # Total force before contact
        f_total = f_elastic + f_gravity + f_damp

        # Contact + friction
        (fric_n, fric_fn, fric_ft, fric_rv,
         fric_ms, fric_mk, fric_act) = self._apply_friction_and_contact(
            self.pos, self.vel, f_total, dt
        )

        # Store friction data for current frame
        self._last_fric = (fric_n, fric_fn, fric_ft, fric_rv, fric_ms, fric_mk, fric_act)

        # Semi-implicit Euler: update velocity then position
        acc = np.zeros_like(self.vel)
        active = self.mass > 1e-15
        acc[active] = f_total[active] / self.mass[active, None]

        self.vel += acc * dt
        self.pos += self.vel * dt

        # Re-enforce root constraints after integration
        for s in range(ns):
            root_idx = s * nps
            self.vel[root_idx] = self._head_velocity(self.current_time + dt)

        # Accumulate external impulse (gravity only as external)
        ext_impulse = np.sum(f_gravity * dt, axis=0, keepdims=True)
        self.cum_ext_impulse += ext_impulse

    def _make_frame(self, time: float) -> StateFrame:
        """Build a StateFrame at the given time."""
        # Ensure we have friction data
        if not hasattr(self, '_last_fric'):
            C = 1
            fric_n = np.zeros((C, 3))
            fric_fn = np.zeros(C)
            fric_ft = np.zeros((C, 3))
            fric_rv = np.zeros((C, 3))
            fric_ms = np.array([self.mu_static])
            fric_mk = np.array([self.mu_kinetic])
            fric_act = np.zeros(C)
        else:
            fric_n, fric_fn, fric_ft, fric_rv, fric_ms, fric_mk, fric_act = self._last_fric

        C = len(fric_fn)
        G = 1  # single global domain
        K = 1  # single force term (gravity)

        # Domain weights: all hair nodes belong to single domain
        domain_weights = np.ones((G, self.N), dtype=float)

        # External force per domain per term: gravity
        ext_force = np.zeros((G, K, 3), dtype=float)
        ext_force[0, 0, :] = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)

        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "strand_id": self.strand_id.copy(),
                "node_index": self.node_index.copy(),
                "is_root": self.is_root.astype(float),
                "kinematic_root": self.is_root.astype(float),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "friction_active_mask": fric_act.copy(),
                "friction_dynamic_coefficient": fric_mk.copy(),
                "friction_normal": fric_n.copy(),
                "friction_normal_force": fric_fn.copy(),
                "friction_relative_velocity": fric_rv.copy(),
                "friction_static_coefficient": fric_ms.copy(),
                "friction_tangential_force": fric_ft.copy(),
            },
            scalars={
                "friction_stick_speed_tolerance": self.stick_tol,
                "reference_force": self.ref_force,
                "reference_momentum": self.ref_momentum,
                "reference_velocity": self.ref_velocity,
                "strand_radius": self.strand_radius,
                "static_friction": self.mu_static,
                "kinetic_friction": self.mu_kinetic,
                "head_stop_time": self.t_stop,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_nodes": self.N,
            "n_strands": self.n_strands,
            "final_time": self.current_time,
        }
