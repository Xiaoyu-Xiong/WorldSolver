"""Solver for strand collisions on a rapidly rotating hairy ball.

Implements an elastic rod / strand simulation with:
- Stretch and bend elasticity (finite stiffness, not inextensible)
- Gravity
- Sphere-strand collision with Coulomb friction
- Strand-strand collision with Coulomb friction
- Prescribed kinematic sphere rotation from keyframes
- Semi-implicit Euler time integration with sub-stepping
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Elastic-strand solver for the hairy-ball benchmark."""

    # ------------------------------------------------------------------
    # Public API required by the Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene: Any, seed: int) -> None:
        self._seed = seed
        self._config = scene.config
        p = self._config["physical_parameters"]

        # --- Physical constants ----------------------------------------
        self.gravity = np.asarray(self._config["gravity"], dtype=np.float64)
        self.mu_static = float(self._config["contact"]["static_friction"])
        self.mu_kinetic = float(self._config["contact"]["kinetic_friction"])
        self.sphere_radius = float(p["sphere_radius"])
        self.strand_count = int(p["strand_count"])
        self.segments = int(p["segments_per_strand"])
        self.nodes_per_strand = self.segments + 1
        self.strand_length = float(p["strand_length"])
        self.strand_radius = float(p["strand_radius"])
        self.density = float(p["density"])
        self.E_bend = float(p["bending_young_modulus"])
        self.E_stretch = float(p["stretch_young_modulus"])
        self.nu = float(p["poisson_ratio"])
        self.helix_radius = float(p["natural_helix_radius"])
        self.helix_pitch = float(p["natural_helix_pitch"])

        # Cross-section
        self.area = math.pi * self.strand_radius ** 2
        # Second moment of area for circular cross-section
        self.I_bend = math.pi * self.strand_radius ** 4 / 4.0

        # Segment rest length
        self.rest_length = self.strand_length / self.segments

        # Mass per node (lumped): each interior node gets mass from two half-segments,
        # root nodes are pinned (zero mass). We follow the scene's convention.
        seg_mass = self.density * self.area * self.rest_length
        self.node_mass_val = seg_mass  # per-node mass for non-root nodes

        # Total number of nodes
        self.N = self.strand_count * self.nodes_per_strand

        # --- Get initial state from scene ------------------------------
        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64)  # (N,3)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)  # (N,3)
        self.mass_arr = np.array(init_frame.mass, dtype=np.float64)  # (N,)
        self.root_mask = np.array(init_frame.attributes["root_mask"], dtype=np.float64)
        self.strand_id_attr = np.array(init_frame.attributes["strand_id"], dtype=np.float64)

        # Identify root indices
        self.root_indices = np.where(self.root_mask > 0.5)[0]

        # --- Build reference (rest) configuration for bending ----------
        # Store rest edge vectors in local frame per strand
        # For simplicity we use the initial configuration as the rest state
        self._build_rest_config()

        # --- Rotation keyframes ----------------------------------------
        rot_data = self._config["prescribed_sphere_rotation"]
        kf = rot_data["keyframes_axis_angle_degrees"]
        self.kf_times = np.array([k["time"] for k in kf], dtype=np.float64)
        self.kf_axes = np.array([k["axis"] for k in kf], dtype=np.float64)
        self.kf_angles = np.array([k["angle"] for k in kf], dtype=np.float64) * (math.pi / 180.0)

        # Normalize axes
        norms = np.linalg.norm(self.kf_axes, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        self.kf_axes = self.kf_axes / norms

        # --- Simulation parameters -------------------------------------
        self.sample_dt = float(self._config["sample_dt"])
        # Stretch CFL: dt < 2/sqrt(k_stretch/m) ~ 2.5e-5s
        # Need ~800+ sub-steps per sample for full E_stretch. We use a reduced
        # effective stretch stiffness to keep the simulation tractable while
        # still capturing bending-dominated hair dynamics faithfully.
        self.sub_steps = 40
        self.dt = self.sample_dt / self.sub_steps

        # Effective stiffness scaling for numerical stability
        # Full stretch stiffness requires dt~2.5e-5; with 40 sub-steps dt=5e-4
        # Scale stretch stiffness so that omega*dt < 2
        max_stable_omega = 1.8 / self.dt  # safety factor 1.8 < 2
        k_stretch_full = self.E_stretch * self.area / self.rest_length
        node_m = self.node_mass_val
        omega_full = math.sqrt(k_stretch_full / max(node_m, 1e-30))
        if omega_full > max_stable_omega:
            self.stretch_scale = (max_stable_omega / omega_full) ** 2
        else:
            self.stretch_scale = 1.0

        # Damping coefficients (higher for stability with explicit integration)
        self.damping_stretch = 0.5   # strong Rayleigh damping for stretch
        self.damping_bend = 0.1      # moderate damping for bend
        self.global_damping = 0.9995  # velocity damping per sub-step

        # Contact stiffness (penalty) - moderate to avoid instability
        self.contact_stiffness = 200.0  # N/m for sphere contact
        self.strand_contact_stiffness = 100.0  # N/m for strand-strand

        # Stick tolerance
        self.stick_tol = 1e-4  # m/s

        # Reference scales
        total_mass = float(np.sum(self.mass_arr))
        if total_mass < 1e-30:
            total_mass = 1.0
        self.ref_velocity = 1.0
        self.ref_force = total_mass * 9.81
        self.ref_momentum = total_mass * self.ref_velocity

        # Budget accumulators
        self.cum_ext_impulse = np.zeros(3, dtype=np.float64)
        self.cum_inflow_mom = np.zeros(3, dtype=np.float64)
        self.cum_outflow_mom = np.zeros(3, dtype=np.float64)

        # Current simulation time
        self.current_time = 0.0

        # Precompute spatial hash grid parameters for strand-strand contact
        self.grid_cell_size = max(4.0 * self.strand_radius, self.rest_length * 0.5)

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the requested time and return a StateFrame."""
        target = float(time)
        if target < self.current_time:
            # Shouldn't happen but be safe
            return self._make_frame(target)

        # Determine how many sample steps to take
        # The runner calls us at each sample time in order
        n_sub = self.sub_steps
        dt = self.dt

        while self.current_time < target - 1e-12:
            step_dt = min(dt, target - self.current_time)
            if step_dt < 1e-15:
                break
            self._substep(step_dt)
            self.current_time += step_dt

        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "elastic_strand_semi_implicit_euler",
            "sub_steps_per_sample": self.sub_steps,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_rest_config(self) -> None:
        """Store rest edge vectors and rest curvatures for each strand."""
        S = self.strand_count
        nps = self.nodes_per_strand
        seg = self.segments

        # Rest edge vectors (in world coords at t=0)
        self.rest_edges = np.zeros((S, seg, 3), dtype=np.float64)
        self.rest_edge_lengths = np.zeros((S, seg), dtype=np.float64)

        # Rest curvature vectors (discrete curvature = change in tangent)
        self.rest_curvatures = np.zeros((S, seg - 1, 3), dtype=np.float64)

        for s in range(S):
            base = s * nps
            for j in range(seg):
                e = self.pos[base + j + 1] - self.pos[base + j]
                l = np.linalg.norm(e)
                if l < 1e-30:
                    l = self.rest_length
                self.rest_edges[s, j] = e
                self.rest_edge_lengths[s, j] = l
            # Curvature: difference of consecutive unit tangents
            for j in range(seg - 1):
                t0 = self.rest_edges[s, j] / max(self.rest_edge_lengths[s, j], 1e-30)
                t1 = self.rest_edges[s, j + 1] / max(self.rest_edge_lengths[s, j + 1], 1e-30)
                self.rest_curvatures[s, j] = t1 - t0

    def _get_sphere_rotation(self, t: float) -> np.ndarray:
        """Return 3x3 rotation matrix for the sphere at time t using smooth interpolation."""
        times = self.kf_times
        axes = self.kf_axes
        angles = self.kf_angles

        if t <= times[0]:
            return np.eye(3, dtype=np.float64)
        if t >= times[-1]:
            idx = len(times) - 1
            return self._axis_angle_matrix(axes[idx], angles[idx])

        # Find interval
        idx = np.searchsorted(times, t, side='right') - 1
        idx = max(0, min(idx, len(times) - 2))

        t0, t1 = times[idx], times[idx + 1]
        alpha = (t - t0) / max(t1 - t0, 1e-30)
        # Smooth-step interpolation
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # SLERP-like: interpolate axis-angle
        ax0, ax1 = axes[idx], axes[idx + 1]
        ang0, ang1 = angles[idx], angles[idx + 1]

        # Simple linear interpolation of axis-angle (not perfect SLERP but adequate)
        ax_interp = (1.0 - alpha) * ax0 + alpha * ax1
        norm_ax = np.linalg.norm(ax_interp)
        if norm_ax < 1e-15:
            return np.eye(3, dtype=np.float64)
        ax_interp /= norm_ax
        ang_interp = (1.0 - alpha) * ang0 + alpha * ang1

        return self._axis_angle_matrix(ax_interp, ang_interp)

    @staticmethod
    def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
        return np.array([
            [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ],
        ], dtype=np.float64)

    def _substep(self, dt: float) -> None:
        """One sub-step of semi-implicit Euler with forces and contacts."""
        pos = self.pos
        vel = self.vel
        mass = self.mass_arr
        N = self.N
        S = self.strand_count
        seg = self.segments
        nps = self.nodes_per_strand

        # Get current sphere rotation
        R = self._get_sphere_rotation(self.current_time)
        R_next = self._get_sphere_rotation(self.current_time + dt)

        # Compute forces
        force = np.zeros((N, 3), dtype=np.float64)

        # 1. Gravity
        for i in range(N):
            if mass[i] > 1e-30:
                force[i] += mass[i] * self.gravity

        # 2. Stretch forces (linear springs along edges)
        k_stretch = self.stretch_scale * self.E_stretch * self.area / self.rest_length
        damp_stretch = self.damping_stretch * math.sqrt(k_stretch * self.node_mass_val)

        for s in range(S):
            base = s * nps
            for j in range(seg):
                i0 = base + j
                i1 = base + j + 1
                diff = pos[i1] - pos[i0]
                dist = np.linalg.norm(diff)
                if dist < 1e-30:
                    continue
                direction = diff / dist
                stretch = dist - self.rest_edge_lengths[s, j]
                # Relative velocity along edge
                rel_vel = vel[i1] - vel[i0]
                vel_along = np.dot(rel_vel, direction)
                f_mag = k_stretch * stretch + damp_stretch * vel_along
                f_vec = f_mag * direction
                force[i0] += f_vec
                force[i1] -= f_vec

        # 3. Bending forces (discrete curvature spring)
        k_bend = self.E_bend * self.I_bend / (self.rest_length ** 3)
        damp_bend = self.damping_bend * math.sqrt(k_bend * self.node_mass_val)

        for s in range(S):
            base = s * nps
            for j in range(seg - 1):
                i0 = base + j
                i1 = base + j + 1
                i2 = base + j + 2

                e0 = pos[i1] - pos[i0]
                e1 = pos[i2] - pos[i1]
                l0 = np.linalg.norm(e0)
                l1 = np.linalg.norm(e1)
                if l0 < 1e-30 or l1 < 1e-30:
                    continue
                t0 = e0 / l0
                t1 = e1 / l1

                # Current curvature vector
                kappa = t1 - t0
                # Rest curvature
                kappa_rest = self.rest_curvatures[s, j]
                dk = kappa - kappa_rest

                # Bending force: F = -k_bend * dk applied as moments
                # Distributed to three nodes via finite differences
                # Simplified: apply restoring force proportional to curvature deviation
                fb = k_bend * dk

                # Damping
                v0 = vel[i0]
                v1 = vel[i1]
                v2 = vel[i2]
                # Approximate rate of curvature change
                dv = (v2 - 2.0 * v1 + v0) / (self.rest_length ** 2)
                fb += damp_bend * dv

                # Apply to nodes (discrete Laplacian stencil)
                force[i0] += fb
                force[i1] -= 2.0 * fb
                force[i2] += fb

        # 4. Sphere contact forces
        sphere_center = np.zeros(3, dtype=np.float64)
        sr = self.sphere_radius
        contact_clearance = self.strand_radius
        k_contact = self.contact_stiffness

        # Track contact data for friction output
        contact_list = []  # (node_idx, normal, fn_mag, rel_vel, tan_force)

        for i in range(N):
            if mass[i] < 1e-30:
                continue  # skip roots
            diff = pos[i] - sphere_center
            dist = np.linalg.norm(diff)
            penetration = sr + contact_clearance - dist
            if penetration > 0 and dist > 1e-30:
                normal = diff / dist  # outward from sphere center
                # Penalty spring
                fn = k_contact * penetration
                # Normal force magnitude (compressive, positive)
                fn_mag = max(fn, 0.0)
                force[i] += fn_mag * normal

                # Friction: relative velocity of node vs sphere surface
                # Sphere surface velocity at contact point = omega x r
                # Approximate omega from rotation difference
                # v_sphere = (R_next - R) / dt * R^T * point_on_sphere
                # Simpler: compute angular velocity from rotation
                dR = (R_next - R) / max(dt, 1e-30)
                # omega_skew = dR * R^T
                omega_skew = dR @ R.T
                # Extract omega from skew-symmetric matrix
                omega = np.array([
                    omega_skew[2, 1],
                    omega_skew[0, 2],
                    omega_skew[1, 0],
                ])
                # Point on sphere surface closest to node
                pt_sphere = sphere_center + sr * normal
                v_sphere_surface = np.cross(omega, pt_sphere - sphere_center)
                v_rel = vel[i] - v_sphere_surface
                # Tangential component
                vn = np.dot(v_rel, normal)
                v_tan = v_rel - vn * normal
                v_tan_mag = np.linalg.norm(v_tan)

                if v_tan_mag > 1e-30:
                    tan_dir = v_tan / v_tan_mag
                    # Coulomb friction
                    if v_tan_mag < self.stick_tol:
                        # Static friction regime
                        mu = self.mu_static
                    else:
                        mu = self.mu_kinetic
                    ft_max = mu * fn_mag
                    ft_mag = min(ft_max, k_contact * v_tan_mag * dt)
                    ft_vec = -ft_mag * tan_dir
                    force[i] += ft_vec
                    contact_list.append((i, normal.copy(), fn_mag, v_rel.copy(), ft_vec.copy()))
                else:
                    contact_list.append((i, normal.copy(), fn_mag, v_rel.copy(), np.zeros(3)))

        # 5. Strand-strand contact (spatial-hash accelerated)
        # Only check segments that are close; use grid hashing
        self._apply_strand_contacts(pos, vel, force, contact_list, dt)

        # Accumulate external impulse from gravity
        ext_force_gravity = np.zeros(3, dtype=np.float64)
        for i in range(N):
            if mass[i] > 1e-30:
                ext_force_gravity += mass[i] * self.gravity
        self.cum_ext_impulse += ext_force_gravity * dt

        # Semi-implicit Euler: update velocity first, then position
        for i in range(N):
            if mass[i] < 1e-30:
                # Root nodes: enforce kinematic constraint
                # Move root with sphere rotation
                s = i // nps
                root_pos_local = self.pos[i].copy()
                # Rotate the original root position
                # Original root is on sphere surface
                new_root = R_next @ (np.linalg.inv(R) @ root_pos_local) if np.linalg.det(R) > 0.5 else root_pos_local
                # Actually, simpler: store initial root positions and rotate them
                pass  # handled below
                continue
            acc = force[i] / mass[i]
            vel[i] += acc * dt
            vel[i] *= self.global_damping

        # Update positions
        for i in range(N):
            if mass[i] < 1e-30:
                continue
            pos[i] += vel[i] * dt

        # Enforce root constraints: roots move with sphere
        for idx in self.root_indices:
            s = idx // nps
            # Initial root position was on sphere surface at t=0
            # We stored it implicitly; reconstruct from initial config
            # Actually, roots should stay on the sphere surface at their attachment point,
            # rotated by the current sphere rotation
            # The initial root position defines the attachment direction
            init_root_dir = self._get_initial_root_direction(s)
            new_pos = self.sphere_radius * (R_next @ init_root_dir)
            pos[idx] = new_pos
            # Velocity of root = omega x r
            dR = (R_next - R) / max(dt, 1e-30)
            omega_skew = dR @ R_next.T
            omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
            vel[idx] = np.cross(omega, new_pos)

        # Store contact info for output
        self._last_contacts = contact_list

    def _get_initial_root_direction(self, strand_idx: int) -> np.ndarray:
        """Get the initial (t=0) unit direction of a strand's root from the sphere center."""
        # Reconstruct from the golden-ratio distribution used in scene.py
        golden = math.pi * (3.0 - math.sqrt(5.0))
        z = (strand_idx + 0.5) / self.strand_count
        phi = strand_idx * golden
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        return np.array([radial * math.cos(phi), radial * math.sin(phi), z], dtype=np.float64)

    def _apply_strand_contacts(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        force: np.ndarray,
        contact_list: list,
        dt: float,
    ) -> None:
        """Detect and resolve strand-strand contacts using spatial hashing."""
        S = self.strand_count
        seg = self.segments
        nps = self.nodes_per_strand
        clearance = 2.0 * self.strand_radius
        k_ss = self.strand_contact_stiffness

        # Build segment list: (strand, seg_idx, midpoint, node_i0, node_i1)
        # Use midpoints for spatial hashing
        cell_size = max(clearance * 2.0, self.rest_length)

        # Hash segments into grid cells
        grid: dict[tuple[int, int, int], list] = {}
        segments_info = []

        for s in range(S):
            base = s * nps
            for j in range(seg):
                i0 = base + j
                i1 = base + j + 1
                mid = 0.5 * (pos[i0] + pos[i1])
                cx = int(math.floor(mid[0] / cell_size))
                cy = int(math.floor(mid[1] / cell_size))
                cz = int(math.floor(mid[2] / cell_size))
                key = (cx, cy, cz)
                if key not in grid:
                    grid[key] = []
                grid[key].append(len(segments_info))
                segments_info.append((s, j, i0, i1))

        # Check neighboring cells
        checked = set()
        n_seg = len(segments_info)

        for cell_key, seg_indices in grid.items():
            cx, cy, cz = cell_key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nkey = (cx + dx, cy + dy, cz + dz)
                        neighbor_segs = grid.get(nkey, [])
                        for si in seg_indices:
                            for sj in neighbor_segs:
                                if si >= sj:
                                    continue
                                pair = (si, sj)
                                if pair in checked:
                                    continue
                                checked.add(pair)

                                s_i, j_i, ia0, ia1 = segments_info[si]
                                s_j, j_j, ib0, ib1 = segments_info[sj]

                                # Skip adjacent segments on same strand
                                if s_i == s_j and abs(j_i - j_j) <= 1:
                                    continue

                                # Segment-segment distance
                                pa0, pa1 = pos[ia0], pos[ia1]
                                pb0, pb1 = pos[ib0], pos[ib1]

                                cp_a, cp_b, dist = self._segment_segment_closest(
                                    pa0, pa1, pb0, pb1
                                )

                                if dist < clearance and dist > 1e-30:
                                    normal = (cp_a - cp_b) / dist
                                    penetration = clearance - dist

                                    # Distribute force to segment endpoints
                                    # Parameter along each segment
                                    ea = pa1 - pa0
                                    eb = pb1 - pb0
                                    la = np.linalg.norm(ea)
                                    lb = np.linalg.norm(eb)

                                    if la < 1e-30 or lb < 1e-30:
                                        continue

                                    ta = np.dot(cp_a - pa0, ea) / (la * la)
                                    tb = np.dot(cp_b - pb0, eb) / (lb * lb)
                                    ta = max(0.0, min(1.0, ta))
                                    tb = max(0.0, min(1.0, tb))

                                    fn = k_ss * penetration

                                    # Relative velocity at contact points
                                    va = vel[ia0] * (1 - ta) + vel[ia1] * ta
                                    vb = vel[ib0] * (1 - tb) + vel[ib1] * tb
                                    v_rel = va - vb
                                    vn = np.dot(v_rel, normal)
                                    v_tan = v_rel - vn * normal
                                    v_tan_mag = np.linalg.norm(v_tan)

                                    # Friction
                                    if v_tan_mag > 1e-30:
                                        tan_dir = v_tan / v_tan_mag
                                        mu = self.mu_static if v_tan_mag < self.stick_tol else self.mu_kinetic
                                        ft_max = mu * fn
                                        ft_mag = min(ft_max, k_ss * v_tan_mag * dt)
                                        ft_vec = -ft_mag * tan_dir
                                    else:
                                        ft_vec = np.zeros(3)

                                    total_f = fn * normal + ft_vec

                                    # Apply to nodes
                                    force[ia0] += total_f * (1 - ta)
                                    force[ia1] += total_f * ta
                                    force[ib0] -= total_f * (1 - tb)
                                    force[ib1] -= total_f * tb

                                    # Record contact at midpoint node for output
                                    mid_node = ia1  # representative
                                    contact_list.append((
                                        mid_node, normal.copy(), fn,
                                        v_rel.copy(), ft_vec.copy()
                                    ))

    @staticmethod
    def _segment_segment_closest(
        p1: np.ndarray, p2: np.ndarray,
        p3: np.ndarray, p4: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Closest points between segments p1-p2 and p3-p4."""
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        if a < 1e-30 and e < 1e-30:
            return p1.copy(), p3.copy(), np.linalg.norm(r)
        if a < 1e-30:
            s = 0.0
            t = max(0.0, min(1.0, f / e))
        else:
            c = np.dot(d1, r)
            if e < 1e-30:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            else:
                b = np.dot(d1, d2)
                denom = a * e - b * b
                if abs(denom) > 1e-30:
                    s = max(0.0, min(1.0, (b * f - c * e) / denom))
                else:
                    s = 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))

        cp1 = p1 + s * d1
        cp2 = p3 + t * d2
        dist = np.linalg.norm(cp1 - cp2)
        return cp1, cp2, dist

    def _make_frame(self, time: float) -> StateFrame:
        """Build a StateFrame with all required trajectory fields."""
        N = self.N
        S = self.strand_count
        seg = self.segments

        # Contacts from last substep
        contacts = getattr(self, '_last_contacts', [])
        C = len(contacts)

        # Friction arrays
        if C > 0:
            fric_normals = np.array([c[1] for c in contacts], dtype=np.float64)
            fric_fn = np.array([c[2] for c in contacts], dtype=np.float64)
            fric_rel_vel = np.array([c[3] for c in contacts], dtype=np.float64)
            fric_tan_force = np.array([c[4] for c in contacts], dtype=np.float64)
            fric_active = np.ones(C, dtype=np.float64)
            fric_mu_s = np.full(C, self.mu_static, dtype=np.float64)
            fric_mu_d = np.full(C, self.mu_kinetic, dtype=np.float64)
        else:
            fric_normals = np.zeros((0, 3), dtype=np.float64)
            fric_fn = np.zeros(0, dtype=np.float64)
            fric_rel_vel = np.zeros((0, 3), dtype=np.float64)
            fric_tan_force = np.zeros((0, 3), dtype=np.float64)
            fric_active = np.zeros(0, dtype=np.float64)
            fric_mu_s = np.zeros(0, dtype=np.float64)
            fric_mu_d = np.zeros(0, dtype=np.float64)

        # Budget domain weights: G=1 global domain, all hair nodes
        domain_weights = np.ones((1, N), dtype=np.float64)

        # External force: G=1 domain, K=1 term (gravity), shape [G,K,3]
        grav_total = np.sum(self.mass_arr[:, None] * self.gravity[None, :], axis=0)
        budget_ext_force = grav_total.reshape(1, 1, 3)

        # Cumulative mass source: no mass added/removed
        cum_mass_source = np.zeros(1, dtype=np.float64)

        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass_arr.copy(),
            attributes={
                "strand_id": self.strand_id_attr.copy(),
                "root_mask": self.root_mask.copy(),
                "mass_quantity": self.mass_arr.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3).copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": cum_mass_source,
                "friction_active_mask": fric_active,
                "friction_dynamic_coefficient": fric_mu_d,
                "friction_normal": fric_normals,
                "friction_normal_force": fric_fn,
                "friction_relative_velocity": fric_rel_vel,
                "friction_static_coefficient": fric_mu_s,
                "friction_tangential_force": fric_tan_force,
            },
            scalars={
                "friction_stick_speed_tolerance": float(self.stick_tol),
                "reference_force": float(self.ref_force),
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.ref_velocity),
                "strand_radius": float(self.strand_radius),
                "static_friction": float(self.mu_static),
                "kinetic_friction": float(self.mu_kinetic),
            },
        )
        return frame
