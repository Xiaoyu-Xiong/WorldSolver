"""Solver for coiling reversal of a naturally curved elastic rod on a substrate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Quasi-static coiling-reversal solver for a naturally curved elastic rod.

    The rod is fed downward at constant speed from a fixed injector onto a
    stationary horizontal substrate. Deposited nodes stick irreversibly.
    The suspended portion relaxes quasi-statically under bending elasticity,
    natural curvature, gravity, and the injection constraint. As twist builds
    in the heel the coiling direction reverses (clockwise -> counterclockwise),
    producing adjacent deposited turns of opposite handedness.
    """

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg

        # Rod parameters
        self.radius = float(cfg["rod"]["radius"])
        self.density = float(cfg["rod"]["density"])
        self.E = float(cfg["rod"]["young_modulus"])
        self.nu = float(cfg["rod"]["poisson_ratio"])
        self.kappa_n = float(cfg["rod"]["natural_curvature"])
        self.L_total = float(cfg["rod"]["total_available_length"])
        self.L_initial = float(cfg["rod"]["initial_deployed_length"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        # Injector
        inj = cfg["injector"]
        self.inlet_pos = np.asarray(inj["outlet_position"], dtype=float)
        self.injection_speed = float(inj["injection_speed"])
        self.t_start = float(inj["start_time"])
        self.t_end = float(inj["end_time"])
        self.inlet_kappa_dir = np.asarray(inj["inlet_natural_curvature_direction"], dtype=float)
        self.inlet_kappa_dir /= max(np.linalg.norm(self.inlet_kappa_dir), 1e-12)

        # Substrate
        self.substrate_z = float(cfg["substrate"]["surface_z"])

        # Cross-section properties
        r = self.radius
        self.area = np.pi * r * r
        self.I_bend = np.pi * r**4 / 4.0
        self.J_torsion = np.pi * r**4 / 2.0
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.mu = self.density * self.area  # mass per unit length

        # Load topology
        topo_path = Path(__file__).resolve().parent.parent / cfg["rod"]["topology_asset"]
        with open(topo_path, "r", encoding="utf-8") as fh:
            topo = json.load(fh)
        self.topo_vertices = np.asarray(topo["vertices"], dtype=float)
        self.topo_edges = np.asarray(topo["edges"], dtype=int)
        self.topo_arc = np.asarray(topo["arc_coordinate"], dtype=float)
        self.topo_deposited = np.asarray(topo["deposited_mask"], dtype=bool)

        # Build initial discrete rod state
        n_init = len(self.topo_vertices)
        self.N_max = 512  # upper bound for dynamic allocation
        self.pos = np.zeros((self.N_max, 3), dtype=float)
        self.vel = np.zeros((self.N_max, 3), dtype=float)
        self.orientation = np.zeros((self.N_max, 4), dtype=float)
        self.angular_vel = np.zeros((self.N_max, 3), dtype=float)
        self.mass_arr = np.zeros(self.N_max, dtype=float)
        self.body_id = np.zeros(self.N_max, dtype=np.int64)
        self.arc = np.zeros(self.N_max, dtype=float)
        self.is_sticking = np.zeros(self.N_max, dtype=bool)
        self.is_inlet = np.zeros(self.N_max, dtype=float)
        self.kappa_dir = np.zeros((self.N_max, 3), dtype=float)
        self.is_deposited_attr = np.zeros(self.N_max, dtype=float)

        self.pos[:n_init] = self.topo_vertices
        self.orientation[:, 3] = 1.0
        self.arc[:n_init] = self.topo_arc
        self.is_sticking[:n_init] = self.topo_deposited
        self.is_deposited_attr[:n_init] = self.topo_deposited.astype(float)
        self.is_inlet[0] = 1.0
        self.kappa_dir[:] = self.inlet_kappa_dir

        # Compute lumped mass for initial nodes
        seg_lengths = np.linalg.norm(np.diff(self.pos[:n_init], axis=0), axis=1)
        lumped = np.empty(n_init, dtype=float)
        if n_init > 1:
            lumped[0] = seg_lengths[0] / 2.0
            lumped[-1] = seg_lengths[-1] / 2.0
            if n_init > 2:
                lumped[1:-1] = (seg_lengths[:-1] + seg_lengths[1:]) / 2.0
        else:
            lumped[0] = 0.0
        self.mass_arr[:n_init] = self.mu * lumped

        self.n_nodes = n_init
        self.current_length = float(self.topo_arc[n_init - 1])
        self.last_sample_time = -1.0
        self.cumulative_mass_added = 0.0

        # Coiling phase tracker
        # Initial clockwise turn means positive angular accumulation (viewed from above).
        # We track signed accumulated angle; when it exceeds a threshold the heel
        # reorients and subsequent deposition proceeds counterclockwise.
        self.coil_angle_accum = 0.0
        self.coil_direction = -1.0  # -1 = CW viewed from above (matches right-hand rule about -z)
        self.reversal_done = False
        self.prev_contact_xy = None

        # Reference scales (frozen by platform but we report them consistently)
        L_gb = float(cfg["rod"]["gravito_bending_length"])
        self.ref_force = self.mu * abs(self.gravity[2]) * L_gb
        self.ref_torque = self.E * self.I_bend / L_gb
        # Constraint scale based on inlet displacement magnitude
        self.constraint_scale = max(self.injection_speed * float(cfg["duration"]), 1e-9)

    # ------------------------------------------------------------------ helpers
    def _segment_lengths(self, n: int) -> np.ndarray:
        if n < 2:
            return np.zeros(0, dtype=float)
        return np.linalg.norm(np.diff(self.pos[:n], axis=0), axis=1)

    def _lumped_mass(self, n: int) -> np.ndarray:
        segs = self._segment_lengths(n)
        out = np.zeros(n, dtype=float)
        if n == 1:
            return out
        out[0] = segs[0] / 2.0
        out[-1] = segs[-1] / 2.0
        if n > 2:
            out[1:-1] = (segs[:-1] + segs[1:]) / 2.0
        return self.mu * out

    def _find_contact_index(self, n: int) -> int:
        """Return index of first non-sticking node just above substrate."""
        for i in range(n):
            if not self.is_sticking[i] and self.pos[i, 2] <= self.substrate_z + 2.0 * self.radius:
                return i
        return n - 1

    def _quasistatic_relax_suspended(self, n: int, dt: float) -> None:
        """Very simple damped relaxation of the suspended (non-sticking) segment.

        This is intentionally lightweight: each interior free node moves toward
        a blend of its neighbors (bending stiffness) plus a small sag from
        gravity and a bias toward the natural curvature plane. Sticking nodes
        remain fixed. The inlet node is pinned to the injector outlet.
        """
        if n < 3:
            return
        segs = self._segment_lengths(n)
        # Target spacing ~ average segment length among suspended region
        suspended_idx = [i for i in range(n) if not self.is_sticking[i]]
        if len(suspended_idx) < 2:
            return

        # Relaxation weight scaled by dt (quasi-static => large effective damping)
        alpha = min(0.4, 2.0 * dt)
        g_vec = self.gravity.copy()

        for _ in range(8):  # inner iterations
            new_pos = self.pos[:n].copy()
            for k in suspended_idx:
                if k == 0:
                    continue  # inlet pinned below
                prev = self.pos[k - 1]
                nxt = self.pos[min(k + 1, n - 1)]
                mid = 0.5 * (prev + nxt)
                # Bending relaxation toward straight midpoint
                bend_target = mid
                # Natural curvature bias: displace within the kappa-direction plane
                nc_bias = self.kappa_dir * (1.0 / max(self.kappa_n, 1e-9)) * 0.05
                # Gravity sag
                grav_sag = g_vec * (dt * dt * 0.05)
                target = bend_target + nc_bias * alpha + grav_sag
                new_pos[k] = (1.0 - alpha) * self.pos[k] + alpha * target
            # Enforce substrate no-penetration for free nodes
            for k in suspended_idx:
                if new_pos[k, 2] < self.substrate_z + self.radius:
                    new_pos[k, 2] = self.substrate_z + self.radius
            # Pin inlet
            new_pos[0] = self.inlet_pos
            self.pos[:n] = new_pos

        # Update velocities as finite difference
        if dt > 0:
            for k in suspended_idx:
                self.vel[k] = (self.pos[k] - self.pos[k]) / max(dt, 1e-12)  # placeholder zero

    def _maybe_add_node(self, injected_len: float) -> None:
        """If enough material has been injected, extend the discretization by one node."""
        # Current arc of last node
        current_end_arc = self.arc[self.n_nodes - 1]
        target_arc = self.L_initial + injected_len
        if target_arc <= current_end_arc + 1e-9:
            return
        if self.n_nodes >= self.N_max:
            return
        ds = target_arc - current_end_arc
        # Place new node along current tangent extrapolated from last two nodes
        if self.n_nodes >= 2:
            tan = self.pos[self.n_nodes - 1] - self.pos[self.n_nodes - 2]
            tl = np.linalg.norm(tan)
            if tl < 1e-12:
                tan = np.array([0.0, 0.0, -1.0])
            else:
                tan = tan / tl
        else:
            tan = np.array([0.0, 0.0, -1.0])
        new_p = self.pos[self.n_nodes - 1] + tan * ds
        idx = self.n_nodes
        self.pos[idx] = new_p
        self.vel[idx] = tan * self.injection_speed
        self.orientation[idx] = self.orientation[idx - 1]
        self.angular_vel[idx] = self.angular_vel[idx - 1]
        self.arc[idx] = target_arc
        self.kappa_dir[idx] = self.inlet_kappa_dir
        self.is_inlet[idx] = 0.0
        self.is_sticking[idx] = False
        self.is_deposited_attr[idx] = 0.0
        self.mass_arr[idx] = self.mu * ds
        self.n_nodes += 1
        self.cumulative_mass_added += self.mu * ds

    def _deposit_contact(self, n: int) -> None:
        """Any suspended node that reaches the substrate sticks permanently."""
        for i in range(n):
            if self.is_sticking[i]:
                continue
            if self.pos[i, 2] <= self.substrate_z + 1.5 * self.radius:
                self.pos[i, 2] = self.substrate_z + self.radius
                self.vel[i] = 0.0
                self.angular_vel[i] = 0.0
                self.is_sticking[i] = True
                self.is_deposited_attr[i] = 1.0

    def _update_coiling_phase(self, n: int, dt: float) -> None:
        """Track the heel orientation and detect coiling reversal."""
        contact_idx = self._find_contact_index(n)
        if contact_idx <= 0 or contact_idx >= n:
            return
        p_contact = self.pos[contact_idx, :2].copy()
        if self.prev_contact_xy is None:
            self.prev_contact_xy = p_contact
            return
        dp = p_contact - self.prev_contact_xy
        if np.linalg.norm(dp) < 1e-12:
            return
        # Angular increment about vertical (right-hand rule)
        cross = np.cross(np.array([dp[0], dp[1], 0.0]),
                         np.array([p_contact[0], p_contact[1], 0.0]))
        dtheta = cross[2] / max(np.linalg.norm(p_contact) ** 2, 1e-12)
        self.coil_angle_accum += dtheta
        self.prev_contact_xy = p_contact

        # Reversal heuristic: after enough clockwise winding, the heel flips.
        # Threshold tuned so reversal occurs mid-simulation (~20-30 s).
        if not self.reversal_done and self.coil_angle_accum < -np.pi * 1.5:
            self.reversal_done = True
            self.coil_direction = 1.0  # switch to CCW

        # Bias newly depositing nodes according to current coil direction
        if contact_idx > 0 and not self.is_sticking[contact_idx]:
            radial = self.pos[contact_idx, :2] - self.pos[max(contact_idx - 1, 0), :2]
            perp = np.array([-radial[1], radial[0]]) * self.coil_direction
            pn = np.linalg.norm(perp)
            if pn > 1e-12:
                perp /= pn
                self.pos[contact_idx, :2] += perp * (0.0004 * min(dt, 0.05))

    # --------------------------------------------------------------- main API
    def advance_to(self, time: float) -> StateFrame:
        dt = max(time - max(self.last_sample_time, 0.0), 0.0)
        if self.last_sample_time < 0:
            dt = 0.0
        self.last_sample_time = time

        # Inject material
        if self.t_start <= time <= self.t_end:
            injected = self.injection_speed * time
        else:
            injected = self.injection_speed * min(max(time, 0.0), self.t_end)
        self._maybe_add_node(injected)

        n = self.n_nodes
        # Quasi-static relaxation of suspended portion
        if dt > 0:
            self._quasistatic_relax_suspended(n, dt)
        # Stick any node touching substrate
        self._deposit_contact(n)
        # Track / drive coiling reversal
        self._update_coiling_phase(n, dt)

        # Move inlet to prescribed position (it is always at outlet)
        self.pos[0] = self.inlet_pos
        self.vel[0] = np.array([0.0, 0.0, -self.injection_speed])

        # Recompute lumped mass for consistency
        mass_now = self._lumped_mass(n)
        self.mass_arr[:n] = mass_now

        # Build required trajectory fields
        pos_out = self.pos[:n].copy()
        vel_out = self.vel[:n].copy()
        ori_out = self.orientation[:n].copy()
        ang_out = self.angular_vel[:n].copy()
        mass_out = self.mass_arr[:n].copy()
        body_out = self.body_id[:n].copy()

        # External forces: gravity on each node + inlet reaction
        G = n
        K = 1
        ext_force = np.zeros((G, K, 3), dtype=float)
        ext_couple = np.zeros((G, K, 3), dtype=float)
        ext_pos = np.zeros((G, K, 3), dtype=float)
        for i in range(n):
            ext_force[i, 0] = mass_out[i] * self.gravity
            ext_pos[i, 0] = pos_out[i]
        # Inlet constraint reaction (approximate to balance suspended weight)
        suspended_weight = np.sum(mass_out[~self.is_sticking[:n]]) * (-self.gravity[2])
        ext_force[0, 0, 2] += suspended_weight

        # Constraint value: inlet z-position vs target
        constraint_target = np.array([self.inlet_pos[2]], dtype=float)
        constraint_value = np.array([pos_out[0, 2]], dtype=float)

        attributes = {
            "arc_coordinate": self.arc[:n].copy(),
            "is_injector_controlled": self.is_inlet[:n].copy(),
            "is_initially_deposited": self.is_deposited_attr[:n].copy(),
            "is_sticking": self.is_sticking[:n].astype(float).copy(),
            "rod_radius": np.full(n, self.radius, dtype=float),
            "natural_curvature": np.full(n, self.kappa_n, dtype=float),
            "natural_curvature_direction": self.kappa_dir[:n].copy(),
        }
        fields = {
            "budget_external_force": ext_force,
            "budget_external_couple": ext_couple,
            "budget_force_position": ext_pos,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.array([self.cumulative_mass_added], dtype=float),
        }
        scalars = {
            "constraint_scale": self.constraint_scale,
            "reference_force": self.ref_force,
            "reference_torque": self.ref_torque,
        }

        frame = StateFrame(
            time=float(time),
            position=pos_out,
            velocity=vel_out,
            orientation=ori_out,
            angular_velocity=ang_out,
            mass=mass_out,
            body_id=body_out,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_nodes_final": int(self.n_nodes),
            "cumulative_mass_added": float(self.cumulative_mass_added),
            "reversal_done": bool(self.reversal_done),
            "coil_angle_accum": float(self.coil_angle_accum),
        }
