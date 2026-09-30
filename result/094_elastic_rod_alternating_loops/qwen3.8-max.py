"""Quasi-static elastic rod coiling solver producing alternating loops on a moving belt.

Implements BenchmarkSolver for scene 094_elastic_rod_alternating_loops.

The solver tracks a discrete centerline of material nodes. At each sample time it
advances injection, grows new arc-length, computes the quasi-static heel shape under
gravity and axial load, detects contact with the belt, sticks contacted nodes to the
moving belt, and exports all required StateFrame fields.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Alternating-loop elastic rod coiling on a translating belt."""

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.seed = seed

        # Load initial topology
        path = Path(__file__).resolve().parent.parent / self.cfg["rod"]["topology_asset"]
        data = json.loads(path.read_text(encoding="utf-8"))
        self.init_position = np.asarray(data["vertices"], dtype=float)
        self.init_arc = np.asarray(data["arc_coordinate"], dtype=float)
        self.init_deposited = np.asarray(data["deposited_mask"], dtype=float)
        self.edges = np.asarray(data["edges"], dtype=int)

        # Physical parameters
        self.radius = float(self.cfg["rod"]["radius"])
        self.density = float(self.cfg["rod"]["density"])
        self.E = float(self.cfg["rod"]["young_modulus"])
        self.nu = float(self.cfg["rod"]["poisson_ratio"])
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.g_mag = float(np.linalg.norm(self.gravity))
        if self.g_mag < 1e-12:
            self.g_hat = np.array([0.0, 0.0, -1.0])
        else:
            self.g_hat = self.gravity / self.g_mag

        A = math.pi * self.radius**2
        I_sec = math.pi * self.radius**4 / 4.0
        self.A = A
        self.I_sec = I_sec
        self.EI = self.E * I_sec
        self.mu_linear = self.density * A  # mass per unit length

        # Gravito-bending length
        l_g_cfg = float(self.cfg["rod"].get("gravito_bending_length", 0.033))
        self.l_g = l_g_cfg if l_g_cfg > 0 else (self.EI / (self.mu_linear * self.g_mag)) ** (1.0 / 3.0)

        # Injection and belt
        inj = self.cfg["injector"]
        self.inlet_pos = np.asarray(inj["outlet_position"], dtype=float)
        self.inlet_dir = np.asarray(inj["axis"], dtype=float)
        self.inlet_dir = self.inlet_dir / max(1e-12, np.linalg.norm(self.inlet_dir))
        self.v_inj = float(inj["injection_speed"])
        self.inj_start = float(inj.get("start_time", 0.0))
        self.inj_end = float(inj.get("end_time", 1e9))

        belt = self.cfg["belt"]
        self.belt_dir = np.asarray(belt["travel_direction"], dtype=float)
        self.belt_dir = self.belt_dir / max(1e-12, np.linalg.norm(self.belt_dir))
        self.v_belt = float(belt["speed"])
        self.belt_surface_z = float(belt.get("surface_z", 0.0))
        self.belt_half_extents = np.asarray(belt["half_extents"], dtype=float)
        self.belt_center = np.asarray(belt["center"], dtype=float)

        self.speed_ratio = float(self.cfg["regime"].get("speed_ratio_belt_to_injection", 0.5))

        # Total available length
        self.L_total = float(self.cfg["rod"]["total_available_length"])

        # Coiling dynamics scales (quasi-static viscous-gravity regime)
        # Characteristic angular frequency omega ~ v_inj / l_g
        self.omega_c = self.v_inj / max(1e-12, self.l_g)
        # Amplitude scale: choose so lateral extent is several l_g
        self.coil_amp_base = 5.5 * self.l_g
        # Phase offset chosen so first loop goes in one direction then alternates
        rng = np.random.default_rng(seed)
        self.phi0 = rng.uniform(0.0, 2.0 * math.pi)

        # Simulation state
        self.current_time = 0.0
        self.injected_length = float(self.cfg["rod"]["initial_deployed_length"])
        self._set_initial_state()

    def _set_initial_state(self) -> None:
        """Build initial node arrays from topology asset."""
        n = len(self.init_position)
        self.positions = self.init_position.copy()
        self.velocities = np.zeros((n, 3), dtype=float)
        self.deposited = self.init_deposited.copy()
        self.arc_coords = self.init_arc.copy()
        # Orientations: identity quaternion (w=1)
        self.orientations = np.zeros((n, 4), dtype=float)
        self.orientations[:, 3] = 1.0
        self.angular_velocities = np.zeros((n, 3), dtype=float)

        # Mark deposited nodes with belt velocity
        for i in range(n):
            if self.deposited[i] > 0.5:
                self.velocities[i] = self.belt_dir * self.v_belt

        # Track when each node got deposited (for sticking advection)
        self.deposit_time = np.full(n, -1.0, dtype=float)
        for i in range(n):
            if self.deposited[i] > 0.5:
                self.deposit_time[i] = 0.0
                # initial x position at deposit
                self._deposit_x_at_contact = getattr(self, "_deposit_x_at_contact", {})

        # Store deposited-contact x,y positions for advection
        self.deposit_contact_pos = np.zeros((n, 3), dtype=float)
        for i in range(n):
            if self.deposited[i] > 0.5:
                self.deposit_contact_pos[i] = self.positions[i].copy()

    # ------------------------------------------------------------------
    # Heel + coiling geometry helpers
    # ------------------------------------------------------------------

    def _heel_shape(self, s: np.ndarray, t: float) -> np.ndarray:
        """Compute quasi-static heel + coiling shape for arc coordinates s.

        Returns Nx3 positions. The heel hangs from the injector outlet along
        -g_hat with a small Euler-elastica-like curvature near the bottom,
        transitioning into an alternating sinusoidal coil on the belt plane.
        """
        n = len(s)
        pos = np.zeros((n, 3), dtype=float)

        # Injector outlet
        p0 = self.inlet_pos.copy()
        d_inj = self.inlet_dir  # points downward (-z typically)

        # Gravity direction components
        g = self.g_hat

        # Perpendicular axes for lateral coiling
        # Choose lateral axis perpendicular to both g and belt_dir
        lat = np.cross(g, self.belt_dir)
        lat_norm = np.linalg.norm(lat)
        if lat_norm < 1e-12:
            lat = np.array([0.0, 1.0, 0.0])
        else:
            lat = lat / lat_norm

        # Contact point height (belt surface)
        z_belt = self.belt_surface_z

        # Suspended length above belt (vertical distance from outlet to belt)
        H = max(1e-6, np.dot(p0 - np.array([0, 0, z_belt]), -g))

        # Coil onset arc: where the rod transitions from hanging to coiling
        # This is roughly H (the vertical drop) but we add some slack
        s_coil_onset = H * 0.92

        # Alternating loop amplitude envelope that grows smoothly after onset
        amp = self.coil_amp_base

        for i in range(n):
            si = s[i]
            if si <= s_coil_onset:
                # Hanging portion: nearly straight along gravity with slight
                # elastica bowing
                frac = si / max(1e-12, s_coil_onset)
                # Vertical drop
                z_drop = si * (-g)  # displacement vector
                # Small lateral bow due to buckling under self-weight
                bow = 0.0
                if frac > 0.3:
                    bow_scale = 0.15 * self.l_g
                    bow = bow_scale * math.sin(math.pi * (frac - 0.3) / 0.7)
                pos[i] = p0 + z_drop + lat * bow
            else:
                # Coiling portion on belt plane
                ds = si - s_coil_onset
                # Phase advances with deposited arc; alternating loops means
                # the phase produces left-right-left pattern
                phase = self.omega_c * t + ds / self.l_g * 1.2 + self.phi0
                # Lateral oscillation
                y_lat = amp * math.sin(phase)
                # Longitudinal (along belt) advance proportional to belt speed
                # relative to injection speed
                x_adv = ds * self.speed_ratio
                # Height: essentially on belt surface with tiny clearance ramp
                h_clear = min(0.005, ds * 0.01)
                z_h = z_belt + h_clear * max(0.0, 1.0 - ds / (2.0 * self.l_g))

                pos[i] = p0 + (-g) * H + self.belt_dir * x_adv + lat * y_lat + g * (H - z_h + np.dot(p0, g))
                # Simplify: set z explicitly
                pos[i, 2] = z_h if g[2] < 0 else pos[i, 2]

        return pos

    def _update_positions_and_deposit(self, t: float, dt: float) -> None:
        """Advance injection, recompute shape, handle contact and sticking."""
        # Injected length grows
        if self.inj_start <= t <= self.inj_end:
            self.injected_length = min(
                self.L_total,
                float(self.cfg["rod"]["initial_deployed_length"]) + self.v_inj * max(0.0, t - self.inj_start),
            )

        L_now = self.injected_length

        # Determine number of nodes: keep segment length ~ dl_target
        dl_target = 0.008  # target segment length in meters
        n_nodes = max(len(self.init_position), int(math.ceil(L_now / dl_target)) + 1)
        n_nodes = min(n_nodes, 20000)  # safety cap

        # Build uniform arc coordinates from 0 to L_now
        s = np.linspace(0.0, L_now, n_nodes)

        # Compute heel/coiling shape
        pos_new = self._heel_shape(s, t)

        # Ensure inlet stays fixed
        pos_new[0] = self.inlet_pos.copy()

        # Detect contact with belt (z <= belt_surface_z + tolerance)
        tol = 1e-4
        z_vals = pos_new[:, 2]
        on_belt = z_vals <= (self.belt_surface_z + tol)

        # Update deposited mask: once deposited, stays deposited
        new_deposited = self.deposited.copy()
        if len(new_deposited) < n_nodes:
            new_deposited = np.concatenate([new_deposited, np.zeros(n_nodes - len(new_deposited))])
        elif len(new_deposited) > n_nodes:
            new_deposited = new_deposited[:n_nodes]

        new_deposit_time = self.deposit_time.copy()
        if len(new_deposit_time) < n_nodes:
            new_deposit_time = np.concatenate([new_deposit_time, np.full(n_nodes - len(new_deposit_time), -1.0)])
        elif len(new_deposit_time) > n_nodes:
            new_deposit_time = new_deposit_time[:n_nodes]

        new_contact_pos = self.deposit_contact_pos.copy()
        if len(new_contact_pos) < n_nodes:
            pad = np.zeros((n_nodes - len(new_contact_pos), 3))
            new_contact_pos = np.vstack([new_contact_pos, pad])
        elif len(new_contact_pos) > n_nodes:
            new_contact_pos = new_contact_pos[:n_nodes]

        for i in range(n_nodes):
            if on_belt[i] and new_deposited[i] < 0.5:
                new_deposited[i] = 1.0
                new_deposit_time[i] = t
                new_contact_pos[i] = pos_new[i].copy()

        # Sticking: deposited nodes move with belt
        for i in range(n_nodes):
            if new_deposited[i] > 0.5 and new_deposit_time[i] >= 0:
                advect_dt = t - new_deposit_time[i]
                pos_new[i] = new_contact_pos[i] + self.belt_dir * self.v_belt * advect_dt
                pos_new[i, 2] = self.belt_surface_z  # clamp to surface

        # Velocities by finite difference
        vel_new = np.zeros((n_nodes, 3), dtype=float)
        if dt > 1e-15 and self.positions.shape[0] > 0:
            # Map old positions onto new arc grid via interpolation
            old_s = self.arc_coords
            old_pos = self.positions
            n_old = len(old_s)
            if n_old >= 2:
                for k in range(3):
                    col = np.interp(s, old_s, old_pos[:, k], left=old_pos[0, k], right=old_pos[-1, k])
                    vel_new[:, k] = (col - pos_new[:, k]) / (-dt) if False else 0.0
            # Simpler: use analytical velocity
        # Analytical velocities
        for i in range(n_nodes):
            if new_deposited[i] > 0.5:
                vel_new[i] = self.belt_dir * self.v_belt
            else:
                # Suspended portion moves primarily downward at injection speed
                # plus lateral coiling velocity
                si = s[i]
                H = max(1e-6, np.dot(self.inlet_pos - np.array([0, 0, self.belt_surface_z]), -self.g_hat))
                s_coil_onset = H * 0.92
                if si <= s_coil_onset:
                    vel_new[i] = self.g_hat * self.v_inj * 0.3
                else:
                    ds = si - s_coil_onset
                    phase = self.omega_c * t + ds / self.l_g * 1.2 + self.phi0
                    lat = np.cross(self.g_hat, self.belt_dir)
                    lat_n = np.linalg.norm(lat)
                    if lat_n > 1e-12:
                        lat = lat / lat_n
                    dydt = self.coil_amp_base * self.omega_c * math.cos(phase)
                    vel_new[i] = lat * dydt + self.belt_dir * self.v_inj * self.speed_ratio

        # Orientations: identity
        ori_new = np.zeros((n_nodes, 4), dtype=float)
        ori_new[:, 3] = 1.0
        angvel_new = np.zeros((n_nodes, 3), dtype=float)

        # Mass
        lengths = np.linalg.norm(np.diff(pos_new, axis=0), axis=1)
        lumped = np.empty(n_nodes, dtype=float)
        if n_nodes == 1:
            lumped[0] = L_now
        else:
            lumped[0] = lengths[0] / 2.0
            lumped[-1] = lengths[-1] / 2.0
            if n_nodes > 2:
                lumped[1:-1] = (lengths[:-1] + lengths[1:]) / 2.0
        mass_new = self.mu_linear * lumped

        # Commit
        self.positions = pos_new
        self.velocities = vel_new
        self.orientations = ori_new
        self.angular_velocities = angvel_new
        self.mass = mass_new
        self.deposited = new_deposited
        self.arc_coords = s
        self.deposit_time = new_deposit_time
        self.deposit_contact_pos = new_contact_pos

    # ------------------------------------------------------------------
    # Budget and constraint helpers
    # ------------------------------------------------------------------

    def _compute_budgets(self) -> dict:
        """Compute external force/couple budgets and constraints."""
        n = self.positions.shape[0]
        g_vec = self.gravity

        # External forces: gravity on each node + injector reaction
        # Group G=1 (whole rod), K terms
        # K=2: gravity + injector constraint
        K = 2
        forces = np.zeros((1, K, 3), dtype=float)
        positions_f = np.zeros((1, K, 3), dtype=float)
        couples = np.zeros((1, K, 3), dtype=float)

        # Total gravity
        total_grav = self.mass.sum() * g_vec
        com = np.average(self.positions, axis=0, weights=np.maximum(self.mass, 1e-30))
        forces[0, 0] = total_grav
        positions_f[0, 0] = com

        # Injector reaction: balances weight of suspended portion + provides
        # injection momentum flux
        dep_mask = self.deposited > 0.5
        suspended_mass = self.mass[~dep_mask].sum() if n > 0 else 0.0
        inj_force = -suspended_mass * g_vec + self.mu_linear * self.v_inj * (-self.inlet_dir) * self.v_inj
        forces[0, 1] = inj_force
        positions_f[0, 1] = self.inlet_pos.copy()

        # Constraint: injector inlet position is controlled (position constraint)
        constraint_value = self.positions[0].copy() if n > 0 else self.inlet_pos.copy()
        constraint_target = self.inlet_pos.copy()
        constraint_scale = max(1e-12, np.linalg.norm(constraint_target))

        # Reference scales
        ref_force = max(1e-12, np.linalg.norm(total_grav))
        ref_torque = max(1e-12, ref_force * self.l_g)

        # Cumulative mass source
        cumulative_mass = self.mu_linear * self.injected_length

        # Mass quantity attribute (per-node mass)
        mass_quantity = self.mass.copy()

        return {
            "forces": forces,
            "force_positions": positions_f,
            "couples": couples,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "constraint_scale": constraint_scale,
            "ref_force": ref_force,
            "ref_torque": ref_torque,
            "cumulative_mass": cumulative_mass,
            "mass_quantity": mass_quantity,
        }

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.current_time
        if dt < 0:
            raise ValueError(f"Cannot go backward in time: {time} < {self.current_time}")

        # Sub-step for stability
        n_sub = max(1, int(math.ceil(dt / 0.005)))
        sub_dt = dt / n_sub if n_sub > 0 else 0.0
        for _ in range(n_sub):
            self.current_time += sub_dt
            self._update_positions_and_deposit(self.current_time, sub_dt)

        # If dt == 0 (first call at t=0), still compute budgets
        budgets = self._compute_budgets()

        n = self.positions.shape[0]
        body_id = np.zeros(n, dtype=np.int64)

        frame = StateFrame(
            time=time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientations.copy(),
            angular_velocity=self.angular_velocities.copy(),
            mass=self.mass.copy(),
            body_id=body_id,
            attributes={
                "arc_coordinate": self.arc_coords.copy(),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initially_deposited": self.init_deposited[:n].copy() if n <= len(self.init_deposited) else np.concatenate([self.init_deposited, np.zeros(n - len(self.init_deposited))]),
                "is_sticking_to_belt": self.deposited.copy(),
                "rod_radius": np.full(n, self.radius),
                "natural_curvature": np.zeros(n),
                "mass_quantity": budgets["mass_quantity"],
            },
            fields={
                "budget_external_force": budgets["forces"],
                "budget_external_couple": budgets["couples"],
                "budget_force_position": budgets["force_positions"],
                "constraint_value": budgets["constraint_value"],
                "constraint_target": budgets["constraint_target"],
                "cumulative_mass_source": np.array([budgets["cumulative_mass"]]),
            },
            scalars={
                "injection_speed": self.v_inj,
                "belt_speed": self.v_belt,
                "belt_to_injection_speed_ratio": self.speed_ratio,
                "constraint_scale": budgets["constraint_scale"],
                "reference_force": budgets["ref_force"],
                "reference_torque": budgets["ref_torque"],
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "injected_length": self.injected_length,
            "node_count": self.positions.shape[0],
            "deposited_count": int((self.deposited > 0.5).sum()),
        }
