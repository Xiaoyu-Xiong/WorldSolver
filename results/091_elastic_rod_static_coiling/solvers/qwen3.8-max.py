"""Quasi-static elastic rod coiling solver (discrete elastic rod, implicit Euler)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BenchmarkSolver:
    """Minimal discrete-elastic-rod solver for static coiling."""

    # ------------------------------------------------------------------ #
    #  Solver contract                                                    #
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg

        # Rod parameters -------------------------------------------------
        self.radius = float(cfg["rod"]["radius"])
        self.density = float(cfg["rod"]["density"])
        self.E = float(cfg["rod"]["young_modulus"])
        self.nu = float(cfg["rod"]["poisson_ratio"])
        self.total_length = float(cfg["rod"]["total_available_length"])
        self.initial_deployed = float(cfg["rod"]["initial_deployed_length"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        # Injector / substrate -------------------------------------------
        inj = cfg["injector"]
        self.injection_speed = float(inj["injection_speed"])
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.inject_axis = np.asarray(inj["axis"], dtype=float)
        self.inject_start = float(inj.get("start_time", 0.0))
        self.inject_end = float(inj.get("end_time", 180.0))
        self.substrate_z = float(cfg["substrate"]["surface_z"])

        ic = cfg.get("initial_conditions", {})
        self.sym_break = np.asarray(ic.get("symmetry_break_offset", [0.0, 0.0, 0.0]), dtype=float)

        # Cross-section properties
        area = np.pi * self.radius ** 2
        self.mass_per_length = self.density * area
        self.bending_stiffness = self.E * (np.pi * self.radius ** 4 / 4.0)
        G = self.E / (2.0 * (1.0 + self.nu))
        self.torsion_stiffness = G * (np.pi * self.radius ** 4 / 2.0)

        # Topology -------------------------------------------------------
        asset_path = Path(__file__).resolve().parent.parent / cfg["rod"]["topology_asset"]
        topo = json.loads(asset_path.read_text(encoding="utf-8"))
        self.topo_vertices = np.asarray(topo["vertices"], dtype=float)
        self.topo_edges = np.asarray(topo["edges"], dtype=int)
        self.topo_arc = np.asarray(topo["arc_coordinate"], dtype=float)

        # Build initial deployed state -----------------------------------
        self._init_rod()

        # Sample times bookkeeping
        self.sample_times_list = list(scene.sample_times())
        self.current_index = 0
        self.last_frame = None

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        # Determine how many samples to take since last call
        idx = self._time_index(time)
        if self.last_frame is not None and abs(time - self.last_frame.time) < 1e-12:
            return self._make_frame(time)

        dt_target = max(time - (self.last_frame.time if self.last_frame is not None else 0.0), 0.0)
        if dt_target <= 0.0:
            frame = self._make_frame(time)
            self.last_frame = frame
            return frame

        # Sub-step for stability
        max_dt = 5e-3
        n_sub = max(1, int(np.ceil(dt_target / max_dt)))
        dt = dt_target / n_sub
        for _ in range(n_sub):
            self._step(dt)

        frame = self._make_frame(time)
        self.last_frame = frame
        return frame

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict[str, object]:
        return {"solver": "quasi_static_elastic_rod_coiling"}

    # ================================================================== #
    #  Internal helpers                                                   #
    # ================================================================== #
    def _time_index(self, t: float) -> int:
        # find closest sample index >= current
        while self.current_index < len(self.sample_times_list) - 1 and self.sample_times_list[self.current_index] < t - 1e-12:
            self.current_index += 1
        return self.current_index

    # ------------------------------------------------------------------ #
    def _init_rod(self) -> None:
        """Create initial straight deployed rod from injector to substrate."""
        L0 = self.initial_deployed
        z_top = self.outlet[2]
        z_bot = self.substrate_z
        H = z_top - z_bot  # should be ~0.5 m

        # Use topology resolution
        n_nodes = len(self.topo_vertices)
        seg_len = H / max(n_nodes - 1, 1)

        x = np.zeros((n_nodes, 3), dtype=float)
        for i in range(n_nodes):
            frac = i / max(n_nodes - 1, 1)
            x[i] = self.outlet + self.inject_axis * (frac * H)

        # Apply small symmetry break at bottom node(s) to seed coiling direction
        x[-1] += self.sym_break
        if n_nodes > 2:
            x[-2] += 0.5 * self.sym_break

        # Velocity
        v = np.zeros_like(x)

        # Material frame twist (scalar per edge)
        self.twist = np.zeros(n_nodes - 1, dtype=float)

        # Deployed length so far
        self.deployed_length = L0
        self.injected_length = L0

        # Which nodes are on substrate (stuck)
        tol = 1e-9
        self.on_substrate = (x[:, 2] <= self.substrate_z + tol).astype(float)

        # Rest lengths per segment
        self.rest_lengths = np.full(n_nodes - 1, seg_len, dtype=float)

        # Positions & velocities
        self.x = x.copy()
        self.v = v.copy()

        # Store cumulative injected mass
        self.cumulative_mass_added = self.mass_per_length * self.injected_length

    # ------------------------------------------------------------------ #
    def _step(self, dt: float) -> None:
        """One quasi-static sub-step with semi-implicit Euler + projection."""
        n = len(self.x)
        if n < 2:
            return

        # Inject new material at top: move top node downward along axis by injection_speed*dt
        # but clamp outlet position; effectively we feed length into the suspended portion.
        inject_rate = 0.0
        current_time_est = (self.last_frame.time if self.last_frame is not None else 0.0) + dt
        if self.inject_start <= current_time_est <= self.inject_end:
            inject_rate = self.injection_speed

        dl = inject_rate * dt
        self.injected_length += dl
        self.cumulative_mass_added += self.mass_per_length * dl

        # Extend deployed length up to total available
        self.deployed_length = min(self.deployed_length + dl, self.total_length)

        # --- Forces ---
        f_ext = np.zeros_like(self.x)
        # Gravity on each node
        seg_lens = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        lumped = np.empty(n, dtype=float)
        lumped[0] = seg_lens[0] / 2.0 if len(seg_lens) > 0 else 0.0
        lumped[-1] = seg_lens[-1] / 2.0 if len(seg_lens) > 0 else 0.0
        if n > 2:
            lumped[1:-1] = (seg_lens[:-1] + seg_lens[1:]) / 2.0
        masses = self.mass_per_length * lumped

        f_ext += masses[:, None] * self.gravity[None, :]

        # Elastic bending forces (discrete Euler-Bernoulli via adjacent-segment angle)
        f_bend = self._bending_forces()
        f_ext += f_bend

        # Axial stretching forces (penalty to preserve rest length)
        f_stretch = self._stretching_forces()
        f_ext += f_stretch

        # --- Semi-implicit Euler integration ---
        # Damping for quasi-static behavior
        damping = 0.97
        self.v = damping * self.v + (f_ext / np.maximum(masses[:, None], 1e-12)) * dt
        self.x += self.v * dt

        # --- Constraints ---
        # Injector clamps first node
        self.x[0] = self.outlet.copy()
        self.v[0] = self.inject_axis * inject_rate

        # Substrate contact + sticking
        eps = 1e-12
        below = self.x[:, 2] < self.substrate_z
        self.x[below, 2] = self.substrate_z
        self.v[below, :] = 0.0
        self.on_substrate = np.where(below, 1.0, self.on_substrate)
        # Sticking: once on substrate, remain fixed
        stuck_mask = self.on_substrate > 0.5
        self.v[stuck_mask] = 0.0
        # Keep z pinned for stuck nodes
        self.x[stuck_mask, 2] = self.substrate_z

    # ------------------------------------------------------------------ #
    def _bending_forces(self) -> np.ndarray:
        """Discrete bending forces from Euler-Bernoulli energy E_b = 0.5*B*sum(kappa^2 * ds)."""
        n = len(self.x)
        f = np.zeros_like(self.x)
        if n < 3:
            return f
        B = self.bending_stiffness

        e = np.diff(self.x, axis=0)  # (n-1, 3)
        le = np.linalg.norm(e, axis=1, keepdims=True) + 1e-30
        t = e / le  # unit tangents
        le_flat = le[:, 0]

        for i in range(1, n - 1):
            t_prev = t[i - 1]
            t_next = t[i]
            ds = 0.5 * (le_flat[i - 1] + le_flat[i])
            # Discrete curvature vector (difference of tangents / ds)
            kappa_vec = (t_next - t_prev) / ds
            # Bending moment-like force: dE/dx_i ~ B * (kappa contributions)
            # Using finite-difference gradient of 0.5*B*kappa^2*ds
            # Approximate force on interior node from two adjacent segments
            fb = B * kappa_vec / ds
            f[i - 1] -= fb
            f[i] += 2.0 * fb
            f[i + 1] -= fb
        return f

    # ------------------------------------------------------------------ #
    def _stretching_forces(self) -> np.ndarray:
        """Linear penalty springs preserving rest segment lengths."""
        n = len(self.x)
        f = np.zeros_like(self.x)
        if n < 2:
            return f
        k_axial = self.E * np.pi * self.radius ** 2  # EA
        e = np.diff(self.x, axis=0)
        le = np.linalg.norm(e, axis=1, keepdims=True) + 1e-30
        stretch = (le - self.rest_lengths[:, None]) / self.rest_lengths[:, None]
        fs = k_axial * stretch * (e / le)
        f[:-1] -= fs
        f[1:] += fs
        return f

    # ------------------------------------------------------------------ #
    def _make_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        x = self.x.copy()
        v = self.v.copy()
        n = len(x)

        # Mass array
        seg_lens = np.linalg.norm(np.diff(x, axis=0), axis=1)
        lumped = np.empty(n, dtype=float)
        if n == 1:
            lumped[0] = self.deployed_length
        else:
            lumped[0] = seg_lens[0] / 2.0
            lumped[-1] = seg_lens[-1] / 2.0
            if n > 2:
                lumped[1:-1] = (seg_lens[:-1] + seg_lens[1:]) / 2.0
        mass = self.mass_per_length * lumped

        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0
        ang_vel = np.zeros((n, 3), dtype=float)

        # Arc coordinate attribute (material coordinate along rod)
        arc = np.zeros(n, dtype=float)
        if n > 1:
            ds = np.linalg.norm(np.diff(x, axis=0), axis=1)
            arc[1:] = np.cumsum(ds)

        # Attributes required by geometry bindings / scene
        attributes = {
            "arc_coordinate": arc,
            "is_injector_controlled": (np.arange(n) == 0).astype(float),
            "is_initial_contact": (np.arange(n) == n - 1).astype(float),
            "rod_radius": np.full(n, self.radius),
        }

        # External force budget (gravity + bending treated as external couples placeholder)
        # Shape [G, K, 3]; use G=1 (combined domain), K=1 term
        g_force = (mass[:, None] * self.gravity[None, :]).sum(axis=0, keepdims=True)  # (1,3)
        fields = {
            "budget_external_force": g_force.reshape(1, 1, 3),
            "budget_external_couple": np.zeros((1, 1, 3), dtype=float),
            "budget_force_position": np.mean(x, axis=0, keepdims=True).reshape(1, 1, 3),
            "constraint_target": np.zeros(1, dtype=float),
            "constraint_value": np.array([self.deployed_length], dtype=float),
            "cumulative_mass_source": np.array([self.cumulative_mass_added], dtype=float),
        }

        scalars = {
            "injection_speed": self.injection_speed,
            "natural_curvature": 0.0,
            "substrate_speed": 0.0,
            "constraint_scale": max(self.deployed_length, 1e-9),
            "reference_force": float(self.E * np.pi * self.radius ** 2),
            "reference_torque": float(self.bending_stiffness / max(self.radius, 1e-9)),
        }

        frame = StateFrame(
            time=time,
            position=x,
            velocity=v,
            orientation=orientation,
            angular_velocity=ang_vel,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
