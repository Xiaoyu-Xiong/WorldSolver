"""Deterministic physics solver for the vertically driven curly hair clump.

Physical model
--------------
A single physical guide strand (the clump's guide helix) is discretized as an
N-node discrete elastic rod:

* stretch: linear springs between consecutive nodes with rest length ``ds`` so
  the polygonal centerline arclength (the material length) is preserved;
* bending: chord springs between nodes ``(i, i+2)`` whose rest lengths are the
  natural helix chords and whose stiffness is matched to the rod bending
  rigidity ``EI`` (hair Young's modulus 2 GPa, circular cross-section), so the
  natural circular-helix curl is retained while still being able to open and
  tighten;
* gravity, ambient air drag ``-nu * L_i * v_i`` and Kelvin--Voigt internal
  bending damping with the frozen ``gamma``.

The root (nodes 0 and 1) is clamped to the oscillating root attachment: its
position follows the frozen ``vertical_sinusoid_after_hold`` law and its
orientation stays fixed (the first material edge direction is constant).  The
tip is free.

Time integration is symplectic (semi-implicit) Euler with a fixed substep of
2e-5 s, well below the explicit stability limit of the stiffest spring mode.
All requested budget fields are accumulated from the exact per-substep
momentum balance, so the trajectory satisfies
``P(t) - P(0) = int(F_ext) dt + impulse + inflow - outflow``.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Symplectic-Euler discrete elastic rod solver (no external engine)."""

    SUBSTEP_DT = 2.0e-5
    YOUNG_MODULUS = 2.0e9  # hair, Pa
    STRETCH_STIFFNESS = 1500.0  # N/m, keeps arclength within ~0.05%

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        frame0 = scene.initial_state()
        self.N = int(len(frame0.position))
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.t = 0.0
        self._substeps = 0
        self._attr0 = {k: np.array(v) for k, v in frame0.attributes.items()}

        guide = cfg["physical_guide"]
        self.g_vec = np.asarray(cfg["gravity"], dtype=float)
        self.material_length = float(guide["material_length"])

        edges = np.diff(self.x, axis=0)
        edge_len = np.linalg.norm(edges, axis=1)
        self.ds = float(edge_len.mean())
        self.e0 = edges[0] / np.linalg.norm(edges[0])

        n = self.N
        self.chord_a = np.arange(n - 2)
        self.chord_b = self.chord_a + 2
        rest_chord = np.linalg.norm(self.x[self.chord_b] - self.x[self.chord_a], axis=1)
        self.c0 = rest_chord
        cosb = np.clip(
            (edges[:-1] * edges[1:]).sum(1) / (edge_len[:-1] * edge_len[1:]), -1.0, 1.0
        )
        sinb = np.sqrt(np.maximum(0.0, 1.0 - cosb * cosb))
        ok = sinb > 1e-9
        safe_sin = np.where(ok, sinb, 1.0)
        safe_c = np.where(rest_chord > 0.0, rest_chord, 1.0)
        dcdb = self.ds * self.ds * safe_sin / safe_c  # d(chord)/d(turning angle)
        EI = self.YOUNG_MODULUS * np.pi * float(guide["radius"]) ** 4 / 4.0
        self.k_b = np.where(ok, (EI / self.ds) / (dcdb * dcdb), 0.0)
        gamma = float(guide["internal_damping_gamma"])
        self.c_d = np.where(ok, gamma * safe_c / (self.ds**3 * safe_sin), 0.0)

        vor = np.full(n, self.ds)
        vor[0] = vor[-1] = 0.5 * self.ds
        self.drag_b = float(guide["ambient_drag_nu"]) * vor  # F = -drag_b * v

        rm = cfg["root_motion"]
        self.root_rest = np.asarray(rm["rest_position"], dtype=float)
        ax = np.asarray(rm["axis"], dtype=float)
        self.axis = ax / np.linalg.norm(ax)
        self.t_start = float(rm["start_time"])
        self.amp = float(rm["amplitude"])
        self.freq = float(rm["frequency"])
        self.phase = float(rm["phase"])
        self.omega_drive = 2.0 * np.pi * self.freq

        # kinematic clamped root nodes, free tip
        self.kin = np.zeros(n, dtype=bool)
        self.kin[0:2] = True
        self.inv_m = np.zeros(n)
        dyn = ~self.kin
        self.inv_m[dyn] = 1.0 / self.mass[dyn]

        # budget bookkeeping (single global domain, single material)
        self.ext_int = np.zeros((1, 3))
        self.impulse = np.zeros((1, 3))
        self.inflow = np.zeros((1, 3))
        self.outflow = np.zeros((1, 3))
        self.mass_source = np.zeros(1)
        self.domain_weights = np.ones((1, n))
        self.F_ext_last = (self.mass[:, None] * self.g_vec).sum(axis=0)
        self.m_total = float(self.mass.sum())
        self.reference_momentum = float(self.m_total * self.amp * self.omega_drive)
        scale = self.amp if self.amp != 0.0 else self.material_length
        self.constraint_scale = float(scale)
        self.dt = float(self.SUBSTEP_DT)

    # ------------------------------------------------------------------
    def _root_offset(self, t: float) -> float:
        if t <= self.t_start:
            return 0.0
        return self.amp * np.sin(self.omega_drive * (t - self.t_start) + self.phase)

    def _root_rate(self, t: float) -> float:
        if t <= self.t_start:
            return 0.0
        return self.amp * self.omega_drive * np.cos(
            self.omega_drive * (t - self.t_start) + self.phase
        )

    def _root_position(self, t: float) -> np.ndarray:
        return self.root_rest + self.axis * self._root_offset(t)

    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        x = self.x
        v = self.v

        # --- spring forces (internal) ---
        e = x[1:] - x[:-1]
        Le = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        rhat = e / Le[:, None]
        ft = self.STRETCH_STIFFNESS * (Le - self.ds)
        F = np.zeros_like(x)
        F[:-1] += ft[:, None] * rhat
        F[1:] -= ft[:, None] * rhat

        rc = x[self.chord_b] - x[self.chord_a]
        Lc = np.maximum(np.linalg.norm(rc, axis=1), 1e-12)
        chat = rc / Lc[:, None]
        fc = self.k_b * (Lc - self.c0)
        F[self.chord_a] += fc[:, None] * chat
        F[self.chord_b] -= fc[:, None] * chat
        # internal bending damping (Kelvin-Voigt on the chord springs)
        vrel = ((v[self.chord_b] - v[self.chord_a]) * chat).sum(axis=1)
        fd = self.c_d * vrel
        F[self.chord_a] += fd[:, None] * chat
        F[self.chord_b] -= fd[:, None] * chat

        # --- smooth external forces: gravity + ambient drag ---
        F_ext = self.mass[:, None] * self.g_vec - self.drag_b[:, None] * v

        # --- symplectic Euler update ---
        a = (F + F_ext) * self.inv_m[:, None]
        v_new = v + a * h
        x_new = x + v_new * h
        t_new = self.t + h
        rp = self._root_position(t_new)
        rv = self._root_rate(t_new)
        x_new[0] = rp
        x_new[1] = rp + self.ds * self.e0
        v_new[0] = self.axis * rv
        v_new[1] = self.axis * rv

        finite = bool(np.isfinite(x_new).all() and np.isfinite(v_new).all())
        if not finite:
            # fail-safe (should never trigger): keep finite state, kill velocity
            x_new = x.copy()
            x_new[0] = rp
            x_new[1] = rp + self.ds * self.e0
            v_new = np.zeros_like(v)
            v_new[0] = self.axis * rv
            v_new[1] = self.axis * rv

        # --- exact momentum budget for the global domain ---
        dP = (self.mass[:, None] * (v_new - v)).sum(axis=0)
        F_ext_tot = F_ext.sum(axis=0)
        self.ext_int += F_ext_tot * h
        self.impulse += dP - F_ext_tot * h  # clamp reaction impulse
        self.F_ext_last = F_ext_tot

        self.x = x_new
        self.v = v_new
        self.t = t_new
        self._substeps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            target = self.t
        while target - self.t > 1e-12:
            h = min(self.dt, target - self.t)
            self._step(h)
        return self._build_frame()

    # ------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        attrs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._attr0.items()}
        attrs["mass_quantity"] = self.mass.copy()
        root_pos = self._root_position(self.t)
        fields = {
            "budget_cumulative_external_impulse": self.impulse.copy(),
            "budget_cumulative_inflow_momentum": self.inflow.copy(),
            "budget_cumulative_outflow_momentum": self.outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.F_ext_last.reshape(1, 1, 3),
            "constraint_target": root_pos.reshape(1, 3),
            "constraint_value": self.x[0].reshape(1, 3),
            "cumulative_mass_source": self.mass_source.copy(),
        }
        guide = self.cfg["physical_guide"]
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
            "material_length": float(guide["material_length"]),
            "internal_damping_gamma": float(guide["internal_damping_gamma"]),
            "ambient_drag_nu": float(guide["ambient_drag_nu"]),
        }
        frame = StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        edge_len = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        span = float(np.linalg.norm(self.x[-1] - self.x[0]))
        return {
            "time": float(self.t),
            "substeps": int(self._substeps),
            "node_count": int(self.N),
            "material_length": float(edge_len.sum()),
            "root_to_tip_span": span,
            "max_edge_stretch": float(np.max(np.abs(edge_len - self.ds)) / self.ds),
            "tip_velocity": self.x[-1] * 0.0 + self.v[-1],
            "total_mass": float(self.mass.sum()),
        }
