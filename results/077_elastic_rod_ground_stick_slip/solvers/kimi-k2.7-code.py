"""Elastic rod stick-slip solver.

A lightweight mass-spring / discrete-shell simulation of the surface mesh from
`assets/rod_surface.obj`.  The top ring is driven kinematically by the frozen
press-then-drag boundary condition.  The bottom ring contacts the horizontal
ground plane with a penalty normal force and Coulomb friction, producing the
requested stick-slip cycles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        """Set up the solver from the frozen scene."""
        self._seed = seed

        # ---- configuration -------------------------------------------------
        self._config = scene.config
        self._dt = 2.0e-5  # internal physics step (seconds)
        self._mu = float(
            self._config.get("contact", {}).get("ground_friction_coefficient", 0.35)
        )

        # ---- initial scene frame ------------------------------------------
        frame0 = scene.initial_state()
        self._time = 0.0
        self._x = frame0.position.copy().astype(float)
        self._v = frame0.velocity.copy().astype(float)
        self._m = frame0.mass.copy().astype(float)
        self._top_mask = frame0.attributes["driven_top"].astype(bool)
        self._n = self._x.shape[0]

        # ---- reference quantities -----------------------------------------
        self._x0 = self._x.copy()

        # ---- load mesh and build elements / springs -----------------------
        self._elements, self._elem_rest_areas = self._load_elements()
        self._springs = self._build_springs()

        # ---- material / model parameters ----------------------------------
        # Chosen to keep the rod light, flexible, and numerically stable with
        # the explicit sub-step above.
        self._k_edge = 5.0e4        # N/m  (membrane / axial springs)
        self._k_hoop = 2.0e5        # N/m  (cross-section shape springs)
        self._k_ground = 1.0e6      # N/m  (ground penalty stiffness)
        self._c_ground = 30.0      # ground normal damping (1/s)
        self._c_damp = 8.0          # global mass-proportional damping (1/s)
        self._v_tol = 1.0e-3        # m/s stick/slip speed tolerance

        # The mesh vertices are on the surface of the rod; offset by the rod
        # radius so the physical surface touches the ground plane z=0.
        self._rod_radius = float(frame0.scalars.get("rod_radius", 0.055))

        # Shift the reference shape so the rod surface rests on the ground.
        self._x0[:, 2] -= self._rod_radius
        self._x[:, 2] -= self._rod_radius
        self._v[:, 2] = 0.0

        # ---- budget bookkeeping -------------------------------------------
        self._budget_K = 4  # [gravity, contact_normal, contact_friction, driving]
        self._budget_G = 1
        self._cum_external_impulse = np.zeros((self._budget_G, 3), dtype=float)
        self._last_budget_external_force = np.zeros(
            (self._budget_G, self._budget_K, 3), dtype=float
        )
        self._cum_inflow = np.zeros((self._budget_G, 3), dtype=float)
        self._cum_outflow = np.zeros((self._budget_G, 3), dtype=float)
        self._cum_mass_source = np.zeros((1,), dtype=float)

        # ---- reference scales ---------------------------------------------
        self._ref_force = 1.0
        self._ref_momentum = 1.0
        self._ref_velocity = 1.0

    # ------------------------------------------------------------------ #
    # Scene / mesh helpers
    # ------------------------------------------------------------------ #
    def _load_elements(self) -> tuple[np.ndarray, np.ndarray]:
        """Parse the rod surface mesh and triangulate it into [Ne,3]."""
        asset = Path("assets/rod_surface.obj")
        if not asset.is_file():
            raise FileNotFoundError(f"Rod asset not found: {asset}")

        faces: list[list[int]] = []
        with asset.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "f":
                    # 1-based OBJ indices, ignore any texture/normal suffixes
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    faces.append(idx)

        # Triangulate n-gons by fanning from the first vertex.
        triangles: list[list[int]] = []
        for face in faces:
            if len(face) < 3:
                continue
            if len(face) == 3:
                triangles.append(face)
            else:
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

        elements = np.asarray(triangles, dtype=int)

        # Compute reference areas (used only as a diagnostic / for F weighting)
        x0 = self._x0
        i0, i1, i2 = elements[:, 0], elements[:, 1], elements[:, 2]
        e1 = x0[i1] - x0[i0]
        e2 = x0[i2] - x0[i0]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

        return elements, areas

    def _build_springs(self) -> list[tuple[int, int, float, float]]:
        """Return list of (i, j, rest_length, stiffness) springs."""
        x0 = self._x0
        springs: list[tuple[int, int, float, float]] = []
        added: set[tuple[int, int]] = set()

        def add(i: int, j: int, k: float) -> None:
            if i == j:
                return
            key = (min(i, j), max(i, j))
            if key in added:
                return
            added.add(key)
            r = float(np.linalg.norm(x0[i] - x0[j]))
            springs.append((i, j, r, k))

        # Edges from the triangulated surface mesh.
        for tri in self._elements:
            add(tri[0], tri[1], self._k_edge)
            add(tri[1], tri[2], self._k_edge)
            add(tri[2], tri[0], self._k_edge)

        # Hoop springs: opposite vertices on each ring, keeps cross-section round.
        sides = 6
        rings = self._n // sides
        for k in range(rings):
            for s in range(sides):
                a = k * sides + s
                b = k * sides + (s + 3) % sides
                add(a, b, self._k_hoop)

        return springs

    # ------------------------------------------------------------------ #
    # Prescribed boundary motion
    # ------------------------------------------------------------------ #
    def _driven_target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return target displacement and velocity for every driven vertex."""
        if t <= 0.8:
            s = t / 0.8
            disp = np.array([0.0, 0.0, -0.05 * s], dtype=float)
            vel = np.array([0.0, 0.0, -0.05 / 0.8], dtype=float)
        else:
            disp = np.array([0.22 * (t - 0.8), 0.0, -0.05], dtype=float)
            vel = np.array([0.22, 0.0, 0.0], dtype=float)

        target_pos = self._x0.copy()
        target_pos[self._top_mask] += disp
        target_vel = np.zeros_like(self._v)
        target_vel[self._top_mask] = vel
        return target_pos, target_vel

    # ------------------------------------------------------------------ #
    # Physics step
    # ------------------------------------------------------------------ #
    def _spring_forces(self) -> np.ndarray:
        f = np.zeros_like(self._x)
        x = self._x
        for i, j, r0, k in self._springs:
            dx = x[i] - x[j]
            L = np.linalg.norm(dx)
            if L < 1.0e-12:
                continue
            mag = k * (L - r0) / L
            fs = mag * dx
            f[i] -= fs
            f[j] += fs
        return f

    def _deformation_gradients(self) -> np.ndarray:
        """Compute per-element 3x3 deformation gradients."""
        x0 = self._x0
        x = self._x
        ne = len(self._elements)
        F = np.zeros((ne, 3, 3), dtype=float)

        i0, i1, i2 = self._elements[:, 0], self._elements[:, 1], self._elements[:, 2]
        E1 = x0[i1] - x0[i0]
        E2 = x0[i2] - x0[i0]
        N_ref = np.cross(E1, E2)
        N_ref /= np.linalg.norm(N_ref, axis=1, keepdims=True) + 1.0e-12

        e1 = x[i1] - x[i0]
        e2 = x[i2] - x[i0]
        n_cur = np.cross(e1, e2)
        n_cur /= np.linalg.norm(n_cur, axis=1, keepdims=True) + 1.0e-12

        # Build 3x3 matrices column-wise: M[:,:,d] is the d-th basis vector.
        D = np.stack([E1, E2, N_ref], axis=2)
        D_inv = np.linalg.inv(D)

        M = np.stack([e1, e2, n_cur], axis=2)
        F[:] = np.einsum("nik,nkj->nij", M, D_inv)
        return F

    def _step(self, dt: float) -> None:
        x = self._x
        v = self._v
        m = self._m

        # Spring and gravity forces (per-vertex)
        f_spring = self._spring_forces()
        f_grav = m[:, None] * np.array([0.0, 0.0, -9.81], dtype=float)

        # Global damping
        f_damp = -self._c_damp * m[:, None] * v

        # Predicted velocity without contact / friction / driving
        f_non = f_spring + f_grav + f_damp
        v_pred = v + dt * f_non / m[:, None]

        # ---- ground contact and friction --------------------------------
        # Ground plane at z = 0, normal pointing up.
        penetration = -x[:, 2]
        in_contact = penetration > 0.0

        f_contact_normal = np.zeros_like(f_non)
        f_contact_friction = np.zeros_like(f_non)

        if np.any(in_contact):
            # Penalty normal force with damping; always non-attractive.
            v_z = v_pred[:, 2]
            normal_mag = self._k_ground * penetration - self._c_ground * v_z
            normal_mag = np.clip(normal_mag, 0.0, None)
            f_contact_normal[in_contact, 2] = normal_mag[in_contact]

            # Tangential Coulomb friction on active contacts.
            # Relative velocity is just the rod vertex velocity (ground is static).
            v_tan = v_pred.copy()
            v_tan[:, 2] = 0.0
            v_tan_mag = np.linalg.norm(v_tan, axis=1) + 1.0e-12

            # Maximum friction force at each contact.
            f_max = self._mu * normal_mag

            # Static: try to cancel tangential velocity; dynamic: oppose motion.
            for i in np.where(in_contact)[0]:
                if v_tan_mag[i] < self._v_tol:
                    # Sticking: impulse needed to remove tangential velocity.
                    impulse = -m[i] * v_tan[i]
                    if np.linalg.norm(impulse) <= dt * f_max[i]:
                        f_contact_friction[i] = impulse / dt
                    else:
                        f_contact_friction[i] = -f_max[i] * v_tan[i] / v_tan_mag[i]
                else:
                    f_contact_friction[i] = -f_max[i] * v_tan[i] / v_tan_mag[i]

        # ---- total force and explicit update ------------------------------
        f_total = f_non + f_contact_normal + f_contact_friction
        v_new = v + dt * f_total / m[:, None]
        x_new = x + dt * v_new

        # ---- hard boundary condition on the driven top --------------------
        target_pos, target_vel = self._driven_target(self._time + dt)
        x_new[self._top_mask] = target_pos[self._top_mask]
        v_new[self._top_mask] = target_vel[self._top_mask]

        # ---- budget driving force (force needed to impose the BC) ---------
        f_drive = np.zeros_like(f_total)
        if np.any(self._top_mask):
            a_drive = (v_new[self._top_mask] - v[self._top_mask]) / dt
            f_drive[self._top_mask] = (
                m[self._top_mask, None] * a_drive - f_non[self._top_mask]
            )

        # ---- update bookkeeping -------------------------------------------
        self._x = x_new
        self._v = v_new
        self._time += dt

        # ---- external force budgets ---------------------------------------
        # term 0: gravity
        # term 1: contact normal
        # term 2: contact friction
        # term 3: driving constraint
        budget = np.zeros((self._budget_G, self._budget_K, 3), dtype=float)
        budget[0, 0] = f_grav.sum(axis=0)
        budget[0, 1] = f_contact_normal.sum(axis=0)
        budget[0, 2] = f_contact_friction.sum(axis=0)
        budget[0, 3] = f_drive.sum(axis=0)

        self._last_budget_external_force = budget
        self._cum_external_impulse += budget.sum(axis=1) * dt

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        while self._time < time - 0.5 * self._dt:
            self._step(self._dt)
        # Partial step to land exactly on the requested sample time.
        if self._time < time:
            self._step(time - self._time)

        # ---- friction contact data ---------------------------------------
        x = self._x
        v = self._v
        penetration = self._rod_radius - x[:, 2]
        active_idx = np.nonzero(penetration > 0.0)[0]
        C = len(active_idx)

        if C == 0:
            friction_active_mask = np.zeros((0,), dtype=float)
            friction_normal = np.zeros((0, 3), dtype=float)
            friction_normal_force = np.zeros((0,), dtype=float)
            friction_relative_velocity = np.zeros((0, 3), dtype=float)
            friction_static_coefficient = np.zeros((0,), dtype=float)
            friction_dynamic_coefficient = np.zeros((0, 3), dtype=float)
            friction_tangential_force = np.zeros((0, 3), dtype=float)
        else:
            friction_active_mask = np.ones((C,), dtype=float)
            friction_normal = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=float), (C, 1)
            )
            fn = self._k_ground * penetration[active_idx]
            friction_normal_force = np.clip(fn, 0.0, None)
            v_rel = v[active_idx].copy()
            v_rel[:, 2] = 0.0
            friction_relative_velocity = v_rel
            friction_static_coefficient = np.full((C,), self._mu, dtype=float)
            friction_dynamic_coefficient = np.full((C,), self._mu, dtype=float)
            v_tan_mag = np.linalg.norm(v_rel, axis=1) + 1.0e-12
            friction_tangential_force = np.zeros_like(v_rel)
            for k, i in enumerate(active_idx):
                if v_tan_mag[k] < self._v_tol:
                    f_elastic_tan = -self._v[i].copy()
                    f_elastic_tan[2] = 0.0
                    cap = self._mu * friction_normal_force[k]
                    mag = np.linalg.norm(f_elastic_tan)
                    if mag > 1.0e-12:
                        friction_tangential_force[k] = f_elastic_tan / mag * min(mag, cap)
                    else:
                        friction_tangential_force[k] = np.zeros(3)
                else:
                    friction_tangential_force[k] = (
                        -self._mu * friction_normal_force[k] * v_rel[k] / v_tan_mag[k]
                    )

        # ---- deformation gradient for shell triangles --------------------
        F_elem = self._deformation_gradients()

        # ---- assemble frame ------------------------------------------------
        domain_weights = np.ones((self._budget_G, self._n), dtype=float)

        frame = StateFrame(
            time=time,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            attributes={
                "mass_quantity": self._m.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self._cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self._cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self._cum_outflow.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": self._last_budget_external_force.copy(),
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "element_deformation_gradient": F_elem,
                "friction_active_mask": friction_active_mask,
                "friction_dynamic_coefficient": friction_dynamic_coefficient,
                "friction_normal": friction_normal,
                "friction_normal_force": friction_normal_force,
                "friction_relative_velocity": friction_relative_velocity,
                "friction_static_coefficient": friction_static_coefficient,
                "friction_tangential_force": friction_tangential_force,
            },
            scalars={
                "friction_stick_speed_tolerance": float(self._v_tol),
                "reference_force": float(self._ref_force),
                "reference_momentum": float(self._ref_momentum),
                "reference_velocity": float(self._ref_velocity),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self._dt,
            "num_vertices": self._n,
            "num_elements": len(self._elements),
            "num_springs": len(self._springs),
        }
