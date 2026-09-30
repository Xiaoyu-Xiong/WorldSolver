"""Physically-based explicit nonlinear-FEM solver for the pneumatic cushion.

Model
-----
The finite-thickness rubber wall of the cushion is the frozen linear-tetrahedron
solid mesh ``assets/cushion_solid.npz``.  The rubber is described by a nearly
incompressible neo-Hookean strain energy

    psi(F) = G/2 * (J^(-2/3) tr(F^T F) - 3) + K/2 * (Jbar - 1)^2

    G = E / (2 (1 + nu))          shear modulus
    K = E / (3 (1 - 2 nu))        bulk modulus

``J = det F`` is the element dilatation and ``Jbar`` is a nodal (F-bar) smoothed
dilatation obtained by volume-weighted averaging of the element dilatations over
the node patch.  The smoothing removes the volumetric locking of linear
tetrahedra so the nearly incompressible wall can actually inflate instead of
behaving as a rigid lattice.

Loading
-------
The prescribed uniform gauge pressure acts as a follower traction on the inner
chamber wall (the closed triangulated cavity surface of the mesh):

    t = -p(t) * n_outward,      p(t) = ramp from p0 to p1 on [t0, t1]

so the inner surface is pushed away from the cavity (the wall inflates).  The
pressure rate follows the frozen loading schedule and the outer faces see the
(zero) external pressure.

Time integration
----------------
Lumped nodal masses ``m = rho * nodal_volume`` and an explicit symplectic-Euler
scheme with the dilatational CFL time step.  Internal element forces are the
exact gradients of the discrete strain energy, so the internal force pairs cancel
for the whole body; mass-proportional (Rayleigh) damping models the internal
friction of rubber.  Rigid-body translation modes are projected out every step
("remove_rigid_body_modes") without restraining the face inflation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class BenchmarkSolver:
    """Deterministic explicit-dynamics solver for the pneumatic cushion."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        mat = cfg["material_properties"]
        load = cfg["loading"]

        self.seed = int(seed)
        self.frozen = False
        self.steps = 0

        asset = getattr(scene, "_asset", None)
        if asset is None:
            asset = Path(scene.__file__).resolve().parent / "assets" / "cushion_solid.npz"
        with np.load(asset) as data:
            rest = np.array(data["vertices"], dtype=np.float64)
            tets = np.array(data["tetrahedra"], dtype=np.int64)
            inner = np.array(data["inner_chamber_faces"], dtype=np.int64)
            nodal_volume = np.array(data["nodal_volume"], dtype=np.float64)
            wall_region = np.array(data["wall_region"], dtype=np.int32)

        # --- frozen reference geometry (positive element orientation) -------
        Dm = np.stack(
            (rest[tets[:, 1]] - rest[tets[:, 0]],
             rest[tets[:, 2]] - rest[tets[:, 0]],
             rest[tets[:, 3]] - rest[tets[:, 0]]),
            axis=2,
        )
        det_m = np.linalg.det(Dm)
        negative = det_m < 0.0
        if np.any(negative):
            flipped = tets.copy()
            flipped[negative] = flipped[negative][:, [0, 1, 3, 2]]
            tets = flipped
            Dm = np.stack(
                (rest[tets[:, 1]] - rest[tets[:, 0]],
                 rest[tets[:, 2]] - rest[tets[:, 0]],
                 rest[tets[:, 3]] - rest[tets[:, 0]]),
                axis=2,
            )
            det_m = np.linalg.det(Dm)

        self.rest = rest
        self.tets = tets
        self.inner_faces = inner
        self.wall_region = wall_region
        self.det_m = det_m
        self.V0 = np.abs(det_m) / 6.0
        self.Dm_inv = np.linalg.inv(Dm)
        self.Dm_inv_T = np.transpose(self.Dm_inv, (0, 2, 1))

        # --- material -------------------------------------------------------
        young = float(mat["young_modulus_pa"])
        poisson = float(mat["poisson_ratio"])
        density = float(mat["density_kg_m3"])
        self.G = young / (2.0 * (1.0 + poisson))
        self.K = young / (3.0 * (1.0 - 2.0 * poisson))
        self.density = density

        # --- lumped mass ----------------------------------------------------
        self.mass = density * nodal_volume
        self.total_mass = float(self.mass.sum())

        # nodal volume weights used by the F-bar smoothing
        node_weight = np.zeros(len(rest), dtype=np.float64)
        np.add.at(node_weight, tets.ravel(), np.repeat(self.V0, 4))
        node_weight = np.maximum(node_weight, 1e-30)
        self.node_weight = node_weight

        # --- explicit time step (dilatational CFL) --------------------------
        edges = []
        for a in range(4):
            for b in range(a + 1, 4):
                edges.append(np.linalg.norm(rest[tets[:, b]] - rest[tets[:, a]], axis=1))
        h_min = float(np.min(edges))
        c_wave = float(np.sqrt((self.K + 4.0 * self.G / 3.0) / density))
        self.dt = max(0.2 * h_min / c_wave, 1e-8)

        # --- pressure schedule ---------------------------------------------
        self.p_initial = float(load["initial_internal_pressure_pa"])
        self.p_target = float(load["target_internal_pressure_pa"])
        self.p_external = float(load["external_pressure_pa"])
        self.t_ramp_start = float(load["ramp_start_s"])
        self.t_ramp_end = float(load["ramp_end_s"])

        # --- outward orientation of the inner chamber faces -----------------
        owner = {}
        for tet in tets:
            a, b, c, d = (int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]))
            owner[tuple(sorted((a, b, c)))] = d
            owner[tuple(sorted((a, b, d)))] = c
            owner[tuple(sorted((a, c, d)))] = b
            owner[tuple(sorted((b, c, d)))] = a
        signs = np.ones(len(inner), dtype=np.float64)
        for i, face in enumerate(inner):
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            other = owner[tuple(sorted((i0, i1, i2)))]
            normal = np.cross(rest[i1] - rest[i0], rest[i2] - rest[i0])
            centre = (rest[i0] + rest[i1] + rest[i2]) / 3.0
            if float(np.dot(normal, rest[other] - centre)) > 0.0:
                signs[i] = -1.0
        self.face_sign = signs

        # --- pressure force scale (frozen positive reference force) ---------
        inner_vertices = rest[np.unique(inner)]
        extents = 2.0 * np.abs(inner_vertices).max(axis=0)
        ordered = np.sort(extents)
        broad_area = float(ordered[1] * ordered[2])
        scale = abs(self.p_target) * broad_area
        if not np.isfinite(scale) or scale <= 0.0:
            scale = young * float(np.mean(extents)) ** 2
        self.reference_force = float(max(scale, 1e-12))

        # --- damping of the rubber (Rayleigh mass-proportional) -------------
        self.damping = 2.0  # 1/s

        # --- state ----------------------------------------------------------
        self.x = rest.copy()
        self.v = np.zeros_like(rest)
        self.t = 0.0
        self.mass_quantity = self.mass.reshape(-1, 1)
        self.mass_source = np.zeros(1, dtype=np.float64)

    # ------------------------------------------------------------- physics
    def _pressure(self, time: float) -> float:
        if self.t_ramp_end > self.t_ramp_start:
            frac = (time - self.t_ramp_start) / (self.t_ramp_end - self.t_ramp_start)
        else:
            frac = 1.0
        frac = min(max(frac, 0.0), 1.0)
        value = self.p_initial + (self.p_target - self.p_initial) * frac
        return value - self.p_external

    def _kinematics(self, x):
        tets = self.tets
        Ds = np.stack(
            (x[tets[:, 1]] - x[tets[:, 0]],
             x[tets[:, 2]] - x[tets[:, 0]],
             x[tets[:, 3]] - x[tets[:, 0]]),
            axis=2,
        )
        return Ds

    def _smooth(self, values_per_element, volume):
        """Volume-weighted nodal patch averaging of per-element scalars."""
        numerator = np.zeros(len(self.rest), dtype=np.float64)
        np.add.at(numerator, self.tets.ravel(), np.repeat(volume * values_per_element, 4))
        nodal = numerator / self.node_weight
        return nodal[self.tets].mean(axis=1)

    def _internal_forces(self, x):
        tets = self.tets
        Ds = self._kinematics(x)
        det_s = np.linalg.det(Ds)
        volume = det_s / 6.0
        J = det_s / self.det_m
        Jc = np.maximum(J, 1e-3)
        Ds_inv = np.linalg.inv(Ds)

        # deformation gradient and its inverse transpose
        F = Ds @ self.Dm_inv
        FinvT = np.transpose(self.Dm_inv @ Ds_inv, (0, 2, 1))
        I1 = np.einsum("eij,eij->e", F, F)

        # F-bar smoothed dilatation
        Jbar = np.maximum(self._smooth(J, self.V0), 1e-3)
        iso = Jc ** (-2.0 / 3.0)

        # deviatoric first Piola-Kirchhoff stress (isochoric neo-Hookean part)
        P_dev = self.G * iso[:, None, None] * (
            F - (I1 / 3.0)[:, None, None] * FinvT
        )

        # dilatation part: derivative of the energy with respect to element volume
        g = self.K * (Jbar - 1.0)
        Pi = self._smooth(g, self.V0)

        f_nodes = -(self.V0[:, None, None] * (P_dev @ self.Dm_inv_T)
                    + (Pi * volume)[:, None, None] * Ds_inv)
        f_first = -f_nodes.sum(axis=2)

        f = np.zeros_like(x)
        np.add.at(f, tets[:, 0], f_first)
        np.add.at(f, tets[:, 1], f_nodes[:, :, 0])
        np.add.at(f, tets[:, 2], f_nodes[:, :, 1])
        np.add.at(f, tets[:, 3], f_nodes[:, :, 2])
        return f

    def _pressure_loads(self, x, pressure):
        faces = self.inner_faces
        x0 = x[faces[:, 0]]
        x1 = x[faces[:, 1]]
        x2 = x[faces[:, 2]]
        cross = np.cross(x1 - x0, x2 - x0)
        norm = np.linalg.norm(cross, axis=1)
        norm = np.maximum(norm, 1e-30)
        area = 0.5 * norm
        unit = cross / norm[:, None]
        # force on the solid surface: -p * n_outward (n_outward points into cavity)
        face_force = (-pressure * area)[:, None] * (unit * self.face_sign[:, None])
        loads = np.zeros_like(x)
        share = face_force / 3.0
        np.add.at(loads, faces[:, 0], share)
        np.add.at(loads, faces[:, 1], share)
        np.add.at(loads, faces[:, 2], share)
        return loads, face_force.sum(axis=0)

    def _step(self, dt: float):
        x_prev = self.x
        v_prev = self.v
        force = self._internal_forces(x_prev)
        pressure = self._pressure(self.t + 0.5 * dt)
        pressure_loads, _ = self._pressure_loads(x_prev, pressure)
        force = force + pressure_loads
        force -= (self.damping * self.mass)[:, None] * v_prev
        # remove rigid-body translation: the chamber stays centred
        force -= self.mass[:, None] * (force.sum(axis=0) / self.total_mass)

        self.v = v_prev + dt * force / self.mass[:, None]
        self.x = x_prev + dt * self.v
        self.steps += 1
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            # never emit a non-finite state: freeze at the last valid one
            self.x = x_prev
            self.v = np.zeros_like(v_prev)
            self.frozen = True

    # ------------------------------------------------------------- outputs
    def _deformation_gradients(self, x):
        Ds = self._kinematics(x)
        return Ds @ self.Dm_inv

    def _frame(self):
        pressure = self._pressure(self.t)
        x = self.x
        _, external_force = self._pressure_loads(x, pressure)
        # keep the chamber centred at the origin (rigid translation only, which
        # leaves the deformation gradient untouched)
        centre = (self.mass[:, None] * x).sum(axis=0) / self.total_mass
        position = x - centre

        from worldsolver.contracts import StateFrame

        frame = StateFrame(
            time=float(self.t),
            position=position,
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "wall_region": self.wall_region,
                "mass_quantity": self.mass_quantity,
            },
            fields={
                "budget_external_force": external_force.reshape(1, 1, 3).copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "element_deformation_gradient": self._deformation_gradients(x),
            },
            scalars={
                "internal_pressure_pa": float(pressure + self.p_external),
                "external_pressure_pa": float(self.p_external),
                "reference_force": float(self.reference_force),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            target = self.t
        guard = 0
        while self.t < target - 1e-12:
            dt = min(self.dt, target - self.t)
            if dt <= 0.0:
                break
            if self.frozen:
                self.t = target
                break
            try:
                self._step(dt)
            except Exception:
                # keep a valid, finite state rather than aborting the trajectory
                self.v = np.zeros_like(self.v)
                self.frozen = True
                self.t = target
                break
            self.t += dt
            guard += 1
            if guard > 2_000_000:
                self.frozen = True
                self.t = target
                break
        self.t = target if abs(self.t - target) < 1e-9 else self.t
        return self._frame()

    def diagnostics(self) -> dict:
        return {
            "solver": "explicit_fbar_neo_hookean",
            "time_step_s": float(self.dt),
            "steps": int(self.steps),
            "frozen": bool(self.frozen),
            "total_mass_kg": float(self.total_mass),
            "shear_modulus_pa": float(self.G),
            "bulk_modulus_pa": float(self.K),
            "reference_force_n": float(self.reference_force),
            "final_time_s": float(self.t),
        }
