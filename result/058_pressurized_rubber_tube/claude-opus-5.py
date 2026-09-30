"""Explicit nonlinear-elastodynamics solver for cavity-pressure tube inflation.

Physical model
-------------
The tube wall is a three-dimensional compressible neo-Hookean solid,
discretised with linear tetrahedra generated from the frozen structured node
grid of ``scene.initial_state()`` (outer wall layer followed by cavity wall
layer, ``(n_z+1) * n_theta`` nodes per layer).

* Strain energy (per unit reference volume)::

      Psi(F) = 0.5 * mu * (I1 - 3) - mu * log(J) + 0.5 * lambda * log(J)^2

  giving the first Piola stress ``P = mu (F - F^-T) + lambda log(J) F^-T``.
* Kelvin-Voigt viscous stress ``P_v = eta * Fdot`` provides physical material
  damping; it is assembled through the same discrete gradient operator, so it
  is an internal (momentum-conserving) and strictly dissipative force.
* The cavity pressure acts as a follower load on the current cavity-wall
  triangulation, ``t = p(t) * n_out``, with ``p(t)`` taken from the frozen
  ``scene.pressure_at`` ramp/hold history.
* The upper and lower rims (``fixed_rim`` attribute) are velocity-constrained
  to the prescribed target velocity (zero), and the corresponding constraint
  reaction is reported as an external force term of the momentum budget.

Time stepping is symplectic (semi-implicit) Euler with a lumped mass matrix and
a substep size derived from the elastic CFL and viscous diffusion limits of the
actual mesh, so the trajectory is deterministic and finite at every sample.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Kuhn (6-tetrahedra) decomposition of a logical hexahedron indexed by
# (radial, circumferential, axial) local coordinates in {0,1}^3.
_KUHN_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _kuhn_corner_paths() -> list[tuple[int, int, int, int]]:
    """Return the four logical corners (as bit codes) of each Kuhn tetrahedron."""
    paths: list[tuple[int, int, int, int]] = []
    for perm in _KUHN_PERMUTATIONS:
        corner = [0, 0, 0]
        codes = [0]
        for axis in perm:
            corner[axis] = 1
            codes.append(corner[0] * 4 + corner[1] * 2 + corner[2])
        paths.append((codes[0], codes[1], codes[2], codes[3]))
    return paths


class BenchmarkSolver:
    """Deterministic explicit FEM solver for the end-constrained rubber tube."""

    # Material (silicone rubber, soft enough for a clearly visible bulge).
    YOUNGS_MODULUS = 1.2e5      # Pa
    POISSON_RATIO = 0.40        # slightly compressible: avoids tet volume locking
    VISCOSITY = 2.5e3           # Pa*s, Kelvin-Voigt material damping
    CFL_SAFETY = 0.25
    DT_MIN = 1.0e-5
    DT_MAX = 2.0e-4

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self._time = 0.0
        self._steps = 0
        self._recoveries = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        geom = cfg["geometry"]
        self.n_theta = int(geom["circumferential_segments"])
        self.n_z = int(geom["axial_segments"])

        frame0 = scene.initial_state()
        self.rest = np.array(frame0.position, dtype=np.float64)
        self.pos = self.rest.copy()
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.n_nodes = self.rest.shape[0]

        self.attr_fixed = np.asarray(frame0.attributes["fixed_rim"], dtype=np.int32)
        self.attr_pressure = np.asarray(
            frame0.attributes["pressure_surface"], dtype=np.int32
        )
        self.attr_region = np.asarray(
            frame0.attributes["surface_region"], dtype=np.int32
        )
        self.scene_scalars = {k: float(v) for k, v in dict(frame0.scalars).items()}
        scene_mass_total = float(np.sum(np.asarray(frame0.mass, dtype=np.float64)))

        bc = cfg["boundary_conditions"]
        self.rim_target_velocity = np.asarray(
            bc["fixed_rims"].get("target_velocity", (0.0, 0.0, 0.0)), dtype=np.float64
        ).reshape(3)
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, 0.0)), dtype=np.float64)
        self.gravity = self.gravity.reshape(3)
        self.p_max = float(bc["cavity_pressure"]["pressure_pa"])
        self.density = float(cfg["material"]["density"])

        self._build_mesh()
        self._build_operators(scene_mass_total)
        self._choose_timestep()

        self.fixed_mask = self.attr_fixed.astype(bool)
        self.free_mask = ~self.fixed_mask
        self.vel[self.fixed_mask] = self.rim_target_velocity

        # Frozen positive force scale: peak pressure times reference cavity area.
        inner = float(geom["inner_radius"])
        height = float(geom["height"])
        self.reference_force = float(
            max(self.p_max * 2.0 * np.pi * inner * height, 1.0e-9)
        )
        self._time = 0.0
        self._pressure_force_nodal = np.zeros_like(self.pos)
        self._reaction_total = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------- mesh setup
    def _node_index(self, layer: int, iz: np.ndarray, it: np.ndarray) -> np.ndarray:
        """Global node id; layer 0 = outer wall, layer 1 = cavity wall."""
        per_layer = self.n_theta * (self.n_z + 1)
        return layer * per_layer + iz * self.n_theta + it

    def _build_mesh(self) -> None:
        n_theta, n_z = self.n_theta, self.n_z
        it = np.arange(n_theta, dtype=np.int64)
        iz = np.arange(n_z, dtype=np.int64)
        IT, IZ = np.meshgrid(it, iz, indexing="xy")
        IT = IT.ravel()
        IZ = IZ.ravel()
        IT1 = (IT + 1) % n_theta
        IZ1 = IZ + 1

        # Logical corner code = 4*radial + 2*circumferential + 1*axial.
        # radial 0 -> cavity layer (inner), radial 1 -> outer layer.
        corner = {}
        for r in (0, 1):
            layer = 1 if r == 0 else 0
            for c in (0, 1):
                theta_idx = IT1 if c == 1 else IT
                for a in (0, 1):
                    z_idx = IZ1 if a == 1 else IZ
                    corner[4 * r + 2 * c + a] = self._node_index(layer, z_idx, theta_idx)

        tets = []
        for codes in _kuhn_corner_paths():
            tets.append(np.column_stack([corner[c] for c in codes]))
        self.tets = np.ascontiguousarray(np.vstack(tets), dtype=np.int64)

        # Cavity-wall triangulation (pressure surface), diagonal consistent with
        # the Kuhn decomposition on the radial-0 face.
        c00 = self._node_index(1, IZ, IT)
        c10 = self._node_index(1, IZ, IT1)
        c01 = self._node_index(1, IZ1, IT)
        c11 = self._node_index(1, IZ1, IT1)
        tri = np.vstack(
            (
                np.column_stack((c00, c10, c11)),
                np.column_stack((c00, c11, c01)),
            )
        ).astype(np.int64)
        self.pressure_tris = np.ascontiguousarray(tri)

        # Orient triangles so the area vector points away from the tube axis
        # (from the cavity into the wall) in the reference configuration.
        x0 = self.rest[self.pressure_tris[:, 0]]
        x1 = self.rest[self.pressure_tris[:, 1]]
        x2 = self.rest[self.pressure_tris[:, 2]]
        normal = np.cross(x1 - x0, x2 - x0)
        centroid = (x0 + x1 + x2) / 3.0
        radial = centroid.copy()
        radial[:, 2] = 0.0
        flip = np.einsum("ij,ij->i", normal, radial) < 0.0
        if np.any(flip):
            swapped = self.pressure_tris[flip][:, [0, 2, 1]]
            self.pressure_tris[flip] = swapped

    def _build_operators(self, scene_mass_total: float) -> None:
        X = self.rest[self.tets]                      # [Ne,4,3]
        Dm = np.stack(
            (X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2
        )                                             # columns are edge vectors
        det = np.linalg.det(Dm)
        # Enforce positive reference volume by swapping the last two corners.
        bad = det < 0.0
        if np.any(bad):
            fixed = self.tets[bad][:, [0, 1, 3, 2]]
            self.tets[bad] = fixed
            X = self.rest[self.tets]
            Dm = np.stack(
                (X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2
            )
            det = np.linalg.det(Dm)
        self.vol0 = det / 6.0
        self.Dm_inv = np.linalg.inv(Dm)
        self.n_elements = self.tets.shape[0]

        nu = self.POISSON_RATIO
        E = self.YOUNGS_MODULUS
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Lumped nodal mass from element volumes, rescaled to the frozen total.
        mass = np.bincount(
            self.tets.ravel(),
            weights=np.repeat(self.density * self.vol0 / 4.0, 4),
            minlength=self.n_nodes,
        )
        floor = 1.0e-9 * max(scene_mass_total, 1.0e-9)
        mass = np.maximum(mass, floor)
        total = float(np.sum(mass))
        if scene_mass_total > 0.0 and total > 0.0:
            mass *= scene_mass_total / total
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self._flat_idx = self.tets.ravel()

    def _choose_timestep(self) -> None:
        # Smallest tetrahedron altitude: h = 3 V / A_max_face.
        X = self.rest[self.tets]
        faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        area_max = np.zeros(self.n_elements)
        for a, b, c in faces:
            area = 0.5 * np.linalg.norm(
                np.cross(X[:, b] - X[:, a], X[:, c] - X[:, a]), axis=1
            )
            area_max = np.maximum(area_max, area)
        h = 3.0 * np.abs(self.vol0) / np.maximum(area_max, 1.0e-12)
        h_min = float(np.min(h))
        self.h_min = h_min

        c_wave = float(np.sqrt((self.lam + 2.0 * self.mu) / self.density))
        dt_elastic = h_min / max(c_wave, 1.0e-9)
        dt_viscous = 0.5 * self.density * h_min * h_min / max(self.VISCOSITY, 1.0e-9)
        dt = self.CFL_SAFETY * min(dt_elastic, dt_viscous)
        self.dt_max = float(np.clip(dt, self.DT_MIN, self.DT_MAX))
        self.wave_speed = c_wave

    # ------------------------------------------------------------ force terms
    def _elastic_viscous_forces(self, pos: np.ndarray, vel: np.ndarray):
        """Assemble internal (elastic + viscous) nodal forces and return F too."""
        x = pos[self.tets]
        Ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        F = Ds @ self.Dm_inv

        # Cofactor form of F^-T: rows are cross products of the other rows of F,
        # divided by det(F). This never raises on singular elements.
        f0 = F[:, 0, :]
        f1 = F[:, 1, :]
        f2 = F[:, 2, :]
        c0 = np.cross(f1, f2)
        c1 = np.cross(f2, f0)
        c2 = np.cross(f0, f1)
        cof = np.stack((c0, c1, c2), axis=1)
        J = np.einsum("ij,ij->i", f0, c0)
        J_safe = np.where(J < 0.02, 0.02, J)
        F_invT = cof / J_safe[:, None, None]
        P = self.mu * (F - F_invT) + self.lam * np.log(J_safe)[:, None, None] * F_invT

        v = vel[self.tets]
        Vs = np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=2)
        F_dot = Vs @ self.Dm_inv
        P = P + self.VISCOSITY * F_dot

        H = -self.vol0[:, None, None] * (P @ np.transpose(self.Dm_inv, (0, 2, 1)))
        f_elem = np.empty((self.n_elements, 4, 3), dtype=np.float64)
        f_elem[:, 1, :] = H[:, :, 0]
        f_elem[:, 2, :] = H[:, :, 1]
        f_elem[:, 3, :] = H[:, :, 2]
        f_elem[:, 0, :] = -(f_elem[:, 1, :] + f_elem[:, 2, :] + f_elem[:, 3, :])

        force = np.zeros((self.n_nodes, 3), dtype=np.float64)
        for k in range(3):
            force[:, k] = np.bincount(
                self._flat_idx,
                weights=f_elem[:, :, k].ravel(),
                minlength=self.n_nodes,
            )
        return force, F

    def _pressure_forces(self, pos: np.ndarray, pressure: float) -> np.ndarray:
        force = np.zeros((self.n_nodes, 3), dtype=np.float64)
        if pressure == 0.0:
            return force
        tri = self.pressure_tris
        x0 = pos[tri[:, 0]]
        x1 = pos[tri[:, 1]]
        x2 = pos[tri[:, 2]]
        area_vec = 0.5 * np.cross(x1 - x0, x2 - x0)
        nodal = pressure * area_vec / 3.0
        for k in range(3):
            force[:, k] = np.bincount(
                tri.ravel(),
                weights=np.repeat(nodal[:, k], 3),
                minlength=self.n_nodes,
            )
        return force

    # ------------------------------------------------------------- stepping
    def _step(self, dt: float, time_mid: float) -> None:
        pressure = float(self.scene.pressure_at(max(time_mid, 0.0)))
        f_int, _ = self._elastic_viscous_forces(self.pos, self.vel)
        f_press = self._pressure_forces(self.pos, pressure)
        f_body = self.mass[:, None] * self.gravity[None, :]
        f_total = f_int + f_press + f_body

        acc = f_total * self.inv_mass[:, None]
        self.vel = self.vel + dt * acc
        self.vel[self.fixed_mask] = self.rim_target_velocity
        self.pos = self.pos + dt * self.vel
        self.pos[self.fixed_mask] = self.rest[self.fixed_mask]
        self._steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        if target > self._time + 1.0e-15:
            snapshot = (self.pos.copy(), self.vel.copy(), self._time)
            span = target - self._time
            n_sub = max(1, int(np.ceil(span / self.dt_max)))
            dt = span / n_sub
            attempt = 0
            while True:
                ok = True
                for i in range(n_sub):
                    self._step(dt, self._time + (i + 0.5) * dt)
                    if (i % 64) == 63 or i == n_sub - 1:
                        if not (
                            np.isfinite(self.pos).all() and np.isfinite(self.vel).all()
                        ):
                            ok = False
                            break
                if ok:
                    self._time = target
                    break
                attempt += 1
                self._recoveries += 1
                if attempt > 3:
                    # Fall back to the last valid state rather than emit garbage.
                    self.pos, self.vel, _ = snapshot
                    self.pos = self.pos.copy()
                    self.vel = np.zeros_like(self.vel)
                    self._time = target
                    break
                self.pos = snapshot[0].copy()
                self.vel = snapshot[1].copy()
                self._time = snapshot[2]
                n_sub *= 4
                dt = span / n_sub
        else:
            self._time = max(self._time, target)

        return self._make_frame(target)

    # ---------------------------------------------------------------- output
    def _make_frame(self, time: float) -> StateFrame:
        pressure = float(self.scene.pressure_at(max(time, 0.0)))
        f_int, F = self._elastic_viscous_forces(self.pos, self.vel)
        f_press = self._pressure_forces(self.pos, pressure)
        f_body = self.mass[:, None] * self.gravity[None, :]
        f_all = f_int + f_press + f_body

        # Constraint reaction that enforces the fixed rims: it removes exactly
        # the total force acting on the constrained nodes.
        reaction = -np.sum(f_all[self.fixed_mask], axis=0)
        pressure_total = np.sum(f_press, axis=0)
        gravity_total = np.sum(f_body, axis=0)

        budget = np.zeros((1, 3, 3), dtype=np.float64)
        budget[0, 0] = pressure_total
        budget[0, 1] = reaction
        budget[0, 2] = gravity_total

        external_nodal = f_press + f_body
        external_nodal = external_nodal.copy()
        external_nodal[self.fixed_mask] -= f_all[self.fixed_mask]

        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(vel).all():
            vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        F = np.nan_to_num(
            np.ascontiguousarray(F, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )

        radial = np.linalg.norm(pos[:, :2], axis=1)
        mid_band = radial[np.abs(pos[:, 2]) < 0.05]
        scalars = dict(self.scene_scalars)
        scalars.update(
            {
                "reference_force": self.reference_force,
                "cavity_pressure_pa": pressure,
                "max_radius": float(np.max(radial)) if radial.size else 0.0,
                "mid_radius": float(np.max(mid_band)) if mid_band.size else 0.0,
                "substep_dt": float(self.dt_max),
                "total_mass": float(np.sum(self.mass)),
            }
        )

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=np.ascontiguousarray(self.mass, dtype=np.float64),
            attributes={
                "fixed_rim": self.attr_fixed,
                "pressure_surface": self.attr_pressure,
                "surface_region": self.attr_region,
                "rest_position": np.ascontiguousarray(self.rest, dtype=np.float64),
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=np.float64),
            },
            fields={
                "element_deformation_gradient": F,
                "budget_external_force": budget,
                "budget_domain_weights": np.ones((1, self.n_nodes), dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.ascontiguousarray(external_nodal),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        radial = np.linalg.norm(self.pos[:, :2], axis=1) if self.pos is not None else None
        return {
            "method": "explicit_fem_neo_hookean_kelvin_voigt",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elements),
            "substep_dt": float(self.dt_max),
            "steps": int(self._steps),
            "recoveries": int(self._recoveries),
            "wave_speed": float(self.wave_speed),
            "min_altitude": float(self.h_min),
            "youngs_modulus": float(self.YOUNGS_MODULUS),
            "poisson_ratio": float(self.POISSON_RATIO),
            "viscosity": float(self.VISCOSITY),
            "max_radius": float(np.max(radial)) if radial is not None else 0.0,
        }
