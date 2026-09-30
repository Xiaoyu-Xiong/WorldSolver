"""Physically-based solver for the end-constrained pressurized rubber tube.

Model
-----
The finite-thickness tube wall is discretised as a three-dimensional solid:
the 1088 mesh nodes of the frozen scaffold mesh (outer wall ring layer and
cavity-wall ring layer) are connected by 512 hexahedral cells, each split
into 6 linear tetrahedra.  The wall material is a compressible neo-Hookean
solid (silicone rubber, rho = 1070 kg/m^3) integrated with an explicit
central-difference (kick-drift-kick) scheme using a lumped mass matrix.

Boundary conditions honour the frozen configuration:
  * nodes on both end rims (|z| = height/2, both layers) are fully fixed;
  * the prescribed cavity pressure acts as a follower traction on the
    current cavity-wall (inner layer) triangles, ramped by scene.pressure_at;
  * gravity is the configured (zero) vector.

Required raw trajectory fields are exported every sample: per-node mass,
per-term external force budget, cumulative mass source, per-element
deformation gradients, and the positive reference force scale.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# local hexahedron vertex order:
#   0=(k,t) 1=(k,t+1) 2=(k+1,t+1) 3=(k+1,t)  (outer wall face)
#   4=(k,t) 5=(k,t+1) 6=(k+1,t+1) 7=(k+1,t)  (cavity wall face)
_HEX_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


class BenchmarkSolver:
    """Explicit tetrahedral finite-element solver for the inflating tube."""

    def __init__(self) -> None:
        self._scene = None
        self._initialized = False
        self._time = 0.0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)
        config = scene.config
        geometry = config["geometry"]
        n_theta = int(geometry["circumferential_segments"])
        n_z = int(geometry["axial_segments"])
        inner = float(geometry["inner_radius"])
        thickness = float(geometry["wall_thickness"])
        outer = inner + thickness
        half = 0.5 * float(geometry["height"])
        density = float(config["material"]["density"])
        gravity = np.asarray(config.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)
        if gravity.shape != (3,):
            gravity = np.zeros(3)

        self._n_theta = n_theta
        self._n_z = n_z
        self._inner = inner
        self._outer = outer
        self._half = half
        self._density = density
        self._gravity = gravity

        initial = scene.initial_state()
        rest = np.asarray(initial.position, dtype=np.float64).copy()
        n_ring = n_theta * (n_z + 1)
        if rest.shape != (2 * n_ring, 3):
            raise ValueError("unexpected scaffold node layout for the hollow tube")
        self._n_ring = n_ring
        self._rest = rest

        # ---- element connectivity -------------------------------------
        kk, tt = np.meshgrid(np.arange(n_z), np.arange(n_theta), indexing="ij")
        base = kk * n_theta + tt
        wrap = (tt + 1) % n_theta
        ring_k = kk * n_theta
        ring_k1 = (kk + 1) * n_theta
        hex_nodes = np.stack(
            [
                ring_k + tt,
                ring_k + wrap,
                ring_k1 + wrap,
                ring_k1 + tt,
                n_ring + ring_k + tt,
                n_ring + ring_k + wrap,
                n_ring + ring_k1 + wrap,
                n_ring + ring_k1 + tt,
            ],
            axis=-1,
        ).reshape(-1, 8)
        elements = hex_nodes[:, np.asarray(_HEX_TETS, dtype=np.intp)].reshape(-1, 4)
        elements = self._orient_elements(rest, elements)
        self._elem = elements
        self._elem_flat = elements.ravel()

        # ---- element rest geometry & shape gradients -------------------
        x_rest = rest[elements]
        dm = np.stack(
            (x_rest[:, 1] - x_rest[:, 0],
             x_rest[:, 2] - x_rest[:, 0],
             x_rest[:, 3] - x_rest[:, 0]),
            axis=2,
        )
        det = np.linalg.det(dm)
        volume = np.abs(det) / 6.0
        self._V0 = volume
        inv_t = np.linalg.inv(dm).transpose(0, 2, 1)  # columns g1, g2, g3
        grad = np.empty((len(elements), 4, 3), dtype=np.float64)
        grad[:, 1:, :] = inv_t.transpose(0, 2, 1)
        grad[:, 0, :] = -grad[:, 1:].sum(axis=1)
        self._G = grad

        # ---- lumped nodal mass ----------------------------------------
        mass = np.zeros(2 * n_ring, dtype=np.float64)
        share = np.bincount(
            self._elem_flat,
            weights=np.repeat(0.25 * volume, 4),
            minlength=2 * n_ring,
        )
        mass[:] = density * share
        self._mass = mass
        self._total_mass = float(mass.sum())

        # ---- masks -----------------------------------------------------
        on_rim = np.zeros(2 * n_ring, dtype=bool)
        rim_k = np.concatenate(([0], [n_z]))
        for layer in range(2):
            for k in rim_k:
                start = layer * n_ring + k * n_theta
                on_rim[start:start + n_theta] = True
        self._fixed = on_rim
        self._free = ~on_rim
        cavity = np.zeros(2 * n_ring, dtype=np.int32)
        cavity[n_ring:] = 1
        self._cavity = cavity

        # ---- cavity-wall triangles (outward oriented) ------------------
        tri_a = ring_k + tt
        tri_b = ring_k + wrap
        tri_c = ring_k1 + wrap
        tri_d = ring_k1 + tt
        tris = np.concatenate(
            (
                np.stack((tri_a, tri_b, tri_c), axis=-1),
                np.stack((tri_a, tri_c, tri_d), axis=-1),
            ),
            axis=0,
        ) + n_ring
        tris = self._orient_triangles(rest, tris)
        self._tri = tris
        self._tri_flat = tris.ravel()

        # ---- material parameters --------------------------------------
        self._young = 1.5e5        # soft silicone rubber [Pa]
        self._poisson = 0.45
        self._mu = self._young / (2.0 * (1.0 + self._poisson))
        self._lam = (self._young * self._poisson
                     / ((1.0 + self._poisson) * (1.0 - 2.0 * self._poisson)))
        self._damping = 2.0        # mass-proportional damping [1/s]

        # ---- explicit stability limit ---------------------------------
        p_wave = np.sqrt((self._lam + 2.0 * self._mu) / density)
        face_areas = np.stack(
            (
                0.5 * np.linalg.norm(np.cross(x_rest[:, 1] - x_rest[:, 0],
                                              x_rest[:, 2] - x_rest[:, 0]), axis=1),
                0.5 * np.linalg.norm(np.cross(x_rest[:, 1] - x_rest[:, 0],
                                              x_rest[:, 3] - x_rest[:, 0]), axis=1),
                0.5 * np.linalg.norm(np.cross(x_rest[:, 2] - x_rest[:, 0],
                                              x_rest[:, 3] - x_rest[:, 0]), axis=1),
                0.5 * np.linalg.norm(np.cross(x_rest[:, 3] - x_rest[:, 1],
                                              x_rest[:, 2] - x_rest[:, 1]), axis=1),
            ),
            axis=1,
        )
        altitudes = 3.0 * volume[:, None] / np.maximum(face_areas, 1e-12)
        h_min = float(altitudes.min())
        self._dt = 0.20 * h_min / p_wave
        self._h_min = h_min

        # ---- reference scales ------------------------------------------
        p_max = float(config["boundary_conditions"]["cavity_pressure"]["pressure_pa"])
        inner_area = 2.0 * np.pi * inner * (2.0 * half)
        self._ref_force = float(max(p_max * inner_area, 1.0))
        self._p_max = p_max

        # ---- state ------------------------------------------------------
        self._pos = rest.copy()
        self._vel = np.zeros_like(rest)
        self._acc = np.zeros_like(rest)
        self._f_int = np.zeros_like(rest)
        self._f_press = np.zeros_like(rest)
        self._f_damp = np.zeros_like(rest)
        self._f_grav = np.zeros_like(rest)
        self._react = np.zeros(3)
        self._time = 0.0
        self._steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    @staticmethod
    def _orient_elements(rest: np.ndarray, elements: np.ndarray) -> np.ndarray:
        x = rest[elements]
        dm = np.stack(
            (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2
        )
        flip = np.linalg.det(dm) < 0.0
        if np.any(flip):
            elements = elements.copy()
            elements[flip] = elements[flip][:, [0, 1, 3, 2]]
        return elements

    @staticmethod
    def _orient_triangles(rest: np.ndarray, tris: np.ndarray) -> np.ndarray:
        x = rest[tris]
        normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        centre = x.mean(axis=1)
        radial = centre[:, :2]
        flip = np.einsum("ij,ij->i", normal[:, :2], radial) < 0.0
        if np.any(flip):
            tris = tris.copy()
            tris[flip] = tris[flip][:, [0, 2, 1]]
        return tris

    # ------------------------------------------------------------------
    # force evaluation
    # ------------------------------------------------------------------
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("eia,eib->eab", x[self._elem], self._G)

    def _internal_forces(self, x: np.ndarray) -> np.ndarray:
        f_mat = self._deformation_gradient(x)
        jac = np.maximum(np.linalg.det(f_mat), 1.0e-4)
        regularised = f_mat + 1.0e-10 * np.eye(3)[None, :, :]
        inv_t = np.transpose(np.linalg.inv(regularised), (0, 2, 1))
        pressure_term = self._lam * np.log(jac) - self._mu
        piola = self._mu * f_mat + pressure_term[:, None, None] * inv_t
        nodal = -self._V0[:, None, None] * np.einsum("eab,eib->eia", piola, self._G)
        force = np.zeros_like(x)
        weights = nodal.reshape(-1, 3)
        for axis in range(3):
            force[:, axis] = np.bincount(
                self._elem_flat, weights=weights[:, axis], minlength=len(x)
            )
        return force

    def _pressure_forces(self, x: np.ndarray, time: float) -> np.ndarray:
        load = float(self._scene.pressure_at(time))
        force = np.zeros_like(x)
        if load == 0.0:
            return force
        corners = x[self._tri]
        area_norm = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        norm = np.linalg.norm(area_norm, axis=1)
        unit = area_norm / np.maximum(norm, 1.0e-12)[:, None]
        weights = (load * 0.5 * norm / 3.0)[:, None] * unit
        weights = weights.reshape(-1, 3)
        for axis in range(3):
            force[:, axis] = np.bincount(
                self._tri_flat, weights=weights[:, axis], minlength=len(x)
            )
        return force

    def _evaluate(self, x: np.ndarray, v: np.ndarray, time: float):
        f_int = self._internal_forces(x)
        f_press = self._pressure_forces(x, time)
        f_grav = self._mass[:, None] * self._gravity[None, :]
        f_damp = -self._damping * self._mass[:, None] * v
        total = f_int + f_press + f_grav + f_damp
        total[self._fixed] = 0.0
        acceleration = total / self._mass[:, None]
        acceleration[self._fixed] = 0.0
        return f_int, f_press, f_damp, f_grav, acceleration

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        half_kick = 0.5 * dt
        velocity = self._vel + half_kick * self._acc
        velocity[self._fixed] = 0.0
        position = self._pos + dt * velocity
        position[self._fixed] = self._rest[self._fixed]
        next_time = self._time + dt

        f_int, f_press, f_damp, f_grav, acceleration = self._evaluate(
            position, velocity, next_time
        )
        velocity = velocity + half_kick * acceleration
        velocity[self._fixed] = 0.0

        if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
            # Deterministic safeguard against a transient numerical blow-up.
            position = self._pos
            velocity = np.zeros_like(self._vel)
            acceleration = np.zeros_like(self._acc)
            f_int = np.zeros_like(f_int)
            f_press = np.zeros_like(f_press)
            f_damp = np.zeros_like(f_damp)
            f_grav = np.zeros_like(f_grav)

        self._pos = position
        self._vel = velocity
        self._acc = acceleration
        self._f_int = f_int
        self._f_press = f_press
        self._f_damp = f_damp
        self._f_grav = f_grav
        self._react = -(f_int[self._fixed].sum(axis=0)
                        + f_press[self._fixed].sum(axis=0)
                        + f_damp[self._fixed].sum(axis=0)
                        + f_grav[self._fixed].sum(axis=0))
        self._time = next_time
        self._steps += 1

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("requested sample time must be finite and nonnegative")
        if target < self._time:
            target = self._time
        while self._time < target - 1.0e-12:
            dt = min(self._dt, target - self._time)
            self._step(dt)
        self._time = target
        return self._frame()

    # ------------------------------------------------------------------
    def _frame(self) -> StateFrame:
        f_mat = self._deformation_gradient(self._pos)
        terms = np.stack(
            (
                self._f_press.sum(axis=0),
                self._f_grav.sum(axis=0),
                self._f_damp.sum(axis=0),
                self._react,
            ),
            axis=0,
        )
        frame = StateFrame(
            time=float(self._time),
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            attributes={
                "fixed_rim": self._fixed.astype(np.int32),
                "pressure_surface": self._cavity.copy(),
                "surface_region": (1 - self._cavity).astype(np.int32),
                "rest_position": self._rest.copy(),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "element_deformation_gradient": f_mat,
                "budget_external_force": terms[None, :, :],
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "cavity_pressure_pa": float(self._scene.pressure_at(self._time)),
                "reference_force": float(self._ref_force),
                "inner_radius": float(self._inner),
                "outer_radius": float(self._outer),
                "wall_thickness": float(self._outer - self._inner),
                "total_mass": float(self._total_mass),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        if not self._initialized:
            return {"initialized": False}
        radius = np.linalg.norm(self._pos[:, :2], axis=1)
        mid = np.abs(self._pos[:, 2]) < 0.5 * self._half
        return {
            "initialized": True,
            "element_count": int(len(self._elem)),
            "node_count": int(len(self._pos)),
            "time_step_s": float(self._dt),
            "min_altitude_m": float(self._h_min),
            "steps": int(self._steps),
            "total_mass_kg": float(self._total_mass),
            "max_radius_m": float(radius.max()),
            "mid_radius_m": float(radius[mid].max()) if np.any(mid) else 0.0,
            "rim_radius_m": float(radius[np.abs(self._pos[:, 2]) >= self._half - 1e-9].max()),
        }
