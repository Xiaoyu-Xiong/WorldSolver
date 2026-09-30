"""Self-contained finite-element solver for the compressed rubber cube."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Lumped-mass tetrahedra with nonlinear neo-Hookean elasticity."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.rest = np.asarray(scene.vertices, dtype=float).copy()
        self.tets = np.asarray(scene.tetrahedra, dtype=np.int64).copy()
        self.surface_faces = np.asarray(scene.faces, dtype=np.int64)
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)
        self.time = 0.0

        X = self.rest[self.tets]
        Dm = np.stack((X[:, 1] - X[:, 0], X[:, 2] - X[:, 0],
                       X[:, 3] - X[:, 0]), axis=2)
        self.volume = np.abs(np.linalg.det(Dm)) / 6.0
        if np.any(self.volume <= 1.0e-14):
            raise ValueError("rubber mesh contains a degenerate tetrahedron")
        self.inv_dm = np.linalg.inv(Dm)

        # Soft, nearly incompressible rubber.  nu below 0.5 avoids volumetric
        # locking while still producing the characteristic lateral bulge.
        density, young, poisson = 1000.0, 2.0e5, 0.45
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.mass = np.zeros(len(self.rest), dtype=float)
        tet_mass_quarter = density * self.volume / 4.0
        for local in range(4):
            np.add.at(self.mass, self.tets[:, local], tet_mass_quarter)
        if np.any(self.mass <= 0.0):
            raise ValueError("rubber mesh contains an unreferenced vertex")

        z_min, z_max = np.min(self.rest[:, 2]), np.max(self.rest[:, 2])
        self.fixed = np.isclose(self.rest[:, 2], z_min, atol=1.0e-10)
        self.top = np.isclose(self.rest[:, 2], z_max, atol=1.0e-10)

        # Consistent nodal weights for uniform traction on the reference face.
        self.top_area_weights = np.zeros(len(self.rest), dtype=float)
        for tri in self.surface_faces:
            if np.all(self.top[tri]):
                a, b, c = self.rest[tri]
                area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
                self.top_area_weights[tri] += area / 3.0
        self.loaded_area = float(np.sum(self.top_area_weights))
        if self.loaded_area <= 0.0:
            raise ValueError("mesh has no top surface for the traction")

        bc = scene.config["boundary_conditions"]["top_compressive_traction"]
        self.peak_traction = float(bc["peak_magnitude_pa"])
        self.traction_direction = np.asarray(bc["direction"], dtype=float)
        self.traction_direction /= np.linalg.norm(self.traction_direction)
        schedule = bc["schedule"]
        self.schedule_t = np.asarray([p["time"] for p in schedule], dtype=float)
        self.schedule_s = np.asarray([p["scale"] for p in schedule], dtype=float)

        self.max_dt = 2.0e-4
        self.damping_rate = 7.0
        self.step_count = 0
        self.last_reaction = np.zeros(3, dtype=float)

    def _traction_scale(self, time: float) -> float:
        return float(np.interp(time, self.schedule_t, self.schedule_s,
                               left=self.schedule_s[0], right=self.schedule_s[-1]))

    def _deformation_gradient(self) -> np.ndarray:
        x = self.x[self.tets]
        ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0],
                       x[:, 3] - x[:, 0]), axis=2)
        return np.einsum("nij,njk->nik", ds, self.inv_dm)

    @staticmethod
    def _cofactor(F: np.ndarray) -> np.ndarray:
        cof = np.empty_like(F)
        cof[:, :, 0] = np.cross(F[:, :, 1], F[:, :, 2])
        cof[:, :, 1] = np.cross(F[:, :, 2], F[:, :, 0])
        cof[:, :, 2] = np.cross(F[:, :, 0], F[:, :, 1])
        return cof

    def _forces(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = self._deformation_gradient()
        cof = self._cofactor(F)
        J = np.einsum("ni,ni->n", F[:, :, 0], cof[:, :, 0])
        # Barrier continuation: ordinary neo-Hookean stress for physical J,
        # finite and strongly restoring if an explicit step distorts an element.
        J_safe = np.maximum(J, 0.12)
        finv_t = cof / J_safe[:, None, None]
        P = (self.mu * (F - finv_t)
             + self.lam * np.log(J_safe)[:, None, None] * finv_t)
        H = -self.volume[:, None, None] * np.einsum(
            "nij,nkj->nik", P, self.inv_dm)

        internal = np.zeros_like(self.x)
        np.add.at(internal, self.tets[:, 1], H[:, :, 0])
        np.add.at(internal, self.tets[:, 2], H[:, :, 1])
        np.add.at(internal, self.tets[:, 3], H[:, :, 2])
        np.add.at(internal, self.tets[:, 0], -np.sum(H, axis=2))

        traction = self.peak_traction * self._traction_scale(time)
        external = (self.top_area_weights[:, None] * traction
                    * self.traction_direction)
        return internal, external, F

    def _step(self, dt: float, end_time: float) -> None:
        internal, external, _ = self._forces(end_time - 0.5 * dt)
        total = internal + external
        self.last_reaction = -np.sum(total[self.fixed], axis=0)
        free = ~self.fixed
        self.v[free] += dt * total[free] / self.mass[free, None]
        self.v[free] *= np.exp(-self.damping_rate * dt)
        self.x[free] += dt * self.v[free]
        self.x[self.fixed] = self.rest[self.fixed]
        self.v[self.fixed] = 0.0
        self.step_count += 1

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-14:
            dt = min(self.max_dt, target - self.time)
            self._step(dt, self.time + dt)
            self.time += dt
        self.time = target

        internal, external, F = self._forces(target)
        reaction = -np.sum((internal + external)[self.fixed], axis=0)
        net_boundary_force = np.sum(external, axis=0) + reaction
        self.last_reaction = reaction
        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest.copy(),
                "bottom_face": self.fixed.astype(float),
                "top_face": self.top.astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": net_boundary_force.reshape(1, 1, 3),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": F.copy(),
            },
            scalars={
                "applied_traction_pa": self.peak_traction * self._traction_scale(target),
                "reference_force": self.peak_traction * self.loaded_area,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit_lumped_tet_neo_hookean",
            "steps": int(self.step_count),
            "total_mass_kg": float(np.sum(self.mass)),
            "loaded_area_m2": float(self.loaded_area),
            "support_reaction_N": self.last_reaction.tolist(),
        }
