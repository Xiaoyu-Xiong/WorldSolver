"""Finite-element inflation solver for the pressurised rubber tube benchmark.

The tube wall is modelled as a Saint-Venant Kirchhoff elastic solid
discretised with linear tetrahedra.  Internal pressure is ramped on the
cavity surface while the end rims are held fixed.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit-dynamics FEM solver for a pressurised hollow cylinder."""

    def __init__(self) -> None:
        self._initialized = False
        self.scene: object = None
        self.seed = 0

    def initialize(self, scene: object, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        frame0 = scene.initial_state()
        self.x0 = np.asarray(frame0.position, dtype=np.float64).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.m = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.n = self.x.shape[0]

        self.fixed = np.asarray(frame0.attributes["fixed_rim"], dtype=bool)
        self.pressure_surface = np.asarray(
            frame0.attributes["pressure_surface"], dtype=bool
        )

        # Infer grid resolution from the reference geometry.
        n_total = self.n
        if n_total % 2 != 0:
            raise ValueError("Expected an even number of shell nodes")
        surface_size = n_total // 2
        z_vals = np.unique(np.round(self.x0[:, 2], 6))
        n_z_layers = z_vals.size
        n_z = n_z_layers - 1
        if surface_size % n_z_layers != 0:
            raise ValueError("Could not infer circumferential resolution")
        n_theta = surface_size // n_z_layers
        self.n_theta = int(n_theta)
        self.n_z = int(n_z)
        self.surface_size = surface_size

        # Build a tetrahedral mesh of the wall by splitting each hexahedral
        # annular brick into two triangular prisms and each prism into
        # three tetrahedra.
        self.elements = self._build_tet_mesh()
        self.n_elements = self.elements.shape[0]

        # Reference shape matrices and volumes.
        self._init_element_data()

        # Material: silicone-like rubber (soft, nearly incompressible).
        self.density = 1070.0
        self.young = 5.0e5
        self.poisson = 0.45
        self.mu = self.young / (2.0 * (1.0 + self.poisson))
        self.lam = (
            self.young
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )

        # Stable explicit time step (longitudinal wave speed).
        h_min = self._min_edge_length()
        c = np.sqrt((self.lam + 2.0 * self.mu) / self.density)
        self.dt = min(0.2 * h_min / c, 1.0e-4)
        if self.dt <= 0.0:
            raise ValueError("Computed non-positive time step")

        # Viscous damping to reach a steady inflated state quickly.
        self.damping = 40.0

        # Cavity pressure normal/area for each pressure-surface node.
        self.pressure_normal = self._build_pressure_normals()

        # Reference state attributes for trajectory export.
        self._attrs = {
            "fixed_rim": np.asarray(frame0.attributes["fixed_rim"], dtype=np.int32),
            "pressure_surface": np.asarray(
                frame0.attributes["pressure_surface"], dtype=np.int32
            ),
            "surface_region": np.asarray(
                frame0.attributes["surface_region"], dtype=np.int32
            ),
            "rest_position": self.x0.copy(),
            "mass_quantity": self.m[:, None].copy(),
        }
        self._scalars = dict(frame0.scalars)
        self._scalars.setdefault("reference_force", 1.0)

        self.t = 0.0
        self._initialized = True

    def _node_index(self, layer: int, i: int, k: int) -> int:
        """Return node index for a grid layer, axial ring and angular sample.

        layer: 0 = outer, 1 = inner.
        """
        return layer * self.surface_size + i * self.n_theta + (k % self.n_theta)

    def _build_tet_mesh(self) -> np.ndarray:
        tets = []
        n_theta = self.n_theta
        for i in range(self.n_z):
            for k in range(n_theta):
                kp1 = (k + 1) % n_theta
                # Outer/inner node aliases.
                a0 = self._node_index(0, i, k)
                a1 = self._node_index(0, i, kp1)
                d0 = self._node_index(0, i, k)
                d1 = self._node_index(0, i, kp1)
                e0 = self._node_index(0, i + 1, k)
                e1 = self._node_index(0, i + 1, kp1)

                b0 = self._node_index(1, i, k)
                b1 = self._node_index(1, i, kp1)
                c0 = self._node_index(1, i + 1, k)
                c1 = self._node_index(1, i + 1, kp1)
                f0 = self._node_index(1, i + 1, k)
                f1 = self._node_index(1, i + 1, kp1)

                # Wedge A: (outer_i, inner_i, inner_{i+1}) x theta.
                tets.append([a0, b0, c0, a1])
                tets.append([b0, c0, a1, b1])
                tets.append([c0, a1, b1, c1])

                # Wedge B: (outer_i, outer_{i+1}, inner_{i+1}) x theta.
                tets.append([d0, e0, f0, d1])
                tets.append([e0, f0, d1, e1])
                tets.append([f0, d1, e1, f1])

        tets = np.asarray(tets, dtype=np.int32)

        # Ensure positive rest volumes by swapping node pairs if necessary.
        X = self.x0
        i0 = tets[:, 0]
        i1 = tets[:, 1]
        i2 = tets[:, 2]
        i3 = tets[:, 3]
        Dm = np.stack(
            [X[i1] - X[i0], X[i2] - X[i0], X[i3] - X[i0]], axis=2
        )  # (Ne, 3, 3)
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip, [0, 1]] = tets[flip, [1, 0]]

        return tets

    def _init_element_data(self) -> None:
        X = self.x0
        elems = self.elements
        i0, i1, i2, i3 = elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3]
        self.Dm = np.stack(
            [X[i1] - X[i0], X[i2] - X[i0], X[i3] - X[i0]], axis=2
        )
        self.Dm_inv = np.linalg.inv(self.Dm)
        self.rest_volume = np.abs(np.linalg.det(self.Dm)) / 6.0

        # Precompute Dm^{-T} used in nodal force assembly.
        self.Dm_inv_T = np.transpose(self.Dm_inv, (0, 2, 1))

    def _build_pressure_normals(self) -> np.ndarray:
        """Return area-weighted outward normals for the inner cavity surface."""
        normal = np.zeros((self.n, 3), dtype=np.float64)
        X = self.x0
        n_theta = self.n_theta
        for i in range(self.n_z):
            for k in range(n_theta):
                kp1 = (k + 1) % n_theta
                a = self._node_index(1, i, k)
                b = self._node_index(1, i, kp1)
                c = self._node_index(1, i + 1, kp1)
                d = self._node_index(1, i + 1, k)

                p_a = X[a]
                p_b = X[b]
                p_c = X[c]
                p_d = X[d]

                a1 = 0.5 * np.cross(p_b - p_a, p_d - p_a)
                a2 = 0.5 * np.cross(p_c - p_b, p_d - p_b)

                # Outward direction from the cavity is radially outward.
                center = 0.25 * (p_a + p_b + p_c + p_d)
                radial = center.copy()
                radial[2] = 0.0
                outward = radial / (np.linalg.norm(radial) + 1e-12)

                def orient(v: np.ndarray) -> np.ndarray:
                    if np.dot(v, outward) < 0.0:
                        return -v
                    return v

                n1 = orient(a1)
                n2 = orient(a2)

                normal[a] += n1 / 3.0
                normal[b] += n1 / 3.0 + n2 / 3.0
                normal[d] += n1 / 3.0 + n2 / 3.0
                normal[c] += n2 / 3.0

        # Only inner-surface nodes carry pressure; zero the rest.
        mask = self.pressure_surface[:, None]
        normal *= mask
        return normal

    def _min_edge_length(self) -> float:
        elems = self.elements
        X = self.x0
        h_min = float("inf")
        for j in range(1, 4):
            d = np.linalg.norm(X[elems[:, j]] - X[elems[:, 0]], axis=1)
            h_min = min(h_min, float(d.min()))
        return h_min

    def _elastic_forces(self) -> np.ndarray:
        """Assemble nodal elastic forces using a St. Venant-Kirchhoff model."""
        x = self.x
        elems = self.elements
        i0, i1, i2, i3 = elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3]

        Ds = np.stack(
            [x[i1] - x[i0], x[i2] - x[i0], x[i3] - x[i0]], axis=2
        )  # (Ne, 3, 3)
        F = Ds @ self.Dm_inv

        # Green-Lagrange strain E = 0.5 (F^T F - I).
        C = np.transpose(F, (0, 2, 1)) @ F
        I = np.eye(3)
        E = 0.5 * (C - I)

        # 2nd Piola-Kirchhoff stress S = 2 mu E + lambda tr(E) I.
        trace_E = np.trace(E, axis1=1, axis2=2)
        S = 2.0 * self.mu * E + self.lam * trace_E[:, None, None] * I

        # 1st Piola-Kirchhoff stress P = F S.
        P = F @ S

        # Nodal forces for nodes 1-3: -V0 P Dm^{-T}; node 0 gets the balance.
        H = P @ self.Dm_inv_T  # (Ne, 3, 3)
        f_tet = -self.rest_volume[:, None, None] * H

        f = np.zeros((self.n, 3), dtype=np.float64)
        np.add.at(f, i1, f_tet[:, :, 0])
        np.add.at(f, i2, f_tet[:, :, 1])
        np.add.at(f, i3, f_tet[:, :, 2])
        np.add.at(f, i0, -(f_tet[:, :, 0] + f_tet[:, :, 1] + f_tet[:, :, 2]))
        return f

    def _step(self, dt: float) -> None:
        f = self._elastic_forces()

        # Cavity pressure traction (ramp provided by the scene).
        pressure = float(self.scene.pressure_at(self.t))
        f += pressure * self.pressure_normal

        # Velocity-proportional damping.
        f -= self.damping * self.v

        # Explicit central-difference update.
        inv_m = 1.0 / self.m
        a = f * inv_m[:, None]
        self.v += a * dt
        self.x += self.v * dt

        # Enforce fixed rim constraints.
        self.v[self.fixed] = 0.0
        self.x[self.fixed] = self.x0[self.fixed]

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")
        if time < self.t:
            raise ValueError("advance_to time must be non-decreasing")

        while self.t < time - 1e-12:
            dt = min(self.dt, time - self.t)
            self._step(dt)
            self.t += dt

        return self._make_frame()

    def _compute_element_F(self) -> np.ndarray:
        x = self.x
        elems = self.elements
        i0, i1, i2, i3 = elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3]
        Ds = np.stack(
            [x[i1] - x[i0], x[i2] - x[i0], x[i3] - x[i0]], axis=2
        )
        return Ds @ self.Dm_inv

    def _make_frame(self) -> StateFrame:
        F = self._compute_element_F()

        # Net external force on the whole body (pressure only, gravity is zero).
        pressure = float(self.scene.pressure_at(self.t))
        net_external = np.sum(pressure * self.pressure_normal, axis=0, keepdims=True)
        budget_external_force = net_external.reshape(1, 1, 3)

        fields = {
            "element_deformation_gradient": F,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
        }

        scalars = dict(self._scalars)
        scalars["reference_force"] = max(1.0, float(np.linalg.norm(net_external)))

        return StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={k: np.copy(v) for k, v in self._attrs.items()},
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict:
        return {
            "solver": "BenchmarkSolver",
            "dt": self.dt if self._initialized else None,
            "elements": self.n_elements if self._initialized else 0,
        }
