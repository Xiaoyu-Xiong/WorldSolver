"""Corotational FEM solver for the gravity sag of a segmented soft spine cantilever.

Physics
-------
The rubber spine arm is discretized as a three-dimensional volumetric solid: a
swept structured grid (3x3 cross-section nodes per axial level) follows the
frozen ``assets/spine_arm.obj`` envelope, alternating broad vertebra sections
with thin flexible necks.  Every structured cell is split into six conforming
tetrahedra (Kuhn fan around the body diagonal).

Each tetrahedron uses corotational linear elasticity built from the frozen
Lame parameters (E, nu of the config) with warped stiffness
``K_e = R K0 R^T`` and force ``f_e = -R K0 (R^T x - X)``, where ``R`` comes from
the polar/SVD factorisation of the element deformation gradient.  Gravity acts
as a body force, Rayleigh damping models the rubber viscosity, and the root
block nodes inside the frozen fixed region are Dirichlet-clamped.

Time integration is linearly-implicit (semi-implicit) Euler,

    (M (1 + h a) + (h b + h^2) K) v_{n+1} = M v_n + h (f_int(x_n) + f_ext)
    x_{n+1} = x_n + h v_{n+1}

solved with a sparse LU factorisation on the free degrees of freedom, which is
unconditionally stable for this stiff, nearly incompressible material.

The 168 surface samples requested by the state binding are embedded in the
simulation mesh with fixed barycentric weights, so exported positions,
velocities and accelerations are affine interpolations of the simulated field.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame

# Six-tetrahedron conforming decomposition of a structured cell (fan around the
# 0-6 diagonal); identical local orientation everywhere keeps faces conforming.
HEX_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def _segments() -> list[tuple[float, float, float]]:
    """(length, half_y, half_z) of the root block, necks and vertebrae."""
    segs = [(0.13, 0.075, 0.115)]
    for i in range(10):
        half_y = 0.075 - 0.0025 * i
        segs.append((0.032, half_y * 0.58, 0.027))
        segs.append((0.092 - 0.003 * i, half_y, 0.105 - 0.0065 * i))
    return segs


def _axial_levels(target_h: float = 0.022, transition: float = 0.004):
    """Axial sample levels with cross-section, using short transition slabs."""
    xs: list[float] = []
    hy: list[float] = []
    hz: list[float] = []
    segs = _segments()
    x = 0.0
    for k, (length, sy, sz) in enumerate(segs):
        x0, x1 = x, x + length
        a = x0 + (0.5 * transition if k > 0 else 0.0)
        b = x1 - (0.5 * transition if k < len(segs) - 1 else 0.0)
        n = max(2, int(round((b - a) / target_h)))
        for value in np.linspace(a, b, n + 1):
            xs.append(float(value))
            hy.append(float(sy))
            hz.append(float(sz))
        x = x1
    return (
        np.asarray(xs, dtype=float),
        np.asarray(hy, dtype=float),
        np.asarray(hz, dtype=float),
    )


def _build_mesh(nyz: int = 3):
    """Swept structured tetrahedral mesh of the spine arm (local coordinates)."""
    xs, hy, hz = _axial_levels()
    nlev = len(xs)
    unit = np.linspace(-1.0, 1.0, nyz)
    yy, zz = np.meshgrid(unit, unit, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    per = nyz * nyz

    nodes = np.empty((nlev * per, 3), dtype=float)
    for level in range(nlev):
        base = level * per
        nodes[base:base + per, 0] = xs[level]
        nodes[base:base + per, 1] = yy * hy[level]
        nodes[base:base + per, 2] = zz * hz[level]

    def nid(level: int, j: int, k: int) -> int:
        return level * per + j * nyz + k

    tets: list[tuple[int, int, int, int]] = []
    for level in range(nlev - 1):
        for j in range(nyz - 1):
            for k in range(nyz - 1):
                corner = (
                    nid(level, j, k),
                    nid(level + 1, j, k),
                    nid(level + 1, j + 1, k),
                    nid(level, j + 1, k),
                    nid(level, j, k + 1),
                    nid(level + 1, j, k + 1),
                    nid(level + 1, j + 1, k + 1),
                    nid(level, j + 1, k + 1),
                )
                for tet in HEX_TETS:
                    tets.append(tuple(corner[t] for t in tet))

    elements = np.asarray(tets, dtype=np.int64)
    edges = nodes[elements[:, 1:]] - nodes[elements[:, :1]]
    detj = np.linalg.det(edges)
    flip = detj < 0.0
    if np.any(flip):
        elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    return nodes, elements


def _rest_operators(nodes: np.ndarray, elements: np.ndarray):
    """Inverse edge matrices and element volumes of the rest configuration."""
    x0 = nodes[elements[:, 0]]
    edges = np.stack(
        [nodes[elements[:, i]] - x0 for i in (1, 2, 3)], axis=2
    )  # [Ne,3,3], columns are rest edges
    volume = np.linalg.det(edges) / 6.0
    inv_edges = np.linalg.inv(edges)
    return inv_edges, volume


def _shape_gradients(inv_edges: np.ndarray) -> np.ndarray:
    """dN_i/dX for the four linear tetrahedron shape functions, shape [Ne,4,3]."""
    ne = inv_edges.shape[0]
    grad = np.zeros((ne, 4, 3), dtype=float)
    # X = X_0 + E lambda with E holding the rest edges as columns, so
    # lambda = E^-1 (X - X_0) and grad lambda_i is row i-1 of E^-1.
    grad[:, 1:, :] = inv_edges
    grad[:, 0, :] = -grad[:, 1:, :].sum(axis=1)
    return grad


def _element_stiffness(grad: np.ndarray, volume: np.ndarray, lam: float, mu: float):
    """Rest-frame linear-elastic element stiffness matrices, shape [Ne,12,12]."""
    ne = grad.shape[0]
    b = np.zeros((ne, 6, 12), dtype=float)
    for i in range(4):
        gx = grad[:, i, 0]
        gy = grad[:, i, 1]
        gz = grad[:, i, 2]
        c = 3 * i
        b[:, 0, c + 0] = gx
        b[:, 1, c + 1] = gy
        b[:, 2, c + 2] = gz
        b[:, 3, c + 0] = gy
        b[:, 3, c + 1] = gx
        b[:, 4, c + 1] = gz
        b[:, 4, c + 2] = gy
        b[:, 5, c + 0] = gz
        b[:, 5, c + 2] = gx
    d = np.zeros((6, 6), dtype=float)
    d[:3, :3] = lam
    d[0, 0] = d[1, 1] = d[2, 2] = lam + 2.0 * mu
    d[3, 3] = d[4, 4] = d[5, 5] = mu
    k0 = np.einsum("eki,kl,elj->eij", b, d, b) * volume[:, None, None]
    return 0.5 * (k0 + np.transpose(k0, (0, 2, 1)))


def _embed(points: np.ndarray, nodes: np.ndarray, elements: np.ndarray,
           inv_edges: np.ndarray):
    """Barycentric embedding of arbitrary points into the tetrahedral mesh."""
    x0 = nodes[elements[:, 0]]
    npts = len(points)
    host = np.zeros(npts, dtype=np.int64)
    weights = np.zeros((npts, 4), dtype=float)
    for p in range(npts):
        rel = points[p][None, :] - x0
        bary = np.einsum("eij,ej->ei", inv_edges, rel)
        full = np.concatenate([1.0 - bary.sum(axis=1, keepdims=True), bary], axis=1)
        outside = np.maximum(-full, 0.0).sum(axis=1)
        best = int(np.argmin(outside))
        host[p] = best
        weights[p] = full[best]
    return host, weights


class BenchmarkSolver:
    """Corotational tetrahedral FEM with linearly-implicit Euler time stepping."""

    # Time stepping (a sample interval of 0.02 s is an integer multiple of this).
    substep_dt = 1.0e-3
    # Rayleigh damping: mild mass term plus rubber viscosity on the stiffness.
    damp_alpha = 0.3
    damp_beta = 1.0e-2

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.config = config
        self.seed = int(seed)

        material = config["material"]
        youngs = float(material["youngs_modulus_pa"])
        poisson = float(material["poisson_ratio"])
        self.density = float(material["density"])
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.gravity = np.asarray(config["gravity"], dtype=float)

        # --- volumetric simulation mesh (object-local coordinates) -----------
        nodes, elements = _build_mesh()
        self.rest_nodes = nodes
        self.elements = elements
        self.n_nodes = len(nodes)
        self.n_elem = len(elements)
        self.inv_edges, self.volume = _rest_operators(nodes, elements)
        grad = _shape_gradients(self.inv_edges)
        self.k0 = _element_stiffness(grad, np.abs(self.volume), self.lam, self.mu)

        # Lumped nodal masses of the simulated solid.
        elem_mass = self.density * np.abs(self.volume) * 0.25
        node_mass = np.zeros(self.n_nodes, dtype=float)
        for corner in range(4):
            node_mass += np.bincount(
                elements[:, corner], weights=elem_mass, minlength=self.n_nodes
            )
        self.node_mass = node_mass
        self.body_force = node_mass[:, None] * self.gravity[None, :]

        # --- frozen Dirichlet clamp -----------------------------------------
        x_max = float(config["boundary_conditions"]["fixed_region"]["x_max"])
        self.fixed_nodes = nodes[:, 0] <= x_max
        free_node = ~self.fixed_nodes
        free_dof = np.repeat(free_node, 3)
        self.free_dof = np.flatnonzero(free_dof)
        self.n_free = len(self.free_dof)
        reduced = np.full(3 * self.n_nodes, -1, dtype=np.int64)
        reduced[self.free_dof] = np.arange(self.n_free, dtype=np.int64)

        dofs = (3 * elements[:, :, None] + np.arange(3)[None, None, :]).reshape(
            self.n_elem, 12
        )
        rows = np.repeat(dofs, 12, axis=1).ravel()
        cols = np.tile(dofs, (1, 12)).ravel()
        keep = free_dof[rows] & free_dof[cols]
        self.keep = keep
        self.rows_r = reduced[rows[keep]]
        self.cols_r = reduced[cols[keep]]
        self.mass_free = np.repeat(node_mass, 3)[self.free_dof]

        # --- exported surface samples embedded in the volume ----------------
        initial = scene.initial_state()
        self.export_points = np.asarray(initial.position, dtype=float)
        self.n_export = len(self.export_points)
        self.export_mass = np.asarray(initial.mass, dtype=float)
        self.export_fixed = np.asarray(initial.attributes["fixed"], dtype=float)
        translation = np.zeros(3, dtype=float)
        for item in (config.get("scene_geometry") or {}).get("objects", []):
            if item.get("id") == "spine_arm":
                translation = np.asarray(
                    (item.get("transform") or {}).get("translation", [0.0, 0.0, 0.0]),
                    dtype=float,
                )
        self.translation = translation
        self.host, self.weights = _embed(
            self.export_points, nodes, elements, self.inv_edges
        )
        self.host_nodes = elements[self.host]

        # --- state ----------------------------------------------------------
        self.position = nodes.copy()
        self.velocity = np.zeros_like(nodes)
        self.acceleration = np.where(
            self.fixed_nodes[:, None], 0.0, self.gravity[None, :]
        )
        self.time = 0.0
        self.reference_force = float(
            abs(self.export_mass.sum() * np.linalg.norm(self.gravity))
        )
        if not np.isfinite(self.reference_force) or self.reference_force <= 0.0:
            self.reference_force = 1.0
        self.reverts = 0
        self.steps = 0

    # ------------------------------------------------------------------ physics
    def _deformation_gradients(self, position: np.ndarray) -> np.ndarray:
        """Per-element deformation gradient F = dx/dX in material coordinates."""
        x0 = position[self.elements[:, 0]]
        edges = np.stack(
            [position[self.elements[:, i]] - x0 for i in (1, 2, 3)], axis=2
        )
        return edges @ self.inv_edges

    @staticmethod
    def _rotations(defgrad: np.ndarray) -> np.ndarray:
        """Proper rotation of the polar decomposition of F, via SVD."""
        u, _, vt = np.linalg.svd(defgrad)
        rot = u @ vt
        bad = np.linalg.det(rot) < 0.0
        if np.any(bad):
            u_bad = u[bad].copy()
            u_bad[:, :, 2] *= -1.0
            rot[bad] = u_bad @ vt[bad]
        return rot

    def _internal(self, position: np.ndarray, rot: np.ndarray):
        """Corotational internal forces [n,3] and warped element stiffness."""
        elements = self.elements
        xe = position[elements]
        rest = self.rest_nodes[elements]
        # Material-frame nodal coordinates: (R^T x)_i = R_ai x_a.
        local = np.einsum("eai,ena->eni", rot, xe) - rest
        ku = np.einsum("eij,ej->ei", self.k0, local.reshape(-1, 12)).reshape(-1, 4, 3)
        # f_e = -R K0 (R^T x - X): rotate the material-frame force back to world.
        felem = -np.einsum("eij,enj->eni", rot, ku)

        force = np.zeros((self.n_nodes, 3), dtype=float)
        for comp in range(3):
            for corner in range(4):
                force[:, comp] += np.bincount(
                    elements[:, corner],
                    weights=felem[:, corner, comp],
                    minlength=self.n_nodes,
                )

        k0r = self.k0.reshape(-1, 4, 3, 4, 3)
        tmp = np.einsum("eia,emanb->eminb", rot, k0r)
        kelem = np.einsum("eminb,ejb->eminj", tmp, rot).reshape(-1, 12, 12)
        return force, kelem

    def _step(self, dt: float) -> None:
        prev_position = self.position
        prev_velocity = self.velocity
        rot = self._rotations(self._deformation_gradients(prev_position))
        f_int, kelem = self._internal(prev_position, rot)

        values = kelem.ravel()[self.keep]
        stiffness = sp.coo_matrix(
            (values, (self.rows_r, self.cols_r)), shape=(self.n_free, self.n_free)
        ).tocsc()
        system = (dt * dt + dt * self.damp_beta) * stiffness
        system = system + sp.diags(
            self.mass_free * (1.0 + dt * self.damp_alpha), format="csc"
        )

        total = (f_int + self.body_force).ravel()[self.free_dof]
        rhs = self.mass_free * prev_velocity.ravel()[self.free_dof] + dt * total

        try:
            solution = spla.splu(system).solve(rhs)
        except Exception:
            solution = spla.spsolve(system.tocsr(), rhs)

        velocity = np.zeros((3 * self.n_nodes,), dtype=float)
        velocity[self.free_dof] = solution
        velocity = velocity.reshape(self.n_nodes, 3)
        position = prev_position + dt * velocity
        if not (np.isfinite(velocity).all() and np.isfinite(position).all()):
            # Deterministic safeguard: keep the last valid configuration at rest.
            self.velocity = np.zeros_like(prev_velocity)
            self.acceleration = np.zeros_like(prev_velocity)
            self.reverts += 1
            return
        self.acceleration = (velocity - prev_velocity) / dt
        self.velocity = velocity
        self.position = position
        self.steps += 1

    # ------------------------------------------------------------------ export
    def _interpolate(self, nodal: np.ndarray) -> np.ndarray:
        return np.einsum("pn,pnc->pc", self.weights, nodal[self.host_nodes])

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        eps = 0.25 * self.substep_dt
        while self.time < target - eps:
            dt = min(self.substep_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target

        position = self._interpolate(self.position) + self.translation[None, :]
        velocity = self._interpolate(self.velocity)
        acceleration = self._interpolate(self.acceleration)
        defgrad = self._deformation_gradients(self.position)

        budget = (self.export_mass[:, None] * acceleration).sum(axis=0)
        frame = StateFrame(
            time=target,
            position=np.ascontiguousarray(position, dtype=float),
            velocity=np.ascontiguousarray(velocity, dtype=float),
            mass=np.ascontiguousarray(self.export_mass, dtype=float),
            attributes={
                "mass_quantity": self.export_mass[:, None].copy(),
                "fixed": self.export_fixed.copy(),
                "acceleration": np.ascontiguousarray(acceleration, dtype=float),
                "external_force": self.export_mass[:, None] * self.gravity[None, :],
                "rest_position": self.export_points + self.translation[None, :],
            },
            fields={
                "budget_external_force": budget.reshape(1, 1, 3),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": np.ascontiguousarray(
                    defgrad, dtype=float
                ),
            },
            scalars={
                "reference_force": self.reference_force,
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        tip = float(self.position[:, 0].max())
        tip_nodes = self.position[self.position[:, 0] > tip - 1e-9]
        return {
            "method": "corotational linear tetrahedra, linearly-implicit Euler",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elem),
            "free_dofs": int(self.n_free),
            "fixed_nodes": int(self.fixed_nodes.sum()),
            "substep_dt": float(self.substep_dt),
            "substeps": int(self.steps),
            "reverted_steps": int(self.reverts),
            "lame_mu": float(self.mu),
            "lame_lambda": float(self.lam),
            "simulated_mass_kg": float(self.node_mass.sum()),
            "exported_mass_kg": float(self.export_mass.sum()),
            "tip_z_local": float(tip_nodes[:, 2].mean()) if len(tip_nodes) else 0.0,
        }
