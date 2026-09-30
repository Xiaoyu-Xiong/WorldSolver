"""Physically based FEM solver: incompressible elastic sphere compressed between plates.

Method
------
The deformable body is discretized as a tetrahedral finite-element solid built
from the frozen ``assets/soft_sphere.obj`` icosahedron surface (subdivided twice,
with one inner shell and a center node).  The material is a compressible
neo-Hookean solid with a stiff volumetric penalty

    psi(F) = mu/2 (I1 - 3) - mu log(J) + kappa/2 (J - 1)^2,
    P      = mu (F - F^-T) + kappa (J - 1) J F^-T,

which approaches the incompressible (volume conserving) limit for kappa >> mu.
The dynamics are advanced with a symplectic (semi-implicit) Euler scheme, an
internal (momentum conserving) pairwise viscous damping term, and impulse/positional
contact against the two kinematic compression plates (Signorini condition with
Coulomb friction).  The solver is fully deterministic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# ----------------------------------------------------------------------------
# mesh utilities
# ----------------------------------------------------------------------------
def _load_obj(path: Path):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _subdivide(vertices: np.ndarray, faces: np.ndarray):
    """Loop-style 1-to-4 triangle subdivision; original vertices keep their indices."""
    new_vertices = [np.asarray(v, dtype=float) for v in vertices]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        found = cache.get(key)
        if found is not None:
            return found
        index = len(new_vertices)
        new_vertices.append(0.5 * (vertices[i] + vertices[j]))
        cache[key] = index
        return index

    new_faces: list[list[int]] = []
    for a, b, c in faces:
        ab = midpoint(int(a), int(b))
        bc = midpoint(int(b), int(c))
        ca = midpoint(int(c), int(a))
        new_faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return np.asarray(new_vertices), np.asarray(new_faces, dtype=np.int64)


class BenchmarkSolver:
    """Explicit FEM solver for the plate-compression scene."""

    SUBDIVISIONS = 2
    SHELL_SCALE = 0.5
    MU = 3.0e2  # shear modulus [Pa]
    KAPPA = 1.5e5  # volumetric penalty modulus [Pa] (near-incompressible, kappa/mu = 500)
    FRICTION = 0.35
    DAMP_RATE = 10.0  # target internal viscous decay rate [1/s]
    CFL = 0.30
    VELOCITY_LIMIT = 30.0  # m/s; far above any physical speed in this scene

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)

        config = getattr(scene, "config", None)
        if config is None:
            import yaml  # type: ignore

            config = yaml.safe_load(
                (Path(__file__).resolve().parent.parent / "config.yaml").read_text("utf-8")
            )
        self.config = config

        initial = scene.initial_state()
        self.initial_position = np.asarray(initial.position, dtype=float).copy()
        self.n_samples = len(self.initial_position)

        workspace = Path(__file__).resolve().parent.parent
        obj_path = workspace / "assets" / "soft_sphere.obj"
        if obj_path.is_file():
            verts, faces = _load_obj(obj_path)
        else:  # deterministic fallback: convex hull of the rest vertices
            from scipy.spatial import ConvexHull

            verts = self.initial_position.copy()
            faces = np.asarray(ConvexHull(verts).simplices, dtype=np.int64)

        # outer shell = subdivided frozen surface (original vertices keep indices 0..11)
        for _ in range(self.SUBDIVISIONS):
            verts, faces = _subdivide(verts, faces)
        outer = verts
        n_outer = len(outer)
        inner = self.SHELL_SCALE * verts
        center = np.zeros((1, 3))
        nodes = np.vstack((outer, inner, center))
        self.n_outer = n_outer

        # tetrahedra: prisms between the shells (3 tets) + star tets to the center
        tets: list[list[int]] = []
        for a, b, c in faces:
            ia, ib, ic = n_outer + int(a), n_outer + int(b), n_outer + int(c)
            tets.append((int(a), int(b), int(c), ia))
            tets.append((ia, int(b), int(c), ib))
            tets.append((ia, ib, int(c), ic))
            tets.append((2 * n_outer, ia, ib, ic))
        tets = np.asarray(tets, dtype=np.int64)

        # fix orientation so that det(Dm) > 0
        p0 = nodes[tets[:, 0]]
        dm = np.stack(
            (nodes[tets[:, 1]] - p0, nodes[tets[:, 2]] - p0, nodes[tets[:, 3]] - p0), axis=2
        )
        det = np.linalg.det(dm)
        flip = det < 0.0
        if np.any(flip):
            swapped = tets[flip].copy()
            tmp = swapped[:, 0].copy()
            swapped[:, 0] = swapped[:, 1]
            swapped[:, 1] = tmp
            tets[flip] = swapped
            p0 = nodes[tets[:, 0]]
            dm = np.stack(
                (nodes[tets[:, 1]] - p0, nodes[tets[:, 2]] - p0, nodes[tets[:, 3]] - p0), axis=2
            )
            det = np.linalg.det(dm)
        assert np.all(det > 0.0), "degenerate rest element"

        self.tets = tets
        self.nodes = nodes
        self.dm_inv = np.linalg.inv(dm)
        self.dm_inv_t = np.transpose(self.dm_inv, (0, 2, 1))
        self.rest_volume = np.abs(det) / 6.0
        self.rest_total_volume = float(self.rest_volume.sum())

        # lumped masses from the frozen total mass
        total_mass = float(config["sphere"]["mass"])
        density = total_mass / self.rest_total_volume
        masses = np.zeros(len(nodes))
        for corner in range(4):
            np.add.at(masses, tets[:, corner], self.rest_volume / 4.0)
        self.node_mass = masses * density

        # unique edges (for internal viscous damping)
        pairs = np.vstack(
            (
                tets[:, [0, 1]],
                tets[:, [0, 2]],
                tets[:, [0, 3]],
                tets[:, [1, 2]],
                tets[:, [1, 3]],
                tets[:, [2, 3]],
            )
        )
        pairs = np.sort(pairs, axis=1)
        self.edges = np.unique(pairs, axis=0)
        degree = np.zeros(len(nodes))
        np.add.at(degree, self.edges[:, 0], 1.0)
        np.add.at(degree, self.edges[:, 1], 1.0)
        self.degree = np.maximum(degree, 1.0)

        # sparse scatter operators for fast force assembly
        from scipy.sparse import csr_matrix

        n_elements = len(tets)
        element_corner = np.arange(n_elements * 4)
        self.assemble = csr_matrix(
            (np.ones(n_elements * 4), (tets.reshape(-1), element_corner)),
            shape=(len(nodes), n_elements * 4),
        )
        n_edges = len(self.edges)
        edge_index = np.arange(n_edges)
        self.edge_scatter = csr_matrix(
            (
                np.concatenate((np.ones(n_edges), -np.ones(n_edges))),
                (
                    np.concatenate((self.edges[:, 0], self.edges[:, 1])),
                    np.concatenate((edge_index, edge_index)),
                ),
            ),
            shape=(len(nodes), n_edges),
        )

        # damping coefficient: per-node decay rate ~ c * deg / m, target DAMP_RATE
        ratio = self.node_mass / self.degree
        self.damp_c = self.DAMP_RATE * float(ratio.min())

        # CFL limit from the smallest element altitude and pressure wave speed
        areas = np.zeros((len(tets), 4))
        for corner in range(4):
            others = [k for k in range(4) if k != corner]
            e1 = nodes[tets[:, others[0]]] - nodes[tets[:, corner]]
            e2 = nodes[tets[:, others[1]]] - nodes[tets[:, corner]]
            areas[:, corner] = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        altitude = 3.0 * self.rest_volume / np.maximum(areas.max(axis=1), 1e-12)
        h_min = float(altitude.min())
        wave = math.sqrt((self.KAPPA + 4.0 * self.MU / 3.0) / density)
        dt_cfl = self.CFL * h_min / wave
        dt_damp = 0.2 * float((self.node_mass / (self.damp_c * self.degree)).min())
        self.dt_max = float(min(dt_cfl, dt_damp, 1.0e-3))

        # kinematic plates from the frozen configuration
        plates = config["plates"]
        z_initial = float(plates["initial_inner_faces_z"][1])
        z_final = float(plates["final_inner_faces_z"][1])
        t0 = float(plates["motion_start"])
        t1 = float(plates["motion_end"])
        self.gap_initial = 2.0 * z_initial
        self.plate_travel = z_initial - z_final
        self.motion_start = t0
        self.motion_window = max(t1 - t0, 1e-9)
        self.plate_speed = self.plate_travel / self.motion_window
        self.z_initial = z_initial
        self.gap_final = 2.0 * z_final

        self.reference_momentum = total_mass * 2.0 * self.plate_speed
        self.sample_mass = np.full(self.n_samples, total_mass / self.n_samples)

        # state
        self.x = nodes.copy()
        self.v = np.zeros_like(nodes)
        self.time = 0.0
        self.element_f = np.tile(np.eye(3), (len(tets), 1, 1))
        self.element_j = np.ones(len(tets))
        self.impulse_total = np.zeros(3)
        self.window_impulse = np.zeros(3)
        self.window_dt = 0.0
        self.volume_error_max = 0.0

    # ------------------------------------------------------------- kinematics
    def _plate_faces(self, time: float):
        progress = (time - self.motion_start) / self.motion_window
        progress = min(max(progress, 0.0), 1.0)
        face = self.z_initial - self.plate_travel * progress
        speed = self.plate_speed if 0.0 < progress < 1.0 else 0.0
        return -face, face, speed  # lower face z, upper face z, |plate speed|

    # ---------------------------------------------------------------- physics
    def _element_deformation(self, x: np.ndarray):
        p0 = x[self.tets[:, 0]]
        ds = np.stack(
            (x[self.tets[:, 1]] - p0, x[self.tets[:, 2]] - p0, x[self.tets[:, 3]] - p0), axis=2
        )
        return ds @ self.dm_inv

    @staticmethod
    def _det_and_inverse_transpose(m: np.ndarray):
        """Batched det(M) and M^-T for 3x3 matrices (closed form)."""
        a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
        d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
        g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
        cof00 = e * i - f * h
        cof01 = -(d * i - f * g)
        cof02 = d * h - e * g
        determinant = a * cof00 + b * cof01 + c * cof02
        safe = np.where(np.abs(determinant) < 1e-12, 1e-12, determinant)
        inverse = np.empty_like(m)
        inverse[:, 0, 0] = cof00
        inverse[:, 0, 1] = -(b * i - c * h)
        inverse[:, 0, 2] = b * f - c * e
        inverse[:, 1, 0] = cof01
        inverse[:, 1, 1] = a * i - c * g
        inverse[:, 1, 2] = -(a * f - c * d)
        inverse[:, 2, 0] = cof02
        inverse[:, 2, 1] = -(a * h - b * g)
        inverse[:, 2, 2] = a * e - b * d
        inverse /= safe[:, None, None]
        return determinant, np.transpose(inverse, (0, 2, 1))

    def _internal_forces(self, x: np.ndarray, v: np.ndarray):
        deformation = self._element_deformation(x)
        jacobian, inverse_transpose = self._det_and_inverse_transpose(deformation)
        inverse_transpose = np.clip(inverse_transpose, -1.0e6, 1.0e6)
        volumetric = self.KAPPA * (jacobian - 1.0) * jacobian
        piola = self.MU * (deformation - inverse_transpose)
        piola = piola + volumetric[:, None, None] * inverse_transpose
        q = self.rest_volume[:, None, None] * (piola @ self.dm_inv_t)

        # scatter element contributions to nodes: f_0 = sum(columns), f_c = -column c
        contribution = np.empty((len(self.tets) * 4, 3))
        contribution[0::4] = q.sum(axis=2)
        contribution[1::4] = -q[:, :, 0]
        contribution[2::4] = -q[:, :, 1]
        contribution[3::4] = -q[:, :, 2]
        forces = self.assemble @ contribution

        # internal (momentum conserving) viscous damping: c * (v_j - v_i) per edge
        relative = v[self.edges[:, 1]] - v[self.edges[:, 0]]
        forces = forces + self.damp_c * (self.edge_scatter @ relative)

        self.element_f = deformation
        self.element_j = jacobian
        return np.asarray(forces)

    def _snapshot(self):
        return (
            self.x.copy(),
            self.v.copy(),
            self.element_f.copy(),
            self.element_j.copy(),
            self.impulse_total.copy(),
            self.window_impulse.copy(),
            self.volume_error_max,
            self.time,
        )

    def _restore(self, state) -> None:
        (
            self.x,
            self.v,
            self.element_f,
            self.element_j,
            self.impulse_total,
            self.window_impulse,
            self.volume_error_max,
            self.time,
        ) = state

    def _substep(self, dt: float) -> None:
        t_end = self.time + dt
        z_lower, z_upper, speed = self._plate_faces(t_end)

        forces = self._internal_forces(self.x, self.v)
        velocity = self.v + dt * forces / self.node_mass[:, None]

        # --- contact against the moving plates (impulse + Coulomb friction) ---
        predicted = self.x + dt * velocity
        before = velocity.copy()

        for sign, face_z, plate_v in ((1, z_lower, speed), (-1, z_upper, -speed)):
            penetration = sign * (face_z - predicted[:, 2])  # >0 when inside the plate
            touching = penetration > 0.0
            if not np.any(touching):
                continue
            normal_velocity = velocity[:, 2] - plate_v
            sticking = touching & (sign * normal_velocity < 0.0)
            velocity[sticking, 2] = plate_v

            # normal impulse magnitude (velocity fix + positional push-out)
            normal_impulse = np.zeros(len(velocity))
            normal_impulse[touching] = self.node_mass[touching] * penetration[touching] / dt
            normal_impulse[sticking] += self.node_mass[sticking] * np.abs(
                normal_velocity[sticking]
            )

            tangential = velocity[:, :2].copy()
            tangential_speed = np.linalg.norm(tangential, axis=1)
            active = touching & (tangential_speed > 1e-12)
            if np.any(active):
                limit = self.FRICTION * normal_impulse[active]
                magnitude = np.minimum(limit, self.node_mass[active] * tangential_speed[active])
                scale = 1.0 - magnitude / (self.node_mass[active] * tangential_speed[active])
                velocity[active, 0] *= scale
                velocity[active, 1] *= scale

        delta = velocity - before
        contact_impulse = np.sum(self.node_mass[:, None] * delta, axis=0)
        self.impulse_total += contact_impulse
        self.window_impulse += contact_impulse

        # --- integrate positions and enforce non-penetration ---
        self.x = self.x + dt * velocity
        np.clip(self.x[:, 2], z_lower, z_upper, out=self.x[:, 2])
        self.v = velocity
        self.time = t_end

        # deformation measure of the reached configuration (used for export and
        # for the element-inversion guard)
        deformation = self._element_deformation(self.x)
        self.element_f = deformation
        self.element_j = np.linalg.det(deformation)
        jmin = float(self.element_j.min())
        jmax = float(self.element_j.max())
        self.volume_error_max = max(self.volume_error_max, abs(jmin - 1.0), abs(jmax - 1.0))

    def _step_accepted(self, dt: float, min_j_before: float) -> bool:
        """Take one substep; reject (and roll back) on blow-up or element inversion."""
        state = self._snapshot()
        self._substep(dt)
        finite = np.isfinite(self.x).all() and np.isfinite(self.v).all()
        if not finite or float(np.abs(self.v).max()) > self.VELOCITY_LIMIT:
            self._restore(state)
            return False
        if float(self.element_j.min()) < 0.0 <= min_j_before:
            self._restore(state)
            return False
        return True

    def _advance(self, span: float) -> None:
        """Advance exactly `span` seconds, shrinking dt when a substep is rejected."""
        steps = max(int(math.ceil(span / self.dt_max - 1e-12)), 1)
        step_dt = span / steps
        remaining = span
        while remaining > 1e-15:
            h = min(step_dt, remaining)
            min_j_before = float(self.element_j.min())
            depth = 0
            while not self._step_accepted(h, min_j_before):
                h *= 0.5
                depth += 1
                if depth > 30 or h < 1e-13:
                    raise RuntimeError("substep refinement failed to produce a stable step")
            step_dt = h
            remaining -= h
            self.window_dt += h

    # ----------------------------------------------------------------- output
    def _sample_deformation_gradient(self):
        """Volume-weighted per-sample deformation gradient (surface vertices)."""
        weight = np.zeros(self.n_samples)
        accumulated = np.zeros((self.n_samples, 3, 3))
        for corner in range(4):
            idx = self.tets[:, corner]
            mask = idx < self.n_samples
            np.add.at(weight, idx[mask], self.rest_volume[mask])
            np.add.at(
                accumulated,
                idx[mask],
                self.rest_volume[mask, None, None] * self.element_f[mask],
            )
        scale = np.where(weight > 0.0, 1.0 / np.maximum(weight, 1e-30), 0.0)
        return accumulated * scale[:, None, None]

    def _frame(self, external_force: np.ndarray) -> StateFrame:
        z_lower, z_upper, _ = self._plate_faces(self.time)
        gap = z_upper - z_lower
        surface_z = self.x[: self.n_outer, 2]
        height = float(surface_z.max() - surface_z.min())

        deformation = self._sample_deformation_gradient()
        return StateFrame(
            time=float(self.time),
            position=self.x[: self.n_samples].copy(),
            velocity=self.v[: self.n_samples].copy(),
            mass=self.sample_mass.copy(),
            body_id=np.zeros(self.n_samples, dtype=np.int64),
            attributes={
                "mass_quantity": self.sample_mass.copy(),
                "rest_position": self.initial_position.copy(),
                "rest_volume_share": np.full(
                    self.n_samples,
                    float(self.config["sphere"]["rest_volume"]) / self.n_samples,
                ),
                "deformation_gradient": deformation,
            },
            fields={
                "budget_cumulative_external_impulse": self.impulse_total[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, self.n_samples)),
                "budget_external_force": external_force[None, None, :].copy(),
                "constraint_target": np.array([gap], dtype=float),
                "constraint_value": np.array([height], dtype=float),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": self.element_f.copy(),
            },
            scalars={
                "constraint_scale": float(self.gap_initial),
                "reference_momentum": float(abs(self.reference_momentum)),
                "rest_volume": float(self.config["sphere"]["rest_volume"]),
                "plate_gap": float(gap),
                "solid_volume": float(self.rest_total_volume),
            },
        )

    # ------------------------------------------------------------------- api
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-9:
            raise ValueError("sample times must be non-decreasing")
        span = target - self.time
        if span > 1e-12:
            self._advance(span)
            self.time = target  # remove floating point drift from the sample grid

        if self.window_dt > 1e-12:
            external_force = self.window_impulse / self.window_dt
        else:
            external_force = np.zeros(3)
        self.window_impulse[:] = 0.0
        self.window_dt = 0.0
        frame = self._frame(external_force)
        # window for the next sample starts now
        return frame

    def diagnostics(self) -> dict[str, object]:
        z_lower, z_upper, _ = self._plate_faces(self.time)
        jacobian = np.linalg.det(self.element_f)
        return {
            "method": "explicit fem, neo-hookean with volumetric penalty (near-incompressible)",
            "nodes": int(len(self.nodes)),
            "elements": int(len(self.tets)),
            "substep_dt_max": float(self.dt_max),
            "current_time": float(self.time),
            "plate_gap": float(z_upper - z_lower),
            "mean_jacobian": float(jacobian.mean()),
            "min_jacobian": float(jacobian.min()),
            "max_jacobian": float(jacobian.max()),
            "max_volume_error": float(self.volume_error_max),
            "cumulative_external_impulse": self.impulse_total.tolist(),
        }
