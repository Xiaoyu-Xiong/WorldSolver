"""Pressure-driven bending of a chambered soft pneumatic finger.

Physical model
--------------
The finger is a three-dimensional hyperelastic solid.  A conforming hexahedral
lattice is generated over the frozen scaffold geometry (a thick continuous
passive underside slab, six chamber lobes separated by compliant hinge gaps and
a terminal tip block).  Six enclosed internal cavities are carved out of the
lobes, leaving a thin top wall, thin side walls and a thick passive bottom
layer -- the asymmetric wall structure that converts uniform pressure into
bending.  Every hexahedron is split into six tetrahedra (Kuhn decomposition),
giving a conforming tetrahedral finite-element mesh.

Elasticity is St. Venant-Kirchhoff (rotation invariant, large-rotation capable)
with Kelvin-Voigt viscosity for material damping.  Loads are:

* gravity,
* follower pressure tractions on all cavity-wall faces (current deformed area
  vectors, so a closed cavity is self-equilibrated),
* unilateral penalty self-contact between facing chamber walls across the
  hinge gaps,
* unilateral penalty contact against the static base support block,
* homogeneous Dirichlet clamping of every node inside the frozen base support
  region x <= 0.030 m.

Time stepping is an explicit symplectic-Euler scheme with a CFL-limited step
and implicit mass-proportional drag, sub-stepped exactly onto the requested
sample times.

Exported samples are the 64 scaffold OBJ vertices, transported by barycentric
embedding in the simulation tetrahedra (exact affine interpolation, exact at
the rest configuration).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])

# Silicone-like elastomer.  Poisson ratio is kept moderately compressible so
# that the linear tetrahedra do not lock in bending.
DENSITY = 1070.0
YOUNGS_MODULUS = 5.0e5
POISSON_RATIO = 0.35
KELVIN_VOIGT_ETA = 20.0
MASS_DRAG = 30.0

# Scaffold geometry (assets/generate_pneumatic_finger.py), meters.
SLAB = (0.000, 0.210, -0.024, 0.024, 0.000, 0.014)
LOBE_X0 = [0.008 + 0.032 * i for i in range(6)]
LOBE_WIDTH = 0.027
LOBE_Y = (-0.024, 0.024)
LOBE_Z = (0.014, 0.050)
TIP = (0.200, 0.220, -0.024, 0.024, 0.008, 0.038)
WALL = 0.005            # chamber wall thickness
CAVITY_FLOOR = 0.019    # top of the passive bottom layer
CAVITY_ROOF = 0.045     # bottom of the thin top wall

SUPPORT_MIN = np.array([-0.003, -0.026, 0.000])
SUPPORT_MAX = np.array([0.031, 0.026, 0.052])

CONTACT_STIFFNESS = 3.0e4
CONTACT_GAP = 5.0e-4
CFL_SAFETY = 0.25
MAX_SPEED = 50.0


def _axis_nodes(breaks, max_spacing: float) -> np.ndarray:
    values = sorted({round(float(b), 9) for b in breaks})
    nodes = [values[0]]
    for lo, hi in zip(values[:-1], values[1:]):
        count = max(1, int(np.ceil((hi - lo) / max_spacing - 1e-9)))
        for i in range(1, count + 1):
            nodes.append(lo + (hi - lo) * i / count)
    return np.asarray(nodes, dtype=float)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)  # determinism anchor (unused noise)
        del rng

        pressure = config["pressure"]
        self.p_target = float(pressure["chamber_pressure_pa"])
        self.p_t0 = float(pressure["ramp_start_s"])
        self.p_t1 = float(pressure["ramp_end_s"])
        self.p_hold = float(pressure["hold_until_s"])
        base_range = config["initial_conditions"]["base_support_region_x"]
        self.clamp_x = float(base_range[1])

        self._build_mesh()
        self._build_pressure_faces()
        self._build_contact_pairs()
        self._build_embedding(scene)

        v0 = np.asarray(config["initial_conditions"]["initial_velocity_m_s"], dtype=float)
        self.x = self.X0.copy()
        self.v = np.tile(v0.reshape(1, 3), (len(self.X0), 1))
        self.v[self.fixed] = 0.0

        mu = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        lam = (
            YOUNGS_MODULUS
            * POISSON_RATIO
            / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
        )
        self.mu = mu
        self.lam = lam
        wave = np.sqrt((lam + 2.0 * mu) / DENSITY)
        self.dt_max = float(CFL_SAFETY * self.min_edge / wave)
        self.dt_max = min(self.dt_max, 1.0e-4)

        self.total_mass = float(self.node_mass.sum())
        self.reference_force = max(self.total_mass * abs(GRAVITY[2]), 1.0e-6)
        self.sample_mass = np.full(
            self.n_samples, self.total_mass / float(self.n_samples), dtype=float
        )

        self.time = 0.0
        self.steps = 0
        self._terms = self._force_terms(self.x, self.v, 0.0)

    # -------------------------------------------------------------- mesh build
    def _build_mesh(self) -> None:
        x_breaks = [SLAB[0], SLAB[1], TIP[0], TIP[1]]
        for x0 in LOBE_X0:
            x1 = x0 + LOBE_WIDTH
            x_breaks += [x0, x0 + WALL, x1 - WALL, x1]
        y_breaks = [LOBE_Y[0], LOBE_Y[0] + WALL, LOBE_Y[1] - WALL, LOBE_Y[1]]
        z_breaks = [0.0, 0.007, LOBE_Z[0], CAVITY_FLOOR, CAVITY_ROOF, LOBE_Z[1]]

        gx = _axis_nodes(x_breaks, 0.009)
        gy = _axis_nodes(y_breaks, 0.020)
        gz = _axis_nodes(z_breaks, 0.014)
        self.gx, self.gy, self.gz = gx, gy, gz
        nx, ny, nz = len(gx), len(gy), len(gz)

        solids = [SLAB, TIP]
        cavities = []
        for x0 in LOBE_X0:
            x1 = x0 + LOBE_WIDTH
            solids.append((x0, x1, LOBE_Y[0], LOBE_Y[1], LOBE_Z[0], LOBE_Z[1]))
            cavities.append(
                (
                    x0 + WALL,
                    x1 - WALL,
                    LOBE_Y[0] + WALL,
                    LOBE_Y[1] - WALL,
                    CAVITY_FLOOR,
                    CAVITY_ROOF,
                )
            )
        self.cavities = cavities

        cx = 0.5 * (gx[:-1] + gx[1:])
        cy = 0.5 * (gy[:-1] + gy[1:])
        cz = 0.5 * (gz[:-1] + gz[1:])
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")

        def inside(box):
            return (
                (CX > box[0]) & (CX < box[1])
                & (CY > box[2]) & (CY < box[3])
                & (CZ > box[4]) & (CZ < box[5])
            )

        solid = np.zeros(CX.shape, dtype=bool)
        for box in solids:
            solid |= inside(box)
        cavity = np.zeros(CX.shape, dtype=bool)
        for box in cavities:
            cavity |= inside(box)
        occupied = solid & ~cavity
        self.occupied = occupied
        self.cavity_cells = cavity

        lattice = np.stack(
            np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1
        ).reshape(-1, 3)

        def nid(i, j, k):
            return (i * ny + j) * nz + k

        hex_offsets = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ]
        kuhn = [
            (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
        ]

        tets = []
        cells = np.argwhere(occupied)
        for i, j, k in cells:
            corners = [nid(i + di, j + dj, k + dk) for di, dj, dk in hex_offsets]
            for a, b, c, d in kuhn:
                tets.append((corners[a], corners[b], corners[c], corners[d]))
        tets = np.asarray(tets, dtype=np.int64)

        used = np.unique(tets)
        remap = -np.ones(nx * ny * nz, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.lattice_remap = remap
        self.X0 = lattice[used].copy()
        self.tets = remap[tets]

        # Orient every tetrahedron positively.
        P = self.X0[self.tets]
        Dm = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=2)
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            swapped = self.tets.copy()
            swapped[flip, 0] = self.tets[flip, 1]
            swapped[flip, 1] = self.tets[flip, 0]
            self.tets = swapped
            P = self.X0[self.tets]
            Dm = np.stack(
                [P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=2
            )
            det = np.linalg.det(Dm)
        keep = det > 1e-14
        self.tets = self.tets[keep]
        Dm = Dm[keep]
        det = det[keep]
        self.Dm_inv = np.linalg.inv(Dm)
        self.Dm_invT = np.transpose(self.Dm_inv, (0, 2, 1))
        self.volume0 = det / 6.0
        self.n_nodes = len(self.X0)
        self.n_elements = len(self.tets)
        self.flat_tets = self.tets.reshape(-1)

        mass = np.zeros(self.n_nodes)
        contribution = DENSITY * self.volume0 / 4.0
        for corner in range(4):
            np.add.at(mass, self.tets[:, corner], contribution)
        mass = np.maximum(mass, 1e-9)
        self.node_mass = mass
        self.inv_mass = 1.0 / mass

        self.fixed = self.X0[:, 0] <= self.clamp_x + 1e-9
        self.free = ~self.fixed

        self.min_edge = float(
            min(np.diff(gx).min(), np.diff(gy).min(), np.diff(gz).min())
        )

    # ---------------------------------------------------------- cavity loading
    def _build_pressure_faces(self) -> None:
        gx, gy, gz = self.gx, self.gy, self.gz
        ny, nz = len(gy), len(gz)
        occupied = self.occupied
        cavity = self.cavity_cells
        remap = self.lattice_remap

        def nid(i, j, k):
            return (i * ny + j) * nz + k

        faces = []
        directions = []
        shape = occupied.shape
        for i, j, k in np.argwhere(cavity):
            for axis, step in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
                idx = [i, j, k]
                idx[axis] += step
                if not (0 <= idx[axis] < shape[axis]):
                    continue
                if not occupied[idx[0], idx[1], idx[2]]:
                    continue
                if axis == 0:
                    plane = i + 1 if step > 0 else i
                    quad = [
                        nid(plane, j, k), nid(plane, j + 1, k),
                        nid(plane, j + 1, k + 1), nid(plane, j, k + 1),
                    ]
                    normal = np.array([float(step), 0.0, 0.0])
                elif axis == 1:
                    plane = j + 1 if step > 0 else j
                    quad = [
                        nid(i, plane, k), nid(i + 1, plane, k),
                        nid(i + 1, plane, k + 1), nid(i, plane, k + 1),
                    ]
                    normal = np.array([0.0, float(step), 0.0])
                else:
                    plane = k + 1 if step > 0 else k
                    quad = [
                        nid(i, j, plane), nid(i + 1, j, plane),
                        nid(i + 1, j + 1, plane), nid(i, j + 1, plane),
                    ]
                    normal = np.array([0.0, 0.0, float(step)])
                local = remap[np.asarray(quad, dtype=np.int64)]
                if np.any(local < 0):
                    continue
                faces.append(local)
                directions.append(normal)

        if faces:
            quads = np.asarray(faces, dtype=np.int64)
            dirs = np.asarray(directions, dtype=float)
            P = self.X0[quads]
            area_vec = 0.5 * np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            sign = np.sum(area_vec * dirs, axis=1)
            flip = sign < 0.0
            if np.any(flip):
                fixed = quads.copy()
                fixed[flip, 1] = quads[flip, 3]
                fixed[flip, 3] = quads[flip, 1]
                quads = fixed
            self.pressure_quads = quads
            self.flat_quads = quads.reshape(-1)
        else:  # pragma: no cover - geometry always yields cavity faces
            self.pressure_quads = np.zeros((0, 4), dtype=np.int64)
            self.flat_quads = self.pressure_quads.reshape(-1)

    def _build_contact_pairs(self) -> None:
        gx, gy, gz = self.gx, self.gy, self.gz
        ny, nz = len(gy), len(gz)
        remap = self.lattice_remap

        def x_index(value):
            return int(np.argmin(np.abs(gx - value)))

        left_ids = []
        right_ids = []
        for lobe in range(len(LOBE_X0) - 1):
            i_right = x_index(LOBE_X0[lobe] + LOBE_WIDTH)
            i_left = x_index(LOBE_X0[lobe + 1])
            for j in range(ny):
                for k in range(nz):
                    if gz[k] < LOBE_Z[0] - 1e-9:
                        continue
                    a = remap[(i_right * ny + j) * nz + k]
                    b = remap[(i_left * ny + j) * nz + k]
                    if a < 0 or b < 0:
                        continue
                    right_ids.append(a)
                    left_ids.append(b)
        self.contact_a = np.asarray(right_ids, dtype=np.int64)
        self.contact_b = np.asarray(left_ids, dtype=np.int64)

    # ---------------------------------------------------------- mesh embedding
    def _build_embedding(self, scene) -> None:
        frame = scene.initial_state()
        targets = np.asarray(frame.position, dtype=float)
        self.n_samples = len(targets)
        self.sample_rest = targets.copy()
        self.sample_fixed = np.asarray(
            frame.attributes["fixed_base"], dtype=float
        ).copy()

        anchor = self.X0[self.tets[:, 3]]
        best_tet = np.zeros(self.n_samples, dtype=np.int64)
        best_w = np.zeros((self.n_samples, 4))
        for s in range(self.n_samples):
            rel = targets[s] - anchor
            lam = np.einsum("eij,ej->ei", self.Dm_inv, rel)
            w = np.concatenate([lam, (1.0 - lam.sum(axis=1))[:, None]], axis=1)
            score = w.min(axis=1)
            e = int(np.argmax(score))
            best_tet[s] = e
            best_w[s] = w[e]
        self.embed_nodes = self.tets[best_tet]
        self.embed_weights = best_w

    # ------------------------------------------------------------------ forces
    def _pressure(self, time: float) -> float:
        if time <= self.p_t0:
            return 0.0
        if time >= self.p_t1:
            return self.p_target
        span = max(self.p_t1 - self.p_t0, 1e-12)
        return self.p_target * (time - self.p_t0) / span

    def _force_terms(self, x, v, time):
        elastic = self._elastic_force(x, v)
        gravity = self.node_mass[:, None] * GRAVITY[None, :]
        pressure_value = self._pressure(time)
        pressure = np.zeros_like(x)
        if pressure_value != 0.0 and len(self.pressure_quads):
            P = x[self.pressure_quads]
            area_vec = 0.5 * np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            nodal = 0.25 * pressure_value * area_vec
            per_node = np.repeat(nodal[:, None, :], 4, axis=1).reshape(-1, 3)
            for c in range(3):
                pressure[:, c] = np.bincount(
                    self.flat_quads, weights=per_node[:, c], minlength=self.n_nodes
                )
        contact = self._contact_force(x)
        drag = -MASS_DRAG * self.node_mass[:, None] * v
        return {
            "elastic": elastic,
            "gravity": gravity,
            "pressure": pressure,
            "contact": contact,
            "drag": drag,
            "pressure_value": pressure_value,
        }

    def _elastic_force(self, x, v):
        P = x[self.tets]
        Ds = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=2)
        F = Ds @ self.Dm_inv
        Ft = np.transpose(F, (0, 2, 1))
        green = 0.5 * (Ft @ F)
        green[:, 0, 0] -= 0.5
        green[:, 1, 1] -= 0.5
        green[:, 2, 2] -= 0.5
        trace = green[:, 0, 0] + green[:, 1, 1] + green[:, 2, 2]
        S = 2.0 * self.mu * green
        S[:, 0, 0] += self.lam * trace
        S[:, 1, 1] += self.lam * trace
        S[:, 2, 2] += self.lam * trace

        Vs = v[self.tets]
        Dv = np.stack(
            [Vs[:, 0] - Vs[:, 3], Vs[:, 1] - Vs[:, 3], Vs[:, 2] - Vs[:, 3]], axis=2
        )
        Fdot = Dv @ self.Dm_inv
        strain_rate = 0.5 * (Ft @ Fdot + np.transpose(Fdot, (0, 2, 1)) @ F)
        S = S + 2.0 * KELVIN_VOIGT_ETA * strain_rate

        piola = F @ S
        H = -self.volume0[:, None, None] * (piola @ self.Dm_invT)
        f0 = H[:, :, 0]
        f1 = H[:, :, 1]
        f2 = H[:, :, 2]
        f3 = -(f0 + f1 + f2)
        stacked = np.stack([f0, f1, f2, f3], axis=1).reshape(-1, 3)
        out = np.zeros_like(x)
        for c in range(3):
            out[:, c] = np.bincount(
                self.flat_tets, weights=stacked[:, c], minlength=self.n_nodes
            )
        return out

    def _contact_force(self, x):
        force = np.zeros_like(x)
        if len(self.contact_a):
            a = self.contact_a
            b = self.contact_b
            gap = x[b, 0] - x[a, 0]
            active = gap < CONTACT_GAP
            if np.any(active):
                magnitude = CONTACT_STIFFNESS * (CONTACT_GAP - gap[active])
                ia = a[active]
                ib = b[active]
                np.add.at(force[:, 0], ia, -magnitude)
                np.add.at(force[:, 0], ib, magnitude)
        inside = (
            (x[:, 0] < SUPPORT_MAX[0])
            & (x[:, 1] > SUPPORT_MIN[1]) & (x[:, 1] < SUPPORT_MAX[1])
            & (x[:, 2] > SUPPORT_MIN[2]) & (x[:, 2] < SUPPORT_MAX[2])
            & self.free
        )
        if np.any(inside):
            force[inside, 0] += CONTACT_STIFFNESS * (SUPPORT_MAX[0] - x[inside, 0])
        return force

    # -------------------------------------------------------------- integrator
    def _step(self, dt: float) -> None:
        terms = self._force_terms(self.x, self.v, self.time)
        total = (
            terms["elastic"] + terms["gravity"] + terms["pressure"] + terms["contact"]
        )
        v_new = (self.v + dt * total * self.inv_mass[:, None]) / (1.0 + MASS_DRAG * dt)
        speed = np.linalg.norm(v_new, axis=1)
        excess = speed > MAX_SPEED
        if np.any(excess):
            v_new[excess] *= (MAX_SPEED / speed[excess])[:, None]
        v_new[self.fixed] = 0.0
        self.x = self.x + dt * v_new
        self.x[self.fixed] = self.X0[self.fixed]
        self.v = v_new
        self.time += dt
        self.steps += 1
        self._terms = terms

    def advance_to(self, time: float):
        target = float(time)
        while target - self.time > 1e-12:
            remaining = target - self.time
            count = max(1, int(np.ceil(remaining / self.dt_max - 1e-9)))
            dt = remaining / count
            for _ in range(count):
                self._step(dt)
            self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------ output
    def _deformation_gradients(self):
        P = self.x[self.tets]
        Ds = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=2)
        return np.ascontiguousarray(Ds @ self.Dm_inv)

    def _budget(self):
        terms = self._force_terms(self.x, self.v, self.time)
        self._terms = terms
        gravity = terms["gravity"].sum(axis=0)
        pressure = terms["pressure"].sum(axis=0)
        contact = terms["contact"].sum(axis=0)
        drag = terms["drag"].sum(axis=0)
        nodal_total = (
            terms["elastic"] + terms["gravity"] + terms["pressure"]
            + terms["contact"] + terms["drag"]
        )
        reaction = -nodal_total[self.fixed].sum(axis=0)
        forces = np.stack([gravity, pressure, contact, drag, reaction], axis=0)

        com = np.average(self.x, axis=0, weights=self.node_mass)
        base = self.x[self.fixed].mean(axis=0) if np.any(self.fixed) else com
        if len(self.pressure_quads):
            quad_centroid = self.x[self.pressure_quads].mean(axis=1).mean(axis=0)
        else:  # pragma: no cover
            quad_centroid = com
        positions = np.stack([com, quad_centroid, base, com, base], axis=0)
        return forces[None, :, :], positions[None, :, :]

    def _frame(self, time: float) -> StateFrame:
        position = np.einsum(
            "sk,skc->sc", self.embed_weights, self.x[self.embed_nodes]
        )
        velocity = np.einsum(
            "sk,skc->sc", self.embed_weights, self.v[self.embed_nodes]
        )
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        gradients = np.nan_to_num(
            self._deformation_gradients(), nan=0.0, posinf=0.0, neginf=0.0
        )
        forces, force_positions = self._budget()
        forces = np.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        force_positions = np.nan_to_num(
            force_positions, nan=0.0, posinf=0.0, neginf=0.0
        )

        frame = StateFrame(
            time=float(max(time, 0.0)),
            position=np.ascontiguousarray(position, dtype=float),
            velocity=np.ascontiguousarray(velocity, dtype=float),
            mass=self.sample_mass.copy(),
            attributes={
                "mass_quantity": self.sample_mass.copy(),
                "fixed_base": self.sample_fixed.copy(),
                "rest_position": self.sample_rest.copy(),
            },
            fields={
                "element_deformation_gradient": gradients,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "budget_external_force": forces,
                "budget_force_position": force_positions,
                "budget_domain_weights": np.ones((1, self.n_samples), dtype=float),
            },
            scalars={
                "reference_force": float(self.reference_force),
                "chamber_pressure_pa": float(self._pressure(self.time)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        tip = int(np.argmax(self.sample_rest[:, 0]))
        displacement = float(
            np.linalg.norm(
                np.einsum("k,kc->c", self.embed_weights[tip], self.x[self.embed_nodes[tip]])
                - self.sample_rest[tip]
            )
        )
        return {
            "method": "embedded_StVK_tetrahedral_FEM_explicit",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elements),
            "cavity_faces": int(len(self.pressure_quads)),
            "contact_pairs": int(len(self.contact_a)),
            "dt_max_s": float(self.dt_max),
            "steps": int(self.steps),
            "total_mass_kg": float(self.total_mass),
            "final_time_s": float(self.time),
            "tip_displacement_m": displacement,
        }
