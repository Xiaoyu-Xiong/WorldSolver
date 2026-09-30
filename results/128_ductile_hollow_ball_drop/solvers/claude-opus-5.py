"""Elastoplastic mass-spring shell solver for the ductile hollow ball drop and tear.

Physical model
--------------
The frozen hollow-ball mesh vertices are welded into unique dynamic nodes carrying
lumped mass. Nodes are connected by a distance-bond network built from the frozen
rest configuration with a cutoff that spans the shell thickness, so the network
supplies membrane *and* bending stiffness.

Each bond is elastoplastic (rate independent perfect plasticity in 1D): the elastic
strain is limited by a yield strain, and any excess flows into the bond rest length,
which is a permanent (plastic) change.  Accumulated plastic strain is tracked per
bond, and a bond fails (ductile tear) once that accumulation passes a failure
threshold while under tension.  This reproduces flattening, permanent buckling /
necking, and a tear that only opens after substantial deformation.

Contacts: the rigid floor (top face of the frozen ground box, z = 0) is enforced by
inelastic position/velocity projection with Coulomb friction.  Self contact of the
collapsing shell uses a penalty repulsion on non-bonded close pairs.

Time stepping: symplectic (semi-implicit) Euler with a fixed small substep, bond
Rayleigh damping and weak drag.  Everything is deterministic (no random numbers).
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is available in the runtime; a NumPy fallback keeps the solver robust.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None


def _pairs_within(points: np.ndarray, radius: float) -> np.ndarray:
    """Deterministic sorted list of unique index pairs closer than ``radius``."""
    if _KDTree is not None:
        tree = _KDTree(points)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    else:  # pragma: no cover - brute force fallback
        n = len(points)
        rows: list[np.ndarray] = []
        for i in range(n):
            d = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
            j = np.nonzero(d < radius)[0] + i + 1
            if len(j):
                rows.append(np.column_stack((np.full(len(j), i, np.int64), j)))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.concatenate(rows)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return np.ascontiguousarray(pairs[order])


class BenchmarkSolver:
    # ---- numerical / material parameters (frozen constants) ----
    SUBSTEP = 2.0e-4          # s, explicit substep
    BOND_CUTOFF = 0.170       # m, spans the 0.16 m shell thickness
    STIFFNESS = 6.0e4         # N/m per bond
    DAMPING_RATIO = 0.08      # bond Rayleigh damping ratio
    YIELD_STRAIN = 0.040      # elastic strain limit before plastic flow
    CREEP = 0.60              # plastic return-map relaxation per substep
    PLASTIC_LIMIT = 0.30      # accumulated plastic strain at ductile failure
    REST_MIN = 0.45           # rest-length clamps (relative to initial rest length)
    REST_MAX = 2.60
    DRAG = 0.15               # 1/s weak ambient drag
    SELF_DIST = 0.060         # m, self-contact activation distance
    SELF_STIFFNESS = 3.0e4    # N/m self-contact penalty
    SELF_REBUILD = 40         # substeps between self-contact neighbour rebuilds
    FRICTION = 0.60           # Coulomb friction with the floor
    FLOOR_Z = 0.0             # top of the frozen ground box
    CONTACT_BAND = 3.0        # near-contact reporting band, in contact length scales

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        frame = scene.initial_state()
        config = getattr(scene, "config", {}) or {}

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.n_raw = int(len(frame.position))
        self.raw_position0 = np.asarray(frame.position, dtype=float).copy()
        self.raw_mass = (
            np.asarray(frame.mass, dtype=float).copy()
            if frame.mass is not None
            else np.full(self.n_raw, 1.0)
        )
        self.body_id = (
            np.asarray(frame.body_id).astype(np.int32).copy()
            if frame.body_id is not None
            else np.zeros(self.n_raw, np.int32)
        )
        attributes = dict(frame.attributes)
        radius = attributes.get("particle_radius")
        self.particle_radius = (
            np.asarray(radius, dtype=float).copy()
            if radius is not None
            else np.full(self.n_raw, 0.05)
        )
        material = attributes.get("material_label")
        self.material_label = (
            np.asarray(material).astype(np.int32).copy()
            if material is not None
            else np.zeros(self.n_raw, np.int32)
        )
        scalars = dict(frame.scalars)
        self.contact_length_scale = float(scalars.get("contact_length_scale", 0.14))
        self.grid_spacing = float(scalars.get("grid_spacing", self.contact_length_scale))
        extent = float(np.ptp(self.raw_position0, axis=0).max())
        self.characteristic_length = float(
            scalars.get("characteristic_length", max(extent, self.contact_length_scale))
        )
        self.reference_velocity = float(scalars.get("reference_velocity", 1.0))

        # ---- weld coincident mesh vertices into unique dynamic nodes ----
        keys = np.round(self.raw_position0, 6)
        _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        self.inverse = np.asarray(inverse, dtype=np.int64).ravel()
        self.node_x0 = self.raw_position0[first_index].copy()
        self.n_node = int(len(self.node_x0))
        self.node_mass = np.bincount(
            self.inverse, weights=self.raw_mass, minlength=self.n_node
        ).astype(float)
        self.node_mass = np.maximum(self.node_mass, 1e-9)
        self.inv_mass = 1.0 / self.node_mass

        self.node_x = self.node_x0.copy()
        self.node_v = np.zeros_like(self.node_x)
        if frame.velocity is not None:
            raw_v = np.asarray(frame.velocity, dtype=float)
            acc = np.zeros((self.n_node, 3))
            for c in range(3):
                acc[:, c] = np.bincount(
                    self.inverse, weights=self.raw_mass * raw_v[:, c], minlength=self.n_node
                )
            self.node_v = acc * self.inv_mass[:, None]

        # ---- bond network from the frozen rest configuration ----
        pairs = _pairs_within(self.node_x0, self.BOND_CUTOFF)
        d0 = np.linalg.norm(self.node_x0[pairs[:, 1]] - self.node_x0[pairs[:, 0]], axis=1)
        keep = d0 > 1e-6
        pairs, d0 = pairs[keep], d0[keep]
        self.bond_i = np.ascontiguousarray(pairs[:, 0])
        self.bond_j = np.ascontiguousarray(pairs[:, 1])
        self.bond_len0 = d0.copy()          # initial (reference) rest length
        self.bond_rest = d0.copy()          # current, plastically evolving rest length
        self.bond_plastic = np.zeros(len(d0))
        self.bond_alive = np.ones(len(d0), dtype=bool)
        # per-bond stiffness scaled by rest length so the network behaves like a
        # continuum (force = k * (l - l0), stiffness ~ EA / l0)
        self.bond_k = self.STIFFNESS * (np.mean(d0) / d0)
        reduced = 1.0 / (self.inv_mass[self.bond_i] + self.inv_mass[self.bond_j])
        self.bond_c = 2.0 * self.DAMPING_RATIO * np.sqrt(self.bond_k * reduced)
        # sorted bond keys for fast self-contact exclusion
        self.bond_key = self.bond_i.astype(np.int64) * self.n_node + self.bond_j
        key_order = np.argsort(self.bond_key, kind="stable")
        self.bond_key_sorted = self.bond_key[key_order]
        self.bond_key_index = key_order

        # ---- least-squares operator for the per-node deformation gradient ----
        self._build_shape_operator()

        # self-contact bookkeeping
        self.self_i = np.zeros(0, dtype=np.int64)
        self.self_j = np.zeros(0, dtype=np.int64)
        self._steps_since_rebuild = self.SELF_REBUILD

        self.time = 0.0
        self.substeps = 0
        self.max_speed = 0.0
        self._rebuild_self_contacts()

    # ------------------------------------------------------------------
    def _build_shape_operator(self) -> None:
        i, j = self.bond_i, self.bond_j
        dX = self.node_x0[j] - self.node_x0[i]
        w = 1.0 / np.maximum(self.bond_len0, 1e-9) ** 2
        A = np.zeros((self.n_node, 3, 3))
        for a in range(3):
            for b in range(3):
                contrib = w * dX[:, a] * dX[:, b]
                A[:, a, b] = np.bincount(i, weights=contrib, minlength=self.n_node) + np.bincount(
                    j, weights=contrib, minlength=self.n_node
                )
        trace = np.trace(A, axis1=1, axis2=2)
        reg = np.maximum(trace, 1e-9)[:, None, None] * 1e-6
        A = A + reg * np.eye(3)[None]
        self.shape_inv = np.linalg.inv(A)
        self.shape_weight = w
        self.shape_dX = dX

    def _deformation_gradient(self) -> np.ndarray:
        i, j = self.bond_i, self.bond_j
        dx = self.node_x[j] - self.node_x[i]
        w = self.shape_weight
        dX = self.shape_dX
        B = np.zeros((self.n_node, 3, 3))
        for a in range(3):
            for b in range(3):
                contrib = w * dx[:, a] * dX[:, b]
                B[:, a, b] = np.bincount(i, weights=contrib, minlength=self.n_node) + np.bincount(
                    j, weights=contrib, minlength=self.n_node
                )
        F = B @ self.shape_inv
        F = np.where(np.isfinite(F), F, np.eye(3)[None])
        return np.clip(F, -50.0, 50.0)

    # ------------------------------------------------------------------
    def _rebuild_self_contacts(self) -> None:
        pairs = _pairs_within(self.node_x, self.SELF_DIST)
        if len(pairs) == 0:
            self.self_i = np.zeros(0, dtype=np.int64)
            self.self_j = np.zeros(0, dtype=np.int64)
            self._steps_since_rebuild = 0
            return
        keys = pairs[:, 0].astype(np.int64) * self.n_node + pairs[:, 1]
        pos = np.searchsorted(self.bond_key_sorted, keys)
        pos_clipped = np.clip(pos, 0, len(self.bond_key_sorted) - 1)
        is_bond = self.bond_key_sorted[pos_clipped] == keys
        # Exclude every pair that was bonded in the rest configuration, including
        # pairs whose bond has since torn.  Those neighbours start out closer than
        # SELF_DIST, so activating a penalty on them would inject a spurious
        # impulse ("pop") at the instant of failure.
        keep = ~is_bond
        self.self_i = np.ascontiguousarray(pairs[keep, 0])
        self.self_j = np.ascontiguousarray(pairs[keep, 1])
        self._steps_since_rebuild = 0

    # ------------------------------------------------------------------
    def _accumulate(self, force: np.ndarray, idx_i, idx_j, f_vec: np.ndarray) -> None:
        n = self.n_node
        for c in range(3):
            force[:, c] += np.bincount(idx_i, weights=-f_vec[:, c], minlength=n)
            force[:, c] += np.bincount(idx_j, weights=f_vec[:, c], minlength=n)

    def _step(self, dt: float) -> None:
        x, v = self.node_x, self.node_v
        force = self.node_mass[:, None] * self.gravity[None, :]

        alive = self.bond_alive
        if np.any(alive):
            i = self.bond_i[alive]
            j = self.bond_j[alive]
            d = x[j] - x[i]
            length = np.linalg.norm(d, axis=1)
            length = np.maximum(length, 1e-9)
            axis = d / length[:, None]
            rest = self.bond_rest[alive]
            k = self.bond_k[alive]
            c = self.bond_c[alive]
            dv = np.einsum("ij,ij->i", v[j] - v[i], axis)
            magnitude = k * (length - rest) + c * dv
            # f_vec acts on node j along -axis and on node i along +axis
            f_vec = magnitude[:, None] * axis
            n = self.n_node
            for comp in range(3):
                force[:, comp] += np.bincount(i, weights=f_vec[:, comp], minlength=n)
                force[:, comp] -= np.bincount(j, weights=f_vec[:, comp], minlength=n)

        # self contact repulsion (non-bonded close pairs)
        if len(self.self_i):
            si, sj = self.self_i, self.self_j
            d = x[sj] - x[si]
            dist = np.linalg.norm(d, axis=1)
            active = dist < self.SELF_DIST
            if np.any(active):
                si, sj = si[active], sj[active]
                dist = np.maximum(dist[active], 1e-6)
                axis = d[active] / dist[:, None]
                penetration = self.SELF_DIST - dist
                dv = np.einsum("ij,ij->i", v[sj] - v[si], axis)
                magnitude = self.SELF_STIFFNESS * penetration - 0.2 * np.sqrt(
                    self.SELF_STIFFNESS * np.mean(self.node_mass)
                ) * np.minimum(dv, 0.0)
                magnitude = np.maximum(magnitude, 0.0)
                f_vec = magnitude[:, None] * axis  # pushes j along +axis, i along -axis
                n = self.n_node
                for comp in range(3):
                    force[:, comp] += np.bincount(sj, weights=f_vec[:, comp], minlength=n)
                    force[:, comp] -= np.bincount(si, weights=f_vec[:, comp], minlength=n)

        # symplectic Euler with weak drag
        v += dt * force * self.inv_mass[:, None]
        v *= max(0.0, 1.0 - self.DRAG * dt)
        x += dt * v

        # rigid floor: inelastic projection with Coulomb friction
        below = x[:, 2] < self.FLOOR_Z
        if np.any(below):
            vz = v[below, 2]
            impact = np.minimum(vz, 0.0)
            x[below, 2] = self.FLOOR_Z
            v[below, 2] = np.maximum(vz, 0.0)
            vt = v[below, :2]
            speed = np.linalg.norm(vt, axis=1)
            limit = self.FRICTION * np.abs(impact)
            scale = np.where(speed > 1e-9, np.maximum(speed - limit, 0.0) / np.maximum(speed, 1e-9), 0.0)
            v[below, :2] = vt * scale[:, None]

        # plastic flow and ductile failure on the bonds
        if np.any(self.bond_alive):
            idx = np.nonzero(self.bond_alive)[0]
            i = self.bond_i[idx]
            j = self.bond_j[idx]
            length = np.linalg.norm(x[j] - x[i], axis=1)
            rest = self.bond_rest[idx]
            strain = (length - rest) / np.maximum(rest, 1e-9)
            ys = self.YIELD_STRAIN
            target = rest.copy()
            tension = strain > ys
            compression = strain < -ys
            if np.any(tension):
                target[tension] = length[tension] / (1.0 + ys)
            if np.any(compression):
                target[compression] = length[compression] / (1.0 - ys)
            new_rest = rest + self.CREEP * (target - rest)
            lo = self.REST_MIN * self.bond_len0[idx]
            hi = self.REST_MAX * self.bond_len0[idx]
            new_rest = np.clip(new_rest, lo, hi)
            increment = np.abs(new_rest - rest) / np.maximum(self.bond_len0[idx], 1e-9)
            self.bond_rest[idx] = new_rest
            self.bond_plastic[idx] += increment
            fail = (self.bond_plastic[idx] > self.PLASTIC_LIMIT) & (strain > 0.5 * ys)
            if np.any(fail):
                broken = idx[fail]
                self.bond_alive[broken] = False
                self._steps_since_rebuild = self.SELF_REBUILD

        # keep the state bounded and finite
        np.clip(v, -200.0, 200.0, out=v)
        if not np.isfinite(x).all():
            x[:] = np.where(np.isfinite(x), x, self.node_x0)
        if not np.isfinite(v).all():
            v[:] = np.where(np.isfinite(v), v, 0.0)

        self.substeps += 1
        self._steps_since_rebuild += 1
        if self._steps_since_rebuild >= self.SELF_REBUILD:
            self._rebuild_self_contacts()
        speed = float(np.abs(v).max()) if len(v) else 0.0
        self.max_speed = max(self.max_speed, speed)

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        remaining = target - self.time
        if remaining > 1e-12:
            count = max(1, int(np.ceil(remaining / self.SUBSTEP)))
            h = remaining / count
            for _ in range(count):
                self._step(h)
            self.time = target
        else:
            self.time = max(self.time, target)
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        position = self.node_x[self.inverse]
        velocity = self.node_v[self.inverse]
        F = self._deformation_gradient()[self.inverse]

        external_force = self.raw_mass[:, None] * self.gravity[None, :]

        gap = position[:, 2] - self.FLOOR_Z
        band = self.CONTACT_BAND * self.contact_length_scale
        near = np.nonzero(gap < band)[0]
        if len(near) == 0:
            near = np.asarray([int(np.argmin(gap))])
        contact_signed_distance = np.ascontiguousarray(gap[near])

        frame = StateFrame(
            float(time),
            np.ascontiguousarray(position),
            np.ascontiguousarray(velocity),
            mass=self.raw_mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.raw_mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": np.ascontiguousarray(F),
            },
            fields={
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": np.zeros(1),
                "external_force": np.ascontiguousarray(external_force),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "grid_spacing": self.grid_spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        alive = int(np.count_nonzero(self.bond_alive))
        total = int(len(self.bond_alive))
        return {
            "method": "elastoplastic mass-spring shell, symplectic Euler",
            "nodes": int(self.n_node),
            "samples": int(self.n_raw),
            "bonds_total": total,
            "bonds_broken": total - alive,
            "substeps": int(self.substeps),
            "substep_dt": float(self.SUBSTEP),
            "max_speed": float(self.max_speed),
            "mean_plastic_strain": float(np.mean(self.bond_plastic)),
            "final_time": float(self.time),
        }
