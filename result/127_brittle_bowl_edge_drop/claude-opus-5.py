"""Brittle bowl edge drop solver.

Physical model
--------------
The bowl surface samples are treated as a bonded particle (peridynamic /
lattice-spring) discretisation of a thin brittle shell:

* every pair of samples inside a horizon (plus the mesh edges) carries an
  axial elastic bond with a dashpot,
* a bond is irreversibly severed when its axial strain exceeds a
  heterogeneous critical strain (brittle failure) -> crack networks and
  fragments emerge from the dynamics, never from a script,
* severed pairs and non-bonded pairs interact through a penalty
  contact + friction law so fragments collide and settle,
* the frozen rigid floor (box top at z = 0) acts through a
  Hertz-like penalty normal force with Coulomb friction,
* gravity from the frozen configuration is the only external force.

Time integration is an explicit symplectic (semi-implicit) Euler scheme with
a fixed sub-step chosen well inside the elastic CFL limit; sample frames are
emitted by stepping up to each requested sample time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # SciPy is optional; a NumPy fallback keeps the solver runnable.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None

from worldsolver.contracts import StateFrame


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pairs_within(points: np.ndarray, radius: float) -> np.ndarray:
    """All unique index pairs closer than ``radius`` (deterministic order)."""
    if _KDTree is not None:
        pairs = _KDTree(points).query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    else:  # brute force fallback
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        iu, ju = np.triu_indices(len(points), k=1)
        mask = dist[iu, ju] < radius
        pairs = np.stack([iu[mask], ju[mask]], axis=1).astype(np.int64)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _encode(pairs: np.ndarray, n: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * np.int64(n) + pairs[:, 1].astype(np.int64)


class BenchmarkSolver:
    """Bonded-particle brittle shell solver (no external physics engine)."""

    # --- numerical / material constants (frozen at initialize) --------------
    SUBSTEP = 1.0e-4          # s, explicit sub-step
    HORIZON = 0.17            # m, bond horizon
    WEIGHT_LENGTH = 0.15      # m, bond weight decay length
    PARTICLE_STIFFNESS = 1.0e8  # N/m, per-sample summed bond stiffness
    BOND_DAMPING_RATIO = 0.07
    STRAIN_REF_MIN = 0.05     # m, floor on the bond gauge length
    CRITICAL_STRAIN = 0.0025  # brittle tensile failure strain
    COMPRESSIVE_FACTOR = 3.0  # crushing is harder than tensile failure
    FLOOR_STIFFNESS = 8.0e5   # N/m
    FLOOR_DAMPING_RATIO = 0.30
    FLOOR_FRICTION = 0.45
    SELF_STIFFNESS = 4.0e5    # N/m fragment-fragment penalty
    SELF_DAMPING_RATIO = 0.25
    SELF_FRICTION = 0.35
    SELF_CONTACT_GAP = 0.11   # m, contact distance for unbonded pairs
    NEIGHBOR_REBUILD = 20     # sub-steps between contact-list rebuilds
    LINEAR_DRAG = 0.02        # 1/s, mild ambient dissipation
    MAX_SPEED = 40.0          # m/s safety clamp

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(scene.config)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.reference = self.position.copy()
        self.n = int(len(self.position))

        self.mass = np.array(frame0.mass, dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.material_label = np.array(
            frame0.attributes.get("material_label", np.zeros(self.n)), dtype=np.int32
        )
        self.particle_radius = np.array(
            frame0.attributes.get(
                "particle_radius",
                np.full(self.n, 0.45 * float(self.config["sampling"]["spacing"])),
            ),
            dtype=float,
        )
        self.spacing = float(self.config["sampling"]["spacing"])
        self.base_scalars = dict(frame0.scalars)
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # frozen floor geometry (rigid box; its top face is the contact plane)
        self.floor_top = 0.0
        self.floor_half = (3.2, 1.4)
        for item in self.config.get("scene_geometry", {}).get("objects", []) or []:
            if item.get("id") == "ground":
                half = item.get("geometry", {}).get("half_extents", [3.2, 1.4, 0.05])
                trans = item.get("transform", {}).get("translation", [0.0, 0.0, -0.05])
                self.floor_top = float(trans[2]) + float(half[2])
                self.floor_half = (float(half[0]), float(half[1]))
        # contact offset of a surface sample against the floor
        self.floor_offset = float(np.minimum(self.particle_radius, 0.045).mean())

        self._build_bonds()

        self.time = 0.0
        self.dt = float(self.SUBSTEP)
        self.step_count = 0
        self.broken_count = 0
        self.contact_pairs = np.zeros((0, 2), dtype=np.int64)
        self.contact_gap0 = np.zeros(0)
        self._refresh_contacts()

        self._deformation = np.repeat(np.eye(3)[None], self.n, axis=0)
        self._frame_emitted = False

    # ------------------------------------------------------------- topology
    def _mesh_edges(self) -> np.ndarray:
        try:
            path = Path(__file__).resolve().parent.parent / "assets" / "bowl_topology.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            faces = np.asarray(payload.get("faces", []), dtype=np.int64)
        except Exception:
            return np.zeros((0, 2), dtype=np.int64)
        if faces.size == 0 or faces.shape[-1] != 3:
            return np.zeros((0, 2), dtype=np.int64)
        edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        edges = np.sort(edges, axis=1)
        edges = edges[(edges[:, 0] < self.n) & (edges[:, 1] < self.n)]
        return np.unique(edges, axis=0)

    def _build_bonds(self) -> None:
        X0 = self.reference
        pairs = _pairs_within(X0, self.HORIZON)
        extra = self._mesh_edges()
        if len(extra):
            keys = set(_encode(pairs, self.n).tolist())
            keep = np.array(
                [k not in keys for k in _encode(extra, self.n).tolist()], dtype=bool
            )
            if keep.any():
                pairs = np.concatenate([pairs, extra[keep]], axis=0)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        if len(pairs) == 0:  # degenerate safeguard
            pairs = np.zeros((0, 2), dtype=np.int64)

        self.bond_i = pairs[:, 0].astype(np.int64)
        self.bond_j = pairs[:, 1].astype(np.int64)
        d = X0[self.bond_j] - X0[self.bond_i]
        self.bond_ref_vec = d
        self.bond_length0 = np.linalg.norm(d, axis=1)
        self.bond_length0 = np.maximum(self.bond_length0, 1e-6)
        self.bond_gauge = np.maximum(self.bond_length0, self.STRAIN_REF_MIN)

        w = np.exp(-((self.bond_length0 / self.WEIGHT_LENGTH) ** 2))
        self.bond_weight = w
        node_w = np.zeros(self.n)
        np.add.at(node_w, self.bond_i, w)
        np.add.at(node_w, self.bond_j, w)
        node_w = np.maximum(node_w, 1e-9)
        norm = 0.5 * (node_w[self.bond_i] + node_w[self.bond_j])
        self.bond_stiffness = self.PARTICLE_STIFFNESS * w / norm

        m_eff = 0.5 * (self.mass[self.bond_i] + self.mass[self.bond_j])
        self.bond_damping = (
            2.0 * self.BOND_DAMPING_RATIO * np.sqrt(self.bond_stiffness * m_eff)
        )

        # heterogeneous brittle strength (seeded -> deterministic)
        jitter = self.rng.uniform(0.75, 1.35, size=len(self.bond_i))
        self.bond_strain_limit = self.CRITICAL_STRAIN * jitter
        self.bond_intact = np.ones(len(self.bond_i), dtype=bool)

        # reference shape tensors for the deformation gradient
        R = self.bond_ref_vec
        outer = R[:, :, None] * R[:, None, :]
        self.bond_ref_outer = outer

    # ------------------------------------------------------- contact lists
    def _refresh_contacts(self) -> None:
        radius = max(self.SELF_CONTACT_GAP * 1.4, 0.16)
        pairs = _pairs_within(self.position, radius)
        if len(pairs) == 0:
            self.contact_pairs = np.zeros((0, 2), dtype=np.int64)
            self.contact_gap0 = np.zeros(0)
            return
        intact = np.stack(
            [self.bond_i[self.bond_intact], self.bond_j[self.bond_intact]], axis=1
        )
        if len(intact):
            bonded_keys = _encode(intact, self.n)
            mask = ~np.isin(_encode(pairs, self.n), bonded_keys)
            pairs = pairs[mask]
        self.contact_pairs = pairs
        if len(pairs) == 0:
            self.contact_gap0 = np.zeros(0)
            return
        d0 = np.linalg.norm(
            self.reference[pairs[:, 1]] - self.reference[pairs[:, 0]], axis=1
        )
        # never push apart pairs that were already close in the intact bowl
        self.contact_gap0 = np.minimum(self.SELF_CONTACT_GAP, 0.9 * d0)

    # ------------------------------------------------------------- forces
    def _bond_forces(self, x: np.ndarray, v: np.ndarray, force: np.ndarray) -> None:
        live = np.flatnonzero(self.bond_intact)
        if len(live) == 0:
            return
        i = self.bond_i[live]
        j = self.bond_j[live]
        d = x[j] - x[i]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-9)
        nvec = d / length[:, None]
        stretch = length - self.bond_length0[live]
        strain = stretch / self.bond_gauge[live]

        rel = v[j] - v[i]
        rate = np.einsum("ij,ij->i", rel, nvec)
        magnitude = self.bond_stiffness[live] * stretch + self.bond_damping[live] * rate
        f = magnitude[:, None] * nvec
        np.add.at(force, i, f)
        np.add.at(force, j, -f)

        limit = self.bond_strain_limit[live]
        failed = (strain > limit) | (strain < -self.COMPRESSIVE_FACTOR * limit)
        if failed.any():
            self.bond_intact[live[failed]] = False
            self.broken_count += int(failed.sum())

    def _self_contact(self, x: np.ndarray, v: np.ndarray, force: np.ndarray) -> None:
        pairs = self.contact_pairs
        if len(pairs) == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = x[j] - x[i]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        active = length < self.contact_gap0
        if not np.any(active):
            return
        i = i[active]
        j = j[active]
        d = d[active]
        length = np.maximum(length[active], 1e-9)
        nvec = d / length[:, None]
        pen = self.contact_gap0[active] - length
        m_eff = 0.5 * (self.mass[i] + self.mass[j])
        damp = 2.0 * self.SELF_DAMPING_RATIO * np.sqrt(self.SELF_STIFFNESS * m_eff)
        rel = v[j] - v[i]
        rate = np.einsum("ij,ij->i", rel, nvec)
        fn = self.SELF_STIFFNESS * pen - damp * rate
        fn = np.maximum(fn, 0.0)
        f = -fn[:, None] * nvec  # pushes i back, j forward
        np.add.at(force, i, f)
        np.add.at(force, j, -f)

        # Coulomb friction on the tangential relative velocity
        vt = rel - rate[:, None] * nvec
        speed = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        moving = speed > 1e-8
        if np.any(moving):
            ft = np.zeros_like(vt)
            mu_fn = self.SELF_FRICTION * fn[moving]
            visc = damp[moving] * speed[moving]
            mag = np.minimum(mu_fn, visc)
            ft[moving] = (mag / speed[moving])[:, None] * vt[moving]
            np.add.at(force, i, ft)
            np.add.at(force, j, -ft)

    def _floor_contact(self, x: np.ndarray, v: np.ndarray, force: np.ndarray) -> None:
        gap = x[:, 2] - self.floor_offset - self.floor_top
        inside = (np.abs(x[:, 0]) <= self.floor_half[0]) & (
            np.abs(x[:, 1]) <= self.floor_half[1]
        )
        active = (gap < 0.0) & inside
        if not np.any(active):
            return
        idx = np.flatnonzero(active)
        pen = -gap[idx]
        m = self.mass[idx]
        damp = 2.0 * self.FLOOR_DAMPING_RATIO * np.sqrt(self.FLOOR_STIFFNESS * m)
        vz = v[idx, 2]
        fn = self.FLOOR_STIFFNESS * pen - damp * vz
        fn = np.maximum(fn, 0.0)
        force[idx, 2] += fn

        vt = v[idx, :2]
        speed = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        moving = speed > 1e-8
        if np.any(moving):
            sel = idx[moving]
            mag = np.minimum(self.FLOOR_FRICTION * fn[moving], damp[moving] * speed[moving])
            force[sel, 0] -= mag * vt[moving, 0] / speed[moving]
            force[sel, 1] -= mag * vt[moving, 1] / speed[moving]

    def _forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :]
        force = np.array(force, dtype=float)
        self._bond_forces(x, v, force)
        self._self_contact(x, v, force)
        self._floor_contact(x, v, force)
        force -= (self.LINEAR_DRAG * self.mass)[:, None] * v
        return force

    # --------------------------------------------------------------- stepping
    def _step(self) -> None:
        dt = self.dt
        x = self.position
        v = self.velocity
        force = self._forces(x, v)
        v += (force * self.inv_mass[:, None]) * dt
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        hot = speed > self.MAX_SPEED
        if np.any(hot):
            v[hot] *= (self.MAX_SPEED / speed[hot])[:, None]
        x += v * dt
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.step_count += 1
        if self.step_count % self.NEIGHBOR_REBUILD == 0:
            self._refresh_contacts()

    # ------------------------------------------------ deformation gradient
    def _compute_deformation_gradient(self) -> np.ndarray:
        A = np.zeros((self.n, 3, 3))
        B = np.zeros((self.n, 3, 3))
        live = np.flatnonzero(self.bond_intact)
        if len(live):
            i = self.bond_i[live]
            j = self.bond_j[live]
            w = self.bond_weight[live]
            r = self.position[j] - self.position[i]
            R = self.bond_ref_vec[live]
            a = w[:, None, None] * (r[:, :, None] * R[:, None, :])
            b = w[:, None, None] * self.bond_ref_outer[live]
            np.add.at(A, i, a)
            np.add.at(A, j, a)
            np.add.at(B, i, b)
            np.add.at(B, j, b)
        trace = np.trace(B, axis1=1, axis2=2)
        scale = np.maximum(trace, 1e-12) * 1e-6 + 1e-12
        reg = B + scale[:, None, None] * np.eye(3)[None]
        F = np.repeat(np.eye(3)[None], self.n, axis=0)
        ok = trace > 1e-9
        if np.any(ok):
            try:
                solved = np.linalg.solve(
                    np.transpose(reg[ok], (0, 2, 1)), np.transpose(A[ok], (0, 2, 1))
                )
                F[ok] = np.transpose(solved, (0, 2, 1))
            except np.linalg.LinAlgError:
                pass
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F = np.clip(F, -50.0, 50.0)
        bad = ~np.isfinite(F).all(axis=(1, 2))
        if np.any(bad):
            F[bad] = np.eye(3)
        return F

    # ------------------------------------------------------------- contacts
    def _contact_signed_distance(self) -> np.ndarray:
        gap = self.position[:, 2] - self.floor_offset - self.floor_top
        band = gap < 5.0 * self.floor_offset + 0.05
        if not np.any(band):
            band = np.zeros(self.n, dtype=bool)
            band[int(np.argmin(gap))] = True
        values = gap[band]
        return np.ascontiguousarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))

    # ---------------------------------------------------------------- frame
    def _frame(self, time: float) -> StateFrame:
        F = self._compute_deformation_gradient()
        self._deformation = F
        external = self.mass[:, None] * self.gravity[None, :]
        scalars = dict(self.base_scalars)
        scalars["contact_length_scale"] = float(self.spacing)
        scalars.setdefault("grid_spacing", float(self.spacing))
        scalars["broken_bond_fraction"] = float(
            1.0 - (self.bond_intact.sum() / max(len(self.bond_intact), 1))
        )
        frame = StateFrame(
            float(max(time, 0.0)),
            np.ascontiguousarray(self.position.copy()),
            np.ascontiguousarray(self.velocity.copy()),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": np.ascontiguousarray(F),
            },
            fields={
                "external_force": np.ascontiguousarray(external),
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        guard = 0
        max_steps = int(abs(target - self.time) / self.dt) + 8
        while self.time < target - 0.5 * self.dt and guard < max_steps:
            self._step()
            self.time += self.dt
            guard += 1
        if target > 0.0:
            self.time = target
        self._frame_emitted = True
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        intact = int(self.bond_intact.sum())
        total = int(len(self.bond_intact))
        return {
            "method": "bonded-particle brittle shell, symplectic Euler",
            "samples": self.n,
            "bonds_total": total,
            "bonds_intact": intact,
            "bonds_broken": total - intact,
            "substep": self.dt,
            "substeps_taken": int(self.step_count),
            "final_time": float(self.time),
            "max_speed": float(np.abs(self.velocity).max()) if self.n else 0.0,
            "min_height": float(self.position[:, 2].min()) if self.n else 0.0,
        }
