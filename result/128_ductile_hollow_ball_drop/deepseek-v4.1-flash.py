"""Ductile hollow-shell drop, plastic flattening, buckling and tearing.

Agent-authored physically based solver for scene
``128_ductile_hollow_ball_drop``.

Physical model
--------------
The frozen ``subject`` geometry is a sampled hollow shell: two concentric
material surfaces (outer radius 0.64 m, inner radius 0.48 m) triangulated as a
closed shell, plus interior sampled material points that lie on the same
surfaces.  The shell is simulated with a Lagrangian mass/spring
finite-element-style discretisation in material coordinates:

* per-edge axial springs with one-dimensional elasto-plasticity
  (radial-return / linear isotropic hardening) so that deformation is
  permanent and forces stay bounded by the yield surface;
* per-edge bending (dihedral cross) springs that resist folding, so the shell
  buckles instead of crumpling freely;
* ductile failure by element deletion once the accumulated plastic strain
  exceeds the material failure strain -- this is what opens the tear after
  large deformation;
* damped penalty contact against the frozen rigid floor box with Coulomb
  friction, plus short-range self contact that keeps the two shell surfaces
  from interpenetrating (the shell thickness is preserved by that contact);
* the interior sampled material points are kinematically (affinely) attached
  to their nearest shell element, so all N samples belong to one continuous
  body and no sample is left free floating.

Time integration is explicit symplectic (kick/drift) Euler with a fixed
stable substep; the last substep of every sampling interval is shortened so
the state is produced exactly at the requested sample times.  The scheme is
deterministic and every exported quantity is finite by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, 1e-12)


def _scatter(index: np.ndarray, vec: np.ndarray, count: int) -> np.ndarray:
    """Accumulate per-particle vector contributions with np.bincount."""
    out = np.empty((count, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.bincount(index, weights=vec[:, axis], minlength=count)
    return out


class _PlasticState:
    """Plastic bookkeeping for one family of axial springs."""

    __slots__ = ("ep", "acc", "alive", "yield_strain", "hardening", "failure")

    def __init__(self, size: int, yield_strain: float, hardening: float, failure: float):
        self.ep = np.zeros(size, dtype=float)
        self.acc = np.zeros(size, dtype=float)
        self.alive = np.ones(size, dtype=bool)
        self.yield_strain = float(yield_strain)
        self.hardening = float(hardening)
        self.failure = float(failure)


class BenchmarkSolver:
    """Runnable ductile shell drop solver (state contract implementation)."""

    # --- material parameters (ductile metal-like shell) -------------------
    E_MODULUS = 3.0e7          # Pa (effective in-plane stiffness modulus)
    SHELL_THICKNESS = 0.16     # m, frozen geometry inner/outer radius gap
    YIELD_STRAIN = 0.012       # onset of plastic flow
    HARDENING = 0.06           # linear isotropic hardening modulus
    FAILURE_STRAIN = 0.30      # accumulated plastic strain at ductile failure
    BEND_STIFFNESS_RATIO = 0.02
    BEND_YIELD_STRAIN = 0.05
    BEND_HARDENING = 0.02
    BEND_FAILURE_STRAIN = 0.60
    DAMPING_RATIO = 0.10

    # --- contact parameters ----------------------------------------------
    SELF_DISTANCE = 0.10       # m, shell self-contact range
    SELF_STIFFNESS = 5.0e5     # N/m per contact pair
    SELF_PEN_CAP = 0.01        # m, cap on penalty penetration
    FLOOR_STIFFNESS = 1.0e6    # N/m per particle against the rigid floor
    FLOOR_PEN_CAP = 0.02       # m
    FLOOR_MU = 0.45
    FLOOR_V_EPS = 0.05         # m/s regularisation of Coulomb friction

    # --- integration ------------------------------------------------------
    DT = 3.0e-4                # s, explicit substep
    CONTACT_EVERY = 8          # refresh self-contact pairs every k substeps
    GLOBAL_DAMPING = 0.3       # 1/s bulk velocity damping (dissipative only)
    MAX_SPEED = 40.0           # m/s numerical safety clamp

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        sampling = dict(config.get("sampling", {}) or {})
        self.spacing = float(sampling.get("spacing", 0.14))
        self.gravity = np.asarray(
            config.get("gravity", (0.0, 0.0, -9.81)), dtype=float
        ).reshape(3)

        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=float).copy()
        if self.x.ndim != 2 or self.x.shape[1] != 3:
            raise ValueError("scene initial positions must be [N,3]")
        self.N = int(self.x.shape[0])
        self.v = np.zeros_like(self.x)
        if frame.mass is None:
            raise ValueError("scene must provide per-sample mass")
        self.m = np.maximum(np.array(frame.mass, dtype=float).reshape(-1), 1e-9)
        if self.m.shape[0] != self.N:
            raise ValueError("scene mass must have N entries")
        self.body_id = np.zeros(self.N, dtype=np.int32)
        self.time = 0.0

        # frozen reference (material) configuration
        self.x_ref = self.x.copy()
        self.material_radius = float(
            np.max(np.linalg.norm(self.x_ref - self.x_ref.mean(axis=0), axis=1))
        )
        self.r_particle = 0.25 * self.spacing
        self.r_floor = self.r_particle

        self._assemble()

        # integration scratch
        self.m_inv = 1.0 / self.m
        self._contact_tick = 0
        self._refresh_self_contact()

    # ------------------------------------------------------------------
    # assembly of the discrete shell
    # ------------------------------------------------------------------
    def _load_faces(self) -> np.ndarray:
        base = Path(__file__).resolve().parent.parent
        candidates: list[Path] = []
        geometry = dict(getattr(self.scene, "config", {}) or {}).get("scene_geometry", {})
        for item in (geometry or {}).get("objects", []) or []:
            geom = item.get("geometry", {}) if isinstance(item, dict) else {}
            value = geom.get("topology_asset") or geom.get("asset")
            if value:
                candidates.append((base / value).resolve())
        candidates.append(base / "assets" / "hollow_ball_topology.json")
        candidates.append(base / "assets" / "hollow_ball.obj")
        faces = None
        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.casefold() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if "faces" in data:
                    faces = np.asarray(data["faces"], dtype=np.int64)
                    break
            elif path.suffix.casefold() == ".obj":
                corners: list[list[int]] = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] == "f":
                        corners.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
                if corners:
                    faces = np.asarray(corners, dtype=np.int64)
                    break
        if faces is None:
            raise FileNotFoundError("cannot locate shell topology asset")
        faces = faces.reshape(-1, 3)
        faces = faces[np.all((faces >= 0) & (faces < self.N), axis=1)]
        if len(faces) == 0:
            raise ValueError("shell topology has no faces inside the sample set")
        return faces

    def _assemble(self) -> None:
        n = self.N
        faces = self._load_faces()
        ref = self.x_ref
        tri = ref[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        keep = area > 1e-9
        self.faces = faces[keep]
        self.faces_area_ref = area[keep]
        faces = self.faces

        # ---- axial (stretch/shear) springs on the shell edges -----------
        half = np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
        )
        opposite = np.concatenate((faces[:, 2], faces[:, 0], faces[:, 1]))
        key = np.minimum(half[:, 0], half[:, 1]).astype(np.int64) * n + np.maximum(
            half[:, 0], half[:, 1]
        )
        _, first = np.unique(key, return_index=True)
        edges = half[first]
        length0 = np.linalg.norm(ref[edges[:, 1]] - ref[edges[:, 0]], axis=1)
        valid = length0 > 1.0e-6
        self.edges = edges[valid]
        self.edge_length0 = length0[valid]
        stiffness = self.E_MODULUS * self.SHELL_THICKNESS
        self.edge_k = stiffness
        self.edge_c = (
            2.0
            * self.DAMPING_RATIO
            * np.sqrt(stiffness * 0.5 * float(np.median(self.m)))
        )
        self.edge_state = _PlasticState(
            len(self.edges), self.YIELD_STRAIN, self.HARDENING, self.FAILURE_STRAIN
        )

        # ---- bending (dihedral cross) springs ---------------------------
        order = np.argsort(key, kind="stable")
        ordered = key[order]
        dup = np.nonzero(ordered[1:] == ordered[:-1])[0]
        if len(dup):
            left = order[dup]
            right = order[dup + 1]
            pairs = np.stack((opposite[left], opposite[right]), axis=1)
            blen = np.linalg.norm(ref[pairs[:, 1]] - ref[pairs[:, 0]], axis=1)
            ok = (pairs[:, 0] != pairs[:, 1]) & (blen > 1.0e-6)
            pairs = pairs[ok]
            blen = blen[ok]
        else:
            pairs = np.zeros((0, 2), dtype=np.int64)
            blen = np.zeros(0)
        self.bend_pairs = pairs
        self.bend_length0 = blen
        bend_k = stiffness * self.BEND_STIFFNESS_RATIO
        self.bend_k = bend_k
        self.bend_c = (
            2.0 * self.DAMPING_RATIO * np.sqrt(bend_k * 0.5 * float(np.median(self.m)))
        )
        self.bend_state = _PlasticState(
            len(pairs),
            self.BEND_YIELD_STRAIN,
            self.BEND_HARDENING,
            self.BEND_FAILURE_STRAIN,
        )

        # ---- self contact bookkeeping -----------------------------------
        self.self_c = (
            2.0 * self.DAMPING_RATIO * np.sqrt(self.SELF_STIFFNESS * 0.5 * float(np.median(self.m)))
        )
        self.floor_c = (
            2.0
            * self.DAMPING_RATIO
            * np.sqrt(self.FLOOR_STIFFNESS * 0.5 * float(np.median(self.m)))
        )
        exclude = self._near_reference_pairs(ref, max(self.SELF_DISTANCE * 1.3, 0.13))
        if len(pairs):
            bkey = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64) * n + np.maximum(
                pairs[:, 0], pairs[:, 1]
            )
            exclude = np.unique(np.concatenate((exclude, bkey)))
        self.exclude_keys = np.sort(np.unique(exclude))
        self.pair_i = np.zeros(0, dtype=np.int64)
        self.pair_j = np.zeros(0, dtype=np.int64)

        # ---- kinematic attachment of interior sampled material points ---
        used = np.zeros(n, dtype=bool)
        used[faces.ravel()] = True
        self.tracer_idx = np.nonzero(~used)[0]
        self.tracer_tri = np.zeros(0, dtype=np.int64)
        self.tracer_bary = np.zeros((0, 3))
        self.tracer_off = np.zeros((0, 3))
        if len(self.tracer_idx):
            centroids = ref[faces].mean(axis=1)
            from scipy.spatial import cKDTree

            tree = cKDTree(centroids)
            k = 6 if len(centroids) >= 6 else len(centroids)
            _, neighbours = tree.query(ref[self.tracer_idx], k=k)
            neighbours = np.atleast_2d(neighbours.astype(np.int64)).reshape(
                len(self.tracer_idx), -1
            )
            best_d = np.full(len(self.tracer_idx), np.inf)
            best_tri = neighbours[:, 0].copy()
            best_bary = np.zeros((len(self.tracer_idx), 3))
            for column in range(neighbours.shape[1]):
                tri_idx = neighbours[:, column]
                bary, dist = self._barycentric(ref[self.tracer_idx], tri_idx)
                better = dist < best_d
                best_d[better] = dist[better]
                best_tri[better] = tri_idx[better]
                best_bary[better] = bary[better]
            self.tracer_tri = best_tri
            self.tracer_bary = best_bary
            attached = np.einsum("mk,mkj->mj", best_bary, ref[faces[best_tri]])
            frame0 = self._triangle_frames(best_tri, ref)
            self.tracer_frame0 = frame0
            offset = self.x_ref[self.tracer_idx] - attached
            self.tracer_off = np.einsum("mji,mj->mi", frame0, offset)

    def _near_reference_pairs(self, ref: np.ndarray, radius: float) -> np.ndarray:
        from scipy.spatial import cKDTree

        pairs = cKDTree(ref).query_pairs(r=radius, output_type="ndarray")
        if not len(pairs):
            return np.zeros(0, dtype=np.int64)
        a = pairs[:, 0].astype(np.int64)
        b = pairs[:, 1].astype(np.int64)
        return np.minimum(a, b) * self.N + np.maximum(a, b)

    def _barycentric(self, points: np.ndarray, tri_idx: np.ndarray):
        """Closest point on each referenced reference triangle + barycentrics."""
        ref = self.x_ref
        tri = self.faces[tri_idx]
        a = ref[tri[:, 0]]
        b = ref[tri[:, 1]]
        c = ref[tri[:, 2]]
        e1 = b - a
        e2 = c - a
        normal = _unit(np.cross(e1, e2))
        rel = points - a
        height = np.einsum("ij,ij->i", rel, normal)
        q = rel - height[:, None] * normal
        d00 = np.einsum("ij,ij->i", e1, e1)
        d01 = np.einsum("ij,ij->i", e1, e2)
        d11 = np.einsum("ij,ij->i", e2, e2)
        dp0 = np.einsum("ij,ij->i", q, e1)
        dp1 = np.einsum("ij,ij->i", q, e2)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        u = (d11 * dp0 - d01 * dp1) / den
        v = (d00 * dp1 - d01 * dp0) / den
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        scale = np.maximum(u + v, 1.0)
        u = u / scale
        v = v / scale
        bary = np.stack((1.0 - u - v, u, v), axis=1)
        projected = np.einsum("mk,mkj->mj", bary, ref[tri])
        return bary, np.linalg.norm(points - projected, axis=1)

    # ------------------------------------------------------------------
    # force evaluation
    # ------------------------------------------------------------------
    def _spring_forces(self, force, pairs, length0, stiffness, damping, state):
        if len(pairs) == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        delta = self.x[j] - self.x[i]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-9)
        strain = (length - length0) / length0
        alive = state.alive
        elastic = strain - state.ep
        threshold = state.yield_strain + state.hardening * state.acc
        flowing = alive & (np.abs(elastic) > threshold)
        if np.any(flowing):
            step = (np.abs(elastic[flowing]) - threshold[flowing]) / (
                1.0 + state.hardening
            )
            state.acc[flowing] += step
            state.ep[flowing] += np.sign(elastic[flowing]) * step
            elastic = strain - state.ep
            failed = alive & (state.acc > state.failure)
            if np.any(failed):
                alive[failed] = False
        elastic = np.where(alive, elastic, 0.0)
        direction = delta / safe[:, None]
        relative = self.v[j] - self.v[i]
        along = np.einsum("ij,ij->i", relative, direction)
        magnitude = stiffness * length0 * elastic + damping * along
        magnitude = np.where(alive, magnitude, 0.0)
        contribution = magnitude[:, None] * direction
        force += _scatter(i, contribution, self.N)
        force -= _scatter(j, contribution, self.N)

    def _self_contact_force(self) -> np.ndarray:
        force = np.zeros_like(self.x)
        i = self.pair_i
        j = self.pair_j
        if len(i) == 0:
            return force
        delta = self.x[i] - self.x[j]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1e-9)[:, None]
        penetration = np.clip(self.SELF_DISTANCE - length, 0.0, self.SELF_PEN_CAP)
        active = penetration > 0.0
        if not np.any(active):
            return force
        relative = np.einsum("ij,ij->i", self.v[i] - self.v[j], direction)
        magnitude = self.SELF_STIFFNESS * penetration - self.self_c * relative
        magnitude = np.where(active, np.maximum(magnitude, 0.0), 0.0)
        contribution = magnitude[:, None] * direction
        force += _scatter(i, contribution, self.N)
        force -= _scatter(j, contribution, self.N)
        return force

    def _floor_force(self) -> np.ndarray:
        """Damped penalty contact with Coulomb friction on the frozen floor."""
        force = np.zeros_like(self.x)
        penetration = self.r_floor - self.x[:, 2]
        active = penetration > 0.0
        if not np.any(active):
            return force
        depth = np.clip(penetration, 0.0, self.FLOOR_PEN_CAP)
        normal = self.FLOOR_STIFFNESS * depth - self.floor_c * self.v[:, 2]
        normal = np.where(active, np.maximum(normal, 0.0), 0.0)
        tangential = self.v[:, :2]
        speed = np.linalg.norm(tangential, axis=1)
        regularised = np.tanh(speed / self.FLOOR_V_EPS) / np.maximum(speed, 1e-12)
        friction = -self.FLOOR_MU * normal[:, None] * tangential * regularised[:, None]
        force[:, 2] += normal
        force[:, :2] += friction
        return force

    def _refresh_self_contact(self) -> None:
        from scipy.spatial import cKDTree

        pairs = cKDTree(self.x).query_pairs(
            r=self.SELF_DISTANCE, output_type="ndarray"
        )
        if not len(pairs):
            self.pair_i = np.zeros(0, dtype=np.int64)
            self.pair_j = np.zeros(0, dtype=np.int64)
            return
        a = pairs[:, 0].astype(np.int64)
        b = pairs[:, 1].astype(np.int64)
        key = np.minimum(a, b) * self.N + np.maximum(a, b)
        if len(self.exclude_keys):
            keep = ~np.isin(key, self.exclude_keys)
        else:
            keep = np.ones(len(key), dtype=bool)
        self.pair_i = a[keep]
        self.pair_j = b[keep]

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        mass = self.m[:, None]
        force = mass * self.gravity
        self._spring_forces(
            force, self.edges, self.edge_length0, self.edge_k, self.edge_c, self.edge_state
        )
        self._spring_forces(
            force,
            self.bend_pairs,
            self.bend_length0,
            self.bend_k,
            self.bend_c,
            self.bend_state,
        )
        force += self._self_contact_force()
        force += self._floor_force()
        if len(self.tracer_idx):
            force[self.tracer_idx] = 0.0  # kinematically slaved material points

        self.v += (force * self.m_inv[:, None]) * dt
        self.v *= max(0.0, 1.0 - self.GLOBAL_DAMPING * dt)
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > self.MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (self.MAX_SPEED / speed[fast])[:, None]

        self.x += self.v * dt
        self._project_tracers()

        self._contact_tick += 1
        if self._contact_tick >= self.CONTACT_EVERY:
            self._contact_tick = 0
            self._refresh_self_contact()

    def _project_tracers(self) -> None:
        idx = self.tracer_idx
        if len(idx) == 0:
            return
        tri = self.faces[self.tracer_tri]
        bary = self.tracer_bary
        self.x[idx] = np.einsum("mk,mkj->mj", bary, self.x[tri])
        self.v[idx] = np.einsum("mk,mkj->mj", bary, self.v[tri])
        frames = self._triangle_frames(self.tracer_tri)
        self.x[idx] += np.einsum("mij,mj->mi", frames, self.tracer_off)

    def _triangle_frames(self, tri_idx: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        """Orthonormal element frames (columns t1, t2, n) for the given triangles."""
        position = self.x if x is None else x
        tri = self.faces[tri_idx]
        a = position[tri[:, 0]]
        e1 = position[tri[:, 1]] - a
        e2 = position[tri[:, 2]] - a
        t1 = _unit(e1)
        normal = _unit(np.cross(e1, e2))
        t2 = np.cross(normal, t1)
        return np.stack((t1, t2, normal), axis=2)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("requested sample time must be finite")
        if target < self.time - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing times")
        while self.time < target - 1e-12:
            dt = min(self.DT, target - self.time)
            self._step(dt)
            self.time = min(self.time + dt, target)
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------
    # exported state
    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        faces = self.faces
        ref = self.x_ref
        dm1 = ref[faces[:, 1]] - ref[faces[:, 0]]
        dm2 = ref[faces[:, 2]] - ref[faces[:, 0]]
        ds1 = self.x[faces[:, 1]] - self.x[faces[:, 0]]
        ds2 = self.x[faces[:, 2]] - self.x[faces[:, 0]]
        n_ref = _unit(np.cross(dm1, dm2))
        n_cur = _unit(np.cross(ds1, ds2))
        d_ref = np.stack((dm1, dm2), axis=2)
        d_cur = np.stack((ds1, ds2), axis=2)
        pinv = np.linalg.pinv(d_ref)
        grad = d_cur @ pinv + n_cur[:, :, None] * n_ref[:, None, :]
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        # blend toward the identity where the current element is degenerate
        area_cur = 0.5 * np.linalg.norm(np.cross(ds1, ds2), axis=1)
        degenerate = area_cur < 1e-10
        if np.any(degenerate):
            eye = np.eye(3)[None, :, :]
            grad = np.where(degenerate[:, None, None], eye, grad)

        accum = np.zeros((self.N, 3, 3))
        weight = np.zeros(self.N)
        for corner in range(3):
            np.add.at(accum, faces[:, corner], grad)
            np.add.at(weight, faces[:, corner], 1.0)
        empty = weight <= 0.0
        weight_safe = np.where(empty, 1.0, weight)
        out = accum / weight_safe[:, None, None]
        out[empty] = np.eye(3)[None, :, :]
        if len(self.tracer_idx):
            out[self.tracer_idx] = grad[self.tracer_tri]
        return out

    def _plastic_strain_per_sample(self) -> np.ndarray:
        accum = np.zeros(self.N)
        weight = np.zeros(self.N)
        if len(self.edges):
            active = self.edge_state.alive
            np.add.at(accum, self.edges[:, 0], np.where(active, self.edge_state.acc, 0.0))
            np.add.at(accum, self.edges[:, 1], np.where(active, self.edge_state.acc, 0.0))
            np.add.at(weight, self.edges[:, 0], 1.0)
            np.add.at(weight, self.edges[:, 1], 1.0)
        out = accum / np.maximum(weight, 1.0)
        if len(self.tracer_idx):
            tri = self.faces[self.tracer_tri]
            out[self.tracer_idx] = np.einsum("mk,mk->m", self.tracer_bary, out[tri])
        return out

    def _frame(self, time: float) -> StateFrame:
        external = self.m[:, None] * self.gravity + self._floor_force()
        gradient = self._deformation_gradient()
        gap = self.x[:, 2] - self.r_floor
        attributes = {
            "mass_quantity": self.m.copy(),
            "deformation_gradient": gradient,
            "particle_radius": np.full(self.N, self.r_particle),
            "plastic_strain": self._plastic_strain_per_sample(),
        }
        fields = {
            "contact_signed_distance": gap,
            "external_force": external,
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "contact_length_scale": self.spacing,
            "grid_spacing": self.spacing,
            "characteristic_length": max(2.0 * self.material_radius, self.spacing),
            "reference_velocity": 1.0,
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "samples": self.N,
            "edges": int(len(self.edges)),
            "bending_pairs": int(len(self.bend_pairs)),
            "tracers": int(len(self.tracer_idx)),
            "broken_edges": int(np.count_nonzero(~self.edge_state.alive)),
            "max_plastic_strain": float(self.edge_state.acc.max())
            if len(self.edge_state.acc)
            else 0.0,
            "substep": self.DT,
            "final_time": float(self.time),
        }
