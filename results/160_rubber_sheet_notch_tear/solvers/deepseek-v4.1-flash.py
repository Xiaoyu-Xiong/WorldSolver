"""Physically-based solver for scene 160_rubber_sheet_notch_tear.

Model
-----
A notched thin rubber sheet is discretised as a mass-spring shell built directly
from the frozen cloth topology (``assets/cloth_topology.json``):

  * tensile (Kelvin-Voigt) springs on every mesh edge of the (already notched)
    triangulation; the notch is a real material void in the topology, so no
    spring crosses it and the notch tip is a genuine stress concentrator,
  * second-neighbour / bending springs that give the sheet out-of-plane and
    shear rigidity,
  * explicit symplectic (semi-implicit) Euler stepping with a CFL-limited fixed
    inner time step derived from the stiffest spring and the lightest sample,
  * adaptive tearing: a spring fails irreversibly when its stretch ratio exceeds
    the material failure strain (with a small deterministic material disorder),
    which reproduces crack initiation at the notch tip and its propagation,
  * kinematic grips: the two boundary vertex columns are prescribed to the grip
    separation law (prescribed tensile loading) with matching velocity,
  * rigid contact against the ground slab and the two grip boxes, resolved by
    projection plus removal of the normal velocity component.

Everything exported is raw physical state; no keyframes, no scripted motion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# frozen scene constants (mirrors config.yaml / scene_geometry.yaml)
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
TOTAL_MASS = 0.32
THICKNESS = 0.015
GRID_NX = 17
GRID_NY = 17
SIZE_X = 3.0
SIZE_Y = 2.5
CENTER = np.array([0.0, 0.0, 1.6], dtype=float)

# tensile spring stiffness: k = E * t * w / L for the sheet element size, i.e.
# a rubber-like Young's modulus of ~1 MPa with the frozen thickness.
K_TENSILE = 1.2e4          # N/m
K_BEND = 1.2e2             # N/m (bending / second-neighbour springs)
DAMPING_RATIO = 0.12       # fraction of critical used for the spring dashpots
BULK_DRAG = 0.25           # 1/s, mild global drag for coherence
CFL_SAFETY = 0.35
FAIL_STRAIN = 0.20         # tensile failure strain of the rubber compound
MATERIAL_DISORDER = 0.06   # +/- relative spread of the local failure strain
MAX_SPEED = 40.0           # m/s safety clamp (never reached in practice)

# prescribed tensile loading (grips separate, with a smooth velocity ramp)
GRIP_SPEED = 0.042         # m/s per grip (far-field strain rate ~0.028/s)
GRIP_RAMP = 1.0            # s, smooth start-up
GRIP_HALF_X = 1.5

GROUND_CENTER = np.array([0.0, 0.0, -0.05], dtype=float)
GROUND_HALF = np.array([2.4, 2.0, 0.05], dtype=float)
GRIP_HALF = np.array([0.12, 0.18, 1.25], dtype=float)

CONTACT_NEAR = 2.0 * THICKNESS
CONTACT_MIN_SAMPLES = 8


def _workspace_file(rel: str):
    """Locate a frozen asset relative to cwd or to this file's workspace."""
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / rel]
    for base in (here.parent.parent, here.parent.parent.parent, here.parent):
        candidates.append(base / rel)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _load_geometry_boxes():
    """Read the frozen obstacle boxes if the geometry file is reachable."""
    ground_c, ground_h = GROUND_CENTER, GROUND_HALF
    grip_h = GRIP_HALF
    path = _workspace_file("scene_geometry.yaml")
    if path is not None:
        try:
            import yaml  # optional, scaffold ships yaml files

            spec = yaml.safe_load(path.read_text())
            for obj in spec.get("objects", []):
                geom = obj.get("geometry", {})
                half = np.asarray(geom.get("half_extents", []), dtype=float)
                trans = np.asarray(
                    obj.get("transform", {}).get("translation", []), dtype=float
                )
                if obj.get("id") == "ground" and half.size == 3:
                    ground_c, ground_h = trans, half
                elif obj.get("id") in ("grip_left", "grip_right") and half.size == 3:
                    grip_h = half
        except Exception:
            pass
    return ground_c, ground_h, grip_h


def _grid_faces(nx: int, ny: int) -> np.ndarray:
    """Triangulated full grid (fallback only; the frozen asset is notched)."""
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return np.asarray(faces, dtype=np.int64)


def _notched_grid_faces(nx: int, ny: int) -> np.ndarray:
    """Fallback notched triangulation (elliptical slit opening on the -x edge)."""
    limit = {5: 1, 6: 2, 7: 3, 8: 3, 9: 2, 10: 1}
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if i < limit.get(j, 0):
                continue
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return np.asarray(faces, dtype=np.int64)


def _load_topology(n_expected: int):
    """Return (material_2d, faces) from the frozen topology asset."""
    path = _workspace_file("assets/cloth_topology.json")
    if path is not None:
        try:
            data = json.loads(path.read_text())
            verts = np.asarray(data["vertices"], dtype=float)
            faces = np.asarray(data["faces"], dtype=np.int64)
            if verts.shape[0] == n_expected and faces.ndim == 2 and faces.shape[1] == 3:
                return verts[:, :2].copy(), faces
        except Exception:
            pass
    nx, ny = GRID_NX, GRID_NY
    xs = np.linspace(-SIZE_X * 0.5, SIZE_X * 0.5, nx)
    ys = np.linspace(-SIZE_Y * 0.5, SIZE_Y * 0.5, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    material = np.column_stack((xx.ravel(), yy.ravel()))
    return material, _notched_grid_faces(nx, ny)


def _unique_pairs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    key = pairs[:, 0] * np.int64(1 << 20) + pairs[:, 1]
    order = np.argsort(key, kind="stable")
    key = key[order]
    keep = np.ones(key.shape[0], dtype=bool)
    keep[1:] = key[1:] != key[:-1]
    return pairs[order][keep]


def _springs_from_faces(faces: np.ndarray, material: np.ndarray):
    lo = np.minimum(faces[:, 0], faces[:, 1])
    hi = np.maximum(faces[:, 0], faces[:, 1])
    edges = _unique_pairs(np.column_stack((lo, hi)))

    second = np.concatenate(
        [
            np.column_stack((faces[:, 0], faces[:, 1])),
            np.column_stack((faces[:, 1], faces[:, 2])),
            np.column_stack((faces[:, 2], faces[:, 0])),
        ],
        axis=0,
    )
    second = np.column_stack(
        (np.minimum(second[:, 0], second[:, 1]), np.maximum(second[:, 0], second[:, 1]))
    )
    second = _unique_pairs(second)
    return edges, second


class BenchmarkSolver:
    """Explicit mass-spring thin sheet with adaptive strain-driven tearing."""

    # -- construction -------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(getattr(scene, "config", {}))
        self.seed = int(seed)
        self.time = 0.0

        cloth = self.config.get("cloth", {})
        thickness = float(cloth.get("thickness", THICKNESS))
        total_mass = float(cloth.get("total_mass", TOTAL_MASS))
        center = np.asarray(cloth.get("center", CENTER), dtype=float)
        self.gravity = np.asarray(self.config.get("gravity", GRAVITY), dtype=float)

        state0 = scene.initial_state()
        pos0 = np.array(state0.position, dtype=float)
        n = pos0.shape[0]
        self.n = n
        self.mass = (
            np.array(state0.mass, dtype=float)
            if state0.mass is not None
            else np.full(n, total_mass / n)
        )
        self.inv_mass = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)
        self.body_id = np.zeros(n, dtype=np.int32)
        self.sample_dt = float(self.config.get("sample_dt", 1.0 / 30.0))

        material, faces = _load_topology(n)
        if material.shape[0] != n or faces.size == 0:
            # rebuild a matching (notched) grid discretisation
            nx, ny = map(int, cloth.get("grid_shape", (GRID_NX, GRID_NY)))
            xs = np.linspace(-0.5 * float(cloth.get("size", [SIZE_X, SIZE_Y])[0]),
                             0.5 * float(cloth.get("size", [SIZE_X, SIZE_Y])[0]), nx)
            ys = np.linspace(-0.5 * float(cloth.get("size", [SIZE_X, SIZE_Y])[1]),
                             0.5 * float(cloth.get("size", [SIZE_X, SIZE_Y])[1]), ny)
            xx, yy = np.meshgrid(xs, ys, indexing="xy")
            material = np.column_stack((xx.ravel(), yy.ravel()))
            faces = _notched_grid_faces(nx, ny)
        self.material = material
        self.faces = faces

        # --- springs -------------------------------------------------------
        edges, second = _springs_from_faces(faces, material)
        rng = np.random.default_rng(self.seed + 1601)

        def build_pairs(pairs, stiffness, fail_scale):
            if pairs.shape[0] == 0:
                return None
            diff = material[pairs[:, 1]] - material[pairs[:, 0]]
            rest = np.linalg.norm(diff, axis=1)
            good = rest > 1e-9
            pairs, rest = pairs[good], rest[good]
            k = np.full(rest.shape[0], stiffness)
            disorder = 1.0 + MATERIAL_DISORDER * (2.0 * rng.random(rest.shape[0]) - 1.0)
            fail = FAIL_STRAIN * fail_scale * disorder
            c = DAMPING_RATIO * np.sqrt(k * float(np.min(self.mass)))
            return {
                "i": pairs[:, 0].copy(),
                "j": pairs[:, 1].copy(),
                "rest": rest,
                "k": k,
                "c": c,
                "fail": fail,
            }

        s_tens = build_pairs(edges, K_TENSILE, 1.0)
        s_bend = build_pairs(second, K_BEND, 1.25)
        parts = [s for s in (s_tens, s_bend) if s is not None]
        self.spring = {
            key: np.concatenate([p[key] for p in parts])
            for key in ("i", "j", "rest", "k", "c", "fail")
        }
        self.n_springs = int(self.spring["i"].shape[0])
        self.active = np.ones(self.n_springs, dtype=bool)
        self._cache_dirty = True
        self.broken_count = 0
        self.crack_tip_x = float("nan")
        self.max_stretch = 1.0

        # --- boundary (grip) binding ---------------------------------------
        bound = np.zeros(n, dtype=bool)
        if "boundary_vertex" in (state0.attributes or {}):
            bound = np.asarray(state0.attributes["boundary_vertex"]).astype(bool)
        if not bound.any():
            tol = 1e-6
            bound = np.abs(material[:, 0] - material[:, 0].min()) < tol
            bound |= np.abs(material[:, 0] - material[:, 0].max()) < tol
        self.pinned = bound
        self.free = ~bound
        self.pinned_rest = pos0[bound].copy()
        self.pinned_side = np.where(
            self.pinned_rest[:, 0] < float(center[0]), -1.0, 1.0
        )
        # material-coordinate blend used for the deformation gradient
        self.boundary_flag = bound.astype(np.int32)

        # --- state ---------------------------------------------------------
        self.pos = pos0.copy()
        self.vel = np.array(state0.velocity, dtype=float).copy()
        self.force = np.zeros_like(self.pos)

        # --- obstacles -----------------------------------------------------
        self.ground_c, self.ground_h, self.grip_h = _load_geometry_boxes()
        self.grip_centers = np.array(
            [
                [-GRIP_HALF_X, 0.0, float(center[2])],
                [GRIP_HALF_X, 0.0, float(center[2])],
            ],
            dtype=float,
        )

        # --- deformation-gradient neighbour table --------------------------
        pairs = np.concatenate(
            [np.column_stack((self.spring["i"], self.spring["j"])),
             np.column_stack((self.spring["j"], self.spring["i"]))],
            axis=0,
        )
        self.dg_pairs = pairs
        self.dg_rest = material[pairs[:, 1]] - material[pairs[:, 0]]
        counts = np.bincount(pairs[:, 0], minlength=n)
        self.dg_isolated = counts == 0

        # --- fixed inner step from the CFL condition -----------------------
        k_max = float(np.max(self.spring["k"]))
        m_min = float(np.min(self.mass[self.mass > 0.0]))
        omega = np.sqrt(2.0 * k_max / max(m_min, 1e-12))
        dt_cfl = 2.0 * CFL_SAFETY / omega
        substeps = max(1, int(np.ceil(self.sample_dt / dt_cfl)))
        self.substeps = int(substeps)
        self.dt = self.sample_dt / float(substeps)
        self.contact_length_scale = thickness

        self._grip_shift_value = 0.0
        self._refresh_cache()
        self.diagnostics_extra = {}

    # -- helpers ------------------------------------------------------------
    def _grip_offset(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t < GRIP_RAMP:
            u = t / GRIP_RAMP
            return GRIP_SPEED * GRIP_RAMP * (u ** 3 - 0.5 * u ** 4)
        return 0.5 * GRIP_SPEED * GRIP_RAMP + GRIP_SPEED * (t - GRIP_RAMP)

    def _grip_velocity(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t < GRIP_RAMP:
            u = t / GRIP_RAMP
            return GRIP_SPEED * (3.0 * u * u - 2.0 * u ** 3)
        return GRIP_SPEED

    def _refresh_cache(self) -> None:
        idx = np.flatnonzero(self.active)
        self._ia = self.spring["i"][idx]
        self._ja = self.spring["j"][idx]
        self._iL0 = self.spring["rest"][idx]
        self._ik = self.spring["k"][idx]
        self._ic = self.spring["c"][idx]
        self._ifail = self.spring["fail"][idx]
        self._iidx = idx
        self._cache_dirty = False

    def _box_sdf(self, p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        d = np.abs(p - center[None, :]) - half[None, :]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return outside + inside

    def _min_signed_distance(self, p: np.ndarray) -> np.ndarray:
        """Signed distance (negative inside) to the union of frozen obstacles."""
        sd = self._box_sdf(p, self.ground_c, self.ground_h)
        shift = self._grip_offset(self.time)
        for g, sign in ((0, -1.0), (1, 1.0)):
            center = self.grip_centers[g].copy()
            center[0] += sign * shift
            sd = np.minimum(sd, self._box_sdf(p, center, self.grip_h))
        return sd

    # -- integration --------------------------------------------------------
    def _spring_forces(self) -> np.ndarray:
        pos, vel = self.pos, self.vel
        ii, jj = self._ia, self._ja
        d = pos[ii] - pos[jj]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1e-12)
        inv = 1.0 / safe
        stretch = safe / self._iL0
        self.max_stretch = float(max(self.max_stretch, float(np.max(stretch)) if stretch.size else 1.0))

        axial = (self._ik * (safe - self._iL0) * inv)[:, None] * d
        rel_v = np.sum((vel[ii] - vel[jj]) * d, axis=1) * inv
        axial += (self._ic * rel_v * inv)[:, None] * d

        n = self.n
        f = np.zeros((n, 3), dtype=float)
        for c in range(3):
            f[:, c] = np.bincount(jj, weights=axial[:, c], minlength=n) - np.bincount(
                ii, weights=axial[:, c], minlength=n
            )

        # irreversible failure of overstretched springs (adaptive tearing)
        failed = stretch > self._ifail
        if failed.any():
            self.active[self._iidx[failed]] = False
            self.broken_count += int(np.count_nonzero(failed))
            self._cache_dirty = True
        return f

    def _step(self, dt: float) -> None:
        self._grip_shift_value = self._grip_offset(self.time)
        if self._cache_dirty:
            self._refresh_cache()

        force = self._spring_forces()
        acc = force * self.inv_mass[:, None] + self.gravity[None, :]
        self.vel += acc * dt
        self.vel *= max(0.0, 1.0 - BULK_DRAG * dt)
        np.clip(self.vel, -MAX_SPEED, MAX_SPEED, out=self.vel)
        self.pos += self.vel * dt

        self._resolve_contacts()
        self._apply_grips(self.time + dt)
        self.time += dt
        self.force = force

    def _resolve_contacts(self) -> None:
        free = self.free
        p = self.pos[free]
        if p.shape[0] == 0:
            return
        sd = self._min_signed_distance(p)
        pen = sd < 0.0
        if not np.any(pen):
            return
        pp = p[pen]
        h = 1e-5
        nrm = np.empty_like(pp)
        for axis in range(3):
            dp = np.zeros_like(pp)
            dm = np.zeros_like(pp)
            dp[:, axis] = h
            dm[:, axis] = h
            nrm[:, axis] = (
                self._min_signed_distance(pp + dp) - self._min_signed_distance(pp - dm)
            ) / (2.0 * h)
        norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(norm, 1e-12)
        depth = -sd[pen]
        p[pen] = pp + nrm * depth[:, None]
        v = self.vel[free]
        vv = v[pen]
        vn = np.sum(vv * nrm, axis=1)
        moving = vn < 0.0
        if np.any(moving):
            vv[moving] -= nrm[moving] * vn[moving][:, None]
        v[pen] = vv
        self.vel[free] = v
        self.pos[free] = p

    def _apply_grips(self, t: float) -> None:
        shift = self._grip_offset(t)
        speed = self._grip_velocity(t)
        pinned = self.pinned
        if not np.any(pinned):
            return
        side = np.where(self.pinned_rest[:, 0] < 0.0, -1.0, 1.0)
        target = self.pinned_rest.copy()
        target[:, 0] += side * shift
        self.pos[pinned] = target
        self.vel[pinned] = 0.0
        self.vel[pinned, 0] = side * speed

    # -- public contract ----------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        budget = 4 * (self.substeps + 1) * (int(round(target / self.sample_dt)) + 2)
        steps = 0
        while self.time < target - 1e-12 and steps < budget:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-15:
                break
            self._step(dt)
            steps += 1
        if self._cache_dirty:
            self._refresh_cache()
        return self._build_frame(float(target))

    def _build_frame(self, time: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()

        # per-sample deformation gradient in material (rest uv) coordinates
        F = np.tile(np.eye(2), (self.n, 1, 1))
        keep = np.concatenate([self.active, self.active])
        pairs = self.dg_pairs[keep]
        if pairs.shape[0]:
            dc = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            dr = self.dg_rest
            n = self.n
            M2 = np.zeros((n, 2, 2))
            M3 = np.zeros((n, 3, 2))
            for a in range(2):
                for b in range(2):
                    M2[:, a, b] = np.bincount(
                        pairs[:, 0], weights=dr[:, a] * dr[:, b], minlength=n
                    )
                for a3 in range(3):
                    M3[:, a3, a] = np.bincount(
                        pairs[:, 0], weights=dc[:, a3] * dr[:, a], minlength=n
                    )
            M2 += np.eye(2)[None, :, :] * 1e-9
            A = np.linalg.solve(M2, np.transpose(M3, (0, 2, 1)))
            A = np.transpose(A, (0, 2, 1))  # (n,3,2)
            C = np.einsum("nij,nik->njk", A, A)
            w, V = np.linalg.eigh(C)
            s = np.clip(np.sqrt(np.clip(w, 0.0, None)), 1e-3, 1e3)
            F = np.einsum("nij,nj,nkj->nik", V, s, V)
            F[self.dg_isolated] = np.eye(2)

        # contact reporting: active + nearest free samples
        sd = self._min_signed_distance(pos)
        sd_free_idx = np.flatnonzero(self.free)
        if sd_free_idx.size:
            sd_free = sd[sd_free_idx]
            order = np.argsort(sd_free, kind="stable")
            near = sd_free[order] < CONTACT_NEAR
            picked = list(sd_free_idx[order][near])
            for k in order[:CONTACT_MIN_SAMPLES]:
                idx = int(sd_free_idx[k])
                if idx not in picked:
                    picked.append(idx)
            picked = sorted(set(picked), key=lambda i: (float(sd[i]), i))
            contact_sd = sd[np.asarray(picked, dtype=np.int64)]
        else:
            contact_sd = np.zeros(0)

        external = mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            float(time),
            pos,
            vel,
            mass=mass,
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": np.asarray(F, dtype=float),
                "boundary_vertex": self.boundary_flag.copy(),
            },
            fields={
                "external_force": external,
                "contact_signed_distance": np.asarray(contact_sd, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "time": float(self.time),
            "samples": int(self.n),
            "springs": int(self.n_springs),
            "broken_springs": int(self.broken_count),
            "torn": bool(self.broken_count > 0),
            "max_stretch": float(self.max_stretch),
            "inner_dt": float(self.dt),
        }
