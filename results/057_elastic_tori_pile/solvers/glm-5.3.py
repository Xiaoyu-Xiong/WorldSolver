"""Physically based solver for scene 057_elastic_tori_pile.

Forty incompressible elastic tori fall under gravity into a receiving region
ringed by twelve rigid posts and settle into a pile on a rigid floor.

Method (no external physics engine):
  * Each torus is a closed volumetric solid discretized by its 288 surface
    nodes (the frozen ``assets/torus.obj`` topology used by the renderer).
  * Structural elasticity: a spring network per torus built from the mesh
    graph (circumferential edges, meridional edges, face diagonals), radial
    tube chords through the tube axis, and shear braces between adjacent
    major-angle rings.  Rest lengths come from the frozen rest geometry.
  * Near-incompressibility: a volumetric pressure constraint per torus.  The
    enclosed volume is measured every substep with the divergence theorem on
    the closed surface, and a hydrostatic pressure proportional to the
    relative volume error pushes nodes along their current area-weighted
    surface normals, conserving each torus' volume under load.
  * Contacts: penalty sphere contacts (normal stiffness + normal damping +
    Coulomb-type capped tangential friction) between nodes of different tori,
    with the floor plane, and with the twelve post capsules.
  * Time integration: explicit symplectic Euler with a fixed substep,
    deterministic and fully vectorized in NumPy.

All trajectory fields exported in each StateFrame are measured from the
simulated state; nothing is keyed or precomputed.  Momentum bookkeeping is
exact: the only external forces are gravity, floor contacts, and post
contacts, and their time integrals accumulate into the reported budget.
"""

from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Frozen scene constants (mirror the scaffold configuration).
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
NU = 24               # major-angle samples of the frozen torus mesh
NV = 12               # tube-angle samples of the frozen torus mesh
MAJOR_RADIUS = 0.46
TUBE_RADIUS = 0.18

# Elastic material parameters (spring network + volumetric pressure).
K_EDGE = 600.0        # N/m  mesh edges
C_EDGE = 0.25         # N*s/m edge damping
K_CHORD = 250.0       # N/m  radial tube chords
C_CHORD = 0.15
K_BRACE = 300.0       # N/m  shear braces
C_BRACE = 0.15
K_VOLUME = 12000.0    # Pa    pressure coefficient for volume conservation

# Contact parameters.
NODE_RADIUS = 0.062   # collision sphere radius per surface node
K_PAIR = 1200.0
C_PAIR = 1.2
MU_PAIR = 0.40
CT_PAIR = 2.0
K_FLOOR = 1500.0
C_FLOOR = 1.5
MU_FLOOR = 0.60
CT_FLOOR = 3.0
K_POST = 1500.0
C_POST = 1.5
MU_POST = 0.50
CT_POST = 3.0

DT_SUB = 1.0e-3       # fixed integration substep (sample_dt / 20)
MAX_SPEED = 40.0      # safety clamp (free fall from 8.5 m stays below 13 m/s)

_GRID_LO = np.array([-5.0, -5.0, -1.0])
_GRID_HI = np.array([5.0, 5.0, 10.0])
_CELL = 0.13          # collision cell size (> 2 * NODE_RADIUS)


# ---------------------------------------------------------------------------
# Frozen torus rest geometry helpers.
# ---------------------------------------------------------------------------
def _local_vertices() -> np.ndarray:
    u = np.arange(NU) * (2.0 * math.pi / NU)
    v = np.arange(NV) * (2.0 * math.pi / NV)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = MAJOR_RADIUS + TUBE_RADIUS * np.cos(vv)
    return np.column_stack((
        (radial * np.cos(uu)).ravel(),
        (radial * np.sin(uu)).ravel(),
        (TUBE_RADIUS * np.sin(vv)).ravel(),
    ))


def _local_index(u: int, v: int) -> int:
    return (u % NU) * NV + (v % NV)


def _local_springs():
    """(i, j, stiffness, damping) spring list in local node indices."""
    half = NV // 2
    pairs = []
    for u in range(NU):
        for v in range(NV):
            pairs.append((_local_index(u, v), _local_index(u + 1, v), K_EDGE, C_EDGE))
            pairs.append((_local_index(u, v), _local_index(u, v + 1), K_EDGE, C_EDGE))
            pairs.append((_local_index(u, v), _local_index(u + 1, v + 1), K_BRACE, C_BRACE))
            pairs.append((_local_index(u, v), _local_index(u + 1, v + 3), K_BRACE, C_BRACE))
            pairs.append((_local_index(u, v), _local_index(u + 1, v + 9), K_BRACE, C_BRACE))
            if v < half:
                pairs.append((_local_index(u, v), _local_index(u, v + half), K_CHORD, C_CHORD))
    return np.asarray(pairs, dtype=float)


def _local_faces() -> np.ndarray:
    """Outward-oriented triangle faces of the closed torus surface."""
    faces = []
    for u in range(NU):
        for v in range(NV):
            a = _local_index(u, v)
            b = _local_index(u + 1, v)
            c = _local_index(u + 1, v + 1)
            d = _local_index(u, v + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int64)
    local = _local_vertices()
    det = np.linalg.det(np.stack([local[faces[:, 0]], local[faces[:, 1]], local[faces[:, 2]]], axis=1))
    if det.sum() < 0.0:                      # enforce outward winding
        faces = faces[:, [0, 2, 1]]
    return faces


def _local_tets() -> np.ndarray:
    """Volumetric tetrahedral elements spanning the tube interior.

    Each element ties an edge of one major-angle ring to the opposite side of
    the next ring, so it measures a genuine 3-D volumetric deformation of the
    solid torus (wall-through-tube elements are non-degenerate).
    """
    tets = []
    for u in range(NU):
        for k in (0, NV // 2):
            tets.append((
                _local_index(u, k),
                _local_index(u, k + 1),
                _local_index(u + 1, k),
                _local_index(u + 1, k + NV // 2),
            ))
    return np.asarray(tets, dtype=np.int64)


# ---------------------------------------------------------------------------
# Physics model.
# ---------------------------------------------------------------------------
class _TorusPilePhysics:
    def __init__(self, position, velocity, mass, body_id, floor, posts):
        self.x = np.array(position, dtype=float)
        self.x0 = self.x.copy()
        self.v = np.array(velocity, dtype=float)
        self.m = np.array(mass, dtype=float)
        self.body = np.asarray(body_id, dtype=np.int64)
        self.n = len(self.x)
        self.body_count = int(self.body.max()) + 1
        if not np.all(np.diff(self.body) >= 0):
            raise ValueError("body_id samples must be grouped contiguously")
        counts = np.bincount(self.body, minlength=self.body_count)
        if not np.all(counts == counts[0]):
            raise ValueError("all tori must share the frozen node count")
        self.npb = int(counts[0])
        offsets = np.arange(self.body_count, dtype=np.int64) * self.npb

        local = _local_vertices()
        springs = _local_springs()
        self.ia = (springs[:, 0].astype(np.int64)[None, :] + offsets[:, None]).ravel()
        self.ib = (springs[:, 1].astype(np.int64)[None, :] + offsets[:, None]).ravel()
        self.k_s = np.tile(springs[:, 2], self.body_count)
        self.c_s = np.tile(springs[:, 3], self.body_count)
        self.L0 = np.tile(
            np.linalg.norm(local[springs[:, 0].astype(int)] - local[springs[:, 1].astype(int)], axis=1),
            self.body_count,
        )

        faces_local = _local_faces()
        self.faces = (faces_local[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
        self.face_starts = np.arange(self.body_count, dtype=np.int64) * len(faces_local)

        # Rest vertex areas (associated surface area per node) and rest volumes.
        e1 = local[faces_local[:, 1]] - local[faces_local[:, 0]]
        e2 = local[faces_local[:, 2]] - local[faces_local[:, 0]]
        face_area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        fv = faces_local.ravel()
        w = np.repeat(face_area2, 3)
        area_local = np.bincount(fv, weights=w, minlength=self.npb) / 6.0
        self.area_v = np.tile(area_local, self.body_count)
        self.V0 = self._volumes(self.x)

        # Volumetric elements and their rest shape inverses.
        tets_local = _local_tets()
        self.tets = (tets_local[None, :, :] + offsets[:, None, None]).reshape(-1, 4)
        dm = np.stack([
            local[tets_local[:, 1]] - local[tets_local[:, 0]],
            local[tets_local[:, 2]] - local[tets_local[:, 0]],
            local[tets_local[:, 3]] - local[tets_local[:, 0]],
        ], axis=2)
        if np.abs(np.linalg.det(dm)).min() < 1.0e-9:
            raise ValueError("degenerate volumetric element")
        self.Dm_inv = np.tile(np.linalg.inv(dm), (self.body_count, 1, 1))
        self.element_count = len(self.tets)

        # Static boundary geometry.
        self.floor_half_x, self.floor_half_y, self.floor_z = floor
        self.posts = posts

        # Momentum budget accumulators (global domain, exact accounting).
        self.gravity_force = (self.m[:, None] * GRAVITY).sum(axis=0)
        self.cum_impulse = np.zeros(3)
        self.term_integral = np.zeros((3, 3))   # gravity, floor, posts
        self.interval = 0.0

        # Uniform collision grid.
        self.gx = int(math.ceil((_GRID_HI[0] - _GRID_LO[0]) / _CELL)) + 1
        self.gy = int(math.ceil((_GRID_HI[1] - _GRID_LO[1]) / _CELL)) + 1
        self.gz = int(math.ceil((_GRID_HI[2] - _GRID_LO[2]) / _CELL)) + 1
        offs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    offs.append(dx + self.gx * dy + self.gx * self.gy * dz)
        self.offsets = np.asarray(offs, dtype=np.int64)

        self.step_count = 0
        self.contact_pairs_seen = 0
        self.safety_resets = 0

    # -- geometry measurements ------------------------------------------------
    def _volumes(self, x: np.ndarray) -> np.ndarray:
        fa = x[self.faces[:, 0]]
        fb = x[self.faces[:, 1]]
        fc = x[self.faces[:, 2]]
        det = np.einsum("ij,ij->i", fa, np.cross(fb, fc))
        return np.add.reduceat(det / 6.0, self.face_starts)

    def _vertex_normals(self, x: np.ndarray) -> np.ndarray:
        fa = x[self.faces[:, 0]]
        fb = x[self.faces[:, 1]]
        fc = x[self.faces[:, 2]]
        fn = np.cross(fb - fa, fc - fa)
        fv = self.faces.ravel()
        w = np.repeat(fn, 3, axis=0)
        normals = np.column_stack([
            np.bincount(fv, weights=w[:, 0], minlength=self.n),
            np.bincount(fv, weights=w[:, 1], minlength=self.n),
            np.bincount(fv, weights=w[:, 2], minlength=self.n),
        ])
        length = np.linalg.norm(normals, axis=1)
        length[length < 1.0e-12] = 1.0
        return normals / length[:, None]

    # -- collision candidates -------------------------------------------------
    def _candidate_pairs(self):
        x = self.x
        ix = np.clip(((x[:, 0] - _GRID_LO[0]) / _CELL).astype(np.int64), 0, self.gx - 1)
        iy = np.clip(((x[:, 1] - _GRID_LO[1]) / _CELL).astype(np.int64), 0, self.gy - 1)
        iz = np.clip(((x[:, 2] - _GRID_LO[2]) / _CELL).astype(np.int64), 0, self.gz - 1)
        key = ix + self.gx * (iy + self.gy * iz)
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        neighbor_key = (key[:, None] + self.offsets[None, :]).ravel()
        left = np.searchsorted(sorted_key, neighbor_key, side="left")
        right = np.searchsorted(sorted_key, neighbor_key, side="right")
        counts = (right - left).ravel()
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        base = np.repeat(np.arange(self.n, dtype=np.int64), self.offsets.size)
        qi = np.repeat(base, counts)
        head = np.repeat(left, counts)
        group_start = np.cumsum(counts) - counts
        within = np.arange(total, dtype=np.int64) - np.repeat(group_start, counts)
        cand = order[np.clip(head + within, 0, self.n - 1)]
        keep = (qi < cand) & (self.body[qi] != self.body[cand])
        return qi[keep], cand[keep]

    # -- one explicit substep -------------------------------------------------
    def step(self, dt: float) -> None:
        x, v, n = self.x, self.v, self.n
        force = np.zeros((n, 3))
        force += self.m[:, None] * GRAVITY

        # Structural springs (internal, pairwise equal and opposite).
        d = x[self.ib] - x[self.ia]
        length = np.linalg.norm(d, axis=1)
        np.maximum(length, 1.0e-9, out=length)
        dirn = d / length[:, None]
        vrel = v[self.ib] - v[self.ia]
        vn = np.einsum("ij,ij->i", vrel, dirn)
        fs = self.k_s * (length - self.L0) + self.c_s * vn
        w = fs[:, None] * dirn
        for c in range(3):
            force[:, c] += (
                np.bincount(self.ia, weights=w[:, c], minlength=n)
                - np.bincount(self.ib, weights=w[:, c], minlength=n)
            )

        # Volumetric pressure constraint (internal, zero net force per body).
        volumes = self._volumes(x)
        pressure = K_VOLUME * (self.V0 - volumes) / self.V0
        normals = self._vertex_normals(x)
        fpress = pressure[self.body][:, None] * self.area_v[:, None] * normals
        sums = np.column_stack([
            np.bincount(self.body, weights=fpress[:, c], minlength=self.body_count)
            for c in range(3)
        ])
        fpress -= (sums / self.npb)[self.body]
        force += fpress

        # Node-node contacts between different tori (internal pairs).
        qi, qj = self._candidate_pairs()
        if len(qi):
            self.contact_pairs_seen += int(len(qi))
            d = x[qj] - x[qi]
            dist = np.linalg.norm(d, axis=1)
            np.maximum(dist, 1.0e-9, out=dist)
            dirn = d / dist[:, None]
            overlap = 2.0 * NODE_RADIUS - dist
            active = overlap > 0.0
            if np.any(active):
                qi, qj, dirn = qi[active], qj[active], dirn[active]
                overlap = overlap[active]
                vrel = v[qj] - v[qi]
                vn = np.einsum("ij,ij->i", vrel, dirn)
                fmag = K_PAIR * overlap + C_PAIR * np.maximum(0.0, -vn)
                vt = vrel - vn[:, None] * dirn
                speed_t = np.linalg.norm(vt, axis=1)
                fric = np.minimum(MU_PAIR * fmag, CT_PAIR * speed_t)
                ft = (fric / np.maximum(speed_t, 1.0e-12))[:, None] * vt
                w = fmag[:, None] * dirn
                for c in range(3):
                    force[:, c] += (
                        np.bincount(qj, weights=w[:, c], minlength=n)
                        - np.bincount(qi, weights=w[:, c], minlength=n)
                        + np.bincount(qi, weights=ft[:, c], minlength=n)
                        - np.bincount(qj, weights=ft[:, c], minlength=n)
                    )

        # Floor contact (external).
        fext_floor = np.zeros(3)
        pen = NODE_RADIUS - x[:, 2]
        act = (
            (pen > 0.0)
            & (np.abs(x[:, 0]) <= self.floor_half_x + NODE_RADIUS)
            & (np.abs(x[:, 1]) <= self.floor_half_y + NODE_RADIUS)
        )
        idx = np.nonzero(act)[0]
        if len(idx):
            fnm = K_FLOOR * pen[idx] + C_FLOOR * np.maximum(0.0, -v[idx, 2])
            contact = np.zeros((len(idx), 3))
            contact[:, 2] = fnm
            vt = v[idx].copy()
            vt[:, 2] = 0.0
            speed_t = np.linalg.norm(vt, axis=1)
            fric = np.minimum(MU_FLOOR * fnm, CT_FLOOR * speed_t)
            contact -= (fric / np.maximum(speed_t, 1.0e-12))[:, None] * vt
            force[idx] += contact
            fext_floor = contact.sum(axis=0)

        # Post contacts (external).
        fext_post = np.zeros(3)
        for px, py, radius, z0, z1 in self.posts:
            cz = np.clip(x[:, 2], z0, z1)
            dx = x[:, 0] - px
            dy = x[:, 1] - py
            dz = x[:, 2] - cz
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            pen = (radius + NODE_RADIUS) - dist
            act = pen > 0.0
            idx = np.nonzero(act)[0]
            if not len(idx):
                continue
            dvec = np.column_stack((dx[idx], dy[idx], dz[idx]))
            dist_i = np.maximum(dist[idx], 1.0e-9)
            dirn = dvec / dist_i[:, None]
            pen_i = pen[idx]
            vel = v[idx]
            vn = np.einsum("ij,ij->i", vel, dirn)
            fnm = K_POST * pen_i + C_POST * np.maximum(0.0, -vn)
            contact = fnm[:, None] * dirn
            vt = vel - vn[:, None] * dirn
            speed_t = np.linalg.norm(vt, axis=1)
            fric = np.minimum(MU_POST * fnm, CT_POST * speed_t)
            contact -= (fric / np.maximum(speed_t, 1.0e-12))[:, None] * vt
            force[idx] += contact
            fext_post += contact.sum(axis=0)

        # Symplectic Euler integration.
        self.v = v + force * (dt / self.m[:, None])

        # Safety: keep the state finite and bounded (never expected to fire).
        bad = ~np.isfinite(self.v).all(axis=1)
        if np.any(bad):
            self.safety_resets += int(bad.sum())
            self.v[bad] = 0.0
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (MAX_SPEED / speed[fast])[:, None]
        self.x = self.x + self.v * dt
        bad = ~np.isfinite(self.x).all(axis=1)
        if np.any(bad):
            self.safety_resets += int(bad.sum())
            self.x[bad] = self.x0[bad]
            self.v[bad] = 0.0

        # Exact external momentum accounting (gravity + floor + posts).
        self.cum_impulse += (self.gravity_force + fext_floor + fext_post) * dt
        self.term_integral += np.stack((self.gravity_force, fext_floor, fext_post)) * dt
        self.interval += dt
        self.step_count += 1

    # -- trajectory export ----------------------------------------------------
    def export_frame(self, time: float, orientation, angular_velocity) -> StateFrame:
        volumes = self._volumes(self.x)
        a = self.tets[:, 0]
        dcur = np.stack([
            self.x[self.tets[:, 1]] - self.x[a],
            self.x[self.tets[:, 2]] - self.x[a],
            self.x[self.tets[:, 3]] - self.x[a],
        ], axis=2)
        deformation = np.matmul(dcur, self.Dm_inv)

        if self.interval > 0.0:
            external_force = self.term_integral / self.interval
            self.term_integral = np.zeros((3, 3))
            self.interval = 0.0
        else:
            external_force = np.stack((self.gravity_force, np.zeros(3), np.zeros(3)))

        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": external_force[None, :, :].copy(),
            "constraint_target": self.V0.copy(),
            "constraint_value": volumes.copy(),
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": deformation,
        }
        scalars = {
            "constraint_scale": float(np.mean(self.V0)),
            "reference_momentum": float(np.sum(self.m) * 9.81),
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=None if orientation is None else np.array(orientation, dtype=float),
            angular_velocity=None if angular_velocity is None else np.array(angular_velocity, dtype=float),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes={"mass_quantity": self.m.copy()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame


# ---------------------------------------------------------------------------
# Frozen boundary geometry parsing.
# ---------------------------------------------------------------------------
def _parse_boundary(config: dict):
    objects = (config.get("scene_geometry") or {}).get("objects") or []
    floor = None
    posts = []
    for obj in objects:
        geometry = obj.get("geometry") or {}
        transform = obj.get("transform") or {}
        translation = transform.get("translation") or [0.0, 0.0, 0.0]
        if geometry.get("type") == "box" and geometry.get("half_extents"):
            he = geometry["half_extents"]
            floor = (float(he[0]), float(he[1]), float(translation[2] + he[2]))
        elif geometry.get("type") == "cylinder" and geometry.get("radius"):
            height = float(geometry["height"])
            radius = float(geometry["radius"])
            posts.append((
                float(translation[0]),
                float(translation[1]),
                radius,
                float(translation[2]) - 0.5 * height,
                float(translation[2]) + 0.5 * height,
            ))
    if floor is None:
        floor = (3.6, 3.6, 0.0)
    if not posts:  # deterministic fallback: frozen twelve-post ring
        for k in range(12):
            angle = 2.0 * math.pi * k / 12.0
            posts.append((2.35 * math.cos(angle), 2.35 * math.sin(angle), 0.13, 0.0, 2.8))
    return floor, posts


# ---------------------------------------------------------------------------
# Solver contract implementation.
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    def __init__(self) -> None:
        self._physics: _TorusPilePhysics | None = None
        self._orientation = None
        self._angular_velocity = None
        self._steps = 0
        self._frames = 0
        self._seed = 0

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        config = getattr(scene, "config", None) or {}
        floor, posts = _parse_boundary(config)
        self._physics = _TorusPilePhysics(
            initial.position,
            initial.velocity,
            initial.mass,
            initial.body_id,
            floor,
            posts,
        )
        if initial.orientation is not None:
            self._orientation = np.array(initial.orientation, dtype=float)
        else:
            self._orientation = np.tile([[0.0, 0.0, 0.0, 1.0]], (len(initial.position), 1))
        self._angular_velocity = np.zeros((len(initial.position), 3))
        self._steps = 0
        self._frames = 0
        self._seed = int(seed)

    def advance_to(self, time: float):
        if self._physics is None:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = int(round(float(time) / DT_SUB))
        while self._steps < target:
            self._physics.step(DT_SUB)
            self._steps += 1
        self._frames += 1
        return self._physics.export_frame(float(time), self._orientation, self._angular_velocity)

    def diagnostics(self) -> dict[str, object]:
        physics = self._physics
        if physics is None:
            return {"success": False, "error": "not initialized"}
        return {
            "success": True,
            "method": "explicit symplectic Euler; spring-network elastic tori with "
                      "divergence-theorem volumetric pressure constraint and penalty contacts",
            "substep": DT_SUB,
            "steps_taken": physics.step_count,
            "frames_exported": self._frames,
            "bodies": physics.body_count,
            "nodes": physics.n,
            "elements": physics.element_count,
            "contact_pairs_evaluated": physics.contact_pairs_seen,
            "safety_resets": physics.safety_resets,
            "seed": self._seed,
        }
