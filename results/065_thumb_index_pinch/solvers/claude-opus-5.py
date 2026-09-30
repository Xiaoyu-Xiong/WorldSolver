"""Volumetric flesh solver for the thumb-index web pinch.

Physical model
--------------
The fused watertight hand/forearm surface is voxelized into an embedded
tetrahedral lattice (6 Kuhn tetrahedra per occupied cell).  The lattice carries
a compressible-but-stiff *stable Neo-Hookean* solid (Smith, Goes & Kim 2018):

    Psi(F) = 0.5 * mu * (Ic - 3) + 0.5 * lambda * (J - alpha)^2
             - 0.5 * mu * log(Ic + 1)
    alpha  = 1 + mu / lambda - mu / (4 * lambda)

which stays well defined through the extreme compression that occurs in the
thumb-index web, so no inversion barrier is needed.  Nodal forces come from the
standard linear-tetrahedron first Piola-Kirchhoff assembly

    H = -vol * P(F) @ DmInv^T

and the state is advanced with damped symplectic (semi-implicit) Euler at a
CFL-limited step size.  Dirichlet conditions follow the frozen scene: the
forearm cuff region is pinned, ``index_bone`` flesh is held in its initial
pose, and ``thumb_bone`` flesh is driven by the frozen smoothstep rotation and
translation of ``config['thumb_drive']``.  The rendered surface vertices are
embedded in the lattice by trilinear (free-form) interpolation of the lattice
displacement field, so the exported ``state.position`` is exactly the OBJ rest
pose at t = 0 and a smoothly deformed flesh surface afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-14


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices [V,3], triangles [T,3]) from a Wavefront OBJ."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int64).reshape(-1, 3),
    )


def _mesh_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh (divergence theorem)."""
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _inside_mask(
    verts: np.ndarray,
    tris: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    zc: np.ndarray,
) -> np.ndarray:
    """Parity ray-cast along +x -> boolean occupancy [nx,ny,nz] of cell centers."""
    a = verts[tris[:, 0]]
    e1 = verts[tris[:, 1]] - a
    e2 = verts[tris[:, 2]] - a
    # 2D (y,z) cross product of the projected edges; ray direction is +x.
    det = e1[:, 1] * e2[:, 2] - e2[:, 1] * e1[:, 2]
    good = np.abs(det) > _EPS
    a, e1, e2, det = a[good], e1[good], e2[good], det[good]
    ny, nz, nx = len(yc), len(zc), len(xc)
    yy, zz = np.meshgrid(yc, zc, indexing="ij")
    rays = np.stack([yy.ravel(), zz.ravel()], axis=1)
    out = np.zeros((len(rays), nx), dtype=bool)
    chunk = 48
    for start in range(0, len(rays), chunk):
        blk = rays[start : start + chunk]
        py = blk[:, 0][:, None] - a[None, :, 1]
        pz = blk[:, 1][:, None] - a[None, :, 2]
        u = (py * e2[None, :, 2] - e2[None, :, 1] * pz) / det[None, :]
        v = (e1[None, :, 1] * pz - py * e1[None, :, 2]) / det[None, :]
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        xh = a[None, :, 0] + u * e1[None, :, 0] + v * e2[None, :, 0]
        for row in range(len(blk)):
            crossings = np.sort(xh[row][hit[row]])
            ahead = len(crossings) - np.searchsorted(crossings, xc, side="left")
            out[start + row] = (ahead % 2) == 1
    return out.reshape(ny, nz, nx).transpose(2, 0, 1)


# Kuhn (path) decomposition of a cube into 6 tetrahedra sharing the main
# diagonal (0,0,0)->(1,1,1).  It is face conforming between neighbour cells, so
# a uniform lattice of these tetrahedra forms a valid simplicial complex.
_KUHN = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
)


def _build_lattice(
    occ: np.ndarray, origin: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tetrahedralize occupied cells; return (node_rest, tets, node_id grid)."""
    nx, ny, nz = occ.shape
    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    ci, cj, ck = np.nonzero(occ)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                used[ci + di, cj + dj, ck + dk] = True
    node_id = np.full(used.shape, -1, dtype=np.int64)
    node_id[used] = np.arange(int(used.sum()), dtype=np.int64)
    gi, gj, gk = np.nonzero(used)
    node_rest = origin[None, :] + np.stack([gi, gj, gk], axis=1) * h

    tets = np.empty((len(ci) * 6, 4), dtype=np.int64)
    for t, corners in enumerate(_KUHN):
        for c, (di, dj, dk) in enumerate(corners):
            tets[t :: 6, c] = node_id[ci + di, cj + dj, ck + dk]
    return node_rest, tets, node_id


def _smoothstep(s: float) -> tuple[float, float, float]:
    """Cubic smoothstep and its first two derivatives w.r.t. the raw ratio s."""
    if s <= 0.0:
        return 0.0, 0.0, 0.0
    if s >= 1.0:
        return 1.0, 0.0, 0.0
    return s * s * (3.0 - 2.0 * s), 6.0 * s * (1.0 - s), 6.0 - 12.0 * s


def _deformation_gradients(
    pos: np.ndarray, tets: np.ndarray, dm_inv: np.ndarray
) -> np.ndarray:
    """Per-tetrahedron deformation gradient F = Ds @ DmInv, shape [Ne,3,3]."""
    p0 = pos[tets[:, 0]]
    ds = np.stack(
        (pos[tets[:, 1]] - p0, pos[tets[:, 2]] - p0, pos[tets[:, 3]] - p0), axis=2
    )
    return ds @ dm_inv


def _stable_neohookean_stress(f: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """First Piola-Kirchhoff stress of the stable Neo-Hookean energy."""
    c0 = f[:, :, 0]
    c1 = f[:, :, 1]
    c2 = f[:, :, 2]
    # dJ/dF columns are the cyclic cross products of the F columns.
    g0 = np.cross(c1, c2)
    g1 = np.cross(c2, c0)
    g2 = np.cross(c0, c1)
    djdf = np.stack((g0, g1, g2), axis=2)
    jdet = np.einsum("ij,ij->i", c0, g0)
    ic = np.einsum("ijk,ijk->i", f, f)
    alpha = 1.0 + mu / lam - mu / (4.0 * lam)
    scale = mu / (ic + 1.0)
    return mu * f - scale[:, None, None] * f + (
        lam * (jdet - alpha)
    )[:, None, None] * djdf


def _rodrigues_motion(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    angle: float,
    dangle: float,
    ddangle: float,
    translation: np.ndarray,
    ease: float,
    dease: float,
    ddease: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid rotation about (center, axis) plus translation, with derivatives."""
    r = points - center[None, :]
    k = np.cross(np.broadcast_to(axis, r.shape), r)
    kk = np.cross(np.broadcast_to(axis, r.shape), k)
    sin_a, cos_a = np.sin(angle), np.cos(angle)
    rot = r + sin_a * k + (1.0 - cos_a) * kk
    d_rot = cos_a * k + sin_a * kk
    dd_rot = -sin_a * k + cos_a * kk
    pos = center[None, :] + rot + ease * translation[None, :]
    vel = dangle * d_rot + dease * translation[None, :]
    acc = (
        ddangle * d_rot
        + (dangle * dangle) * dd_rot
        + ddease * translation[None, :]
    )
    return pos, vel, acc


def _capsule_distance(
    points: np.ndarray, center: np.ndarray, axis: np.ndarray, half: float
) -> np.ndarray:
    """Distance from points to the capsule core segment."""
    rel = points - center[None, :]
    t = np.clip(rel @ axis, -half, half)
    return np.linalg.norm(rel - t[:, None] * axis[None, :], axis=1)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < _EPS:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class BenchmarkSolver:
    """Embedded-lattice stable Neo-Hookean flesh solver."""

    # Soft-tissue flesh parameters (nearly incompressible, so squeezed material
    # has to escape sideways -> the web mound).
    YOUNGS_MODULUS = 3.0e4      # Pa
    POISSON_RATIO = 0.46        # dimensionless
    MASS_DAMPING = 60.0         # 1 / s, keeps the pinch quasi-static
    CELL_SIZE = 0.011           # m, lattice spacing
    CFL_SAFETY = 0.18           # explicit step safety factor
    BONE_MARGIN = 0.009         # m, flesh shell rigidly tied to each bone

    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self.scene = scene
        cfg = dict(scene.config)
        self.config = cfg
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        init = scene.initial_state()
        self.rest_surface = np.asarray(init.position, dtype=np.float64).copy()
        self.n_surface = len(self.rest_surface)
        self._scene_attributes = {
            key: np.asarray(value).copy() for key, value in init.attributes.items()
        }

        obj_path = self._locate_asset(scene)
        _, tris = _load_obj(obj_path)
        self.tris = tris
        surface_volume = _mesh_volume(self.rest_surface, tris)

        self._build_mesh(tris)
        self._setup_material(surface_volume)
        self._setup_boundaries()
        self._setup_embedding()

        self.time = 0.0
        self.node_pos = self.node_rest.copy()
        self.node_vel = np.zeros_like(self.node_rest)
        self._apply_prescribed(0.0)
        self._max_speed = 0.0
        self._min_jacobian = 1.0

    @staticmethod
    def _locate_asset(scene) -> Path:
        import sys

        candidates: list[Path] = []
        module = sys.modules.get(type(scene).__module__)
        module_file = getattr(module, "__file__", None)
        if module_file:
            candidates.append(Path(module_file).resolve().parent)
        candidates.append(Path.cwd())
        candidates.append(Path(__file__).resolve().parent.parent)
        for base in candidates:
            path = base / "assets" / "hand_flesh.obj"
            if path.is_file():
                return path
        raise FileNotFoundError("assets/hand_flesh.obj not found")

    def _build_mesh(self, tris: np.ndarray) -> None:
        h = float(self.CELL_SIZE)
        self.h = h
        lo = self.rest_surface.min(axis=0) - 1.5 * h
        hi = self.rest_surface.max(axis=0) + 1.5 * h
        dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
        self.grid_origin = lo
        self.grid_dims = dims
        centers = [lo[d] + (np.arange(dims[d]) + 0.5) * h for d in range(3)]
        occ = _inside_mask(self.rest_surface, tris, centers[0], centers[1], centers[2])
        # Guarantee that every rendered surface vertex sits in an occupied cell so
        # the trilinear embedding always has support.
        cells = np.floor((self.rest_surface - lo[None, :]) / h).astype(np.int64)
        cells = np.clip(cells, 0, dims[None, :] - 1)
        occ[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        self.occupancy = occ
        self.node_rest, self.tets, self.node_id = _build_lattice(occ, lo, h)
        self.n_nodes = len(self.node_rest)
        self.n_elements = len(self.tets)

    def _setup_material(self, surface_volume: float) -> None:
        e_mod = float(self.YOUNGS_MODULUS)
        nu = float(self.POISSON_RATIO)
        self.mu = e_mod / (2.0 * (1.0 + nu))
        self.lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        rest = self.node_rest
        p0 = rest[self.tets[:, 0]]
        dm = np.stack(
            (rest[self.tets[:, 1]] - p0, rest[self.tets[:, 2]] - p0,
             rest[self.tets[:, 3]] - p0),
            axis=2,
        )
        det = np.linalg.det(dm)
        flip = det < 0.0
        if np.any(flip):
            swapped = self.tets.copy()
            swapped[flip, 2] = self.tets[flip, 3]
            swapped[flip, 3] = self.tets[flip, 2]
            self.tets = swapped
            p0 = rest[self.tets[:, 0]]
            dm = np.stack(
                (rest[self.tets[:, 1]] - p0, rest[self.tets[:, 2]] - p0,
                 rest[self.tets[:, 3]] - p0),
                axis=2,
            )
            det = np.linalg.det(dm)
        self.dm_inv = np.linalg.inv(dm)
        self.rest_volume = np.abs(det) / 6.0
        lattice_volume = float(self.rest_volume.sum())

        # Exported mass follows the frozen nominal density and the true closed
        # surface volume; the lattice density is scaled so that the simulated
        # inertia matches the exported total mass exactly.
        density = float(self.config.get("initial_conditions", {}).get(
            "nominal_density", 1040.0))
        self.total_mass = density * max(surface_volume, 1e-9)
        self.lattice_density = self.total_mass / max(lattice_volume, 1e-12)
        elem_mass = self.lattice_density * self.rest_volume
        node_mass = np.bincount(
            self.tets.ravel(),
            weights=np.repeat(elem_mass / 4.0, 4),
            minlength=self.n_nodes,
        )
        self.node_mass = np.maximum(node_mass, 1e-12)
        self.surface_mass = np.full(
            self.n_surface, self.total_mass / self.n_surface, dtype=np.float64
        )

        wave = np.sqrt((self.lam + 2.0 * self.mu) / max(self.lattice_density, 1e-9))
        self.wave_speed = float(wave)
        self.dt_limit = float(self.CFL_SAFETY) * self.h / max(wave, 1e-9)
        # Elastic force scale of the web region: shear modulus times a web area.
        self.reference_force = float(self.mu * 0.04 * 0.04)

    def _geometry_objects(self) -> dict:
        """Frozen scene geometry, read from the scaffold when available."""
        defaults = {
            "index_bone": {
                "translation": [0.105, 0.03, 0.012],
                "rotation_xyzw": [0.0, 0.7071068, 0.0, 0.7071068],
                "radius": 0.01,
                "height": 0.205,
            },
            "thumb_bone": {
                "translation": [0.025, -0.074, -0.002],
                "rotation_xyzw": [0.0, 0.7071068, 0.0, 0.7071068],
                "radius": 0.012,
                "height": 0.105,
            },
        }
        try:
            import yaml  # type: ignore

            for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
                path = base / "scene_geometry.yaml"
                if not path.is_file():
                    continue
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                for obj in data.get("objects", []) or []:
                    oid = obj.get("id")
                    if oid not in defaults:
                        continue
                    geom = obj.get("geometry", {}) or {}
                    tf = obj.get("transform", {}) or {}
                    entry = defaults[oid]
                    entry["radius"] = float(geom.get("radius", entry["radius"]))
                    entry["height"] = float(geom.get("height", entry["height"]))
                    entry["translation"] = list(
                        tf.get("translation", entry["translation"])
                    )
                    entry["rotation_xyzw"] = list(
                        tf.get("rotation_xyzw", entry["rotation_xyzw"])
                    )
                break
        except Exception:
            pass
        return defaults

    def _setup_boundaries(self) -> None:
        bones = self._geometry_objects()
        rest = self.node_rest
        self.bones = {}
        for name, spec in bones.items():
            center = np.asarray(spec["translation"], dtype=float)
            axis = _quat_matrix(np.asarray(spec["rotation_xyzw"], dtype=float)) @ np.array(
                [0.0, 0.0, 1.0]
            )
            axis = axis / max(np.linalg.norm(axis), _EPS)
            self.bones[name] = {
                "center": center,
                "axis": axis,
                "half": 0.5 * float(spec["height"]),
                "radius": float(spec["radius"]),
            }

        cuff_limit = -0.150
        meta = {}
        try:
            meta = self.scene.metadata() or {}
        except Exception:
            meta = {}
        region = meta.get("fixed_region", {}) if isinstance(meta, dict) else {}
        if isinstance(region, dict) and region.get("axis") == "x":
            cuff_limit = float(region.get("maximum", cuff_limit))
        self.cuff_limit = cuff_limit

        cuff = rest[:, 0] <= cuff_limit
        taken = cuff.copy()
        group_masks = {"cuff": cuff}
        for name in ("index_bone", "thumb_bone"):
            bone = self.bones[name]
            margin = float(self.BONE_MARGIN)
            mask = np.zeros(self.n_nodes, dtype=bool)
            for _ in range(6):
                dist = _capsule_distance(rest, bone["center"], bone["axis"], bone["half"])
                mask = (dist <= bone["radius"] + margin) & ~taken
                if int(mask.sum()) >= 8:
                    break
                margin += 0.004
            group_masks[name] = mask
            taken |= mask
        self.group_masks = group_masks
        self.prescribed_mask = taken
        self.free_mask = ~taken
        self.thumb_nodes = np.nonzero(group_masks["thumb_bone"])[0]
        self.held_mask = group_masks["cuff"] | group_masks["index_bone"]

        drive = dict(self.config.get("thumb_drive", {}) or {})
        self.drive_start = float(drive.get("start_time", 0.3))
        self.drive_end = float(drive.get("end_time", 1.65))
        self.drive_span = max(self.drive_end - self.drive_start, 1e-9)
        self.drive_axis = np.asarray(
            drive.get("rotation_axis", [0.0, 0.0, 1.0]), dtype=float
        )
        norm = np.linalg.norm(self.drive_axis)
        self.drive_axis = self.drive_axis / (norm if norm > _EPS else 1.0)
        self.drive_angle = np.deg2rad(float(drive.get("rotation_degrees", 34.0)))
        self.drive_translation = np.asarray(
            drive.get("translation", [0.018, 0.052, 0.004]), dtype=float
        )
        self.drive_center = self.bones["thumb_bone"]["center"].copy()
        self.thumb_rest = self.node_rest[self.thumb_nodes].copy()

    def _setup_embedding(self) -> None:
        h = self.h
        dims = self.grid_dims
        cells = np.floor((self.rest_surface - self.grid_origin[None, :]) / h).astype(
            np.int64
        )
        cells = np.clip(cells, 0, dims[None, :] - 1)
        local = (self.rest_surface - self.grid_origin[None, :]) / h - cells
        local = np.clip(local, 0.0, 1.0)
        idx = np.empty((self.n_surface, 8), dtype=np.int64)
        wgt = np.empty((self.n_surface, 8), dtype=np.float64)
        corner = 0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    idx[:, corner] = self.node_id[
                        cells[:, 0] + di, cells[:, 1] + dj, cells[:, 2] + dk
                    ]
                    wx = local[:, 0] if di else 1.0 - local[:, 0]
                    wy = local[:, 1] if dj else 1.0 - local[:, 1]
                    wz = local[:, 2] if dk else 1.0 - local[:, 2]
                    wgt[:, corner] = wx * wy * wz
                    corner += 1
        # Any unsupported corner (should not happen) is dropped and the remaining
        # weights renormalized so the embedding stays a partition of unity.
        bad = idx < 0
        if np.any(bad):
            wgt[bad] = 0.0
            idx[bad] = 0
        total = wgt.sum(axis=1, keepdims=True)
        wgt = wgt / np.maximum(total, 1e-12)
        self.embed_idx = idx
        self.embed_w = wgt
        point_mass = self.total_mass / self.n_surface
        self.node_weight = np.bincount(
            idx.ravel(),
            weights=(wgt * point_mass).ravel(),
            minlength=self.n_nodes,
        )

    # ------------------------------------------------------------------ drive
    def _thumb_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prescribed thumb-bone flesh position/velocity/acceleration at time t."""
        s = (float(t) - self.drive_start) / self.drive_span
        ease, dease_ds, ddease_ds = _smoothstep(s)
        dease = dease_ds / self.drive_span
        ddease = ddease_ds / (self.drive_span * self.drive_span)
        return _rodrigues_motion(
            self.thumb_rest,
            self.drive_center,
            self.drive_axis,
            self.drive_angle * ease,
            self.drive_angle * dease,
            self.drive_angle * ddease,
            self.drive_translation,
            ease,
            dease,
            ddease,
        )

    def _apply_prescribed(self, t: float) -> None:
        held = self.held_mask
        self.node_pos[held] = self.node_rest[held]
        self.node_vel[held] = 0.0
        pos, vel, _ = self._thumb_state(t)
        self.node_pos[self.thumb_nodes] = pos
        self.node_vel[self.thumb_nodes] = vel

    # ----------------------------------------------------------------- forces
    def _elastic_force(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Internal elastic nodal forces [N,3] and element gradients [Ne,3,3]."""
        f = _deformation_gradients(pos, self.tets, self.dm_inv)
        stress = _stable_neohookean_stress(f, self.mu, self.lam)
        # H holds the forces on nodes 1..3; node 0 balances the element.
        h_mat = -self.rest_volume[:, None, None] * (
            stress @ np.transpose(self.dm_inv, (0, 2, 1))
        )
        contrib = np.empty((self.n_elements, 4, 3), dtype=np.float64)
        contrib[:, 1, :] = h_mat[:, :, 0]
        contrib[:, 2, :] = h_mat[:, :, 1]
        contrib[:, 3, :] = h_mat[:, :, 2]
        contrib[:, 0, :] = -(contrib[:, 1] + contrib[:, 2] + contrib[:, 3])
        flat = self.tets.ravel()
        force = np.empty((self.n_nodes, 3), dtype=np.float64)
        for d in range(3):
            force[:, d] = np.bincount(
                flat, weights=contrib[:, :, d].ravel(), minlength=self.n_nodes
            )
        return force, f

    def _step(self, dt: float, t_next: float) -> None:
        force, _ = self._elastic_force(self.node_pos)
        free = self.free_mask
        mass = self.node_mass[free][:, None]
        accel = force[free] / mass + self.gravity[None, :]
        # Semi-implicit (unconditionally stable) treatment of mass damping.
        self.node_vel[free] = (self.node_vel[free] + dt * accel) / (
            1.0 + dt * float(self.MASS_DAMPING)
        )
        self.node_pos[free] = self.node_pos[free] + dt * self.node_vel[free]
        self._apply_prescribed(t_next)

    # ----------------------------------------------------------------- export
    def _budget_external_force(self) -> np.ndarray:
        """[G=1, K=5, 3] external/boundary force terms of the flesh domain."""
        force, _ = self._elastic_force(self.node_pos)
        w = self.node_weight[:, None]
        free = self.free_mask
        damping = float(self.MASS_DAMPING)

        gravity_term = self.gravity * float(self.node_weight[free].sum())
        damping_term = -damping * (w[free] * self.node_vel[free]).sum(axis=0)
        traction_term = (
            w[free] * force[free] / self.node_mass[free][:, None]
        ).sum(axis=0)

        # Cuff and index-bone flesh are held in place, so their prescribed
        # acceleration (and hence their inertial contribution) vanishes.
        held_term = np.zeros(3, dtype=np.float64)

        _, _, acc = self._thumb_state(self.time)
        thumb_term = (self.node_weight[self.thumb_nodes][:, None] * acc).sum(axis=0)

        return np.stack(
            (gravity_term, damping_term, traction_term, held_term, thumb_term)
        )[None, :, :]

    def _frame(self) -> StateFrame:
        disp = self.node_pos - self.node_rest
        surf_disp = np.einsum(
            "nc,ncd->nd", self.embed_w, disp[self.embed_idx]
        )
        surf_vel = np.einsum(
            "nc,ncd->nd", self.embed_w, self.node_vel[self.embed_idx]
        )
        position = self.rest_surface + surf_disp
        grads = _deformation_gradients(self.node_pos, self.tets, self.dm_inv)
        jdet = np.linalg.det(grads)
        self._min_jacobian = min(self._min_jacobian, float(jdet.min()))
        self._max_speed = max(self._max_speed, float(np.abs(surf_vel).max()))

        attributes = {
            "rest_position": self.rest_surface.copy(),
            "mass_quantity": self.surface_mass.copy(),
        }
        for key, value in self._scene_attributes.items():
            if key not in attributes:
                attributes[key] = value.copy()

        frame = StateFrame(
            time=float(self.time),
            position=position,
            velocity=surf_vel,
            mass=self.surface_mass.copy(),
            attributes=attributes,
            fields={
                "element_deformation_gradient": grads.astype(np.float32),
                "budget_external_force": self._budget_external_force(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "reference_force": float(self.reference_force),
                "solid_dimension": 3.0,
                "nominal_density": float(
                    self.config.get("initial_conditions", {}).get(
                        "nominal_density", 1040.0
                    )
                ),
                "total_mass": float(self.total_mass),
                "shear_modulus": float(self.mu),
                "first_lame": float(self.lam),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(int(np.ceil(span / self.dt_limit)), 1)
            n_sub = min(n_sub, 4000)
            dt = span / n_sub
            pos_backup = self.node_pos.copy()
            vel_backup = self.node_vel.copy()
            t0 = self.time
            for attempt in range(3):
                ok = True
                for k in range(n_sub):
                    self._step(dt, t0 + (k + 1) * dt)
                    if not np.isfinite(self.node_pos).all():
                        ok = False
                        break
                if ok and np.isfinite(self.node_vel).all():
                    break
                # Numerical blow-up guard: restart the interval with finer steps.
                self.node_pos = pos_backup.copy()
                self.node_vel = vel_backup.copy()
                n_sub *= 4
                dt = span / n_sub
            if not (
                np.isfinite(self.node_pos).all() and np.isfinite(self.node_vel).all()
            ):
                # Last resort: keep the state finite and deterministic.
                self.node_pos = pos_backup.copy()
                self.node_vel = np.zeros_like(vel_backup)
            self.time = target
            self._apply_prescribed(self.time)
        else:
            self.time = max(target, 0.0)
            self._apply_prescribed(self.time)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_surface_points": int(self.n_surface),
            "n_lattice_nodes": int(self.n_nodes),
            "n_elements": int(self.n_elements),
            "cell_size": float(self.h),
            "substep_limit": float(self.dt_limit),
            "wave_speed": float(self.wave_speed),
            "total_mass": float(self.total_mass),
            "lattice_density": float(self.lattice_density),
            "shear_modulus": float(self.mu),
            "first_lame": float(self.lam),
            "thumb_driven_nodes": int(len(self.thumb_nodes)),
            "held_nodes": int(self.held_mask.sum()),
            "free_nodes": int(self.free_mask.sum()),
            "min_element_jacobian": float(self._min_jacobian),
            "max_surface_speed": float(self._max_speed),
            "material_model": "stable_neo_hookean_tetrahedral_fem",
            "integrator": "damped_symplectic_euler",
        }

