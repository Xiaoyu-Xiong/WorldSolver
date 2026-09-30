"""Incompressible FLIP liquid solver for the U-corridor dam break scene.

Written from scratch on NumPy/SciPy:

* marker particles carrying mass and velocity (FLIP/PIC transfer),
* staggered MAC velocity grid, gravity body force,
* exact pressure projection via a sparse symmetric Poisson solve with p = 0
  Dirichlet conditions in air cells and no-flux Neumann conditions on solid
  faces (so the represented liquid is divergence free and does not leak),
* CFL limited explicit stepping with RK2 particle advection,
* frozen rigid geometry (ground, corridor walls, dam gate) enforced on the grid
  through a solid-cell mask and on the particles through analytic box push-out
  plus a swept test that prevents tunnelling through the thin gate.

The confined column collapses under gravity, drains through the corridor legs
beside the dam gate, runs up the far end wall, and returns as interacting waves.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt

from worldsolver.contracts import StateFrame

_OFF_U = np.array([0.0, 0.5, 0.5])
_OFF_V = np.array([0.5, 0.0, 0.5])
_OFF_W = np.array([0.5, 0.5, 0.0])
_OFF_C = np.array([0.5, 0.5, 0.5])


class BenchmarkSolver:
    CFL = 0.55
    FLIP_BLEND = 0.94
    DT_MAX = 0.01
    DT_MIN = 2.0e-4
    MAX_SPEED = 20.0
    SUBSTEPS = 2
    DENSITY_STIFFNESS = 0.05
    MAX_BOUNDARY_SAMPLES = 1800

    # ----------------------------------------------------------------- set-up
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        frame0 = scene.initial_state()
        sampling = dict(cfg.get("sampling") or {})
        self.spacing = float(sampling.get("spacing", 0.14))
        self.density = float(sampling.get("density", 1000.0))
        self.blocks = [list(map(float, b)) for b in (sampling.get("blocks") or [])]
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        ref_pos = np.asarray(frame0.position, float)
        if frame0.mass is not None:
            self.total_mass = float(np.sum(np.asarray(frame0.mass, float)))
        else:
            self.total_mass = float(len(ref_pos) * self.density * self.spacing ** 3)
        self._build_geometry(cfg, ref_pos)
        self._build_grid()
        self._build_solid()
        self._seed_particles(ref_pos)
        self._build_boundary_samples()
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.q = np.zeros((self.nx, self.ny, self.nz))
        self.fluid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        self.count = np.zeros((self.nx, self.ny, self.nz))
        self.time = 0.0
        self.steps = 0
        self.char_len = max(float(np.ptp(ref_pos, axis=0).max()), self.spacing)
        head = max(float(ref_pos[:, 2].max()) - float(self.lo[2]), self.spacing)
        self.ref_vel = float(np.sqrt(2.0 * abs(float(self.gravity[2])) * head))
    # ------------------------------------------------------- frozen geometry
    def _build_geometry(self, cfg, ref_pos):
        """Collect axis-aligned collision boxes and the open simulation domain."""
        geom = cfg.get("scene_geometry") or {}
        objects = geom.get("objects") or []
        boxes = []          # (lo, hi, id) analytic solid boxes
        floors = []
        subject_bounds = None
        for obj in objects:
            gspec = dict(obj.get("geometry") or {})
            tr = dict(obj.get("transform") or {})
            trans = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), float)
            scale = np.asarray(tr.get("scale", [1.0, 1.0, 1.0]), float)
            gtype = gspec.get("type")
            if gtype == "box" and bool(obj.get("collision_enabled", True)) and \
                    obj.get("semantic") in ("boundary", "rigid"):
                he = np.asarray(gspec.get("half_extents", [0.0, 0.0, 0.0]), float) * scale
                lo, hi = trans - he, trans + he
                ext = hi - lo
                flat = ext[2] <= 0.25 * max(ext[0], ext[1]) and hi[2] <= 0.2
                boxes.append((lo, hi, str(obj.get("id", "solid"))))
                if flat:
                    floors.append((lo, hi))
            elif gtype in ("particle_domain", "voxel_grid"):
                bmin = gspec.get("bounds_min")
                bmax = gspec.get("bounds_max")
                if bmin is not None and bmax is not None:
                    subject_bounds = (np.asarray(bmin, float) * scale + trans,
                                      np.asarray(bmax, float) * scale + trans)
        self.boxes_analytic = boxes
        floor_set = {id(f) for f in floors}
        # Domain: enclosing box of the walls / gates plus the liquid region.
        mins, maxs = [], []
        for lo, hi, _ in boxes:
            ext = hi - lo
            if ext[2] <= 0.25 * max(ext[0], ext[1]) and hi[2] <= 0.2:
                continue  # skip the wide floor slab
            mins.append(lo)
            maxs.append(hi)
        for b in self.blocks:
            mins.append(np.asarray(b[:3], float))
            maxs.append(np.asarray(b[3:], float))
        if subject_bounds is not None:
            mins.append(subject_bounds[0])
            maxs.append(subject_bounds[1])
        if ref_pos.size:
            mins.append(ref_pos.min(axis=0) - 0.5 * self.spacing)
            maxs.append(ref_pos.max(axis=0) + 0.5 * self.spacing)
        if not mins:
            mins = [np.array([-1.0, -1.0, 0.0])]
            maxs = [np.array([1.0, 1.0, 1.0])]
        dom_lo = np.min(np.asarray(mins, float), axis=0)
        dom_hi = np.max(np.asarray(maxs, float), axis=0)
        floor_top = max((float(hi[2]) for lo, hi in floors), default=float(dom_lo[2]))
        dom_lo[2] = floor_top
        if dom_hi[2] <= dom_lo[2] + self.spacing:
            dom_hi[2] = dom_lo[2] + 2.0 * self.spacing
        self.dom_lo = dom_lo
        self.dom_hi = dom_hi
        _ = floor_set

    def _build_grid(self) -> None:
        self.dx = float(self.spacing / 2.8)
        while True:
            lo = self.dom_lo.copy()
            lo[2] -= self.dx                  # one solid ground layer
            n = np.maximum(np.ceil((self.dom_hi - lo) / self.dx - 1e-9).astype(int), 4)
            if int(np.prod(n)) <= 300000:
                break
            self.dx *= 1.2
        self.lo = lo
        self.nx, self.ny, self.nz = int(n[0]), int(n[1]), int(n[2])
        self.shape = (self.nx, self.ny, self.nz)
        self.hi = lo + n * self.dx
        self.ushape = (self.nx + 1, self.ny, self.nz)
        self.vshape = (self.nx, self.ny + 1, self.nz)
        self.wshape = (self.nx, self.ny, self.nz + 1)

    def _build_solid(self) -> None:
        """Conservative solid-cell mask: a cell is solid if any rigid box overlaps it."""
        solid = np.zeros(self.shape, dtype=bool)
        edges = [self.lo[d] + np.arange(self.shape[d] + 1) * self.dx for d in range(3)]
        for lo_b, hi_b, _name in self.boxes_analytic:
            masks = []
            for d in range(3):
                cell_lo = edges[d][:-1]
                cell_hi = edges[d][1:]
                masks.append((cell_hi > lo_b[d] + 1e-9) & (cell_lo < hi_b[d] - 1e-9))
            solid |= masks[0][:, None, None] & masks[1][None, :, None] & masks[2][None, None, :]
        self.solid = solid
        # Solid faces: a face is solid when either adjacent cell (outside = solid) is solid.
        pad = np.ones((self.nx + 2, self.ny + 2, self.nz + 2), dtype=bool)
        pad[1:-1, 1:-1, 1:-1] = solid
        self.solid_pad = pad
        self.usolid = pad[:-1, 1:-1, 1:-1] | pad[1:, 1:-1, 1:-1]
        self.vsolid = pad[1:-1, :-1, 1:-1] | pad[1:-1, 1:, 1:-1]
        self.wsolid = pad[1:-1, 1:-1, :-1] | pad[1:-1, 1:-1, 1:]

    # ------------------------------------------------------------- particles
    def _inside_solid(self, pos, margin=0.0):
        inside = np.zeros(len(pos), dtype=bool)
        for lo_b, hi_b, _name in self.boxes_analytic:
            inside |= np.all((pos > lo_b - margin) & (pos < hi_b + margin), axis=1)
        return inside

    def _seed_particles(self, ref_pos) -> None:
        h = self.dx / 1.6
        rng = np.random.default_rng(self.seed)
        clouds = []
        blocks = self.blocks if self.blocks else []
        if not blocks and ref_pos.size:
            blocks = [list(ref_pos.min(axis=0) - 0.5 * self.spacing)
                      + list(ref_pos.max(axis=0) + 0.5 * self.spacing)]
        for b in blocks:
            lo_b = np.asarray(b[:3], float)
            hi_b = np.asarray(b[3:], float)
            axes = []
            for d in range(3):
                count = max(int(np.floor((hi_b[d] - lo_b[d]) / h)), 1)
                step = (hi_b[d] - lo_b[d]) / count
                axes.append(lo_b[d] + (np.arange(count) + 0.5) * step)
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            pts = pts + (rng.random(pts.shape) - 0.5) * (0.35 * h)
            clouds.append(pts)
        pos = np.concatenate(clouds) if clouds else ref_pos.copy()
        keep = ~self._inside_solid(pos)
        keep &= np.all(pos > self.lo + 1e-4, axis=1) & np.all(pos < self.hi - 1e-4, axis=1)
        if not np.any(keep):
            keep = np.ones(len(pos), dtype=bool)
        self.pos = np.ascontiguousarray(pos[keep], dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.n_particles = int(len(self.pos))
        self.p_mass = self.total_mass / max(self.n_particles, 1)
        n_out = int(min(self.n_particles, 9000))
        self.export_idx = np.unique(
            np.linspace(0, self.n_particles - 1, n_out).astype(np.int64))
        self.n_export = int(len(self.export_idx))
        self.export_mass = np.full(self.n_export, self.total_mass / self.n_export)
        self.export_labels = np.zeros(self.n_export, dtype=np.int32)
        self.export_radius = np.full(self.n_export, 0.45 * self.spacing)

    def _build_boundary_samples(self) -> None:
        pad = self.solid_pad
        pos_list, nrm_list = [], []
        # x faces
        left = pad[:-1, 1:-1, 1:-1]
        right = pad[1:, 1:-1, 1:-1]
        sel = np.argwhere(left != right)
        if sel.size:
            p = self.lo + np.column_stack((sel[:, 0] * self.dx,
                                           (sel[:, 1] + 0.5) * self.dx,
                                           (sel[:, 2] + 0.5) * self.dx))
            sgn = np.where(left[sel[:, 0], sel[:, 1], sel[:, 2]], 1.0, -1.0)
            nrm = np.zeros((len(sel), 3))
            nrm[:, 0] = sgn
            pos_list.append(p)
            nrm_list.append(nrm)
        # y faces
        low = pad[1:-1, :-1, 1:-1]
        high = pad[1:-1, 1:, 1:-1]
        sel = np.argwhere(low != high)
        if sel.size:
            p = self.lo + np.column_stack(((sel[:, 0] + 0.5) * self.dx,
                                           sel[:, 1] * self.dx,
                                           (sel[:, 2] + 0.5) * self.dx))
            sgn = np.where(low[sel[:, 0], sel[:, 1], sel[:, 2]], 1.0, -1.0)
            nrm = np.zeros((len(sel), 3))
            nrm[:, 1] = sgn
            pos_list.append(p)
            nrm_list.append(nrm)
        # z faces
        low = pad[1:-1, 1:-1, :-1]
        high = pad[1:-1, 1:-1, 1:]
        sel = np.argwhere(low != high)
        if sel.size:
            p = self.lo + np.column_stack(((sel[:, 0] + 0.5) * self.dx,
                                           (sel[:, 1] + 0.5) * self.dx,
                                           sel[:, 2] * self.dx))
            sgn = np.where(low[sel[:, 0], sel[:, 1], sel[:, 2]], 1.0, -1.0)
            nrm = np.zeros((len(sel), 3))
            nrm[:, 2] = sgn
            pos_list.append(p)
            nrm_list.append(nrm)
        if pos_list:
            bpos = np.concatenate(pos_list)
            bnrm = np.concatenate(nrm_list)
        else:
            bpos = np.zeros((1, 3))
            bnrm = np.zeros((1, 3))
            bnrm[0, 2] = 1.0
        if len(bpos) > self.MAX_BOUNDARY_SAMPLES:
            pick = np.linspace(0, len(bpos) - 1, self.MAX_BOUNDARY_SAMPLES).astype(np.int64)
            pick = np.unique(pick)
            bpos, bnrm = bpos[pick], bnrm[pick]
        self.b_pos = np.ascontiguousarray(bpos)
        self.b_normal = np.ascontiguousarray(bnrm)
        self.b_target = np.zeros_like(self.b_normal)   # all frozen walls are stationary
# __TAIL__
