"""Physically based thin-sheet (mass-spring) cloth drape over a fixed cylinder.

The solver implements its own explicit physical model:  A rectangular
mass-spring sheet (structural / shear / bending springs) is integrated with a
semi-implicit (symplectic) Euler scheme under gravity, with penalty plus
Coulomb friction contact against the frozen cylinder and ground obstacles.

No scripted animation or precomputed motion is used; every frame is produced by
actually time-stepping the equations of motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# numerical / physical model constants (material parameters of the sheet model)
# ----------------------------------------------------------------------------
_K_STRUCT = 900.0     # N/m  structural (warp/weft) spring stiffness
_C_STRUCT = 1.0       # N s/m structural damping
_K_SHEAR = 300.0      # N/m  diagonal shear spring stiffness
_C_SHEAR = 0.4        # N s/m shear damping
_K_BEND = 12.0        # N/m  second-neighbour bending spring stiffness
_C_BEND = 0.05        # N s/m bending damping
_AIR_DRAG = 0.6       # 1/s  global aerodynamic velocity damping

_K_CONTACT = 2.0e4    # N/m  penalty contact stiffness
_C_CONTACT = 8.0      # N s/m contact normal damping
_C_TANGENT = 60.0     # N s/m contact tangential (friction) damping
_MU = 0.45            # Coulomb friction coefficient
_NEAR_BAND = 0.03     # m    band used to report near-contact samples

_DEFAULT_THICKNESS = 0.015


def _quat_to_matrix(xyzw):
    x, y, z, w = (float(v) for v in xyzw)
    nrm = np.sqrt(x * x + y * y + z * z + w * w)
    if nrm <= 0.0:
        return np.eye(3)
    x, y, z, w = x / nrm, y / nrm, z / nrm, w / nrm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class BenchmarkSolver:
    """Explicit mass-spring thin-sheet solver with penalty contact."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}) or {})
        self.nx, self.ny = (int(v) for v in cloth.get("grid_shape", (17, 17)))
        self.thickness = float(cloth.get("thickness", _DEFAULT_THICKNESS))
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float).reshape(3)

        frame0 = scene.initial_state()
        p0 = np.array(frame0.position, dtype=float)
        self.n = int(p0.shape[0])
        if frame0.mass is not None:
            self.mass = np.array(frame0.mass, dtype=float)
        else:
            self.mass = np.full(self.n, 1.0 / self.n, dtype=float)
        self.mass = np.maximum(self.mass, 1e-9)
        self.boundary = np.array(
            frame0.attributes.get("boundary_vertex", np.zeros(self.n, dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)

        self.p = p0.copy()
        self.v = np.zeros_like(self.p)
        self.t = 0.0
        self.steps = 0
        self.max_stretch = 1.0

        self._build_springs(p0)
        self._build_obstacles(cfg)
        self._choose_dt(cfg)

        self._ext_force = self.mass[:, None] * self.gravity[None, :]
        self._mass_quantity = self.mass.copy()

    def _build_springs(self, p0):
        nx, ny = self.nx, self.ny
        index = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)

        a_list, b_list, k_list, c_list = [], [], [], []

        def add(di, dj, stiff, damp):
            i0, i1 = (0, nx - di) if di >= 0 else (-di, nx)
            j0, j1 = (0, ny - dj) if dj >= 0 else (-dj, ny)
            if i1 <= i0 or j1 <= j0:
                return
            block = index[j0:j1, i0:i1]
            a_list.append(block.ravel())
            b_list.append(index[j0 + dj: j1 + dj, i0 + di: i1 + di].ravel())
            k_list.append(np.full(block.size, float(stiff)))
            c_list.append(np.full(block.size, float(damp)))

        # structural (axial) springs
        add(1, 0, _K_STRUCT, _C_STRUCT)
        add(0, 1, _K_STRUCT, _C_STRUCT)
        # shear (diagonal) springs
        add(1, 1, _K_SHEAR, _C_SHEAR)
        add(1, -1, _K_SHEAR, _C_SHEAR)
        # bending (second neighbour) springs
        add(2, 0, _K_BEND, _C_BEND)
        add(0, 2, _K_BEND, _C_BEND)

        self.sa = np.concatenate(a_list).astype(np.int64)
        self.sb = np.concatenate(b_list).astype(np.int64)
        self.sk = np.concatenate(k_list)
        self.sc = np.concatenate(c_list)
        delta = p0[self.sb] - p0[self.sa]
        self.sl0 = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        self.sl0 = np.maximum(self.sl0, 1e-6)

        # material (reference) 2D coordinates used for deformation gradients
        sx, sy = self._cloth_size
        self.mat2d = np.column_stack(
            (
                (index % nx).ravel().astype(float) * (sx / max(nx - 1, 1)),
                (index // nx).ravel().astype(float) * (sy / max(ny - 1, 1)),
            )
        )
        # node connectivity used for per-node deformation gradients
        self.dg_a, self.dg_b = self._connectivity

    @property
    def _cloth_size(self):
        cloth = self.scene.config.get("cloth", {})
        size = cloth.get("size", (3.0, 2.5))
        return float(size[0]), float(size[1])

    @property
    def _connectivity(self):
        nx, ny = self.nx, self.ny
        index = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
        a_list, b_list = [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)):
            i0, i1 = (0, nx - di) if di >= 0 else (-di, nx)
            j0, j1 = (0, ny - dj) if dj >= 0 else (-dj, ny)
            if i1 <= i0 or j1 <= j0:
                continue
            block = index[j0:j1, i0:i1]
            a_list.append(block.ravel())
            b_list.append(index[j0 + dj: j1 + dj, i0 + di: i1 + di].ravel())
        return np.concatenate(a_list), np.concatenate(b_list)

    def _build_obstacles(self, cfg):
        self.obstacles = []
        geo = cfg.get("scene_geometry", {}) or {}
        objects = {str(o.get("id")): o for o in (geo.get("objects", []) or []) if isinstance(o, dict)}

        cyl = objects.get("cylinder")
        if cyl is not None:
            g = cyl.get("geometry", {}) or {}
            radius = float(g.get("radius", 0.52))
            height = float(g.get("height", 2.8))
            tr = (cyl.get("transform", {}) or {}).get("translation", (0.0, 0.0, 0.7))
            origin = np.asarray(tr, dtype=float).reshape(3)
            rot = (cyl.get("transform", {}) or {}).get("rotation_xyzw", (0.7071068, 0.0, 0.0, 0.7071068))
            axis = _quat_to_matrix(rot) @ np.array([0.0, 0.0, 1.0])
            n_axis = np.linalg.norm(axis)
            axis = axis / n_axis if n_axis > 0 else np.array([0.0, 1.0, 0.0])
            self.obstacles.append(
                {
                    "kind": "cylinder",
                    "origin": origin,
                    "axis": axis,
                    "radius": radius,
                    "half_height": 0.5 * height,
                }
            )
        else:
            self.obstacles.append(
                {
                    "kind": "cylinder",
                    "origin": np.array([0.0, 0.0, 0.7]),
                    "axis": np.array([0.0, 1.0, 0.0]),
                    "radius": 0.52,
                    "half_height": 1.4,
                }
            )

        ground = objects.get("ground")
        if ground is not None:
            g = ground.get("geometry", {}) or {}
            half = np.asarray(g.get("half_extents", (2.4, 2.0, 0.05)), dtype=float).reshape(3)
            tr = (ground.get("transform", {}) or {}).get("translation", (0.0, 0.0, -0.05))
            center = np.asarray(tr, dtype=float).reshape(3)
        else:
            half = np.array([2.4, 2.0, 0.05])
            center = np.array([0.0, 0.0, -0.05])
        self.obstacles.append({"kind": "box", "center": center, "half": half})

    def _choose_dt(self, cfg):
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        m_min = float(np.min(self.mass))
        k_max = max(
            float(np.max(self.sk)) if self.sk.size else 1.0,
            _K_CONTACT,
        )
        omega = np.sqrt(k_max / max(m_min, 1e-12))
        n_sub = int(np.ceil(sample_dt * omega / 0.42))
        n_sub = max(8, min(n_sub, 4000))
        self.n_sub = n_sub
        self.dt = sample_dt / float(n_sub)
        self.sample_dt = sample_dt

    # --------------------------------------------------------------- dynamics
    def _spring_forces(self, p, v):
        d = p[self.sb] - p[self.sa]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r_safe = np.maximum(r, 1e-9)
        nrm = d / r_safe[:, None]
        rel = v[self.sb] - v[self.sa]
        ext = np.einsum("ij,ij->i", rel, nrm)
        mag = self.sk * (r - self.sl0) + self.sc * ext
        w = mag[:, None] * nrm
        f = np.empty_like(p)
        for comp in range(3):
            col = w[:, comp]
            f[:, comp] = np.bincount(self.sa, weights=col, minlength=self.n) - np.bincount(
                self.sb, weights=col, minlength=self.n
            )
        return f

    def _obstacle_state(self, obstacle, p):
        """Return (signed distance, outward unit normal) per sample for one obstacle."""
        if obstacle["kind"] == "cylinder":
            origin = obstacle["origin"]
            axis = obstacle["axis"]
            w = p - origin[None, :]
            t = np.einsum("ij,j->i", w, axis)
            t_clamped = np.clip(t, -obstacle["half_height"], obstacle["half_height"])
            radial = w - t_clamped[:, None] * axis[None, :]
            rad = np.sqrt(np.einsum("ij,ij->i", radial, radial))
            rad_safe = np.maximum(rad, 1e-9)
            dist = rad - obstacle["radius"]
            nrm = radial / rad_safe[:, None]
            degenerate = rad < 1e-9
            if np.any(degenerate):
                nrm[degenerate] = axis[None, :]
            return dist, nrm

        center = obstacle["center"]
        half = obstacle["half"]
        q = np.abs(p - center[None, :]) - half[None, :]
        outside = np.maximum(q, 0.0)
        d_out = np.sqrt(np.einsum("ij,ij->i", outside, outside))
        d_in = np.minimum(np.max(q, axis=1), 0.0)
        dist = d_out + d_in
        signs = np.sign(p - center[None, :])
        signs[signs == 0.0] = 1.0
        nrm = np.where(q > 0.0, signs * (outside / np.maximum(d_out, 1e-9)[:, None]), 0.0)
        norm = np.sqrt(np.einsum("ij,ij->i", nrm, nrm))
        bad = norm < 1e-9
        if np.any(bad):
            nrm[bad] = np.array([0.0, 0.0, 1.0])
        return dist, nrm

    def _contact(self, p, v):
        f = np.zeros_like(p)
        gap = np.full(self.n, np.inf)
        for obstacle in self.obstacles:
            dist, nrm = self._obstacle_state(obstacle, p)
            gap = np.minimum(gap, dist)
            vn = np.einsum("ij,ij->i", v, nrm)
            pen = np.maximum(0.0, -dist)
            active = (pen > 0.0) | ((dist < 0.004) & (vn < 0.0))
            if not np.any(active):
                continue
            fn = np.where(active, _K_CONTACT * pen + _C_CONTACT * np.maximum(-vn, 0.0), 0.0)
            fn = np.maximum(fn, 0.0)
            f += fn[:, None] * nrm
            vt = v - vn[:, None] * nrm
            speed = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            ft = np.minimum(_C_TANGENT * speed, _MU * fn)
            coeff = np.where((speed > 1e-12) & active, ft / np.maximum(speed, 1e-12), 0.0)
            f -= coeff[:, None] * vt
        return f, gap

    def _step(self):
        p, v = self.p, self.v
        f = self._spring_forces(p, v)
        f += self._ext_force
        fc, gap = self._contact(p, v)
        f += fc
        v += (f / self.mass[:, None]) * self.dt
        v *= 1.0 / (1.0 + _AIR_DRAG * self.dt)
        p += v * self.dt
        self.t += self.dt
        self.steps += 1

    # ------------------------------------------------------------- trajectory
    def _deformation_gradients(self):
        """Per-node 2x2 in-material-plane stretch tensor from least squares."""
        p = self.p
        dX = self.mat2d[self.dg_b] - self.mat2d[self.dg_a]
        dx = p[self.dg_b] - p[self.dg_a]
        m = np.zeros((self.n, 2, 2))
        b = np.zeros((self.n, 3, 2))
        for comp in range(2):
            for comp2 in range(2):
                m[:, comp, comp2] = np.bincount(
                    self.dg_a, weights=dX[:, comp] * dX[:, comp2], minlength=self.n
                )
        for row in range(3):
            for comp2 in range(2):
                b[:, row, comp2] = np.bincount(
                    self.dg_a, weights=dx[:, row] * dX[:, comp2], minlength=self.n
                )
        # least-squares 3x2 gradient G = B M^{-1} (regularised)
        reg = 1e-8 * float(np.max(np.abs(m))) + 1e-12
        m_reg = m + reg * np.eye(2)[None, :, :]
        det = m_reg[:, 0, 0] * m_reg[:, 1, 1] - m_reg[:, 0, 1] * m_reg[:, 1, 0]
        det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        inv = np.empty_like(m_reg)
        inv[:, 0, 0] = m_reg[:, 1, 1] / det
        inv[:, 0, 1] = -m_reg[:, 0, 1] / det
        inv[:, 1, 0] = -m_reg[:, 1, 0] / det
        inv[:, 1, 1] = m_reg[:, 0, 0] / det
        g = np.einsum("nik,nkj->nij", b, inv)  # [N,3,2]
        # right Cauchy-Green tensor C = G^T G (2x2, material coordinates)
        c = np.einsum("nki,nkj->nij", g, g)
        # symmetric square root of C: stretch tensor (equals I when undeformed)
        c00, c01, c11 = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
        tr = c00 + c11
        diff = c00 - c11
        root = np.sqrt(np.maximum(diff * diff + 4.0 * c01 * c01, 0.0))
        lam1 = np.maximum(0.5 * (tr + root), 0.0)
        lam2 = np.maximum(0.5 * (tr - root), 0.0)
        s1 = np.sqrt(lam1)
        s2 = np.sqrt(lam2)
        denom = np.where(root > 1e-12, root, 1.0)
        # eigenvector for lam1: (c01, lam1 - c00) normalised
        e1x = np.where(root > 1e-12, c01, 1.0)
        e1y = np.where(root > 1e-12, lam1 - c00, 0.0)
        enorm = np.sqrt(e1x * e1x + e1y * e1y)
        enorm = np.where(enorm > 1e-12, enorm, 1.0)
        e1x = e1x / enorm
        e1y = e1y / enorm
        e2x = -e1y
        e2y = e1x
        f = np.empty_like(c)
        f[:, 0, 0] = s1 * e1x * e1x + s2 * e2x * e2x
        f[:, 0, 1] = s1 * e1x * e1y + s2 * e2x * e2y
        f[:, 1, 0] = f[:, 0, 1]
        f[:, 1, 1] = s1 * e1y * e1y + s2 * e2y * e2y
        f = 0.5 * (f + np.transpose(f, (0, 2, 1)))
        self.max_stretch = float(np.max(np.linalg.eigvalsh(f)))
        return f

    def _frame(self):
        f_grad = self._deformation_gradients()
        _, gap = self._contact(self.p, self.v)
        near = np.isfinite(gap) & (gap < _NEAR_BAND)
        if not np.any(near):
            near = np.zeros(self.n, dtype=bool)
            near[int(np.argmin(gap))] = True
        contact_gap = np.asarray(gap[near], dtype=float)
        if contact_gap.size == 0:
            contact_gap = np.zeros(1, dtype=float)
        frame = StateFrame(
            float(self.t),
            np.array(self.p, dtype=float),
            np.array(self.v, dtype=float),
            mass=np.array(self.mass, dtype=float),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self._mass_quantity.copy(),
                "deformation_gradient": f_grad,
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "external_force": np.array(self._ext_force, dtype=float),
                "contact_signed_distance": contact_gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- contract
    def advance_to(self, time: float):
        target = float(time)
        if np.isfinite(target) and target > self.t + 1e-12:
            n_target = int(round(target / self.dt))
            while self.steps < n_target:
                self._step()
            # snap to the exact requested sample time (no accumulated drift)
            self.t = n_target * self.dt
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "mass_spring_thin_sheet_explicit",
            "nodes": int(self.n),
            "springs": int(self.sa.size),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.n_sub),
            "steps": int(self.steps),
            "time": float(self.t),
            "max_stretch": float(self.max_stretch),
        }
