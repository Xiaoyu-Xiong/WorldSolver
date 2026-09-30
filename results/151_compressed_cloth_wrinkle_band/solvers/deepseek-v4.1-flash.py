"""Physically based solver for the "Compressed Cloth Wrinkle Band" scene.

Model
-----
The frozen rectangular cloth (17x17 grid, 3.0 m x 2.5 m, 0.32 kg) is
discretised as a triangular mass-spring thin sheet with a real physical
time-stepping procedure (no external simulation engine is used):

* membrane  : axial (x / y) springs of stiffness per unit width ``Et`` plus
              weaker diagonal shear springs; Kelvin-Voigt viscous damping
              along each spring.
* bending   : discrete biharmonic (Laplacian-of-Laplacian) curvature energy
              with plate bending modulus ``D = E t^3 / (12 (1 - nu^2))``.
              A quadratic curvature energy (instead of soft distance
              "bending" springs) gives a genuine, mesh independent wrinkle
              wavelength.  Because the frozen mesh is coarse, the modulus is
              raised to the smallest value that keeps the fastest unstable
              wavelength above ~4 element sizes, which is exactly the
              anti mesh-locked-faceting requirement of the task.
* external  : uniform gravity taken from the frozen configuration.
* contact   : one-sided penalty with normal damping and Coulomb friction
              against the frozen ground box and the two kinematic grip
              boxes (which follow the gripped cloth edges inward).

The two opposite gripped edge columns are kinematic: their position and
velocity follow a C1 smoothstep grip motion ``|x| = 1.5 - d(t)``, which
places the sheet in compression (``~8%`` axial strain).  The flat state is
geometrically unstable in the out-of-plane direction for any compressive
load, so a band of wrinkles nucleates and grows, with a wavelength selected
by the ``D`` / membrane-tension balance.

Integration is explicit velocity-Verlet with a fixed step ``h = 1e-4 s``
(well inside the ``2/omega_max`` stability limit) plus a semi-implicit
linear drag.  Everything is deterministic and finite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------
# frozen fallback descriptions of the scene boxes (used if the geometry
# payload is not attached to the config)
# --------------------------------------------------------------------------
_FALLBACK_BOXES = (
    ("ground", (2.4, 2.0, 0.05), (0.0, 0.0, -0.05), False),
    ("grip_left", (0.12, 1.25, 0.12), (-1.5, 0.0, 2.15), True),
    ("grip_right", (0.12, 1.25, 0.12), (1.5, 0.0, 2.15), True),
)


class BenchmarkSolver:
    """Compressed cloth wrinkle band solver (mass-spring thin sheet)."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        cloth = cfg["cloth"]

        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.sx, self.sy = (float(v) for v in cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)

        nx, ny = self.nx, self.ny
        n = nx * ny
        self.n = n

        # ---- reference (rest) geometry, row-major i = iy*nx + ix ----------
        hx = self.sx / float(nx - 1)
        hy = self.sy / float(ny - 1)
        self.hx, self.hy = hx, hy
        xs = -0.5 * self.sx + hx * np.arange(nx)
        ys = -0.5 * self.sy + hy * np.arange(ny)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        self.ix = np.tile(np.arange(nx), ny)
        self.iy = np.repeat(np.arange(ny), nx)
        rest2d = np.column_stack((gx.ravel(), gy.ravel()))
        self.rest2d = rest2d
        faces = self._triangulation(n, nx, ny)

        # ---- masses -------------------------------------------------------
        self.mass = np.full(n, self.total_mass / float(n), dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.zeros(n, dtype=np.int32)

        # ---- material parameters -----------------------------------------
        # membrane stiffness per unit width  [N/m]  (E * t)
        self.Et = 2500.0
        self.nu = 0.30
        self.shear_ratio = 0.22
        # effective plate bending modulus  [N m]
        D_phys = self.Et * self.thickness ** 2 / (12.0 * (1.0 - self.nu ** 2))
        # coarse-mesh effective bending: keep shortest unstable wavelength
        # (2 pi sqrt(D / (Et * eps))) above ~4 element sizes.
        D_min = (4.0 * hx) ** 2 * (self.Et * 0.08) / (4.0 * np.pi ** 2)
        self.bending_modulus = float(max(D_phys, 1.0 * D_min))
        self.kappa_bx = self.bending_modulus * hy / hx ** 3
        self.kappa_by = self.bending_modulus * hx / hy ** 3
        self.zeta = 0.25          # spring damping ratio
        self.gamma = 0.8          # global linear drag  [1/s]

        # ---- springs ------------------------------------------------------
        si: list[int] = []
        sj: list[int] = []
        sl0: list[float] = []
        sk: list[float] = []
        for iy in range(ny):
            for ix in range(nx):
                i = iy * nx + ix
                if ix + 1 < nx:
                    si.append(i); sj.append(i + 1); sl0.append(hx); sk.append(self.Et)
                if iy + 1 < ny:
                    si.append(i); sj.append(i + nx); sl0.append(hy); sk.append(self.Et)
                if ix + 1 < nx and iy + 1 < ny:
                    d = np.hypot(hx, hy)
                    si.append(i); sj.append(i + nx + 1); sl0.append(d)
                    sk.append(self.Et * self.shear_ratio)
                    si.append(i + 1); sj.append(i + nx); sl0.append(d)
                    sk.append(self.Et * self.shear_ratio)
        self.si = np.asarray(si, dtype=np.int64)
        self.sj = np.asarray(sj, dtype=np.int64)
        self.sl0 = np.asarray(sl0, dtype=float)
        self.sk = np.asarray(sk, dtype=float)
        # Kelvin-Voigt damping coefficient c = 2 zeta sqrt(k m)
        self.sc = 2.0 * self.zeta * np.sqrt(self.sk * self.mass.mean())

        # ---- bending stencil supports ------------------------------------
        self._build_bending_supports()

        # ---- contacts -----------------------------------------------------
        self.boxes = self._load_boxes(cfg)
        self.k_contact = 2.0e5
        self.c_contact = 2.0 * 0.25 * np.sqrt(self.k_contact * self.mass.mean())
        self.mu = 0.4

        # ---- kinematic grips ---------------------------------------------
        self.grip_left = np.where(self.ix == 0)[0]
        self.grip_right = np.where(self.ix == nx - 1)[0]
        self.grip_idx = np.concatenate((self.grip_left, self.grip_right))
        self.grip_mask = np.zeros(n, dtype=bool)
        self.grip_mask[self.grip_idx] = True
        self.grip_y0 = rest2d[self.grip_idx, 1].copy()
        self.grip_z0 = np.full(len(self.grip_idx), float(center[2]))
        self.grip_half = 0.5 * self.sx
        self.grip_dmax = 0.12          # inward travel of each grip  [m]
        self.grip_t0 = 0.35
        self.grip_t1 = 2.80

        # ---- state --------------------------------------------------------
        self.pos = np.column_stack((rest2d[:, 0], rest2d[:, 1],
                                    np.full(n, float(center[2]))))
        # tiny deterministic out-of-plane imperfection: seeds the physically
        # selected buckling mode (it does not prescribe any motion).
        X = (rest2d[:, 0] + 0.5 * self.sx) / self.sx
        Y = (rest2d[:, 1] + 0.5 * self.sy) / self.sy
        m = 5.0
        amp = 1.5e-3
        w = amp * (
            np.sin(m * np.pi * X) * (1.0 + 0.15 * np.cos(2.0 * np.pi * Y))
            + 0.45 * np.sin(3.0 * np.pi * X)
            + 0.30 * np.sin(2.0 * np.pi * Y + 0.7) * np.sin(m * np.pi * X)
        )
        self.pos[:, 2] += w
        self.vel = np.zeros_like(self.pos)

        # ---- triangles (deformation gradient output) ----------------------
        self.faces = faces

        # ---- integration control ------------------------------------------
        self.dt = 1.0e-4
        self._t = 0.0
        self._steps = 0
        self.seed = int(seed)

    # ------------------------------------------------------------ utilities
    def _triangulation(self, n: int, nx: int, ny: int) -> np.ndarray:
        """Use the frozen topology asset when available, else triangulate."""
        asset = Path(__file__).resolve().parent.parent / "assets" / "cloth_topology.json"
        try:
            data = json.loads(asset.read_text(encoding="utf-8"))
            verts = np.asarray(data["vertices"], dtype=float)
            fcs = np.asarray(data["faces"], dtype=np.int64)
            if verts.shape == (n, 3) and fcs.ndim == 2 and fcs.shape[1] == 3:
                if np.allclose(verts[:, :2], self.rest2d, atol=1e-9):
                    return fcs
        except Exception:
            pass
        tris = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                i = iy * nx + ix
                tris.append((i, i + 1, i + nx))
                tris.append((i + 1, i + nx + 1, i + nx))
        return np.asarray(tris, dtype=np.int64)

    def _build_bending_supports(self) -> None:
        """Grid slice helpers for the biharmonic curvature operator."""
        self._Lx = np.zeros((self.ny, self.nx, 3))
        self._Ly = np.zeros((self.ny, self.nx, 3))
        self._Bx = np.zeros((self.ny, self.nx, 3))
        self._By = np.zeros((self.ny, self.nx, 3))

    def _load_boxes(self, cfg: dict):
        boxes = []
        payload = cfg.get("scene_geometry") or {}
        objects = payload.get("objects") if isinstance(payload, dict) else None
        if objects:
            for item in objects:
                geom = item.get("geometry", {})
                if geom.get("type") != "box":
                    continue
                half = geom.get("half_extents")
                tr = item.get("transform", {}) or {}
                tvec = tr.get("translation", (0.0, 0.0, 0.0))
                oid = item.get("id", "")
                moving = str(item.get("motion", "static")) == "kinematic"
                boxes.append((oid, np.asarray(half, float),
                              np.asarray(tvec, float), moving))
        if not boxes:
            boxes = [(oid, np.asarray(h, float), np.asarray(t, float), mv)
                     for oid, h, t, mv in _FALLBACK_BOXES]
        return boxes

    # ------------------------------------------------------- grip kinematics
    def _grip_offset(self, t: float):
        """Inward grip travel d(t) and its time derivative (C1 smoothstep)."""
        if t <= self.grip_t0:
            return 0.0, 0.0
        if t >= self.grip_t1:
            return self.grip_dmax, 0.0
        s = (t - self.grip_t0) / (self.grip_t1 - self.grip_t0)
        d = self.grip_dmax * s * s * (3.0 - 2.0 * s)
        dd = self.grip_dmax * 6.0 * s * (1.0 - s) / (self.grip_t1 - self.grip_t0)
        return d, dd

    def _apply_grips(self, pos: np.ndarray, vel: np.ndarray, t: float) -> None:
        d, dd = self._grip_offset(t)
        xl = -(self.grip_half - d)
        xr = (self.grip_half - d)
        nl = len(self.grip_left)
        idx = self.grip_idx
        pos[idx, 0] = np.concatenate((np.full(nl, xl), np.full(nl, xr)))
        pos[idx, 1] = self.grip_y0
        pos[idx, 2] = self.grip_z0
        vx = np.concatenate((np.full(nl, -dd), np.full(nl, dd)))
        vel[idx, 0] = vx
        vel[idx, 1] = 0.0
        vel[idx, 2] = 0.0

    # ----------------------------------------------------------------- forces
    def _accel(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        n = self.n
        acc = np.zeros((n, 3))

        # membrane springs (Kelvin-Voigt)
        d = pos[self.sj] - pos[self.si]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        length = np.maximum(length, 1e-9)
        inv = 1.0 / length
        nrm = d * inv[:, None]
        rel = vel[self.sj] - vel[self.si]
        vn = np.einsum("ij,ij->i", rel, nrm)
        fmag = self.sk * (length - self.sl0) + self.sc * vn
        force = nrm * fmag[:, None]
        for c in range(3):
            acc[:, c] = (np.bincount(self.si, weights=force[:, c], minlength=n)
                         - np.bincount(self.sj, weights=force[:, c], minlength=n)) * self.inv_mass

        # bending: discrete biharmonic curvature energy
        acc += self._bending_accel(pos)

        # gravity
        acc += self.gravity[None, :]

        # contacts (gripped samples are kinematic, not contacts)
        acc += self._contact_accel(pos, vel)

        acc[self.grip_mask] = 0.0
        return acc

    def _bending_accel(self, pos: np.ndarray) -> np.ndarray:
        nx, ny = self.nx, self.ny
        if nx < 5 and ny < 5:
            return np.zeros_like(pos)
        pg = pos.reshape(ny, nx, 3)
        lx = np.zeros_like(pg)
        ly = np.zeros_like(pg)
        if nx >= 5:
            lx[:, 2:nx - 2, :] = (pg[:, 1:nx - 3, :]
                                  - 2.0 * pg[:, 2:nx - 2, :]
                                  + pg[:, 3:nx - 1, :])
        if ny >= 5:
            ly[2:ny - 2, :, :] = (pg[1:ny - 3, :, :]
                                  - 2.0 * pg[2:ny - 2, :, :]
                                  + pg[3:ny - 1, :, :])
        fb = np.zeros_like(pg)
        if nx >= 5:
            bx = np.zeros_like(pg)
            bx[:, 1:nx - 1, :] = lx[:, 2:, :] - 2.0 * lx[:, 1:nx - 1, :] + lx[:, :-2, :]
            fb -= self.kappa_bx * bx
        if ny >= 5:
            by = np.zeros_like(pg)
            by[1:ny - 1, :, :] = ly[2:, :, :] - 2.0 * ly[1:ny - 1, :, :] + ly[:-2, :, :]
            fb -= self.kappa_by * by
        return fb.reshape(-1, 3) * self.inv_mass[:, None]

    @staticmethod
    def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray):
        q = np.abs(points - center[None, :]) - half[None, :]
        pos_part = np.maximum(q, 0.0)
        outside = np.sqrt(np.sum(pos_part * pos_part, axis=1))
        inside = np.minimum(np.max(q, axis=1), 0.0)
        sdf = outside + inside
        nrm = np.sign(points - center[None, :]) * pos_part
        norm = np.linalg.norm(nrm, axis=1)
        safe = np.where(norm > 1e-12, norm, 1.0)
        nrm = nrm / safe[:, None]
        rows = np.arange(len(points))
        idx = np.argmax(q, axis=1)
        axis_n = np.zeros_like(points)
        sgn = np.sign(points[rows, idx] - center[idx])
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        axis_n[rows, idx] = sgn
        deep = outside <= 1e-12
        if np.any(deep):
            nrm[deep] = axis_n[deep]
        return sdf, nrm

    def _contact_state(self, pos: np.ndarray, t: float):
        """Minimum signed distance / outward normal over all frozen boxes."""
        d, _ = self._grip_offset(t)
        best = np.full(self.n, np.inf)
        best_n = np.zeros((self.n, 3))
        for oid, half, trans, moving in self.boxes:
            center = trans.copy()
            if moving:
                if "left" in oid:
                    center[0] = -(self.grip_half - d)
                elif "right" in oid:
                    center[0] = (self.grip_half - d)
            sdf, nrm = self._box_sdf(pos, center, half)
            closer = sdf < best
            if np.any(closer):
                best[closer] = sdf[closer]
                best_n[closer] = nrm[closer]
        return best, best_n

    def _contact_accel(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        acc = np.zeros((self.n, 3))
        # use the grip offset at the current (internal) time
        sdf, nrm = self._contact_state(pos, self._t)
        pen = sdf < 0.0
        pen[self.grip_mask] = False
        if not np.any(pen):
            return acc
        idx = np.where(pen)[0]
        nn = nrm[idx]
        vn = np.einsum("ij,ij->i", vel[idx], nn)
        fn = self.k_contact * (-sdf[idx]) - self.c_contact * np.minimum(vn, 0.0)
        fn = np.maximum(fn, 0.0)
        force = nn * fn[:, None]
        # Coulomb friction on the tangential slip
        vt = vel[idx] - vn[:, None] * nn
        vt_mag = np.linalg.norm(vt, axis=1)
        ft_mag = np.minimum(self.mu * fn, self.c_contact * vt_mag)
        safe = np.where(vt_mag > 1e-12, vt_mag, 1.0)
        force -= (vt / safe[:, None]) * ft_mag[:, None]
        acc[idx] = force * self.inv_mass[idx, None]
        return acc

    # -------------------------------------------------------------- stepping
    def _step(self, h: float) -> None:
        t_new = self._t + h
        damp = 1.0 / (1.0 + self.gamma * h)

        acc = self._accel(self.pos, self.vel)
        vel = (self.vel + 0.5 * h * acc) * damp
        pos = self.pos + h * vel
        self._apply_grips(pos, vel, t_new)

        acc = self._accel(pos, vel)
        vel = (vel + 0.5 * h * acc) * damp
        self._apply_grips(pos, vel, t_new)

        self.pos = pos
        self.vel = vel
        self._t = t_new
        self._steps += 1

    def _advance(self, target: float) -> None:
        while self._t < target - 1e-12:
            h = min(self.dt, target - self._t)
            if h <= 1e-14:
                break
            self._step(h)
        self._t = float(target)
        self._apply_grips(self.pos, self.vel, self._t)

    # ---------------------------------------------------------------- outputs
    def _deformation_gradient(self, pos: np.ndarray) -> np.ndarray:
        tri = self.faces
        p0 = pos[tri[:, 0]]
        p1 = pos[tri[:, 1]]
        p2 = pos[tri[:, 2]]
        xm = np.stack((p1 - p0, p2 - p0), axis=2)              # (T,3,2)
        rm = np.stack((self.rest2d[tri[:, 1]] - self.rest2d[tri[:, 0]],
                       self.rest2d[tri[:, 2]] - self.rest2d[tri[:, 0]]), axis=2)
        try:
            rm_inv = np.linalg.inv(rm)
        except np.linalg.LinAlgError:
            rm_inv = np.linalg.pinv(rm)
        f = np.einsum("tij,tjk->tik", xm, rm_inv)               # (T,3,2)
        e1 = p1 - p0
        e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-12)[:, None]
        nrm = np.cross(p1 - p0, p2 - p0)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        e2 = np.cross(nrm, e1)
        fu = f[:, :, 0]
        fv = f[:, :, 1]
        f2 = np.empty((len(tri), 2, 2))
        f2[:, 0, 0] = np.einsum("ij,ij->i", fu, e1)
        f2[:, 1, 0] = np.einsum("ij,ij->i", fu, e2)
        f2[:, 0, 1] = np.einsum("ij,ij->i", fv, e1)
        f2[:, 1, 1] = np.einsum("ij,ij->i", fv, e2)
        out = np.zeros((self.n, 2, 2))
        count = np.zeros(self.n)
        for c in range(2):
            for r in range(2):
                out[:, r, c] = np.bincount(tri[:, 0], weights=f2[:, r, c],
                                           minlength=self.n)
        for v in (1, 2):
            for c in range(2):
                for r in range(2):
                    out[:, r, c] += np.bincount(tri[:, v], weights=f2[:, r, c],
                                                minlength=self.n)
        np.add.at(count, tri.ravel(), 1.0)
        count = np.maximum(count, 1.0)
        return out / count[:, None, None]

    def _contact_distance(self) -> np.ndarray:
        sdf, _ = self._contact_state(self.pos, self._t)
        sdf = np.where(np.isfinite(sdf), sdf, 0.0)
        sdf[self.grip_mask] = 0.0
        return sdf

    def _make_frame(self, t: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        frame = StateFrame(
            float(t),
            pos,
            vel,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self._deformation_gradient(pos),
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.grip_mask.astype(np.int32),
            },
            fields={
                "contact_signed_distance": self._contact_distance(),
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "mean_element_size": float(0.5 * (self.hx + self.hy)),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self._t:
            self._advance(target)
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        d, dd = self._grip_offset(self._t)
        return {
            "solver": "mass_spring_thin_sheet_biharmonic_bending",
            "steps": int(self._steps),
            "dt": float(self.dt),
            "internal_time": float(self._t),
            "samples": int(self.n),
            "elements": int(len(self.faces)),
            "springs": int(len(self.si)),
            "bending_modulus": float(self.bending_modulus),
            "membrane_stiffness_per_width": float(self.Et),
            "grip_travel": float(d),
            "grip_velocity": float(dd),
            "seed": int(self.seed),
        }
