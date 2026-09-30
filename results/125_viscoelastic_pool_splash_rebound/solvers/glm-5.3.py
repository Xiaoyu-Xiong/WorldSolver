"""Viscoelastic pool splash and rebound solver.

Explicit Material Point Method (MPM) with quadratic B-spline grid weights and a
fixed-corotated viscoelastic-plastic constitutive model (elastic shear/bulk
response, plastic yield on the singular values of the deformation gradient, and
Maxwell-type stress relaxation).  All static solids are axis-aligned boxes taken
from the frozen scene geometry; they are enforced by blocking grid nodes inside
the (slightly expanded) boxes and by projecting particles out of the boxes after
advection.

State is advanced with a fixed physical substep (CFL-safe), and every requested
sample time is produced by stepping deterministically to that time.
"""
from __future__ import annotations

import os

import numpy as np

from worldsolver.contracts import StateFrame

_FLIP = 0.88              # FLIP/PIC blend (1 = pure FLIP)
_VMAX_GRID = 12.0         # grid velocity clamp [m/s]
_VMAX_PART = 12.0         # particle velocity clamp [m/s]
_MAX_GRAD = 80.0          # velocity gradient clamp [1/s]
_MASS_EPS = 1e-8
_SUBSTEPS = 24            # substeps per sample interval

# Fallback solids (mirrors the frozen scene_geometry.yaml, used only if the
# file cannot be read).
_FALLBACK_BOXES = (
    ((0.0, 0.0, -0.05), (3.2, 1.4, 0.05)),
    ((0.0, 0.72, 0.30), (1.45, 0.08, 0.34)),
    ((-1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),
    ((1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),
)


def _load_boxes(root: str):
    boxes = []
    try:
        import yaml  # noqa: PLC0415

        with open(os.path.join(root, "scene_geometry.yaml"), "r") as fh:
            data = yaml.safe_load(fh) or {}
        for obj in data.get("objects", []):
            if obj.get("motion") != "static":
                continue
            if not obj.get("collision_enabled", False):
                continue
            geom = obj.get("geometry", {})
            if geom.get("type") != "box":
                continue
            center = np.asarray(
                obj.get("transform", {}).get("translation", (0.0, 0.0, 0.0)),
                dtype=float,
            )
            half = np.asarray(geom["half_extents"], dtype=float)
            boxes.append((center, half))
    except Exception:
        boxes = []
    if not boxes:
        boxes = [
            (np.asarray(c, float), np.asarray(h, float)) for c, h in _FALLBACK_BOXES
        ]
    return boxes


class BenchmarkSolver:
    """Explicit MPM solver for the viscoelastic pool splash scene."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame = scene.initial_state()

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        sample_dt = float(cfg["sample_dt"])
        self.dt_sub = sample_dt / float(_SUBSTEPS)

        self.pos = np.array(frame.position, dtype=float)
        self.n = int(len(self.pos))
        self.vel = np.zeros((self.n, 3), dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        if frame.body_id is not None:
            self.body_id = np.asarray(frame.body_id).astype(np.int32)
        else:
            self.body_id = np.zeros(self.n, dtype=np.int32)

        attrs = dict(frame.attributes)
        if "particle_radius" in attrs:
            self.particle_radius = np.array(attrs["particle_radius"], dtype=float)
        else:
            self.particle_radius = np.full(self.n, 0.45 * self.spacing)
        if "material_label" in attrs:
            self.material_label = np.asarray(attrs["material_label"])
        else:
            self.material_label = self.body_id.copy()
        self.base_scalars = {k: float(v) for k, v in frame.scalars.items()}

        # per-particle reference volume (mass / reference density)
        self.pvol = self.spacing ** 3

        # deformation gradient and cached first-Piola force matrix P F^T
        self.F = np.tile(np.eye(3), (self.n, 1, 1))
        self.pft = np.zeros((self.n, 3, 3))

        # viscoelastic material parameters (SI units)
        self.mu = 6000.0           # shear modulus [Pa]
        self.lam = 20000.0         # bulk penalty [Pa]
        self.yield_strain = 0.12   # plastic yield strain
        self.relax_rate = 2.5      # Maxwell relaxation rate [1/s]

        # background grid
        self.dx = 0.16
        self.origin = np.array([-2.00, -1.44, -0.16])
        self.shape = (26, 19, 23)  # node counts per axis
        self.nnodes = self.shape[0] * self.shape[1] * self.shape[2]

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.boxes = _load_boxes(root)

        # static blocked-node mask (nodes inside slightly expanded solids)
        nodes = self._node_positions()
        blocked = np.zeros(self.nnodes, dtype=bool)
        for center, half in self.boxes:
            inside = np.all(
                np.abs(nodes - center[None, :]) <= (half[None, :] + 1e-2), axis=1
            )
            blocked |= inside
        # outer grid shell acts as a closed boundary
        ix, iy, iz = np.unravel_index(np.arange(self.nnodes), self.shape)
        shell = (
            (ix == 0) | (ix == self.shape[0] - 1)
            | (iy == 0) | (iy == self.shape[1] - 1)
            | (iz == 0) | (iz == self.shape[2] - 1)
        )
        blocked |= shell
        self.blocked = blocked

        self.t = 0.0
        self._ready = True

    # ------------------------------------------------------------- utilities
    def _node_positions(self) -> np.ndarray:
        gx = self.origin[0] + self.dx * np.arange(self.shape[0])
        gy = self.origin[1] + self.dx * np.arange(self.shape[1])
        gz = self.origin[2] + self.dx * np.arange(self.shape[2])
        mx, my, mz = np.meshgrid(gx, gy, gz, indexing="ij")
        return np.column_stack((mx.ravel(), my.ravel(), mz.ravel()))

    def _solid_sdf(self, points: np.ndarray) -> np.ndarray:
        """Signed distance to the nearest solid box (negative inside)."""
        sdf = np.full(len(points), 1e9)
        for center, half in self.boxes:
            q = np.abs(points - center[None, :]) - half[None, :]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            sdf = np.minimum(sdf, outside + inside)
        return sdf

    # ------------------------------------------------------------ time steps
    def advance_to(self, time):
        if not self._ready:
            raise RuntimeError("solver.initialize() was not called")
        target = float(time)
        while target > self.t + 1e-12:
            dt = min(self.dt_sub, target - self.t)
            self._step(dt)
            self.t += dt
        return self._frame(target)

    def _step(self, dt: float) -> None:
        n = self.n
        dx = self.dx
        pos, vel, mass = self.pos, self.vel, self.mass

        # ---- particle stencil (quadratic B-spline, 3x3x3 nodes)
        xi = (pos - self.origin[None, :]) / dx
        base = np.floor(xi - 0.5).astype(np.int64)
        d = xi - base
        W = np.empty((n, 3, 3))
        dW = np.empty((n, 3, 3))
        W[:, :, 0] = 0.5 * (1.5 - d) ** 2
        W[:, :, 1] = 0.75 - (d - 1.0) ** 2
        W[:, :, 2] = 0.5 * (d - 0.5) ** 2
        dW[:, :, 0] = 1.5 - d
        dW[:, :, 1] = 2.0 * (d - 1.0)
        dW[:, :, 2] = -(d - 0.5)

        idx = np.empty((n, 27), dtype=np.int64)
        wgt = np.empty((n, 27))
        gw = np.empty((n, 27, 3))
        b0, b1, b2 = base[:, 0], base[:, 1], base[:, 2]
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    wx, wy, wz = W[:, 0, i], W[:, 1, j], W[:, 2, k]
                    wgt[:, c] = wx * wy * wz
                    gw[:, c, 0] = dW[:, 0, i] * wy * wz / dx
                    gw[:, c, 1] = wx * dW[:, 1, j] * wz / dx
                    gw[:, c, 2] = wx * wy * dW[:, 2, k] / dx
                    idx[:, c] = (b0 + i) * (self.shape[1] * self.shape[2]) + (
                        b1 + j
                    ) * self.shape[2] + (b2 + k)
                    c += 1

        # ---- particle-to-grid (mass, momentum, internal force)
        idx3 = (idx[:, :, None] * 3 + np.arange(3)[None, None, :]).ravel()
        m_g = np.bincount(
            idx.ravel(),
            weights=(mass[:, None] * wgt).ravel(),
            minlength=self.nnodes,
        )
        mv_w = (mass[:, None, None] * wgt[:, :, None] * vel[:, None, :]).ravel()
        mv_g = np.bincount(idx3, weights=mv_w, minlength=3 * self.nnodes).reshape(
            self.nnodes, 3
        )
        # node force:  f_i = sum_p V0 (P F^T) grad_{x_i} w_ip
        f_w = (self.pvol * np.einsum("pab,pib->pia", self.pft, gw)).ravel()
        f_g = np.bincount(idx3, weights=f_w, minlength=3 * self.nnodes).reshape(
            self.nnodes, 3
        )

        # ---- grid update (explicit)
        active = m_g > _MASS_EPS
        v_old = np.zeros((self.nnodes, 3))
        v_old[active] = mv_g[active] / m_g[active, None]
        v_new = v_old + dt * self.gravity[None, :]
        v_new[active] += dt * (f_g[active] / m_g[active, None])
        v_new[~active] = 0.0
        v_new[self.blocked] = 0.0
        speed = np.linalg.norm(v_new, axis=1)
        over = speed > _VMAX_GRID
        if over.any():
            v_new[over] *= (_VMAX_GRID / speed[over])[:, None]

        # ---- grid-to-particle
        vn = v_new[idx]            # (n, 27, 3)
        vo = v_old[idx]
        w3 = wgt[:, :, None]
        v_pic = (w3 * vn).sum(axis=1)
        dv_flip = (w3 * (vn - vo)).sum(axis=1)
        v_part = _FLIP * (vel + dv_flip) + (1.0 - _FLIP) * v_pic

        # velocity gradient  L = dF/dt F^-1 = -sum_i v_i (x) grad_{x_i} w
        L = -np.einsum("pia,pib->pab", vn, gw)
        ln = np.linalg.norm(L, axis=(1, 2))
        bad = ln > _MAX_GRAD
        if bad.any():
            L[bad] *= (_MAX_GRAD / ln[bad])[:, None, None]
        F_new = (np.eye(3)[None, :, :] + dt * L) @ self.F

        # ---- plastic yield + viscoelastic relaxation on singular values
        U, S, Vt = np.linalg.svd(F_new)
        s = np.clip(S, 1.0 - self.yield_strain, 1.0 + self.yield_strain)
        s = 1.0 + (s - 1.0) * np.exp(-self.relax_rate * dt)
        s = np.clip(s, 0.2, 5.0)
        self.F = (U * s[:, None, :]) @ Vt

        # cache P F^T = U diag(2 mu (s-1) s + lam (J-1) J) U^T for next step
        detF = np.linalg.det(self.F)
        J = np.clip(detF, 0.05, 4.0)
        coef = 2.0 * self.mu * (s - 1.0) * s + self.lam * (J - 1.0) * J
        self.pft = (U * coef[:, None, :]) @ U.transpose(0, 2, 1)

        # ---- advect
        pos += dt * v_part
        vel[:] = v_part

        # keep particles on the grid
        for a in range(3):
            lo = self.origin[a] + 0.5 * self.dx + 1e-9
            hi = self.origin[a] + (self.shape[a] - 1.5) * self.dx - 1e-6
            low = pos[:, a] < lo
            high = pos[:, a] > hi
            if low.any():
                pos[low, a] = lo
                vel[low, a] = 0.0
            if high.any():
                pos[high, a] = hi
                vel[high, a] = 0.0

        # solid contact: project out of boxes, kill inward normal velocity
        for center, half in self.boxes:
            rel = np.abs(pos - center[None, :]) - half[None, :]
            inside = np.all(rel < 0.0, axis=1)
            if not inside.any():
                continue
            p_in = pos[inside]
            r_in = rel[inside]
            rows = np.arange(len(p_in))
            axis = np.argmax(r_in, axis=1)
            sgn = np.sign(p_in[rows, axis])
            sgn = np.where(sgn == 0.0, 1.0, sgn)
            p_in[rows, axis] = center[axis] + sgn * (half[axis] + 1e-4)
            pos[inside] = p_in
            v_in = vel[inside]
            normal = np.zeros_like(v_in)
            normal[rows, axis] = sgn
            vn_dot = (v_in * normal).sum(axis=1)
            hit = vn_dot < 0.0
            if hit.any():
                v_in[hit] -= vn_dot[hit, None] * normal[hit]
                v_in[hit] *= 0.9  # contact friction/damping
            vel[inside] = v_in

        # particle velocity clamp + finiteness guard
        sp = np.linalg.norm(vel, axis=1)
        over = sp > _VMAX_PART
        if over.any():
            vel[over] *= (_VMAX_PART / sp[over])[:, None]
        np.nan_to_num(pos, copy=False)
        np.nan_to_num(vel, copy=False)
        np.nan_to_num(self.F, copy=False)
        np.nan_to_num(self.pft, copy=False)

    # ------------------------------------------------------------ export
    def _frame(self, time: float):
        sdf = self._solid_sdf(self.pos)
        external = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self.F.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label,
            },
            fields={
                "contact_signed_distance": sdf,
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": external,
            },
            scalars=dict(
                self.base_scalars, contact_length_scale=self.spacing
            ),
        )

    def diagnostics(self) -> dict:
        out = {
            "time": float(self.t),
            "num_particles": int(self.n),
            "substep": float(self.dt_sub),
        }
        if self._ready:
            out["max_speed"] = float(np.max(np.linalg.norm(self.vel, axis=1)))
            out["max_height"] = float(np.max(self.pos[:, 2]))
            out["min_solid_sdf"] = float(np.min(self._solid_sdf(self.pos)))
        return out
