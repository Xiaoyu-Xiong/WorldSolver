"""Knitted-patch uniaxial-stretch solver (self-contained physical model).

Phenomenon
----------
A rectangular knitted patch (17x17 vertex grid, held along its two short
opposite edges) is stretched uniaxially: the left grip is fixed, the right
grip translates away with a smooth, prescribed extension profile.  Yarn-level
knit response is represented by an anisotropic, strongly nonlinear spring
network on the triangle mesh:

* course-direction (pull) springs: very compliant while the loops straighten,
  then locking / jamming stiffening at large extension,
* wale-direction (transverse) springs: compliant in tension, strongly locking
  in compression (loop jamming),
* diagonal shear springs: geometrically couple the two directions and produce
  the transverse contraction (Poisson-like effect) under uniaxial extension,
* weak bending springs on second neighbours keep the sheet smooth and coherent.

Boundary conditions
-------------------
The two gripped edge columns are kinematic: the left column stays fixed at the
left grip, the right column is rigidly attached to the moving grip.

Numerics
--------
Backward-Euler (linearly implicit) time stepping with an SPD spring stiffness
matrix; the reduced free-DOF system is solved by sparse LU factorisation each
substep.  Ground/grip contact uses an implicit one-sided penalty constraint and
the signed geometric gap is exported per vertex.  All quantities are
deterministic and finite.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame

_GRAVITY_DEFAULT = np.array([0.0, 0.0, -9.81], dtype=float)
_GEOMETRY_FILE = "scene_geometry.yaml"


def _smoothstep(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau * tau * (3.0 - 2.0 * tau)


def _smoothstep_d(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return 6.0 * tau * (1.0 - tau)


class BenchmarkSolver:
    """Deterministic physical solver for the knitted patch uniaxial stretch."""

    # --- integration -----------------------------------------------------
    SUBSTEPS = 12                 # substeps per sample interval
    DAMPING = 120.0               # 1/s, velocity proportional (per unit mass)

    # --- prescribed uniaxial extension -----------------------------------
    T_EXT_START = 1.0             # s, extension begins after initial settling
    T_EXT_END = 5.5               # s, smooth ramp duration
    MAX_EXTENSION = 1.2           # m, total grip travel (40% of the 3 m width)

    # --- contact ----------------------------------------------------------
    CONTACT_MARGIN = 0.002        # m
    CONTACT_STIFFNESS = 400.0     # N/m
    CONTACT_DAMPING = 0.6         # N s/m

    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        self.config = cfg
        self.seed = int(seed)

        cloth = cfg["cloth"]
        self.nx = int(cloth["grid_shape"][0])
        self.ny = int(cloth["grid_shape"][1])
        self.thickness = float(cloth["thickness"])
        gravity = np.asarray(cfg.get("gravity", _GRAVITY_DEFAULT), dtype=float)
        self.gravity = gravity.reshape(3) if gravity.size == 3 else _GRAVITY_DEFAULT.copy()
        self.sample_dt = float(cfg["sample_dt"])
        self.dt_sub = self.sample_dt / float(self.SUBSTEPS)

        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=float)
        self.x0 = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.m = np.asarray(frame.mass, dtype=float).reshape(-1).copy()
        self.N = int(self.x.shape[0])
        self.time = float(frame.time)
        boundary = frame.attributes.get("boundary_vertex")
        if boundary is None:
            boundary = np.zeros(self.N, dtype=np.int32)
        self.boundary = np.asarray(boundary).astype(np.int32)

        self._setup_topology()
        self._setup_springs()
        self._setup_colliders()
        self._setup_dofs()

    # ------------------------------------------------------------------ #
    def _setup_topology(self) -> None:
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        v00 = idx[:-1, :-1].reshape(-1)
        v10 = idx[:-1, 1:].reshape(-1)
        v01 = idx[1:, :-1].reshape(-1)
        v11 = idx[1:, 1:].reshape(-1)
        tris = np.concatenate(
            [np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)],
            axis=0,
        ).astype(np.int64)
        self.tris = tris

        # incident triangle weights from the flat rest configuration
        p = self.x0
        e1 = p[tris[:, 1]] - p[tris[:, 0]]
        e2 = p[tris[:, 2]] - p[tris[:, 0]]
        rest_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        rest_area = np.maximum(rest_area, 1e-12)
        self.rest_area = rest_area
        r1 = e1[:, :2]
        r2 = e2[:, :2]
        det = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
        safe_det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        rinv = np.empty((tris.shape[0], 2, 2), dtype=float)
        rinv[:, 0, 0] = r2[:, 1] / safe_det
        rinv[:, 0, 1] = -r2[:, 0] / safe_det
        rinv[:, 1, 0] = -r1[:, 1] / safe_det
        rinv[:, 1, 1] = r1[:, 0] / safe_det
        self.rest_edge_inv = rinv
        self.vertex_area = np.zeros(self.N, dtype=float)
        for k in range(3):
            np.add.at(self.vertex_area, tris[:, k], rest_area)
        self.vertex_area = np.maximum(self.vertex_area, 1e-12)

        # gripped columns: extreme x on each side (grid indices)
        i = np.arange(self.N) % nx
        self.left_nodes = np.where(i == 0)[0]
        self.right_nodes = np.where(i == nx - 1)[0]
        self.pin_nodes = np.concatenate([self.left_nodes, self.right_nodes])
        self.n_right = self.right_nodes.size
        self.pin_pos0 = self.x0[self.pin_nodes].copy()

    # ------------------------------------------------------------------ #
    def _setup_springs(self) -> None:
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)

        def edge_pairs(a, b):
            return a.reshape(-1).astype(np.int64), b.reshape(-1).astype(np.int64)

        pairs = []
        # course direction (pull direction): soft while loops straighten,
        # then pronounced jamming lock at large extension
        pairs.append((*edge_pairs(idx[:, :-1], idx[:, 1:]),
                      25.0, 30.0, 0.06, 100.0, 0.10))
        # wale direction (transverse): soft tension, locking compression
        pairs.append((*edge_pairs(idx[:-1, :], idx[1:, :]),
                      12.0, 15.0, 0.25, 200.0, 0.10))
        # shear diagonals (both diagonals of every cell)
        pairs.append((*edge_pairs(idx[:-1, :-1], idx[1:, 1:]),
                      10.0, 40.0, 0.10, 100.0, 0.08))
        pairs.append((*edge_pairs(idx[:-1, 1:], idx[1:, :-1]),
                      10.0, 40.0, 0.10, 100.0, 0.08))
        # weak bending springs (second neighbours)
        pairs.append((*edge_pairs(idx[:, :-2], idx[:, 2:]),
                      0.2, 0.0, 1e9, 0.0, 1e9))
        pairs.append((*edge_pairs(idx[:-2, :], idx[2:, :]),
                      0.2, 0.0, 1e9, 0.0, 1e9))

        sa = np.concatenate([p[0] for p in pairs])
        sb = np.concatenate([p[1] for p in pairs])
        self.sa = sa
        self.sb = sb
        self.s_k0 = np.concatenate([np.full(p[0].size, p[2]) for p in pairs])
        self.s_at = np.concatenate([np.full(p[0].size, p[3]) for p in pairs])
        self.s_ej = np.concatenate([np.full(p[0].size, p[4]) for p in pairs])
        self.s_ac = np.concatenate([np.full(p[0].size, p[5]) for p in pairs])
        self.s_ejc = np.concatenate([np.full(p[0].size, p[6]) for p in pairs])
        d = self.x0[sb] - self.x0[sa]
        self.s_l0 = np.maximum(np.linalg.norm(d, axis=1), 1e-9)

    # ------------------------------------------------------------------ #
    def _setup_colliders(self) -> None:
        """Static collider boxes; the right grip translates with the extension."""
        self.ground_center = np.array([0.0, 0.0, -0.05], dtype=float)
        self.ground_half = np.array([2.4, 2.0, 0.05], dtype=float)
        self.grip_left_center0 = np.array([-1.5, 0.0, 2.15], dtype=float)
        self.grip_right_center0 = np.array([1.5, 0.0, 2.15], dtype=float)
        self.grip_half = np.array([0.12, 1.25, 0.12], dtype=float)

        path = os.path.join(os.getcwd(), _GEOMETRY_FILE)
        try:  # optional: honour the frozen scene geometry exactly
            import yaml

            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            for obj in data.get("objects", []):
                geom = obj.get("geometry", {})
                if geom.get("type") != "box":
                    continue
                center = np.asarray(obj["transform"]["translation"], dtype=float)
                half = np.asarray(geom["half_extents"], dtype=float)
                if obj["id"] == "ground":
                    self.ground_center, self.ground_half = center, half
                elif obj["id"] == "grip_left":
                    self.grip_left_center0, self.grip_half = center, half
                elif obj["id"] == "grip_right":
                    self.grip_right_center0, self.grip_half = center, half
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _setup_dofs(self) -> None:
        n3 = 3 * self.N
        is_pin = np.zeros(self.N, dtype=bool)
        is_pin[self.pin_nodes] = True
        self.pin_node_mask = is_pin
        dof_pin = np.zeros(n3, dtype=bool)
        dof_pin[3 * self.pin_nodes[:, None] + np.arange(3)[None, :]] = True
        self.pin_dof = np.where(dof_pin)[0]
        self.free_dof = np.where(~dof_pin)[0]
        self.free_node = ~is_pin

    # ------------------------------------------------------------------ #
    # kinematics of the grips
    # ------------------------------------------------------------------ #
    def _extension(self, time: float) -> float:
        tau = (time - self.T_EXT_START) / (self.T_EXT_END - self.T_EXT_START)
        return float(self.MAX_EXTENSION * _smoothstep(tau))

    def _extension_rate(self, time: float) -> float:
        tau = (time - self.T_EXT_START) / (self.T_EXT_END - self.T_EXT_START)
        return float(self.MAX_EXTENSION * _smoothstep_d(tau)
                     / (self.T_EXT_END - self.T_EXT_START))

    def _pin_state(self, time: float):
        pos = self.pin_pos0.copy()
        vel = np.zeros_like(pos)
        if self.n_right:
            pos[self.left_nodes.size:, 0] += self._extension(time)
            vel[self.left_nodes.size:, 0] = self._extension_rate(time)
        return pos, vel

    # ------------------------------------------------------------------ #
    # signed distances / contact
    # ------------------------------------------------------------------ #
    @staticmethod
    def _box_sdf_normal(points, center, half):
        q = np.abs(points - center) - half
        pos = np.maximum(q, 0.0)
        norm = np.linalg.norm(pos, axis=1)
        dist = norm + np.minimum(np.max(q, axis=1), 0.0)
        normal = np.zeros_like(points)
        outside = norm > 1e-12
        if np.any(outside):
            normal[outside] = pos[outside] / norm[outside, None]
        inside = ~outside
        if np.any(inside):
            axes = np.argmax(q[inside], axis=1)
            local = np.zeros((int(inside.sum()), 3))
            local[np.arange(axes.size), axes] = 1.0
            normal[inside] = local
        sign = np.sign(points - center)
        sign[sign == 0.0] = 1.0
        normal = normal * sign
        nlen = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(nlen, 1e-12)
        return dist, normal

    def _signed_distances(self, points, time: float):
        disp = self._extension(time)
        boxes = (
            (self.ground_center, self.ground_half),
            (self.grip_left_center0, self.grip_half),
            (self.grip_right_center0 + np.array([disp, 0.0, 0.0]), self.grip_half),
        )
        best = None
        best_n = None
        for center, half in boxes:
            d, n = self._box_sdf_normal(points, center, half)
            if best is None:
                best, best_n = d, n
            else:
                closer = d < best
                best = np.where(closer, d, best)
                best_n = np.where(closer[:, None], n, best_n)
        return best, best_n

    # ------------------------------------------------------------------ #
    # one backward-Euler substep
    # ------------------------------------------------------------------ #
    def _step(self, dt: float, time: float) -> None:
        x, v, m = self.x, self.v, self.m
        n = self.N
        n3 = 3 * n

        force = m[:, None] * self.gravity[None, :]

        # --- spring forces and tangent stiffness --------------------------
        d = x[self.sb] - x[self.sa]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / length[:, None]
        eps = length / self.s_l0 - 1.0
        tension = np.maximum(0.0, eps - self.s_ej)
        compression = np.maximum(0.0, -eps - self.s_ejc)
        k = self.s_k0 * (1.0 + self.s_at * tension * tension
                         + self.s_ac * compression * compression)
        f = k * (length - self.s_l0)
        fs = f[:, None] * u
        force = force + np.zeros_like(force)
        np.add.at(force, self.sa, fs)
        np.add.at(force, self.sb, -fs)

        # --- contact (implicit one-sided penalty anchored to the surface) --
        sdf, snormal = self._signed_distances(x, time)
        active = (sdf < self.CONTACT_MARGIN) & self.free_node
        act = np.where(active)[0]
        kc = None
        nc = None
        if act.size:
            kc = np.full(act.size, self.CONTACT_STIFFNESS)
            nc = snormal[act]
            depth = self.CONTACT_MARGIN - sdf[act]
            force[act] += (kc[:, None] * depth[:, None]) * nc
            vn = np.sum(v[act] * nc, axis=1)
            force[act] -= (self.CONTACT_DAMPING * np.minimum(vn, 0.0))[:, None] * nc

        # --- assemble SPD stiffness blocks -------------------------------
        rows = []
        cols = []
        vals = []
        block = (k[:, None, None] * u[:, :, None] * u[:, None, :]).reshape(-1)
        for ra, rb, sgn in ((self.sa, self.sa, 1.0), (self.sa, self.sb, -1.0),
                            (self.sb, self.sa, -1.0), (self.sb, self.sb, 1.0)):
            rr = np.repeat(3 * ra[:, None] + np.arange(3)[None, :], 3, axis=1).reshape(-1)
            cc = np.tile(3 * rb[:, None] + np.arange(3)[None, :], (1, 3)).reshape(-1)
            rows.append(rr)
            cols.append(cc)
            vals.append(sgn * block)
        if kc is not None:
            ablock = (kc[:, None, None] * nc[:, :, None] * nc[:, None, :]).reshape(-1)
            rr = np.repeat(3 * act[:, None] + np.arange(3)[None, :], 3, axis=1).reshape(-1)
            cc = np.tile(3 * act[:, None] + np.arange(3)[None, :], (1, 3)).reshape(-1)
            rows.append(rr)
            cols.append(cc)
            vals.append(ablock)
        K = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n3, n3),
        ).tocsr()

        # --- reduced backward-Euler system on free DOFs -------------------
        diag = np.repeat(m * (1.0 + dt * self.DAMPING), 3)
        A = sp.diags(diag) + (dt * dt) * K
        pin_pos, pin_vel = self._pin_state(time + dt)
        rhs = (m[:, None] * v).reshape(-1) + dt * force.reshape(-1)
        Kfp = K[self.free_dof][:, self.pin_dof]
        rhs[self.free_dof] -= (dt * dt) * (Kfp @ pin_vel.reshape(-1))
        Af = A[self.free_dof][:, self.free_dof].tocsc()
        v_new = v.copy()
        try:
            lu = splu(Af)
            vf = lu.solve(rhs[self.free_dof])
        except Exception:
            vf = np.linalg.lstsq(Af.toarray(), rhs[self.free_dof], rcond=None)[0]
        if not np.all(np.isfinite(vf)):
            vf = np.zeros_like(vf)
        v_new[self.free_node] = vf.reshape(-1, 3)
        v_new[self.pin_node_mask] = pin_vel
        self.v = v_new
        self.x = x + dt * v_new
        self.x[self.pin_node_mask] = pin_pos
        self.time = time + dt

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            target = self.time
        guard = 0
        while self.time < target - 1e-9 and guard < 500000:
            dt = min(self.dt_sub, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt, self.time)
            guard += 1
        return self._frame(target)

    # ------------------------------------------------------------------ #
    def _deformation_gradient(self) -> np.ndarray:
        p = self.x
        tris = self.tris
        e1 = p[tris[:, 1]] - p[tris[:, 0]]
        e2 = p[tris[:, 2]] - p[tris[:, 0]]
        e = np.stack([e1, e2], axis=2)                              # (T,3,2)
        m_mat = np.einsum("tik,tkj->tij", e, self.rest_edge_inv)    # (T,3,2)
        c = np.einsum("tki,tkj->tij", m_mat, m_mat)                 # right Cauchy-Green
        c = 0.5 * (c + np.transpose(c, (0, 2, 1)))
        w, vec = np.linalg.eigh(c)
        w = np.clip(w, 1e-12, None)
        sq = np.sqrt(w)
        ftri = np.einsum("tij,tj,tkj->tik", vec, sq, vec)
        ftri = 0.5 * (ftri + np.transpose(ftri, (0, 2, 1)))

        weight = self.rest_area
        fv = np.zeros((self.N, 2, 2), dtype=float)
        wsum = np.zeros(self.N, dtype=float)
        for k in range(3):
            np.add.at(fv, tris[:, k], weight[:, None, None] * ftri)
            np.add.at(wsum, tris[:, k], weight)
        wsum = np.maximum(wsum, 1e-12)
        fv = fv / wsum[:, None, None]
        bad = ~np.isfinite(fv).all(axis=(1, 2))
        if np.any(bad):
            fv[bad] = np.eye(2)
        return fv

    # ------------------------------------------------------------------ #
    def _frame(self, time: float) -> StateFrame:
        sdf, _ = self._signed_distances(self.x, time)
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.m.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": np.asarray(sdf, dtype=float),
                "cumulative_mass_source": np.zeros(self.N, dtype=float),
                "external_force": self.m[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict[str, object]:
        sdf, _ = self._signed_distances(self.x, self.time)
        return {
            "time": float(self.time),
            "extension": self._extension(self.time),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self.N else 0.0,
            "min_contact_signed_distance": float(np.min(sdf)) if sdf.size else 0.0,
            "n_nodes": int(self.N),
            "n_elements": int(self.tris.shape[0]),
        }
