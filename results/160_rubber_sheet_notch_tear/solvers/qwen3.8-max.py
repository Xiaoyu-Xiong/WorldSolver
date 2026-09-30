"""Physically-based solver for the notched rubber sheet tear scene.

Model: explicit mass-spring thin sheet on the provided triangle topology
(which already contains the edge notch).  Structural springs follow the mesh
edges; weak bending springs span two grid cells.  Grips are kinematic: the
left/right boundary vertex columns are prescribed to translate apart with a
smooth ramp.  Tearing is a physical brittle-fracture criterion: a structural
spring fails permanently once its stretch exceeds a critical value; stress
concentration at the notch tip nucleates the crack and it then propagates
through the sheet as load sheds onto neighbouring springs.  Penalty contact
against the ground box and the two grip boxes provides contact response and
signed-distance output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        cloth = cfg["cloth"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        nx, ny = map(int, cloth["grid_shape"])
        self.nx, self.ny = nx, ny

        frame0 = scene.initial_state()
        self.X0 = np.asarray(frame0.position, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        N = len(self.X0)
        self.N = N

        # ---- topology (contains the notch cut)
        topo_path = Path(__file__).resolve().parents[1] / "assets" / "cloth_topology.json"
        topo = json.loads(topo_path.read_text())
        self.faces = np.asarray(topo["faces"], dtype=np.int64)

        # ---- structural edges from triangles
        edge_map: dict[tuple[int, int], int] = {}
        e0_list: list[int] = []
        e1_list: list[int] = []
        tri_edges = np.zeros((len(self.faces), 3), dtype=np.int64)
        for fi, tri in enumerate(self.faces):
            for k in range(3):
                a = int(tri[k])
                b = int(tri[(k + 1) % 3])
                key = (a, b) if a < b else (b, a)
                eid = edge_map.get(key)
                if eid is None:
                    eid = len(e0_list)
                    edge_map[key] = eid
                    e0_list.append(key[0])
                    e1_list.append(key[1])
                tri_edges[fi, k] = eid
        self.e0 = np.asarray(e0_list, dtype=np.int64)
        self.e1 = np.asarray(e1_list, dtype=np.int64)
        self.tri_edges = tri_edges
        d0 = self.X0[self.e1] - self.X0[self.e0]
        self.L0 = np.linalg.norm(d0, axis=1)
        self.spring_active = np.ones(len(self.e0), dtype=bool)

        # ---- bending springs (span two grid cells in u and v)
        b0: list[int] = []
        b1: list[int] = []
        for j in range(ny):
            for i in range(nx - 2):
                a = j * nx + i
                b0.append(a)
                b1.append(a + 2)
        for j in range(ny - 2):
            for i in range(nx):
                a = j * nx + i
                b0.append(a)
                b1.append(a + 2 * nx)
        self.b0 = np.asarray(b0, dtype=np.int64)
        self.b1 = np.asarray(b1, dtype=np.int64)
        db0 = self.X0[self.b1] - self.X0[self.b0]
        self.Lb0 = np.linalg.norm(db0, axis=1)
        self.bend_active = np.ones(len(self.b0), dtype=bool)

        # ---- grips / clamped boundary columns
        x = self.X0[:, 0]
        self.left_mask = x < (x.min() + 1e-6)
        self.right_mask = x > (x.max() - 1e-6)
        self.clamped = self.left_mask | self.right_mask
        self.free = ~self.clamped
        self.left_idx = np.flatnonzero(self.left_mask)
        self.right_idx = np.flatnonzero(self.right_mask)

        # grip boxes (kinematic): half extents and initial centres
        self.grip_h = np.array([0.12, 0.18, 1.25])
        self.grip_c0 = np.array([[-1.5, 0.0, 1.6], [1.5, 0.0, 1.6]])
        self.grip_sign = np.array([-1.0, +1.0])  # motion direction along x
        self.ground_h = np.array([2.4, 2.0, 0.05])
        self.ground_c = np.array([0.0, 0.0, -0.05])

        # ---- prescribed grip separation schedule
        self.S = 0.30          # travel of each grip [m]
        self.t_start = 0.3     # pull begins [s]
        self.t_end = 4.5       # full extension [s]

        # ---- material parameters
        self.k_s = 600.0       # structural spring stiffness [N/m]
        self.c_s = 0.08        # spring damping [N s/m]
        self.k_b = 12.0        # bending stiffness
        self.c_b = 0.05
        self.break_stretch = 1.38   # brittle tear threshold (structural)
        self.break_bend = 2.0       # (weak) bending failure
        self.k_contact = 3000.0
        self.c_contact = 0.5
        self.drag = 0.5        # gentle global drag [1/s]
        self.dt_sub = 5e-4
        self.v_max = 60.0

        # ---- state
        self.x = self.X0.copy()
        self.v = np.zeros_like(self.X0)
        self.f = np.zeros_like(self.X0)
        self.t = 0.0
        self.steps_done = 0
        self.n_broken = 0

        # ---- precomputed triangle material frames for deformation gradient
        U = self.X0[:, [0, 2]]  # material coords (x, z)
        t0, t1, t2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        a1 = U[t1] - U[t0]
        a2 = U[t2] - U[t0]
        det = a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0]
        self.tri_area0 = 0.5 * np.abs(det)
        inv = np.empty((len(self.faces), 2, 2))
        inv[:, 0, 0] = a2[:, 1] / det
        inv[:, 0, 1] = -a1[:, 1] / det
        inv[:, 1, 0] = -a2[:, 0] / det
        inv[:, 1, 1] = a1[:, 0] / det
        self.tri_Ainv = inv
        self.vert_F_wsum = np.zeros((N, 2, 2))
        self.vert_w = np.zeros(N)

        self.contact_len_scale = self.thickness
        self._frame0 = frame0

    # ------------------------------------------------------- grip kinematics
    def _grip_s(self, t: float) -> tuple[float, float]:
        t0, t1, S = self.t_start, self.t_end, self.S
        tau = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        s = S * (3.0 * tau * tau - 2.0 * tau ** 3)
        if 0.0 < (t - t0) < (t1 - t0):
            v = S * (6.0 * tau - 6.0 * tau * tau) / (t1 - t0)
        else:
            v = 0.0
        return s, v

    def _apply_grips(self, t: float) -> None:
        s, v = self._grip_s(t)
        self.x[self.left_idx, 0] = self.X0[self.left_idx, 0] - s
        self.x[self.left_idx, 1] = self.X0[self.left_idx, 1]
        self.x[self.left_idx, 2] = self.X0[self.left_idx, 2]
        self.v[self.left_idx] = np.array([-v, 0.0, 0.0])
        self.x[self.right_idx, 0] = self.X0[self.right_idx, 0] + s
        self.x[self.right_idx, 1] = self.X0[self.right_idx, 1]
        self.x[self.right_idx, 2] = self.X0[self.right_idx, 2]
        self.v[self.right_idx] = np.array([v, 0.0, 0.0])

    # ---------------------------------------------------------------- contact
    def _box_sdf(self, p: np.ndarray, c: np.ndarray, h: np.ndarray):
        d = p - c
        ad = np.abs(d)
        q = ad - h
        qc = np.maximum(q, 0.0)
        outside = np.linalg.norm(qc, axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        sdf = outside + inside
        # outward normal (gradient of sdf)
        n = np.zeros_like(d)
        out = outside > 1e-12
        n[out] = qc[out] * np.sign(d[out]) / np.maximum(outside[out, None], 1e-12)
        if np.any(~out):
            idx = np.flatnonzero(~out)
            axis = np.argmax(q[idx], axis=1)
            rows = np.arange(len(idx))
            n[idx, axis] = np.sign(d[idx, axis])
            zero = n[idx, axis] == 0
            n[idx[zero], axis[zero]] = 1.0
        return sdf, n

    def _contact(self, t: float):
        s, _ = self._grip_s(t)
        centres = self.grip_c0 + np.outer(self.grip_sign * s, np.array([1.0, 0.0, 0.0]))
        sdf_min = np.full(self.N, np.inf)
        fn = np.zeros((self.N, 3))
        boxes = [(self.ground_c, self.ground_h, None)]
        boxes.append((centres[0], self.grip_h, self.left_mask))
        boxes.append((centres[1], self.grip_h, self.right_mask))
        for c, h, excl in boxes:
            sdf, n = self._box_sdf(self.x, c, h)
            sdf_min = np.minimum(sdf_min, sdf)
            pen = sdf < 0.0
            if excl is not None:
                pen = pen & (~excl)
            if np.any(pen):
                idx = np.flatnonzero(pen)
                depth = -sdf[idx]
                nn = n[idx]
                vn = np.sum(self.v[idx] * nn, axis=1)
                fmag = self.k_contact * depth - self.c_contact * np.minimum(vn, 0.0)
                fn[idx] += fmag[:, None] * nn
        return sdf_min, fn

    # ------------------------------------------------------------ integration
    def _step(self, h: float) -> None:
        t_new = self.t + h
        f = np.zeros_like(self.x)

        # structural springs
        act = self.spring_active
        if np.any(act):
            i0 = self.e0[act]
            i1 = self.e1[act]
            d = self.x[i1] - self.x[i0]
            L = np.linalg.norm(d, axis=1)
            L = np.maximum(L, 1e-12)
            dirv = d / L[:, None]
            L0 = self.L0[act]
            rv = self.v[i1] - self.v[i0]
            dl = np.sum(rv * dirv, axis=1)
            fmag = self.k_s * (L - L0) + self.c_s * dl
            fv = fmag[:, None] * dirv
            np.add.at(f, i0, fv)
            np.add.at(f, i1, -fv)

        # bending springs
        bact = self.bend_active
        if np.any(bact):
            j0 = self.b0[bact]
            j1 = self.b1[bact]
            d = self.x[j1] - self.x[j0]
            L = np.linalg.norm(d, axis=1)
            L = np.maximum(L, 1e-12)
            dirv = d / L[:, None]
            Lb0 = self.Lb0[bact]
            rv = self.v[j1] - self.v[j0]
            dl = np.sum(rv * dirv, axis=1)
            fmag = self.k_b * (L - Lb0) + self.c_b * dl
            fv = fmag[:, None] * dirv
            np.add.at(f, j0, fv)
            np.add.at(f, j1, -fv)

        # gravity + contact
        sdf, fc = self._contact(self.t)
        f += fc
        f += self.mass[:, None] * self.gravity[None, :]

        # semi-implicit Euler on free vertices
        a = f / self.mass[:, None]
        self.v += h * a
        self.v *= (1.0 - self.drag * h)
        vmax = np.linalg.norm(self.v, axis=1)
        scale = np.minimum(1.0, self.v_max / np.maximum(vmax, 1e-12))
        self.v *= scale[:, None]
        self.x += h * self.v

        # kinematic grips override
        self._apply_grips(t_new)

        # ---- tearing: fail overstretched springs (permanent)
        i0 = self.e0
        i1 = self.e1
        d = self.x[i1] - self.x[i0]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        stretch = L / self.L0
        newly = self.spring_active & (stretch > self.break_stretch)
        if np.any(newly):
            self.spring_active &= ~newly
            self.n_broken += int(newly.sum())
        db = self.x[self.b1] - self.x[self.b0]
        Lb = np.maximum(np.linalg.norm(db, axis=1), 1e-12)
        bnew = self.bend_active & (Lb / self.Lb0 > self.break_bend)
        if np.any(bnew):
            self.bend_active &= ~bnew

        self.t = t_new
        self.steps_done += 1

    # ------------------------------------------------------------- F per vert
    def _deformation_gradients(self) -> np.ndarray:
        t0, t1, t2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        p0, p1, p2 = self.x[t0], self.x[t1], self.x[t2]
        b1 = p1 - p0
        b2 = p2 - p0
        Bxz = np.stack((b1[:, [0, 2]], b2[:, [0, 2]]), axis=2)  # (F,2,2) cols
        Ftri = Bxz @ self.tri_Ainv  # (F,2,2)
        tri_ok = np.all(self.spring_active[self.tri_edges], axis=1)
        w = np.where(tri_ok, self.tri_area0, 0.0)
        vF = np.zeros((self.N, 2, 2))
        vw = np.zeros(self.N)
        for k in range(3):
            np.add.at(vF, self.faces[:, k], w[:, None, None] * Ftri)
            np.add.at(vw, self.faces[:, k], w)
        has = vw > 1e-12
        vF[has] /= vw[has, None, None]
        vF[~has] = np.eye(2)
        return vF

    # ----------------------------------------------------------------- frame
    def _make_frame(self, t: float) -> StateFrame:
        sdf, _ = self._contact(t)
        ext_force = self.mass[:, None] * self.gravity[None, :]
        F = self._deformation_gradients()
        frame = StateFrame(
            float(t),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
                "boundary_vertex": self.clamped.astype(np.int32),
            },
            fields={
                "contact_signed_distance": sdf,
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": ext_force,
            },
            scalars={"contact_length_scale": float(self.contact_len_scale)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        t_target = float(time)
        if t_target > self.t + 1e-12:
            span = t_target - self.t
            n = max(1, int(np.ceil(span / self.dt_sub)))
            h = span / n
            for _ in range(n):
                self._step(h)
        return self._make_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        tri_ok = np.all(self.spring_active[self.tri_edges], axis=1)
        return {
            "time": float(self.t),
            "steps": int(self.steps_done),
            "springs_broken": int(self.n_broken),
            "springs_active": int(self.spring_active.sum()),
            "triangles_intact": int(tri_ok.sum()),
            "mass_total": float(self.mass.sum()),
        }
