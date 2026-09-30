"""Knitted patch uniaxial stretch solver.

Physically-based cloth simulation of a knitted patch held between two grips.
The left grip is fixed, the right grip prescribes a smooth uniaxial extension.
The knit character is captured with an anisotropic, nonlinear spring network:
very low compression resistance (loops buckle / wrinkle), moderate tension
resistance while loops rotate and tighten, and strong stiffening once the
loops jam and the yarn itself is stretched.  Time integration is semi-implicit
(symplectic) Euler with sub-stepping, spring damping, and penalty contact
against the (kinematic) grip boxes and the static ground box.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    """Mass-spring / thin-sheet solver for the knit patch stretch scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        cloth = cfg["cloth"]
        nx, ny = map(int, cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        sx, sy = map(float, cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        x = np.linspace(-sx * 0.5, sx * 0.5, nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        pos0 = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) + center
        self.n = len(pos0)

        # Tiny deterministic knit-scale thickness ripple so wrinkle modes are
        # well defined (stitch-scale) without altering the flat sheet shape.
        ripple = (0.004
                  * np.sin(xx.ravel() / (sx / nx) * np.pi)
                  * np.sin(yy.ravel() / (sy / ny) * np.pi))
        pos0 = pos0 + np.column_stack((np.zeros(self.n), np.zeros(self.n), ripple))

        self.pos = pos0.copy()
        self.vel = np.zeros_like(pos0)
        self.mass = np.full(self.n, float(cloth["total_mass"]) / self.n)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.thickness = float(cloth["thickness"])

        # ---- topology ---------------------------------------------------
        topo_path = os.path.join("assets", "cloth_topology.json")
        faces = None
        if os.path.exists(topo_path):
            with open(topo_path) as fh:
                topo = json.load(fh)
            faces = np.asarray(topo.get("faces", topo.get("triangles", [])), dtype=int)
        if faces is None or len(faces) == 0:
            fl = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    a = j * nx + i
                    b, c, d = a + 1, a + nx, a + nx + 1
                    fl.append((a, b, d))
                    fl.append((a, d, c))
            faces = np.asarray(fl, dtype=int)
        self.faces = faces
        self.ref = pos0.copy()  # material coordinates

        # ---- obstacles ---------------------------------------------------
        # axis-aligned boxes: [cx, cy, cz, hx, hy, hz]
        self.ground = np.array([0.0, 0.0, -0.05, 2.4, 2.0, 0.05])
        self.grip_left = np.array([-1.5, 0.0, 2.15, 0.12, 1.25, 0.12])
        self.grip_right0 = np.array([1.5, 0.0, 2.15, 0.12, 1.25, 0.12])

        # Prescribed extension profile: smooth ramp, then hold.
        self.t_start = 0.5
        self.t_end = 5.0
        self.grip_disp_max = 1.8  # m: patch stretches from 3.0 m to 4.8 m

        # ---- grip attachment (boundary conditions) -----------------------
        self.left_nodes = np.where(np.isclose(self.pos[:, 0], x[0]))[0]
        self.right_nodes = np.where(np.isclose(self.pos[:, 0], x[-1]))[0]
        self.left_anchor = self.pos[self.left_nodes].copy()
        self.right_anchor = self.pos[self.right_nodes].copy()

        # ---- spring network ----------------------------------------------
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        diag = float(np.hypot(dx, dy))

        # Knit anisotropy: course direction (x, stretch direction) shows the
        # nonlinear loop behavior (soft then jamming); wale direction (y) is
        # stiffer in tension. Compression is weak everywhere -> buckling.
        K_STRUCT_X, JAM_X = 18.0, 1.55
        K_STRUCT_Y, JAM_Y = 90.0, 1.35
        K_SHEAR = 6.0
        K_BEND = 1.5
        JAM_SOFT = 0.3  # strain window over which jamming stiffening engages

        spr = []  # (i, j, rest, k_lo, k_hi, jam, k_comp)
        add = lambda i, j, r, k, jam, kc: spr.append(
            (i, j, r, k, k * 20.0, jam, kc))

        for j in range(ny):
            for i in range(nx - 1):
                add(j * nx + i, j * nx + i + 1, dx, K_STRUCT_X, JAM_X, 0.4)
        for j in range(ny - 1):
            for i in range(nx):
                add(j * nx + i, (j + 1) * nx + i, dy, K_STRUCT_Y, JAM_Y, 1.5)
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b, c, d = a + 1, a + nx, a + nx + 1
                add(a, d, diag, K_SHEAR, 2.5, 0.5)
                add(b, c, diag, K_SHEAR, 2.5, 0.5)
        for j in range(ny):
            for i in range(nx - 2):
                add(j * nx + i, j * nx + i + 2, 2.0 * dx, K_BEND, 1.2, 0.3)
        for j in range(ny - 2):
            for i in range(nx):
                add(j * nx + i, (j + 2) * nx + i, 2.0 * dy, K_BEND, 1.2, 0.3)

        arr = np.asarray(spr, dtype=float)
        self.si = arr[:, 0].astype(np.int64)
        self.sj = arr[:, 1].astype(np.int64)
        self.rest = arr[:, 2].copy()
        self.k_lo = arr[:, 3].copy()
        self.k_hi = arr[:, 4].copy()
        self.jam = arr[:, 5].copy()
        self.k_comp = arr[:, 6].copy()
        self.jam_soft = JAM_SOFT

        m_node = self.mass[0]
        # ~0.3 x critical damping per spring family
        self.cdamp = 0.6 * np.sqrt(self.k_lo * m_node)

        # per-node triangle lists (deformation-gradient averaging)
        nf = [[] for _ in range(self.n)]
        for t, f in enumerate(self.faces):
            for v in f:
                nf[v].append(t)
        self.node_faces = [np.asarray(fs, dtype=int) for fs in nf]

        self.time = 0.0
        self.h = 5.0e-4
        self._contact_force = np.zeros((self.n, 3))
        self._contact_sd = np.full(self.n, 1e9)

    # ------------------------------------------------------------------
    def grip_displacement(self, t: float) -> float:
        u = (t - self.t_start) / (self.t_end - self.t_start)
        return self.grip_disp_max * _smoothstep(u)

    def grip_velocity(self, t: float) -> float:
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        u = (t - self.t_start) / (self.t_end - self.t_start)
        # d/dt smoothstep(u) = 6 u (1-u) / (t_end - t_start)
        return self.grip_disp_max * 6.0 * u * (1.0 - u) / (self.t_end - self.t_start)

    # ------------------------------------------------------------------
    @staticmethod
    def _box_sdf(p: np.ndarray, box: np.ndarray):
        """Signed distance of points p ([N,3]) to an axis-aligned box."""
        q = p - box[None, :3]
        d = np.abs(q) - box[None, 3:6]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return outside + inside, q, d

    def _contact(self, pos: np.ndarray, vel: np.ndarray, boxes):
        """Penalty contact (with viscous friction) against obstacle boxes."""
        f = np.zeros_like(pos)
        margin = self.thickness
        best_sd = np.full(self.n, 1e9)
        for box in boxes:
            sd, q, d = self._box_sdf(pos, box)
            best_sd = np.minimum(best_sd, sd)
            depth = margin - sd
            active = depth > 0.0
            if not np.any(active):
                continue
            # outward normal from the closest face
            viol = np.abs(q) - box[None, 3:6]
            axis = np.argmin(viol, axis=1)
            normal = np.zeros((self.n, 3))
            rows = np.arange(self.n)
            sign = np.sign(q[rows, axis])
            sign[sign == 0] = 1.0
            normal[rows, axis] = sign
            fn = 4000.0 * np.maximum(depth, 0.0) ** 1.5
            f += fn[:, None] * normal
            # friction on penetrating samples: damp tangential velocity
            touching = sd < 0.0
            if np.any(touching):
                n_t = normal[touching]
                vt = (vel[touching]
                      - np.sum(vel[touching] * n_t, axis=1, keepdims=True) * n_t)
                f[touching] -= 2.0 * vt * self.mass[touching, None]
        return f, best_sd

    # ------------------------------------------------------------------
    def _spring_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        d = pos[self.sj] - pos[self.si]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / L[:, None]
        s = L / self.rest  # stretch ratio

        # Nonlinear knit stiffness:
        #  - compression: weak (k_comp) -> loops buckle, wrinkles form
        #  - tension below jam strain: k_lo (loops rotate / tighten)
        #  - beyond jam: ramp up to k_hi (yarns inextensible)
        k_soft = np.where(s < 1.0, self.k_comp, self.k_lo)
        jam_frac = np.clip((s - self.jam) / self.jam_soft, 0.0, 1.0)
        k_eff = k_soft + (self.k_hi - self.k_lo) * jam_frac

        strain = L - self.rest
        strain = np.where(strain < 0.0, 0.15 * strain, strain)  # buckle freely
        fs = k_eff * strain

        dv = vel[self.sj] - vel[self.si]
        fd = self.cdamp * np.sum(dv * u, axis=1)

        ft = fs + fd
        f = np.zeros_like(pos)
        np.add.at(f, self.si, -ft[:, None] * u)
        np.add.at(f, self.sj, ft[:, None] * u)
        return f

    # ------------------------------------------------------------------
    def _integrate(self, h: float) -> None:
        t = self.time
        grip_d = self.grip_displacement(t + 0.5 * h)
        grip_v = self.grip_velocity(t + 0.5 * h)
        grip_right = self.grip_right0.copy()
        grip_right[0] += grip_d
        boxes = [self.ground, self.grip_left, grip_right]

        f_int = self._spring_forces(self.pos, self.vel)
        f_contact, best_sd = self._contact(self.pos, self.vel, boxes)
        f_ext = self.mass[:, None] * self.gravity[None, :] + f_contact

        acc = (f_int + f_ext) / self.mass[:, None]
        self.vel += h * acc
        self.vel *= (1.0 - 0.02 * h)  # small numerical damping
        self.pos += h * self.vel

        # exact boundary conditions at the grips
        self.pos[self.left_nodes] = self.left_anchor
        self.vel[self.left_nodes] = 0.0
        self.pos[self.right_nodes] = self.right_anchor + np.array([grip_d, 0.0, 0.0])
        self.vel[self.right_nodes] = np.array([grip_v, 0.0, 0.0])

        # non-penetration projection against obstacles
        for box in boxes:
            sd, q, _ = self._box_sdf(self.pos, box)
            inside = sd < 0.0
            if not np.any(inside):
                continue
            viol = np.abs(q[inside]) - box[None, 3:6]
            axis = np.argmin(viol, axis=1)
            idx = np.where(inside)[0]
            for a in range(3):
                sel = idx[axis == a]
                if len(sel) == 0:
                    continue
                sign = np.sign(self.pos[sel, a] - box[a])
                sign[sign == 0] = 1.0
                self.pos[sel, a] = box[a] + sign * (box[3 + a] + self.thickness)

        self._contact_force = f_contact
        self._contact_sd = best_sd
        self.time = t + h

    # ------------------------------------------------------------------
    def _deformation_gradients(self, pos: np.ndarray) -> np.ndarray:
        """Per-node 2x2 in-plane deformation gradient (material xy -> xy).

        Edge columns: P_cols = F @ M_cols  =>  F = P_cols @ inv(M_cols).
        """
        a, b, c = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        M = np.stack((self.ref[b, :2] - self.ref[a, :2],
                      self.ref[c, :2] - self.ref[a, :2]), axis=1)  # [T,2,2]
        P = np.stack((pos[b, :2] - pos[a, :2],
                      pos[c, :2] - pos[a, :2]), axis=1)  # [T,2,2]
        try:
            Ftri = P @ np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Ftri = np.repeat(np.eye(2)[None], len(self.faces), axis=0)
        Ftri = np.nan_to_num(Ftri, nan=0.0, posinf=0.0, neginf=0.0)

        Fnode = np.zeros((self.n, 2, 2))
        eye = np.eye(2)
        for v in range(self.n):
            fs = self.node_faces[v]
            Fnode[v] = Ftri[fs].mean(axis=0) if len(fs) else eye
        return Fnode

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        if time > self.time + 1e-12:
            while self.time < time - 1e-12:
                h = min(self.h, time - self.time)
                self._integrate(h)
        return self._frame(time)

    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        F = self._deformation_gradients(self.pos)
        sd = self._contact_sd
        near = np.where(sd < 0.08)[0]
        if len(near) == 0:
            near = np.array([int(np.argmin(sd))])
        contact_sd = sd[near]

        f_ext = self.mass[:, None] * self.gravity[None, :] + self._contact_force

        bnd = np.zeros(self.n, dtype=np.int32)
        bnd[self.left_nodes] = 1
        bnd[self.right_nodes] = 1

        frame = StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": bnd,
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(1),
                "external_force": f_ext,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "nodes": int(self.n),
            "grip_displacement": float(self.grip_displacement(self.time)),
        }
