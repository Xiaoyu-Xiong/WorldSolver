"""Physically-based solver for 027_torus_through_hole.

A deformable torus (surface mesh, mass-spring/position-based elasticity) is
pushed by a kinematic cylindrical pusher through a smaller hole in a rigid
plate, under gravity, with penalty-free projection-based contact against the
ground, the holed plate, and the pusher.  The solver exports per-sample
physical state (positions, velocities, masses, deformation gradients, external
forces, contact signed distances, mass sources) in the required StateFrame
layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        base = Path(__file__).resolve().parent.parent
        with np.load(base / cfg["subject_asset"]) as data:
            self.x0 = np.asarray(data["vertices"], dtype=float).copy()
            self.faces = np.asarray(data["faces"], dtype=np.int64)

        n = len(self.x0)
        self.n = n
        self.x = self.x0.copy()
        self.v = np.tile(
            np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float), (n, 1)
        )
        self.t = 0.0

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        total_mass = float(cfg["subject_mass"])
        self.mass = np.full(n, total_mass / n, dtype=float)

        self.mu = float(cfg.get("friction", 0.4))
        self.rest = float(cfg.get("restitution", 0.0))
        self.damp = float(cfg.get("damping", 0.0))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        stiffness = float(cfg.get("stiffness", 11.0))

        # ---- topology: mesh edges + bending (cross-edge) pairs ----
        faces = self.faces
        edge_set = set()
        for a, b, c in faces:
            for i, j in ((a, b), (b, c), (c, a)):
                edge_set.add((min(i, j), max(i, j)))
        edges = np.array(sorted(edge_set), dtype=np.int64)
        self.edges = edges
        d = np.linalg.norm(self.x0[edges[:, 0]] - self.x0[edges[:, 1]], axis=1)
        self.edge_rest = d

        # bending pairs: for each edge, the two opposite vertices of the
        # adjacent triangles.
        f_of_edge: dict[tuple[int, int], list[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                third = ({a, b, c} - {i, j}).pop()
                f_of_edge.setdefault(key, []).append(third)
        bend = []
        for key, thirds in f_of_edge.items():
            if len(thirds) == 2 and thirds[0] != thirds[1]:
                bend.append((min(thirds), max(thirds)))
        bend = np.array(sorted(set(bend)), dtype=np.int64)
        # drop any pairs that are already distance edges
        eset = set(map(tuple, edges.tolist()))
        bend = np.array(
            [p for p in bend.tolist() if tuple(p) not in eset], dtype=np.int64
        )
        self.bend = bend
        self.bend_rest = np.linalg.norm(
            self.x0[bend[:, 0]] - self.x0[bend[:, 1]], axis=1
        )

        self.k_edge = float(min(1.0, 0.05 + stiffness / 12.0))
        self.k_bend = 0.35 * self.k_edge

        # ---- per-face material frame pseudo-inverses for deformation gradient
        X = self.x0
        a, b, c = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
        Dm = np.stack([b - a, c - a], axis=2)  # (F,3,2)
        self.face_pinv = np.linalg.pinv(Dm)  # (F,2,3)
        e1 = Dm[:, :, 0]
        e2 = Dm[:, :, 1]
        nrm = np.cross(e1, e2)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        self.face_n0 = nrm  # (F,3)

        # collider parameters (frozen scene geometry)
        self.hole_radius = float(cfg.get("hole_radius", 0.5))
        self.hole_z = float(cfg.get("hole_z", 0.62))
        self.plate_z0 = 0.39
        self.plate_z1 = 0.85
        self.plate_r_out = 0.96
        self.pusher_radius = 0.48
        self.pusher_hh = 0.09
        self.pusher_z0 = 1.88
        self.push_speed = float(cfg.get("push_speed", 0.48))
        self.ground_z = float(cfg.get("ground_z", 0.0))

        self.sub_dt = 1.0 / 240.0
        self.iterations = 8

        self._diag = {
            "steps": 0,
            "constraint_iterations": 0,
            "max_penetration": 0.0,
            "contacts_total": 0,
        }

    # ------------------------------------------------------------- kinematics
    def _pusher_center(self, t: float) -> float:
        zc = self.pusher_z0 - self.push_speed * t
        # pusher is kinematic but cannot pass through the ground plane
        return max(zc, self.ground_z + self.pusher_hh)

    # --------------------------------------------------------------- contacts
    def _project_colliders(self, p: np.ndarray, t: float):
        """Project positions out of ground / holed plate / pusher.

        Returns contact mask info for friction/restitution handling.
        """
        contacts = []  # (mask, normal)

        # ---- ground plane z >= ground_z ----
        gz = self.ground_z
        pen = gz - p[:, 2]
        m = pen > 0.0
        if m.any():
            p[m, 2] = gz
            nrm = np.zeros((int(m.sum()), 3))
            nrm[:, 2] = 1.0
            contacts.append((m, nrm))

        # ---- holed plate: solid annular slab r in [hole_r, plate_r_out],
        #      z in [plate_z0, plate_z1]; hole is free passage.
        r = np.hypot(p[:, 0], p[:, 1])
        r_safe = np.maximum(r, 1e-9)
        inside = (r > self.hole_radius) & (p[:, 2] > self.plate_z0) & (
            p[:, 2] < self.plate_z1
        ) & (r < self.plate_r_out + 0.5)
        if inside.any():
            idx = np.nonzero(inside)[0]
            p_down = p[idx, 2] - self.plate_z0   # push out bottom
            p_up = self.plate_z1 - p[idx, 2]     # push out top
            p_in = r[idx] - self.hole_radius     # push into hole (radially in)
            k = np.argmin(np.stack([p_down, p_up, p_in]), axis=0)
            nrm = np.zeros((len(idx), 3))
            # bottom face
            sel = idx[k == 0]
            if len(sel):
                p[sel, 2] = self.plate_z0
                nrm[k == 0, 2] = -1.0
            # top face
            sel = idx[k == 1]
            if len(sel):
                p[sel, 2] = self.plate_z1
                nrm[k == 1, 2] = 1.0
            # hole wall: move radially inward to r = hole_radius
            sel = idx[k == 2]
            if len(sel):
                s = self.hole_radius / r_safe[sel]
                p[sel, 0] *= s
                p[sel, 1] *= s
                rr2 = np.maximum(np.hypot(p[sel, 0], p[sel, 1]), 1e-9)
                nrm[k == 2, 0] = -p[sel, 0] / rr2
                nrm[k == 2, 1] = -p[sel, 1] / rr2
            contacts.append((_mask_from_idx(idx, len(p)), nrm))

        # ---- pusher: solid cylinder, kinematic ----
        zc = self._pusher_center(t)
        zb, zt = zc - self.pusher_hh, zc + self.pusher_hh
        r = np.hypot(p[:, 0], p[:, 1])
        inside = (r < self.pusher_radius) & (p[:, 2] > zb) & (p[:, 2] < zt)
        if inside.any():
            idx = np.nonzero(inside)[0]
            p_bot = p[idx, 2] - zb     # push out bottom (downward)
            p_top = zt - p[idx, 2]     # push out top
            p_side = self.pusher_radius - r[idx]  # push out side
            k = np.argmin(np.stack([p_bot, p_top, p_side]), axis=0)
            nrm = np.zeros((len(idx), 3))
            sel = idx[k == 0]
            if len(sel):
                p[sel, 2] = zb
                nrm[k == 0, 2] = -1.0
            sel = idx[k == 1]
            if len(sel):
                p[sel, 2] = zt
                nrm[k == 1, 2] = 1.0
            sel = idx[k == 2]
            if len(sel):
                s = self.pusher_radius / np.maximum(r[sel], 1e-9)
                p[sel, 0] *= s
                p[sel, 1] *= s
                rr = np.hypot(p[sel, 0], p[sel, 1]).clip(1e-9)
                nrm[k == 2, 0] = p[sel, 0] / rr
                nrm[k == 2, 1] = p[sel, 1] / rr
            contacts.append((_mask_from_idx(idx, len(p)), nrm))

        return contacts

    def _signed_distance(self, p: np.ndarray, t: float) -> np.ndarray:
        """Approximate signed distance to nearest collider (negative inside)."""
        sd = p[:, 2] - self.ground_z  # ground

        # plate slab (solid where r>hole_r and z0<z<z1)
        r = np.hypot(p[:, 0], p[:, 1])
        sd_plate = np.maximum(
            np.maximum(self.plate_z0 - p[:, 2], p[:, 2] - self.plate_z1),
            self.hole_radius - r,
        )
        sd = np.minimum(sd, sd_plate)

        # pusher cylinder
        zc = self._pusher_center(t)
        zb, zt = zc - self.pusher_hh, zc + self.pusher_hh
        sd_push = np.maximum(
            np.maximum(zb - p[:, 2], p[:, 2] - zt), self.pusher_radius - r
        )
        sd = np.minimum(sd, sd_push)
        return sd

    # ------------------------------------------------------------------ step
    def _solve_distance(self, p, ia, ib, rest, k):
        d = p[ib] - p[ia]
        L = np.linalg.norm(d, axis=1)
        Ls = np.maximum(L, 1e-12)
        corr = (k * (Ls - rest) / Ls)[:, None] * d * 0.5
        np.add.at(p, ia, corr)
        np.add.at(p, ib, -corr)

    def _step(self, h: float) -> None:
        t1 = self.t + h
        # semi-implicit prediction with gravity + mild drag damping
        self.v += self.gravity[None, :] * h
        if self.damp > 0.0:
            self.v /= 1.0 + self.damp * h
        p = self.x + self.v * h

        for _ in range(self.iterations):
            self._solve_distance(
                p, self.edges[:, 0], self.edges[:, 1], self.edge_rest, self.k_edge
            )
            self._solve_distance(
                p, self.bend[:, 0], self.bend[:, 1], self.bend_rest, self.k_bend
            )
            self._diag["constraint_iterations"] += 1

        x_old = self.x
        contacts = self._project_colliders(p, t1)

        # velocity from position change (includes kinematic pusher work)
        v_new = (p - x_old) / h

        # friction + restitution at contacts
        fric = 1.0 / (1.0 + self.mu)
        for mask, nrm in contacts:
            idx = np.nonzero(mask)[0]
            nn = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            vv = v_new[idx]
            vn = np.einsum("ij,ij->i", vv, nn)
            vt = vv - vn[:, None] * nn
            vt *= fric
            vn_out = np.where(vn < 0.0, -self.rest * vn, vn)
            v_new[idx] = vt + vn_out[:, None] * nn

        self.x = p
        self.v = v_new
        self.t = t1
        self._diag["steps"] += 1
        sd = self._signed_distance(self.x, self.t)
        self._diag["max_penetration"] = max(
            self._diag["max_penetration"], float(-sd.min())
        )
        self._diag["contacts_total"] += int((sd < self.contact_length_scale).sum())

    # -------------------------------------------------------------- sampling
    def _deformation_gradient(self) -> np.ndarray:
        f = self.faces
        x = self.x
        a, b, c = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        Ds = np.stack([b - a, c - a], axis=2)  # (F,3,2)
        Ff = np.einsum("fij,fjk->fik", Ds, self.face_pinv)  # (F,3,3)
        n1 = np.cross(Ds[:, :, 0], Ds[:, :, 1])
        nn = np.linalg.norm(n1, axis=1, keepdims=True)
        n1 = n1 / np.maximum(nn, 1e-12)
        Ff = Ff + n1[:, :, None] * self.face_n0[:, None, :]
        Fv = np.zeros((self.n, 3, 3))
        cnt = np.zeros(self.n)
        for corner in range(3):
            vi = f[:, corner]
            np.add.at(Fv, vi, Ff)
            np.add.at(cnt, vi, 1.0)
        Fv /= np.maximum(cnt, 1.0)[:, None, None]
        return Fv

    def _make_frame(self, time: float) -> StateFrame:
        sd = self._signed_distance(self.x, time)
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": sd,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.t:
            time = self.t
        eps = 1e-12
        while self.t < time - eps:
            h = min(self.sub_dt, time - self.t)
            self._step(h)
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)


def _mask_from_idx(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m
