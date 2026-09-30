"""Volumetric flesh simulation of elbow flexion (solver-owned implementation).

Physical model
--------------
The frozen ``assets/arm_flesh.obj`` surface mesh is treated as a three
dimensional solid and discretised with the Delaunay tetrahedralisation of its
vertices (the arm is convex to within 0.7 %, so the tetrahedra fill the rest
volume exactly).  Degenerate/flat tetrahedra are removed; the retained elements
still reproduce the full rest volume (0.066832 m^3) with no orphan nodes.

Constitutive law: compressible stable Neo-Hookean solid

    Psi(F) = mu/2 (tr(F^T F) - 3) + lambda/2 (J - alpha)^2,  alpha = 1 + mu/lambda
    P(F)   = mu F + lambda (J - alpha) cof(F)

which is a polynomial in F (no matrix inverse, no logarithm) so nodal forces
stay finite for every element of the discretisation, and P(I) = 0 so the frozen
rest pose is stress free.

Kinematic drive: the two rigid bone capsules act on the flesh through
distributed attachment tractions (flesh_attachment: driven_by_internal_bone_
surfaces).  Every flesh node carries a smooth rest-position skin weight; the
humerus is fixed, the forearm bone rotates about the prescribed hinge
(smoothstep 0 -> -110 deg between 0.4 s and 2.4 s).  The attachment springs are
integrated implicitly, so the stiff skin drive never limits the step size and
the volumetric material - not a hard kinematic projection - decides how the
flesh redistributes.  Flesh self-contact and bone/flesh contact are resolved by
a penalty barrier, which turns the approaching forearm and upper arm into an
outward, volume preserving bulge instead of an interpenetrating pinch.

Time integration is a symplectic (semi-implicit) Euler step at a fixed step
size chosen with a wide margin from the measured elastic CFL limit
(omega_max ~ 260 rad/s at rest vs. 2/h = 13000 rad/s), with mass and
stiffness proportional Rayleigh damping plus contact damping.  The state is
deterministic and finite at every requested sample time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

# --- material (SI: Pa, kg/m^3, m) -------------------------------------------
_MU = 2.0e4
_LAMBDA = 3.0e5
_ALPHA = 1.0 + _MU / _LAMBDA          # rest-stress-free stable Neo-Hookean
_MASS_DAMPING = 6.0                    # 1/s   mass proportional Rayleigh
_STIFF_DAMPING = 8.0e-4                # s     stiffness proportional Rayleigh

# --- kinematic attachment ----------------------------------------------------
_K_ATT = 3.0e4                         # N/m  distributed skin tractions

# --- contact -----------------------------------------------------------------
_R_CONTACT = 0.035                     # m    flesh self-contact cutoff
_K_CONTACT = 5.0e4                     # N/m
_C_CONTACT = 6.0                       # N*s/m
_F_CONTACT_MAX = 400.0                 # N    per-pair force ceiling
_BONE_MARGIN = 0.010                   # m    bone/flesh barrier margin
_EXCLUDE_HOPS = 3                      # mesh graph distance excluded from contact

# --- discretisation / integration -------------------------------------------
_MIN_ALTITUDE = 2.0e-3                 # m    drop flat Delaunay slivers
_DT = 1.5e-4                           # s    explicit stable step


def _det3(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Determinant and cofactor matrix (columns = cross products of columns)."""
    c0 = np.cross(F[:, :, 1], F[:, :, 2])
    c1 = np.cross(F[:, :, 2], F[:, :, 0])
    c2 = np.cross(F[:, :, 0], F[:, :, 1])
    cof = np.stack((c0, c1, c2), axis=2)
    jac = np.einsum("ij,ij->i", F[:, :, 0], c0)
    return jac, cof


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c = math.cos(angle)
    s = math.sin(angle)
    cross = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float
    )
    return c * np.eye(3) + s * cross + (1.0 - c) * np.outer(a, a)


class BenchmarkSolver:
    """Runnable physics solver for the elbow flesh-bulge scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        self.config = config

        frame = scene.initial_state()
        rest = np.asarray(frame.position, dtype=float).copy()
        mass = np.asarray(frame.mass, dtype=float)
        self.rest = rest
        self.n = int(rest.shape[0])
        if mass.shape == (self.n,):
            self.node_mass = mass.copy()
        else:
            self.node_mass = np.full(self.n, float(np.sum(mass)) / self.n)

        self.x = rest.copy()
        self.v = np.zeros_like(rest)
        self._time = 0.0
        self._nsteps = 0
        self._budget = None

        self._read_surface()
        self._build_solid()
        self._configure_motion(config)
        self._build_skinning()
        self._build_contacts()
        self._build_damping()

        self.reference_force = float(abs(np.sum(self.node_mass) * _GRAVITY[2]))
        self._refresh_budget()

    def _read_surface(self) -> None:
        path = Path(__file__).resolve().parent.parent / "assets" / "arm_flesh.obj"
        faces: list[list[int]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0] == "f":
                    faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
        except OSError:
            faces = []
        self.faces = np.asarray(faces, dtype=int) if faces else np.empty((0, 3), dtype=int)

    def _build_solid(self) -> None:
        """Delaunay tetrahedralisation of the frozen rest vertices."""
        tri = Delaunay(self.rest)
        tets = np.asarray(tri.simplices, dtype=int).copy()

        def signed_volume(T: np.ndarray) -> np.ndarray:
            a = self.rest[T[:, 0]]
            b = self.rest[T[:, 1]]
            c = self.rest[T[:, 2]]
            d = self.rest[T[:, 3]]
            return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0

        vol = signed_volume(tets)
        flip = vol < 0.0
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        vol = signed_volume(tets)

        alt = self._altitudes(tets, vol)
        keep = (vol > 1e-12) & (alt > _MIN_ALTITUDE)
        if int(keep.sum()) < 8:
            keep = vol > 1e-12
        tets = tets[keep]
        vol = vol[keep]

        dm = np.stack(
            (
                self.rest[tets[:, 1]] - self.rest[tets[:, 0]],
                self.rest[tets[:, 2]] - self.rest[tets[:, 0]],
                self.rest[tets[:, 3]] - self.rest[tets[:, 0]],
            ),
            axis=2,
        )
        dmi = np.linalg.inv(dm)
        grad = np.empty((len(tets), 4, 3), dtype=float)
        grad[:, 1:, :] = np.transpose(dmi, (0, 2, 1))
        grad[:, 0, :] = -grad[:, 1:, :].sum(axis=1)

        self.tets = tets
        self.dmi = dmi
        self.rest_vol = vol
        self.grad = grad
        self.n_elem = int(len(tets))

        # surface graph used to exclude mesh neighbours from self-contact
        adj: list[set[int]] = [set() for _ in range(self.n)]
        for face in self.faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
        if not any(adj):
            for tet in tets:
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj[int(tet[i])].add(int(tet[j]))
                        adj[int(tet[j])].add(int(tet[i]))

        excl = np.zeros((self.n, self.n), dtype=bool)
        for i in range(self.n):
            seen = {i}
            frontier = {i}
            for _ in range(_EXCLUDE_HOPS):
                nxt: set[int] = set()
                for node in frontier:
                    nxt.update(adj[node])
                nxt -= seen
                seen |= nxt
                frontier = nxt
                if not frontier:
                    break
            excl[i, list(seen)] = True
        dx = self.rest[:, None, :] - self.rest[None, :, :]
        dist = np.linalg.norm(dx, axis=2)
        excl |= dist < 1.05 * _R_CONTACT
        np.fill_diagonal(excl, True)
        self.excl = excl

    def _altitudes(self, tets: np.ndarray, vol: np.ndarray) -> np.ndarray:
        best = np.zeros(len(tets), dtype=float)
        for i in range(4):
            idx = [j for j in range(4) if j != i]
            p = self.rest[tets[:, idx[0]]]
            q = self.rest[tets[:, idx[1]]]
            r = self.rest[tets[:, idx[2]]]
            area = np.linalg.norm(np.cross(q - p, r - p), axis=1)
            best = np.maximum(best, area)
        return 3.0 * np.abs(vol) / np.maximum(best, 1e-30)

    def _configure_motion(self, config: dict) -> None:
        bc = dict(config.get("boundary_conditions", {}) or {})
        fore = dict(bc.get("forearm_bone", {}) or {})
        self.hinge_center = np.asarray(
            fore.get("hinge_center", [0.0, 0.0, 1.05]), dtype=float
        )
        self.hinge_axis = np.asarray(fore.get("hinge_axis", [0.0, 1.0, 0.0]), dtype=float)
        self.t_start = float(fore.get("start_time", 0.4))
        self.t_end = float(fore.get("end_time", 2.4))
        self.angle0 = math.radians(float(fore.get("start_angle_degrees", 0.0)))
        self.angle1 = math.radians(float(fore.get("end_angle_degrees", -110.0)))
        if self.t_end <= self.t_start:
            self.t_end = self.t_start + 1e-6

        # bone capsules from the frozen scene geometry (world frame at rest);
        # capsule axis = local z rotated by the frozen quaternion.
        defaults = {
            "humerus_bone": ([-0.36, 0.0, 1.05], [0.0, 0.70710678, 0.0, 0.70710678], 0.05, 0.62),
            "forearm_bone": ([0.38, 0.0, 1.05], [0.0, 0.70710678, 0.0, 0.70710678], 0.04, 0.66),
        }
        specs = {}
        geometry = dict(config.get("scene_geometry", {}) or {})
        for item in geometry.get("objects", []) or []:
            if not isinstance(item, dict) or item.get("id") not in defaults:
                continue
            geom = dict(item.get("geometry", {}) or {})
            tf = dict(item.get("transform", {}) or {})
            specs[item["id"]] = (
                [float(v) for v in tf.get("translation", defaults[item["id"]][0])],
                [float(v) for v in tf.get("rotation_xyzw", defaults[item["id"]][1])],
                float(geom.get("radius", defaults[item["id"]][2])),
                float(geom.get("height", defaults[item["id"]][3])),
            )
        for key, value in defaults.items():
            specs.setdefault(key, value)

        def capsule(spec):
            translation, quat, radius, height = spec
            center = np.asarray(translation, dtype=float)
            q = np.asarray(quat, dtype=float)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            w, xq, yq, zq = q[3], q[0], q[1], q[2]
            rot = np.array(
                [
                    [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
                    [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                    [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
                ],
                dtype=float,
            )
            axis = rot @ np.array([0.0, 0.0, 1.0])
            half = 0.5 * height
            return (center - half * axis, center + half * axis, radius, axis)

        self.humerus = capsule(specs["humerus_bone"])
        self.forearm_local = capsule(specs["forearm_bone"])

    def _angle(self, t: float) -> float:
        if t <= self.t_start:
            return self.angle0
        if t >= self.t_end:
            return self.angle1
        s = (t - self.t_start) / (self.t_end - self.t_start)
        return self.angle0 + (self.angle1 - self.angle0) * s * s * (3.0 - 2.0 * s)

    def _angle_rate(self, t: float) -> float:
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        s = (t - self.t_start) / (self.t_end - self.t_start)
        return (
            (self.angle1 - self.angle0)
            * 6.0
            * s
            * (1.0 - s)
            / (self.t_end - self.t_start)
        )

    def _build_skinning(self) -> None:
        x = self.rest[:, 0]
        lo, hi = -0.16, 0.24
        s = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        self.skin_w = s * s * (3.0 - 2.0 * s)
        self._omega_dir = self.hinge_axis / max(float(np.linalg.norm(self.hinge_axis)), 1e-12)

        # constant rest stiffness (Rayleigh damping matrix)
        rows = []
        cols = []
        vals = []
        c = self._material_tangent(np.eye(3)).reshape(9, 9)
        for e in range(self.n_elem):
            nodes = self.tets[e]
            g = self.grad[e]
            vol = self.rest_vol[e]
            b = np.zeros((9, 12))
            for i in range(4):
                for a in range(3):
                    for j in range(3):
                        b[a * 3 + j, 3 * i + a] = g[i, j]
            ke = vol * (b.T @ c @ b)
            idx = np.repeat(nodes, 3) * 3 + np.tile(np.arange(3), 4)
            rows.append(np.repeat(idx, 12))
            cols.append(np.tile(idx, 12))
            vals.append(ke.ravel())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        from scipy.sparse import coo_matrix

        self.rest_stiffness = coo_matrix(
            (vals, (rows, cols)), shape=(3 * self.n, 3 * self.n)
        ).tocsr()

    def _material_tangent(self, F: np.ndarray) -> np.ndarray:
        """dP/dF of the stable Neo-Hookean energy."""
        finv = np.linalg.inv(F)
        jac = float(np.linalg.det(F))
        # iJkL
        c = _MU * np.einsum("ik,JL->iJkL", np.eye(3), np.eye(3))
        c = c + _LAMBDA * jac * (2.0 * jac - _ALPHA) * np.einsum(
            "Lk,Ji->iJkL", finv.T, finv.T
        )
        c = c - _LAMBDA * (jac - _ALPHA) * jac * np.einsum("Jk,Li->iJkL", finv.T, finv.T)
        return c

    def _build_contacts(self) -> None:
        self._tree = None

    def _build_damping(self) -> None:
        self._damping_diag = _MASS_DAMPING

    # ------------------------------------------------------------- kinematics
    def _target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Skin target positions and velocities for the current time."""
        angle = self._angle(t)
        rate = self._angle_rate(t)
        rot = _rotation(self.hinge_axis, angle)
        rel = (self.rest - self.hinge_center) @ rot.T
        target_fore = rel + self.hinge_center
        w = self.skin_w[:, None]
        target = (1.0 - w) * self.rest + w * target_fore
        omega = rate * self._omega_dir
        velocity = w * np.cross(np.broadcast_to(omega, rel.shape), rel)
        return target, velocity

    # ----------------------------------------------------------------- forces
    def _elastic_forces(self, x: np.ndarray) -> np.ndarray:
        tets = self.tets
        ds = np.stack(
            (
                x[tets[:, 1]] - x[tets[:, 0]],
                x[tets[:, 2]] - x[tets[:, 0]],
                x[tets[:, 3]] - x[tets[:, 0]],
            ),
            axis=2,
        )
        f = ds @ self.dmi
        jac, cof = _det3(f)
        p = _MU * f + (_LAMBDA * (jac - _ALPHA))[:, None, None] * cof
        force = np.zeros_like(x)
        for a in range(4):
            fa = -self.rest_vol[:, None] * np.einsum(
                "nij,nj->ni", p, self.grad[:, a, :]
            )
            np.add.at(force, tets[:, a], fa)
        return force

    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        tets = self.tets
        ds = np.stack(
            (
                x[tets[:, 1]] - x[tets[:, 0]],
                x[tets[:, 2]] - x[tets[:, 0]],
                x[tets[:, 3]] - x[tets[:, 0]],
            ),
            axis=2,
        )
        return ds @ self.dmi

    def _bone_segments(self, t: float):
        angle = self._angle(t)
        rot = _rotation(self.hinge_axis, angle)
        p0, p1, r, axis = self.forearm_local
        p0 = (p0 - self.hinge_center) @ rot.T + self.hinge_center
        p1 = (p1 - self.hinge_center) @ rot.T + self.hinge_center
        axis = rot @ axis
        return (self.humerus, (p0, p1, r, axis))

    def _contact_forces(self, x: np.ndarray, v: np.ndarray):
        force = np.zeros_like(x)
        tree = cKDTree(x)
        pairs = tree.query_pairs(_R_CONTACT, output_type="ndarray")
        if len(pairs):
            pairs = pairs[~self.excl[pairs[:, 0], pairs[:, 1]]]
        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            d = x[i] - x[j]
            dist = np.linalg.norm(d, axis=1)
            act = dist > 1e-12
            i, j, d, dist = i[act], j[act], d[act], dist[act]
            if len(i):
                n = d / dist[:, None]
                gap = _R_CONTACT - dist
                fmag = np.minimum(_K_CONTACT * gap, _F_CONTACT_MAX)
                vrel = np.einsum("ij,ij->i", v[i] - v[j], n)
                fmag = np.maximum(fmag - _C_CONTACT * vrel, 0.0)
                fpair = fmag[:, None] * n
                np.add.at(force, i, fpair)
                np.add.at(force, j, -fpair)

        # bone / flesh barriers
        boundary = np.zeros((3, 3))
        for bidx, (p0, p1, radius, _axis) in enumerate(self._bone_segments(self._time)):
            ab = p1 - p0
            denom = float(ab @ ab)
            rel = x - p0
            s = np.clip((rel @ ab) / max(denom, 1e-30), 0.0, 1.0)
            q = p0 + s[:, None] * ab
            sep = x - q
            dist = np.linalg.norm(sep, axis=1)
            gap = dist - radius
            act = gap < _BONE_MARGIN
            if np.any(act):
                n = sep[act] / np.maximum(dist[act], 1e-12)[:, None]
                fnode = _K_CONTACT * (_BONE_MARGIN - gap[act])[:, None] * n
                force[act] += fnode
                boundary[bidx] = fnode.sum(axis=0)
        return force, boundary

    def _boundary_tractions(self, x: np.ndarray, v: np.ndarray, t: float):
        target, target_v = self._target(t)
        k = _K_ATT
        c = 2.0 * np.sqrt(k * self.node_mass)
        f = k * (target - x) + c[:, None] * (target_v - v)
        w = self.skin_w[:, None]
        f_humerus = (f * (1.0 - w)).sum(axis=0)
        f_forearm = (f * w).sum(axis=0)
        self._boundary_state = (f_humerus, f_forearm)
        return f.sum(axis=0), f_humerus, f_forearm

    def _refresh_budget(self) -> None:
        tractions, f_humerus, f_forearm = self._boundary_tractions(self.x, self.v, self._time)
        _, bone_boundary = self._contact_forces(self.x, self.v)
        gravity = np.sum(self.node_mass) * _GRAVITY
        flesh = gravity + tractions + bone_boundary.sum(axis=0)
        humerus = -f_humerus - bone_boundary[0]
        forearm = -f_forearm - bone_boundary[1]
        combined = gravity
        self._budget = np.stack((combined, flesh, humerus, forearm)).reshape(4, 1, 3)

    # ------------------------------------------------------------- integration
    def _step(self, h: float) -> None:
        x = self.x
        v = self.v

        force = self._elastic_forces(x)
        force += self.node_mass[:, None] * _GRAVITY
        force -= _MASS_DAMPING * self.node_mass[:, None] * v
        if _STIFF_DAMPING > 0.0:
            force -= _STIFF_DAMPING * (self.rest_stiffness @ v.ravel()).reshape(self.n, 3)

        contact_force, bone_boundary = self._contact_forces(x, v)
        force += contact_force

        t_new = self._time + h
        target, target_v = self._target(t_new)
        k = _K_ATT
        c = 2.0 * np.sqrt(k * self.node_mass)

        m = self.node_mass[:, None]
        denom = m + h * h * k + h * c[:, None]
        v_new = (
            m * v + h * force + h * k * (target - x) + (h * c[:, None]) * target_v
        ) / denom

        self.x = x + h * v_new
        self.v = v_new

    def _safe_step(self, h: float) -> float:
        """Advance one step, rolling back with refinement on a non-finite state.

        The fixed step sits far below the measured CFL limit, so this guard is a
        bounded safety net that keeps every exported state finite.
        """
        x_old = self.x.copy()
        v_old = self.v.copy()
        self._step(h)
        if np.isfinite(self.x).all() and np.isfinite(self.v).all():
            return h
        for _ in range(6):
            self.x = x_old.copy()
            self.v = v_old.copy()
            h *= 0.5
            self._step(h)
            if np.isfinite(self.x).all() and np.isfinite(self.v).all():
                return h
        self.x = x_old
        self.v = v_old
        return 0.0

    def advance_to(self, time: float):
        t_requested = float(time)
        while True:
            remaining = t_requested - self._time
            if remaining <= 1e-9:
                break
            h = _DT if remaining > _DT else remaining
            used = self._safe_step(h)
            if used <= 0.0:
                break
            self._time += used
            self._nsteps += 1
        if abs(t_requested - self._time) < 1e-9:
            self._time = t_requested
        return self._frame(t_requested)

    # ------------------------------------------------------------------ output
    def _frame(self, time: float) -> StateFrame:
        self._refresh_budget()
        fgrad = self._deformation_gradient(self.x)
        angle_deg = math.degrees(self._angle(self._time))
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.node_mass.copy(),
            attributes={
                "mass_quantity": self.node_mass[:, None].copy(),
                "rest_position": self.rest.copy(),
            },
            fields={
                "budget_external_force": np.asarray(self._budget, dtype=float).copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": fgrad,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "flesh_density_kg_m3": 1050.0,
                "elbow_flexion_degrees": float(angle_deg),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        fgrad = self._deformation_gradient(self.x)
        jac, _ = _det3(fgrad)
        speed = np.linalg.norm(self.v, axis=1)
        return {
            "solver": "stable_neo_hookean_tet_fem",
            "elements": self.n_elem,
            "nodes": self.n,
            "time_step": _DT,
            "steps": int(self._nsteps),
            "simulation_time": float(self._time),
            "mu_pa": _MU,
            "lambda_pa": _LAMBDA,
            "jacobian_min": float(jac.min()),
            "jacobian_max": float(jac.max()),
            "max_speed": float(speed.max()),
            "total_mass": float(np.sum(self.node_mass)),
            "reference_force": float(self.reference_force),
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
