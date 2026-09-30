"""Physically-based rigid-body solver for the self-supporting parabolic arch.

Model
-----
15 rigid voussoir blocks (wedge hexahedra loaded from the scene assets) rest on
a static ground plane at z=0 and press against their neighbours along shared
(mating) radial faces.  Gravity is the only external force.  Contacts are
handled with a penalty (compliant) contact model integrated by semi-implicit
Euler with small substeps:

* normal direction  : unilateral spring-damper, F_n = max(0, k_n*pen - c_n*gdot)
* tangential        : regularised static-friction spring (brush model) with a
  Coulomb cap |F_t| <= mu * F_n  (mu = 0.6 from the scene config)

Contact points are the corners of the mating faces, frozen in each body's local
frame; because the arch is statically stable the relative motion stays in the
sub-millimetre regime, so the linearised contact geometry is exact to within
the penetration depth.  The assembled arch therefore settles slightly and then
maintains a stable compressive load path -- the observable phenomenon.

Exported trajectory data (per StateFrame):
  state.position/velocity/orientation/angular_velocity/mass/body_id
  fields.external_force          [N,3]  gravity per block
  fields.contact_signed_distance [C]    geometric gap at every contact sample
  fields.constraint_value        [N,3]  current block positions (drift measure)
  fields.constraint_target       [N,3]  frozen initial positions
  scalars.constraint_scale       arch rise (keystone height)
  scalars.contact_length_scale   mean block width
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame

_WORKSPACE = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# contact tuning (stable for h = 5e-4 with k_n = 2e6, m_min = 27 kg)
# --------------------------------------------------------------------------
_H = 5.0e-4           # physics substep [s]
_KN = 2.0e6           # normal penalty stiffness [N/m]
_ZETA = 0.4           # normal contact damping ratio
_KT = _KN / 3.0       # tangential (friction) spring stiffness [N/m]
_MU = 0.6             # Coulomb friction coefficient (paper value)
_GROUND_TOL = 1.0e-4  # vertices this close to z=0 touch the ground


def _load_obj(path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
            tris.append((idx[0], idx[1], idx[2]))
    return np.asarray(verts, dtype=float), tris


def _planar_faces(verts: np.ndarray, tris) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group triangles into planar polygonal faces with outward unit normals.

    Returns list of (normal[3], corners[K,3]) in local coordinates.
    """
    groups: dict[tuple, dict] = {}
    for a, b, c in tris:
        p0, p1, p2 = verts[a], verts[b], verts[c]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        fc = (p0 + p1 + p2) / 3.0
        # mesh centroid is ~local origin: outward normal has positive dot with fc
        if np.dot(n, fc) < 0.0:
            n = -n
        key = (
            round(float(n[0]), 4), round(float(n[1]), 4), round(float(n[2]), 4),
            round(float(np.dot(n, fc)), 4),
        )
        g = groups.setdefault(key, {"n": n, "pts": {}})
        for p in (p0, p1, p2):
            g["pts"][tuple(np.round(p, 9))] = p
    return [(g["n"], np.array(list(g["pts"].values()))) for g in groups.values()]


class BenchmarkSolver:
    """Rigid-body frictional-contact solver for the voussoir arch."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self._seed = int(seed)
        g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._gravity = g
        try:
            self._mu = float(cfg.get("friction", {}).get("coefficient", _MU))
        except AttributeError:
            self._mu = _MU

        state0 = scene.initial_state()
        self._pos0 = np.array(state0.position, dtype=float)      # [N,3]
        self._mass = np.array(state0.mass, dtype=float)          # [N]
        self._body_id = np.array(state0.body_id, dtype=int)      # [N]
        n = len(self._mass)
        self._n = n

        # dynamic state
        self._pos = self._pos0.copy()
        self._vel = np.zeros((n, 3))
        self._quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))  # xyzw
        self._omeg = np.zeros((n, 3))
        self._t = 0.0

        # inertia tensors from local bounding boxes (wedge ~ box approximation)
        meshes = []
        self._inv_inertia = np.zeros((n, 3, 3))
        extents = []
        for i in range(n):
            verts, tris = _load_obj(_WORKSPACE / "assets" / f"voussoir_{i:02d}.obj")
            meshes.append((verts, tris))
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            ex, ey, ez = hi - lo
            extents.append((ex, ey, ez))
            m = self._mass[i]
            I = (m / 12.0) * np.diag([ey**2 + ez**2, ex**2 + ez**2, ex**2 + ey**2])
            self._inv_inertia[i] = np.linalg.inv(I)
        mean_w = float(np.mean([e[0] for e in extents]))
        self._contact_length_scale = max(mean_w, 1e-3)
        self._constraint_scale = float(np.max(np.abs(self._pos0[:, 2])))

        # external force: gravity only
        self._ext_force = self._mass[:, None] * self._gravity[None, :]

        # ---------------- build contacts ----------------
        # pair contacts: mating face of block i facing block i+1
        # each entry: dict(a, b, pA, pB, n) with local points; b >= 0
        self._pair_contacts = []
        for i in range(n - 1):
            verts_a, tris_a = meshes[i]
            faces = _planar_faces(verts_a, tris_a)
            xa = self._pos0[i]
            xb = self._pos0[i + 1]
            for nrm, corners in faces:
                c_f = corners.mean(axis=0) + xa          # world face centroid
                d = xb - c_f
                if np.dot(nrm, d) > 0.05 and np.linalg.norm(d) < 1.2:
                    # mating face; corners are shared world points
                    world = corners + xa                 # [K,3]
                    for w in world:
                        self._pair_contacts.append(
                            (i, i + 1, w - xa, w - xb, nrm.copy())
                        )
        # ground contacts: local vertices resting on z = 0
        self._ground_contacts = []
        for i in range(n):
            verts, _ = meshes[i]
            wz = verts[:, 2] + self._pos0[i, 2]
            for j in np.where(wz < _GROUND_TOL)[0]:
                self._ground_contacts.append((i, verts[j].copy()))

        self._n_contacts = len(self._pair_contacts) + len(self._ground_contacts)
        # friction spring (tangential displacement) state
        self._delta_t = np.zeros((self._n_contacts, 3))

        # signed distance buffer, fixed ordering across frames
        self._sd = np.zeros(self._n_contacts)

        # damping coefficient from lightest effective mass
        self._cn = 2.0 * _ZETA * np.sqrt(_KN * self._mass.min())

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _substep(self, h: float) -> None:
        n = self._n
        rot = [self._quat_to_rot(self._quat[i]) for i in range(n)]
        forces = self._ext_force.copy()
        torques = np.zeros((n, 3))
        cn, kt, mu = self._cn, _KT, self._mu

        ci = 0
        # ---- block-block pair contacts ----
        for (a, b, pA_l, pB_l, nrm) in self._pair_contacts:
            Ra, Rb = rot[a], rot[b]
            rA = Ra @ pA_l
            rB = Rb @ pB_l
            pA = self._pos[a] + rA
            pB = self._pos[b] + rB
            gap = float(np.dot(pB - pA, nrm))
            self._sd[ci] = gap
            vA = self._vel[a] + np.cross(self._omeg[a], rA)
            vB = self._vel[b] + np.cross(self._omeg[b], rB)
            gdot = float(np.dot(vB - vA, nrm))
            pen = -gap
            fn = _KN * pen - cn * gdot
            if fn > 0.0:
                f_n_vec = fn * nrm                    # A->B normal
                # friction (brush) spring on relative tangential motion
                vrel = vA - vB
                vt = vrel - np.dot(vrel, nrm) * nrm
                d = self._delta_t[ci]
                d += vt * h
                f_t = -kt * d                         # force on A (tangential)
                mag = np.linalg.norm(f_t)
                cap = mu * fn
                if mag > cap and mag > 0.0:
                    scale = cap / mag
                    d *= scale
                    f_t *= scale
                fA = -f_n_vec + f_t
                forces[a] += fA
                forces[b] -= fA
                torques[a] += np.cross(rA, fA)
                torques[b] += np.cross(rB, -fA)
            else:
                # separated: relax the friction spring
                self._delta_t[ci] *= 0.0
            ci += 1

        # ---- block-ground contacts (normal +z, ground static) ----
        up = np.array([0.0, 0.0, 1.0])
        for (a, p_l) in self._ground_contacts:
            Ra = rot[a]
            rA = Ra @ p_l
            pA = self._pos[a] + rA
            gap = pA[2]
            self._sd[ci] = gap
            vA = self._vel[a] + np.cross(self._omeg[a], rA)
            gdot = vA[2]
            pen = -gap
            fn = _KN * pen - cn * gdot
            if fn > 0.0:
                f_n_vec = fn * up                     # pushes block up
                vt = vA - gdot * up
                d = self._delta_t[ci]
                d += vt * h
                f_t = -kt * d
                mag = np.linalg.norm(f_t)
                cap = mu * fn
                if mag > cap and mag > 0.0:
                    scale = cap / mag
                    d *= scale
                    f_t *= scale
                fA = f_n_vec + f_t
                forces[a] += fA
                torques[a] += np.cross(rA, fA)
            else:
                self._delta_t[ci] *= 0.0
            ci += 1

        # ---- semi-implicit integration ----
        acc = forces / self._mass[:, None]
        self._vel += acc * h
        for i in range(n):
            dw = rot[i] @ (self._inv_inertia[i] @ (rot[i].T @ torques[i]))
            self._omeg[i] += dw * h
        self._pos += self._vel * h
        for i in range(n):
            wx, wy, wz = self._omeg[i]
            x, y, z, w = self._quat[i]
            qd = 0.5 * np.array([
                w * wx + y * wz - z * wy,
                w * wy + z * wx - x * wz,
                w * wz + x * wy - y * wx,
                -(x * wx + y * wy + z * wz),
            ])
            q = self._quat[i] + qd * h
            self._quat[i] = q / np.linalg.norm(q)

    # ----------------------------------------------------------------- output
    def _make_frame(self, time: float) -> StateFrame:
        # refresh signed distances at the current configuration (no forces)
        ci = 0
        rot = [self._quat_to_rot(self._quat[i]) for i in range(self._n)]
        for (a, b, pA_l, pB_l, nrm) in self._pair_contacts:
            pA = self._pos[a] + rot[a] @ pA_l
            pB = self._pos[b] + rot[b] @ pB_l
            self._sd[ci] = float(np.dot(pB - pA, nrm))
            ci += 1
        for (a, p_l) in self._ground_contacts:
            self._sd[ci] = float((self._pos[a] + rot[a] @ p_l)[2])
            ci += 1

        frame = StateFrame(
            time=float(time),
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            orientation=self._quat.copy(),
            angular_velocity=self._omeg.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields={
                "external_force": self._ext_force.copy(),
                "contact_signed_distance": self._sd.copy(),
                "constraint_value": self._pos.copy(),
                "constraint_target": self._pos0.copy(),
            },
            scalars={
                "constraint_scale": self._constraint_scale,
                "contact_length_scale": self._contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        while self._t < target - 1e-12:
            h = min(_H, target - self._t)
            self._substep(h)
            self._t += h
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        drift = float(np.max(np.linalg.norm(self._pos - self._pos0, axis=1))) \
            if hasattr(self, "_pos") else 0.0
        min_gap = float(np.min(self._sd)) if hasattr(self, "_sd") else 0.0
        return {
            "max_block_drift_m": drift,
            "min_contact_gap_m": min_gap,
            "num_contacts": int(getattr(self, "_n_contacts", 0)),
            "collapsed": drift > 0.25,
        }
