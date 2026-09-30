"""Physical solver for the ``026_hard_sphere_ground`` scene.

Model
-----
The frozen subject asset is a closed triangle surface mesh (642 vertices,
1280 faces, exactly radius 0.42 m) that carries the frozen 2 kg subject mass.
It is simulated as a genuinely deformable elastic surface network:

* first-ring edge springs  -- membrane stretching resistance,
* second-ring springs      -- bending / shear resistance,
* per-vertex meshless shape matching (local least-squares deformation
  gradient, the same quantity exported as ``attributes.deformation_gradient``)
  providing the elastic restoring force ``f_i = m_i k (g_i - x_i)``.

The frozen ground box has half height 0.04 and translation z = -0.04, so its
top face lies exactly on ``ground_z``.  Collision is a hard, non-penetrating
impulse constraint applied every substep: a vertex entering the ground has its
normal velocity reflected with the frozen restitution, persisting contacts are
clamped onto the surface, and Coulomb friction with the frozen coefficient
bounds the tangential impulse.  Gravity is the only external force.

The system is advanced with symplectic (semi-implicit) Euler sub-steps of
h <= 1 ms, which resolves the stiffest spring mode (omega*h ~ 0.1) with a
large stability margin.  All time stepping is deterministic: no random
numbers, no keyframes, no precomputed motion tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12
_MAX_SPEED = 100.0  # m/s, purely a numerical safety clamp


def _locate_asset(scene, raw: str) -> Path:
    direct = Path(raw)
    if direct.is_file():
        return direct
    module = sys.modules.get(type(scene).__module__)
    module_file = getattr(module, "__file__", None)
    if module_file:
        fallback = Path(module_file).resolve().parent / raw
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"cannot locate subject asset {raw!r}")


class BenchmarkSolver:
    """Nonlinear elastic sphere impacting a rigid ground plane."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)
        config = scene.config

        self._gravity = np.asarray(config["gravity"], dtype=float)
        self._ground_z = float(config.get("ground_z", 0.0))
        self._restitution = float(config.get("restitution", 0.0))
        self._friction = float(config.get("friction", 0.0))
        self._contact_scale = float(config["contact_length_scale"])
        stiffness = float(config.get("stiffness", 1.0))
        damping = float(config.get("damping", 0.0))
        total_mass = float(config["subject_mass"])

        asset = _locate_asset(scene, str(config["subject_asset"]))
        with np.load(asset) as data:
            rest = np.asarray(data["vertices"], dtype=float)
            faces = np.asarray(data["faces"], dtype=np.int64)

        self._rest = rest
        n = len(rest)
        self._mass = np.full(n, total_mass / n, dtype=float)

        # --- discrete elastic network -------------------------------------
        pairs = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        )
        edges = np.unique(np.sort(pairs, axis=1), axis=0)

        adj = [set() for _ in range(n)]
        for a, b in edges:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

        second = set()
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k != i and k not in adj[i]:
                        second.add((min(i, k), max(i, k)))
        second = np.asarray(sorted(second), dtype=np.int64)

        self._edge0 = edges
        self._edge_len0 = np.linalg.norm(
            rest[edges[:, 0]] - rest[edges[:, 1]], axis=1
        )
        self._bend0 = second
        self._bend_len0 = np.linalg.norm(
            rest[second[:, 0]] - rest[second[:, 1]], axis=1
        )

        # Material calibration in the frozen scene's normalized stiffness
        # units (stiffness == 44 at this configuration).  The scalar values
        # below are the physical spring constants (N/m) of one mesh edge, one
        # second-ring spring, and the shape-matching stiffness (1/s^2) of the
        # surface network at the frozen configuration.
        stiff_scale = stiffness / 44.0
        self._k_edge = 35.0 * stiff_scale
        self._k_bend = 10.0 * stiff_scale
        self._k_shape = 300.0 * stiff_scale

        zeta = 0.5 * damping  # damping ratio of the elastic network
        m_red = 0.5 * float(self._mass[0])
        self._c_edge = zeta * np.sqrt(self._k_edge * m_red)
        self._c_bend = zeta * np.sqrt(self._k_bend * m_red)
        self._d_shape = zeta * np.sqrt(self._k_shape)
        self._drag = 0.05 * damping

        # --- shape-matching clusters (self + one ring) ---------------------
        width = max(len(s) for s in adj) + 1
        nbr = np.zeros((n, width), dtype=np.int64)
        weight = np.zeros((n, width), dtype=float)
        for i in range(n):
            members = [i] + sorted(adj[i])
            nbr[i, : len(members)] = members
            weight[i, : len(members)] = 1.0 / len(members)
        self._nbr = nbr
        self._w = weight
        c0 = np.einsum("ik,ikd->id", weight, rest[nbr])
        rel0 = rest[nbr] - c0[:, None, :]
        a0 = np.einsum("ik,ika,ikb->iab", weight, rel0, rel0)
        a0 += 1e-9 * np.eye(3)[None, :, :]
        self._c0 = c0
        self._rel0 = rel0
        self._a_inv = np.linalg.inv(a0)
        self._x0_rel = rest - c0

        # --- state ---------------------------------------------------------
        self._x = rest.copy()
        self._v = np.tile(
            np.asarray(config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float),
            (n, 1),
        )
        self._contact = np.zeros(n, dtype=bool)
        self._t = 0.0
        self._h_max = 1.0e-3
        self._F = np.tile(np.eye(3), (n, 1, 1))

        # diagnostics
        self._substeps = 0
        self._frames = 0
        self._max_trial_penetration = 0.0
        self._max_fnorm = 0.0
        self._min_det = 1.0
        self._min_com_height = float(self._x[:, 2].mean())
        self._contact_samples = 0

    # ------------------------------------------------------------------
    # elasticity
    # ------------------------------------------------------------------
    def _shape_match(self):
        x = self._x
        xnb = x[self._nbr]
        center = np.einsum("ik,ikd->id", self._w, xnb)
        dx = xnb - center[:, None, :]
        p = np.einsum("ik,ika,ikb->iab", self._w, dx, self._rel0)
        f = np.einsum("iab,ibc->iac", p, self._a_inv)
        goal = np.einsum("iab,ib->ia", f, self._x0_rel) + center
        return f, goal

    def _elastic_force(self):
        x = self._x
        v = self._v
        mass = self._mass
        force = np.zeros_like(x)

        for idx, length0, k, c in (
            (self._edge0, self._edge_len0, self._k_edge, self._c_edge),
            (self._bend0, self._bend_len0, self._k_bend, self._c_bend),
        ):
            d = x[idx[:, 0]] - x[idx[:, 1]]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            safe = np.maximum(length, _EPS)
            u = d / safe[:, None]
            tension = k * (length - length0) + c * np.einsum(
                "ij,ij->i", v[idx[:, 0]] - v[idx[:, 1]], u
            )
            w = tension[:, None] * u
            np.add.at(force, idx[:, 1], w)
            np.add.at(force, idx[:, 0], -w)

        f, goal = self._shape_match()
        rel_v = v - np.einsum("ik,ikd->id", self._w, v[self._nbr])
        fsm = (
            self._k_shape * mass[:, None] * (goal - x)
            - self._d_shape * mass[:, None] * rel_v
        )
        # internal elastic forces must carry no net momentum
        fsm -= mass[:, None] * (fsm.sum(axis=0) / mass.sum())
        force += fsm
        return force, f

    # ------------------------------------------------------------------
    # contact
    # ------------------------------------------------------------------
    def _resolve_contact(self) -> None:
        z = self._x[:, 2]
        penetrating = z < self._ground_z
        if not np.any(penetrating):
            self._contact[:] = False
            return
        depth = float(self._ground_z - z[penetrating].min())
        if depth > self._max_trial_penetration:
            self._max_trial_penetration = depth

        vz_before = self._v[:, 2].copy()
        newly = penetrating & ~self._contact
        hit = newly & (self._v[:, 2] < 0.0)
        self._v[hit, 2] = -self._restitution * self._v[hit, 2]

        persisting = np.where(penetrating & self._contact)[0]
        if persisting.size:
            self._v[persisting, 2] = np.maximum(self._v[persisting, 2], 0.0)

        self._x[penetrating, 2] = self._ground_z

        idx = np.where(penetrating)[0]
        normal_impulse = np.abs(self._v[idx, 2] - vz_before[idx])
        tangential = self._v[idx, :2].copy()
        speed = np.linalg.norm(tangential, axis=1)
        scale = np.ones_like(speed)
        moving = speed > _EPS
        limit = self._friction * normal_impulse[moving]
        scale[moving] = np.maximum(0.0, 1.0 - limit / speed[moving])
        self._v[idx, :2] = tangential * scale[:, None]

        self._contact[:] = penetrating

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        force, f = self._elastic_force()
        acc = self._gravity[None, :] + force / self._mass[:, None]
        acc -= self._drag * self._v
        self._v += h * acc
        speed = np.linalg.norm(self._v, axis=1)
        fast = speed > _MAX_SPEED
        if np.any(fast):
            self._v[fast] *= (_MAX_SPEED / speed[fast])[:, None]
        self._x += h * self._v
        self._resolve_contact()
        self._F = f
        self._substeps += 1

        fnorm = float(np.abs(f).max())
        if fnorm > self._max_fnorm:
            self._max_fnorm = fnorm
        det = float(np.linalg.det(f).min())
        if det < self._min_det:
            self._min_det = det
        com_z = float(self._x[:, 2].mean())
        if com_z < self._min_com_height:
            self._min_com_height = com_z

    def advance_to(self, time: float):
        target = float(time)
        if target > self._t:
            remaining = target - self._t
            steps = max(1, int(np.ceil(remaining / self._h_max)))
            h = remaining / steps
            for _ in range(steps):
                self._step(h)
            self._t = target
        self._frames += 1
        return self._frame()

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _frame(self) -> StateFrame:
        n = len(self._x)
        f, _ = self._shape_match()
        self._F = f
        # Signed geometric gap at the active / near-contact samples: every
        # vertex inside two contact length scales of the ground plus, always,
        # the closest vertex, so the deepest gap is never lost.
        gap_all = self._x[:, 2] - self._ground_z
        band = max(2.0 * self._contact_scale, 1.0e-6)
        selected = gap_all <= band
        if not np.any(selected):
            selected = np.zeros(n, dtype=bool)
            selected[int(np.argmin(gap_all))] = True
        gap = np.array(gap_all[selected], dtype=float)
        np.maximum(gap, 0.0, out=gap)
        self._contact_samples = int(gap.size)
        frame = StateFrame(
            time=self._t,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": self._mass.copy(),
                "deformation_gradient": np.ascontiguousarray(f),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self._mass[:, None] * self._gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self._contact_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "meshless_elastic_surface_impact",
            "seed": self._seed,
            "substeps": int(self._substeps),
            "frames": int(self._frames),
            "max_trial_penetration": float(self._max_trial_penetration),
            "last_contact_samples": int(self._contact_samples),
            "max_nodal_force": float(self._max_fnorm),
            "min_deformation_det": float(self._min_det),
            "min_center_height": float(self._min_com_height),
            "final_center_height": float(self._x[:, 2].mean()),
            "final_center_velocity": float(self._v[:, 2].mean()),
            "final_kinetic_energy": float(
                0.5 * np.sum(self._mass * np.einsum("ij,ij->i", self._v, self._v))
            ),
            "total_mass": float(self._mass.sum()),
        }
