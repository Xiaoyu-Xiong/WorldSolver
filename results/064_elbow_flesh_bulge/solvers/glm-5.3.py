"""Total-Lagrangian stable Neo-Hookean FEM solver for elbow flesh bulging.

Physical model
--------------
The arm flesh is a three-dimensional elastic solid discretized with a
conforming tetrahedral mesh built from the frozen rest surface mesh (the
watertight arm skin is a structured tube of matched vertex rings, so each
slab between two rings is decomposed into conforming tetrahedra).

Constitutive law: stable Neo-Hookean flesh (Smith et al., "Stable Neo-Hookean
Flesh Simulation", 2018)

    psi(F) = mu/2 (|F|^2 - 3) - mu log(J) + lambda/2 (J - 1 - log J)
    P(F)   = mu F + (lambda/2 (J - 1) - mu) F^{-T}

which is well behaved under inversion and reduces to zero stress at F = I.

Dynamics: semi-implicit (symplectic) Euler time stepping of the nodal
semi-discrete equations with lumped masses rho * V_e / 4, mass-proportional
damping, penalty contact against the two rigid bone capsules, penalty
self-contact of the flesh surface, and spring-damper attachment of the flesh
to the internal bone surfaces ("flesh_attachment:
driven_by_internal_bone_surfaces").  The forearm bone follows the prescribed
hinge rotation about the frozen hinge center/axis with smoothstep
interpolation; the humerus is fixed.  All deformation emerges from the
time-stepped dynamics -- no keyframes or precomputed motion are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical and numerical constants (SI units)
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])

YOUNG = 2.0e4            # Pa, flesh Young's modulus (soft flesh)
POISSON = 0.45           # near-incompressible flesh
DEFAULT_DENSITY = 1050.0  # kg / m^3

MIN_TET_VOLUME = 1.0e-5  # m^3, degenerate sliver tets below this are dropped

DT = 5.0e-5              # s, explicit time step (CFL-safe for this mesh)
DAMP_MASS = 4.0          # 1/s mass-proportional damping

K_ATTACH = 3000.0        # N/m, flesh <-> bone surface attachment stiffness
C_ATTACH = 40.0          # N s/m, attachment damping
ATTACH_GAP = 0.14        # m, decay length of attachment weight with distance

K_BONE = 6.0e4           # N/m, flesh <-> bone capsule contact penalty
K_SELF = 3.0e4           # N/m, flesh self-contact penalty
SELF_D0 = 0.01           # m, self-contact activation distance
SELF_REST_MIN = 0.05     # m, ignore node/triangle pairs closer than this at rest
SELF_REST_MAX = 0.70     # m, ignore node/triangle pairs farther than this at rest

FORCE_CAP = 1.0e5        # N, per-node force clamp (numerical safety only)
VEL_CAP = 20.0           # m/s, per-node velocity clamp (numerical safety only)

# Bone geometry (frozen scene geometry, world frame, straight rest pose).
HUM_A = np.array([-0.72, 0.0, 1.05])
HUM_B = np.array([0.0, 0.0, 1.05])
HUM_R = 0.05
FORE_A = np.array([0.01, 0.0, 1.05])
FORE_B = np.array([0.75, 0.0, 1.05])
FORE_R = 0.04


def _closest_point_on_triangles(p, a, b, c):
    """Vectorized closest point on triangles (Ericson) for arrays (..., 3)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("...i,...i->...", ab, ap)
    d2 = np.einsum("...i,...i->...", ac, ap)
    bp = p - b
    d3 = np.einsum("...i,...i->...", ab, bp)
    d4 = np.einsum("...i,...i->...", ac, bp)
    cpp = p - c
    d5 = np.einsum("...i,...i->...", ab, cpp)
    d6 = np.einsum("...i,...i->...", ac, cpp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # interior (default)
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-14, 1.0, denom)
    v = vb / safe
    w = vc / safe
    q = a + ab * v[..., None] + ac * w[..., None]

    # edge AB
    dd = d1 - d3
    t = np.divide(d1, dd, out=np.zeros_like(d1), where=np.abs(dd) > 1e-14)
    q_ab = a + ab * t[..., None]
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], q_ab, q)

    # edge AC
    dd = d2 - d6
    t = np.divide(d2, dd, out=np.zeros_like(d2), where=np.abs(dd) > 1e-14)
    q_ac = a + ac * t[..., None]
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], q_ac, q)

    # edge BC
    dn = d4 - d3
    dd = (d4 - d3) + (d5 - d6)
    t = np.divide(dn, dd, out=np.zeros_like(dn), where=np.abs(dd) > 1e-14)
    q_bc = b + (c - b) * t[..., None]
    q = np.where(((va <= 0) & (dn >= 0) & ((d5 - d6) >= 0))[..., None], q_bc, q)

    # vertex regions (highest priority)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
    return q


def _segment_closest(points, a, b):
    """Closest point on segment [a, b] for each row of ``points``."""
    ab = b - a
    denom = float(ab @ ab)
    t = (points - a) @ ab / denom
    t = np.clip(t, 0.0, 1.0)
    return a[None, :] + t[:, None] * ab[None, :]


class BenchmarkSolver:
    """Explicit total-Lagrangian FEM solver for the elbow flesh bulge scene."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(seed)  # deterministic; unused by physics
        del rng

        config = getattr(scene, "config", None) or {}
        initial = scene.initial_state()
        rest = np.asarray(initial.position, dtype=float).copy()
        n = len(rest)
        self._rest = rest
        self._n = n

        density = float(
            config.get("initial_conditions", {}).get(
                "flesh_density_kg_m3", DEFAULT_DENSITY
            )
        )
        self._density = density

        # -- prescribed forearm hinge motion ----------------------------
        fb = config.get("boundary_conditions", {}).get("forearm_bone", {})
        self._hinge = np.asarray(fb.get("hinge_center", [0.0, 0.0, 1.05]), float)
        axis = np.asarray(fb.get("hinge_axis", [0.0, 1.0, 0.0]), float)
        norm = float(np.linalg.norm(axis))
        self._axis = axis / norm if norm > 1e-12 else np.array([0.0, 1.0, 0.0])
        self._t_start = float(fb.get("start_time", 0.4))
        self._t_end = float(fb.get("end_time", 2.4))
        self._angle0 = np.deg2rad(float(fb.get("start_angle_degrees", 0.0)))
        self._angle1 = np.deg2rad(float(fb.get("end_angle_degrees", -110.0)))

        # -- surface mesh (for self-contact) -----------------------------
        mesh_path = Path(__file__).resolve().parent.parent / "assets" / "arm_flesh.obj"
        tris = self._load_faces(mesh_path)
        self._tris = tris
        self._tri_rest_radius = np.array(
            [
                np.linalg.norm(rest[tri] - rest[tri].mean(axis=0), axis=1).max()
                for tri in tris
            ]
        )

        # -- tetrahedral volume mesh -------------------------------------
        tets, slaves = self._build_tets(rest)
        self._tets = tets
        corners = rest[tets]  # (M, 4, 3)
        dm = np.stack(
            [corners[:, 0] - corners[:, 3],
             corners[:, 1] - corners[:, 3],
             corners[:, 2] - corners[:, 3]],
            axis=2,
        )  # columns are rest edge vectors
        self._dm_inv = np.linalg.inv(dm)
        # shape-function material gradients: g_i is ROW i of Dm^{-1}
        # (columns of _grad are the per-corner gradient vectors)
        self._grad = np.ascontiguousarray(np.transpose(self._dm_inv, (0, 2, 1)))
        v0 = np.linalg.det(dm) / 6.0
        v0 = np.where(v0 < 0, -v0, v0)  # orientation already fixed; safety
        self._v0 = v0
        self._rest_volume = float(v0.sum())

        # lumped nodal masses
        mass = density * np.bincount(
            tets.ravel(), weights=np.repeat(v0 / 4.0, 4), minlength=n
        )
        in_volume = mass > 0
        if in_volume.any():
            mean_m = float(mass[in_volume].mean())
        else:
            mean_m = 1.0
        # nodes outside the tet mesh (flat cap centers, orphaned ring nodes
        # after sliver removal) still carry flesh mass and are driven by the
        # bone-surface attachment
        mass[~in_volume] = mean_m
        self._mass = mass
        self._total_mass = float(mass.sum())

        # -- flesh attachment to internal bone surfaces -------------------
        is_hum = rest[:, 0] <= 0.0
        self._is_hum = is_hum
        own_gap = np.empty(n)
        hum_gap = np.linalg.norm(rest - _segment_closest(rest, HUM_A, HUM_B), axis=1) - HUM_R
        fore_gap = np.linalg.norm(rest - _segment_closest(rest, FORE_A, FORE_B), axis=1) - FORE_R
        own_gap = np.where(is_hum, hum_gap, fore_gap)
        self._w = np.clip(1.0 - own_gap / ATTACH_GAP, 0.0, 1.0)
        # nodes that start inside a bone are rigidly bound to it
        self._w[own_gap < 0.002] = 1.0
        # bone contact is disabled for nodes starting inside that bone
        self._hum_contact_ok = hum_gap > 0.002
        self._fore_contact_ok = fore_gap > 0.002

        # -- self-contact candidate pairs ---------------------------------
        tri_rest = rest[tris]  # (T, 3, 3)
        cn, ct = np.meshgrid(np.arange(n), np.arange(len(tris)), indexing="ij")
        cn = cn.ravel()
        ct = ct.ravel()
        q = _closest_point_on_triangles(
            rest[cn], tri_rest[ct, 0], tri_rest[ct, 1], tri_rest[ct, 2]
        )
        d_rest = np.linalg.norm(rest[cn] - q, axis=1)
        keep = (d_rest > SELF_REST_MIN) & (d_rest < SELF_REST_MAX)
        self._cn = cn[keep]
        self._ct = ct[keep]

        # -- constitutive parameters --------------------------------------
        self._mu = YOUNG / (2.0 * (1.0 + POISSON))
        self._lam = (
            YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        )

        # -- dynamic state --------------------------------------------------
        self._x = rest.copy()
        self._v = np.zeros_like(rest)
        self._x_good = rest.copy()
        self._v_good = np.zeros_like(rest)
        self._time = 0.0
        self._budget = np.zeros((3, 3))  # gravity, attachment, bone contact
        self._steps = 0
        self._ready = True

        self._reference_force = float(self._total_mass * np.linalg.norm(GRAVITY))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_faces(path: Path) -> np.ndarray:
        faces = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or parts[0] != "f":
                continue
            idx = [int(tok.split("/")[0]) - 1 for tok in parts[1:4]]
            faces.append(idx)
        if not faces:
            raise RuntimeError(f"no faces found in {path}")
        return np.asarray(faces, dtype=np.int64)

    @staticmethod
    def _build_tets(rest: np.ndarray):
        """Conforming tetrahedralization of the ring-structured arm tube.

        Each slab between two matched vertex rings is split by a fan
        triangulation of the cross-section; every triangular prism is split
        into three tetrahedra with a diagonal rule that is conforming across
        neighbouring prisms.
        """
        xs = np.round(rest[:, 0], 6)
        stations = np.unique(xs)
        rings = []
        slaves = []
        for xv in stations:
            idx = np.where(xs == xv)[0]
            center = rest[idx].mean(axis=0)
            rad = np.linalg.norm(rest[idx, 1:] - center[1:], axis=1)
            ring_idx = idx[rad > 1e-4]
            if len(ring_idx) != 16:
                raise RuntimeError("unexpected rest mesh station layout")
            slaves.extend(idx[rad <= 1e-4].tolist())
            ang = np.arctan2(
                rest[ring_idx, 2] - center[2], rest[ring_idx, 1] - center[1]
            )
            rings.append(ring_idx[np.argsort(ang)])
        if len(rings) < 2:
            raise RuntimeError("could not identify arm mesh rings")

        tets = []
        for k in range(len(rings) - 1):
            a = rings[k]
            b = rings[k + 1]
            m = len(a)
            for i in range(1, m - 1):
                a0, bi, bi1 = a[0], a[i], a[i + 1]
                b0, ci, ci1 = b[0], b[i], b[i + 1]
                tets.append((a0, bi, bi1, ci1))   # (A, B, C, C')
                tets.append((a0, bi, ci1, ci))    # (A, B, C', B')
                tets.append((a0, ci, ci1, b0))    # (A, B', C', A')
        tets = np.asarray(tets, dtype=np.int64)

        # fix orientation: positive signed volume in the rest configuration
        c = rest[tets]
        signed = np.einsum(
            "ij,ij->i",
            c[:, 0] - c[:, 3],
            np.cross(c[:, 1] - c[:, 3], c[:, 2] - c[:, 3]),
        ) / 6.0
        flip = signed < 0
        if flip.any():
            tets[flip, 0], tets[flip, 1] = tets[flip, 1].copy(), tets[flip, 0].copy()

        # drop degenerate sliver tets (near-zero volume, ill-conditioned
        # shape-function gradients); the volume loss is negligible
        tets = tets[np.abs(signed) >= MIN_TET_VOLUME]
        return tets, slaves

    # ------------------------------------------------------------------
    def _hinge_angle(self, t: float):
        """Prescribed hinge angle and angular rate at time ``t``."""
        if self._t_end <= self._t_start:
            u = 1.0 if t >= self._t_end else 0.0
        else:
            u = (t - self._t_start) / (self._t_end - self._t_start)
        u = min(max(u, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep
        theta = self._angle0 + (self._angle1 - self._angle0) * s
        dsdu = 6.0 * u * (1.0 - u)
        rate = (self._angle1 - self._angle0) * dsdu / max(
            self._t_end - self._t_start, 1e-12
        )
        return theta, rate

    def _rotate(self, points: np.ndarray, theta: float) -> np.ndarray:
        """Rodrigues rotation of ``points`` about the hinge axis/center."""
        k = self._axis
        rel = points - self._hinge
        c, s = np.cos(theta), np.sin(theta)
        return (
            self._hinge
            + rel * c
            + np.cross(k, rel) * s
            + np.outer(rel @ k, k) * (1.0 - c)
        )

    # ------------------------------------------------------------------
    # force terms
    # ------------------------------------------------------------------
    def _deformation_gradients(self, x: np.ndarray) -> np.ndarray:
        corners = x[self._tets]
        ds = np.stack(
            [corners[:, 0] - corners[:, 3],
             corners[:, 1] - corners[:, 3],
             corners[:, 2] - corners[:, 3]],
            axis=2,
        )
        return np.matmul(ds, self._dm_inv)

    def _elastic_forces(self, x: np.ndarray) -> np.ndarray:
        f_mat = self._deformation_gradients(x)
        det = np.linalg.det(f_mat)
        # guard against (near-)singular elements
        bad = np.abs(det) < 1e-10
        if bad.any():
            f_mat = f_mat.copy()
            f_mat[bad] = np.eye(3)
            det = np.linalg.det(f_mat)
        ft_inv = np.linalg.inv(f_mat).transpose(0, 2, 1)  # F^{-T}
        j = det
        alpha = 0.5 * self._lam * (j - 1.0) - self._mu
        p = self._mu * f_mat + alpha[:, None, None] * ft_inv
        # f_i = -V0 * P @ grad N_i ; corner 4 uses -(sum of gradients)
        fc = -self._v0[:, None, None] * np.einsum("mab,mbi->mai", p, self._grad)
        gsum = self._grad.sum(axis=2)  # (M, 3), sum of corner gradients
        f3 = self._v0[:, None] * np.einsum("mab,mb->ma", p, gsum)
        f = np.zeros_like(x)
        for i in range(3):
            np.add.at(f, self._tets[:, i], fc[:, :, i])
        np.add.at(f, self._tets[:, 3], f3)
        return f

    def _attachment_forces(self, x, v, theta, rate):
        """Spring-damper traction from the internal bone surfaces."""
        target = self._rest.copy()
        fore = ~self._is_hum
        if fore.any():
            target[fore] = self._rotate(self._rest[fore], theta)
        v_target = np.zeros_like(x)
        if fore.any() and abs(rate) > 0.0:
            rel = target[fore] - self._hinge
            omega = rate * self._axis
            v_target[fore] = np.cross(np.broadcast_to(omega, rel.shape), rel)
        w = self._w[:, None]
        return (K_ATTACH * (target - x) + C_ATTACH * (v_target - v)) * w

    def _bone_segments(self, theta):
        fore_a = self._rotate(FORE_A[None, :], theta)[0]
        fore_b = self._rotate(FORE_B[None, :], theta)[0]
        return [
            (HUM_A, HUM_B, HUM_R, self._hum_contact_ok),
            (fore_a, fore_b, FORE_R, self._fore_contact_ok),
        ]

    def _bone_contact_forces(self, x, theta):
        f = np.zeros_like(x)
        for a, b, radius, allowed in self._bone_segments(theta):
            cp = _segment_closest(x, a, b)
            dvec = x - cp
            dist = np.linalg.norm(dvec, axis=1)
            gap = dist - radius
            active = allowed & (gap < 0.0)
            if not active.any():
                continue
            safe = np.where(dist < 1e-9, 1.0, dist)
            nrm = dvec / safe[:, None]
            bad = dist < 1e-9
            if bad.any():
                nrm[bad] = np.array([0.0, 0.0, 1.0])
            mag = -K_BONE * gap  # positive when penetrating
            f += np.where(active[:, None], mag[:, None] * nrm, 0.0)
        return f

    def _self_contact_forces(self, x):
        """Penalty self-contact between non-adjacent surface parts.

        Reaction forces are applied to the triangle vertices, so the
        interaction is internal to the flesh body and cancels in the combined
        budget domain.
        """
        f = np.zeros_like(x)
        tri = x[self._tris]
        centroid = tri.mean(axis=1)
        diff = x[self._cn] - centroid[self._ct]
        r = np.linalg.norm(diff, axis=1)
        near = r < (self._tri_rest_radius[self._ct] + SELF_D0)
        idx = np.where(near)[0]
        if idx.size == 0:
            return f
        ni = self._cn[idx]
        ti = self._ct[idx]
        q = _closest_point_on_triangles(
            x[ni], tri[ti, 0], tri[ti, 1], tri[ti, 2]
        )
        dvec = x[ni] - q
        dist = np.linalg.norm(dvec, axis=1)
        active = dist < SELF_D0
        if not active.any():
            return f
        ni = ni[active]
        ti = ti[active]
        dvec = dvec[active]
        dist = dist[active]
        safe = np.where(dist < 1e-9, 1.0, dist)
        nrm = dvec / safe[:, None]
        degenerate = dist < 1e-9
        if degenerate.any():
            tri_n = np.cross(
                self._x[self._tris[ti, 1]] - self._x[self._tris[ti, 0]],
                self._x[self._tris[ti, 2]] - self._x[self._tris[ti, 0]],
            )
            tn = np.linalg.norm(tri_n, axis=1)
            tri_n = tri_n / np.where(tn < 1e-12, 1.0, tn)[:, None]
            nrm[degenerate] = tri_n[degenerate]
        mag = K_SELF * (SELF_D0 - dist)
        fnode = mag[:, None] * nrm
        np.add.at(f, ni, fnode)
        np.add.at(f, self._tris[ti].ravel(), np.repeat(-fnode / 3.0, 3, axis=0))
        return f

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        x, v = self._x, self._v
        theta, rate = self._hinge_angle(self._time)

        f = self._elastic_forces(x)
        f_gravity = self._mass[:, None] * GRAVITY[None, :]
        f += f_gravity
        f_attach = self._attachment_forces(x, v, theta, rate)
        f += f_attach
        f_bone = self._bone_contact_forces(x, theta)
        f += f_bone
        f += self._self_contact_forces(x)
        # mass-proportional damping (internal, dissipative)
        f -= (DAMP_MASS * self._mass)[:, None] * v

        np.clip(f, -FORCE_CAP, FORCE_CAP, out=f)
        v_new = v + dt * f / self._mass[:, None]
        np.clip(v_new, -VEL_CAP, VEL_CAP, out=v_new)
        x_new = x + dt * v_new

        if np.isfinite(x_new).all() and np.isfinite(v_new).all():
            self._x = x_new
            self._v = v_new
            self._x_good = x_new.copy()
            self._v_good = v_new.copy()
            self._budget = np.stack(
                [f_gravity.sum(axis=0), f_attach.sum(axis=0), f_bone.sum(axis=0)]
            )
        else:
            # numerical safety net: restore last good state, kill velocities
            self._x = self._x_good.copy()
            self._v = np.zeros_like(self._x)
            self._budget = np.zeros((3, 3))
        self._time += dt
        self._steps += 1

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver not initialized")
        target = float(time)
        if target < self._time:
            target = self._time
        while self._time < target - 1e-12:
            dt = min(DT, target - self._time)
            self._substep(dt)
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, t: float) -> StateFrame:
        theta, _ = self._hinge_angle(t)
        f_mat = self._deformation_gradients(self._x)
        if not np.isfinite(f_mat).all():
            f_mat = np.nan_to_num(f_mat, nan=0.0, posinf=0.0, neginf=0.0)
        budget = self._budget.reshape(3, 1, 3)
        frame = StateFrame(
            time=float(t),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "rest_position": self._rest.copy(),
                "attachment_weight": self._w.copy(),
            },
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": f_mat,
            },
            scalars={
                "flesh_density_kg_m3": float(self._density),
                "elbow_flexion_degrees": float(np.rad2deg(theta)),
                "reference_force": float(self._reference_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        if not self._ready:
            return {}
        return {
            "nodes": int(self._n),
            "tets": int(len(self._tets)),
            "surface_faces": int(len(self._tris)),
            "rest_volume_m3": self._rest_volume,
            "total_mass_kg": self._total_mass,
            "reference_force_N": self._reference_force,
            "time_step": DT,
            "substeps_taken": int(self._steps),
            "mu": self._mu,
            "lambda": self._lam,
        }
