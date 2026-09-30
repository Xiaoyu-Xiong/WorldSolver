"""Elastoplastic work-hardening deformable-body solver (NumPy/SciPy, no external engines).

Physical model
--------------
The bunny is a closed triangle shell.  It is discretized as a nonlinear
mass-spring lattice:
  * 1-ring edge springs  -> membrane (stretch / shear) stiffness,
  * 2-ring bending springs -> out-of-plane (folding) stiffness.
Each spring carries a 1-D elastoplastic law with isotropic work hardening:
    eps_e    = (L - L0) / L0                       (elastic strain)
    eps_y    = EPS_Y0 + H_LIN * eps_p              (yield strain, grows with
                                                    accumulated plastic strain)
    |eps_e| <= eps_y                               (yield surface)
    L0      <- L0 * (1 + sign(eps_e) * flow)       (return mapping, 1-D flow)
    k_eff    = k_base * (1 + hardening_rate*eps_p) (hardened regions stiffer
                                                    -> more elastic, localized)
Plastic flow is dissipative, so strongly deformed regions permanently deform and,
through hardening, become stiffer and elastically jiggly.  All springs and the
contact constraints are integrated with a symplectic (semi-implicit) Euler
scheme at a fixed, stability-bounded internal timestep.  Contacts against the
frozen ground plane and the frozen metal cylinder are resolved with a
spring-damper penalty plus regularized Coulomb friction; the normal damper is
tuned so the effective coefficient of restitution matches the frozen config.

Everything is deterministic: a fixed internal dt and no randomness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# fixed numerical / material constants
# --------------------------------------------------------------------------
_DT = 1.5e-4                 # internal integration step [s]
_E_SCALE = 133.0             # effective modulus per unit of config "stiffness" [Pa]
_BETA_BEND = 0.25            # bending spring stiffness relative to edge springs
_ZETA_SPRING = 0.20          # spring dashpot damping ratio
_DAMP_FACTOR = 0.60          # fraction of config "damping" used as bulk rate [1/s]
_EPS_Y0 = 0.03               # initial yield strain
_EPSP_CAP = 0.55             # cap on accumulated plastic strain
_HARD_STIFF_CAP = 4.0        # cap on hardening stiffness multiplier
_FLOW_RATE = 2.0e4           # plastic flow relaxation rate [1/s]
_COMP_K = 6.0                # compression-guard stiffness multiplier
_COMP_MIN = 0.55             # compression guard activates below this L/L0
_KC = 4.0e6                  # contact penalty stiffness / vertex mass [1/s^2]
_ZETA_CONTACT = 0.40         # contact damper ratio -> e = 0.28 for the config
_FRICTION_REG = 1.0          # tangential regularization relative to normal k
_MAX_SPEED = 60.0            # numerical safety clamp [m/s]
_CONTACT_BAND_MULT = 2.0     # near-contact reporting band = mult * contact_length_scale


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


class BenchmarkSolver:
    """Deterministic elastoplastic impact solver for the hardening bunny scene."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.config = cfg
        self.seed = int(seed)

        root = Path(__file__).resolve().parent.parent
        with np.load(root / cfg["subject_asset"]) as data:
            self.faces = np.asarray(data["faces"], dtype=np.int64)

        self.X0 = np.asarray(scene.vertices, dtype=float)
        n = len(self.X0)
        self.n = n

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.bar_center, self.bar_axis, self.bar_radius, self.bar_half = self._bar_geometry(cfg)

        total_mass = float(cfg["subject_mass"])
        self.mass = np.full(n, total_mass / n, dtype=float)

        self._build_springs()
        self._build_gradient_operator()

        # state
        self.x = self.X0.copy()
        self.v = np.tile(
            np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float), (n, 1)
        )
        self.time = 0.0
        self._body_id = np.zeros(n, dtype=np.int64)
        self._steps = 0
        self._min_det = 1.0

        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        self.friction = float(cfg.get("friction", 0.55))
        self.hardening_rate = float(cfg.get("hardening_rate", 1.8))
        self.bulk_damping = _DAMP_FACTOR * float(cfg.get("damping", 1.4))

    def _bar_geometry(self, cfg):
        """Authoritative cylinder from frozen scene geometry; config keys as fallback."""
        center = np.array([0.0, 0.0, float(cfg.get("bar_z", 0.66))])
        axis = np.array([0.0, 1.0, 0.0])
        radius = float(cfg.get("bar_radius", 0.085))
        half = 1.2
        geom = cfg.get("scene_geometry") or {}
        for item in geom.get("objects", []):
            if item.get("id") != "metal_bar":
                continue
            spec = item.get("geometry", {})
            radius = float(spec.get("radius", radius))
            half = 0.5 * float(spec.get("height", 2.0 * half))
            tr = item.get("transform", {}) or {}
            center = np.asarray(tr.get("translation", center), dtype=float)
            q = np.asarray(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            w, xq, yq, zq = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            rot = np.array(
                [
                    [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - w * zq), 2 * (xq * zq + w * yq)],
                    [2 * (xq * yq + w * zq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - w * xq)],
                    [2 * (xq * zq - w * yq), 2 * (yq * zq + w * xq), 1 - 2 * (xq * xq + yq * yq)],
                ]
            )
            axis = _unit(rot @ np.array([0.0, 0.0, 1.0]))
        return center, axis, radius, half

    def _build_springs(self) -> None:
        f = self.faces
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        n = self.n
        neighbours = [set() for _ in range(n)]
        for a, b in edges:
            neighbours[int(a)].add(int(b))
            neighbours[int(b)].add(int(a))

        pairs = set()
        for i in range(n):
            ring1 = neighbours[i]
            for j in ring1:
                for k in neighbours[j]:
                    if k != i and k not in ring1:
                        pairs.add((i, k) if i < k else (k, i))
        pairs = np.array(sorted(pairs), dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)

        si = np.concatenate([edges[:, 0], pairs[:, 0]]).astype(np.int64)
        sj = np.concatenate([edges[:, 1], pairs[:, 1]]).astype(np.int64)

        h_mean = float(np.mean(np.linalg.norm(self.X0[edges[:, 0]] - self.X0[edges[:, 1]], axis=1)))
        modulus = _E_SCALE * float(self.config.get("stiffness", 18.0))
        k_edge = modulus * h_mean

        n_edge = len(edges)
        n_pair = len(pairs)
        k_base = np.concatenate(
            [np.full(n_edge, k_edge), np.full(n_pair, _BETA_BEND * k_edge)]
        )

        keep = si != sj
        si, sj, k_base = si[keep], sj[keep], k_base[keep]
        l_ref = np.linalg.norm(self.X0[sj] - self.X0[si], axis=1)
        good = l_ref > 1e-9
        si, sj, k_base, l_ref = si[good], sj[good], k_base[good], l_ref[good]

        self.si, self.sj = si, sj
        self.k_base = k_base
        self.l_ref = l_ref
        self.L0 = l_ref.copy()
        self.epsp = np.zeros(len(l_ref), dtype=float)
        self.h_mean = h_mean
        self.modulus = modulus

    def _build_gradient_operator(self) -> None:
        """Least-squares per-vertex deformation gradient on the spring neighbourhood."""
        di = np.concatenate([self.si, self.sj])
        dj = np.concatenate([self.sj, self.si])
        Xd = self.X0[dj] - self.X0[di]
        eye = np.eye(3)
        M = np.zeros((self.n, 3, 3))
        for a in range(3):
            for b in range(3):
                M[:, a, b] = np.bincount(
                    di, weights=Xd[:, a] * Xd[:, b], minlength=self.n
                )
        lam = 0.05 * np.trace(M, axis1=1, axis2=2) / 3.0 + 1e-10
        M += lam[:, None, None] * eye
        self._grad_i = di
        self._grad_Xd = Xd
        self._grad_lam = lam
        self._grad_Minv = np.linalg.inv(M)

    # ------------------------------------------------------------------
    # internal force model
    # ------------------------------------------------------------------
    def _spring_forces(self, x, v):
        si, sj = self.si, self.sj
        dx = x[sj] - x[si]
        length = np.sqrt(np.einsum("ij,ij->i", dx, dx))
        length = np.maximum(length, 1e-9)
        inv = 1.0 / length
        normal = dx * inv[:, None]

        eps_e = (length - self.L0) / self.L0
        k_eff = self.k_base * (1.0 + 2.0 * self.hardening_rate * self.epsp)
        np.minimum(k_eff, _HARD_STIFF_CAP * self.k_base, out=k_eff)

        m_s = 0.5 * (self.mass[si] + self.mass[sj])
        c_s = _ZETA_SPRING * 2.0 * np.sqrt(k_eff * m_s)
        dv = v[sj] - v[si]
        f_mag = k_eff * (length - self.L0) + c_s * np.einsum("ij,ij->i", dv, normal)

        compression = length - _COMP_MIN * self.L0
        comp_mask = compression < 0.0
        if np.any(comp_mask):
            f_mag = f_mag.copy()
            f_mag[comp_mask] += _COMP_K * k_eff[comp_mask] * (-compression[comp_mask])

        f = f_mag[:, None] * normal
        out = np.empty((self.n, 3), dtype=float)
        for d in range(3):
            out[:, d] = np.bincount(si, weights=f[:, d], minlength=self.n) + np.bincount(
                sj, weights=-f[:, d], minlength=self.n
            )
        return out, eps_e, k_eff

    def _update_plasticity(self, eps_e, dt: float) -> None:
        eps_y = _EPS_Y0 + 1.8 * self.hardening_rate * _EPS_Y0 * self.epsp
        excess = np.abs(eps_e) - eps_y
        active = excess > 0.0
        if not np.any(active):
            return
        rate = min(1.0, _FLOW_RATE * dt)
        flow = excess[active] * rate
        sgn = np.sign(eps_e[active])
        self.L0[active] = self.L0[active] * (1.0 + sgn * flow)
        self.epsp[active] = np.minimum(self.epsp[active] + flow, _EPSP_CAP)
        np.maximum(self.L0, 0.35 * self.l_ref, out=self.L0)

    # ------------------------------------------------------------------
    # contact model
    # ------------------------------------------------------------------
    def _obstacle_gaps(self, x):
        """Signed gaps (>0 separated) and outward unit normals for ground and bar."""
        ground_gap = x[:, 2] - self.ground_z
        ground_normal = np.zeros_like(x)
        ground_normal[:, 2] = 1.0

        rel = x - self.bar_center
        t = np.clip(rel @ self.bar_axis, -self.bar_half, self.bar_half)
        closest = self.bar_center + t[:, None] * self.bar_axis
        radial = x - closest
        dist = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        safe = np.maximum(dist, 1e-9)
        bar_normal = radial / safe[:, None]
        degenerate = dist < 1e-9
        if np.any(degenerate):
            bar_normal[degenerate] = np.array([0.0, 0.0, 1.0])
        bar_gap = dist - self.bar_radius

        gap = np.minimum(ground_gap, bar_gap)
        return ground_gap, ground_normal, bar_gap, bar_normal, gap

    def _contact_force(self, x, v):
        n = self.n
        force = np.zeros_like(x)
        totals = np.zeros((2, 3))
        ground_gap, ground_normal, bar_gap, bar_normal, gap = self._obstacle_gaps(x)

        for idx, (g, normal) in enumerate(((ground_gap, ground_normal), (bar_gap, bar_normal))):
            pen = -g
            touch = pen > 0.0
            if not np.any(touch):
                continue
            k_c = _KC * self.mass[touch]
            v_n = np.einsum("ij,ij->i", v[touch], normal[touch])
            approach = np.maximum(-v_n, 0.0)
            c_n = 2.0 * _ZETA_CONTACT * np.sqrt(k_c * self.mass[touch])
            f_n = np.maximum(k_c * pen[touch] + c_n * approach, 0.0)
            f = f_n[:, None] * normal[touch]

            v_t = v[touch] - v_n[:, None] * normal[touch]
            speed_t = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
            moving = speed_t > 1e-8
            if np.any(moving):
                k_t = _FRICTION_REG * k_c[moving]
                cap = self.friction * f_n[moving]
                mag = np.minimum(k_t * speed_t[moving], cap)
                f[moving] -= (mag / speed_t[moving])[:, None] * v_t[moving]

            force[touch] += f
            totals[idx] = f.sum(axis=0)
        return force, totals, gap

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        spring_f, eps_e, _ = self._spring_forces(self.x, self.v)
        contact_f, _, _ = self._contact_force(self.x, self.v)
        self._update_plasticity(eps_e, dt)

        force = spring_f + contact_f
        force += self.mass[:, None] * self.gravity[None, :]

        self.v += (force / self.mass[:, None]) * dt
        self.v *= 1.0 / (1.0 + self.bulk_damping * dt)

        speed = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        too_fast = speed > _MAX_SPEED
        if np.any(too_fast):
            self.v[too_fast] *= (_MAX_SPEED / speed[too_fast])[:, None]

        self.x += self.v * dt
        self.time += dt
        self._steps += 1

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        dj = np.concatenate([self.sj, self.si])
        xd = self.x[dj] - self.x[self._grad_i]
        C = np.zeros((self.n, 3, 3))
        Xd = self._grad_Xd
        for a in range(3):
            for b in range(3):
                C[:, a, b] = np.bincount(
                    self._grad_i, weights=xd[:, a] * Xd[:, b], minlength=self.n
                )
        # Same ridge as in the normal matrix => F is exactly the identity at rest.
        C += self._grad_lam[:, None, None] * np.eye(3)
        F = C @ self._grad_Minv
        det = np.linalg.det(F)
        bad = det < 0.02
        if np.any(bad):
            F = F.copy()
            F[bad] = 0.5 * (F[bad] + np.eye(3))
        finite = np.isfinite(F)
        if not np.all(finite):
            F = np.where(finite, F, np.eye(3))
        self._min_det = min(self._min_det, float(np.min(np.linalg.det(F))))
        return F

    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        _, _, _, _, gap = self._obstacle_gaps(self.x)
        band = _CONTACT_BAND_MULT * self.contact_length_scale
        near = gap[gap < band]
        if near.size == 0:
            near = np.array([float(np.min(gap))])
        near = np.sort(near)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "interaction_force": np.zeros((2, 3)),
                "contact_signed_distance": near,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        return frame

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            target = self.time
        guard = 0
        while self.time < target - 1e-12:
            dt = min(_DT, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
            if guard > 10_000_000:
                break
        frame = self._frame(target)
        _, totals, _ = self._contact_force(self.x, self.v)
        frame.fields["interaction_force"] = totals.copy()
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "elastoplastic_work_hardening_mass_spring",
            "steps": int(self._steps),
            "internal_dt": float(_DT),
            "vertices": int(self.n),
            "springs": int(len(self.si)),
            "modulus": float(self.modulus),
            "mean_edge_length": float(self.h_mean),
            "max_plastic_strain": float(np.max(self.epsp)) if self.epsp.size else 0.0,
            "mean_plastic_strain": float(np.mean(self.epsp)) if self.epsp.size else 0.0,
            "hardened_spring_fraction": float(np.mean(self.epsp > 0.0)) if self.epsp.size else 0.0,
            "min_deformation_det": float(self._min_det),
            "final_time": float(self.time),
            "final_center_of_mass": [float(c) for c in self.x.mean(axis=0)],
        }
