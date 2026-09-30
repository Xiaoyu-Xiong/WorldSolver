"""Physics solver for scene 074: stick-slip of a volumetric elastic Armadillo
between driven high-friction rollers.

Model
-----
* The volumetric deformable body is discretised with linear tetrahedra
  (built from the frozen closed surface mesh with a Delaunay tetrahedralisation,
  so the node set is exactly the frozen mesh node set).
* Material: stable compressible neo-Hookean (Smith/Goes/Kim form) with
  E = 5.0e5 Pa, nu = 0.4 -> Lame parameters mu, lam.
* Four kinematic rigid cylinders ("rollers") with the frozen geometry.  Their
  angular speed follows the frozen start time / ramp, and their sign is chosen so
  that the contact surface velocity is +x, i.e. consistent with the frozen
  ``intended_transport_direction: positive_x`` (the armadillo rests at x=-0.72,
  to the -x side of both roller pairs, and must be drawn into the gap).
* Contact: penalty normal law + regularised Coulomb friction (stick below a
  speed tolerance, capped tangential force otherwise).  Contacts are with the
  finite cylinder lateral surface (radial + end-cap excess distance).
* Time integration: explicit symplectic Euler with a small fixed step, plus
  light viscous damping.  Deterministic reference values, no randomness beyond
  the frozen seed (which is not needed for deterministic dynamics).

All frozen quantities (geometry, masses, roller placement, coefficients) are
read from the scene contract; nothing is animation-scripted.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# Frozen / calibrated constants
# ----------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
YOUNGS_MODULUS = 5.0e5          # frozen moderate stiffness reference (Pa)
POISSON = 0.4
MU_FRICTION = 0.5               # frozen roller/deformable pairing

ROLLER_RADIUS = 0.22
ROLLER_HALF_HEIGHT = 0.36
ROLLER_START_TIME = 0.35
ROLLER_RAMP_DURATION = 0.25
ROLLER_OMEGA_MAG = 5.0

DT = 4.0e-5
K_NORMAL = 5.0e5                # penalty normal stiffness [N/m]
C_NORMAL = 8.0e2                # contact normal damping [N*s/m]
K_TANGENT = 5.0e5               # tangential regularisation stiffness [N*s/m]
V_STICK = 0.02                  # stick/slip speed tolerance [m/s]
VEL_DAMPING = 3.0               # global viscous damping [1/s]
MAX_CONTACT_FORCE = 2.0e5       # safety valve, rarely reached [N]

REFERENCE_VELOCITY = ROLLER_RADIUS * ROLLER_OMEGA_MAG   # 1.1 m/s

# Roller centres from the frozen scene geometry (id order preserved).
ROLLER_DEFS = [
    ("upper_left_roller", (-0.2, 0.0, 0.3), -1.0),
    ("upper_right_roller", (0.3, 0.0, 0.3), -1.0),
    ("lower_left_roller", (-0.2, 0.0, -0.3), 1.0),
    ("lower_right_roller", (0.3, 0.0, -0.3), 1.0),
]


def _ramp(t: float) -> float:
    if t <= ROLLER_START_TIME:
        return 0.0
    if t >= ROLLER_START_TIME + ROLLER_RAMP_DURATION:
        return 1.0
    s = (t - ROLLER_START_TIME) / ROLLER_RAMP_DURATION
    return s * s * (3.0 - 2.0 * s)


def _cofactor(F: np.ndarray) -> np.ndarray:
    """Adjugate-free cofactor matrix (det(F) * F^{-T}) computed from 2x2 minors."""
    c = np.empty_like(F)
    c[:, 0, 0] = F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1]
    c[:, 0, 1] = F[:, 1, 2] * F[:, 2, 0] - F[:, 1, 0] * F[:, 2, 2]
    c[:, 0, 2] = F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]
    c[:, 1, 0] = F[:, 0, 2] * F[:, 2, 1] - F[:, 0, 1] * F[:, 2, 2]
    c[:, 1, 1] = F[:, 0, 0] * F[:, 2, 2] - F[:, 0, 2] * F[:, 2, 0]
    c[:, 1, 2] = F[:, 0, 1] * F[:, 2, 0] - F[:, 0, 0] * F[:, 2, 1]
    c[:, 2, 0] = F[:, 0, 1] * F[:, 1, 2] - F[:, 0, 2] * F[:, 1, 1]
    c[:, 2, 1] = F[:, 0, 2] * F[:, 1, 0] - F[:, 0, 0] * F[:, 1, 2]
    c[:, 2, 2] = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    return c


class BenchmarkSolver:
    """Deterministic volumetric FEM solver with roller contact."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        init = scene.initial_state()
        self.n = int(len(init.position))
        self.x0 = np.array(init.position, dtype=float)
        self.mass = np.array(init.mass, dtype=float)
        if self.mass.shape != (self.n,):
            self.mass = np.full(self.n, float(np.sum(self.mass)) / self.n)
        self.body_id = np.ones(self.n, dtype=np.int64)
        self.total_mass = float(np.sum(self.mass))

        # --- volumetric tetrahedral mesh (same node set as the frozen mesh) ---
        tri = Delaunay(self.x0)
        tets = np.array(tri.simplices, dtype=np.int64)
        p = self.x0[tets]
        Dm = np.stack(
            [p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2
        )
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            p = self.x0[tets]
            Dm = np.stack(
                [p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2
            )
            det = np.linalg.det(Dm)
        # drop (near-)degenerate sliver tets; they carry ~zero volume
        keep = det > 6.0e-10
        self.tets = tets[keep]
        Dm = Dm[keep]
        self.ne = int(len(self.tets))

        self.Dm_inv = np.linalg.inv(Dm)                  # (Ne,3,3)
        self.Dm_inv_T = np.transpose(self.Dm_inv, (0, 2, 1))
        self.rest_vol = det[keep] / 6.0                  # tet volume

        # per-element explicit-stability scaling: for a linear tet the highest
        # frequency scales like sqrt(E * A_max / (3 * h * m)).  Elements whose
        # critical step is below DT get their stiffness reduced (they are
        # sliver-like with negligible volume, so this changes nothing physical
        # but keeps the explicit scheme stable).
        m_rep = float(np.mean(self.mass))
        p_rest = self.x0[self.tets]
        amax = np.zeros(self.ne)
        for (i, j, k) in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            cr = np.cross(p_rest[:, j] - p_rest[:, i], p_rest[:, k] - p_rest[:, i])
            amax = np.maximum(amax, 0.5 * np.linalg.norm(cr, axis=1))
        h_alt = 3.0 * self.rest_vol / np.maximum(amax, 1e-12)
        omega = np.sqrt(YOUNGS_MODULUS * amax / (3.0 * h_alt * m_rep))
        dt_crit = 2.0 / np.maximum(omega, 1e-12)
        self.elem_scale = np.minimum(1.0, (dt_crit / DT) ** 2)

        # node gather / scatter operators for vectorised force assembly
        gather = self.tets.reshape(-1)                   # (4*Ne,)
        self._gather = gather
        scatter = np.zeros((self.n, 4 * self.ne))
        scatter[gather, np.arange(4 * self.ne)] = 1.0
        self._scatter = scatter

        # --- Lame parameters (stable compressible neo-Hookean) ---
        mu = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON))
        lam = YOUNGS_MODULUS * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        self.mu = mu
        self.lam = lam
        self.alpha = 1.0 + mu / lam - mu / (4.0 * lam)

        # --- kinematic rollers ---
        self.roller_center = np.array([d[1] for d in ROLLER_DEFS], dtype=float)
        self.roller_sign = np.array([d[2] for d in ROLLER_DEFS], dtype=float)

        # --- state ---
        self.t = 0.0
        self.x = self.x0.copy()
        self.v = np.zeros((self.n, 3))
        self.dt = DT
        self._dt_hint = DT

        # --- budget accumulators (G domains, K external terms) ---
        self.n_domains = 2
        self.domain_weights = np.ones((self.n_domains, self.n))
        self.n_terms = 3                       # gravity, contact, damping
        self.impulse = np.zeros((self.n_domains, 3))
        self.inflow = np.zeros((self.n_domains, 3))
        self.outflow = np.zeros((self.n_domains, 3))
        self.mass_source = np.zeros(1)
        self._last_contacts = {
            "normal": np.zeros((0, 3)),
            "normal_force": np.zeros((0,)),
            "rel_vel": np.zeros((0, 3)),
            "tangential_force": np.zeros((0, 3)),
            "active": np.zeros((0,), dtype=bool),
        }

    # ------------------------------------------------------------- kinematics
    def _roller_omega(self, t: float) -> np.ndarray:
        return self.roller_sign * ROLLER_OMEGA_MAG * _ramp(t)

    def _roller_angle(self, t: float) -> float:
        """Integrated roller angle (for diagnostics / rendering)."""
        if t <= ROLLER_START_TIME:
            return 0.0
        end = min(t, ROLLER_START_TIME + ROLLER_RAMP_DURATION)
        # integral of smoothstep ramp
        s = (end - ROLLER_START_TIME) / ROLLER_RAMP_DURATION
        ramp_int = ROLLER_RAMP_DURATION * (s ** 3 - 0.5 * s ** 4)
        total = ramp_int + max(0.0, t - ROLLER_START_TIME - ROLLER_RAMP_DURATION)
        return ROLLER_OMEGA_MAG * total

    # ---------------------------------------------------------------- physics
    def _elastic_forces(self) -> np.ndarray:
        xe = self.x[self.tets]                              # (Ne,4,3)
        Ds = np.stack(
            [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], axis=2
        )
        F = Ds @ self.Dm_inv
        J = np.linalg.det(F)
        ic = np.einsum("eij,eij->e", F, F)
        cof = _cofactor(F)
        cfac = self.lam * (J - self.alpha) * J
        mfac = self.mu * (1.0 - 1.0 / (ic + 1.0))
        P = np.einsum("e,ij->eij", mfac, F) + np.einsum("e,eij->eij", cfac, cof)
        H = P @ self.Dm_inv_T                               # (Ne,3,3)
        f_tet = np.empty((self.ne, 4, 3))
        f_tet[:, 0, :] = self.rest_vol[:, None] * np.einsum("eij->ei", H)
        for k in range(3):
            f_tet[:, k + 1, :] = -self.rest_vol[:, None] * H[:, :, k]
        f_tet *= self.elem_scale[:, None, None]
        return f_tet.reshape(-1, 3)

    def _contact(self, t: float):
        """Penalty normal + Coulomb friction over the frozen candidate set.

        The candidate set is every (deformable node, roller) pair, so the output
        shapes are constant across frames; ``friction_active_mask`` selects the
        pairs currently in persistent dry contact.
        """
        omega = self._roller_omega(t)
        n_pairs = len(ROLLER_DEFS)
        C = self.n * n_pairs
        f_ct = np.zeros((self.n, 3))
        normals = np.zeros((C, 3))
        normals_f = np.zeros(C)
        rel_vels = np.zeros((C, 3))
        tang_f = np.zeros((C, 3))
        active = np.zeros(C, dtype=bool)
        yhat = np.array([0.0, 1.0, 0.0])
        for k in range(n_pairs):
            c = self.roller_center[k]
            d = self.x - c
            ay = d[:, 1]
            a_cl = np.clip(ay, -ROLLER_HALF_HEIGHT, ROLLER_HALF_HEIGHT)
            rad = d[:, [0, 2]]
            r = np.linalg.norm(rad, axis=1)
            good = r > 1e-12
            rad_u = np.zeros_like(rad)
            rad_u[good] = rad[good] / r[good, None]
            rad_u[~good, 0] = 1.0
            base = k * self.n
            normals[base:base + self.n, 0] = rad_u[:, 0]
            normals[base:base + self.n, 2] = rad_u[:, 1]

            radial_excess = np.maximum(ROLLER_RADIUS - r, 0.0)
            axial_excess = np.maximum(np.abs(ay) - ROLLER_HALF_HEIGHT, 0.0)
            pen = np.sqrt(radial_excess ** 2 + axial_excess ** 2)
            mask = pen > 0.0
            if not np.any(mask):
                continue
            idx = np.nonzero(mask)[0]
            nrm = np.zeros((len(idx), 3))
            nrm[:, [0, 2]] = (
                radial_excess[idx, None] * rad_u[idx] / pen[idx, None]
            )
            nrm[:, 1] = axial_excess[idx] * np.sign(ay[idx]) / pen[idx]

            # closest point on the roller surface and its rigid velocity
            r_cl = np.minimum(r[idx], ROLLER_RADIUS)
            p_c = np.array(c) + np.outer(a_cl[idx], yhat)
            p_c[:, [0, 2]] += rad_u[idx] * r_cl[:, None]
            rel_c = p_c - c
            v_roll = omega[k] * np.stack(
                [rel_c[:, 2], np.zeros(len(idx)), -rel_c[:, 0]], axis=1
            )

            v_rel = self.v[idx] - v_roll
            vn = np.einsum("ij,ij->i", v_rel, nrm)
            fn_mag = np.maximum(
                K_NORMAL * pen[idx] - C_NORMAL * np.minimum(vn, 0.0), 0.0
            )
            fn_mag = np.minimum(fn_mag, MAX_CONTACT_FORCE)
            f_n = fn_mag[:, None] * nrm

            vt = v_rel - vn[:, None] * nrm
            vt_mag = np.linalg.norm(vt, axis=1)
            f_t = -K_TANGENT * vt
            cap = MU_FRICTION * fn_mag
            ft_mag = np.linalg.norm(f_t, axis=1)
            scale = np.ones(len(idx))
            over = ft_mag > 1e-12
            scale[over] = np.minimum(1.0, cap[over] / ft_mag[over])
            f_t = f_t * scale[:, None]

            np.add.at(f_ct, idx, f_n + f_t)

            rows = base + idx
            normals[rows] = nrm
            normals_f[rows] = fn_mag
            rel_vels[rows] = v_rel
            tang_f[rows] = f_t
            active[rows] = True

        self._last_contacts = {
            "normal": normals,
            "normal_force": normals_f,
            "rel_vel": rel_vels,
            "tangential_force": tang_f,
            "active": active,
        }
        return f_ct

    def _step(self, h: float) -> None:
        f_tet = self._elastic_forces()
        f_int = self._scatter @ f_tet
        f_ct = self._contact(self.t)
        f_grav = self.mass[:, None] * GRAVITY[None, :]
        f_damp = -VEL_DAMPING * self.mass[:, None] * self.v
        f_ext = f_ct + f_grav + f_damp

        # cumulative external impulse per budget domain (gravity, contact, damping)
        for g in range(self.n_domains):
            w = self.domain_weights[g]
            self.impulse[g] += h * (w @ f_ext)

        a = (f_int + f_ext) / self.mass[:, None]
        self.v += h * a
        self.v *= max(0.0, 1.0 - VEL_DAMPING * h)
        self.x += h * self.v
        self.t += h

        # guard against non-finite excursions (never triggered in practice)
        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all():
            self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------- driving
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-12:
            h = min(self.dt, target - self.t)
            if h <= 1e-12:
                break
            self._step(h)
            guard += 1
            if guard > 40_000_000:
                break
        return self._frame(target)

    # ------------------------------------------------------------------ frame
    def _deformation_gradient(self) -> np.ndarray:
        xe = self.x[self.tets]
        Ds = np.stack(
            [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], axis=2
        )
        return Ds @ self.Dm_inv

    def _frame(self, t: float) -> StateFrame:
        # contact term is the force acting at the current configuration
        f_ct = self._contact(t)
        ct = self._last_contacts
        exts = np.zeros((self.n_domains, self.n_terms, 3))
        w = self.domain_weights
        f_damp = -VEL_DAMPING * self.mass[:, None] * self.v
        for g in range(self.n_domains):
            exts[g, 0] = w[g] @ (self.mass[:, None] * GRAVITY[None, :])
            exts[g, 1] = w[g] @ f_ct
            exts[g, 2] = w[g] @ f_damp

        frame = StateFrame(
            time=t,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "rest_position": self.x0.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.impulse.copy(),
                "budget_cumulative_inflow_momentum": self.inflow.copy(),
                "budget_cumulative_outflow_momentum": self.outflow.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": exts,
                "cumulative_mass_source": self.mass_source.copy(),
                "element_deformation_gradient": self._deformation_gradient(),
                "friction_active_mask": ct["active"].copy(),
                "friction_dynamic_coefficient": np.full(len(ct["active"]), MU_FRICTION),
                "friction_static_coefficient": np.full(len(ct["active"]), MU_FRICTION),
                "friction_normal": ct["normal"].copy(),
                "friction_normal_force": ct["normal_force"].copy(),
                "friction_relative_velocity": ct["rel_vel"].copy(),
                "friction_tangential_force": ct["tangential_force"].copy(),
            },
            scalars={
                "friction_stick_speed_tolerance": V_STICK,
                "reference_force": 1.0e3,
                "reference_momentum": self.total_mass * REFERENCE_VELOCITY,
                "reference_velocity": REFERENCE_VELOCITY,
                "roller_friction": MU_FRICTION,
                "reference_youngs_modulus_pa": YOUNGS_MODULUS,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "dt": float(self.dt),
            "nodes": int(self.n),
            "tets": int(self.ne),
            "total_mass_kg": float(self.total_mass),
            "friction_coefficient": MU_FRICTION,
            "roller_angular_velocity_rad_s": float(ROLLER_OMEGA_MAG),
            "roller_angle_rad": float(self._roller_angle(self.t)),
            "intended_transport_direction": [1.0, 0.0, 0.0],
            "active_contacts": int(np.count_nonzero(
                self._last_contacts["active"])),
            "max_element_jacobian": float(
                np.max(np.linalg.det(self._deformation_gradient()))
            ),
        }
