"""Physics solver for scene 069_solid_beam_twist.

Total-Lagrangian finite-element simulation of a volumetric elastic beam
(St. Venant-Kirchhoff material, 8-node hexahedral elements, 2x2x2 Gauss
quadrature) whose two end faces are prescribed to rotate in opposite
directions about the beam (x) axis. Time integration is semi-implicit
(symplectic) Euler with mass-proportional Rayleigh damping; the prescribed
end rotations are enforced kinematically and the associated reaction
impulses are accumulated for the momentum-budget trajectory fields.
"""
from __future__ import annotations

import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback if import path differs
    StateFrame = object


def _smoothstep_interp(keys, t):
    """Smoothstep interpolation of a [(time, angle_degrees), ...] schedule."""
    t0, a0 = float(keys[0][0]), float(keys[0][1])
    t1, a1 = float(keys[-1][0]), float(keys[-1][1])
    if t1 <= t0:
        return a0, 0.0
    u = (t - t0) / (t1 - t0)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    s = u * u * (3.0 - 2.0 * u)
    angle = a0 + (a1 - a0) * s
    rate = (a1 - a0) * 6.0 * u * (1.0 - u) / (t1 - t0)
    return angle, rate


class BenchmarkSolver:
    """Explicit FEM solver for the opposed-end-rotation beam twist."""

    def __init__(self) -> None:
        self.dt = 1.0e-3
        self.alpha = 3.0  # Rayleigh mass-damping coefficient [1/s]
        self.young = 1.0e5  # Pa
        self.poisson = 0.35

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config if hasattr(scene, "config") else scene
        self.seed = int(seed)
        self.config = cfg

        frame = scene.initial_state()
        self.rest = np.array(frame.position, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.end_region = np.array(frame.attributes["end_region"], dtype=float)
        self.n = len(self.rest)

        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0

        bc = cfg["boundary_conditions"]
        self.left_center = np.asarray(bc["left_end"]["center"], dtype=float)
        self.right_center = np.asarray(bc["right_end"]["center"], dtype=float)
        self.left_keys = bc["left_end"]["prescribed_rotation_degrees"]
        self.right_keys = bc["right_end"]["prescribed_rotation_degrees"]
        self.duration = float(cfg["duration"])
        axis = np.asarray(bc["rotation_axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        # Beam axis is x per config; rotations act about the x direction.
        self.axis_index = int(np.argmax(np.abs(axis)))

        self.left_idx = np.nonzero(self.end_region < -0.5)[0]
        self.right_idx = np.nonzero(self.end_region > 0.5)[0]
        self.left_d = self.rest[self.left_idx] - self.left_center
        self.right_d = self.rest[self.right_idx] - self.right_center

        # Lamé parameters.
        e, nu = self.young, self.poisson
        self.mu = e / (2.0 * (1.0 + nu))
        self.lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        self._build_elements(cfg)

        # Budget domains: [whole beam, left end band, right end band].
        left_w = np.zeros(self.n)
        left_w[self.left_idx] = 1.0
        right_w = np.zeros(self.n)
        right_w[self.right_idx] = 1.0
        self.domain_weights = np.vstack(
            [np.ones(self.n), left_w, right_w]
        )
        self.g = self.domain_weights.shape[0]
        self.k = 1  # one conserved material
        self.cum_impulse = np.zeros((self.g, 3))
        self.frame_impulse = np.zeros((self.g, 3))
        self.frame_time = 0.0

        # Positive momentum scale (kinematic scale of the driven ends).
        a1 = float(self.right_keys[-1][1]) - float(self.right_keys[0][1])
        t0 = float(self.right_keys[0][0])
        t1 = float(self.right_keys[-1][0])
        omega_max = np.deg2rad(abs(a1)) * 1.5 / max(t1 - t0, 1e-9)
        beam = cfg["beam"]
        half_w = 0.5 * max(float(beam["dimensions"][1]), float(beam["dimensions"][2]))
        v_char = omega_max * half_w
        self.reference_momentum = float(np.sum(self.mass) * max(v_char, 1e-6))

    def _build_elements(self, cfg) -> None:
        nx, ny, nz = (int(v) for v in cfg["beam"]["volumetric_grid_shape"])
        self.nx, self.ny, self.nz = nx, ny, nz

        def nid(i, j, k):
            return (i * ny + j) * nz + k

        elems = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    elems.append(
                        [
                            nid(i, j, k),
                            nid(i + 1, j, k),
                            nid(i + 1, j + 1, k),
                            nid(i, j + 1, k),
                            nid(i, j, k + 1),
                            nid(i + 1, j, k + 1),
                            nid(i + 1, j + 1, k + 1),
                            nid(i, j + 1, k + 1),
                        ]
                    )
        self.elems = np.asarray(elems, dtype=np.int64)
        ne = len(self.elems)

        # Trilinear hex shape-function gradients in reference coordinates.
        corners = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        g = 1.0 / np.sqrt(3.0)
        gp = np.array(
            [[sx * g, sy * g, sz * g] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        ng = len(gp)
        # dNdXi[gauss, node, dir] = dN_node / dXi_dir
        dNdXi = np.zeros((ng, 8, 3))
        for q in range(ng):
            xi, eta, zeta = gp[q]
            for a in range(8):
                ca = corners[a]
                dNdXi[q, a, 0] = 0.125 * ca[0] * (1 + ca[1] * eta) * (1 + ca[2] * zeta)
                dNdXi[q, a, 1] = 0.125 * ca[1] * (1 + ca[0] * xi) * (1 + ca[2] * zeta)
                dNdXi[q, a, 2] = 0.125 * ca[2] * (1 + ca[0] * xi) * (1 + ca[1] * eta)

        x0 = self.rest[self.elems]  # (ne, 8, 3)
        # J[e,g,l,i] = sum_a X[a,i] dN_a/dXi_l
        jac = np.einsum("eai,gal->egli", x0, dNdXi)
        det = (
            jac[..., 0, 0] * (jac[..., 1, 1] * jac[..., 2, 2] - jac[..., 1, 2] * jac[..., 2, 1])
            - jac[..., 0, 1] * (jac[..., 1, 0] * jac[..., 2, 2] - jac[..., 1, 2] * jac[..., 2, 0])
            + jac[..., 0, 2] * (jac[..., 1, 0] * jac[..., 2, 1] - jac[..., 1, 1] * jac[..., 2, 0])
        )
        if np.any(det <= 0.0):
            raise ValueError("non-positive reference Jacobian in beam mesh")
        inv = np.empty_like(jac)
        inv[..., 0, 0] = jac[..., 1, 1] * jac[..., 2, 2] - jac[..., 1, 2] * jac[..., 2, 1]
        inv[..., 0, 1] = jac[..., 0, 2] * jac[..., 2, 1] - jac[..., 0, 1] * jac[..., 2, 2]
        inv[..., 0, 2] = jac[..., 0, 1] * jac[..., 1, 2] - jac[..., 0, 2] * jac[..., 1, 1]
        inv[..., 1, 0] = jac[..., 1, 2] * jac[..., 2, 0] - jac[..., 1, 0] * jac[..., 2, 2]
        inv[..., 1, 1] = jac[..., 0, 0] * jac[..., 2, 2] - jac[..., 0, 2] * jac[..., 2, 0]
        inv[..., 1, 2] = jac[..., 0, 2] * jac[..., 1, 0] - jac[..., 0, 0] * jac[..., 1, 2]
        inv[..., 2, 0] = jac[..., 1, 0] * jac[..., 2, 1] - jac[..., 1, 1] * jac[..., 2, 0]
        inv[..., 2, 1] = jac[..., 0, 1] * jac[..., 2, 0] - jac[..., 0, 0] * jac[..., 2, 1]
        inv[..., 2, 2] = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
        inv /= det[..., None, None]

        # dNdX[e,g,a,i] = sum_l dN_a/dXi_l * invJ[l,i]
        self.dndx = np.einsum("gal,egli->egai", dNdXi, inv).astype(np.float64)
        self.wv = det  # unit Gauss weights times reference element volume
        self.ne = ne
        self.elem_flat = self.elems.ravel()

    # ------------------------------------------------------------- mechanics
    def _stress_forces(self, x):
        """Internal elastic nodal forces (and per-element mean F)."""
        xe = x[self.elems]  # (ne,8,3)
        # deformation gradient per Gauss point
        fgp = np.einsum("eai,egaj->egij", xe, self.dndx)
        # Green-Lagrange strain
        cgreen = np.einsum("egij,egik->egjk", fgp, fgp)
        strain = 0.5 * (cgreen - np.eye(3))
        tr = strain[..., 0, 0] + strain[..., 1, 1] + strain[..., 2, 2]
        stress = 2.0 * self.mu * strain
        stress[..., 0, 0] += self.lam * tr
        stress[..., 1, 1] += self.lam * tr
        stress[..., 2, 2] += self.lam * tr
        pio = np.einsum("egij,egjk->egik", fgp, stress)
        fe = -np.einsum("eg,egij,egaj->eai", self.wv, pio, self.dndx)
        f = np.empty((self.n, 3))
        flat = self.elem_flat
        for d in range(3):
            f[:, d] = np.bincount(flat, fe[:, :, d].ravel(), minlength=self.n)
        elem_f = np.einsum("eg,egij->eij", self.wv, fgp) / self.wv.sum(axis=1)[:, None, None]
        return f, elem_f

    def _end_state(self, t):
        """Prescribed angles (rad) and angular rates (rad/s) at time t."""
        al, rl = _smoothstep_interp(self.left_keys, t)
        ar, rr = _smoothstep_interp(self.right_keys, t)
        return np.deg2rad(al), np.deg2rad(rl), np.deg2rad(ar), np.deg2rad(rr)

    def _apply_rotation(self, d, theta, omega):
        """Rotate offsets d about the x axis; return (pos_offset, vel)."""
        c, s = np.cos(theta), np.sin(theta)
        rx = d[:, 0]
        ry = c * d[:, 1] - s * d[:, 2]
        rz = s * d[:, 1] + c * d[:, 2]
        vel = np.stack(
            [np.zeros_like(rx), -omega * rz, omega * ry], axis=1
        )
        return np.stack([rx, ry, rz], axis=1), vel

    def _substep(self, h, t_new):
        fint, elem_f = self._stress_forces(self.x)
        self._last_elem_f = elem_f
        acc = fint / self.mass[:, None] - self.alpha * self.v
        v_star = self.v + h * acc
        x_star = self.x + h * v_star

        thl, wl, thr, wr = self._end_state(t_new)

        imp = np.zeros((self.g, 3))
        if len(self.left_idx):
            off, vel = self._apply_rotation(self.left_d, thl, wl)
            pos = self.left_center + off
            m = self.mass[self.left_idx]
            imp[0] += np.sum(m[:, None] * (vel - v_star[self.left_idx]), axis=0)
            imp[1] += np.sum(m[:, None] * (vel - v_star[self.left_idx]), axis=0)
            x_star[self.left_idx] = pos
            v_star[self.left_idx] = vel
        if len(self.right_idx):
            off, vel = self._apply_rotation(self.right_d, thr, wr)
            pos = self.right_center + off
            m = self.mass[self.right_idx]
            imp[0] += np.sum(m[:, None] * (vel - v_star[self.right_idx]), axis=0)
            imp[2] += np.sum(m[:, None] * (vel - v_star[self.right_idx]), axis=0)
            x_star[self.right_idx] = pos
            v_star[self.right_idx] = vel

        self.x = x_star
        self.v = v_star
        self.cum_impulse += imp
        self.frame_impulse += imp

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float):
        target = float(time)
        if target < 0.0 or not np.isfinite(target):
            raise ValueError("advance_to requires a finite non-negative time")
        while self.t < target - 1e-9:
            nxt = min(self.t + self.dt, target)
            if target - nxt < 1e-12:
                nxt = target
            self._substep(nxt - self.t, nxt)
            self.t = nxt

        elapsed = self.t - self.frame_time
        if elapsed > 1e-12:
            ext_force = self.frame_impulse / elapsed
        else:
            ext_force = np.zeros((self.g, 3))
        ext_force = ext_force[:, None, :]  # (G, K=1, 3)
        self.frame_impulse[:] = 0.0
        self.frame_time = self.t

        if getattr(self, "_last_elem_f", None) is None:
            _, elem_f = self._stress_forces(self.x)
        else:
            elem_f = self._last_elem_f

        thl, _wl, thr, _wr = self._end_state(self.t)
        rel_deg = float(np.rad2deg(thr - thl))

        attributes = {
            "mass_quantity": self.mass.copy(),
            "end_region": self.end_region.copy(),
            "rest_position": self.rest.copy(),
        }
        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((self.g, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((self.g, 3)),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": ext_force,
            "cumulative_mass_source": np.zeros(self.k),
            "element_deformation_gradient": elem_f.copy(),
        }
        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "relative_end_rotation_degrees": rel_deg,
        }
        frame = StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "num_nodes": int(self.n),
            "num_elements": int(self.ne),
            "dt": float(self.dt),
            "time": float(self.t),
            "young_modulus": float(self.young),
            "seed": int(self.seed),
        }
