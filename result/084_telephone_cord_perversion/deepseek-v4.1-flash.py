"""Discrete elastic rod solver for the telephone-cord helical perversion scene.

The rod is modelled as a Kirchhoff rod discretised on the frozen 161-vertex
centerline: a discrete stretch (inextensibility) constraint, discrete bending
and twisting energies with intrinsic natural curvature/torsion taken from the
configured circular helix, Rayleigh (mass-proportional) damping and the
prescribed quasi-static end grips.  Time advance is a semi-implicit
(linearly implicit in the bending operator) damped dynamics step on the
inextensible constraint manifold, with the material twist field relaxed
toward elastic equilibrium at every step.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _skew(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def _rotate(q, v):
    """Rotate vector(s) v by unit quaternion q (xyzw)."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    v = np.asarray(v, dtype=float)
    return v @ R.T


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        rod = self.config["rod"]
        self.n = int(rod["vertex_count"])
        self.L = float(rod["total_length"])
        self.mass_total = float(rod["mass"])
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        shape = rod["natural_shape"]
        self.helix_radius = float(shape["radius"])
        self.helix_pitch = float(shape["pitch_per_turn"])
        self.helix_hand = 1.0 if str(shape.get("handedness", "right")).lower() == "right" else -1.0

        cs = rod["cross_section"]
        self.width = float(cs["width"])
        self.thickness = float(cs["thickness"])

        ctl = self.config["endpoint_controls"]
        self.left0 = np.asarray(ctl["left"]["initial_position"], dtype=float)
        self.left1 = np.asarray(ctl["left"]["final_position"], dtype=float)
        self.right0 = np.asarray(ctl["right"]["initial_position"], dtype=float)
        self.right1 = np.asarray(ctl["right"]["final_position"], dtype=float)
        self.motion = ctl["motion"]

        # --- geometric discretisation of the frozen centerline -------------
        self.s = np.linspace(0.0, 1.0, self.n)
        self.h = self.L / (self.n - 1)
        edges = np.diff(self.s)
        self.edge_length0 = self.h
        self.vertex_measure = np.full(self.n, self.h)
        self.vertex_measure[0] = 0.5 * self.h
        self.vertex_measure[-1] = 0.5 * self.h
        self.vertex_mass = self.mass_total * self.vertex_measure / self.L

        # --- isotropic bending / torsional rigidities (rectangular section) -
        A = self.width * self.thickness
        I1 = self.thickness ** 3 * self.width / 12.0
        I2 = self.width ** 3 * self.thickness / 12.0
        J = I1 + I2
        self.E = 6.0e7
        self.G = 3.0e7
        # compliant bending axis carries the natural curvature; the stiff axis
        # penalises transverse curvature (anisotropic rectangular section)
        self.k_a = self.E * I1
        self.k_b = self.E * I2
        self.tau_rig = self.G * J
        self.sigma = 20.0
        self.dt_sub = 2.5e-3
        self.n_twist = 25
        self.alpha_psi = 2.0e-4
        self.material_axis_world = np.asarray(
            rod["initial_preparation"]["material_axis"], dtype=float
        )
        self.material_axis_world = self.material_axis_world / np.linalg.norm(
            self.material_axis_world
        )
        self.constraint_scale = float(np.linalg.norm(self.right0 - self.left0))
        self._grip_force = np.zeros((2, 3))

        # --- natural (stress-free) shape: circular helix --------------------
        hh = self.helix_hand
        R = self.helix_radius
        p = self.helix_pitch
        denom = np.sqrt(p * p + 4.0 * np.pi * np.pi * R * R)
        self.kappa_n = 4.0 * np.pi * np.pi * R / (denom * denom)
        self.tau_n = hh * 2.0 * np.pi * p / (denom * denom)
        self.reference_force = float(max(self.k_a * self.kappa_n ** 2, 1.0e-6))
        self.reference_torque = float(max(self.tau_rig * abs(self.tau_n), 1.0e-9))

        # --- reference material frame: flat, anisotropic axis = +y ----------
        m1_0 = np.tile(np.array([0.0, 1.0, 0.0]), (self.n, 1))
        self.m1_ref = m1_0

        # --- initial (prepared, near-straight, stressed) state --------------
        s = self.s
        pos = self.left0[None, :] * (1.0 - s[:, None]) + self.right0[None, :] * s[:, None]
        amp = float(rod["initial_preparation"]["centerline_wave_amplitude"])
        waves = int(rod["initial_preparation"]["centerline_wave_count"])
        pos[:, 2] += amp * np.sin(2.0 * np.pi * waves * s) * np.sin(np.pi * s)
        self.x = pos

        tangent0 = np.zeros_like(pos)
        tangent0[:, 0] = 1.0
        self.tangent_ref = tangent0
        m2_0 = np.cross(tangent0, m1_0)

        self.psi = np.zeros(self.n)
        self.m1 = m1_0.copy()
        self.m2 = m2_0.copy()

        self._last_time = 0.0
        self._relax_count = 0
        self._diag = {}
        self._build_stencil()
        _, t0v = self._tangents(self.x)
        b1, b2 = self._bishop(t0v)
        self.psi[0] = self._end_twist(t0v, b1, b2, 0)
        self.psi[-1] = self._end_twist(t0v, b1, b2, -1)
        m1c, m2c = self._material_frames(t0v, b1, b2, self.psi)
        _, _, _, _, ebend = self._bend_state(self.x, m1c, m2c)
        self._energy_initial = float(ebend + self._twist_energy(self.psi))
        self._energy_prev = self._energy_initial
        self.v = np.zeros_like(self.x)

    # ------------------------------------------------------------ geometry
    def _tangents(self, x):
        d = x[1:] - x[:-1]
        l = np.linalg.norm(d, axis=1)
        l = np.maximum(l, 1.0e-12)
        te = d / l[:, None]
        t = np.empty_like(x)
        t[1:-1] = te[:-1] + te[1:]
        t[0] = te[0]
        t[-1] = te[-1]
        nn = np.maximum(np.linalg.norm(t, axis=1), 1.0e-12)
        return te, t / nn[:, None]

    @staticmethod
    def _transport(v, t0, t1):
        c = float(np.dot(t0, t1))
        if c > 1.0 - 1.0e-12:
            return v
        if c < -1.0 + 1.0e-8:
            axis = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(t0, axis)
            u = u / max(np.linalg.norm(u), 1.0e-12)
            u = np.cross(t0, u)
            return u
        w = np.cross(t0, t1)
        K = _skew(w)
        R = np.eye(3) + K + K @ K / (1.0 + c)
        u = R @ v
        u = u - t1 * float(np.dot(u, t1))
        n = np.linalg.norm(u)
        if n < 1.0e-12:
            axis = np.array([0.0, 0.0, 1.0]) if abs(t1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(t1, axis)
            n = max(np.linalg.norm(u), 1.0e-12)
        return u / n

    def _bishop(self, t):
        b1 = np.empty_like(t)
        a = self.material_axis_world
        v = a - t[0] * float(np.dot(a, t[0]))
        n = np.linalg.norm(v)
        if n < 1.0e-9:
            v = np.cross(t[0], np.array([1.0, 0.0, 0.0]))
            n = max(np.linalg.norm(v), 1.0e-12)
        b1[0] = v / n
        for i in range(self.n - 1):
            b1[i + 1] = self._transport(b1[i], t[i], t[i + 1])
        return b1, np.cross(t, b1)

    def _end_twist(self, t, b1, b2, idx):
        a = self.material_axis_world
        m = a - t[idx] * float(np.dot(a, t[idx]))
        n = np.linalg.norm(m)
        if n < 1.0e-9:
            m = b1[idx]
            n = 1.0
        m = m / n
        return float(np.arctan2(np.dot(np.cross(b1[idx], m), t[idx]), np.dot(b1[idx], m)))

    def _material_frames(self, t, b1, b2, psi):
        m1 = np.cos(psi)[:, None] * b1 + np.sin(psi)[:, None] * b2
        m2 = np.cross(t, m1)
        return m1, m2

    # ------------------------------------------------- discrete rod energy
    def _build_stencil(self):
        n = self.n
        idx = np.zeros((n, 3), dtype=int)
        coef = np.zeros((n, 3))
        inv = 1.0 / (self.h * self.h)
        for i in range(n):
            if i == 0:
                idx[i] = [0, 1, 2]
                coef[i] = [1.0, -2.0, 1.0]
            elif i == n - 1:
                idx[i] = [n - 3, n - 2, n - 1]
                coef[i] = [1.0, -2.0, 1.0]
            else:
                idx[i] = [i - 1, i, i + 1]
                coef[i] = [-1.0, 2.0, -1.0]
        self.sten_idx = idx
        self.sten_coef = coef * inv
        w = np.full(n, self.h)
        w[0] = 0.5 * self.h
        w[-1] = 0.5 * self.h
        self.bend_weight = w

    def _curvature(self, x):
        kap = np.zeros_like(x)
        for a in range(3):
            np.add.at(kap, self.sten_idx[:, a], self.sten_coef[:, a][:, None] * x)
        return kap

    def _bend_state(self, x, m1, m2):
        kap = self._curvature(x)
        k1 = np.einsum("ij,ij->i", kap, m1)
        k2 = np.einsum("ij,ij->i", kap, m2)
        strain = (self.k_a * (k1 - self.kappa_n))[:, None] * m1 + (self.k_b * k2)[:, None] * m2
        c = (self.bend_weight[:, None]) * strain
        energy = 0.5 * float(
            np.sum(self.bend_weight * (self.k_a * (k1 - self.kappa_n) ** 2 + self.k_b * k2 ** 2))
        )
        return kap, k1, k2, c, energy

    def _bend_force(self, c):
        F = np.zeros((self.n, 3))
        for a in range(3):
            np.add.at(F, self.sten_idx[:, a], -self.sten_coef[:, a][:, None] * c)
        return F

    def _bend_stiffness(self, m1, m2):
        n = self.n
        B = self.k_a * m1[:, :, None] * m1[:, None, :] + self.k_b * m2[:, :, None] * m2[:, None, :]
        rows = []
        cols = []
        vals = []
        w = self.bend_weight
        for a in range(3):
            for b in range(3):
                base = w * self.sten_coef[:, a] * self.sten_coef[:, b]
                for p in range(3):
                    for q in range(3):
                        rows.append(3 * self.sten_idx[:, a] + p)
                        cols.append(3 * self.sten_idx[:, b] + q)
                        vals.append(base * B[:, p, q])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        return sp.coo_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n)).tocsc()

    def _twist_energy(self, psi):
        g = np.diff(psi) / self.h - self.tau_n
        return 0.5 * self.tau_rig * self.h * float(np.sum(g * g))

    def _twist_relax(self, psi, nsteps):
        tau = self.tau_rig
        for _ in range(nsteps):
            g = np.diff(psi) / self.h - self.tau_n
            grad = tau * (g[1:] - g[:-1])
            psi[1:-1] -= self.alpha_psi * grad
        return psi

    # ---------------------------------------------------------- integration
    def _project(self, x, xL, xR):
        h = self.h
        for _ in range(4):
            d = x[1:] - x[:-1]
            l = np.linalg.norm(d, axis=1)
            l = np.maximum(l, 1.0e-12)
            corr = 0.5 * (l - h) / l
            x[1:] -= d * corr[:, None]
            x[:-1] += d * corr[:, None]
            x[0] = xL
            x[-1] = xR
        return x

    def _targets(self, time):
        m = self.motion
        t0 = float(m["move_start"])
        t1 = float(m["move_end"])
        span = max(t1 - t0, 1.0e-9)
        u = float(np.clip((time - t0) / span, 0.0, 1.0))
        su = u * u * (3.0 - 2.0 * u)
        dsu = 6.0 * u * (1.0 - u) / span if 0.0 < u < 1.0 else 0.0
        xL = (1.0 - su) * self.left0 + su * self.left1
        xR = (1.0 - su) * self.right0 + su * self.right1
        vL = dsu * (self.left1 - self.left0)
        vR = dsu * (self.right1 - self.right0)
        return xL, xR, vL, vR

    def _substep(self, dt, x, v, xL, xR, vL, vR):
        te, t = self._tangents(x)
        b1, b2 = self._bishop(t)
        self.psi[0] = self._end_twist(t, b1, b2, 0)
        self.psi[-1] = self._end_twist(t, b1, b2, -1)
        m1, m2 = self._material_frames(t, b1, b2, self.psi)
        kap, k1, k2, c, ebend = self._bend_state(x, m1, m2)
        etw = self._twist_energy(self.psi)

        F = self._bend_force(c) + self.vertex_mass[:, None] * self.gravity[None, :]
        M = self.vertex_mass
        A = self._bend_stiffness(m1, m2) * dt + sp.diags(
            np.repeat(M * (1.0 / dt + self.sigma), 3)
        )
        rhs = (F + (M * v) / dt).reshape(-1)

        dof = np.arange(3, 3 * (self.n - 1))
        bnd = np.concatenate(
            [np.arange(3), np.arange(3 * (self.n - 1), 3 * self.n)]
        )
        vb = np.concatenate([vL, vR])
        A_ii = A[dof][:, dof].tocsc()
        rhs_i = rhs[dof] - A[dof][:, bnd] @ vb
        vnew = np.empty((self.n, 3))
        vnew.reshape(-1)[dof] = spla.splu(A_ii).solve(rhs_i)
        vnew[0] = vL
        vnew[-1] = vR

        # external (grip) actions holding the controlled ends
        F_grip = np.zeros((2, 3))
        F_grip[0] = M[0] * (vnew[0] - v[0]) / dt - F[0] + self.sigma * M[0] * vnew[0]
        F_grip[1] = (
            M[-1] * (vnew[-1] - v[-1]) / dt - F[-1] + self.sigma * M[-1] * vnew[-1]
        )

        xnew = x + dt * vnew
        xnew = self._project(xnew, xL, xR)
        vnew[0] = vL
        vnew[-1] = vR

        # relax the material twist field toward equilibrium
        te2, t2 = self._tangents(xnew)
        b1b, b2b = self._bishop(t2)
        self.psi[0] = self._end_twist(t2, b1b, b2b, 0)
        self.psi[-1] = self._end_twist(t2, b1b, b2b, -1)
        self._twist_relax(self.psi, self.n_twist)

        return xnew, vnew, F_grip, ebend + etw

    # ------------------------------------------------------------ sampling
    def _end_couple(self, t, k1, k2):
        """Free couples the grips apply to hold the end material frames."""
        w = self.bend_weight
        g = np.diff(self.psi) / self.h - self.tau_n
        tau = np.zeros((2, 3))
        for slot, i, edge, sign in ((0, 0, 0, -1.0), (1, self.n - 1, self.n - 2, 1.0)):
            bend = w[i] * (
                self.k_a * (k1[i] - self.kappa_n) * k2[i] - self.k_b * k2[i] * k1[i]
            )
            tau[slot] = -(bend + sign * self.tau_rig * g[edge]) * t[i]
        return tau

    def _frame(self, time):
        x = self.x
        te, t = self._tangents(x)
        b1, b2 = self._bishop(t)
        psi = self.psi
        m1, m2 = self._material_frames(t, b1, b2, psi)
        kap, k1, k2, c, ebend = self._bend_state(x, m1, m2)
        etw = self._twist_energy(psi)
        fint = self._bend_force(c)
        couple = self._end_couple(t, k1, k2)

        xL, xR, _, _ = self._targets(time)
        sep = float(np.linalg.norm(x[-1] - x[0]))
        sep_t = float(np.linalg.norm(xR - xL))
        attrs = {
            "arc_fraction": self.s,
            "mass_quantity": self.vertex_mass,
            "material_axis_1": m1,
            "material_axis_2": m2,
            "tangent": t,
            "twist_angle": psi,
            "natural_torsion": np.full(self.n, self.tau_n),
            "natural_curvature": np.full(self.n, self.kappa_n),
            "cross_section_width": np.full(self.n, self.width),
            "cross_section_thickness": np.full(self.n, self.thickness),
            "bend_stiffness": np.full(self.n, self.k_a),
            "twist_stiffness": np.full(self.n, self.tau_rig),
            "elastic_energy": np.full(self.n, (ebend + etw) / self.n),
            "endpoint_controlled": np.isin(np.arange(self.n), [0, self.n - 1]).astype(float),
        }
        fields = {
            "budget_external_force": np.asarray(self._grip_force).reshape(1, 2, 3),
            "budget_force_position": np.stack([x[0], x[-1]]).reshape(1, 2, 3),
            "budget_external_couple": couple.reshape(1, 2, 3),
            "constraint_value": np.array([sep, float(x[0][0]), float(x[-1][0])]),
            "constraint_target": np.array([sep_t, float(xL[0]), float(xR[0])]),
            "cumulative_mass_source": np.zeros(1),
            "elastic_internal_force": fint,
            "elastic_internal_couple": (self.tau_rig * (np.diff(psi) / self.h - self.tau_n))[
                :, None
            ]
            * 0.5
            * (t[:-1] + t[1:]),
            "gravity_acceleration": self.gravity.reshape(1, 3),
        }
        scalars = {
            "constraint_scale": self.constraint_scale,
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
            "rod_length": self.L,
            "elastic_energy_total": float(ebend + etw),
        }
        frame = StateFrame(
            float(time),
            x.copy(),
            self.v.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1)),
            angular_velocity=np.zeros((self.n, 3)),
            mass=self.vertex_mass.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        dt_sample = float(self.config.get("sample_dt", 0.02))
        if time <= self._last_time + 1.0e-12:
            return self._frame(time)
        duration = time - self._last_time
        nsub = max(1, int(round(duration / self.dt_sub)))
        dt = duration / nsub
        F_acc = np.zeros((2, 3))
        e_acc = 0.0
        for k in range(nsub):
            t_now = self._last_time + (k + 1.0) * dt
            xL, xR, vL, vR = self._targets(t_now)
            self.x, self.v, F_grip, energy = self._substep(
                dt, self.x, self.v, xL, xR, vL, vR
            )
            F_acc += F_grip
            e_acc = energy
        self._grip_force = F_acc / nsub
        self._last_time = time
        self._relax_count += nsub
        self._energy_prev = e_acc
        self._diag["last_energy"] = e_acc
        self._diag["grip_force_magnitude"] = float(np.linalg.norm(self._grip_force.sum(axis=0)))
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)
