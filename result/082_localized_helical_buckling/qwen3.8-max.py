"""Physically-based quasi-static solver for localized helical buckling of a twisted rod.

Model: inextensible, isotropic Kirchhoff elastic rod discretized as a discrete
elastic rod (DER-style): nodes x_i with material twist angles phi_e per edge
defined relative to a parallel-transported frame.  Energy

    E = sum_edges  EA/(2 l0) (l - l0)^2                       (inextensibility)
      + sum_hinges B theta^2 / (2 l_v0)                       (bending)
      + sum_hinges C (dphi)^2 / (2 l_v0)                      (twist)
      + self-contact penalty

with clamped ends: node positions x_0, x_1, x_{N-2}, x_{N-1} fixed
(position + tangent clamp), and edge twist angles phi_0 = 0,
phi_{M-1} = Psi(t) prescribed (material-frame clamp).  Because the material
frame is the parallel-transport frame, the elastic twist rate is dphi/ds and
the twist energy is minimized by a uniform twist distribution for given
total imposed twist Psi, so phi_e = Psi * e/(M-1); all buckling physics
(bending/stretch/contact) is carried by the node positions, solved
quasi-statically with L-BFGS at every load increment (warm started), which
reproduces the twist-induced localized helical buckling of the source
experiment.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        init = scene.initial_state()
        self.N = int(init.position.shape[0])
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.X_seed = np.asarray(init.position, dtype=float).copy()

        rod = cfg["rod"]
        proto = cfg["loading_protocol"]
        bc = cfg["boundary_conditions"]

        self.L = float(rod["length"])
        r = float(rod["radius"])
        self.radius = r
        E_mod = 1.0e9  # Pa, stiff polymer scale
        I_area = np.pi * r**4 / 4.0
        self.B = float(rod["bending_modulus_ratio"]) * E_mod * I_area
        self.C = float(rod["twisting_modulus_ratio"]) * E_mod * I_area
        self.EA = E_mod * np.pi * r**2
        self.l0 = self.L / (self.N - 1)
        self.lv0 = self.l0  # Voronoi length (l0+l0)/2

        tw = proto["twist_stage"]
        sh = proto["shortening_stage"]
        self.twist_t0 = float(tw["start_time"])
        self.twist_t1 = float(tw["end_time"])
        self.psi_final = float(tw["final_relative_rotation_rad"])
        self.short_t0 = float(sh["start_time"])
        self.short_t1 = float(sh["end_time"])

        self.left = np.array(bc["left_clamp"]["fixed_position"], dtype=float)
        self.xr0 = float(bc["right_clamp"]["initial_position"][0])
        self.xr1 = float(bc["right_clamp"]["final_position"][0])
        self.tan = np.array(bc["right_clamp"]["tangent"], dtype=float)
        self.tan /= np.linalg.norm(self.tan)

        # frozen scales
        self.ref_force = self.B / self.L**2
        self.ref_torque = self.C / self.L
        self.constraint_scale = self.L

        # self contact
        self.dmin = 2.2 * r
        self.kc = 1.0e5
        ii, jj = np.triu_indices(self.N, k=3)
        self.pi_idx = ii
        self.pj_idx = jj

        self.nfree = self.N - 4  # nodes 2 .. N-3
        self._prev_t = None
        self._prev_X = None
        self._q = None
        self._last = None
        self._iters = []

    # ---------------- loading protocol ----------------
    def _load(self, t: float):
        f_tw = np.clip((t - self.twist_t0) / max(self.twist_t1 - self.twist_t0, 1e-12), 0.0, 1.0)
        psi = self.psi_final * f_tw
        f_sh = np.clip((t - self.short_t0) / max(self.short_t1 - self.short_t0, 1e-12), 0.0, 1.0)
        xr = self.xr0 + (self.xr1 - self.xr0) * f_sh
        return psi, xr

    def _fixed_nodes(self, xr: float):
        N = self.N
        x0 = self.left.copy()
        x1 = x0 + self.l0 * self.tan
        xN = np.array([xr, 0.0, 0.0])
        xNm1 = xN - self.l0 * self.tan
        X = np.empty((N, 3))
        X[0] = x0
        X[1] = x1
        X[N - 1] = xN
        X[N - 2] = xNm1
        return X

    def _assemble(self, q: np.ndarray, xr: float) -> np.ndarray:
        X = self._fixed_nodes(xr)
        X[2:self.N - 2] = q.reshape(-1, 3)
        return X

    # ---------------- energy and gradient ----------------
    def _energy_grad(self, X: np.ndarray, psi: float, need_grad: bool):
        B, C, EA, l0 = self.B, self.C, self.EA, self.l0
        e = X[1:] - X[:-1]
        l = np.linalg.norm(e, axis=1)
        l = np.maximum(l, 1e-12)
        t = e / l[:, None]

        E = float(np.sum(0.5 * EA / l0 * (l - l0) ** 2))

        a = t[:-1]
        b = t[1:]
        c = np.einsum("ij,ij->i", a, b)
        c = np.clip(c, -1.0, 1.0)
        cr = np.cross(a, b)
        sc = np.linalg.norm(cr, axis=1)  # = sin(theta)
        theta = 2.0 * np.arctan2(sc, 1.0 + c)
        E += float(np.sum(0.5 * B * theta**2 / self.lv0))

        # twist energy (constant for given psi: uniform distribution)
        M = self.N - 1
        dphi = psi / (M - 1)
        E += float((self.N - 2) * 0.5 * C * dphi**2 / self.lv0)

        # self contact
        dvec = X[self.pj_idx] - X[self.pi_idx]
        dn = np.linalg.norm(dvec, axis=1)
        act = dn < self.dmin
        if np.any(act):
            pen = self.dmin - dn[act]
            E += float(np.sum(0.5 * self.kc * pen**2))

        if not need_grad:
            return E, None

        g = np.zeros_like(X)
        # stretch
        cs = EA * (l - l0) / l0
        gs = (cs / l)[:, None] * e
        g[1:] += gs
        g[:-1] -= gs
        # bending: dE/dx = f * dc/dx with f = -(B/lv0) * theta/sin(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sc > 1e-9, theta / np.maximum(sc, 1e-12), 1.0)
        f = -(B / self.lv0) * ratio
        H = self.N - 2
        la = l[:-1][:, None]
        lb = l[1:][:, None]
        u = (b - c[:, None] * a) / la  # d c / d x_{i} partial pieces
        v = (a - c[:, None] * b) / lb
        g[0:H] += -f[:, None] * u
        g[1:H + 1] += f[:, None] * (u - v)
        g[2:H + 2] += f[:, None] * v
        # contact
        if np.any(act):
            ia = self.pi_idx[act]
            ja = self.pj_idx[act]
            da = dn[act]
            unit = dvec[act] / np.maximum(da, 1e-12)[:, None]
            gmag = -self.kc * (self.dmin - da)  # dE/d(dn)
            np.add.at(g, ja, gmag[:, None] * unit)
            np.add.at(g, ia, -gmag[:, None] * unit)
        return E, g

    def _objective(self, q, xr, psi):
        X = self._assemble(q, xr)
        E, g = self._energy_grad(X, psi, True)
        gfree = g[2:self.N - 2].ravel()
        return E, gfree

    # ---------------- stepping ----------------
    def advance_to(self, time: float):
        time = float(time)
        psi, xr = self._load(time)

        # initial guess
        if self._q is None:
            X0 = self.X_seed.copy()
            X0[0], X0[1], X0[self.N - 1], X0[self.N - 2] = self._fixed_nodes(xr)[
                [0, 1, self.N - 1, self.N - 2]
            ]
            q = X0[2:self.N - 2].ravel().copy()
        else:
            q = self._q.copy()

        res = minimize(
            self._objective,
            q,
            args=(xr, psi),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 800, "maxfun": 1500, "ftol": 1e-15, "gtol": 1e-9},
        )
        q = np.asarray(res.x, dtype=float)
        self._q = q
        self._iters.append(int(res.nit))
        X = self._assemble(q, xr)

        if self._prev_X is None or time <= self._prev_t:
            V = np.zeros_like(X)
        else:
            V = (X - self._prev_X) / (time - self._prev_t)

        # reactions at clamped nodes: external force = dE/dx at fixed nodes
        _, gfull = self._energy_grad(X, psi, True)
        N = self.N
        fix_idx = np.array([0, 1, N - 2, N - 1])
        F = gfull[fix_idx]  # [4,3]
        # clamp couples about the tangent needed to hold prescribed material frame
        M = N - 1
        omega3 = psi / ((M - 1) * self.lv0)
        coup = np.zeros((4, 3))
        coup[0] = -self.C * omega3 * self.tan
        coup[3] = +self.C * omega3 * self.tan

        dist = float(np.linalg.norm(X[N - 1] - X[0]))
        constraint_value = np.array([dist, psi])
        constraint_target = np.array([abs(xr - self.left[0]), psi])

        frame = StateFrame(
            time=time,
            position=X,
            velocity=V,
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_length": np.linspace(0.0, self.L, N),
            },
            fields={
                "budget_external_force": F.reshape(1, 4, 3),
                "budget_external_couple": coup.reshape(1, 4, 3),
                "budget_force_position": X[fix_idx].reshape(1, 4, 3),
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "reference_force": float(self.ref_force),
                "reference_torque": float(self.ref_torque),
            },
        )
        frame.validate()
        self._prev_t = time
        self._prev_X = X.copy()
        self._last = {
            "time": time,
            "psi": psi,
            "xr": xr,
            "energy": float(res.fun),
            "max_lateral": float(np.max(np.abs(X[2:N - 2, 1:]))),
            "min_pair_dist": float(np.min(np.linalg.norm(
                X[self.pj_idx] - X[self.pi_idx], axis=1))),
        }
        return frame

    # ---------------- diagnostics ----------------
    def diagnostics(self) -> dict:
        d = {"seed": self.seed, "solver": "quasi-static DER L-BFGS"}
        if self._iters:
            d["lbfgs_max_iters_per_frame"] = int(np.max(self._iters))
            d["lbfgs_mean_iters_per_frame"] = float(np.mean(self._iters))
        if self._last is not None:
            d.update(self._last)
        return d
