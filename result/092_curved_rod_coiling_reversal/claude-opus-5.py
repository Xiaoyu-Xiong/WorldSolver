"""Discrete elastic rod (Kirchhoff) solver for coiling reversal of a naturally curved rod.

Physics
-------
Inextensible Kirchhoff rod discretised as a polyline of nodes (positions) plus a
per-edge material-frame angle (twist degree of freedom), i.e. a discrete elastic
rod in the sense of Bergou et al.:

  * bending energy   sum_i  (EI / 2 l_i) | kb_i - kb0_i |^2   with a *natural*
    curvature target kb0_i = kappa0 * l_i * m2_i (natural bending plane carried
    by the material frame, hence twist reorients the intrinsic curvature),
  * twisting energy  sum_i  (GJ / 2 l_i) ( theta_i - theta_{i-1} + refTwist_i )^2,
  * uniform gravity, viscous + neighbour-relative dissipation (quasi-static
    regime requested by the configuration),
  * exact inextensibility by a linearised length-constraint projection solved
    with a banded (tridiagonal) direct solve; the Lagrange multipliers give the
    axial tension used for the external-force budget,
  * rigid stationary substrate with the source *sticking* condition: a node that
    reaches the substrate is deposited and pinned there for good,
  * injector: node 0 is clamped at the outlet, node 1 is driven along the
    injector axis so material is extruded at the prescribed speed; new nodes are
    created at the outlet with a frozen inlet material orientation.

Everything is time stepped explicitly (symplectic Euler + implicit linear drag +
constraint projection), no scripted motion of any kind.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, _EPS)


def _parallel_transport(u: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Rotate u (perp. to t0) onto the plane perp. to t1 without twist."""
    b = np.cross(t0, t1)
    sn = np.linalg.norm(b, axis=-1)
    cs = np.clip(np.einsum("ij,ij->i", t0, t1), -1.0, 1.0)
    k = b / np.maximum(sn, _EPS)[:, None]
    kdu = np.einsum("ij,ij->i", k, u)
    out = (u * cs[:, None] + np.cross(k, u) * sn[:, None]
           + k * (kdu * (1.0 - cs))[:, None])
    small = sn < 1e-10
    if np.any(small):
        out[small] = u[small]
    return _unit(out - np.einsum("ij,ij->i", out, t1)[:, None] * t1)


class BenchmarkSolver:
    """Discrete elastic rod with intrinsic curvature, twist, injection and sticking."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)

        rod = cfg["rod"]
        inj = cfg["injector"]
        sub = cfg["substrate"]

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.E = float(rod["young_modulus"])
        self.nu = float(rod["poisson_ratio"])
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.area = np.pi * self.radius ** 2
        inertia = np.pi * self.radius ** 4 / 4.0          # second moment
        self.EI = self.E * inertia
        self.GJ = self.G * 2.0 * inertia                  # J = 2I
        self.rhoA = self.density * self.area
        self.kappa0 = float(rod["natural_curvature"])
        self.total_length = float(rod["total_available_length"])
        self.L_gb = float(rod["gravito_bending_length"])

        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.axis = _unit(np.asarray(inj["axis"], dtype=float)[None, :])[0]
        self.v_inj = float(inj["injection_speed"])
        self.t_inj_start = float(inj["start_time"])
        self.t_inj_end = float(inj["end_time"])
        self.inlet_dir = _unit(np.asarray(
            inj["inlet_natural_curvature_direction"], dtype=float)[None, :])[0]

        self.surface_z = float(sub["surface_z"])
        self.z_contact = self.surface_z + self.radius

        # ---- initial rod state straight from the frozen topology asset
        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        n = len(self.x)
        self.v = np.zeros((n, 3))
        self.l_rest = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self.theta = np.zeros(n - 1)
        self.thetadot = np.zeros(n - 1)
        dep = np.asarray(frame0.attributes["is_initially_deposited"], dtype=float) > 0.5
        self.initially_deposited = dep.copy()
        self.stuck = dep.copy()
        self.x[self.stuck, 2] = self.z_contact
        self.arc0 = np.asarray(frame0.attributes["arc_coordinate"], dtype=float)

        # ---- injection bookkeeping
        self.L0 = float(self.l_rest[0])
        self.l_split = 0.014
        self.l_seed = 0.002
        self.dx_nom = 0.012
        self.deployed0 = float(self.l_rest.sum())
        self.deployed = self.deployed0

        # ---- time stepping (explicit, quasi-static regularisation of inertia)
        self.dt = 4.0e-4
        self.time = 0.0
        self.inertia_scale = 200.0                        # twist inertia scaling
        self.tau_lin = 0.02                               # translational drag time
        self.tau_rot = 0.01                               # rotational drag time
        self._update_mass()
        self._laplace_c = 0.05 * float(self.mass.min()) / self.dt

        # ---- frozen reporting scales
        self.reference_force = self.rhoA * abs(self.gravity[2]) * self.total_length
        self.reference_torque = max(self.reference_force * self.L_gb, 1e-12)
        self.constraint_scale = float(np.mean(self.l_rest))
        self._steps = 0
        self._prev_sample = (self.x.copy(), self.theta.copy(), 0.0)
        self._frame0_position = self.x.copy()

    def _update_mass(self) -> None:
        l = self.l_rest
        n = len(self.x)
        lump = np.empty(n)
        lump[0] = 0.5 * l[0]
        lump[-1] = 0.5 * l[-1]
        lump[1:-1] = 0.5 * (l[:-1] + l[1:])
        self.lump_len = lump
        self.mass = self.rhoA * lump
        self.inertia_t = self.inertia_scale * self.density * (
            np.pi * self.radius ** 4 / 2.0) * l

    # ------------------------------------------------------- material frames
    def _init_frames(self) -> None:
        e = np.diff(self.x, axis=0)
        t = _unit(e)
        d1 = np.zeros_like(t)
        # seed the first edge director with the frozen inlet curvature direction
        u = self.inlet_dir - np.dot(self.inlet_dir, t[0]) * t[0]
        if np.linalg.norm(u) < 1e-8:
            u = np.array([1.0, 0.0, 0.0]) - t[0][0] * t[0]
        d1[0] = _unit(u[None, :])[0]
        for i in range(1, len(t)):
            d1[i] = _parallel_transport(d1[i - 1][None, :], t[i - 1][None, :],
                                        t[i][None, :])[0]
        self.d1 = d1
        self.t_edge = t

    def _frames(self):
        e = np.diff(self.x, axis=0)
        L = np.linalg.norm(e, axis=1)
        t = e / np.maximum(L, _EPS)[:, None]
        # time-parallel transport of the reference directors
        d1 = _parallel_transport(self.d1, self.t_edge, t)
        self.d1 = d1
        self.t_edge = t
        d2 = np.cross(t, d1)
        c = np.cos(self.theta)[:, None]
        s = np.sin(self.theta)[:, None]
        m1 = c * d1 + s * d2
        m2 = -s * d1 + c * d2
        return e, L, t, d1, d2, m1, m2

    def _ref_twist(self, t, d1):
        p = _parallel_transport(d1[:-1], t[:-1], t[1:])
        cs = np.einsum("ij,ij->i", p, d1[1:])
        sn = np.einsum("ij,ij->i", np.cross(p, d1[1:]), t[1:])
        return np.arctan2(sn, cs)
