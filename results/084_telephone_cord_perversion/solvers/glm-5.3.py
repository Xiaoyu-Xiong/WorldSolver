"""Solver for scene 084_telephone_cord_perversion.

Physical model
--------------
A discrete Cosserat rod (the "slinky ribbon") with

* an inextensible centerline (stretch stiffness + per-substep length
  projection),
* an anisotropic rectangular cross-section: bending about the width axis
  is soft (E*w*t^3/12), bending about the thickness axis is stiff
  (E*t*w^3/12), plus Saint-Venant torsion (G*w*t^3/3),
* a naturally helical rest shape given by a constant rest Darboux vector
  (kappa, 0, tau) expressed in the material frame.  The stress-free
  configuration is the right-handed circular helix of the source spring
  (radius 0.155 m, pitch 0.055 m per turn).  The scaffold's prepared
  initial state (straightened, writhe removed, material orientation
  controlled) is therefore a highly stressed configuration, exactly as
  the task requires,
* material frames carried as dynamic director pairs with an alignment
  energy tying frame director d3 to the edge tangent, so bending and
  twist of the frames couple back onto the centerline,
* kinematic endpoint grips: prescribed smoothstep separation history,
  both end material orientations held fixed in world space, axial spin
  forbidden (this is what makes the helical perversion persistent),
* point self-contact between non-adjacent vertices.

Time integration: strongly damped (overdamped / quasi-static viscous)
explicit stepping, i.e. every degree of freedom advances by
mobility * (-grad E) * dt, the small-inertia limit of damped elastic rod
dynamics.  This matches the quasi-static endpoint protocol of the source
experiment (Discrete Elastic Rods, Sec. 9.3, helical perversion).

Frame convention (derived from the natural helix)
-------------------------------------------------
Material frame per edge: d1 = strip width axis (soft bending axis),
d2 = strip thickness axis, d3 = d1 x d2 ~ edge tangent.  For the
right-handed natural helix posed with its initial tangent along +x, the
compatible material frame at the rod start is d1 = (0,0,-1) (width),
d2 = (0,1,0) (thickness), d3 = (1,0,0); the scaffold's controlled
material axis (0,1,0) is exactly this thickness axis.  With this frame
the rest Darboux vector per unit length is (kappa, 0, tau) in material
coordinates, kappa = R/(R^2+b^2), tau = b/(R^2+b^2), b = pitch/(2*pi).
"""
from __future__ import annotations

import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - standalone fallback
    from dataclasses import dataclass, field as _dcfield

    @dataclass
    class StateFrame:  # type: ignore
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: object = _dcfield(default_factory=dict)
        fields: object = _dcfield(default_factory=dict)
        scalars: object = _dcfield(default_factory=dict)

        def validate(self) -> None:  # pragma: no cover
            pass


def _unit_rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-12)


def _rotate_rows(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rotate each row vector V[i] by rotation vector W[i] (Rodrigues)."""
    theta = np.linalg.norm(W, axis=1)
    small = theta < 1e-12
    ths = np.where(small, 1.0, theta)
    k = W / ths[:, None]
    c = np.cos(ths)[:, None]
    s = np.sin(ths)[:, None]
    dot = np.sum(k * V, axis=1)[:, None]
    out = V * c + np.cross(k, V) * s + k * dot * (1.0 - c)
    out = np.where(small[:, None], V, out)
    return out


def _quat_from_frame(d1, d2, d3):
    """Quaternion (xyzw) of the rotation with columns d1, d2, d3."""
    R = np.column_stack((d1, d2, d3))
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    return q / max(np.linalg.norm(q), 1e-12)


class BenchmarkSolver:
    """Overdamped discrete Cosserat rod solver for the telephone cord."""

    def __init__(self) -> None:
        self._ready = False
        self._rollback = 0

    # ------------------------------------------------------------------ #
    # initialization
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if cfg is None and hasattr(scene, "__getitem__"):
            cfg = scene["config"]
        if cfg is None:  # pragma: no cover
            cfg = {}
        self.cfg = cfg

        # ---- initial centerline / attributes from the frozen scaffold ----
        frame0 = None
        try:
            frame0 = scene.initial_state()
        except Exception:
            frame0 = None
        rod = cfg.get("rod", {})
        n = int(rod.get("vertex_count", 161))
        total_len = float(rod.get("total_length", 3.6))
        left = np.asarray(cfg["endpoint_controls"]["left"]["initial_position"], float)
        right = np.asarray(cfg["endpoint_controls"]["right"]["initial_position"], float)
        if frame0 is not None:
            x0 = np.array(frame0.position, dtype=float)
            attrs0 = {k: np.asarray(v, dtype=float) for k, v in dict(frame0.attributes).items()}
        else:
            s = np.linspace(0.0, 1.0, n)
            x0 = left[None, :] * (1.0 - s[:, None]) + right[None, :] * s[:, None]
            amp = float(rod.get("initial_preparation", {}).get("centerline_wave_amplitude", 0.015))
            wav = int(rod.get("initial_preparation", {}).get("centerline_wave_count", 2))
            x0[:, 2] += amp * np.sin(2.0 * np.pi * wav * s) * np.sin(np.pi * s)
            attrs0 = {}
        self.n = n = len(x0)
        self.m = m = n - 1
        self.l0 = l0 = total_len / m
        self.mass = np.full(n, float(rod.get("mass", 0.085)) / n)

        # ---- natural helix geometry -> constant rest Darboux vector ----
        nat = rod.get("natural_shape", {})
        R = float(nat.get("radius", 0.155))
        pitch = float(nat.get("pitch_per_turn", 0.055))
        b = pitch / (2.0 * np.pi)
        denom = R * R + b * b
        self.kappa = R / denom              # natural curvature (1/m)
        self.torsion = b / denom            # natural torsion (1/m)
        cs = rod.get("cross_section", {})
        w = float(cs.get("width", 0.018))
        th = float(cs.get("thickness", 0.0045))
        # Bending stiffnesses.  Young's modulus is scaled down from spring
        # steel purely for numerical stability of the explicit scheme; the
        # rest geometry, anisotropy ratio and cross-section are exact.
        self.k1 = 0.1                                              # soft axis (N*m^2)
        self.k2 = self.k1 * (w / th) ** 2                              # stiff axis
        self.kt = self.k1 * 12.0 / (3.0 * 2.6)                         # torsion GJ
        # rest per-edge rotation vector in material coordinates
        self.u_bar = np.array([self.kappa * l0, 0.0, self.torsion * l0])

        # ---- numerics parameters (stability-checked overdamped stepping) ----
        self.k_s = 1.0e5            # stretch stiffness (N/m)
        self.k_a = 2.0 * self.k1 / (l0 * l0)   # frame/tangent alignment stiffness
        self.k_c = 1.5e5            # self-contact stiffness (N/m)
        self.r_c = 0.014            # contact distance (m)
        self.c_p = 1.0e-3           # centerline mobility (overdamped)
        self.c_f = 5.0              # frame rotational mobility
        self.dt = 2.0e-4            # substep
        self.dx_max = 5.0e-3        # per-substep displacement clamp
        self.dw_max = 5.0e-2        # per-substep frame rotation clamp

        # ---- material frames ----
        # width d1 = (0,0,-1), thickness d2 = (0,1,0) (see module docstring);
        # d3 = d1 x d2 = (1,0,0) matches the initial tangents.
        self.d1 = np.tile(np.array([0.0, 0.0, -1.0]), (m, 1))
        self.d2 = np.tile(np.array([0.0, 1.0, 0.0]), (m, 1))
        self.pin_d1 = self.d1[0].copy()
        self.pin_d2 = self.d2[0].copy()
        self.x = x0.copy()

        # ---- endpoint schedule ----
        ctl = cfg.get("endpoint_controls", {})
        self.p_left0 = np.asarray(ctl.get("left", {}).get("initial_position", left), float)
        self.p_left1 = np.asarray(ctl.get("left", {}).get("final_position", left), float)
        self.p_right0 = np.asarray(ctl.get("right", {}).get("initial_position", right), float)
        self.p_right1 = np.asarray(ctl.get("right", {}).get("final_position", right), float)
        motion = ctl.get("motion", {})
        self.t0 = float(motion.get("move_start", 0.8))
        self.t1 = float(motion.get("move_end", 6.4))

        # ---- masks and contact pairs ----
        self.free = np.ones(n, dtype=bool)
        self.free[0] = False
        self.free[-1] = False
        self.free_f = np.ones(m, dtype=bool)
        self.free_f[0] = False
        self.free_f[-1] = False
        skip = 3
        self.ci, self.cj = np.triu_indices(n, k=skip)

        # ---- bookkeeping ----
        self.t_now = 0.0
        self._prev = None  # (t, x, d1, d2) of previous sample
        self.attrs0 = attrs0
        self.seed = int(seed)
        self._ready = True

    # ------------------------------------------------------------------ #
    # physics: energy gradients (forces on vertices, torques on frames)
    # ------------------------------------------------------------------ #
    def _endpoints(self, t: float):
        u = (t - self.t0) / max(self.t1 - self.t0, 1e-12)
        u = min(max(u, 0.0), 1.0)
        u = u * u * (3.0 - 2.0 * u)
        pL = (1.0 - u) * self.p_left0 + u * self.p_left1
        pR = (1.0 - u) * self.p_right0 + u * self.p_right1
        return pL, pR

    def _forces(self, x, d1, d2):
        """Return (F on vertices [n,3], torque on frames [m,3]) = -grad E."""
        n, m = self.n, self.m
        F = np.zeros((n, 3))
        T = np.zeros((m, 3))

        e = x[1:] - x[:-1]
        l = np.linalg.norm(e, axis=1)
        ls = np.maximum(l, 1e-9)
        t3 = e / ls[:, None]

        # --- stretch: E = 1/2 k_s (l - l0)^2 ---
        Fs = (self.k_s * (self.l0 - l))[:, None] * t3
        F[1:] += Fs
        F[:-1] -= Fs

        d3 = _unit_rows(np.cross(d1, d2))

        # --- frame/tangent alignment: E = 1/2 k_a |d3 - t|^2 ---
        g = d3 - t3
        gp = g - np.sum(g * t3, axis=1)[:, None] * t3   # (I - t t^T)(d3 - t)
        Fa = (self.k_a / ls)[:, None] * gp
        F[1:] += Fa
        F[:-1] -= Fa
        T += self.k_a * np.cross(d3, t3)                # rotates d3 toward t

        # --- bending + twist between adjacent frames ---
        # pair q (q = 0..m-2) connects frames q and q+1; the relative
        # rotation vector is expressed in frame q's material coordinates.
        a1, a2, a3 = d1[:-1], d2[:-1], d3[:-1]
        b1, b2, b3 = d1[1:], d2[1:], d3[1:]
        u1 = 0.5 * (np.sum(a2 * b3, axis=1) - np.sum(a3 * b2, axis=1))
        u2 = 0.5 * (np.sum(a3 * b1, axis=1) - np.sum(a1 * b3, axis=1))
        u3 = 0.5 * (np.sum(a1 * b2, axis=1) - np.sum(a2 * b1, axis=1))
        g1 = self.k1 * (u1 - self.u_bar[0])
        g2 = self.k2 * (u2 - self.u_bar[1])
        g3 = self.kt * (u3 - self.u_bar[2])
        # GW[q] = R_q g_{q+1} expressed in world coordinates
        GW = g1[:, None] * a1 + g2[:, None] * a2 + g3[:, None] * a3
        # torque on frame j = R_j g_{j+1} - R_{j-1} g_j
        T[0] += GW[0]
        T[1:m - 1] += GW[1:m - 1] - GW[0:m - 2]
        T[m - 1] -= GW[m - 2]

        # --- point self-contact between non-adjacent vertices ---
        if self.ci.size:
            dif = x[self.ci] - x[self.cj]
            d2p = np.sum(dif * dif, axis=1)
            hit = d2p < self.r_c * self.r_c
            if np.any(hit):
                dd = np.sqrt(d2p[hit])
                nn = dif[hit] / np.maximum(dd, 1e-12)[:, None]
                fc = (self.k_c * (self.r_c - dd))[:, None] * nn
                np.add.at(F, self.ci[hit], fc)
                np.add.at(F, self.cj[hit], -fc)

        return F, T

    def _project_lengths(self, x) -> None:
        """Enforce inextensibility: edges relax toward the rest length."""
        wfree = self.free.astype(float)
        for _ in range(3):
            e = x[1:] - x[:-1]
            l = np.linalg.norm(e, axis=1)
            ls = np.maximum(l, 1e-9)
            t3 = e / ls[:, None]
            corr = l - self.l0
            wl = wfree[:-1]
            wr = wfree[1:]
            denom = np.maximum(wl + wr, 1e-12)
            dx = np.zeros_like(x)
            dx[:-1] += (corr * (wl / denom))[:, None] * t3
            dx[1:] -= (corr * (wr / denom))[:, None] * t3
            x += dx

    # ------------------------------------------------------------------ #
    # time stepping
    # ------------------------------------------------------------------ #
    def _substep(self, dt: float) -> None:
        x_prev = self.x.copy()
        d1_prev = self.d1.copy()
        d2_prev = self.d2.copy()

        F, T = self._forces(self.x, self.d1, self.d2)

        # overdamped centerline update (free interior vertices only)
        dx = self.c_p * F * dt
        nrm = np.linalg.norm(dx, axis=1)
        scale = np.minimum(1.0, self.dx_max / np.maximum(nrm, 1e-30))
        dx = dx * scale[:, None]
        dx[~self.free] = 0.0
        self.x += dx

        # overdamped frame update (interior frames only; end frames are
        # gripped: orientation preserved, axial spin forbidden)
        dw = self.c_f * T * dt
        wn = np.linalg.norm(dw, axis=1)
        wsc = np.minimum(1.0, self.dw_max / np.maximum(wn, 1e-30))
        dw = dw * wsc[:, None]
        dw[~self.free_f] = 0.0
        self.d1 = _rotate_rows(self.d1, dw)
        self.d2 = _rotate_rows(self.d2, dw)

        # kinematic endpoint grips
        t_new = self.t_now + dt
        pL, pR = self._endpoints(t_new)
        self.x[0] = pL
        self.x[-1] = pR
        self.d1[0] = self.pin_d1
        self.d2[0] = self.pin_d2
        self.d1[-1] = self.pin_d1
        self.d2[-1] = self.pin_d2

        # inextensibility projection and frame re-orthonormalization
        self._project_lengths(self.x)
        d1n = _unit_rows(self.d1)
        d3n = _unit_rows(np.cross(d1n, self.d2))
        self.d1 = d1n
        self.d2 = np.cross(d3n, d1n)

        # finite-state guard (roll back a bad step if it ever occurs)
        if not (np.isfinite(self.x).all() and np.isfinite(self.d1).all()
                and np.isfinite(self.d2).all()):
            self.x = x_prev
            self.d1 = d1_prev
            self.d2 = d2_prev
            self._rollback += 1
        self.t_now = t_new

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        t = float(time)
        if t < self.t_now:
            t = self.t_now
        guard = 0
        while self.t_now < t - 1e-9 and guard < 10_000_000:
            dt = min(self.dt, t - self.t_now)
            self._substep(dt)
            guard += 1
        return self._make_frame(t)

    def _make_frame(self, t: float):
        n, m = self.n, self.m
        x = self.x.copy()
        d1 = self.d1.copy()
        d2 = self.d2.copy()
        d3 = _unit_rows(np.cross(d1, d2))

        # sampled velocities (finite differences across sample times)
        if self._prev is not None and t > self._prev[0] + 1e-12:
            dts = t - self._prev[0]
            v = (x - self._prev[1]) / dts
            w_edge = 0.5 * (np.cross(self._prev[2], d1)
                            + np.cross(self._prev[3], d2)
                            + np.cross(_unit_rows(np.cross(self._prev[2], self._prev[3])), d3)) / dts
        else:
            v = np.zeros_like(x)
            w_edge = np.zeros((m, 3))
        av = np.zeros_like(x)
        if m >= 2:
            av[0] = w_edge[0]
            av[-1] = w_edge[-1]
            av[1:-1] = 0.5 * (w_edge[:-1] + w_edge[1:])
        else:
            av[:] = w_edge

        # per-vertex orientation quaternions from the material frames
        quat = np.zeros((n, 4))
        for i in range(n):
            j = min(i, m - 1)
            quat[i] = _quat_from_frame(d1[j], d2[j], d3[j])

        # ---- grip (external) reactions: quasi-static constraint forces ----
        F, T = self._forces(x, d1, d2)
        F_ext = -F[[0, n - 1]]                # applied by the kinematic grips
        C_ext = -T[[0, m - 1]]                # end-orientation constraint couples
        pL, pR = self._endpoints(t)
        F_pos = np.stack((pL, pR))

        fields = {
            "budget_external_force": F_ext.reshape(2, 1, 3),
            "budget_external_couple": C_ext.reshape(2, 1, 3),
            "budget_force_position": F_pos.reshape(2, 1, 3),
            "constraint_value": np.stack((x[0], x[n - 1])),
            "constraint_target": np.stack((pL, pR)),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "constraint_scale": float(3.6),
            "reference_force": float(max(self.k1 * self.kappa / self.l0, 1e-6)),
            "reference_torque": float(max(self.k1 * self.kappa, 1e-6)),
        }

        attributes = {}
        for k, val in self.attrs0.items():
            attributes[k] = np.asarray(val, dtype=float)
        attributes["mass_quantity"] = self.mass.copy()
        # controlled material axes mapped from per-edge frames onto vertices
        # (scaffold's axis 1 = strip thickness direction, initially (0,1,0))
        idx = np.clip(np.arange(n), 0, m - 1)
        attributes["material_axis_1"] = d2[idx].copy()
        attributes["material_axis_width"] = d1[idx].copy()

        frame = StateFrame(float(t), x, v, orientation=quat, angular_velocity=av,
                           mass=self.mass.copy(), attributes=attributes,
                           fields=fields, scalars=scalars)
        frame.validate()
        self._prev = (float(t), x, d1, d2)
        return frame

    def diagnostics(self) -> dict:
        out = {
            "model": "overdamped_discrete_cosserat_rod",
            "time": float(self.t_now),
            "vertex_count": int(self.n),
            "natural_curvature": float(self.kappa),
            "natural_torsion": float(self.torsion),
            "rest_darboux_per_edge": self.u_bar.tolist(),
            "rollbacks": int(self._rollback),
        }
        try:
            F, T = self._forces(self.x, self.d1, self.d2)
            out["max_vertex_force"] = float(np.max(np.linalg.norm(F, axis=1)))
            out["max_frame_torque"] = float(np.max(np.linalg.norm(T, axis=1)))
            e = self.x[1:] - self.x[:-1]
            out["max_length_error"] = float(np.max(np.abs(np.linalg.norm(e, axis=1) - self.l0)))
        except Exception:
            pass
        return out
