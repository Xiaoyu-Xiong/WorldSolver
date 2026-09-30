"""Twist localization in a V-shaped anisotropic rod.

Damped geometrically-exact (Cosserat) rod dynamics on a discrete chain.

State
-----
* vertex positions ``x[i]`` (both endpoints position-clamped),
* orthonormal material frames ``R[i]`` (columns = width / thickness / tangent
  directors of the flattened cross-section),
* translational velocities and angular velocities.

Elastic energy (reference = untwisted V-shaped natural configuration)
--------------------------------------------------------------------
For every edge the elastic relative rotation between the adjacent material
frames is evaluated in material coordinates,

    Lambda_i = (R_i^T R_{i+1}) (R_i^0^T R_{i+1}^0)^T ,   sigma_i = log(Lambda_i),

which yields bending (sigma_1, sigma_2) and twist (sigma_3) measures that
vanish in the reference shape.  With the anisotropic bending stiffnesses
``B1 = E w t^3 / 12`` (edgewise), ``B2 = E t w^3 / 12`` (flatwise) and the
torsional stiffness ``GJ`` of the rectangular cross-section,

    E_bend+twist = 1/2 sum_i (B1 sigma_1^2 + B2 sigma_2^2 + GJ sigma_3^2) / l_i^0 .

A stretch term ``1/2 EA l (l/l0 - 1)^2`` keeps the rod inextensible and a
small shear regularization ties the frame directors to the edge tangents.
The schedule rotates the driven clamp about the (fixed) endpoint tangent
while the other end is fully clamped; heavy modal damping makes the sampled
states track the equilibrated twist increments of the source experiment.
Gradients of the total energy are obtained by exact finite differencing of
the (analytic, smooth) energy over all nodal degrees of freedom.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# rotation algebra (vectorised over a leading batch axis)
# ---------------------------------------------------------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    shape = v.shape[:-1] + (3, 3)
    out = np.zeros(shape)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def exp_rotations(vec: np.ndarray) -> np.ndarray:
    """Rotation matrices from rotation vectors (Rodrigues, batched)."""
    vec = np.asarray(vec, dtype=float)
    theta = np.linalg.norm(vec, axis=-1, keepdims=True)
    tiny = theta < 1e-8
    safe = np.where(tiny, 1.0, theta)
    axis = vec / safe
    K = _skew(axis)
    th2 = theta * theta
    a = np.where(tiny, 1.0 - th2 / 6.0, np.sin(theta) / safe)
    b = np.where(tiny, 0.5 - th2 / 24.0, (1.0 - np.cos(theta)) / (safe * safe))
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def log_rotations(R: np.ndarray) -> np.ndarray:
    """Rotation vectors from rotation matrices (batched, robust)."""
    w = 0.5 * np.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        axis=-1,
    )
    c = 0.5 * (np.einsum("...ii->...", R) - 1.0)
    s = np.linalg.norm(w, axis=-1)
    small = s < 1e-9
    theta = np.arctan2(s, np.clip(c, -1.0, 1.0))
    scale = np.where(small, 1.0, theta / np.where(small, 1.0, s))
    vec = w * scale[..., None]
    near_pi = small & (c < -0.9)
    if np.any(near_pi):
        block = R[near_pi]
        diag = np.clip(0.5 * (np.diagonal(block, axis1=1, axis2=2) + 1.0), 0.0, None)
        axis = np.sqrt(diag)
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-300)
        vec[near_pi] = (math.pi - 1e-6) * axis
    return vec


def quaternions_from_rotations(R: np.ndarray) -> np.ndarray:
    """xyzw quaternions of rotation matrices (batched)."""
    half = 0.5 * log_rotations(R)
    n = np.linalg.norm(half, axis=-1, keepdims=True)
    tiny = n < 1e-12
    safe = np.where(tiny, 1.0, n)
    qv = np.where(tiny, half, half * (np.sin(safe) / safe))
    qw = np.cos(n[..., 0])
    return np.concatenate([qv, qw[..., None]], axis=-1)


class BenchmarkSolver:
    """Damped Cosserat-rod solver for the V-shaped anisotropic rod scene."""

    def __init__(self) -> None:
        self._ready = False

    # -- setup --------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        rod = cfg["rod"]
        self.n = int(rod["vertex_count"])
        self.apex = int(rod["apex_vertex"])
        self.width = float(rod["cross_section_width"])
        self.thickness = float(rod["cross_section_thickness"])
        density = float(rod["density"])

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame0.mass, dtype=float)

        # --- material parameters (unstiffened engineering constants) -------
        E = 1.0e6
        nu = 0.3
        G = E / (2.0 * (1.0 + nu))
        w, th = self.width, self.thickness
        self.B1 = E * w * th**3 / 12.0        # edgewise bending stiffness
        self.B2 = E * th * w**3 / 12.0        # flatwise bending stiffness (stiff)
        jt = w * th**3 / 3.0 * (1.0 - 0.63 * th / w)
        self.GJ = G * jt                       # torsional stiffness
        self.EA = E * w * th
        self.k_shear = 2.0                     # shear regularization (N/m)
        self.damping = 40.0                    # modal damping rate (1/s)

        # --- reference (untwisted V) geometry ------------------------------
        edges = self.x[1:] - self.x[:-1]
        self.l0 = np.linalg.norm(edges, axis=1)
        edge_dirs = edges / self.l0[:, None]
        tangents = np.zeros_like(self.x)
        tangents[0] = edge_dirs[0]
        tangents[-1] = edge_dirs[-1]
        tangents[1:-1] = edge_dirs[:-1] + edge_dirs[1:]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # Material frames: width director transported from the driven end,
        # staying inside the x-z centreline plane (untwisted marking plane).
        d1 = np.zeros_like(self.x)
        ex = np.array([1.0, 0.0, 0.0])
        u = ex - np.dot(ex, tangents[0]) * tangents[0]
        d1[0] = u / np.linalg.norm(u)
        for i in range(1, self.n):
            u = d1[i - 1] - np.dot(d1[i - 1], tangents[i]) * tangents[i]
            nu_ = np.linalg.norm(u)
            d1[i] = d1[i - 1] if nu_ < 1e-9 else u / nu_
        d2 = np.cross(tangents, d1)
        self.R0 = np.stack((d1, d2, tangents), axis=-1)   # columns = directors
        self.R = self.R0.copy()
        self.omega = np.zeros_like(self.x)

        self.C0 = np.einsum("nij,nkj->nik", self.R0[:-1], self.R0[1:])  # R_i^T R_{i+1}
        self.C0T = np.transpose(self.C0, (0, 2, 1))
        d3 = self.R0[:, :, 2]
        self.gamma0 = 0.5 * (d3[:-1] + d3[1:]) - edge_dirs

        # --- lumped inertia -------------------------------------------------
        lv = np.empty(self.n)
        lv[0] = 0.5 * self.l0[0]
        lv[-1] = 0.5 * self.l0[-1]
        lv[1:-1] = 0.5 * (self.l0[:-1] + self.l0[1:])
        i1 = self.mass * (th**2 + lv**2) / 12.0
        i2 = self.mass * (w**2 + lv**2) / 12.0
        i3 = self.mass * (w**2 + th**2) / 12.0
        self.Jmat = np.zeros((self.n, 3, 3))
        self.Jmat[:, 0, 0] = i1
        self.Jmat[:, 1, 1] = i2
        self.Jmat[:, 2, 2] = i3

        # --- boundary conditions --------------------------------------------
        self.driven = 0
        self.fixed = self.n - 1
        self.free = np.arange(1, self.n - 1)
        self.x_driven = self.x[self.driven].copy()
        self.x_fixed = self.x[self.fixed].copy()
        self.t_axis = edge_dirs[0].copy()      # driven clamp rotation axis
        sched = np.asarray(
            cfg["boundary_conditions"]["driven_end"]["rotation_schedule"], dtype=float
        )
        order = np.argsort(sched[:, 0])
        self.sched_t = sched[order, 0]
        self.sched_a = sched[order, 1]

        # --- stable internal timestep ---------------------------------------
        jmin = float(np.min(i1 + i2 + i3) / 3.0) + 1e-12
        mmin = float(np.min(self.mass)) + 1e-12
        lmin = float(np.min(self.l0))
        k_rot = 2.0 * max(self.B1, self.B2, self.GJ) / lmin + self.k_shear * lmin
        k_trn = 2.0 * self.EA / lmin
        omega = max(math.sqrt(k_rot / jmin), math.sqrt(k_trn / mmin))
        self.dt_max = min(4.0e-4, 0.3 * 2.0 / omega)
        self.dt_max = max(self.dt_max, 1.0e-5)

        # --- static per-frame attributes ------------------------------------
        self.attr_arc = np.array(frame0.attributes["arc_length"], dtype=float)
        self.attr_w = np.array(frame0.attributes["cross_section_width"], dtype=float)
        self.attr_t = np.array(frame0.attributes["cross_section_thickness"], dtype=float)
        self.attr_apex = np.array(frame0.attributes["is_apex"], dtype=float)
        self.attr_driven = np.array(frame0.attributes["is_driven_end"], dtype=float)
        self.attr_fixed = np.array(frame0.attributes["is_fixed_end"], dtype=float)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        self.t_cur = 0.0
        self.alpha_cur = 0.0
        self.steps = 0
        self.resets = 0
        self._prev_angle = 0.0
        self._last_energy = 0.0
        self._ready = True

    # -- prescribed rotation -------------------------------------------------
    def _alpha(self, t: float) -> float:
        ts, vs = self.sched_t, self.sched_a
        if t <= ts[0]:
            return float(vs[0])
        if t >= ts[-1]:
            return float(vs[-1])
        j = int(np.searchsorted(ts, t, side="right")) - 1
        f = (t - ts[j]) / (ts[j + 1] - ts[j])
        return float(vs[j] + f * (vs[j + 1] - vs[j]))

    # -- energy ---------------------------------------------------------------
    def _energy(self, x: np.ndarray, R: np.ndarray) -> float:
        e = x[1:] - x[:-1]
        l = np.linalg.norm(e, axis=1)
        l = np.maximum(l, 1e-12)
        that = e / l[:, None]

        C = np.einsum("nji,njk->nik", R[:-1], R[1:])       # R_i^T R_{i+1}
        Lam = C @ self.C0T
        sig = log_rotations(Lam)
        e_bt = 0.5 * float(
            np.sum((self.B1 * sig[:, 0] ** 2 + self.B2 * sig[:, 1] ** 2 + self.GJ * sig[:, 2] ** 2) / self.l0)
        )

        eps = l / self.l0 - 1.0
        e_st = 0.5 * float(np.sum(self.EA * self.l0 * eps**2))

        d3 = R[:, :, 2]
        gam = 0.5 * (d3[:-1] + d3[1:]) - that
        e_sh = 0.5 * float(np.sum(self.k_shear * self.l0 * np.sum((gam - self.gamma0) ** 2, axis=1)))
        return e_bt + e_st + e_sh

    def _gradient(self, x: np.ndarray, R: np.ndarray):
        """dE/dx and dE/d(delta) (world left-perturbation of each frame)."""
        h = 1.0e-6
        n = self.n
        gx = np.zeros((n, 3))
        gd = np.zeros((n, 3))
        e0 = self._energy(x, R)
        for i in range(n):
            for k in range(3):
                xp = x.copy()
                xp[i, k] += h
                gx[i, k] = (self._energy(xp, R) - e0) / h
        axis = np.zeros(3)
        for i in range(n):
            for k in range(3):
                axis[k] = h
                Rp = R.copy()
                Rp[i] = exp_rotations(axis) @ R[i]
                gd[i, k] = (self._energy(x, Rp) - e0) / h
                axis[k] = 0.0
        return gx, gd

    # -- inertia ---------------------------------------------------------------
    def _world_inertia(self, idx: np.ndarray) -> np.ndarray:
        R = self.R[idx]
        return np.einsum("nij,njk,nlk->nil", R, self.Jmat[idx], R)

    # -- time stepping -----------------------------------------------------------
    def _step(self, dt: float, t_new: float) -> None:
        gx, gd = self._gradient(self.x, self.R)
        self._last_energy = self._energy(self.x, self.R)
        free = self.free

        x_old = self.x.copy()
        R_old = self.R.copy()
        v_old = self.v.copy()
        w_old = self.omega.copy()

        force = -gx[free]
        acc = force / self.mass[free][:, None] - self.damping * self.v[free]
        v_new = (self.v[free] + dt * acc) / (1.0 + dt * self.damping)
        x_new = self.x[free] + dt * v_new

        Jw = self._world_inertia(free)
        torque = -gd[free]
        w_new = (self.omega[free] + dt * np.linalg.solve(Jw, torque)) / (1.0 + dt * self.damping)
        R_new = exp_rotations(w_new * dt) @ self.R[free]

        self.x[free] = x_new
        self.v[free] = v_new
        self.omega[free] = w_new
        self.R[free] = R_new

        # clamped degrees of freedom
        alpha_new = self._alpha(t_new)
        self.x[self.driven] = self.x_driven
        self.x[self.fixed] = self.x_fixed
        self.v[self.driven] = 0.0
        self.v[self.fixed] = 0.0
        self.R[self.driven] = exp_rotations(alpha_new * self.t_axis) @ self.R0[self.driven]
        self.R[self.fixed] = self.R0[self.fixed]
        alpha_rate = (alpha_new - self.alpha_cur) / dt
        self.omega[self.driven] = alpha_rate * self.t_axis
        self.omega[self.fixed] = 0.0
        self.alpha_cur = alpha_new
        self.t_cur = t_new
        self.steps += 1

        if self.steps % 25 == 0:
            self._orthonormalize()

        if not (
            np.isfinite(self.x).all()
            and np.isfinite(self.v).all()
            and np.isfinite(self.R).all()
            and np.isfinite(self.omega).all()
        ):
            self.x, self.R, self.v, self.omega = x_old, R_old, v_old, w_old
            self.v[free] = 0.0
            self.omega[free] = 0.0
            self.t_cur = t_new - dt
            self.resets += 1

    def _orthonormalize(self) -> None:
        U, _, Vt = np.linalg.svd(self.R)
        det = np.linalg.det(np.einsum("nij,njk->nik", U, Vt))
        D = np.zeros_like(self.R)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = det
        self.R = U @ D @ Vt
        self.R[self.driven] = exp_rotations(self.alpha_cur * self.t_axis) @ self.R0[self.driven]
        self.R[self.fixed] = self.R0[self.fixed]

    # -- public interface ---------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver not initialized")
        t = float(time)
        if t < self.t_cur:
            t = self.t_cur
        span = t - self.t_cur
        if span > 1e-12:
            nsteps = max(1, int(math.ceil(span / self.dt_max)))
            h = span / nsteps
            for j in range(nsteps):
                self._step(h, self.t_cur + h)
            self.t_cur = t
            # exact schedule value at the sample time
            self.R[self.driven] = exp_rotations(self._alpha(t) * self.t_axis) @ self.R0[self.driven]
            self.alpha_cur = self._alpha(t)
        return self._build_frame()

    # -- frame export -----------------------------------------------------------
    def _edge_strains(self):
        C = np.einsum("nji,njk->nik", self.R[:-1], self.R[1:])
        Lam = C @ self.C0T
        sig = log_rotations(Lam)
        return sig

    def _build_frame(self) -> StateFrame:
        x = self.x.copy()
        v = self.v.copy()
        omega = self.omega.copy()
        R = self.R.copy()
        free = self.free
        gx, gd = self._gradient(x, R)
        self._last_energy = self._energy(x, R)

        # measured driven-end rotation about the clamp axis (unwrapped)
        rel = R[self.driven] @ self.R0[self.driven].T
        vec = log_rotations(rel[None, :, :])[0]
        ang = float(np.linalg.norm(vec))
        if ang > 1e-12 and float(np.dot(vec / ang, self.t_axis)) < 0.0:
            ang = -ang
        k = round((self._prev_angle - ang) / (2.0 * math.pi))
        meas = ang + 2.0 * math.pi * k
        self._prev_angle = meas

        # cumulative material twist (elastic twist about edge tangents)
        sig = self._edge_strains()
        sig_world = np.einsum("nij,nj->ni", R[:-1], sig)
        e = x[1:] - x[:-1]
        that = e / np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
        tw = np.sum(sig_world * that, axis=1)
        material_twist = np.concatenate(([0.0], np.cumsum(tw)))

        # external-force budget (single domain: the whole rod, one material)
        Jw = self._world_inertia(free)
        f_damp = -self.damping * np.sum(self.mass[free][:, None] * v[free], axis=0)
        f_ext = gx[self.driven] + gx[self.fixed] + f_damp
        p_app = x[self.driven]
        tau_origin = (
            np.cross(x[self.driven], gx[self.driven])
            + np.cross(x[self.fixed], gx[self.fixed])
            + gd[self.driven]
            + gd[self.fixed]
            - self.damping
            * (
                np.sum(np.cross(x[free], self.mass[free][:, None] * v[free]), axis=0)
                + np.einsum("nij,nj->ni", Jw, omega[free]).sum(axis=0)
            )
        )
        couple = tau_origin - np.cross(p_app, f_ext)

        target = self._alpha(self.t_cur)
        fields = {
            "budget_external_couple": couple.reshape(1, 1, 3),
            "budget_external_force": f_ext.reshape(1, 1, 3),
            "budget_force_position": p_app.reshape(1, 1, 3),
            "constraint_target": np.array([target]),
            "constraint_value": np.array([meas]),
            "cumulative_mass_source": np.zeros(1),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "arc_length": self.attr_arc.copy(),
            "material_twist": material_twist,
            "cross_section_width": self.attr_w.copy(),
            "cross_section_thickness": self.attr_t.copy(),
            "is_apex": self.attr_apex.copy(),
            "is_driven_end": self.attr_driven.copy(),
            "is_fixed_end": self.attr_fixed.copy(),
        }
        scalars = {
            "driven_end_angle": float(meas),
            "constraint_scale": 2.0 * math.pi,
            "reference_force": 0.1,
            "reference_torque": 0.01,
        }
        frame = StateFrame(
            time=float(self.t_cur),
            position=x,
            velocity=v,
            orientation=quaternions_from_rotations(R),
            angular_velocity=omega,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "damped_cosserat_rod_fd_gradient",
            "internal_steps": int(self.steps),
            "stability_resets": int(self.resets),
            "internal_dt": float(self.dt_max),
            "final_energy_J": float(self._last_energy),
            "driven_end_angle_rad": float(self.alpha_cur),
        }
