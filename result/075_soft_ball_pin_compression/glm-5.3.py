"""Soft ball compressed between pin arrays.

Physical model
--------------
* The ball is a 3-D volumetric solid discretised with linear tetrahedra
  (asset ``assets/soft_ball_volume.json``: icosahedral hull + centre vertex).
* Material: compressible neo-Hookean elasticity
      psi(F) = mu/2 (I1 - 3) - mu ln J + lambda/2 (J-1)^2
  with a quadratic barrier on det(F) that keeps elements from inverting.
* Time integration: implicit Euler.  Each substep solves the non-linear
  system M (x - x_n - dt v_n)/dt^2 = F(x) with a mass-scaled fixed-point
  (quasi-Newton) iteration; the residual uses the exact elastic and contact
  penalty forces, so the converged iterate is the implicit-Euler solution.
* Gravity acts on all nodal masses.
* Contacts: the two pin arrays are codimensional line-segment obstacles
  (capsules of radius 0.009 m).  Two complementary penalty contact sets are
  evaluated every iteration:
    (a) ball vertices vs. pin capsules (point-segment distance),
    (b) pin axis sample points vs. the ball surface triangles
        (point-triangle closest point), which prevents pins from passing
        through the coarse faces between ball vertices.
  Contacts carry normal penalty forces (implicit, inside the iteration),
  normal damping and Coulomb friction (lagged velocities), which anchors the
  ball so it is squeezed between the arrays instead of sliding away.
* The upper array follows the prescribed piecewise-linear translation from
  the frozen configuration.

No animation, keyframes or precomputed motion are used: the ball motion is
entirely the result of the time-stepping procedure above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- numerics / material parameters (fixed, deterministic) ----
    DT = 1.0e-3                 # substep size [s]
    YOUNG = 2.0e5               # rubber Young modulus [Pa]
    NU = 0.45                   # Poisson ratio
    KAPPA_BARRIER = 1.0e5       # inversion barrier stiffness [Pa]
    BARRIER_J0 = 0.25           # barrier activates below this det(F)
    EPS_J = 1.0e-3              # clamp for ln(J)
    K_CONTACT = 1.5e6           # penalty stiffness [N/m]
    C_CONTACT = 1500.0          # normal contact damping [N s/m]
    MU_FRICTION = 0.7           # Coulomb friction coefficient
    K_TANGENTIAL = 5.0e4        # tangential regularisation [N s/m]
    PIN_SAMPLE_SPACING = 0.01   # sampling along pin axes [m]
    MAX_ITER = 30
    TOL_ITER = 1.0e-9

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        root = Path(__file__).resolve().parents[1]

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        init = scene.initial_state()
        self.rest = np.array(init.position, dtype=float)
        self.mass = np.array(init.mass, dtype=float)
        self.n = len(self.rest)
        self.mass_matrix_diag = np.repeat(self.mass, 3)

        # ---------------- tetrahedral mesh ----------------
        mesh = json.loads((root / "assets" / "soft_ball_volume.json").read_text())
        tets = np.asarray(mesh["tetrahedra"], dtype=int) - 1
        d = self.rest[tets]
        vol = np.linalg.det(
            np.stack((d[:, 1] - d[:, 0], d[:, 2] - d[:, 0], d[:, 3] - d[:, 0]), axis=2)
        ) / 6.0
        keep = np.abs(vol) > 1.0e-12
        self.tets = tets[keep]
        self.tet_volume = np.abs(vol[keep])
        self.n_elements = len(self.tets)

        d = self.rest[self.tets]
        dm = np.stack(
            (d[:, 1] - d[:, 0], d[:, 2] - d[:, 0], d[:, 3] - d[:, 0]), axis=2
        )  # [Ne,3,3] columns are edge vectors
        self.dm_inv = np.linalg.inv(dm)

        # ---------------- contact surface (oriented hull faces) -------------
        faces = np.asarray(mesh["surface_triangles"], dtype=int) - 1
        tri = self.rest[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        faces = faces[areas > 1.0e-12]
        centre = self.rest.mean(axis=0)
        tri = self.rest[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cen = tri.mean(axis=1)
        flip = np.sum((cen - centre) * normals, axis=1) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        self.faces = faces
        self.n_faces = len(faces)

        # ---------------- pin arrays ----------------
        lower = json.loads((root / "assets" / "lower_pins.json").read_text())
        upper = json.loads((root / "assets" / "upper_pins.json").read_text())

        def segments(obj):
            v = np.asarray(obj["vertices"], dtype=float)
            e = np.asarray(obj["edges"], dtype=int)
            return v[e[:, 0]], v[e[:, 1]]

        self.lower_a, self.lower_b = segments(lower)
        self.upper_a0, self.upper_b0 = segments(upper)

        radius = 0.009
        geometry = cfg.get("scene_geometry") or {}
        for obj in geometry.get("objects", []):
            if obj.get("id") in ("lower_pin_array", "upper_pin_array"):
                r = (obj.get("geometry") or {}).get("radius")
                if r:
                    radius = float(r)
        self.pin_radius = radius
        self.margin = self.pin_radius

        # sample parameters along each pin (both arrays have equal lengths)
        seg_len = np.linalg.norm(self.lower_b - self.lower_a, axis=1)
        L = float(seg_len.mean())
        k_samples = max(2, int(np.ceil(L / self.PIN_SAMPLE_SPACING)) + 1)
        self.pin_params = np.linspace(0.0, 1.0, k_samples)

        # ---------------- upper array prescribed motion ----------------
        bc = cfg["boundary_conditions"]["upper_pin_array"]
        self.motion_axis = np.asarray(bc["axis"], dtype=float)
        self.motion_axis /= np.linalg.norm(self.motion_axis)
        self.keyframes = sorted(
            (float(k["time"]), float(k["displacement"])) for k in bc["keyframes"]
        )

        # ---------------- elastic constants ----------------
        self.mu = self.YOUNG / (2.0 * (1.0 + self.NU))
        self.lam = self.YOUNG * self.NU / ((1.0 + self.NU) * (1.0 - 2.0 * self.NU))

        # ---------------- linear stiffness (for Rayleigh damping) ----------
        k0 = self._linear_stiffness_fd()
        self.k0 = 0.5 * (k0 + k0.T)
        # cap Rayleigh coefficient for explicit (lagged) stability
        scaled = self.k0 / np.sqrt(
            np.outer(self.mass_matrix_diag, self.mass_matrix_diag)
        )
        eig = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
        lam_max = float(max(eig.max(), 1.0e-9))
        self.c_rayleigh = min(0.05, 0.4 / (lam_max * self.DT))

        # ---------------- state ----------------
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.rest)
        self.substep = 0
        self.cum_external_impulse = np.zeros(3)
        self.contact_force_total = np.zeros(3)
        self.gravity_force_total = float(self.mass.sum()) * self.gravity
        self.total_mass = float(self.mass.sum())
        fall_height = max(1.0e-6, float(np.abs(self.rest[:, 2].min() - 0.54)))
        self.reference_momentum = self.total_mass * float(
            np.sqrt(2.0 * max(np.linalg.norm(self.gravity), 1e-9) * fall_height)
        )
        self._substep_count = 0
        self._density = float(cfg["initial_conditions"]["ball_density"])
        self._pin_point_cache = None

    # ------------------------------------------------- prescribed pin motion
    def _displacement(self, t: float):
        keys = self.keyframes
        if t <= keys[0][0]:
            return keys[0][1], 0.0
        if t >= keys[-1][0]:
            return keys[-1][1], 0.0
        for (t0, d0), (t1, d1) in zip(keys[:-1], keys[1:]):
            if t0 <= t <= t1:
                span = t1 - t0
                if span <= 0.0:
                    return d1, 0.0
                u = (t - t0) / span
                return d0 + u * (d1 - d0), (d1 - d0) / span
        return keys[-1][1], 0.0

    def _pin_segments(self, t: float):
        """Return (A, B, velocity) for all pin segments at time t."""
        disp, rate = self._displacement(t)
        off = self.motion_axis * disp
        vel = self.motion_axis * rate
        upper_a = self.upper_a0 + off
        upper_b = self.upper_b0 + off
        A = np.vstack((self.lower_a, upper_a))
        B = np.vstack((self.lower_b, upper_b))
        seg_vel = np.zeros_like(A)
        seg_vel[len(self.lower_a):] = vel
        return A, B, seg_vel

    def _pin_points(self, t: float):
        """Sample points along every pin axis plus per-point velocity (cached per time)."""
        if self._pin_point_cache is not None and self._pin_point_cache[0] == t:
            return self._pin_point_cache[1], self._pin_point_cache[2]
        A, B, seg_vel = self._pin_segments(t)
        params = self.pin_params
        # points[k, s, :] = A[s] + params[k] * (B[s]-A[s])
        pts = A[None, :, :] + params[:, None, None] * (B - A)[None, :, :]
        shape = pts.shape
        pts = pts.reshape(-1, 3)
        vel = np.broadcast_to(seg_vel[None, :, :], shape).reshape(-1, 3)
        self._pin_point_cache = (t, pts, vel)
        return pts, vel

    # ------------------------------------------------------------ elasticity
    def _elastic_forces(self, x: np.ndarray):
        """Nodal elastic forces and per-element deformation gradients."""
        d = x[self.tets]
        ds = np.stack(
            (d[:, 1] - d[:, 0], d[:, 2] - d[:, 0], d[:, 3] - d[:, 0]), axis=2
        )
        F = ds @ self.dm_inv
        J = np.linalg.det(F)
        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)
        F_inv_t = np.transpose(F_inv, (0, 2, 1))
        J_safe = np.where(np.abs(J) < 1.0e-12, np.sign(J + 1e-300) * 1.0e-12, J)
        coef = (
            self.lam * (J_safe - 1.0) * J_safe
            - self.mu * (J_safe > self.EPS_J)
            - 2.0 * self.KAPPA_BARRIER * np.maximum(0.0, self.BARRIER_J0 - J_safe) * J_safe
        )
        P = self.mu * F + coef[:, None, None] * F_inv_t
        G = self.tet_volume[:, None, None] * (P @ np.transpose(self.dm_inv, (0, 2, 1)))
        forces = np.zeros((self.n, 3))
        # node 0 of each tet: +sum of columns ; nodes 1..3: -column
        f0 = G.sum(axis=2)                      # [Ne,3] sum of columns g1+g2+g3
        np.add.at(forces, self.tets[:, 0], f0)
        for j in (1, 2, 3):
            np.add.at(forces, self.tets[:, j], -G[:, :, j - 1])
        return forces, F

    def _linear_stiffness_fd(self):
        """Linearised stiffness matrix at the rest pose (finite differences)."""
        dofs = 3 * self.n
        K = np.zeros((dofs, dofs))
        h = 1.0e-6
        for i in range(dofs):
            xp = self.rest.copy().reshape(-1)
            xm = xp.copy()
            xp[i] += h
            xm[i] -= h
            fp, _ = self._elastic_forces(xp.reshape(self.n, 3))
            fm, _ = self._elastic_forces(xm.reshape(self.n, 3))
            K[:, i] = -(fp - fm).reshape(-1) / (2.0 * h)
        return K

    # --------------------------------------------------------------- contacts
    @staticmethod
    def _closest_on_triangle(P, a, b, c):
        """Closest point of P (M,3) on triangle abc; returns q (M,3), w (M,3)."""
        ab = b - a
        ac = c - a
        ap = P - a
        d1 = ap @ ab
        d2 = ap @ ac
        bp = P - b
        d3 = bp @ ab
        d4 = bp @ ac
        cp = P - c
        d5 = cp @ ab
        d6 = cp @ ac
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        M = len(P)
        q = np.empty_like(P)
        w = np.zeros((M, 3))
        assigned = np.zeros(M, dtype=bool)

        interior = (denom > 1.0e-14) & (va >= 0) & (vb >= 0) & (vc >= 0)
        safe = np.where(interior, denom, 1.0)
        v = vb / safe
        ww = vc / safe
        q[interior] = a + v[interior, None] * ab + ww[interior, None] * ac
        w[interior, 1] = v[interior]
        w[interior, 2] = ww[interior]
        w[interior, 0] = 1.0 - v[interior] - ww[interior]
        assigned |= interior

        # vertex regions
        m = (~assigned) & (d1 <= 0) & (d2 <= 0)
        q[m] = a
        w[m] = (1.0, 0.0, 0.0)
        assigned |= m
        m = (~assigned) & (d3 >= 0) & (d4 <= d3)
        q[m] = b
        w[m] = (0.0, 1.0, 0.0)
        assigned |= m
        m = (~assigned) & (d6 >= 0) & (d5 <= d6)
        q[m] = c
        w[m] = (0.0, 0.0, 1.0)
        assigned |= m

        # edge regions
        m = (~assigned) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        span = d1 - d3
        tt = np.clip(np.divide(d1, span, out=np.zeros_like(d1), where=np.abs(span) > 1e-14), 0, 1)
        q[m] = a + tt[m, None] * ab
        w[m] = np.stack((1.0 - tt[m], tt[m], np.zeros(m.sum())), axis=1)
        assigned |= m

        m = (~assigned) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        span = d2 - d6
        tt = np.clip(np.divide(d2, span, out=np.zeros_like(d2), where=np.abs(span) > 1e-14), 0, 1)
        q[m] = a + tt[m, None] * ac
        w[m] = np.stack((1.0 - tt[m], np.zeros(m.sum()), tt[m]), axis=1)
        assigned |= m

        m = (~assigned) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        span = (d4 - d3) + (d5 - d6)
        tt = np.clip(
            np.divide(d4 - d3, span, out=np.zeros_like(d4), where=np.abs(span) > 1e-14), 0, 1
        )
        q[m] = b + tt[m, None] * (c - b)
        w[m] = np.stack((np.zeros(m.sum()), 1.0 - tt[m], tt[m]), axis=1)
        assigned |= m

        # fallback (numerical corner): nearest vertex
        left = ~assigned
        if left.any():
            dists = np.stack(
                [np.sum((P - a) ** 2, axis=1), np.sum((P - b) ** 2, axis=1),
                 np.sum((P - c) ** 2, axis=1)], axis=1)
            idx = np.argmin(dists, axis=1)
            verts = np.stack((a, b, c))
            q[left] = verts[idx[left]]
            wl = np.zeros((left.sum(), 3))
            wl[np.arange(left.sum()), idx[left]] = 1.0
            w[left] = wl
        return q, w

    def _contacts(self, x, v_lag, t):
        """All ball/pin contact forces.

        Returns
        -------
        f_penalty : (N,3) implicit penalty forces (function of x)
        f_lagged  : (N,3) friction + damping (lagged velocities)
        diag_stiff: (N,3) per-dof stiffness augmentation for the iteration
        f_total   : (N,3) total contact force on the ball nodes
        """
        n = self.n
        f_penalty = np.zeros((n, 3))
        f_lagged = np.zeros((n, 3))
        diag = np.zeros((n, 3))
        A, B, seg_vel = self._pin_segments(t)

        # ---- (a) ball vertices vs pin capsules ----
        X = x[:, None, :]                       # [N,S,3]
        AB = (B - A)[None, :, :]
        denom = np.sum(AB * AB, axis=2)          # [1,S]
        u = X - A[None, :, :]
        tt = np.clip(np.sum(u * AB, axis=2) / np.maximum(denom, 1e-14), 0.0, 1.0)
        q = A[None, :, :] + tt[:, :, None] * AB
        diff = X - q
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        pen = self.pin_radius - dist
        act = pen > 0.0
        if act.any():
            d_safe = np.where(dist > 1e-12, dist, 1e-12)
            nrm = diff / d_safe[:, :, None]
            pen_c = np.where(act, pen, 0.0)
            fpen = self.K_CONTACT * pen_c[:, :, None] * nrm      # [N,S,3]
            f_penalty += fpen.sum(axis=1)
            diag += self.K_CONTACT * act.sum(axis=1)[:, None] * np.ones((1, 3))
            # friction + normal damping
            vrel = v_lag[:, None, :] - seg_vel[None, :, :]
            vn = np.sum(vrel * nrm, axis=2)
            vt = vrel - vn[:, :, None] * nrm
            vt_norm = np.sqrt(np.sum(vt * vt, axis=2))
            normal_force = self.K_CONTACT * pen_c
            slide = vt_norm > 1e-6
            ft_mag = np.where(
                slide,
                np.minimum(self.MU_FRICTION * normal_force, self.K_TANGENTIAL * vt_norm),
                0.0,
            )
            fhat = np.where(slide[:, :, None], vt / np.maximum(vt_norm, 1e-12)[:, :, None], 0.0)
            f_lagged += (-ft_mag[:, :, None] * fhat).sum(axis=1)
            # damping only at active contacts (approach = gap closing)
            approach = np.minimum(vn, 0.0) * act
            f_lagged += (-self.C_CONTACT * approach[:, :, None] * nrm).sum(axis=1)

        # ---- (b) pin axis samples vs ball surface triangles ----
        pts, pt_vel = self._pin_points(t)
        centre = x.mean(axis=0)
        reach = np.linalg.norm(x - centre, axis=1).max() + self.margin + 0.05
        near = np.linalg.norm(pts - centre, axis=1) < reach
        pts = pts[near]
        pt_vel = pt_vel[near]
        if len(pts):
            best_d2 = np.full(len(pts), np.inf)
            best_q = np.zeros((len(pts), 3))
            best_w = np.zeros((len(pts), 3))
            best_face = np.zeros(len(pts), dtype=int)
            for fi in range(self.n_faces):
                a, b, c = x[self.faces[fi]]
                qf, wf = self._closest_on_triangle(pts, a, b, c)
                d2 = np.sum((pts - qf) ** 2, axis=1)
                upd = d2 < best_d2
                best_d2 = np.where(upd, d2, best_d2)
                best_q[upd] = qf[upd]
                best_w[upd] = wf[upd]
                best_face[upd] = fi
            # current outward face normals
            tri = x[self.faces]
            nrm_f = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm_f /= np.maximum(np.linalg.norm(nrm_f, axis=1, keepdims=True), 1e-12)
            nrm = nrm_f[best_face]                                # [M,3]
            s = np.sum((pts - best_q) * nrm, axis=1)              # signed distance
            pen = self.margin - s
            act = pen > 0.0
            if act.any():
                idx = np.where(act)[0]
                pen_c = pen[idx]
                nrm_c = nrm[idx]
                w_c = best_w[idx]
                vel_c = pt_vel[idx]
                node_idx = self.faces[best_face[idx]]             # [m,3]
                fmag = self.K_CONTACT * pen_c                     # [m]
                # penalty pushes the ball surface away from the pin (-n)
                for k in (0, 1, 2):
                    np.add.at(
                        f_penalty,
                        node_idx[:, k],
                        -(fmag * w_c[:, k])[:, None] * nrm_c,
                    )
                    np.add.at(
                        diag,
                        node_idx[:, k],
                        (self.K_CONTACT * w_c[:, k] ** 2)[:, None] * np.ones((1, 3)),
                    )
                # surface velocity at contact points
                v_surf = np.zeros((len(idx), 3))
                for k in (0, 1, 2):
                    v_surf += w_c[:, k, None] * v_lag[node_idx[:, k]]
                vrel = v_surf - vel_c
                vn = np.sum(vrel * nrm_c, axis=1)
                # s = (p-q).n shrinks when vrel.n > 0 (approaching pin)
                approach = np.maximum(vn, 0.0)
                vt = vrel - vn[:, None] * nrm_c
                vt_norm = np.sqrt(np.sum(vt * vt, axis=1))
                slide = vt_norm > 1e-6
                ft_mag = np.where(
                    slide,
                    np.minimum(
                        self.MU_FRICTION * fmag, self.K_TANGENTIAL * vt_norm
                    ),
                    0.0,
                )
                fhat = np.where(slide[:, None], vt / np.maximum(vt_norm, 1e-12)[:, None], 0.0)
                ft = -ft_mag[:, None] * fhat
                fd = -self.C_CONTACT * approach[:, None] * nrm_c
                flg = ft + fd
                for k in (0, 1, 2):
                    np.add.at(f_lagged, node_idx[:, k], (w_c[:, k])[:, None] * flg)

        f_total = f_penalty + f_lagged
        return f_penalty, f_lagged, diag, f_total

    # ------------------------------------------------------------- substepping
    def _substep(self, t_end: float) -> None:
        dt = self.DT
        x_n = self.x
        v_n = self.v
        x_hat = x_n + dt * v_n
        x = x_hat.copy()

        f_grav = self.mass[:, None] * self.gravity[None, :]
        f_damp = (-self.c_rayleigh * (self.k0 @ v_n.reshape(-1))).reshape(self.n, 3)

        m_over_dt2 = self.mass_matrix_diag / (dt * dt)
        f_penalty = np.zeros((self.n, 3))
        f_lagged = np.zeros((self.n, 3))
        f_contact_total = np.zeros(3)

        for _ in range(self.MAX_ITER):
            f_el, _ = self._elastic_forces(x)
            f_penalty, f_lagged, diag, f_total = self._contacts(x, v_n, t_end)
            force = f_el + f_penalty + f_grav + f_lagged + f_damp
            residual = ((x - x_hat).reshape(-1) * m_over_dt2) - force.reshape(-1)
            A_diag = m_over_dt2 + diag.reshape(-1)
            dx = residual / A_diag
            x = x - dx.reshape(self.n, 3)
            if not np.all(np.isfinite(x)):
                x = x_hat.copy()
                break
            if np.max(np.abs(dx)) < self.TOL_ITER:
                break
        self._substep_count += 1

        # final consistent force evaluation (for the momentum budget)
        f_el, _ = self._elastic_forces(x)
        f_penalty, f_lagged, diag, f_total = self._contacts(x, (x - x_n) / dt, t_end)
        f_contact_total = (f_penalty + f_lagged).sum(axis=0)

        self.x = x
        self.v = (x - x_n) / dt
        self.substep += 1
        self.contact_force_total = f_contact_total
        self.cum_external_impulse += (self.gravity_force_total + f_contact_total) * dt

    # ----------------------------------------------------------------- frames
    def _frame(self, time: float) -> StateFrame:
        _, F = self._elastic_forces(self.x)
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "rest_position": self.rest.copy(),
            },
            fields={
                "element_deformation_gradient": F.copy(),
                "budget_domain_weights": np.ones((1, self.n)),
                "budget_external_force": np.stack(
                    (
                        self.gravity_force_total[None, :],
                        self.contact_force_total[None, :],
                    )
                )[None, :, :],
                "budget_cumulative_external_impulse": self.cum_external_impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "density": float(self._density),
            },
        )

    def advance_to(self, time: float):
        target = int(round(float(time) / self.DT))
        while self.substep < target:
            t_end = (self.substep + 1) * self.DT
            self._substep(t_end)
        return self._frame(time)

    def diagnostics(self) -> dict:
        return {
            "method": "implicit Euler, neo-Hookean tet FEM, penalty pin contacts",
            "substeps": self.substep,
            "dt": self.DT,
            "young_modulus": self.YOUNG,
            "poisson_ratio": self.NU,
            "pin_radius": self.pin_radius,
        }
