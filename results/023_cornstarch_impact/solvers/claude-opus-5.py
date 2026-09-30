"""Shear-thickening viscoplastic deformable-solid solver (cornstarch impact).

Physical model
--------------
The subject is a closed triangle mesh whose surface samples carry the mass.
Internal mechanics are an explicitly integrated elastoplastic network:

* axial edge springs with plastically evolving rest lengths (deviatoric part),
* an enclosed-volume pressure term (near-incompressible bulk part),
* an umbrella/bending term with a plastically relaxing rest offset,
* gravity plus a penalty ground contact with Coulomb friction whose normal
  damping ratio reproduces the frozen coefficient of restitution.

Plastic flow uses a rate dependent (shear-thickening) Bingham-type law::

    eps_y(edot) = eps_y0 * (1 + (|edot| / edot_ref) ** m)
    gammadot    = max(0, |eps| - eps_y(edot)) / tau_p
    dL0/dt      = L0 * gammadot * sign(eps)

During the high-rate impact the apparent yield strain is large, the network
responds elastically and the blob rebounds; once stress and strain rate decay
the yield strain collapses, the network flows and the blob slumps into a puddle
while the enclosed volume (and therefore mass) is conserved.

Integration is a symplectic (semi-implicit) Euler scheme with a fixed maximum
substep, so trajectories are deterministic and finite at every sample time.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_DT_MAX = 2.0e-4


def _scatter3(index, values, n):
    """Accumulate ``[M,3]`` values into an ``[n,3]`` array at rows ``index``."""
    out = np.empty((n, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.bincount(index, weights=values[:, c], minlength=n)
    return out


class BenchmarkSolver:
    """Explicit elastoplastic solver for the frozen deformable scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg

        verts, faces = self._load_mesh(scene, cfg)
        self.n = int(verts.shape[0])
        self.faces = faces
        self.x0 = verts.copy()          # material (reference) coordinates
        self.x = verts.copy()
        self.ground_z = float(cfg.get("ground_z", 0.0))

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n, 1)).astype(float)

        total_mass = float(cfg.get("subject_mass", 1.0))
        self.mass = np.full(self.n, total_mass / max(self.n, 1), dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.total_mass = total_mass
        self.contact_len = float(cfg.get("contact_length_scale", 0.04))

        # consistent outward face winding
        if self._mesh_volume(self.x0) < 0.0:
            self.faces = self.faces[:, ::-1].copy()
        self.V0 = abs(self._mesh_volume(self.x0))
        if not np.isfinite(self.V0) or self.V0 <= 1e-12:
            self.V0 = 1e-6

        self._build_topology()
        self._build_material_frames()
        self._set_parameters()

        self.time = 0.0
        self._frames_emitted = 0
        self._nan_events = 0
        self._last_good = (self.x.copy(), self.v.copy())

    # ------------------------------------------------------------- mesh input
    def _load_mesh(self, scene, cfg):
        verts = getattr(scene, "vertices", None)
        faces = getattr(scene, "faces", None)
        if verts is None or faces is None:
            asset = cfg.get("subject_asset", "assets/subject.npz")
            root = None
            try:
                root = Path(inspect.getfile(type(scene))).resolve().parent
            except (TypeError, OSError):
                root = Path.cwd()
            path = Path(asset)
            if not path.is_absolute():
                cand = root / path
                path = cand if cand.exists() else Path.cwd() / path
            with np.load(path) as data:
                if verts is None:
                    verts = data["vertices"]
                if faces is None:
                    faces = data["faces"]
        return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

    def _build_topology(self):
        f = self.faces
        pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        pairs = np.sort(pairs, axis=1)
        edges = np.unique(pairs, axis=0)
        self.e_i = edges[:, 0].astype(np.int64)
        self.e_j = edges[:, 1].astype(np.int64)
        self.n_edges = int(edges.shape[0])
        d0 = self.x0[self.e_j] - self.x0[self.e_i]
        self.L_init = np.linalg.norm(d0, axis=1)
        self.L_init = np.maximum(self.L_init, 1e-9)
        self.L0 = self.L_init.copy()
        self.mean_edge = float(np.mean(self.L_init))

        # 1-ring adjacency (both directions) for the umbrella operator and F fit
        self.a_i = np.concatenate([self.e_i, self.e_j])
        self.a_j = np.concatenate([self.e_j, self.e_i])
        deg = np.bincount(self.a_i, minlength=self.n).astype(float)
        self.deg = np.maximum(deg, 1.0)
        # rest umbrella offset (plastically relaxing)
        mean_nb0 = _scatter3(self.a_i, self.x0[self.a_j], self.n) / self.deg[:, None]
        self.u_rest = self.x0 - mean_nb0

    def _build_material_frames(self):
        """Precompute regularized least-squares operators for F per sample."""
        r0 = self.x0[self.a_j] - self.x0[self.a_i]           # [2E,3]
        w = 1.0 / np.maximum(np.linalg.norm(r0, axis=1) ** 2, 1e-12)
        self.f_w = w
        self.f_r0 = r0
        A = np.zeros((self.n, 3, 3), dtype=float)
        wr = r0 * w[:, None]
        for a in range(3):
            for b in range(3):
                A[:, a, b] = np.bincount(self.a_i, weights=wr[:, a] * r0[:, b],
                                         minlength=self.n)
        evals, evecs = np.linalg.eigh(A)
        thresh = np.maximum(evals[:, -1:], 1e-12) * 0.05
        inv = np.where(evals > thresh, 1.0 / np.maximum(evals, 1e-30), 0.0)
        keep = (evals > thresh).astype(float)
        self.A_pinv = np.einsum('nab,nb,ncb->nac', evecs, inv, evecs)
        # tangent-space projector in material coordinates
        self.P0 = np.einsum('nab,nb,ncb->nac', evecs, keep, evecs)
        self.nrm0 = self._vertex_normals(self.x0)

    def _surface_area(self, x):
        f = self.faces
        cr = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
        return float(0.5 * np.linalg.norm(cr, axis=1).sum())

    def _set_parameters(self):
        cfg = self.cfg
        k_cfg = float(cfg.get("stiffness", 16.0))
        d_cfg = float(cfg.get("damping", 1.0))
        rest = float(np.clip(float(cfg.get("restitution", 0.5)), 0.0, 0.99))
        self.mu = max(0.0, float(cfg.get("friction", 0.5)))

        self.k_s = 25.0 * k_cfg                 # axial spring stiffness [N/m]
        self.k_b = 0.1 * self.k_s               # umbrella/bending stiffness
        self.k_p = 1200.0 * k_cfg               # volumetric modulus [Pa]

        m = float(np.mean(self.mass))
        A0 = max(self._surface_area(self.x0), 1e-9)
        self.A0 = A0

        # light deviatoric dissipation; volumetric dashpot set from a modal ratio
        self.c_e = 2.0 * (0.04 * d_cfg) * np.sqrt(max(self.k_s, 1e-9) * m)
        zeta_v = 0.15 * d_cfg
        self.c_pv = 2.0 * zeta_v * np.sqrt(self.V0 * self.k_p * self.total_mass) / A0

        # ground penalty; normal dashpot reproduces the frozen restitution
        self.k_c = 3000.0
        ln_e = np.log(max(rest, 1e-3))
        zeta_c = -ln_e / np.sqrt(np.pi ** 2 + ln_e ** 2)
        self.c_c = 2.0 * zeta_c * np.sqrt(self.k_c * m)

        # ---- shear-thickening viscoplasticity ----------------------------
        # Characteristic scales of the frozen scene.
        g = float(np.linalg.norm(self.gravity)) or 9.81
        rc = self.x0 - self.x0.mean(axis=0)
        R_char = max(float(np.mean(np.linalg.norm(rc, axis=1))), 1e-6)
        rho = self.total_mass / self.V0
        E_eff = self.k_s / max(self.mean_edge, 1e-9)          # network modulus [Pa]
        yt = max(float(cfg.get("yield_time", 1.0)), 1e-3)

        # Yield strain fixes the arrested puddle thickness: sigma_y ~ rho g h.
        h_target = 0.5 * R_char
        self.eps_y0 = float(np.clip(rho * g * h_target / E_eff, 1e-4, 0.05))
        # Relaxation time set so the gravity-driven slump timescale equals
        # the frozen yield_time: t_slump ~ (E_eff tau_p) / (rho g R).
        self.tau_p = float(np.clip(yt * rho * g * R_char / E_eff, 5e-4, 0.05))
        self.tau_b = max(0.05, yt)

        # Jamming state: thickening reference rate chosen so the first impact
        # stays below yield (elastic rebound) with a safety factor of three.
        h_fall = max(float(np.min(self.x0[:, 2])) - self.ground_z, 0.0)
        vz0 = abs(float(self.v[0, 2])) if self.n else 0.0
        v_imp = float(np.sqrt(vz0 * vz0 + 2.0 * g * h_fall)) + 1e-6
        edot_imp = v_imp / max(2.0 * R_char, 1e-6)
        sumL2 = float(np.sum(self.L_init ** 2))
        eps_imp = v_imp * np.sqrt(self.total_mass / max(self.k_s * sumL2, 1e-12))
        self.thicken_m = 2.0
        need = max((3.0 * eps_imp / self.eps_y0 - 1.0) / 0.4, 1.0)
        self.edot_ref = float(np.clip(edot_imp / need ** (1.0 / self.thicken_m),
                                      0.02, 5.0))
        self.tau_j = 0.06                       # jamming memory [s]
        self.phi = np.zeros(self.n_edges, dtype=float)
        self.eps_imp = float(eps_imp)

        self.v_max = 60.0
        self.L0_lo = 0.35 * self.L_init
        self.L0_hi = 3.0 * self.L_init


    # ------------------------------------------------------------- geometry
    def _mesh_volume(self, x):
        f = self.faces
        c = x.mean(axis=0)
        a = x[f[:, 0]] - c
        b = x[f[:, 1]] - c
        d = x[f[:, 2]] - c
        return float(np.einsum('ij,ij->i', a, np.cross(b, d)).sum() / 6.0)

    def _volume_and_gradient(self, x):
        """Enclosed volume and dV/dx (also proportional to vertex normals)."""
        f = self.faces
        c = x.mean(axis=0)
        a = x[f[:, 0]] - c
        b = x[f[:, 1]] - c
        d = x[f[:, 2]] - c
        vol = float(np.einsum('ij,ij->i', a, np.cross(b, d)).sum() / 6.0)
        g0 = np.cross(b, d) / 6.0
        g1 = np.cross(d, a) / 6.0
        g2 = np.cross(a, b) / 6.0
        idx = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        grad = _scatter3(idx, np.concatenate([g0, g1, g2], axis=0), self.n)
        return vol, grad

    def _vertex_normals(self, x):
        _, grad = self._volume_and_gradient(x)
        nrm = np.linalg.norm(grad, axis=1)
        bad = nrm < 1e-14
        out = grad / np.maximum(nrm, 1e-14)[:, None]
        if bad.any():
            radial = x[bad] - x.mean(axis=0)
            rn = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
            out[bad] = radial / rn[:, None]
        return out

    # --------------------------------------------------------------- dynamics
    def _step(self, dt):
        x, v = self.x, self.v

        force = self.mass[:, None] * self.gravity[None, :]

        # ---- axial springs with plastic rest lengths -----------------------
        d = x[self.e_j] - x[self.e_i]
        L = np.linalg.norm(d, axis=1)
        Ls = np.maximum(L, 1e-9)
        dirn = d / Ls[:, None]
        eps = (L - self.L0) / self.L0
        dv = v[self.e_j] - v[self.e_i]
        rate = np.einsum('ij,ij->i', dv, dirn)
        edot = rate / self.L0
        f_ax = (self.k_s * self.L0 * eps + self.c_e * rate)[:, None] * dirn
        force += _scatter3(self.e_i, f_ax, self.n)
        force -= _scatter3(self.e_j, f_ax, self.n)

        # ---- enclosed-volume pressure (near incompressible) ---------------
        vol, grad = self._volume_and_gradient(x)
        voldot = float(np.einsum('ij,ij->', v, grad))
        press = self.k_p * (self.V0 - vol) / self.V0 - self.c_pv * voldot / self.V0
        press = float(np.clip(press, -20.0 * self.k_p, 20.0 * self.k_p))
        force += press * grad

        # ---- umbrella (bending) term with relaxing rest offset ------------
        mean_nb = _scatter3(self.a_i, x[self.a_j], self.n) / self.deg[:, None]
        u = x - mean_nb
        fb = -self.k_b * (u - self.u_rest)
        force += fb
        share = (fb / self.deg[:, None])[self.a_i]
        force -= _scatter3(self.a_j, share, self.n)

        # ---- ground contact: penalty normal + capped Coulomb friction -----
        pen = self.ground_z - x[:, 2]
        active = pen > 0.0
        fn = np.zeros(self.n, dtype=float)
        if active.any():
            fn_a = self.k_c * pen[active] - self.c_c * v[active, 2]
            fn_a = np.maximum(fn_a, 0.0)
            fn[active] = fn_a
            force[active, 2] += fn_a

            vt = v[active].copy()
            vt[:, 2] = 0.0
            sp = np.linalg.norm(vt, axis=1)
            moving = sp > 1e-9
            if moving.any():
                idx = np.flatnonzero(active)[moving]
                tdir = vt[moving] / sp[moving][:, None]
                f_coul = self.mu * fn_a[moving]
                f_stop = self.mass[idx] * sp[moving] / max(dt, 1e-12)
                mag = np.minimum(f_coul, f_stop)
                force[idx] -= mag[:, None] * tdir

        # ---- symplectic Euler --------------------------------------------
        v += dt * force * self.inv_mass[:, None]
        np.clip(v, -self.v_max, self.v_max, out=v)
        x += dt * v

        # anti-tunneling safety clamp
        z_floor = self.ground_z - 4.0 * self.contact_len
        deep = x[:, 2] < z_floor
        if deep.any():
            x[deep, 2] = z_floor
            vz = v[deep, 2]
            v[deep, 2] = np.where(vz < 0.0, 0.0, vz)

        # ---- shear-thickening jamming state -------------------------------
        # phi builds up with strain rate and decays with memory tau_j, so the
        # rate-stiffened (jammed) yield surface survives the instant of peak
        # compression where the axial strain rate momentarily vanishes.
        phi_t = np.minimum((np.abs(edot) / self.edot_ref) ** self.thicken_m, 1e6)
        self.phi += (dt / self.tau_j) * (phi_t - self.phi)
        np.clip(self.phi, 0.0, 1e6, out=self.phi)

        # ---- plastic flow (Bingham-type return mapping) -------------------
        eps_y = self.eps_y0 * (1.0 + self.phi)
        excess = np.maximum(np.abs(eps) - eps_y, 0.0)
        gdot = np.minimum(excess / self.tau_p, 0.2 / max(dt, 1e-12))
        self.L0 *= 1.0 + dt * gdot * np.sign(eps)
        np.clip(self.L0, self.L0_lo, self.L0_hi, out=self.L0)

        # bending memory relaxes with the local plastic rate
        gnode = np.bincount(self.a_i, weights=np.concatenate([gdot, gdot]),
                            minlength=self.n) / self.deg
        alpha = np.clip(dt * (gnode + 1.0 / self.tau_b), 0.0, 0.5)
        self.u_rest += alpha[:, None] * (u - self.u_rest)

    def _integrate_to(self, target):
        remaining = target - self.time
        if remaining <= 0.0:
            return
        nsub = max(1, int(np.ceil(remaining / _DT_MAX)))
        dt = remaining / nsub
        for _ in range(nsub):
            self._step(dt)
            if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
                self._nan_events += 1
                self.x[:], self.v[:] = self._last_good[0], 0.0 * self._last_good[1]
                break
        self.time = target
        if np.isfinite(self.x).all() and np.isfinite(self.v).all():
            self._last_good = (self.x.copy(), self.v.copy())


    # -------------------------------------------------- deformation gradient
    def _deformation_gradient(self):
        """Per-sample F from a regularized 1-ring fit, closed by incompressibility."""
        x = self.x
        dx = x[self.a_j] - x[self.a_i]
        wdx = dx * self.f_w[:, None]
        B = np.empty((self.n, 3, 3), dtype=float)
        for a in range(3):
            for b in range(3):
                B[:, a, b] = np.bincount(self.a_i,
                                         weights=wdx[:, a] * self.f_r0[:, b],
                                         minlength=self.n)
        F_tan = np.einsum('nab,nbc->nac', B, self.A_pinv)
        F_tan = np.einsum('nab,nbc->nac', F_tan, self.P0)

        nrm = self._vertex_normals(x)
        outer = np.einsum('na,nb->nab', nrm, self.nrm0)
        d1 = np.linalg.det(F_tan + outer)
        vol = abs(self._mesh_volume(x))
        J = float(np.clip(vol / self.V0, 1e-3, 1e3))
        s = np.where(np.abs(d1) > 1e-9, J / np.where(np.abs(d1) > 1e-9, d1, 1.0), 1.0)
        s = np.clip(s, -50.0, 50.0)
        F = F_tan + s[:, None, None] * outer
        F = np.clip(np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        eye = np.tile(np.eye(3), (self.n, 1, 1))
        bad = ~np.isfinite(F).all(axis=(1, 2))
        if bad.any():
            F[bad] = eye[bad]
        return F

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        t = max(0.0, float(time))
        self._integrate_to(t)

        pos = np.nan_to_num(np.asarray(self.x, dtype=float), nan=0.0,
                            posinf=1e3, neginf=-1e3)
        vel = np.nan_to_num(np.asarray(self.v, dtype=float), nan=0.0,
                            posinf=self.v_max, neginf=-self.v_max)
        gap = pos[:, 2] - self.ground_z
        frame = StateFrame(
            time=t,
            position=pos.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": np.ascontiguousarray(gap, dtype=float),
            },
            scalars={"contact_length_scale": float(self.contact_len)},
        )
        frame.validate()
        self._frames_emitted += 1
        return frame

    def diagnostics(self) -> dict[str, object]:
        vol = abs(self._mesh_volume(self.x)) if hasattr(self, "x") else 0.0
        return {
            "method": "explicit elastoplastic surface network (springs + volume pressure)",
            "integrator": "symplectic Euler",
            "max_substep": _DT_MAX,
            "n_samples": int(getattr(self, "n", 0)),
            "n_edges": int(getattr(self, "n_edges", 0)),
            "frames_emitted": int(self._frames_emitted),
            "nan_recoveries": int(self._nan_events),
            "volume_ratio": float(vol / self.V0) if getattr(self, "V0", 0.0) else 0.0,
            "mean_rest_length_ratio": float(np.mean(self.L0 / self.L_init))
            if hasattr(self, "L0") else 1.0,
            "yield_strain": float(getattr(self, "eps_y0", 0.0)),
            "plastic_relaxation_time": float(getattr(self, "tau_p", 0.0)),
            "thickening_reference_rate": float(getattr(self, "edot_ref", 0.0)),
            "mean_jamming_state": float(np.mean(self.phi))
            if hasattr(self, "phi") else 0.0,
            "min_height": float(np.min(self.x[:, 2])) if hasattr(self, "x") else 0.0,
            "max_height": float(np.max(self.x[:, 2])) if hasattr(self, "x") else 0.0,
        }
