"""Frictionless spaghetti-in-a-plate solver.

Self-contained physical solver (NumPy/SciPy only, no Genesis):

* Each of the 31 strands is a discrete elastic rod / mass-spring chain with
  axial (stretch) springs, a discrete bending energy
  ``E = 0.5*kb*sum |x_{i-1} - 2 x_i + x_{i+1}|^2`` (kb = EI / l0^3) and light
  internal Rayleigh damping.  Both endpoints are free.
* Finite thickness is resolved with collision spheres of the frozen strand
  radius sampled along every segment (3 per segment).  Strand-strand and
  strand-plate contacts are unilateral normal penalty contacts with
  normal damping and *exactly zero* tangential (friction) force, so the
  strands slide freely and spread laterally.
* The plate is a frozen static surface of revolution built from the
  ``inner_profile`` of ``scene_geometry.yaml`` (shallow interior + raised
  rim).  Contact uses the true surface normal from the closest point on the
  axisymmetric profile polyline.
* Time stepping is symplectic (semi-implicit) Euler with a fixed sub-step,
  deterministic and stable for the chosen stiffnesses.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is only used for the contact broad phase.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None


def _profile_points(geometry: dict, plate_cfg: dict) -> np.ndarray:
    """Axisymmetric inner profile as an (r, z) polyline, axis outward."""
    profile = None
    if isinstance(geometry, dict):
        for item in geometry.get("objects", ()) or ():
            if item.get("id") == "plate":
                profile = (item.get("geometry") or {}).get("inner_profile")
                break
    if not profile:
        profile = [
            [0.10, float(plate_cfg.get("interior_floor_height", 0.08)) - 0.005],
            [0.88, float(plate_cfg.get("interior_floor_height", 0.08))],
            [1.05, 0.135],
            [float(plate_cfg.get("interior_radius", 1.05)) + 0.09,
             float(plate_cfg.get("rim_height", 0.22))],
        ]
    pts = [(float(r), float(z)) for r, z in profile]
    pts.sort(key=lambda p: p[0])
    # Close the profile on the axis (flat interior floor) and add the flat rim
    # top annulus out to the plate outer radius.
    first = pts[0]
    if first[0] > 1e-9:
        pts.insert(0, (0.0, first[1]))
    outer = float(plate_cfg.get("outer_radius", pts[-1][0] + 0.05))
    if outer > pts[-1][0] + 1e-9:
        pts.append((outer, pts[-1][1]))
    return np.asarray(pts, dtype=float)


def _plate_contact(xyz: np.ndarray, prof: np.ndarray, r_max: float):
    """Signed distance and outward unit normal of the axisymmetric inner surface.

    Returns ``(signed_distance, normal3)``.  ``signed_distance`` is positive on
    the free (fluid) side of the container surface.  Points outside the plate
    outer radius are reported as far away (no contact).
    """
    rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    q = np.column_stack((rho, xyz[:, 2]))
    a = prof[:-1]
    b = prof[1:]
    ab = b - a
    seg_len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-18)
    diff = q[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nsd,sd->ns", diff, ab) / seg_len2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dvec = q[:, None, :] - closest
    dist = np.sqrt(np.sum(dvec * dvec, axis=2))
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    dmin = dist[rows, j]
    dsel = dvec[rows, j]
    # Above/below classification from the profile height at this radius.
    z_surf = np.interp(rho, prof[:, 0], prof[:, 1])
    sign = np.where(q[:, 1] >= z_surf, 1.0, -1.0)
    # Fallback normal (segment normal, oriented with positive z component).
    seg_n = np.column_stack((-ab[:, 1], ab[:, 0])) / np.sqrt(seg_len2)[:, None]
    n2 = np.where(dmin[:, None] > 1e-12, sign[:, None] * dsel / np.maximum(dmin, 1e-12)[:, None], seg_n[j])
    signed = sign * dmin
    outside = rho > r_max
    signed = np.where(outside, 1.0e3, signed)
    inv_rho = 1.0 / np.maximum(rho, 1e-12)
    ex = np.where(rho > 1e-12, xyz[:, 0] * inv_rho, 1.0)
    ey = np.where(rho > 1e-12, xyz[:, 1] * inv_rho, 0.0)
    normal3 = np.column_stack((n2[:, 0] * ex, n2[:, 0] * ey, n2[:, 1]))
    nrm = np.maximum(np.linalg.norm(normal3, axis=1), 1e-12)
    return signed, normal3 / nrm[:, None]


class BenchmarkSolver:
    """Discrete elastic rods + frictionless penalty contact, symplectic Euler."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        frame0 = scene.initial_state()

        self.n_strand = int(cfg["strand_count"])
        strand = cfg["strand"]
        self.n_vert = int(strand["vertex_count"])
        self.radius = float(strand["radius"])
        self.length = float(strand["length"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        S, P = self.n_strand, self.n_vert
        self.pos = np.asarray(frame0.position, dtype=float).reshape(S, P, 3).copy()
        self.vel = np.zeros_like(self.pos)
        node_mass = float(strand["total_mass_each"]) / P
        self.node_mass = node_mass
        self.mass_flat = np.full(S * P, node_mass)
        self.inv_mass = 1.0 / node_mass
        self.body_id = np.repeat(np.arange(S), P)
        self.attr0 = {k: np.asarray(v, dtype=float) for k, v in frame0.attributes.items()}
        self.l0 = self.length / (P - 1)

        # Deterministic sub-millimetre placement imperfection (real bundles are
        # never perfectly aligned).  Far smaller than the 1 mm clearance between
        # neighbouring strands, so the bundle stays contact-free and at rest.
        rng = np.random.default_rng(int(seed))
        jitter = rng.normal(0.0, 1.0, size=(S, 1, 3)) * 4.0e-5
        self.pos += jitter

        # ---- material parameters (soft, spaghetti-like flexible rods) ----
        self.k_stretch = 800.0                     # N/m per axial spring
        self.c_stretch = 0.05                      # N*s/m axial damping
        self.k_bend = 60.0                         # N/m on the 2nd difference
        self.c_bend = 0.010                        # N*s/m bending damping
        self.k_contact = 500.0                     # N/m normal penalty
        self.c_contact = 0.25                      # N*s/m normal damping
        self.c_air = 2.0e-5                        # N*s/m weak ambient drag
        self.mu_s = float(cfg["contact"]["strand_strand"]["static_friction_coefficient"])
        self.mu_k = float(cfg["contact"]["strand_strand"]["kinetic_friction_coefficient"])
        self.stick_tol = 1.0e-4                    # m/s stick-slip tolerance

        # ---- plate geometry (frozen axisymmetric container) ----
        geom = cfg.get("scene_geometry", {}) or {}
        self.profile = _profile_points(geom, cfg["plate"])
        self.plate_rmax = float(cfg["plate"]["outer_radius"])

        # ---- collision sphere sampling along every segment ----
        sub = 3
        seg = P - 1
        s_local = np.arange(sub, dtype=float) / sub
        idx_a, idx_b, wa = [], [], []
        for e in range(seg):
            for s in s_local:
                idx_a.append(e)
                idx_b.append(e + 1)
                wa.append(1.0 - s)
        idx_a.append(P - 2)
        idx_b.append(P - 1)
        wa.append(0.0)
        self.samp_a = np.asarray(idx_a, dtype=int)
        self.samp_b = np.asarray(idx_b, dtype=int)
        self.samp_wa = np.asarray(wa, dtype=float)
        self.samp_wb = 1.0 - self.samp_wa
        self.n_samp = len(self.samp_a)
        self.samp_arc = (self.samp_a + (1.0 - self.samp_wa))  # monotone arc index
        self.samp_strand = np.repeat(np.arange(S), self.n_samp)

        # ---- time stepping ----
        self.substeps = 100
        self.dt = self.sample_dt / self.substeps
        self.time = 0.0
        self.step_index = 0
        self.pair_i = np.zeros(0, dtype=int)
        self.pair_j = np.zeros(0, dtype=int)
        self._pair_age = 10 ** 9
        self.pair_refresh = 15
        self.pair_margin = 0.012
        self._diag = {
            "method": "discrete elastic rods, symplectic Euler, frictionless penalty contact",
            "dt": self.dt,
            "collision_spheres": int(S * self.n_samp),
            "friction_coefficients": [self.mu_s, self.mu_k],
        }

        # reference scales
        total_mass = node_mass * S * P
        self.total_mass = total_mass
        g = float(np.linalg.norm(self.gravity))
        drop = max(float(cfg["initial_conditions"]["center"][2])
                   - float(cfg["plate"]["interior_floor_height"]), 1e-3)
        self.v_ref = float(np.sqrt(2.0 * g * drop))
        self.f_ref = total_mass * g
        self.p_ref = total_mass * self.v_ref

    # ------------------------------------------------------- internal forces
    def _internal_forces(self, pos, vel):
        """Axial + bending elastic forces with Rayleigh damping (Nnode form)."""
        f = np.zeros_like(pos)
        # axial springs
        d = pos[:, 1:, :] - pos[:, :-1, :]
        ln = np.linalg.norm(d, axis=2)
        ln_safe = np.maximum(ln, 1e-12)
        u = d / ln_safe[:, :, None]
        dv = vel[:, 1:, :] - vel[:, :-1, :]
        stretch = self.k_stretch * (ln - self.l0)
        damp = self.c_stretch * np.sum(dv * u, axis=2)
        mag = (stretch + damp)[:, :, None] * u
        f[:, :-1, :] += mag
        f[:, 1:, :] -= mag
        # bending: second difference operator
        dd = pos[:, :-2, :] - 2.0 * pos[:, 1:-1, :] + pos[:, 2:, :]
        ddv = vel[:, :-2, :] - 2.0 * vel[:, 1:-1, :] + vel[:, 2:, :]
        g = self.k_bend * dd + self.c_bend * ddv
        f[:, :-2, :] -= g
        f[:, 1:-1, :] += 2.0 * g
        f[:, 2:, :] -= g
        return f

    # ------------------------------------------------------- contact helpers
    def _sphere_state(self, pos, vel):
        wa = self.samp_wa[None, :, None]
        wb = self.samp_wb[None, :, None]
        q = wa * pos[:, self.samp_a, :] + wb * pos[:, self.samp_b, :]
        w = wa * vel[:, self.samp_a, :] + wb * vel[:, self.samp_b, :]
        return q.reshape(-1, 3), w.reshape(-1, 3)

    def _scatter(self, f_nodes, f_sph):
        """Distribute sphere forces to the two supporting nodes."""
        S, P = self.n_strand, self.n_vert
        fs = f_sph.reshape(S, self.n_samp, 3)
        wa = self.samp_wa[None, :, None]
        wb = self.samp_wb[None, :, None]
        np.add.at(f_nodes, (slice(None), self.samp_a), wa * fs)
        np.add.at(f_nodes, (slice(None), self.samp_b), wb * fs)

    def _refresh_pairs(self, q):
        rad = 2.0 * self.radius + self.pair_margin
        if _KDTree is not None:
            tree = _KDTree(q)
            pairs = tree.query_pairs(rad, output_type="ndarray")
        else:  # pragma: no cover - fallback broad phase
            plist = []
            block = 512
            for start in range(0, len(q), block):
                stop = min(start + block, len(q))
                d = np.linalg.norm(q[start:stop, None, :] - q[None, :, :], axis=2)
                ii, jj = np.nonzero(d < rad)
                ii = ii + start
                keep = ii < jj
                plist.append(np.column_stack((ii[keep], jj[keep])))
            pairs = np.vstack(plist) if plist else np.zeros((0, 2), dtype=int)
        if len(pairs) == 0:
            self.pair_i = np.zeros(0, dtype=int)
            self.pair_j = np.zeros(0, dtype=int)
            return
        i = pairs[:, 0].astype(int)
        j = pairs[:, 1].astype(int)
        same = self.samp_strand[i] == self.samp_strand[j]
        arc_gap = np.abs(self.samp_arc[i % self.n_samp] - self.samp_arc[j % self.n_samp])
        keep = (~same) | (arc_gap > 3.0)
        self.pair_i = i[keep]
        self.pair_j = j[keep]

    def _contact_forces(self, q, w, collect=False):
        """Frictionless unilateral normal contacts (plate + strand/strand)."""
        f_sph = np.zeros_like(q)
        rec = {"n": [], "fn": [], "vrel": []} if collect else None

        # --- strand / plate ---
        signed, normal = _plate_contact(q, self.profile, self.plate_rmax)
        pen = np.clip(self.radius - signed, None, 3.0 * self.radius)
        act = pen > 0.0
        if np.any(act):
            na = normal[act]
            vn = np.sum(w[act] * na, axis=1)
            fn = self.k_contact * pen[act] + self.c_contact * np.maximum(-vn, 0.0)
            fn = np.maximum(fn, 0.0)
            fvec = fn[:, None] * na
            idx = np.nonzero(act)[0]
            np.add.at(f_sph, idx, fvec)
            self._plate_force_step = fvec.sum(axis=0)
            if collect:
                rec["n"].append(na)
                rec["fn"].append(fn)
                rec["vrel"].append(w[act])
        else:
            self._plate_force_step = np.zeros(3)

        # --- strand / strand ---
        if len(self.pair_i):
            i, j = self.pair_i, self.pair_j
            dvec = q[i] - q[j]
            dist = np.linalg.norm(dvec, axis=1)
            pen2 = np.clip(2.0 * self.radius - dist, None, 2.0 * self.radius)
            act2 = (pen2 > 0.0) & (dist > 1e-12)
            if np.any(act2):
                ia, ja = i[act2], j[act2]
                nb = dvec[act2] / dist[act2][:, None]
                vrel = w[ia] - w[ja]
                vn = np.sum(vrel * nb, axis=1)
                fn = self.k_contact * pen2[act2] + self.c_contact * np.maximum(-vn, 0.0)
                fn = np.maximum(fn, 0.0)
                fvec = fn[:, None] * nb
                np.add.at(f_sph, ia, fvec)
                np.add.at(f_sph, ja, -fvec)
                if collect:
                    rec["n"].append(nb)
                    rec["fn"].append(fn)
                    rec["vrel"].append(vrel)
        return f_sph, rec

    # ------------------------------------------------------------- time step
    def _step(self, dt):
        pos, vel = self.pos, self.vel
        q, w = self._sphere_state(pos, vel)
        if self._pair_age >= self.pair_refresh:
            self._refresh_pairs(q)
            self._pair_age = 0
        else:
            self._pair_age += 1
        f = self._internal_forces(pos, vel)
        f_sph, _ = self._contact_forces(q, w)
        self._scatter(f, f_sph)
        f += self.node_mass * self.gravity[None, None, :]
        f -= self.c_air * vel
        vel += (dt * self.inv_mass) * f
        pos += dt * vel
        self.step_index += 1

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target

        pos = self.pos.reshape(-1, 3)
        vel = self.vel.reshape(-1, 3)
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise FloatingPointError("solver produced non-finite state")

        # instantaneous diagnostic contact set at the sample time
        q, w = self._sphere_state(self.pos, self.vel)
        _, rec = self._contact_forces(q, w, collect=True)
        if rec["fn"]:
            normal = np.vstack(rec["n"])
            fn = np.concatenate(rec["fn"])
            vrel = np.vstack(rec["vrel"])
        else:
            normal = np.zeros((0, 3))
            fn = np.zeros(0)
            vrel = np.zeros((0, 3))
        if len(fn) == 0:
            # No candidate contact exists yet: emit one inactive placeholder so
            # every frame carries a well-formed contact record.
            normal = np.array([[0.0, 0.0, 1.0]])
            fn = np.zeros(1)
            vrel = np.zeros((1, 3))
            mask = np.zeros(1)
        else:
            mask = (fn > 0.0).astype(float)
        n_c = len(fn)
        tangential = np.zeros((n_c, 3))  # frictionless: mu = 0 exactly

        n_node = pos.shape[0]
        mass = self.mass_flat.copy()
        gravity_term = self.total_mass * self.gravity
        plate_term = np.asarray(self._plate_force_step, dtype=float)
        ext_force = np.zeros((1, 2, 3))
        ext_force[0, 0] = gravity_term
        ext_force[0, 1] = plate_term

        attrs = {k: v.copy() for k, v in self.attr0.items()}
        attrs["mass_quantity"] = mass.copy()

        frame = StateFrame(
            time=target,
            position=pos.copy(),
            velocity=vel.copy(),
            mass=mass,
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields={
                "budget_domain_weights": np.ones((1, n_node)),
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": np.zeros((1, 3)),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "cumulative_mass_source": np.zeros(1),
                "friction_active_mask": mask,
                "friction_static_coefficient": np.full(n_c, self.mu_s),
                "friction_dynamic_coefficient": np.full(n_c, self.mu_k),
                "friction_normal": normal,
                "friction_normal_force": fn,
                "friction_relative_velocity": vrel,
                "friction_tangential_force": tangential,
            },
            scalars={
                "friction_stick_speed_tolerance": self.stick_tol,
                "reference_force": self.f_ref,
                "reference_momentum": self.p_ref,
                "reference_velocity": self.v_ref,
                "strand_strand_static_friction": self.mu_s,
                "strand_strand_kinetic_friction": self.mu_k,
                "strand_plate_static_friction": self.mu_s,
                "strand_plate_kinetic_friction": self.mu_k,
            },
        )
        frame.validate()
        self._diag["last_time"] = target
        self._diag["last_contact_count"] = int(n_c)
        return frame

    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diag)
        out["steps"] = int(self.step_index)
        out["mean_height"] = float(np.mean(self.pos[:, :, 2]))
        out["lateral_extent"] = float(
            np.max(np.sqrt(self.pos[:, :, 0] ** 2 + self.pos[:, :, 1] ** 2))
        )
        return out
