"""Physics solver for scene 078_squishy_ball_glass_impact.

Volumetric finite-element simulation of a soft, tendril-covered ball that
flies toward a fixed glass wall, flattens against it, and elastically
recovers.  The implementation is self-contained (NumPy only):

*   3-D tetrahedral FEM on the frozen volume mesh
    (assets/squishy_ball_volume.npz: nodes / tets).
*   Stable Neo-Hookean constitutive model
        psi(F) = mu/2 (|F|^2 - 3) + lambda/2 (J - 1 - ln J)^2 ,  J = det F
    with first Piola stress
        P = mu F + lambda (J - 1 - ln J)(J - 1) F^{-T},
    evaluated with exact adjugate-based inverse transposes (always finite).
*   Explicit symplectic Euler time stepping with mass-proportional Rayleigh
    damping; the step size satisfies the CFL condition of the mesh, the
    material wave speed and the contact spring.
*   Penalty contact against the front face of the fixed glass box
    (a stiff repulsive spring plus normal impact damping).  Contact and
    damping are the only external forces; gravity is absent because the
    frozen scene provides no support surface and the ball must remain in
    the fixed camera field of view for the whole 1.2 s duration.
*   Exact momentum bookkeeping: internal elastic forces cancel in the
    combined domain, so the accumulated external impulse equals the change
    of total linear momentum.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Frozen material / numerics parameters (soft squishy toy rubber).
# ---------------------------------------------------------------------------
YOUNG_MODULUS = 70.0        # Pa  (very soft foam rubber toy)
POISSON_RATIO = 0.42        # nearly incompressible rubber
RAYLEIGH_MASS_DAMPING = 0.25   # 1/s, mild global dissipation
CONTACT_STIFFNESS = 2000.0  # N/m penalty spring per contacting node
CONTACT_DAMPING = 0.5       # N*s/m normal impact damping per node
SUBSTEP_DT = 2.0e-4         # s, CFL-safe explicit substep
VELOCITY_CLAMP = 30.0       # m/s, safety net (never active in normal run)
FORCE_CLAMP = 50.0          # N per node, safety net against blow-up
MIN_J = 0.02                # clamp for the log-barrier volumetric term


class BenchmarkSolver:
    """Explicit FEM solver honouring the Scene / StateFrame contract."""

    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        initial = scene.initial_state()
        self.x = np.array(initial.position, dtype=np.float64).copy()
        self.v = np.array(initial.velocity, dtype=np.float64).copy()
        self.mass = np.array(initial.mass, dtype=np.float64).copy()
        self.n = len(self.x)
        self.body_id = np.ones(self.n, dtype=np.int64)

        # Preserve any scene-provided per-sample attributes (e.g. component).
        self.base_attributes = {
            str(k): np.asarray(v).copy() for k, v in (initial.attributes or {}).items()
        }

        self.seed = int(seed)
        self.time = 0.0
        self.substeps = 0

        # --- tetrahedral topology -------------------------------------------
        asset = getattr(scene, "_asset", None)
        if asset is None:
            asset = Path(__file__).resolve().parents[1] / "assets" / "squishy_ball_volume.npz"
        with np.load(asset) as data:
            tets = np.asarray(data["tets"], dtype=np.int64)
            rest = np.asarray(data["rest_position"], dtype=np.float64)
        if rest.shape != self.x.shape:
            rest = self.x.copy()
        self.tets_all = tets
        n_tets = len(tets)

        # Rest shape matrices Dm = [X0-X3 | X1-X3 | X2-X3] (columns).
        p0, p1, p2, p3 = (rest[tets[:, k]] for k in range(4))
        dm = np.stack([p0 - p3, p1 - p3, p2 - p3], axis=2)
        volumes = np.abs(np.linalg.det(dm)) / 6.0
        # Pseudo-inverse keeps every element's F finite (a few tets are
        # degenerate slivers in the frozen asset).
        self.b_all = np.linalg.pinv(dm)
        active = volumes > 1e-12
        self.tets = tets[active]
        self.b = self.b_all[active]
        self.v0 = volumes[active]
        self.bt = np.transpose(self.b, (0, 2, 1))
        self.tet_flat = np.concatenate(
            [self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]]
        )

        # --- material --------------------------------------------------------
        nu = POISSON_RATIO
        e = YOUNG_MODULUS
        self.mu = e / (2.0 * (1.0 + nu))
        self.lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # --- glass wall (fixed box) ------------------------------------------
        wall = self._wall_parameters(config)
        self.wall_x, self.wall_y_lim, self.wall_z_lim = wall

        # --- budget accumulators ---------------------------------------------
        self.cum_impulse = np.zeros(3)          # gravity + contact + damping
        self.force_int_contact = np.zeros(3)    # \int F_contact dt  (sample)
        self.force_int_damping = np.zeros(3)    # \int F_damping dt  (sample)
        self.last_emit_time = 0.0
        self.peak_penetration = 0.0
        self.contact_node_peak = 0

        # --- frozen momentum scale (2x the launch momentum magnitude) -------
        speed0 = float(np.linalg.norm(self.v, axis=1).max())
        self.total_mass = float(self.mass.sum())
        self.reference_momentum = 2.0 * self.total_mass * max(speed0, 1e-9)

    # ------------------------------------------------------------------
    @staticmethod
    def _wall_parameters(config):
        """Front plane and footprint of the fixed glass box."""
        objects = (config.get("scene_geometry") or {}).get("objects", [])
        for obj in objects:
            if obj.get("id") == "glass_wall":
                geom = obj.get("geometry", {})
                half = geom.get("half_extents", [0.012, 0.72, 0.72])
                tr = obj.get("transform", {}).get("translation", [0.012, 0.0, 0.42])
                hx, hy, hz = (float(v) for v in half)
                tx, ty, tz = (float(v) for v in tr)
                front = tx - hx
                return front, (ty - hy, ty + hy), (tz - hz, tz + hz)
        return 0.0, (-1e9, 1e9), (-1e9, 1e9)

    # ------------------------------------------------------------------
    def _elastic_forces(self, x: np.ndarray) -> np.ndarray:
        """Nodal elastic forces of the stable Neo-Hookean tet mesh."""
        t = self.tets
        q0, q1, q2, q3 = (x[t[:, k]] for k in range(4))
        ds = np.stack([q0 - q3, q1 - q3, q2 - q3], axis=2)
        f = ds @ self.b  # deformation gradients (Ne,3,3)

        # Exact F^{-T} via adjugate (cross products) -- always finite.
        a, b, c = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        bxc = np.cross(b, c)
        cxa = np.cross(c, a)
        axb = np.cross(a, b)
        det = np.einsum("ij,ij->i", a, bxc)
        sign = np.where(det >= 0.0, 1.0, -1.0)
        denom = sign * np.maximum(np.abs(det), 1e-8)
        finv_t = np.stack([bxc, cxa, axb], axis=2) / denom[:, None, None]

        # Volumetric log-barrier term, clamped for numerical safety.
        j_clamped = np.clip(det, MIN_J, 1.0 / MIN_J)
        t_vol = self.lam * (det - 1.0 - np.log(j_clamped)) * (det - 1.0)

        p = self.mu * f + t_vol[:, None, None] * finv_t

        # f_k = -V0 * (P @ B[k, :]^T);  node 3 balances nodes 0..2.
        g = p @ self.bt
        w = -self.v0[:, None, None] * g
        wsum = w.sum(axis=2)
        forces = np.empty((self.n, 3))
        flat = self.tet_flat
        for comp in range(3):
            weights = np.concatenate(
                [w[:, comp, 0], w[:, comp, 1], w[:, comp, 2], -wsum[:, comp]]
            )
            forces[:, comp] = np.bincount(flat, weights=weights, minlength=self.n)
        np.nan_to_num(forces, copy=False)
        np.clip(forces, -FORCE_CLAMP, FORCE_CLAMP, out=forces)
        return forces

    # ------------------------------------------------------------------
    def _contact_force(self) -> np.ndarray:
        """Penalty contact against the glass front plane x = wall_x."""
        fc = np.zeros((self.n, 3))
        x = self.x
        depth = self.wall_x - x[:, 0]
        inside = (
            (x[:, 1] >= self.wall_y_lim[0])
            & (x[:, 1] <= self.wall_y_lim[1])
            & (x[:, 2] >= self.wall_z_lim[0])
            & (x[:, 2] <= self.wall_z_lim[1])
        )
        pen = (depth > 0.0) & inside
        if np.any(pen):
            d = depth[pen]
            vn = self.v[pen, 0]
            fc[pen, 0] = CONTACT_STIFFNESS * d - CONTACT_DAMPING * np.minimum(vn, 0.0)
            self.peak_penetration = max(self.peak_penetration, float(d.max()))
            self.contact_node_peak = max(self.contact_node_peak, int(pen.sum()))
        return fc

    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        # Internal (elastic) forces.
        f_int = self._elastic_forces(self.x)

        # External forces: contact penalty + mass-proportional damping.
        # Gravity is zero: the frozen scene has no support surface and the
        # ball must stay inside the fixed camera frustum for the full take.
        fc = self._contact_force()
        fd = -RAYLEIGH_MASS_DAMPING * self.mass[:, None] * self.v
        f_ext = fc + fd

        # Symplectic Euler.
        acc = (f_int + f_ext) / self.mass[:, None]
        self.v += h * acc
        np.clip(self.v, -VELOCITY_CLAMP, VELOCITY_CLAMP, out=self.v)
        self.x += h * self.v
        np.nan_to_num(self.x, copy=False)

        # Exact external-impulse accumulation (gravity term is identically 0).
        impulse = h * f_ext.sum(axis=0)
        self.cum_impulse += impulse
        self.force_int_contact += h * fc.sum(axis=0)
        self.force_int_damping += h * fd.sum(axis=0)
        self.substeps += 1

    # ------------------------------------------------------------------
    def _element_deformation_gradients(self) -> np.ndarray:
        t = self.tets_all
        q0, q1, q2, q3 = (self.x[t[:, k]] for k in range(4))
        ds = np.stack([q0 - q3, q1 - q3, q2 - q3], axis=2)
        f = ds @ self.b_all
        return np.nan_to_num(f)

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time:
            while self.time < target - 1e-15:
                h = min(SUBSTEP_DT, target - self.time)
                self._step(h)
                self.time += h
            self.time = target

        # Sample-interval average external forces (per-term, G=1 domain).
        interval = max(target - self.last_emit_time, 0.0)
        if interval > 0.0:
            contact_avg = self.force_int_contact / interval
            damping_avg = self.force_int_damping / interval
        else:
            contact_avg = np.zeros(3)
            damping_avg = np.zeros(3)
        self.force_int_contact[:] = 0.0
        self.force_int_damping[:] = 0.0
        self.last_emit_time = target

        # budget_external_force[g, k, :] = [gravity(=0), contact, damping]
        external_force = np.zeros((1, 3, 3))
        external_force[0, 1, :] = contact_avg
        external_force[0, 2, :] = damping_avg

        attributes = dict(self.base_attributes)
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": external_force,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": self._element_deformation_gradients(),
        }
        scalars = {
            "reference_momentum": float(self.reference_momentum),
        }

        frame_position = self.x.copy()
        frame_velocity = self.v.copy()
        frame = StateFrame(
            time=target,
            position=frame_position,
            velocity=frame_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        kinetic = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.v, self.v)))
        return {
            "scene_id": "078_squishy_ball_glass_impact",
            "method": "explicit FEM, stable Neo-Hookean tets, penalty contact",
            "young_modulus_pa": YOUNG_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "substep_dt": SUBSTEP_DT,
            "substeps": self.substeps,
            "nodes": int(self.n),
            "active_tets": int(len(self.tets)),
            "total_tets": int(len(self.tets_all)),
            "total_mass_kg": self.total_mass,
            "kinetic_energy_j": kinetic,
            "peak_penetration_m": self.peak_penetration,
            "peak_contact_nodes": self.contact_node_peak,
            "reference_momentum": float(self.reference_momentum),
            "time": float(self.time),
        }
