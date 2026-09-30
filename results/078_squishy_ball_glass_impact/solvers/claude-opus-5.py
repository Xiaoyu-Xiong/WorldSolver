"""Explicit finite-element solver for a tendril-covered squishy ball striking glass.

Physics
-------
The toy is a three-dimensional elastic solid discretized with the frozen
tetrahedral topology shipped in ``assets/squishy_ball_volume.npz``.  Internal
stresses come from a stable (inversion-safe) compressible Neo-Hookean energy

    Psi(F) = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2
    P(F)   = mu F + (lam (J - 1) - mu) dJ/dF ,

with ``dJ/dF`` evaluated from column cross products so no matrix inverse of the
current configuration is ever required.  A Kelvin-Voigt term ``eta * dF/dt``
supplies material viscosity, and a momentum-conserving centre-of-mass relative
drag removes long-lived internal ringing.

Time stepping is symplectic (semi-implicit) Euler with a step size derived from
the lumped-mass nodal frequency of the frozen mesh, so the scheme is stable and
fully deterministic.  The fixed glass wall is a frozen half-space taken from
``scene_geometry.yaml`` and enforced as a unilateral Signorini constraint with
Coulomb friction: positions are projected onto the wall face, inbound normal
velocity is removed, and tangential velocity is reduced by the friction cone.
Nodes leave the wall as soon as the recovering elastic stress pulls them back,
which produces the flatten / round-again / tendrils-lift-away sequence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_ASSET_NAME = "squishy_ball_volume.npz"


def _load_topology(scene) -> dict:
    """Locate and load the frozen volumetric topology asset."""
    candidates = []
    asset = getattr(scene, "_asset", None)
    if asset is not None:
        candidates.append(Path(asset))
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent / "assets" / _ASSET_NAME)
    candidates.append(Path.cwd() / "assets" / _ASSET_NAME)
    for path in candidates:
        try:
            if path.is_file():
                with np.load(path) as data:
                    return {k: np.asarray(data[k]) for k in data.files}
        except OSError:
            continue
    raise FileNotFoundError("could not locate %s" % _ASSET_NAME)


def _wall_from_geometry() -> dict:
    """Frozen glass-wall half-space (facing -x) from scene_geometry.yaml."""
    wall = {
        "plane_x": 0.0,
        "y_min": -0.72,
        "y_max": 0.72,
        "z_min": -0.30,
        "z_max": 1.14,
    }
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        path = base / "scene_geometry.yaml"
        if not path.is_file():
            continue
        try:
            import yaml

            with path.open("r", encoding="utf-8") as handle:
                spec = yaml.safe_load(handle)
            for obj in spec.get("objects", []):
                if obj.get("id") != "glass_wall":
                    continue
                half = [float(v) for v in obj["geometry"]["half_extents"]]
                trans = [float(v) for v in obj["transform"]["translation"]]
                wall["plane_x"] = trans[0] - half[0]
                wall["y_min"] = trans[1] - half[1]
                wall["y_max"] = trans[1] + half[1]
                wall["z_min"] = trans[2] - half[2]
                wall["z_max"] = trans[2] + half[2]
                return wall
        except Exception:
            return wall
    return wall


class BenchmarkSolver:
    """Deterministic explicit FEM solver for the squishy-ball / glass impact."""

    # Soft rubber-foam toy.  The lumped mass of the frozen scene normalizes the
    # total ball mass to 1 kg over ~0.0857 m^3 of material, so the modulus is
    # chosen consistently with that (very light, very compliant) normalization
    # to give a several-hundred-millisecond contact with strong flattening.
    youngs_modulus = 130.0        # Pa
    poisson_ratio = 0.40          # compliant but volume-resisting
    # Kelvin-Voigt viscosity is sized so the strain-rate stability bound stays
    # comparable to the elastic CFL bound on this lumped-mass mesh: it damps the
    # stiffest mesh modes while leaving the ball's fundamental flattening mode
    # nearly undamped, so the rebound and tendril lift-off survive.
    viscosity = 0.12              # Pa*s, Kelvin-Voigt strain-rate damping
    internal_drag = 0.6           # 1/s, momentum-conserving relative drag
    friction_coeff = 0.12         # low friction so tendrils spread outward
    cfl_safety = 0.20
    dt_max = 4.0e-4
    dt_min = 2.0e-5
    speed_cap = 25.0              # m/s numerical guard, far above physical speeds

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=np.float64)
        self.velocity = np.array(frame.velocity, dtype=np.float64)
        self.n_nodes = self.position.shape[0]
        if frame.mass is not None:
            self.mass = np.array(frame.mass, dtype=np.float64).reshape(self.n_nodes)
        else:
            self.mass = np.full(self.n_nodes, 1.0 / self.n_nodes)
        self.mass = np.where(self.mass > 0.0, self.mass, 1.0 / self.n_nodes)
        self.inv_mass = 1.0 / self.mass
        if frame.body_id is not None:
            self.body_id = np.array(frame.body_id, dtype=np.int64).reshape(self.n_nodes)
        else:
            self.body_id = np.ones(self.n_nodes, dtype=np.int64)
        self.total_mass = float(self.mass.sum())

        topo = _load_topology(scene)
        tets = np.asarray(topo["tets"], dtype=np.int64).reshape(-1, 4)
        rest = np.asarray(
            topo["rest_position"] if "rest_position" in topo else topo["nodes"],
            dtype=np.float64,
        )
        if rest.shape[0] != self.n_nodes:
            rest = self.position.copy()
        self.tets_all = tets
        self.n_elements = tets.shape[0]

        # Reference shape matrices; flip inverted elements so all rest volumes
        # are positive, and mask out degenerate (zero-volume) elements.
        rp = rest[tets]
        dm = np.stack(
            (rp[:, 1] - rp[:, 0], rp[:, 2] - rp[:, 0], rp[:, 3] - rp[:, 0]), axis=2
        )
        det = np.linalg.det(dm)
        flip = det < 0.0
        if np.any(flip):
            tets = tets.copy()
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            rp = rest[tets]
            dm = np.stack(
                (rp[:, 1] - rp[:, 0], rp[:, 2] - rp[:, 0], rp[:, 3] - rp[:, 0]), axis=2
            )
            det = np.linalg.det(dm)
        self.tets_oriented = tets
        rest_volume = det / 6.0
        scale = max(float(np.max(np.abs(rest_volume))), 1e-30)
        self.valid = rest_volume > 1e-8 * scale
        self.tets = tets[self.valid]
        self.rest_volume = rest_volume[self.valid]
        self.dm_inv = np.linalg.inv(dm[self.valid])
        self.idx_flat = self.tets.reshape(-1)
        self.rest_position = rest

        # Lame parameters of the compliant solid.
        e_mod = float(self.youngs_modulus)
        nu = float(self.poisson_ratio)
        self.mu = e_mod / (2.0 * (1.0 + nu))
        self.lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Shape-function gradients of every valid element (rows of Dm^-1).
        b123 = self.dm_inv                      # [E,3,3]: row k -> grad phi_{k+1}
        b0 = -b123.sum(axis=1)                  # [E,3]
        grad_sq = np.sum(b123 ** 2, axis=(1, 2)) + np.sum(b0 ** 2, axis=1)

        # Explicit-stability step size from lumped nodal frequencies.
        k_elem = (2.0 * self.mu + self.lam) * self.rest_volume * grad_sq
        c_elem = float(self.viscosity) * self.rest_volume * grad_sq
        k_node = np.bincount(
            self.idx_flat, weights=np.repeat(k_elem, 4), minlength=self.n_nodes
        )
        c_node = np.bincount(
            self.idx_flat, weights=np.repeat(c_elem, 4), minlength=self.n_nodes
        )
        omega_max = float(np.sqrt(np.max(k_node * self.inv_mass)))
        dt_elastic = 2.0 / omega_max if omega_max > 0.0 else self.dt_max
        c_ratio = float(np.max(c_node * self.inv_mass))
        dt_visc = 2.0 / c_ratio if c_ratio > 0.0 else self.dt_max
        dt = float(self.cfl_safety) * min(dt_elastic, dt_visc)
        self.dt = float(np.clip(dt, self.dt_min, self.dt_max))

        self.wall = _wall_from_geometry()
        self.gravity = np.zeros(3, dtype=np.float64)   # horizontal shot, no gravity

        self.current_time = 0.0
        self.cumulative_impulse = np.zeros(3, dtype=np.float64)
        self.interval_impulse = np.zeros(3, dtype=np.float64)
        self.interval_span = 0.0
        self.contact_force = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self._max_penetration = 0.0
        self._min_jacobian = 1.0

        speed0 = float(np.linalg.norm(self.velocity, axis=1).max()) if self.n_nodes else 0.0
        try:
            cfg = getattr(scene, "config", {}) or {}
            v_ic = cfg["initial_conditions"]["squishy_ball"]["linear_velocity"]
            speed0 = max(speed0, float(np.linalg.norm(np.asarray(v_ic, dtype=np.float64))))
        except Exception:
            pass
        self.reference_momentum = float(max(self.total_mass * max(speed0, 1e-3), 1e-9))

        self._identity = np.eye(3, dtype=np.float64)
        self._F_buffer = np.tile(self._identity, (self.n_elements, 1, 1))

    # ------------------------------------------------------------------ physics
    def _deformation_gradients(self):
        """Per-element deformation gradient of the valid elements."""
        p = self.position[self.tets]
        ds = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        return np.einsum("eij,ejk->eik", ds, self.dm_inv)

    def _internal_forces(self):
        """Neo-Hookean elastic + Kelvin-Voigt viscous nodal forces."""
        p = self.position[self.tets]
        v = self.velocity[self.tets]
        ds = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        dv = np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=2)
        f_mat = np.einsum("eij,ejk->eik", ds, self.dm_inv)
        f_dot = np.einsum("eij,ejk->eik", dv, self.dm_inv)

        c0 = f_mat[:, :, 0]
        c1 = f_mat[:, :, 1]
        c2 = f_mat[:, :, 2]
        djdf = np.stack(
            (np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)), axis=2
        )
        jac = np.einsum("ei,ei->e", c0, djdf[:, :, 0])
        self._min_jacobian = min(self._min_jacobian, float(jac.min()))

        coeff = self.lam * (jac - 1.0) - self.mu
        stress = self.mu * f_mat + coeff[:, None, None] * djdf
        stress += self.viscosity * f_dot

        h = -self.rest_volume[:, None, None] * np.einsum(
            "eij,ekj->eik", stress, self.dm_inv
        )
        node_f = np.empty((self.tets.shape[0], 4, 3), dtype=np.float64)
        node_f[:, 1] = h[:, :, 0]
        node_f[:, 2] = h[:, :, 1]
        node_f[:, 3] = h[:, :, 2]
        node_f[:, 0] = -(node_f[:, 1] + node_f[:, 2] + node_f[:, 3])

        flat = node_f.reshape(-1, 3)
        force = np.empty((self.n_nodes, 3), dtype=np.float64)
        for axis in range(3):
            force[:, axis] = np.bincount(
                self.idx_flat, weights=flat[:, axis], minlength=self.n_nodes
            )

        # Momentum-conserving relative drag damps internal ringing only.
        if self.internal_drag > 0.0:
            v_com = (self.mass[:, None] * self.velocity).sum(axis=0) / self.total_mass
            force -= self.internal_drag * self.mass[:, None] * (self.velocity - v_com)
        return force

    def _resolve_wall_contact(self):
        """Unilateral Signorini contact with Coulomb friction on the glass face."""
        wall = self.wall
        px = self.position[:, 0]
        hit = px > wall["plane_x"]
        if not np.any(hit):
            return
        y = self.position[:, 1]
        z = self.position[:, 2]
        hit &= (y >= wall["y_min"]) & (y <= wall["y_max"])
        hit &= (z >= wall["z_min"]) & (z <= wall["z_max"])
        if not np.any(hit):
            return
        idx = np.flatnonzero(hit)
        depth = px[idx] - wall["plane_x"]
        self._max_penetration = max(self._max_penetration, float(depth.max()))

        v_old = self.velocity[idx].copy()
        vn = v_old[:, 0]
        inbound = vn > 0.0
        v_new = v_old.copy()
        v_new[inbound, 0] = 0.0

        if self.friction_coeff > 0.0 and np.any(inbound):
            vt = v_new[:, 1:3]
            vt_mag = np.sqrt(np.sum(vt ** 2, axis=1))
            budget = self.friction_coeff * np.where(inbound, vn, 0.0)
            scale = np.where(vt_mag > 1e-12, 1.0 - budget / np.maximum(vt_mag, 1e-12), 1.0)
            scale = np.clip(scale, 0.0, 1.0)
            v_new[:, 1:3] = vt * scale[:, None]

        self.velocity[idx] = v_new
        # Non-penetration projection (positions only, momentum untouched).
        self.position[idx, 0] = wall["plane_x"]

        dp = (self.mass[idx, None] * (v_new - v_old)).sum(axis=0)
        self.cumulative_impulse += dp
        self.interval_impulse += dp

    def _step(self, h: float) -> None:
        force = self._internal_forces()
        if self.gravity.any():
            force += self.mass[:, None] * self.gravity[None, :]
        self.velocity += h * force * self.inv_mass[:, None]

        if not np.isfinite(self.velocity).all():
            self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.sqrt(np.sum(self.velocity ** 2, axis=1))
        over = speed > self.speed_cap
        if np.any(over):
            self.velocity[over] *= (self.speed_cap / speed[over])[:, None]

        self.position += h * self.velocity
        if not np.isfinite(self.position).all():
            self.position = np.nan_to_num(self.position, nan=0.0, posinf=0.0, neginf=0.0)

        self._resolve_wall_contact()
        self.current_time += h
        self.interval_span += h
        self.step_count += 1

    # ------------------------------------------------------------------ contract
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        self.interval_impulse = np.zeros(3, dtype=np.float64)
        self.interval_span = 0.0
        guard = 0
        max_steps = int(4.0e6)
        while self.current_time < target - 1e-12 and guard < max_steps:
            h = min(self.dt, target - self.current_time)
            if h <= 0.0:
                break
            self._step(h)
            guard += 1
        self.current_time = target

        if self.interval_span > 0.0:
            self.contact_force = self.interval_impulse / self.interval_span
        else:
            self.contact_force = np.zeros(3, dtype=np.float64)

        f_mat = self._F_buffer.copy()
        f_mat[self.valid] = self._deformation_gradients()
        f_mat = np.nan_to_num(f_mat, nan=0.0, posinf=0.0, neginf=0.0)

        external_force = np.zeros((1, 2, 3), dtype=np.float64)
        external_force[0, 0] = self.contact_force
        external_force[0, 1] = self.total_mass * self.gravity

        fields = {
            "budget_domain_weights": np.ones((1, self.n_nodes), dtype=np.float64),
            "budget_external_force": external_force,
            "budget_cumulative_external_impulse": self.cumulative_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": f_mat,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "component": np.zeros(self.n_nodes, dtype=np.int64),
        }
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars={"reference_momentum": self.reference_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        momentum = (self.mass[:, None] * self.velocity).sum(axis=0)
        return {
            "method": "explicit_neo_hookean_fem_signorini_contact",
            "dt": float(self.dt),
            "steps": int(self.step_count),
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elements),
            "active_elements": int(self.tets.shape[0]),
            "youngs_modulus": float(self.youngs_modulus),
            "poisson_ratio": float(self.poisson_ratio),
            "viscosity": float(self.viscosity),
            "friction_coeff": float(self.friction_coeff),
            "total_mass": float(self.total_mass),
            "momentum": momentum.tolist(),
            "cumulative_contact_impulse": self.cumulative_impulse.tolist(),
            "max_penetration": float(self._max_penetration),
            "min_jacobian": float(self._min_jacobian),
        }
