"""Physically-based solver for the suspended elastic bunny pendulum.

Model
-----
The bunny surface mesh (triangle soup from ``assets/bunny.obj``) is treated as a
three-dimensional elastic solid discretised by its edges: an XPBD (extended
position based dynamics) compliant distance-constraint network.  This is a
conservative, unconditionally stable discretisation of a soft elastic body:

* Each mesh edge is a compliant axial constraint with stiffness
  ``k = E_eff * L0`` derived from an effective Young's modulus so that the
  self-weight strain ``rho_eff * g * L / E_eff`` reproduces the very soft
  material of the source experiment (large stretch near the suspension).
* Gravity acts on every node; the crown patch nodes flagged ``fixed`` in the
  scene initial state are Dirichlet-fixed at their attachment positions
  (infinite mass / zero inverse mass), which produces attachment reaction
  impulses that are accumulated for the momentum-budget fields.
* The initial pose is a rigid displacement of the rest shape, so the rest
  (material) configuration equals the initial pose and the initial per-element
  deformation gradient is the identity-on-the-tangent-plane map.

Time integration uses fixed sub-steps (``dt_sub``) with symplectic XPBD
projection; sample frames are produced exactly at the requested sample times.

Momentum budget bookkeeping (single global domain, G=1):
  term 0: gravity acting on all nodes, ``M_total * g``;
  term 1: attachment reactions = ``-M_fixed * g`` plus the negative of the
          constraint impulses injected into free nodes by mixed (fixed-free)
          edges, so that ``sum_k F_k == dP/dt`` holds to sub-step accuracy and
          the cumulative impulse equals the delivered impulse ``P(t) - P(0)``.
There is no mass inflow/outflow or mass source, so those fields are zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Model parameters (SI).  The scene freezes total mass to 1 kg, so the
# effective modulus is scaled to keep the self-weight strain of the source
# experiment (very soft bunny, visible stretch near the suspension).
# ---------------------------------------------------------------------------
_E_EFFECTIVE = 1.0e3      # Pa, effective axial modulus of the constraint net
_DT_SUB = 2.0e-3          # s, internal sub-step
_ITERATIONS = 3           # XPBD constraint iterations per sub-step
_DAMPING = 0.3            # 1/s, light viscous damping (air drag-like)


def _load_faces(path: Path) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(tuple(idx))  # type: ignore[arg-type]
    if not faces:
        raise ValueError(f"No faces found in {path}")
    return np.asarray(faces, dtype=np.int64)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        meta = scene.metadata()
        self._gravity = np.asarray(meta.get("gravity", [0.0, 0.0, -9.81]),
                                   dtype=np.float64)

        x0 = np.asarray(frame0.position, dtype=np.float64).copy()
        self._x = x0.copy()
        self._x_prev = x0.copy()
        self._v = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self._mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        n = len(self._x)

        fixed_attr = np.asarray(frame0.attributes["fixed"], dtype=np.float64)
        self._fixed_mask = fixed_attr > 0.5
        self._fixed_attr = fixed_attr.copy()

        asset = Path(__file__).resolve().parent.parent / "assets" / "bunny.obj"
        faces = _load_faces(asset)
        self._faces = faces
        edges = _unique_edges(faces)
        self._ea = edges[:, 0]
        self._eb = edges[:, 1]

        # Inverse masses; fixed crown nodes are Dirichlet (w = 0).
        self._w = np.where(self._fixed_mask, 0.0, 1.0 / np.maximum(self._mass, 1e-12))

        # Rest lengths and per-edge stiffness/compliance.
        d0 = x0[self._eb] - x0[self._ea]
        self._rest_len = np.linalg.norm(d0, axis=1)
        k_edge = _E_EFFECTIVE * np.maximum(self._rest_len, 1e-9)
        self._alpha = 1.0 / k_edge                      # compliance [m/N]
        self._lam = np.zeros(len(edges), dtype=np.float64)

        wsum = self._w[self._ea] + self._w[self._eb]
        self._active = wsum > 0.0                       # skip fixed-fixed edges
        self._mixed = self._active & (
            self._fixed_mask[self._ea] != self._fixed_mask[self._eb]
        )
        # For mixed edges identify the free endpoint.
        a_free = ~self._fixed_mask[self._ea]
        self._free_idx = np.where(a_free, self._ea, self._eb)
        self._free_sign = np.where(a_free, -1.0, 1.0)   # impulse sign on free node

        # Reference triangle data for the deformation gradient F = Ds * pinv(Dm).
        pa, pb, pc = x0[faces[:, 0]], x0[faces[:, 1]], x0[faces[:, 2]]
        dm = np.stack([pb - pa, pc - pa], axis=2)       # (Ne,3,2)
        self._dm_pinv = np.linalg.pinv(dm)              # (Ne,2,3)

        # Masses and totals.
        self._mass_total = float(self._mass.sum())
        self._mass_fixed = float(self._mass[self._fixed_mask].sum())

        # Frozen positive reference momentum scale: M * sqrt(g * L_pendulum).
        attachment = np.array([0.0, 0.0, 2.85])
        com = float(np.dot(self._mass, x0[:, 2]) / max(self._mass_total, 1e-12))
        l_pend = max(abs(attachment[2] - com), 1e-3)
        g_norm = float(np.linalg.norm(self._gravity))
        self._ref_momentum = self._mass_total * np.sqrt(g_norm * l_pend)

        # Time / budget state.
        self._t = 0.0
        self._impulse_grav = np.zeros(3)
        self._impulse_react = np.zeros(3)
        self._interval_react = np.zeros(3)
        self._initialized = True

    # ------------------------------------------------------------- stepping
    def _substep(self, h: float) -> None:
        x, v, w = self._x, self._v, self._w
        grav_dt = self._gravity * h

        # Cumulative gravity impulse on every node of the domain.
        self._impulse_grav += self._mass_total * grav_dt

        # Semi-implicit prediction.
        v += grav_dt
        x_prev = x + v * h
        x_next = x_prev.copy()

        alpha_t = self._alpha / (h * h)
        lam = np.zeros_like(self._lam)
        ea, eb = self._ea, self._eb
        wa, wb = w[ea], w[eb]
        rest = self._rest_len
        active = self._active
        n_nodes = len(x)

        react_imp = np.zeros(3)
        for _ in range(_ITERATIONS):
            d = x_next[eb] - x_next[ea]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            dist_safe = np.where(dist > 1e-12, dist, 1e-12)
            nrm = d / dist_safe[:, None]
            c = dist - rest
            denom = wa + wb + alpha_t
            dlam = np.where(active, (-c - alpha_t * lam) / denom, 0.0)
            lam += dlam
            # Position corrections.
            corr_a = (wa * dlam)[:, None] * nrm
            corr_b = (wb * dlam)[:, None] * nrm
            xa = np.zeros((n_nodes, 3))
            xb = np.zeros((n_nodes, 3))
            for j in range(3):
                xa[:, j] = np.bincount(ea, weights=-corr_a[:, j], minlength=n_nodes)
                xb[:, j] = np.bincount(eb, weights=corr_b[:, j], minlength=n_nodes)
            x_next += xa + xb
            # Impulse delivered to free nodes of mixed edges (reaction source).
            mix = self._mixed
            if np.any(mix):
                imp = (dlam[mix] / h)[:, None] * nrm[mix]
                imp *= self._free_sign[mix][:, None]
                react_imp += imp.sum(axis=0)

        # Velocity update from projected positions.
        v_new = (x_next - x_prev) / h
        v_new *= 1.0 / (1.0 + _DAMPING * h)
        # Dirichlet crown nodes.
        v_new[self._fixed_mask] = 0.0
        x_next[self._fixed_mask] = x[self._fixed_mask]

        self._x = x_next
        self._v = v_new

        # Attachment reaction impulse over this sub-step:
        #   cancel gravity on fixed nodes + constraint pull transmitted
        #   through mixed edges (equal and opposite on the attachment).
        self._impulse_react += -self._mass_fixed * grav_dt + react_imp
        self._interval_react += -self._mass_fixed * grav_dt + react_imp

    def _advance(self, target: float) -> None:
        remaining = target - self._t
        while remaining > 1e-12:
            h = min(_DT_SUB, remaining)
            self._substep(h)
            self._t += h
            remaining = target - self._t

    # -------------------------------------------------------------- framing
    def _deformation_gradients(self) -> np.ndarray:
        f = self._faces
        pa, pb, pc = self._x[f[:, 0]], self._x[f[:, 1]], self._x[f[:, 2]]
        ds = np.stack([pb - pa, pc - pa], axis=2)       # (Ne,3,2)
        return np.matmul(ds, self._dm_pinv)             # (Ne,3,3)

    def _make_frame(self, time: float, first: bool):
        from worldsolver.contracts import StateFrame

        momentum = self._mass[:, None] * self._v
        p_total = momentum.sum(axis=0)

        if first:
            # No elapsed interval yet: reactions just support the fixed patch.
            force_grav = self._mass_total * self._gravity
            force_react = -self._mass_fixed * self._gravity
        else:
            dt = max(time - self._last_time, 1e-9)
            force_grav = self._mass_total * self._gravity
            force_react = self._interval_react / dt

        external_force = np.stack([force_grav, force_react], axis=0)[None, :, :]
        cum_impulse = (self._impulse_grav + self._impulse_react)[None, :]
        zeros_g3 = np.zeros((1, 3))

        frame = StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={
                "fixed": self._fixed_attr.copy(),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "budget_domain_weights": np.ones((1, len(self._x))),
                "budget_external_force": external_force,
                "budget_cumulative_external_impulse": cum_impulse,
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3.copy(),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": self._deformation_gradients(),
            },
            scalars={
                "reference_momentum": float(self._ref_momentum),
                "momentum_total": float(np.linalg.norm(p_total)),
            },
        )
        frame.validate()
        self._last_time = float(time)
        self._interval_react = np.zeros(3)
        return frame

    def advance_to(self, time: float):
        time = float(time)
        first = not hasattr(self, "_last_time")
        if time > self._t + 1e-12:
            self._advance(time)
        return self._make_frame(time, first)

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "XPBD compliant edge-constraint elastic solid",
            "youngs_modulus_effective_pa": _E_EFFECTIVE,
            "substep_s": _DT_SUB,
            "iterations": _ITERATIONS,
            "nodes": int(len(self._x)) if hasattr(self, "_x") else 0,
            "edges": int(len(self._ea)) if hasattr(self, "_ea") else 0,
            "elements": int(len(self._faces)) if hasattr(self, "_faces") else 0,
            "fixed_nodes": int(self._fixed_mask.sum()) if hasattr(self, "_fixed_mask") else 0,
        }
