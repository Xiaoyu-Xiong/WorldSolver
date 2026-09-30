"""Mass-spring cloth solver for the 036_cloth_stack_5mm scene.

Ten square panels fall under gravity, collide with a board and each other,
and settle into a frictional multilayer stack.  The model uses an explicit
symplectic Euler time step with in-plane edge springs, bending springs,
velocity damping and stiff contact penalties.  No Genesis code is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver implementing the WorldSolver Solver protocol."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------
    def initialize(self, scene: object, seed: int) -> None:
        self.config = getattr(scene, "config", {}) if scene is not None else {}
        init = scene.initial_state()

        self.x = np.asarray(init.position, dtype=float).copy()
        self.v = np.asarray(init.velocity, dtype=float).copy()
        if self.x.shape != self.v.shape or self.x.ndim != 2 or self.x.shape[1] != 3:
            raise ValueError("position/velocity must be [N,3]")

        self.N = self.x.shape[0]
        self.n_panels = 10
        self.n_panel = self.N // self.n_panels
        if self.N != self.n_panels * self.n_panel:
            raise ValueError(f"expected {self.n_panels * self.n_panel} particles, got {self.N}")

        # body_id: one integer per panel, 0..9
        if init.body_id is not None:
            self.body_id = np.asarray(init.body_id, dtype=int)
        else:
            self.body_id = np.repeat(np.arange(self.n_panels, dtype=int), self.n_panel)

        # mass: equal distribution of the panel mass over its vertices
        panel_mass = float(self.config.get("panel", {}).get("total_mass", 0.12))
        if init.mass is not None:
            self.mass = np.asarray(init.mass, dtype=float)
        else:
            self.mass = np.full(self.N, panel_mass / self.n_panel, dtype=float)

        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.board_top = float(self.config.get("board_top", 0.05))
        self.thickness = float(self.config.get("panel", {}).get("thickness", 0.005))
        self.friction = float(self.config.get("friction", 0.1))

        # Load the shared panel topology and reference shape
        root = Path(__file__).resolve().parent.parent / "assets"
        topo = json.loads((root / "panel_topology.json").read_text(encoding="utf-8"))
        faces = np.asarray(topo["faces"], dtype=int)
        ref = np.asarray(topo["vertices"], dtype=float)

        edges, edge_rest = self._build_edges(faces, ref)
        bend_pairs, bend_rest = self._build_bending_pairs(faces, ref)

        # Tile topology across all panels
        self.edges = np.vstack([edges + b * self.n_panel for b in range(self.n_panels)])
        self.edge_rest = np.tile(edge_rest, self.n_panels)
        self.bend_pairs = np.vstack([bend_pairs + b * self.n_panel for b in range(self.n_panels)])
        self.bend_rest = np.tile(bend_rest, self.n_panels)

        # Material / numerical parameters
        self.k_stretch = 1.0e4
        self.k_bend = 5.0e2
        self.c_damp = 1.0

        self.k_contact = 5.0e3
        self.c_contact = 20.0
        self.contact_cutoff = self.thickness

        self.k_board = 1.0e4
        self.c_board = 20.0

        # Fixed sub-step chosen to divide the 1/30 s sample interval evenly
        self.dt = 1.0 / 2400.0
        self.time = float(init.time)

        self.constraint_scale = float(np.mean(edge_rest)) if edge_rest.size else 1.0
        self.contact_length_scale = self.thickness

        self._initialized = True

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("solver not initialized")

        target = float(time)
        if target < self.time:
            raise ValueError("target time is before current time")

        dt = self.dt
        while self.time < target:
            remaining = target - self.time
            if remaining < dt:
                self._substep(remaining)
            else:
                self._substep(dt)

        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "k_stretch": self.k_stretch,
            "k_bend": self.k_bend,
            "k_contact": self.k_contact,
            "time": self.time,
        }

    # ------------------------------------------------------------------
    # Substep
    # ------------------------------------------------------------------
    def _substep(self, h: float) -> None:
        # Start with gravitational body force
        F = self.mass[:, None] * self.gravity[None, :]

        # Membrane / stretch forces
        F += self._spring_force(self.x, self.v, self.edges, self.edge_rest, self.k_stretch)
        # Bending resistance (opposite-vertex springs)
        F += self._spring_force(self.x, self.v, self.bend_pairs, self.bend_rest, self.k_bend)

        # Particle-particle contact (self and inter-panel)
        F += self._particle_contact_force(self.x, self.v)

        # Global velocity damping
        F -= self.c_damp * self.v

        # Board contact force
        F += self._board_contact_force(self.x, self.v)

        # Integrate
        a = F / self.mass[:, None]
        self.v += h * a
        self.x += h * self.v

        # Hard projection / friction on the board
        self._enforce_board()

        self.time += h

    # ------------------------------------------------------------------
    # Force models
    # ------------------------------------------------------------------
    def _spring_force(
        self,
        x: np.ndarray,
        v: np.ndarray,
        pairs: np.ndarray,
        rest: np.ndarray,
        k: float,
    ) -> np.ndarray:
        """Return forces for an array of spring elements (i, j)."""
        if pairs.size == 0:
            return np.zeros_like(x)

        i = pairs[:, 0]
        j = pairs[:, 1]
        d = x[j] - x[i]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-12)

        strain = (L - rest) / rest
        f_mag = k * strain

        f_vec = f_mag[:, None] * (d / L[:, None])
        F = np.zeros_like(x)
        np.add.at(F, i, -f_vec)
        np.add.at(F, j, f_vec)
        return F

    def _particle_contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Penalty force for particle pairs closer than the cloth thickness."""
        tree = cKDTree(x)
        pairs = tree.query_pairs(r=self.contact_cutoff, output_type="ndarray")
        if pairs.shape[0] == 0:
            return np.zeros_like(x)

        i = pairs[:, 0]
        j = pairs[:, 1]
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1)
        mask = dist < self.contact_cutoff
        if not np.any(mask):
            return np.zeros_like(x)

        i = i[mask]
        j = j[mask]
        d = d[mask]
        dist = dist[mask]
        n = d / dist[:, None]
        pen = self.contact_cutoff - dist

        dv = v[j] - v[i]
        v_rel = np.sum(dv * n, axis=1)
        f_mag = self.k_contact * pen - self.c_contact * v_rel
        f_mag = np.maximum(f_mag, 0.0)  # repulsive only
        f_vec = f_mag[:, None] * n

        F = np.zeros_like(x)
        np.add.at(F, i, -f_vec)
        np.add.at(F, j, f_vec)
        return F

    def _board_contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Soft penalty preventing cloth from crossing the board top."""
        F = np.zeros_like(x)
        pen = self.board_top - x[:, 2]
        mask = pen > 0.0
        if np.any(mask):
            F[mask, 2] += self.k_board * pen[mask]
            F[mask, 2] -= self.c_board * v[mask, 2]
        return F

    def _enforce_board(self) -> None:
        """Project particles onto the board and apply Coulomb friction."""
        mask = self.x[:, 2] < self.board_top
        if not np.any(mask):
            return

        self.x[mask, 2] = self.board_top
        self.v[mask, 2] = 0.0

        vt = self.v[mask, :2]
        vtn = np.linalg.norm(vt, axis=1)
        if np.any(vtn > 0.0):
            # Tangential deceleration from friction: |dv_t| = mu * g * dt
            dv_t = self.friction * 9.81 * self.dt
            scale = np.maximum(0.0, 1.0 - dv_t / vtn)
            self.v[mask, :2] = (vt.T * scale).T

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_edges(self, faces: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_set = set()
        for f in faces:
            a, b, c = f
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(b, c), max(b, c)))
            edge_set.add((min(c, a), max(c, a)))
        edges = np.array(sorted(edge_set), dtype=int)
        d = ref[edges[:, 1]] - ref[edges[:, 0]]
        rest = np.linalg.norm(d, axis=1)
        return edges, rest

    def _build_bending_pairs(
        self, faces: np.ndarray, ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create springs between the two opposite vertices of each interior edge."""
        opposite: dict[tuple[int, int], list[int]] = {}
        for f in faces:
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                key = (min(a, b), max(a, b))
                c = f[(k + 2) % 3]
                opposite.setdefault(key, []).append(c)

        pairs = []
        for _key, opp in opposite.items():
            if len(opp) == 2:
                pairs.append((opp[0], opp[1]))
        pairs = np.array(pairs, dtype=int)
        if pairs.size == 0:
            return pairs, np.array([], dtype=float)
        d = ref[pairs[:, 0]] - ref[pairs[:, 1]]
        rest = np.linalg.norm(d, axis=1)
        return pairs, rest

    def _make_frame(self, time: float) -> StateFrame:
        # Constraint data: edge lengths vs. rest lengths
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        edge_lengths = np.linalg.norm(self.x[j] - self.x[i], axis=1)

        # Signed distance to the board (negative means penetration)
        contact_signed_distance = self.x[:, 2] - self.board_top

        # No mass added/removed during the simulation
        cumulative_mass_source = np.array([0.0], dtype=float)

        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "constraint_target": self.edge_rest.copy(),
                "constraint_value": edge_lengths,
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
