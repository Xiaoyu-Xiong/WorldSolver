"""Physical solver for the wet porous plug / funnel scene.

Implements a particle-based two-way coupled simulation:
- The porous plug is a Lagrangian elastic network built from the provided OBJ topology.
- Water is a particle fluid with short-range repulsive pressure, solved with a
type spatial hash.
- Funnel contact uses a signed-distance field built from the axisymmetric profile.
- Wetting softens the plug material progressively as water contacts it.

The solver advances with explicit symplectic Euler sub-steps and exports every
required trajectory field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.config = scene.config
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Substepping / sampling parameters.
        sample_dt = float(self.config["sample_dt"])
        self.dt = 1.0e-3
        self.substeps = max(1, int(round(sample_dt / self.dt)))
        self.dt = sample_dt / self.substeps  # adjust to exactly divide sample_dt

        # Initial state from the scaffold.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.phase_id = np.asarray(frame0.attributes["phase_id"], dtype=float).copy()
        self.saturation = np.asarray(frame0.attributes["saturation"], dtype=float).copy()
        self.porosity = np.asarray(frame0.attributes["porosity"], dtype=float).copy()
        self.deformation_gradient = np.asarray(
            frame0.attributes["deformation_gradient"], dtype=float
        ).copy()
        self.external_force_field = np.asarray(
            frame0.fields["external_force"], dtype=float
        ).copy()

        # Scalar passthrough + reference momentum.
        self.frame_scalars = dict(frame0.scalars)
        self._set_reference_momentum()

        # Domain masks.
        self.plug_mask = self.body_id == 1
        self.water_mask = self.body_id == 2
        self.plug_idx = np.nonzero(self.plug_mask)[0]
        self.water_idx = np.nonzero(self.water_mask)[0]
        self.num_groups = 3  # 0: plug, 1: water, 2: total

        # Per-sample mass split into material columns: [solid, liquid].
        n = len(self.mass)
        self.mass_quantity = np.zeros((n, 2), dtype=float)
        self.mass_quantity[self.plug_idx, 0] = self.mass[self.plug_idx]
        self.mass_quantity[self.water_idx, 1] = self.mass[self.water_idx]

        # Build the plug elastic edge network from the OBJ file.
        asset = Path(__file__).resolve().parent.parent / "assets" / "porous_plug.obj"
        verts, faces = self._parse_obj(asset)
        self.rest_pos = verts.astype(float)
        self.edges = self._build_edges(faces)
        if len(self.edges):
            p0 = self.rest_pos[self.edges[:, 0]]
            p1 = self.rest_pos[self.edges[:, 1]]
            self.rest_len = np.linalg.norm(p0 - p1, axis=1)
        else:
            self.rest_len = np.zeros(0)

        # Material/interaction parameters.
        plug_cfg = self.config["plug"]
        self.E_dry = float(plug_cfg["dry_young_modulus"])
        self.softening_factor = float(plug_cfg["wet_softening_factor"])
        self.wetting_rate = 0.6
        self.permeability = float(plug_cfg["permeability"])

        liquid_cfg = self.config["liquid"]
        self.rho0 = float(liquid_cfg["density"])
        self.dx = float(self.config["particle_spacing"])

        # Interaction cut-off and stiffnesses (tuned for m/s^2 units).
        self.cutoff = 2.0 * self.dx
        self.k_ww = 4.0e3   # water-water repulsion
        self.k_wp = 8.0e3   # water-plug repulsion
        self.k_wall = 2.0e4
        self.k_floor = 2.0e4
        self.c_pair = 20.0
        self.c_wall = 80.0
        self.c_floor = 80.0
        self.friction_mu = 0.6
        self.damping_global = 0.5  # mild velocity damping on the plug

        # Budget accumulators (groups: plug, water, total).
        self.cum_external_impulse = np.zeros((self.num_groups, 3), dtype=float)
        self.cum_inflow_momentum = np.zeros((self.num_groups, 3), dtype=float)
        self.cum_outflow_momentum = np.zeros((self.num_groups, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)

        self.current_time = 0.0
        self.diagnostics_data: dict[str, object] = {
            "dt": self.dt,
            "substeps_per_sample": self.substeps,
            "n_particles": n,
            "n_plug": int(self.plug_mask.sum()),
            "n_water": int(self.water_mask.sum()),
            "n_edges": len(self.edges),
        }

        # Small deterministic jitter to avoid perfectly aligned lattice artifacts.
        if len(self.water_idx):
            jitter = (rng.random((len(self.water_idx), 3)) - 0.5) * self.dx * 0.02
            self.position[self.water_idx] += jitter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        steps = int(round((time - self.current_time) / self.dt))
        for _ in range(steps):
            self._step()
        self.current_time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return self.diagnostics_data

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------
    def _step(self) -> None:
        pos = self.position
        vel = self.velocity

        # Accumulate forces.
        force = np.zeros_like(pos)

        # Gravity (also the per-sample external force field).
        force += self.mass[:, None] * self.gravity

        # Funnel and floor contact.
        wall_force, wall_pen = self._funnel_contact_force(pos, vel)
        force += wall_force
        floor_force = self._floor_contact_force(pos, vel)
        force += floor_force

        # Plug elastic forces.
        force += self._plug_elastic_force()

        # Water-water and water-plug repulsion.
        pair_force, pw_net = self._pair_forces(pos, vel)
        force += pair_force

        # Update budget accumulators before integrating.
        self._update_budgets(force, wall_force + floor_force, pw_net)

        # Integrate (symplectic Euler).
        # Apply mild global damping to the plug to suppress high-frequency modes.
        plug_idx = self.plug_idx
        vel[plug_idx] *= 1.0 - self.damping_global * self.dt

        acc = force / self.mass[:, None]
        vel += acc * self.dt
        pos += vel * self.dt

        # Update wetting.
        self._update_wetting()

        self.current_time += self.dt

    # ------------------------------------------------------------------
    # Force models
    # ------------------------------------------------------------------
    def _plug_elastic_force(self) -> np.ndarray:
        f = np.zeros_like(self.position)
        if len(self.edges) == 0:
            return f
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        pi = self.position[i]
        pj = self.position[j]
        dx = pi - pj
        r = np.linalg.norm(dx, axis=1)
        r = np.where(r < 1e-12, 1e-12, r)
        u = dx / r[:, None]

        # Wetness-dependent stiffness.
        s_avg = 0.5 * (self.saturation[i] + self.saturation[j])
        E = self.E_dry * (
            self.softening_factor
            + (1.0 - self.softening_factor) * (1.0 - s_avg)
        )
        # Cross-sectional area based on particle spacing, divided by rest length.
        A = self.dx * self.dx
        k = E * A / self.rest_len

        # Spring force along the edge.
        f_mag = k * (r - self.rest_len)
        f_edge = f_mag[:, None] * u

        # Add axial damping between the two endpoints.
        dv = self.velocity[i] - self.velocity[j]
        vdot = np.einsum("ij,ij->i", dv, u)
        f_damp = 2.0 * np.sqrt(k * self.mass[i]) * vdot
        f_edge += f_damp[:, None] * u

        np.add.at(f, i, f_edge)
        np.add.at(f, j, -f_edge)
        return f

    def _pair_forces(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return total force array and the net plug<->water force vector."""
        f = np.zeros_like(pos)
        if cKDTree is None:
            return f, np.zeros(3)

        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=self.cutoff, output_type="ndarray")
        if pairs.size == 0:
            return f, np.zeros(3)

        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = pos[i] - pos[j]
        r = np.linalg.norm(dx, axis=1)
        mask = r > 1e-12
        i = i[mask]
        j = j[mask]
        dx = dx[mask]
        r = r[mask]
        u = dx / r[:, None]

        bi = self.body_id[i]
        bj = self.body_id[j]
        ww = (bi == 2) & (bj == 2)
        wp = ((bi == 1) & (bj == 2)) | ((bi == 2) & (bj == 1))

        mag = np.zeros(len(r))
        mag[ww] = self.k_ww * (self.cutoff - r[ww])
        mag[wp] = self.k_wp * (self.cutoff - r[wp])

        # Repulsion acts along the line connecting the two particles.
        f_rep = mag[:, None] * u

        # Damping for closing velocities.
        dv = vel[i] - vel[j]
        vdot = np.einsum("ij,ij->i", dv, u)
        closing = np.minimum(0.0, vdot)
        f_damp = -self.c_pair * closing
        f_rep += f_damp[:, None] * u

        np.add.at(f, i, f_rep)
        np.add.at(f, j, -f_rep)

        # Net force on the plug exerted by water (used for interaction budget).
        # For each water-plug pair, the force on the plug is the force applied to
        # the plug particle. The pair force above is the force on i (pointing
        # from j to i). Recompute the plug-side contribution directly.
        pw = wp
        f_pw = np.zeros((len(r), 3), dtype=float)
        m_pw = self.k_wp * (self.cutoff - r[pw])
        f_pw[pw] = (m_pw[:, None] * u[pw]) + (-self.c_pair * closing[pw])[:, None] * u[pw]
        # Identify which index is the plug particle.
        plug_is_i = (bi == 1)
        # Force on plug from water is f_pw when plug is i, otherwise -f_pw.
        pw_force_i = f_pw[plug_is_i]
        pw_force_j = -f_pw[~plug_is_i]
        net = pw_force_i.sum(axis=0) + pw_force_j.sum(axis=0)
        return f, net

    def _funnel_contact_force(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Contact force and signed distance/penetration for the axisymmetric funnel."""
        profile = np.asarray(self.config["funnel"]["inner_profile"], dtype=float)
        z_nodes = profile[:, 1]
        # Only apply wall for z inside the profile range.
        z = pos[:, 2]
        in_range = (z >= z_nodes[0]) & (z <= z_nodes[-1])
        # Segment index for each point.
        seg = np.searchsorted(z_nodes, z, side="right") - 1
        seg = np.clip(seg, 0, len(z_nodes) - 2)

        r0 = profile[seg, 0]
        z0 = profile[seg, 1]
        r1 = profile[seg + 1, 0]
        z1 = profile[seg + 1, 1]
        dz_seg = z1 - z0
        dr_seg = r1 - r0
        t = np.where(dz_seg > 0, (z - z0) / dz_seg, 0.0)
        t = np.clip(t, 0.0, 1.0)
        r_wall = r0 + t * dr_seg

        xy = pos[:, :2]
        r = np.linalg.norm(xy, axis=1)
        # Signed distance (positive inside the container).
        d = r - r_wall
        pen = np.maximum(0.0, -d)

        # Unit normal pointing into the fluid domain.
        L = np.sqrt(dr_seg * dr_seg + dz_seg * dz_seg)
        L = np.where(L < 1e-12, 1.0, L)
        nr = dz_seg / L
        nz = -dr_seg / L
        r_safe = np.where(r > 1e-12, r, 1.0)
        cos = np.where(r > 1e-12, xy[:, 0] / r_safe, 1.0)
        sin = np.where(r > 1e-12, xy[:, 1] / r_safe, 0.0)
        normal = np.column_stack((cos * nr, sin * nr, nz))

        f = np.zeros_like(pos)
        active = in_range & (pen > 0.0)
        if not np.any(active):
            return f, d

        v_n = np.einsum("ij,ij->i", vel, normal)
        f_n = self.k_wall * pen - self.c_wall * np.minimum(0.0, v_n)
        f_n = np.where(active, f_n, 0.0)

        f[active] += f_n[active, None] * normal[active]

        # Coulomb friction along the wall surface.
        v_t = vel - v_n[:, None] * normal
        v_t_mag = np.linalg.norm(v_t, axis=1)
        slip = v_t_mag > 1e-12
        if np.any(slip & active):
            dir_t = v_t / np.where(v_t_mag[:, None] > 1e-12, v_t_mag[:, None], 1.0)
            f_t = -self.friction_mu * f_n
            f_t = np.where((slip & active), f_t, 0.0)
            f[slip & active] += f_t[slip & active, None] * dir_t[slip & active]
        return f, d

    def _floor_contact_force(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Catch-floor contact at z = 0.06."""
        f = np.zeros_like(pos)
        floor_z = 0.06
        pen = floor_z - pos[:, 2]
        active = pen > 0.0
        if not np.any(active):
            return f
        v_n = vel[:, 2]
        f_n = self.k_floor * pen - self.c_floor * np.minimum(0.0, v_n)
        f[active, 2] += f_n[active]

        # Tangential friction on the floor.
        v_t = vel[:, :2].copy()
        v_t_mag = np.linalg.norm(v_t, axis=1)
        slip = (v_t_mag > 1e-12) & active
        if np.any(slip):
            dir_t = v_t / np.where(v_t_mag[:, None] > 1e-12, v_t_mag[:, None], 1.0)
            f_t = -self.friction_mu * f_n
            f[slip, :2] += f_t[slip, None] * dir_t[slip]
        return f

    def _update_wetting(self) -> None:
        """Increase plug saturation when water particles are nearby."""
        if cKDTree is None or len(self.water_idx) == 0 or len(self.plug_idx) == 0:
            return
        water_pos = self.position[self.water_idx]
        plug_pos = self.position[self.plug_idx]
        tree = cKDTree(water_pos)
        # Count nearby water particles for each plug particle.
        dist, _ = tree.query(plug_pos, k=1, distance_upper_bound=self.cutoff * 1.5)
        dist = np.asarray(dist)
        close = dist < self.cutoff * 1.5
        if not np.any(close):
            return
        # Saturation rises with proximity.
        self.saturation[self.plug_idx[close]] += self.wetting_rate * self.dt * (
            1.0 + (self.cutoff * 1.5 - dist[close]) / (self.cutoff * 1.5)
        )
        np.clip(self.saturation, 0.0, 1.0, out=self.saturation)

    # ------------------------------------------------------------------
    # Budgets and required fields
    # ------------------------------------------------------------------
    def _update_budgets(
        self,
        total_force: np.ndarray,
        wall_force: np.ndarray,
        plug_water_force: np.ndarray,
    ) -> None:
        # Domain membership weights [G, N].
        weights = np.zeros((self.num_groups, len(self.mass)), dtype=float)
        weights[0, self.plug_idx] = 1.0
        weights[1, self.water_idx] = 1.0
        weights[2, :] = 1.0

        # External force per domain, split into gravity and wall/floor terms.
        # shape [G, K, 3] with K=2.
        budget_force = np.zeros((self.num_groups, 2, 3), dtype=float)
        gravity = self.mass[:, None] * self.gravity
        for g in range(self.num_groups):
            w = weights[g]
            budget_force[g, 0] = (w[:, None] * gravity).sum(axis=0)
            budget_force[g, 1] = (w[:, None] * wall_force).sum(axis=0)

        total_external = budget_force.sum(axis=1)  # [G, 3]
        self.cum_external_impulse += total_external * self.dt

        # Interaction force pairs (plug-water, plug-funnel, water-funnel).
        # plug-water is computed from pair repulsion (force on plug from water).
        plug_wall = wall_force[self.plug_mask].sum(axis=0)
        water_wall = wall_force[self.water_mask].sum(axis=0)
        self._interaction = np.vstack(
            [plug_water_force, plug_wall, water_wall]
        ).astype(float)

        self._budget_domain_weights = weights
        self._budget_external_force = budget_force
        self._budget_wall_force = wall_force

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        # Ensure scalar saturation/porosity are bounded.
        self.saturation = np.clip(self.saturation, 0.0, 1.0)
        # Copy dynamic arrays to avoid exposing internal state.
        fields = {
            "external_force": self.external_force_field.copy(),
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
            "budget_domain_weights": self._budget_domain_weights.copy(),
            "budget_external_force": self._budget_external_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self._interaction.copy(),
        }
        attributes = {
            "phase_id": self.phase_id.copy(),
            "saturation": self.saturation.copy(),
            "porosity": self.porosity.copy(),
            "deformation_gradient": self.deformation_gradient.copy(),
            "mass_quantity": self.mass_quantity.copy(),
        }
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=self.frame_scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_reference_momentum(self) -> None:
        """Set a positive reference momentum scale."""
        g = np.linalg.norm(self.gravity)
        L = float(self.config["plug"]["height"])
        total_m = float(self.mass.sum())
        ref = max(1.0, total_m * np.sqrt(g * L))
        self.frame_scalars["reference_momentum"] = ref

    def _parse_obj(self, path: Path) -> tuple[np.ndarray, list[list[int]]]:
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(idx)
        return np.asarray(vertices, dtype=float), faces

    def _build_edges(self, faces: Sequence[Sequence[int]]) -> np.ndarray:
        edges: set[tuple[int, int]] = set()
        for face in faces:
            # Triangulate n-gons by fan from the first vertex.
            verts = list(face)
            n = len(verts)
            for k in range(n - 2):
                tri = [verts[0], verts[k + 1], verts[k + 2]]
                for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    if a == b:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    edges.add(edge)
        if not edges:
            return np.zeros((0, 2), dtype=int)
        return np.asarray(sorted(edges), dtype=int)
