"""Volumetric elastic ball compressed between codimensional pin arrays.

Physical model
--------------
* Linear tetrahedral FEM on the frozen ball asset, lumped nodal mass from the
  frozen density.
* Stable Neo-Hookean elasticity ``P = mu*F + lam*(J - 1 - mu/lam)*cof(F)`` plus
  Kelvin-Voigt strain-rate viscosity ``P_d = beta*dF/dt``. Both are assembled
  antisymmetrically per element so internal forces sum to zero exactly.
* Codimensional obstacles: every pin segment is sampled into contact points
  that interact with the deformed surface triangles through a one-sided
  penalty layer (normal penalty + normal damping + regularized Coulomb
  friction) distributed to triangle nodes with barycentric weights. Ball nodes
  additionally collide with the pin capsules (frozen radius), so the ball is
  supported by the arrays instead of passing through them.
* Upper array follows the frozen piecewise-linear prescribed translation.
* Symplectic-Euler stepping with a fixed CFL-safe step; each sample interval is
  subdivided into equal substeps so sample times are hit exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    dt_max = 2.0e-4           # s
    young = 8.0e4             # Pa (soft rubber ball)
    poisson = 0.42
    kv_damping = 1.2e3        # Pa*s Kelvin-Voigt viscosity
    contact_k = 1.0e5         # N/m per pin sample point
    contact_c = 3.0e2         # N*s/m per pin sample point
    contact_layer = 0.25      # m one-sided penalty layer thickness
    contact_depth_cap = 0.06  # m penalty saturation depth
    friction_mu = 0.45
    friction_ct = 3.0e3       # N*s/m tangential regularization
    node_k = 2.0e5            # N/m node-vs-pin-capsule penalty
    node_c = 3.0e2            # N*s/m node-vs-pin-capsule damping
    node_margin = 0.004       # m clearance added to pin radius
    pin_sample_ds = 0.02      # m sampling distance along pin segments
    bbox_margin = 0.03        # m broad-phase padding

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.config = cfg
        self.rng = np.random.default_rng(int(seed))
        root = Path(getattr(scene, "_root", Path.cwd()))
        if not (root / "assets").is_dir():
            root = Path.cwd()
        self.root = root

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.rest_position = np.array(
            frame0.attributes.get("rest_position", frame0.position), dtype=float
        )
        self.n_nodes = int(len(self.position))

        ball = json.loads((root / "assets" / "soft_ball_volume.json").read_text("utf-8"))
        self.tets = np.asarray(ball["tetrahedra"], dtype=int)
        self.faces = np.asarray(ball.get("surface_triangles", ball["faces"]), dtype=int)

        # Reference element matrices; orient tets so det(Dm) > 0.
        X = self.rest_position
        tets = self.tets.copy()
        Dm = np.stack(
            (
                X[tets[:, 1]] - X[tets[:, 0]],
                X[tets[:, 2]] - X[tets[:, 0]],
                X[tets[:, 3]] - X[tets[:, 0]],
            ),
            axis=2,
        )
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip, 2], tets[flip, 3] = tets[flip, 3].copy(), tets[flip, 2].copy()
            self.tets = tets
            Dm = np.stack(
                (
                    X[tets[:, 1]] - X[tets[:, 0]],
                    X[tets[:, 2]] - X[tets[:, 0]],
                    X[tets[:, 3]] - X[tets[:, 0]],
                ),
                axis=2,
            )
            det = np.linalg.det(Dm)
        self.dm_inv = np.linalg.inv(Dm)
        self.rest_volume = np.abs(det) / 6.0
        self.n_elements = int(len(self.tets))

        # Orient surface triangles outward w.r.t. the rest centroid.
        centroid = X.mean(axis=0)
        faces = self.faces.copy()
        a, b, c = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
        nrm = np.cross(b - a, c - a)
        out = (a + b + c) / 3.0 - centroid
        bad = np.einsum("ij,ij->i", nrm, out) < 0.0
        if np.any(bad):
            faces[bad, 1], faces[bad, 2] = faces[bad, 2].copy(), faces[bad, 1].copy()
        self.faces = faces

        # ---- pin arrays (codimensional line-segment obstacles) ----
        geom = cfg.get("scene_geometry") or {}
        radii = {}
        for obj in geom.get("objects", []) or []:
            g = obj.get("geometry", {}) or {}
            if g.get("type") == "curve_network" and g.get("radius"):
                radii[obj.get("id")] = float(g["radius"])
        self.pin_radius = float(radii.get("lower_pin_array", 0.009))
        upper_radius = float(radii.get("upper_pin_array", self.pin_radius))
        self.pin_radius = 0.5 * (self.pin_radius + upper_radius)

        self.pin_groups = []
        for name, moving in (("lower_pins", False), ("upper_pins", True)):
            data = json.loads((root / "assets" / f"{name}.json").read_text("utf-8"))
            verts = np.asarray(data["vertices"], dtype=float)
            edges = np.asarray(data["edges"], dtype=int)
            p0 = verts[edges[:, 0]]
            p1 = verts[edges[:, 1]]
            samples = []
            for s, e in zip(p0, p1):
                length = float(np.linalg.norm(e - s))
                count = max(2, int(np.ceil(length / self.pin_sample_ds)) + 1)
                t = np.linspace(0.0, 1.0, count)[:, None]
                samples.append(s[None, :] * (1.0 - t) + e[None, :] * t)
            self.pin_groups.append(
                {
                    "moving": moving,
                    "p0": p0,
                    "p1": p1,
                    "samples": np.concatenate(samples, axis=0),
                }
            )

        # ---- prescribed motion of the upper array ----
        bc = cfg.get("boundary_conditions", {}) or {}
        upper = bc.get("upper_pin_array", {}) or {}
        axis = np.asarray(upper.get("axis", [0.0, 0.0, -1.0]), dtype=float)
        norm = float(np.linalg.norm(axis))
        self.press_axis = axis / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
        keys = upper.get("keyframes", []) or [{"time": 0.0, "displacement": 0.0}]
        self.key_t = np.asarray([float(k["time"]) for k in keys], dtype=float)
        self.key_d = np.asarray([float(k["displacement"]) for k in keys], dtype=float)
        order = np.argsort(self.key_t, kind="stable")
        self.key_t = self.key_t[order]
        self.key_d = self.key_d[order]

        # ---- material ----
        e_mod, nu = float(self.young), float(self.poisson)
        self.mu = e_mod / (2.0 * (1.0 + nu))
        self.lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.total_mass = float(self.mass.sum())
        self.gravity_force = self.total_mass * self.gravity

        # ---- accumulators / frozen scales ----
        self.time = 0.0
        self.steps = 0
        self.max_depth = 0.0
        self.impulse = np.zeros(3, dtype=float)
        self.contact_total = np.zeros(3, dtype=float)
        self.inflow = np.zeros((1, 3), dtype=float)
        self.outflow = np.zeros((1, 3), dtype=float)
        self.domain_weights = np.ones((1, self.n_nodes), dtype=float)
        self.mass_source = np.zeros(1, dtype=float)

        ball_z = float(np.asarray(
            cfg.get("initial_conditions", {}).get("ball_center", [0.0, 0.0, 1.0]),
            dtype=float,
        )[2])
        tip_z = float((bc.get("lower_pin_array", {}) or {}).get("tip_height", 0.0))
        drop = max(0.05, ball_z - tip_z)
        g_mag = max(1e-9, float(np.linalg.norm(self.gravity)))
        self.reference_momentum = float(self.total_mass * np.sqrt(2.0 * g_mag * drop))

    # ------------------------------------------------------- prescribed motion
    def _press(self, t: float):
        """Return (displacement vector, velocity vector) of the upper array."""
        tt = float(t)
        d = float(np.interp(tt, self.key_t, self.key_d))
        rate = 0.0
        if len(self.key_t) > 1:
            idx = int(np.searchsorted(self.key_t, tt, side="right")) - 1
            idx = min(max(idx, 0), len(self.key_t) - 2)
            dt_key = self.key_t[idx + 1] - self.key_t[idx]
            if dt_key > 0.0 and self.key_t[0] <= tt <= self.key_t[-1]:
                rate = float((self.key_d[idx + 1] - self.key_d[idx]) / dt_key)
        return self.press_axis * d, self.press_axis * rate

    # -------------------------------------------------------------- mechanics
    def _deformation_gradient(self, position):
        tets = self.tets
        Ds = np.stack(
            (
                position[tets[:, 1]] - position[tets[:, 0]],
                position[tets[:, 2]] - position[tets[:, 0]],
                position[tets[:, 3]] - position[tets[:, 0]],
            ),
            axis=2,
        )
        return Ds @ self.dm_inv

    def _internal_forces(self, position, velocity):
        """Neo-Hookean + Kelvin-Voigt nodal forces (sum to zero by construction)."""
        tets = self.tets
        F = self._deformation_gradient(position)
        f0, f1, f2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        c0 = np.cross(f1, f2)
        c1 = np.cross(f2, f0)
        c2 = np.cross(f0, f1)
        cof = np.stack((c0, c1, c2), axis=2)          # d(det F)/dF
        J = np.einsum("ij,ij->i", f0, c0)
        alpha = 1.0 + self.mu / max(self.lam, 1e-12)
        P = self.mu * F + (self.lam * (J - alpha))[:, None, None] * cof

        # strain-rate (Kelvin-Voigt) viscous stress
        Ls = np.stack(
            (
                velocity[tets[:, 1]] - velocity[tets[:, 0]],
                velocity[tets[:, 2]] - velocity[tets[:, 0]],
                velocity[tets[:, 3]] - velocity[tets[:, 0]],
            ),
            axis=2,
        )
        Fdot = Ls @ self.dm_inv
        P = P + self.kv_damping * Fdot

        H = -(self.rest_volume[:, None, None] * P) @ np.transpose(self.dm_inv, (0, 2, 1))
        forces = np.zeros_like(position)
        np.add.at(forces, tets[:, 1], H[:, :, 0])
        np.add.at(forces, tets[:, 2], H[:, :, 1])
        np.add.at(forces, tets[:, 3], H[:, :, 2])
        np.add.at(forces, tets[:, 0], -(H[:, :, 0] + H[:, :, 1] + H[:, :, 2]))
        return forces, F

    # ---------------------------------------------------------------- contact
    def _face_contact(self, position, velocity, points, point_vel, forces):
        """One-sided penalty layer between pin sample points and ball triangles.

        Each sample point that lies inside the inward prism of a surface
        triangle pushes that triangle outward-inverted (i.e. dents the ball) and
        the reaction supports the ball. Forces are distributed to the triangle
        nodes with barycentric weights, so momentum bookkeeping is exact.
        """
        total = np.zeros(3, dtype=float)
        if len(points) == 0:
            return total
        lo = position.min(axis=0) - self.bbox_margin
        hi = position.max(axis=0) + self.bbox_margin
        inside_box = np.all((points >= lo) & (points <= hi), axis=1)
        if not np.any(inside_box):
            return total
        pts = points[inside_box]

        faces = self.faces
        a = position[faces[:, 0]]
        b = position[faces[:, 1]]
        c = position[faces[:, 2]]
        nrm = np.cross(b - a, c - a)
        area2 = np.linalg.norm(nrm, axis=1)
        valid = area2 > 1e-12
        if not np.any(valid):
            return total
        unit = np.zeros_like(nrm)
        unit[valid] = nrm[valid] / area2[valid, None]

        # signed distance of every candidate point to every face plane
        rel = pts[:, None, :] - a[None, :, :]
        sd = np.einsum("pfk,fk->pf", rel, unit)
        depth = -sd                                  # >0 means inside the ball
        near = (depth > 0.0) & (depth < self.contact_layer) & valid[None, :]
        if not np.any(near):
            return total

        e1 = b - a
        e2 = c - a
        d11 = np.einsum("fk,fk->f", e1, e1)
        d12 = np.einsum("fk,fk->f", e1, e2)
        d22 = np.einsum("fk,fk->f", e2, e2)
        denom = d11 * d22 - d12 * d12
        denom = np.where(np.abs(denom) < 1e-16, 1e-16, denom)
        r1 = np.einsum("pfk,fk->pf", rel, e1)
        r2 = np.einsum("pfk,fk->pf", rel, e2)
        v = (d22[None, :] * r1 - d12[None, :] * r2) / denom[None, :]
        w = (d11[None, :] * r2 - d12[None, :] * r1) / denom[None, :]
        u = 1.0 - v - w
        tol = -0.02
        active = near & (u > tol) & (v > tol) & (w > tol)
        pi, fi = np.nonzero(active)
        if len(pi) == 0:
            return total

        wu = np.clip(u[pi, fi], 0.0, 1.0)
        wv = np.clip(v[pi, fi], 0.0, 1.0)
        ww = np.clip(w[pi, fi], 0.0, 1.0)
        wsum = wu + wv + ww
        wsum = np.where(wsum < 1e-9, 1.0, wsum)
        wu, wv, ww = wu / wsum, wv / wsum, ww / wsum

        n = unit[fi]
        dep = depth[pi, fi]
        self.max_depth = max(self.max_depth, float(dep.max()))
        dep_eff = np.minimum(dep, self.contact_depth_cap)

        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        surf_v = (
            wu[:, None] * velocity[i0]
            + wv[:, None] * velocity[i1]
            + ww[:, None] * velocity[i2]
        )
        pv = np.broadcast_to(point_vel, (len(points), 3))[inside_box][pi]
        rel_v = pv - surf_v
        closing = np.einsum("ij,ij->i", rel_v, n)     # >0: pin driving inward
        fn = self.contact_k * dep_eff + self.contact_c * np.maximum(closing, 0.0)
        fn = np.maximum(fn, 0.0)

        # ball surface is pushed along -n (dent inward, support outward)
        f_normal = -fn[:, None] * n
        v_t = rel_v - closing[:, None] * n
        speed = np.linalg.norm(v_t, axis=1)
        safe = np.maximum(speed, 1e-9)
        ft = np.minimum(self.friction_mu * fn, self.friction_ct * speed)
        f_fric = ft[:, None] * (v_t / safe[:, None])
        f_total = f_normal + f_fric

        np.add.at(forces, i0, wu[:, None] * f_total)
        np.add.at(forces, i1, wv[:, None] * f_total)
        np.add.at(forces, i2, ww[:, None] * f_total)
        total += f_total.sum(axis=0)
        return total

    def _node_capsule_contact(self, position, velocity, p0, p1, seg_vel, forces):
        """Ball nodes against pin capsules of the frozen radius."""
        total = np.zeros(3, dtype=float)
        lo = position.min(axis=0) - 0.05
        hi = position.max(axis=0) + 0.05
        keep = np.all(
            (np.maximum(p0, p1) >= lo) & (np.minimum(p0, p1) <= hi), axis=1
        )
        if not np.any(keep):
            return total
        s0 = p0[keep]
        s1 = p1[keep]
        d = s1 - s0
        dd = np.einsum("ij,ij->i", d, d)
        dd = np.where(dd < 1e-16, 1e-16, dd)
        rel = position[:, None, :] - s0[None, :, :]
        t = np.clip(np.einsum("nsk,sk->ns", rel, d) / dd[None, :], 0.0, 1.0)
        closest = s0[None, :, :] + t[:, :, None] * d[None, :, :]
        diff = position[:, None, :] - closest
        dist = np.linalg.norm(diff, axis=2)
        limit = self.pin_radius + self.node_margin
        ni, si = np.nonzero((dist < limit) & (dist > 1e-12))
        if len(ni) == 0:
            return total
        n = diff[ni, si] / dist[ni, si][:, None]
        pen = limit - dist[ni, si]
        rel_v = velocity[ni] - np.broadcast_to(seg_vel, (3,))[None, :]
        approach = -np.einsum("ij,ij->i", rel_v, n)
        fmag = np.maximum(self.node_k * pen + self.node_c * np.maximum(approach, 0.0), 0.0)
        f = fmag[:, None] * n
        np.add.at(forces, ni, f)
        total += f.sum(axis=0)
        return total

    def _contact_forces(self, time, position, velocity):
        forces = np.zeros_like(position)
        total = np.zeros(3, dtype=float)
        disp, vel = self._press(time)
        for group in self.pin_groups:
            if group["moving"]:
                pts = group["samples"] + disp[None, :]
                q0 = group["p0"] + disp[None, :]
                q1 = group["p1"] + disp[None, :]
                pv = vel
            else:
                pts = group["samples"]
                q0 = group["p0"]
                q1 = group["p1"]
                pv = np.zeros(3, dtype=float)
            total += self._face_contact(position, velocity, pts, pv, forces)
            total += self._node_capsule_contact(position, velocity, q0, q1, pv, forces)
        return forces, total

    # ------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        t_end = self.time + dt
        internal, _ = self._internal_forces(self.position, self.velocity)
        contact, contact_total = self._contact_forces(t_end, self.position, self.velocity)
        gravity = self.mass[:, None] * self.gravity[None, :]
        total = internal + contact + gravity
        self.velocity = self.velocity + dt * (total / self.mass[:, None])
        self.position = self.position + dt * self.velocity
        # external (gravity + obstacle contact) impulse on the ball domain
        self.impulse = self.impulse + dt * (self.gravity_force + contact_total)
        self.contact_total = contact_total
        self.contact_nodal = contact
        self.time = t_end
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        remaining = target - self.time
        if remaining > 1e-15:
            n = max(1, int(np.ceil(remaining / self.dt_max)))
            dt = remaining / n
            for _ in range(n):
                self._step(dt)
            self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------ output
    def _frame(self, time: float) -> StateFrame:
        contact_nodal = getattr(self, "contact_nodal", None)
        if contact_nodal is None:
            contact_nodal, contact_total = self._contact_forces(
                time, self.position, self.velocity
            )
        else:
            contact_total = self.contact_total
        F = self._deformation_gradient(self.position)
        gravity = self.mass[:, None] * self.gravity[None, :]

        budget_external_force = np.zeros((1, 2, 3), dtype=float)
        budget_external_force[0, 0] = gravity.sum(axis=0)
        budget_external_force[0, 1] = contact_total

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.position, dtype=float),
            velocity=np.ascontiguousarray(self.velocity, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=float),
                "rest_position": np.ascontiguousarray(self.rest_position, dtype=float),
            },
            fields={
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": budget_external_force,
                "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.inflow.copy(),
                "budget_cumulative_outflow_momentum": self.outflow.copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "element_deformation_gradient": np.ascontiguousarray(F, dtype=float),
                "external_force": np.ascontiguousarray(gravity + contact_nodal, dtype=float),
                "contact_force": np.ascontiguousarray(contact_nodal, dtype=float),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "density": float(self.config["initial_conditions"]["ball_density"]),
                "total_mass": float(self.total_mass),
                "upper_array_displacement": float(np.linalg.norm(self._press(time)[0])),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "dt_max": float(self.dt_max),
            "elements": int(self.n_elements),
            "nodes": int(self.n_nodes),
            "young_modulus": float(self.young),
            "poisson_ratio": float(self.poisson),
            "total_mass": float(self.total_mass),
            "max_contact_depth": float(self.max_depth),
            "final_center_z": float(self.position.mean(axis=0)[2]),
            "final_min_z": float(self.position[:, 2].min()),
            "final_max_radius_xy": float(
                np.linalg.norm(self.position[:, :2], axis=1).max()
            ),
            "reference_momentum": float(self.reference_momentum),
        }
