"""BenchmarkSolver: frictional stacking of 31 spaghetti strands in a plate.

Implements an explicit mass-spring chain with bending resistance, gravity,
strand-plate contact (axisymmetric container), and strand-strand contact with
Coulomb friction. The plate is treated as a rigid static boundary described by
its inner_profile polyline revolved about the z-axis.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Plate geometry helpers
# ---------------------------------------------------------------------------

_PLATE_INNER_PROFILE = np.array(
    [[0.10, 0.000], [0.40, 0.018], [0.52, 0.075], [0.57, 0.125]], dtype=float
)
_WALL_THICKNESS = 0.025


def _plate_floor_height(r: float) -> float:
    """Return floor z for a given cylindrical radius r (linear interp)."""
    profile = _PLATE_INNER_PROFILE
    if r <= profile[0, 0]:
        return profile[0, 1]
    if r >= profile[-1, 0]:
        # Outside rim: clamp to rim height (no support beyond rim).
        return profile[-1, 1]
    # Linear segment search
    for i in range(len(profile) - 1):
        r0, z0 = profile[i]
        r1, z1 = profile[i + 1]
        if r0 <= r <= r1:
            t = (r - r0) / (r1 - r0)
            return z0 + t * (z1 - z0)
    return profile[-1, 1]


def _plate_normal_at(r: float, z: float) -> np.ndarray:
    """Approximate outward unit normal of the plate surface at (r,z).

    For the bowl interior we use the local slope of the profile; on the flat
    bottom the normal is +z. Returns a 3-vector in (r_dir, 0, z).
    """
    profile = _PLATE_INNER_PROFILE
    # Bottom region
    if r <= profile[0, 0] + 1e-9:
        return np.array([0.0, 0.0, 1.0])
    # Find segment
    for i in range(len(profile) - 1):
        r0, z0 = profile[i]
        r1, z1 = profile[i + 1]
        if r0 <= r <= r1 + 1e-9:
            dr = r1 - r0
            dz = z1 - z0
            # Tangent along surface: (dr, dz). Outward normal points toward
            # increasing r and decreasing z for this concave-up bowl.
            nr = dz
            nz = -dr
            n = np.array([nr, 0.0, nz])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n /= norm
            return n
    # Beyond rim: treat as vertical wall pointing inward
    return np.array([-1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=int)

        meta = scene.metadata()
        self.strand_count = int(meta.get("strand_count", 31))
        self.nodes_per_strand = int(meta.get("nodes_per_strand", 16))
        self.strand_length = float(meta.get("strand_length", 0.72))
        self.strand_radius = float(meta.get("strand_radius", 0.006))
        self.gravity = np.asarray(meta.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        contact = meta.get("contact", {})
        ss = contact.get("strand_strand", {})
        sp = contact.get("strand_plate", {})
        self.mu_ss = float(ss.get("static_friction", 0.3))
        self.mu_sp = float(sp.get("static_friction", 0.3))

        # Segment rest length (uniform spacing from initial state)
        n_seg = self.nodes_per_strand - 1
        self.rest_length = self.strand_length / n_seg

        # Elastic parameters – tuned for dry spaghetti-like behaviour
        # Linear density already encoded in mass; choose stiffness so that
        # axial waves are stiff but integrable with small explicit dt.
        linear_density = 0.010  # kg/m (matches scene.py)
        self.EA = 50.0  # axial stiffness N
        self.EI = 2.0e-4  # bending stiffness N*m^2

        # Damping (proportional to mass) for settling
        self.damping = 0.6  # 1/s

        # Sub-stepping
        self.dt_physics = 2.0e-5
        self.stick_tol = 1.0e-4  # m/s stick-slip tolerance

        # Budget accumulators
        self._cum_ext_impulse = np.zeros((1, 3), dtype=float)
        self._cum_inflow_mom = np.zeros((1, 3), dtype=float)
        self._cum_outflow_mom = np.zeros((1, 3), dtype=float)
        self._cum_mass_src = np.zeros((1,), dtype=float)

        # Current time
        self.time = 0.0

        # Domain weights: single global domain, weight=1 for every sample
        self._domain_weights = np.ones((1, len(self.mass)), dtype=float)

        # Reference scales
        total_mass = float(np.sum(self.mass))
        g_mag = float(np.linalg.norm(self.gravity))
        self.ref_force = max(total_mass * g_mag, 1.0e-9)
        self.ref_velocity = max(math.sqrt(g_mag * self.strand_length), 1.0e-9)
        self.ref_momentum = max(total_mass * self.ref_velocity, 1.0e-9)

        # Friction contact cache (rebuilt each advance_to call)
        self._friction_contacts: list[tuple] = []

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _step(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity
        mass = self.mass
        body_id = self.body_id
        N = len(pos)
        ns = self.nodes_per_strand
        sc = self.strand_count

        # Accumulate forces
        force = np.zeros_like(pos)

        # Gravity
        force += mass[:, None] * self.gravity[None, :]

        # Linear damping
        force -= self.damping * mass[:, None] * vel

        # Axial spring forces along each strand
        EA = self.EA
        L0 = self.rest_length
        for s in range(sc):
            sl = slice(s * ns, (s + 1) * ns)
            xs = pos[sl]
            for i in range(ns - 1):
                d = xs[i + 1] - xs[i]
                L = np.linalg.norm(d)
                if L < 1e-14:
                    continue
                f_mag = EA * (L - L0) / L0
                f_vec = f_mag * (d / L)
                force[s * ns + i] += f_vec
                force[s * ns + i + 1] -= f_vec

        # Bending forces (discrete Euler-Bernouville via angle springs)
        EI = self.EI
        for s in range(sc):
            base = s * ns
            for i in range(1, ns - 1):
                p0 = pos[base + i - 1]
                p1 = pos[base + i]
                p2 = pos[base + i + 1]
                e1 = p1 - p0
                e2 = p2 - p1
                l1 = np.linalg.norm(e1)
                l2 = np.linalg.norm(e2)
                if l1 < 1e-14 or l2 < 1e-14:
                    continue
                t1 = e1 / l1
                t2 = e2 / l2
                cos_th = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
                # Rest angle = 0 (straight), so theta_rest = pi (cos=-1? no)
                # Actually for straight rod, consecutive tangents align: theta=0 => cos=1
                # Bending moment ~ EI * kappa; kappa ~ theta / L_avg
                theta = math.acos(cos_th)
                # Discrete bending energy: E_b = EI/(2*L_avg) * theta^2
                # Moment M = dE/dtheta = EI/L_avg * theta
                L_avg = 0.5 * (l1 + l2)
                if L_avg < 1e-14:
                    continue
                M = EI * theta / L_avg
                # Torque direction: perpendicular to both tangents
                cross = np.cross(t1, t2)
                cross_norm = np.linalg.norm(cross)
                if cross_norm < 1e-14:
                    # Nearly collinear – apply small perturbation in arbitrary perp dir
                    # Choose a vector perpendicular to t1
                    arb = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(arb, t1)) > 0.9:
                        arb = np.array([0.0, 1.0, 0.0])
                    axis = np.cross(t1, arb)
                    axis /= np.linalg.norm(axis)
                else:
                    axis = cross / cross_norm
                # Apply equal-and-opposite torques to adjacent segments
                # Convert torque to nodal forces via virtual work
                # For simplicity: apply restoring forces that reduce theta
                # F on middle node pushes it toward straight config
                # Use finite-difference style: perturb p1 and compute gradient
                eps = 1e-8
                # Numerical gradient of theta w.r.t. p0, p1, p2
                grad = np.zeros((3, 3))
                for node_local, pn in enumerate((p0, p1, p2)):
                    for dim in range(3):
                        pp = np.copy(pn)
                        pp[dim] += eps
                        pts = [p0.copy(), p1.copy(), p2.copy()]
                        pts[node_local] = pp
                        e1p = pts[1] - pts[0]
                        e2p = pts[2] - pts[1]
                        l1p = np.linalg.norm(e1p)
                        l2p = np.linalg.norm(e2p)
                        if l1p < 1e-14 or l2p < 1e-14:
                            continue
                        c = float(np.clip(np.dot(e1p / l1p, e2p / l2p), -1.0, 1.0))
                        th_p = math.acos(c)
                        grad[node_local, dim] = (th_p - theta) / eps
                # Force = -dE/dx = -(EI/L_avg)*theta * d(theta)/dx
                f_bend_0 = -M * grad[0]
                f_bend_1 = -M * grad[1]
                f_bend_2 = -M * grad[2]
                force[base + i - 1] += f_bend_0
                force[base + i] += f_bend_1
                force[base + i + 1] += f_bend_2

        # Contact: strand-plate
        contacts_this_step: list[tuple] = []
        mu_plate = self.mu_sp
        radius = self.strand_radius
        k_contact = 500.0  # penalty stiffness N/m
        for idx in range(N):
            x, y, z = pos[idx]
            r = math.sqrt(x * x + y * y)
            h_floor = _plate_floor_height(r)
            pen = (h_floor + radius) - z
            if pen > 0:
                # Normal at contact
                n_local = _plate_normal_at(r, z)
                # Map to 3D: radial direction is (x,y,0)/r
                if r > 1e-12:
                    rad = np.array([x / r, y / r, 0.0])
                else:
                    rad = np.array([1.0, 0.0, 0.0])
                normal = n_local[0] * rad + n_local[2] * np.array([0.0, 0.0, 1.0])
                nn = np.linalg.norm(normal)
                if nn < 1e-12:
                    normal = np.array([0.0, 0.0, 1.0])
                else:
                    normal /= nn
                fn_mag = k_contact * pen
                f_n = fn_mag * normal
                force[idx] += f_n

                # Friction
                v_rel = vel[idx].copy()  # plate is static
                vn = np.dot(v_rel, normal)
                v_tang = v_rel - vn * normal
                vt_mag = np.linalg.norm(v_tang)
                if vt_mag < self.stick_tol:
                    # Stick: oppose tangential velocity
                    f_t = -k_contact * v_tang * dt  # impulse-like
                    ft_mag = np.linalg.norm(f_t)
                    if ft_mag > mu_plate * fn_mag:
                        f_t = -mu_plate * fn_mag * (v_tang / max(vt_mag, 1e-14))
                else:
                    f_t = -mu_plate * fn_mag * (v_tang / vt_mag)
                force[idx] += f_t
                contacts_this_step.append((idx, -1, normal, fn_mag, v_rel, f_t, mu_plate, mu_plate))

        # Contact: strand-strand (only nearby nodes to keep O(N^2) tractable)
        mu_ss = self.mu_ss
        two_r = 2.0 * radius
        for a in range(N):
            xa = pos[a]
            ba = body_id[a]
            for b in range(a + 1, N):
                if body_id[b] == ba:
                    continue
                d = xa - pos[b]
                dist2 = float(np.dot(d, d))
                if dist2 > (two_r * 3) ** 2:
                    continue
                dist = math.sqrt(dist2)
                pen = two_r - dist
                if pen <= 0:
                    continue
                if dist < 1e-14:
                    continue
                normal = d / dist
                fn_mag = k_contact * pen
                f_n = fn_mag * normal
                force[a] += f_n
                force[b] -= f_n

                v_rel = vel[a] - vel[b]
                vn = np.dot(v_rel, normal)
                v_tang = v_rel - vn * normal
                vt_mag = np.linalg.norm(v_tang)
                if vt_mag < self.stick_tol:
                    f_t = -k_contact * v_tang * dt
                    ft_mag = np.linalg.norm(f_t)
                    if ft_mag > mu_ss * fn_mag:
                        f_t = -mu_ss * fn_mag * (v_tang / max(vt_mag, 1e-14))
                else:
                    f_t = -mu_ss * fn_mag * (v_tang / vt_mag)
                force[a] += f_t
                force[b] -= f_t
                contacts_this_step.append((a, b, normal, fn_mag, v_rel, f_t, mu_ss, mu_ss))

        # Semi-implicit Euler
        acc = force / mass[:, None]
        vel += acc * dt
        pos += vel * dt

        # Update external impulse accumulator (gravity only)
        grav_impulse = np.sum(mass[:, None] * self.gravity[None, :], axis=0) * dt
        self._cum_ext_impulse[0] += grav_impulse

        self._friction_contacts = contacts_this_step

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt_physics, target - self.time)
            if dt <= 0:
                break
            self._step(dt)
            self.time += dt

        # Build friction arrays from last-step contacts
        contacts = self._friction_contacts
        C = len(contacts)
        if C == 0:
            fric_active = np.zeros(0, dtype=float)
            fric_mu_s = np.zeros(0, dtype=float)
            fric_mu_k = np.zeros(0, dtype=float)
            fric_normal = np.zeros((0, 3), dtype=float)
            fric_fn = np.zeros(0, dtype=float)
            fric_vrel = np.zeros((0, 3), dtype=float)
            fric_ft = np.zeros((0, 3), dtype=float)
        else:
            fric_active = np.ones(C, dtype=float)
            fric_mu_s = np.array([c[6] for c in contacts], dtype=float)
            fric_mu_k = np.array([c[7] for c in contacts], dtype=float)
            fric_normal = np.array([c[2] for c in contacts], dtype=float)
            fric_fn = np.array([c[3] for c in contacts], dtype=float)
            fric_vrel = np.array([c[4] for c in contacts], dtype=float)
            fric_ft = np.array([c[5] for c in contacts], dtype=float)

        N = len(self.position)
        # External force per domain per term: just gravity (1 term)
        ext_force = np.zeros((1, 1, 3), dtype=float)
        ext_force[0, 0, :] = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)

        frame = StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self._cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self._cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self._cum_outflow_mom.copy(),
                "budget_domain_weights": self._domain_weights.copy(),
                "budget_external_force": ext_force,
                "cumulative_mass_source": self._cum_mass_src.copy(),
                "friction_active_mask": fric_active,
                "friction_dynamic_coefficient": fric_mu_k,
                "friction_normal": fric_normal,
                "friction_normal_force": fric_fn,
                "friction_relative_velocity": fric_vrel,
                "friction_static_coefficient": fric_mu_s,
                "friction_tangential_force": fric_ft,
            },
            scalars={
                "friction_stick_speed_tolerance": self.stick_tol,
                "reference_force": self.ref_force,
                "reference_momentum": self.ref_momentum,
                "reference_velocity": self.ref_velocity,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "dt_physics": self.dt_physics,
            "EA": self.EA,
            "EI": self.EI,
            "damping": self.damping,
        }
