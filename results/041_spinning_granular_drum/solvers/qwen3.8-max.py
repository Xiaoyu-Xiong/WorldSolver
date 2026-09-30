"""Physically-based DEM solver for the spinning granular drum (041).

Soft-sphere discrete element method:
  - grain-grain contacts: linear normal spring-dashpot + regularized Coulomb friction
  - rotating cylindrical drum wall (axis = y): kinematic wall with frictional drag
    that lifts grains, producing avalanches down the free surface
  - lateral (y) containment planes matching the drum interior
  - gravity, mild local damping, symplectic Euler time stepping
Neighbor search uses a uniform spatial hash grid rebuilt every step (vectorized).
"""
from __future__ import annotations

import itertools

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        frame0 = scene.initial_state()
        self._frame0 = frame0
        meta = scene.metadata()

        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.n = self.pos.shape[0]
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.rad = np.array(frame0.attributes["radius"], dtype=np.float64)
        self.r0 = float(self.rad[0])
        self.invm = 1.0 / self.mass

        grav = np.asarray(meta.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.g = grav
        self.gforce = self.mass[:, None] * grav[None, :]

        drum = scene.config["initial_conditions"]["drum"]
        center = np.asarray(drum["center"], dtype=np.float64)
        self.axis_pt = np.array([center[0], 0.0, center[2]])  # point on y-axis
        self.R = float(drum["radius"])                 # inner wall radius
        self.omega = float(drum["angular_velocity_rad_s"])
        depth = float(drum["depth"])
        self.y_lo = -0.5 * depth - 0.015 + self.r0     # containment planes
        self.y_hi = 0.5 * depth + 0.015 - self.r0
        self.y_lo_c = -0.5 * depth - 0.015
        self.y_hi_c = 0.5 * depth + 0.015

        # contact parameters (stable for dt below)
        self.kn = 2.0e4
        self.cn = 30.0
        self.kn_w = 4.0e4
        self.cn_w = 50.0
        self.mu = 0.5          # grain-grain friction
        self.mu_w = 0.65       # grain-wall friction (drives lifting)
        self.vt_eps = 1.0e-3   # tangential regularization
        self.drag = 0.15       # mild local velocity damping (1/s)

        self.dt = 4.0e-4
        self.t = 0.0

        # spatial hash grid
        cell = 2.2 * self.r0 + 2.0e-3
        self.cell = cell
        p = self.pos
        self.gmin = np.array([-0.95, -0.35, self.axis_pt[2] - 0.95])
        gmax = np.array([0.95, 0.35, self.axis_pt[2] + 0.95])
        self.gdims = np.ceil((gmax - self.gmin) / cell).astype(np.int64) + 1
        self.cut = 2.0 * self.r0 + 1.5e-3
        self.cut2 = self.cut * self.cut

        # 13 half offsets + zero offset for pair enumeration
        offs = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]
        self.offsets = [(0, 0, 0)] + offs

        self.escaped = 0
        self.drum_angle = 0.0

    # ------------------------------------------------------------------ grid
    def _cell_keys(self):
        d = self.gdims
        idx = np.floor((self.pos - self.gmin) / self.cell).astype(np.int64)
        idx = np.clip(idx, 0, d - 1)
        return (idx[:, 0] * d[1] + idx[:, 1]) * d[2] + idx[:, 2]

    def _neighbor_pairs(self, keys):
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        d = self.gdims
        sy = int(d[2]); sxy = int(d[1] * d[2])
        src_all, dst_all = [], []
        n = self.n
        arange_n = np.arange(n)
        for ox, oy, oz in self.offsets:
            shifted = keys + ox * sxy + oy * sy + oz
            lo = np.searchsorted(skeys, shifted, "left")
            hi = np.searchsorted(skeys, shifted, "right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            src = np.repeat(arange_n, cnt)
            starts = np.repeat(lo, cnt)
            within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            dst = order[starts + within]
            if ox == 0 and oy == 0 and oz == 0:
                keep = src < dst
                src, dst = src[keep], dst[keep]
            if src.size:
                src_all.append(src)
                dst_all.append(dst)
        if src_all:
            return np.concatenate(src_all), np.concatenate(dst_all)
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))

    # ------------------------------------------------------------------ step
    def _step(self):
        dt = self.dt
        pos, vel = self.pos, self.vel
        forces = self.gforce.copy()
        # mild local damping
        forces -= (self.drag * self.mass)[:, None] * vel

        keys = self._cell_keys()
        src, dst = self._neighbor_pairs(keys)

        # ---------------- grain-grain contacts
        if src.size:
            dvec = pos[src] - pos[dst]
            d2 = np.einsum("ij,ij->i", dvec, dvec)
            m = d2 < self.cut2
            if m.any():
                src, dst = src[m], dst[m]
                dvec, d2 = dvec[m], d2[m]
                dist = np.sqrt(np.maximum(d2, 1e-16))
                nrm = dvec / dist[:, None]
                delta = 2.0 * self.r0 - dist
                np.clip(delta, 0.0, None, out=delta)
                vrel = vel[src] - vel[dst]
                vn = np.einsum("ij,ij->i", vrel, nrm)
                vt = vrel - vn[:, None] * nrm
                fn = self.kn * delta - self.cn * vn
                np.clip(fn, 0.0, None, out=fn)
                vt_norm = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                ft_scale = -self.mu * fn / (vt_norm + self.vt_eps)
                f = fn[:, None] * nrm + ft_scale[:, None] * vt
                np.add.at(forces, src, f)
                np.add.at(forces, dst, -f)

        # ---------------- rotating cylindrical wall (axis along y)
        rx = pos[:, 0] - self.axis_pt[0]
        rz = pos[:, 2] - self.axis_pt[2]
        rho = np.sqrt(rx * rx + rz * rz)
        rho_s = np.maximum(rho, 1e-12)
        delta_w = rho + self.r0 - self.R
        cw = delta_w > 0.0
        if cw.any():
            nx = np.zeros(self.n); nz = np.zeros(self.n)
            nx[cw] = -rx[cw] / rho_s[cw]
            nz[cw] = -rz[cw] / rho_s[cw]
            # wall surface velocity: omega * yhat x r
            # wall surface velocity: omega * (yhat x r) = omega * (rz, 0, -rx)
            wvx = np.zeros(self.n); wvz = np.zeros(self.n)
            wvx[cw] = self.omega * rz[cw]
            wvz[cw] = -self.omega * rx[cw]
            rvx = vel[cw, 0] - wvx[cw]
            rvy = vel[cw, 1]
            rvz = vel[cw, 2] - wvz[cw]
            nxc, nzc = nx[cw], nz[cw]
            vn = rvx * nxc + rvz * nzc
            vtx = rvx - vn * nxc
            vty = rvy
            vtz = rvz - vn * nzc
            fn = self.kn_w * delta_w[cw] - self.cn_w * vn
            np.clip(fn, 0.0, None, out=fn)
            vt_norm = np.sqrt(vtx * vtx + vty * vty + vtz * vtz)
            s = -self.mu_w * fn / (vt_norm + self.vt_eps)
            fx = fn * nxc + s * vtx
            fz = fn * nzc + s * vtz
            fy = s * vty
            forces[cw, 0] += fx
            forces[cw, 1] += fy
            forces[cw, 2] += fz

        # ---------------- lateral containment planes (y = const)
        d_lo = self.y_lo - pos[:, 1]        # >0 when penetrating lower plane
        c1 = d_lo > 0.0
        if c1.any():
            vy = vel[c1, 1]
            fn = self.kn_w * d_lo[c1] - self.cn_w * vy  # n = +y, vn = vy
            np.clip(fn, 0.0, None, out=fn)
            vtx = vel[c1, 0]; vtz = vel[c1, 2]
            vt_norm = np.sqrt(vtx * vtx + vtz * vtz)
            s = -self.mu_w * fn / (vt_norm + self.vt_eps)
            forces[c1, 0] += s * vtx
            forces[c1, 1] += fn
            forces[c1, 2] += s * vtz
        d_hi = pos[:, 1] - self.y_hi
        c2 = d_hi > 0.0
        if c2.any():
            vy = vel[c2, 1]
            fn = self.kn_w * d_hi[c2] + self.cn_w * vy  # n = -y, vn = -vy
            np.clip(fn, 0.0, None, out=fn)
            vtx = vel[c2, 0]; vtz = vel[c2, 2]
            vt_norm = np.sqrt(vtx * vtx + vtz * vtz)
            s = -self.mu_w * fn / (vt_norm + self.vt_eps)
            forces[c2, 0] += s * vtx
            forces[c2, 1] += -fn
            forces[c2, 2] += s * vtz

        # ---------------- symplectic Euler integration
        self.vel = vel + forces * self.invm[:, None] * dt
        self.pos = pos + self.vel * dt

        # hard clamp so nothing can tunnel out of the drum (safety net)
        p = self.pos
        rx = p[:, 0] - self.axis_pt[0]
        rz = p[:, 2] - self.axis_pt[2]
        rho = np.sqrt(rx * rx + rz * rz)
        over = rho > (self.R - self.r0)
        if over.any():
            s = (self.R - self.r0) / np.maximum(rho[over], 1e-12)
            p[over, 0] = self.axis_pt[0] + rx[over] * s
            p[over, 2] = self.axis_pt[2] + rz[over] * s
            radial = np.zeros((int(over.sum()), 3))
            radial[:, 0] = rx[over]
            radial[:, 2] = rz[over]
            radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-12)[:, None]
            vr = np.einsum("ij,ij->i", self.vel[over], radial)
            if np.any(vr > 0):
                k = vr > 0
                idx = np.nonzero(over)[0][k]
                self.vel[idx] -= (vr[k] * 0.5)[:, None] * radial[k]
        np.clip(p[:, 1], self.y_lo_c + self.r0 * 0.5, self.y_hi_c - self.r0 * 0.5, out=p[:, 1])

        self.t += dt
        self.drum_angle += self.omega * dt
        self._last_forces = forces

    # ------------------------------------------------------------------ frame
    def _make_frame(self, time: float) -> StateFrame:
        f0 = self._frame0
        pos = self.pos
        rx = pos[:, 0] - self.axis_pt[0]
        rz = pos[:, 2] - self.axis_pt[2]
        rho = np.sqrt(rx * rx + rz * rz)
        gap = self.R - self.r0 - rho          # negative => wall penetration
        escaped = int(np.sum((rho + self.r0 > self.R + 2e-3) |
                             (pos[:, 1] < self.y_lo_c - 2e-3) |
                             (pos[:, 1] > self.y_hi_c + 2e-3)))
        self.escaped = escaped

        ext = self.gforce.copy()  # external force excludes internal contacts

        attrs = {
            "radius": self.rad.copy(),
            "particle_radius": self.rad.copy(),
            "material_label": np.ones(self.n, dtype=np.int32),
            "tracer_grain": np.array(f0.attributes["tracer_grain"], dtype=np.float64),
            "initial_volume": np.array(f0.attributes["initial_volume"], dtype=np.float64),
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "external_force": ext,
            "contact_signed_distance": gap,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
        }
        scalars = {
            "drum_radius": self.R,
            "drum_depth": float(f0.scalars["drum_depth"]),
            "drum_angular_velocity_rad_s": self.omega,
            "initial_particle_count": float(self.n),
            "contact_length_scale": 2.0 * self.r0,
            "drum_angle": self.drum_angle,
            "escaped_particle_count": float(escaped),
            "lifted_grain_height": float(np.max(pos[:, 2]) - self.axis_pt[2]),
            "time": float(self.t),
        }
        frame = StateFrame(
            time=float(time),
            position=pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.ones(self.n, dtype=np.int32),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        # step with fixed dt
        guard = 0
        while self.t < target - 1e-12 and guard < 200000:
            dt_left = target - self.t
            dt = self.dt
            if dt_left < dt:
                dt = dt_left
                old = self.dt
                self.dt = dt
                self._step()
                self.dt = old
            else:
                self._step()
            guard += 1
        return self._make_frame(max(target, 0.0))

    def diagnostics(self) -> dict:
        return {
            "particles": int(self.n),
            "drum_angle_rad": float(self.drum_angle),
            "escaped_particle_count": int(self.escaped),
            "sim_time": float(self.t),
            "dt": self.dt,
        }
