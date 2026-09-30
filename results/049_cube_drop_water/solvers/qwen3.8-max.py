"""Physically-based solver for scene 049_cube_drop_water.

Water: weakly-compressible SPH (cubic spline kernel, Tait equation of state,
Monaghan artificial viscosity, XSPH position smoothing).
Cube: rigid body with quaternion orientation, gravity, and fluid reaction.
Coupling: Becker-style rigid-fluid coupling -- fluid particles penetrating the
cube are projected onto the nearest cube face, receive the local rigid-body
velocity, and the momentum change is applied back to the cube (giving drag,
splash deflection, and buoyancy).
Tank: closed floor/side walls handled by position clamping with zero normal
restitution (no penetration, no sticking); open top.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

G = np.array([0.0, 0.0, -9.81])

# --- SPH parameters ---------------------------------------------------------
DP = 0.12                       # particle spacing (scene grid)
H = 1.5 * DP                    # kernel support radius
H2 = H * H
SIGMA3 = 1.0 / (np.pi * H ** 3)  # 3D cubic spline normalization
C0 = 20.0                       # artificial sound speed
RHO0 = 1000.0
B_TAIT = RHO0 * C0 * C0 / 7.0
ALPHA_V = 0.05                  # artificial viscosity coefficient
XSPH_EPS = 0.2
DT = 0.001
CONTACT_BAND = 0.2              # signed-distance reporting band (m)

# Tank interior: planes given as (inward normal n, offset c) with s = n.p + c
_PLANES_N = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
])
_PLANES_C = np.array([1.38, 1.38, 1.08, 1.08, 0.0, 2.5])
_LO = np.array([-1.38, -1.08, 0.0])
_HI = np.array([1.38, 1.08, 2.5])

_OFFS = np.array([(i, j, k) for i in (-1, 0, 1)
                  for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)


def _quat_to_rot(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        self.seed = int(seed)

        pos = np.array(frame0.position, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.mass = np.array(frame0.mass, dtype=float)

        self.n_water = int(np.sum(self.body_id == 1))
        self.cube_idx = self.n_water  # single rigid sample appended at the end

        wc = cfg["initial_conditions"]["cube"]
        self.cube_pos = np.array(wc["initial_position"], dtype=float)
        self.cube_vel = np.array(wc["initial_velocity"], dtype=float)
        self.cube_quat = np.array(wc["initial_orientation_xyzw"], dtype=float)
        self.cube_quat /= np.linalg.norm(self.cube_quat)
        self.cube_omega = np.zeros(3)
        self.cube_mass = float(self.mass[self.cube_idx])
        a = 0.5 * float(wc["side_length"])
        self.cube_half = np.array([a, a, a])
        I_diag = self.cube_mass * (2 * a) ** 2 / 6.0
        self.cube_I_inv_local = np.eye(3) / I_diag

        self.wpos = np.array(pos[:self.n_water], dtype=float)
        self.wvel = np.array(frame0.velocity[:self.n_water], dtype=float)
        self.wmass = self.mass[:self.n_water]
        self.rho = np.full(self.n_water, RHO0)

        self.time = 0.0
        self._interaction_accum = np.zeros(3)
        self._last_frame_time = 0.0
        self._diag = {"steps": 0}

    # ------------------------------------------------------------- neighbors
    def _build_pairs(self):
        pos = self.wpos
        ci = np.floor((pos - _LO) / H).astype(np.int64) + 4
        keys = (ci[:, 0] * 1024 + ci[:, 1]) * 1024 + ci[:, 2]
        order = np.argsort(keys, kind="stable")
        ukeys, starts = np.unique(keys[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        c0 = ukeys // 1048576
        c1 = (ukeys // 1024) % 1024
        c2 = ukeys % 1024

        I_list = []
        J_list = []
        for off in _OFFS:
            nk = ((c0 + off[0]) * 1024 + (c1 + off[1])) * 1024 + (c2 + off[2])
            idx = np.searchsorted(ukeys, nk)
            idx_c = np.clip(idx, 0, len(ukeys) - 1)
            hit = ukeys[idx_c] == nk
            if not hit.any():
                continue
            src = np.nonzero(hit)[0]
            loc = idx_c[src]
            cnt = ends[loc] - starts[loc]
            base = np.repeat(np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
            rep = np.repeat(src, cnt)
            dst = np.repeat(starts[loc], cnt) + (np.arange(len(rep)) - base)
            I_list.append(order[rep])
            J_list.append(order[dst])
        if I_list:
            I = np.concatenate(I_list)
            J = np.concatenate(J_list)
            dx = pos[I] - pos[J]
            r2 = np.einsum("ij,ij->i", dx, dx)
            keep = (r2 < H2) & (r2 > 1e-12)
            self.pair_I = I[keep]
            self.pair_J = J[keep]
        else:
            self.pair_I = np.zeros(0, dtype=np.int64)
            self.pair_J = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------- SPH forces
    def _sph_step(self):
        nw = self.n_water
        I, J = self.pair_I, self.pair_J
        acc = np.zeros((nw, 3))
        vavg = np.zeros((nw, 3))
        if len(I) > 0:
            dx = self.wpos[I] - self.wpos[J]
            r2 = np.einsum("ij,ij->i", dx, dx)
            r = np.sqrt(r2)
            q = r / H
            W = SIGMA3 * (1.0 - 1.5 * q * q + 0.75 * q ** 3)
            dWdr = SIGMA3 / H * (-3.0 * q + 2.25 * q * q)
            mj = self.wmass[J]

            # density (sum formulation; self term included as m_i * W(0))
            rho = self.wmass.copy()
            np.add.at(rho, I, mj * W)
            np.add.at(rho, np.arange(nw), self.wmass * SIGMA3)
            self.rho = rho

            # Tait pressure, clamped non-negative (free surface stability)
            P = B_TAIT * (np.maximum((rho / RHO0) ** 7, 1.0) - 1.0)

            rhoI = rho[I]
            rhoJ = rho[J]
            rho_bar = 0.5 * (rhoI + rhoJ)
            gW = (dWdr / np.maximum(r, 1e-9))[:, None] * dx  # grad_i W_ij

            # pressure gradient force
            coef = mj * (P[I] / rhoI ** 2 + P[J] / rhoJ ** 2)
            np.add.at(acc, I, -coef[:, None] * gW)

            # artificial viscosity
            dv = self.wvel[I] - self.wvel[J]
            vdotr = np.einsum("ij,ij->i", dv, dx)
            mu = H * vdotr / (r2 + 0.01 * H2)
            Pi = np.where(vdotr < 0.0, (-ALPHA_V * C0 * mu) / rho_bar, 0.0)
            np.add.at(acc, I, -(Pi * mj)[:, None] * gW)

            # XSPH velocity smoothing (used for position update only)
            np.add.at(vavg, I,
                      mj[:, None] * (self.wvel[J] - self.wvel[I])
                      / rho_bar[:, None] * W[:, None])

        acc += G[None, :]
        self.wvel += acc * DT
        self.wpos += (self.wvel + XSPH_EPS * vavg) * DT

        # hard clamp against tank walls/floor: no penetration, no sticking
        for k in range(3):
            lo_hit = self.wpos[:, k] < _LO[k]
            if lo_hit.any():
                self.wpos[lo_hit, k] = _LO[k]
                np.maximum(self.wvel[lo_hit, k], 0.0, out=self.wvel[lo_hit, k])
            hi_hit = self.wpos[:, k] > _HI[k]
            if hi_hit.any():
                self.wpos[hi_hit, k] = _HI[k]
                np.minimum(self.wvel[hi_hit, k], 0.0, out=self.wvel[hi_hit, k])

    # -------------------------------------------------------------- coupling
    def _couple_cube(self):
        # rigid-body integration (gravity; coupling impulse applied below)
        self.cube_vel = self.cube_vel + G * DT
        self.cube_pos = self.cube_pos + self.cube_vel * DT

        wx, wy, wz = self.cube_omega
        x, y, z, w = self.cube_quat
        qd = 0.5 * np.array([
            w * wx + wy * z - wz * y,
            w * wy + wz * x - wx * z,
            w * wz + wx * y - wy * x,
            -(wx * x + wy * y + wz * z),
        ])
        self.cube_quat = self.cube_quat + qd * DT
        self.cube_quat /= np.linalg.norm(self.cube_quat)
        self.cube_omega *= (1.0 - 0.5 * DT)  # mild angular damping

        R = _quat_to_rot(self.cube_quat)

        # Becker-style coupling: project penetrating fluid particles onto the
        # nearest cube face, assign the local rigid velocity, react on the cube.
        rel = self.wpos - self.cube_pos[None, :]
        loc = rel @ R                                  # world -> local
        al = np.abs(loc)
        inside = np.all(al <= self.cube_half[None, :], axis=1)
        imp_lin = np.zeros(3)
        imp_ang = np.zeros(3)
        if inside.any():
            ii = np.nonzero(inside)[0]
            li = loc[ii]
            pen = self.cube_half[None, :] - np.abs(li)
            axis = np.argmin(pen, axis=1)
            rows = np.arange(len(ii))
            newl = li.copy()
            newl[rows, axis] = np.where(
                li[rows, axis] >= 0.0,
                self.cube_half[axis], -self.cube_half[axis])
            delta = (newl - li) @ R.T                  # local -> world
            self.wpos[ii] += delta
            r_world = self.wpos[ii] - self.cube_pos[None, :]
            v_rigid = (self.cube_vel[None, :]
                       + np.cross(np.broadcast_to(self.cube_omega, r_world.shape),
                                  r_world))
            dnorm = np.linalg.norm(delta, axis=1)
            nrm = delta / np.maximum(dnorm, 1e-12)[:, None]
            vn_fluid = np.einsum("ij,ij->i", self.wvel[ii], nrm)
            vn_rigid = np.einsum("ij,ij->i", v_rigid, nrm)
            # never reduce an already-faster outward normal motion
            v_new = v_rigid + np.maximum(0.0, vn_fluid - vn_rigid)[:, None] * nrm
            dp = self.wmass[ii, None] * (v_new - self.wvel[ii])  # impulse on fluid
            imp_lin = -dp.sum(axis=0)                            # reaction impulse
            imp_ang = -np.cross(r_world, dp).sum(axis=0)
            self.wvel[ii] = v_new

        self.cube_vel += imp_lin / self.cube_mass
        I_inv_w = R @ self.cube_I_inv_local @ R.T
        self.cube_omega += I_inv_w @ imp_ang

        # keep the cube inside the tank
        ext = np.abs(R) @ self.cube_half
        self.cube_pos = np.clip(self.cube_pos, _LO + ext, _HI - ext)
        self._interaction_accum += imp_lin
        return imp_lin

    # -------------------------------------------------------------- advancing
    def _substep(self):
        self._build_pairs()
        self._sph_step()
        self._couple_cube()
        self.time += DT
        self._diag["steps"] += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_steps = int(round((target - self.time) / DT))
        self._interaction_accum = np.zeros(3)
        self._last_frame_time = self.time
        for _ in range(max(n_steps, 0)):
            self._substep()
        return self._make_frame(target)

    # ------------------------------------------------------------------ frame
    def _make_frame(self, t: float) -> StateFrame:
        nw = self.n_water
        n = nw + 1
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        position[:nw] = self.wpos
        velocity[:nw] = self.wvel
        position[nw] = self.cube_pos
        velocity[nw] = self.cube_vel

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0
        orientation[nw] = self.cube_quat
        angular_velocity = np.zeros((n, 3))
        angular_velocity[nw] = self.cube_omega

        mass_quantity = np.zeros((n, 2))
        mass_quantity[:nw, 0] = self.wmass
        mass_quantity[nw, 1] = self.cube_mass

        external_force = self.mass[:, None] * G[None, :]

        dt_interval = max(t - self._last_frame_time, DT)
        f_on_cube = self._interaction_accum / dt_interval
        interaction_force = np.zeros((3, 3))
        interaction_force[0] = -f_on_cube   # water <- cube
        interaction_force[1] = f_on_cube    # cube  <- water
        interaction_force[2] = 0.0          # boundary pair (clamp, no adhesion)

        # contact signed distances: water near walls + cube vs each plane
        s_w = self.wpos @ _PLANES_N.T + _PLANES_C[None, :]
        s_min_w = s_w.min(axis=1)
        near = s_min_w < CONTACT_BAND
        R = _quat_to_rot(self.cube_quat)
        support = np.abs(R) @ self.cube_half
        sup_per_plane = np.array([support[0], support[0], support[1],
                                  support[1], support[2], support[2]])
        s_cube = (_PLANES_N @ self.cube_pos + _PLANES_C) - sup_per_plane
        contact = np.concatenate((s_min_w[near], s_cube))

        fields = {
            "contact_signed_distance": contact,
            "cumulative_mass_source": np.zeros(2),
            "external_force": external_force,
            "interaction_force": interaction_force,
        }
        scalars = {
            "contact_length_scale": float(DP),
            "water_rest_density": float(RHO0),
            "cube_density": 400.0,
        }
        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"phase": np.concatenate((np.zeros(nw), [1.0])),
                        "mass_quantity": mass_quantity},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return dict(self._diag)
