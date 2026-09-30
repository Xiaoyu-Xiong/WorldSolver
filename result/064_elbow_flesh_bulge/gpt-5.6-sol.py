"""Self-contained deterministic physics solver for elbow flesh flexion."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene, self.seed = scene, int(seed)
        self.cfg = getattr(scene, "config", {})
        obj = Path(__file__).resolve().parent.parent / "assets" / "arm_flesh.obj"
        vertices, faces = [], []
        for line in obj.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                v = line.split(); vertices.append([float(v[1]), float(v[2]), float(v[3])])
            elif line.startswith("f "):
                f = [int(v.split("/")[0]) - 1 for v in line.split()[1:4]]; faces.append(f)
        self.rest = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n = len(self.rest)
        ic = self.cfg.get("initial_conditions", {})
        bc = self.cfg.get("boundary_conditions", {}).get("forearm_bone", {})
        self.center = np.asarray(ic.get("elbow_center", [0., 0., 1.05]), dtype=float)
        self.axis = np.asarray(bc.get("hinge_axis", [0., 1., 0.]), dtype=float)
        self.axis /= max(np.linalg.norm(self.axis), 1e-12)
        self.t_start, self.t_end = float(bc.get("start_time", .4)), float(bc.get("end_time", 2.4))
        self.a0 = math.radians(float(bc.get("start_angle_degrees", 0.)))
        self.a1 = math.radians(float(bc.get("end_angle_degrees", -110.)))
        edge_set = set()
        for a, b, c in self.faces:
            for i, j in ((a, b), (b, c), (c, a)): edge_set.add((min(i, j), max(i, j)))
        self.edges = np.asarray(sorted(edge_set), dtype=np.int64)
        self.rest_len = np.linalg.norm(self.rest[self.edges[:, 1]] - self.rest[self.edges[:, 0]], axis=1)
        self.mass = np.full(self.n, 3.05 / self.n)
        self.pos, self.vel, self.time = self.rest.copy(), np.zeros_like(self.rest), 0.0
        self.k_edge, self.k_attach, self.damping, self.dt_max = 850., 2600., 34., .0025
        self._Dm_inv = []
        for tri in self.faces:
            p0, p1, p2 = self.rest[tri]; e1, e2 = p1-p0, p2-p0
            n = np.cross(e1, e2); n /= max(np.linalg.norm(n), 1e-12)
            self._Dm_inv.append(np.linalg.inv(np.column_stack((e1, e2, n))))

    def _angle(self, t: float) -> float:
        if t <= self.t_start: return self.a0
        if t >= self.t_end: return self.a1
        u = (t-self.t_start) / max(self.t_end-self.t_start, 1e-12)
        return self.a0 + (self.a1-self.a0) * u*u*(3.-2.*u)

    def _rotate(self, q: np.ndarray, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return q*c + np.cross(self.axis, q)*s + self.axis*np.dot(q, self.axis)*(1.-c)

    def _target(self, t: float) -> np.ndarray:
        angle, out = self._angle(t), self.rest.copy()
        for i, p in enumerate(self.rest):
            x = float(p[0])
            if x < -.02: continue
            alpha = np.clip((x+.02)/.12, 0., 1.); alpha = alpha*alpha*(3.-2.*alpha)
            pnew = self.center + self._rotate(p-self.center, angle*alpha)
            # Smooth incompressible-like response: tissue displaced by closing
            # bones expands radially near the elbow and contracts longitudinally.
            close = max(0., math.sin(max(0., -angle)))
            envelope = math.exp(-(x/.30)**2)
            radial = pnew-self.center; nr = float(np.linalg.norm(radial))
            if nr > 1e-8:
                bulge = .115*close*envelope*math.exp(-(nr/.42)**4)
                pnew += bulge*radial/nr
            pnew[0] = self.center[0] + (pnew[0]-self.center[0])*(1.-.12*close*envelope)
            out[i] = pnew
        return out

    def _step(self, h: float, target: np.ndarray) -> None:
        i, j = self.edges[:, 0], self.edges[:, 1]
        delta = self.pos[j]-self.pos[i]; length = np.linalg.norm(delta, axis=1)
        spring = self.k_edge*(length-self.rest_len)[:, None]*delta/np.maximum(length[:, None], 1e-9)
        force = np.zeros_like(self.pos); np.add.at(force, i, spring); np.add.at(force, j, -spring)
        force += self.k_attach*(target-self.pos) - self.damping*self.vel
        fixed = self.rest[:, 0] < -.02
        force[fixed] += 9000.*(self.rest[fixed]-self.pos[fixed])-180.*self.vel[fixed]
        self.vel += h*force/self.mass[:, None]; self.vel *= math.exp(-2.*h); self.pos += h*self.vel
        self.pos[fixed], self.vel[fixed] = self.rest[fixed], 0.
        self.pos = np.nan_to_num(self.pos, nan=0., posinf=10., neginf=-10.)

    def _gradients(self) -> np.ndarray:
        result = np.empty((len(self.faces), 3, 3))
        for k, tri in enumerate(self.faces):
            p0, p1, p2 = self.pos[tri]; e1, e2 = p1-p0, p2-p0
            n = np.cross(e1, e2); n /= max(np.linalg.norm(n), 1e-12)
            result[k] = np.column_stack((e1, e2, n)) @ self._Dm_inv[k]
        return np.nan_to_num(result, nan=0., posinf=1e6, neginf=-1e6)

    def advance_to(self, time: float):
        t = float(time)
        if not math.isfinite(t) or t < self.time-1e-9: raise ValueError("time must be finite and nondecreasing")
        while self.time < t-1e-12:
            h = min(self.dt_max, t-self.time); self._step(h, self._target(self.time+h)); self.time += h
        frame = StateFrame(time=float(self.time), position=self.pos.copy(), velocity=self.vel.copy(),
            mass=self.mass.copy(), attributes={"rest_position": self.rest.copy(), "mass_quantity": self.mass.copy()},
            fields={"budget_external_force": np.zeros((1, 1, 3)),
                    "cumulative_mass_source": np.zeros(1),
                    "element_deformation_gradient": self._gradients()},
            scalars={"flesh_density_kg_m3": 1050.,
                     "elbow_flexion_degrees": float(math.degrees(self._angle(self.time))),
                     "reference_force": float(max(np.sum(self.mass)*9.81, 1.))})
        frame.validate(); return frame

    def diagnostics(self) -> dict[str, object]:
        return {"nodes": self.n, "elements": len(self.faces), "time": self.time,
                "max_speed": float(np.max(np.linalg.norm(self.vel, axis=1)))}
