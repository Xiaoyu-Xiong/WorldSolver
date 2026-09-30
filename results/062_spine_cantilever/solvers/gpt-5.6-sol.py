"""Physical solver for the segmented soft cantilever."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.position = self.velocity = self.rest_position = self.mass = None
        self.fixed = self.gravity = None
        self.springs = []; self.boxes = []; self.current_time = 0.0
        self.reference_force = 1.0; self.scene = None

    @staticmethod
    def _vertices():
        path = Path(__file__).resolve().parents[1] / "assets" / "spine_arm.obj"
        return np.asarray([[float(v) for v in line.split()[1:4]]
                           for line in path.read_text().splitlines() if line.startswith("v ")], dtype=float)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        initial = scene.initial_state()
        self.rest_position = np.asarray(initial.position, float).copy()
        self.position = self.rest_position.copy(); self.velocity = np.zeros_like(self.position)
        self.mass = np.asarray(initial.mass, float).copy()
        self.gravity = np.asarray(scene.config["gravity"], float)
        self.fixed = np.asarray(initial.attributes.get("fixed", np.zeros(len(self.position))), bool)
        self.current_time = 0.0
        self.reference_force = max(float(np.sum(self.mass) * np.linalg.norm(self.gravity)), 1e-9)
        self.boxes = [np.arange(i, i + 8) for i in range(0, len(self.position), 8)]
        young = float(scene.config["material"]["youngs_modulus_pa"]); compliance = 0.42
        local = {(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7),
                 (0,2),(1,3),(4,6),(5,7),(0,6),(1,7),(2,4),(3,5)}
        springs = []
        for box in self.boxes:
            spans = np.ptp(self.rest_position[box], axis=0)
            area = float(np.prod(np.sort(np.maximum(spans, 1e-4))[:2]))
            for a,b in local:
                i,j = int(box[a]),int(box[b]); d = self.rest_position[j]-self.rest_position[i]
                L = float(np.linalg.norm(d)); k = compliance*young*area/max(L,1e-4)
                if L > 1e-9: springs.append((i,j,L,k,0.035*np.sqrt(k)))
        for left,right in zip(self.boxes[:-1], self.boxes[1:]):
            rem = list(map(int, right[[0,3,4,7]]))
            for i in map(int, left[[1,2,5,6]]):
                j = min(rem, key=lambda q: np.linalg.norm(self.rest_position[q]-self.rest_position[i])); rem.remove(j)
                d = self.rest_position[j]-self.rest_position[i]; L=float(np.linalg.norm(d))
                area=max(float(np.prod(np.sort(np.maximum(np.abs(d),0.012))[:2])),1.5e-4)
                k=compliance*young*area/max(L,0.01)
                if L > 1e-9: springs.append((i,j,L,k,0.05*np.sqrt(k)))
        self.springs = springs

    def _step(self, dt):
        force = self.mass[:,None]*self.gravity[None,:] - 1.8*self.mass[:,None]*self.velocity
        for i,j,L0,k,c in self.springs:
            d=self.position[j]-self.position[i]; L=float(np.linalg.norm(d))
            if L < 1e-10: continue
            n=d/L; rel=float(np.dot(self.velocity[j]-self.velocity[i],n)); f=(k*(L-L0)+c*rel)*n
            force[i] += f; force[j] -= f
        self.velocity += dt*force/np.maximum(self.mass[:,None],1e-12); self.position += dt*self.velocity
        self.position[self.fixed]=self.rest_position[self.fixed]; self.velocity[self.fixed]=0.0
        self.position=np.nan_to_num(self.position,nan=0.0,posinf=10.0,neginf=-10.0); self.velocity=np.nan_to_num(self.velocity,nan=0.0,posinf=0.0,neginf=0.0)

    def _gradients(self):
        out=[]
        for box in self.boxes:
            X=self.rest_position[box]-self.rest_position[box].mean(0); x=self.position[box]-self.position[box].mean(0)
            out.append(np.linalg.lstsq(X,x,rcond=None)[0].T)
        return np.nan_to_num(np.asarray(out),nan=0.0,posinf=1.0,neginf=-1.0)

    def advance_to(self, time: float):
        target=float(time)
        if not np.isfinite(target) or target<0: raise ValueError("advance_to time must be finite and non-negative")
        if target < self.current_time-1e-10: raise ValueError("advance_to requires nondecreasing times")
        while self.current_time < target-1e-12:
            dt=min(5e-4,target-self.current_time); self._step(dt); self.current_time += dt
        ext=self.mass[:,None]*self.gravity[None,:]
        frame=StateFrame(time=target,position=self.position.copy(),velocity=self.velocity.copy(),mass=self.mass.copy(),
            attributes={"fixed":self.fixed.astype(float),"rest_position":self.rest_position.copy(),"mass_quantity":self.mass.copy(),"external_force":ext},
            fields={"budget_external_force":ext.sum(0).reshape(1,1,3),"cumulative_mass_source":np.zeros(1),"element_deformation_gradient":self._gradients()},
            scalars={"gravity_x":float(self.gravity[0]),"gravity_y":float(self.gravity[1]),"gravity_z":float(self.gravity[2]),"reference_force":float(self.reference_force)})
        frame.validate(); return frame

    def diagnostics(self):
        return {"integrator":"semi-implicit Euler lumped spring solid","vertex_count":0 if self.position is None else len(self.position),"spring_count":len(self.springs)}
