"""Deterministic mass--spring deformable-solid solver for the octopus scene."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init = scene.initial_state()
        self.rest = np.asarray(init.position, float).copy()
        self.x = self.rest.copy(); self.v = np.zeros_like(self.x)
        self.mass = np.asarray(init.mass, float).copy(); self.n = len(self.x)
        self.time = 0.0
        self.gravity = np.asarray(scene.config.get("gravity", [0., 0., -9.81]), float)
        self.faces = []
        obj = Path(__file__).resolve().parent.parent / "assets" / "octopus.obj"
        for line in obj.read_text(encoding="utf-8").splitlines():
            if line.startswith("f "):
                q = [int(s.split("/")[0])-1 for s in line.split()[1:4]]
                if len(q) == 3: self.faces.append(q)
        es = set()
        for t in self.faces:
            for a,b in ((t[0],t[1]), (t[1],t[2]), (t[2],t[0])): es.add(tuple(sorted((a,b))))
        for i in range(self.n):
            d = np.linalg.norm(self.rest-self.rest[i], axis=1)
            for j in np.argsort(d)[1:5]:
                if d[j] < 2.2: es.add(tuple(sorted((i,int(j)))))
        self.edges = np.asarray(sorted(es), dtype=np.int32)
        self.edge_rest = np.linalg.norm(self.rest[self.edges[:,1]]-self.rest[self.edges[:,0]], axis=1)
        self.k = 1150.; self.damp = 7.; self.tether = 18.; self.dt_max = .004
        self.supports = [(-.72,.72,.58), (.72,.72,.58), (0.,1.72,.58)]
        self.keys = scene.config["upper_cylinder_motion"]["keyframes"]
        self.cum_impulse = np.zeros((1,3)); self.last_force = np.zeros((1,2,3))
        self.last_force[0,0] = np.sum(self.mass[:,None]*self.gravity[None,:], axis=0)
        self.reference_momentum = float(np.sum(self.mass)*3.)

    def _upper(self, t):
        if t <= self.keys[0]["time"]: return float(self.keys[0]["translation"][2]), 0.
        for a,b in zip(self.keys[:-1], self.keys[1:]):
            ta,tb=float(a["time"]),float(b["time"])
            if t <= tb:
                za,zb=float(a["translation"][2]),float(b["translation"][2])
                return za+(zb-za)*(t-ta)/(tb-ta), (zb-za)/(tb-ta)
        return float(self.keys[-1]["translation"][2]),0.

    def _forces(self, t):
        f = self.mass[:,None]*self.gravity[None,:]
        for e,(i,j) in enumerate(self.edges):
            d=self.x[j]-self.x[i]; L=np.linalg.norm(d)
            if L < 1e-9: continue
            n=d/L; rel=np.dot(self.v[j]-self.v[i],n)
            fs=(self.k*(L-self.edge_rest[e])+self.damp*rel)*n
            f[i]+=fs; f[j]-=fs
        # Translation-invariant shape matching adds volumetric coherence without
        # pinning the animal in space, so gravity can produce a genuine drop.
        xrel=self.x-np.average(self.x,axis=0,weights=self.mass)
        rrel=self.rest-np.average(self.rest,axis=0,weights=self.mass)
        f += -self.tether*(xrel-rrel)
        contact=np.zeros_like(f); uz,uzv=self._upper(t)
        cylinders=self.supports+[(0.,uz,.62)]
        for i in range(self.n):
            for ci,(cx,cz,r) in enumerate(cylinders):
                if abs(self.x[i,1])>3.05: continue
                q=self.x[i,[0,2]]-np.array([cx,cz]); rr=np.linalg.norm(q); pen=r-rr
                if pen>0:
                    n2=q/rr if rr>1e-8 else np.array([0.,1.]); n3=np.array([n2[0],0.,n2[1]])
                    vc=np.array([0.,0.,uzv]) if ci==3 else np.zeros(3)
                    contact[i]+=2600.*pen*n3; reln=np.dot(self.v[i]-vc,n3)
                    if reln<0: contact[i]+=-28.*reln*n3
                    self.x[i,0]+=.72*pen*n2[0]; self.x[i,2]+=.72*pen*n2[1]
        f += contact
        terms=np.zeros((1,2,3)); terms[:,0,:]=np.sum(self.mass[:,None]*self.gravity[None,:],axis=0); terms[:,1,:]=np.sum(contact,axis=0)
        return f,terms

    def _step(self, dt):
        f,terms=self._forces(self.time+dt)
        self.v += dt*f/self.mass[:,None]; self.x += dt*self.v
        self._forces(self.time+dt); self.last_force=terms
        self.cum_impulse += np.sum(terms,axis=1)*dt; self.time += dt

    def _F(self):
        out=np.empty((len(self.faces),3,3))
        for k,(a,b,c) in enumerate(self.faces):
            dm1=self.rest[b]-self.rest[a]; dm2=self.rest[c]-self.rest[a]
            ds1=self.x[b]-self.x[a]; ds2=self.x[c]-self.x[a]; nm=np.cross(dm1,dm2); ns=np.cross(ds1,ds2)
            lm=np.linalg.norm(nm)
            if lm<1e-10: out[k]=np.eye(3); continue
            Dm=np.column_stack((dm1,dm2,nm/lm)); Ds=np.column_stack((ds1,ds2,ns/max(np.linalg.norm(ns),1e-10)))
            out[k]=Ds@np.linalg.inv(Dm)
        return np.nan_to_num(out)

    def advance_to(self, time: float):
        target=float(time)
        if target < self.time-1e-9: raise ValueError("times must be nondecreasing")
        while self.time+1e-10 < target: self._step(min(self.dt_max,target-self.time))
        frame=StateFrame(time=target, position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(), body_id=np.zeros(self.n,dtype=np.int32), attributes={"rest_position":self.rest.copy(), "mass_quantity":self.mass.copy()}, fields={"element_deformation_gradient":self._F(), "budget_cumulative_external_impulse":self.cum_impulse.copy(), "budget_cumulative_inflow_momentum":np.zeros((1,3)), "budget_cumulative_outflow_momentum":np.zeros((1,3)), "budget_domain_weights":np.ones((1,self.n)), "budget_external_force":self.last_force.copy(), "cumulative_mass_source":np.zeros(1)}, scalars={"reference_momentum":self.reference_momentum, "gravity_z":float(self.gravity[2])})
        frame.validate(); return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver":"damped_mass_spring_3d", "vertices":int(self.n), "edges":int(len(self.edges))}
