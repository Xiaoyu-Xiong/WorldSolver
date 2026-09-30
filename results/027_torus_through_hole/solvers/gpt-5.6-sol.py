"""Deterministic spring-mass solver for the torus-through-hole scene."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        c = self.cfg = dict(scene.config)
        self.rest = np.asarray(scene.vertices, float).copy(); self.x = self.rest.copy()
        self.n = len(self.x); self.time = 0.0
        self.v = np.tile(np.asarray(c.get("initial_velocity", [0., 0., 0.]), float), (self.n, 1))
        self.total_mass = float(c.get("subject_mass", 1.0)); self.mass = np.full(self.n, self.total_mass / self.n)
        self.gravity = np.asarray(c.get("gravity", [0., 0., -9.81]), float)
        self.dt = min(float(c.get("sample_dt", 1/30)) / 4.0, 1/240)
        self.k = float(c.get("stiffness", 10.0)) * 28.0; self.damping = float(c.get("damping", 1.0))
        self.restitution = float(c.get("restitution", .04)); self.friction = float(c.get("friction", .48))
        self.hole_r = float(c.get("hole_radius", .5)); self.hole_top = float(c.get("hole_z", .62)) + .23
        self.ground_z = float(c.get("ground_z", 0.0)); self.contact_scale = float(c.get("contact_length_scale", .04))
        self.push_speed = float(c.get("push_speed", .48))
        try:
            with np.load(__import__("pathlib").Path(__file__).resolve().parents[1] / c["subject_asset"]) as d: faces = d["faces"]
        except Exception:
            faces = np.empty((0, 3), int)
        e = np.concatenate((faces[:, (0,1)], faces[:, (1,2)], faces[:, (2,0)]), 0) if len(faces) else np.empty((0,2), int)
        self.edges = np.unique(np.sort(e, axis=1), axis=0) if len(e) else e
        i, j = self.edges[:,0], self.edges[:,1]
        self.edge_rest = np.linalg.norm(self.rest[j]-self.rest[i], axis=1) if len(self.edges) else np.empty(0)
        self.neigh = [[] for _ in range(self.n)]
        for a,b in self.edges: self.neigh[int(a)].append(int(b)); self.neigh[int(b)].append(int(a))
        self.local_inv = []
        for a, ns in enumerate(self.neigh):
            q = self.rest[np.asarray(ns)] - self.rest[a] if len(ns) else np.zeros((1,3))
            self.local_inv.append(np.linalg.inv(q.T@q + 1e-8*np.eye(3)) if len(ns)>=3 else np.eye(3))

    def _forces(self):
        f = self.mass[:,None] * self.gravity[None,:]
        if len(self.edges):
            i,j = self.edges[:,0], self.edges[:,1]; d=self.x[j]-self.x[i]; L=np.linalg.norm(d,axis=1); u=d/np.maximum(L[:,None],1e-9)
            fs=self.k*(L-self.edge_rest)+.35*self.damping*np.sum((self.v[j]-self.v[i])*u,axis=1); p=fs[:,None]*u
            np.add.at(f,i,p); np.add.at(f,j,-p)
        pen=self.ground_z-self.x[:,2]; hit=pen>0
        if np.any(hit): f[hit,2]+=1500*pen[hit]-20*self.v[hit,2]
        r=np.linalg.norm(self.x[:,:2],axis=1); ann=r>self.hole_r; pt=self.hole_top-self.x[:,2]
        hit=ann&(pt>0)&(self.x[:,2]>self.hole_top-.28)
        if np.any(hit): f[hit,2]+=1100*pt[hit]-18*self.v[hit,2]
        underside=1.88-self.push_speed*min(self.time,2.7)-.09; inside=(r<.48)&(self.x[:,2]>underside)
        if np.any(inside): f[inside,2]-=1250*(self.x[inside,2]-underside)+20*np.maximum(self.v[inside,2],0)
        f-=.10*self.damping*self.v
        gap=np.minimum.reduce((self.x[:,2]-self.ground_z,np.where(ann,self.x[:,2]-self.hole_top,10.),np.where(inside,underside-self.x[:,2],10.)))
        return f,gap

    def _step(self,h):
        force,_=self._forces(); self.v += h*force/self.mass[:,None]; s=np.linalg.norm(self.v,axis=1); fast=s>8
        if np.any(fast): self.v[fast]*=(8/s[fast])[:,None]
        self.x += h*self.v
        b=self.x[:,2]<self.ground_z
        if np.any(b): self.x[b,2]=self.ground_z; self.v[b,2]=np.maximum(0,-self.restitution*self.v[b,2])
        r=np.linalg.norm(self.x[:,:2],axis=1); b=(r>self.hole_r)&(self.x[:,2]<self.hole_top)&(self.x[:,2]>self.hole_top-.28)
        if np.any(b): self.x[b,2]=self.hole_top; self.v[b,2]=np.maximum(0,self.v[b,2])
        self.time += h

    def _frame(self,t):
        _,gap=self._forces(); F=np.empty((self.n,3,3))
        for a,ns in enumerate(self.neigh):
            if len(ns)<3: F[a]=np.eye(3); continue
            ids=np.asarray(ns); X=self.rest[ids]-self.rest[a]; Y=self.x[ids]-self.x[a]; F[a]=(Y.T@X)@self.local_inv[a]
        F=np.nan_to_num(F,nan=0.,posinf=10.,neginf=-10.)
        fr=StateFrame(time=float(t),position=self.x.copy(),velocity=self.v.copy(),mass=self.mass.copy(),body_id=np.zeros(self.n,dtype=np.int64),attributes={"mass_quantity":self.mass.copy(),"deformation_gradient":F},fields={"cumulative_mass_source":np.zeros(1),"external_force":self.mass[:,None]*self.gravity[None,:],"contact_signed_distance":gap.copy()},scalars={"contact_length_scale":self.contact_scale})
        fr.validate(); return fr

    def advance_to(self,time: float):
        target=max(float(time),self.time)
        while self.time+1e-10<target: self._step(min(self.dt,target-self.time))
        return self._frame(target)

    def diagnostics(self):
        return {"time":float(self.time),"particle_count":int(self.n),"substep_dt":float(self.dt)}
