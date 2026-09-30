"""Deterministic viscoplastic particle/edge solver for the bunny impact."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.cfg = dict(scene.config); self.rest0 = np.asarray(scene.vertices, float).copy(); self.pos = self.rest0.copy()
        self.n = len(self.pos); self.mass = np.full(self.n, float(self.cfg["subject_mass"]) / self.n)
        self.gravity = np.asarray(self.cfg.get("gravity", [0,0,-9.81]), float)
        self.vel = np.tile(np.asarray(self.cfg.get("initial_velocity", [0,0,0]), float), (self.n,1)); self.t = 0.0
        self.plastic = np.zeros_like(self.pos); self.hardening = np.zeros(self.n)
        try:
            with np.load(Path(__file__).resolve().parents[1] / self.cfg["subject_asset"]) as d: faces = np.asarray(d["faces"], dtype=np.int64)
            e = np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]])); self.edges = np.unique(np.sort(e,axis=1),axis=0)
        except Exception: self.edges = np.empty((0,2), dtype=np.int64)
        self.ei, self.ej = (self.edges[:,0], self.edges[:,1]) if len(self.edges) else (np.empty(0,dtype=int), np.empty(0,dtype=int))
        if len(self.edges): self.rest_len = np.linalg.norm(self.rest0[self.ej]-self.rest0[self.ei],axis=1).clip(1e-5)
        else: self.rest_len = np.empty(0)
        self._last_contact_force = np.zeros(3)

    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-10: raise ValueError("advance_to requires nondecreasing time")
        while self.t < target - 1e-12:
            h = min(1/600, target-self.t); self._step(h); self.t += h
        return self._frame(target)

    def _forces(self):
        f = self.mass[:,None] * self.gravity[None,:]; internal = np.zeros_like(self.pos)
        if len(self.edges):
            d=self.pos[self.ej]-self.pos[self.ei]; L=np.linalg.norm(d,axis=1).clip(1e-8); u=d/L[:,None]
            k=float(self.cfg.get("stiffness",18))*8*(1+0.5*(self.hardening[self.ei]+self.hardening[self.ej])); ef=(k*(L-self.rest_len))[:,None]*u
            np.add.at(internal,self.ei,ef); np.add.at(internal,self.ej,-ef)
        internal += -float(self.cfg.get("stiffness",18))*1.8*(self.pos-(self.rest0+self.plastic)); f += internal-float(self.cfg.get("damping",1.4))*self.vel
        contact=np.zeros_like(self.pos); ground=float(self.cfg.get("ground_z",0)); kn=600.; cn=4.
        gap=self.pos[:,2]-ground; hit=gap<0
        if np.any(hit):
            fn=-kn*gap[hit]-cn*np.minimum(self.vel[hit,2],0); contact[hit,2]+=fn
        bz=float(self.cfg.get("bar_z",.66)); br=float(self.cfg.get("bar_radius",.085)); rv=self.pos[:,[0,2]]-np.array([0,bz]); rr=np.linalg.norm(rv,axis=1); bgap=rr-br; near=bgap<0
        if np.any(near):
            n=np.zeros((near.sum(),3)); n[:,[0,2]]=rv[near]/np.maximum(np.linalg.norm(rv[near],axis=1)[:,None],1e-8); vn=np.sum(self.vel[near]*n,axis=1); fn=-kn*bgap[near]-cn*np.minimum(vn,0); contact[near]+=fn[:,None]*n
        f += contact; self._last_contact_force=contact.sum(axis=0); return f/self.mass[:,None], gap, bgap

    def _step(self, dt):
        a,_,_=self._forces(); self.vel += dt*a; s=np.linalg.norm(self.vel,axis=1); self.vel *= np.minimum(1,18/np.maximum(s,1e-9))[:,None]; self.pos += dt*self.vel
        if len(self.edges):
            L=np.linalg.norm(self.pos[self.ej]-self.pos[self.ei],axis=1); strain=(L-self.rest_len)/self.rest_len; active=np.abs(strain)>.055; alpha=min(.08,float(self.cfg.get("hardening_rate",1.8))*dt*.22)
            if np.any(active):
                self.rest_len[active]+=alpha*(L[active]-self.rest_len[active]); inc=np.zeros(self.n); np.add.at(inc,self.ei[active],np.abs(strain[active])*alpha); np.add.at(inc,self.ej[active],np.abs(strain[active])*alpha); self.hardening=np.clip(self.hardening+inc,0,8); w=np.clip(inc*3,0,.18); self.plastic=(1-w[:,None])*self.plastic+w[:,None]*(self.pos-self.rest0)

    def _deformation_gradient(self):
        # Local least-squares affine map from rest edge vectors to current ones.
        F=np.tile(np.eye(3),(self.n,1,1));
        for i in range(self.n):
            mask=(self.ei==i)|(self.ej==i); ids=np.where(mask)[0]
            if len(ids)>=3:
                X=[]; Y=[]
                for q in ids:
                    j=self.ej[q] if self.ei[q]==i else self.ei[q]; X.append(self.rest0[j]-self.rest0[i]); Y.append(self.pos[j]-self.pos[i])
                X=np.asarray(X); Y=np.asarray(Y); F[i]=np.linalg.lstsq(X,Y,rcond=None)[0].T
        return np.nan_to_num(F,nan=0,posinf=0,neginf=0)

    def _frame(self,time,gap=None,bgap=None):
        if gap is None: _,gap,bgap=self._forces()
        fr=StateFrame(time=float(time),position=self.pos.copy(),velocity=self.vel.copy(),mass=self.mass.copy(),body_id=np.zeros(self.n,dtype=np.int64),attributes={"mass_quantity":self.mass.copy(),"deformation_gradient":self._deformation_gradient()},fields={"cumulative_mass_source":np.zeros(1),"external_force":self.mass[:,None]*self.gravity[None,:],"interaction_force":self._last_contact_force.reshape(1,3),"contact_signed_distance":np.concatenate((gap,bgap))},scalars={"contact_length_scale":float(self.cfg.get("contact_length_scale",.04))}); fr.validate(); return fr

    def diagnostics(self) -> dict[str, object]:
        return {"solver":"edge_spring_viscoplastic","vertex_count":int(self.n),"max_hardening":float(np.max(self.hardening))}
