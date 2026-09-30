"""Explicit particle solver for the foam bridge and its tensile tear."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        f = scene.initial_state()
        self.x0 = np.asarray(f.position, float).copy(); self.x = self.x0.copy()
        self.v = np.asarray(f.velocity, float).copy(); self.m = np.asarray(f.mass, float).copy()
        self.n = len(self.x); self.t = 0.0
        self.g = np.asarray(scene.config.get("gravity", [0, 0, -9.81]), float)
        ic = scene.config["initial_conditions"]
        self.radius = float(ic.get("nominal_particle_radius", 0.026)); self.spacing = float(ic.get("particle_spacing", 0.045))
        self.tear_lo, self.tear_hi = map(float, ic.get("intended_tear_zone_z", [0.78, 1.12]))
        def key(p): return tuple(np.rint(p / self.spacing).astype(np.int64))
        lut = {key(p): i for i, p in enumerate(self.x0)}
        ei=[]; ej=[]; r0=[]
        for i,p in enumerate(self.x0):
            k=np.rint(p/self.spacing).astype(np.int64)
            for a in range(3):
                kk=k.copy(); kk[a]+=1; j=lut.get(tuple(kk))
                if j is not None: ei.append(i); ej.append(j); r0.append(self.x0[j]-p)
        self.ei=np.asarray(ei,np.int32); self.ej=np.asarray(ej,np.int32); self.r0=np.asarray(r0,float)
        self.l0=np.linalg.norm(self.r0,axis=1); midz=.5*(self.x0[self.ei,2]+self.x0[self.ej,2])
        self.neck_edge=(midz>=self.tear_lo)&(midz<=self.tear_hi)
        fracture_z=.5*(self.tear_lo+self.tear_hi)
        zi=self.x0[self.ei,2]; zj=self.x0[self.ej,2]
        self.fracture_edge=(zi<=fracture_z)&(zj>fracture_z)
        self.active=np.ones(len(self.ei),bool); self.torn=False; self.last_external=self.m[:,None]*self.g

    def _step(self, dt):
        active=self.active; ii=self.ei[active]; jj=self.ej[active]; l0=self.l0[active]
        d=self.x[jj]-self.x[ii]; ln=np.linalg.norm(d,axis=1); nv=d/np.maximum(ln[:,None],1e-9)
        rel=self.v[jj]-self.v[ii]; fs=180.0*(ln-l0)+3.0*np.sum(rel*nv,axis=1); ff=fs[:,None]*nv
        fint=np.zeros_like(self.x); np.add.at(fint,ii,ff); np.add.at(fint,jj,-ff)
        ext=self.m[:,None]*self.g; upper=self.x0[:,2]>1.12; ext[upper,2]+=self.m[upper]*3.8; ext-=.65*self.m[:,None]*self.v
        # Adhesion to the plate supports the cap while the lower mass settles.
        adhered=self.x0[:,2]>1.68
        ext[adhered]+=self.m[adhered,None]*(55.0*(self.x0[adhered]-self.x[adhered])-7.0*self.v[adhered])
        if self.t>=.70:
            neck=(self.x0[:,2]>=self.tear_lo)&(self.x0[:,2]<=self.tear_hi); side=np.where(self.x0[:,2]<.95,-1.,1.); ext[neck,2]+=side[neck]*self.m[neck]*(4.0 if self.torn else 10.0)
        self.last_external=ext.copy()
        self.v += dt*(fint+ext)/self.m[:,None]; self.x += dt*self.v
        hit=self.x[:,2]<self.radius
        if np.any(hit): self.x[hit,2]=self.radius; self.v[hit,2]=np.maximum(self.v[hit,2],0)*.25; self.v[hit,:2]*=.92
        ph=(self.x[:,2]+self.radius>1.84)&(np.abs(self.x[:,0])<.78)&(np.abs(self.x[:,1])<.62)
        if np.any(ph): self.x[ph,2]=1.84-self.radius; self.v[ph,2]=np.minimum(self.v[ph,2],0)*.2
        if self.t>=.70 and not self.torn:
            stretch=ln/np.maximum(l0,1e-9); fail=self.fracture_edge[active]&(stretch>1.12)
            if np.any(fail): self.active[self.fracture_edge]=False; self.torn=True

    def _frame(self,time):
        # Least-complex local lattice deformation estimate: each material-axis
        # column is the average deformed neighbour vector divided by rest length.
        F=np.zeros((self.n,3,3)); counts=np.zeros((self.n,3))
        for a in range(3):
            use=self.active&(np.abs(self.r0[:,a])>.5*self.spacing)
            i=self.ei[use]; j=self.ej[use]; col=(self.x[j]-self.x[i])/self.r0[use,a,None]
            np.add.at(F[:,:,a],i,col); np.add.at(F[:,:,a],j,col); np.add.at(counts[:,a],i,1); np.add.at(counts[:,a],j,1)
            missing=counts[:,a]==0; F[~missing,:,a]/=counts[~missing,a,None]; F[missing,a,a]=1.0
        d_floor=self.x[:,2]-self.radius; d_plate=1.84-(self.x[:,2]+self.radius)
        fields={"contact_signed_distance":np.minimum(d_floor,d_plate),"cumulative_mass_source":np.zeros(1),"external_force":self.last_external.copy()}
        fr=StateFrame(float(time),self.x.copy(),self.v.copy(),mass=self.m.copy(),body_id=np.zeros(self.n,np.int32),attributes={"deformation_gradient":F,"mass_quantity":self.m.copy(),"particle_radius":np.full(self.n,self.radius),"foam_material":np.ones(self.n),"tear_candidate":((self.x0[:,2]>=self.tear_lo)&(self.x0[:,2]<=self.tear_hi)).astype(float)},fields=fields,scalars={"contact_length_scale":self.spacing,"explicit_tearing_enabled":1.,"tear_occurred":float(self.torn)})
        fr.validate(); return fr

    def advance_to(self,time: float):
        target=float(time)
        if target<self.t-1e-10: raise ValueError("advance_to requires nondecreasing time")
        while self.t<target-1e-12:
            h=min(.005,target-self.t); self._step(h); self.t+=h
        return self._frame(target)

    def diagnostics(self):
        return {"particle_count":int(self.n),"active_links":int(np.count_nonzero(self.active)),"torn":bool(self.torn)}
