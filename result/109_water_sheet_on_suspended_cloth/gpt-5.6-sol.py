"""Self-contained two-way particle-water / suspended-cloth solver."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene, self.cfg = scene, scene.config
        f = scene.initial_state()
        self.position = np.asarray(f.position, float).copy()
        self.velocity = np.asarray(f.velocity, float).copy()
        self.mass = np.asarray(f.mass, float).copy()
        self.body_id = np.asarray(f.body_id, np.int32).copy()
        self.base_attributes = {k: np.asarray(v).copy() for k, v in f.attributes.items()}
        self.nc = int(np.count_nonzero(self.body_id == 0)); self.ws = slice(self.nc, None)
        self.nw = len(self.position) - self.nc
        self.gravity = np.asarray(self.cfg["gravity"], float)
        self.time = 0.0
        cc = self.cfg["cloth"]
        self.nx, self.ny = map(int, cc["grid_shape"])
        self.rest = self.position[:self.nc].copy()
        self.pins = np.asarray(cc["pinned_vertices"], np.int64)
        self.pin_position = self.rest[self.pins].copy()
        self.inv_mass = 1.0 / self.mass[:self.nc]; self.inv_mass[self.pins] = 0.0
        self._make_constraints()
        self.spacing = float(self.cfg["water"]["initial_particle_spacing"])
        self.radius = .32 * self.spacing
        self.fluid_clock, self.fluid_interval = 0.0, .005
        self._make_emitter()
        self.emit_period = self.spacing / float(self.cfg["inflow"]["speed"])
        self.next_emit, self.emit_phase = self.emit_period, 0
        self.wetness = np.zeros(self.nc)
        self.cum_ext = np.zeros((3, 3)); self.cum_in = np.zeros((3, 3)); self.cum_out = np.zeros((3, 3))
        self.cum_in_mass = self.cum_out_mass = 0.0
        self.last_ext = np.zeros((3, 3, 3)); self.last_interaction = np.zeros((2, 3))
        self.steps = self.emitted_layers = self.contacts = 0

    def _make_constraints(self):
        edges = []
        for j in range(self.ny):
            for i in range(self.nx):
                a = j*self.nx+i
                if i+1 < self.nx: edges.append((a,a+1,.92))
                if j+1 < self.ny: edges.append((a,a+self.nx,.92))
                if i+1 < self.nx and j+1 < self.ny:
                    edges.extend(((a,a+self.nx+1,.72),(a+1,a+self.nx,.72)))
                if i+2 < self.nx: edges.append((a,a+2,.16))
                if j+2 < self.ny: edges.append((a,a+2*self.nx,.16))
        self.ea=np.asarray([x[0] for x in edges],np.int64)
        self.eb=np.asarray([x[1] for x in edges],np.int64)
        self.es=np.asarray([x[2] for x in edges],float)
        self.el=np.linalg.norm(self.rest[self.eb]-self.rest[self.ea],axis=1)

    def _make_emitter(self):
        r=float(self.cfg["inflow"]["radius"])
        a=np.arange(-r,r+.25*self.spacing,self.spacing)
        x,y=np.meshgrid(a,a,indexing="xy"); q=np.column_stack((x.ravel(),y.ravel()))
        self.emit_offsets=q[np.sum(q*q,axis=1)<=r*r]

    def _emit(self, end):
        pin=np.zeros(3); pout=np.zeros(3); mi=mo=0.0
        cfg=self.cfg["inflow"]; active=float(cfg["active_interval"][1])
        center=np.asarray(cfg["center"],float); prescribed=np.asarray(cfg["axis"],float)*float(cfg["speed"])
        while self.next_emit <= end+1e-12 and self.next_emit <= active:
            p=self.position[self.ws]
            candidates=np.flatnonzero((p[:,2]<.24)&(np.abs(p[:,0])<1.95)&(np.abs(p[:,1])<1.60))
            count=min(len(candidates),len(self.emit_offsets))
            if count:
                angle=(self.emit_phase%8)*math.pi/16; c,s=math.cos(angle),math.sin(angle)
                xy=self.emit_offsets[:count]@np.array(((c,-s),(s,c))).T
                ids=candidates[:count]+self.nc; m=self.mass[ids]
                pout+=np.sum(m[:,None]*self.velocity[ids],axis=0); mo+=float(m.sum())
                self.position[ids,:2]=center[:2]+xy; self.position[ids,2]=center[2]-.035
                self.velocity[ids]=prescribed
                pin+=np.sum(m[:,None]*prescribed,axis=0); mi+=float(m.sum())
                self.emit_phase+=1; self.emitted_layers+=1
            self.next_emit+=self.emit_period
        return pin,pout,mi,mo

    def _fluid_step(self, elapsed):
        p=self.position[self.ws]; v=self.velocity[self.ws]
        pairs=cKDTree(p).query_pairs(1.08*self.spacing,output_type="ndarray")
        if pairs.size==0:return
        ia,ib=pairs[:,0],pairs[:,1]; d=p[ib]-p[ia]; length=np.linalg.norm(d,axis=1)
        ok=length>1e-8; ia,ib,d,length=ia[ok],ib[ok],d[ok],length[ok]
        if not len(length):return
        overlap=np.maximum(1.08*self.spacing-length,0); direction=d/length[:,None]
        dv=np.zeros_like(v); impulse=1050*overlap[:,None]*direction*elapsed
        np.add.at(dv,ia,-impulse); np.add.at(dv,ib,impulse)
        visc=.035*(overlap/(1.08*self.spacing))[:,None]*(v[ib]-v[ia])
        np.add.at(dv,ia,visc); np.add.at(dv,ib,-visc); v+=dv

    def _surface(self, xy):
        width,depth=map(float,self.cfg["cloth"]["size"])
        u=(xy[:,0]/width+.5)*(self.nx-1); q=(xy[:,1]/depth+.5)*(self.ny-1)
        inside=(u>=0)&(u<=self.nx-1)&(q>=0)&(q<=self.ny-1)
        uc=np.clip(u,0,self.nx-1-1e-9); qc=np.clip(q,0,self.ny-1-1e-9)
        i=np.floor(uc).astype(np.int64); j=np.floor(qc).astype(np.int64); fu=uc-i; fq=qc-j
        ids=np.column_stack((j*self.nx+i,j*self.nx+i+1,(j+1)*self.nx+i,(j+1)*self.nx+i+1))
        w=np.column_stack(((1-fu)*(1-fq),fu*(1-fq),(1-fu)*fq,fu*fq))
        cp=self.position[:self.nc]; cv=self.velocity[:self.nc]
        h=np.sum(w*cp[ids,2],axis=1); sv=np.sum(w[:,:,None]*cv[ids],axis=1)
        z=cp[:,2].reshape(self.ny,self.nx)
        dy,dx=np.gradient(z,depth/(self.ny-1),width/(self.nx-1))
        gx=np.sum(w*dx.ravel()[ids],axis=1); gy=np.sum(w*dy.ravel()[ids],axis=1)
        n=np.column_stack((-gx,-gy,np.ones(len(xy)))); n/=np.linalg.norm(n,axis=1)[:,None]
        return inside,h,sv,n,ids,w

    def _cloth_impulse(self, ids, w, impulse):
        node=np.zeros((self.nc,3))
        for c in range(4): np.add.at(node,ids[:,c],-impulse*w[:,c,None])
        self.velocity[:self.nc]+=node*self.inv_mass[:,None]

    def _contact(self, oldp, dt):
        p=self.position[self.ws]; v=self.velocity[self.ws]
        inside,h,sv,n,ids,w=self._surface(p[:,:2]); oldinside,oldh,*_=self._surface(oldp[:,:2])
        sn=p[:,2]-h; so=oldp[:,2]-oldh; rel=v-sv; vn=np.sum(rel*n,axis=1); total=np.zeros(3)
        near=inside&(sn>self.radius)&(sn<.13)&(vn<0)
        if np.any(near):
            k=np.flatnonzero(near); dec=np.minimum(24,.55*vn[k]**2/np.maximum(2*sn[k],.025))
            dv=dec[:,None]*n[k]*dt; impulse=self.mass[self.nc+k,None]*dv
            v[k]+=dv; self._cloth_impulse(ids[k],w[k],impulse); total+=impulse.sum(axis=0)
        hit=inside&oldinside&(sn<=self.radius)&(so>=-.045)&(vn<0)&(p[:,2]>.45)
        if np.any(hit):
            k=np.flatnonzero(hit); before=v[k].copy(); rr=before-sv[k]
            vn2=np.sum(rr*n[k],axis=1); after=before-1.025*vn2[:,None]*n[k]
            tangent=after-sv[k]-np.sum((after-sv[k])*n[k],axis=1)[:,None]*n[k]
            after-=.025*tangent; v[k]=after; p[k,2]=h[k]+self.radius
            impulse=self.mass[self.nc+k,None]*(after-before)
            self._cloth_impulse(ids[k],w[k],impulse); total+=impulse.sum(axis=0)
            np.add.at(self.wetness,ids[k].ravel(),w[k].ravel()); self.contacts+=len(k)
        return total

    def _basin(self):
        p=self.position[self.ws]; v=self.velocity[self.ws]; before=v.copy(); r=self.radius
        hit=p[:,2]<r; p[hit,2]=r; v[hit,2]=np.maximum(-.08*v[hit,2],0); v[hit,:2]*=.78
        low=p[:,2]<.78
        for axis,bound,direction in ((0,-2.11+r,1),(0,2.11-r,-1),(1,-1.86+r,1),(1,1.86-r,-1)):
            hit=low&((p[:,axis]<bound) if direction>0 else (p[:,axis]>bound))
            p[hit,axis]=bound; inward=direction*v[hit,axis]
            v[hit,axis]+=direction*np.maximum(-1.12*inward,0); v[hit,1-axis]*=.92
        for axis,lo,hi in ((0,-2.32,2.32),(1,-2.12,2.12)):
            left=p[:,axis]<lo; right=p[:,axis]>hi; p[left,axis]=lo; p[right,axis]=hi
            v[left,axis]=np.abs(v[left,axis])*.1; v[right,axis]=-np.abs(v[right,axis])*.1
        return np.sum(self.mass[self.ws,None]*(v-before),axis=0)

    def _cloth_step(self,dt):
        old=self.position[:self.nc].copy(); oldmom=np.sum(self.mass[:self.nc,None]*self.velocity[:self.nc],axis=0)
        grav=self.mass[:self.nc].sum()*self.gravity*dt
        self.velocity[:self.nc]+=self.gravity*dt; self.velocity[:self.nc]*=math.exp(-.65*dt)
        x=old+self.velocity[:self.nc]*dt; x[self.pins]=self.pin_position
        wa=self.inv_mass[self.ea]; wb=self.inv_mass[self.eb]; denom=wa+wb
        for _ in range(7):
            d=x[self.eb]-x[self.ea]; length=np.linalg.norm(d,axis=1); direction=d/np.maximum(length[:,None],1e-12)
            corr=(self.es*(length-self.el)/np.maximum(denom,1e-15))[:,None]*direction
            delta=np.zeros_like(x); np.add.at(delta,self.ea,wa[:,None]*corr); np.add.at(delta,self.eb,-wb[:,None]*corr)
            x+=.72*delta; x[self.pins]=self.pin_position
        x[:,2]=np.maximum(x[:,2],.08); x[self.pins]=self.pin_position
        self.position[:self.nc]=x; self.velocity[:self.nc]=(x-old)/dt; self.velocity[self.pins]=0
        speed=np.linalg.norm(self.velocity[:self.nc],axis=1); fast=speed>35
        self.velocity[:self.nc][fast]*=(35/speed[fast])[:,None]
        newmom=np.sum(self.mass[:self.nc,None]*self.velocity[:self.nc],axis=0)
        return newmom-oldmom,grav

    def _substep(self,dt):
        pin,pout,mi,mo=self._emit(self.time+dt); oldp=self.position[self.ws].copy()
        self.velocity[self.ws]+=self.gravity*dt; self.fluid_clock+=dt
        if self.fluid_clock>=self.fluid_interval-1e-12:
            self._fluid_step(self.fluid_clock); self.fluid_clock=0
        self.position[self.ws]+=self.velocity[self.ws]*dt
        water_j=self._contact(oldp,dt); boundary_j=self._basin(); cloth_dp,cloth_g=self._cloth_step(dt)
        # cloth_dp starts after contact has already applied -water_j; therefore
        # it contains only gravity, damping/constraint, and support response.
        support_j=cloth_dp-cloth_g; water_g=self.mass[self.ws].sum()*self.gravity*dt
        self.wetness*=math.exp(-.035*dt); np.clip(self.wetness,0,1,out=self.wetness)
        self.time+=dt; self.steps+=1
        return water_j,cloth_g,water_g,support_j,boundary_j,pin,pout,mi,mo

    def advance_to(self,time:float):
        target=float(time)
        if target<self.time-1e-10: raise ValueError("advance_to times must be nondecreasing")
        start=self.time; ext=np.zeros((3,3,3)); inter=np.zeros((2,3)); pin=np.zeros(3); pout=np.zeros(3)
        while self.time<target-1e-12:
            dt=min(.001,target-self.time); wj,cg,wg,sj,bj,pi,po,mi,mo=self._substep(dt)
            inter[0]-=wj; inter[1]+=wj; ext[0,0]+=cg; ext[0,1]+=sj; ext[1,0]+=wg; ext[1,1]+=bj
            pin+=pi; pout+=po; self.cum_in_mass+=mi; self.cum_out_mass+=mo
        ext[2]=ext[0]+ext[1]; elapsed=target-start
        if elapsed>0:
            self.last_ext=ext/elapsed; self.last_interaction=inter/elapsed; self.cum_ext+=ext.sum(axis=1)
            self.cum_in[1]+=pin; self.cum_in[2]+=pin; self.cum_out[1]+=pout; self.cum_out[2]+=pout
        return self._frame(target)

    def _normals(self):
        p=self.position[:self.nc].reshape(self.ny,self.nx,3); ty=np.gradient(p,axis=0); tx=np.gradient(p,axis=1)
        n=np.cross(tx,ty).reshape(-1,3); n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
        n[n[:,2]<0]*=-1; return n

    def _frame(self,time):
        weights=np.zeros((3,len(self.position))); weights[0,:self.nc]=1; weights[1,self.nc:]=1; weights[2]=1
        static=np.asarray(((0,0,1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)),float)
        bn=np.vstack((static,self._normals())); bv=np.vstack((np.zeros((5,3)),self.velocity[:self.nc]))
        attrs={k:v.copy() for k,v in self.base_attributes.items()}; attrs["mass_quantity"]=self.mass.copy()
        attrs["wetness"]=np.r_[self.wetness,np.ones(self.nw)]
        fields={"boundary_normal":bn,"boundary_target_velocity":bv.copy(),"boundary_velocity":bv,
          "budget_cumulative_external_impulse":self.cum_ext.copy(),"budget_cumulative_inflow_momentum":self.cum_in.copy(),
          "budget_cumulative_outflow_momentum":self.cum_out.copy(),"budget_domain_weights":weights,
          "budget_external_force":self.last_ext.copy(),"cumulative_mass_source":np.asarray((self.cum_in_mass-self.cum_out_mass,)),
          "interaction_force":self.last_interaction.copy()}
        rv=float(self.cfg["inflow"]["speed"])
        scalars={"water_density":float(self.cfg["water"]["density"]),"cloth_thickness":float(self.cfg["cloth"]["thickness"]),
          "nominal_inflow_speed":rv,"nozzle_radius":float(self.cfg["inflow"]["radius"]),
          "reference_velocity":rv,"reference_momentum":max(float(self.mass.sum())*rv,1e-9)}
        frame=StateFrame(time=float(time),position=self.position.copy(),velocity=self.velocity.copy(),mass=self.mass.copy(),
          body_id=self.body_id.copy(),attributes=attrs,fields=fields,scalars=scalars)
        frame.validate(); return frame

    def diagnostics(self)->dict[str,object]:
        return {"method":"PBD membrane and Lagrangian pressure particles","time":float(self.time),"substeps":self.steps,
          "emitted_layers":self.emitted_layers,"cloth_water_contacts":self.contacts,
          "finite":bool(np.isfinite(self.position).all() and np.isfinite(self.velocity).all())}
