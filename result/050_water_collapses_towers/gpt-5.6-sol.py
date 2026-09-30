"""Deterministic particle-fluid / rigid-body solver for the tower collapse scene."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


def _R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)


def _qstep(q, omega, dt):
    x, y, z, w = q
    ox, oy, oz = omega
    dq = .5*np.array([ox*w+oy*z-oz*y, -ox*z+oy*w+oz*x,
                      ox*y-oy*x+oz*w, -(ox*x+oy*y+oz*z)])
    q = q + dt*dq
    n = np.linalg.norm(q)
    return q/n if n > 1e-12 else np.array([0., 0., 0., 1.])


def _closest(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    aa, ee = u@u, v@v
    if aa < 1e-12 and ee < 1e-12: return a, c
    if aa < 1e-12: s, t = 0., np.clip(v@w/ee, 0., 1.)
    else:
        cc = u@w
        if ee < 1e-12: t, s = 0., np.clip(-cc/aa, 0., 1.)
        else:
            bb, den = u@v, aa*ee-(u@v)**2
            s = np.clip((bb*(v@w)-cc*ee)/den, 0., 1.) if abs(den)>1e-12 else 0.
            t = (bb*s+v@w)/ee
            if t < 0.: t, s = 0., np.clip(-cc/aa, 0., 1.)
            elif t > 1.: t, s = 1., np.clip((bb-cc)/aa, 0., 1.)
    return a+s*u, c+t*v


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        s = scene.initial_state(); self.time = 0.; self.seed = int(seed)
        self.g = np.asarray(scene.config.get("gravity", [0., 0., -9.81]), float)
        self.groups = np.asarray(s.attributes["scene_group"], float)
        self.nw = int(np.count_nonzero(self.groups == 10)); self.nr = len(s.position)-self.nw
        self.wx = s.position[:self.nw].copy(); self.wv = s.velocity[:self.nw].copy(); self.wm = s.mass[:self.nw].copy()
        sl = slice(self.nw, None); self.rx = s.position[sl].copy(); self.rv = s.velocity[sl].copy(); self.rq = s.orientation[sl].copy(); self.rw = np.zeros((self.nr,3)); self.rm = s.mass[sl].copy(); self.rgroups = self.groups[sl].copy(); self.box = self.rgroups % 10 == 1
        self.mass = s.mass.copy(); self.body_id = s.body_id.copy(); self.density0 = np.asarray(s.attributes["density"], float).copy()
        self.rad, self.hh, self.seg = .115, .925, .81
        self.h, self.rho0, self.cs2, self.visc = .24, 1000., 144., .65
        self.wrad, self.contact_len, self.dtmax = .068, .16, .005
        self.fk, self.fd, self.rk, self.rd = 3.2e4, 85., 8e4, 1.2e3
        self.inertia = np.empty((self.nr,3))
        for b,m in enumerate(self.rm):
            self.inertia[b] = ([m*(self.rad**2+self.hh**2)/3]*2+[2*m*self.rad**2/3] if self.box[b] else [m*(3*self.rad**2+1.85**2)/12]*2+[.5*m*self.rad**2])
        self.rho = np.full(self.nw, self.rho0); self.ext = self.mass[:,None]*self.g; self.inter = np.zeros((3,3)); self.steps = 0

    def _sph(self):
        x,v,m,h = self.wx,self.wv,self.wm,self.h; n=len(x); tree=cKDTree(x); pairs=tree.query_pairs(h, output_type="ndarray")
        poly=315./(64.*math.pi*h**9); rho=m*(poly*h**6)
        if len(pairs)==0: return np.zeros_like(x),rho
        i,j=pairs[:,0],pairs[:,1]; d=x[i]-x[j]; dist=np.linalg.norm(d,axis=1); ok=dist>1e-9; i,j,d,dist=i[ok],j[ok],d[ok],dist[ok]
        w=poly*np.maximum(h*h-dist*dist,0.)**3; np.add.at(rho,i,m[j]*w); np.add.at(rho,j,m[i]*w); rho=np.clip(rho,250.,1800.)
        p=np.clip(self.cs2*(rho-self.rho0),-250.,5e4); unit=d/dist[:,None]; grad=-45./(math.pi*h**6)*(h-dist)**2; com=p[i]/rho[i]**2+p[j]/rho[j]**2
        ai=-(m[j]*com*grad)[:,None]*unit; aj=(m[i]*com*grad)[:,None]*unit; lap=45./(math.pi*h**6)*(h-dist); dv=v[j]-v[i]
        ai+=(self.visc*m[j]*lap/(rho[i]*rho[j]))[:,None]*dv; aj-=(self.visc*m[i]*lap/(rho[i]*rho[j]))[:,None]*dv
        a=np.zeros((n,3)); np.add.at(a,i,ai); np.add.at(a,j,aj); return a,rho

    def _fluid_contacts(self):
        ff=np.zeros_like(self.wx); bf=np.zeros_like(self.rx); bt=np.zeros_like(self.rx); gaps=np.full(self.nr,10.)
        for b in range(self.nr):
            R=_R(self.rq[b]); relall=self.wx-self.rx[b]; ids=np.flatnonzero(np.linalg.norm(relall,axis=1)<1.15)
            if not len(ids): continue
            rel=relall[ids]
            if self.box[b]:
                local=rel@R; half=np.array([self.rad,self.rad,self.hh]); cl=np.clip(local,-half,half); od=local-cl; odist=np.linalg.norm(od,axis=1); inside=np.all(np.abs(local)<=half,axis=1); nl=np.zeros_like(local); out=~inside; good=out&(odist>1e-10); nl[good]=od[good]/odist[good,None]; signed=odist-self.wrad
                if np.any(inside):
                    ii=np.flatnonzero(inside); mar=half[None,:]-np.abs(local[ii]); ax=np.argmin(mar,axis=1); sg=np.sign(local[ii,ax]); sg[sg==0]=1.; nl[ii,ax]=sg; cl[ii,ax]=sg*half[ax]; signed[ii]=-mar[np.arange(len(ii)),ax]-self.wrad
                normal=nl@R.T; cp=self.rx[b]+cl@R.T
            else:
                axis=R[:,2]; axl=np.clip(rel@axis,-self.seg,self.seg); cp=self.rx[b]+axl[:,None]*axis; radial=self.wx[ids]-cp; rd=np.linalg.norm(radial,axis=1); normal=np.zeros_like(radial); good=rd>1e-10; normal[good]=radial[good]/rd[good,None]; normal[~good]=R[:,0]; signed=rd-self.rad-self.wrad
            gaps[b]=np.min(signed); active=signed<0
            if not np.any(active): continue
            ids2=ids[active]; nrm=normal[active]; lever=cp[active]-self.rx[b]; sv=self.rv[b]+np.cross(self.rw[b],lever); relv=self.wv[ids2]-sv; vn=np.sum(relv*nrm,axis=1); mag=np.clip(self.fk*(-signed[active])+self.fd*np.maximum(-vn,0),0,8e3); F=mag[:,None]*nrm; tan=relv-vn[:,None]*nrm; tn=np.linalg.norm(tan,axis=1); mov=tn>1e-9; fr=np.zeros_like(F); fr[mov]=-np.minimum(22*tn[mov],.22*mag[mov])[:,None]*tan[mov]/tn[mov,None]; F+=fr; np.add.at(ff,ids2,F); bf[b]-=F.sum(axis=0); bt[b]+=np.sum(np.cross(lever,-F),axis=0)
        return ff,bf,bt,gaps

    def _rigid_contacts(self):
        f=np.zeros_like(self.rx); t=np.zeros_like(self.rx); ext=np.zeros_like(self.rx); axes=np.array([_R(q)[:,2] for q in self.rq])
        for a in range(self.nr):
            pa,qa=self.rx[a]-self.seg*axes[a],self.rx[a]+self.seg*axes[a]
            for b in range(a+1,self.nr):
                if np.linalg.norm(self.rx[a]-self.rx[b])>2.25: continue
                pb,qb=self.rx[b]-self.seg*axes[b],self.rx[b]+self.seg*axes[b]; ca,cb=_closest(pa,qa,pb,qb); d=ca-cb; dist=np.linalg.norm(d); pen=2*self.rad-dist
                if pen<=0: continue
                n=d/dist if dist>1e-9 else np.array([1.,0.,0.]); la,lb=ca-self.rx[a],cb-self.rx[b]; va=self.rv[a]+np.cross(self.rw[a],la); vb=self.rv[b]+np.cross(self.rw[b],lb); mag=min(self.rk*pen+self.rd*max(-(va-vb)@n,0),2e5); F=mag*n; f[a]+=F; f[b]-=F; t[a]+=np.cross(la,F); t[b]+=np.cross(lb,-F)
        for b in range(self.nr):
            R=_R(self.rq[b]); pts=(np.array([[sx*self.rad,sy*self.rad,sz*self.hh] for sx in(-1.,1.) for sy in(-1.,1.) for sz in(-1.,1.)]) if self.box[b] else np.array([[self.rad*math.cos(th),self.rad*math.sin(th),sz*self.hh] for sz in(-1.,1.) for th in np.linspace(0,2*math.pi,8,endpoint=False)])); lev=pts@R.T; world=self.rx[b]+lev; pv=self.rv[b]+np.cross(self.rw[b],lev)
            for p,l,vel in zip(world,lev,pv):
                if p[2]<0:
                    normal_mag=min(.18*self.rk*(-p[2])+.12*self.rd*max(-vel[2],0),5e4); F=np.array([0.,0.,normal_mag]); speed=np.linalg.norm(vel[:2])
                    if speed>1e-9: F[:2]=-min(50.*speed,.55*normal_mag)*vel[:2]/speed
                    f[b]+=F; ext[b]+=F; t[b]+=np.cross(l,F)
            sx=self.rad+self.hh*abs(axes[b,0]); sy=self.rad+self.hh*abs(axes[b,1]); walls=((self.rx[b,0]-sx+4,np.array([1.,0,0])),(4-self.rx[b,0]-sx,np.array([-1.,0,0])),(self.rx[b,1]-sy+2,np.array([0.,1,0])),(2-self.rx[b,1]-sy,np.array([0.,-1,0])))
            for gap,n in walls:
                if gap<0:
                    F=(self.rk*(-gap)+self.rd*max(-self.rv[b]@n,0))*n; f[b]+=F; ext[b]+=F
        return f,t,ext

    def _bound(self, old, dt):
        v=self.wv; r=self.wrad
        for ax,lo,sg in ((0,-4+r,1),(0,4-r,-1),(1,-2+r,1),(1,2-r,-1),(2,r,1)):
            hit=self.wx[:,ax]<lo if sg>0 else self.wx[:,ax]>lo
            if np.any(hit):
                self.wx[hit,ax]=lo; ids=np.flatnonzero(hit & (v[:,ax]*sg<0)); v[ids,ax]*=-.16; oth=[k for k in range(3) if k!=ax]; v[np.ix_(ids,oth)]*=.94
        return self.wm[:,None]*(v-old)/max(dt,1e-12)

    def _step(self,dt):
        sa,rho=self._sph(); ff,bf,bt,_=self._fluid_contacts(); rf,rt,re=self._rigid_contacts(); self.wv+=dt*(sa+self.g+ff/self.wm[:,None]); self.wv*=math.exp(-.025*dt); old=self.wv.copy(); self.wx+=dt*self.wv; wb=self._bound(old,dt); self.rv+=dt*(bf+rf+self.rm[:,None]*self.g)/self.rm[:,None]; self.rv*=math.exp(-.035*dt)
        for b in range(self.nr):
            R=_R(self.rq[b]); inv=R@np.diag(1/self.inertia[b])@R.T; self.rw[b]+=dt*(inv@(bt[b]+rt[b])); self.rw[b]*=math.exp(-.08*dt); self.rq[b]=_qstep(self.rq[b],self.rw[b],dt)
        self.rx+=dt*self.rv; self.rho=rho; self.ext=self.mass[:,None]*self.g; self.ext[:self.nw]+=wb; self.ext[self.nw:]+=re; internal=rf-re; light=~self.box; q=np.sum(ff,axis=0); self.inter=np.vstack((q,np.sum(bf[light]+internal[light],axis=0),np.sum(bf[self.box]+internal[self.box],axis=0))); self.time+=dt; self.steps+=1
        for a in (self.wx,self.wv,self.rx,self.rv,self.rq,self.rw): np.nan_to_num(a,copy=False); np.clip(a,-100,100,out=a) if a.ndim==2 and a.shape[1]==3 else None

    def _gaps(self):
        g=np.full(self.nr,10.); floor=np.empty(self.nr)
        for b in range(self.nr):
            R=_R(self.rq[b]); rel=self.wx-self.rx[b]
            if self.box[b]:
                l=rel@R; h=np.array([self.rad,self.rad,self.hh]); out=np.maximum(np.abs(l)-h,0); sd=np.linalg.norm(out,axis=1)+np.minimum(np.max(np.abs(l)-h,axis=1),0)-self.wrad; verts=np.array([[sx*self.rad,sy*self.rad,sz*self.hh] for sx in(-1.,1.) for sy in(-1.,1.) for sz in(-1.,1.)]); floor[b]=np.min(self.rx[b,2]+(verts@R.T)[:,2])
            else:
                ax=R[:,2]; z=np.clip(rel@ax,-self.seg,self.seg); sd=np.linalg.norm(rel-z[:,None]*ax,axis=1)-self.rad-self.wrad; floor[b]=self.rx[b,2]-self.seg*abs(ax[2])-self.rad
            g[b]=np.min(sd)
        return np.r_[g,floor]

    def advance_to(self,time:float):
        target=float(time)
        if target<self.time-1e-10: raise ValueError("advance_to times must be nondecreasing")
        while self.time<target-1e-12: self._step(min(self.dtmax,target-self.time))
        self.time=target; pos=np.vstack((self.wx,self.rx)); vel=np.vstack((self.wv,self.rv)); ori=np.tile([0.,0.,0.,1.],(len(pos),1)); ori[self.nw:]=self.rq; ang=np.zeros_like(pos); ang[self.nw:]=self.rw; den=self.density0.copy(); den[:self.nw]=self.rho
        fr=StateFrame(time=target,position=pos.copy(),velocity=vel.copy(),orientation=ori,angular_velocity=ang,mass=self.mass.copy(),body_id=self.body_id.copy(),attributes={"scene_group":self.groups.copy(),"density":den,"mass_quantity":self.mass.copy()},fields={"density":den.copy(),"contact_signed_distance":self._gaps(),"cumulative_mass_source":np.zeros(3),"external_force":self.ext.copy(),"interaction_force":self.inter.copy()},scalars={"water_rest_density":self.rho0,"initial_water_speed":3.2,"contact_length_scale":self.contact_len}); fr.validate(); return fr

    def diagnostics(self):
        return {"method":"weakly_compressible_sph_with_six_dof_rigid_contact","physics_steps":int(self.steps),"water_samples":int(self.nw),"rigid_bodies":int(self.nr),"simulated_time":float(self.time)}
