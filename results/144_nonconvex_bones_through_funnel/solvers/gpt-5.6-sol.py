"""Deterministic rigid-body solver for bones falling through the funnel."""
from __future__ import annotations
import math
import numpy as np
from worldsolver.contracts import StateFrame

def _quat_matrix(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]],float)

def _integrate_quaternion(q,omega,dt):
    speed=float(np.linalg.norm(omega))
    if speed*dt<1e-12:return q.copy()
    axis=omega/speed;s=math.sin(.5*speed*dt);dv=axis*s;dw=math.cos(.5*speed*dt);qv=q[:3];qw=q[3]
    out=np.empty(4);out[:3]=dw*qv+qw*dv+np.cross(dv,qv);out[3]=dw*qw-np.dot(dv,qv)
    n=np.linalg.norm(out);return out/n if n>1e-14 else np.array([0.,0.,0.,1.])

class BenchmarkSolver:
    def initialize(self,scene,seed:int)->None:
        c=scene.config;s=scene.initial_state()
        self.position=np.asarray(s.position,float).copy();self.velocity=np.asarray(s.velocity,float).copy();self.orientation=np.asarray(s.orientation,float).copy();self.omega=np.asarray(s.angular_velocity,float).copy();self.mass=np.asarray(s.mass,float).copy();self.body_id=np.asarray(s.body_id,np.int32).copy();self.inv_mass=1/self.mass
        self.gravity=np.asarray(c['gravity'],float);ct=c['contact'];self.friction=float(ct['friction']);self.restitution=float(ct['restitution']);self.contact_scale=float(ct['length_scale']);self.external_force=self.mass[:,None]*self.gravity[None,:]
        self.time=0.;self.steps=0;self.contact_count=0;self.local_centers=np.array([[-.1,0,-.38],[0,0,-.235],[0,0,0],[0,0,.235],[.1,0,.38]],float);self.radii=np.array([.158,.106,.106,.106,.158]);self.inv_inertia_body=1/(self.mass[:,None]/.35*np.array([.04,.0425,.0068])[None,:]);self.funnel_bottom=.48;self.funnel_top=1.72;self.slope_x=(1.27-.30)/(self.funnel_top-self.funnel_bottom);self.slope_y=(.87-.20)/(self.funnel_top-self.funnel_bottom);self.substep=.0025;self._last_forces=np.empty((0,3));self._last_distances=np.array([1.]);self._refresh_diagnostics()

    def _world_geometry(self):
        n=len(self.mass);R=np.empty((n,3,3));C=np.empty((n,5,3));I=np.empty((n,3,3))
        for i in range(n):R[i]=_quat_matrix(self.orientation[i]);C[i]=self.position[i]+self.local_centers@R[i].T;I[i]=R[i]@np.diag(self.inv_inertia_body[i])@R[i].T
        return C,I

    def _contacts(self,margin=.012):
        C,I=self._world_geometry();contacts=[];dist=[];n=len(self.mass)
        for i in range(n-1):
            for j in range(i+1,n):
                if np.dot(self.position[i]-self.position[j],self.position[i]-self.position[j])>1.44:continue
                cand=[];near=float('inf')
                for si,ri in enumerate(self.radii):
                    for sj,rj in enumerate(self.radii):
                        dvec=C[i,si]-C[j,sj];d=float(np.linalg.norm(dvec));gap=d-ri-rj;near=min(near,gap)
                        if gap<margin:
                            normal=dvec/d if d>1e-10 else (np.array([1.,0,0]) if (i+j)%2==0 else np.array([0.,1.,0]));ai=C[i,si]-self.position[i]-normal*ri;aj=C[j,sj]-self.position[j]+normal*rj;cand.append((gap,i,j,ai,aj,normal))
                if near<.08:dist.append(near)
                cand.sort(key=lambda x:x[0]);pts=[]
                for q in cand:
                    p=self.position[i]+q[3]
                    if all(np.linalg.norm(p-o)>.10 for o in pts):contacts.append((q[1],q[2],q[3],q[4],q[5],q[0]));pts.append(p)
                    if len(pts)==2:break
        nx=math.sqrt(1+self.slope_x**2);ny=math.sqrt(1+self.slope_y**2);walls=((0,1,self.slope_x,nx,.30),(0,-1,self.slope_x,nx,.30),(1,1,self.slope_y,ny,.20),(1,-1,self.slope_y,ny,.20))
        for i in range(n):
            for si,r in enumerate(self.radii):
                c=C[i,si];fg=c[2]-r
                if fg<.08:dist.append(fg)
                if fg<margin:
                    no=np.array([0.,0.,1.]);contacts.append((i,-1,c-self.position[i]-no*r,np.zeros(3),no,fg))
                if not(self.funnel_bottom-r<=c[2]<=self.funnel_top+r):continue
                for axis,side,sl,no,bottom in walls:
                    gap=(bottom+sl*(c[2]-self.funnel_bottom)-side*c[axis])/no-r
                    if gap<.08:dist.append(gap)
                    if gap<margin:
                        normal=np.array([0.,0.,sl]);normal[axis]=-side;normal/=no;contacts.append((i,-1,c-self.position[i]-normal*r,np.zeros(3),normal,gap))
        return contacts,np.asarray(dist if dist else [1.]),I

    @staticmethod
    def _effective(im_a,im_b,ii_a,ii_b,aa,ab,d):
        ca=np.cross(aa,d);v=im_a+np.dot(ca,ii_a@ca)
        if ii_b is not None:cb=np.cross(ab,d);v+=im_b+np.dot(cb,ii_b@cb)
        return max(float(v),1e-12)

    def _apply(self,b,arm,imp,I):self.velocity[b]+=self.inv_mass[b]*imp;self.omega[b]+=I[b]@np.cross(arm,imp)

    def _step(self,dt):
        self.velocity+=self.gravity[None,:]*dt;contacts,dist,I=self._contacts();J=np.zeros((len(contacts),3));touch=np.zeros(len(self.mass),bool)
        for _ in range(7):
            for k,(a,b,aa,ab,no,gap) in enumerate(contacts):
                va=self.velocity[a]+np.cross(self.omega[a],aa)
                if b>=0:vb=self.velocity[b]+np.cross(self.omega[b],ab);imb=self.inv_mass[b];iib=I[b]
                else:vb=np.zeros(3);imb=0.;iib=None
                rel=va-vb;vn=float(np.dot(rel,no));target=max(.18*max(-gap-.001,0)/dt,-self.restitution*vn if vn<-.65 else 0.);den=self._effective(self.inv_mass[a],imb,I[a],iib,aa,ab,no);old=float(np.dot(J[k],no));new=max(0.,old+(target-vn)/den);dn=new-old;imp=dn*no
                if dn:self._apply(a,aa,imp,I);self._apply(b,ab,-imp,I) if b>=0 else None;J[k]+=imp
                va=self.velocity[a]+np.cross(self.omega[a],aa);vb=self.velocity[b]+np.cross(self.omega[b],ab) if b>=0 else np.zeros(3);tv=va-vb;tv-=no*np.dot(tv,no);ts=float(np.linalg.norm(tv))
                if ts>1e-10 and new>0:
                    t=tv/ts;td=self._effective(self.inv_mass[a],imb,I[a],iib,aa,ab,t);ot=J[k]-no*np.dot(J[k],no);trial=ot-ts/td*t;lim=self.friction*new;tn=np.linalg.norm(trial)
                    if tn>lim:trial*=lim/tn
                    dtr=trial-ot;self._apply(a,aa,dtr,I);self._apply(b,ab,-dtr,I) if b>=0 else None;J[k]+=dtr
                if np.linalg.norm(J[k])>0:
                    touch[a]=True
                    if b>=0: touch[b]=True
        self.omega[touch]*=math.exp(-1.1*dt);self.position+=self.velocity*dt
        for i in range(len(self.mass)):self.orientation[i]=_integrate_quaternion(self.orientation[i],self.omega[i],dt)
        sp=np.linalg.norm(self.velocity,axis=1);m=sp>35;self.velocity[m]*=(35/sp[m])[:,None];sp=np.linalg.norm(self.omega,axis=1);m=sp>90;self.omega[m]*=(90/sp[m])[:,None]
        paired=[]
        for j in J:
            if np.linalg.norm(j)>1e-10:paired.extend((j/dt,-j/dt))
        self._last_forces=np.asarray(paired).reshape(-1,3) if paired else np.empty((0,3));self._last_distances=dist;self.contact_count+=int(np.count_nonzero(np.linalg.norm(J,axis=1)>0));self.steps+=1

    def _refresh_diagnostics(self):_,self._last_distances,_=self._contacts()

    def advance_to(self,time:float):
        target=float(time)
        if not math.isfinite(target) or target<self.time-1e-10:raise ValueError('advance_to requires finite, nondecreasing times')
        while target-self.time>1e-12:dt=min(self.substep,target-self.time);self._step(dt);self.time+=dt
        self.time=target
        if target==0:self._refresh_diagnostics()
        f=StateFrame(target,self.position.copy(),self.velocity.copy(),orientation=self.orientation.copy(),angular_velocity=self.omega.copy(),mass=self.mass.copy(),body_id=self.body_id.copy(),fields={'external_force':self.external_force.copy(),'interaction_force':self._last_forces.copy(),'contact_signed_distance':self._last_distances.copy()},scalars={'contact_length_scale':self.contact_scale});f.validate();return f

    def diagnostics(self)->dict[str,object]:return {'method':'rigid sphere-union sequential impulses','substeps':int(self.steps),'resolved_contacts':int(self.contact_count),'simulated_time':float(self.time)}
