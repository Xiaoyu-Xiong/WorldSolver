"""Small physical FEM solver for the armadillo roller experiment."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.config = scene.config
        self.x = np.asarray(initial.position, float).copy()
        self.v = np.asarray(initial.velocity, float).copy()
        self.rest = np.asarray(initial.attributes["rest_position"], float).copy()
        self.mass = np.asarray(initial.mass, float).copy()
        self.body_id = np.asarray(initial.body_id, np.int64).copy()
        self.n, self.time, self.steps = len(self.x), 0.0, 0
        tets, invs, vols = [], [], []
        for tet in np.asarray(Delaunay(self.rest).simplices, np.int64):
            dm = np.column_stack((self.rest[tet[1]]-self.rest[tet[0]], self.rest[tet[2]]-self.rest[tet[0]], self.rest[tet[3]]-self.rest[tet[0]]))
            det = float(np.linalg.det(dm)); vol = abs(det)/6.0
            if vol < 2e-5: continue
            if det < 0: tet = tet[[0,2,1,3]]; dm = np.column_stack((self.rest[tet[1]]-self.rest[tet[0]], self.rest[tet[2]]-self.rest[tet[0]], self.rest[tet[3]]-self.rest[tet[0]]))
            tets.append(tet); invs.append(np.linalg.inv(dm)); vols.append(vol)
        self.tets, self.inv_dm, self.vols = np.asarray(tets), np.asarray(invs), np.asarray(vols)
        young = float(self.config.get("phenomenon_targets", {}).get("moderate_stiffness_reference_pa", 5e5)); nu = .30
        self.mu, self.lam = young/(2*(1+nu)), young*nu/((1+nu)*(1-2*nu))
        self.gravity = np.asarray(self.config.get("gravity", [0,0,-9.81]), float)
        bc = self.config.get("boundary_conditions", {})
        self.start, self.ramp = float(bc.get("start_time", .35)), max(float(bc.get("ramp_duration", .25)), 1e-9)
        self.ou, self.ol = abs(float(bc.get("upper_angular_velocity_rad_s", 5))), abs(float(bc.get("lower_angular_velocity_rad_s", -5)))
        self.mu_d = float(bc.get("contact_friction_coefficient", .5)); self.mu_s = self.mu_d
        self.centers = np.asarray([[-.2,0,.3],[.3,0,.3],[-.2,0,-.3],[.3,0,-.3]], float)
        self.radius, self.half_h = .22, .36
        self.C = self.n*4; self.mem = np.zeros((self.C,3)); self.active_prev = np.zeros(self.C, bool)
        self.kn, self.cn, self.kt, self.ct, self.drag, self.max_step = 3.5e4, 55., 1.8e4, 8., .16, 4e-4
        self.impulse = np.zeros(3); self.last_contact = {}

    def _ramp(self, t):
        q = np.clip((t-self.start)/self.ramp, 0., 1.); return q*q*(3-2*q)

    def _elastic(self):
        f = np.zeros_like(self.x); I = np.eye(3)
        for e, tet in enumerate(self.tets):
            ds = np.column_stack((self.x[tet[1]]-self.x[tet[0]], self.x[tet[2]]-self.x[tet[0]], self.x[tet[3]]-self.x[tet[0]])); F = ds@self.inv_dm[e]
            u,_,vt = np.linalg.svd(F); R = u@vt
            if np.linalg.det(R)<0: u[:,-1]*=-1; R=u@vt
            P = 2*self.mu*(F-R) + self.lam*np.trace(R.T@F-I)*R; h = -self.vols[e]*P@self.inv_dm[e].T
            f[tet[1]] += h[:,0]; f[tet[2]] += h[:,1]; f[tet[3]] += h[:,2]; f[tet[0]] -= h.sum(axis=1)
            lv = self.v[tet]-self.v[tet].mean(axis=0); f[tet] += -.008*self.mu*self.vols[e]*lv
        return f

    def _contacts(self, update=True):
        total=np.zeros_like(self.x); A=np.zeros(self.C); N=np.zeros((self.C,3)); FN=np.zeros(self.C); RV=np.zeros((self.C,3)); FT=np.zeros((self.C,3)); sf=self._ramp(self.time)
        for ri, c0 in enumerate(self.centers):
            om=np.array([0., (-self.ou if ri<2 else self.ol)*sf, 0.])
            for i in range(self.n):
                j=ri*self.n+i; radial=self.x[i]-c0; radial[1]=0.; d=np.linalg.norm(radial); n=radial/d if d>1e-12 else np.array([-1.,0,0]); N[j]=n
                surf=np.cross(om,self.radius*n); rel=self.v[i]-surf; RV[j]=rel; pen=self.radius-d
                if pen<=0 or abs(self.x[i,1]-c0[1])>self.half_h:
                    if update: self.mem[j]=0; self.active_prev[j]=False
                    continue
                A[j]=1; vn=np.dot(rel,n); fn=max(0.,self.kn*pen-self.cn*min(vn,0.)); FN[j]=fn; tv=rel-vn*n; m=self.mem[j].copy() if self.active_prev[j] else np.zeros(3)
                if update: m += self._dt*tv
                m -= np.dot(m,n)*n; trial=-self.kt*m-self.ct*tv
                if np.linalg.norm(trial)<=self.mu_s*fn: ft=trial
                else:
                    speed=np.linalg.norm(tv); direction=-tv/speed if speed>1e-10 else (trial/np.linalg.norm(trial) if np.linalg.norm(trial)>1e-12 else np.zeros(3)); ft=self.mu_d*fn*direction
                    if update: m=-(ft+self.ct*tv)/self.kt
                if update: self.mem[j]=m; self.active_prev[j]=True
                FT[j]=ft; total[i]+=fn*n+ft
        return total,{"friction_active_mask":A,"friction_dynamic_coefficient":np.full(self.C,self.mu_d),"friction_normal":N,"friction_normal_force":FN,"friction_relative_velocity":RV,"friction_static_coefficient":np.full(self.C,self.mu_s),"friction_tangential_force":FT}

    def _step(self, dt):
        self._dt=dt; cf,cv=self._contacts(True); ext=self.mass[:,None]*self.gravity+cf-self.drag*self.v; self.v += dt*(self._elastic()+ext)/self.mass[:,None]; self.x += dt*self.v; self.impulse += dt*ext.sum(0); self.time += dt; self.steps += 1; self.last_contact=cv

    def _F(self):
        out=np.empty((len(self.tets),3,3))
        for e,t in enumerate(self.tets): out[e]=np.column_stack((self.x[t[1]]-self.x[t[0]],self.x[t[2]]-self.x[t[0]],self.x[t[3]]-self.x[t[0]]))@self.inv_dm[e]
        return out

    def advance_to(self, time: float):
        target=float(time)
        if not np.isfinite(target) or target<self.time-1e-10: raise ValueError("times must be finite and nondecreasing")
        while self.time<target-1e-12: self._step(min(self.max_step,target-self.time))
        cf, c = self._contacts(False); g=np.sum(self.mass[:,None]*self.gravity,0); ct=cf.sum(0); dr=np.sum(-self.drag*self.v,0); mass=float(self.mass.sum()); rv=self.radius*max(self.ou,self.ol)
        fields=dict(c); fields.update({"budget_cumulative_external_impulse":self.impulse[None,:],"budget_cumulative_inflow_momentum":np.zeros((1,3)),"budget_cumulative_outflow_momentum":np.zeros((1,3)),"budget_domain_weights":np.ones((1,self.n)),"budget_external_force":np.asarray([[g,ct,dr]]),"cumulative_mass_source":np.zeros(1),"element_deformation_gradient":self._F()})
        frame=StateFrame(time=target,position=self.x.copy(),velocity=self.v.copy(),mass=self.mass.copy(),body_id=self.body_id.copy(),attributes={"rest_position":self.rest.copy(),"mass_quantity":self.mass.copy()},fields=fields,scalars={"roller_friction":self.mu_d,"reference_youngs_modulus_pa":self.mu*2*1.3,"friction_stick_speed_tolerance":.025,"reference_force":max(mass*np.linalg.norm(self.gravity),1.),"reference_momentum":max(mass*rv,1e-6),"reference_velocity":max(rv,1e-6)})
        frame.validate(); return frame

    def diagnostics(self):
        return {"method":"corotational_tetrahedral_fem_coulomb_contact","tetrahedra":int(len(self.tets)),"substeps":int(self.steps),"contact_slots":int(self.C)}
