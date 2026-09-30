"""Small explicit finite-element solver for pneumatic cushion inflation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        with np.load(scene._asset) as d:
            self.x0 = np.asarray(d["vertices"], float)
            self.tets = np.asarray(d["tetrahedra"], np.int64)
            self.inner_faces = np.asarray(d["inner_chamber_faces"], np.int64)
            self.nodal_volume = np.asarray(d["nodal_volume"], float)
            self.wall_region = np.asarray(d["wall_region"], np.int32)
        self.n, self.ne = len(self.x0), len(self.tets)
        rho = float(cfg["material_properties"]["density_kg_m3"])
        self.mass = np.maximum(rho * self.nodal_volume, 1e-9)
        self.inv_mass = 1.0 / self.mass
        self.inv_dm = np.empty((self.ne, 3, 3)); self.volume = np.empty(self.ne)
        for q, ids in enumerate(self.tets):
            X = self.x0[ids]
            dm = np.column_stack((X[1]-X[0], X[2]-X[0], X[3]-X[0]))
            self.volume[q] = max(abs(np.linalg.det(dm))/6.0, 1e-12)
            self.inv_dm[q] = np.linalg.inv(dm)
        E = float(cfg["material_properties"]["young_modulus_pa"])
        nu = float(cfg["material_properties"]["poisson_ratio"])
        self.mu = E/(2*(1+nu)); self.lam = min(E*nu/((1+nu)*(1-2*nu)), 5e5)
        load = cfg["loading"]
        self.p0 = float(load["initial_internal_pressure_pa"])
        self.ptarget = float(load["target_internal_pressure_pa"])
        self.ramp_start = float(load["ramp_start_s"]); self.ramp_end = float(load["ramp_end_s"])
        self.external_pressure = float(load["external_pressure_pa"])
        self.center = np.asarray(cfg["geometry"]["chamber_center_m"], float)
        dims = np.asarray(cfg["geometry"]["outer_dimensions_m"], float)
        self.reference_force = max(abs(self.ptarget) * float(np.max(dims[:,None]*dims[None,:])), 1.0)
        self.x = self.x0.copy(); self.v = np.zeros_like(self.x); self.current_time = 0.0
        self._last_external = np.zeros((1, 1, 3)); self._steps = 0
        tri = self.x0[self.inner_faces]
        normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        flip = np.einsum("ij,ij->i", normals, tri.mean(1)-self.center) < 0
        self.inner_faces_oriented = self.inner_faces.copy()
        self.inner_faces_oriented[flip] = self.inner_faces_oriented[flip][:,[0,2,1]]

    def advance_to(self, time: float):
        time = float(time)
        if time < self.current_time - 1e-10: raise ValueError("time must be nondecreasing")
        while self.current_time < time - 1e-12:
            dt = min(5e-4, time-self.current_time)
            self._step(dt, self.current_time+dt); self.current_time += dt
        frame = StateFrame(time=time, position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
            attributes={"wall_region":self.wall_region.copy(), "mass_quantity":self.mass.copy()},
            fields={"budget_external_force":self._last_external.copy(),
                    "cumulative_mass_source":np.zeros(1),
                    "element_deformation_gradient":self._gradients()},
            scalars={"internal_pressure_pa":self._pressure(time), "external_pressure_pa":self.external_pressure,
                     "reference_force":self.reference_force})
        frame.validate(); return frame

    def _pressure(self, t):
        if t <= self.ramp_start: return self.p0
        if t >= self.ramp_end: return self.ptarget
        a=(t-self.ramp_start)/max(self.ramp_end-self.ramp_start,1e-12); a=a*a*(3-2*a)
        return self.p0+(self.ptarget-self.p0)*a

    def _forces(self, t):
        f=np.zeros_like(self.x)
        for q, ids in enumerate(self.tets):
            xx=self.x[ids]; ds=np.column_stack((xx[1]-xx[0],xx[2]-xx[0],xx[3]-xx[0]))
            F=ds@self.inv_dm[q]; J=float(np.linalg.det(F))
            if not np.isfinite(J) or not np.isfinite(F).all():
                F=np.eye(3); J=1.0
            elif J < .15:
                F += np.eye(3)*(.15-J); J=max(float(np.linalg.det(F)),.15)
            try: invT=np.linalg.inv(F).T
            except np.linalg.LinAlgError: invT=np.eye(3)
            P=self.mu*(F-invT)+self.lam*np.log(J)*invT; g=self.inv_dm[q].T
            ft=np.empty((4,3)); ft[1:]=-(self.volume[q]*(P@g).T); ft[0]=-ft[1:].sum(0)
            np.add.at(f,ids,ft)
        tri=self.x[self.inner_faces_oriented]; avec=.5*np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        ff=(self._pressure(t)-self.external_pressure)*avec
        for j in range(3): np.add.at(f,self.inner_faces_oriented[:,j],ff/3)
        return f, ff.sum(0).reshape(1,1,3)

    def _step(self, dt, t):
        force, self._last_external=self._forces(t)
        a=force*self.inv_mass[:,None]-10.0*self.v; a=np.clip(a,-2e4,2e4)
        self.v += dt*a; self.x += dt*self.v
        m=self.mass.sum(); com=(self.x*self.mass[:,None]).sum(0)/m; vc=(self.v*self.mass[:,None]).sum(0)/m
        self.x += self.center-com; self.v -= vc; self._steps += 1

    def _gradients(self):
        out=np.empty((self.ne,3,3))
        for q,ids in enumerate(self.tets):
            xx=self.x[ids]; ds=np.column_stack((xx[1]-xx[0],xx[2]-xx[0],xx[3]-xx[0])); out[q]=ds@self.inv_dm[q]
        return out

    def diagnostics(self) -> dict[str, object]:
        return {"solver":"explicit_neo_hookean_tetrahedral", "node_count":int(self.n), "element_count":int(self.ne), "internal_steps":int(self._steps)}
