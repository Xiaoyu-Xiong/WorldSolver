"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic damped mass-spring discretization of the flesh surface."""

    def initialize(self, scene, seed: int) -> None:
        self.scene, self.cfg = scene, scene.config
        state = scene.initial_state()
        self.x0 = np.asarray(state.position, float).copy()
        self.x = self.x0.copy(); self.v = np.zeros_like(self.x)
        self.n = len(self.x); self.mass = np.asarray(state.mass, float).copy()
        self.fixed = np.asarray(state.attributes.get("fixed_mask", np.zeros(self.n)), bool)
        self.web = np.asarray(state.attributes.get("web_region", np.zeros(self.n)), bool)
        asset = Path(__file__).resolve().parents[1] / "assets" / "hand_flesh.obj"
        faces = []
        with asset.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("f "):
                    ids = [int(tok.split("/")[0])-1 for tok in line.split()[1:4]]
                    if len(ids) == 3: faces.append(ids)
        self.faces = np.asarray(faces, np.int32)
        edges = np.vstack((self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]))
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        d = self.x0[self.edges[:,1]] - self.x0[self.edges[:,0]]
        self.rest_len = np.maximum(np.linalg.norm(d, axis=1), 1e-8)
        a = self.x0[self.faces[:,1]] - self.x0[self.faces[:,0]]
        b = self.x0[self.faces[:,2]] - self.x0[self.faces[:,0]]
        ref = np.stack((a, b, np.cross(a,b)), axis=2)
        # Pseudoinverse keeps the export finite even for nearly coplanar facets.
        self.ref_inv = np.linalg.pinv(ref)
        p = self.x0
        tc, ic = np.array([.025,-.045,0.]), np.array([.055,.025,.012])
        wt = np.exp(-((p[:,0]-tc[0])**2/.055**2 + (p[:,1]-tc[1])**2/.065**2 + (p[:,2]-tc[2])**2/.045**2))
        wi = np.exp(-((p[:,0]-ic[0])**2/.060**2 + (p[:,1]-ic[1])**2/.070**2 + (p[:,2]-ic[2])**2/.050**2))
        self.thumb_weight = np.clip(wt,0,1) * (~self.fixed)
        self.web_weight = np.clip(wt*wi*2.5,0,1) * self.web
        self.last_time = 0.0; self.total_drive_force = np.zeros(3)
        self.k_spring, self.k_drive, self.k_bulge, self.damping = .22, .03, .004, .01

    @staticmethod
    def _smoothstep(u):
        u = float(np.clip(u,0,1)); return u*u*(3-2*u)

    def _drive(self, t):
        d = self.cfg["thumb_drive"]; s = self._smoothstep((t-float(d["start_time"])) / max(float(d["end_time"])-float(d["start_time"]),1e-9))
        ang = np.deg2rad(float(d["rotation_degrees"])) * s; c, si = np.cos(ang), np.sin(ang)
        R = np.array([[c,-si,0],[si,c,0],[0,0,1.]])
        center = np.array([.025,-.045,0.]); trans = np.asarray(d["translation"],float)*s
        target = (self.x0-center) @ R.T + center + trans
        target += self.web_weight[:,None] * (.010*s) * np.array([0.,-1.,.75])
        return s, target

    def _step(self, dt, t):
        s, target = self._drive(t); i0,i1 = self.edges[:,0], self.edges[:,1]
        delta = self.x[i1]-self.x[i0]; length = np.linalg.norm(delta,axis=1)
        fedge = self.k_spring*(length-self.rest_len)[:,None]*delta/np.maximum(length,1e-8)[:,None]
        force = np.zeros_like(self.x); np.add.at(force,i0,fedge); np.add.at(force,i1,-fedge)
        drive = self.k_drive*self.thumb_weight[:,None]*(target-self.x)
        bulge = self.k_bulge*s*self.web_weight[:,None]*np.array([0.,-1.,.8])
        force += drive - self.damping*self.v + bulge
        force[self.fixed] = 0.; self.v += dt*force/self.mass[:,None]; self.v *= np.exp(-2.5*dt); self.x += dt*self.v
        self.x[self.fixed] = self.x0[self.fixed]; self.v[self.fixed] = 0.; self.total_drive_force = np.sum(drive + bulge,axis=0)

    def _deformation_gradient(self):
        a = self.x[self.faces[:,1]]-self.x[self.faces[:,0]]; b = self.x[self.faces[:,2]]-self.x[self.faces[:,0]]
        cur = np.stack((a,b,np.cross(a,b)),axis=2)
        return np.einsum("nij,njk->nik",cur,self.ref_inv)

    def advance_to(self, time: float):
        time = float(time)
        if time < self.last_time-1e-9: raise ValueError("advance_to requires nondecreasing time")
        rem = time-self.last_time
        while rem > 1e-10:
            h = min(.004, rem); self._step(h,self.last_time+h); self.last_time += h; rem -= h
        frame = StateFrame(time=time, position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
            attributes={"rest_position":self.x0.copy(),"fixed_mask":self.fixed.astype(np.int32),"web_region":self.web.astype(np.int32),"mass_quantity":self.mass.copy()},
            fields={"budget_external_force":self.total_drive_force.reshape(1,1,3),"cumulative_mass_source":np.zeros(1),"element_deformation_gradient":self._deformation_gradient()},
            scalars={"solid_dimension":3.,"nominal_density":1040.,"reference_force":1.})
        frame.validate(); return frame

    def diagnostics(self):
        return {"integrator":"damped_explicit_mass_spring","node_count":int(self.n),"element_count":int(len(self.faces)),"fixed_nodes":int(self.fixed.sum())}
