"""Deterministic particle/shape-matching solver for the heterogeneous gummy bear."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _read_obj(name: str):
    vertices, edges = [], set()
    path = Path(__file__).resolve().parents[1] / "assets" / name
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(a.split("/")[0]) - 1 for a in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                for a, b in ((ids[0], ids[i]), (ids[i], ids[i + 1])):
                    edges.add((min(a, b), max(a, b)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(sorted(edges), dtype=np.int64)


def _mat_to_quat(r):
    q = np.empty(4, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[3], q[0], q[1], q[2] = .25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s
    else:
        k = int(np.argmax(np.diag(r))); j, l = (k+1)%3, (k+2)%3
        s = np.sqrt(max(1e-12, 1+r[k,k]-r[j,j]-r[l,l]))*2
        q[k], q[3], q[j], q[l] = .25*s, (r[l,j]-r[j,l])/s, (r[j,k]+r[k,j])/s, (r[l,k]+r[k,l])/s
    return q / max(np.linalg.norm(q), 1e-12)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        outer, oe = _read_obj("gummy_bear_outer.obj")
        core, ce = _read_obj("gummy_bear_stiff.obj")
        self.n_outer, self.n_core = len(outer), len(core)
        self.n = self.n_outer + self.n_core
        self.ref = np.vstack((outer, core)); self.x = self.ref.copy()
        self.v = np.empty_like(self.x)
        xx, zz = self.x[:, 0], self.x[:, 2]
        self.v[:, 0] = .30*(zz-1.15); self.v[:, 1] = 1.20*xx+.22; self.v[:, 2] = -1.45*xx-.38*xx*xx
        self.body = np.r_[np.zeros(self.n_outer, dtype=np.int32), np.ones(self.n_core, dtype=np.int32)]
        self.mass = np.full(self.n, 1.0/self.n); self.inv_mass = 1.0/self.mass
        self.gravity = np.asarray(scene.config["initial_conditions"]["gravity"], dtype=float)
        self.ground = float(scene.config["boundary_conditions"]["ground_height"])
        self.time = 0.0; self.dt_max = .0008; self.damping = .16
        self.edges = np.vstack((oe, ce+self.n_outer))
        self.rest_len = np.linalg.norm(self.ref[self.edges[:,1]]-self.ref[self.edges[:,0]], axis=1)
        self.edge_k = np.r_[np.full(len(oe), 95.0), np.zeros(len(ce))]
        self.edge_d = np.r_[np.full(len(oe), .45), np.zeros(len(ce))]
        d2 = ((core[:,None,:]-outer[None,:,:])**2).sum(2)
        self.core_to_outer = np.argmin(d2, axis=1); ci = self.n_outer+np.arange(self.n_core)
        self.bond_rest = self.ref[ci]-self.ref[self.core_to_outer]; self.bond_k, self.bond_d = 34., .8
        self.impulse = np.zeros((2,3)); self.interaction = np.zeros((2,3)); self._last_force = np.zeros((2,2,3))
        self._core_rot = np.eye(3); self._core_omega = np.zeros(3)
        self._weights = np.zeros((2,self.n)); self._weights[0,:self.n_outer] = 1; self._weights[1,self.n_outer:] = 1

    def _step(self, h):
        f = self.mass[:,None]*self.gravity[None,:]
        ext = np.zeros((2,2,3)); ext[:,0,:] = np.array([self.mass[:self.n_outer].sum(), self.mass[self.n_outer:].sum()])[:,None]*self.gravity
        a,b = self.edges[:,0], self.edges[:,1]; d = self.x[b]-self.x[a]; ln = np.linalg.norm(d,axis=1); nv = d/np.maximum(ln[:,None],1e-9)
        rel = np.sum((self.v[b]-self.v[a])*nv,axis=1); sf = nv*(self.edge_k*(ln-self.rest_len)+self.edge_d*rel)[:,None]
        np.add.at(f,a,sf); np.add.at(f,b,-sf)
        oi = self.core_to_outer; ci = self.n_outer+np.arange(self.n_core); cur = self.x[ci]-self.x[oi]; diff = cur-self.bond_rest
        bl = np.linalg.norm(cur,axis=1); bn = cur/np.maximum(bl[:,None],1e-9); br = np.sum((self.v[ci]-self.v[oi])*bn,axis=1)
        bf = bn*(self.bond_k*np.sum(diff*bn,axis=1)+self.bond_d*br)[:,None]; np.add.at(f,oi,bf); np.add.at(f,ci,-bf)
        net = bf.sum(0); self.interaction[0], self.interaction[1] = net, -net
        f -= self.damping*self.mass[:,None]*self.v
        pen = self.ground-self.x[:self.n_outer,2]; active = pen>0
        if np.any(active):
            normal = np.maximum(180*pen[active]-12*np.minimum(self.v[active,2],0),0); f[np.where(active)[0],2] += normal; ext[0,1,2] = normal.sum()
        old_core = self.x[ci].copy(); self.v += h*f*self.inv_mass[:,None]; self.x += h*self.v
        q = self.ref[self.n_outer:]; qc = q.mean(0); yc = self.x[ci].mean(0); H = (q-qc).T@(self.x[ci]-yc); U,_,Vt = np.linalg.svd(H); R = Vt.T@U.T
        if np.linalg.det(R)<0: Vt[-1] *= -1; R = Vt.T@U.T
        projected = (q-qc)@R.T+yc; self.x[ci] = projected; self.v[ci] = (projected-old_core)/h
        # Unilateral ground constraint.  The shell is the only colliding object;
        # remove any residual penetration after the penalty impulse and prevent
        # downward rebound velocities.
        below = self.x[:self.n_outer, 2] < self.ground
        if np.any(below):
            self.x[np.where(below)[0], 2] = self.ground
            self.v[np.where(below)[0], 2] = np.maximum(self.v[np.where(below)[0], 2], 0.0)
        dR = R @ self._core_rot.T
        self._core_omega = np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])/(2*h); self._core_rot = R
        self._last_force = ext; self.impulse += ext.sum(1)*h

    def _frame(self, time):
        F = np.empty((self.n,3,3))
        for sl in (slice(0,self.n_outer), slice(self.n_outer,self.n)):
            X = self.ref[sl]-self.ref[sl].mean(0); Y = self.x[sl]-self.x[sl].mean(0); F[sl] = (Y.T@X)@np.linalg.inv(X.T@X+1e-8*np.eye(3))
        ori = np.tile([0.,0.,0.,1.], (self.n,1)); ori[self.n_outer:] = _mat_to_quat(self._core_rot)
        ang = np.zeros((self.n,3)); ang[self.n_outer:] = self._core_omega
        fr = StateFrame(float(time), self.x.copy(), self.v.copy(), orientation=ori, angular_velocity=ang, mass=self.mass.copy(), body_id=self.body.copy(), attributes={"mass_quantity":self.mass.copy()}, fields={
            "budget_cumulative_external_impulse":self.impulse.copy(), "budget_cumulative_inflow_momentum":np.zeros((2,3)), "budget_cumulative_outflow_momentum":np.zeros((2,3)), "budget_domain_weights":self._weights.copy(), "budget_external_force":self._last_force.copy(), "cumulative_mass_source":np.zeros(1), "element_deformation_gradient":F, "interaction_force":self.interaction.copy()}, scalars={"reference_momentum":1.0})
        fr.validate(); return fr

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time-1e-9: raise ValueError("advance_to requires nondecreasing times")
        while self.time < target-1e-12:
            h = min(self.dt_max, target-self.time); self._step(h); self.time += h
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"integrator":"semi_implicit_euler_shape_matching", "particle_count":int(self.n), "time":float(self.time)}
