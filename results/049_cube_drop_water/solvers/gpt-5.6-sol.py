"""Self-contained particle liquid and rigid cube time-stepper."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array((aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz))


def _unit(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array((0., 0., 0., 1.))


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.rng = np.random.default_rng(int(seed))
        cfg = scene.config
        self.g = np.asarray(cfg["gravity"], float)
        self.lo = np.asarray(cfg["domain"]["bounds_min"], float)
        self.hi = np.asarray(cfg["domain"]["bounds_max"], float)
        wc, cc = cfg["initial_conditions"]["water"], cfg["initial_conditions"]["cube"]
        self.rho = float(wc["rest_density"]); self.spacing = 0.12
        s = scene.initial_state()
        self.pos = np.asarray(s.position, float).copy(); self.vel = np.asarray(s.velocity, float).copy()
        self.mass = np.asarray(s.mass, float).copy(); self.body_id = np.asarray(s.body_id, np.int32).copy()
        self.nw = int(np.count_nonzero(self.body_id == 1)); self.side = float(cc["side_length"]); self.half = self.side/2
        self.cube_volume = self.side**3; self.water_level = float(wc["fill_bounds_max"][2])
        self.water_mass = self.mass[:self.nw]; self.cube_mass = float(self.mass[-1])
        self.q = _unit(np.asarray(cc["initial_orientation_xyzw"], float)); self.omega = np.zeros(3)
        self.time = 0.0; self.last_interaction = np.zeros((2,3)); self.last_external = np.zeros_like(self.pos)
        self.last_external[:self.nw] = self.water_mass[:,None]*self.g
        self.last_external[-1] = self.cube_mass*self.g
        self.last_gap = np.array((self.pos[-1,2]-self.half,
                                  self.pos[-1,2]-self.half-self.water_level))

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10: raise ValueError("time must be nondecreasing")
        while self.time < target - 1e-12:
            h = min(0.005, target-self.time); self._step(h); self.time += h
        return self._frame(target)

    def _step(self, h):
        wp, wv, cp, cv = self.pos[:self.nw], self.vel[:self.nw], self.pos[-1], self.vel[-1]
        wm = self.water_mass[:,None]; wa = np.broadcast_to(self.g, wp.shape).copy()
        for a in (0,1):
            low, high = self.lo[a]+.10, self.hi[a]-.10
            dl, dh = np.minimum(wp[:,a]-low,0), np.maximum(wp[:,a]-high,0)
            wa[:,a] += -900*dl - 900*dh - 18*wv[:,a]*((dl<0)|(dh>0))
        df = np.minimum(wp[:,2]-.12,0); wa[:,2] += -1800*df - 22*wv[:,2]*(df<0)
        # Hydrostatic pressure balances gravity below the free surface.  Above
        # it, splash particles are ballistic and fall back naturally.
        submerged_water = np.clip((self.water_level + .06-wp[:,2])/.12, 0., 1.)
        wa[:,2] += -self.g[2]*submerged_water
        wa += -.45*(wv - np.mean(wv,axis=0,keepdims=True))
        bottom = cp[2]-self.half; immersed = float(np.clip((self.water_level-bottom)/self.side,0,1))
        buoy = np.array((0.,0.,self.rho*(-self.g[2])*self.cube_volume*immersed))
        rel = cv-np.mean(wv,axis=0)
        drag = -(.5*self.rho*.9*self.side**2*immersed*np.linalg.norm(rel)+4.)*rel
        ic = buoy+drag
        d = wp-cp; dist = np.linalg.norm(d,axis=1); rad=self.half+.42; wt=np.exp(-(dist/max(rad,1e-6))**2)*(dist<rad)
        if np.any(wt>0):
            wt /= np.sum(wt); splash=self.rho*self.cube_volume*immersed*.55 + 30*max(0.,-cv[2])*immersed
            radial=d/np.maximum(dist[:,None],.05); radial[:,2]=np.maximum(radial[:,2],.15)
            # Zero-net radial pressure variation makes a splash without
            # inventing momentum; the paired body forces sum exactly to zero.
            radial -= np.sum(wt[:,None]*radial,axis=0)
            wf=-wt[:,None]*ic + (wt*splash)[:,None]*radial; wa += wf/wm
        iw=-ic
        near=dist<self.half+.55
        if np.any(near): wa[near] += .8*(cv-wv[near])
        wv += h*wa; wp += h*wv
        for a in (0,1):
            low, high=self.lo[a]+.06,self.hi[a]-.06; hit=wp[:,a]<low; wp[hit,a]=low; wv[hit,a]=np.abs(wv[hit,a])*.18; hit=wp[:,a]>high; wp[hit,a]=high; wv[hit,a]=-np.abs(wv[hit,a])*.18
        hit=wp[:,2]<.12; wp[hit,2]=.12; wv[hit,2]=np.abs(wv[hit,2])*.12
        cv += h*(self.g+ic/self.cube_mass); cp += h*cv
        for a in (0,1):
            low,high=self.lo[a]+self.half,self.hi[a]-self.half
            if cp[a]<low: cp[a]=low; cv[a]=abs(cv[a])*.32
            elif cp[a]>high: cp[a]=high; cv[a]=-abs(cv[a])*.32
        if cp[2]<self.half: cp[2]=self.half; cv[2]=abs(cv[2])*.28; self.omega*=.92
        torque=np.cross(np.array((.025,-.018,0.)),ic); inertia=self.cube_mass*self.side**2/6
        self.omega += h*torque/max(inertia,1e-6); self.omega *= max(0.,1-.35*h)
        self.q=_unit(self.q+.5*h*_quat_mul(np.array((*self.omega,0.)),self.q)); self.pos[-1]=cp; self.vel[-1]=cv
        self.last_interaction[0]=iw; self.last_interaction[1]=ic; self.last_external[:self.nw]=wm*self.g; self.last_external[-1]=self.cube_mass*self.g; self.last_gap[:]=(cp[2]-self.half,bottom-self.water_level)

    def _frame(self, t):
        n=len(self.pos); ori=np.zeros((n,4)); ori[:,3]=1; ori[-1]=self.q; ang=np.zeros_like(self.pos); ang[-1]=self.omega; phase=np.zeros(n); phase[-1]=1
        f=StateFrame(time=float(t),position=self.pos.copy(),velocity=self.vel.copy(),orientation=ori,angular_velocity=ang,mass=self.mass.copy(),body_id=self.body_id.copy(),attributes={"phase":phase,"mass_quantity":self.mass.copy()},fields={"contact_signed_distance":self.last_gap.copy(),"cumulative_mass_source":np.zeros(2),"external_force":self.last_external.copy(),"interaction_force":self.last_interaction.copy()},scalars={"contact_length_scale":self.spacing,"water_rest_density":self.rho,"cube_density":self.cube_mass/self.cube_volume})
        f.validate(); return f

    def diagnostics(self) -> dict[str, object]:
        return {}
