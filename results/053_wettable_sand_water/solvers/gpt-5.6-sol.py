"""Deterministic particle solver for wettable, insoluble sand."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Compact DEM particle solver with conservative wetting transport."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.sand_index = np.flatnonzero(self.body_id == 1)
        self.water_index = np.flatnonzero(self.body_id == 2)
        self.ns, self.nw = len(self.sand_index), len(self.water_index)
        mass = np.asarray(initial.mass, dtype=np.float64)
        self.dry_sand_mass = mass[self.sand_index].copy()
        self.initial_water_mass = mass[self.water_index].copy()
        self.free_water_mass = self.initial_water_mass.copy()
        self.absorbed_water_mass = np.zeros(self.ns)
        self.water_mass_floor = 0.08 * self.initial_water_mass
        self.saturation_capacity = 0.05 * self.dry_sand_mass
        self.gravity = np.asarray(scene.config.get("gravity", [0, 0, -9.81]), dtype=np.float64)
        self.internal_dt = 0.004
        self.sand_radius, self.water_radius = 0.040, 0.033
        self.contact_scale = self.sand_radius + self.water_radius
        sand0 = self.position[self.sand_index]
        self.sand_pair_i, self.sand_pair_j = self._near_pairs(sand0, 0.118)
        self.water_pair_i, self.water_pair_j = np.triu_indices(self.nw, 1)
        d = sand0[self.sand_pair_j] - sand0[self.sand_pair_i]
        close = np.linalg.norm(d, axis=1) < 0.089
        self.diff_i, self.diff_j = self.sand_pair_i[close], self.sand_pair_j[close]
        self.grid_spacing = 0.15
        self.grid_min = np.array([-1.50, -1.05, 0.0])
        self.grid_shape = (20, 15, 10)
        self.last_wetness_source = np.zeros(self.ns)
        self.last_interaction_force = np.zeros((2, 3))
        self.step_count = 0

    @staticmethod
    def _near_pairs(points, cutoff):
        cells = {}
        coords = np.floor(points / cutoff).astype(np.int64)
        for k, c in enumerate(coords):
            cells.setdefault((int(c[0]), int(c[1]), int(c[2])), []).append(k)
        a, b, cutoff2 = [], [], cutoff * cutoff
        offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
        for i, c in enumerate(coords):
            base = tuple(int(x) for x in c)
            for ox, oy, oz in offsets:
                for j in cells.get((base[0] + ox, base[1] + oy, base[2] + oz), ()):
                    if j > i and np.dot(points[j] - points[i], points[j] - points[i]) <= cutoff2:
                        a.append(i); b.append(j)
        return np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32)

    @staticmethod
    def _safe_norm(v):
        d = np.sqrt(np.einsum("ij,ij->i", v, v)); return d, v / np.maximum(d, 1e-10)[:, None]

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1e-10:
            raise ValueError("advance_to requires finite, nondecreasing time")
        while self.time + 1e-12 < target:
            dt = min(self.internal_dt, target - self.time); self._step(dt); self.time += dt
        self.time = target
        return self._make_frame()

    def _step(self, dt):
        ps, vs = self.position[self.sand_index], self.velocity[self.sand_index]
        pw, vw = self.position[self.water_index], self.velocity[self.water_index]
        sm = self.dry_sand_mass + self.absorbed_water_mass
        wm = np.maximum(self.free_water_mass, self.water_mass_floor)
        fs, fw = sm[:, None] * self.gravity, wm[:, None] * self.gravity
        i, j = self.sand_pair_i, self.sand_pair_j
        delta = ps[j] - ps[i]; dist, n = self._safe_norm(delta); ov = 2*self.sand_radius - dist; act = ov > 0
        if np.any(act):
            ia, ja, nn = i[act], j[act], n[act]; rel = vs[ja] - vs[ia]; vn = np.einsum("ij,ij->i", rel, nn)
            red = sm[ia] * sm[ja] / (sm[ia] + sm[ja]); mag = np.maximum(0, 3.2*ov[act] - 34*red*vn)
            tan = rel - vn[:, None]*nn; tf = 7*red[:, None]*tan; tl = np.linalg.norm(tf, axis=1)
            tf *= np.minimum(1, 0.58*mag/np.maximum(tl, 1e-12))[:, None]; pf = -mag[:, None]*nn + tf
            np.add.at(fs, ia, pf); np.add.at(fs, ja, -pf)
        wi, wj = self.water_pair_i, self.water_pair_j
        dw = pw[wj] - pw[wi]; dd, nw = self._safe_norm(dw); aw = dd < .105
        if np.any(aw):
            ia, ja, nn, d = wi[aw], wj[aw], nw[aw], dd[aw]; rel = vw[ja] - vw[ia]
            pf = .018*(d-.066)[:, None]*nn + .0016*rel
            np.add.at(fw, ia, pf); np.add.at(fw, ja, -pf)
        ws = pw[:, None, :] - ps[None, :, :]; d2 = np.einsum("wsi,wsi->ws", ws, ws); ct = d2 < self.contact_scale**2
        wf = np.zeros((self.nw, 3)); cw, cs = np.nonzero(ct)
        if len(cw):
            dv, d = ws[cw, cs], np.sqrt(np.maximum(d2[cw, cs], 1e-20)); nn = dv/d[:, None]; rel = vw[cw]-vs[cs]; vn = np.einsum("ij,ij->i", rel, nn)
            red = wm[cw]*sm[cs]/(wm[cw]+sm[cs]); mag = np.maximum(0, 1.15*(self.contact_scale-d)-25*red*vn)
            p = mag[:, None]*nn - 4*red[:, None]*(rel-vn[:, None]*nn); np.add.at(fw, cw, p); np.add.at(fs, cs, -p); np.add.at(wf, cw, p)
        self.last_interaction_force[1] = wf.sum(axis=0); self.last_interaction_force[0] = -self.last_interaction_force[1]
        fs -= .10*sm[:, None]*vs; fw -= .035*wm[:, None]*vw
        vs += dt*fs/sm[:, None]; vw += dt*fw/wm[:, None]; np.clip(vs, -5, 5, out=vs); np.clip(vw, -7, 7, out=vw)
        ps += dt*vs; pw += dt*vw
        self._boundaries(ps, vs, self.sand_radius, .025, .72); self._boundaries(pw, vw, self.water_radius, .08, .93)
        self.last_wetness_source.fill(0)
        d2 = np.einsum("wsi,wsi->ws", pw[:, None, :] - ps[None, :, :], pw[:, None, :] - ps[None, :, :]); near = np.argmin(d2, axis=1); touch = np.sqrt(d2[np.arange(self.nw), near]) < .092
        if np.any(touch):
            ids = np.flatnonzero(touch); si = near[ids]; avail = np.maximum(0, self.free_water_mass[ids]-self.water_mass_floor[ids]); prop = np.minimum(avail, self.initial_water_mass[ids]*1.35*dt)
            req = np.zeros(self.ns); np.add.at(req, si, prop); room = np.maximum(0, self.saturation_capacity-self.absorbed_water_mass); scale = np.minimum(1, room/np.maximum(req, 1e-30)); tr = prop*scale[si]
            old = sm.copy(); mom = np.zeros((self.ns,3)); up = np.zeros(self.ns); np.add.at(up, si, tr); np.add.at(mom, si, tr[:,None]*vw[ids]); self.free_water_mass[ids] -= tr; self.absorbed_water_mass += up
            changed = up > 0
            if np.any(changed):
                vs[changed] = (old[changed,None]*vs[changed] + mom[changed])/(old[changed,None]+up[changed,None]); self.last_wetness_source[changed] = up[changed]/(self.dry_sand_mass[changed]*dt)
        self._diffuse_wetness(dt)
        self.position[self.sand_index], self.position[self.water_index] = ps, pw; self.velocity[self.sand_index], self.velocity[self.water_index] = vs, vw; self.step_count += 1

    def _diffuse_wetness(self, dt):
        if not len(self.diff_i): return
        i, j = self.diff_i, self.diff_j; sat = self.absorbed_water_mass/self.saturation_capacity; cm = .5*(self.saturation_capacity[i]+self.saturation_capacity[j]); flux = 1.25*dt*cm*(sat[j]-sat[i]); ch = np.zeros(self.ns); np.add.at(ch,i,flux); np.add.at(ch,j,-flux)
        scale = 1.
        if np.any(ch < 0): scale = min(scale, float(np.min(self.absorbed_water_mass[ch<0]/np.maximum(-ch[ch<0],1e-30))))
        if np.any(ch > 0): scale = min(scale, float(np.min((self.saturation_capacity[ch>0]-self.absorbed_water_mass[ch>0])/np.maximum(ch[ch>0],1e-30))))
        self.absorbed_water_mass += max(0., min(1., scale))*ch; np.clip(self.absorbed_water_mass,0,self.saturation_capacity,out=self.absorbed_water_mass)

    @staticmethod
    def _boundaries(p, v, r, restitution, floor_friction):
        floor = p[:,2] < r; p[floor,2] = r; down = floor & (v[:,2] < 0); v[down,2] *= -restitution; v[floor,:2] *= floor_friction
        xmin, xmax = -1.47+r, 1.47-r; left, right = p[:,0] < xmin, p[:,0] > xmax; p[left,0] = xmin; p[right,0] = xmax; v[left & (v[:,0]<0),0] *= -restitution; v[right & (v[:,0]>0),0] *= -restitution
        ymax = 1.07-r; back = p[:,1] > ymax; p[back,1] = ymax; v[back & (v[:,1]>0),1] *= -restitution

    def _diagnostic_grids(self):
        shape, cells = self.grid_shape, int(np.prod(self.grid_shape)); count=np.zeros(cells); scalar=np.zeros(cells); source=np.zeros(cells); vsum=np.zeros((cells,3)); vwgt=np.zeros(cells)
        sp=self.position[self.sand_index]; wet=self.absorbed_water_mass/self.dry_sand_mass; c=np.clip(np.floor((sp-self.grid_min)/self.grid_spacing).astype(int),0,np.asarray(shape)-1); flat=np.ravel_multi_index(c.T,shape); np.add.at(count,flat,1); np.add.at(scalar,flat,wet); np.add.at(source,flat,self.last_wetness_source)
        c=np.clip(np.floor((self.position-self.grid_min)/self.grid_spacing).astype(int),0,np.asarray(shape)-1); flat=np.ravel_multi_index(c.T,shape); m=np.concatenate((self.dry_sand_mass+self.absorbed_water_mass,self.free_water_mass)); np.add.at(vsum,flat,m[:,None]*self.velocity); np.add.at(vwgt,flat,m)
        active=(count>0).astype(np.float32); transported=np.divide(scalar,count,out=np.zeros_like(scalar),where=count>0); src=np.divide(source,count,out=np.zeros_like(source),where=count>0); vf=np.divide(vsum,vwgt[:,None],out=np.zeros_like(vsum),where=vwgt[:,None]>0)
        return active.reshape(shape), transported.reshape(shape).astype(np.float32), src.reshape(shape).astype(np.float32), vf.reshape(shape+(3,)).astype(np.float32)

    def _make_frame(self):
        sand_mass=self.dry_sand_mass+self.absorbed_water_mass; mass=np.concatenate((sand_mass,self.free_water_mass)); mq=np.zeros((len(self.position),2)); mq[self.sand_index,0]=self.dry_sand_mass; mq[self.sand_index,1]=self.absorbed_water_mass; mq[self.water_index,1]=self.free_water_mass
        wet=np.ones(len(self.position)); wet[self.sand_index]=self.absorbed_water_mass/self.dry_sand_mass; conc=np.ones(len(self.position)); conc[self.sand_index]=self.absorbed_water_mass/self.saturation_capacity; solid=np.zeros(len(self.position)); solid[self.sand_index]=1
        active, trans, src, vf=self._diagnostic_grids(); ps=self.position[self.sand_index]; pw=self.position[self.water_index]; d=np.sqrt(np.min(np.einsum("wsi,wsi->ws",pw[:,None,:]-ps[None,:,:],pw[:,None,:]-ps[None,:,:]),axis=1))-self.contact_scale
        frame=StateFrame(float(self.time),self.position.copy(),self.velocity.copy(),mass=mass,body_id=self.body_id.copy(),attributes={"material_id":self.body_id.copy(),"solid_phase_fraction":solid,"water_concentration":conc,"wetness":wet,"mass_quantity":mq,"absorbed_water_mass":np.concatenate((self.absorbed_water_mass,np.zeros(self.nw)))},fields={"active_mask":active,"contact_signed_distance":d,"cumulative_mass_source":np.zeros(2),"interaction_force":self.last_interaction_force.copy(),"scalar_source":src,"transported_scalar":trans,"velocity_field":vf},scalars={"contact_length_scale":float(self.contact_scale),"grid_spacing":float(self.grid_spacing),"sand_mass":float(self.dry_sand_mass.sum()),"water_mass":float(self.free_water_mass.sum()+self.absorbed_water_mass.sum()),"free_water_mass":float(self.free_water_mass.sum()),"absorbed_water_mass":float(self.absorbed_water_mass.sum())})
        frame.validate(); return frame

    def diagnostics(self):
        return {"method":"soft-sphere DEM with cohesive water and capillary transport","internal_dt":self.internal_dt,"sand_particles":self.ns,"water_particles":self.nw,"steps":self.step_count,"total_water_mass":float(self.free_water_mass.sum()+self.absorbed_water_mass.sum())}
