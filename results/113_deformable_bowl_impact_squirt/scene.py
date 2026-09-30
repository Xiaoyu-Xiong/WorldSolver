from __future__ import annotations
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        a = np.load(Path(__file__).parent / 'assets' / 'bowl.npz')
        self.vertices = np.asarray(a['vertices'], float)
        self.faces = np.asarray(a['faces'], np.int32)

    def metadata(self) -> dict:
        return {'scene_id': self.config['scene_id'], 'seed': self.config['seed'], 'phenomenon': 'two-way coupled deformable bowl impact, liquid squirting into shallow water', 'topology': {'bowl_faces': self.faces.tolist()}, 'coupling_requirement': 'Fluid traction changes bowl motion and deformation; the evolving impermeable open bowl changes both contained and surrounding liquid flow.'}

    def initial_state(self) -> StateFrame:
        bowl = self.vertices.copy()
        bowl[:,2] += 2.35
        s = 0.11
        vals = np.arange(-0.62, 0.63, s)
        liquid = np.array([(x,y,z) for x in vals for y in vals for z in np.arange(1.78,2.31,s) if (x/0.72)**2+(y/0.72)**2 < 0.82], float)
        w = 0.13
        water = np.array([(x,y,z) for x in np.arange(-2.08,2.09,w) for y in np.arange(-1.38,1.39,w) for z in np.arange(0.07,0.23,w)], float)
        pos = np.vstack([bowl, liquid, water]); vel = np.zeros_like(pos); vel[:len(bowl)+len(liquid),2] = -1.2
        ids = np.r_[np.full(len(bowl),1,np.int32), np.full(len(liquid),2,np.int32), np.full(len(water),3,np.int32)]
        mass = np.r_[np.full(len(bowl),0.45/len(bowl)), np.full(len(liquid),1000*s**3), np.full(len(water),1000*w**3)]
        f = StateFrame(0.0,pos,vel,mass=mass,body_id=ids,scalars={'gravity_z':-9.81,'bowl_impermeable':1.0,'shallow_water_surface_z':0.22})
        f.validate(); return f

    def sample_times(self) -> list[float]:
        return np.linspace(0.0,float(self.config['duration']),round(float(self.config['duration'])/float(self.config['sample_dt']))+1).tolist()
