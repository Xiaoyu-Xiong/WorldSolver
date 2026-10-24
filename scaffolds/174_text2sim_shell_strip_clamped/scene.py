"""Frozen initial data only; no solution or source simulator."""
from pathlib import Path
import json
import numpy as np
from worldsolver.contracts import StateFrame
class BenchmarkScene:
    def __init__(self, config):
        self.config=config
        self.root=Path(__file__).resolve().parent
        self.contract=json.loads((self.root/'assets/scene_contract.json').read_text())
    def metadata(self):
        return {'scene_id':self.config['scene_id'],'seed':self.config['seed'],'scene_contract':self.contract,'initial_state_asset':'assets/initial_state.npz'}
    def initial_state(self):
        with np.load(self.root/'assets/initial_state.npz',allow_pickle=False) as data:
            return StateFrame(time=0.0,**{k:data[k].copy() for k in ['position','velocity','mass','body_id','orientation','angular_velocity']})
    def sample_times(self):
        return np.linspace(0,self.config['duration'],round(self.config['duration']/self.config['sample_dt'])+1).tolist()
