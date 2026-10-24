"""Frozen initial data; no numerical solver or reference trajectory."""
from pathlib import Path
import json
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config):
        self.config = config
        self.root = Path(__file__).resolve().parent
        self.contract = json.loads((self.root / "assets/scene_contract.json").read_text())
    def metadata(self):
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"],
                "scene_contract": self.contract, "initial_state_asset": "assets/initial_state.npz"}
    def initial_state(self):
        with np.load(self.root / "assets/initial_state.npz", allow_pickle=False) as data:
            return StateFrame(time=0.0, position=data["position"].copy(),
                velocity=data["velocity"].copy(), mass=data["mass"].copy(),
                body_id=data["body_id"].copy(), orientation=data["orientation"].copy(),
                angular_velocity=data["angular_velocity"].copy())
    def sample_times(self):
        # Include all 2 s of physical start-up and the subsequent 12 s.
        return np.linspace(0.0, self.config["duration"],
            round(self.config["duration"] / self.config["sample_dt"]) + 1).tolist()
