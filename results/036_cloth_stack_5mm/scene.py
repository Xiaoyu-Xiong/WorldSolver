"""Frozen deterministic benchmark scaffold."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        root = Path(__file__).resolve().parent / "assets"
        self.data = json.loads((root / "initial_state.json").read_text(encoding="utf-8"))
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    def metadata(self) -> dict:
        return self.meta
    def initial_state(self) -> StateFrame:
        d = self.data
        optional = {}
        for key in ("mass", "body_id", "orientation", "angular_velocity"):
            if key in d:
                optional[key] = np.asarray(d[key], dtype=int if key == "body_id" else float)
        frame = StateFrame(0.0, np.asarray(d["position"], float), np.asarray(d["velocity"], float),
            attributes={k: np.asarray(v) for k,v in d.get("attributes",{}).items()},
            fields={k: np.asarray(v) for k,v in d.get("fields",{}).items()},
            scalars={k: float(v) for k,v in d.get("scalars",{}).items()}, **optional)
        frame.validate()
        return frame
    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration/dt)+1).tolist()

