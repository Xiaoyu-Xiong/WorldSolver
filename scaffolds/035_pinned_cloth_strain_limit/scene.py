"""Frozen deterministic scaffold for the Figure 15 sphere-drape experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        asset_root = Path(__file__).resolve().parent / "assets"
        self.data = json.loads((asset_root / "initial_state.json").read_text(encoding="utf-8"))
        self.meta = json.loads((asset_root / "metadata.json").read_text(encoding="utf-8"))

    def metadata(self) -> dict:
        return self.meta

    def initial_state(self) -> StateFrame:
        data = self.data
        frame = StateFrame(
            0.0,
            np.asarray(data["position"], dtype=float),
            np.asarray(data["velocity"], dtype=float),
            mass=np.asarray(data["mass"], dtype=float),
            attributes={
                name: np.asarray(value) for name, value in data.get("attributes", {}).items()
            },
            fields={name: np.asarray(value) for name, value in data.get("fields", {}).items()},
            scalars={name: float(value) for name, value in data.get("scalars", {}).items()},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        sample_dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / sample_dt) + 1).tolist()
