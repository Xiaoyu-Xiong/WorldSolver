from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from worldsolver.contracts import validate_required_trajectory_fields
from worldsolver.trajectory.reader import TrajectoryReader


@dataclass(frozen=True, slots=True)
class ValidationReport:
    frame_count: int
    start_time: float
    end_time: float
    max_entities: int
    bytes: int


def validate_trajectory(
    path: Path,
    *,
    expected_times: list[float] | None = None,
    max_bytes: int = 4 * 1024**3,
    time_tolerance: float = 1e-8,
    required_paths: tuple[str, ...] = (),
) -> ValidationReport:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Trajectory exceeds byte budget: {size} > {max_bytes}")
    frames = list(TrajectoryReader(path))
    if not frames:
        raise ValueError("Trajectory has no frames")
    times = np.asarray([frame.time for frame in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory times are not strictly increasing")
    if expected_times is not None and not np.allclose(
        times, np.asarray(expected_times), rtol=0.0, atol=time_tolerance
    ):
        raise ValueError("Trajectory timestamps do not match the requested sample times")
    for frame in frames:
        frame.validate()
        validate_required_trajectory_fields(frame, required_paths)
    with h5py.File(path, "r") as handle:
        if len(handle["frames"]) != len(handle["time"]):
            raise ValueError("Frame group count does not match time dataset")
    return ValidationReport(
        len(frames),
        float(times[0]),
        float(times[-1]),
        max(map(lambda x: len(x.position), frames)),
        size,
    )
