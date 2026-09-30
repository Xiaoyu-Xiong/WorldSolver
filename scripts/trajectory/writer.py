from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from worldsolver.contracts import StateFrame

SCHEMA_VERSION = "2.0"


class TrajectoryWriter:
    """Append validated frames without retaining the full simulation in memory."""

    def __init__(
        self,
        destination: Path,
        metadata: dict[str, object] | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.temporary = destination.with_suffix(destination.suffix + ".tmp")
        self.handle = h5py.File(self.temporary, "w", libver="latest")
        self.handle.attrs["worldsolver_schema"] = SCHEMA_VERSION
        self.handle.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)
        self.times = self.handle.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8")
        self.frames = self.handle.create_group("frames", track_order=True)
        self.frame_count = 0
        self.previous_time = -float("inf")
        self.closed = False

    def append(self, frame: StateFrame) -> None:
        if self.closed:
            raise RuntimeError("Cannot append to a closed trajectory")
        frame.validate()
        if self.frame_count and frame.time <= self.previous_time:
            raise ValueError("Frame times must be strictly increasing")
        self.previous_time = frame.time
        self.times.resize((self.frame_count + 1,))
        self.times[self.frame_count] = frame.time
        group = self.frames.create_group(f"{self.frame_count:08d}")
        for name in (
            "position",
            "velocity",
            "orientation",
            "angular_velocity",
            "mass",
            "body_id",
        ):
            value = getattr(frame, name)
            if value is not None:
                group.create_dataset(name, data=np.asarray(value), compression="gzip", shuffle=True)
        scalars = group.create_group("scalars")
        for name, value in sorted(frame.scalars.items()):
            scalars.attrs[name] = float(value)
        attributes = group.create_group("attributes")
        for name, value in sorted(frame.attributes.items()):
            attributes.create_dataset(
                name, data=np.asarray(value), compression="gzip", shuffle=True
            )
        fields = group.create_group("fields")
        for name, value in sorted(frame.fields.items()):
            fields.create_dataset(name, data=np.asarray(value), compression="gzip", shuffle=True)
        self.frame_count += 1

    def close(self) -> None:
        if self.closed:
            return
        if self.frame_count == 0:
            self.handle.close()
            self.temporary.unlink(missing_ok=True)
            self.closed = True
            raise ValueError("A trajectory requires at least one StateFrame")
        self.handle.flush()
        self.handle.close()
        os.replace(self.temporary, self.destination)
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.handle.close()
            self.temporary.unlink(missing_ok=True)
            self.closed = True

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


def write_trajectory(
    destination: Path,
    frames: Iterable[StateFrame],
    metadata: dict[str, object] | None = None,
) -> None:
    with TrajectoryWriter(destination, metadata) as writer:
        for frame in frames:
            writer.append(frame)
