from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from worldsolver.contracts import StateFrame
from worldsolver.trajectory.writer import SCHEMA_VERSION


class TrajectoryReader:
    def __init__(self, path: Path):
        self.path = path

    def metadata(self) -> dict[str, object]:
        with h5py.File(self.path, "r") as handle:
            self._check_schema(handle)
            return json.loads(handle.attrs.get("metadata_json", "{}"))

    def __iter__(self) -> Iterator[StateFrame]:
        with h5py.File(self.path, "r") as handle:
            self._check_schema(handle)
            times = handle["time"][:]
            for index, time in enumerate(times):
                group = handle["frames"][f"{index:08d}"]
                values = {
                    name: group[name][:]
                    for name in (
                        "position",
                        "velocity",
                        "orientation",
                        "angular_velocity",
                        "mass",
                        "body_id",
                    )
                    if name in group
                }
                scalars = dict(group["scalars"].attrs) if "scalars" in group else {}
                attributes = (
                    {name: group["attributes"][name][:] for name in group["attributes"]}
                    if "attributes" in group
                    else {}
                )
                fields = (
                    {name: group["fields"][name][:] for name in group["fields"]}
                    if "fields" in group
                    else {}
                )
                yield StateFrame(
                    time=float(time),
                    scalars=scalars,
                    attributes=attributes,
                    fields=fields,
                    **values,
                )

    def series(self, path: str) -> list[np.ndarray | float]:
        """Read one canonical StateFrame path across all frames."""
        namespace, separator, name = path.partition(".")
        if not separator or not name:
            raise ValueError(f"Invalid trajectory field path: {path}")
        values: list[np.ndarray | float] = []
        for frame in self:
            if namespace == "state":
                value = getattr(frame, name, None)
            elif namespace in {"attributes", "fields", "scalars"}:
                value = getattr(frame, namespace).get(name)
            else:
                raise ValueError(f"Not a trajectory field path: {path}")
            if value is None:
                raise KeyError(path)
            values.append(value)
        return values

    @staticmethod
    def _check_schema(handle: h5py.File) -> None:
        if handle.attrs.get("worldsolver_schema") != SCHEMA_VERSION:
            raise ValueError("Unsupported or missing WorldSolver HDF5 schema")
