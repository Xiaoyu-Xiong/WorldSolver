"""Deterministic solver-neutral wet-towel scene contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Provide initial coupled state and the two prescribed grasp motions."""

    def __init__(self, config: dict):
        self.config = config
        self._towel_position, self._faces, self._uv = self._make_towel()

    def _make_towel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        towel = self.config["towel"]
        nx, ny = map(int, towel["grid_shape"])
        length, width = map(float, towel["size"])
        x = np.linspace(-0.5 * length, 0.5 * length, nx)
        y = np.linspace(-0.5 * width, 0.5 * width, ny)
        end_z = float(towel["initial_end_height"])
        center_z = float(towel["initial_center_height"])
        z = center_z + (end_z - center_z) * (2.0 * np.abs(x) / length) ** 1.65
        position = np.asarray([[xx, yy, zz] for yy in y for xx, zz in zip(x, z)], dtype=float)
        uv = np.asarray([[ix / (nx - 1), iy / (ny - 1)] for iy in range(ny) for ix in range(nx)])
        faces = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                faces.extend(((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)))
        return position, np.asarray(faces, dtype=np.int32), uv

    def _make_water(self) -> np.ndarray:
        water = self.config["water"]
        lo = np.asarray(water["initial_bounds_min"], dtype=float)
        hi = np.asarray(water["initial_bounds_max"], dtype=float)
        spacing = float(water["particle_spacing"])
        axes = [np.arange(lo[k] + 0.5 * spacing, hi[k], spacing) for k in range(3)]
        return np.asarray([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]], dtype=float)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "gravity": self.config["gravity"],
            "controls": self.config["controls"],
            "coupling_requirements": self.config["coupling_requirements"],
            "towel_topology": {"faces": self._faces.tolist()},
            "boundary_condition": "only grasp-labeled towel end regions follow grip poses",
        }

    def grip_kinematics(self, time: float) -> dict[str, dict[str, object]]:
        controls = self.config["controls"]
        lift, twist = controls["lift"], controls["twist"]
        lift_u = np.clip((time - float(lift["start_time"])) /
                         (float(lift["end_time"]) - float(lift["start_time"])), 0.0, 1.0)
        smooth_lift = lift_u * lift_u * (3.0 - 2.0 * lift_u)
        z0 = float(self.config["towel"]["initial_end_height"])
        z = z0 + (float(lift["end_height"]) - z0) * smooth_lift
        twist_u = np.clip((time - float(twist["start_time"])) /
                          (float(twist["end_time"]) - float(twist["start_time"])), 0.0, 1.0)
        smooth_twist = twist_u * twist_u * (3.0 - 2.0 * twist_u)
        return {
            "grip_left": {"translation": [-1.25, 0.0, z],
                          "rotation_about_x_radians": np.deg2rad(float(twist["left_total_degrees"])) * smooth_twist},
            "grip_right": {"translation": [1.25, 0.0, z],
                           "rotation_about_x_radians": np.deg2rad(float(twist["right_total_degrees"])) * smooth_twist},
        }

    def initial_state(self) -> StateFrame:
        towel, water = self._towel_position, self._make_water()
        position = np.vstack((towel, water))
        nt, nw = len(towel), len(water)
        body_id = np.concatenate((np.zeros(nt, dtype=np.int32), np.ones(nw, dtype=np.int32)))
        towel_cfg, water_cfg = self.config["towel"], self.config["water"]
        dry_mass = (float(towel_cfg["dry_areal_density"]) * np.prod(towel_cfg["size"]) / nt)
        stored_mass = (float(towel_cfg["liquid_storage_capacity_kg_m2"]) *
                       float(towel_cfg["initial_saturation"]) * np.prod(towel_cfg["size"]) / nt)
        water_mass = float(water_cfg["density"]) * float(water_cfg["particle_spacing"]) ** 3
        mass = np.concatenate((np.full(nt, dry_mass + stored_mass), np.full(nw, water_mass)))
        grasp = float(towel_cfg["grasp_length"])
        left = (towel[:, 0] <= towel[:, 0].min() + grasp).astype(float)
        right = (towel[:, 0] >= towel[:, 0].max() - grasp).astype(float)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            body_id=body_id,
            attributes={
                "rest_position": position.copy(),
                "uv": np.vstack((self._uv, np.zeros((nw, 2)))),
                "left_grasp": np.concatenate((left, np.zeros(nw))),
                "right_grasp": np.concatenate((right, np.zeros(nw))),
                "porosity": np.concatenate((np.full(nt, float(towel_cfg["porosity"])), np.zeros(nw))),
                "permeability_m2": np.concatenate((np.full(nt, float(towel_cfg["permeability_m2"])), np.zeros(nw))),
                "saturation": np.concatenate((np.full(nt, float(towel_cfg["initial_saturation"])), np.ones(nw))),
            },
            scalars={"water_density": float(water_cfg["density"]), "water_free_surface_z": float(water_cfg["free_surface_z"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1, dtype=float) * dt).tolist()


def _write_asset(path: Path) -> None:
    nx, ny = 41, 15
    vertices = [[x, y, 0.0] for y in np.linspace(-0.36, 0.36, ny)
                for x in np.linspace(-1.25, 1.25, nx)]
    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix + 1
            faces.extend(((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)))
    lines = ["# 41x15 towel topology; coordinates in meters"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    _write_asset(Path(__file__).parent / "assets" / "towel_topology.obj")
