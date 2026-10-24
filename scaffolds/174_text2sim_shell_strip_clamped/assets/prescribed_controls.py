"""Original prescribed invisible end-frame targets only; no dynamics."""
from __future__ import annotations
import math
import numpy as np
from typing import Any
INITIAL_HOLD_END_SEC = 0.50
TWIST_END_SEC = 5.00
SHORTEN_END_SEC = 6.80
FINAL_END_SEC = 10.0
FINAL_ANGLE_RAD = 3.5 * math.pi
END_APPROACH_M = 0.12
LEFT_SEED_OFFSET = (0.0, 0.014, 0.008)
RIGHT_SEED_OFFSET = (0.0, -0.011, -0.006)
def _targets(contract: dict[str, Any], t: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    command = _command(t)
    angle = float(command["angle_magnitude_rad"])
    left_angle = -angle
    right_angle = angle
    approach = float(command["approach_each_m"])
    left_seed = np.asarray(command["left_seed_offset"], dtype=np.float64)
    right_seed = np.asarray(command["right_seed_offset"], dtype=np.float64)
    left_center = contract["left_origin"] + np.array((approach, 0.0, 0.0)) + left_seed
    right_center = contract["right_origin"] + np.array((-approach, 0.0, 0.0)) + right_seed
    left = left_center + _rotate_x(contract["left_offsets"], left_angle)
    right = right_center + _rotate_x(contract["right_offsets"], right_angle)
    command.update(
        {
            "left_angle_rad": left_angle,
            "right_angle_rad": right_angle,
            "relative_angle_rad": right_angle - left_angle,
            "left_center": left_center,
            "right_center": right_center,
        }
    )
    return left, right, command

def _command(t: float) -> dict[str, Any]:
    if t <= INITIAL_HOLD_END_SEC:
        phase = "initial_hold"
        angle = 0.0
        approach = 0.0
        seed_scale = 0.0
    elif t <= TWIST_END_SEC:
        phase = "helical_twist_at_fixed_span"
        q = _quintic((t - INITIAL_HOLD_END_SEC) / (TWIST_END_SEC - INITIAL_HOLD_END_SEC))
        angle = FINAL_ANGLE_RAD * q
        approach = 0.0
        seed_scale = 0.0
    elif t <= SHORTEN_END_SEC:
        phase = "twist_to_writhe_shortening"
        q = _quintic((t - TWIST_END_SEC) / (SHORTEN_END_SEC - TWIST_END_SEC))
        angle = FINAL_ANGLE_RAD
        approach = END_APPROACH_M * q
        seed_scale = q
    else:
        phase = "fixed_target_elastic_settle"
        angle = FINAL_ANGLE_RAD
        approach = END_APPROACH_M
        seed_scale = 1.0
    return {
        "phase": phase,
        "angle_magnitude_rad": angle,
        "approach_each_m": approach,
        "left_seed_offset": np.asarray(LEFT_SEED_OFFSET, dtype=np.float64) * seed_scale,
        "right_seed_offset": np.asarray(RIGHT_SEED_OFFSET, dtype=np.float64) * seed_scale,
    }

def _rotate_x(offsets: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = offsets.copy()
    result[:, 1] = cosine * offsets[:, 1] - sine * offsets[:, 2]
    result[:, 2] = sine * offsets[:, 1] + cosine * offsets[:, 2]
    return result

def _quintic(value: float) -> float:
    u = min(1.0, max(0.0, float(value)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
def target_at(time_sec, rest_vertices, left_indices, right_indices):
    left=rest_vertices[left_indices]; right=rest_vertices[right_indices]
    contract={'left_origin':left.mean(axis=0),'right_origin':right.mean(axis=0),
              'left_offsets':left-left.mean(axis=0),'right_offsets':right-right.mean(axis=0)}
    return _targets(contract,time_sec)
