"""Source prescribed actuator targets only; no dynamics or trajectory."""
from __future__ import annotations
import math
import numpy as np
from typing import Sequence,Mapping,Any
ActionContractError=ValueError
REQUESTED_DURATION_SEC = 10.0
REQUIRED_STEPS = 1000
SEATING_DWELL_END_SEC = 1.00
FOUNDATION_END_SEC = 2.20
BASE_SEAT_END_SEC = 3.40
BASE_DWELL_END_SEC = 3.70
FIRST_RESEAT_END_SEC = 4.40
FIRST_CONTACT_DWELL_END_SEC = 4.70
FIRST_RELEASE_END_SEC = 5.30
SECOND_RESEAT_END_SEC = 6.10
SECOND_CONTACT_DWELL_END_SEC = 6.35
SECOND_RELEASE_END_SEC = 6.90
THIRD_RESEAT_END_SEC = 7.60
THIRD_CONTACT_DWELL_END_SEC = 8.00
RELAXATION_END_SEC = 8.90
DOF_LABELS = ("approach_slide_m", "writhe_slide_m", "twist_hinge_rad")
APPROACH = 0
WRITHE = 1
TWIST = 2
SIDES = ("left", "right")
KP = np.asarray((120.0, 80.0, 8.0), dtype=np.float64)
KV = np.asarray((18.0, 12.0, 1.2), dtype=np.float64)
FORCE_LIMIT = np.asarray((32.0, 22.0, 2.5), dtype=np.float64)
MAX_COMMAND_INCREMENT = np.asarray((0.0022, 0.0012, 0.065), dtype=np.float64)
KEYFRAMES=[(0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), (1.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), (2.2, [[0.04, 0.016, 2.5], [0.04, -0.016, -2.5]]), (3.4, [[0.122, 0.034, 5.3], [0.118, -0.034, -5.3]]), (3.7, [[0.122, 0.034, 5.3], [0.118, -0.034, -5.3]]), (4.4, [[0.182, 0.046, 6.65], [0.178, -0.046, -6.65]]), (4.7, [[0.182, 0.046, 6.65], [0.178, -0.046, -6.65]]), (5.3, [[0.152, 0.01, 6.05], [0.148, -0.01, -6.05]]), (6.1, [[0.184, -0.036, 6.85], [0.18, 0.036, -6.85]]), (6.35, [[0.184, -0.036, 6.85], [0.18, 0.036, -6.85]]), (6.9, [[0.158, -0.004, 6.2], [0.154, 0.004, -6.2]]), (7.6, [[0.188, 0.038, 7.05], [0.182, -0.038, -7.05]]), (8.0, [[0.188, 0.038, 7.05], [0.182, -0.038, -7.05]]), (8.9, [[0.173, 0.006, 6.2], [0.172, -0.006, -6.2]]), (10.0, [[0.173, 0.006, 6.2], [0.172, -0.006, -6.2]])]
def _targets(time_sec: float, schedule: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target = np.empty((2, 3), dtype=np.float64)
    rate = np.empty((2, 3), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        for channel, label in enumerate(DOF_LABELS):
            value, velocity = _trajectory(time_sec, schedule["trajectories"][f"{side}_{label}"])
            target[side_index, channel] = value
            rate[side_index, channel] = velocity
    return target, rate

def _trajectory(time_sec: float, keyframes: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not math.isfinite(time_sec):
        raise ActionContractError("trajectory time must be finite")
    if time_sec <= keyframes[0][0]:
        return float(keyframes[0][1]), 0.0
    for (start, value_start), (end, value_end) in zip(keyframes[:-1], keyframes[1:]):
        if end <= start:
            raise ActionContractError("trajectory keyframe times must increase")
        if time_sec <= end:
            u = min(1.0, max(0.0, (time_sec - start) / (end - start)))
            blend = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
            blend_rate = 30.0 * u * u * (1.0 - u) ** 2 / (end - start)
            delta = value_end - value_start
            return value_start + delta * blend, delta * blend_rate
    return float(keyframes[-1][1]), 0.0
def target_at(time_sec):
    trajectories={f'{side}_{label}':[(t,values[i][j]) for t,values in KEYFRAMES] for i,side in enumerate(SIDES) for j,label in enumerate(DOF_LABELS)}
    return _targets(time_sec,{'trajectories':trajectories})
