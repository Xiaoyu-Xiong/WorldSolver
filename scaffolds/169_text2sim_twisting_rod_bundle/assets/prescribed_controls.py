"""Source actuator target schedule only; no dynamics or answer trajectory."""
import math
import numpy as np
SETTLE_END_SEC = 1.0
BRAID_END_SEC = 6.0
COMPRESS_END_SEC = 7.0
SHAPE_END_SEC = 8.0
MAIN_TWIST_RAD = 2.0 * math.pi
FINAL_TWIST_RAD = 6.8
AXIAL_COMPRESSION_M = 0.030
LATERAL_OFFSET_M = 0.025
FINAL_OFFSET = np.asarray(
    (AXIAL_COMPRESSION_M, LATERAL_OFFSET_M, FINAL_TWIST_RAD,
     AXIAL_COMPRESSION_M, -LATERAL_OFFSET_M, -FINAL_TWIST_RAD),
    dtype=np.float64,
)
MAIN_OFFSET = np.asarray((0.0, 0.0, MAIN_TWIST_RAD, 0.0, 0.0, -MAIN_TWIST_RAD), dtype=np.float64)
COMPRESSED_OFFSET = np.asarray((0.026, 0.0, 6.65, 0.026, 0.0, -6.65), dtype=np.float64)
def _target_at(time_sec: float, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time_sec <= SETTLE_END_SEC:
        return baseline.copy(), np.zeros(6, dtype=np.float64)
    if time_sec < BRAID_END_SEC:
        progress, rate = _minimum_jerk(time_sec, SETTLE_END_SEC, BRAID_END_SEC)
        return baseline + progress * MAIN_OFFSET, rate * MAIN_OFFSET
    if time_sec < COMPRESS_END_SEC:
        progress, rate = _minimum_jerk(time_sec, BRAID_END_SEC, COMPRESS_END_SEC)
        delta = COMPRESSED_OFFSET - MAIN_OFFSET
        return baseline + MAIN_OFFSET + progress * delta, rate * delta
    if time_sec < SHAPE_END_SEC:
        progress, rate = _minimum_jerk(time_sec, COMPRESS_END_SEC, SHAPE_END_SEC)
        delta = FINAL_OFFSET - COMPRESSED_OFFSET
        return baseline + COMPRESSED_OFFSET + progress * delta, rate * delta
    return baseline + FINAL_OFFSET, np.zeros(6, dtype=np.float64)

def _minimum_jerk(time_sec: float, start: float, finish: float) -> tuple[float, float]:
    if time_sec <= start:
        return 0.0, 0.0
    if time_sec >= finish:
        return 1.0, 0.0
    duration = finish - start
    u = (time_sec - start) / duration
    value = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    derivative = 30.0 * u**2 * (1.0 - u) ** 2 / duration
    return value, derivative
