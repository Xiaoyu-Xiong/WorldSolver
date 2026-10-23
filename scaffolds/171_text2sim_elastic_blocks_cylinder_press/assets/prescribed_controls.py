"""Original target schedule; no dynamics or recorded trajectory."""
from __future__ import annotations
import numpy as np
from typing import Any,Mapping
def _r(x): return round(float(x),7)
__all__ = ["run_actions"]
_SETTLE_END = 1.0
_FREE_AUDIT_RAMP_END = 1.7
_FREE_AUDIT_END = 2.0
_PRESS_END = 4.1
_DWELL_END = 4.8
_SPIN_RAMP_END = 5.5
_SPIN_HOLD_END = 9.0
_MOTION_END = 9.5
_FREE_AUDIT_SPIN_SPEED = 0.32
_SPIN_SPEED = 0.85
_PLANNED_SPIN = _SPIN_SPEED * (
    0.5 * (_SPIN_RAMP_END - _DWELL_END)
    + (_SPIN_HOLD_END - _SPIN_RAMP_END)
    + 0.5 * (_MOTION_END - _SPIN_HOLD_END)
)
_PRESS_KP = 3000.0
_PRESS_INTERNAL_KV = 215.0
_PRESS_KV_APPROACH = 215.0
_PRESS_KV_LOADED = 55.0
_PRESS_FORCE = 480.0
_SPIN_KP = 0.0
_SPIN_KV = 18.0
_SPIN_TORQUE = 28.0
_PRESS_VELOCITY_FILTER_TAU = 0.04
_PRESS_VELOCITY_CLAMP = 0.45
_PRESS_INTERNAL_VELOCITY_TARGET_LIMIT = 1.0
_LOADED_PHASES = frozenset(
    {
        "pressure_dwell",
        "shear_ramp",
        "shear_hold",
        "shear_decelerate",
        "terminal_pressure_hold",
    }
)
_METRIC_PERIOD = 0.20
_EPS = 1.0e-9
def _command(now: float, control: Mapping[str, Any]) -> dict[str, Any]:
    t = max(0.0, float(now))
    z0 = float(control["home_target_z"])
    audit_start = float(control["audit_start_z"])
    za = float(control["audit_z"])
    z1 = float(control["working_z"])
    if t < _SETTLE_END:
        phase = "gravity_settle"
        z_target, z_velocity, press_progress = z0, 0.0, 0.0
        spin_velocity = 0.0
    elif t < _FREE_AUDIT_RAMP_END:
        phase = "free_space_tracking"
        z_target, z_velocity, press_progress = _minimum_jerk(
            audit_start,
            za,
            t,
            _SETTLE_END,
            _FREE_AUDIT_RAMP_END,
        )
        audit_progress = float(
            np.clip(
                (t - _SETTLE_END) / (_FREE_AUDIT_RAMP_END - _SETTLE_END),
                0.0,
                1.0,
            )
        )
        spin_velocity = _FREE_AUDIT_SPIN_SPEED * 4.0 * audit_progress * (1.0 - audit_progress)
    elif t < _FREE_AUDIT_END:
        phase = "free_space_audit_dwell"
        z_target, z_velocity = za, 0.0
        press_progress = control["audit_travel"] / max(control["stroke"], _EPS)
        spin_velocity = 0.0
    elif t < _PRESS_END:
        phase = "press_descent"
        z_target, z_velocity, continuation = _minimum_jerk(
            za,
            z1,
            t,
            _FREE_AUDIT_END,
            _PRESS_END,
        )
        press_progress = (
            control["audit_travel"]
            + (control["stroke"] - control["audit_travel"]) * continuation
        ) / max(control["stroke"], _EPS)
        spin_velocity = 0.0
    elif t < _DWELL_END:
        phase = "pressure_dwell"
        z_target, z_velocity, press_progress = z1, 0.0, 1.0
        spin_velocity = 0.0
    elif t < _SPIN_RAMP_END:
        phase = "shear_ramp"
        spin_progress, _ = _minimum_jerk_unit(t, _DWELL_END, _SPIN_RAMP_END)
        z_target, z_velocity, press_progress = z1, 0.0, 1.0
        spin_velocity = _SPIN_SPEED * spin_progress
    elif t < _SPIN_HOLD_END:
        phase = "shear_hold"
        z_target, z_velocity, press_progress = z1, 0.0, 1.0
        spin_velocity = _SPIN_SPEED
    elif t < _MOTION_END:
        phase = "shear_decelerate"
        spin_progress, _ = _minimum_jerk_unit(t, _SPIN_HOLD_END, _MOTION_END)
        z_target, z_velocity, press_progress = z1, 0.0, 1.0
        spin_velocity = _SPIN_SPEED * (1.0 - spin_progress)
    else:
        phase = "terminal_pressure_hold"
        z_target, z_velocity, press_progress = z1, 0.0, 1.0
        spin_velocity = 0.0
    z_target = float(np.clip(z_target, *control["z_limits"]))
    return {
        "phase": phase,
        "time_sec": _r(t),
        "press_z_target_m": z_target,
        "press_z_velocity_target_m_s": z_velocity,
        "spin_velocity_target_rad_s": spin_velocity,
        "press_progress": press_progress,
    }

def _minimum_jerk(start: float, finish: float, now: float, begin: float, end: float) -> tuple[float, float, float]:
    blend, derivative = _minimum_jerk_unit(now, begin, end)
    delta = finish - start
    return start + delta * blend, delta * derivative, blend

def _minimum_jerk_unit(now: float, begin: float, end: float) -> tuple[float, float]:
    u = float(np.clip((now - begin) / (end - begin), 0.0, 1.0))
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    derivative = 30.0 * u**2 * (1.0 - u) ** 2 / (end - begin)
    return blend, derivative
